import os
import os.path as op
import mne
import numpy as np
import subprocess
import multiprocessing
import pickle

HEMIS = ['rh', 'lh']


def namebase(file_name):
    return op.splitext(op.basename(file_name))[0]


def get_fol_name(fname):
    return op.sep.join(fname.split(op.sep)[:-1])


def make_dir(fol):
    if not os.path.isdir(fol):
        os.makedirs(fol)
    return fol


def both_hemi_files_exist(file_template):
    return op.isfile(file_template.format(hemi='rh')) and op.isfile(file_template.format(hemi='lh'))


def morph_labels_from_fsaverage(subject, subjects_dir='', aparc_name='aparc250', fs_labels_fol='',
            sub_labels_fol='', n_jobs=6, fsaverage='fsaverage', overwrite=False):
    if subjects_dir=='':
        subjects_dir = os.environ['SUBJECTS_DIR']
    subject_dir = op.join(subjects_dir, subject)
    labels_fol = op.join(subjects_dir, fsaverage, 'label', aparc_name) if fs_labels_fol=='' else fs_labels_fol
    sub_labels_fol = op.join(subject_dir, 'label', aparc_name) if sub_labels_fol=='' else sub_labels_fol
    if not op.isdir(sub_labels_fol):
        os.makedirs(sub_labels_fol)
    labels = mne.read_labels_from_annot(fsaverage, aparc_name, subjects_dir=subjects_dir)
    if len(labels) == 0:
        raise Exception('morph_labels_from_fsaverage: No labels files found in annot file!'.format(labels_fol))
    surf_loaded = False
    # for label_file in labels_files:
    for fs_label in labels:
        label_file = op.join(labels_fol, '{}.label'.format(fs_label.name))
        local_label_name = op.join(sub_labels_fol, '{}.label'.format(op.splitext(op.split(label_file)[1])[0]))
        if not op.isfile(local_label_name) or overwrite:
            # fs_label = mne.read_label(label_file)
            fs_label.values.fill(1.0)
            sub_label = fs_label.morph(fsaverage, subject, grade=None, n_jobs=n_jobs, subjects_dir=subjects_dir)
            if np.all(sub_label.pos == 0):
                if not surf_loaded:
                    verts = {}
                    for hemi in HEMIS:
                        d = np.load(op.join(subjects_dir, subject, 'mmvt', '{}.pial.npz'.format(hemi)))
                        verts[hemi] = d['verts']
                    surf_loaded = True
                sub_label.pos = verts[sub_label.hemi][sub_label.vertices]
            sub_label.save(local_label_name)


def run_script(cmd, verbose=False):
    if verbose:
        print('running: {}'.format(cmd))
    output = subprocess.check_output('{} | tee /dev/stderr'.format(cmd), shell=True)
    print(output)
    return output


def srf2ply(srf_file, ply_file):
    # print('convert {} to {}'.format(namebase(srf_file), namebase(ply_file)))
    verts, faces, verts_num, faces_num = read_srf_file(srf_file)
    write_ply_file(verts, faces, ply_file)
    return ply_file


def read_srf_file(srf_file):
    with open(srf_file, 'r') as f:
        lines = f.readlines()
        verts_num, faces_num = map(int, lines[1].strip().split(' '))
        sep = '  ' if len(lines[2].split('  ')) > 1 else ' '
        verts = np.array([list(map(float, l.strip().split(sep))) for l in lines[2:verts_num+2]])[:,:-1]
        faces = np.array([list(map(int, l.strip().split(' '))) for l in lines[verts_num+2:]])[:,:-1]
    return verts, faces, verts_num, faces_num


def write_ply_file(verts, faces, ply_file_name):
    PLY_HEADER = 'ply\nformat ascii 1.0\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nelement face {}\nproperty list uchar int vertex_index\nend_header\n'
    verts_num = verts.shape[0]
    faces_num = faces.shape[0]
    faces = np.hstack((np.ones((faces_num, 1)) * 3, faces))
    with open(ply_file_name, 'w') as f:
        f.write(PLY_HEADER.format(verts_num, faces_num))
    with open(ply_file_name, 'ab') as f:
        np.savetxt(f, verts, fmt='%.5f', delimiter=' ')
        np.savetxt(f, faces, fmt='%d', delimiter=' ')


def read_ply_file(ply_file, npz_fname=''):
    with open(ply_file, 'r') as f:
        lines = f.readlines()
        verts_num = int(lines[2].split(' ')[-1])
        faces_num = int(lines[6].split(' ')[-1])
        verts_lines = lines[9:9 + verts_num]
        faces_lines = lines[9 + verts_num:]
        verts = np.array([list(map(float, l.strip().split(' '))) for l in verts_lines])
        faces = np.array([list(map(int, l.strip().split(' '))) for l in faces_lines])[:,1:]
    return verts, faces


def save(obj, fname):
    with open(fname, 'wb') as fp:
        # protocol=2 so we'll be able to load in python 2.7
        pickle.dump(obj, fp)


def load(fname):
    with open(fname, 'rb') as fp:
        obj = pickle.load(fp)
    return obj


def get_n_jobs(n_jobs):
    cpu_num = multiprocessing.cpu_count()
    n_jobs = int(n_jobs)
    if n_jobs > cpu_num:
        n_jobs = cpu_num
    elif n_jobs < 0:
        n_jobs = cpu_num + n_jobs
    return n_jobs


def sms_generator(root_fol, subjects=()):
    import glob
    if len(subjects) == 0:
        subjects = glob.glob(op.join(root_fol, 'nmr*'))
    for subject_fol in subjects:
        smss = glob.glob(op.join(subject_fol, '3mm_SMS*'))
        # smss = ['3mm_SMS1_pa', '3mm_SMS4_ipat1_pa', '3mm_SMS4_ipat2_pa', '3mm_SMS8_pa']
        for sms_fol in smss:
            runs = glob.glob(op.join(op.join(sms_fol, '*')))
            for run_fol in runs:
                run = namebase(run_fol)
                if not run.isdigit():
                    continue
                sms = namebase(sms_fol)
                subject = namebase(subject_fol)
                print(subject, sms, run)
                yield run_fol, subject, sms, run


def l2norm(x):
    return np.sum(np.abs(x)**2,axis=-1)**(1./2)


def maximize_figure(plt):
    import matplotlib
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
    elif backend == 'wxAgg':
        mng = plt.get_current_fig_manager()
        mng.frame.Maximize(True)
    elif backend == 'Qt4Agg':
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
    else:
        print ('{} not supported!'.format(backend))


def chunks(l, n):
    n = int(max(1, n))
    return [l[i:i + n] for i in range(0, len(l), n)]


def run_parallel(func, params, njobs=1):
    import multiprocessing
    if njobs == 1:
        results = [func(p) for p in params]
    else:
        pool = multiprocessing.Pool(processes=njobs)
        results = pool.map(func, params)
        pool.close()
    return results


def get_n_jobs(n_jobs):
    import multiprocessing
    cpu_num = multiprocessing.cpu_count()
    n_jobs = int(n_jobs)
    if n_jobs > cpu_num:
        n_jobs = cpu_num
    elif n_jobs < 0:
        n_jobs = cpu_num + n_jobs
    return n_jobs


def existing_fol(fols):
    for fol in fols:
        if op.isdir(fol):
            return fol
    raise Exception('None of the folders exist!')