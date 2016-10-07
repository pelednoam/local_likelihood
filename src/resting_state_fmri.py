import os
import os.path as op
import nibabel as nib
import mne
import numpy as np
import sklearn.decomposition as deco
import matplotlib.pyplot as plt
import scipy.signal
import scipy.io as sio
import shutil
import glob
from src import freesurfer_utils as fu
from src import utils

FREE_SURFER_HOME = os.environ.get('FREESURFER_HOME', '')
if FREE_SURFER_HOME == '':
    print('Freesurfer is not sourced!')


def convert_fmri_file(volume_fname_template, from_format, to_format):
    output_fname = volume_fname_template.format(format=to_format)
    if not op.isfile(output_fname):
        utils.run_script('mri_convert {} {}'.format(volume_fname_template.format(format=from_format), output_fname))
    return output_fname


def morph_labels_to_all_vertices(subject, atlas, subjects_dir, hemi, n_jobs=1, local_subjects_dir=''):
    morphed_labels_fol = op.join(subjects_dir, subject, 'label') if local_subjects_dir == '' else \
        op.join(local_subjects_dir, subject, 'label')
    morphed_labels_fname = op.join(morphed_labels_fol, '{}.{}_all_vertices.pkl'.format(hemi, atlas))
    if not op.isfile(morphed_labels_fname):
        labels = mne.read_labels_from_annot(subject, atlas, subjects_dir=subjects_dir, hemi=hemi)
        morphed_labels = []
        for label in labels:
            label.values.fill(1.0)
            morphed_label = label.morph(subject, subject, 5, None, subjects_dir, n_jobs)
            morphed_labels.append(morphed_label)
        utils.save(morphed_labels, morphed_labels_fname)


def load_first_vertice(fmri_fname):
    x = nib.load(fmri_fname).get_data()
    x0 = x[0, 0, 0, :]
    np.save(op.join(utils.get_fol_name(fmri_fname), 'first_vertice.npy'), x0)
    plt.plot(x0)
    plt.savefig(op.join(utils.get_fol_name(fmri_fname), 'first_vertice.jpg'))



def calc_measure_across_labels(fmri_fname, subject, atlas, hemi, subjects_dir, measure='PCA', do_plot=False, local_subjects_dir=''):
    output_fname = op.join(utils.get_fol_name(fmri_fname), 'labels_data_{}_{}.npz'.format(aparc_name, measure))
    if op.isfile(output_fname):
        return
    x = nib.load(fmri_fname).get_data()
    print(x.shape)
    morphed_labels_fol = op.join(subjects_dir, subject, 'label') if local_subjects_dir == '' else \
        op.join(local_subjects_dir, subject, 'label')
    morphed_labels_fname = op.join(morphed_labels_fol, '{}.{}_all_vertices.pkl'.format(hemi, atlas))
    labels = utils.load(morphed_labels_fname)
    print(max([max(label.vertices) for label in labels]))
    if measure != 'coef_of_variation_across_time':
        labels_data = np.zeros((len(labels), x.shape[-1]))
    else:
        labels_data = np.zeros((len(labels)))
    labels_names = []
    utils.make_dir(op.join(utils.get_fol_name(fmri_fname), 'figures'))
    for ind, label in enumerate(labels):
        if measure == 'mean':
            labels_data[ind, :] = np.mean(x[label.vertices, 0, 0, :], 0)
        elif measure == 'PCA':
            print(label)
            _x = x[label.vertices, 0, 0, :].T
            remove_cols = np.where(np.all(_x == np.mean(_x, 0), 0))[0]
            _x = np.delete(_x, remove_cols, 1)
            _x = (_x - np.mean(_x, 0)) / np.std(_x, 0)
            pca = deco.PCA(1)
            x_r = pca.fit(_x).transform(_x)
            labels_data[ind, :] = x_r.ravel()
        elif measure == 'coef_of_variation':
            label_mean = np.mean(x[label.vertices, 0, 0, :], 0)
            label_std = np.std(x[label.vertices, 0, 0, :], 0)
            labels_data[ind, :] = label_std / label_mean
        elif measure == 'coef_of_variation_across_time':
            label_mean = np.mean(x[label.vertices, 0, 0, :], 1)
            label_std = np.std(x[label.vertices, 0, 0, :], 1)
            labels_data[ind] = np.mean(label_std) / np.mean(label_mean)
        labels_names.append(label.name)
        if do_plot:
            plt.figure()
            plt.plot(x[label.vertices, 0, 0, :].T)
            plt.savefig(op.join(utils.get_fol_name(fmri_fname), 'figures', '{}.jpg'.format(label.name)))
            plt.close()
    np.savez(output_fname, data=labels_data, names=labels_names)


def calc_cov_and_power_spectrum(fol, aparc_name, tr, measure='PCA'):
    input_fname = op.join(fol, 'labels_data_{}_{}.npz'.format(aparc_name, measure))
    output_fname = op.join(fol, 'cov_pxx_{}_{}.npz'.format(aparc_name, measure))
    d = np.load(input_fname)
    labels_data = d['data']
    labels_names = d['names']
    cov = np.cov(labels_data)
    f, Pxx = scipy.signal.welch(labels_data, tr / 1000, nperseg=32)
    mean_Pxx = np.mean(Pxx, 0)
    np.savez(output_fname, f=f, Pxx=Pxx, mean_Pxx=mean_Pxx, cov=cov)
    sio.savemat(op.join(fol, 'cov_pxx_{}_{}.mat'.format(aparc_name, measure)), dict(
        timelength=500.0, TRin=(tr/1000), TRout=(tr/1000), P_target=mean_Pxx, cov_target=cov))


def calc_simulated_labels(fol, root_fol, aparc_name, tr, measure='PCA', data_len=500):
    out_fname = op.join(fol, 'fmri_timecourse_sim.mat')
    if op.isfile(out_fname):
        return
    d = np.load(op.join(fol, 'cov_pxx_{}_{}.npz'.format(aparc_name, measure)))
    matlab_command = op.join(root_fol, 'simulate_BOLD_timecourse_func_v2.m')
    matlab_command = "'{}'".format(matlab_command)
    #todo: data_len should be like the data
    sio.savemat(op.join(root_fol, 'params.mat'), mdict={
        'cov': d['cov'], 'tr': (tr / 1000), 'mean_Pxx': d['mean_Pxx'], 'data_len': float(data_len)})
    cmd = 'matlab -nodisplay -nosplash -nodesktop -r "run({}); exit;"'.format(matlab_command)
    utils.run_script(cmd)
    shutil.move(op.join(root_fol, 'fmri_timecourse_sim.mat'), out_fname)


def plot_fmri_time_series(fmri_fname, aparc_name, measure):
    d = np.load(op.join(utils.get_fol_name(fmri_fname), 'labels_data_{}_{}.npz'.format(aparc_name, measure)))
    fig = plt.figure()
    if measure != 'coef_of_variation_across_time':
        for label_data in d['data']:
            plt.plot(label_data)
    else:
        width = 0.35
        ind = np.arange(len(d['data']))
        plt.bar(ind, d['data'], width=width)
        plt.xticks(ind + width / 2, d['names'])
        fig.autofmt_xdate()

    plt.savefig(op.join(utils.get_fol_name(fmri_fname), 'labels_data_{}_{}.jpg'.format(aparc_name, measure)))


def project_volume_data(subject, volume_file, hemi):
    surf_output_fname = op.join(utils.get_fol_name(volume_file), '{}.mgz'.format(utils.namebase(utils.namebase(volume_file))))
    surf_data = fu.project_volume_data(volume_file, hemi, subject_id=subject, smooth_fwhm=3,
                                       output_fname=surf_output_fname.format(hemi=hemi))


def create_annotation_from_fsaverage(subject, subjects_dir, aparc_name='aparc250', fsaverage='fsaverage',
        overwrite_morphing=False, fs_labels_fol='', n_jobs=6):
    annotations_exist = utils.both_hemi_files_exist(op.join(subjects_dir, subject, 'label', '{}.{}.annot'.format(
        '{hemi}', aparc_name)))
    if not annotations_exist:
        existing_freesurfer_annotations = ['aparc.DKTatlas40.annot', 'aparc.annot', 'aparc.a2009s.annot']
        if '{}.annot'.format(aparc_name) in existing_freesurfer_annotations:
            utils.make_dir(op.join(subjects_dir, subject, 'label'))
            fu.create_annotation_file(
                subject, aparc_name, subjects_dir=subjects_dir, freesurfer_home=FREE_SURFER_HOME)
        utils.morph_labels_from_fsaverage(subject, subjects_dir, aparc_name, n_jobs=n_jobs,
            fsaverage=fsaverage, overwrite=overwrite_morphing, fs_labels_fol=fs_labels_fol)


def get_tr(fmri_fname):
    output_fname = op.join(utils.get_fol_name(fmri_fname), 'tr.pkl')
    if not op.isfile(output_fname):
        img = nib.load(fmri_fname)
        hdr = img.get_header()
        tr = float(hdr._header_data.tolist()[-1][0])
        utils.save(tr, output_fname)
    else:
        tr = utils.load(output_fname)
    print('tr: {}'.format(tr))
    return tr


if __name__ == '__main__':
    n_jobs = utils.get_n_jobs(-1)
    # subject = 'nmr00956'
    fsaverage = 'fsaverage'
    # sms = '3mm_SMS1_pa'
    # run = '006'
    root_fol = utils.existing_fol(
        ['/home/noam/vic', '/cluster/neuromind/dwakeman/sequence_analysis/sms_study_bay8/raw/func', '/homes/5/npeled/space1/vic'])
    hemi = 'lh'
    local_subjects_dir = os.environ['SUBJECTS_DIR']

    subjects_dir = '/cluster/neuromind/dwakeman/sequence_analysis/sms_study_bay8/subjects'
    aparc_name = 'aparc' # 'laus250'
    volume_fname_template = 'fmcpr.sm5.{}.{}.{}'.format(fsaverage, hemi, '{format}')
    measure = 'PCA'

    for fol, subject, sms, run in utils.sms_generator(root_fol):
        image_name = convert_fmri_file(op.join(fol, volume_fname_template), 'nii.gz', 'mgz')
        tr = get_tr(image_name)
        # morph_labels_to_all_vertices(fsaverage, aparc_name, subjects_dir, hemi, n_jobs, local_subjects_dir=local_subjects_dir)
        # calc_measure_across_labels(image_name, fsaverage, aparc_name, hemi, subjects_dir, measure, local_subjects_dir=local_subjects_dir)
        # plot_fmri_time_series(image_name, aparc_name, measure)
        # calc_cov_and_power_spectrum(fol, aparc_name, tr, measure)
        calc_simulated_labels(fol, root_fol, aparc_name, tr, measure)
