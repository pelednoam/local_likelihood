import os.path as op
import numpy as np
import shutil
import matplotlib.pyplot as plt
from functools import partial
from src import utils
import time

def init_figures():
    from matplotlib import rcParams

    # set plot attributes
    fig_width = 12  # width in inches
    fig_height = 9  # height in inches
    fig_size = [fig_width, fig_height]
    params = {'backend': 'Agg',
              'axes.labelsize': 22,
              'axes.titlesize': 20,
              'text.fontsize': 20,
              'legend.fontsize': 22,
              'xtick.labelsize': 20,
              'ytick.labelsize': 20,
              'lines.linewidth': 5,
              'figure.figsize': fig_size,
              'savefig.dpi': 600,
              'font.family': 'sans-serif'}
    rcParams.update(params)


def K(t_i, t, h, type='triangular'):
    return K_func((t_i - t) / h, type)


def K_func(x, type='triangular'):
    if abs(x) > 1:
        return 0
    if type == 'triangular':
        return 1 - abs(x)
    elif type == 'Epanechnikov':
        return 3/4 * (1 - pow(x, 2))
    elif type == 'tricube':
        return pow((1 - pow(abs(x), 3)), 3)


def est_mean_t(y, t, t_j, h, k_type='triangular'):
    n = len(y)
    nom = sum([K(t[i], t_j, h, k_type) * y[i] for i in range(n)])
    dom = sum([K(t[i], t_j, h, k_type) for i in range(n)])
    return nom / dom


def est_var_t(y, mue, t, t_j, h, k_type='triangular'):
    n = len(y)
    nom = sum([K(t[i], t_j, h, k_type) * pow(y[i] - mue[i], 2) for i in range(n)])
    dom = sum([K(t[i], t_j, h, k_type) for i in range(n)])
    return nom / dom


def est_mean(y, h, t, k_type='triangular'):
    n = len(y)
    return [est_mean_t(y, t, t[j], h, k_type) for j in range(n)]


def est_var(y, mue, h, t, k_type='triangular'):
    n = len(y)
    return [est_var_t(y, mue, t, t[j], h, k_type) for j in range(n)]


# def est_mean_t_equispaced(y, k, m, k_type='triangular'):
#     # lls: local least square
#     nom = sum([K_func(l, k_type) * y[l+k] for l in range(-m, m + 1)])
#     dom = sum([K_func(l, k_type) for l in range(-m, m + 1)])
#     return nom / dom


def vector_mean_ll(ys, k, h, n, k_type='triangular'):
    nom = sum([K(i, k, h, k_type) * ys[:, i] for i in range(n)])
    dom = sum([K(i, k, h, k_type) for i in range(n)])
    return nom / dom


def vector_est_var_t(ys, mues, k, h, n, k_type='triangular'):
    nom = sum([K(i, k, h, k_type) * (ys[:, i] - mues[:, i])*(ys[:, i] - mues[:, i]).T for i in range(n)])
    dom = sum([K(i, k, h, k_type) for i in range(n)])
    return nom / dom


def calc_tr_M(n, h, k_type='triangular'):
    tr_M = sum([K_func(0, k_type) / sum([K(j, k, h, k_type) for j in range(n)]) for k in range(n)])
    return tr_M


def mean_cl_stat(y, mue, h, k_type='triangular'):
    n = len(y)
    var = np.var(y)
    trM = calc_tr_M(n, h, k_type)
    C_L = (1 / n) * sum([(y[k] - mue[k]) / var for k in range(n)]) + 2 * trM / n
    return C_L


def vector_mean_cl_stat(ys, mues, h, k_type='triangular'):
    n = ys.shape[1]
    d = ys.shape[0]
    var = np.var(ys)
    trM = calc_tr_M(n, h, k_type)
    C_L = (1 / n) * sum([pow(np.linalg.norm([ys[:, k] - mues[:, k]]), 2) / var for k in range(n)]) + \
          ((2 * d) / n) * trM
    # C_L = (1 / n) * sum([(ys[:, k] - mues[:, k]) / var for k in range(n)]) + \
    #        2 * d * trM / n
    return C_L


def eit(ys, mues, k):
    return ys[:, k] - mues[:, k]


def sum_eit(ys, mues, n):
    return sum([eit(ys, mues, k) for k in range(n)])


def sum_Kit(t, n, h, k_type='triangular'):
    return sum([K(j, t, h, k_type) for j in range(n)])


def vector_mean_and_cov_cl_stat(ys, mues, h, k_type='triangular'):
    n = ys.shape[1]
    d = ys.shape[0]
    K0 = K_func(0, k_type)
    _eit = partial(eit, ys=ys, mues=mues)
    _est_var = partial(vector_est_var_t, ys=ys, mues=mues, h=h, n=n, k_type=k_type)
    _sum_Kit = partial(sum_Kit, n=n, h=h, k_type=k_type)

    AIC_mue = (1 / n) * sum([_eit(k) * 1 / _est_var(k) * _eit(k) for k in range(n)]) + \
        (1 / n) * d * K0 * sum([1 / _sum_Kit(k) for k in range(n)])

    AIC_var = (1/ n) * sum([np.log(abs(_est_var(k))) for k in range(n)]) + \
              (K0 / n) * sum([pow(_eit(k) * 1 / _est_var(k) * _eit(k), 2) / _sum_Kit(k) for k in range(n)])

    return AIC_mue + AIC_var


def mean_var_cl_stat(y, mue, var, h, k_type='triangular'):
    if np.all(var == 0):
        return np.nan, np.nan
    n = len(y)
    K0 = K_func(0, k_type)
    # const_var = np.var(y)
    # trM = calc_tr_M(n, h, k_type) #  = nK(0) / sum(K[(t_j - t_i) / h]) = nK(0) / sum(K_j,t_i)
    # CL_mean = (1 / n) * sum([np.power(y[k] - mue[k], 2) / const_var for k in range(n)]) + \
    CL_mean = (1 / n) * sum([pow(y[t] - mue[t], 2) / var[t] for t in range(n)]) + \
              (1 / n) * sum([2 * K0 / sum([K(j, t, h, k_type) for j in range(n)]) for t in range(n)])
    CL_var = (1 / n) * sum([np.log(var[t]) for t in range(n)]) + \
             (1 / n) * sum([(2 * K0 * pow(y[t] - mue[t], 4)) /
                            (sum([K(j, t, h, k_type) for j in range(n)]) * pow(var[t], 2)) for t in range(n)])
    return CL_mean, CL_var


def est_vector_mean_ll(ys, h, k_type='triangular'):
    n = ys.shape[1]
    return np.array([vector_mean_ll(ys, k, h, n, k_type) for k in range(n)]).T


def est_vector_est_var_t(ys, mues, h, k_type='triangular'):
    n = ys.shape[1]
    return np.array([vector_est_var_t(ys, mues, k, h, n, k_type) for k in range(n)]).T


def plot_mean_var(y, hs, t_axis, fol, k_type='triangular'):
    if not op.isfile(op.join(fol, 'ests_mean_{}.pkl'.format(k_type))):
        ests_mean, ests_var = {}, {}
        for h in hs:
            ests_mean[h] = est_mean(y, h, t_axis, k_type)
            ests_var[h] = est_var(y, ests_mean[h], h, t_axis, k_type)
        utils.save(ests_mean, op.join(fol, 'ests_mean_{}.pkl'.format(k_type)))
        utils.save(ests_var, op.join(fol, 'ests_var_{}.pkl'.format(k_type)))
    else:
        ests_mean = utils.load(op.join(fol, 'ests_mean_{}.pkl'.format(k_type)))
        ests_var = utils.load(op.join(fol, 'ests_var_{}.pkl'.format(k_type)))

    boynton_colors = ["red", "green", "yellow", "magenta", "pink", "orange", "brown", "gray"]
    t = range(len(y))
    plt.figure()
    plt.plot(t, y, 'b', label='real')
    for h, color in zip(hs, boynton_colors):
        # plt.errorbar(t, ests_mean[h], c=color, yerr=np.power(ests_var[h], 0.5), label='w {}'.format(h))
        plt.plot(t, ests_mean[h], color, label='w {}'.format(h))
        error = np.power(ests_var[h], 0.5)
        plt.fill_between(t, ests_mean[h] - error, ests_mean[h] + error,
                        alpha=0.2, edgecolor=color, facecolor=color)
    plt.legend()
    plt.savefig(op.join(fol, 'first_vertice_h_var_k_{}.jpg'.format(k_type)))


def est_mean_and_var(ys, names, hs_tr, hs_s, t_axis, fol, k_type='triangular', sim=False, overwrite=False,
                     specific_label='', n_jobs=1):
    output_fname = op.join(fol, 'mean_var{}_{}_{}-{}.npz'.format('_sim' if sim else '', k_type, hs_s[0], hs_s[-1]))
    if op.isfile(output_fname) and not overwrite:
        d = np.load(op.join(fol, 'mean_var_sim_{}.npz'.format(k_type)))
        if np.any(np.array(d['hs_tr']) != np.array(hs_tr)):
            overwrite = True
            print('The parameter hs_tr is not the same as in the saved file, recalculating.')
    W = len(hs_s)
    if not op.isfile(output_fname) or overwrite:
        # all_means, all_vars = np.zeros((ys.shape[0], W)), np.zeros((ys.shape[0], W))
        all_means = np.zeros((len(hs_tr), len(names), ys.shape[1]))
        all_vars = np.zeros((len(hs_tr), len(names), ys.shape[1]))

        h_chunks = utils.chunks(list(zip(hs_tr, hs_s, range(W))), W / n_jobs)
        params = [(ys, names, h_chunk, t_axis, fol, k_type, specific_label) for h_chunk in h_chunks]
        results = utils.run_parallel(_est_mean_and_var_parallel, params, n_jobs)
        # for chunk_means, chunk_vars in results:
        #     for (h_ind, mean), var in zip(chunk_means.items(), chunk_vars.values()):
        #         all_means[h_ind] = mean
        #         all_vars[h_ind] = var
        for (chunk_means, chunk_vars) in results:
            for h_ind, label_ind in chunk_means.keys():
                all_means[h_ind, label_ind, :] = chunk_means[(h_ind, label_ind)]
                all_vars[h_ind, label_ind, :] = chunk_vars[(h_ind, label_ind)]

        # for ind, (h_tr, h_s) in enumerate(zip(hs_tr, hs_s)):
        #     print('h: {}s'.format(h_s))
        #     all_means[ind] = est_vector_mean_ll(ys, h_tr, k_type)
        #     all_vars[ind] = est_vector_est_var_t(ys, all_means[ind], h_tr, k_type)
        np.savez(output_fname, means=all_means, vars=all_vars, hs_tr=hs_tr, hs_ms=hs_s)


def _est_mean_and_var_parallel(p):
    ys, labels_names, h_chunk, time_axis, fol, k_type, specific_label = p
    all_means, all_vars = {}, {}
    for h_tr, h_s, h_ind in h_chunk:
        print('h: {}s'.format(h_s))
        for label_id, label_name in enumerate(labels_names):
            if specific_label != '' and label_name != specific_label:
                continue
            mue = all_means[(h_ind, label_id)] = est_mean(ys[label_id], h_s, time_axis, k_type)
            all_vars[(h_ind, label_id)] = est_var(ys[label_id], mue, h_s, time_axis, k_type)
    return all_means, all_vars


def est_vector_mean_and_var(ys, names, hs_tr, hs_s, fol, k_type='triangular', sim=False, overwrite=False, n_jobs=1):
    output_fname = op.join(fol, 'vector_mean_var{}_{}.npz'.format('_sim' if sim else '', k_type))
    if op.isfile(output_fname) and not overwrite:
        d = np.load(output_fname)
        if np.any(np.array(d['hs_tr']) != np.array(hs_tr)):
            overwrite = True
            print('The parameter hs_tr is not the same as in the saved file, recalculating.')
    if not op.isfile(output_fname) or overwrite:
        all_means = np.zeros((len(hs_tr), len(names), ys.shape[1]))
        all_vars = np.zeros((len(hs_tr), len(names), ys.shape[1]))
        h_chunks = utils.chunks(list(zip(hs_tr, hs_s, range(len(hs_tr)))), len(hs_tr) / n_jobs)
        params = [(ys, names, h_chunk, fol, k_type) for h_chunk in h_chunks]
        results = utils.run_parallel(_est_vector_mean_and_var_parallel, params, n_jobs)
        for chunk_means, chunk_vars in results:
            for (h_ind, mean), var in zip(chunk_means.items(), chunk_vars.values()):
                all_means[h_ind] = mean
                all_vars[h_ind] = var
        # for ind, (h_tr, h_s) in enumerate(zip(hs_tr, hs_s)):
        #     print('h: {}s'.format(h_s))
        #     all_means[ind] = est_vector_mean_ll(ys, h_tr, k_type)
        #     all_vars[ind] = est_vector_est_var_t(ys, all_means[ind], h_tr, k_type)
        np.savez(output_fname, means=all_means, vars=all_vars, hs_tr=hs_tr, hs_ms=hs_s)


def _est_vector_mean_and_var_parallel(p):
    ys, names, h_chunk, fol, k_type = p
    all_means, all_vars = {}, {}
    for h_tr, h_s, h_ind in h_chunk:
        print('h: {}s'.format(h_s))
        all_means[h_ind] = est_vector_mean_ll(ys, h_tr, k_type)
        all_vars[h_ind] = est_vector_est_var_t(ys, all_means[h_ind], h_tr, k_type)
    return all_means, all_vars


def calc_vector_mean_cl(ys, fol, hs_tr, k_type='triangular', sim=False, overwrite=False, n_jobs=1):
    output_fname = op.join(fol, 'vector_mean_cl{}_{}.npz'.format('_sim' if sim else '', k_type))
    if op.isfile(output_fname) and not overwrite:
        d = np.load(output_fname)
        if np.any(np.array(d['hs_tr']) != np.array(hs_tr)):
            overwrite = True
    if not op.isfile(output_fname) or overwrite:
        d = np.load(op.join(fol, 'vector_mean_var{}_{}.npz'.format('_sim' if sim else '', k_type)))
        means_est, vars_est, hs_tr, hs_ms = d['means'], d['vars'], d['hs_tr'], d['hs_ms']
        mean_cl, var_cl = np.zeros((len(hs_tr))), np.zeros((len(hs_tr)))

        h_chunks = utils.chunks(list(enumerate(hs_tr)), len(hs_tr) / n_jobs)
        params = [(ys, means_est, h_chunk, k_type) for h_chunk in h_chunks]
        results = utils.run_parallel(_calc_vector_mean_cl_parallel, params, n_jobs)
        for chunk_mean_cl in results:
            for h_ind in chunk_mean_cl.keys():
                mean_cl[h_ind] = chunk_mean_cl[h_ind]
        # for hs_ind, hs_tr in enumerate(hs_tr):
        #     mean_cl[hs_ind] = vector_mean_cl_stat(ys, means[hs_ind], hs_tr, k_type)
        np.savez(output_fname, mean_cl=mean_cl, var_cl=var_cl, hs_tr=hs_tr, hs_ms=hs_ms)


def _calc_vector_mean_cl_parallel(p):
    ys, means_est, h_chunk, k_type = p
    mean_cl = {}
    for h_ind, h_tr in h_chunk:
        mean_cl[h_ind] = vector_mean_cl_stat(ys, means_est[h_ind], h_tr, k_type)
    return mean_cl


def calc_vector_mean_cov_cl(ys, fol, hs_tr, k_type='triangular', sim=False, overwrite=False, n_jobs=1):
    output_fname = op.join(fol, 'vector_mean_cl_cov{}_{}.npz'.format('_sim' if sim else '', k_type))
    if op.isfile(output_fname) and not overwrite:
        d = np.load(output_fname)
        if np.any(np.array(d['hs_tr']) != np.array(hs_tr)):
            overwrite = True
    if not op.isfile(output_fname) or overwrite:
        d = np.load(op.join(fol, 'vector_mean_var{}_{}.npz'.format('_sim' if sim else '', k_type)))
        means_est, vars_est, hs_tr, hs_ms = d['means'], d['vars'], d['hs_tr'], d['hs_ms']
        mean_cl, var_cl = np.zeros((len(hs_tr))), np.zeros((len(hs_tr)))

        h_chunks = utils.chunks(list(enumerate(hs_tr)), len(hs_tr) / n_jobs)
        params = [(ys, means_est, h_chunk, k_type) for h_chunk in h_chunks]
        results = utils.run_parallel(_calc_vector_mean_and_cov_cl_parallel, params, n_jobs)
        for chunk_mean_cl in results:
            for h_ind in chunk_mean_cl.keys():
                mean_cl[h_ind] = chunk_mean_cl[h_ind]
        np.savez(output_fname, mean_cl=mean_cl, var_cl=var_cl, hs_tr=hs_tr, hs_ms=hs_ms)


def _calc_vector_mean_and_cov_cl_parallel(p):
    ys, means_est, h_chunk, k_type = p
    mean_cov_cl = {}
    for h_ind, h_tr in h_chunk:
        mean_cov_cl[h_ind] = vector_mean_and_cov_cl_stat(ys, means_est[h_ind], h_tr, k_type)
    return mean_cov_cl


def calc_mean_var_cl(ys, fol, hs_tr, hs_s, labels_names, k_type='triangular', sim=False, overwrite=False,
                     specific_label='', n_jobs=1):
    from itertools import product
    output_fname = op.join(fol, 'mean_var{}_cl_{}_{}-{}.npz'.format('_sim' if sim else '', k_type, hs_s[0], hs_s[-1]))
    if op.isfile(output_fname) and not overwrite:
        d = np.load(output_fname)
        if np.any(np.array(d['hs_tr']) != np.array(hs_tr)):
            overwrite = True
    if not op.isfile(output_fname) or overwrite:
        intpu_fname = op.join(fol, 'mean_var{}_{}_{}-{}.npz'.format('_sim' if sim else '', k_type, hs_s[0], hs_s[-1]))
        d = np.load(intpu_fname)
        means_est, vars_est, hs_tr, hs_ms = d['means'], d['vars'], d['hs_tr'], d['hs_ms']
        mean_cl, var_cl = np.zeros((ys.shape[0], len(hs_tr))), np.zeros((ys.shape[0], len(hs_tr)))
        params_to_chunk = []
        for (h_tr_ind, h_tr), (label_ind, label_name) in product(enumerate(hs_tr), enumerate(labels_names)):# range(ys.shape[0])):
            params_to_chunk.append((h_tr_ind, h_tr, label_ind, label_name))
        h_l_chunks = utils.chunks(params_to_chunk, len(params_to_chunk) / n_jobs)
        params = [(ys, means_est, vars_est, h_l_chunk, k_type, specific_label) for h_l_chunk in h_l_chunks]
        results = utils.run_parallel(_calc_mean_var_cl_parallel, params, n_jobs)
        for (chunk_mean_cl, chunk_var_cl) in results:
            for h_ind, label_ind in chunk_mean_cl.keys():
                mean_cl[label_ind, h_ind] = chunk_mean_cl[(h_ind, label_ind)]
                var_cl[label_ind, h_ind] = chunk_var_cl[(h_ind, label_ind)]
        np.savez(output_fname, mean_cl=mean_cl, var_cl=var_cl, hs_tr=hs_tr, hs_ms=hs_ms)


def _calc_mean_var_cl_parallel(p):
    ys, means_est, vars_est, h_l_chunk, k_type, specific_label = p
    mean_cl, var_cl = {}, {}
    for h_ind, h_tr, label_ind, label_name in h_l_chunk:
        if specific_label != '' and label_name != specific_label:
            continue
        # print(h_tr)
        mean_cl[(h_ind, label_ind)], var_cl[(h_ind, label_ind)] = mean_var_cl_stat(
            ys[label_ind, :], means_est[h_ind, label_ind, :], vars_est[h_ind, label_ind, :], h_tr, k_type)
    return mean_cl, var_cl


def plot_vector_mean_var(subject, sms, run, ys, names, label_ids, fol, tr, hs_plot, k_type='triangular', sim=False,
                         overwrite=False, ax=None, plot_legend=True, xlim=None):
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    from itertools import cycle

    majorLocator = MultipleLocator(100)
    majorFormatter = FormatStrFormatter('%d')
    minorLocator = MultipleLocator(50)

    d = np.load(op.join(fol, 'vector_mean_var{}_{}.npz'.format('_sim' if sim else '', k_type)))
    means, vars, hs_tr, hs_ms = d['means'], d['vars'], d['hs_tr'], d['hs_ms']
    boynton_colors = cycle(["red", "green",  "magenta", "yellow", "pink", "orange", "brown", "gray"])
    t0 = int(2 * (max(hs_tr) - 1))
    # t = range(ys.shape[1])[t0:]
    t = np.arange(ys.shape[1])[t0:] * tr / 1000
    print(sms, tr, t[0], t[-1], ys.shape[1])
    utils.make_dir(op.join(fol, 'pca_labels'))

    for y, label_name, label_ind in zip(ys, names, label_ids):
        fig_fname = op.join(fol, 'pca_labels', '{}_{}.jpg'.format(label_name, k_type))
        # if op.isfile(fig_fname) and not overwrite:
        #     continue
        if ax is None:
            fig, ax = plt.subplots()
        lines = []
        l, = ax.plot(t, y[t0:], 'b', label='simulated' if sim else 'real')
        lines.append(l)
        for h_s in hs_plot:
            h_ind = np.where(np.array(hs_ms) == h_s)[0][0]
            h_ms = hs_ms[h_ind]
            color = next(boynton_colors)
            mean = means[h_ind, label_ind, t0:]
            error = np.power(vars[h_ind, label_ind, t0:], 0.5)
            l, = ax.plot(t, mean, color, label='w {:d}s'.format(h_ms))
            lines.append(l)
            ax.fill_between(t, mean - error, mean + error,
                            alpha=0.2, edgecolor=color, facecolor=color)
        if plot_legend:
            ax.legend(handles=lines, bbox_to_anchor=(1.05, 1.1))

        ax.set_xlabel('Time (s)')
        if ax is None:
            ax.set_title('{} {} {} {}'.format(subject, sms, run, label_name))
        else:
            ax.set_title('{}'.format(sms.replace('_', ' ')))
            utils.maximize_figure(plt)
            plt.tight_layout()

        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_major_formatter(majorFormatter)
        ax.xaxis.set_minor_locator(minorLocator)
        if not xlim is None:
            ax.set_xlim(xlim)

        if ax is None:
            plt.savefig(fig_fname, dpi=200)
            plt.close()


def plot_mean_var_cl(fol, root_fol, subject, sms, run, labels_names, k_type='triangular', sim=False):
    # if not op.isfile(input_fname):
    #     shutil.move(op.join(fol, 'mean_var_cl_{}.npz'.format(k_type)), input_fname)
    input_fname = op.join(fol, 'mean_var_cl{}_{}.npz'.format('_sim' if sim else '', k_type))
    d = np.load(input_fname)
    mean_cl, var_cl, hs_tr, hs_ms = d['mean_cl'], d['var_cl'], d['hs_tr'], d['hs_ms']
    for cl_vector, cl_name in zip([mean_cl + var_cl, mean_cl, var_cl], ['AIC', 'AIC_mean', 'AIC_var']):
        for label_ind in range(mean_cl.shape[0]):
            cl = cl_vector[label_ind]
            label_name = labels_names[label_ind]
            figs_fol = op.join(root_fol, 'figures', 'mean_var{}_cl'.format('_sim' if sim else ''), label_name, cl_name)
            utils.make_dir(figs_fol)
            fig_fname = op.join(figs_fol, '{}_{}_{}_{}.jpg'.format(subject, sms, cl_name, label_name))
            # if op.isfile(fig_fname):
            #     continue
            ind = np.arange(len(hs_ms))
            if np.any(np.isnan(cl)):
                ind = ind[~np.isnan(cl)]
                cl = cl[ind]
            fig = plt.figure()
            width = 0.35
            # plt.bar(ind, cl, width=width)
            plt.scatter(ind, cl, marker='o', facecolors='none')
            # plt.xticks(ind + width / 2, hs_ms[ind])
            plt.xlim([-0.5, len(cl) + 0.5])
            plt.title('{} {} {} {} {}{}'.format(cl_name, subject, label_name, sms, run, ' sim' if sim else ''))
            plt.xlabel('window-width (s)')
            # plt.text(ind[cl.argmin()], cl.min() * 0.97 + .03 * cl.max(), '*', fontsize=14)
            plt.scatter(ind[cl.argmin()], cl.min(), marker='o')
            plt.plot(ind, cl, '--')
            # utils.maximize_figure(plt)
            # plt.tight_layout()
            # plt.show()
            plt.savefig(fig_fname, dpi=100)
            plt.close()


def plot_vector_mean_var_cl(fol, root_fol, subject, sms, run, k_type='triangular', sim=False):
    d = np.load(op.join(fol, 'vector_mean_cl{}_{}.npz'.format('_sim' if sim else '', k_type)))
    mean_cl, var_cl, hs_ms = d['mean_cl'], d['var_cl'], d['hs_ms']
    # for cl, cl_name in zip([mean_cl, var_cl], ['AIC mean', 'AIC var']):
    for cl, cl_name in zip([mean_cl], ['AIC mean']):
        fig = plt.figure()
        width = 0.35
        ind = np.arange(len(hs_ms))
        # plt.bar(ind, cl, width=width)
        plt.scatter(ind, cl, marker='o', facecolors='none')
        # plt.xticks(ind + width / 2, hs_ms)
        plt.title('{} {} {} {}{}'.format(cl_name, subject, sms, run, ' sim' if sim else ''))
        plt.xlabel('window-width (s)')
        # plt.text(cl.argmin(), cl.min() * 0.97 + .03 * cl.max(), '*', fontsize=14)
        plt.scatter(cl.argmin(), cl.min(), marker='o')
        plt.xlim([-0.5, len(cl) + 0.5])
        utils.maximize_figure(plt)
        plt.tight_layout()
        # plt.show()
        figures_fol = op.join(root_fol, 'figures', 'cl_mean_figures{}'.format('_sim' if sim else ''))
        utils.make_dir(figures_fol)
        plt.savefig(op.join(figures_fol, 'vector_mean_cl_{}_{}_{}{}.jpg'.format(
            subject, run, sms, '_sim' if sim else '')), dpi=200)
        plt.close()


def copy_figures(subject, sms, run, fol, root_fol, label_name, k_type='triangular'):
    utils.make_dir(op.join(root_fol, 'mean_var_figures'))
    shutil.copy(op.join(fol, 'pca_labels', '{}_{}.jpg'.format(label_name, k_type)),
                op.join(root_fol, 'mean_var_figures', '{}_{}_{}_{}_{}.jpg'.format(subject, sms, run, label_name, k_type)))


def compare_vector_mean_var_cl(subject, sms, run, fol, root_fol, k_types=['triangular']):

    for k_type in k_types:
        d = np.load(op.join(fol, 'vector_mean_cl_{}_{}-{}.npz'.format(k_type, )))
        cl, hs_ms = d['mean_cl'], d['hs_ms']
        d = np.load(op.join(fol, 'vector_mean_cl_sim_{}_{}-{}.npz'.format(k_type)))
        cl_sim, hs_ms_sim = d['mean_cl'], d['hs_ms']

        fig = plt.figure()
        width = 0.35
        ind = np.arange(len(hs_ms))
        # plt.bar(ind, cl, width=width)
        plt.scatter(ind, cl, marker='o', facecolors='none', label='real data')
        plt.scatter(ind, cl_sim, marker='^', facecolors='none', label='simulated data')
        # plt.xticks(ind + width / 2, hs_ms)
        plt.title('{} {} {} {}{}'.format('AIC mean', subject, sms, run, ' sim' if sim else ''))
        plt.xlabel('window-width (s)')
        # plt.text(cl.argmin(), cl.min() * 0.97 + .03 * cl.max(), '*', fontsize=14)
        plt.legend()
        plt.scatter(cl.argmin(), cl.min(), marker='o')
        plt.scatter(cl_sim.argmin(), cl_sim.min(), marker='^')
        plt.plot(ind, cl, '--')
        plt.plot(ind, cl_sim, '--')
        plt.xlim([-0.5, len(cl) + 0.5])

        utils.maximize_figure(plt)
        plt.tight_layout()
        # plt.show()
        figures_fol = op.join(root_fol, 'figures', 'cl_mean_comparison_figures')
        utils.make_dir(figures_fol)
        plt.savefig(op.join(figures_fol, 'vector_mean_cl_{}_{}_{}.jpg'.format(subject, sms, run)), dpi=200)
        plt.close()


def merge_mean_var_cl_files(fol, exclude=[], k_type='triangular'):

    def merge(search_str, output_name_format):
        import glob
        files = glob.glob(op.join(fol, search_str))
        for ind, input_fname in enumerate(files):
            d = np.load(input_fname)
            print(d['hs_ms'][0], d['hs_ms'][-1])
            if len(exclude) == 2 and d['hs_ms'][0] == exclude[0] and d['hs_ms'][-1] == exclude[1]:
                continue
            all_hs_ms = d['hs_ms'] if ind == 0 else np.hstack((all_hs_ms, d['hs_ms']))
            all_hs_tr = d['hs_tr'] if ind == 0 else np.hstack((all_hs_tr, d['hs_tr']))
            all_mean_cl = d['mean_cl'] if ind == 0 else np.hstack((all_mean_cl, d['mean_cl']))
            all_var_cl = d['var_cl'] if ind == 0 else np.hstack((all_var_cl, d['var_cl']))

        inds = np.argsort(all_hs_ms)
        all_hs_ms = all_hs_ms[inds]
        all_mean_cl = all_mean_cl[:, inds]
        all_var_cl = all_var_cl[:, inds]
        all_hs_tr = all_hs_tr[inds]
        output_fname = op.join(fol, output_name_format.format(k_type, all_hs_ms[0], all_hs_ms[-1]))
        np.savez(output_fname, mean_cl=all_mean_cl, var_cl=all_var_cl, hs_tr=all_hs_tr, hs_ms=all_hs_ms)

    merge('mean_var_cl_{}*.npz'.format(k_type), 'mean_var_cl_{}_{}-{}.npz')
    merge('mean_var_sim_cl_{}*.npz'.format(k_type), 'mean_var_sim_cl_{}_{}-{}.npz')


def compare_mean_var_cl(fol, root_fol, subject, sms, run, labels_names, hs, top_hs=-1, labels_ids=None,
                        hs_sim=None, k_type='triangular'):
    from scipy.signal import argrelextrema

    input_fname = op.join(fol, 'mean_var_cl_{}_{}-{}.npz'.format(k_type, hs[0], hs[1]))
    if not op.isfile(input_fname):
        print('{} does not exist!'.format(input_fname))
        return
    d = np.load(input_fname)
    utils.print_modif_time(input_fname)
    # cl_vector = d['mean_cl'] + d['var_cl']
    hs_tr, hs_ms = d['hs_tr'], d['hs_ms']
    if hs_sim is None:
        hs_sim = hs
    sim_fname = op.join(fol, 'mean_var_sim_cl_{}_{}-{}.npz'.format(k_type, hs_sim[0], hs_sim[1]))
    if op.isfile(sim_fname):
        d_sim = np.load(sim_fname)
        utils.print_modif_time(sim_fname)
        no_sim_data = False
    else:
        print("Can't find sim data!")
        d_sim = np.load(input_fname)
        no_sim_data = True
    # cl_vector_sim = d_sim['mean_cl'] + d_sim['var_cl']
    hs_tr_sim, hs_ms_sim = d['hs_tr'], d['hs_ms']
    # cl_name = 'AIC'
    # fig, axes = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(12, 8))
    # axs = list(itertools.chain(*axes))
    root_figs_fol = op.join(root_fol, 'figures', subject, 'mean_var_cl_compare_{}_{}'.format(hs[0], top_hs))# hs[-1]))
    utils.make_dir(root_figs_fol)
    if labels_ids is None:
        labels_ids = range(len(labels_names))
    for cl_name, cl_vector, cl_vector_sim in zip(['AIC', 'AIC_mean', 'AIC_var'],
                                                 [d['mean_cl'] + d['var_cl'], d['mean_cl'], d['var_cl']],
                                                 [d_sim['mean_cl'] + d_sim['var_cl'], d_sim['mean_cl'], d_sim['var_cl']]):
        for ind, label_ind in enumerate(labels_ids):
            fig = plt.figure()
            cl_real = cl_vector[label_ind]
            cl_sim = cl_vector_sim[label_ind]
            label_name = labels_names[ind]
            figs_fol = op.join(root_figs_fol, label_name, cl_name)
            utils.make_dir(figs_fol)
            fig_fname = op.join(figs_fol, '{}_{}_{}_{}_{}.jpg'.format(subject, sms, run, label_name, cl_name))
            # if op.isfile(fig_fname):
            #     continue
            # ind_real = np.arange(len(hs_ms))
            # ind_sim = np.arange(len(hs_ms_sim))
            ind_real = hs_ms[:top_hs]
            ind_sim = hs_ms_sim[:top_hs]
            # print('ind real len: {}, ind sim len: {}'.format(len(ind_real), len(ind_sim)))
            # print('cl real shape: {}, cl sim shape: {}'.format(cl_real.shape, cl_sim.shape))
            lines = []
            if no_sim_data:
                itr = zip([cl_real], ['o'], [ind_real], ['real'], ['b'])
            else:
                itr = zip([cl_real, cl_sim], ['o', '^'], [ind_real, ind_sim], ['real', 'sim'], ['b', 'g'])
            for cl, marker, ind, label, c in itr:
            # for cl, marker, ind, label in zip([cl_real], ['o'], [ind_real], ['real']):
                cl = cl[:top_hs]
                if np.any(np.isnan(cl)):
                    ind = ind[~np.isnan(cl)]
                    cl = cl[~np.isnan(cl)]
                # plt.bar(ind, cl, width=width)
                # plt.scatter(ind, cl, marker=marker, facecolors='none', label=label)
            # for cl, ind, c, label in zip([cl_real, cl_sim], [ind_real, ind_sim], ['b', 'g'], ['real', 'sim']):
            # for cl, ind in zip([cl_real], [ind_real]):
                # plt.text(ind[cl.argmin()], cl.min() * 0.97 + .03 * cl.max(), '*', fontsize=14)
                # plt.scatter(ind[cl.argmin()], cl.min(), marker='o')
                ind = ind[:len(cl)]
                l, = plt.plot(ind, cl, '--', label=label)
                lines.append(l)
                # plt.legend()
                minm = argrelextrema(cl, np.less)  # (array([2, 5, 7]),)
                for cl_min in minm[0]:
                    plt.scatter(ind[cl_min], cl[cl_min], marker='o', c=c)
            # plt.xlim([-0.5, len(cl_real) + 0.5])
            plt.legend(handles=lines)
            plt.xlim([-0.5, max(ind_real) + 0.5])
            # plt.title('{} {} {} {} {}'.format(cl_name, subject, label_name, sms, run))
            plt.title('{}'.format(sms.replace('_', ' ')))
            plt.ylabel(cl_name)
            plt.xlabel('window-width (s)')
            # utils.maximize_figure(plt)
            # plt.tight_layout()
            # plt.show()
            print('Saving figure {}'.format(fig_fname))
            plt.savefig(fig_fname, dpi=100)
            plt.close()


def plot_mean_var(k_type='triangular', sim=False, labels_names=None, label_ids=None):
    from collections import defaultdict, OrderedDict
    import itertools

    xlim = None #[200, 300]
    figures_fol = op.join(root_fol, 'figures', 'smss_per_label{}{}'.format(
        '_window' if not xlim is None else '', '_sim' if sim else ''))
    utils.make_dir(figures_fol)
    gen, data = {}, {}
    hs_plot = [5, 10, 15, 25]
    if labels_names is None:
        labels_names = utils.load(op.join(root_fol, 'labels_names.pkl'))
        label_ids = range(len(labels_names))
    for fol, subject, sms, run in utils.sms_generator(root_fol):
        print(fol, subject, sms, run)
        if subject not in gen:
            gen[subject] = OrderedDict()
        if sms not in gen[subject]:
            gen[subject][sms] = []
        gen[subject][sms].append((fol, run))
        if not sim:
            d = np.load(op.join(fol, 'labels_data_{}_{}.npz'.format(atlas, measure)))
            data[fol] = d['data']
        else:
            import scipy.io as sio
            d_sim = sio.loadmat(op.join(fol, 'fmri_timecourse_sim.mat'))
            data[fol] = d_sim['timecourse_use_sim'].T

    # labels_names = [0]
    now = time.time()
    for subject, (label_id, label_name) in itertools.product(gen.keys(), zip(label_ids, labels_names)):
        utils.time_to_go(now, label_id, len(labels_names), 5)
        fig, axes = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(12, 8))
        axs = list(itertools.chain(*axes))
        # fig.suptitle('{} {} {}'.format(subject, label_name, 'sim' if sim else ''))
        for index, (ax, sms) in enumerate(zip(axs, gen[subject].keys())):
            fol, run = gen[subject][sms][0]
            tr = utils.load(op.join(fol, 'tr.pkl'))
            print(tr)
            ys = np.array([data[fol][label_id]])
            plot_vector_mean_var(subject, sms, run, ys, [label_name], [label_id], fol, tr, hs_plot, k_type, sim,
                                 overwrite=False, ax=ax, plot_legend=index==1, xlim=None)

        utils.maximize_figure(plt)
        plt.tight_layout()
        plt.savefig(op.join(figures_fol, label_name))
        plt.close()


def combine_mean_var_sim_plots(root_fol, labels_names):
    figure_fol = op.join(root_fol, 'figures')
    utils.make_dir(op.join(figure_fol, 'smss_per_label_compare'))
    now = time.time()
    for ind, label in enumerate(labels_names):
        utils.time_to_go(now, ind, len(labels_names), 5)
        figure_real_data = op.join(figure_fol, 'smss_per_label', '{}.png'.format(label))
        figure_sim_data = op.join(figure_fol, 'smss_per_label_sim', '{}.png'.format(label))
        new_image_fname = op.join(figure_fol, 'smss_per_label_compare', '{}.png'.format(label))
        utils.combine_two_images(figure_real_data, figure_sim_data, new_image_fname)


def combine_mean_var_cl_sim_plots(root_fol, subject, labels_names, hs):
    import glob
    import os
    figure_fol = op.join(root_fol, 'figures', subject, 'mean_var_cl_compare_{}_{}'.format(hs[0], hs[-1]))
    # figure_fol = op.join(root_fol, 'figures', 'mean_var_cl_compare')
    # utils.make_dir(op.join(figure_fol, 'smss_per_label_compare'))
    now = time.time()
    for ind, label in enumerate(labels_names):
        utils.time_to_go(now, ind, len(labels_names), 5)
        figs = sorted(glob.glob(op.join(figure_fol, label, '*.jpg')))
        if len(figs) == 0:
            continue
        elif len(figs) == 4:
            utils.combine_four_images(figs, op.join(figure_fol, '{}.png'.format(label)))
        else:
            utils.combine_nine_images(figs, op.join(figure_fol, '{}.png'.format(label)))


def main(subject, sms, run, fmri_fname, fol, root_fol, atlas, tr, hs_s, k_types=['triangular'], measure='PCA',
         sim=False, labels_names=None, labels_ids=None, only_one_trace=False, overwrite=False,
         specific_label='', n_jobs=-2):
    if only_one_trace:
        y = np.load(op.join(utils.get_fol_name(fmri_fname), 'first_vertice.npy'))
    else:
        if not sim:
            d = np.load(op.join(utils.get_fol_name(fmri_fname), 'labels_data_{}_{}.npz'.format(atlas, measure)))
            ys = d['data']
        else:
            import scipy.io as sio
            d_sim = sio.loadmat(op.join(fol, 'fmri_timecourse_sim.mat'))
            ys = d_sim['timecourse_use_sim'].T
        if labels_names is None:
            labels_names = d['names']
            labels_ids = range(len(labels_names))
        else:
            ys = ys[labels_ids, :]

    hs_tr = np.array(hs_s) * 1000 / tr
    t_axis = np.arange(0, tr / 1000 * ys.shape[1], tr / 1000)
    for k_type in k_types:
        if only_one_trace:
            plot_mean_var(y, hs_s, fol, k_type=k_type)
        else:
            print(subject, sms, run)
            # est_mean_and_var(ys, labels_names, hs_tr, hs_s, t_axis, fol, k_type, sim, overwrite=overwrite,
            #                  specific_label=specific_label, n_jobs=n_jobs)
            # calc_mean_var_cl(ys, fol, hs_tr, hs_s, labels_names, k_type=k_type, sim=sim, overwrite=overwrite,
            #                  specific_label=specific_label, n_jobs=n_jobs)
            # plot_mean_var_cl(fol, root_fol, subject, sms, run, labels_names, k_type, sim)

            # est_vector_mean_and_var(ys, labels_names, hs_tr, hs_s, fol, k_type, sim, overwrite=overwrite, n_jobs=n_jobs)
            # calc_vector_mean_cl(ys, fol, hs_tr, k_type, sim, overwrite=False, n_jobs=1)
            calc_vector_mean_cov_cl(ys, fol, hs_tr, k_type, sim, overwrite=False, n_jobs=1)
            # plot_vector_mean_var_cl(fol, root_fol, subject, sms, run, k_type, sim)


if __name__ == '__main__':
    # subject = 'nmr00956'
    fsaverage = 'fsaverage'
    root_fol = utils.existing_fol(
        ['/home/noam/vic', '/space/violet/1/neuromind/dwakeman/sequence_analysis/sms_study_bay8/raw/func', '/homes/5/npeled/space1/vic'])
    root_fol = '/homes/5/npeled/space1/vic'
    hemi = 'lh'
    atlas = 'aparc' # 'laus250'
    measure = 'PCA'
    # hs_plot = [8,  13,  18, 23]
    # hs_plot = [4, 6, 12, 22]
    hs = range(1, 59)
    hs_to_compare = (1, 59) # (1, 450)
    top_hs = 59

    hs = range(1, 450)
    hs_to_compare = (1, 450) # (1, 450)
    top_hs = 120

    # hs_plot = [5, 10, 15, 25]
    # hs = [8,  13,  18, 23]
    k_types = ['triangular'] # 'Epanechnikov', 'tricube'
    only_one_trace = False
    # legend_index = 1
    # labels_names = ['posteriorcingulate-lh']
    # figures_fol = op.join(root_fol, 'figures', 'smss_per_label_window')
    # utils.make_dir(figures_fol)
    overwrite = False
    sim = True
    n_jobs = -1
    n_jobs = utils.get_n_jobs(n_jobs)
    specific_label = 'posteriorcingulate-lh'
    labels_names = utils.load(op.join(root_fol, 'labels_names.pkl'))
    labels_ids = range(len(labels_names))

    specific_label = 'posteriorcingulate-lh'
    # for sim in [False, True]:
    subjects = set()
    for fol, subject, sms, run in utils.sms_generator(root_fol):
        if subject != 'nmr00956':
            continue
        subjects.add(subject)
        fmri_fname = op.join(fol, 'fmcpr.sm5.{}.{}.mgz'.format(fsaverage, hemi))
        tr = utils.load(op.join(fol, 'tr.pkl'))
        print(subject, sms, run, tr)
        main(subject, sms, run, fmri_fname, fol, root_fol, atlas, tr, hs, k_types, measure, sim, labels_names,
             labels_ids, only_one_trace, overwrite, specific_label, n_jobs=n_jobs)
        # compare_vector_mean_var_cl(subject, sms, run, fol, root_fol, k_types)

    label = 'posteriorcingulate-lh' # 'fusiform-lh'
    labels_ids = np.where(labels_names == label)[0]
    labels_names = [label]
    init_figures()
    for fol, subject, sms, run in utils.sms_generator(root_fol):
        if subject != 'nmr00956':
            continue
        # merge_mean_var_cl_files(fol, hs_to_compare, k_types[0])
        # compare_mean_var_cl(fol, root_fol, subject, sms, run, labels_names, hs_to_compare, top_hs, labels_ids,
        #                     hs, k_types[0])
        pass


    for sim in [True, False]:
        # plot_mean_var(k_types[0], sim, labels_names, labels_ids)
        pass

    for subject in subjects:
        # combine_mean_var_sim_plots(root_fol, labels_names)
        # combine_mean_var_cl_sim_plots(root_fol, subject, labels_names, hs_to_compare)
        pass

    print('Finish!')

