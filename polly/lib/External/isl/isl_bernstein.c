/*
 * Copyright 2006-2007 Universiteit Leiden
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 * Copyright 2010      INRIA Saclay
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, Leiden Institute of Advanced Computer Science,
 * Universiteit Leiden, Niels Bohrweg 1, 2333 CA Leiden, The Netherlands
 * and K.U.Leuven, Departement Computerwetenschappen, Celestijnenlaan 200A,
 * B-3001 Leuven, Belgium
 * and INRIA Saclay - Ile-de-France, Parc Club Orsay Universite,
 * ZAC des vignes, 4 rue Jacques Monod, 91893 Orsay, France
 */

#include <isl_ctx_private.h>
#include <isl_map_private.h>
#include <isl/set.h>
#include <isl_seq.h>
#include <isl_morph.h>
#include <isl_factorization.h>
#include <isl_vertices_private.h>
#include <isl_polynomial_private.h>
#include <isl_options_private.h>
#include <isl_vec_private.h>
#include <isl_bernstein.h>

struct bernstein_data {
	enum isl_fold type;
	isl_qpolynomial *poly;
	int check_tight;

	isl_cell *cell;

	isl_qpolynomial_fold *fold;
	isl_qpolynomial_fold *fold_tight;
	isl_pw_qpolynomial_fold *pwf;
	isl_pw_qpolynomial_fold *pwf_tight;
};

static isl_bool vertex_is_integral(__isl_keep isl_basic_set *vertex)
{
	isl_size nvar;
	isl_size nparam;
	int i;

	nvar = isl_basic_set_dim(vertex, isl_dim_set);
	nparam = isl_basic_set_dim(vertex, isl_dim_param);
	if (nvar < 0 || nparam < 0)
		return isl_bool_error;
	for (i = 0; i < nvar; ++i) {
		int r = nvar - 1 - i;
		if (!isl_int_is_one(vertex->eq[r][1 + nparam + i]) &&
		    !isl_int_is_negone(vertex->eq[r][1 + nparam + i]))
			return isl_bool_false;
	}

	return isl_bool_true;
}

static __isl_give isl_qpolynomial *vertex_coordinate(
	__isl_keep isl_basic_set *vertex, int i, __isl_take isl_space *space)
{
	isl_size nvar;
	isl_size nparam;
	isl_size total;
	int r;
	isl_int denom;
	isl_qpolynomial *v;

	isl_int_init(denom);

	nvar = isl_basic_set_dim(vertex, isl_dim_set);
	nparam = isl_basic_set_dim(vertex, isl_dim_param);
	total = isl_basic_set_dim(vertex, isl_dim_all);
	if (nvar < 0 || nparam < 0 || total < 0)
		goto error;
	r = nvar - 1 - i;

	isl_int_set(denom, vertex->eq[r][1 + nparam + i]);
	isl_assert(vertex->ctx, !isl_int_is_zero(denom), goto error);

	if (isl_int_is_pos(denom))
		isl_seq_neg(vertex->eq[r], vertex->eq[r], 1 + total);
	else
		isl_int_neg(denom, denom);

	v = isl_qpolynomial_from_affine(space, vertex->eq[r], denom);
	isl_int_clear(denom);

	return v;
error:
	isl_space_free(space);
	isl_int_clear(denom);
	return NULL;
}

/* Check whether the bound associated to the selection "k" is tight,
 * which is the case if we select exactly one vertex (i.e., one of the
 * exponents in "k" is exactly "d") and if that vertex
 * is integral for all values of the parameters.
 *
 * If the degree "d" is zero, then there are no exponents.
 * Since the polynomial is a constant expression in this case,
 * the bound is necessarily tight.
 */
static isl_bool is_tight(int *k, int n, int d, isl_cell *cell)
{
	int i;

	if (d == 0)
		return isl_bool_true;

	for (i = 0; i < n; ++i) {
		int v;
		if (!k[i])
			continue;
		if (k[i] != d)
			return isl_bool_false;
		v = cell->ids[n - 1 - i];
		return vertex_is_integral(cell->vertices->v[v].vertex);
	}

	return isl_bool_false;
}

static isl_stat add_fold(__isl_take isl_qpolynomial *b, __isl_keep isl_set *dom,
	int *k, int n, int d, struct bernstein_data *data)
{
	isl_qpolynomial_fold *fold;
	isl_bool tight;

	fold = isl_qpolynomial_fold_alloc(data->type, b);

	tight = isl_bool_false;
	if (data->check_tight)
		tight = is_tight(k, n, d, data->cell);
	if (tight < 0)
		return isl_stat_error;
	if (tight)
		data->fold_tight = isl_qpolynomial_fold_fold_on_domain(dom,
							data->fold_tight, fold);
	else
		data->fold = isl_qpolynomial_fold_fold_on_domain(dom,
							data->fold, fold);
	return isl_stat_ok;
}

/* Extract the coefficients of the Bernstein base polynomials and store
 * them in data->fold and data->fold_tight.
 *
 * In particular, the coefficient of each monomial
 * of multi-degree (k[0], k[1], ..., k[n-1]) is divided by the corresponding
 * multinomial coefficient d!/k[0]! k[1]! ... k[n-1]!
 *
 * c[i] contains the coefficient of the selected powers of the first i+1 vars.
 * multinom[i] contains the partial multinomial coefficient.
 */
static isl_stat extract_coefficients(isl_qpolynomial *poly,
	__isl_keep isl_set *dom, struct bernstein_data *data)
{
	int i;
	int d;
	isl_size n;
	isl_ctx *ctx;
	isl_qpolynomial **c = NULL;
	int *k = NULL;
	int *left = NULL;
	isl_vec *multinom = NULL;

	n = isl_qpolynomial_dim(poly, isl_dim_in);
	if (n < 0)
		return isl_stat_error;

	ctx = isl_qpolynomial_get_ctx(poly);
	d = isl_qpolynomial_degree(poly);
	isl_assert(ctx, n >= 2, return isl_stat_error);

	c = isl_calloc_array(ctx, isl_qpolynomial *, n);
	k = isl_alloc_array(ctx, int, n);
	left = isl_alloc_array(ctx, int, n);
	multinom = isl_vec_alloc(ctx, n);
	if (!c || !k || !left || !multinom)
		goto error;

	isl_int_set_si(multinom->el[0], 1);
	for (k[0] = d; k[0] >= 0; --k[0]) {
		int i = 1;
		isl_qpolynomial_free(c[0]);
		c[0] = isl_qpolynomial_coeff(poly, isl_dim_in, n - 1, k[0]);
		left[0] = d - k[0];
		k[1] = -1;
		isl_int_set(multinom->el[1], multinom->el[0]);
		while (i > 0) {
			if (i == n - 1) {
				int j;
				isl_space *space;
				isl_qpolynomial *b;
				isl_qpolynomial *f;
				for (j = 2; j <= left[i - 1]; ++j)
					isl_int_divexact_ui(multinom->el[i],
						multinom->el[i], j);
				b = isl_qpolynomial_coeff(c[i - 1], isl_dim_in,
					n - 1 - i, left[i - 1]);
				b = isl_qpolynomial_project_domain_on_params(b);
				space = isl_qpolynomial_get_domain_space(b);
				f = isl_qpolynomial_rat_cst_on_domain(space,
					ctx->one, multinom->el[i]);
				b = isl_qpolynomial_mul(b, f);
				k[n - 1] = left[n - 2];
				if (add_fold(b, dom, k, n, d, data) < 0)
					goto error;
				--i;
				continue;
			}
			if (k[i] >= left[i - 1]) {
				--i;
				continue;
			}
			++k[i];
			if (k[i])
				isl_int_divexact_ui(multinom->el[i],
					multinom->el[i], k[i]);
			isl_qpolynomial_free(c[i]);
			c[i] = isl_qpolynomial_coeff(c[i - 1], isl_dim_in,
					n - 1 - i, k[i]);
			left[i] = left[i - 1] - k[i];
			k[i + 1] = -1;
			isl_int_set(multinom->el[i + 1], multinom->el[i]);
			++i;
		}
		isl_int_mul_ui(multinom->el[0], multinom->el[0], k[0]);
	}

	for (i = 0; i < n; ++i)
		isl_qpolynomial_free(c[i]);

	isl_vec_free(multinom);
	free(left);
	free(k);
	free(c);
	return isl_stat_ok;
error:
	isl_vec_free(multinom);
	free(left);
	free(k);
	if (c)
		for (i = 0; i < n; ++i)
			isl_qpolynomial_free(c[i]);
	free(c);
	return isl_stat_error;
}

/* Perform bernstein expansion on the parametric vertices that are active
 * on "cell".
 *
 * data->poly has been homogenized in the calling function.
 *
 * We plug in the barycentric coordinates for the set variables
 *
 *		\vec x = \sum_i \alpha_i v_i(\vec p)
 *
 * and the constant "1 = \sum_i \alpha_i" for the homogeneous dimension.
 * Next, we extract the coefficients of the Bernstein base polynomials.
 */
static isl_stat bernstein_coefficients_cell(__isl_take isl_cell *cell,
	void *user)
{
	int i, j;
	struct bernstein_data *data = (struct bernstein_data *)user;
	isl_space *space_param;
	isl_space *space_dst;
	isl_qpolynomial *poly = data->poly;
	isl_size n_in;
	unsigned nvar;
	int n_vertices;
	isl_qpolynomial **subs;
	isl_pw_qpolynomial_fold *pwf;
	isl_set *dom;
	isl_ctx *ctx;

	n_in = isl_qpolynomial_dim(poly, isl_dim_in);
	if (n_in < 0)
		goto error;

	nvar = n_in - 1;
	n_vertices = cell->n_vertices;

	ctx = isl_qpolynomial_get_ctx(poly);
	if (n_vertices > nvar + 1 && ctx->opt->bernstein_triangulate)
		return isl_cell_foreach_simplex(cell,
					    &bernstein_coefficients_cell, user);

	subs = isl_alloc_array(ctx, isl_qpolynomial *, 1 + nvar);
	if (!subs)
		goto error;

	space_param = isl_basic_set_get_space(cell->dom);
	space_dst = isl_qpolynomial_get_domain_space(poly);
	space_dst = isl_space_add_dims(space_dst, isl_dim_set, n_vertices);

	for (i = 0; i < 1 + nvar; ++i)
		subs[i] =
		    isl_qpolynomial_zero_on_domain(isl_space_copy(space_dst));

	for (i = 0; i < n_vertices; ++i) {
		isl_qpolynomial *c;
		c = isl_qpolynomial_var_on_domain(isl_space_copy(space_dst),
					isl_dim_set, 1 + nvar + i);
		for (j = 0; j < nvar; ++j) {
			int k = cell->ids[i];
			isl_qpolynomial *v;
			v = vertex_coordinate(cell->vertices->v[k].vertex, j,
						isl_space_copy(space_param));
			v = isl_qpolynomial_add_dims(v, isl_dim_in,
							1 + nvar + n_vertices);
			v = isl_qpolynomial_mul(v, isl_qpolynomial_copy(c));
			subs[1 + j] = isl_qpolynomial_add(subs[1 + j], v);
		}
		subs[0] = isl_qpolynomial_add(subs[0], c);
	}
	isl_space_free(space_dst);

	poly = isl_qpolynomial_copy(poly);

	poly = isl_qpolynomial_add_dims(poly, isl_dim_in, n_vertices);
	poly = isl_qpolynomial_substitute(poly, isl_dim_in, 0, 1 + nvar, subs);
	poly = isl_qpolynomial_drop_dims(poly, isl_dim_in, 0, 1 + nvar);

	data->cell = cell;
	dom = isl_set_from_basic_set(isl_basic_set_copy(cell->dom));
	data->fold = isl_qpolynomial_fold_empty(data->type,
						isl_space_copy(space_param));
	data->fold_tight = isl_qpolynomial_fold_empty(data->type, space_param);
	if (extract_coefficients(poly, dom, data) < 0) {
		data->fold = isl_qpolynomial_fold_free(data->fold);
		data->fold_tight = isl_qpolynomial_fold_free(data->fold_tight);
	}

	pwf = isl_pw_qpolynomial_fold_alloc(data->type, isl_set_copy(dom),
					    data->fold);
	data->pwf = isl_pw_qpolynomial_fold_fold(data->pwf, pwf);
	pwf = isl_pw_qpolynomial_fold_alloc(data->type, dom, data->fold_tight);
	data->pwf_tight = isl_pw_qpolynomial_fold_fold(data->pwf_tight, pwf);

	isl_qpolynomial_free(poly);
	isl_cell_free(cell);
	for (i = 0; i < 1 + nvar; ++i)
		isl_qpolynomial_free(subs[i]);
	free(subs);
	return isl_stat_ok;
error:
	isl_cell_free(cell);
	return isl_stat_error;
}

/* Base case of applying bernstein expansion.
 *
 * We compute the chamber decomposition of the parametric polytope "bset"
 * and then perform bernstein expansion on the parametric vertices
 * that are active on each chamber.
 *
 * If the polynomial does not depend on the set variables
 * (and in particular if the number of set variables is zero)
 * then the bound is equal to the polynomial and
 * no actual bernstein expansion needs to be performed.
 */
static __isl_give isl_pw_qpolynomial_fold *bernstein_coefficients_base(
	__isl_take isl_basic_set *bset,
	__isl_take isl_qpolynomial *poly, struct bernstein_data *data,
	isl_bool *tight)
{
	int degree;
	isl_size nvar;
	isl_space *space;
	isl_vertices *vertices;
	isl_bool covers;

	nvar = isl_basic_set_dim(bset, isl_dim_set);
	if (nvar < 0)
		bset = isl_basic_set_free(bset);
	if (nvar == 0)
		return isl_qpolynomial_cst_bound(bset, poly, data->type, tight);

	degree = isl_qpolynomial_degree(poly);
	if (degree < -1)
		bset = isl_basic_set_free(bset);
	if (degree <= 0)
		return isl_qpolynomial_cst_bound(bset, poly, data->type, tight);

	space = isl_basic_set_get_space(bset);
	space = isl_space_params(space);
	space = isl_space_from_domain(space);
	space = isl_space_add_dims(space, isl_dim_set, 1);
	data->pwf = isl_pw_qpolynomial_fold_zero(isl_space_copy(space),
						data->type);
	data->pwf_tight = isl_pw_qpolynomial_fold_zero(space, data->type);
	data->poly = isl_qpolynomial_homogenize(isl_qpolynomial_copy(poly));
	vertices = isl_basic_set_compute_vertices(bset);
	if (isl_vertices_foreach_disjoint_cell(vertices,
					&bernstein_coefficients_cell, data) < 0)
		data->pwf = isl_pw_qpolynomial_fold_free(data->pwf);
	isl_vertices_free(vertices);
	isl_qpolynomial_free(data->poly);

	isl_basic_set_free(bset);
	isl_qpolynomial_free(poly);

	covers = isl_pw_qpolynomial_fold_covers(data->pwf_tight, data->pwf);
	if (covers < 0)
		goto error;

	if (tight)
		*tight = covers;

	if (covers) {
		isl_pw_qpolynomial_fold_free(data->pwf);
		return data->pwf_tight;
	}

	data->pwf = isl_pw_qpolynomial_fold_fold(data->pwf, data->pwf_tight);

	return data->pwf;
error:
	isl_pw_qpolynomial_fold_free(data->pwf_tight);
	isl_pw_qpolynomial_fold_free(data->pwf);
	return NULL;
}

/* Apply bernstein expansion recursively by working in on len[i]
 * set variables at a time, with i ranging from n_group - 1 to 0.
 */
static __isl_give isl_pw_qpolynomial_fold *bernstein_coefficients_recursive(
	__isl_take isl_pw_qpolynomial *pwqp,
	int n_group, int *len, struct bernstein_data *data, isl_bool *tight)
{
	int i;
	isl_size nparam;
	isl_size nvar;
	isl_pw_qpolynomial_fold *pwf;

	nparam = isl_pw_qpolynomial_dim(pwqp, isl_dim_param);
	nvar = isl_pw_qpolynomial_dim(pwqp, isl_dim_in);
	if (nparam < 0 || nvar < 0)
		goto error;

	pwqp = isl_pw_qpolynomial_move_dims(pwqp, isl_dim_param, nparam,
					isl_dim_in, 0, nvar - len[n_group - 1]);
	pwf = isl_pw_qpolynomial_bound(pwqp, data->type, tight);

	for (i = n_group - 2; i >= 0; --i) {
		nparam = isl_pw_qpolynomial_fold_dim(pwf, isl_dim_param);
		if (nparam < 0)
			return isl_pw_qpolynomial_fold_free(pwf);
		pwf = isl_pw_qpolynomial_fold_move_dims(pwf, isl_dim_in, 0,
				isl_dim_param, nparam - len[i], len[i]);
		if (tight && !*tight)
			tight = NULL;
		pwf = isl_pw_qpolynomial_fold_bound(pwf, tight);
	}

	return pwf;
error:
	isl_pw_qpolynomial_free(pwqp);
	return NULL;
}

static __isl_give isl_pw_qpolynomial_fold *bernstein_coefficients_factors(
	__isl_take isl_basic_set *bset,
	__isl_take isl_qpolynomial *poly, struct bernstein_data *data,
	isl_bool *tight)
{
	isl_factorizer *f;
	isl_set *set;
	isl_pw_qpolynomial *pwqp;
	isl_pw_qpolynomial_fold *pwf;

	f = isl_basic_set_factorizer(bset);
	if (!f)
		goto error;
	if (f->n_group == 0) {
		isl_factorizer_free(f);
		return bernstein_coefficients_base(bset, poly, data, tight);
	}

	set = isl_set_from_basic_set(bset);
	pwqp = isl_pw_qpolynomial_alloc(set, poly);
	pwqp = isl_pw_qpolynomial_morph_domain(pwqp, isl_morph_copy(f->morph));

	pwf = bernstein_coefficients_recursive(pwqp, f->n_group, f->len, data,
						tight);

	isl_factorizer_free(f);

	return pwf;
error:
	isl_basic_set_free(bset);
	isl_qpolynomial_free(poly);
	return NULL;
}

static __isl_give isl_pw_qpolynomial_fold *bernstein_coefficients_full_recursive(
	__isl_take isl_basic_set *bset,
	__isl_take isl_qpolynomial *poly, struct bernstein_data *data,
	isl_bool *tight)
{
	int i;
	int *len;
	isl_size nvar;
	isl_pw_qpolynomial_fold *pwf;
	isl_set *set;
	isl_pw_qpolynomial *pwqp;

	nvar = isl_basic_set_dim(bset, isl_dim_set);
	if (nvar < 0 || !poly)
		goto error;
	
	len = isl_alloc_array(bset->ctx, int, nvar);
	if (nvar && !len)
		goto error;

	for (i = 0; i < nvar; ++i)
		len[i] = 1;

	set = isl_set_from_basic_set(bset);
	pwqp = isl_pw_qpolynomial_alloc(set, poly);

	pwf = bernstein_coefficients_recursive(pwqp, nvar, len, data, tight);

	free(len);

	return pwf;
error:
	isl_basic_set_free(bset);
	isl_qpolynomial_free(poly);
	return NULL;
}

/* Compute a bound on the polynomial defined over the parametric polytope
 * using bernstein expansion and store the result
 * in bound->pwf and bound->pwf_tight.
 *
 * If bernstein_recurse is set to ISL_BERNSTEIN_FACTORS, we check if
 * the polytope can be factorized and apply bernstein expansion recursively
 * on the factors.
 * If bernstein_recurse is set to ISL_BERNSTEIN_INTERVALS, we apply
 * bernstein expansion recursively on each dimension.
 * Otherwise, we apply bernstein expansion on the entire polytope.
 */
isl_stat isl_qpolynomial_bound_on_domain_bernstein(
	__isl_take isl_basic_set *bset, __isl_take isl_qpolynomial *poly,
	struct isl_bound *bound)
{
	struct bernstein_data data;
	isl_pw_qpolynomial_fold *pwf;
	isl_size nvar;
	isl_bool tight = isl_bool_false;
	isl_bool *tp = bound->check_tight ? &tight : NULL;

	nvar = isl_basic_set_dim(bset, isl_dim_set);
	if (nvar < 0 || !poly)
		goto error;

	data.type = bound->type;
	data.check_tight = bound->check_tight;

	if (bset->ctx->opt->bernstein_recurse & ISL_BERNSTEIN_FACTORS)
		pwf = bernstein_coefficients_factors(bset, poly, &data, tp);
	else if (nvar > 1 &&
	    (bset->ctx->opt->bernstein_recurse & ISL_BERNSTEIN_INTERVALS))
		pwf = bernstein_coefficients_full_recursive(bset, poly, &data, tp);
	else
		pwf = bernstein_coefficients_base(bset, poly, &data, tp);

	if (tight)
		return isl_bound_add_tight(bound, pwf);
	else
		return isl_bound_add(bound, pwf);
error:
	isl_basic_set_free(bset);
	isl_qpolynomial_free(poly);
	return isl_stat_error;
}
