/*
 * Copyright 2010      INRIA Saclay
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, INRIA Saclay - Ile-de-France,
 * Parc Club Orsay Universite, ZAC des vignes, 4 rue Jacques Monod,
 * 91893 Orsay, France 
 */

#include <isl/aff.h>
#include <isl/val.h>
#include <isl_ctx_private.h>
#include <isl_map_private.h>
#include <isl_bound.h>
#include <isl_bernstein.h>
#include <isl_range.h>
#include <isl_polynomial_private.h>
#include <isl_options_private.h>

/* Given a polynomial "poly" that is constant in terms
 * of the domain variables, construct a polynomial reduction
 * of type "type" that is equal to "poly" on "bset",
 * with the domain projected onto the parameters.
 */
__isl_give isl_pw_qpolynomial_fold *isl_qpolynomial_cst_bound(
	__isl_take isl_basic_set *bset, __isl_take isl_qpolynomial *poly,
	enum isl_fold type, isl_bool *tight)
{
	isl_set *dom;
	isl_qpolynomial_fold *fold;
	isl_pw_qpolynomial_fold *pwf;

	fold = isl_qpolynomial_fold_alloc(type, poly);
	dom = isl_set_from_basic_set(bset);
	if (tight)
		*tight = isl_bool_true;
	pwf = isl_pw_qpolynomial_fold_alloc(type, dom, fold);
	return isl_pw_qpolynomial_fold_project_domain_on_params(pwf);
}

/* Add the bound "pwf", which is not known to be tight,
 * to the output of "bound".
 */
isl_stat isl_bound_add(struct isl_bound *bound,
	__isl_take isl_pw_qpolynomial_fold *pwf)
{
	bound->pwf = isl_pw_qpolynomial_fold_fold(bound->pwf, pwf);
	return isl_stat_non_null(bound->pwf);
}

/* Add the bound "pwf", which is known to be tight,
 * to the output of "bound".
 */
isl_stat isl_bound_add_tight(struct isl_bound *bound,
	__isl_take isl_pw_qpolynomial_fold *pwf)
{
	bound->pwf_tight = isl_pw_qpolynomial_fold_fold(bound->pwf_tight, pwf);
	return isl_stat_non_null(bound->pwf);
}

/* Given a polynomial "poly" that is constant in terms
 * of the domain variables and the domain "bset",
 * construct the corresponding polynomial reduction and
 * add it to the tight bounds of "bound".
 */
static isl_stat add_constant_poly(__isl_take isl_basic_set *bset,
	__isl_take isl_qpolynomial *poly, struct isl_bound *bound)
{
	isl_pw_qpolynomial_fold *pwf;

	pwf = isl_qpolynomial_cst_bound(bset, poly, bound->type, NULL);
	return isl_bound_add_tight(bound, pwf);
}

/* Compute a bound on the polynomial defined over the parametric polytope
 * using either range propagation or bernstein expansion and
 * store the result in bound->pwf and bound->pwf_tight.
 * Since bernstein expansion requires bounded domains, we apply
 * range propagation on unbounded domains.  Otherwise, we respect the choice
 * of the user.
 *
 * If the polynomial does not depend on the set variables
 * then the bound is equal to the polynomial and
 * it can be added to "bound" directly.
 */
static isl_stat compressed_guarded_poly_bound(__isl_take isl_basic_set *bset,
	__isl_take isl_qpolynomial *poly, struct isl_bound *bound)
{
	isl_ctx *ctx;
	int bounded;
	int degree;

	if (!bset || !poly)
		goto error;

	degree = isl_qpolynomial_degree(poly);
	if (degree < -1)
		goto error;
	if (degree <= 0)
		return add_constant_poly(bset, poly, bound);

	ctx = isl_basic_set_get_ctx(bset);
	if (ctx->opt->bound == ISL_BOUND_RANGE)
		return isl_qpolynomial_bound_on_domain_range(bset, poly, bound);

	bounded = isl_basic_set_is_bounded(bset);
	if (bounded < 0)
		goto error;
	if (bounded)
		return isl_qpolynomial_bound_on_domain_bernstein(bset, poly, bound);
	else
		return isl_qpolynomial_bound_on_domain_range(bset, poly, bound);
error:
	isl_basic_set_free(bset);
	isl_qpolynomial_free(poly);
	return isl_stat_error;
}

static isl_stat unwrapped_guarded_poly_bound(__isl_take isl_basic_set *bset,
	__isl_take isl_qpolynomial *poly, struct isl_bound *bound)
{
	isl_pw_qpolynomial_fold *top_pwf;
	isl_pw_qpolynomial_fold *top_pwf_tight;
	isl_space *space;
	isl_morph *morph;
	isl_stat r;

	bset = isl_basic_set_detect_equalities(bset);

	if (!bset)
		goto error;

	if (bset->n_eq == 0)
		return compressed_guarded_poly_bound(bset, poly, bound);

	morph = isl_basic_set_full_compression(bset);

	bset = isl_morph_basic_set(isl_morph_copy(morph), bset);
	poly = isl_qpolynomial_morph_domain(poly, isl_morph_copy(morph));

	space = isl_morph_get_ran_space(morph);
	space = isl_space_params(space);

	top_pwf = bound->pwf;
	top_pwf_tight = bound->pwf_tight;

	space = isl_space_from_domain(space);
	space = isl_space_add_dims(space, isl_dim_out, 1);
	bound->pwf = isl_pw_qpolynomial_fold_zero(isl_space_copy(space),
						  bound->type);
	bound->pwf_tight = isl_pw_qpolynomial_fold_zero(space, bound->type);

	r = compressed_guarded_poly_bound(bset, poly, bound);

	morph = isl_morph_dom_params(morph);
	morph = isl_morph_ran_params(morph);
	morph = isl_morph_inverse(morph);

	bound->pwf = isl_pw_qpolynomial_fold_morph_domain(bound->pwf,
							isl_morph_copy(morph));
	bound->pwf_tight = isl_pw_qpolynomial_fold_morph_domain(
						bound->pwf_tight, morph);

	isl_bound_add(bound, top_pwf);
	isl_bound_add_tight(bound, top_pwf_tight);

	return r;
error:
	isl_basic_set_free(bset);
	isl_qpolynomial_free(poly);
	return isl_stat_error;
}

/* Update bound->pwf and bound->pwf_tight with a bound
 * of type bound->type on the (quasi-)polynomial "qp" over the domain "bset",
 * by calling "unwrapped" on unwrapped versions of "bset and "qp".
 * If "qp" is a polynomial, then "unwrapped" will also be called
 * on a polynomial.
 *
 * If the original problem did not have a wrapped relation in the domain,
 * then call "unwrapped" directly.
 *
 * Otherwise, the bound should be computed over the range
 * of the wrapped relation.  Temporarily treat the domain dimensions
 * of this wrapped relation as parameters, compute a bound using "unwrapped"
 * in terms of these and the original parameters,
 * turn the parameters back into set dimensions and
 * add the results to bound->pwf and bound->pwf_tight.
 *
 * Note that even though "bset" is known to live in the same space
 * as the domain of "qp", the names of the set dimensions
 * may be different (or missing).  Make sure the naming is exactly
 * the same before turning these dimensions into parameters
 * to ensure that the spaces are still the same after
 * this operation.
 */
static isl_stat unwrap(__isl_take isl_basic_set *bset,
	__isl_take isl_qpolynomial *qp,
	isl_stat (*unwrapped)(__isl_take isl_basic_set *bset,
		__isl_take isl_qpolynomial *qp, struct isl_bound *bound),
	struct isl_bound *bound)
{
	isl_space *space;
	isl_pw_qpolynomial_fold *top_pwf;
	isl_pw_qpolynomial_fold *top_pwf_tight;
	isl_size nparam;
	isl_size n_in;
	isl_stat r;

	if (!bound->wrapping)
		return unwrapped(bset, qp, bound);

	nparam = isl_space_dim(bound->dim, isl_dim_param);
	n_in = isl_space_dim(bound->dim, isl_dim_in);
	if (nparam < 0 || n_in < 0)
		goto error;

	space = isl_qpolynomial_get_domain_space(qp);
	bset = isl_basic_set_reset_space(bset, space);

	bset = isl_basic_set_move_dims(bset, isl_dim_param, nparam,
					isl_dim_set, 0, n_in);
	qp = isl_qpolynomial_move_dims(qp, isl_dim_param, nparam,
					isl_dim_in, 0, n_in);

	space = isl_basic_set_get_space(bset);
	space = isl_space_params(space);

	top_pwf = bound->pwf;
	top_pwf_tight = bound->pwf_tight;

	space = isl_space_from_domain(space);
	space = isl_space_add_dims(space, isl_dim_out, 1);
	bound->pwf = isl_pw_qpolynomial_fold_zero(isl_space_copy(space),
						  bound->type);
	bound->pwf_tight = isl_pw_qpolynomial_fold_zero(space, bound->type);

	r = unwrapped(bset, qp, bound);

	bound->pwf = isl_pw_qpolynomial_fold_reset_space(bound->pwf,
						    isl_space_copy(bound->dim));
	bound->pwf_tight = isl_pw_qpolynomial_fold_reset_space(bound->pwf_tight,
						    isl_space_copy(bound->dim));

	isl_bound_add(bound, top_pwf);
	isl_bound_add_tight(bound, top_pwf_tight);

	return r;
error:
	isl_basic_set_free(bset);
	isl_qpolynomial_free(qp);
	return isl_stat_error;
}

/* Update bound->pwf and bound->pwf_tight with a bound
 * of type bound->type on the polynomial "poly" over the domain "bset",
 * handling any wrapping in the domain.
 */
static isl_stat guarded_poly_bound(__isl_take isl_basic_set *bset,
	__isl_take isl_qpolynomial *poly, void *user)
{
	struct isl_bound *bound = (struct isl_bound *)user;

	return unwrap(bset, poly, &unwrapped_guarded_poly_bound, bound);
}

/* Is "bset" bounded and is "qp" a quasi-affine expression?
 */
static isl_bool is_bounded_affine(__isl_keep isl_basic_set *bset,
	__isl_keep isl_qpolynomial *qp)
{
	isl_bool affine;

	affine = isl_qpolynomial_isa_aff(qp);
	if (affine < 0 || !affine)
		return affine;
	return isl_basic_set_is_bounded(bset);
}

/* Update bound->pwf and bound->pwf_tight with a bound
 * of type bound->type on the quasi-polynomial "qp" over the domain "bset",
 * for the case where "bset" is bounded and
 * "qp" is a quasi-affine expression and
 * they have both been unwrapped already if needed.
 *
 * Consider the set of possible function values of "qp" over "bset" and
 * take the minimum or maximum value in this set, depending
 * on whether a lower or an upper bound is being computed.
 * Do this by calling isl_set_lexmin_pw_multi_aff or
 * isl_set_lexmax_pw_multi_aff, which compute a regular minimum or maximum
 * since the set is one-dimensional.
 * Since this computation is exact, the bound is always tight.
 *
 * Note that the minimum or maximum integer value is being computed,
 * so if "qp" has some non-trivial denominator, then it needs
 * to be multiplied out first and then taken into account again
 * after computing the minimum or maximum.
 */
static isl_stat unwrapped_affine_qp(__isl_take isl_basic_set *bset,
	__isl_take isl_qpolynomial *qp, struct isl_bound *bound)
{
	isl_val *d;
	isl_aff *aff;
	isl_basic_map *bmap;
	isl_set *range;
	isl_pw_multi_aff *opt;
	isl_pw_aff *pa;
	isl_pw_qpolynomial *pwqp;
	isl_pw_qpolynomial_fold *pwf;

	aff = isl_qpolynomial_as_aff(qp);
	d = isl_aff_get_denominator_val(aff);
	aff = isl_aff_scale_val(aff, isl_val_copy(d));
	bmap = isl_basic_map_from_aff(aff);
	bmap = isl_basic_map_intersect_domain(bmap, bset);
	range = isl_set_from_basic_set(isl_basic_map_range(bmap));
	if (bound->type == isl_fold_min)
		opt = isl_set_lexmin_pw_multi_aff(range);
	else
		opt = isl_set_lexmax_pw_multi_aff(range);
	pa = isl_pw_multi_aff_get_at(opt, 0);
	isl_pw_multi_aff_free(opt);
	pa = isl_pw_aff_scale_down_val(pa, d);
	pwqp = isl_pw_qpolynomial_from_pw_aff(pa);
	pwf = isl_pw_qpolynomial_fold_from_pw_qpolynomial(bound->type, pwqp);

	bound->pwf_tight = isl_pw_qpolynomial_fold_fold(bound->pwf_tight, pwf);

	return isl_stat_non_null(bound->pwf_tight);
}

/* Update bound->pwf and bound->pwf_tight with a bound
 * of type bound->type on the quasi-polynomial "qp" over the domain bound->bset,
 * for the case where bound->bset is bounded and
 * "qp" is a quasi-affine expression,
 * handling any wrapping in the domain.
 */
static isl_stat affine_qp(__isl_take isl_qpolynomial *qp,
	struct isl_bound *bound)
{
	isl_basic_set *bset;

	bset = isl_basic_set_copy(bound->bset);
	return unwrap(bset, qp, &unwrapped_affine_qp, bound);
}

/* Update bound->pwf and bound->pwf_tight with a bound
 * of type bound->type on the quasi-polynomial "qp" over the domain bound->bset.
 *
 * If bound->bset is bounded and if "qp" is a quasi-affine expression,
 * then use a specialized version.
 *
 * Otherwise, treat the integer divisions as extra variables and
 * compute a bound over the polynomial in terms of the original and
 * the extra variables.
 */
static isl_stat guarded_qp(__isl_take isl_qpolynomial *qp, void *user)
{
	struct isl_bound *bound = (struct isl_bound *)user;
	isl_stat r;
	isl_bool bounded_affine;

	bounded_affine = is_bounded_affine(bound->bset, qp);
	if (bounded_affine < 0)
		qp = isl_qpolynomial_free(qp);
	else if (bounded_affine)
		return affine_qp(qp, bound);

	r = isl_qpolynomial_as_polynomial_on_domain(qp, bound->bset,
						    &guarded_poly_bound, user);
	isl_qpolynomial_free(qp);
	return r;
}

static isl_stat basic_guarded_fold(__isl_take isl_basic_set *bset, void *user)
{
	struct isl_bound *bound = (struct isl_bound *)user;
	isl_stat r;

	bound->bset = bset;
	r = isl_qpolynomial_fold_foreach_qpolynomial(bound->fold,
							&guarded_qp, user);
	isl_basic_set_free(bset);
	return r;
}

static isl_stat guarded_fold(__isl_take isl_set *set,
	__isl_take isl_qpolynomial_fold *fold, void *user)
{
	struct isl_bound *bound = (struct isl_bound *)user;

	if (!set || !fold)
		goto error;

	set = isl_set_make_disjoint(set);

	bound->fold = fold;
	bound->type = isl_qpolynomial_fold_get_type(fold);

	if (isl_set_foreach_basic_set(set, &basic_guarded_fold, bound) < 0)
		goto error;

	isl_set_free(set);
	isl_qpolynomial_fold_free(fold);

	return isl_stat_ok;
error:
	isl_set_free(set);
	isl_qpolynomial_fold_free(fold);
	return isl_stat_error;
}

__isl_give isl_pw_qpolynomial_fold *isl_pw_qpolynomial_fold_bound(
	__isl_take isl_pw_qpolynomial_fold *pwf, isl_bool *tight)
{
	isl_size nvar;
	struct isl_bound bound;
	isl_bool covers;

	if (!pwf)
		return NULL;

	bound.dim = isl_pw_qpolynomial_fold_get_domain_space(pwf);

	bound.wrapping = isl_space_is_wrapping(bound.dim);
	if (bound.wrapping)
		bound.dim = isl_space_unwrap(bound.dim);
	nvar = isl_space_dim(bound.dim, isl_dim_out);
	if (nvar < 0)
		bound.dim = isl_space_free(bound.dim);
	bound.dim = isl_space_domain(bound.dim);
	bound.dim = isl_space_from_domain(bound.dim);
	bound.dim = isl_space_add_dims(bound.dim, isl_dim_out, 1);

	if (nvar == 0) {
		if (tight)
			*tight = isl_bool_true;
		return isl_pw_qpolynomial_fold_reset_space(pwf, bound.dim);
	}

	if (isl_pw_qpolynomial_fold_is_zero(pwf)) {
		enum isl_fold type = pwf->type;
		isl_pw_qpolynomial_fold_free(pwf);
		if (tight)
			*tight = isl_bool_true;
		return isl_pw_qpolynomial_fold_zero(bound.dim, type);
	}

	bound.pwf = isl_pw_qpolynomial_fold_zero(isl_space_copy(bound.dim),
							pwf->type);
	bound.pwf_tight = isl_pw_qpolynomial_fold_zero(isl_space_copy(bound.dim),
							pwf->type);
	bound.check_tight = !!tight;

	if (isl_pw_qpolynomial_fold_foreach_lifted_piece(pwf,
							guarded_fold, &bound) < 0)
		goto error;

	covers = isl_pw_qpolynomial_fold_covers(bound.pwf_tight, bound.pwf);
	if (covers < 0)
		goto error;

	if (tight)
		*tight = covers;

	isl_space_free(bound.dim);
	isl_pw_qpolynomial_fold_free(pwf);

	if (covers) {
		isl_pw_qpolynomial_fold_free(bound.pwf);
		return bound.pwf_tight;
	}

	bound.pwf = isl_pw_qpolynomial_fold_fold(bound.pwf, bound.pwf_tight);

	return bound.pwf;
error:
	isl_pw_qpolynomial_fold_free(bound.pwf_tight);
	isl_pw_qpolynomial_fold_free(bound.pwf);
	isl_pw_qpolynomial_fold_free(pwf);
	isl_space_free(bound.dim);
	return NULL;
}

__isl_give isl_pw_qpolynomial_fold *isl_pw_qpolynomial_bound(
	__isl_take isl_pw_qpolynomial *pwqp, enum isl_fold type,
	isl_bool *tight)
{
	isl_pw_qpolynomial_fold *pwf;

	pwf = isl_pw_qpolynomial_fold_from_pw_qpolynomial(type, pwqp);
	return isl_pw_qpolynomial_fold_bound(pwf, tight);
}

struct isl_union_bound_data {
	enum isl_fold type;
	isl_bool tight;
	isl_union_pw_qpolynomial_fold *res;
};

static isl_stat bound_pw(__isl_take isl_pw_qpolynomial *pwqp, void *user)
{
	struct isl_union_bound_data *data = user;
	isl_pw_qpolynomial_fold *pwf;

	pwf = isl_pw_qpolynomial_bound(pwqp, data->type,
					data->tight ? &data->tight : NULL);
	data->res = isl_union_pw_qpolynomial_fold_fold_pw_qpolynomial_fold(
								data->res, pwf);

	return isl_stat_ok;
}

__isl_give isl_union_pw_qpolynomial_fold *isl_union_pw_qpolynomial_bound(
	__isl_take isl_union_pw_qpolynomial *upwqp,
	enum isl_fold type, isl_bool *tight)
{
	isl_space *space;
	struct isl_union_bound_data data = { type, 1, NULL };

	if (!upwqp)
		return NULL;

	if (!tight)
		data.tight = isl_bool_false;

	space = isl_union_pw_qpolynomial_get_space(upwqp);
	data.res = isl_union_pw_qpolynomial_fold_zero(space, type);
	if (isl_union_pw_qpolynomial_foreach_pw_qpolynomial(upwqp,
						    &bound_pw, &data) < 0)
		goto error;

	isl_union_pw_qpolynomial_free(upwqp);
	if (tight)
		*tight = data.tight;

	return data.res;
error:
	isl_union_pw_qpolynomial_free(upwqp);
	isl_union_pw_qpolynomial_fold_free(data.res);
	return NULL;
}
