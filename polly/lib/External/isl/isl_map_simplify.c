/*
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 * Copyright 2012-2013 Ecole Normale Superieure
 * Copyright 2014-2015 INRIA Rocquencourt
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, K.U.Leuven, Departement
 * Computerwetenschappen, Celestijnenlaan 200A, B-3001 Leuven, Belgium
 * and Ecole Normale Superieure, 45 rue d’Ulm, 75230 Paris, France
 * and Inria Paris - Rocquencourt, Domaine de Voluceau - Rocquencourt,
 * B.P. 105 - 78153 Le Chesnay, France
 */

#include <isl_ctx_private.h>
#include <isl_map_private.h>
#include "isl_equalities.h"
#include <isl/map.h>
#include <isl_seq.h>
#include "isl_tab.h"
#include <isl_space_private.h>
#include <isl_mat_private.h>
#include <isl_vec_private.h>

static void swap_equality(struct isl_basic_map *bmap, int a, int b)
{
	isl_int *t = bmap->eq[a];
	bmap->eq[a] = bmap->eq[b];
	bmap->eq[b] = t;
}

static void swap_inequality(struct isl_basic_map *bmap, int a, int b)
{
	if (a != b) {
		isl_int *t = bmap->ineq[a];
		bmap->ineq[a] = bmap->ineq[b];
		bmap->ineq[b] = t;
	}
}

static void constraint_drop_vars(isl_int *c, unsigned n, unsigned rem)
{
	isl_seq_cpy(c, c + n, rem);
	isl_seq_clr(c + rem, n);
}

/* Drop n dimensions starting at first.
 *
 * In principle, this frees up some extra variables as the number
 * of columns remains constant, but we would have to extend
 * the div array too as the number of rows in this array is assumed
 * to be equal to extra.
 */
struct isl_basic_set *isl_basic_set_drop_dims(
		struct isl_basic_set *bset, unsigned first, unsigned n)
{
	int i;

	if (!bset)
		goto error;

	isl_assert(bset->ctx, first + n <= bset->dim->n_out, goto error);

	if (n == 0 && !isl_space_get_tuple_name(bset->dim, isl_dim_set))
		return bset;

	bset = isl_basic_set_cow(bset);
	if (!bset)
		return NULL;

	for (i = 0; i < bset->n_eq; ++i)
		constraint_drop_vars(bset->eq[i]+1+bset->dim->nparam+first, n,
				     (bset->dim->n_out-first-n)+bset->extra);

	for (i = 0; i < bset->n_ineq; ++i)
		constraint_drop_vars(bset->ineq[i]+1+bset->dim->nparam+first, n,
				     (bset->dim->n_out-first-n)+bset->extra);

	for (i = 0; i < bset->n_div; ++i)
		constraint_drop_vars(bset->div[i]+1+1+bset->dim->nparam+first, n,
				     (bset->dim->n_out-first-n)+bset->extra);

	bset->dim = isl_space_drop_outputs(bset->dim, first, n);
	if (!bset->dim)
		goto error;

	ISL_F_CLR(bset, ISL_BASIC_SET_NORMALIZED);
	bset = isl_basic_set_simplify(bset);
	return isl_basic_set_finalize(bset);
error:
	isl_basic_set_free(bset);
	return NULL;
}

struct isl_set *isl_set_drop_dims(
		struct isl_set *set, unsigned first, unsigned n)
{
	int i;

	if (!set)
		goto error;

	isl_assert(set->ctx, first + n <= set->dim->n_out, goto error);

	if (n == 0 && !isl_space_get_tuple_name(set->dim, isl_dim_set))
		return set;
	set = isl_set_cow(set);
	if (!set)
		goto error;
	set->dim = isl_space_drop_outputs(set->dim, first, n);
	if (!set->dim)
		goto error;

	for (i = 0; i < set->n; ++i) {
		set->p[i] = isl_basic_set_drop_dims(set->p[i], first, n);
		if (!set->p[i])
			goto error;
	}

	ISL_F_CLR(set, ISL_SET_NORMALIZED);
	return set;
error:
	isl_set_free(set);
	return NULL;
}

/* Move "n" divs starting at "first" to the end of the list of divs.
 */
static struct isl_basic_map *move_divs_last(struct isl_basic_map *bmap,
	unsigned first, unsigned n)
{
	isl_int **div;
	int i;

	if (first + n == bmap->n_div)
		return bmap;

	div = isl_alloc_array(bmap->ctx, isl_int *, n);
	if (!div)
		goto error;
	for (i = 0; i < n; ++i)
		div[i] = bmap->div[first + i];
	for (i = 0; i < bmap->n_div - first - n; ++i)
		bmap->div[first + i] = bmap->div[first + n + i];
	for (i = 0; i < n; ++i)
		bmap->div[bmap->n_div - n + i] = div[i];
	free(div);
	return bmap;
error:
	isl_basic_map_free(bmap);
	return NULL;
}

/* Drop "n" dimensions of type "type" starting at "first".
 *
 * In principle, this frees up some extra variables as the number
 * of columns remains constant, but we would have to extend
 * the div array too as the number of rows in this array is assumed
 * to be equal to extra.
 */
struct isl_basic_map *isl_basic_map_drop(struct isl_basic_map *bmap,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	int i;
	unsigned dim;
	unsigned offset;
	unsigned left;

	if (!bmap)
		goto error;

	dim = isl_basic_map_dim(bmap, type);
	isl_assert(bmap->ctx, first + n <= dim, goto error);

	if (n == 0 && !isl_space_is_named_or_nested(bmap->dim, type))
		return bmap;

	bmap = isl_basic_map_cow(bmap);
	if (!bmap)
		return NULL;

	offset = isl_basic_map_offset(bmap, type) + first;
	left = isl_basic_map_total_dim(bmap) - (offset - 1) - n;
	for (i = 0; i < bmap->n_eq; ++i)
		constraint_drop_vars(bmap->eq[i]+offset, n, left);

	for (i = 0; i < bmap->n_ineq; ++i)
		constraint_drop_vars(bmap->ineq[i]+offset, n, left);

	for (i = 0; i < bmap->n_div; ++i)
		constraint_drop_vars(bmap->div[i]+1+offset, n, left);

	if (type == isl_dim_div) {
		bmap = move_divs_last(bmap, first, n);
		if (!bmap)
			goto error;
		isl_basic_map_free_div(bmap, n);
	} else
		bmap->dim = isl_space_drop_dims(bmap->dim, type, first, n);
	if (!bmap->dim)
		goto error;

	ISL_F_CLR(bmap, ISL_BASIC_MAP_NORMALIZED);
	bmap = isl_basic_map_simplify(bmap);
	return isl_basic_map_finalize(bmap);
error:
	isl_basic_map_free(bmap);
	return NULL;
}

__isl_give isl_basic_set *isl_basic_set_drop(__isl_take isl_basic_set *bset,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	return (isl_basic_set *)isl_basic_map_drop((isl_basic_map *)bset,
							type, first, n);
}

struct isl_basic_map *isl_basic_map_drop_inputs(
		struct isl_basic_map *bmap, unsigned first, unsigned n)
{
	return isl_basic_map_drop(bmap, isl_dim_in, first, n);
}

struct isl_map *isl_map_drop(struct isl_map *map,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	int i;

	if (!map)
		goto error;

	isl_assert(map->ctx, first + n <= isl_map_dim(map, type), goto error);

	if (n == 0 && !isl_space_get_tuple_name(map->dim, type))
		return map;
	map = isl_map_cow(map);
	if (!map)
		goto error;
	map->dim = isl_space_drop_dims(map->dim, type, first, n);
	if (!map->dim)
		goto error;

	for (i = 0; i < map->n; ++i) {
		map->p[i] = isl_basic_map_drop(map->p[i], type, first, n);
		if (!map->p[i])
			goto error;
	}
	ISL_F_CLR(map, ISL_MAP_NORMALIZED);

	return map;
error:
	isl_map_free(map);
	return NULL;
}

struct isl_set *isl_set_drop(struct isl_set *set,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	return (isl_set *)isl_map_drop((isl_map *)set, type, first, n);
}

struct isl_map *isl_map_drop_inputs(
		struct isl_map *map, unsigned first, unsigned n)
{
	return isl_map_drop(map, isl_dim_in, first, n);
}

/*
 * We don't cow, as the div is assumed to be redundant.
 */
static struct isl_basic_map *isl_basic_map_drop_div(
		struct isl_basic_map *bmap, unsigned div)
{
	int i;
	unsigned pos;

	if (!bmap)
		goto error;

	pos = 1 + isl_space_dim(bmap->dim, isl_dim_all) + div;

	isl_assert(bmap->ctx, div < bmap->n_div, goto error);

	for (i = 0; i < bmap->n_eq; ++i)
		constraint_drop_vars(bmap->eq[i]+pos, 1, bmap->extra-div-1);

	for (i = 0; i < bmap->n_ineq; ++i) {
		if (!isl_int_is_zero(bmap->ineq[i][pos])) {
			isl_basic_map_drop_inequality(bmap, i);
			--i;
			continue;
		}
		constraint_drop_vars(bmap->ineq[i]+pos, 1, bmap->extra-div-1);
	}

	for (i = 0; i < bmap->n_div; ++i)
		constraint_drop_vars(bmap->div[i]+1+pos, 1, bmap->extra-div-1);

	if (div != bmap->n_div - 1) {
		int j;
		isl_int *t = bmap->div[div];

		for (j = div; j < bmap->n_div - 1; ++j)
			bmap->div[j] = bmap->div[j+1];

		bmap->div[bmap->n_div - 1] = t;
	}
	ISL_F_CLR(bmap, ISL_BASIC_MAP_NORMALIZED);
	isl_basic_map_free_div(bmap, 1);

	return bmap;
error:
	isl_basic_map_free(bmap);
	return NULL;
}

struct isl_basic_map *isl_basic_map_normalize_constraints(
	struct isl_basic_map *bmap)
{
	int i;
	isl_int gcd;
	unsigned total = isl_basic_map_total_dim(bmap);

	if (!bmap)
		return NULL;

	isl_int_init(gcd);
	for (i = bmap->n_eq - 1; i >= 0; --i) {
		isl_seq_gcd(bmap->eq[i]+1, total, &gcd);
		if (isl_int_is_zero(gcd)) {
			if (!isl_int_is_zero(bmap->eq[i][0])) {
				bmap = isl_basic_map_set_to_empty(bmap);
				break;
			}
			isl_basic_map_drop_equality(bmap, i);
			continue;
		}
		if (ISL_F_ISSET(bmap, ISL_BASIC_MAP_RATIONAL))
			isl_int_gcd(gcd, gcd, bmap->eq[i][0]);
		if (isl_int_is_one(gcd))
			continue;
		if (!isl_int_is_divisible_by(bmap->eq[i][0], gcd)) {
			bmap = isl_basic_map_set_to_empty(bmap);
			break;
		}
		isl_seq_scale_down(bmap->eq[i], bmap->eq[i], gcd, 1+total);
	}

	for (i = bmap->n_ineq - 1; i >= 0; --i) {
		isl_seq_gcd(bmap->ineq[i]+1, total, &gcd);
		if (isl_int_is_zero(gcd)) {
			if (isl_int_is_neg(bmap->ineq[i][0])) {
				bmap = isl_basic_map_set_to_empty(bmap);
				break;
			}
			isl_basic_map_drop_inequality(bmap, i);
			continue;
		}
		if (ISL_F_ISSET(bmap, ISL_BASIC_MAP_RATIONAL))
			isl_int_gcd(gcd, gcd, bmap->ineq[i][0]);
		if (isl_int_is_one(gcd))
			continue;
		isl_int_fdiv_q(bmap->ineq[i][0], bmap->ineq[i][0], gcd);
		isl_seq_scale_down(bmap->ineq[i]+1, bmap->ineq[i]+1, gcd, total);
	}
	isl_int_clear(gcd);

	return bmap;
}

struct isl_basic_set *isl_basic_set_normalize_constraints(
	struct isl_basic_set *bset)
{
	return (struct isl_basic_set *)isl_basic_map_normalize_constraints(
		(struct isl_basic_map *)bset);
}

/* Assuming the variable at position "pos" has an integer coefficient
 * in integer division "div", extract it from this integer division.
 * "pos" is as determined by isl_basic_map_offset, i.e., pos == 0
 * corresponds to the constant term.
 *
 * That is, the integer division is of the form
 *
 *	floor((... + c * d * x_pos + ...)/d)
 *
 * Replace it by
 *
 *	floor((... + 0 * x_pos + ...)/d) + c * x_pos
 */
static __isl_give isl_basic_map *remove_var_from_div(
	__isl_take isl_basic_map *bmap, int div, int pos)
{
	isl_int shift;

	isl_int_init(shift);
	isl_int_divexact(shift, bmap->div[div][1 + pos], bmap->div[div][0]);
	isl_int_neg(shift, shift);
	bmap = isl_basic_map_shift_div(bmap, div, pos, shift);
	isl_int_clear(shift);

	return bmap;
}

/* Check if integer division "div" has any integral coefficient
 * (or constant term).  If so, extract them from the integer division.
 */
static __isl_give isl_basic_map *remove_independent_vars_from_div(
	__isl_take isl_basic_map *bmap, int div)
{
	int i;
	unsigned total = 1 + isl_basic_map_total_dim(bmap);

	for (i = 0; i < total; ++i) {
		if (isl_int_is_zero(bmap->div[div][1 + i]))
			continue;
		if (!isl_int_is_divisible_by(bmap->div[div][1 + i],
					     bmap->div[div][0]))
			continue;
		bmap = remove_var_from_div(bmap, div, i);
		if (!bmap)
			break;
	}

	return bmap;
}

/* Check if any known integer division has any integral coefficient
 * (or constant term).  If so, extract them from the integer division.
 */
static __isl_give isl_basic_map *remove_independent_vars_from_divs(
	__isl_take isl_basic_map *bmap)
{
	int i;

	if (!bmap)
		return NULL;
	if (bmap->n_div == 0)
		return bmap;

	for (i = 0; i < bmap->n_div; ++i) {
		if (isl_int_is_zero(bmap->div[i][0]))
			continue;
		bmap = remove_independent_vars_from_div(bmap, i);
		if (!bmap)
			break;
	}

	return bmap;
}

/* Remove any common factor in numerator and denominator of the div expression,
 * not taking into account the constant term.
 * That is, if the div is of the form
 *
 *	floor((a + m f(x))/(m d))
 *
 * then replace it by
 *
 *	floor((floor(a/m) + f(x))/d)
 *
 * The difference {a/m}/d in the argument satisfies 0 <= {a/m}/d < 1/d
 * and can therefore not influence the result of the floor.
 */
static void normalize_div_expression(__isl_keep isl_basic_map *bmap, int div)
{
	unsigned total = isl_basic_map_total_dim(bmap);
	isl_ctx *ctx = bmap->ctx;

	if (isl_int_is_zero(bmap->div[div][0]))
		return;
	isl_seq_gcd(bmap->div[div] + 2, total, &ctx->normalize_gcd);
	isl_int_gcd(ctx->normalize_gcd, ctx->normalize_gcd, bmap->div[div][0]);
	if (isl_int_is_one(ctx->normalize_gcd))
		return;
	isl_int_fdiv_q(bmap->div[div][1], bmap->div[div][1],
			ctx->normalize_gcd);
	isl_int_divexact(bmap->div[div][0], bmap->div[div][0],
			ctx->normalize_gcd);
	isl_seq_scale_down(bmap->div[div] + 2, bmap->div[div] + 2,
			ctx->normalize_gcd, total);
}

/* Remove any common factor in numerator and denominator of a div expression,
 * not taking into account the constant term.
 * That is, look for any div of the form
 *
 *	floor((a + m f(x))/(m d))
 *
 * and replace it by
 *
 *	floor((floor(a/m) + f(x))/d)
 *
 * The difference {a/m}/d in the argument satisfies 0 <= {a/m}/d < 1/d
 * and can therefore not influence the result of the floor.
 */
static __isl_give isl_basic_map *normalize_div_expressions(
	__isl_take isl_basic_map *bmap)
{
	int i;

	if (!bmap)
		return NULL;
	if (bmap->n_div == 0)
		return bmap;

	for (i = 0; i < bmap->n_div; ++i)
		normalize_div_expression(bmap, i);

	return bmap;
}

/* Assumes divs have been ordered if keep_divs is set.
 */
static void eliminate_var_using_equality(struct isl_basic_map *bmap,
	unsigned pos, isl_int *eq, int keep_divs, int *progress)
{
	unsigned total;
	unsigned space_total;
	int k;
	int last_div;

	total = isl_basic_map_total_dim(bmap);
	space_total = isl_space_dim(bmap->dim, isl_dim_all);
	last_div = isl_seq_last_non_zero(eq + 1 + space_total, bmap->n_div);
	for (k = 0; k < bmap->n_eq; ++k) {
		if (bmap->eq[k] == eq)
			continue;
		if (isl_int_is_zero(bmap->eq[k][1+pos]))
			continue;
		if (progress)
			*progress = 1;
		isl_seq_elim(bmap->eq[k], eq, 1+pos, 1+total, NULL);
		isl_seq_normalize(bmap->ctx, bmap->eq[k], 1 + total);
	}

	for (k = 0; k < bmap->n_ineq; ++k) {
		if (isl_int_is_zero(bmap->ineq[k][1+pos]))
			continue;
		if (progress)
			*progress = 1;
		isl_seq_elim(bmap->ineq[k], eq, 1+pos, 1+total, NULL);
		isl_seq_normalize(bmap->ctx, bmap->ineq[k], 1 + total);
		ISL_F_CLR(bmap, ISL_BASIC_MAP_NORMALIZED);
	}

	for (k = 0; k < bmap->n_div; ++k) {
		if (isl_int_is_zero(bmap->div[k][0]))
			continue;
		if (isl_int_is_zero(bmap->div[k][1+1+pos]))
			continue;
		if (progress)
			*progress = 1;
		/* We need to be careful about circular definitions,
		 * so for now we just remove the definition of div k
		 * if the equality contains any divs.
		 * If keep_divs is set, then the divs have been ordered
		 * and we can keep the definition as long as the result
		 * is still ordered.
		 */
		if (last_div == -1 || (keep_divs && last_div < k)) {
			isl_seq_elim(bmap->div[k]+1, eq,
					1+pos, 1+total, &bmap->div[k][0]);
			normalize_div_expression(bmap, k);
		} else
			isl_seq_clr(bmap->div[k], 1 + total);
		ISL_F_CLR(bmap, ISL_BASIC_MAP_NORMALIZED);
	}
}

/* Assumes divs have been ordered if keep_divs is set.
 */
static __isl_give isl_basic_map *eliminate_div(__isl_take isl_basic_map *bmap,
	isl_int *eq, unsigned div, int keep_divs)
{
	unsigned pos = isl_space_dim(bmap->dim, isl_dim_all) + div;

	eliminate_var_using_equality(bmap, pos, eq, keep_divs, NULL);

	bmap = isl_basic_map_drop_div(bmap, div);

	return bmap;
}

/* Check if elimination of div "div" using equality "eq" would not
 * result in a div depending on a later div.
 */
static int ok_to_eliminate_div(struct isl_basic_map *bmap, isl_int *eq,
	unsigned div)
{
	int k;
	int last_div;
	unsigned space_total = isl_space_dim(bmap->dim, isl_dim_all);
	unsigned pos = space_total + div;

	last_div = isl_seq_last_non_zero(eq + 1 + space_total, bmap->n_div);
	if (last_div < 0 || last_div <= div)
		return 1;

	for (k = 0; k <= last_div; ++k) {
		if (isl_int_is_zero(bmap->div[k][0]))
			return 1;
		if (!isl_int_is_zero(bmap->div[k][1 + 1 + pos]))
			return 0;
	}

	return 1;
}

/* Elimininate divs based on equalities
 */
static struct isl_basic_map *eliminate_divs_eq(
		struct isl_basic_map *bmap, int *progress)
{
	int d;
	int i;
	int modified = 0;
	unsigned off;

	bmap = isl_basic_map_order_divs(bmap);

	if (!bmap)
		return NULL;

	off = 1 + isl_space_dim(bmap->dim, isl_dim_all);

	for (d = bmap->n_div - 1; d >= 0 ; --d) {
		for (i = 0; i < bmap->n_eq; ++i) {
			if (!isl_int_is_one(bmap->eq[i][off + d]) &&
			    !isl_int_is_negone(bmap->eq[i][off + d]))
				continue;
			if (!ok_to_eliminate_div(bmap, bmap->eq[i], d))
				continue;
			modified = 1;
			*progress = 1;
			bmap = eliminate_div(bmap, bmap->eq[i], d, 1);
			if (isl_basic_map_drop_equality(bmap, i) < 0)
				return isl_basic_map_free(bmap);
			break;
		}
	}
	if (modified)
		return eliminate_divs_eq(bmap, progress);
	return bmap;
}

/* Elimininate divs based on inequalities
 */
static struct isl_basic_map *eliminate_divs_ineq(
		struct isl_basic_map *bmap, int *progress)
{
	int d;
	int i;
	unsigned off;
	struct isl_ctx *ctx;

	if (!bmap)
		return NULL;

	ctx = bmap->ctx;
	off = 1 + isl_space_dim(bmap->dim, isl_dim_all);

	for (d = bmap->n_div - 1; d >= 0 ; --d) {
		for (i = 0; i < bmap->n_eq; ++i)
			if (!isl_int_is_zero(bmap->eq[i][off + d]))
				break;
		if (i < bmap->n_eq)
			continue;
		for (i = 0; i < bmap->n_ineq; ++i)
			if (isl_int_abs_gt(bmap->ineq[i][off + d], ctx->one))
				break;
		if (i < bmap->n_ineq)
			continue;
		*progress = 1;
		bmap = isl_basic_map_eliminate_vars(bmap, (off-1)+d, 1);
		if (!bmap || ISL_F_ISSET(bmap, ISL_BASIC_MAP_EMPTY))
			break;
		bmap = isl_basic_map_drop_div(bmap, d);
		if (!bmap)
			break;
	}
	return bmap;
}

struct isl_basic_map *isl_basic_map_gauss(
	struct isl_basic_map *bmap, int *progress)
{
	int k;
	int done;
	int last_var;
	unsigned total_var;
	unsigned total;

	bmap = isl_basic_map_order_divs(bmap);

	if (!bmap)
		return NULL;

	total = isl_basic_map_total_dim(bmap);
	total_var = total - bmap->n_div;

	last_var = total - 1;
	for (done = 0; done < bmap->n_eq; ++done) {
		for (; last_var >= 0; --last_var) {
			for (k = done; k < bmap->n_eq; ++k)
				if (!isl_int_is_zero(bmap->eq[k][1+last_var]))
					break;
			if (k < bmap->n_eq)
				break;
		}
		if (last_var < 0)
			break;
		if (k != done)
			swap_equality(bmap, k, done);
		if (isl_int_is_neg(bmap->eq[done][1+last_var]))
			isl_seq_neg(bmap->eq[done], bmap->eq[done], 1+total);

		eliminate_var_using_equality(bmap, last_var, bmap->eq[done], 1,
						progress);

		if (last_var >= total_var &&
		    isl_int_is_zero(bmap->div[last_var - total_var][0])) {
			unsigned div = last_var - total_var;
			isl_seq_neg(bmap->div[div]+1, bmap->eq[done], 1+total);
			isl_int_set_si(bmap->div[div][1+1+last_var], 0);
			isl_int_set(bmap->div[div][0],
				    bmap->eq[done][1+last_var]);
			if (progress)
				*progress = 1;
			ISL_F_CLR(bmap, ISL_BASIC_MAP_NORMALIZED);
		}
	}
	if (done == bmap->n_eq)
		return bmap;
	for (k = done; k < bmap->n_eq; ++k) {
		if (isl_int_is_zero(bmap->eq[k][0]))
			continue;
		return isl_basic_map_set_to_empty(bmap);
	}
	isl_basic_map_free_equality(bmap, bmap->n_eq-done);
	return bmap;
}

struct isl_basic_set *isl_basic_set_gauss(
	struct isl_basic_set *bset, int *progress)
{
	return (struct isl_basic_set*)isl_basic_map_gauss(
			(struct isl_basic_map *)bset, progress);
}


static unsigned int round_up(unsigned int v)
{
	int old_v = v;

	while (v) {
		old_v = v;
		v ^= v & -v;
	}
	return old_v << 1;
}

/* Hash table of inequalities in a basic map.
 * "index" is an array of addresses of inequalities in the basic map, some
 * of which are NULL.  The inequalities are hashed on the coefficients
 * except the constant term.
 * "size" is the number of elements in the array and is always a power of two
 * "bits" is the number of bits need to represent an index into the array.
 * "total" is the total dimension of the basic map.
 */
struct isl_constraint_index {
	unsigned int size;
	int bits;
	isl_int ***index;
	unsigned total;
};

/* Fill in the "ci" data structure for holding the inequalities of "bmap".
 */
static isl_stat create_constraint_index(struct isl_constraint_index *ci,
	__isl_keep isl_basic_map *bmap)
{
	isl_ctx *ctx;

	ci->index = NULL;
	if (!bmap)
		return isl_stat_error;
	ci->total = isl_basic_set_total_dim(bmap);
	if (bmap->n_ineq == 0)
		return isl_stat_ok;
	ci->size = round_up(4 * (bmap->n_ineq + 1) / 3 - 1);
	ci->bits = ffs(ci->size) - 1;
	ctx = isl_basic_map_get_ctx(bmap);
	ci->index = isl_calloc_array(ctx, isl_int **, ci->size);
	if (!ci->index)
		return isl_stat_error;

	return isl_stat_ok;
}

/* Free the memory allocated by create_constraint_index.
 */
static void constraint_index_free(struct isl_constraint_index *ci)
{
	free(ci->index);
}

/* Return the position in ci->index that contains the address of
 * an inequality that is equal to *ineq up to the constant term,
 * provided this address is not identical to "ineq".
 * If there is no such inequality, then return the position where
 * such an inequality should be inserted.
 */
static int hash_index_ineq(struct isl_constraint_index *ci, isl_int **ineq)
{
	int h;
	uint32_t hash = isl_seq_get_hash_bits((*ineq) + 1, ci->total, ci->bits);
	for (h = hash; ci->index[h]; h = (h+1) % ci->size)
		if (ineq != ci->index[h] &&
		    isl_seq_eq((*ineq) + 1, ci->index[h][0]+1, ci->total))
			break;
	return h;
}

/* Return the position in ci->index that contains the address of
 * an inequality that is equal to the k'th inequality of "bmap"
 * up to the constant term, provided it does not point to the very
 * same inequality.
 * If there is no such inequality, then return the position where
 * such an inequality should be inserted.
 */
static int hash_index(struct isl_constraint_index *ci,
	__isl_keep isl_basic_map *bmap, int k)
{
	return hash_index_ineq(ci, &bmap->ineq[k]);
}

static int set_hash_index(struct isl_constraint_index *ci,
			  struct isl_basic_set *bset, int k)
{
	return hash_index(ci, bset, k);
}

/* Fill in the "ci" data structure with the inequalities of "bset".
 */
static isl_stat setup_constraint_index(struct isl_constraint_index *ci,
	__isl_keep isl_basic_set *bset)
{
	int k, h;

	if (create_constraint_index(ci, bset) < 0)
		return isl_stat_error;

	for (k = 0; k < bset->n_ineq; ++k) {
		h = set_hash_index(ci, bset, k);
		ci->index[h] = &bset->ineq[k];
	}

	return isl_stat_ok;
}

/* Is the inequality ineq (obviously) redundant with respect
 * to the constraints in "ci"?
 *
 * Look for an inequality in "ci" with the same coefficients and then
 * check if the contant term of "ineq" is greater than or equal
 * to the constant term of that inequality.  If so, "ineq" is clearly
 * redundant.
 *
 * Note that hash_index_ineq ignores a stored constraint if it has
 * the same address as the passed inequality.  It is ok to pass
 * the address of a local variable here since it will never be
 * the same as the address of a constraint in "ci".
 */
static isl_bool constraint_index_is_redundant(struct isl_constraint_index *ci,
	isl_int *ineq)
{
	int h;

	h = hash_index_ineq(ci, &ineq);
	if (!ci->index[h])
		return isl_bool_false;
	return isl_int_ge(ineq[0], (*ci->index[h])[0]);
}

/* If we can eliminate more than one div, then we need to make
 * sure we do it from last div to first div, in order not to
 * change the position of the other divs that still need to
 * be removed.
 */
static struct isl_basic_map *remove_duplicate_divs(
	struct isl_basic_map *bmap, int *progress)
{
	unsigned int size;
	int *index;
	int *elim_for;
	int k, l, h;
	int bits;
	struct isl_blk eq;
	unsigned total_var;
	unsigned total;
	struct isl_ctx *ctx;

	bmap = isl_basic_map_order_divs(bmap);
	if (!bmap || bmap->n_div <= 1)
		return bmap;

	total_var = isl_space_dim(bmap->dim, isl_dim_all);
	total = total_var + bmap->n_div;

	ctx = bmap->ctx;
	for (k = bmap->n_div - 1; k >= 0; --k)
		if (!isl_int_is_zero(bmap->div[k][0]))
			break;
	if (k <= 0)
		return bmap;

	size = round_up(4 * bmap->n_div / 3 - 1);
	if (size == 0)
		return bmap;
	elim_for = isl_calloc_array(ctx, int, bmap->n_div);
	bits = ffs(size) - 1;
	index = isl_calloc_array(ctx, int, size);
	if (!elim_for || !index)
		goto out;
	eq = isl_blk_alloc(ctx, 1+total);
	if (isl_blk_is_error(eq))
		goto out;

	isl_seq_clr(eq.data, 1+total);
	index[isl_seq_get_hash_bits(bmap->div[k], 2+total, bits)] = k + 1;
	for (--k; k >= 0; --k) {
		uint32_t hash;

		if (isl_int_is_zero(bmap->div[k][0]))
			continue;

		hash = isl_seq_get_hash_bits(bmap->div[k], 2+total, bits);
		for (h = hash; index[h]; h = (h+1) % size)
			if (isl_seq_eq(bmap->div[k],
				       bmap->div[index[h]-1], 2+total))
				break;
		if (index[h]) {
			*progress = 1;
			l = index[h] - 1;
			elim_for[l] = k + 1;
		}
		index[h] = k+1;
	}
	for (l = bmap->n_div - 1; l >= 0; --l) {
		if (!elim_for[l])
			continue;
		k = elim_for[l] - 1;
		isl_int_set_si(eq.data[1+total_var+k], -1);
		isl_int_set_si(eq.data[1+total_var+l], 1);
		bmap = eliminate_div(bmap, eq.data, l, 1);
		if (!bmap)
			break;
		isl_int_set_si(eq.data[1+total_var+k], 0);
		isl_int_set_si(eq.data[1+total_var+l], 0);
	}

	isl_blk_free(ctx, eq);
out:
	free(index);
	free(elim_for);
	return bmap;
}

static int n_pure_div_eq(struct isl_basic_map *bmap)
{
	int i, j;
	unsigned total;

	total = isl_space_dim(bmap->dim, isl_dim_all);
	for (i = 0, j = bmap->n_div-1; i < bmap->n_eq; ++i) {
		while (j >= 0 && isl_int_is_zero(bmap->eq[i][1 + total + j]))
			--j;
		if (j < 0)
			break;
		if (isl_seq_first_non_zero(bmap->eq[i] + 1 + total, j) != -1)
			return 0;
	}
	return i;
}

/* Normalize divs that appear in equalities.
 *
 * In particular, we assume that bmap contains some equalities
 * of the form
 *
 *	a x = m * e_i
 *
 * and we want to replace the set of e_i by a minimal set and
 * such that the new e_i have a canonical representation in terms
 * of the vector x.
 * If any of the equalities involves more than one divs, then
 * we currently simply bail out.
 *
 * Let us first additionally assume that all equalities involve
 * a div.  The equalities then express modulo constraints on the
 * remaining variables and we can use "parameter compression"
 * to find a minimal set of constraints.  The result is a transformation
 *
 *	x = T(x') = x_0 + G x'
 *
 * with G a lower-triangular matrix with all elements below the diagonal
 * non-negative and smaller than the diagonal element on the same row.
 * We first normalize x_0 by making the same property hold in the affine
 * T matrix.
 * The rows i of G with a 1 on the diagonal do not impose any modulo
 * constraint and simply express x_i = x'_i.
 * For each of the remaining rows i, we introduce a div and a corresponding
 * equality.  In particular
 *
 *	g_ii e_j = x_i - g_i(x')
 *
 * where each x'_k is replaced either by x_k (if g_kk = 1) or the
 * corresponding div (if g_kk != 1).
 *
 * If there are any equalities not involving any div, then we
 * first apply a variable compression on the variables x:
 *
 *	x = C x''	x'' = C_2 x
 *
 * and perform the above parameter compression on A C instead of on A.
 * The resulting compression is then of the form
 *
 *	x'' = T(x') = x_0 + G x'
 *
 * and in constructing the new divs and the corresponding equalities,
 * we have to replace each x'', i.e., the x'_k with (g_kk = 1),
 * by the corresponding row from C_2.
 */
static struct isl_basic_map *normalize_divs(
	struct isl_basic_map *bmap, int *progress)
{
	int i, j, k;
	int total;
	int div_eq;
	struct isl_mat *B;
	struct isl_vec *d;
	struct isl_mat *T = NULL;
	struct isl_mat *C = NULL;
	struct isl_mat *C2 = NULL;
	isl_int v;
	int *pos;
	int dropped, needed;

	if (!bmap)
		return NULL;

	if (bmap->n_div == 0)
		return bmap;

	if (bmap->n_eq == 0)
		return bmap;

	if (ISL_F_ISSET(bmap, ISL_BASIC_MAP_NORMALIZED_DIVS))
		return bmap;

	total = isl_space_dim(bmap->dim, isl_dim_all);
	div_eq = n_pure_div_eq(bmap);
	if (div_eq == 0)
		return bmap;

	if (div_eq < bmap->n_eq) {
		B = isl_mat_sub_alloc6(bmap->ctx, bmap->eq, div_eq,
					bmap->n_eq - div_eq, 0, 1 + total);
		C = isl_mat_variable_compression(B, &C2);
		if (!C || !C2)
			goto error;
		if (C->n_col == 0) {
			bmap = isl_basic_map_set_to_empty(bmap);
			isl_mat_free(C);
			isl_mat_free(C2);
			goto done;
		}
	}

	d = isl_vec_alloc(bmap->ctx, div_eq);
	if (!d)
		goto error;
	for (i = 0, j = bmap->n_div-1; i < div_eq; ++i) {
		while (j >= 0 && isl_int_is_zero(bmap->eq[i][1 + total + j]))
			--j;
		isl_int_set(d->block.data[i], bmap->eq[i][1 + total + j]);
	}
	B = isl_mat_sub_alloc6(bmap->ctx, bmap->eq, 0, div_eq, 0, 1 + total);

	if (C) {
		B = isl_mat_product(B, C);
		C = NULL;
	}

	T = isl_mat_parameter_compression(B, d);
	if (!T)
		goto error;
	if (T->n_col == 0) {
		bmap = isl_basic_map_set_to_empty(bmap);
		isl_mat_free(C2);
		isl_mat_free(T);
		goto done;
	}
	isl_int_init(v);
	for (i = 0; i < T->n_row - 1; ++i) {
		isl_int_fdiv_q(v, T->row[1 + i][0], T->row[1 + i][1 + i]);
		if (isl_int_is_zero(v))
			continue;
		isl_mat_col_submul(T, 0, v, 1 + i);
	}
	isl_int_clear(v);
	pos = isl_alloc_array(bmap->ctx, int, T->n_row);
	if (!pos)
		goto error;
	/* We have to be careful because dropping equalities may reorder them */
	dropped = 0;
	for (j = bmap->n_div - 1; j >= 0; --j) {
		for (i = 0; i < bmap->n_eq; ++i)
			if (!isl_int_is_zero(bmap->eq[i][1 + total + j]))
				break;
		if (i < bmap->n_eq) {
			bmap = isl_basic_map_drop_div(bmap, j);
			isl_basic_map_drop_equality(bmap, i);
			++dropped;
		}
	}
	pos[0] = 0;
	needed = 0;
	for (i = 1; i < T->n_row; ++i) {
		if (isl_int_is_one(T->row[i][i]))
			pos[i] = i;
		else
			needed++;
	}
	if (needed > dropped) {
		bmap = isl_basic_map_extend_space(bmap, isl_space_copy(bmap->dim),
				needed, needed, 0);
		if (!bmap)
			goto error;
	}
	for (i = 1; i < T->n_row; ++i) {
		if (isl_int_is_one(T->row[i][i]))
			continue;
		k = isl_basic_map_alloc_div(bmap);
		pos[i] = 1 + total + k;
		isl_seq_clr(bmap->div[k] + 1, 1 + total + bmap->n_div);
		isl_int_set(bmap->div[k][0], T->row[i][i]);
		if (C2)
			isl_seq_cpy(bmap->div[k] + 1, C2->row[i], 1 + total);
		else
			isl_int_set_si(bmap->div[k][1 + i], 1);
		for (j = 0; j < i; ++j) {
			if (isl_int_is_zero(T->row[i][j]))
				continue;
			if (pos[j] < T->n_row && C2)
				isl_seq_submul(bmap->div[k] + 1, T->row[i][j],
						C2->row[pos[j]], 1 + total);
			else
				isl_int_neg(bmap->div[k][1 + pos[j]],
								T->row[i][j]);
		}
		j = isl_basic_map_alloc_equality(bmap);
		isl_seq_neg(bmap->eq[j], bmap->div[k]+1, 1+total+bmap->n_div);
		isl_int_set(bmap->eq[j][pos[i]], bmap->div[k][0]);
	}
	free(pos);
	isl_mat_free(C2);
	isl_mat_free(T);

	if (progress)
		*progress = 1;
done:
	ISL_F_SET(bmap, ISL_BASIC_MAP_NORMALIZED_DIVS);

	return bmap;
error:
	isl_mat_free(C);
	isl_mat_free(C2);
	isl_mat_free(T);
	return bmap;
}

static struct isl_basic_map *set_div_from_lower_bound(
	struct isl_basic_map *bmap, int div, int ineq)
{
	unsigned total = 1 + isl_space_dim(bmap->dim, isl_dim_all);

	isl_seq_neg(bmap->div[div] + 1, bmap->ineq[ineq], total + bmap->n_div);
	isl_int_set(bmap->div[div][0], bmap->ineq[ineq][total + div]);
	isl_int_add(bmap->div[div][1], bmap->div[div][1], bmap->div[div][0]);
	isl_int_sub_ui(bmap->div[div][1], bmap->div[div][1], 1);
	isl_int_set_si(bmap->div[div][1 + total + div], 0);

	return bmap;
}

/* Check whether it is ok to define a div based on an inequality.
 * To avoid the introduction of circular definitions of divs, we
 * do not allow such a definition if the resulting expression would refer to
 * any other undefined divs or if any known div is defined in
 * terms of the unknown div.
 */
static int ok_to_set_div_from_bound(struct isl_basic_map *bmap,
	int div, int ineq)
{
	int j;
	unsigned total = 1 + isl_space_dim(bmap->dim, isl_dim_all);

	/* Not defined in terms of unknown divs */
	for (j = 0; j < bmap->n_div; ++j) {
		if (div == j)
			continue;
		if (isl_int_is_zero(bmap->ineq[ineq][total + j]))
			continue;
		if (isl_int_is_zero(bmap->div[j][0]))
			return 0;
	}

	/* No other div defined in terms of this one => avoid loops */
	for (j = 0; j < bmap->n_div; ++j) {
		if (div == j)
			continue;
		if (isl_int_is_zero(bmap->div[j][0]))
			continue;
		if (!isl_int_is_zero(bmap->div[j][1 + total + div]))
			return 0;
	}

	return 1;
}

/* Would an expression for div "div" based on inequality "ineq" of "bmap"
 * be a better expression than the current one?
 *
 * If we do not have any expression yet, then any expression would be better.
 * Otherwise we check if the last variable involved in the inequality
 * (disregarding the div that it would define) is in an earlier position
 * than the last variable involved in the current div expression.
 */
static int better_div_constraint(__isl_keep isl_basic_map *bmap,
	int div, int ineq)
{
	unsigned total = 1 + isl_space_dim(bmap->dim, isl_dim_all);
	int last_div;
	int last_ineq;

	if (isl_int_is_zero(bmap->div[div][0]))
		return 1;

	if (isl_seq_last_non_zero(bmap->ineq[ineq] + total + div + 1,
				  bmap->n_div - (div + 1)) >= 0)
		return 0;

	last_ineq = isl_seq_last_non_zero(bmap->ineq[ineq], total + div);
	last_div = isl_seq_last_non_zero(bmap->div[div] + 1,
					 total + bmap->n_div);

	return last_ineq < last_div;
}

/* Given two constraints "k" and "l" that are opposite to each other,
 * except for the constant term, check if we can use them
 * to obtain an expression for one of the hitherto unknown divs or
 * a "better" expression for a div for which we already have an expression.
 * "sum" is the sum of the constant terms of the constraints.
 * If this sum is strictly smaller than the coefficient of one
 * of the divs, then this pair can be used define the div.
 * To avoid the introduction of circular definitions of divs, we
 * do not use the pair if the resulting expression would refer to
 * any other undefined divs or if any known div is defined in
 * terms of the unknown div.
 */
static struct isl_basic_map *check_for_div_constraints(
	struct isl_basic_map *bmap, int k, int l, isl_int sum, int *progress)
{
	int i;
	unsigned total = 1 + isl_space_dim(bmap->dim, isl_dim_all);

	for (i = 0; i < bmap->n_div; ++i) {
		if (isl_int_is_zero(bmap->ineq[k][total + i]))
			continue;
		if (isl_int_abs_ge(sum, bmap->ineq[k][total + i]))
			continue;
		if (!better_div_constraint(bmap, i, k))
			continue;
		if (!ok_to_set_div_from_bound(bmap, i, k))
			break;
		if (isl_int_is_pos(bmap->ineq[k][total + i]))
			bmap = set_div_from_lower_bound(bmap, i, k);
		else
			bmap = set_div_from_lower_bound(bmap, i, l);
		if (progress)
			*progress = 1;
		break;
	}
	return bmap;
}

__isl_give isl_basic_map *isl_basic_map_remove_duplicate_constraints(
	__isl_take isl_basic_map *bmap, int *progress, int detect_divs)
{
	struct isl_constraint_index ci;
	int k, l, h;
	unsigned total = isl_basic_map_total_dim(bmap);
	isl_int sum;

	if (!bmap || bmap->n_ineq <= 1)
		return bmap;

	if (create_constraint_index(&ci, bmap) < 0)
		return bmap;

	h = isl_seq_get_hash_bits(bmap->ineq[0] + 1, total, ci.bits);
	ci.index[h] = &bmap->ineq[0];
	for (k = 1; k < bmap->n_ineq; ++k) {
		h = hash_index(&ci, bmap, k);
		if (!ci.index[h]) {
			ci.index[h] = &bmap->ineq[k];
			continue;
		}
		if (progress)
			*progress = 1;
		l = ci.index[h] - &bmap->ineq[0];
		if (isl_int_lt(bmap->ineq[k][0], bmap->ineq[l][0]))
			swap_inequality(bmap, k, l);
		isl_basic_map_drop_inequality(bmap, k);
		--k;
	}
	isl_int_init(sum);
	for (k = 0; k < bmap->n_ineq-1; ++k) {
		isl_seq_neg(bmap->ineq[k]+1, bmap->ineq[k]+1, total);
		h = hash_index(&ci, bmap, k);
		isl_seq_neg(bmap->ineq[k]+1, bmap->ineq[k]+1, total);
		if (!ci.index[h])
			continue;
		l = ci.index[h] - &bmap->ineq[0];
		isl_int_add(sum, bmap->ineq[k][0], bmap->ineq[l][0]);
		if (isl_int_is_pos(sum)) {
			if (detect_divs)
				bmap = check_for_div_constraints(bmap, k, l,
								 sum, progress);
			continue;
		}
		if (isl_int_is_zero(sum)) {
			/* We need to break out of the loop after these
			 * changes since the contents of the hash
			 * will no longer be valid.
			 * Plus, we probably we want to regauss first.
			 */
			if (progress)
				*progress = 1;
			isl_basic_map_drop_inequality(bmap, l);
			isl_basic_map_inequality_to_equality(bmap, k);
		} else
			bmap = isl_basic_map_set_to_empty(bmap);
		break;
	}
	isl_int_clear(sum);

	constraint_index_free(&ci);
	return bmap;
}

/* Detect all pairs of inequalities that form an equality.
 *
 * isl_basic_map_remove_duplicate_constraints detects at most one such pair.
 * Call it repeatedly while it is making progress.
 */
__isl_give isl_basic_map *isl_basic_map_detect_inequality_pairs(
	__isl_take isl_basic_map *bmap, int *progress)
{
	int duplicate;

	do {
		duplicate = 0;
		bmap = isl_basic_map_remove_duplicate_constraints(bmap,
								&duplicate, 0);
		if (progress && duplicate)
			*progress = 1;
	} while (duplicate);

	return bmap;
}

/* Eliminate knowns divs from constraints where they appear with
 * a (positive or negative) unit coefficient.
 *
 * That is, replace
 *
 *	floor(e/m) + f >= 0
 *
 * by
 *
 *	e + m f >= 0
 *
 * and
 *
 *	-floor(e/m) + f >= 0
 *
 * by
 *
 *	-e + m f + m - 1 >= 0
 *
 * The first conversion is valid because floor(e/m) >= -f is equivalent
 * to e/m >= -f because -f is an integral expression.
 * The second conversion follows from the fact that
 *
 *	-floor(e/m) = ceil(-e/m) = floor((-e + m - 1)/m)
 *
 *
 * Note that one of the div constraints may have been eliminated
 * due to being redundant with respect to the constraint that is
 * being modified by this function.  The modified constraint may
 * no longer imply this div constraint, so we add it back to make
 * sure we do not lose any information.
 *
 * We skip integral divs, i.e., those with denominator 1, as we would
 * risk eliminating the div from the div constraints.  We do not need
 * to handle those divs here anyway since the div constraints will turn
 * out to form an equality and this equality can then be use to eliminate
 * the div from all constraints.
 */
static __isl_give isl_basic_map *eliminate_unit_divs(
	__isl_take isl_basic_map *bmap, int *progress)
{
	int i, j;
	isl_ctx *ctx;
	unsigned total;

	if (!bmap)
		return NULL;

	ctx = isl_basic_map_get_ctx(bmap);
	total = 1 + isl_space_dim(bmap->dim, isl_dim_all);

	for (i = 0; i < bmap->n_div; ++i) {
		if (isl_int_is_zero(bmap->div[i][0]))
			continue;
		if (isl_int_is_one(bmap->div[i][0]))
			continue;
		for (j = 0; j < bmap->n_ineq; ++j) {
			int s;

			if (!isl_int_is_one(bmap->ineq[j][total + i]) &&
			    !isl_int_is_negone(bmap->ineq[j][total + i]))
				continue;

			*progress = 1;

			s = isl_int_sgn(bmap->ineq[j][total + i]);
			isl_int_set_si(bmap->ineq[j][total + i], 0);
			if (s < 0)
				isl_seq_combine(bmap->ineq[j],
					ctx->negone, bmap->div[i] + 1,
					bmap->div[i][0], bmap->ineq[j],
					total + bmap->n_div);
			else
				isl_seq_combine(bmap->ineq[j],
					ctx->one, bmap->div[i] + 1,
					bmap->div[i][0], bmap->ineq[j],
					total + bmap->n_div);
			if (s < 0) {
				isl_int_add(bmap->ineq[j][0],
					bmap->ineq[j][0], bmap->div[i][0]);
				isl_int_sub_ui(bmap->ineq[j][0],
					bmap->ineq[j][0], 1);
			}

			bmap = isl_basic_map_extend_constraints(bmap, 0, 1);
			if (isl_basic_map_add_div_constraint(bmap, i, s) < 0)
				return isl_basic_map_free(bmap);
		}
	}

	return bmap;
}

struct isl_basic_map *isl_basic_map_simplify(struct isl_basic_map *bmap)
{
	int progress = 1;
	if (!bmap)
		return NULL;
	while (progress) {
		progress = 0;
		if (!bmap)
			break;
		if (isl_basic_map_plain_is_empty(bmap))
			break;
		bmap = isl_basic_map_normalize_constraints(bmap);
		bmap = remove_independent_vars_from_divs(bmap);
		bmap = normalize_div_expressions(bmap);
		bmap = remove_duplicate_divs(bmap, &progress);
		bmap = eliminate_unit_divs(bmap, &progress);
		bmap = eliminate_divs_eq(bmap, &progress);
		bmap = eliminate_divs_ineq(bmap, &progress);
		bmap = isl_basic_map_gauss(bmap, &progress);
		/* requires equalities in normal form */
		bmap = normalize_divs(bmap, &progress);
		bmap = isl_basic_map_remove_duplicate_constraints(bmap,
								&progress, 1);
		if (bmap && progress)
			ISL_F_CLR(bmap, ISL_BASIC_MAP_REDUCED_COEFFICIENTS);
	}
	return bmap;
}

struct isl_basic_set *isl_basic_set_simplify(struct isl_basic_set *bset)
{
	return (struct isl_basic_set *)
		isl_basic_map_simplify((struct isl_basic_map *)bset);
}


int isl_basic_map_is_div_constraint(__isl_keep isl_basic_map *bmap,
	isl_int *constraint, unsigned div)
{
	unsigned pos;

	if (!bmap)
		return -1;

	pos = 1 + isl_space_dim(bmap->dim, isl_dim_all) + div;

	if (isl_int_eq(constraint[pos], bmap->div[div][0])) {
		int neg;
		isl_int_sub(bmap->div[div][1],
				bmap->div[div][1], bmap->div[div][0]);
		isl_int_add_ui(bmap->div[div][1], bmap->div[div][1], 1);
		neg = isl_seq_is_neg(constraint, bmap->div[div]+1, pos);
		isl_int_sub_ui(bmap->div[div][1], bmap->div[div][1], 1);
		isl_int_add(bmap->div[div][1],
				bmap->div[div][1], bmap->div[div][0]);
		if (!neg)
			return 0;
		if (isl_seq_first_non_zero(constraint+pos+1,
					    bmap->n_div-div-1) != -1)
			return 0;
	} else if (isl_int_abs_eq(constraint[pos], bmap->div[div][0])) {
		if (!isl_seq_eq(constraint, bmap->div[div]+1, pos))
			return 0;
		if (isl_seq_first_non_zero(constraint+pos+1,
					    bmap->n_div-div-1) != -1)
			return 0;
	} else
		return 0;

	return 1;
}

int isl_basic_set_is_div_constraint(__isl_keep isl_basic_set *bset,
	isl_int *constraint, unsigned div)
{
	return isl_basic_map_is_div_constraint(bset, constraint, div);
}


/* If the only constraints a div d=floor(f/m)
 * appears in are its two defining constraints
 *
 *	f - m d >=0
 *	-(f - (m - 1)) + m d >= 0
 *
 * then it can safely be removed.
 */
static int div_is_redundant(struct isl_basic_map *bmap, int div)
{
	int i;
	unsigned pos = 1 + isl_space_dim(bmap->dim, isl_dim_all) + div;

	for (i = 0; i < bmap->n_eq; ++i)
		if (!isl_int_is_zero(bmap->eq[i][pos]))
			return 0;

	for (i = 0; i < bmap->n_ineq; ++i) {
		if (isl_int_is_zero(bmap->ineq[i][pos]))
			continue;
		if (!isl_basic_map_is_div_constraint(bmap, bmap->ineq[i], div))
			return 0;
	}

	for (i = 0; i < bmap->n_div; ++i) {
		if (isl_int_is_zero(bmap->div[i][0]))
			continue;
		if (!isl_int_is_zero(bmap->div[i][1+pos]))
			return 0;
	}

	return 1;
}

/*
 * Remove divs that don't occur in any of the constraints or other divs.
 * These can arise when dropping constraints from a basic map or
 * when the divs of a basic map have been temporarily aligned
 * with the divs of another basic map.
 */
static struct isl_basic_map *remove_redundant_divs(struct isl_basic_map *bmap)
{
	int i;

	if (!bmap)
		return NULL;

	for (i = bmap->n_div-1; i >= 0; --i) {
		if (!div_is_redundant(bmap, i))
			continue;
		bmap = isl_basic_map_drop_div(bmap, i);
	}
	return bmap;
}

/* Mark "bmap" as final, without checking for obviously redundant
 * integer divisions.  This function should be used when "bmap"
 * is known not to involve any such integer divisions.
 */
__isl_give isl_basic_map *isl_basic_map_mark_final(
	__isl_take isl_basic_map *bmap)
{
	if (!bmap)
		return NULL;
	ISL_F_SET(bmap, ISL_BASIC_SET_FINAL);
	return bmap;
}

/* Mark "bmap" as final, after removing obviously redundant integer divisions.
 */
struct isl_basic_map *isl_basic_map_finalize(struct isl_basic_map *bmap)
{
	bmap = remove_redundant_divs(bmap);
	bmap = isl_basic_map_mark_final(bmap);
	return bmap;
}

struct isl_basic_set *isl_basic_set_finalize(struct isl_basic_set *bset)
{
	return (struct isl_basic_set *)
		isl_basic_map_finalize((struct isl_basic_map *)bset);
}

struct isl_set *isl_set_finalize(struct isl_set *set)
{
	int i;

	if (!set)
		return NULL;
	for (i = 0; i < set->n; ++i) {
		set->p[i] = isl_basic_set_finalize(set->p[i]);
		if (!set->p[i])
			goto error;
	}
	return set;
error:
	isl_set_free(set);
	return NULL;
}

struct isl_map *isl_map_finalize(struct isl_map *map)
{
	int i;

	if (!map)
		return NULL;
	for (i = 0; i < map->n; ++i) {
		map->p[i] = isl_basic_map_finalize(map->p[i]);
		if (!map->p[i])
			goto error;
	}
	ISL_F_CLR(map, ISL_MAP_NORMALIZED);
	return map;
error:
	isl_map_free(map);
	return NULL;
}


/* Remove definition of any div that is defined in terms of the given variable.
 * The div itself is not removed.  Functions such as
 * eliminate_divs_ineq depend on the other divs remaining in place.
 */
static struct isl_basic_map *remove_dependent_vars(struct isl_basic_map *bmap,
									int pos)
{
	int i;

	if (!bmap)
		return NULL;

	for (i = 0; i < bmap->n_div; ++i) {
		if (isl_int_is_zero(bmap->div[i][0]))
			continue;
		if (isl_int_is_zero(bmap->div[i][1+1+pos]))
			continue;
		isl_int_set_si(bmap->div[i][0], 0);
	}
	return bmap;
}

/* Eliminate the specified variables from the constraints using
 * Fourier-Motzkin.  The variables themselves are not removed.
 */
struct isl_basic_map *isl_basic_map_eliminate_vars(
	struct isl_basic_map *bmap, unsigned pos, unsigned n)
{
	int d;
	int i, j, k;
	unsigned total;
	int need_gauss = 0;

	if (n == 0)
		return bmap;
	if (!bmap)
		return NULL;
	total = isl_basic_map_total_dim(bmap);

	bmap = isl_basic_map_cow(bmap);
	for (d = pos + n - 1; d >= 0 && d >= pos; --d)
		bmap = remove_dependent_vars(bmap, d);
	if (!bmap)
		return NULL;

	for (d = pos + n - 1;
	     d >= 0 && d >= total - bmap->n_div && d >= pos; --d)
		isl_seq_clr(bmap->div[d-(total-bmap->n_div)], 2+total);
	for (d = pos + n - 1; d >= 0 && d >= pos; --d) {
		int n_lower, n_upper;
		if (!bmap)
			return NULL;
		for (i = 0; i < bmap->n_eq; ++i) {
			if (isl_int_is_zero(bmap->eq[i][1+d]))
				continue;
			eliminate_var_using_equality(bmap, d, bmap->eq[i], 0, NULL);
			isl_basic_map_drop_equality(bmap, i);
			need_gauss = 1;
			break;
		}
		if (i < bmap->n_eq)
			continue;
		n_lower = 0;
		n_upper = 0;
		for (i = 0; i < bmap->n_ineq; ++i) {
			if (isl_int_is_pos(bmap->ineq[i][1+d]))
				n_lower++;
			else if (isl_int_is_neg(bmap->ineq[i][1+d]))
				n_upper++;
		}
		bmap = isl_basic_map_extend_constraints(bmap,
				0, n_lower * n_upper);
		if (!bmap)
			goto error;
		for (i = bmap->n_ineq - 1; i >= 0; --i) {
			int last;
			if (isl_int_is_zero(bmap->ineq[i][1+d]))
				continue;
			last = -1;
			for (j = 0; j < i; ++j) {
				if (isl_int_is_zero(bmap->ineq[j][1+d]))
					continue;
				last = j;
				if (isl_int_sgn(bmap->ineq[i][1+d]) ==
				    isl_int_sgn(bmap->ineq[j][1+d]))
					continue;
				k = isl_basic_map_alloc_inequality(bmap);
				if (k < 0)
					goto error;
				isl_seq_cpy(bmap->ineq[k], bmap->ineq[i],
						1+total);
				isl_seq_elim(bmap->ineq[k], bmap->ineq[j],
						1+d, 1+total, NULL);
			}
			isl_basic_map_drop_inequality(bmap, i);
			i = last + 1;
		}
		if (n_lower > 0 && n_upper > 0) {
			bmap = isl_basic_map_normalize_constraints(bmap);
			bmap = isl_basic_map_remove_duplicate_constraints(bmap,
								    NULL, 0);
			bmap = isl_basic_map_gauss(bmap, NULL);
			bmap = isl_basic_map_remove_redundancies(bmap);
			need_gauss = 0;
			if (!bmap)
				goto error;
			if (ISL_F_ISSET(bmap, ISL_BASIC_MAP_EMPTY))
				break;
		}
	}
	ISL_F_CLR(bmap, ISL_BASIC_MAP_NORMALIZED);
	if (need_gauss)
		bmap = isl_basic_map_gauss(bmap, NULL);
	return bmap;
error:
	isl_basic_map_free(bmap);
	return NULL;
}

struct isl_basic_set *isl_basic_set_eliminate_vars(
	struct isl_basic_set *bset, unsigned pos, unsigned n)
{
	return (struct isl_basic_set *)isl_basic_map_eliminate_vars(
			(struct isl_basic_map *)bset, pos, n);
}

/* Eliminate the specified n dimensions starting at first from the
 * constraints, without removing the dimensions from the space.
 * If the set is rational, the dimensions are eliminated using Fourier-Motzkin.
 * Otherwise, they are projected out and the original space is restored.
 */
__isl_give isl_basic_map *isl_basic_map_eliminate(
	__isl_take isl_basic_map *bmap,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	isl_space *space;

	if (!bmap)
		return NULL;
	if (n == 0)
		return bmap;

	if (first + n > isl_basic_map_dim(bmap, type) || first + n < first)
		isl_die(bmap->ctx, isl_error_invalid,
			"index out of bounds", goto error);

	if (ISL_F_ISSET(bmap, ISL_BASIC_MAP_RATIONAL)) {
		first += isl_basic_map_offset(bmap, type) - 1;
		bmap = isl_basic_map_eliminate_vars(bmap, first, n);
		return isl_basic_map_finalize(bmap);
	}

	space = isl_basic_map_get_space(bmap);
	bmap = isl_basic_map_project_out(bmap, type, first, n);
	bmap = isl_basic_map_insert_dims(bmap, type, first, n);
	bmap = isl_basic_map_reset_space(bmap, space);
	return bmap;
error:
	isl_basic_map_free(bmap);
	return NULL;
}

__isl_give isl_basic_set *isl_basic_set_eliminate(
	__isl_take isl_basic_set *bset,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	return isl_basic_map_eliminate(bset, type, first, n);
}

/* Remove all constraints from "bmap" that reference any unknown local
 * variables (directly or indirectly).
 *
 * Dropping all constraints on a local variable will make it redundant,
 * so it will get removed implicitly by
 * isl_basic_map_drop_constraints_involving_dims.  Some other local
 * variables may also end up becoming redundant if they only appear
 * in constraints together with the unknown local variable.
 * Therefore, start over after calling
 * isl_basic_map_drop_constraints_involving_dims.
 */
__isl_give isl_basic_map *isl_basic_map_drop_constraint_involving_unknown_divs(
	__isl_take isl_basic_map *bmap)
{
	isl_bool known;
	int i, n_div, o_div;

	known = isl_basic_map_divs_known(bmap);
	if (known < 0)
		return isl_basic_map_free(bmap);
	if (known)
		return bmap;

	n_div = isl_basic_map_dim(bmap, isl_dim_div);
	o_div = isl_basic_map_offset(bmap, isl_dim_div) - 1;

	for (i = 0; i < n_div; ++i) {
		known = isl_basic_map_div_is_known(bmap, i);
		if (known < 0)
			return isl_basic_map_free(bmap);
		if (known)
			continue;
		bmap = remove_dependent_vars(bmap, o_div + i);
		bmap = isl_basic_map_drop_constraints_involving_dims(bmap,
							    isl_dim_div, i, 1);
		if (!bmap)
			return NULL;
		n_div = isl_basic_map_dim(bmap, isl_dim_div);
		i = -1;
	}

	return bmap;
}

/* Remove all constraints from "map" that reference any unknown local
 * variables (directly or indirectly).
 *
 * Since constraints may get dropped from the basic maps,
 * they may no longer be disjoint from each other.
 */
__isl_give isl_map *isl_map_drop_constraint_involving_unknown_divs(
	__isl_take isl_map *map)
{
	int i;
	isl_bool known;

	known = isl_map_divs_known(map);
	if (known < 0)
		return isl_map_free(map);
	if (known)
		return map;

	map = isl_map_cow(map);
	if (!map)
		return NULL;

	for (i = 0; i < map->n; ++i) {
		map->p[i] =
		    isl_basic_map_drop_constraint_involving_unknown_divs(
								    map->p[i]);
		if (!map->p[i])
			return isl_map_free(map);
	}

	if (map->n > 1)
		ISL_F_CLR(map, ISL_MAP_DISJOINT);

	return map;
}

/* Don't assume equalities are in order, because align_divs
 * may have changed the order of the divs.
 */
static void compute_elimination_index(struct isl_basic_map *bmap, int *elim)
{
	int d, i;
	unsigned total;

	total = isl_space_dim(bmap->dim, isl_dim_all);
	for (d = 0; d < total; ++d)
		elim[d] = -1;
	for (i = 0; i < bmap->n_eq; ++i) {
		for (d = total - 1; d >= 0; --d) {
			if (isl_int_is_zero(bmap->eq[i][1+d]))
				continue;
			elim[d] = i;
			break;
		}
	}
}

static void set_compute_elimination_index(struct isl_basic_set *bset, int *elim)
{
	compute_elimination_index((struct isl_basic_map *)bset, elim);
}

static int reduced_using_equalities(isl_int *dst, isl_int *src,
	struct isl_basic_map *bmap, int *elim)
{
	int d;
	int copied = 0;
	unsigned total;

	total = isl_space_dim(bmap->dim, isl_dim_all);
	for (d = total - 1; d >= 0; --d) {
		if (isl_int_is_zero(src[1+d]))
			continue;
		if (elim[d] == -1)
			continue;
		if (!copied) {
			isl_seq_cpy(dst, src, 1 + total);
			copied = 1;
		}
		isl_seq_elim(dst, bmap->eq[elim[d]], 1 + d, 1 + total, NULL);
	}
	return copied;
}

static int set_reduced_using_equalities(isl_int *dst, isl_int *src,
	struct isl_basic_set *bset, int *elim)
{
	return reduced_using_equalities(dst, src,
					(struct isl_basic_map *)bset, elim);
}

static struct isl_basic_set *isl_basic_set_reduce_using_equalities(
	struct isl_basic_set *bset, struct isl_basic_set *context)
{
	int i;
	int *elim;

	if (!bset || !context)
		goto error;

	if (context->n_eq == 0) {
		isl_basic_set_free(context);
		return bset;
	}

	bset = isl_basic_set_cow(bset);
	if (!bset)
		goto error;

	elim = isl_alloc_array(bset->ctx, int, isl_basic_set_n_dim(bset));
	if (!elim)
		goto error;
	set_compute_elimination_index(context, elim);
	for (i = 0; i < bset->n_eq; ++i)
		set_reduced_using_equalities(bset->eq[i], bset->eq[i],
							context, elim);
	for (i = 0; i < bset->n_ineq; ++i)
		set_reduced_using_equalities(bset->ineq[i], bset->ineq[i],
							context, elim);
	isl_basic_set_free(context);
	free(elim);
	bset = isl_basic_set_simplify(bset);
	bset = isl_basic_set_finalize(bset);
	return bset;
error:
	isl_basic_set_free(bset);
	isl_basic_set_free(context);
	return NULL;
}

/* For each inequality in "ineq" that is a shifted (more relaxed)
 * copy of an inequality in "context", mark the corresponding entry
 * in "row" with -1.
 * If an inequality only has a non-negative constant term, then
 * mark it as well.
 */
static isl_stat mark_shifted_constraints(__isl_keep isl_mat *ineq,
	__isl_keep isl_basic_set *context, int *row)
{
	struct isl_constraint_index ci;
	int n_ineq;
	unsigned total;
	int k;

	if (!ineq || !context)
		return isl_stat_error;
	if (context->n_ineq == 0)
		return isl_stat_ok;
	if (setup_constraint_index(&ci, context) < 0)
		return isl_stat_error;

	n_ineq = isl_mat_rows(ineq);
	total = isl_mat_cols(ineq) - 1;
	for (k = 0; k < n_ineq; ++k) {
		int l;
		isl_bool redundant;

		l = isl_seq_first_non_zero(ineq->row[k] + 1, total);
		if (l < 0 && isl_int_is_nonneg(ineq->row[k][0])) {
			row[k] = -1;
			continue;
		}
		redundant = constraint_index_is_redundant(&ci, ineq->row[k]);
		if (redundant < 0)
			goto error;
		if (!redundant)
			continue;
		row[k] = -1;
	}
	constraint_index_free(&ci);
	return isl_stat_ok;
error:
	constraint_index_free(&ci);
	return isl_stat_error;
}

static struct isl_basic_set *remove_shifted_constraints(
	struct isl_basic_set *bset, struct isl_basic_set *context)
{
	struct isl_constraint_index ci;
	int k;

	if (!bset || !context)
		return bset;

	if (context->n_ineq == 0)
		return bset;
	if (setup_constraint_index(&ci, context) < 0)
		return bset;

	for (k = 0; k < bset->n_ineq; ++k) {
		isl_bool redundant;

		redundant = constraint_index_is_redundant(&ci, bset->ineq[k]);
		if (redundant < 0)
			goto error;
		if (!redundant)
			continue;
		bset = isl_basic_set_cow(bset);
		if (!bset)
			goto error;
		isl_basic_set_drop_inequality(bset, k);
		--k;
	}
	constraint_index_free(&ci);
	return bset;
error:
	constraint_index_free(&ci);
	return bset;
}

/* Remove constraints from "bmap" that are identical to constraints
 * in "context" or that are more relaxed (greater constant term).
 *
 * We perform the test for shifted copies on the pure constraints
 * in remove_shifted_constraints.
 */
static __isl_give isl_basic_map *isl_basic_map_remove_shifted_constraints(
	__isl_take isl_basic_map *bmap, __isl_take isl_basic_map *context)
{
	isl_basic_set *bset, *bset_context;

	if (!bmap || !context)
		goto error;

	if (bmap->n_ineq == 0 || context->n_ineq == 0) {
		isl_basic_map_free(context);
		return bmap;
	}

	context = isl_basic_map_align_divs(context, bmap);
	bmap = isl_basic_map_align_divs(bmap, context);

	bset = isl_basic_map_underlying_set(isl_basic_map_copy(bmap));
	bset_context = isl_basic_map_underlying_set(context);
	bset = remove_shifted_constraints(bset, bset_context);
	isl_basic_set_free(bset_context);

	bmap = isl_basic_map_overlying_set(bset, bmap);

	return bmap;
error:
	isl_basic_map_free(bmap);
	isl_basic_map_free(context);
	return NULL;
}

/* Does the (linear part of a) constraint "c" involve any of the "len"
 * "relevant" dimensions?
 */
static int is_related(isl_int *c, int len, int *relevant)
{
	int i;

	for (i = 0; i < len; ++i) {
		if (!relevant[i])
			continue;
		if (!isl_int_is_zero(c[i]))
			return 1;
	}

	return 0;
}

/* Drop constraints from "bset" that do not involve any of
 * the dimensions marked "relevant".
 */
static __isl_give isl_basic_set *drop_unrelated_constraints(
	__isl_take isl_basic_set *bset, int *relevant)
{
	int i, dim;

	dim = isl_basic_set_dim(bset, isl_dim_set);
	for (i = 0; i < dim; ++i)
		if (!relevant[i])
			break;
	if (i >= dim)
		return bset;

	for (i = bset->n_eq - 1; i >= 0; --i)
		if (!is_related(bset->eq[i] + 1, dim, relevant))
			isl_basic_set_drop_equality(bset, i);

	for (i = bset->n_ineq - 1; i >= 0; --i)
		if (!is_related(bset->ineq[i] + 1, dim, relevant))
			isl_basic_set_drop_inequality(bset, i);

	return bset;
}

/* Update the groups in "group" based on the (linear part of a) constraint "c".
 *
 * In particular, for any variable involved in the constraint,
 * find the actual group id from before and replace the group
 * of the corresponding variable by the minimal group of all
 * the variables involved in the constraint considered so far
 * (if this minimum is smaller) or replace the minimum by this group
 * (if the minimum is larger).
 *
 * At the end, all the variables in "c" will (indirectly) point
 * to the minimal of the groups that they referred to originally.
 */
static void update_groups(int dim, int *group, isl_int *c)
{
	int j;
	int min = dim;

	for (j = 0; j < dim; ++j) {
		if (isl_int_is_zero(c[j]))
			continue;
		while (group[j] >= 0 && group[group[j]] != group[j])
			group[j] = group[group[j]];
		if (group[j] == min)
			continue;
		if (group[j] < min) {
			if (min >= 0 && min < dim)
				group[min] = group[j];
			min = group[j];
		} else
			group[group[j]] = min;
	}
}

/* Allocate an array of groups of variables, one for each variable
 * in "context", initialized to zero.
 */
static int *alloc_groups(__isl_keep isl_basic_set *context)
{
	isl_ctx *ctx;
	int dim;

	dim = isl_basic_set_dim(context, isl_dim_set);
	ctx = isl_basic_set_get_ctx(context);
	return isl_calloc_array(ctx, int, dim);
}

/* Drop constraints from "context" that only involve variables that are
 * not related to any of the variables marked with a "-1" in "group".
 *
 * We construct groups of variables that collect variables that
 * (indirectly) appear in some common constraint of "context".
 * Each group is identified by the first variable in the group,
 * except for the special group of variables that was already identified
 * in the input as -1 (or are related to those variables).
 * If group[i] is equal to i (or -1), then the group of i is i (or -1),
 * otherwise the group of i is the group of group[i].
 *
 * We first initialize groups for the remaining variables.
 * Then we iterate over the constraints of "context" and update the
 * group of the variables in the constraint by the smallest group.
 * Finally, we resolve indirect references to groups by running over
 * the variables.
 *
 * After computing the groups, we drop constraints that do not involve
 * any variables in the -1 group.
 */
static __isl_give isl_basic_set *group_and_drop_irrelevant_constraints(
	__isl_take isl_basic_set *context, __isl_take int *group)
{
	int dim;
	int i;
	int last;

	dim = isl_basic_set_dim(context, isl_dim_set);

	last = -1;
	for (i = 0; i < dim; ++i)
		if (group[i] >= 0)
			last = group[i] = i;
	if (last < 0) {
		free(group);
		return context;
	}

	for (i = 0; i < context->n_eq; ++i)
		update_groups(dim, group, context->eq[i] + 1);
	for (i = 0; i < context->n_ineq; ++i)
		update_groups(dim, group, context->ineq[i] + 1);

	for (i = 0; i < dim; ++i)
		if (group[i] >= 0)
			group[i] = group[group[i]];

	for (i = 0; i < dim; ++i)
		group[i] = group[i] == -1;

	context = drop_unrelated_constraints(context, group);

	free(group);
	return context;
}

/* Drop constraints from "context" that are irrelevant for computing
 * the gist of "bset".
 *
 * In particular, drop constraints in variables that are not related
 * to any of the variables involved in the constraints of "bset"
 * in the sense that there is no sequence of constraints that connects them.
 *
 * We first mark all variables that appear in "bset" as belonging
 * to a "-1" group and then continue with group_and_drop_irrelevant_constraints.
 */
static __isl_give isl_basic_set *drop_irrelevant_constraints(
	__isl_take isl_basic_set *context, __isl_keep isl_basic_set *bset)
{
	int *group;
	int dim;
	int i, j;

	if (!context || !bset)
		return isl_basic_set_free(context);

	group = alloc_groups(context);

	if (!group)
		return isl_basic_set_free(context);

	dim = isl_basic_set_dim(bset, isl_dim_set);
	for (i = 0; i < dim; ++i) {
		for (j = 0; j < bset->n_eq; ++j)
			if (!isl_int_is_zero(bset->eq[j][1 + i]))
				break;
		if (j < bset->n_eq) {
			group[i] = -1;
			continue;
		}
		for (j = 0; j < bset->n_ineq; ++j)
			if (!isl_int_is_zero(bset->ineq[j][1 + i]))
				break;
		if (j < bset->n_ineq)
			group[i] = -1;
	}

	return group_and_drop_irrelevant_constraints(context, group);
}

/* Drop constraints from "context" that are irrelevant for computing
 * the gist of the inequalities "ineq".
 * Inequalities in "ineq" for which the corresponding element of row
 * is set to -1 have already been marked for removal and should be ignored.
 *
 * In particular, drop constraints in variables that are not related
 * to any of the variables involved in "ineq"
 * in the sense that there is no sequence of constraints that connects them.
 *
 * We first mark all variables that appear in "bset" as belonging
 * to a "-1" group and then continue with group_and_drop_irrelevant_constraints.
 */
static __isl_give isl_basic_set *drop_irrelevant_constraints_marked(
	__isl_take isl_basic_set *context, __isl_keep isl_mat *ineq, int *row)
{
	int *group;
	int dim;
	int i, j, n;

	if (!context || !ineq)
		return isl_basic_set_free(context);

	group = alloc_groups(context);

	if (!group)
		return isl_basic_set_free(context);

	dim = isl_basic_set_dim(context, isl_dim_set);
	n = isl_mat_rows(ineq);
	for (i = 0; i < dim; ++i) {
		for (j = 0; j < n; ++j) {
			if (row[j] < 0)
				continue;
			if (!isl_int_is_zero(ineq->row[j][1 + i]))
				break;
		}
		if (j < n)
			group[i] = -1;
	}

	return group_and_drop_irrelevant_constraints(context, group);
}

/* Do all "n" entries of "row" contain a negative value?
 */
static int all_neg(int *row, int n)
{
	int i;

	for (i = 0; i < n; ++i)
		if (row[i] >= 0)
			return 0;

	return 1;
}

/* Update the inequalities in "bset" based on the information in "row"
 * and "tab".
 *
 * In particular, the array "row" contains either -1, meaning that
 * the corresponding inequality of "bset" is redundant, or the index
 * of an inequality in "tab".
 *
 * If the row entry is -1, then drop the inequality.
 * Otherwise, if the constraint is marked redundant in the tableau,
 * then drop the inequality.  Similarly, if it is marked as an equality
 * in the tableau, then turn the inequality into an equality and
 * perform Gaussian elimination.
 */
static __isl_give isl_basic_set *update_ineq(__isl_take isl_basic_set *bset,
	__isl_keep int *row, struct isl_tab *tab)
{
	int i;
	unsigned n_ineq;
	unsigned n_eq;
	int found_equality = 0;

	if (!bset)
		return NULL;
	if (tab && tab->empty)
		return isl_basic_set_set_to_empty(bset);

	n_ineq = bset->n_ineq;
	for (i = n_ineq - 1; i >= 0; --i) {
		if (row[i] < 0) {
			if (isl_basic_set_drop_inequality(bset, i) < 0)
				return isl_basic_set_free(bset);
			continue;
		}
		if (!tab)
			continue;
		n_eq = tab->n_eq;
		if (isl_tab_is_equality(tab, n_eq + row[i])) {
			isl_basic_map_inequality_to_equality(bset, i);
			found_equality = 1;
		} else if (isl_tab_is_redundant(tab, n_eq + row[i])) {
			if (isl_basic_set_drop_inequality(bset, i) < 0)
				return isl_basic_set_free(bset);
		}
	}

	if (found_equality)
		bset = isl_basic_set_gauss(bset, NULL);
	bset = isl_basic_set_finalize(bset);
	return bset;
}

/* Update the inequalities in "bset" based on the information in "row"
 * and "tab" and free all arguments (other than "bset").
 */
static __isl_give isl_basic_set *update_ineq_free(
	__isl_take isl_basic_set *bset, __isl_take isl_mat *ineq,
	__isl_take isl_basic_set *context, __isl_take int *row,
	struct isl_tab *tab)
{
	isl_mat_free(ineq);
	isl_basic_set_free(context);

	bset = update_ineq(bset, row, tab);

	free(row);
	isl_tab_free(tab);
	return bset;
}

/* Remove all information from bset that is redundant in the context
 * of context.
 * "ineq" contains the (possibly transformed) inequalities of "bset",
 * in the same order.
 * The (explicit) equalities of "bset" are assumed to have been taken
 * into account by the transformation such that only the inequalities
 * are relevant.
 * "context" is assumed not to be empty.
 *
 * "row" keeps track of the constraint index of a "bset" inequality in "tab".
 * A value of -1 means that the inequality is obviously redundant and may
 * not even appear in  "tab".
 *
 * We first mark the inequalities of "bset"
 * that are obviously redundant with respect to some inequality in "context".
 * Then we remove those constraints from "context" that have become
 * irrelevant for computing the gist of "bset".
 * Note that this removal of constraints cannot be replaced by
 * a factorization because factors in "bset" may still be connected
 * to each other through constraints in "context".
 *
 * If there are any inequalities left, we construct a tableau for
 * the context and then add the inequalities of "bset".
 * Before adding these inequalities, we freeze all constraints such that
 * they won't be considered redundant in terms of the constraints of "bset".
 * Then we detect all redundant constraints (among the
 * constraints that weren't frozen), first by checking for redundancy in the
 * the tableau and then by checking if replacing a constraint by its negation
 * would lead to an empty set.  This last step is fairly expensive
 * and could be optimized by more reuse of the tableau.
 * Finally, we update bset according to the results.
 */
static __isl_give isl_basic_set *uset_gist_full(__isl_take isl_basic_set *bset,
	__isl_take isl_mat *ineq, __isl_take isl_basic_set *context)
{
	int i, r;
	int *row = NULL;
	isl_ctx *ctx;
	isl_basic_set *combined = NULL;
	struct isl_tab *tab = NULL;
	unsigned n_eq, context_ineq;
	unsigned total;

	if (!bset || !ineq || !context)
		goto error;

	if (bset->n_ineq == 0 || isl_basic_set_is_universe(context)) {
		isl_basic_set_free(context);
		isl_mat_free(ineq);
		return bset;
	}

	ctx = isl_basic_set_get_ctx(context);
	row = isl_calloc_array(ctx, int, bset->n_ineq);
	if (!row)
		goto error;

	if (mark_shifted_constraints(ineq, context, row) < 0)
		goto error;
	if (all_neg(row, bset->n_ineq))
		return update_ineq_free(bset, ineq, context, row, NULL);

	context = drop_irrelevant_constraints_marked(context, ineq, row);
	if (!context)
		goto error;
	if (isl_basic_set_is_universe(context))
		return update_ineq_free(bset, ineq, context, row, NULL);

	n_eq = context->n_eq;
	context_ineq = context->n_ineq;
	combined = isl_basic_set_cow(isl_basic_set_copy(context));
	combined = isl_basic_set_extend_constraints(combined, 0, bset->n_ineq);
	tab = isl_tab_from_basic_set(combined, 0);
	for (i = 0; i < context_ineq; ++i)
		if (isl_tab_freeze_constraint(tab, n_eq + i) < 0)
			goto error;
	if (isl_tab_extend_cons(tab, bset->n_ineq) < 0)
		goto error;
	r = context_ineq;
	for (i = 0; i < bset->n_ineq; ++i) {
		if (row[i] < 0)
			continue;
		combined = isl_basic_set_add_ineq(combined, ineq->row[i]);
		if (isl_tab_add_ineq(tab, ineq->row[i]) < 0)
			goto error;
		row[i] = r++;
	}
	if (isl_tab_detect_implicit_equalities(tab) < 0)
		goto error;
	if (isl_tab_detect_redundant(tab) < 0)
		goto error;
	total = isl_basic_set_total_dim(bset);
	for (i = bset->n_ineq - 1; i >= 0; --i) {
		isl_basic_set *test;
		int is_empty;

		if (row[i] < 0)
			continue;
		r = row[i];
		if (tab->con[n_eq + r].is_redundant)
			continue;
		test = isl_basic_set_dup(combined);
		if (isl_inequality_negate(test, r) < 0)
			test = isl_basic_set_free(test);
		test = isl_basic_set_update_from_tab(test, tab);
		is_empty = isl_basic_set_is_empty(test);
		isl_basic_set_free(test);
		if (is_empty < 0)
			goto error;
		if (is_empty)
			tab->con[n_eq + r].is_redundant = 1;
	}
	bset = update_ineq_free(bset, ineq, context, row, tab);
	if (bset) {
		ISL_F_SET(bset, ISL_BASIC_SET_NO_IMPLICIT);
		ISL_F_SET(bset, ISL_BASIC_SET_NO_REDUNDANT);
	}

	isl_basic_set_free(combined);
	return bset;
error:
	free(row);
	isl_mat_free(ineq);
	isl_tab_free(tab);
	isl_basic_set_free(combined);
	isl_basic_set_free(context);
	isl_basic_set_free(bset);
	return NULL;
}

/* Extract the inequalities of "bset" as an isl_mat.
 */
static __isl_give isl_mat *extract_ineq(__isl_keep isl_basic_set *bset)
{
	unsigned total;
	isl_ctx *ctx;
	isl_mat *ineq;

	if (!bset)
		return NULL;

	ctx = isl_basic_set_get_ctx(bset);
	total = isl_basic_set_total_dim(bset);
	ineq = isl_mat_sub_alloc6(ctx, bset->ineq, 0, bset->n_ineq,
				    0, 1 + total);

	return ineq;
}

/* Remove all information from "bset" that is redundant in the context
 * of "context", for the case where both "bset" and "context" are
 * full-dimensional.
 */
static __isl_give isl_basic_set *uset_gist_uncompressed(
	__isl_take isl_basic_set *bset, __isl_take isl_basic_set *context)
{
	isl_mat *ineq;

	ineq = extract_ineq(bset);
	return uset_gist_full(bset, ineq, context);
}

/* Remove all information from "bset" that is redundant in the context
 * of "context", for the case where the combined equalities of
 * "bset" and "context" allow for a compression that can be obtained
 * by preapplication of "T".
 *
 * "bset" itself is not transformed by "T".  Instead, the inequalities
 * are extracted from "bset" and those are transformed by "T".
 * uset_gist_full then determines which of the transformed inequalities
 * are redundant with respect to the transformed "context" and removes
 * the corresponding inequalities from "bset".
 *
 * After preapplying "T" to the inequalities, any common factor is
 * removed from the coefficients.  If this results in a tightening
 * of the constant term, then the same tightening is applied to
 * the corresponding untransformed inequality in "bset".
 * That is, if after plugging in T, a constraint f(x) >= 0 is of the form
 *
 *	g f'(x) + r >= 0
 *
 * with 0 <= r < g, then it is equivalent to
 *
 *	f'(x) >= 0
 *
 * This means that f(x) >= 0 is equivalent to f(x) - r >= 0 in the affine
 * subspace compressed by T since the latter would be transformed to
 *
 *	g f'(x) >= 0
 */
static __isl_give isl_basic_set *uset_gist_compressed(
	__isl_take isl_basic_set *bset, __isl_take isl_basic_set *context,
	__isl_take isl_mat *T)
{
	isl_ctx *ctx;
	isl_mat *ineq;
	int i, n_row, n_col;
	isl_int rem;

	ineq = extract_ineq(bset);
	ineq = isl_mat_product(ineq, isl_mat_copy(T));
	context = isl_basic_set_preimage(context, T);

	if (!ineq || !context)
		goto error;
	if (isl_basic_set_plain_is_empty(context)) {
		isl_mat_free(ineq);
		isl_basic_set_free(context);
		return isl_basic_set_set_to_empty(bset);
	}

	ctx = isl_mat_get_ctx(ineq);
	n_row = isl_mat_rows(ineq);
	n_col = isl_mat_cols(ineq);
	isl_int_init(rem);
	for (i = 0; i < n_row; ++i) {
		isl_seq_gcd(ineq->row[i] + 1, n_col - 1, &ctx->normalize_gcd);
		if (isl_int_is_zero(ctx->normalize_gcd))
			continue;
		if (isl_int_is_one(ctx->normalize_gcd))
			continue;
		isl_seq_scale_down(ineq->row[i] + 1, ineq->row[i] + 1,
				    ctx->normalize_gcd, n_col - 1);
		isl_int_fdiv_r(rem, ineq->row[i][0], ctx->normalize_gcd);
		isl_int_fdiv_q(ineq->row[i][0],
				ineq->row[i][0], ctx->normalize_gcd);
		if (isl_int_is_zero(rem))
			continue;
		bset = isl_basic_set_cow(bset);
		if (!bset)
			break;
		isl_int_sub(bset->ineq[i][0], bset->ineq[i][0], rem);
	}
	isl_int_clear(rem);

	return uset_gist_full(bset, ineq, context);
error:
	isl_mat_free(ineq);
	isl_basic_set_free(context);
	isl_basic_set_free(bset);
	return NULL;
}

/* Project "bset" onto the variables that are involved in "template".
 */
static __isl_give isl_basic_set *project_onto_involved(
	__isl_take isl_basic_set *bset, __isl_keep isl_basic_set *template)
{
	int i, n;

	if (!bset || !template)
		return isl_basic_set_free(bset);

	n = isl_basic_set_dim(template, isl_dim_set);

	for (i = 0; i < n; ++i) {
		isl_bool involved;

		involved = isl_basic_set_involves_dims(template,
							isl_dim_set, i, 1);
		if (involved < 0)
			return isl_basic_set_free(bset);
		if (involved)
			continue;
		bset = isl_basic_set_eliminate_vars(bset, i, 1);
	}

	return bset;
}

/* Remove all information from bset that is redundant in the context
 * of context.  In particular, equalities that are linear combinations
 * of those in context are removed.  Then the inequalities that are
 * redundant in the context of the equalities and inequalities of
 * context are removed.
 *
 * First of all, we drop those constraints from "context"
 * that are irrelevant for computing the gist of "bset".
 * Alternatively, we could factorize the intersection of "context" and "bset".
 *
 * We first compute the intersection of the integer affine hulls
 * of "bset" and "context",
 * compute the gist inside this intersection and then reduce
 * the constraints with respect to the equalities of the context
 * that only involve variables already involved in the input.
 *
 * If two constraints are mutually redundant, then uset_gist_full
 * will remove the second of those constraints.  We therefore first
 * sort the constraints so that constraints not involving existentially
 * quantified variables are given precedence over those that do.
 * We have to perform this sorting before the variable compression,
 * because that may effect the order of the variables.
 */
static __isl_give isl_basic_set *uset_gist(__isl_take isl_basic_set *bset,
	__isl_take isl_basic_set *context)
{
	isl_mat *eq;
	isl_mat *T;
	isl_basic_set *aff;
	isl_basic_set *aff_context;
	unsigned total;

	if (!bset || !context)
		goto error;

	context = drop_irrelevant_constraints(context, bset);

	bset = isl_basic_set_detect_equalities(bset);
	aff = isl_basic_set_copy(bset);
	aff = isl_basic_set_plain_affine_hull(aff);
	context = isl_basic_set_detect_equalities(context);
	aff_context = isl_basic_set_copy(context);
	aff_context = isl_basic_set_plain_affine_hull(aff_context);
	aff = isl_basic_set_intersect(aff, aff_context);
	if (!aff)
		goto error;
	if (isl_basic_set_plain_is_empty(aff)) {
		isl_basic_set_free(bset);
		isl_basic_set_free(context);
		return aff;
	}
	bset = isl_basic_set_sort_constraints(bset);
	if (aff->n_eq == 0) {
		isl_basic_set_free(aff);
		return uset_gist_uncompressed(bset, context);
	}
	total = isl_basic_set_total_dim(bset);
	eq = isl_mat_sub_alloc6(bset->ctx, aff->eq, 0, aff->n_eq, 0, 1 + total);
	eq = isl_mat_cow(eq);
	T = isl_mat_variable_compression(eq, NULL);
	isl_basic_set_free(aff);
	if (T && T->n_col == 0) {
		isl_mat_free(T);
		isl_basic_set_free(context);
		return isl_basic_set_set_to_empty(bset);
	}

	aff_context = isl_basic_set_affine_hull(isl_basic_set_copy(context));
	aff_context = project_onto_involved(aff_context, bset);

	bset = uset_gist_compressed(bset, context, T);
	bset = isl_basic_set_reduce_using_equalities(bset, aff_context);

	if (bset) {
		ISL_F_SET(bset, ISL_BASIC_SET_NO_IMPLICIT);
		ISL_F_SET(bset, ISL_BASIC_SET_NO_REDUNDANT);
	}

	return bset;
error:
	isl_basic_set_free(bset);
	isl_basic_set_free(context);
	return NULL;
}

/* Return a basic map that has the same intersection with "context" as "bmap"
 * and that is as "simple" as possible.
 *
 * The core computation is performed on the pure constraints.
 * When we add back the meaning of the integer divisions, we need
 * to (re)introduce the div constraints.  If we happen to have
 * discovered that some of these integer divisions are equal to
 * some affine combination of other variables, then these div
 * constraints may end up getting simplified in terms of the equalities,
 * resulting in extra inequalities on the other variables that
 * may have been removed already or that may not even have been
 * part of the input.  We try and remove those constraints of
 * this form that are most obviously redundant with respect to
 * the context.  We also remove those div constraints that are
 * redundant with respect to the other constraints in the result.
 */
struct isl_basic_map *isl_basic_map_gist(struct isl_basic_map *bmap,
	struct isl_basic_map *context)
{
	isl_basic_set *bset, *eq;
	isl_basic_map *eq_bmap;
	unsigned total, n_div, extra, n_eq, n_ineq;

	if (!bmap || !context)
		goto error;

	if (isl_basic_map_is_universe(bmap)) {
		isl_basic_map_free(context);
		return bmap;
	}
	if (isl_basic_map_plain_is_empty(context)) {
		isl_space *space = isl_basic_map_get_space(bmap);
		isl_basic_map_free(bmap);
		isl_basic_map_free(context);
		return isl_basic_map_universe(space);
	}
	if (isl_basic_map_plain_is_empty(bmap)) {
		isl_basic_map_free(context);
		return bmap;
	}

	bmap = isl_basic_map_remove_redundancies(bmap);
	context = isl_basic_map_remove_redundancies(context);
	if (!context)
		goto error;

	context = isl_basic_map_align_divs(context, bmap);
	n_div = isl_basic_map_dim(context, isl_dim_div);
	total = isl_basic_map_dim(bmap, isl_dim_all);
	extra = n_div - isl_basic_map_dim(bmap, isl_dim_div);

	bset = isl_basic_map_underlying_set(isl_basic_map_copy(bmap));
	bset = isl_basic_set_add_dims(bset, isl_dim_set, extra);
	bset = uset_gist(bset,
		    isl_basic_map_underlying_set(isl_basic_map_copy(context)));
	bset = isl_basic_set_project_out(bset, isl_dim_set, total, extra);

	if (!bset || bset->n_eq == 0 || n_div == 0 ||
	    isl_basic_set_plain_is_empty(bset)) {
		isl_basic_map_free(context);
		return isl_basic_map_overlying_set(bset, bmap);
	}

	n_eq = bset->n_eq;
	n_ineq = bset->n_ineq;
	eq = isl_basic_set_copy(bset);
	eq = isl_basic_set_cow(eq);
	if (isl_basic_set_free_inequality(eq, n_ineq) < 0)
		eq = isl_basic_set_free(eq);
	if (isl_basic_set_free_equality(bset, n_eq) < 0)
		bset = isl_basic_set_free(bset);

	eq_bmap = isl_basic_map_overlying_set(eq, isl_basic_map_copy(bmap));
	eq_bmap = isl_basic_map_remove_shifted_constraints(eq_bmap, context);
	bmap = isl_basic_map_overlying_set(bset, bmap);
	bmap = isl_basic_map_intersect(bmap, eq_bmap);
	bmap = isl_basic_map_remove_redundancies(bmap);

	return bmap;
error:
	isl_basic_map_free(bmap);
	isl_basic_map_free(context);
	return NULL;
}

/*
 * Assumes context has no implicit divs.
 */
__isl_give isl_map *isl_map_gist_basic_map(__isl_take isl_map *map,
	__isl_take isl_basic_map *context)
{
	int i;

	if (!map || !context)
		goto error;

	if (isl_basic_map_plain_is_empty(context)) {
		isl_space *space = isl_map_get_space(map);
		isl_map_free(map);
		isl_basic_map_free(context);
		return isl_map_universe(space);
	}

	context = isl_basic_map_remove_redundancies(context);
	map = isl_map_cow(map);
	if (!map || !context)
		goto error;
	isl_assert(map->ctx, isl_space_is_equal(map->dim, context->dim), goto error);
	map = isl_map_compute_divs(map);
	if (!map)
		goto error;
	for (i = map->n - 1; i >= 0; --i) {
		map->p[i] = isl_basic_map_gist(map->p[i],
						isl_basic_map_copy(context));
		if (!map->p[i])
			goto error;
		if (isl_basic_map_plain_is_empty(map->p[i])) {
			isl_basic_map_free(map->p[i]);
			if (i != map->n - 1)
				map->p[i] = map->p[map->n - 1];
			map->n--;
		}
	}
	isl_basic_map_free(context);
	ISL_F_CLR(map, ISL_MAP_NORMALIZED);
	return map;
error:
	isl_map_free(map);
	isl_basic_map_free(context);
	return NULL;
}

/* Drop all inequalities from "bmap" that also appear in "context".
 * "context" is assumed to have only known local variables and
 * the initial local variables of "bmap" are assumed to be the same
 * as those of "context".
 * The constraints of both "bmap" and "context" are assumed
 * to have been sorted using isl_basic_map_sort_constraints.
 *
 * Run through the inequality constraints of "bmap" and "context"
 * in sorted order.
 * If a constraint of "bmap" involves variables not in "context",
 * then it cannot appear in "context".
 * If a matching constraint is found, it is removed from "bmap".
 */
static __isl_give isl_basic_map *drop_inequalities(
	__isl_take isl_basic_map *bmap, __isl_keep isl_basic_map *context)
{
	int i1, i2;
	unsigned total, extra;

	if (!bmap || !context)
		return isl_basic_map_free(bmap);

	total = isl_basic_map_total_dim(context);
	extra = isl_basic_map_total_dim(bmap) - total;

	i1 = bmap->n_ineq - 1;
	i2 = context->n_ineq - 1;
	while (bmap && i1 >= 0 && i2 >= 0) {
		int cmp;

		if (isl_seq_first_non_zero(bmap->ineq[i1] + 1 + total,
					    extra) != -1) {
			--i1;
			continue;
		}
		cmp = isl_basic_map_constraint_cmp(context, bmap->ineq[i1],
							context->ineq[i2]);
		if (cmp < 0) {
			--i2;
			continue;
		}
		if (cmp > 0) {
			--i1;
			continue;
		}
		if (isl_int_eq(bmap->ineq[i1][0], context->ineq[i2][0])) {
			bmap = isl_basic_map_cow(bmap);
			if (isl_basic_map_drop_inequality(bmap, i1) < 0)
				bmap = isl_basic_map_free(bmap);
		}
		--i1;
		--i2;
	}

	return bmap;
}

/* Drop all equalities from "bmap" that also appear in "context".
 * "context" is assumed to have only known local variables and
 * the initial local variables of "bmap" are assumed to be the same
 * as those of "context".
 *
 * Run through the equality constraints of "bmap" and "context"
 * in sorted order.
 * If a constraint of "bmap" involves variables not in "context",
 * then it cannot appear in "context".
 * If a matching constraint is found, it is removed from "bmap".
 */
static __isl_give isl_basic_map *drop_equalities(
	__isl_take isl_basic_map *bmap, __isl_keep isl_basic_map *context)
{
	int i1, i2;
	unsigned total, extra;

	if (!bmap || !context)
		return isl_basic_map_free(bmap);

	total = isl_basic_map_total_dim(context);
	extra = isl_basic_map_total_dim(bmap) - total;

	i1 = bmap->n_eq - 1;
	i2 = context->n_eq - 1;

	while (bmap && i1 >= 0 && i2 >= 0) {
		int last1, last2;

		if (isl_seq_first_non_zero(bmap->eq[i1] + 1 + total,
					    extra) != -1)
			break;
		last1 = isl_seq_last_non_zero(bmap->eq[i1] + 1, total);
		last2 = isl_seq_last_non_zero(context->eq[i2] + 1, total);
		if (last1 > last2) {
			--i2;
			continue;
		}
		if (last1 < last2) {
			--i1;
			continue;
		}
		if (isl_seq_eq(bmap->eq[i1], context->eq[i2], 1 + total)) {
			bmap = isl_basic_map_cow(bmap);
			if (isl_basic_map_drop_equality(bmap, i1) < 0)
				bmap = isl_basic_map_free(bmap);
		}
		--i1;
		--i2;
	}

	return bmap;
}

/* Remove the constraints in "context" from "bmap".
 * "context" is assumed to have explicit representations
 * for all local variables.
 *
 * First align the divs of "bmap" to those of "context" and
 * sort the constraints.  Then drop all constraints from "bmap"
 * that appear in "context".
 */
__isl_give isl_basic_map *isl_basic_map_plain_gist(
	__isl_take isl_basic_map *bmap, __isl_take isl_basic_map *context)
{
	isl_bool done, known;

	done = isl_basic_map_is_universe(context);
	if (done == isl_bool_false)
		done = isl_basic_map_is_universe(bmap);
	if (done == isl_bool_false)
		done = isl_basic_map_plain_is_empty(context);
	if (done == isl_bool_false)
		done = isl_basic_map_plain_is_empty(bmap);
	if (done < 0)
		goto error;
	if (done) {
		isl_basic_map_free(context);
		return bmap;
	}
	known = isl_basic_map_divs_known(context);
	if (known < 0)
		goto error;
	if (!known)
		isl_die(isl_basic_map_get_ctx(bmap), isl_error_invalid,
			"context has unknown divs", goto error);

	bmap = isl_basic_map_align_divs(bmap, context);
	bmap = isl_basic_map_gauss(bmap, NULL);
	bmap = isl_basic_map_sort_constraints(bmap);
	context = isl_basic_map_sort_constraints(context);

	bmap = drop_inequalities(bmap, context);
	bmap = drop_equalities(bmap, context);

	isl_basic_map_free(context);
	bmap = isl_basic_map_finalize(bmap);
	return bmap;
error:
	isl_basic_map_free(bmap);
	isl_basic_map_free(context);
	return NULL;
}

/* Replace "map" by the disjunct at position "pos" and free "context".
 */
static __isl_give isl_map *replace_by_disjunct(__isl_take isl_map *map,
	int pos, __isl_take isl_basic_map *context)
{
	isl_basic_map *bmap;

	bmap = isl_basic_map_copy(map->p[pos]);
	isl_map_free(map);
	isl_basic_map_free(context);
	return isl_map_from_basic_map(bmap);
}

/* Remove the constraints in "context" from "map".
 * If any of the disjuncts in the result turns out to be the universe,
 * the return this universe.
 * "context" is assumed to have explicit representations
 * for all local variables.
 */
__isl_give isl_map *isl_map_plain_gist_basic_map(__isl_take isl_map *map,
	__isl_take isl_basic_map *context)
{
	int i;
	isl_bool univ, known;

	univ = isl_basic_map_is_universe(context);
	if (univ < 0)
		goto error;
	if (univ) {
		isl_basic_map_free(context);
		return map;
	}
	known = isl_basic_map_divs_known(context);
	if (known < 0)
		goto error;
	if (!known)
		isl_die(isl_map_get_ctx(map), isl_error_invalid,
			"context has unknown divs", goto error);

	map = isl_map_cow(map);
	if (!map)
		goto error;
	for (i = 0; i < map->n; ++i) {
		map->p[i] = isl_basic_map_plain_gist(map->p[i],
						isl_basic_map_copy(context));
		univ = isl_basic_map_is_universe(map->p[i]);
		if (univ < 0)
			goto error;
		if (univ && map->n > 1)
			return replace_by_disjunct(map, i, context);
	}

	isl_basic_map_free(context);
	ISL_F_CLR(map, ISL_MAP_NORMALIZED);
	if (map->n > 1)
		ISL_F_CLR(map, ISL_MAP_DISJOINT);
	return map;
error:
	isl_map_free(map);
	isl_basic_map_free(context);
	return NULL;
}

/* Return a map that has the same intersection with "context" as "map"
 * and that is as "simple" as possible.
 *
 * If "map" is already the universe, then we cannot make it any simpler.
 * Similarly, if "context" is the universe, then we cannot exploit it
 * to simplify "map"
 * If "map" and "context" are identical to each other, then we can
 * return the corresponding universe.
 *
 * If none of these cases apply, we have to work a bit harder.
 * During this computation, we make use of a single disjunct context,
 * so if the original context consists of more than one disjunct
 * then we need to approximate the context by a single disjunct set.
 * Simply taking the simple hull may drop constraints that are
 * only implicitly available in each disjunct.  We therefore also
 * look for constraints among those defining "map" that are valid
 * for the context.  These can then be used to simplify away
 * the corresponding constraints in "map".
 */
static __isl_give isl_map *map_gist(__isl_take isl_map *map,
	__isl_take isl_map *context)
{
	int equal;
	int is_universe;
	isl_basic_map *hull;

	is_universe = isl_map_plain_is_universe(map);
	if (is_universe >= 0 && !is_universe)
		is_universe = isl_map_plain_is_universe(context);
	if (is_universe < 0)
		goto error;
	if (is_universe) {
		isl_map_free(context);
		return map;
	}

	equal = isl_map_plain_is_equal(map, context);
	if (equal < 0)
		goto error;
	if (equal) {
		isl_map *res = isl_map_universe(isl_map_get_space(map));
		isl_map_free(map);
		isl_map_free(context);
		return res;
	}

	context = isl_map_compute_divs(context);
	if (!context)
		goto error;
	if (isl_map_n_basic_map(context) == 1) {
		hull = isl_map_simple_hull(context);
	} else {
		isl_ctx *ctx;
		isl_map_list *list;

		ctx = isl_map_get_ctx(map);
		list = isl_map_list_alloc(ctx, 2);
		list = isl_map_list_add(list, isl_map_copy(context));
		list = isl_map_list_add(list, isl_map_copy(map));
		hull = isl_map_unshifted_simple_hull_from_map_list(context,
								    list);
	}
	return isl_map_gist_basic_map(map, hull);
error:
	isl_map_free(map);
	isl_map_free(context);
	return NULL;
}

__isl_give isl_map *isl_map_gist(__isl_take isl_map *map,
	__isl_take isl_map *context)
{
	return isl_map_align_params_map_map_and(map, context, &map_gist);
}

struct isl_basic_set *isl_basic_set_gist(struct isl_basic_set *bset,
						struct isl_basic_set *context)
{
	return (struct isl_basic_set *)isl_basic_map_gist(
		(struct isl_basic_map *)bset, (struct isl_basic_map *)context);
}

__isl_give isl_set *isl_set_gist_basic_set(__isl_take isl_set *set,
	__isl_take isl_basic_set *context)
{
	return (struct isl_set *)isl_map_gist_basic_map((struct isl_map *)set,
					(struct isl_basic_map *)context);
}

__isl_give isl_set *isl_set_gist_params_basic_set(__isl_take isl_set *set,
	__isl_take isl_basic_set *context)
{
	isl_space *space = isl_set_get_space(set);
	isl_basic_set *dom_context = isl_basic_set_universe(space);
	dom_context = isl_basic_set_intersect_params(dom_context, context);
	return isl_set_gist_basic_set(set, dom_context);
}

__isl_give isl_set *isl_set_gist(__isl_take isl_set *set,
	__isl_take isl_set *context)
{
	return (struct isl_set *)isl_map_gist((struct isl_map *)set,
					(struct isl_map *)context);
}

/* Compute the gist of "bmap" with respect to the constraints "context"
 * on the domain.
 */
__isl_give isl_basic_map *isl_basic_map_gist_domain(
	__isl_take isl_basic_map *bmap, __isl_take isl_basic_set *context)
{
	isl_space *space = isl_basic_map_get_space(bmap);
	isl_basic_map *bmap_context = isl_basic_map_universe(space);

	bmap_context = isl_basic_map_intersect_domain(bmap_context, context);
	return isl_basic_map_gist(bmap, bmap_context);
}

__isl_give isl_map *isl_map_gist_domain(__isl_take isl_map *map,
	__isl_take isl_set *context)
{
	isl_map *map_context = isl_map_universe(isl_map_get_space(map));
	map_context = isl_map_intersect_domain(map_context, context);
	return isl_map_gist(map, map_context);
}

__isl_give isl_map *isl_map_gist_range(__isl_take isl_map *map,
	__isl_take isl_set *context)
{
	isl_map *map_context = isl_map_universe(isl_map_get_space(map));
	map_context = isl_map_intersect_range(map_context, context);
	return isl_map_gist(map, map_context);
}

__isl_give isl_map *isl_map_gist_params(__isl_take isl_map *map,
	__isl_take isl_set *context)
{
	isl_map *map_context = isl_map_universe(isl_map_get_space(map));
	map_context = isl_map_intersect_params(map_context, context);
	return isl_map_gist(map, map_context);
}

__isl_give isl_set *isl_set_gist_params(__isl_take isl_set *set,
	__isl_take isl_set *context)
{
	return isl_map_gist_params(set, context);
}

/* Quick check to see if two basic maps are disjoint.
 * In particular, we reduce the equalities and inequalities of
 * one basic map in the context of the equalities of the other
 * basic map and check if we get a contradiction.
 */
isl_bool isl_basic_map_plain_is_disjoint(__isl_keep isl_basic_map *bmap1,
	__isl_keep isl_basic_map *bmap2)
{
	struct isl_vec *v = NULL;
	int *elim = NULL;
	unsigned total;
	int i;

	if (!bmap1 || !bmap2)
		return isl_bool_error;
	isl_assert(bmap1->ctx, isl_space_is_equal(bmap1->dim, bmap2->dim),
			return isl_bool_error);
	if (bmap1->n_div || bmap2->n_div)
		return isl_bool_false;
	if (!bmap1->n_eq && !bmap2->n_eq)
		return isl_bool_false;

	total = isl_space_dim(bmap1->dim, isl_dim_all);
	if (total == 0)
		return isl_bool_false;
	v = isl_vec_alloc(bmap1->ctx, 1 + total);
	if (!v)
		goto error;
	elim = isl_alloc_array(bmap1->ctx, int, total);
	if (!elim)
		goto error;
	compute_elimination_index(bmap1, elim);
	for (i = 0; i < bmap2->n_eq; ++i) {
		int reduced;
		reduced = reduced_using_equalities(v->block.data, bmap2->eq[i],
							bmap1, elim);
		if (reduced && !isl_int_is_zero(v->block.data[0]) &&
		    isl_seq_first_non_zero(v->block.data + 1, total) == -1)
			goto disjoint;
	}
	for (i = 0; i < bmap2->n_ineq; ++i) {
		int reduced;
		reduced = reduced_using_equalities(v->block.data,
						bmap2->ineq[i], bmap1, elim);
		if (reduced && isl_int_is_neg(v->block.data[0]) &&
		    isl_seq_first_non_zero(v->block.data + 1, total) == -1)
			goto disjoint;
	}
	compute_elimination_index(bmap2, elim);
	for (i = 0; i < bmap1->n_ineq; ++i) {
		int reduced;
		reduced = reduced_using_equalities(v->block.data,
						bmap1->ineq[i], bmap2, elim);
		if (reduced && isl_int_is_neg(v->block.data[0]) &&
		    isl_seq_first_non_zero(v->block.data + 1, total) == -1)
			goto disjoint;
	}
	isl_vec_free(v);
	free(elim);
	return isl_bool_false;
disjoint:
	isl_vec_free(v);
	free(elim);
	return isl_bool_true;
error:
	isl_vec_free(v);
	free(elim);
	return isl_bool_error;
}

int isl_basic_set_plain_is_disjoint(__isl_keep isl_basic_set *bset1,
	__isl_keep isl_basic_set *bset2)
{
	return isl_basic_map_plain_is_disjoint((struct isl_basic_map *)bset1,
					      (struct isl_basic_map *)bset2);
}

/* Are "map1" and "map2" obviously disjoint?
 *
 * If one of them is empty or if they live in different spaces (ignoring
 * parameters), then they are clearly disjoint.
 *
 * If they have different parameters, then we skip any further tests.
 *
 * If they are obviously equal, but not obviously empty, then we will
 * not be able to detect if they are disjoint.
 *
 * Otherwise we check if each basic map in "map1" is obviously disjoint
 * from each basic map in "map2".
 */
isl_bool isl_map_plain_is_disjoint(__isl_keep isl_map *map1,
	__isl_keep isl_map *map2)
{
	int i, j;
	isl_bool disjoint;
	isl_bool intersect;
	isl_bool match;

	if (!map1 || !map2)
		return isl_bool_error;

	disjoint = isl_map_plain_is_empty(map1);
	if (disjoint < 0 || disjoint)
		return disjoint;

	disjoint = isl_map_plain_is_empty(map2);
	if (disjoint < 0 || disjoint)
		return disjoint;

	match = isl_space_tuple_is_equal(map1->dim, isl_dim_in,
				map2->dim, isl_dim_in);
	if (match < 0 || !match)
		return match < 0 ? isl_bool_error : isl_bool_true;

	match = isl_space_tuple_is_equal(map1->dim, isl_dim_out,
				map2->dim, isl_dim_out);
	if (match < 0 || !match)
		return match < 0 ? isl_bool_error : isl_bool_true;

	match = isl_space_match(map1->dim, isl_dim_param,
				map2->dim, isl_dim_param);
	if (match < 0 || !match)
		return match < 0 ? isl_bool_error : isl_bool_false;

	intersect = isl_map_plain_is_equal(map1, map2);
	if (intersect < 0 || intersect)
		return intersect < 0 ? isl_bool_error : isl_bool_false;

	for (i = 0; i < map1->n; ++i) {
		for (j = 0; j < map2->n; ++j) {
			isl_bool d = isl_basic_map_plain_is_disjoint(map1->p[i],
								   map2->p[j]);
			if (d != isl_bool_true)
				return d;
		}
	}
	return isl_bool_true;
}

/* Are "map1" and "map2" disjoint?
 *
 * They are disjoint if they are "obviously disjoint" or if one of them
 * is empty.  Otherwise, they are not disjoint if one of them is universal.
 * If none of these cases apply, we compute the intersection and see if
 * the result is empty.
 */
isl_bool isl_map_is_disjoint(__isl_keep isl_map *map1, __isl_keep isl_map *map2)
{
	isl_bool disjoint;
	isl_bool intersect;
	isl_map *test;

	disjoint = isl_map_plain_is_disjoint(map1, map2);
	if (disjoint < 0 || disjoint)
		return disjoint;

	disjoint = isl_map_is_empty(map1);
	if (disjoint < 0 || disjoint)
		return disjoint;

	disjoint = isl_map_is_empty(map2);
	if (disjoint < 0 || disjoint)
		return disjoint;

	intersect = isl_map_plain_is_universe(map1);
	if (intersect < 0 || intersect)
		return intersect < 0 ? isl_bool_error : isl_bool_false;

	intersect = isl_map_plain_is_universe(map2);
	if (intersect < 0 || intersect)
		return intersect < 0 ? isl_bool_error : isl_bool_false;

	test = isl_map_intersect(isl_map_copy(map1), isl_map_copy(map2));
	disjoint = isl_map_is_empty(test);
	isl_map_free(test);

	return disjoint;
}

/* Are "bmap1" and "bmap2" disjoint?
 *
 * They are disjoint if they are "obviously disjoint" or if one of them
 * is empty.  Otherwise, they are not disjoint if one of them is universal.
 * If none of these cases apply, we compute the intersection and see if
 * the result is empty.
 */
isl_bool isl_basic_map_is_disjoint(__isl_keep isl_basic_map *bmap1,
	__isl_keep isl_basic_map *bmap2)
{
	isl_bool disjoint;
	isl_bool intersect;
	isl_basic_map *test;

	disjoint = isl_basic_map_plain_is_disjoint(bmap1, bmap2);
	if (disjoint < 0 || disjoint)
		return disjoint;

	disjoint = isl_basic_map_is_empty(bmap1);
	if (disjoint < 0 || disjoint)
		return disjoint;

	disjoint = isl_basic_map_is_empty(bmap2);
	if (disjoint < 0 || disjoint)
		return disjoint;

	intersect = isl_basic_map_is_universe(bmap1);
	if (intersect < 0 || intersect)
		return intersect < 0 ? isl_bool_error : isl_bool_false;

	intersect = isl_basic_map_is_universe(bmap2);
	if (intersect < 0 || intersect)
		return intersect < 0 ? isl_bool_error : isl_bool_false;

	test = isl_basic_map_intersect(isl_basic_map_copy(bmap1),
		isl_basic_map_copy(bmap2));
	disjoint = isl_basic_map_is_empty(test);
	isl_basic_map_free(test);

	return disjoint;
}

/* Are "bset1" and "bset2" disjoint?
 */
isl_bool isl_basic_set_is_disjoint(__isl_keep isl_basic_set *bset1,
	__isl_keep isl_basic_set *bset2)
{
	return isl_basic_map_is_disjoint(bset1, bset2);
}

isl_bool isl_set_plain_is_disjoint(__isl_keep isl_set *set1,
	__isl_keep isl_set *set2)
{
	return isl_map_plain_is_disjoint((struct isl_map *)set1,
					(struct isl_map *)set2);
}

/* Are "set1" and "set2" disjoint?
 */
isl_bool isl_set_is_disjoint(__isl_keep isl_set *set1, __isl_keep isl_set *set2)
{
	return isl_map_is_disjoint(set1, set2);
}

/* Is "v" equal to 0, 1 or -1?
 */
static int is_zero_or_one(isl_int v)
{
	return isl_int_is_zero(v) || isl_int_is_one(v) || isl_int_is_negone(v);
}

/* Check if we can combine a given div with lower bound l and upper
 * bound u with some other div and if so return that other div.
 * Otherwise return -1.
 *
 * We first check that
 *	- the bounds are opposites of each other (except for the constant
 *	  term)
 *	- the bounds do not reference any other div
 *	- no div is defined in terms of this div
 *
 * Let m be the size of the range allowed on the div by the bounds.
 * That is, the bounds are of the form
 *
 *	e <= a <= e + m - 1
 *
 * with e some expression in the other variables.
 * We look for another div b such that no third div is defined in terms
 * of this second div b and such that in any constraint that contains
 * a (except for the given lower and upper bound), also contains b
 * with a coefficient that is m times that of b.
 * That is, all constraints (execpt for the lower and upper bound)
 * are of the form
 *
 *	e + f (a + m b) >= 0
 *
 * Furthermore, in the constraints that only contain b, the coefficient
 * of b should be equal to 1 or -1.
 * If so, we return b so that "a + m b" can be replaced by
 * a single div "c = a + m b".
 */
static int div_find_coalesce(struct isl_basic_map *bmap, int *pairs,
	unsigned div, unsigned l, unsigned u)
{
	int i, j;
	unsigned dim;
	int coalesce = -1;

	if (bmap->n_div <= 1)
		return -1;
	dim = isl_space_dim(bmap->dim, isl_dim_all);
	if (isl_seq_first_non_zero(bmap->ineq[l] + 1 + dim, div) != -1)
		return -1;
	if (isl_seq_first_non_zero(bmap->ineq[l] + 1 + dim + div + 1,
				   bmap->n_div - div - 1) != -1)
		return -1;
	if (!isl_seq_is_neg(bmap->ineq[l] + 1, bmap->ineq[u] + 1,
			    dim + bmap->n_div))
		return -1;

	for (i = 0; i < bmap->n_div; ++i) {
		if (isl_int_is_zero(bmap->div[i][0]))
			continue;
		if (!isl_int_is_zero(bmap->div[i][1 + 1 + dim + div]))
			return -1;
	}

	isl_int_add(bmap->ineq[l][0], bmap->ineq[l][0], bmap->ineq[u][0]);
	if (isl_int_is_neg(bmap->ineq[l][0])) {
		isl_int_sub(bmap->ineq[l][0],
			    bmap->ineq[l][0], bmap->ineq[u][0]);
		bmap = isl_basic_map_copy(bmap);
		bmap = isl_basic_map_set_to_empty(bmap);
		isl_basic_map_free(bmap);
		return -1;
	}
	isl_int_add_ui(bmap->ineq[l][0], bmap->ineq[l][0], 1);
	for (i = 0; i < bmap->n_div; ++i) {
		if (i == div)
			continue;
		if (!pairs[i])
			continue;
		for (j = 0; j < bmap->n_div; ++j) {
			if (isl_int_is_zero(bmap->div[j][0]))
				continue;
			if (!isl_int_is_zero(bmap->div[j][1 + 1 + dim + i]))
				break;
		}
		if (j < bmap->n_div)
			continue;
		for (j = 0; j < bmap->n_ineq; ++j) {
			int valid;
			if (j == l || j == u)
				continue;
			if (isl_int_is_zero(bmap->ineq[j][1 + dim + div])) {
				if (is_zero_or_one(bmap->ineq[j][1 + dim + i]))
					continue;
				break;
			}
			if (isl_int_is_zero(bmap->ineq[j][1 + dim + i]))
				break;
			isl_int_mul(bmap->ineq[j][1 + dim + div],
				    bmap->ineq[j][1 + dim + div],
				    bmap->ineq[l][0]);
			valid = isl_int_eq(bmap->ineq[j][1 + dim + div],
					   bmap->ineq[j][1 + dim + i]);
			isl_int_divexact(bmap->ineq[j][1 + dim + div],
					 bmap->ineq[j][1 + dim + div],
					 bmap->ineq[l][0]);
			if (!valid)
				break;
		}
		if (j < bmap->n_ineq)
			continue;
		coalesce = i;
		break;
	}
	isl_int_sub_ui(bmap->ineq[l][0], bmap->ineq[l][0], 1);
	isl_int_sub(bmap->ineq[l][0], bmap->ineq[l][0], bmap->ineq[u][0]);
	return coalesce;
}

/* Given a lower and an upper bound on div i, construct an inequality
 * that when nonnegative ensures that this pair of bounds always allows
 * for an integer value of the given div.
 * The lower bound is inequality l, while the upper bound is inequality u.
 * The constructed inequality is stored in ineq.
 * g, fl, fu are temporary scalars.
 *
 * Let the upper bound be
 *
 *	-n_u a + e_u >= 0
 *
 * and the lower bound
 *
 *	n_l a + e_l >= 0
 *
 * Let n_u = f_u g and n_l = f_l g, with g = gcd(n_u, n_l).
 * We have
 *
 *	- f_u e_l <= f_u f_l g a <= f_l e_u
 *
 * Since all variables are integer valued, this is equivalent to
 *
 *	- f_u e_l - (f_u - 1) <= f_u f_l g a <= f_l e_u + (f_l - 1)
 *
 * If this interval is at least f_u f_l g, then it contains at least
 * one integer value for a.
 * That is, the test constraint is
 *
 *	f_l e_u + f_u e_l + f_l - 1 + f_u - 1 + 1 >= f_u f_l g
 */
static void construct_test_ineq(struct isl_basic_map *bmap, int i,
	int l, int u, isl_int *ineq, isl_int *g, isl_int *fl, isl_int *fu)
{
	unsigned dim;
	dim = isl_space_dim(bmap->dim, isl_dim_all);

	isl_int_gcd(*g, bmap->ineq[l][1 + dim + i], bmap->ineq[u][1 + dim + i]);
	isl_int_divexact(*fl, bmap->ineq[l][1 + dim + i], *g);
	isl_int_divexact(*fu, bmap->ineq[u][1 + dim + i], *g);
	isl_int_neg(*fu, *fu);
	isl_seq_combine(ineq, *fl, bmap->ineq[u], *fu, bmap->ineq[l],
			1 + dim + bmap->n_div);
	isl_int_add(ineq[0], ineq[0], *fl);
	isl_int_add(ineq[0], ineq[0], *fu);
	isl_int_sub_ui(ineq[0], ineq[0], 1);
	isl_int_mul(*g, *g, *fl);
	isl_int_mul(*g, *g, *fu);
	isl_int_sub(ineq[0], ineq[0], *g);
}

/* Remove more kinds of divs that are not strictly needed.
 * In particular, if all pairs of lower and upper bounds on a div
 * are such that they allow at least one integer value of the div,
 * the we can eliminate the div using Fourier-Motzkin without
 * introducing any spurious solutions.
 */
static struct isl_basic_map *drop_more_redundant_divs(
	struct isl_basic_map *bmap, int *pairs, int n)
{
	struct isl_tab *tab = NULL;
	struct isl_vec *vec = NULL;
	unsigned dim;
	int remove = -1;
	isl_int g, fl, fu;

	isl_int_init(g);
	isl_int_init(fl);
	isl_int_init(fu);

	if (!bmap)
		goto error;

	dim = isl_space_dim(bmap->dim, isl_dim_all);
	vec = isl_vec_alloc(bmap->ctx, 1 + dim + bmap->n_div);
	if (!vec)
		goto error;

	tab = isl_tab_from_basic_map(bmap, 0);

	while (n > 0) {
		int i, l, u;
		int best = -1;
		enum isl_lp_result res;

		for (i = 0; i < bmap->n_div; ++i) {
			if (!pairs[i])
				continue;
			if (best >= 0 && pairs[best] <= pairs[i])
				continue;
			best = i;
		}

		i = best;
		for (l = 0; l < bmap->n_ineq; ++l) {
			if (!isl_int_is_pos(bmap->ineq[l][1 + dim + i]))
				continue;
			for (u = 0; u < bmap->n_ineq; ++u) {
				if (!isl_int_is_neg(bmap->ineq[u][1 + dim + i]))
					continue;
				construct_test_ineq(bmap, i, l, u,
						    vec->el, &g, &fl, &fu);
				res = isl_tab_min(tab, vec->el,
						  bmap->ctx->one, &g, NULL, 0);
				if (res == isl_lp_error)
					goto error;
				if (res == isl_lp_empty) {
					bmap = isl_basic_map_set_to_empty(bmap);
					break;
				}
				if (res != isl_lp_ok || isl_int_is_neg(g))
					break;
			}
			if (u < bmap->n_ineq)
				break;
		}
		if (l == bmap->n_ineq) {
			remove = i;
			break;
		}
		pairs[i] = 0;
		--n;
	}

	isl_tab_free(tab);
	isl_vec_free(vec);

	isl_int_clear(g);
	isl_int_clear(fl);
	isl_int_clear(fu);

	free(pairs);

	if (remove < 0)
		return bmap;

	bmap = isl_basic_map_remove_dims(bmap, isl_dim_div, remove, 1);
	return isl_basic_map_drop_redundant_divs(bmap);
error:
	free(pairs);
	isl_basic_map_free(bmap);
	isl_tab_free(tab);
	isl_vec_free(vec);
	isl_int_clear(g);
	isl_int_clear(fl);
	isl_int_clear(fu);
	return NULL;
}

/* Given a pair of divs div1 and div2 such that, except for the lower bound l
 * and the upper bound u, div1 always occurs together with div2 in the form
 * (div1 + m div2), where m is the constant range on the variable div1
 * allowed by l and u, replace the pair div1 and div2 by a single
 * div that is equal to div1 + m div2.
 *
 * The new div will appear in the location that contains div2.
 * We need to modify all constraints that contain
 * div2 = (div - div1) / m
 * The coefficient of div2 is known to be equal to 1 or -1.
 * (If a constraint does not contain div2, it will also not contain div1.)
 * If the constraint also contains div1, then we know they appear
 * as f (div1 + m div2) and we can simply replace (div1 + m div2) by div,
 * i.e., the coefficient of div is f.
 *
 * Otherwise, we first need to introduce div1 into the constraint.
 * Let the l be
 *
 *	div1 + f >=0
 *
 * and u
 *
 *	-div1 + f' >= 0
 *
 * A lower bound on div2
 *
 *	div2 + t >= 0
 *
 * can be replaced by
 *
 *	m div2 + div1 + m t + f >= 0
 *
 * An upper bound
 *
 *	-div2 + t >= 0
 *
 * can be replaced by
 *
 *	-(m div2 + div1) + m t + f' >= 0
 *
 * These constraint are those that we would obtain from eliminating
 * div1 using Fourier-Motzkin.
 *
 * After all constraints have been modified, we drop the lower and upper
 * bound and then drop div1.
 */
static struct isl_basic_map *coalesce_divs(struct isl_basic_map *bmap,
	unsigned div1, unsigned div2, unsigned l, unsigned u)
{
	isl_ctx *ctx;
	isl_int m;
	unsigned dim, total;
	int i;

	ctx = isl_basic_map_get_ctx(bmap);

	dim = isl_space_dim(bmap->dim, isl_dim_all);
	total = 1 + dim + bmap->n_div;

	isl_int_init(m);
	isl_int_add(m, bmap->ineq[l][0], bmap->ineq[u][0]);
	isl_int_add_ui(m, m, 1);

	for (i = 0; i < bmap->n_ineq; ++i) {
		if (i == l || i == u)
			continue;
		if (isl_int_is_zero(bmap->ineq[i][1 + dim + div2]))
			continue;
		if (isl_int_is_zero(bmap->ineq[i][1 + dim + div1])) {
			if (isl_int_is_pos(bmap->ineq[i][1 + dim + div2]))
				isl_seq_combine(bmap->ineq[i], m, bmap->ineq[i],
						ctx->one, bmap->ineq[l], total);
			else
				isl_seq_combine(bmap->ineq[i], m, bmap->ineq[i],
						ctx->one, bmap->ineq[u], total);
		}
		isl_int_set(bmap->ineq[i][1 + dim + div2],
			    bmap->ineq[i][1 + dim + div1]);
		isl_int_set_si(bmap->ineq[i][1 + dim + div1], 0);
	}

	isl_int_clear(m);
	if (l > u) {
		isl_basic_map_drop_inequality(bmap, l);
		isl_basic_map_drop_inequality(bmap, u);
	} else {
		isl_basic_map_drop_inequality(bmap, u);
		isl_basic_map_drop_inequality(bmap, l);
	}
	bmap = isl_basic_map_drop_div(bmap, div1);
	return bmap;
}

/* First check if we can coalesce any pair of divs and
 * then continue with dropping more redundant divs.
 *
 * We loop over all pairs of lower and upper bounds on a div
 * with coefficient 1 and -1, respectively, check if there
 * is any other div "c" with which we can coalesce the div
 * and if so, perform the coalescing.
 */
static struct isl_basic_map *coalesce_or_drop_more_redundant_divs(
	struct isl_basic_map *bmap, int *pairs, int n)
{
	int i, l, u;
	unsigned dim;

	dim = isl_space_dim(bmap->dim, isl_dim_all);

	for (i = 0; i < bmap->n_div; ++i) {
		if (!pairs[i])
			continue;
		for (l = 0; l < bmap->n_ineq; ++l) {
			if (!isl_int_is_one(bmap->ineq[l][1 + dim + i]))
				continue;
			for (u = 0; u < bmap->n_ineq; ++u) {
				int c;

				if (!isl_int_is_negone(bmap->ineq[u][1+dim+i]))
					continue;
				c = div_find_coalesce(bmap, pairs, i, l, u);
				if (c < 0)
					continue;
				free(pairs);
				bmap = coalesce_divs(bmap, i, c, l, u);
				return isl_basic_map_drop_redundant_divs(bmap);
			}
		}
	}

	if (ISL_F_ISSET(bmap, ISL_BASIC_MAP_EMPTY))
		return bmap;

	return drop_more_redundant_divs(bmap, pairs, n);
}

/* Remove divs that are not strictly needed.
 * In particular, if a div only occurs positively (or negatively)
 * in constraints, then it can simply be dropped.
 * Also, if a div occurs in only two constraints and if moreover
 * those two constraints are opposite to each other, except for the constant
 * term and if the sum of the constant terms is such that for any value
 * of the other values, there is always at least one integer value of the
 * div, i.e., if one plus this sum is greater than or equal to
 * the (absolute value) of the coefficent of the div in the constraints,
 * then we can also simply drop the div.
 *
 * We skip divs that appear in equalities or in the definition of other divs.
 * Divs that appear in the definition of other divs usually occur in at least
 * 4 constraints, but the constraints may have been simplified.
 *
 * If any divs are left after these simple checks then we move on
 * to more complicated cases in drop_more_redundant_divs.
 */
struct isl_basic_map *isl_basic_map_drop_redundant_divs(
	struct isl_basic_map *bmap)
{
	int i, j;
	unsigned off;
	int *pairs = NULL;
	int n = 0;

	if (!bmap)
		goto error;
	if (bmap->n_div == 0)
		return bmap;

	off = isl_space_dim(bmap->dim, isl_dim_all);
	pairs = isl_calloc_array(bmap->ctx, int, bmap->n_div);
	if (!pairs)
		goto error;

	for (i = 0; i < bmap->n_div; ++i) {
		int pos, neg;
		int last_pos, last_neg;
		int redundant;
		int defined;

		defined = !isl_int_is_zero(bmap->div[i][0]);
		for (j = i; j < bmap->n_div; ++j)
			if (!isl_int_is_zero(bmap->div[j][1 + 1 + off + i]))
				break;
		if (j < bmap->n_div)
			continue;
		for (j = 0; j < bmap->n_eq; ++j)
			if (!isl_int_is_zero(bmap->eq[j][1 + off + i]))
				break;
		if (j < bmap->n_eq)
			continue;
		++n;
		pos = neg = 0;
		for (j = 0; j < bmap->n_ineq; ++j) {
			if (isl_int_is_pos(bmap->ineq[j][1 + off + i])) {
				last_pos = j;
				++pos;
			}
			if (isl_int_is_neg(bmap->ineq[j][1 + off + i])) {
				last_neg = j;
				++neg;
			}
		}
		pairs[i] = pos * neg;
		if (pairs[i] == 0) {
			for (j = bmap->n_ineq - 1; j >= 0; --j)
				if (!isl_int_is_zero(bmap->ineq[j][1+off+i]))
					isl_basic_map_drop_inequality(bmap, j);
			bmap = isl_basic_map_drop_div(bmap, i);
			free(pairs);
			return isl_basic_map_drop_redundant_divs(bmap);
		}
		if (pairs[i] != 1)
			continue;
		if (!isl_seq_is_neg(bmap->ineq[last_pos] + 1,
				    bmap->ineq[last_neg] + 1,
				    off + bmap->n_div))
			continue;

		isl_int_add(bmap->ineq[last_pos][0],
			    bmap->ineq[last_pos][0], bmap->ineq[last_neg][0]);
		isl_int_add_ui(bmap->ineq[last_pos][0],
			       bmap->ineq[last_pos][0], 1);
		redundant = isl_int_ge(bmap->ineq[last_pos][0],
				bmap->ineq[last_pos][1+off+i]);
		isl_int_sub_ui(bmap->ineq[last_pos][0],
			       bmap->ineq[last_pos][0], 1);
		isl_int_sub(bmap->ineq[last_pos][0],
			    bmap->ineq[last_pos][0], bmap->ineq[last_neg][0]);
		if (!redundant) {
			if (defined ||
			    !ok_to_set_div_from_bound(bmap, i, last_pos)) {
				pairs[i] = 0;
				--n;
				continue;
			}
			bmap = set_div_from_lower_bound(bmap, i, last_pos);
			bmap = isl_basic_map_simplify(bmap);
			free(pairs);
			return isl_basic_map_drop_redundant_divs(bmap);
		}
		if (last_pos > last_neg) {
			isl_basic_map_drop_inequality(bmap, last_pos);
			isl_basic_map_drop_inequality(bmap, last_neg);
		} else {
			isl_basic_map_drop_inequality(bmap, last_neg);
			isl_basic_map_drop_inequality(bmap, last_pos);
		}
		bmap = isl_basic_map_drop_div(bmap, i);
		free(pairs);
		return isl_basic_map_drop_redundant_divs(bmap);
	}

	if (n > 0)
		return coalesce_or_drop_more_redundant_divs(bmap, pairs, n);

	free(pairs);
	return bmap;
error:
	free(pairs);
	isl_basic_map_free(bmap);
	return NULL;
}

struct isl_basic_set *isl_basic_set_drop_redundant_divs(
	struct isl_basic_set *bset)
{
	return (struct isl_basic_set *)
	    isl_basic_map_drop_redundant_divs((struct isl_basic_map *)bset);
}

struct isl_map *isl_map_drop_redundant_divs(struct isl_map *map)
{
	int i;

	if (!map)
		return NULL;
	for (i = 0; i < map->n; ++i) {
		map->p[i] = isl_basic_map_drop_redundant_divs(map->p[i]);
		if (!map->p[i])
			goto error;
	}
	ISL_F_CLR(map, ISL_MAP_NORMALIZED);
	return map;
error:
	isl_map_free(map);
	return NULL;
}

struct isl_set *isl_set_drop_redundant_divs(struct isl_set *set)
{
	return (struct isl_set *)
	    isl_map_drop_redundant_divs((struct isl_map *)set);
}

/* Does "bmap" satisfy any equality that involves more than 2 variables
 * and/or has coefficients different from -1 and 1?
 */
static int has_multiple_var_equality(__isl_keep isl_basic_map *bmap)
{
	int i;
	unsigned total;

	total = isl_basic_map_dim(bmap, isl_dim_all);

	for (i = 0; i < bmap->n_eq; ++i) {
		int j, k;

		j = isl_seq_first_non_zero(bmap->eq[i] + 1, total);
		if (j < 0)
			continue;
		if (!isl_int_is_one(bmap->eq[i][1 + j]) &&
		    !isl_int_is_negone(bmap->eq[i][1 + j]))
			return 1;

		j += 1;
		k = isl_seq_first_non_zero(bmap->eq[i] + 1 + j, total - j);
		if (k < 0)
			continue;
		j += k;
		if (!isl_int_is_one(bmap->eq[i][1 + j]) &&
		    !isl_int_is_negone(bmap->eq[i][1 + j]))
			return 1;

		j += 1;
		k = isl_seq_first_non_zero(bmap->eq[i] + 1 + j, total - j);
		if (k >= 0)
			return 1;
	}

	return 0;
}

/* Remove any common factor g from the constraint coefficients in "v".
 * The constant term is stored in the first position and is replaced
 * by floor(c/g).  If any common factor is removed and if this results
 * in a tightening of the constraint, then set *tightened.
 */
static __isl_give isl_vec *normalize_constraint(__isl_take isl_vec *v,
	int *tightened)
{
	isl_ctx *ctx;

	if (!v)
		return NULL;
	ctx = isl_vec_get_ctx(v);
	isl_seq_gcd(v->el + 1, v->size - 1, &ctx->normalize_gcd);
	if (isl_int_is_zero(ctx->normalize_gcd))
		return v;
	if (isl_int_is_one(ctx->normalize_gcd))
		return v;
	v = isl_vec_cow(v);
	if (!v)
		return NULL;
	if (tightened && !isl_int_is_divisible_by(v->el[0], ctx->normalize_gcd))
		*tightened = 1;
	isl_int_fdiv_q(v->el[0], v->el[0], ctx->normalize_gcd);
	isl_seq_scale_down(v->el + 1, v->el + 1, ctx->normalize_gcd,
				v->size - 1);
	return v;
}

/* If "bmap" is an integer set that satisfies any equality involving
 * more than 2 variables and/or has coefficients different from -1 and 1,
 * then use variable compression to reduce the coefficients by removing
 * any (hidden) common factor.
 * In particular, apply the variable compression to each constraint,
 * factor out any common factor in the non-constant coefficients and
 * then apply the inverse of the compression.
 * At the end, we mark the basic map as having reduced constants.
 * If this flag is still set on the next invocation of this function,
 * then we skip the computation.
 *
 * Removing a common factor may result in a tightening of some of
 * the constraints.  If this happens, then we may end up with two
 * opposite inequalities that can be replaced by an equality.
 * We therefore call isl_basic_map_detect_inequality_pairs,
 * which checks for such pairs of inequalities as well as eliminate_divs_eq
 * and isl_basic_map_gauss if such a pair was found.
 */
__isl_give isl_basic_map *isl_basic_map_reduce_coefficients(
	__isl_take isl_basic_map *bmap)
{
	unsigned total;
	isl_ctx *ctx;
	isl_vec *v;
	isl_mat *eq, *T, *T2;
	int i;
	int tightened;

	if (!bmap)
		return NULL;
	if (ISL_F_ISSET(bmap, ISL_BASIC_MAP_REDUCED_COEFFICIENTS))
		return bmap;
	if (isl_basic_map_is_rational(bmap))
		return bmap;
	if (bmap->n_eq == 0)
		return bmap;
	if (!has_multiple_var_equality(bmap))
		return bmap;

	total = isl_basic_map_dim(bmap, isl_dim_all);
	ctx = isl_basic_map_get_ctx(bmap);
	v = isl_vec_alloc(ctx, 1 + total);
	if (!v)
		return isl_basic_map_free(bmap);

	eq = isl_mat_sub_alloc6(ctx, bmap->eq, 0, bmap->n_eq, 0, 1 + total);
	T = isl_mat_variable_compression(eq, &T2);
	if (!T || !T2)
		goto error;
	if (T->n_col == 0) {
		isl_mat_free(T);
		isl_mat_free(T2);
		isl_vec_free(v);
		return isl_basic_map_set_to_empty(bmap);
	}

	tightened = 0;
	for (i = 0; i < bmap->n_ineq; ++i) {
		isl_seq_cpy(v->el, bmap->ineq[i], 1 + total);
		v = isl_vec_mat_product(v, isl_mat_copy(T));
		v = normalize_constraint(v, &tightened);
		v = isl_vec_mat_product(v, isl_mat_copy(T2));
		if (!v)
			goto error;
		isl_seq_cpy(bmap->ineq[i], v->el, 1 + total);
	}

	isl_mat_free(T);
	isl_mat_free(T2);
	isl_vec_free(v);

	ISL_F_SET(bmap, ISL_BASIC_MAP_REDUCED_COEFFICIENTS);

	if (tightened) {
		int progress = 0;

		bmap = isl_basic_map_detect_inequality_pairs(bmap, &progress);
		if (progress) {
			bmap = eliminate_divs_eq(bmap, &progress);
			bmap = isl_basic_map_gauss(bmap, NULL);
		}
	}

	return bmap;
error:
	isl_mat_free(T);
	isl_mat_free(T2);
	isl_vec_free(v);
	return isl_basic_map_free(bmap);
}

/* Shift the integer division at position "div" of "bmap"
 * by "shift" times the variable at position "pos".
 * "pos" is as determined by isl_basic_map_offset, i.e., pos == 0
 * corresponds to the constant term.
 *
 * That is, if the integer division has the form
 *
 *	floor(f(x)/d)
 *
 * then replace it by
 *
 *	floor((f(x) + shift * d * x_pos)/d) - shift * x_pos
 */
__isl_give isl_basic_map *isl_basic_map_shift_div(
	__isl_take isl_basic_map *bmap, int div, int pos, isl_int shift)
{
	int i;
	unsigned total;

	if (!bmap)
		return NULL;

	total = isl_basic_map_dim(bmap, isl_dim_all);
	total -= isl_basic_map_dim(bmap, isl_dim_div);

	isl_int_addmul(bmap->div[div][1 + pos], shift, bmap->div[div][0]);

	for (i = 0; i < bmap->n_eq; ++i) {
		if (isl_int_is_zero(bmap->eq[i][1 + total + div]))
			continue;
		isl_int_submul(bmap->eq[i][pos],
				shift, bmap->eq[i][1 + total + div]);
	}
	for (i = 0; i < bmap->n_ineq; ++i) {
		if (isl_int_is_zero(bmap->ineq[i][1 + total + div]))
			continue;
		isl_int_submul(bmap->ineq[i][pos],
				shift, bmap->ineq[i][1 + total + div]);
	}
	for (i = 0; i < bmap->n_div; ++i) {
		if (isl_int_is_zero(bmap->div[i][0]))
			continue;
		if (isl_int_is_zero(bmap->div[i][1 + 1 + total + div]))
			continue;
		isl_int_submul(bmap->div[i][1 + pos],
				shift, bmap->div[i][1 + 1 + total + div]);
	}

	return bmap;
}
