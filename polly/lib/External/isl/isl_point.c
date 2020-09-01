/*
 * Copyright 2010      INRIA Saclay
 * Copyright 2013      Ecole Normale Superieure
 * Copyright 2015      Sven Verdoolaege
 * Copyright 2019      Cerebras Systems
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, INRIA Saclay - Ile-de-France,
 * Parc Club Orsay Universite, ZAC des vignes, 4 rue Jacques Monod,
 * 91893 Orsay, France
 * and Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 * and Cerebras Systems, 175 S San Antonio Rd, Los Altos, CA, USA
 */

#include <isl_map_private.h>
#include <isl_point_private.h>
#include <isl/set.h>
#include <isl/union_set.h>
#include <isl_sample.h>
#include <isl_scan.h>
#include <isl_seq.h>
#include <isl_space_private.h>
#include <isl_local_private.h>
#include <isl_val_private.h>
#include <isl_vec_private.h>
#include <isl_output_private.h>

#include <set_to_map.c>

isl_ctx *isl_point_get_ctx(__isl_keep isl_point *pnt)
{
	return pnt ? isl_space_get_ctx(pnt->dim) : NULL;
}

/* Return the space of "pnt".
 */
__isl_keep isl_space *isl_point_peek_space(__isl_keep isl_point *pnt)
{
	return pnt ? pnt->dim : NULL;
}

__isl_give isl_space *isl_point_get_space(__isl_keep isl_point *pnt)
{
	return isl_space_copy(isl_point_peek_space(pnt));
}

#undef TYPE1
#define TYPE1		isl_basic_map
#undef TYPE2
#define TYPE2		isl_point
#undef TYPE_PAIR
#define TYPE_PAIR	isl_basic_map_point

static
#include "isl_type_has_equal_space_templ.c"
static
#include "isl_type_check_equal_space_templ.c"

__isl_give isl_point *isl_point_alloc(__isl_take isl_space *space,
	__isl_take isl_vec *vec)
{
	struct isl_point *pnt;
	isl_size dim;

	dim = isl_space_dim(space, isl_dim_all);
	if (dim < 0 || !vec)
		goto error;

	if (vec->size > 1 + dim) {
		vec = isl_vec_cow(vec);
		if (!vec)
			goto error;
		vec->size = 1 + dim;
	}

	pnt = isl_alloc_type(space->ctx, struct isl_point);
	if (!pnt)
		goto error;

	pnt->ref = 1;
	pnt->dim = space;
	pnt->vec = vec;

	return pnt;
error:
	isl_space_free(space);
	isl_vec_free(vec);
	return NULL;
}

__isl_give isl_point *isl_point_zero(__isl_take isl_space *space)
{
	isl_vec *vec;
	isl_size dim;

	dim = isl_space_dim(space, isl_dim_all);
	if (dim < 0)
		goto error;
	vec = isl_vec_alloc(space->ctx, 1 + dim);
	if (!vec)
		goto error;
	isl_int_set_si(vec->el[0], 1);
	isl_seq_clr(vec->el + 1, vec->size - 1);
	return isl_point_alloc(space, vec);
error:
	isl_space_free(space);
	return NULL;
}

__isl_give isl_point *isl_point_dup(__isl_keep isl_point *pnt)
{
	struct isl_point *pnt2;

	if (!pnt)
		return NULL;
	pnt2 = isl_point_alloc(isl_space_copy(pnt->dim), isl_vec_copy(pnt->vec));
	return pnt2;
}

__isl_give isl_point *isl_point_cow(__isl_take isl_point *pnt)
{
	struct isl_point *pnt2;
	if (!pnt)
		return NULL;

	if (pnt->ref == 1)
		return pnt;

	pnt2 = isl_point_dup(pnt);
	isl_point_free(pnt);
	return pnt2;
}

__isl_give isl_point *isl_point_copy(__isl_keep isl_point *pnt)
{
	if (!pnt)
		return NULL;

	pnt->ref++;
	return pnt;
}

__isl_null isl_point *isl_point_free(__isl_take isl_point *pnt)
{
	if (!pnt)
		return NULL;

	if (--pnt->ref > 0)
		return NULL;

	isl_space_free(pnt->dim);
	isl_vec_free(pnt->vec);
	free(pnt);
	return NULL;
}

__isl_give isl_point *isl_point_void(__isl_take isl_space *space)
{
	if (!space)
		return NULL;

	return isl_point_alloc(space, isl_vec_alloc(space->ctx, 0));
}

isl_bool isl_point_is_void(__isl_keep isl_point *pnt)
{
	if (!pnt)
		return isl_bool_error;

	return isl_bool_ok(pnt->vec->size == 0);
}

/* Return the space of "pnt".
 * This may be either a copy or the space itself
 * if there is only one reference to "pnt".
 * This allows the space to be modified inplace
 * if both the point and its space have only a single reference.
 * The caller is not allowed to modify "pnt" between this call and
 * a subsequent call to isl_point_restore_space.
 * The only exception is that isl_point_free can be called instead.
 */
__isl_give isl_space *isl_point_take_space(__isl_keep isl_point *pnt)
{
	isl_space *space;

	if (!pnt)
		return NULL;
	if (pnt->ref != 1)
		return isl_point_get_space(pnt);
	space = pnt->dim;
	pnt->dim = NULL;
	return space;
}

/* Set the space of "pnt" to "space", where the space of "pnt" may be missing
 * due to a preceding call to isl_point_take_space.
 * However, in this case, "pnt" only has a single reference and
 * then the call to isl_point_cow has no effect.
 */
__isl_give isl_point *isl_point_restore_space(__isl_take isl_point *pnt,
	__isl_take isl_space *space)
{
	if (!pnt || !space)
		goto error;

	if (pnt->dim == space) {
		isl_space_free(space);
		return pnt;
	}

	pnt = isl_point_cow(pnt);
	if (!pnt)
		goto error;
	isl_space_free(pnt->dim);
	pnt->dim = space;

	return pnt;
error:
	isl_point_free(pnt);
	isl_space_free(space);
	return NULL;
}

/* Return the coordinate vector of "pnt".
 */
__isl_keep isl_vec *isl_point_peek_vec(__isl_keep isl_point *pnt)
{
	return pnt ? pnt->vec : NULL;
}

/* Return a copy of the coordinate vector of "pnt".
 */
__isl_give isl_vec *isl_point_get_vec(__isl_keep isl_point *pnt)
{
	return isl_vec_copy(isl_point_peek_vec(pnt));
}

/* Return the coordinate vector of "pnt".
 * This may be either a copy or the coordinate vector itself
 * if there is only one reference to "pnt".
 * This allows the coordinate vector to be modified inplace
 * if both the point and its coordinate vector have only a single reference.
 * The caller is not allowed to modify "pnt" between this call and
 * a subsequent call to isl_point_restore_vec.
 * The only exception is that isl_point_free can be called instead.
 */
__isl_give isl_vec *isl_point_take_vec(__isl_keep isl_point *pnt)
{
	isl_vec *vec;

	if (!pnt)
		return NULL;
	if (pnt->ref != 1)
		return isl_point_get_vec(pnt);
	vec = pnt->vec;
	pnt->vec = NULL;
	return vec;
}

/* Set the coordinate vector of "pnt" to "vec",
 * where the coordinate vector of "pnt" may be missing
 * due to a preceding call to isl_point_take_vec.
 * However, in this case, "pnt" only has a single reference and
 * then the call to isl_point_cow has no effect.
 */
__isl_give isl_point *isl_point_restore_vec(__isl_take isl_point *pnt,
	__isl_take isl_vec *vec)
{
	if (!pnt || !vec)
		goto error;

	if (pnt->vec == vec) {
		isl_vec_free(vec);
		return pnt;
	}

	pnt = isl_point_cow(pnt);
	if (!pnt)
		goto error;
	isl_vec_free(pnt->vec);
	pnt->vec = vec;

	return pnt;
error:
	isl_point_free(pnt);
	isl_vec_free(vec);
	return NULL;
}

/* Return the number of variables of the given type.
 */
static isl_size isl_point_dim(__isl_keep isl_point *pnt, enum isl_dim_type type)
{
	return isl_space_dim(isl_point_peek_space(pnt), type);
}

/* Return the position of the coordinates of the given type
 * within the sequence of coordinates of "pnt".
 */
static isl_size isl_point_var_offset(__isl_keep isl_point *pnt,
	enum isl_dim_type type)
{
	return pnt ? isl_space_offset(pnt->dim, type) : isl_size_error;
}

#undef TYPE
#define TYPE	isl_point
static
#include "check_type_range_templ.c"

/* Return the value of coordinate "pos" of type "type" of "pnt".
 */
__isl_give isl_val *isl_point_get_coordinate_val(__isl_keep isl_point *pnt,
	enum isl_dim_type type, int pos)
{
	isl_ctx *ctx;
	isl_val *v;
	isl_size off;

	if (!pnt)
		return NULL;

	ctx = isl_point_get_ctx(pnt);
	if (isl_point_is_void(pnt))
		isl_die(ctx, isl_error_invalid,
			"void point does not have coordinates", return NULL);
	if (isl_point_check_range(pnt, type, pos, 1) < 0)
		return NULL;

	off = isl_point_var_offset(pnt, type);
	if (off < 0)
		return NULL;
	pos += off;

	v = isl_val_rat_from_isl_int(ctx, pnt->vec->el[1 + pos],
						pnt->vec->el[0]);
	return isl_val_normalize(v);
}

/* Set all entries of "mv" to NaN.
 */
static __isl_give isl_multi_val *set_nan(__isl_take isl_multi_val *mv)
{
	int i;
	isl_size n;
	isl_val *v;

	n = isl_multi_val_size(mv);
	if (n < 0)
		return isl_multi_val_free(mv);
	v = isl_val_nan(isl_multi_val_get_ctx(mv));
	for (i = 0; i < n; ++i)
		mv = isl_multi_val_set_at(mv, i, isl_val_copy(v));
	isl_val_free(v);

	return mv;
}

/* Return the values of the set dimensions of "pnt".
 * Return a sequence of NaNs in case of a void point.
 */
__isl_give isl_multi_val *isl_point_get_multi_val(__isl_keep isl_point *pnt)
{
	int i;
	isl_bool is_void;
	isl_size n;
	isl_multi_val *mv;

	is_void = isl_point_is_void(pnt);
	if (is_void < 0)
		return NULL;

	mv = isl_multi_val_alloc(isl_point_get_space(pnt));
	if (is_void)
		return set_nan(mv);
	n = isl_multi_val_size(mv);
	if (n < 0)
		return isl_multi_val_free(mv);
	for (i = 0; i < n; ++i) {
		isl_val *v;

		v = isl_point_get_coordinate_val(pnt, isl_dim_set, i);
		mv = isl_multi_val_set_at(mv, i, v);
	}

	return mv;
}

/* Replace coordinate "pos" of type "type" of "pnt" by "v".
 */
__isl_give isl_point *isl_point_set_coordinate_val(__isl_take isl_point *pnt,
	enum isl_dim_type type, int pos, __isl_take isl_val *v)
{
	if (!pnt || !v)
		goto error;
	if (isl_point_is_void(pnt))
		isl_die(isl_point_get_ctx(pnt), isl_error_invalid,
			"void point does not have coordinates", goto error);
	if (isl_point_check_range(pnt, type, pos, 1) < 0)
		goto error;
	if (!isl_val_is_rat(v))
		isl_die(isl_point_get_ctx(pnt), isl_error_invalid,
			"expecting rational value", goto error);

	pos += isl_space_offset(isl_point_peek_space(pnt), type);
	if (isl_int_eq(pnt->vec->el[1 + pos], v->n) &&
	    isl_int_eq(pnt->vec->el[0], v->d)) {
		isl_val_free(v);
		return pnt;
	}

	pnt = isl_point_cow(pnt);
	if (!pnt)
		goto error;
	pnt->vec = isl_vec_cow(pnt->vec);
	if (!pnt->vec)
		goto error;

	if (isl_int_eq(pnt->vec->el[0], v->d)) {
		isl_int_set(pnt->vec->el[1 + pos], v->n);
	} else if (isl_int_is_one(v->d)) {
		isl_int_mul(pnt->vec->el[1 + pos], pnt->vec->el[0], v->n);
	} else {
		isl_seq_scale(pnt->vec->el + 1,
				pnt->vec->el + 1, v->d, pnt->vec->size - 1);
		isl_int_mul(pnt->vec->el[1 + pos], pnt->vec->el[0], v->n);
		isl_int_mul(pnt->vec->el[0], pnt->vec->el[0], v->d);
		pnt->vec = isl_vec_normalize(pnt->vec);
		if (!pnt->vec)
			goto error;
	}

	isl_val_free(v);
	return pnt;
error:
	isl_val_free(v);
	isl_point_free(pnt);
	return NULL;
}

__isl_give isl_point *isl_point_add_ui(__isl_take isl_point *pnt,
	enum isl_dim_type type, int pos, unsigned val)
{
	isl_size off;

	if (!pnt || isl_point_is_void(pnt))
		return pnt;

	pnt = isl_point_cow(pnt);
	if (!pnt)
		return NULL;
	pnt->vec = isl_vec_cow(pnt->vec);
	if (!pnt->vec)
		goto error;

	off = isl_point_var_offset(pnt, type);
	if (off < 0)
		goto error;
	pos += off;

	isl_int_add_ui(pnt->vec->el[1 + pos], pnt->vec->el[1 + pos], val);

	return pnt;
error:
	isl_point_free(pnt);
	return NULL;
}

__isl_give isl_point *isl_point_sub_ui(__isl_take isl_point *pnt,
	enum isl_dim_type type, int pos, unsigned val)
{
	isl_size off;

	if (!pnt || isl_point_is_void(pnt))
		return pnt;

	pnt = isl_point_cow(pnt);
	if (!pnt)
		return NULL;
	pnt->vec = isl_vec_cow(pnt->vec);
	if (!pnt->vec)
		goto error;

	off = isl_point_var_offset(pnt, type);
	if (off < 0)
		goto error;
	pos += off;

	isl_int_sub_ui(pnt->vec->el[1 + pos], pnt->vec->el[1 + pos], val);

	return pnt;
error:
	isl_point_free(pnt);
	return NULL;
}

struct isl_foreach_point {
	struct isl_scan_callback callback;
	isl_stat (*fn)(__isl_take isl_point *pnt, void *user);
	void *user;
	isl_space *dim;
};

static isl_stat foreach_point(struct isl_scan_callback *cb,
	__isl_take isl_vec *sample)
{
	struct isl_foreach_point *fp = (struct isl_foreach_point *)cb;
	isl_point *pnt;

	pnt = isl_point_alloc(isl_space_copy(fp->dim), sample);

	return fp->fn(pnt, fp->user);
}

isl_stat isl_set_foreach_point(__isl_keep isl_set *set,
	isl_stat (*fn)(__isl_take isl_point *pnt, void *user), void *user)
{
	struct isl_foreach_point fp = { { &foreach_point }, fn, user };
	int i;

	if (!set)
		return isl_stat_error;

	fp.dim = isl_set_get_space(set);
	if (!fp.dim)
		return isl_stat_error;

	set = isl_set_copy(set);
	set = isl_set_cow(set);
	set = isl_set_make_disjoint(set);
	set = isl_set_compute_divs(set);
	if (!set)
		goto error;

	for (i = 0; i < set->n; ++i)
		if (isl_basic_set_scan(isl_basic_set_copy(set->p[i]),
					&fp.callback) < 0)
			goto error;

	isl_set_free(set);
	isl_space_free(fp.dim);

	return isl_stat_ok;
error:
	isl_set_free(set);
	isl_space_free(fp.dim);
	return isl_stat_error;
}

/* Return 1 if "bmap" contains the point "point".
 * "bmap" is assumed to have known divs.
 * The point is first extended with the divs and then passed
 * to basic_map_contains.
 */
isl_bool isl_basic_map_contains_point(__isl_keep isl_basic_map *bmap,
	__isl_keep isl_point *point)
{
	isl_local *local;
	isl_vec *vec;
	isl_bool contains;

	if (isl_basic_map_point_check_equal_space(bmap, point) < 0)
		return isl_bool_error;
	if (bmap->n_div == 0)
		return isl_basic_map_contains(bmap, point->vec);

	local = isl_local_alloc_from_mat(isl_basic_map_get_divs(bmap));
	vec = isl_point_get_vec(point);
	vec = isl_local_extend_point_vec(local, vec);
	isl_local_free(local);

	contains = isl_basic_map_contains(bmap, vec);

	isl_vec_free(vec);
	return contains;
}

isl_bool isl_map_contains_point(__isl_keep isl_map *map,
	__isl_keep isl_point *point)
{
	int i;
	isl_bool found = isl_bool_false;

	if (!map || !point)
		return isl_bool_error;

	map = isl_map_copy(map);
	map = isl_map_compute_divs(map);
	if (!map)
		return isl_bool_error;

	for (i = 0; i < map->n; ++i) {
		found = isl_basic_map_contains_point(map->p[i], point);
		if (found < 0)
			goto error;
		if (found)
			break;
	}
	isl_map_free(map);

	return found;
error:
	isl_map_free(map);
	return isl_bool_error;
}

isl_bool isl_set_contains_point(__isl_keep isl_set *set,
	__isl_keep isl_point *point)
{
	return isl_map_contains_point(set_to_map(set), point);
}

__isl_give isl_basic_set *isl_basic_set_from_point(__isl_take isl_point *pnt)
{
	isl_basic_set *bset;
	isl_basic_set *model;

	if (!pnt)
		return NULL;

	model = isl_basic_set_empty(isl_space_copy(pnt->dim));
	bset = isl_basic_set_from_vec(isl_vec_copy(pnt->vec));
	bset = isl_basic_set_from_underlying_set(bset, model);
	isl_point_free(pnt);

	return bset;
}

__isl_give isl_set *isl_set_from_point(__isl_take isl_point *pnt)
{
	isl_basic_set *bset;
	bset = isl_basic_set_from_point(pnt);
	return isl_set_from_basic_set(bset);
}

/* Construct a union set, containing the single element "pnt".
 * If "pnt" is void, then return an empty union set.
 */
__isl_give isl_union_set *isl_union_set_from_point(__isl_take isl_point *pnt)
{
	if (!pnt)
		return NULL;
	if (isl_point_is_void(pnt)) {
		isl_space *space;

		space = isl_point_get_space(pnt);
		isl_point_free(pnt);
		return isl_union_set_empty(space);
	}

	return isl_union_set_from_set(isl_set_from_point(pnt));
}

__isl_give isl_basic_set *isl_basic_set_box_from_points(
	__isl_take isl_point *pnt1, __isl_take isl_point *pnt2)
{
	isl_basic_set *bset = NULL;
	isl_size total;
	int i;
	int k;
	isl_int t;

	isl_int_init(t);

	if (!pnt1 || !pnt2)
		goto error;

	isl_assert(pnt1->dim->ctx,
			isl_space_is_equal(pnt1->dim, pnt2->dim), goto error);

	if (isl_point_is_void(pnt1) && isl_point_is_void(pnt2)) {
		isl_space *space = isl_space_copy(pnt1->dim);
		isl_point_free(pnt1);
		isl_point_free(pnt2);
		isl_int_clear(t);
		return isl_basic_set_empty(space);
	}
	if (isl_point_is_void(pnt1)) {
		isl_point_free(pnt1);
		isl_int_clear(t);
		return isl_basic_set_from_point(pnt2);
	}
	if (isl_point_is_void(pnt2)) {
		isl_point_free(pnt2);
		isl_int_clear(t);
		return isl_basic_set_from_point(pnt1);
	}

	total = isl_point_dim(pnt1, isl_dim_all);
	if (total < 0)
		goto error;
	bset = isl_basic_set_alloc_space(isl_space_copy(pnt1->dim), 0, 0, 2 * total);

	for (i = 0; i < total; ++i) {
		isl_int_mul(t, pnt1->vec->el[1 + i], pnt2->vec->el[0]);
		isl_int_submul(t, pnt2->vec->el[1 + i], pnt1->vec->el[0]);

		k = isl_basic_set_alloc_inequality(bset);
		if (k < 0)
			goto error;
		isl_seq_clr(bset->ineq[k] + 1, total);
		if (isl_int_is_pos(t)) {
			isl_int_set_si(bset->ineq[k][1 + i], -1);
			isl_int_set(bset->ineq[k][0], pnt1->vec->el[1 + i]);
		} else {
			isl_int_set_si(bset->ineq[k][1 + i], 1);
			isl_int_neg(bset->ineq[k][0], pnt1->vec->el[1 + i]);
		}
		isl_int_fdiv_q(bset->ineq[k][0], bset->ineq[k][0], pnt1->vec->el[0]);

		k = isl_basic_set_alloc_inequality(bset);
		if (k < 0)
			goto error;
		isl_seq_clr(bset->ineq[k] + 1, total);
		if (isl_int_is_pos(t)) {
			isl_int_set_si(bset->ineq[k][1 + i], 1);
			isl_int_neg(bset->ineq[k][0], pnt2->vec->el[1 + i]);
		} else {
			isl_int_set_si(bset->ineq[k][1 + i], -1);
			isl_int_set(bset->ineq[k][0], pnt2->vec->el[1 + i]);
		}
		isl_int_fdiv_q(bset->ineq[k][0], bset->ineq[k][0], pnt2->vec->el[0]);
	}

	bset = isl_basic_set_finalize(bset);

	isl_point_free(pnt1);
	isl_point_free(pnt2);

	isl_int_clear(t);

	return bset;
error:
	isl_point_free(pnt1);
	isl_point_free(pnt2);
	isl_int_clear(t);
	isl_basic_set_free(bset);
	return NULL;
}

__isl_give isl_set *isl_set_box_from_points(__isl_take isl_point *pnt1,
	__isl_take isl_point *pnt2)
{
	isl_basic_set *bset;
	bset = isl_basic_set_box_from_points(pnt1, pnt2);
	return isl_set_from_basic_set(bset);
}

/* Print the coordinate at position "pos" of the point "pnt".
 */
static __isl_give isl_printer *print_coordinate(__isl_take isl_printer *p,
	struct isl_print_space_data *data, unsigned pos)
{
	isl_point *pnt = data->user;

	pos += isl_space_offset(data->space, data->type);
	p = isl_printer_print_isl_int(p, pnt->vec->el[1 + pos]);
	if (!isl_int_is_one(pnt->vec->el[0])) {
		p = isl_printer_print_str(p, "/");
		p = isl_printer_print_isl_int(p, pnt->vec->el[0]);
	}

	return p;
}

__isl_give isl_printer *isl_printer_print_point(
	__isl_take isl_printer *p, __isl_keep isl_point *pnt)
{
	struct isl_print_space_data data = { 0 };
	int i;
	isl_size nparam;

	if (!pnt)
		return p;
	if (isl_point_is_void(pnt)) {
		p = isl_printer_print_str(p, "void");
		return p;
	}

	nparam = isl_point_dim(pnt, isl_dim_param);
	if (nparam < 0)
		return isl_printer_free(p);
	if (nparam > 0) {
		p = isl_printer_print_str(p, "[");
		for (i = 0; i < nparam; ++i) {
			const char *name;
			if (i)
				p = isl_printer_print_str(p, ", ");
			name = isl_space_get_dim_name(pnt->dim, isl_dim_param, i);
			if (name) {
				p = isl_printer_print_str(p, name);
				p = isl_printer_print_str(p, " = ");
			}
			p = isl_printer_print_isl_int(p, pnt->vec->el[1 + i]);
			if (!isl_int_is_one(pnt->vec->el[0])) {
				p = isl_printer_print_str(p, "/");
				p = isl_printer_print_isl_int(p, pnt->vec->el[0]);
			}
		}
		p = isl_printer_print_str(p, "]");
		p = isl_printer_print_str(p, " -> ");
	}
	data.print_dim = &print_coordinate;
	data.user = pnt;
	p = isl_printer_print_str(p, "{ ");
	p = isl_print_space(pnt->dim, p, 0, &data);
	p = isl_printer_print_str(p, " }");
	return p;
}
