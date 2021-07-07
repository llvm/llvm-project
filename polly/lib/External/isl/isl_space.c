/*
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 * Copyright 2010      INRIA Saclay
 * Copyright 2013-2014 Ecole Normale Superieure
 * Copyright 2018-2019 Cerebras Systems
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, K.U.Leuven, Departement
 * Computerwetenschappen, Celestijnenlaan 200A, B-3001 Leuven, Belgium
 * and INRIA Saclay - Ile-de-France, Parc Club Orsay Universite,
 * ZAC des vignes, 4 rue Jacques Monod, 91893 Orsay, France 
 * and Ecole Normale Superieure, 45 rue d’Ulm, 75230 Paris, France
 * and Cerebras Systems, 175 S San Antonio Rd, Los Altos, CA, USA
 */

#include <stdlib.h>
#include <string.h>
#include <isl_space_private.h>
#include <isl_id_private.h>
#include <isl_reordering.h>

isl_ctx *isl_space_get_ctx(__isl_keep isl_space *space)
{
	return space ? space->ctx : NULL;
}

__isl_give isl_space *isl_space_alloc(isl_ctx *ctx,
			unsigned nparam, unsigned n_in, unsigned n_out)
{
	isl_space *space;

	space = isl_alloc_type(ctx, struct isl_space);
	if (!space)
		return NULL;

	space->ctx = ctx;
	isl_ctx_ref(ctx);
	space->ref = 1;
	space->nparam = nparam;
	space->n_in = n_in;
	space->n_out = n_out;

	space->tuple_id[0] = NULL;
	space->tuple_id[1] = NULL;

	space->nested[0] = NULL;
	space->nested[1] = NULL;

	space->n_id = 0;
	space->ids = NULL;

	return space;
}

/* Mark the space as being that of a set, by setting the domain tuple
 * to isl_id_none.
 */
static __isl_give isl_space *mark_as_set(__isl_take isl_space *space)
{
	space = isl_space_cow(space);
	if (!space)
		return NULL;
	space = isl_space_set_tuple_id(space, isl_dim_in, &isl_id_none);
	return space;
}

/* Is the space that of a set?
 */
isl_bool isl_space_is_set(__isl_keep isl_space *space)
{
	if (!space)
		return isl_bool_error;
	if (space->n_in != 0 || space->nested[0])
		return isl_bool_false;
	if (space->tuple_id[0] != &isl_id_none)
		return isl_bool_false;
	return isl_bool_true;
}

/* Check that "space" is a set space.
 */
isl_stat isl_space_check_is_set(__isl_keep isl_space *space)
{
	isl_bool is_set;

	is_set = isl_space_is_set(space);
	if (is_set < 0)
		return isl_stat_error;
	if (!is_set)
		isl_die(isl_space_get_ctx(space), isl_error_invalid,
			"space is not a set", return isl_stat_error);
	return isl_stat_ok;
}

/* Is the given space that of a map?
 */
isl_bool isl_space_is_map(__isl_keep isl_space *space)
{
	int r;

	if (!space)
		return isl_bool_error;
	r = space->tuple_id[0] != &isl_id_none &&
	    space->tuple_id[1] != &isl_id_none;
	return isl_bool_ok(r);
}

/* Check that "space" is the space of a map.
 */
static isl_stat isl_space_check_is_map(__isl_keep isl_space *space)
{
	isl_bool is_space;

	is_space = isl_space_is_map(space);
	if (is_space < 0)
		return isl_stat_error;
	if (!is_space)
		isl_die(isl_space_get_ctx(space), isl_error_invalid,
			"expecting map space", return isl_stat_error);
	return isl_stat_ok;
}

/* Check that "space" is the space of a map
 * where the domain is a wrapped map space.
 */
isl_stat isl_space_check_domain_is_wrapping(__isl_keep isl_space *space)
{
	isl_bool wrapping;

	wrapping = isl_space_domain_is_wrapping(space);
	if (wrapping < 0)
		return isl_stat_error;
	if (!wrapping)
		isl_die(isl_space_get_ctx(space), isl_error_invalid,
			"domain not a product", return isl_stat_error);
	return isl_stat_ok;
}

/* Check that "space" is the space of a map
 * where the range is a wrapped map space.
 */
isl_stat isl_space_check_range_is_wrapping(__isl_keep isl_space *space)
{
	isl_bool wrapping;

	wrapping = isl_space_range_is_wrapping(space);
	if (wrapping < 0)
		return isl_stat_error;
	if (!wrapping)
		isl_die(isl_space_get_ctx(space), isl_error_invalid,
			"range not a product", return isl_stat_error);
	return isl_stat_ok;
}

__isl_give isl_space *isl_space_set_alloc(isl_ctx *ctx,
			unsigned nparam, unsigned dim)
{
	isl_space *space;
	space = isl_space_alloc(ctx, nparam, 0, dim);
	space = mark_as_set(space);
	return space;
}

/* Mark the space as being that of a parameter domain, by setting
 * both tuples to isl_id_none.
 */
static __isl_give isl_space *mark_as_params(isl_space *space)
{
	if (!space)
		return NULL;
	space = isl_space_set_tuple_id(space, isl_dim_in, &isl_id_none);
	space = isl_space_set_tuple_id(space, isl_dim_out, &isl_id_none);
	return space;
}

/* Is the space that of a parameter domain?
 */
isl_bool isl_space_is_params(__isl_keep isl_space *space)
{
	if (!space)
		return isl_bool_error;
	if (space->n_in != 0 || space->nested[0] ||
	    space->n_out != 0 || space->nested[1])
		return isl_bool_false;
	if (space->tuple_id[0] != &isl_id_none)
		return isl_bool_false;
	if (space->tuple_id[1] != &isl_id_none)
		return isl_bool_false;
	return isl_bool_true;
}

/* Create a space for a parameter domain.
 */
__isl_give isl_space *isl_space_params_alloc(isl_ctx *ctx, unsigned nparam)
{
	isl_space *space;
	space = isl_space_alloc(ctx, nparam, 0, 0);
	space = mark_as_params(space);
	return space;
}

/* Create a space for a parameter domain, without any parameters.
 */
__isl_give isl_space *isl_space_unit(isl_ctx *ctx)
{
	return isl_space_params_alloc(ctx, 0);
}

static isl_size global_pos(__isl_keep isl_space *space,
				 enum isl_dim_type type, unsigned pos)
{
	if (isl_space_check_range(space, type, pos, 1) < 0)
		return isl_size_error;

	switch (type) {
	case isl_dim_param:
		return pos;
	case isl_dim_in:
		return pos + space->nparam;
	case isl_dim_out:
		return pos + space->nparam + space->n_in;
	default:
		isl_assert(isl_space_get_ctx(space), 0, return isl_size_error);
	}
	return isl_size_error;
}

/* Extend length of ids array to the total number of dimensions.
 */
static __isl_give isl_space *extend_ids(__isl_take isl_space *space)
{
	isl_id **ids;
	int i;
	isl_size dim;

	dim = isl_space_dim(space, isl_dim_all);
	if (dim < 0)
		return isl_space_free(space);
	if (dim <= space->n_id)
		return space;

	if (!space->ids) {
		space->ids = isl_calloc_array(space->ctx, isl_id *, dim);
		if (!space->ids)
			goto error;
	} else {
		ids = isl_realloc_array(space->ctx, space->ids, isl_id *, dim);
		if (!ids)
			goto error;
		space->ids = ids;
		for (i = space->n_id; i < dim; ++i)
			space->ids[i] = NULL;
	}

	space->n_id = dim;

	return space;
error:
	isl_space_free(space);
	return NULL;
}

static __isl_give isl_space *set_id(__isl_take isl_space *space,
	enum isl_dim_type type, unsigned pos, __isl_take isl_id *id)
{
	isl_size gpos;

	space = isl_space_cow(space);

	gpos = global_pos(space, type, pos);
	if (gpos < 0)
		goto error;

	if (gpos >= space->n_id) {
		if (!id)
			return space;
		space = extend_ids(space);
		if (!space)
			goto error;
	}

	space->ids[gpos] = id;

	return space;
error:
	isl_id_free(id);
	isl_space_free(space);
	return NULL;
}

static __isl_keep isl_id *get_id(__isl_keep isl_space *space,
				 enum isl_dim_type type, unsigned pos)
{
	isl_size gpos;

	gpos = global_pos(space, type, pos);
	if (gpos < 0)
		return NULL;
	if (gpos >= space->n_id)
		return NULL;
	return space->ids[gpos];
}

/* Return the nested space at the given position.
 */
static __isl_keep isl_space *isl_space_peek_nested(__isl_keep isl_space *space,
	int pos)
{
	if (!space)
		return NULL;
	if (!space->nested[pos])
		isl_die(isl_space_get_ctx(space), isl_error_invalid,
			"no nested space", return NULL);
	return space->nested[pos];
}

static unsigned offset(__isl_keep isl_space *space, enum isl_dim_type type)
{
	switch (type) {
	case isl_dim_param:	return 0;
	case isl_dim_in:	return space->nparam;
	case isl_dim_out:	return space->nparam + space->n_in;
	default:		return 0;
	}
}

static unsigned n(__isl_keep isl_space *space, enum isl_dim_type type)
{
	switch (type) {
	case isl_dim_param:	return space->nparam;
	case isl_dim_in:	return space->n_in;
	case isl_dim_out:	return space->n_out;
	case isl_dim_all:
		return space->nparam + space->n_in + space->n_out;
	default:		return 0;
	}
}

isl_size isl_space_dim(__isl_keep isl_space *space, enum isl_dim_type type)
{
	if (!space)
		return isl_size_error;
	return n(space, type);
}

/* Return the dimensionality of tuple "inner" within the wrapped relation
 * inside tuple "outer".
 */
isl_size isl_space_wrapped_dim(__isl_keep isl_space *space,
	enum isl_dim_type outer, enum isl_dim_type inner)
{
	int pos;

	if (!space)
		return isl_size_error;
	if (outer != isl_dim_in && outer != isl_dim_out)
		isl_die(isl_space_get_ctx(space), isl_error_invalid,
			"only input, output and set tuples "
			"can have nested relations", return isl_size_error);
	pos = outer - isl_dim_in;
	return isl_space_dim(isl_space_peek_nested(space, pos), inner);
}

unsigned isl_space_offset(__isl_keep isl_space *space, enum isl_dim_type type)
{
	if (!space)
		return 0;
	return offset(space, type);
}

static __isl_give isl_space *copy_ids(__isl_take isl_space *dst,
	enum isl_dim_type dst_type, unsigned offset, __isl_keep isl_space *src,
	enum isl_dim_type src_type)
{
	int i;
	isl_id *id;

	if (!dst)
		return NULL;

	for (i = 0; i < n(src, src_type); ++i) {
		id = get_id(src, src_type, i);
		if (!id)
			continue;
		dst = set_id(dst, dst_type, offset + i, isl_id_copy(id));
		if (!dst)
			return NULL;
	}
	return dst;
}

__isl_take isl_space *isl_space_dup(__isl_keep isl_space *space)
{
	isl_space *dup;
	if (!space)
		return NULL;
	dup = isl_space_alloc(space->ctx,
				space->nparam, space->n_in, space->n_out);
	if (!dup)
		return NULL;
	if (space->tuple_id[0] &&
	    !(dup->tuple_id[0] = isl_id_copy(space->tuple_id[0])))
		goto error;
	if (space->tuple_id[1] &&
	    !(dup->tuple_id[1] = isl_id_copy(space->tuple_id[1])))
		goto error;
	if (space->nested[0] &&
	    !(dup->nested[0] = isl_space_copy(space->nested[0])))
		goto error;
	if (space->nested[1] &&
	    !(dup->nested[1] = isl_space_copy(space->nested[1])))
		goto error;
	if (!space->ids)
		return dup;
	dup = copy_ids(dup, isl_dim_param, 0, space, isl_dim_param);
	dup = copy_ids(dup, isl_dim_in, 0, space, isl_dim_in);
	dup = copy_ids(dup, isl_dim_out, 0, space, isl_dim_out);
	return dup;
error:
	isl_space_free(dup);
	return NULL;
}

__isl_give isl_space *isl_space_cow(__isl_take isl_space *space)
{
	if (!space)
		return NULL;

	if (space->ref == 1)
		return space;
	space->ref--;
	return isl_space_dup(space);
}

__isl_give isl_space *isl_space_copy(__isl_keep isl_space *space)
{
	if (!space)
		return NULL;

	space->ref++;
	return space;
}

__isl_null isl_space *isl_space_free(__isl_take isl_space *space)
{
	int i;

	if (!space)
		return NULL;

	if (--space->ref > 0)
		return NULL;

	isl_id_free(space->tuple_id[0]);
	isl_id_free(space->tuple_id[1]);

	isl_space_free(space->nested[0]);
	isl_space_free(space->nested[1]);

	for (i = 0; i < space->n_id; ++i)
		isl_id_free(space->ids[i]);
	free(space->ids);
	isl_ctx_deref(space->ctx);
	
	free(space);

	return NULL;
}

/* Check if "s" is a valid dimension or tuple name.
 * We currently only forbid names that look like a number.
 *
 * s is assumed to be non-NULL.
 */
static int name_ok(isl_ctx *ctx, const char *s)
{
	char *p;
	long dummy;

	dummy = strtol(s, &p, 0);
	if (p != s)
		isl_die(ctx, isl_error_invalid, "name looks like a number",
			return 0);

	return 1;
}

/* Return a copy of the nested space at the given position.
 */
static __isl_keep isl_space *isl_space_get_nested(__isl_keep isl_space *space,
	int pos)
{
	return isl_space_copy(isl_space_peek_nested(space, pos));
}

/* Return the nested space at the given position.
 * This may be either a copy or the nested space itself
 * if there is only one reference to "space".
 * This allows the nested space to be modified inplace
 * if both "space" and the nested space have only a single reference.
 * The caller is not allowed to modify "space" between this call and
 * a subsequent call to isl_space_restore_nested.
 * The only exception is that isl_space_free can be called instead.
 */
static __isl_give isl_space *isl_space_take_nested(__isl_keep isl_space *space,
	int pos)
{
	isl_space *nested;

	if (!space)
		return NULL;
	if (space->ref != 1)
		return isl_space_get_nested(space, pos);
	nested = space->nested[pos];
	space->nested[pos] = NULL;
	return nested;
}

/* Replace the nested space at the given position by "nested",
 * where this nested space of "space" may be missing
 * due to a preceding call to isl_space_take_nested.
 * However, in this case, "space" only has a single reference and
 * then the call to isl_space_cow has no effect.
 */
static __isl_give isl_space *isl_space_restore_nested(
	__isl_take isl_space *space, int pos, __isl_take isl_space *nested)
{
	if (!space || !nested)
		goto error;

	if (space->nested[pos] == nested) {
		isl_space_free(nested);
		return space;
	}

	space = isl_space_cow(space);
	if (!space)
		goto error;
	isl_space_free(space->nested[pos]);
	space->nested[pos] = nested;

	return space;
error:
	isl_space_free(space);
	isl_space_free(nested);
	return NULL;
}

/* Is it possible for the given dimension type to have a tuple id?
 */
static int space_can_have_id(__isl_keep isl_space *space,
	enum isl_dim_type type)
{
	if (!space)
		return 0;
	if (isl_space_is_params(space))
		isl_die(space->ctx, isl_error_invalid,
			"parameter spaces don't have tuple ids", return 0);
	if (isl_space_is_set(space) && type != isl_dim_set)
		isl_die(space->ctx, isl_error_invalid,
			"set spaces can only have a set id", return 0);
	if (type != isl_dim_in && type != isl_dim_out)
		isl_die(space->ctx, isl_error_invalid,
			"only input, output and set tuples can have ids",
			return 0);

	return 1;
}

/* Does the tuple have an id?
 */
isl_bool isl_space_has_tuple_id(__isl_keep isl_space *space,
	enum isl_dim_type type)
{
	if (!space_can_have_id(space, type))
		return isl_bool_error;
	return isl_bool_ok(space->tuple_id[type - isl_dim_in] != NULL);
}

/* Does the domain tuple of the map space "space" have an identifier?
 */
isl_bool isl_space_has_domain_tuple_id(__isl_keep isl_space *space)
{
	if (isl_space_check_is_map(space) < 0)
		return isl_bool_error;
	return isl_space_has_tuple_id(space, isl_dim_in);
}

/* Does the range tuple of the map space "space" have an identifier?
 */
isl_bool isl_space_has_range_tuple_id(__isl_keep isl_space *space)
{
	if (isl_space_check_is_map(space) < 0)
		return isl_bool_error;
	return isl_space_has_tuple_id(space, isl_dim_out);
}

__isl_give isl_id *isl_space_get_tuple_id(__isl_keep isl_space *space,
	enum isl_dim_type type)
{
	int has_id;

	if (!space)
		return NULL;
	has_id = isl_space_has_tuple_id(space, type);
	if (has_id < 0)
		return NULL;
	if (!has_id)
		isl_die(space->ctx, isl_error_invalid,
			"tuple has no id", return NULL);
	return isl_id_copy(space->tuple_id[type - isl_dim_in]);
}

/* Return the identifier of the domain tuple of the map space "space",
 * assuming it has one.
 */
__isl_give isl_id *isl_space_get_domain_tuple_id(
	__isl_keep isl_space *space)
{
	if (isl_space_check_is_map(space) < 0)
		return NULL;
	return isl_space_get_tuple_id(space, isl_dim_in);
}

/* Return the identifier of the range tuple of the map space "space",
 * assuming it has one.
 */
__isl_give isl_id *isl_space_get_range_tuple_id(
	__isl_keep isl_space *space)
{
	if (isl_space_check_is_map(space) < 0)
		return NULL;
	return isl_space_get_tuple_id(space, isl_dim_out);
}

__isl_give isl_space *isl_space_set_tuple_id(__isl_take isl_space *space,
	enum isl_dim_type type, __isl_take isl_id *id)
{
	space = isl_space_cow(space);
	if (!space || !id)
		goto error;
	if (type != isl_dim_in && type != isl_dim_out)
		isl_die(space->ctx, isl_error_invalid,
			"only input, output and set tuples can have names",
			goto error);

	isl_id_free(space->tuple_id[type - isl_dim_in]);
	space->tuple_id[type - isl_dim_in] = id;

	return space;
error:
	isl_id_free(id);
	isl_space_free(space);
	return NULL;
}

/* Replace the identifier of the domain tuple of the map space "space"
 * by "id".
 */
__isl_give isl_space *isl_space_set_domain_tuple_id(
	__isl_take isl_space *space, __isl_take isl_id *id)
{
	if (isl_space_check_is_map(space) < 0)
		space = isl_space_free(space);
	return isl_space_set_tuple_id(space, isl_dim_in, id);
}

/* Replace the identifier of the range tuple of the map space "space"
 * by "id".
 */
__isl_give isl_space *isl_space_set_range_tuple_id(
	__isl_take isl_space *space, __isl_take isl_id *id)
{
	if (isl_space_check_is_map(space) < 0)
		space = isl_space_free(space);
	return isl_space_set_tuple_id(space, isl_dim_out, id);
}

__isl_give isl_space *isl_space_reset_tuple_id(__isl_take isl_space *space,
	enum isl_dim_type type)
{
	space = isl_space_cow(space);
	if (!space)
		return NULL;
	if (type != isl_dim_in && type != isl_dim_out)
		isl_die(space->ctx, isl_error_invalid,
			"only input, output and set tuples can have names",
			goto error);

	isl_id_free(space->tuple_id[type - isl_dim_in]);
	space->tuple_id[type - isl_dim_in] = NULL;

	return space;
error:
	isl_space_free(space);
	return NULL;
}

/* Set the id of the given dimension of "space" to "id".
 * If the dimension already has an id, then it is replaced.
 * If the dimension is a parameter, then we need to change it
 * in the nested spaces (if any) as well.
 */
__isl_give isl_space *isl_space_set_dim_id(__isl_take isl_space *space,
	enum isl_dim_type type, unsigned pos, __isl_take isl_id *id)
{
	space = isl_space_cow(space);
	if (!space || !id)
		goto error;

	if (type == isl_dim_param) {
		int i;

		for (i = 0; i < 2; ++i) {
			if (!space->nested[i])
				continue;
			space->nested[i] =
				isl_space_set_dim_id(space->nested[i],
						type, pos, isl_id_copy(id));
			if (!space->nested[i])
				goto error;
		}
	}

	isl_id_free(get_id(space, type, pos));
	return set_id(space, type, pos, id);
error:
	isl_id_free(id);
	isl_space_free(space);
	return NULL;
}

/* Reset the id of the given dimension of "space".
 * If the dimension already has an id, then it is removed.
 * If the dimension is a parameter, then we need to reset it
 * in the nested spaces (if any) as well.
 */
__isl_give isl_space *isl_space_reset_dim_id(__isl_take isl_space *space,
	enum isl_dim_type type, unsigned pos)
{
	space = isl_space_cow(space);
	if (!space)
		goto error;

	if (type == isl_dim_param) {
		int i;

		for (i = 0; i < 2; ++i) {
			if (!space->nested[i])
				continue;
			space->nested[i] =
				isl_space_reset_dim_id(space->nested[i],
							type, pos);
			if (!space->nested[i])
				goto error;
		}
	}

	isl_id_free(get_id(space, type, pos));
	return set_id(space, type, pos, NULL);
error:
	isl_space_free(space);
	return NULL;
}

isl_bool isl_space_has_dim_id(__isl_keep isl_space *space,
	enum isl_dim_type type, unsigned pos)
{
	if (!space)
		return isl_bool_error;
	return isl_bool_ok(get_id(space, type, pos) != NULL);
}

__isl_give isl_id *isl_space_get_dim_id(__isl_keep isl_space *space,
	enum isl_dim_type type, unsigned pos)
{
	if (!space)
		return NULL;
	if (!get_id(space, type, pos))
		isl_die(space->ctx, isl_error_invalid,
			"dim has no id", return NULL);
	return isl_id_copy(get_id(space, type, pos));
}

__isl_give isl_space *isl_space_set_tuple_name(__isl_take isl_space *space,
	enum isl_dim_type type, const char *s)
{
	isl_id *id;

	if (!space)
		return NULL;

	if (!s)
		return isl_space_reset_tuple_id(space, type);

	if (!name_ok(space->ctx, s))
		goto error;

	id = isl_id_alloc(space->ctx, s, NULL);
	return isl_space_set_tuple_id(space, type, id);
error:
	isl_space_free(space);
	return NULL;
}

/* Does the tuple have a name?
 */
isl_bool isl_space_has_tuple_name(__isl_keep isl_space *space,
	enum isl_dim_type type)
{
	isl_id *id;

	if (!space_can_have_id(space, type))
		return isl_bool_error;
	id = space->tuple_id[type - isl_dim_in];
	return isl_bool_ok(id && id->name);
}

__isl_keep const char *isl_space_get_tuple_name(__isl_keep isl_space *space,
	 enum isl_dim_type type)
{
	isl_id *id;
	if (!space)
		return NULL;
	if (type != isl_dim_in && type != isl_dim_out)
		return NULL;
	id = space->tuple_id[type - isl_dim_in];
	return id ? id->name : NULL;
}

__isl_give isl_space *isl_space_set_dim_name(__isl_take isl_space *space,
				 enum isl_dim_type type, unsigned pos,
				 const char *s)
{
	isl_id *id;

	if (!space)
		return NULL;
	if (!s)
		return isl_space_reset_dim_id(space, type, pos);
	if (!name_ok(space->ctx, s))
		goto error;
	id = isl_id_alloc(space->ctx, s, NULL);
	return isl_space_set_dim_id(space, type, pos, id);
error:
	isl_space_free(space);
	return NULL;
}

/* Does the given dimension have a name?
 */
isl_bool isl_space_has_dim_name(__isl_keep isl_space *space,
	enum isl_dim_type type, unsigned pos)
{
	isl_id *id;

	if (!space)
		return isl_bool_error;
	id = get_id(space, type, pos);
	return isl_bool_ok(id && id->name);
}

__isl_keep const char *isl_space_get_dim_name(__isl_keep isl_space *space,
				 enum isl_dim_type type, unsigned pos)
{
	isl_id *id = get_id(space, type, pos);
	return id ? id->name : NULL;
}

int isl_space_find_dim_by_id(__isl_keep isl_space *space,
	enum isl_dim_type type, __isl_keep isl_id *id)
{
	int i;
	int offset;
	isl_size n;

	n = isl_space_dim(space, type);
	if (n < 0 || !id)
		return -1;

	offset = isl_space_offset(space, type);
	for (i = 0; i < n && offset + i < space->n_id; ++i)
		if (space->ids[offset + i] == id)
			return i;

	return -1;
}

int isl_space_find_dim_by_name(__isl_keep isl_space *space,
	enum isl_dim_type type, const char *name)
{
	int i;
	int offset;
	isl_size n;

	n = isl_space_dim(space, type);
	if (n < 0 || !name)
		return -1;

	offset = isl_space_offset(space, type);
	for (i = 0; i < n && offset + i < space->n_id; ++i) {
		isl_id *id = get_id(space, type, i);
		if (id && id->name && !strcmp(id->name, name))
			return i;
	}

	return -1;
}

/* Reset the user pointer on all identifiers of parameters and tuples
 * of "space".
 */
__isl_give isl_space *isl_space_reset_user(__isl_take isl_space *space)
{
	int i;
	isl_ctx *ctx;
	isl_id *id;
	const char *name;

	if (!space)
		return NULL;

	ctx = isl_space_get_ctx(space);

	for (i = 0; i < space->nparam && i < space->n_id; ++i) {
		if (!isl_id_get_user(space->ids[i]))
			continue;
		space = isl_space_cow(space);
		if (!space)
			return NULL;
		name = isl_id_get_name(space->ids[i]);
		id = isl_id_alloc(ctx, name, NULL);
		isl_id_free(space->ids[i]);
		space->ids[i] = id;
		if (!id)
			return isl_space_free(space);
	}

	for (i = 0; i < 2; ++i) {
		if (!space->tuple_id[i])
			continue;
		if (!isl_id_get_user(space->tuple_id[i]))
			continue;
		space = isl_space_cow(space);
		if (!space)
			return NULL;
		name = isl_id_get_name(space->tuple_id[i]);
		id = isl_id_alloc(ctx, name, NULL);
		isl_id_free(space->tuple_id[i]);
		space->tuple_id[i] = id;
		if (!id)
			return isl_space_free(space);
	}

	for (i = 0; i < 2; ++i) {
		isl_space *nested;

		if (!space->nested[i])
			continue;
		nested = isl_space_take_nested(space, i);
		nested = isl_space_reset_user(nested);
		space = isl_space_restore_nested(space, i, nested);
		if (!space)
			return NULL;
	}

	return space;
}

static __isl_keep isl_id *tuple_id(__isl_keep isl_space *space,
	enum isl_dim_type type)
{
	if (!space)
		return NULL;
	if (type == isl_dim_in)
		return space->tuple_id[0];
	if (type == isl_dim_out)
		return space->tuple_id[1];
	return NULL;
}

static __isl_keep isl_space *nested(__isl_keep isl_space *space,
	enum isl_dim_type type)
{
	if (!space)
		return NULL;
	if (type == isl_dim_in)
		return space->nested[0];
	if (type == isl_dim_out)
		return space->nested[1];
	return NULL;
}

/* Are the two spaces the same, apart from positions and names of parameters?
 */
isl_bool isl_space_has_equal_tuples(__isl_keep isl_space *space1,
	__isl_keep isl_space *space2)
{
	if (!space1 || !space2)
		return isl_bool_error;
	if (space1 == space2)
		return isl_bool_true;
	return isl_space_tuple_is_equal(space1, isl_dim_in,
					space2, isl_dim_in) &&
	       isl_space_tuple_is_equal(space1, isl_dim_out,
					space2, isl_dim_out);
}

/* Check that a match involving "space" was successful.
 * That is, check that "match" is equal to isl_bool_true.
 */
static isl_stat check_match(__isl_keep isl_space *space, isl_bool match)
{
	if (match < 0)
		return isl_stat_error;
	if (!match)
		isl_die(isl_space_get_ctx(space), isl_error_invalid,
			"incompatible spaces", return isl_stat_error);

	return isl_stat_ok;
}

/* Check that the two spaces are the same,
 * apart from positions and names of parameters.
 */
isl_stat isl_space_check_equal_tuples(__isl_keep isl_space *space1,
	__isl_keep isl_space *space2)
{
	isl_bool is_equal;

	is_equal = isl_space_has_equal_tuples(space1, space2);
	return check_match(space1, is_equal);
}

/* Check if the tuple of type "type1" of "space1" is the same as
 * the tuple of type "type2" of "space2".
 *
 * That is, check if the tuples have the same identifier, the same dimension
 * and the same internal structure.
 * The identifiers of the dimensions inside the tuples do not affect the result.
 *
 * Note that this function only checks the tuples themselves.
 * If nested tuples are involved, then we need to be careful not
 * to have result affected by possibly differing parameters
 * in those nested tuples.
 */
isl_bool isl_space_tuple_is_equal(__isl_keep isl_space *space1,
	enum isl_dim_type type1, __isl_keep isl_space *space2,
	enum isl_dim_type type2)
{
	isl_id *id1, *id2;
	isl_space *nested1, *nested2;

	if (!space1 || !space2)
		return isl_bool_error;

	if (space1 == space2 && type1 == type2)
		return isl_bool_true;

	if (n(space1, type1) != n(space2, type2))
		return isl_bool_false;
	id1 = tuple_id(space1, type1);
	id2 = tuple_id(space2, type2);
	if (!id1 ^ !id2)
		return isl_bool_false;
	if (id1 && id1 != id2)
		return isl_bool_false;
	nested1 = nested(space1, type1);
	nested2 = nested(space2, type2);
	if (!nested1 ^ !nested2)
		return isl_bool_false;
	if (nested1 && !isl_space_has_equal_tuples(nested1, nested2))
		return isl_bool_false;
	return isl_bool_true;
}

/* Is the tuple "inner" within the wrapped relation inside tuple "outer"
 * of "space1" equal to tuple "type2" of "space2"?
 */
isl_bool isl_space_wrapped_tuple_is_equal(__isl_keep isl_space *space1,
	enum isl_dim_type outer, enum isl_dim_type inner,
	__isl_keep isl_space *space2, enum isl_dim_type type2)
{
	int pos;
	isl_space *nested;

	if (!space1)
		return isl_bool_error;
	if (outer != isl_dim_in && outer != isl_dim_out)
		isl_die(isl_space_get_ctx(space1), isl_error_invalid,
			"only input, output and set tuples "
			"can have nested relations", return isl_bool_error);
	pos = outer - isl_dim_in;
	nested = isl_space_peek_nested(space1, pos);
	return isl_space_tuple_is_equal(nested, inner, space2, type2);
}

/* Check that the tuple "inner" within the wrapped relation inside tuple "outer"
 * of "space1" is equal to tuple "type2" of "space2".
 */
isl_stat isl_space_check_wrapped_tuple_is_equal(__isl_keep isl_space *space1,
	enum isl_dim_type outer, enum isl_dim_type inner,
	__isl_keep isl_space *space2, enum isl_dim_type type2)
{
	isl_bool is_equal;

	is_equal = isl_space_wrapped_tuple_is_equal(space1, outer, inner,
							space2, type2);
	return check_match(space1, is_equal);
}

static isl_bool match(__isl_keep isl_space *space1, enum isl_dim_type type1,
	__isl_keep isl_space *space2, enum isl_dim_type type2)
{
	int i;
	isl_bool equal;

	if (!space1 || !space2)
		return isl_bool_error;

	if (space1 == space2 && type1 == type2)
		return isl_bool_true;

	equal = isl_space_tuple_is_equal(space1, type1, space2, type2);
	if (equal < 0 || !equal)
		return equal;

	if (!space1->ids && !space2->ids)
		return isl_bool_true;

	for (i = 0; i < n(space1, type1); ++i) {
		if (get_id(space1, type1, i) != get_id(space2, type2, i))
			return isl_bool_false;
	}
	return isl_bool_true;
}

/* Do "space1" and "space2" have the same parameters?
 */
isl_bool isl_space_has_equal_params(__isl_keep isl_space *space1,
	__isl_keep isl_space *space2)
{
	return match(space1, isl_dim_param, space2, isl_dim_param);
}

/* Do "space1" and "space2" have the same identifiers for all
 * the tuple variables?
 */
isl_bool isl_space_has_equal_ids(__isl_keep isl_space *space1,
	__isl_keep isl_space *space2)
{
	isl_bool equal;

	equal = match(space1, isl_dim_in, space2, isl_dim_in);
	if (equal < 0 || !equal)
		return equal;
	return match(space1, isl_dim_out, space2, isl_dim_out);
}

isl_bool isl_space_match(__isl_keep isl_space *space1, enum isl_dim_type type1,
	__isl_keep isl_space *space2, enum isl_dim_type type2)
{
	return match(space1, type1, space2, type2);
}

static void get_ids(__isl_keep isl_space *space, enum isl_dim_type type,
	unsigned first, unsigned n, __isl_keep isl_id **ids)
{
	int i;

	for (i = 0; i < n ; ++i)
		ids[i] = get_id(space, type, first + i);
}

static __isl_give isl_space *space_extend(__isl_take isl_space *space,
			unsigned nparam, unsigned n_in, unsigned n_out)
{
	isl_id **ids = NULL;

	if (!space)
		return NULL;
	if (space->nparam == nparam &&
	    space->n_in == n_in && space->n_out == n_out)
		return space;

	isl_assert(space->ctx, space->nparam <= nparam, goto error);
	isl_assert(space->ctx, space->n_in <= n_in, goto error);
	isl_assert(space->ctx, space->n_out <= n_out, goto error);

	space = isl_space_cow(space);
	if (!space)
		goto error;

	if (space->ids) {
		unsigned n;
		n = nparam + n_in + n_out;
		if (n < nparam || n < n_in || n < n_out)
			isl_die(isl_space_get_ctx(space), isl_error_invalid,
				"overflow in total number of dimensions",
				goto error);
		ids = isl_calloc_array(space->ctx, isl_id *, n);
		if (!ids)
			goto error;
		get_ids(space, isl_dim_param, 0, space->nparam, ids);
		get_ids(space, isl_dim_in, 0, space->n_in, ids + nparam);
		get_ids(space, isl_dim_out, 0, space->n_out,
			ids + nparam + n_in);
		free(space->ids);
		space->ids = ids;
		space->n_id = nparam + n_in + n_out;
	}
	space->nparam = nparam;
	space->n_in = n_in;
	space->n_out = n_out;

	return space;
error:
	free(ids);
	isl_space_free(space);
	return NULL;
}

__isl_give isl_space *isl_space_extend(__isl_take isl_space *space,
	unsigned nparam, unsigned n_in, unsigned n_out)
{
	return space_extend(space, nparam, n_in, n_out);
}

__isl_give isl_space *isl_space_add_dims(__isl_take isl_space *space,
	enum isl_dim_type type, unsigned n)
{
	space = isl_space_reset(space, type);
	if (!space)
		return NULL;
	switch (type) {
	case isl_dim_param:
		space = space_extend(space,
				space->nparam + n, space->n_in, space->n_out);
		if (space && space->nested[0] &&
		    !(space->nested[0] = isl_space_add_dims(space->nested[0],
						    isl_dim_param, n)))
			goto error;
		if (space && space->nested[1] &&
		    !(space->nested[1] = isl_space_add_dims(space->nested[1],
						    isl_dim_param, n)))
			goto error;
		return space;
	case isl_dim_in:
		return space_extend(space,
				space->nparam, space->n_in + n, space->n_out);
	case isl_dim_out:
		return space_extend(space,
				space->nparam, space->n_in, space->n_out + n);
	default:
		isl_die(space->ctx, isl_error_invalid,
			"cannot add dimensions of specified type", goto error);
	}
error:
	isl_space_free(space);
	return NULL;
}

/* Add a parameter with identifier "id" to "space", provided
 * it does not already appear in "space".
 */
__isl_give isl_space *isl_space_add_param_id(__isl_take isl_space *space,
	__isl_take isl_id *id)
{
	isl_size pos;

	if (!space || !id)
		goto error;

	if (isl_space_find_dim_by_id(space, isl_dim_param, id) >= 0) {
		isl_id_free(id);
		return space;
	}

	pos = isl_space_dim(space, isl_dim_param);
	if (pos < 0)
		goto error;
	space = isl_space_add_dims(space, isl_dim_param, 1);
	space = isl_space_set_dim_id(space, isl_dim_param, pos, id);

	return space;
error:
	isl_space_free(space);
	isl_id_free(id);
	return NULL;
}

static int valid_dim_type(enum isl_dim_type type)
{
	switch (type) {
	case isl_dim_param:
	case isl_dim_in:
	case isl_dim_out:
		return 1;
	default:
		return 0;
	}
}

#undef TYPE
#define TYPE	isl_space
#include "check_type_range_templ.c"

/* Insert "n" dimensions of type "type" at position "pos".
 * If we are inserting parameters, then they are also inserted in
 * any nested spaces.
 */
__isl_give isl_space *isl_space_insert_dims(__isl_take isl_space *space,
	enum isl_dim_type type, unsigned pos, unsigned n)
{
	isl_ctx *ctx;
	isl_id **ids = NULL;

	if (!space)
		return NULL;
	if (n == 0)
		return isl_space_reset(space, type);

	ctx = isl_space_get_ctx(space);
	if (!valid_dim_type(type))
		isl_die(ctx, isl_error_invalid,
			"cannot insert dimensions of specified type",
			goto error);

	if (isl_space_check_range(space, type, pos, 0) < 0)
		return isl_space_free(space);

	space = isl_space_cow(space);
	if (!space)
		return NULL;

	if (space->ids) {
		enum isl_dim_type t, o = isl_dim_param;
		int off;
		int s[3];
		ids = isl_calloc_array(ctx, isl_id *,
			     space->nparam + space->n_in + space->n_out + n);
		if (!ids)
			goto error;
		off = 0;
		s[isl_dim_param - o] = space->nparam;
		s[isl_dim_in - o] = space->n_in;
		s[isl_dim_out - o] = space->n_out;
		for (t = isl_dim_param; t <= isl_dim_out; ++t) {
			if (t != type) {
				get_ids(space, t, 0, s[t - o], ids + off);
				off += s[t - o];
			} else {
				get_ids(space, t, 0, pos, ids + off);
				off += pos + n;
				get_ids(space, t, pos, s[t - o] - pos,
					ids + off);
				off += s[t - o] - pos;
			}
		}
		free(space->ids);
		space->ids = ids;
		space->n_id = space->nparam + space->n_in + space->n_out + n;
	}
	switch (type) {
	case isl_dim_param:	space->nparam += n; break;
	case isl_dim_in:	space->n_in += n; break;
	case isl_dim_out:	space->n_out += n; break;
	default:		;
	}
	space = isl_space_reset(space, type);

	if (type == isl_dim_param) {
		if (space && space->nested[0] &&
		    !(space->nested[0] = isl_space_insert_dims(space->nested[0],
						    isl_dim_param, pos, n)))
			goto error;
		if (space && space->nested[1] &&
		    !(space->nested[1] = isl_space_insert_dims(space->nested[1],
						    isl_dim_param, pos, n)))
			goto error;
	}

	return space;
error:
	isl_space_free(space);
	return NULL;
}

__isl_give isl_space *isl_space_move_dims(__isl_take isl_space *space,
	enum isl_dim_type dst_type, unsigned dst_pos,
	enum isl_dim_type src_type, unsigned src_pos, unsigned n)
{
	int i;

	space = isl_space_reset(space, src_type);
	space = isl_space_reset(space, dst_type);
	if (!space)
		return NULL;
	if (n == 0)
		return space;

	if (isl_space_check_range(space, src_type, src_pos, n) < 0)
		return isl_space_free(space);

	if (dst_type == src_type && dst_pos == src_pos)
		return space;

	isl_assert(space->ctx, dst_type != src_type, goto error);

	space = isl_space_cow(space);
	if (!space)
		return NULL;

	if (space->ids) {
		isl_id **ids;
		enum isl_dim_type t, o = isl_dim_param;
		int off;
		int s[3];
		ids = isl_calloc_array(space->ctx, isl_id *,
				 space->nparam + space->n_in + space->n_out);
		if (!ids)
			goto error;
		off = 0;
		s[isl_dim_param - o] = space->nparam;
		s[isl_dim_in - o] = space->n_in;
		s[isl_dim_out - o] = space->n_out;
		for (t = isl_dim_param; t <= isl_dim_out; ++t) {
			if (t == dst_type) {
				get_ids(space, t, 0, dst_pos, ids + off);
				off += dst_pos;
				get_ids(space, src_type, src_pos, n, ids + off);
				off += n;
				get_ids(space, t, dst_pos, s[t - o] - dst_pos,
						ids + off);
				off += s[t - o] - dst_pos;
			} else if (t == src_type) {
				get_ids(space, t, 0, src_pos, ids + off);
				off += src_pos;
				get_ids(space, t, src_pos + n,
					    s[t - o] - src_pos - n, ids + off);
				off += s[t - o] - src_pos - n;
			} else {
				get_ids(space, t, 0, s[t - o], ids + off);
				off += s[t - o];
			}
		}
		free(space->ids);
		space->ids = ids;
		space->n_id = space->nparam + space->n_in + space->n_out;
	}

	switch (dst_type) {
	case isl_dim_param:	space->nparam += n; break;
	case isl_dim_in:	space->n_in += n; break;
	case isl_dim_out:	space->n_out += n; break;
	default:		;
	}

	switch (src_type) {
	case isl_dim_param:	space->nparam -= n; break;
	case isl_dim_in:	space->n_in -= n; break;
	case isl_dim_out:	space->n_out -= n; break;
	default:		;
	}

	if (dst_type != isl_dim_param && src_type != isl_dim_param)
		return space;

	for (i = 0; i < 2; ++i) {
		isl_space *nested;

		if (!space->nested[i])
			continue;
		nested = isl_space_take_nested(space, i);
		nested = isl_space_replace_params(nested, space);
		space = isl_space_restore_nested(space, i, nested);
		if (!space)
			return NULL;
	}

	return space;
error:
	isl_space_free(space);
	return NULL;
}

/* Check that "space1" and "space2" have the same parameters,
 * reporting an error if they do not.
 */
isl_stat isl_space_check_equal_params(__isl_keep isl_space *space1,
	__isl_keep isl_space *space2)
{
	isl_bool equal;

	equal = isl_space_has_equal_params(space1, space2);
	if (equal < 0)
		return isl_stat_error;
	if (!equal)
		isl_die(isl_space_get_ctx(space1), isl_error_invalid,
			"parameters need to match", return isl_stat_error);
	return isl_stat_ok;
}

__isl_give isl_space *isl_space_join(__isl_take isl_space *left,
	__isl_take isl_space *right)
{
	isl_space *space;

	if (isl_space_check_equal_params(left, right) < 0)
		goto error;

	isl_assert(left->ctx,
		isl_space_tuple_is_equal(left, isl_dim_out, right, isl_dim_in),
		goto error);

	space = isl_space_alloc(left->ctx,
				left->nparam, left->n_in, right->n_out);
	if (!space)
		goto error;

	space = copy_ids(space, isl_dim_param, 0, left, isl_dim_param);
	space = copy_ids(space, isl_dim_in, 0, left, isl_dim_in);
	space = copy_ids(space, isl_dim_out, 0, right, isl_dim_out);

	if (space && left->tuple_id[0] &&
	    !(space->tuple_id[0] = isl_id_copy(left->tuple_id[0])))
		goto error;
	if (space && right->tuple_id[1] &&
	    !(space->tuple_id[1] = isl_id_copy(right->tuple_id[1])))
		goto error;
	if (space && left->nested[0] &&
	    !(space->nested[0] = isl_space_copy(left->nested[0])))
		goto error;
	if (space && right->nested[1] &&
	    !(space->nested[1] = isl_space_copy(right->nested[1])))
		goto error;

	isl_space_free(left);
	isl_space_free(right);

	return space;
error:
	isl_space_free(left);
	isl_space_free(right);
	return NULL;
}

/* Given two map spaces { A -> C } and { B -> D }, construct the space
 * { [A -> B] -> [C -> D] }.
 * Given two set spaces { A } and { B }, construct the space { [A -> B] }.
 */
__isl_give isl_space *isl_space_product(__isl_take isl_space *left,
	__isl_take isl_space *right)
{
	isl_space *dom1, *dom2, *nest1, *nest2;
	int is_set;

	if (!left || !right)
		goto error;

	is_set = isl_space_is_set(left);
	if (is_set != isl_space_is_set(right))
		isl_die(isl_space_get_ctx(left), isl_error_invalid,
			"expecting either two set spaces or two map spaces",
			goto error);
	if (is_set)
		return isl_space_range_product(left, right);

	if (isl_space_check_equal_params(left, right) < 0)
		goto error;

	dom1 = isl_space_domain(isl_space_copy(left));
	dom2 = isl_space_domain(isl_space_copy(right));
	nest1 = isl_space_wrap(isl_space_join(isl_space_reverse(dom1), dom2));

	dom1 = isl_space_range(left);
	dom2 = isl_space_range(right);
	nest2 = isl_space_wrap(isl_space_join(isl_space_reverse(dom1), dom2));

	return isl_space_join(isl_space_reverse(nest1), nest2);
error:
	isl_space_free(left);
	isl_space_free(right);
	return NULL;
}

/* Given two spaces { A -> C } and { B -> C }, construct the space
 * { [A -> B] -> C }
 */
__isl_give isl_space *isl_space_domain_product(__isl_take isl_space *left,
	__isl_take isl_space *right)
{
	isl_space *ran, *dom1, *dom2, *nest;

	if (isl_space_check_equal_params(left, right) < 0)
		goto error;

	if (!isl_space_tuple_is_equal(left, isl_dim_out, right, isl_dim_out))
		isl_die(left->ctx, isl_error_invalid,
			"ranges need to match", goto error);

	ran = isl_space_range(isl_space_copy(left));

	dom1 = isl_space_domain(left);
	dom2 = isl_space_domain(right);
	nest = isl_space_wrap(isl_space_join(isl_space_reverse(dom1), dom2));

	return isl_space_join(isl_space_reverse(nest), ran);
error:
	isl_space_free(left);
	isl_space_free(right);
	return NULL;
}

__isl_give isl_space *isl_space_range_product(__isl_take isl_space *left,
	__isl_take isl_space *right)
{
	isl_space *dom, *ran1, *ran2, *nest;

	if (isl_space_check_equal_params(left, right) < 0)
		goto error;

	if (!isl_space_tuple_is_equal(left, isl_dim_in, right, isl_dim_in))
		isl_die(left->ctx, isl_error_invalid,
			"domains need to match", goto error);

	dom = isl_space_domain(isl_space_copy(left));

	ran1 = isl_space_range(left);
	ran2 = isl_space_range(right);
	nest = isl_space_wrap(isl_space_join(isl_space_reverse(ran1), ran2));

	return isl_space_join(isl_space_reverse(dom), nest);
error:
	isl_space_free(left);
	isl_space_free(right);
	return NULL;
}

/* Given a space of the form [A -> B] -> C, return the space A -> C.
 */
__isl_give isl_space *isl_space_domain_factor_domain(
	__isl_take isl_space *space)
{
	isl_space *nested;
	isl_space *domain;

	if (isl_space_check_domain_is_wrapping(space) < 0)
		return isl_space_free(space);

	nested = space->nested[0];
	domain = isl_space_copy(space);
	domain = isl_space_drop_dims(domain, isl_dim_in,
					nested->n_in, nested->n_out);
	if (!domain)
		return isl_space_free(space);
	if (nested->tuple_id[0]) {
		domain->tuple_id[0] = isl_id_copy(nested->tuple_id[0]);
		if (!domain->tuple_id[0])
			goto error;
	}
	if (nested->nested[0]) {
		domain->nested[0] = isl_space_copy(nested->nested[0]);
		if (!domain->nested[0])
			goto error;
	}

	isl_space_free(space);
	return domain;
error:
	isl_space_free(space);
	isl_space_free(domain);
	return NULL;
}

/* Given a space of the form [A -> B] -> C, return the space B -> C.
 */
__isl_give isl_space *isl_space_domain_factor_range(
	__isl_take isl_space *space)
{
	isl_space *nested;
	isl_space *range;

	if (isl_space_check_domain_is_wrapping(space) < 0)
		return isl_space_free(space);

	nested = space->nested[0];
	range = isl_space_copy(space);
	range = isl_space_drop_dims(range, isl_dim_in, 0, nested->n_in);
	if (!range)
		return isl_space_free(space);
	if (nested->tuple_id[1]) {
		range->tuple_id[0] = isl_id_copy(nested->tuple_id[1]);
		if (!range->tuple_id[0])
			goto error;
	}
	if (nested->nested[1]) {
		range->nested[0] = isl_space_copy(nested->nested[1]);
		if (!range->nested[0])
			goto error;
	}

	isl_space_free(space);
	return range;
error:
	isl_space_free(space);
	isl_space_free(range);
	return NULL;
}

/* Internal function that selects the domain of the map that is
 * embedded in either a set space or the range of a map space.
 * In particular, given a space of the form [A -> B], return the space A.
 * Given a space of the form A -> [B -> C], return the space A -> B.
 */
static __isl_give isl_space *range_factor_domain(__isl_take isl_space *space)
{
	isl_space *nested;
	isl_space *domain;

	if (!space)
		return NULL;

	nested = space->nested[1];
	domain = isl_space_copy(space);
	domain = isl_space_drop_dims(domain, isl_dim_out,
					nested->n_in, nested->n_out);
	if (!domain)
		return isl_space_free(space);
	if (nested->tuple_id[0]) {
		domain->tuple_id[1] = isl_id_copy(nested->tuple_id[0]);
		if (!domain->tuple_id[1])
			goto error;
	}
	if (nested->nested[0]) {
		domain->nested[1] = isl_space_copy(nested->nested[0]);
		if (!domain->nested[1])
			goto error;
	}

	isl_space_free(space);
	return domain;
error:
	isl_space_free(space);
	isl_space_free(domain);
	return NULL;
}

/* Given a space of the form A -> [B -> C], return the space A -> B.
 */
__isl_give isl_space *isl_space_range_factor_domain(
	__isl_take isl_space *space)
{
	if (isl_space_check_range_is_wrapping(space) < 0)
		return isl_space_free(space);

	return range_factor_domain(space);
}

/* Given a space of the form [A -> B], return the space A.
 */
static __isl_give isl_space *set_factor_domain(__isl_take isl_space *space)
{
	if (!space)
		return NULL;
	if (!isl_space_is_wrapping(space))
		isl_die(isl_space_get_ctx(space), isl_error_invalid,
			"not a product", return isl_space_free(space));

	return range_factor_domain(space);
}

/* Given a space of the form [A -> B] -> [C -> D], return the space A -> C.
 * Given a space of the form [A -> B], return the space A.
 */
__isl_give isl_space *isl_space_factor_domain(__isl_take isl_space *space)
{
	if (!space)
		return NULL;
	if (isl_space_is_set(space))
		return set_factor_domain(space);
	space = isl_space_domain_factor_domain(space);
	space = isl_space_range_factor_domain(space);
	return space;
}

/* Internal function that selects the range of the map that is
 * embedded in either a set space or the range of a map space.
 * In particular, given a space of the form [A -> B], return the space B.
 * Given a space of the form A -> [B -> C], return the space A -> C.
 */
static __isl_give isl_space *range_factor_range(__isl_take isl_space *space)
{
	isl_space *nested;
	isl_space *range;

	if (!space)
		return NULL;

	nested = space->nested[1];
	range = isl_space_copy(space);
	range = isl_space_drop_dims(range, isl_dim_out, 0, nested->n_in);
	if (!range)
		return isl_space_free(space);
	if (nested->tuple_id[1]) {
		range->tuple_id[1] = isl_id_copy(nested->tuple_id[1]);
		if (!range->tuple_id[1])
			goto error;
	}
	if (nested->nested[1]) {
		range->nested[1] = isl_space_copy(nested->nested[1]);
		if (!range->nested[1])
			goto error;
	}

	isl_space_free(space);
	return range;
error:
	isl_space_free(space);
	isl_space_free(range);
	return NULL;
}

/* Given a space of the form A -> [B -> C], return the space A -> C.
 */
__isl_give isl_space *isl_space_range_factor_range(
	__isl_take isl_space *space)
{
	if (isl_space_check_range_is_wrapping(space) < 0)
		return isl_space_free(space);

	return range_factor_range(space);
}

/* Given a space of the form [A -> B], return the space B.
 */
static __isl_give isl_space *set_factor_range(__isl_take isl_space *space)
{
	if (!space)
		return NULL;
	if (!isl_space_is_wrapping(space))
		isl_die(isl_space_get_ctx(space), isl_error_invalid,
			"not a product", return isl_space_free(space));

	return range_factor_range(space);
}

/* Given a space of the form [A -> B] -> [C -> D], return the space B -> D.
 * Given a space of the form [A -> B], return the space B.
 */
__isl_give isl_space *isl_space_factor_range(__isl_take isl_space *space)
{
	if (!space)
		return NULL;
	if (isl_space_is_set(space))
		return set_factor_range(space);
	space = isl_space_domain_factor_range(space);
	space = isl_space_range_factor_range(space);
	return space;
}

__isl_give isl_space *isl_space_map_from_set(__isl_take isl_space *space)
{
	isl_ctx *ctx;
	isl_id **ids = NULL;
	int n_id;

	if (!space)
		return NULL;
	ctx = isl_space_get_ctx(space);
	if (!isl_space_is_set(space))
		isl_die(ctx, isl_error_invalid, "not a set space", goto error);
	space = isl_space_cow(space);
	if (!space)
		return NULL;
	n_id = space->nparam + space->n_out + space->n_out;
	if (n_id > 0 && space->ids) {
		ids = isl_calloc_array(space->ctx, isl_id *, n_id);
		if (!ids)
			goto error;
		get_ids(space, isl_dim_param, 0, space->nparam, ids);
		get_ids(space, isl_dim_out, 0, space->n_out,
			ids + space->nparam);
	}
	space->n_in = space->n_out;
	if (ids) {
		free(space->ids);
		space->ids = ids;
		space->n_id = n_id;
		space = copy_ids(space, isl_dim_out, 0, space, isl_dim_in);
	}
	isl_id_free(space->tuple_id[0]);
	space->tuple_id[0] = isl_id_copy(space->tuple_id[1]);
	isl_space_free(space->nested[0]);
	space->nested[0] = isl_space_copy(space->nested[1]);
	return space;
error:
	isl_space_free(space);
	return NULL;
}

__isl_give isl_space *isl_space_map_from_domain_and_range(
	__isl_take isl_space *domain, __isl_take isl_space *range)
{
	if (!domain || !range)
		goto error;
	if (!isl_space_is_set(domain))
		isl_die(isl_space_get_ctx(domain), isl_error_invalid,
			"domain is not a set space", goto error);
	if (!isl_space_is_set(range))
		isl_die(isl_space_get_ctx(range), isl_error_invalid,
			"range is not a set space", goto error);
	return isl_space_join(isl_space_reverse(domain), range);
error:
	isl_space_free(domain);
	isl_space_free(range);
	return NULL;
}

static __isl_give isl_space *set_ids(__isl_take isl_space *space,
	enum isl_dim_type type,
	unsigned first, unsigned n, __isl_take isl_id **ids)
{
	int i;

	for (i = 0; i < n ; ++i)
		space = set_id(space, type, first + i, ids[i]);

	return space;
}

__isl_give isl_space *isl_space_reverse(__isl_take isl_space *space)
{
	unsigned t;
	isl_bool equal;
	isl_space *nested;
	isl_id **ids = NULL;
	isl_id *id;

	equal = match(space, isl_dim_in, space, isl_dim_out);
	if (equal < 0)
		return isl_space_free(space);
	if (equal)
		return space;

	space = isl_space_cow(space);
	if (!space)
		return NULL;

	id = space->tuple_id[0];
	space->tuple_id[0] = space->tuple_id[1];
	space->tuple_id[1] = id;

	nested = space->nested[0];
	space->nested[0] = space->nested[1];
	space->nested[1] = nested;

	if (space->ids) {
		int n_id = space->n_in + space->n_out;
		ids = isl_alloc_array(space->ctx, isl_id *, n_id);
		if (n_id && !ids)
			goto error;
		get_ids(space, isl_dim_in, 0, space->n_in, ids);
		get_ids(space, isl_dim_out, 0, space->n_out, ids + space->n_in);
	}

	t = space->n_in;
	space->n_in = space->n_out;
	space->n_out = t;

	if (space->ids) {
		space = set_ids(space, isl_dim_out, 0, space->n_out, ids);
		space = set_ids(space, isl_dim_in, 0, space->n_in,
				ids + space->n_out);
		free(ids);
	}

	return space;
error:
	free(ids);
	isl_space_free(space);
	return NULL;
}

/* Given a space A -> (B -> C), return the corresponding space
 * A -> (C -> B).
 *
 * If the range tuple is named, then the name is only preserved
 * if B and C are equal tuples, in which case the output
 * of this function is identical to the input.
 */
__isl_give isl_space *isl_space_range_reverse(__isl_take isl_space *space)
{
	isl_space *nested;
	isl_bool equal;

	if (isl_space_check_range_is_wrapping(space) < 0)
		return isl_space_free(space);

	nested = isl_space_peek_nested(space, 1);
	equal = isl_space_tuple_is_equal(nested, isl_dim_in,
					nested, isl_dim_out);
	if (equal < 0)
		return isl_space_free(space);

	nested = isl_space_take_nested(space, 1);
	nested = isl_space_reverse(nested);
	space = isl_space_restore_nested(space, 1, nested);
	if (!equal)
		space = isl_space_reset_tuple_id(space, isl_dim_out);

	return space;
}

__isl_give isl_space *isl_space_drop_dims(__isl_take isl_space *space,
	enum isl_dim_type type, unsigned first, unsigned num)
{
	int i;

	if (!space)
		return NULL;

	if (num == 0)
		return isl_space_reset(space, type);

	if (!valid_dim_type(type))
		isl_die(space->ctx, isl_error_invalid,
			"cannot drop dimensions of specified type", goto error);

	if (isl_space_check_range(space, type, first, num) < 0)
		return isl_space_free(space);
	space = isl_space_cow(space);
	if (!space)
		goto error;
	if (space->ids) {
		space = extend_ids(space);
		if (!space)
			goto error;
		for (i = 0; i < num; ++i)
			isl_id_free(get_id(space, type, first + i));
		for (i = first+num; i < n(space, type); ++i)
			set_id(space, type, i - num, get_id(space, type, i));
		switch (type) {
		case isl_dim_param:
			get_ids(space, isl_dim_in, 0, space->n_in,
				space->ids + offset(space, isl_dim_in) - num);
		case isl_dim_in:
			get_ids(space, isl_dim_out, 0, space->n_out,
				space->ids + offset(space, isl_dim_out) - num);
		default:
			;
		}
		space->n_id -= num;
	}
	switch (type) {
	case isl_dim_param:	space->nparam -= num; break;
	case isl_dim_in:	space->n_in -= num; break;
	case isl_dim_out:	space->n_out -= num; break;
	default:		;
	}
	space = isl_space_reset(space, type);
	if (type == isl_dim_param) {
		if (space && space->nested[0] &&
		    !(space->nested[0] = isl_space_drop_dims(space->nested[0],
						    isl_dim_param, first, num)))
			goto error;
		if (space && space->nested[1] &&
		    !(space->nested[1] = isl_space_drop_dims(space->nested[1],
						    isl_dim_param, first, num)))
			goto error;
	}
	return space;
error:
	isl_space_free(space);
	return NULL;
}

__isl_give isl_space *isl_space_drop_inputs(__isl_take isl_space *space,
		unsigned first, unsigned n)
{
	if (!space)
		return NULL;
	return isl_space_drop_dims(space, isl_dim_in, first, n);
}

__isl_give isl_space *isl_space_drop_outputs(__isl_take isl_space *space,
		unsigned first, unsigned n)
{
	if (!space)
		return NULL;
	return isl_space_drop_dims(space, isl_dim_out, first, n);
}

/* Remove all parameters from "space".
 */
__isl_give isl_space *isl_space_drop_all_params(__isl_take isl_space *space)
{
	isl_size nparam;

	nparam = isl_space_dim(space, isl_dim_param);
	if (nparam < 0)
		return isl_space_free(space);
	return isl_space_drop_dims(space, isl_dim_param, 0, nparam);
}

__isl_give isl_space *isl_space_domain(__isl_take isl_space *space)
{
	if (!space)
		return NULL;
	space = isl_space_drop_dims(space, isl_dim_out, 0, space->n_out);
	space = isl_space_reverse(space);
	space = mark_as_set(space);
	return space;
}

__isl_give isl_space *isl_space_from_domain(__isl_take isl_space *space)
{
	if (!space)
		return NULL;
	if (!isl_space_is_set(space))
		isl_die(isl_space_get_ctx(space), isl_error_invalid,
			"not a set space", goto error);
	space = isl_space_reverse(space);
	space = isl_space_reset(space, isl_dim_out);
	return space;
error:
	isl_space_free(space);
	return NULL;
}

__isl_give isl_space *isl_space_range(__isl_take isl_space *space)
{
	if (!space)
		return NULL;
	space = isl_space_drop_dims(space, isl_dim_in, 0, space->n_in);
	space = mark_as_set(space);
	return space;
}

__isl_give isl_space *isl_space_from_range(__isl_take isl_space *space)
{
	if (!space)
		return NULL;
	if (!isl_space_is_set(space))
		isl_die(isl_space_get_ctx(space), isl_error_invalid,
			"not a set space", goto error);
	return isl_space_reset(space, isl_dim_in);
error:
	isl_space_free(space);
	return NULL;
}

/* Given a map space A -> B, return the map space [A -> B] -> A.
 */
__isl_give isl_space *isl_space_domain_map(__isl_take isl_space *space)
{
	isl_space *domain;

	domain = isl_space_from_range(isl_space_domain(isl_space_copy(space)));
	space = isl_space_from_domain(isl_space_wrap(space));
	space = isl_space_join(space, domain);

	return space;
}

/* Given a map space A -> B, return the map space [A -> B] -> B.
 */
__isl_give isl_space *isl_space_range_map(__isl_take isl_space *space)
{
	isl_space *range;

	range = isl_space_from_range(isl_space_range(isl_space_copy(space)));
	space = isl_space_from_domain(isl_space_wrap(space));
	space = isl_space_join(space, range);

	return space;
}

__isl_give isl_space *isl_space_params(__isl_take isl_space *space)
{
	isl_size n_in, n_out;

	if (isl_space_is_params(space))
		return space;
	n_in = isl_space_dim(space, isl_dim_in);
	n_out = isl_space_dim(space, isl_dim_out);
	if (n_in < 0 || n_out < 0)
		return isl_space_free(space);
	space = isl_space_drop_dims(space, isl_dim_in, 0, n_in);
	space = isl_space_drop_dims(space, isl_dim_out, 0, n_out);
	space = mark_as_params(space);
	return space;
}

__isl_give isl_space *isl_space_set_from_params(__isl_take isl_space *space)
{
	if (!space)
		return NULL;
	if (!isl_space_is_params(space))
		isl_die(isl_space_get_ctx(space), isl_error_invalid,
			"not a parameter space", goto error);
	return isl_space_reset(space, isl_dim_set);
error:
	isl_space_free(space);
	return NULL;
}

/* Add an unnamed tuple of dimension "dim" to "space".
 * This requires "space" to be a parameter or set space.
 *
 * In particular, if "space" is a parameter space, then return
 * a set space with the given dimension.
 * If "space" is a set space, then return a map space
 * with "space" as domain and a range of the given dimension.
 */
__isl_give isl_space *isl_space_add_unnamed_tuple_ui(
	__isl_take isl_space *space, unsigned dim)
{
	isl_bool is_params, is_set;

	is_params = isl_space_is_params(space);
	is_set = isl_space_is_set(space);
	if (is_params < 0 || is_set < 0)
		return isl_space_free(space);
	if (!is_params && !is_set)
		isl_die(isl_space_get_ctx(space), isl_error_invalid,
			"cannot add tuple to map space",
			return isl_space_free(space));
	if (is_params)
		space = isl_space_set_from_params(space);
	else
		space = isl_space_from_domain(space);
	space = isl_space_add_dims(space, isl_dim_out, dim);
	return space;
}

/* Add a tuple of dimension "dim" and with tuple identifier "tuple_id"
 * to "space".
 * This requires "space" to be a parameter or set space.
 */
__isl_give isl_space *isl_space_add_named_tuple_id_ui(
	__isl_take isl_space *space, __isl_take isl_id *tuple_id, unsigned dim)
{
	space = isl_space_add_unnamed_tuple_ui(space, dim);
	space = isl_space_set_tuple_id(space, isl_dim_out, tuple_id);
	return space;
}

/* Check that the identifiers in "tuple" do not appear as parameters
 * in "space".
 */
static isl_stat check_fresh_params(__isl_keep isl_space *space,
	__isl_keep isl_multi_id *tuple)
{
	int i;
	isl_size n;

	n = isl_multi_id_size(tuple);
	if (n < 0)
		return isl_stat_error;
	for (i = 0; i < n; ++i) {
		isl_id *id;
		int pos;

		id = isl_multi_id_get_at(tuple, i);
		if (!id)
			return isl_stat_error;
		pos = isl_space_find_dim_by_id(space, isl_dim_param, id);
		isl_id_free(id);
		if (pos >= 0)
			isl_die(isl_space_get_ctx(space), isl_error_invalid,
				"parameters not unique", return isl_stat_error);
	}

	return isl_stat_ok;
}

/* Add the identifiers in "tuple" as parameters of "space"
 * that are known to be fresh.
 */
static __isl_give isl_space *add_bind_params(__isl_take isl_space *space,
	__isl_keep isl_multi_id *tuple)
{
	int i;
	isl_size first, n;

	first = isl_space_dim(space, isl_dim_param);
	n = isl_multi_id_size(tuple);
	if (first < 0 || n < 0)
		return isl_space_free(space);
	space = isl_space_add_dims(space, isl_dim_param, n);
	for (i = 0; i < n; ++i) {
		isl_id *id;

		id = isl_multi_id_get_at(tuple, i);
		space = isl_space_set_dim_id(space,
						isl_dim_param, first + i, id);
	}

	return space;
}

/* Internal function that removes the set tuple of "space",
 * which is assumed to correspond to the range space of "tuple", and
 * adds the identifiers in "tuple" as fresh parameters.
 * In other words, the set dimensions of "space" are reinterpreted
 * as parameters, but stay in the same global positions.
 */
__isl_give isl_space *isl_space_bind_set(__isl_take isl_space *space,
	__isl_keep isl_multi_id *tuple)
{
	isl_space *tuple_space;

	if (isl_space_check_is_set(space) < 0)
		return isl_space_free(space);
	tuple_space = isl_multi_id_peek_space(tuple);
	if (isl_space_check_equal_tuples(tuple_space, space) < 0)
		return isl_space_free(space);
	if (check_fresh_params(space, tuple) < 0)
		return isl_space_free(space);
	space = isl_space_params(space);
	space = add_bind_params(space, tuple);
	return space;
}

/* Internal function that removes the domain tuple of the map space "space",
 * which is assumed to correspond to the range space of "tuple", and
 * adds the identifiers in "tuple" as fresh parameters.
 * In other words, the domain dimensions of "space" are reinterpreted
 * as parameters, but stay in the same global positions.
 */
__isl_give isl_space *isl_space_bind_map_domain(__isl_take isl_space *space,
	__isl_keep isl_multi_id *tuple)
{
	isl_space *tuple_space;

	if (isl_space_check_is_map(space) < 0)
		return isl_space_free(space);
	tuple_space = isl_multi_id_peek_space(tuple);
	if (isl_space_check_domain_tuples(tuple_space, space) < 0)
		return isl_space_free(space);
	if (check_fresh_params(space, tuple) < 0)
		return isl_space_free(space);
	space = isl_space_range(space);
	space = add_bind_params(space, tuple);
	return space;
}

/* Internal function that, given a space of the form [A -> B] -> C and
 * a tuple of identifiers in A, returns a space B -> C with
 * the identifiers in "tuple" added as fresh parameters.
 * In other words, the domain dimensions of the wrapped relation
 * in the domain of "space" are reinterpreted
 * as parameters, but stay in the same global positions.
 */
__isl_give isl_space *isl_space_bind_domain_wrapped_domain(
	__isl_take isl_space *space, __isl_keep isl_multi_id *tuple)
{
	isl_space *tuple_space;

	if (isl_space_check_is_map(space) < 0)
		return isl_space_free(space);
	tuple_space = isl_multi_id_peek_space(tuple);
	if (isl_space_check_domain_wrapped_domain_tuples(tuple_space,
							space) < 0)
		  return isl_space_free(space);
	if (check_fresh_params(space, tuple) < 0)
		return isl_space_free(space);
	space = isl_space_domain_factor_range(space);
	space = add_bind_params(space, tuple);
	return space;
}

/* Insert a domain tuple in "space" corresponding to the set space "domain".
 * In particular, if "space" is a parameter space, then the result
 * is the set space "domain" combined with the parameters of "space".
 * If "space" is a set space, then the result
 * is a map space with "domain" as domain and the original space as range.
 */
static __isl_give isl_space *isl_space_insert_domain(
	__isl_take isl_space *space, __isl_take isl_space *domain)
{
	isl_bool is_params;

	domain = isl_space_replace_params(domain, space);

	is_params = isl_space_is_params(space);
	if (is_params < 0) {
		isl_space_free(domain);
		space = isl_space_free(space);
	} else if (is_params) {
		isl_space_free(space);
		space = domain;
	} else {
		space = isl_space_map_from_domain_and_range(domain, space);
	}
	return space;
}

/* Internal function that introduces a domain in "space"
 * corresponding to the range space of "tuple".
 * In particular, if "space" is a parameter space, then the result
 * is a set space.  If "space" is a set space, then the result
 * is a map space with the original space as range.
 * Parameters that correspond to the identifiers in "tuple" are removed.
 *
 * The parameters are removed in reverse order (under the assumption
 * that they appear in the same order in "multi") because
 * it is slightly more efficient to remove parameters at the end.
 *
 * For pretty-printing purposes, the identifiers of the set dimensions
 * of the introduced domain are set to the identifiers in "tuple".
 */
__isl_give isl_space *isl_space_unbind_params_insert_domain(
	__isl_take isl_space *space, __isl_keep isl_multi_id *tuple)
{
	int i;
	isl_size n;
	isl_space *tuple_space;

	n = isl_multi_id_size(tuple);
	if (!space || n < 0)
		return isl_space_free(space);
	for (i = n - 1; i >= 0; --i) {
		isl_id *id;
		int pos;

		id = isl_multi_id_get_id(tuple, i);
		if (!id)
			return isl_space_free(space);
		pos = isl_space_find_dim_by_id(space, isl_dim_param, id);
		isl_id_free(id);
		if (pos < 0)
			continue;
		space = isl_space_drop_dims(space, isl_dim_param, pos, 1);
	}
	tuple_space = isl_multi_id_get_space(tuple);
	for (i = 0; i < n; ++i) {
		isl_id *id;

		id = isl_multi_id_get_id(tuple, i);
		tuple_space = isl_space_set_dim_id(tuple_space,
						    isl_dim_set, i, id);
	}
	return isl_space_insert_domain(space, tuple_space);
}

__isl_give isl_space *isl_space_underlying(__isl_take isl_space *space,
	unsigned n_div)
{
	int i;
	isl_bool is_set;

	is_set = isl_space_is_set(space);
	if (is_set < 0)
		return isl_space_free(space);
	if (n_div == 0 && is_set &&
	    space->nparam == 0 && space->n_in == 0 && space->n_id == 0)
		return isl_space_reset(space, isl_dim_out);
	space = isl_space_cow(space);
	if (!space)
		return NULL;
	space->n_out += space->nparam + space->n_in + n_div;
	space->nparam = 0;
	space->n_in = 0;

	for (i = 0; i < space->n_id; ++i)
		isl_id_free(get_id(space, isl_dim_out, i));
	space->n_id = 0;
	space = isl_space_reset(space, isl_dim_in);
	space = isl_space_reset(space, isl_dim_out);
	space = mark_as_set(space);

	return space;
}

/* Are the two spaces the same, including positions and names of parameters?
 */
isl_bool isl_space_is_equal(__isl_keep isl_space *space1,
	__isl_keep isl_space *space2)
{
	isl_bool equal;

	if (!space1 || !space2)
		return isl_bool_error;
	if (space1 == space2)
		return isl_bool_true;
	equal = isl_space_has_equal_params(space1, space2);
	if (equal < 0 || !equal)
		return equal;
	return isl_space_has_equal_tuples(space1, space2);
}

/* Do the tuples of "space1" correspond to those of the domain of "space2"?
 * That is, is "space1" equal to the domain of "space2", ignoring parameters.
 *
 * "space2" is allowed to be a set space, in which case "space1"
 * should be a parameter space.
 */
isl_bool isl_space_has_domain_tuples(__isl_keep isl_space *space1,
	__isl_keep isl_space *space2)
{
	isl_bool is_set;

	is_set = isl_space_is_set(space1);
	if (is_set < 0 || !is_set)
		return is_set;
	return isl_space_tuple_is_equal(space1, isl_dim_set,
					space2, isl_dim_in);
}

/* Do the tuples of "space1" correspond to those of the range of "space2"?
 * That is, is "space1" equal to the range of "space2", ignoring parameters.
 *
 * "space2" is allowed to be the space of a set,
 * in which case it should be equal to "space1", ignoring parameters.
 */
isl_bool isl_space_has_range_tuples(__isl_keep isl_space *space1,
	__isl_keep isl_space *space2)
{
	isl_bool is_set;

	is_set = isl_space_is_set(space1);
	if (is_set < 0 || !is_set)
		return is_set;
	return isl_space_tuple_is_equal(space1, isl_dim_set,
					space2, isl_dim_out);
}

/* Check that the tuples of "space1" correspond to those
 * of the domain of "space2".
 * That is, check that "space1" is equal to the domain of "space2",
 * ignoring parameters.
 */
isl_stat isl_space_check_domain_tuples(__isl_keep isl_space *space1,
	__isl_keep isl_space *space2)
{
	isl_bool is_equal;

	is_equal = isl_space_has_domain_tuples(space1, space2);
	return check_match(space1, is_equal);
}

/* Check that the tuples of "space1" correspond to those
 * of the domain of the wrapped relation in the domain of "space2".
 * That is, check that "space1" is equal to this domain,
 * ignoring parameters.
 */
isl_stat isl_space_check_domain_wrapped_domain_tuples(
	__isl_keep isl_space *space1, __isl_keep isl_space *space2)
{
	isl_space *domain;
	isl_stat r;

	domain = isl_space_unwrap(isl_space_domain(isl_space_copy(space2)));
	r = isl_space_check_domain_tuples(space1, domain);
	isl_space_free(domain);

	return r;
}

/* Is space1 equal to the domain of space2?
 *
 * In the internal version we also allow space2 to be the space of a set,
 * provided space1 is a parameter space.
 */
isl_bool isl_space_is_domain_internal(__isl_keep isl_space *space1,
	__isl_keep isl_space *space2)
{
	isl_bool equal_params;

	if (!space1 || !space2)
		return isl_bool_error;
	equal_params = isl_space_has_equal_params(space1, space2);
	if (equal_params < 0 || !equal_params)
		return equal_params;
	return isl_space_has_domain_tuples(space1, space2);
}

/* Is space1 equal to the domain of space2?
 */
isl_bool isl_space_is_domain(__isl_keep isl_space *space1,
	__isl_keep isl_space *space2)
{
	if (!space2)
		return isl_bool_error;
	if (!isl_space_is_map(space2))
		return isl_bool_false;
	return isl_space_is_domain_internal(space1, space2);
}

/* Is space1 equal to the range of space2?
 *
 * In the internal version, space2 is allowed to be the space of a set,
 * in which case it should be equal to space1.
 */
isl_bool isl_space_is_range_internal(__isl_keep isl_space *space1,
	__isl_keep isl_space *space2)
{
	isl_bool equal_params;

	if (!space1 || !space2)
		return isl_bool_error;
	equal_params = isl_space_has_equal_params(space1, space2);
	if (equal_params < 0 || !equal_params)
		return equal_params;
	return isl_space_has_range_tuples(space1, space2);
}

/* Is space1 equal to the range of space2?
 */
isl_bool isl_space_is_range(__isl_keep isl_space *space1,
	__isl_keep isl_space *space2)
{
	if (!space2)
		return isl_bool_error;
	if (!isl_space_is_map(space2))
		return isl_bool_false;
	return isl_space_is_range_internal(space1, space2);
}

/* Update "hash" by hashing in the parameters of "space".
 */
static uint32_t isl_hash_params(uint32_t hash, __isl_keep isl_space *space)
{
	int i;
	isl_id *id;

	if (!space)
		return hash;

	isl_hash_byte(hash, space->nparam % 256);

	for (i = 0; i < space->nparam; ++i) {
		id = get_id(space, isl_dim_param, i);
		hash = isl_hash_id(hash, id);
	}

	return hash;
}

/* Update "hash" by hashing in the tuples of "space".
 * Changes in this function should be reflected in isl_hash_tuples_domain.
 */
static uint32_t isl_hash_tuples(uint32_t hash, __isl_keep isl_space *space)
{
	isl_id *id;

	if (!space)
		return hash;

	isl_hash_byte(hash, space->n_in % 256);
	isl_hash_byte(hash, space->n_out % 256);

	id = tuple_id(space, isl_dim_in);
	hash = isl_hash_id(hash, id);
	id = tuple_id(space, isl_dim_out);
	hash = isl_hash_id(hash, id);

	hash = isl_hash_tuples(hash, space->nested[0]);
	hash = isl_hash_tuples(hash, space->nested[1]);

	return hash;
}

/* Update "hash" by hashing in the domain tuple of "space".
 * The result of this function is equal to the result of applying
 * isl_hash_tuples to the domain of "space".
 */
static uint32_t isl_hash_tuples_domain(uint32_t hash,
	__isl_keep isl_space *space)
{
	isl_id *id;

	if (!space)
		return hash;

	isl_hash_byte(hash, 0);
	isl_hash_byte(hash, space->n_in % 256);

	hash = isl_hash_id(hash, &isl_id_none);
	id = tuple_id(space, isl_dim_in);
	hash = isl_hash_id(hash, id);

	hash = isl_hash_tuples(hash, space->nested[0]);

	return hash;
}

/* Return a hash value that digests the tuples of "space",
 * i.e., that ignores the parameters.
 * Changes in this function should be reflected
 * in isl_space_get_tuple_domain_hash.
 */
uint32_t isl_space_get_tuple_hash(__isl_keep isl_space *space)
{
	uint32_t hash;

	if (!space)
		return 0;

	hash = isl_hash_init();
	hash = isl_hash_tuples(hash, space);

	return hash;
}

/* Return the hash value of "space".
 */
uint32_t isl_space_get_full_hash(__isl_keep isl_space *space)
{
	uint32_t hash;

	if (!space)
		return 0;

	hash = isl_hash_init();
	hash = isl_hash_params(hash, space);
	hash = isl_hash_tuples(hash, space);

	return hash;
}

/* Return the hash value of the domain tuple of "space".
 * That is, isl_space_get_tuple_domain_hash(space) is equal to
 * isl_space_get_tuple_hash(isl_space_domain(space)).
 */
uint32_t isl_space_get_tuple_domain_hash(__isl_keep isl_space *space)
{
	uint32_t hash;

	if (!space)
		return 0;

	hash = isl_hash_init();
	hash = isl_hash_tuples_domain(hash, space);

	return hash;
}

/* Is "space" the space of a set wrapping a map space?
 */
isl_bool isl_space_is_wrapping(__isl_keep isl_space *space)
{
	if (!space)
		return isl_bool_error;

	if (!isl_space_is_set(space))
		return isl_bool_false;

	return isl_bool_ok(space->nested[1] != NULL);
}

/* Is "space" the space of a map where the domain is a wrapped map space?
 */
isl_bool isl_space_domain_is_wrapping(__isl_keep isl_space *space)
{
	if (!space)
		return isl_bool_error;

	if (isl_space_is_set(space))
		return isl_bool_false;

	return isl_bool_ok(space->nested[0] != NULL);
}

/* Is "space" the space of a map where the range is a wrapped map space?
 */
isl_bool isl_space_range_is_wrapping(__isl_keep isl_space *space)
{
	if (!space)
		return isl_bool_error;

	if (isl_space_is_set(space))
		return isl_bool_false;

	return isl_bool_ok(space->nested[1] != NULL);
}

/* Is "space" a product of two spaces?
 * That is, is it a wrapping set space or a map space
 * with wrapping domain and range?
 */
isl_bool isl_space_is_product(__isl_keep isl_space *space)
{
	isl_bool is_set;
	isl_bool is_product;

	is_set = isl_space_is_set(space);
	if (is_set < 0)
		return isl_bool_error;
	if (is_set)
		return isl_space_is_wrapping(space);
	is_product = isl_space_domain_is_wrapping(space);
	if (is_product < 0 || !is_product)
		return is_product;
	return isl_space_range_is_wrapping(space);
}

__isl_give isl_space *isl_space_wrap(__isl_take isl_space *space)
{
	isl_space *wrap;

	if (!space)
		return NULL;

	wrap = isl_space_set_alloc(space->ctx,
				    space->nparam, space->n_in + space->n_out);

	wrap = copy_ids(wrap, isl_dim_param, 0, space, isl_dim_param);
	wrap = copy_ids(wrap, isl_dim_set, 0, space, isl_dim_in);
	wrap = copy_ids(wrap, isl_dim_set, space->n_in, space, isl_dim_out);

	if (!wrap)
		goto error;

	wrap->nested[1] = space;

	return wrap;
error:
	isl_space_free(space);
	return NULL;
}

__isl_give isl_space *isl_space_unwrap(__isl_take isl_space *space)
{
	isl_space *unwrap;

	if (!space)
		return NULL;

	if (!isl_space_is_wrapping(space))
		isl_die(space->ctx, isl_error_invalid, "not a wrapping space",
			goto error);

	unwrap = isl_space_copy(space->nested[1]);
	isl_space_free(space);

	return unwrap;
error:
	isl_space_free(space);
	return NULL;
}

isl_bool isl_space_is_named_or_nested(__isl_keep isl_space *space,
	enum isl_dim_type type)
{
	if (type != isl_dim_in && type != isl_dim_out)
		return isl_bool_false;
	if (!space)
		return isl_bool_error;
	if (space->tuple_id[type - isl_dim_in])
		return isl_bool_true;
	if (space->nested[type - isl_dim_in])
		return isl_bool_true;
	return isl_bool_false;
}

isl_bool isl_space_may_be_set(__isl_keep isl_space *space)
{
	isl_bool nested;
	isl_size n_in;

	if (!space)
		return isl_bool_error;
	if (isl_space_is_set(space))
		return isl_bool_true;
	n_in = isl_space_dim(space, isl_dim_in);
	if (n_in < 0)
		return isl_bool_error;
	if (n_in != 0)
		return isl_bool_false;
	nested = isl_space_is_named_or_nested(space, isl_dim_in);
	if (nested < 0 || nested)
		return isl_bool_not(nested);
	return isl_bool_true;
}

__isl_give isl_space *isl_space_reset(__isl_take isl_space *space,
	enum isl_dim_type type)
{
	if (!isl_space_is_named_or_nested(space, type))
		return space;

	space = isl_space_cow(space);
	if (!space)
		return NULL;

	isl_id_free(space->tuple_id[type - isl_dim_in]);
	space->tuple_id[type - isl_dim_in] = NULL;
	isl_space_free(space->nested[type - isl_dim_in]);
	space->nested[type - isl_dim_in] = NULL;

	return space;
}

__isl_give isl_space *isl_space_flatten(__isl_take isl_space *space)
{
	if (!space)
		return NULL;
	if (!space->nested[0] && !space->nested[1])
		return space;

	if (space->nested[0])
		space = isl_space_reset(space, isl_dim_in);
	if (space && space->nested[1])
		space = isl_space_reset(space, isl_dim_out);

	return space;
}

__isl_give isl_space *isl_space_flatten_domain(__isl_take isl_space *space)
{
	if (!space)
		return NULL;
	if (!space->nested[0])
		return space;

	return isl_space_reset(space, isl_dim_in);
}

__isl_give isl_space *isl_space_flatten_range(__isl_take isl_space *space)
{
	if (!space)
		return NULL;
	if (!space->nested[1])
		return space;

	return isl_space_reset(space, isl_dim_out);
}

/* Replace the parameters of dst by those of src.
 */
__isl_give isl_space *isl_space_replace_params(__isl_take isl_space *dst,
	__isl_keep isl_space *src)
{
	isl_size dst_dim, src_dim;
	isl_bool equal_params;
	enum isl_dim_type type = isl_dim_param;

	equal_params = isl_space_has_equal_params(dst, src);
	if (equal_params < 0)
		return isl_space_free(dst);
	if (equal_params)
		return dst;

	dst = isl_space_cow(dst);

	dst_dim = isl_space_dim(dst, type);
	src_dim = isl_space_dim(src, type);
	if (dst_dim < 0 || src_dim < 0)
		goto error;

	dst = isl_space_drop_dims(dst, type, 0, dst_dim);
	dst = isl_space_add_dims(dst, type, src_dim);
	dst = copy_ids(dst, type, 0, src, type);

	if (dst) {
		int i;
		for (i = 0; i <= 1; ++i) {
			isl_space *nested;

			if (!dst->nested[i])
				continue;
			nested = isl_space_take_nested(dst, i);
			nested = isl_space_replace_params(nested, src);
			dst = isl_space_restore_nested(dst, i, nested);
			if (!dst)
				return NULL;
		}
	}

	return dst;
error:
	isl_space_free(dst);
	return NULL;
}

/* Given two tuples ("dst_type" in "dst" and "src_type" in "src")
 * of the same size, check if any of the dimensions in the "dst" tuple
 * have no identifier, while the corresponding dimensions in "src"
 * does have an identifier,
 * If so, copy the identifier over to "dst".
 */
__isl_give isl_space *isl_space_copy_ids_if_unset(__isl_take isl_space *dst,
	enum isl_dim_type dst_type, __isl_keep isl_space *src,
	enum isl_dim_type src_type)
{
	int i;
	isl_size n;

	n = isl_space_dim(dst, dst_type);
	if (n < 0)
		return isl_space_free(dst);
	for (i = 0; i < n; ++i) {
		isl_bool set;
		isl_id *id;

		set = isl_space_has_dim_id(dst, dst_type, i);
		if (set < 0)
			return isl_space_free(dst);
		if (set)
			continue;

		set = isl_space_has_dim_id(src, src_type, i);
		if (set < 0)
			return isl_space_free(dst);
		if (!set)
			continue;

		id = isl_space_get_dim_id(src, src_type, i);
		dst = isl_space_set_dim_id(dst, dst_type, i, id);
	}

	return dst;
}

/* Given a space "space" of a set, create a space
 * for the lift of the set.  In particular, the result
 * is of the form lifted[space -> local[..]], with n_local variables in the
 * range of the wrapped map.
 */
__isl_give isl_space *isl_space_lift(__isl_take isl_space *space,
	unsigned n_local)
{
	isl_space *local_space;

	if (!space)
		return NULL;

	local_space = isl_space_dup(space);
	local_space = isl_space_drop_dims(local_space, isl_dim_set, 0,
					space->n_out);
	local_space = isl_space_add_dims(local_space, isl_dim_set, n_local);
	local_space = isl_space_set_tuple_name(local_space,
						isl_dim_set, "local");
	space = isl_space_join(isl_space_from_domain(space),
			    isl_space_from_range(local_space));
	space = isl_space_wrap(space);
	space = isl_space_set_tuple_name(space, isl_dim_set, "lifted");

	return space;
}

isl_bool isl_space_can_zip(__isl_keep isl_space *space)
{
	isl_bool is_set;

	is_set = isl_space_is_set(space);
	if (is_set < 0)
		return isl_bool_error;
	if (is_set)
		return isl_bool_false;
	return isl_space_is_product(space);
}

__isl_give isl_space *isl_space_zip(__isl_take isl_space *space)
{
	isl_space *dom, *ran;
	isl_space *dom_dom, *dom_ran, *ran_dom, *ran_ran;

	if (!isl_space_can_zip(space))
		isl_die(space->ctx, isl_error_invalid, "space cannot be zipped",
			goto error);

	if (!space)
		return NULL;
	dom = isl_space_unwrap(isl_space_domain(isl_space_copy(space)));
	ran = isl_space_unwrap(isl_space_range(space));
	dom_dom = isl_space_domain(isl_space_copy(dom));
	dom_ran = isl_space_range(dom);
	ran_dom = isl_space_domain(isl_space_copy(ran));
	ran_ran = isl_space_range(ran);
	dom = isl_space_join(isl_space_from_domain(dom_dom),
			   isl_space_from_range(ran_dom));
	ran = isl_space_join(isl_space_from_domain(dom_ran),
			   isl_space_from_range(ran_ran));
	return isl_space_join(isl_space_from_domain(isl_space_wrap(dom)),
			    isl_space_from_range(isl_space_wrap(ran)));
error:
	isl_space_free(space);
	return NULL;
}

/* Can we apply isl_space_curry to "space"?
 * That is, does is it have a map space with a nested relation in its domain?
 */
isl_bool isl_space_can_curry(__isl_keep isl_space *space)
{
	return isl_space_domain_is_wrapping(space);
}

/* Given a space (A -> B) -> C, return the corresponding space
 * A -> (B -> C).
 */
__isl_give isl_space *isl_space_curry(__isl_take isl_space *space)
{
	isl_space *dom, *ran;
	isl_space *dom_dom, *dom_ran;

	if (!space)
		return NULL;

	if (!isl_space_can_curry(space))
		isl_die(space->ctx, isl_error_invalid,
			"space cannot be curried", goto error);

	dom = isl_space_unwrap(isl_space_domain(isl_space_copy(space)));
	ran = isl_space_range(space);
	dom_dom = isl_space_domain(isl_space_copy(dom));
	dom_ran = isl_space_range(dom);
	ran = isl_space_join(isl_space_from_domain(dom_ran),
			   isl_space_from_range(ran));
	return isl_space_join(isl_space_from_domain(dom_dom),
			    isl_space_from_range(isl_space_wrap(ran)));
error:
	isl_space_free(space);
	return NULL;
}

/* Can isl_space_range_curry be applied to "space"?
 * That is, does it have a nested relation in its range,
 * the domain of which is itself a nested relation?
 */
isl_bool isl_space_can_range_curry(__isl_keep isl_space *space)
{
	isl_bool can;

	if (!space)
		return isl_bool_error;
	can = isl_space_range_is_wrapping(space);
	if (can < 0 || !can)
		return can;
	return isl_space_can_curry(space->nested[1]);
}

/* Given a space A -> ((B -> C) -> D), return the corresponding space
 * A -> (B -> (C -> D)).
 */
__isl_give isl_space *isl_space_range_curry(__isl_take isl_space *space)
{
	isl_space *nested;

	if (!space)
		return NULL;

	if (!isl_space_can_range_curry(space))
		isl_die(isl_space_get_ctx(space), isl_error_invalid,
			"space range cannot be curried",
			return isl_space_free(space));

	nested = isl_space_take_nested(space, 1);
	nested = isl_space_curry(nested);
	space = isl_space_restore_nested(space, 1, nested);

	return space;
}

/* Can we apply isl_space_uncurry to "space"?
 * That is, does it have a map space with a nested relation in its range?
 */
isl_bool isl_space_can_uncurry(__isl_keep isl_space *space)
{
	return isl_space_range_is_wrapping(space);
}

/* Given a space A -> (B -> C), return the corresponding space
 * (A -> B) -> C.
 */
__isl_give isl_space *isl_space_uncurry(__isl_take isl_space *space)
{
	isl_space *dom, *ran;
	isl_space *ran_dom, *ran_ran;

	if (!space)
		return NULL;

	if (!isl_space_can_uncurry(space))
		isl_die(space->ctx, isl_error_invalid,
			"space cannot be uncurried",
			return isl_space_free(space));

	dom = isl_space_domain(isl_space_copy(space));
	ran = isl_space_unwrap(isl_space_range(space));
	ran_dom = isl_space_domain(isl_space_copy(ran));
	ran_ran = isl_space_range(ran);
	dom = isl_space_join(isl_space_from_domain(dom),
			   isl_space_from_range(ran_dom));
	return isl_space_join(isl_space_from_domain(isl_space_wrap(dom)),
			    isl_space_from_range(ran_ran));
}

isl_bool isl_space_has_named_params(__isl_keep isl_space *space)
{
	int i;
	unsigned off;

	if (!space)
		return isl_bool_error;
	if (space->nparam == 0)
		return isl_bool_true;
	off = isl_space_offset(space, isl_dim_param);
	if (off + space->nparam > space->n_id)
		return isl_bool_false;
	for (i = 0; i < space->nparam; ++i)
		if (!space->ids[off + i])
			return isl_bool_false;
	return isl_bool_true;
}

/* Check that "space" has only named parameters, reporting an error
 * if it does not.
 */
isl_stat isl_space_check_named_params(__isl_keep isl_space *space)
{
	isl_bool named;

	named = isl_space_has_named_params(space);
	if (named < 0)
		return isl_stat_error;
	if (!named)
		isl_die(isl_space_get_ctx(space), isl_error_invalid,
			"unexpected unnamed parameters", return isl_stat_error);

	return isl_stat_ok;
}

/* Align the initial parameters of space1 to match the order in space2.
 */
__isl_give isl_space *isl_space_align_params(__isl_take isl_space *space1,
	__isl_take isl_space *space2)
{
	isl_reordering *exp;

	if (isl_space_check_named_params(space1) < 0 ||
	    isl_space_check_named_params(space2) < 0)
		goto error;

	exp = isl_parameter_alignment_reordering(space1, space2);
	exp = isl_reordering_extend_space(exp, space1);
	isl_space_free(space2);
	space1 = isl_reordering_get_space(exp);
	isl_reordering_free(exp);
	return space1;
error:
	isl_space_free(space1);
	isl_space_free(space2);
	return NULL;
}

/* Given the space of set (domain), construct a space for a map
 * with as domain the given space and as range the range of "model".
 */
__isl_give isl_space *isl_space_extend_domain_with_range(
	__isl_take isl_space *space, __isl_take isl_space *model)
{
	isl_size n_out;

	if (!model)
		goto error;

	space = isl_space_from_domain(space);
	n_out = isl_space_dim(model, isl_dim_out);
	if (n_out < 0)
		goto error;
	space = isl_space_add_dims(space, isl_dim_out, n_out);
	if (isl_space_has_tuple_id(model, isl_dim_out))
		space = isl_space_set_tuple_id(space, isl_dim_out,
				isl_space_get_tuple_id(model, isl_dim_out));
	if (!space)
		goto error;
	if (model->nested[1]) {
		isl_space *nested = isl_space_copy(model->nested[1]);
		isl_size n_nested, n_space;
		nested = isl_space_align_params(nested, isl_space_copy(space));
		n_nested = isl_space_dim(nested, isl_dim_param);
		n_space = isl_space_dim(space, isl_dim_param);
		if (n_nested < 0 || n_space < 0)
			goto error;
		if (n_nested > n_space)
			nested = isl_space_drop_dims(nested, isl_dim_param,
						n_space, n_nested - n_space);
		if (!nested)
			goto error;
		space->nested[1] = nested;
	}
	isl_space_free(model);
	return space;
error:
	isl_space_free(model);
	isl_space_free(space);
	return NULL;
}

/* Compare the "type" dimensions of two isl_spaces.
 *
 * The order is fairly arbitrary.
 */
static int isl_space_cmp_type(__isl_keep isl_space *space1,
	__isl_keep isl_space *space2, enum isl_dim_type type)
{
	int cmp;
	isl_size dim1, dim2;
	isl_space *nested1, *nested2;

	dim1 = isl_space_dim(space1, type);
	dim2 = isl_space_dim(space2, type);
	if (dim1 < 0 || dim2 < 0)
		return 0;
	if (dim1 != dim2)
		return dim1 - dim2;

	cmp = isl_id_cmp(tuple_id(space1, type), tuple_id(space2, type));
	if (cmp != 0)
		return cmp;

	nested1 = nested(space1, type);
	nested2 = nested(space2, type);
	if (!nested1 != !nested2)
		return !nested1 - !nested2;

	if (nested1)
		return isl_space_cmp(nested1, nested2);

	return 0;
}

/* Compare two isl_spaces.
 *
 * The order is fairly arbitrary.
 */
int isl_space_cmp(__isl_keep isl_space *space1, __isl_keep isl_space *space2)
{
	int i;
	int cmp;

	if (space1 == space2)
		return 0;
	if (!space1)
		return -1;
	if (!space2)
		return 1;

	cmp = isl_space_cmp_type(space1, space2, isl_dim_param);
	if (cmp != 0)
		return cmp;
	cmp = isl_space_cmp_type(space1, space2, isl_dim_in);
	if (cmp != 0)
		return cmp;
	cmp = isl_space_cmp_type(space1, space2, isl_dim_out);
	if (cmp != 0)
		return cmp;

	if (!space1->ids && !space2->ids)
		return 0;

	for (i = 0; i < n(space1, isl_dim_param); ++i) {
		cmp = isl_id_cmp(get_id(space1, isl_dim_param, i),
				 get_id(space2, isl_dim_param, i));
		if (cmp != 0)
			return cmp;
	}

	return 0;
}
