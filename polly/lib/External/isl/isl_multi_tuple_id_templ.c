/*
 * Copyright 2011      Sven Verdoolaege
 * Copyright 2012-2013 Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 */

#include <isl/space.h>

#include <isl_multi_macro.h>

const char *FN(MULTI(BASE),get_tuple_name)(__isl_keep MULTI(BASE) *multi,
	enum isl_dim_type type)
{
	return multi ? isl_space_get_tuple_name(multi->space, type) : NULL;
}

/* Does the specified tuple have an id?
 */
isl_bool FN(MULTI(BASE),has_tuple_id)(__isl_keep MULTI(BASE) *multi,
	enum isl_dim_type type)
{
	if (!multi)
		return isl_bool_error;
	return isl_space_has_tuple_id(multi->space, type);
}

/* Does the (range) tuple of "multi" have an identifier?
 *
 * Technically, the implementation should use isl_dim_set if "multi"
 * lives in a set space and isl_dim_out if it lives in a map space.
 * Internally, however, it can be assumed that isl_dim_set is equal
 * to isl_dim_out.
 */
isl_bool FN(MULTI(BASE),has_range_tuple_id)(__isl_keep MULTI(BASE) *multi)
{
	return FN(MULTI(BASE),has_tuple_id)(multi, isl_dim_out);
}

/* Return the id of the specified tuple.
 */
__isl_give isl_id *FN(MULTI(BASE),get_tuple_id)(__isl_keep MULTI(BASE) *multi,
	enum isl_dim_type type)
{
	return multi ? isl_space_get_tuple_id(multi->space, type) : NULL;
}

/* Return the identifier of the (range) tuple of "multi", assuming it has one.
 *
 * Technically, the implementation should use isl_dim_set if "multi"
 * lives in a set space and isl_dim_out if it lives in a map space.
 * Internally, however, it can be assumed that isl_dim_set is equal
 * to isl_dim_out.
 */
__isl_give isl_id *FN(MULTI(BASE),get_range_tuple_id)(
	__isl_keep MULTI(BASE) *multi)
{
	return FN(MULTI(BASE),get_tuple_id)(multi, isl_dim_out);
}

__isl_give MULTI(BASE) *FN(MULTI(BASE),set_tuple_name)(
	__isl_keep MULTI(BASE) *multi, enum isl_dim_type type,
	const char *s)
{
	isl_space *space;

	multi = FN(MULTI(BASE),cow)(multi);
	if (!multi)
		return NULL;

	space = FN(MULTI(BASE),get_space)(multi);
	space = isl_space_set_tuple_name(space, type, s);

	return FN(MULTI(BASE),reset_space)(multi, space);
}

__isl_give MULTI(BASE) *FN(MULTI(BASE),set_tuple_id)(
	__isl_take MULTI(BASE) *multi, enum isl_dim_type type,
	__isl_take isl_id *id)
{
	isl_space *space;

	multi = FN(MULTI(BASE),cow)(multi);
	if (!multi)
		goto error;

	space = FN(MULTI(BASE),get_space)(multi);
	space = isl_space_set_tuple_id(space, type, id);

	return FN(MULTI(BASE),reset_space)(multi, space);
error:
	isl_id_free(id);
	return NULL;
}

/* Replace the identifier of the (range) tuple of "multi" by "id".
 *
 * Technically, the implementation should use isl_dim_set if "multi"
 * lives in a set space and isl_dim_out if it lives in a map space.
 * Internally, however, it can be assumed that isl_dim_set is equal
 * to isl_dim_out.
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),set_range_tuple_id)(
	__isl_take MULTI(BASE) *multi, __isl_take isl_id *id)
{
	return FN(MULTI(BASE),set_tuple_id)(multi, isl_dim_out, id);
}

/* Drop the id on the specified tuple.
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),reset_tuple_id)(
	__isl_take MULTI(BASE) *multi, enum isl_dim_type type)
{
	isl_space *space;

	if (!multi)
		return NULL;
	if (!FN(MULTI(BASE),has_tuple_id)(multi, type))
		return multi;

	multi = FN(MULTI(BASE),cow)(multi);
	if (!multi)
		return NULL;

	space = FN(MULTI(BASE),get_space)(multi);
	space = isl_space_reset_tuple_id(space, type);

	return FN(MULTI(BASE),reset_space)(multi, space);
}

/* Drop the identifier of the (range) tuple of "multi".
 *
 * Technically, the implementation should use isl_dim_set if "multi"
 * lives in a set space and isl_dim_out if it lives in a map space.
 * Internally, however, it can be assumed that isl_dim_set is equal
 * to isl_dim_out.
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),reset_range_tuple_id)(
	__isl_take MULTI(BASE) *multi)
{
	return FN(MULTI(BASE),reset_tuple_id)(multi, isl_dim_out);
}
