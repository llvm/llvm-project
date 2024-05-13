/*
 * Copyright 2011      Sven Verdoolaege
 * Copyright 2013      Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 */

#include <isl/space.h>

#include <isl_multi_macro.h>

/* Return the position of the dimension of the given type and name
 * in "multi".
 * Return -1 if no such dimension can be found.
 */
int FN(MULTI(BASE),find_dim_by_name)(__isl_keep MULTI(BASE) *multi,
	enum isl_dim_type type, const char *name)
{
	if (!multi)
		return -1;
	return isl_space_find_dim_by_name(multi->space, type, name);
}

/* Return the position of the first dimension of "type" with id "id".
 * Return -1 if there is no such dimension.
 */
int FN(MULTI(BASE),find_dim_by_id)(__isl_keep MULTI(BASE) *multi,
	enum isl_dim_type type, __isl_keep isl_id *id)
{
	if (!multi)
		return -1;
	return isl_space_find_dim_by_id(multi->space, type, id);
}

/* Return the id of the given dimension.
 */
__isl_give isl_id *FN(MULTI(BASE),get_dim_id)(__isl_keep MULTI(BASE) *multi,
	enum isl_dim_type type, unsigned pos)
{
	return multi ? isl_space_get_dim_id(multi->space, type, pos) : NULL;
}

__isl_give MULTI(BASE) *FN(MULTI(BASE),set_dim_name)(
	__isl_take MULTI(BASE) *multi,
	enum isl_dim_type type, unsigned pos, const char *s)
{
	isl_space *space;

	space = FN(MULTI(BASE),get_space)(multi);
	space = isl_space_set_dim_name(space, type, pos, s);

	return FN(MULTI(BASE),reset_space)(multi, space);
}

/* Set the id of the given dimension of "multi" to "id".
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),set_dim_id)(
	__isl_take MULTI(BASE) *multi,
	enum isl_dim_type type, unsigned pos, __isl_take isl_id *id)
{
	isl_space *space;

	space = FN(MULTI(BASE),get_space)(multi);
	space = isl_space_set_dim_id(space, type, pos, id);

	return FN(MULTI(BASE),reset_space)(multi, space);
}
