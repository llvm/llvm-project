/*
 * Copyright 2011      Sven Verdoolaege
 * Copyright 2012-2013 Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 */

#include <isl_space_private.h>

#include <isl_multi_macro.h>

/* Check whether "multi" has non-zero coefficients for any dimension
 * in the given range or if any of these dimensions appear
 * with non-zero coefficients in any of the integer divisions involved.
 */
isl_bool FN(MULTI(BASE),involves_dims)(__isl_keep MULTI(BASE) *multi,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	int i;

	if (!multi)
		return isl_bool_error;
	if (n == 0)
		return isl_bool_false;

	for (i = 0; i < multi->n; ++i) {
		isl_bool involves;

		involves = FN(EL,involves_dims)(multi->u.p[i], type, first, n);
		if (involves < 0 || involves)
			return involves;
	}

	if (FN(MULTI(BASE),has_explicit_domain)(multi))
		return FN(MULTI(BASE),involves_explicit_domain_dims)(multi,
								type, first, n);

	return isl_bool_false;
}

__isl_give MULTI(BASE) *FN(MULTI(BASE),insert_dims)(
	__isl_take MULTI(BASE) *multi,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	int i;

	if (!multi)
		return NULL;
	if (type == isl_dim_out)
		isl_die(FN(MULTI(BASE),get_ctx)(multi), isl_error_invalid,
			"cannot insert output/set dimensions",
			return FN(MULTI(BASE),free)(multi));
	if (n == 0 && !isl_space_is_named_or_nested(multi->space, type))
		return multi;

	multi = FN(MULTI(BASE),cow)(multi);
	if (!multi)
		return NULL;

	multi->space = isl_space_insert_dims(multi->space, type, first, n);
	if (!multi->space)
		return FN(MULTI(BASE),free)(multi);
	if (FN(MULTI(BASE),has_explicit_domain)(multi))
		multi = FN(MULTI(BASE),insert_explicit_domain_dims)(multi,
								type, first, n);
	if (!multi)
		return NULL;

	for (i = 0; i < multi->n; ++i) {
		multi->u.p[i] = FN(EL,insert_dims)(multi->u.p[i],
							type, first, n);
		if (!multi->u.p[i])
			return FN(MULTI(BASE),free)(multi);
	}

	return multi;
}

__isl_give MULTI(BASE) *FN(MULTI(BASE),add_dims)(__isl_take MULTI(BASE) *multi,
	enum isl_dim_type type, unsigned n)
{
	isl_size pos;

	pos = FN(MULTI(BASE),dim)(multi, type);
	if (pos < 0)
		return FN(MULTI(BASE),free)(multi);

	return FN(MULTI(BASE),insert_dims)(multi, type, pos, n);
}

/* Project the domain of "multi" onto its parameter space.
 * "multi" may not involve any of the domain dimensions.
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),project_domain_on_params)(
	__isl_take MULTI(BASE) *multi)
{
	isl_size n;
	isl_bool involves;
	isl_space *space;

	n = FN(MULTI(BASE),dim)(multi, isl_dim_in);
	if (n < 0)
		return FN(MULTI(BASE),free)(multi);
	involves = FN(MULTI(BASE),involves_dims)(multi, isl_dim_in, 0, n);
	if (involves < 0)
		return FN(MULTI(BASE),free)(multi);
	if (involves)
		isl_die(FN(MULTI(BASE),get_ctx)(multi), isl_error_invalid,
		    "expression involves some of the domain dimensions",
		    return FN(MULTI(BASE),free)(multi));
	multi = FN(MULTI(BASE),drop_dims)(multi, isl_dim_in, 0, n);
	space = FN(MULTI(BASE),get_domain_space)(multi);
	space = isl_space_params(space);
	multi = FN(MULTI(BASE),reset_domain_space)(multi, space);
	return multi;
}
