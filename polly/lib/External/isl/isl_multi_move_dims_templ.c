/*
 * Copyright 2013      Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 */

#include <isl/space.h>

#include <isl_multi_macro.h>

/* Move the "n" dimensions of "src_type" starting at "src_pos" of "multi"
 * to dimensions of "dst_type" at "dst_pos".
 *
 * We only support moving input dimensions to parameters and vice versa.
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),move_dims)(__isl_take MULTI(BASE) *multi,
	enum isl_dim_type dst_type, unsigned dst_pos,
	enum isl_dim_type src_type, unsigned src_pos, unsigned n)
{
	int i;

	if (!multi)
		return NULL;

	if (n == 0 &&
	    !isl_space_is_named_or_nested(multi->space, src_type) &&
	    !isl_space_is_named_or_nested(multi->space, dst_type))
		return multi;

	if (dst_type == isl_dim_out || src_type == isl_dim_out)
		isl_die(FN(MULTI(BASE),get_ctx)(multi), isl_error_invalid,
			"cannot move output/set dimension",
			return FN(MULTI(BASE),free)(multi));
	if (dst_type == isl_dim_div || src_type == isl_dim_div)
		isl_die(FN(MULTI(BASE),get_ctx)(multi), isl_error_invalid,
			"cannot move divs",
			return FN(MULTI(BASE),free)(multi));
	if (FN(MULTI(BASE),check_range)(multi, src_type, src_pos, n) < 0)
		return FN(MULTI(BASE),free)(multi);
	if (dst_type == src_type)
		isl_die(FN(MULTI(BASE),get_ctx)(multi), isl_error_unsupported,
			"moving dims within the same type not supported",
			return FN(MULTI(BASE),free)(multi));

	multi = FN(MULTI(BASE),cow)(multi);
	if (!multi)
		return NULL;

	multi->space = isl_space_move_dims(multi->space, dst_type, dst_pos,
						src_type, src_pos, n);
	if (!multi->space)
		return FN(MULTI(BASE),free)(multi);
	if (FN(MULTI(BASE),has_explicit_domain)(multi))
		multi = FN(MULTI(BASE),move_explicit_domain_dims)(multi,
				dst_type, dst_pos, src_type, src_pos, n);
	if (!multi)
		return NULL;

	for (i = 0; i < multi->n; ++i) {
		multi->u.p[i] = FN(EL,move_dims)(multi->u.p[i],
						dst_type, dst_pos,
						src_type, src_pos, n);
		if (!multi->u.p[i])
			return FN(MULTI(BASE),free)(multi);
	}

	return multi;
}
