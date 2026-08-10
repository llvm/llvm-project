/*
 * Copyright 2012      Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 */

#include <isl/space.h>

#include <isl_multi_macro.h>

/* Given two MULTI(BASE)s A -> B and C -> D,
 * construct a MULTI(BASE) [A -> C] -> [B -> D].
 *
 * If "multi1" and/or "multi2" has an explicit domain, then
 * intersect the domain of the result with these explicit domains.
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),product)(
	__isl_take MULTI(BASE) *multi1, __isl_take MULTI(BASE) *multi2)
{
	int i;
	EL *el;
	isl_space *space;
	MULTI(BASE) *res;
	isl_size in1, in2, out1, out2;

	FN(MULTI(BASE),align_params_bin)(&multi1, &multi2);
	in1 = FN(MULTI(BASE),dim)(multi1, isl_dim_in);
	in2 = FN(MULTI(BASE),dim)(multi2, isl_dim_in);
	out1 = FN(MULTI(BASE),dim)(multi1, isl_dim_out);
	out2 = FN(MULTI(BASE),dim)(multi2, isl_dim_out);
	if (in1 < 0 || in2 < 0 || out1 < 0 || out2 < 0)
		goto error;
	space = isl_space_product(FN(MULTI(BASE),get_space)(multi1),
				  FN(MULTI(BASE),get_space)(multi2));
	res = FN(MULTI(BASE),alloc)(isl_space_copy(space));
	space = isl_space_domain(space);

	for (i = 0; i < out1; ++i) {
		el = FN(FN(MULTI(BASE),get),BASE)(multi1, i);
		el = FN(EL,insert_dims)(el, isl_dim_in, in1, in2);
		el = FN(EL,reset_domain_space)(el, isl_space_copy(space));
		res = FN(FN(MULTI(BASE),set),BASE)(res, i, el);
	}

	for (i = 0; i < out2; ++i) {
		el = FN(FN(MULTI(BASE),get),BASE)(multi2, i);
		el = FN(EL,insert_dims)(el, isl_dim_in, 0, in1);
		el = FN(EL,reset_domain_space)(el, isl_space_copy(space));
		res = FN(FN(MULTI(BASE),set),BASE)(res, out1 + i, el);
	}

	if (FN(MULTI(BASE),has_explicit_domain)(multi1) ||
	    FN(MULTI(BASE),has_explicit_domain)(multi2))
		res = FN(MULTI(BASE),intersect_explicit_domain_product)(res,
								multi1, multi2);

	isl_space_free(space);
	FN(MULTI(BASE),free)(multi1);
	FN(MULTI(BASE),free)(multi2);
	return res;
error:
	FN(MULTI(BASE),free)(multi1);
	FN(MULTI(BASE),free)(multi2);
	return NULL;
}
