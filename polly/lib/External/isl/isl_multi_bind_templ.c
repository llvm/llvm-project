/*
 * Copyright 2018      Cerebras Systems
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Cerebras Systems, 175 S San Antonio Rd, Los Altos, CA, USA
 */

/* Bind the expressions of "multi" to parameters with identifiers
 * specified by "tuple", living in the same space as
 * (the target space of) "multi",
 * returning the elements in the domain where the expressions
 * are equal to the parameters.
 */
__isl_give DOM *FN(MULTI(BASE),bind)(__isl_take MULTI(BASE) *multi,
	__isl_take isl_multi_id *tuple)
{
	int i;
	isl_id *id;
	isl_stat r;
	isl_size n;
	isl_space *multi_space, *tuple_space;
	EL *el;
	DOM *bnd;

	multi_space = isl_space_range(FN(MULTI(BASE),get_space)(multi));
	tuple_space = isl_multi_id_peek_space(tuple);
	r = isl_space_check_equal_tuples(multi_space, tuple_space);
	isl_space_free(multi_space);
	if (r < 0)
		goto error;
	n = FN(MULTI(BASE),dim)(multi, isl_dim_set);
	if (n < 0)
		goto error;

	if (n == 0) {
		isl_multi_id_free(tuple);
		return FN(MULTI(BASE),domain)(multi);
	}

	el = FN(MULTI(BASE),get_at)(multi, 0);
	id = isl_multi_id_get_at(tuple, 0);
	bnd = FN(EL,bind_id)(el, id);

	for (i = 1; i < n; ++i) {
		DOM *bnd_i;

		el = FN(MULTI(BASE),get_at)(multi, i);
		id = isl_multi_id_get_at(tuple, i);
		bnd_i = FN(EL,bind_id)(el, id);

		bnd_i = FN(DOM,align_params)(bnd_i, FN(DOM,get_space)(bnd));
		bnd = FN(DOM,align_params)(bnd, FN(DOM,get_space)(bnd_i));
		bnd = FN(DOM,intersect)(bnd, bnd_i);
	}

	FN(MULTI(BASE),free)(multi);
	isl_multi_id_free(tuple);
	return bnd;
error:
	FN(MULTI(BASE),free)(multi);
	isl_multi_id_free(tuple);
	return NULL;
}
