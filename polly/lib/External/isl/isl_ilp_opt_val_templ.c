/*
 * Copyright 2017      Sven Verdoolaege
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege
 */

#define xCAT(A,B) A ## B
#define CAT(A,B) xCAT(A,B)
#undef TYPE
#define TYPE CAT(isl_,BASE)
#define xBFN(BASE,NAME) isl_ ## BASE ## _ ## NAME
#define BFN(BASE,NAME) xBFN(BASE,NAME)

/* Return the minimal (maximal if "max" is set) value attained
 * by the given set dimension,
 * independently of the parameter values and of any other dimensions.
 *
 * Return infinity or negative infinity if the optimal value is unbounded and
 * NaN if "set" is empty.
 */
static __isl_give isl_val *BFN(BASE,dim_opt_val)(__isl_take TYPE *set, int max,
	int pos)
{
	isl_local_space *ls;
	isl_aff *obj;
	isl_val *v;

	if (BFN(BASE,check_range)(set, isl_dim_set, pos, 1) < 0)
		goto error;
	ls = isl_local_space_from_space(BFN(BASE,get_space)(set));
	obj = isl_aff_var_on_domain(ls, isl_dim_set, pos);
	v = BFN(BASE,opt_val)(set, max, obj);
	isl_aff_free(obj);
	BFN(BASE,free)(set);

	return v;
error:
	BFN(BASE,free)(set);
	return NULL;
}
