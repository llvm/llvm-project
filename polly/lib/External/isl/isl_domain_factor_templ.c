/*
 * Copyright 2012      Ecole Normale Superieure
 * Copyright 2017      Sven Verdoolaege
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 */

#define xFN(TYPE,NAME) TYPE ## _ ## NAME
#define FN(TYPE,NAME) xFN(TYPE,NAME)

/* Drop the "n" domain dimensions starting at "first" from "obj",
 * after checking that they do not appear in the affine expression.
 */
static __isl_give TYPE *FN(TYPE,drop_domain)(__isl_take TYPE *obj,
	unsigned first, unsigned n)
{
	isl_bool involves;

	involves = FN(TYPE,involves_dims)(obj, isl_dim_in, first, n);
	if (involves < 0)
		return FN(TYPE,free)(obj);
	if (involves)
		isl_die(FN(TYPE,get_ctx)(obj), isl_error_invalid,
		    "affine expression involves some of the domain dimensions",
		    return FN(TYPE,free)(obj));
	return FN(TYPE,drop_dims)(obj, isl_dim_in, first, n);
}

/* Check that the domain of "obj" is a product.
 */
static isl_stat FN(TYPE,check_domain_product)(__isl_keep TYPE *obj)
{
	isl_bool is_product;

	is_product = FN(TYPE,domain_is_product)(obj);
	if (is_product < 0)
		return isl_stat_error;
	if (!is_product)
		isl_die(FN(TYPE,get_ctx)(obj), isl_error_invalid,
			"domain is not a product", return isl_stat_error);
	return isl_stat_ok;
}

/* Given an affine function with a domain of the form [A -> B] that
 * does not depend on B, return the same function on domain A.
 */
__isl_give TYPE *FN(TYPE,domain_factor_domain)(__isl_take TYPE *obj)
{
	isl_space *space;
	isl_size n, n_in;

	if (FN(TYPE,check_domain_product)(obj) < 0)
		return FN(TYPE,free)(obj);
	space = FN(TYPE,get_domain_space)(obj);
	n = isl_space_dim(space, isl_dim_set);
	space = isl_space_factor_domain(space);
	n_in = isl_space_dim(space, isl_dim_set);
	if (n < 0 || n_in < 0)
		obj = FN(TYPE,free)(obj);
	else
		obj = FN(TYPE,drop_domain)(obj, n_in, n - n_in);
	obj = FN(TYPE,reset_domain_space)(obj, space);
	return obj;
}
