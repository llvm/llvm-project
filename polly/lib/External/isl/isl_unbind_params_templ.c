/*
 * Copyright 2018      Sven Verdoolaege
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege.
 */

/* Given a function "obj" defined over a parameter domain,
 * convert it to a function defined over a domain corresponding
 * to "domain".
 * Any parameters with identifiers in "domain" are reinterpreted
 * as the corresponding domain dimensions.
 */
__isl_give TYPE *FN(TYPE,unbind_params_insert_domain)(
	__isl_take TYPE *obj, __isl_take isl_multi_id *domain)
{
	isl_bool is_params;
	isl_space *space;
	isl_reordering *r;

	space = FN(TYPE,get_domain_space)(obj);
	is_params = isl_space_is_params(space);
	if (is_params < 0)
		domain = isl_multi_id_free(domain);
	else if (!is_params)
		isl_die(FN(TYPE,get_ctx)(obj), isl_error_invalid,
			"expecting function with parameter domain",
			domain = isl_multi_id_free(domain));
	r = isl_reordering_unbind_params_insert_domain(space, domain);
	isl_space_free(space);
	isl_multi_id_free(domain);

	return FN(TYPE,realign_domain)(obj, r);
}
