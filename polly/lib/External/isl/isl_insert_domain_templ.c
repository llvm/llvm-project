/*
 * Copyright 2019      Cerebras Systems
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Cerebras Systems, 175 S San Antonio Rd, Los Altos, CA, USA
 */

/* Given a function defined over a parameter domain,
 * convert it to a function defined over a domain corresponding
 * to "domain".
 */
__isl_give TYPE *FN(TYPE,insert_domain)(__isl_take TYPE *obj,
	__isl_take isl_space *domain)
{
	isl_size dim;
	isl_space *obj_space;

	obj_space = FN(TYPE,peek_space)(obj);
	if (isl_space_check_is_set(domain) < 0 ||
	    isl_space_check_is_set(obj_space) < 0)
		goto error;
	dim = isl_space_dim(domain, isl_dim_set);
	if (dim < 0)
		goto error;

	domain = isl_space_replace_params(domain, obj_space);

	obj = FN(TYPE,from_range)(obj);
	obj = FN(TYPE,add_dims)(obj, isl_dim_in, dim);
	obj = FN(TYPE,reset_domain_space)(obj, domain);

	return obj;
error:
	isl_space_free(domain);
	FN(TYPE,free)(obj);
	return NULL;
}
