/*
 * Copyright 2017      Sven Verdoolaege
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege.
 */

#include <isl_multi_macro.h>

/* Does the space of "domain" correspond to that of the domain of "multi"?
 * The parameters do not need to be aligned.
 */
static isl_bool FN(MULTI(BASE),compatible_domain)(
	__isl_keep MULTI(BASE) *multi, __isl_keep DOM *domain)
{
	isl_bool ok;
	isl_space *space, *domain_space;

	domain_space = FN(DOM,get_space)(domain);
	space = FN(MULTI(BASE),get_space)(multi);
	ok = isl_space_has_domain_tuples(domain_space, space);
	isl_space_free(space);
	isl_space_free(domain_space);

	return ok;
}

/* Check that the space of "domain" corresponds to
 * that of the domain of "multi", ignoring parameters.
 */
static isl_stat FN(MULTI(BASE),check_compatible_domain)(
	__isl_keep MULTI(BASE) *multi, __isl_keep DOM *domain)
{
	isl_bool ok;

	ok = FN(MULTI(BASE),compatible_domain)(multi, domain);
	if (ok < 0)
		return isl_stat_error;
	if (!ok)
		isl_die(FN(DOM,get_ctx)(domain), isl_error_invalid,
			"incompatible spaces", return isl_stat_error);

	return isl_stat_ok;
}
