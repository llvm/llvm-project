/*
 * Copyright 2011      Sven Verdoolaege
 * Copyright 2012-2013 Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d’Ulm, 75230 Paris, France
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

/* Intersect the explicit domain of "multi" with "domain".
 *
 * The parameters of "multi" and "domain" are assumed to have been aligned.
 *
 * In the case of an isl_multi_union_pw_aff object, the explicit domain
 * is allowed to have only constraints on the parameters, while
 * "domain" contains actual domain elements.  In this case,
 * "domain" is intersected with those parameter constraints and
 * then used as the explicit domain of "multi".
 */
static __isl_give MULTI(BASE) *FN(MULTI(BASE),domain_intersect_aligned)(
	__isl_take MULTI(BASE) *multi, __isl_take DOM *domain)
{
	isl_bool is_params;
	DOM *multi_dom;

	if (FN(MULTI(BASE),check_compatible_domain)(multi, domain) < 0)
		goto error;
	if (FN(MULTI(BASE),check_has_explicit_domain)(multi) < 0)
		goto error;
	is_params = FN(DOM,is_params)(multi->u.dom);
	if (is_params < 0)
		goto error;
	multi_dom = FN(MULTI(BASE),get_explicit_domain)(multi);
	if (!is_params) {
		domain = FN(DOM,intersect)(multi_dom, domain);
	} else {
		isl_set *params;

		params = FN(DOM,params)(multi_dom);
		domain = FN(DOM,intersect_params)(domain, params);
	}
	multi = FN(MULTI(BASE),set_explicit_domain)(multi, domain);
	return multi;
error:
	FN(MULTI(BASE),free)(multi);
	FN(DOM,free)(domain);
	return NULL;
}

/* Intersect the explicit domain of "multi" with "domain".
 * First align the parameters, if needed.
 */
static __isl_give MULTI(BASE) *FN(MULTI(BASE),domain_intersect)(
	__isl_take MULTI(BASE) *multi, __isl_take DOM *domain)
{
	return FN(FN(MULTI(BASE),align_params),DOMBASE)(multi, domain,
				    FN(MULTI(BASE),domain_intersect_aligned));
}

/* Intersect the domain of "multi" with "domain".
 *
 * If "multi" has an explicit domain, then only this domain
 * needs to be intersected.
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),intersect_domain)(
	__isl_take MULTI(BASE) *multi, __isl_take DOM *domain)
{
	if (FN(MULTI(BASE),has_explicit_domain)(multi))
		return FN(MULTI(BASE),domain_intersect)(multi, domain);
	return FN(FN(MULTI(BASE),apply),DOMBASE)(multi, domain,
					&FN(EL,intersect_domain));
}

/* Intersect the parameter domain of the explicit domain of "multi"
 * with "domain".
 */
static __isl_give MULTI(BASE) *FN(MULTI(BASE),domain_intersect_params_aligned)(
	__isl_take MULTI(BASE) *multi, __isl_take isl_set *domain)
{
	DOM *multi_dom;

	multi_dom = FN(MULTI(BASE),get_explicit_domain)(multi);
	multi_dom = FN(DOM,intersect_params)(multi_dom, domain);
	multi = FN(MULTI(BASE),set_explicit_domain)(multi, multi_dom);

	return multi;
}

/* Intersect the parameter domain of the explicit domain of "multi"
 * with "domain".
 * First align the parameters, if needed.
 */
static __isl_give MULTI(BASE) *FN(MULTI(BASE),domain_intersect_params)(
	__isl_take MULTI(BASE) *multi, __isl_take isl_set *domain)
{
	return FN(FN(MULTI(BASE),align_params),set)(multi, domain,
			    FN(MULTI(BASE),domain_intersect_params_aligned));
}

/* Intersect the parameter domain of "multi" with "domain".
 *
 * If "multi" has an explicit domain, then only this domain
 * needs to be intersected.
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),intersect_params)(
	__isl_take MULTI(BASE) *multi, __isl_take isl_set *domain)
{
	if (FN(MULTI(BASE),has_explicit_domain)(multi))
		return FN(MULTI(BASE),domain_intersect_params)(multi, domain);
	return FN(MULTI(BASE),apply_set)(multi, domain,
					&FN(EL,intersect_params));
}
