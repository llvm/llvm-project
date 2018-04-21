/*
 * Copyright 2017      Sven Verdoolaege
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 */

/* Align the parameters of "multi" and "domain" (if needed) and
 * call "fn".
 */
static __isl_give MULTI(BASE) *FN(FN(MULTI(BASE),align_params),ALIGN_DOMBASE)(
	__isl_take MULTI(BASE) *multi, __isl_take ALIGN_DOM *domain,
	__isl_give MULTI(BASE) *fn(__isl_take MULTI(BASE) *multi,
		__isl_take ALIGN_DOM *domain))
{
	isl_bool aligned;
	isl_bool named;
	isl_space *dom_space;

	aligned = FN(ALIGN_DOM,space_has_equal_params)(domain, multi->space);
	if (aligned < 0)
		goto error;
	if (aligned)
		return fn(multi, domain);

	dom_space = FN(ALIGN_DOM,peek_space)(domain);
	named = isl_space_has_named_params(multi->space);
	if (named >= 0 && named)
		named = isl_space_has_named_params(dom_space);
	if (named < 0)
		goto error;
	if (!named)
		isl_die(FN(MULTI(BASE),get_ctx)(multi), isl_error_invalid,
			"unaligned unnamed parameters", goto error);
	multi = FN(MULTI(BASE),align_params)(multi,
					    FN(ALIGN_DOM,get_space)(domain));
	domain = FN(ALIGN_DOM,align_params)(domain,
					    FN(MULTI(BASE),get_space)(multi));
	return fn(multi, domain);
error:
	FN(MULTI(BASE),free)(multi);
	FN(ALIGN_DOM,free)(domain);
	return NULL;
}
