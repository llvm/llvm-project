/*
 * Copyright 2017      Sven Verdoolaege
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege.
 */

/* These versions of the explicit domain functions are used
 * when the multi expression may have an explicit domain.
 */

#include <isl_multi_macro.h>

__isl_give MULTI(BASE) *FN(MULTI(BASE),cow)(__isl_take MULTI(BASE) *multi);

/* Does "multi" have an explicit domain?
 *
 * An explicit domain is only available if "multi" is zero-dimensional.
 */
static int FN(MULTI(BASE),has_explicit_domain)(__isl_keep MULTI(BASE) *multi)
{
	return multi && multi->n == 0;
}

/* Check that "multi" has an explicit domain.
 */
static isl_stat FN(MULTI(BASE),check_has_explicit_domain)(
	__isl_keep MULTI(BASE) *multi)
{
	if (!multi)
		return isl_stat_error;
	if (!FN(MULTI(BASE),has_explicit_domain)(multi))
		isl_die(FN(MULTI(BASE),get_ctx)(multi), isl_error_internal,
			"expression does not have an explicit domain",
			return isl_stat_error);
	return isl_stat_ok;
}

/* Return the explicit domain of "multi", assuming it has one.
 */
static __isl_give DOM *FN(MULTI(BASE),get_explicit_domain)(
	__isl_keep MULTI(BASE) *multi)
{
	if (FN(MULTI(BASE),check_has_explicit_domain)(multi) < 0)
		return NULL;
	return FN(DOM,copy)(multi->u.dom);
}

/* Replace the explicit domain of "multi" by "dom", assuming it has one.
 */
static __isl_give MULTI(BASE) *FN(MULTI(BASE),set_explicit_domain)(
	__isl_take MULTI(BASE) *multi, __isl_take DOM *dom)
{
	if (FN(MULTI(BASE),check_has_explicit_domain)(multi) < 0)
		goto error;
	multi = FN(MULTI(BASE),cow)(multi);
	if (!multi || !dom)
		goto error;
	FN(DOM,free)(multi->u.dom);
	multi->u.dom = dom;
	if (!multi->u.dom)
		return FN(MULTI(BASE),free)(multi);
	return multi;
error:
	FN(MULTI(BASE),free)(multi);
	FN(DOM,free)(dom);
	return NULL;
}

/* Intersect the domain of "dst" with the explicit domain of "src".
 */
static __isl_give MULTI(BASE) *FN(MULTI(BASE),intersect_explicit_domain)(
	__isl_take MULTI(BASE) *dst, __isl_keep MULTI(BASE) *src)
{
	DOM *dom;

	dom = FN(MULTI(BASE),get_explicit_domain)(src);
	dst = FN(MULTI(BASE),intersect_domain)(dst, dom);

	return dst;
}

/* Set the explicit domain of "dst" to that of "src".
 */
static __isl_give MULTI(BASE) *FN(MULTI(BASE),copy_explicit_domain)(
	__isl_take MULTI(BASE) *dst, __isl_keep MULTI(BASE) *src)
{
	DOM *dom;

	dom = FN(MULTI(BASE),get_explicit_domain)(src);
	dst = FN(MULTI(BASE),set_explicit_domain)(dst, dom);

	return dst;
}

/* Align the parameters of the explicit domain of "multi" to those of "space".
 */
static __isl_give MULTI(BASE) *FN(MULTI(BASE),align_explicit_domain_params)(
	__isl_take MULTI(BASE) *multi, __isl_take isl_space *space)
{
	DOM *dom;

	dom = FN(MULTI(BASE),get_explicit_domain)(multi);
	dom = FN(DOM,align_params)(dom, space);
	multi = FN(MULTI(BASE),set_explicit_domain)(multi, dom);

	return multi;
}

/* Replace the space of the explicit domain of "multi" by "space",
 * without modifying its dimension.
 */
static __isl_give MULTI(BASE) *FN(MULTI(BASE),reset_explicit_domain_space)(
	__isl_take MULTI(BASE) *multi, __isl_take isl_space *space)
{
	DOM *dom;

	dom = FN(MULTI(BASE),get_explicit_domain)(multi);
	dom = FN(DOM,reset_equal_dim_space)(dom, space);
	multi = FN(MULTI(BASE),set_explicit_domain)(multi, dom);

	return multi;
}

/* Free the explicit domain of "multi".
 */
static void FN(MULTI(BASE),free_explicit_domain)(__isl_keep MULTI(BASE) *multi)
{
	if (FN(MULTI(BASE),check_has_explicit_domain)(multi) < 0)
		return;
	FN(DOM,free)(multi->u.dom);
}

/* Do "multi1" and "multi2" have the same explicit domain?
 */
static isl_bool FN(MULTI(BASE),equal_explicit_domain)(
	__isl_keep MULTI(BASE) *multi1, __isl_keep MULTI(BASE) *multi2)
{
	DOM *dom1, *dom2;
	isl_bool equal;

	if (FN(MULTI(BASE),check_has_explicit_domain)(multi1) < 0 ||
	    FN(MULTI(BASE),check_has_explicit_domain)(multi2) < 0)
		return isl_bool_error;
	dom1 = FN(MULTI(BASE),get_explicit_domain)(multi1);
	dom2 = FN(MULTI(BASE),get_explicit_domain)(multi2);
	equal = FN(DOM,is_equal)(dom1, dom2);
	FN(DOM,free)(dom1);
	FN(DOM,free)(dom2);

	return equal;
}

static isl_stat FN(MULTI(BASE),check_explicit_domain)(
	__isl_keep MULTI(BASE) *multi) __attribute__ ((unused));

/* Debugging function to check that the explicit domain of "multi"
 * has the correct space.
 */
isl_stat FN(MULTI(BASE),check_explicit_domain)(__isl_keep MULTI(BASE) *multi)
{
	isl_space *space1, *space2;
	isl_bool equal;

	if (FN(MULTI(BASE),check_has_explicit_domain)(multi) < 0)
		return isl_stat_error;
	space1 = isl_space_domain(isl_space_copy(multi->space));
	space2 = FN(DOM,get_space)(multi->u.dom);
	equal = isl_space_is_equal(space1, space2);
	isl_space_free(space1);
	isl_space_free(space2);
	if (equal < 0)
		return isl_stat_error;
	if (!equal)
		isl_die(FN(MULTI(BASE),get_ctx)(multi), isl_error_internal,
			"check failed", return isl_stat_error);
	return isl_stat_ok;
}
