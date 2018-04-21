/*
 * Copyright 2017      Sven Verdoolaege
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege.
 */

/* These versions of the explicit domain functions are used
 * when the multi expression cannot have an explicit domain.
 */

#include <isl/space.h>

#include <isl_multi_macro.h>

/* Does "multi" have an explicit domain?
 *
 * No.
 */
static int FN(MULTI(BASE),has_explicit_domain)(__isl_keep MULTI(BASE) *multi)
{
	return 0;
}

/* Initialize the explicit domain of "multi".
 * "multi" cannot have an explicit domain, so this function is never called.
 */
static __isl_give MULTI(BASE) *FN(MULTI(BASE),init_explicit_domain)(
	__isl_take MULTI(BASE) *multi)
{
	return multi;
}

/* Intersect the domain of "dst" with the explicit domain of "src".
 * "src" cannot have an explicit domain, so this function is never called.
 */
static __isl_give MULTI(BASE) *FN(MULTI(BASE),intersect_explicit_domain)(
	__isl_take MULTI(BASE) *dst, __isl_keep MULTI(BASE) *src)
{
	return dst;
}

/* Set the explicit domain of "dst" to that of "src".
 * "src" and "dst" cannot have an explicit domain,
 * so this function is never called.
 */
static __isl_give MULTI(BASE) *FN(MULTI(BASE),copy_explicit_domain)(
	__isl_take MULTI(BASE) *dst, __isl_keep MULTI(BASE) *src)
{
	return dst;
}

/* Intersect the domain of "dst" with the domain product
 * of the explicit domains of "src1" and "src2".
 * This function is only called if at least one of "src1" or "src2"
 * has an explicit domain.
 * "src1", "src2" and "dst" cannot have an explicit domain,
 * so this function is never called.
 */
static __isl_give MULTI(BASE) *
FN(MULTI(BASE),intersect_explicit_domain_product)(
	__isl_take MULTI(BASE) *dst, __isl_keep MULTI(BASE) *src1,
	__isl_keep MULTI(BASE) *src2)
{
	return dst;
}

/* Align the parameters of the explicit domain of "multi" to those of "space".
 * "multi" cannot have an explicit domain, so this function is never called.
 */
static __isl_give MULTI(BASE) *FN(MULTI(BASE),align_explicit_domain_params)(
	__isl_take MULTI(BASE) *multi, __isl_take isl_space *space)
{
	isl_space_free(space);
	return multi;
}

/* Replace the space of the explicit domain of "multi" by "space",
 * without modifying its dimension.
 * "multi" cannot have an explicit domain, so this function is never called.
 */
static __isl_give MULTI(BASE) *FN(MULTI(BASE),reset_explicit_domain_space)(
	__isl_take MULTI(BASE) *multi, __isl_take isl_space *space)
{
	isl_space_free(space);
	return multi;
}

/* Check whether the explicit domain of "multi" has non-zero coefficients
 * for any dimension in the given range or if any of these dimensions appear
 * with non-zero coefficients in any of the integer divisions involved.
 * "multi" cannot have an explicit domain, so this function is never called.
 */
isl_bool FN(MULTI(BASE),involves_explicit_domain_dims)(
	__isl_keep MULTI(BASE) *multi,
	enum isl_dim_type type, unsigned pos, unsigned n)
{
	return isl_bool_false;
}

/* Insert "n" dimensions of type "type" at position "pos"
 * of the explicit domain of "multi".
 * "multi" cannot have an explicit domain, so this function is never called.
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),insert_explicit_domain_dims)(
	__isl_take MULTI(BASE) *multi,
	enum isl_dim_type type, unsigned pos, unsigned n)
{
	return multi;
}

/* Drop the "n" dimensions of type "type" starting at position "pos"
 * of the explicit domain of "multi".
 * "multi" cannot have an explicit domain, so this function is never called.
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),drop_explicit_domain_dims)(
	__isl_take MULTI(BASE) *multi,
	enum isl_dim_type type, unsigned pos, unsigned n)
{
	return multi;
}

/* Move the "n" dimensions of "src_type" starting at "src_pos" of
 * of the explicit domain of "multi" to dimensions of "dst_type" at "dst_pos".
 * "multi" cannot have an explicit domain, so this function is never called.
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),move_explicit_domain_dims)(
	__isl_take MULTI(BASE) *multi,
	enum isl_dim_type dst_type, unsigned dst_pos,
	enum isl_dim_type src_type, unsigned src_pos, unsigned n)
{
	return multi;
}

/* Free the explicit domain of "multi".
 * "multi" cannot have an explicit domain, so this function is never called.
 */
static void FN(MULTI(BASE),free_explicit_domain)(__isl_keep MULTI(BASE) *multi)
{
}

/* Do "multi1" and "multi2" have the same explicit domain?
 * "multi1" and "multi2" cannot have an explicit domain,
 * so this function is never called.
 */
static isl_bool FN(MULTI(BASE),equal_explicit_domain)(
	__isl_keep MULTI(BASE) *multi1, __isl_keep MULTI(BASE) *multi2)
{
	return isl_bool_true;
}

static isl_stat FN(MULTI(BASE),check_explicit_domain)(
	__isl_keep MULTI(BASE) *multi) __attribute__ ((unused));

/* Debugging function to check that the explicit domain of "multi"
 * has the correct space.
 * "multi" cannot have an explicit domain,
 * so this function should never be called.
 */
static isl_stat FN(MULTI(BASE),check_explicit_domain)(
	__isl_keep MULTI(BASE) *multi)
{
	return isl_stat_ok;
}
