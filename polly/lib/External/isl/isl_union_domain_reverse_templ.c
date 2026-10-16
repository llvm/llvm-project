/*
 * Copyright 2023      Sven Verdoolaege
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege
 */

/* Is "part" defined over a domain wrapping a binary relation?
 */
static isl_bool FN(UNION,select_domain_is_wrapping_entry)(__isl_keep PART *part,
	void *user)
{
	return isl_space_domain_is_wrapping(FN(PART,peek_space)(part));
}

/* Wrapper around PART_domain_reverse for use
 * as an isl_union_*_transform callback.
 */
static __isl_give PART *FN(UNION,domain_reverse_entry)(__isl_take PART *part,
	void *user)
{
	return FN(PART,domain_reverse)(part);
}

/* For each base expression defined on a domain (A -> B),
 * interchange A and B in the wrapped domain
 * to obtain an expression on the domain (B -> A) and
 * collect the results.
 */
__isl_give UNION *FN(UNION,domain_reverse)(__isl_keep UNION *u)
{
	S(UNION,transform_control) control = {
		.filter = &FN(UNION,select_domain_is_wrapping_entry),
		.fn = &FN(UNION,domain_reverse_entry),
	};

	return FN(UNION,transform)(u, &control);
}
