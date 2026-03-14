/*
 * Copyright 2013      Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 */

#include <isl/space.h>
#include <isl/local_space.h>
#include <isl_reordering.h>

#include <isl_multi_macro.h>

/* The functions in this file are meant for base object types
 * that do not have any associated space.  They are only meant to be used
 * in the generic isl_multi_* functions which have to deal with base objects
 * that do have an associated space.
 */


/* Drop the "n" first dimensions of type "type" at position "first".
 *
 * For a base expression without an associated space, this function
 * does not do anything.
 */
static __isl_give EL *FN(EL,drop_dims)(__isl_take EL *el,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	return el;
}

/* Return the space of "el".
 *
 * For a base expression without an associated space,
 * the conditions surrounding the call to this function make sure
 * that this function will never actually get called.  We return a valid
 * space anyway, just in case.
 */
static __isl_give isl_space *FN(EL,get_space)(__isl_keep EL *el)
{
	if (!el)
		return NULL;

	return isl_space_params_alloc(FN(EL,get_ctx)(el), 0);
}

/* Reset the domain space of "el" to "space".
 *
 * For a base expression without an associated space, this function
 * does not do anything, apart from error handling and cleaning up memory.
 */
static __isl_give EL *FN(EL,reset_domain_space)(__isl_take EL *el,
	__isl_take isl_space *space)
{
	if (!space)
		return FN(EL,free)(el);
	isl_space_free(space);
	return el;
}

/* Align the parameters of "el" to those of "space".
 *
 * For a base expression without an associated space, this function
 * does not do anything, apart from error handling and cleaning up memory.
 * Note that the conditions surrounding the call to this function make sure
 * that this function will never actually get called.
 */
static __isl_give EL *FN(EL,align_params)(__isl_take EL *el,
	__isl_take isl_space *space)
{
	if (!space)
		return FN(EL,free)(el);
	isl_space_free(space);
	return el;
}

/* Reorder the dimensions of the domain of "el" according
 * to the given reordering.
 *
 * For a base expression without an associated space, this function
 * does not do anything, apart from error handling and cleaning up memory.
 */
static __isl_give EL *FN(EL,realign_domain)(__isl_take EL *el,
	__isl_take isl_reordering *r)
{
	if (!r)
		return FN(EL,free)(el);
	isl_reordering_free(r);
	return el;
}

/* Do the parameters of "el" match those of "space"?
 *
 * For a base expression without an associated space, this function
 * simply returns true, except if "el" or "space" are NULL.
 */
static isl_bool FN(EL,matching_params)(__isl_keep EL *el,
	__isl_keep isl_space *space)
{
	if (!el || !space)
		return isl_bool_error;
	return isl_bool_true;
}

/* Check that the domain space of "el" matches "space".
 *
 * For a base expression without an associated space, this function
 * simply returns isl_stat_ok, except if "el" or "space" are NULL.
 */
static isl_stat FN(EL,check_match_domain_space)(__isl_keep EL *el,
	__isl_keep isl_space *space)
{
	if (!el || !space)
		return isl_stat_error;
	return isl_stat_ok;
}
