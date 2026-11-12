/*
 * Copyright 2011      Sven Verdoolaege
 * Copyright 2012-2014 Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d’Ulm, 75230 Paris, France
 */

#include <isl/id.h>
#include <isl_space_private.h>
#include <isl/set.h>
#include <isl_reordering.h>

#include <isl_multi_macro.h>

#define MULTI_NAME(BASE) "isl_multi_" #BASE

isl_ctx *FN(MULTI(BASE),get_ctx)(__isl_keep MULTI(BASE) *multi)
{
	return multi ? isl_space_get_ctx(multi->space) : NULL;
}

/* Return the space of "multi".
 */
__isl_keep isl_space *FN(MULTI(BASE),peek_space)(__isl_keep MULTI(BASE) *multi)
{
	return multi ? multi->space : NULL;
}

__isl_give isl_space *FN(MULTI(BASE),get_space)(__isl_keep MULTI(BASE) *multi)
{
	return isl_space_copy(FN(MULTI(BASE),peek_space)(multi));
}

__isl_give isl_space *FN(MULTI(BASE),get_domain_space)(
	__isl_keep MULTI(BASE) *multi)
{
	return multi ? isl_space_domain(isl_space_copy(multi->space)) : NULL;
}

/* Allocate a multi expression living in "space".
 *
 * If the number of base expressions is zero, then make sure
 * there is enough room in the structure for the explicit domain,
 * in case the type supports such an explicit domain.
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),alloc)(__isl_take isl_space *space)
{
	isl_ctx *ctx;
	isl_size n;
	MULTI(BASE) *multi;

	n = isl_space_dim(space, isl_dim_out);
	if (n < 0)
		goto error;

	ctx = isl_space_get_ctx(space);
	if (n > 0)
		multi = isl_calloc(ctx, MULTI(BASE),
			 sizeof(MULTI(BASE)) + (n - 1) * sizeof(struct EL *));
	else
		multi = isl_calloc(ctx, MULTI(BASE), sizeof(MULTI(BASE)));
	if (!multi)
		goto error;

	multi->space = space;
	multi->n = n;
	multi->ref = 1;
	if (FN(MULTI(BASE),has_explicit_domain)(multi))
		multi = FN(MULTI(BASE),init_explicit_domain)(multi);
	return multi;
error:
	isl_space_free(space);
	return NULL;
}

__isl_give MULTI(BASE) *FN(MULTI(BASE),dup)(__isl_keep MULTI(BASE) *multi)
{
	int i;
	MULTI(BASE) *dup;

	if (!multi)
		return NULL;

	dup = FN(MULTI(BASE),alloc)(isl_space_copy(multi->space));
	if (!dup)
		return NULL;

	for (i = 0; i < multi->n; ++i)
		dup = FN(FN(MULTI(BASE),set),BASE)(dup, i,
						    FN(EL,copy)(multi->u.p[i]));
	if (FN(MULTI(BASE),has_explicit_domain)(multi))
		dup = FN(MULTI(BASE),copy_explicit_domain)(dup, multi);

	return dup;
}

__isl_give MULTI(BASE) *FN(MULTI(BASE),cow)(__isl_take MULTI(BASE) *multi)
{
	if (!multi)
		return NULL;

	if (multi->ref == 1)
		return multi;

	multi->ref--;
	return FN(MULTI(BASE),dup)(multi);
}

__isl_give MULTI(BASE) *FN(MULTI(BASE),copy)(__isl_keep MULTI(BASE) *multi)
{
	if (!multi)
		return NULL;

	multi->ref++;
	return multi;
}

__isl_null MULTI(BASE) *FN(MULTI(BASE),free)(__isl_take MULTI(BASE) *multi)
{
	int i;

	if (!multi)
		return NULL;

	if (--multi->ref > 0)
		return NULL;

	isl_space_free(multi->space);
	for (i = 0; i < multi->n; ++i)
		FN(EL,free)(multi->u.p[i]);
	if (FN(MULTI(BASE),has_explicit_domain)(multi))
		FN(MULTI(BASE),free_explicit_domain)(multi);
	free(multi);

	return NULL;
}

/* Return the space of "multi".
 * The caller is not allowed to modify "multi" between this call
 * and the call to *_restore_space because the number
 * of references needs to stay the same.
 * The only exception is that isl_multi_*_free can be called instead.
 * No copy is taken of multi->space if "multi" has only one reference
 * such that it can be modified inplace if both have only a single reference.
 */
__isl_give isl_space *FN(MULTI(BASE),take_space)(__isl_keep MULTI(BASE) *multi)
{
	isl_space *space;

	if (!multi)
		return NULL;
	if (multi->ref != 1)
		return FN(MULTI(BASE),get_space)(multi);
	space = multi->space;
	multi->space = NULL;
	return space;
}

/* Set the space of "multi" to "space", where the space of "multi"
 * may be missing due to a preceding call to isl_multi_*_take_space.
 * However, in this case, "multi" only has a single reference and
 * then the call to isl_multi_*_cow has no effect.
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),restore_space)(
	__isl_take MULTI(BASE) *multi, __isl_take isl_space *space)
{
	if (!multi || !space)
		goto error;

	if (multi->space == space) {
		isl_space_free(space);
		return multi;
	}

	multi = FN(MULTI(BASE),cow)(multi);
	if (!multi)
		goto error;
	isl_space_free(multi->space);
	multi->space = space;

	return multi;
error:
	FN(MULTI(BASE),free)(multi);
	isl_space_free(space);
	return NULL;
}

isl_size FN(MULTI(BASE),dim)(__isl_keep MULTI(BASE) *multi,
	enum isl_dim_type type)
{
	return isl_space_dim(FN(MULTI(BASE),peek_space)(multi), type);
}

/* Return the number of base expressions in "multi".
 */
isl_size FN(MULTI(BASE),size)(__isl_keep MULTI(BASE) *multi)
{
	return multi ? multi->n : isl_size_error;
}

#undef TYPE
#define TYPE	MULTI(BASE)
static
#include "check_type_range_templ.c"

/* Return the base expression at position "pos" in "multi".
 */
static __isl_give EL *FN(MULTI(BASE),peek_at)(__isl_keep MULTI(BASE) *multi,
	int pos)
{
	if (FN(MULTI(BASE),check_range)(multi, isl_dim_out, pos, 1) < 0)
		return NULL;
	return multi->u.p[pos];
}

/* Return a copy of the base expression at position "pos" in "multi".
 */
__isl_give EL *FN(MULTI(BASE),get_at)(__isl_keep MULTI(BASE) *multi, int pos)
{
	return FN(EL,copy)(FN(MULTI(BASE),peek_at)(multi, pos));
}

/* This is an alternative name for the function above.
 */
__isl_give EL *FN(FN(MULTI(BASE),get),BASE)(__isl_keep MULTI(BASE) *multi,
	int pos)
{
	return FN(MULTI(BASE),get_at)(multi, pos);
}

/* Return the base expression at position "pos" in "multi".
 * This may be either a copy or the base expression itself
 * if there is only one reference to "multi".
 * This allows the base expression to be modified inplace
 * if both the multi expression and this base expression
 * have only a single reference.
 * The caller is not allowed to modify "multi" between this call and
 * the subsequent call to isl_multi_*_restore_at_*.
 * The only exception is that isl_multi_*_free can be called instead.
 */
static __isl_give EL *FN(MULTI(BASE),take_at)(__isl_keep MULTI(BASE) *multi,
	int pos)
{
	EL *el;

	if (!multi)
		return NULL;
	if (multi->ref != 1)
		return FN(MULTI(BASE),get_at)(multi, pos);
	if (FN(MULTI(BASE),check_range)(multi, isl_dim_out, pos, 1) < 0)
		return NULL;
	el = multi->u.p[pos];
	multi->u.p[pos] = NULL;
	return el;
}

/* Set the element at position "pos" of "multi" to "el",
 * where the position may be empty if "multi" has only a single reference.
 */
static __isl_give MULTI(BASE) *FN(MULTI(BASE),restore_at)(
	__isl_take MULTI(BASE) *multi, int pos, __isl_take EL *el)
{
	if (FN(MULTI(BASE),check_range)(multi, isl_dim_out, pos, 1) < 0 || !el)
		goto error;

	if (multi->u.p[pos] == el) {
		FN(EL,free)(el);
		return multi;
	}

	multi = FN(MULTI(BASE),cow)(multi);
	if (!multi)
		goto error;

	FN(EL,free)(multi->u.p[pos]);
	multi->u.p[pos] = el;

	return multi;
error:
	FN(MULTI(BASE),free)(multi);
	FN(EL,free)(el);
	return NULL;
}

/* Set the element at position "pos" of "multi" to "el",
 * where the position may be empty if "multi" has only a single reference.
 * However, the space of "multi" is available and is checked
 * for compatibility with "el".
 */
static __isl_give MULTI(BASE) *FN(MULTI(BASE),restore_check_space)(
	__isl_take MULTI(BASE) *multi, int pos, __isl_take EL *el)
{
	isl_space *space;

	space = FN(MULTI(BASE),peek_space)(multi);
	if (FN(EL,check_match_domain_space)(el, space) < 0)
		multi = FN(MULTI(BASE),free)(multi);
	return FN(MULTI(BASE),restore_at)(multi, pos, el);
}

/* Replace the base expression at position "pos" in "multi" with "el".
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),set_at)(
	__isl_take MULTI(BASE) *multi, int pos, __isl_take EL *el)
{
	isl_space *multi_space = NULL;
	isl_space *el_space = NULL;
	isl_bool match;

	multi_space = FN(MULTI(BASE),get_space)(multi);
	match = FN(EL,matching_params)(el, multi_space);
	if (match < 0)
		goto error;
	if (!match) {
		multi = FN(MULTI(BASE),align_params)(multi,
						    FN(EL,get_space)(el));
		isl_space_free(multi_space);
		multi_space = FN(MULTI(BASE),get_space)(multi);
		el = FN(EL,align_params)(el, isl_space_copy(multi_space));
	}

	multi = FN(MULTI(BASE),restore_check_space)(multi, pos, el);

	isl_space_free(multi_space);
	isl_space_free(el_space);

	return multi;
error:
	FN(MULTI(BASE),free)(multi);
	FN(EL,free)(el);
	isl_space_free(multi_space);
	isl_space_free(el_space);
	return NULL;
}

/* This is an alternative name for the function above.
 */
__isl_give MULTI(BASE) *FN(FN(MULTI(BASE),set),BASE)(
	__isl_take MULTI(BASE) *multi, int pos, __isl_take EL *el)
{
	return FN(MULTI(BASE),set_at)(multi, pos, el);
}

/* Return the base expressions of "multi" as a list.
 */
__isl_give LIST(EL) *FN(MULTI(BASE),get_list)(
	__isl_keep MULTI(BASE) *multi)
{
	isl_size n;
	int i;
	LIST(EL) *list;

	n = FN(MULTI(BASE),size)(multi);
	if (n < 0)
		return NULL;
	list = FN(LIST(EL),alloc)(FN(MULTI(BASE),get_ctx(multi)), n);
	for (i = 0; i < n; ++i) {
		EL *el = FN(MULTI(BASE),get_at)(multi, i);
		list = FN(LIST(EL),add)(list, el);
	}

	return list;
}

/* Reset the space of "multi".  This function is called from isl_pw_templ.c
 * and doesn't know if the space of an element object is represented
 * directly or through its domain.  It therefore passes along both,
 * which we pass along to the element function since we don't know how
 * that is represented either.
 *
 * If "multi" has an explicit domain, then the caller is expected
 * to make sure that any modification that would change the dimensions
 * of the explicit domain has bee applied before this function is called.
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),reset_space_and_domain)(
	__isl_take MULTI(BASE) *multi, __isl_take isl_space *space,
	__isl_take isl_space *domain)
{
	isl_size n;
	int i;

	n = FN(MULTI(BASE),size)(multi);
	if (n < 0 || !space || !domain)
		goto error;

	for (i = 0; i < n; ++i) {
		EL *el;

		el = FN(MULTI(BASE),take_at)(multi, i);
		el = FN(EL,reset_domain_space)(el, isl_space_copy(domain));
		multi = FN(MULTI(BASE),restore_at)(multi, i, el);
	}
	if (FN(MULTI(BASE),has_explicit_domain)(multi))
		multi = FN(MULTI(BASE),reset_explicit_domain_space)(multi,
							isl_space_copy(domain));
	isl_space_free(domain);

	multi = FN(MULTI(BASE),restore_space)(multi, space);

	return multi;
error:
	isl_space_free(domain);
	isl_space_free(space);
	FN(MULTI(BASE),free)(multi);
	return NULL;
}

__isl_give MULTI(BASE) *FN(MULTI(BASE),reset_domain_space)(
	__isl_take MULTI(BASE) *multi, __isl_take isl_space *domain)
{
	isl_space *space, *multi_space;

	multi_space = FN(MULTI(BASE),get_space)(multi);
	space = isl_space_extend_domain_with_range(isl_space_copy(domain),
						multi_space);
	return FN(MULTI(BASE),reset_space_and_domain)(multi, space, domain);
}

__isl_give MULTI(BASE) *FN(MULTI(BASE),reset_space)(
	__isl_take MULTI(BASE) *multi, __isl_take isl_space *space)
{
	isl_space *domain;

	domain = isl_space_domain(isl_space_copy(space));
	return FN(MULTI(BASE),reset_space_and_domain)(multi, space, domain);
}

/* Reset the user pointer on all identifiers of parameters and tuples
 * of the space of "multi".
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),reset_user)(
	__isl_take MULTI(BASE) *multi)
{
	isl_space *space;

	space = FN(MULTI(BASE),get_space)(multi);
	space = isl_space_reset_user(space);

	return FN(MULTI(BASE),reset_space)(multi, space);
}

__isl_give MULTI(BASE) *FN(MULTI(BASE),realign_domain)(
	__isl_take MULTI(BASE) *multi, __isl_take isl_reordering *exp)
{
	int i;
	isl_size n;
	isl_space *space;

	n = FN(MULTI(BASE),size)(multi);
	if (n < 0 || !exp)
		goto error;

	for (i = 0; i < n; ++i) {
		EL *el;

		el = FN(MULTI(BASE),take_at)(multi, i);
		el = FN(EL,realign_domain)(el, isl_reordering_copy(exp));
		multi = FN(MULTI(BASE),restore_at)(multi, i, el);
	}

	space = isl_reordering_get_space(exp);
	multi = FN(MULTI(BASE),reset_domain_space)(multi, space);

	isl_reordering_free(exp);
	return multi;
error:
	isl_reordering_free(exp);
	FN(MULTI(BASE),free)(multi);
	return NULL;
}

/* Align the parameters of "multi" to those of "model".
 *
 * If "multi" has an explicit domain, then align the parameters
 * of the domain first.
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),align_params)(
	__isl_take MULTI(BASE) *multi, __isl_take isl_space *model)
{
	isl_ctx *ctx;
	isl_bool equal_params;
	isl_space *domain_space;
	isl_reordering *exp;

	if (!multi || !model)
		goto error;

	equal_params = isl_space_has_equal_params(multi->space, model);
	if (equal_params < 0)
		goto error;
	if (equal_params) {
		isl_space_free(model);
		return multi;
	}

	ctx = isl_space_get_ctx(model);
	if (!isl_space_has_named_params(model))
		isl_die(ctx, isl_error_invalid,
			"model has unnamed parameters", goto error);
	if (!isl_space_has_named_params(multi->space))
		isl_die(ctx, isl_error_invalid,
			"input has unnamed parameters", goto error);

	if (FN(MULTI(BASE),has_explicit_domain)(multi)) {
		multi = FN(MULTI(BASE),align_explicit_domain_params)(multi,
							isl_space_copy(model));
		if (!multi)
			goto error;
	}
	domain_space = FN(MULTI(BASE),get_domain_space)(multi);
	exp = isl_parameter_alignment_reordering(domain_space, model);
	isl_space_free(domain_space);
	multi = FN(MULTI(BASE),realign_domain)(multi, exp);

	isl_space_free(model);
	return multi;
error:
	isl_space_free(model);
	FN(MULTI(BASE),free)(multi);
	return NULL;
}

/* Create a multi expression in the given space with the elements of "list"
 * as base expressions.
 *
 * Since isl_multi_*_restore_* assumes that the element and
 * the multi expression have matching spaces, the alignment
 * (if any) needs to be performed beforehand.
 */
__isl_give MULTI(BASE) *FN(FN(MULTI(BASE),from),LIST(BASE))(
	__isl_take isl_space *space, __isl_take LIST(EL) *list)
{
	int i;
	isl_size n, dim;
	isl_ctx *ctx;
	MULTI(BASE) *multi;

	dim = isl_space_dim(space, isl_dim_out);
	n = FN(FN(LIST(EL),n),BASE)(list);
	if (dim < 0 || n < 0)
		goto error;

	ctx = isl_space_get_ctx(space);
	if (n != dim)
		isl_die(ctx, isl_error_invalid,
			"invalid number of elements in list", goto error);

	for (i = 0; i < n; ++i) {
		EL *el = FN(LIST(EL),peek)(list, i);
		space = isl_space_align_params(space, FN(EL,get_space)(el));
	}
	multi = FN(MULTI(BASE),alloc)(isl_space_copy(space));
	for (i = 0; i < n; ++i) {
		EL *el = FN(FN(LIST(EL),get),BASE)(list, i);
		el = FN(EL,align_params)(el, isl_space_copy(space));
		multi = FN(MULTI(BASE),restore_check_space)(multi, i, el);
	}

	isl_space_free(space);
	FN(LIST(EL),free)(list);
	return multi;
error:
	isl_space_free(space);
	FN(LIST(EL),free)(list);
	return NULL;
}

/* This function performs the same operation as isl_multi_*_from_*_list,
 * but is considered as a function on an isl_space when exported.
 */
__isl_give MULTI(BASE) *FN(isl_space_multi,BASE)(__isl_take isl_space *space,
	__isl_take LIST(EL) *list)
{
	return FN(FN(MULTI(BASE),from),LIST(BASE))(space, list);
}

/* Drop the "n" output dimensions of "multi" starting at "first",
 * where the space is assumed to have been adjusted already.
 */
static __isl_give MULTI(BASE) *FN(MULTI(BASE),drop_output_dims)(
	__isl_take MULTI(BASE) *multi, unsigned first, unsigned n)
{
	int i;

	multi = FN(MULTI(BASE),cow)(multi);
	if (!multi)
		return NULL;

	for (i = 0; i < n; ++i)
		FN(EL,free)(multi->u.p[first + i]);
	for (i = first; i + n < multi->n; ++i)
		multi->u.p[i] = multi->u.p[i + n];
	multi->n -= n;
	if (n > 0 && FN(MULTI(BASE),has_explicit_domain)(multi))
		multi = FN(MULTI(BASE),init_explicit_domain)(multi);

	return multi;
}

__isl_give MULTI(BASE) *FN(MULTI(BASE),drop_dims)(
	__isl_take MULTI(BASE) *multi,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	isl_space *space;
	isl_size size;
	int i;

	if (FN(MULTI(BASE),check_range)(multi, type, first, n) < 0)
		return FN(MULTI(BASE),free)(multi);

	space = FN(MULTI(BASE),take_space)(multi);
	space = isl_space_drop_dims(space, type, first, n);
	multi = FN(MULTI(BASE),restore_space)(multi, space);

	if (type == isl_dim_out)
		return FN(MULTI(BASE),drop_output_dims)(multi, first, n);

	if (FN(MULTI(BASE),has_explicit_domain)(multi))
		multi = FN(MULTI(BASE),drop_explicit_domain_dims)(multi,
								type, first, n);

	size = FN(MULTI(BASE),size)(multi);
	if (size < 0)
		return FN(MULTI(BASE),free)(multi);
	for (i = 0; i < size; ++i) {
		EL *el;

		el = FN(MULTI(BASE),take_at)(multi, i);
		el = FN(EL,drop_dims)(el, type, first, n);
		multi = FN(MULTI(BASE),restore_at)(multi, i, el);
	}

	return multi;
}

#undef TYPE
#define TYPE MULTI(BASE)

#include "isl_check_named_params_templ.c"
static
#include "isl_align_params_bin_templ.c"

/* Given two MULTI(BASE)s A -> B and C -> D,
 * construct a MULTI(BASE) (A * C) -> [B -> D].
 *
 * If "multi1" and/or "multi2" has an explicit domain, then
 * intersect the domain of the result with these explicit domains.
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),range_product)(
	__isl_take MULTI(BASE) *multi1, __isl_take MULTI(BASE) *multi2)
{
	int i;
	isl_size n1, n2;
	EL *el;
	isl_space *space;
	MULTI(BASE) *res;

	FN(MULTI(BASE),align_params_bin)(&multi1, &multi2);
	n1 = FN(MULTI(BASE),size)(multi1);
	n2 = FN(MULTI(BASE),size)(multi2);
	if (n1 < 0 || n2 < 0)
		goto error;

	space = isl_space_range_product(FN(MULTI(BASE),get_space)(multi1),
					FN(MULTI(BASE),get_space)(multi2));
	res = FN(MULTI(BASE),alloc)(space);

	for (i = 0; i < n1; ++i) {
		el = FN(FN(MULTI(BASE),get),BASE)(multi1, i);
		res = FN(FN(MULTI(BASE),set),BASE)(res, i, el);
	}

	for (i = 0; i < n2; ++i) {
		el = FN(FN(MULTI(BASE),get),BASE)(multi2, i);
		res = FN(FN(MULTI(BASE),set),BASE)(res, n1 + i, el);
	}

	if (FN(MULTI(BASE),has_explicit_domain)(multi1))
		res = FN(MULTI(BASE),intersect_explicit_domain)(res, multi1);
	if (FN(MULTI(BASE),has_explicit_domain)(multi2))
		res = FN(MULTI(BASE),intersect_explicit_domain)(res, multi2);

	FN(MULTI(BASE),free)(multi1);
	FN(MULTI(BASE),free)(multi2);
	return res;
error:
	FN(MULTI(BASE),free)(multi1);
	FN(MULTI(BASE),free)(multi2);
	return NULL;
}

/* Is the range of "multi" a wrapped relation?
 */
isl_bool FN(MULTI(BASE),range_is_wrapping)(__isl_keep MULTI(BASE) *multi)
{
	if (!multi)
		return isl_bool_error;
	return isl_space_range_is_wrapping(multi->space);
}

/* Given a function A -> [B -> C], extract the function A -> B.
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),range_factor_domain)(
	__isl_take MULTI(BASE) *multi)
{
	isl_space *space;
	isl_size total, keep;

	total = FN(MULTI(BASE),dim)(multi, isl_dim_out);
	if (total < 0)
		return FN(MULTI(BASE),free)(multi);
	if (!isl_space_range_is_wrapping(multi->space))
		isl_die(FN(MULTI(BASE),get_ctx)(multi), isl_error_invalid,
			"range is not a product",
			return FN(MULTI(BASE),free)(multi));

	space = FN(MULTI(BASE),get_space)(multi);
	space = isl_space_range_factor_domain(space);
	keep = isl_space_dim(space, isl_dim_out);
	if (keep < 0)
		multi = FN(MULTI(BASE),free)(multi);
	multi = FN(MULTI(BASE),drop_dims)(multi,
					isl_dim_out, keep, total - keep);
	multi = FN(MULTI(BASE),reset_space)(multi, space);

	return multi;
}

/* Given a function A -> [B -> C], extract the function A -> C.
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),range_factor_range)(
	__isl_take MULTI(BASE) *multi)
{
	isl_space *space;
	isl_size total, keep;

	total = FN(MULTI(BASE),dim)(multi, isl_dim_out);
	if (total < 0)
		return FN(MULTI(BASE),free)(multi);
	if (!isl_space_range_is_wrapping(multi->space))
		isl_die(FN(MULTI(BASE),get_ctx)(multi), isl_error_invalid,
			"range is not a product",
			return FN(MULTI(BASE),free)(multi));

	space = FN(MULTI(BASE),get_space)(multi);
	space = isl_space_range_factor_range(space);
	keep = isl_space_dim(space, isl_dim_out);
	if (keep < 0)
		multi = FN(MULTI(BASE),free)(multi);
	multi = FN(MULTI(BASE),drop_dims)(multi, isl_dim_out, 0, total - keep);
	multi = FN(MULTI(BASE),reset_space)(multi, space);

	return multi;
}

/* Given a function [B -> C], extract the function C.
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),factor_range)(
	__isl_take MULTI(BASE) *multi)
{
	isl_space *space;
	isl_size total, keep;

	total = FN(MULTI(BASE),dim)(multi, isl_dim_set);
	if (total < 0)
		return FN(MULTI(BASE),free)(multi);
	if (!isl_space_is_wrapping(multi->space))
		isl_die(FN(MULTI(BASE),get_ctx)(multi), isl_error_invalid,
			"not a product", return FN(MULTI(BASE),free)(multi));

	space = FN(MULTI(BASE),get_space)(multi);
	space = isl_space_factor_range(space);
	keep = isl_space_dim(space, isl_dim_set);
	if (keep < 0)
		multi = FN(MULTI(BASE),free)(multi);
	multi = FN(MULTI(BASE),drop_dims)(multi, isl_dim_set, 0, total - keep);
	multi = FN(MULTI(BASE),reset_space)(multi, space);

	return multi;
}

__isl_give MULTI(BASE) *FN(MULTI(BASE),flatten_range)(
	__isl_take MULTI(BASE) *multi)
{
	isl_space *space;

	space = FN(MULTI(BASE),take_space)(multi);
	space = isl_space_flatten_range(space);
	multi = FN(MULTI(BASE),restore_space)(multi, space);

	return multi;
}

/* Given two MULTI(BASE)s A -> B and C -> D,
 * construct a MULTI(BASE) (A * C) -> (B, D).
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),flat_range_product)(
	__isl_take MULTI(BASE) *multi1, __isl_take MULTI(BASE) *multi2)
{
	MULTI(BASE) *multi;

	multi = FN(MULTI(BASE),range_product)(multi1, multi2);
	multi = FN(MULTI(BASE),flatten_range)(multi);
	return multi;
}

/* Given two multi expressions, "multi1"
 *
 *	[A] -> [B1 B2]
 *
 * where B2 starts at position "pos", and "multi2"
 *
 *	[A] -> [D]
 *
 * return the multi expression
 *
 *	[A] -> [B1 D B2]
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),range_splice)(
	__isl_take MULTI(BASE) *multi1, unsigned pos,
	__isl_take MULTI(BASE) *multi2)
{
	MULTI(BASE) *res;
	isl_size dim;

	dim = FN(MULTI(BASE),size)(multi1);
	if (dim < 0 || !multi2)
		goto error;

	if (FN(MULTI(BASE),check_range)(multi1, isl_dim_out, pos, 0) < 0)
		goto error;

	res = FN(MULTI(BASE),copy)(multi1);
	res = FN(MULTI(BASE),drop_dims)(res, isl_dim_out, pos, dim - pos);
	multi1 = FN(MULTI(BASE),drop_dims)(multi1, isl_dim_out, 0, pos);

	res = FN(MULTI(BASE),flat_range_product)(res, multi2);
	res = FN(MULTI(BASE),flat_range_product)(res, multi1);

	return res;
error:
	FN(MULTI(BASE),free)(multi1);
	FN(MULTI(BASE),free)(multi2);
	return NULL;
}

#undef TYPE
#define TYPE	MULTI(BASE)

static
#include "isl_type_has_equal_space_bin_templ.c"
static
#include "isl_type_check_equal_space_templ.c"

/* This function is currently only used from isl_aff.c
 */
static __isl_give MULTI(BASE) *FN(MULTI(BASE),bin_op)(
	__isl_take MULTI(BASE) *multi1, __isl_take MULTI(BASE) *multi2,
	__isl_give EL *(*fn)(__isl_take EL *, __isl_take EL *))
	__attribute__ ((unused));

/* Pairwise perform "fn" to the elements of "multi1" and "multi2" and
 * return the result.
 *
 * If "multi2" has an explicit domain, then
 * intersect the domain of the result with this explicit domain.
 */
static __isl_give MULTI(BASE) *FN(MULTI(BASE),bin_op)(
	__isl_take MULTI(BASE) *multi1, __isl_take MULTI(BASE) *multi2,
	__isl_give EL *(*fn)(__isl_take EL *, __isl_take EL *))
{
	isl_size n;
	int i;

	FN(MULTI(BASE),align_params_bin)(&multi1, &multi2);
	n = FN(MULTI(BASE),size)(multi1);
	if (n < 0 || FN(MULTI(BASE),check_equal_space)(multi1, multi2) < 0)
		goto error;

	for (i = 0; i < n; ++i) {
		EL *el1, *el2;

		el2 = FN(MULTI(BASE),get_at)(multi2, i);
		el1 = FN(MULTI(BASE),take_at)(multi1, i);
		el1 = fn(el1, el2);
		multi1 = FN(MULTI(BASE),restore_at)(multi1, i, el1);
	}

	if (FN(MULTI(BASE),has_explicit_domain)(multi2))
		multi1 = FN(MULTI(BASE),intersect_explicit_domain)(multi1,
								    multi2);

	FN(MULTI(BASE),free)(multi2);
	return multi1;
error:
	FN(MULTI(BASE),free)(multi1);
	FN(MULTI(BASE),free)(multi2);
	return NULL;
}

/* Only used on some multi-expressions.
 */
static isl_bool FN(MULTI(BASE),any)(__isl_keep MULTI(BASE) *multi,
	isl_bool (*test)(__isl_keep EL *)) __attribute__ ((unused));

/* Does "test" succeed on any base expression of "multi"?
 */
static isl_bool FN(MULTI(BASE),any)(__isl_keep MULTI(BASE) *multi,
	isl_bool (*test)(__isl_keep EL *))
{
	isl_size n;
	int i;

	n = FN(MULTI(BASE),size)(multi);
	if (n < 0)
		return isl_bool_error;

	for (i = 0; i < n; ++i) {
		isl_bool any = test(multi->u.p[i]);
		if (any < 0 || any)
			return any;
	}

	return isl_bool_false;
}

/* Only used on some multi-expressions.
 */
static isl_bool FN(MULTI(BASE),every)(__isl_keep MULTI(BASE) *multi,
	isl_bool (*test)(__isl_keep EL *)) __attribute__ ((unused));

/* Does "test" succeed on every base expression of "multi"?
 */
static isl_bool FN(MULTI(BASE),every)(__isl_keep MULTI(BASE) *multi,
	isl_bool (*test)(__isl_keep EL *))
{
	isl_size n;
	int i;

	n = FN(MULTI(BASE),size)(multi);
	if (n < 0)
		return isl_bool_error;

	for (i = 0; i < n; ++i) {
		isl_bool every = test(multi->u.p[i]);
		if (every < 0 || !every)
			return every;
	}

	return isl_bool_true;
}

#undef TYPE
#define TYPE MULTI(BASE)
#include "isl_from_range_templ.c"

/* Are "multi1" and "multi2" obviously equal?
 */
isl_bool FN(MULTI(BASE),plain_is_equal)(__isl_keep MULTI(BASE) *multi1,
	__isl_keep MULTI(BASE) *multi2)
{
	int i;
	isl_bool equal;

	if (!multi1 || !multi2)
		return isl_bool_error;
	if (multi1->n != multi2->n)
		return isl_bool_false;
	equal = isl_space_is_equal(multi1->space, multi2->space);
	if (equal < 0 || !equal)
		return equal;

	for (i = 0; i < multi1->n; ++i) {
		equal = FN(EL,plain_is_equal)(multi1->u.p[i], multi2->u.p[i]);
		if (equal < 0 || !equal)
			return equal;
	}

	if (FN(MULTI(BASE),has_explicit_domain)(multi1) ||
	    FN(MULTI(BASE),has_explicit_domain)(multi2)) {
		equal = FN(MULTI(BASE),equal_explicit_domain)(multi1, multi2);
		if (equal < 0 || !equal)
			return equal;
	}

	return isl_bool_true;
}
