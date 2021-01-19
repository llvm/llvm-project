/*
 * Copyright 2010-2011 INRIA Saclay
 * Copyright 2011      Sven Verdoolaege
 * Copyright 2012-2014 Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, INRIA Saclay - Ile-de-France,
 * Parc Club Orsay Universite, ZAC des vignes, 4 rue Jacques Monod,
 * 91893 Orsay, France
 * and Ecole Normale Superieure, 45 rue d’Ulm, 75230 Paris, France
 */

#include <isl/id.h>
#include <isl/aff.h>
#include <isl_sort.h>
#include <isl_val_private.h>

#include <isl_pw_macro.h>

#include "opt_type.h"

__isl_give PW *FN(PW,alloc_size)(__isl_take isl_space *space
	OPT_TYPE_PARAM, int n)
{
	isl_ctx *ctx;
	struct PW *pw;

	if (!space)
		return NULL;
	ctx = isl_space_get_ctx(space);
	isl_assert(ctx, n >= 0, goto error);
	pw = isl_alloc(ctx, struct PW,
			sizeof(struct PW) + (n - 1) * sizeof(S(PW,piece)));
	if (!pw)
		goto error;

	pw->ref = 1;
	OPT_SET_TYPE(pw->, type);
	pw->size = n;
	pw->n = 0;
	pw->dim = space;
	return pw;
error:
	isl_space_free(space);
	return NULL;
}

__isl_give PW *FN(PW,ZERO)(__isl_take isl_space *space OPT_TYPE_PARAM)
{
	return FN(PW,alloc_size)(space OPT_TYPE_ARG(NO_LOC), 0);
}

/* Add a piece with domain "set" and base expression "el"
 * to the piecewise expression "pw".
 *
 * Do this independently of the values of "set" and "el",
 * such that this function can be used by isl_pw_*_dup.
 */
__isl_give PW *FN(PW,add_dup_piece)(__isl_take PW *pw,
	__isl_take isl_set *set, __isl_take EL *el)
{
	isl_ctx *ctx;
	isl_space *el_dim = NULL;

	if (!pw || !set || !el)
		goto error;

	ctx = isl_set_get_ctx(set);
	if (!OPT_EQUAL_TYPES(pw->, el->))
		isl_die(ctx, isl_error_invalid, "fold types don't match",
			goto error);
	el_dim = FN(EL,get_space(el));
	isl_assert(ctx, isl_space_is_equal(pw->dim, el_dim), goto error);
	isl_assert(ctx, pw->n < pw->size, goto error);

	pw->p[pw->n].set = set;
	pw->p[pw->n].FIELD = el;
	pw->n++;
	
	isl_space_free(el_dim);
	return pw;
error:
	isl_space_free(el_dim);
	FN(PW,free)(pw);
	isl_set_free(set);
	FN(EL,free)(el);
	return NULL;
}

/* Add a piece with domain "set" and base expression "el"
 * to the piecewise expression "pw", provided the domain
 * is not obviously empty and the base expression
 * is not equal to the default value.
 */
__isl_give PW *FN(PW,add_piece)(__isl_take PW *pw,
	__isl_take isl_set *set, __isl_take EL *el)
{
	isl_bool skip;

	skip = isl_set_plain_is_empty(set);
	if (skip >= 0 && !skip)
		skip = FN(EL,EL_IS_ZERO)(el);
	if (skip >= 0 && !skip)
		return FN(PW,add_dup_piece)(pw, set, el);

	isl_set_free(set);
	FN(EL,free)(el);
	if (skip < 0)
		return FN(PW,free)(pw);
	return pw;
}

/* Does the space of "set" correspond to that of the domain of "el".
 */
static isl_bool FN(PW,compatible_domain)(__isl_keep EL *el,
	__isl_keep isl_set *set)
{
	isl_bool ok;
	isl_space *el_space, *set_space;

	if (!set || !el)
		return isl_bool_error;
	set_space = isl_set_get_space(set);
	el_space = FN(EL,get_space)(el);
	ok = isl_space_is_domain_internal(set_space, el_space);
	isl_space_free(el_space);
	isl_space_free(set_space);
	return ok;
}

/* Check that the space of "set" corresponds to that of the domain of "el".
 */
static isl_stat FN(PW,check_compatible_domain)(__isl_keep EL *el,
	__isl_keep isl_set *set)
{
	isl_bool ok;

	ok = FN(PW,compatible_domain)(el, set);
	if (ok < 0)
		return isl_stat_error;
	if (!ok)
		isl_die(isl_set_get_ctx(set), isl_error_invalid,
			"incompatible spaces", return isl_stat_error);

	return isl_stat_ok;
}

__isl_give PW *FN(PW,alloc)(OPT_TYPE_PARAM_FIRST
	__isl_take isl_set *set, __isl_take EL *el)
{
	PW *pw;

	if (FN(PW,check_compatible_domain)(el, set) < 0)
		goto error;

	pw = FN(PW,alloc_size)(FN(EL,get_space)(el) OPT_TYPE_ARG(NO_LOC), 1);

	return FN(PW,add_piece)(pw, set, el);
error:
	isl_set_free(set);
	FN(EL,free)(el);
	return NULL;
}

__isl_give PW *FN(PW,dup)(__isl_keep PW *pw)
{
	int i;
	PW *dup;

	if (!pw)
		return NULL;

	dup = FN(PW,alloc_size)(isl_space_copy(pw->dim)
				OPT_TYPE_ARG(pw->), pw->n);
	if (!dup)
		return NULL;

	for (i = 0; i < pw->n; ++i)
		dup = FN(PW,add_dup_piece)(dup, isl_set_copy(pw->p[i].set),
					    FN(EL,copy)(pw->p[i].FIELD));

	return dup;
}

__isl_give PW *FN(PW,cow)(__isl_take PW *pw)
{
	if (!pw)
		return NULL;

	if (pw->ref == 1)
		return pw;
	pw->ref--;
	return FN(PW,dup)(pw);
}

__isl_give PW *FN(PW,copy)(__isl_keep PW *pw)
{
	if (!pw)
		return NULL;

	pw->ref++;
	return pw;
}

__isl_null PW *FN(PW,free)(__isl_take PW *pw)
{
	int i;

	if (!pw)
		return NULL;
	if (--pw->ref > 0)
		return NULL;

	for (i = 0; i < pw->n; ++i) {
		isl_set_free(pw->p[i].set);
		FN(EL,free)(pw->p[i].FIELD);
	}
	isl_space_free(pw->dim);
	free(pw);

	return NULL;
}

/* Create a piecewise expression with the given base expression on a universe
 * domain.
 */
static __isl_give PW *FN(FN(FN(PW,from),BASE),type_base)(__isl_take EL *el
	OPT_TYPE_PARAM)
{
	isl_set *dom = isl_set_universe(FN(EL,get_domain_space)(el));
	return FN(PW,alloc)(OPT_TYPE_ARG_FIRST(NO_LOC) dom, el);
}

/* Create a piecewise expression with the given base expression on a universe
 * domain.
 *
 * If the default value of this piecewise type is zero and
 * if "el" is effectively zero, then create an empty piecewise expression
 * instead.
 */
static __isl_give PW *FN(FN(FN(PW,from),BASE),type)(__isl_take EL *el
	OPT_TYPE_PARAM)
{
	isl_bool is_zero;
	isl_space *space;

	if (!DEFAULT_IS_ZERO)
		return FN(FN(FN(PW,from),BASE),type_base)(el
							OPT_TYPE_ARG(NO_LOC));
	is_zero = FN(EL,EL_IS_ZERO)(el);
	if (is_zero < 0)
		goto error;
	if (!is_zero)
		return FN(FN(FN(PW,from),BASE),type_base)(el
							OPT_TYPE_ARG(NO_LOC));
	space = FN(EL,get_space)(el);
	FN(EL,free)(el);
	return FN(PW,ZERO)(space OPT_TYPE_ARG(NO_LOC));
error:
	FN(EL,free)(el);
	return NULL;
}

#ifdef HAS_TYPE
/* Create a piecewise expression with the given base expression on a universe
 * domain.
 *
 * Pass along the type as an extra argument for improved uniformity
 * with piecewise types that do not have a fold type.
 */
__isl_give PW *FN(FN(PW,from),BASE)(__isl_take EL *el)
{
	enum isl_fold type = FN(EL,get_type)(el);
	return FN(FN(FN(PW,from),BASE),type)(el, type);
}
#else
__isl_give PW *FN(FN(PW,from),BASE)(__isl_take EL *el)
{
	return FN(FN(FN(PW,from),BASE),type)(el);
}
#endif

const char *FN(PW,get_dim_name)(__isl_keep PW *pw, enum isl_dim_type type,
	unsigned pos)
{
	return pw ? isl_space_get_dim_name(pw->dim, type, pos) : NULL;
}

isl_bool FN(PW,has_dim_id)(__isl_keep PW *pw, enum isl_dim_type type,
	unsigned pos)
{
	return pw ? isl_space_has_dim_id(pw->dim, type, pos) : isl_bool_error;
}

__isl_give isl_id *FN(PW,get_dim_id)(__isl_keep PW *pw, enum isl_dim_type type,
	unsigned pos)
{
	return pw ? isl_space_get_dim_id(pw->dim, type, pos) : NULL;
}

isl_bool FN(PW,has_tuple_name)(__isl_keep PW *pw, enum isl_dim_type type)
{
	return pw ? isl_space_has_tuple_name(pw->dim, type) : isl_bool_error;
}

const char *FN(PW,get_tuple_name)(__isl_keep PW *pw, enum isl_dim_type type)
{
	return pw ? isl_space_get_tuple_name(pw->dim, type) : NULL;
}

isl_bool FN(PW,has_tuple_id)(__isl_keep PW *pw, enum isl_dim_type type)
{
	return pw ? isl_space_has_tuple_id(pw->dim, type) : isl_bool_error;
}

__isl_give isl_id *FN(PW,get_tuple_id)(__isl_keep PW *pw, enum isl_dim_type type)
{
	return pw ? isl_space_get_tuple_id(pw->dim, type) : NULL;
}

isl_bool FN(PW,IS_ZERO)(__isl_keep PW *pw)
{
	if (!pw)
		return isl_bool_error;

	return isl_bool_ok(pw->n == 0);
}

__isl_give PW *FN(PW,realign_domain)(__isl_take PW *pw,
	__isl_take isl_reordering *exp)
{
	int i;

	pw = FN(PW,cow)(pw);
	if (!pw || !exp)
		goto error;

	for (i = 0; i < pw->n; ++i) {
		pw->p[i].set = isl_set_realign(pw->p[i].set,
						    isl_reordering_copy(exp));
		if (!pw->p[i].set)
			goto error;
		pw->p[i].FIELD = FN(EL,realign_domain)(pw->p[i].FIELD,
						    isl_reordering_copy(exp));
		if (!pw->p[i].FIELD)
			goto error;
	}

	pw = FN(PW,reset_domain_space)(pw, isl_reordering_get_space(exp));

	isl_reordering_free(exp);
	return pw;
error:
	isl_reordering_free(exp);
	FN(PW,free)(pw);
	return NULL;
}

#undef TYPE
#define TYPE PW

#include "isl_check_named_params_templ.c"

/* Align the parameters of "pw" to those of "model".
 */
__isl_give PW *FN(PW,align_params)(__isl_take PW *pw, __isl_take isl_space *model)
{
	isl_ctx *ctx;
	isl_bool equal_params;

	if (!pw || !model)
		goto error;

	ctx = isl_space_get_ctx(model);
	if (!isl_space_has_named_params(model))
		isl_die(ctx, isl_error_invalid,
			"model has unnamed parameters", goto error);
	if (FN(PW,check_named_params)(pw) < 0)
		goto error;
	equal_params = isl_space_has_equal_params(pw->dim, model);
	if (equal_params < 0)
		goto error;
	if (!equal_params) {
		isl_reordering *exp;

		exp = isl_parameter_alignment_reordering(pw->dim, model);
		exp = isl_reordering_extend_space(exp,
					FN(PW,get_domain_space)(pw));
		pw = FN(PW,realign_domain)(pw, exp);
	}

	isl_space_free(model);
	return pw;
error:
	isl_space_free(model);
	FN(PW,free)(pw);
	return NULL;
}

#undef TYPE
#define TYPE	PW

static
#include "isl_align_params_bin_templ.c"

#undef SUFFIX
#define SUFFIX	set
#undef ARG1
#define ARG1	PW
#undef ARG2
#define ARG2	isl_set

static
#include "isl_align_params_templ.c"

#undef TYPE
#define TYPE	PW

#include "isl_type_has_equal_space_bin_templ.c"
#include "isl_type_check_equal_space_templ.c"

/* Private version of "union_add".  For isl_pw_qpolynomial and
 * isl_pw_qpolynomial_fold, we prefer to simply call it "add".
 */
static __isl_give PW *FN(PW,union_add_)(__isl_take PW *pw1, __isl_take PW *pw2)
{
	int i, j, n;
	struct PW *res;
	isl_ctx *ctx;
	isl_set *set;

	if (FN(PW,align_params_bin)(&pw1, &pw2) < 0)
		goto error;

	ctx = isl_space_get_ctx(pw1->dim);
	if (!OPT_EQUAL_TYPES(pw1->, pw2->))
		isl_die(ctx, isl_error_invalid,
			"fold types don't match", goto error);
	if (FN(PW,check_equal_space)(pw1, pw2) < 0)
		goto error;

	if (FN(PW,IS_ZERO)(pw1)) {
		FN(PW,free)(pw1);
		return pw2;
	}

	if (FN(PW,IS_ZERO)(pw2)) {
		FN(PW,free)(pw2);
		return pw1;
	}

	n = (pw1->n + 1) * (pw2->n + 1);
	res = FN(PW,alloc_size)(isl_space_copy(pw1->dim)
				OPT_TYPE_ARG(pw1->), n);

	for (i = 0; i < pw1->n; ++i) {
		set = isl_set_copy(pw1->p[i].set);
		for (j = 0; j < pw2->n; ++j) {
			struct isl_set *common;
			EL *sum;
			common = isl_set_intersect(isl_set_copy(pw1->p[i].set),
						isl_set_copy(pw2->p[j].set));
			if (isl_set_plain_is_empty(common)) {
				isl_set_free(common);
				continue;
			}
			set = isl_set_subtract(set,
					isl_set_copy(pw2->p[j].set));

			sum = FN(EL,add_on_domain)(common,
						   FN(EL,copy)(pw1->p[i].FIELD),
						   FN(EL,copy)(pw2->p[j].FIELD));

			res = FN(PW,add_piece)(res, common, sum);
		}
		res = FN(PW,add_piece)(res, set, FN(EL,copy)(pw1->p[i].FIELD));
	}

	for (j = 0; j < pw2->n; ++j) {
		set = isl_set_copy(pw2->p[j].set);
		for (i = 0; i < pw1->n; ++i)
			set = isl_set_subtract(set,
					isl_set_copy(pw1->p[i].set));
		res = FN(PW,add_piece)(res, set, FN(EL,copy)(pw2->p[j].FIELD));
	}

	FN(PW,free)(pw1);
	FN(PW,free)(pw2);

	return res;
error:
	FN(PW,free)(pw1);
	FN(PW,free)(pw2);
	return NULL;
}

/* Make sure "pw" has room for at least "n" more pieces.
 *
 * If there is only one reference to pw, we extend it in place.
 * Otherwise, we create a new PW and copy the pieces.
 */
static __isl_give PW *FN(PW,grow)(__isl_take PW *pw, int n)
{
	int i;
	isl_ctx *ctx;
	PW *res;

	if (!pw)
		return NULL;
	if (pw->n + n <= pw->size)
		return pw;
	ctx = FN(PW,get_ctx)(pw);
	n += pw->n;
	if (pw->ref == 1) {
		res = isl_realloc(ctx, pw, struct PW,
			    sizeof(struct PW) + (n - 1) * sizeof(S(PW,piece)));
		if (!res)
			return FN(PW,free)(pw);
		res->size = n;
		return res;
	}
	res = FN(PW,alloc_size)(isl_space_copy(pw->dim) OPT_TYPE_ARG(pw->), n);
	if (!res)
		return FN(PW,free)(pw);
	for (i = 0; i < pw->n; ++i)
		res = FN(PW,add_piece)(res, isl_set_copy(pw->p[i].set),
					    FN(EL,copy)(pw->p[i].FIELD));
	FN(PW,free)(pw);
	return res;
}

__isl_give PW *FN(PW,add_disjoint)(__isl_take PW *pw1, __isl_take PW *pw2)
{
	int i;
	isl_ctx *ctx;

	if (FN(PW,align_params_bin)(&pw1, &pw2) < 0)
		goto error;

	if (pw1->size < pw1->n + pw2->n && pw1->n < pw2->n)
		return FN(PW,add_disjoint)(pw2, pw1);

	ctx = isl_space_get_ctx(pw1->dim);
	if (!OPT_EQUAL_TYPES(pw1->, pw2->))
		isl_die(ctx, isl_error_invalid,
			"fold types don't match", goto error);
	if (FN(PW,check_equal_space)(pw1, pw2) < 0)
		goto error;

	if (FN(PW,IS_ZERO)(pw1)) {
		FN(PW,free)(pw1);
		return pw2;
	}

	if (FN(PW,IS_ZERO)(pw2)) {
		FN(PW,free)(pw2);
		return pw1;
	}

	pw1 = FN(PW,grow)(pw1, pw2->n);
	if (!pw1)
		goto error;

	for (i = 0; i < pw2->n; ++i)
		pw1 = FN(PW,add_piece)(pw1,
				isl_set_copy(pw2->p[i].set),
				FN(EL,copy)(pw2->p[i].FIELD));

	FN(PW,free)(pw2);

	return pw1;
error:
	FN(PW,free)(pw1);
	FN(PW,free)(pw2);
	return NULL;
}

/* This function is currently only used from isl_aff.c
 */
static __isl_give PW *FN(PW,on_shared_domain_in)(__isl_take PW *pw1,
	__isl_take PW *pw2, __isl_take isl_space *space,
	__isl_give EL *(*fn)(__isl_take EL *el1, __isl_take EL *el2))
	__attribute__ ((unused));

/* Apply "fn" to pairs of elements from pw1 and pw2 on shared domains.
 * The result of "fn" (and therefore also of this function) lives in "space".
 */
static __isl_give PW *FN(PW,on_shared_domain_in)(__isl_take PW *pw1,
	__isl_take PW *pw2, __isl_take isl_space *space,
	__isl_give EL *(*fn)(__isl_take EL *el1, __isl_take EL *el2))
{
	int i, j, n;
	PW *res = NULL;

	if (!pw1 || !pw2)
		goto error;

	n = pw1->n * pw2->n;
	res = FN(PW,alloc_size)(isl_space_copy(space) OPT_TYPE_ARG(pw1->), n);

	for (i = 0; i < pw1->n; ++i) {
		for (j = 0; j < pw2->n; ++j) {
			isl_set *common;
			EL *res_ij;
			int empty;

			common = isl_set_intersect(
					isl_set_copy(pw1->p[i].set),
					isl_set_copy(pw2->p[j].set));
			empty = isl_set_plain_is_empty(common);
			if (empty < 0 || empty) {
				isl_set_free(common);
				if (empty < 0)
					goto error;
				continue;
			}

			res_ij = fn(FN(EL,copy)(pw1->p[i].FIELD),
				    FN(EL,copy)(pw2->p[j].FIELD));
			res_ij = FN(EL,gist)(res_ij, isl_set_copy(common));

			res = FN(PW,add_piece)(res, common, res_ij);
		}
	}

	isl_space_free(space);
	FN(PW,free)(pw1);
	FN(PW,free)(pw2);
	return res;
error:
	isl_space_free(space);
	FN(PW,free)(pw1);
	FN(PW,free)(pw2);
	FN(PW,free)(res);
	return NULL;
}

/* This function is currently only used from isl_aff.c
 */
static __isl_give PW *FN(PW,on_shared_domain)(__isl_take PW *pw1,
	__isl_take PW *pw2,
	__isl_give EL *(*fn)(__isl_take EL *el1, __isl_take EL *el2))
	__attribute__ ((unused));

/* Apply "fn" to pairs of elements from pw1 and pw2 on shared domains.
 * The result of "fn" is assumed to live in the same space as "pw1" and "pw2".
 */
static __isl_give PW *FN(PW,on_shared_domain)(__isl_take PW *pw1,
	__isl_take PW *pw2,
	__isl_give EL *(*fn)(__isl_take EL *el1, __isl_take EL *el2))
{
	isl_space *space;

	if (FN(PW,check_equal_space)(pw1, pw2) < 0)
		goto error;

	space = isl_space_copy(pw1->dim);
	return FN(PW,on_shared_domain_in)(pw1, pw2, space, fn);
error:
	FN(PW,free)(pw1);
	FN(PW,free)(pw2);
	return NULL;
}

/* Return the parameter domain of "pw".
 */
__isl_give isl_set *FN(PW,params)(__isl_take PW *pw)
{
	return isl_set_params(FN(PW,domain)(pw));
}

__isl_give isl_set *FN(PW,domain)(__isl_take PW *pw)
{
	int i;
	isl_set *dom;

	if (!pw)
		return NULL;

	dom = isl_set_empty(FN(PW,get_domain_space)(pw));
	for (i = 0; i < pw->n; ++i)
		dom = isl_set_union_disjoint(dom, isl_set_copy(pw->p[i].set));

	FN(PW,free)(pw);

	return dom;
}

/* Exploit the equalities in the domain of piece "i" of "pw"
 * to simplify the associated function.
 * If the domain of piece "i" is empty, then remove it entirely,
 * replacing it with the final piece.
 */
static int FN(PW,exploit_equalities_and_remove_if_empty)(__isl_keep PW *pw,
	int i)
{
	isl_basic_set *aff;
	int empty = isl_set_plain_is_empty(pw->p[i].set);

	if (empty < 0)
		return -1;
	if (empty) {
		isl_set_free(pw->p[i].set);
		FN(EL,free)(pw->p[i].FIELD);
		if (i != pw->n - 1)
			pw->p[i] = pw->p[pw->n - 1];
		pw->n--;

		return 0;
	}

	aff = isl_set_affine_hull(isl_set_copy(pw->p[i].set));
	pw->p[i].FIELD = FN(EL,substitute_equalities)(pw->p[i].FIELD, aff);
	if (!pw->p[i].FIELD)
		return -1;

	return 0;
}

/* Convert a piecewise expression defined over a parameter domain
 * into one that is defined over a zero-dimensional set.
 */
__isl_give PW *FN(PW,from_range)(__isl_take PW *pw)
{
	isl_space *space;

	if (!pw)
		return NULL;
	if (!isl_space_is_set(pw->dim))
		isl_die(FN(PW,get_ctx)(pw), isl_error_invalid,
			"not living in a set space", return FN(PW,free)(pw));

	space = FN(PW,get_space)(pw);
	space = isl_space_from_range(space);
	pw = FN(PW,reset_space)(pw, space);

	return pw;
}

/* Fix the value of the given parameter or domain dimension of "pw"
 * to be equal to "value".
 */
__isl_give PW *FN(PW,fix_si)(__isl_take PW *pw, enum isl_dim_type type,
	unsigned pos, int value)
{
	int i;

	if (!pw)
		return NULL;

	if (type == isl_dim_out)
		isl_die(FN(PW,get_ctx)(pw), isl_error_invalid,
			"cannot fix output dimension", return FN(PW,free)(pw));

	if (pw->n == 0)
		return pw;

	if (type == isl_dim_in)
		type = isl_dim_set;

	pw = FN(PW,cow)(pw);
	if (!pw)
		return FN(PW,free)(pw);

	for (i = pw->n - 1; i >= 0; --i) {
		pw->p[i].set = isl_set_fix_si(pw->p[i].set, type, pos, value);
		if (FN(PW,exploit_equalities_and_remove_if_empty)(pw, i) < 0)
			return FN(PW,free)(pw);
	}

	return pw;
}

/* Restrict the domain of "pw" by combining each cell
 * with "set" through a call to "fn", where "fn" may be
 * isl_set_intersect, isl_set_intersect_params, isl_set_intersect_factor_domain,
 * isl_set_intersect_factor_range or isl_set_subtract.
 */
static __isl_give PW *FN(PW,restrict_domain_aligned)(__isl_take PW *pw,
	__isl_take isl_set *set,
	__isl_give isl_set *(*fn)(__isl_take isl_set *set1,
				    __isl_take isl_set *set2))
{
	int i;

	if (!pw || !set)
		goto error;

	if (pw->n == 0) {
		isl_set_free(set);
		return pw;
	}

	pw = FN(PW,cow)(pw);
	if (!pw)
		goto error;

	for (i = pw->n - 1; i >= 0; --i) {
		pw->p[i].set = fn(pw->p[i].set, isl_set_copy(set));
		if (FN(PW,exploit_equalities_and_remove_if_empty)(pw, i) < 0)
			goto error;
	}
	
	isl_set_free(set);
	return pw;
error:
	isl_set_free(set);
	FN(PW,free)(pw);
	return NULL;
}

__isl_give PW *FN(PW,intersect_domain)(__isl_take PW *pw,
	__isl_take isl_set *context)
{
	FN(PW,align_params_set)(&pw, &context);
	return FN(PW,restrict_domain_aligned)(pw, context, &isl_set_intersect);
}

/* Intersect the domain of "pw" with the parameter domain "context".
 */
__isl_give PW *FN(PW,intersect_params)(__isl_take PW *pw,
	__isl_take isl_set *context)
{
	FN(PW,align_params_set)(&pw, &context);
	return FN(PW,restrict_domain_aligned)(pw, context,
					&isl_set_intersect_params);
}

/* Given a piecewise expression "pw" with domain in a space [A -> B] and
 * a set in the space A, intersect the domain with the set.
 */
__isl_give PW *FN(PW,intersect_domain_wrapped_domain)(__isl_take PW *pw,
	__isl_take isl_set *set)
{
	FN(PW,align_params_set)(&pw, &set);
	return FN(PW,restrict_domain_aligned)(pw, set,
					    &isl_set_intersect_factor_domain);
}

/* Given a piecewise expression "pw" with domain in a space [A -> B] and
 * a set in the space B, intersect the domain with the set.
 */
__isl_give PW *FN(PW,intersect_domain_wrapped_range)(__isl_take PW *pw,
	__isl_take isl_set *set)
{
	FN(PW,align_params_set)(&pw, &set);
	return FN(PW,restrict_domain_aligned)(pw, set,
					    &isl_set_intersect_factor_range);
}

/* Subtract "domain' from the domain of "pw".
 */
__isl_give PW *FN(PW,subtract_domain)(__isl_take PW *pw,
	__isl_take isl_set *domain)
{
	FN(PW,align_params_set)(&pw, &domain);
	return FN(PW,restrict_domain_aligned)(pw, domain, &isl_set_subtract);
}

/* Compute the gist of "pw" with respect to the domain constraints
 * of "context" for the case where the domain of the last element
 * of "pw" is equal to "context".
 * Call "fn_el" to compute the gist of this element, replace
 * its domain by the universe and drop all other elements
 * as their domains are necessarily disjoint from "context".
 */
static __isl_give PW *FN(PW,gist_last)(__isl_take PW *pw,
	__isl_take isl_set *context,
	__isl_give EL *(*fn_el)(__isl_take EL *el, __isl_take isl_set *set))
{
	int i;
	isl_space *space;

	for (i = 0; i < pw->n - 1; ++i) {
		isl_set_free(pw->p[i].set);
		FN(EL,free)(pw->p[i].FIELD);
	}
	pw->p[0].FIELD = pw->p[pw->n - 1].FIELD;
	pw->p[0].set = pw->p[pw->n - 1].set;
	pw->n = 1;

	space = isl_set_get_space(context);
	pw->p[0].FIELD = fn_el(pw->p[0].FIELD, context);
	context = isl_set_universe(space);
	isl_set_free(pw->p[0].set);
	pw->p[0].set = context;

	if (!pw->p[0].FIELD || !pw->p[0].set)
		return FN(PW,free)(pw);

	return pw;
}

/* Compute the gist of "pw" with respect to the domain constraints
 * of "context".  Call "fn_el" to compute the gist of the elements
 * and "fn_dom" to compute the gist of the domains.
 *
 * If the piecewise expression is empty or the context is the universe,
 * then nothing can be simplified.
 */
static __isl_give PW *FN(PW,gist_aligned)(__isl_take PW *pw,
	__isl_take isl_set *context,
	__isl_give EL *(*fn_el)(__isl_take EL *el,
				    __isl_take isl_set *set),
	__isl_give isl_set *(*fn_dom)(__isl_take isl_set *set,
				    __isl_take isl_basic_set *bset))
{
	int i;
	int is_universe;
	isl_bool aligned;
	isl_basic_set *hull = NULL;

	if (!pw || !context)
		goto error;

	if (pw->n == 0) {
		isl_set_free(context);
		return pw;
	}

	is_universe = isl_set_plain_is_universe(context);
	if (is_universe < 0)
		goto error;
	if (is_universe) {
		isl_set_free(context);
		return pw;
	}

	aligned = isl_set_space_has_equal_params(context, pw->dim);
	if (aligned < 0)
		goto error;
	if (!aligned) {
		pw = FN(PW,align_params)(pw, isl_set_get_space(context));
		context = isl_set_align_params(context, FN(PW,get_space)(pw));
	}

	pw = FN(PW,cow)(pw);
	if (!pw)
		goto error;

	if (pw->n == 1) {
		int equal;

		equal = isl_set_plain_is_equal(pw->p[0].set, context);
		if (equal < 0)
			goto error;
		if (equal)
			return FN(PW,gist_last)(pw, context, fn_el);
	}

	context = isl_set_compute_divs(context);
	hull = isl_set_simple_hull(isl_set_copy(context));

	for (i = pw->n - 1; i >= 0; --i) {
		isl_set *set_i;
		int empty;

		if (i == pw->n - 1) {
			int equal;
			equal = isl_set_plain_is_equal(pw->p[i].set, context);
			if (equal < 0)
				goto error;
			if (equal) {
				isl_basic_set_free(hull);
				return FN(PW,gist_last)(pw, context, fn_el);
			}
		}
		set_i = isl_set_intersect(isl_set_copy(pw->p[i].set),
						 isl_set_copy(context));
		empty = isl_set_plain_is_empty(set_i);
		pw->p[i].FIELD = fn_el(pw->p[i].FIELD, set_i);
		pw->p[i].set = fn_dom(pw->p[i].set, isl_basic_set_copy(hull));
		if (empty < 0 || !pw->p[i].FIELD || !pw->p[i].set)
			goto error;
		if (empty) {
			isl_set_free(pw->p[i].set);
			FN(EL,free)(pw->p[i].FIELD);
			if (i != pw->n - 1)
				pw->p[i] = pw->p[pw->n - 1];
			pw->n--;
		}
	}

	isl_basic_set_free(hull);
	isl_set_free(context);

	return pw;
error:
	FN(PW,free)(pw);
	isl_basic_set_free(hull);
	isl_set_free(context);
	return NULL;
}

__isl_give PW *FN(PW,gist)(__isl_take PW *pw, __isl_take isl_set *context)
{
	FN(PW,align_params_set)(&pw, &context);
	return FN(PW,gist_aligned)(pw, context, &FN(EL,gist),
					&isl_set_gist_basic_set);
}

__isl_give PW *FN(PW,gist_params)(__isl_take PW *pw,
	__isl_take isl_set *context)
{
	FN(PW,align_params_set)(&pw, &context);
	return FN(PW,gist_aligned)(pw, context, &FN(EL,gist_params),
					&isl_set_gist_params_basic_set);
}

/* Return -1 if the piece "p1" should be sorted before "p2"
 * and 1 if it should be sorted after "p2".
 * Return 0 if they do not need to be sorted in a specific order.
 *
 * The two pieces are compared on the basis of their function value expressions.
 */
static int FN(PW,sort_field_cmp)(const void *p1, const void *p2, void *arg)
{
	struct FN(PW,piece) const *pc1 = p1;
	struct FN(PW,piece) const *pc2 = p2;

	return FN(EL,plain_cmp)(pc1->FIELD, pc2->FIELD);
}

/* Sort the pieces of "pw" according to their function value
 * expressions and then combine pairs of adjacent pieces with
 * the same such expression.
 *
 * The sorting is performed in place because it does not
 * change the meaning of "pw", but care needs to be
 * taken not to change any possible other copies of "pw"
 * in case anything goes wrong.
 */
__isl_give PW *FN(PW,sort)(__isl_take PW *pw)
{
	int i, j;
	isl_set *set;

	if (!pw)
		return NULL;
	if (pw->n <= 1)
		return pw;
	if (isl_sort(pw->p, pw->n, sizeof(pw->p[0]),
		    &FN(PW,sort_field_cmp), NULL) < 0)
		return FN(PW,free)(pw);
	for (i = pw->n - 1; i >= 1; --i) {
		if (!FN(EL,plain_is_equal)(pw->p[i - 1].FIELD, pw->p[i].FIELD))
			continue;
		set = isl_set_union(isl_set_copy(pw->p[i - 1].set),
				    isl_set_copy(pw->p[i].set));
		if (!set)
			return FN(PW,free)(pw);
		isl_set_free(pw->p[i].set);
		FN(EL,free)(pw->p[i].FIELD);
		isl_set_free(pw->p[i - 1].set);
		pw->p[i - 1].set = set;
		for (j = i + 1; j < pw->n; ++j)
			pw->p[j - 1] = pw->p[j];
		pw->n--;
	}

	return pw;
}

/* Coalesce the domains of "pw".
 *
 * Prior to the actual coalescing, first sort the pieces such that
 * pieces with the same function value expression are combined
 * into a single piece, the combined domain of which can then
 * be coalesced.
 */
__isl_give PW *FN(PW,coalesce)(__isl_take PW *pw)
{
	int i;

	pw = FN(PW,sort)(pw);
	if (!pw)
		return NULL;

	for (i = 0; i < pw->n; ++i) {
		pw->p[i].set = isl_set_coalesce(pw->p[i].set);
		if (!pw->p[i].set)
			goto error;
	}

	return pw;
error:
	FN(PW,free)(pw);
	return NULL;
}

isl_ctx *FN(PW,get_ctx)(__isl_keep PW *pw)
{
	return pw ? isl_space_get_ctx(pw->dim) : NULL;
}

isl_bool FN(PW,involves_dims)(__isl_keep PW *pw, enum isl_dim_type type,
	unsigned first, unsigned n)
{
	int i;
	enum isl_dim_type set_type;

	if (!pw)
		return isl_bool_error;
	if (pw->n == 0 || n == 0)
		return isl_bool_false;

	set_type = type == isl_dim_in ? isl_dim_set : type;

	for (i = 0; i < pw->n; ++i) {
		isl_bool involves = FN(EL,involves_dims)(pw->p[i].FIELD,
							type, first, n);
		if (involves < 0 || involves)
			return involves;
		involves = isl_set_involves_dims(pw->p[i].set,
							set_type, first, n);
		if (involves < 0 || involves)
			return involves;
	}
	return isl_bool_false;
}

__isl_give PW *FN(PW,set_dim_name)(__isl_take PW *pw,
	enum isl_dim_type type, unsigned pos, const char *s)
{
	int i;
	enum isl_dim_type set_type;

	pw = FN(PW,cow)(pw);
	if (!pw)
		return NULL;

	set_type = type == isl_dim_in ? isl_dim_set : type;

	pw->dim = isl_space_set_dim_name(pw->dim, type, pos, s);
	if (!pw->dim)
		goto error;

	for (i = 0; i < pw->n; ++i) {
		pw->p[i].set = isl_set_set_dim_name(pw->p[i].set,
							set_type, pos, s);
		if (!pw->p[i].set)
			goto error;
		pw->p[i].FIELD = FN(EL,set_dim_name)(pw->p[i].FIELD, type, pos, s);
		if (!pw->p[i].FIELD)
			goto error;
	}

	return pw;
error:
	FN(PW,free)(pw);
	return NULL;
}

__isl_give PW *FN(PW,drop_dims)(__isl_take PW *pw,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	int i;
	enum isl_dim_type set_type;

	if (!pw)
		return NULL;
	if (n == 0 && !isl_space_get_tuple_name(pw->dim, type))
		return pw;

	set_type = type == isl_dim_in ? isl_dim_set : type;

	pw = FN(PW,cow)(pw);
	if (!pw)
		return NULL;
	pw->dim = isl_space_drop_dims(pw->dim, type, first, n);
	if (!pw->dim)
		goto error;
	for (i = 0; i < pw->n; ++i) {
		pw->p[i].FIELD = FN(EL,drop_dims)(pw->p[i].FIELD, type, first, n);
		if (!pw->p[i].FIELD)
			goto error;
		if (type == isl_dim_out)
			continue;
		pw->p[i].set = isl_set_drop(pw->p[i].set, set_type, first, n);
		if (!pw->p[i].set)
			goto error;
	}

	return pw;
error:
	FN(PW,free)(pw);
	return NULL;
}

/* This function is very similar to drop_dims.
 * The only difference is that the cells may still involve
 * the specified dimensions.  They are removed using
 * isl_set_project_out instead of isl_set_drop.
 */
__isl_give PW *FN(PW,project_out)(__isl_take PW *pw,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	int i;
	enum isl_dim_type set_type;

	if (!pw)
		return NULL;
	if (n == 0 && !isl_space_get_tuple_name(pw->dim, type))
		return pw;

	set_type = type == isl_dim_in ? isl_dim_set : type;

	pw = FN(PW,cow)(pw);
	if (!pw)
		return NULL;
	pw->dim = isl_space_drop_dims(pw->dim, type, first, n);
	if (!pw->dim)
		goto error;
	for (i = 0; i < pw->n; ++i) {
		pw->p[i].set = isl_set_project_out(pw->p[i].set,
							set_type, first, n);
		if (!pw->p[i].set)
			goto error;
		pw->p[i].FIELD = FN(EL,drop_dims)(pw->p[i].FIELD, type, first, n);
		if (!pw->p[i].FIELD)
			goto error;
	}

	return pw;
error:
	FN(PW,free)(pw);
	return NULL;
}

/* Project the domain of pw onto its parameter space.
 */
__isl_give PW *FN(PW,project_domain_on_params)(__isl_take PW *pw)
{
	isl_space *space;
	isl_size n;

	n = FN(PW,dim)(pw, isl_dim_in);
	if (n < 0)
		return FN(PW,free)(pw);
	pw = FN(PW,project_out)(pw, isl_dim_in, 0, n);
	space = FN(PW,get_domain_space)(pw);
	space = isl_space_params(space);
	pw = FN(PW,reset_domain_space)(pw, space);
	return pw;
}

/* Drop all parameters not referenced by "pw".
 */
__isl_give PW *FN(PW,drop_unused_params)(__isl_take PW *pw)
{
	isl_size n;
	int i;

	if (FN(PW,check_named_params)(pw) < 0)
		return FN(PW,free)(pw);

	n = FN(PW,dim)(pw, isl_dim_param);
	if (n < 0)
		return FN(PW,free)(pw);
	for (i = n - 1; i >= 0; i--) {
		isl_bool involves;

		involves = FN(PW,involves_dims)(pw, isl_dim_param, i, 1);
		if (involves < 0)
			return FN(PW,free)(pw);
		if (!involves)
			pw = FN(PW,drop_dims)(pw, isl_dim_param, i, 1);
	}

	return pw;
}

__isl_give PW *FN(PW,fix_dim)(__isl_take PW *pw,
	enum isl_dim_type type, unsigned pos, isl_int v)
{
	int i;

	if (!pw)
		return NULL;

	if (type == isl_dim_in)
		type = isl_dim_set;

	pw = FN(PW,cow)(pw);
	if (!pw)
		return NULL;
	for (i = 0; i < pw->n; ++i) {
		pw->p[i].set = isl_set_fix(pw->p[i].set, type, pos, v);
		if (FN(PW,exploit_equalities_and_remove_if_empty)(pw, i) < 0)
			return FN(PW,free)(pw);
	}

	return pw;
}

/* Fix the value of the variable at position "pos" of type "type" of "pw"
 * to be equal to "v".
 */
__isl_give PW *FN(PW,fix_val)(__isl_take PW *pw,
	enum isl_dim_type type, unsigned pos, __isl_take isl_val *v)
{
	if (!v)
		return FN(PW,free)(pw);
	if (!isl_val_is_int(v))
		isl_die(FN(PW,get_ctx)(pw), isl_error_invalid,
			"expecting integer value", goto error);

	pw = FN(PW,fix_dim)(pw, type, pos, v->n);
	isl_val_free(v);

	return pw;
error:
	isl_val_free(v);
	return FN(PW,free)(pw);
}

isl_size FN(PW,dim)(__isl_keep PW *pw, enum isl_dim_type type)
{
	return isl_space_dim(FN(PW,peek_space)(pw), type);
}

__isl_give PW *FN(PW,split_dims)(__isl_take PW *pw,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	int i;

	if (!pw)
		return NULL;
	if (n == 0)
		return pw;

	if (type == isl_dim_in)
		type = isl_dim_set;

	pw = FN(PW,cow)(pw);
	if (!pw)
		return NULL;
	if (!pw->dim)
		goto error;
	for (i = 0; i < pw->n; ++i) {
		pw->p[i].set = isl_set_split_dims(pw->p[i].set, type, first, n);
		if (!pw->p[i].set)
			goto error;
	}

	return pw;
error:
	FN(PW,free)(pw);
	return NULL;
}

/* Return the space of "pw".
 */
__isl_keep isl_space *FN(PW,peek_space)(__isl_keep PW *pw)
{
	return pw ? pw->dim : NULL;
}

__isl_give isl_space *FN(PW,get_space)(__isl_keep PW *pw)
{
	return isl_space_copy(FN(PW,peek_space)(pw));
}

/* Return the space of "pw".
 * This may be either a copy or the space itself
 * if there is only one reference to "pw".
 * This allows the space to be modified inplace
 * if both the piecewise expression and its space have only a single reference.
 * The caller is not allowed to modify "pw" between this call and
 * a subsequent call to isl_pw_*_restore_*.
 * The only exception is that isl_pw_*_free can be called instead.
 */
__isl_give isl_space *FN(PW,take_space)(__isl_keep PW *pw)
{
	isl_space *space;

	if (!pw)
		return NULL;
	if (pw->ref != 1)
		return FN(PW,get_space)(pw);
	space = pw->dim;
	pw->dim = NULL;
	return space;
}

/* Set the space of "pw" to "space", where the space of "pw" may be missing
 * due to a preceding call to isl_pw_*_take_space.
 * However, in this case, "pw" only has a single reference and
 * then the call to isl_pw_*_cow has no effect.
 */
__isl_give PW *FN(PW,restore_space)(__isl_take PW *pw,
	__isl_take isl_space *space)
{
	if (!pw || !space)
		goto error;

	if (pw->dim == space) {
		isl_space_free(space);
		return pw;
	}

	pw = FN(PW,cow)(pw);
	if (!pw)
		goto error;
	isl_space_free(pw->dim);
	pw->dim = space;

	return pw;
error:
	FN(PW,free)(pw);
	isl_space_free(space);
	return NULL;
}

/* Check that "pos" is a valid position for a cell in "pw".
 */
static isl_stat FN(PW,check_pos)(__isl_keep PW *pw, int pos)
{
	if (!pw)
		return isl_stat_error;
	if (pos < 0 || pos >= pw->n)
		isl_die(FN(PW,get_ctx)(pw), isl_error_internal,
			"position out of bounds", return isl_stat_error);
	return isl_stat_ok;
}

/* Return the cell at position "pos" in "pw".
 */
static __isl_keep isl_set *FN(PW,peek_domain_at)(__isl_keep PW *pw, int pos)
{
	if (FN(PW,check_pos)(pw, pos) < 0)
		return NULL;
	return pw->p[pos].set;
}

/* Return a copy of the cell at position "pos" in "pw".
 */
__isl_give isl_set *FN(PW,get_domain_at)(__isl_keep PW *pw, int pos)
{
	return isl_set_copy(FN(PW,peek_domain_at)(pw, pos));
}

/* Return the base expression associated to
 * the cell at position "pos" in "pw".
 */
static __isl_keep EL *FN(PW,peek_base_at)(__isl_keep PW *pw, int pos)
{
	if (FN(PW,check_pos)(pw, pos) < 0)
		return NULL;
	return pw->p[pos].FIELD;
}

/* Return a copy of the base expression associated to
 * the cell at position "pos" in "pw".
 */
__isl_give EL *FN(PW,get_base_at)(__isl_keep PW *pw, int pos)
{
	return FN(EL,copy)(FN(PW,peek_base_at)(pw, pos));
}

/* Return the base expression associated to
 * the cell at position "pos" in "pw".
 * This may be either a copy or the base expression itself
 * if there is only one reference to "pw".
 * This allows the base expression to be modified inplace
 * if both the piecewise expression and this base expression
 * have only a single reference.
 * The caller is not allowed to modify "pw" between this call and
 * a subsequent call to isl_pw_*_restore_*.
 * The only exception is that isl_pw_*_free can be called instead.
 */
__isl_give EL *FN(PW,take_base_at)(__isl_keep PW *pw, int pos)
{
	EL *el;

	if (!pw)
		return NULL;
	if (pw->ref != 1)
		return FN(PW,get_base_at)(pw, pos);
	if (FN(PW,check_pos)(pw, pos) < 0)
		return NULL;
	el = pw->p[pos].FIELD;
	pw->p[pos].FIELD = NULL;
	return el;
}

/* Set the base expression associated to
 * the cell at position "pos" in "pw" to "el",
 * where this base expression may be missing
 * due to a preceding call to isl_pw_*_take_base_at.
 * However, in this case, "pw" only has a single reference and
 * then the call to isl_pw_*_cow has no effect.
 */
__isl_give PW *FN(PW,restore_base_at)(__isl_take PW *pw, int pos,
	__isl_take EL *el)
{
	if (FN(PW,check_pos)(pw, pos) < 0 || !el)
		goto error;

	if (pw->p[pos].FIELD == el) {
		FN(EL,free)(el);
		return pw;
	}

	pw = FN(PW,cow)(pw);
	if (!pw)
		goto error;
	FN(EL,free)(pw->p[pos].FIELD);
	pw->p[pos].FIELD = el;

	return pw;
error:
	FN(PW,free)(pw);
	FN(EL,free)(el);
	return NULL;
}

__isl_give isl_space *FN(PW,get_domain_space)(__isl_keep PW *pw)
{
	return pw ? isl_space_domain(isl_space_copy(pw->dim)) : NULL;
}

/* Return the position of the dimension of the given type and name
 * in "pw".
 * Return -1 if no such dimension can be found.
 */
int FN(PW,find_dim_by_name)(__isl_keep PW *pw,
	enum isl_dim_type type, const char *name)
{
	if (!pw)
		return -1;
	return isl_space_find_dim_by_name(pw->dim, type, name);
}

/* Return the position of the dimension of the given type and identifier
 * in "pw".
 * Return -1 if no such dimension can be found.
 */
static int FN(PW,find_dim_by_id)(__isl_keep PW *pw,
	enum isl_dim_type type, __isl_keep isl_id *id)
{
	isl_space *space;

	space = FN(PW,peek_space)(pw);
	return isl_space_find_dim_by_id(space, type, id);
}

/* Does the piecewise expression "pw" depend in any way
 * on the parameter with identifier "id"?
 */
isl_bool FN(PW,involves_param_id)(__isl_keep PW *pw, __isl_keep isl_id *id)
{
	int pos;

	if (!pw || !id)
		return isl_bool_error;
	if (pw->n == 0)
		return isl_bool_false;

	pos = FN(PW,find_dim_by_id)(pw, isl_dim_param, id);
	if (pos < 0)
		return isl_bool_false;
	return FN(PW,involves_dims)(pw, isl_dim_param, pos, 1);
}

/* Reset the space of "pw".  Since we don't know if the elements
 * represent the spaces themselves or their domains, we pass along
 * both when we call their reset_space_and_domain.
 */
static __isl_give PW *FN(PW,reset_space_and_domain)(__isl_take PW *pw,
	__isl_take isl_space *space, __isl_take isl_space *domain)
{
	int i;

	pw = FN(PW,cow)(pw);
	if (!pw || !space || !domain)
		goto error;

	for (i = 0; i < pw->n; ++i) {
		pw->p[i].set = isl_set_reset_space(pw->p[i].set,
						 isl_space_copy(domain));
		if (!pw->p[i].set)
			goto error;
		pw->p[i].FIELD = FN(EL,reset_space_and_domain)(pw->p[i].FIELD,
			      isl_space_copy(space), isl_space_copy(domain));
		if (!pw->p[i].FIELD)
			goto error;
	}

	isl_space_free(domain);

	isl_space_free(pw->dim);
	pw->dim = space;

	return pw;
error:
	isl_space_free(domain);
	isl_space_free(space);
	FN(PW,free)(pw);
	return NULL;
}

__isl_give PW *FN(PW,reset_domain_space)(__isl_take PW *pw,
	__isl_take isl_space *domain)
{
	isl_space *space;

	space = isl_space_extend_domain_with_range(isl_space_copy(domain),
						   FN(PW,get_space)(pw));
	return FN(PW,reset_space_and_domain)(pw, space, domain);
}

__isl_give PW *FN(PW,reset_space)(__isl_take PW *pw,
	__isl_take isl_space *space)
{
	isl_space *domain;

	domain = isl_space_domain(isl_space_copy(space));
	return FN(PW,reset_space_and_domain)(pw, space, domain);
}

__isl_give PW *FN(PW,set_tuple_id)(__isl_take PW *pw, enum isl_dim_type type,
	__isl_take isl_id *id)
{
	isl_space *space;

	pw = FN(PW,cow)(pw);
	if (!pw)
		goto error;

	space = FN(PW,get_space)(pw);
	space = isl_space_set_tuple_id(space, type, id);

	return FN(PW,reset_space)(pw, space);
error:
	isl_id_free(id);
	return FN(PW,free)(pw);
}

/* Drop the id on the specified tuple.
 */
__isl_give PW *FN(PW,reset_tuple_id)(__isl_take PW *pw, enum isl_dim_type type)
{
	isl_space *space;

	if (!pw)
		return NULL;
	if (!FN(PW,has_tuple_id)(pw, type))
		return pw;

	pw = FN(PW,cow)(pw);
	if (!pw)
		return NULL;

	space = FN(PW,get_space)(pw);
	space = isl_space_reset_tuple_id(space, type);

	return FN(PW,reset_space)(pw, space);
}

__isl_give PW *FN(PW,set_dim_id)(__isl_take PW *pw,
	enum isl_dim_type type, unsigned pos, __isl_take isl_id *id)
{
	pw = FN(PW,cow)(pw);
	if (!pw)
		goto error;
	pw->dim = isl_space_set_dim_id(pw->dim, type, pos, id);
	return FN(PW,reset_space)(pw, isl_space_copy(pw->dim));
error:
	isl_id_free(id);
	return FN(PW,free)(pw);
}

/* Reset the user pointer on all identifiers of parameters and tuples
 * of the space of "pw".
 */
__isl_give PW *FN(PW,reset_user)(__isl_take PW *pw)
{
	isl_space *space;

	space = FN(PW,get_space)(pw);
	space = isl_space_reset_user(space);

	return FN(PW,reset_space)(pw, space);
}

isl_size FN(PW,n_piece)(__isl_keep PW *pw)
{
	return pw ? pw->n : isl_size_error;
}

isl_stat FN(PW,foreach_piece)(__isl_keep PW *pw,
	isl_stat (*fn)(__isl_take isl_set *set, __isl_take EL *el, void *user),
	void *user)
{
	int i;

	if (!pw)
		return isl_stat_error;

	for (i = 0; i < pw->n; ++i)
		if (fn(isl_set_copy(pw->p[i].set),
				FN(EL,copy)(pw->p[i].FIELD), user) < 0)
			return isl_stat_error;

	return isl_stat_ok;
}

/* Does "test" succeed on every cell of "pw"?
 */
isl_bool FN(PW,every_piece)(__isl_keep PW *pw,
	isl_bool (*test)(__isl_keep isl_set *set,
		__isl_keep EL *el, void *user), void *user)
{
	int i;

	if (!pw)
		return isl_bool_error;

	for (i = 0; i < pw->n; ++i) {
		isl_bool r;

		r = test(pw->p[i].set, pw->p[i].FIELD, user);
		if (r < 0 || !r)
			return r;
	}

	return isl_bool_true;
}

/* Is "pw" defined over a single universe domain?
 *
 * If the default value of this piecewise type is zero,
 * then a "pw" with a zero number of cells is also accepted
 * as it represents the default zero value.
 */
isl_bool FN(FN(PW,isa),BASE)(__isl_keep PW *pw)
{
	isl_size n;

	n = FN(PW,n_piece)(pw);
	if (n < 0)
		return isl_bool_error;
	if (DEFAULT_IS_ZERO && n == 0)
		return isl_bool_true;
	if (n != 1)
		return isl_bool_false;
	return isl_set_plain_is_universe(FN(PW,peek_domain_at)(pw, 0));
}

/* Return a zero base expression in the same space (and of the same type)
 * as "pw".
 */
static __isl_give EL *FN(EL,zero_like_type)(__isl_take PW *pw OPT_TYPE_PARAM)
{
	isl_space *space;

	space = FN(PW,get_space)(pw);
	FN(PW,free)(pw);
	return FN(EL,zero_in_space)(space OPT_TYPE_ARG(NO_LOC));
}

#ifndef HAS_TYPE
/* Return a zero base expression in the same space as "pw".
 */
static __isl_give EL *FN(EL,zero_like)(__isl_take PW *pw)
{
	return FN(EL,zero_like_type)(pw);
}
#else
/* Return a zero base expression in the same space and of the same type
 * as "pw".
 *
 * Pass along the type as an explicit argument for uniform handling
 * in isl_*_zero_like_type.
 */
static __isl_give EL *FN(EL,zero_like)(__isl_take PW *pw)
{
	enum isl_fold type;

	type = FN(PW,get_type)(pw);
	if (type < 0)
		goto error;
	return FN(EL,zero_like_type)(pw, type);
error:
	FN(PW,free)(pw);
	return NULL;
}
#endif

/* Given that "pw" is defined over a single universe domain,
 * return the base expression associated to this domain.
 *
 * If the number of cells is zero, then "pw" is of a piecewise type
 * with a default zero value and effectively represents zero.
 * In this case, create a zero base expression in the same space
 * (and with the same type).
 * Otherwise, simply extract the associated base expression.
 */
__isl_give EL *FN(FN(PW,as),BASE)(__isl_take PW *pw)
{
	isl_bool is_total;
	isl_size n;
	EL *el;

	is_total = FN(FN(PW,isa),BASE)(pw);
	if (is_total < 0)
		goto error;
	if (!is_total)
		isl_die(FN(PW,get_ctx)(pw), isl_error_invalid,
			"expecting single total function", goto error);
	n = FN(PW,n_piece)(pw);
	if (n < 0)
		goto error;
	if (n == 0)
		return FN(EL,zero_like)(pw);
	el = FN(PW,take_base_at)(pw, 0);
	FN(PW,free)(pw);
	return el;
error:
	FN(PW,free)(pw);
	return NULL;
}

#ifdef HAS_TYPE
/* Negate the type of "pw".
 */
static __isl_give PW *FN(PW,negate_type)(__isl_take PW *pw)
{
	pw = FN(PW,cow)(pw);
	if (!pw)
		return NULL;
	pw->type = isl_fold_type_negate(pw->type);
	return pw;
}
#else
/* Negate the type of "pw".
 * Since "pw" does not have a type, do nothing.
 */
static __isl_give PW *FN(PW,negate_type)(__isl_take PW *pw)
{
	return pw;
}
#endif

__isl_give PW *FN(PW,mul_isl_int)(__isl_take PW *pw, isl_int v)
{
	int i;

	if (isl_int_is_one(v))
		return pw;
	if (pw && DEFAULT_IS_ZERO && isl_int_is_zero(v)) {
		PW *zero;
		isl_space *space = FN(PW,get_space)(pw);
		zero = FN(PW,ZERO)(space OPT_TYPE_ARG(pw->));
		FN(PW,free)(pw);
		return zero;
	}
	pw = FN(PW,cow)(pw);
	if (isl_int_is_neg(v))
		pw = FN(PW,negate_type)(pw);
	if (!pw)
		return NULL;
	if (pw->n == 0)
		return pw;

	for (i = 0; i < pw->n; ++i) {
		pw->p[i].FIELD = FN(EL,scale)(pw->p[i].FIELD, v);
		if (!pw->p[i].FIELD)
			goto error;
	}

	return pw;
error:
	FN(PW,free)(pw);
	return NULL;
}

/* Multiply the pieces of "pw" by "v" and return the result.
 */
__isl_give PW *FN(PW,scale_val)(__isl_take PW *pw, __isl_take isl_val *v)
{
	int i;

	if (!pw || !v)
		goto error;

	if (isl_val_is_one(v)) {
		isl_val_free(v);
		return pw;
	}
	if (pw && DEFAULT_IS_ZERO && isl_val_is_zero(v)) {
		PW *zero;
		isl_space *space = FN(PW,get_space)(pw);
		zero = FN(PW,ZERO)(space OPT_TYPE_ARG(pw->));
		FN(PW,free)(pw);
		isl_val_free(v);
		return zero;
	}
	if (pw->n == 0) {
		isl_val_free(v);
		return pw;
	}
	pw = FN(PW,cow)(pw);
	if (isl_val_is_neg(v))
		pw = FN(PW,negate_type)(pw);
	if (!pw)
		goto error;

	for (i = 0; i < pw->n; ++i) {
		pw->p[i].FIELD = FN(EL,scale_val)(pw->p[i].FIELD,
						    isl_val_copy(v));
		if (!pw->p[i].FIELD)
			goto error;
	}

	isl_val_free(v);
	return pw;
error:
	isl_val_free(v);
	FN(PW,free)(pw);
	return NULL;
}

/* Divide the pieces of "pw" by "v" and return the result.
 */
__isl_give PW *FN(PW,scale_down_val)(__isl_take PW *pw, __isl_take isl_val *v)
{
	int i;

	if (!pw || !v)
		goto error;

	if (isl_val_is_one(v)) {
		isl_val_free(v);
		return pw;
	}

	if (!isl_val_is_rat(v))
		isl_die(isl_val_get_ctx(v), isl_error_invalid,
			"expecting rational factor", goto error);
	if (isl_val_is_zero(v))
		isl_die(isl_val_get_ctx(v), isl_error_invalid,
			"cannot scale down by zero", goto error);

	if (pw->n == 0) {
		isl_val_free(v);
		return pw;
	}
	pw = FN(PW,cow)(pw);
	if (isl_val_is_neg(v))
		pw = FN(PW,negate_type)(pw);
	if (!pw)
		goto error;

	for (i = 0; i < pw->n; ++i) {
		pw->p[i].FIELD = FN(EL,scale_down_val)(pw->p[i].FIELD,
						    isl_val_copy(v));
		if (!pw->p[i].FIELD)
			goto error;
	}

	isl_val_free(v);
	return pw;
error:
	isl_val_free(v);
	FN(PW,free)(pw);
	return NULL;
}

__isl_give PW *FN(PW,scale)(__isl_take PW *pw, isl_int v)
{
	return FN(PW,mul_isl_int)(pw, v);
}

/* Apply some normalization to "pw".
 * In particular, sort the pieces according to their function value
 * expressions, combining pairs of adjacent pieces with
 * the same such expression, and then normalize the domains of the pieces.
 *
 * We normalize in place, but if anything goes wrong we need
 * to return NULL, so we need to make sure we don't change the
 * meaning of any possible other copies of "pw".
 */
__isl_give PW *FN(PW,normalize)(__isl_take PW *pw)
{
	int i;
	isl_set *set;

	pw = FN(PW,sort)(pw);
	if (!pw)
		return NULL;
	for (i = 0; i < pw->n; ++i) {
		set = isl_set_normalize(isl_set_copy(pw->p[i].set));
		if (!set)
			return FN(PW,free)(pw);
		isl_set_free(pw->p[i].set);
		pw->p[i].set = set;
	}

	return pw;
}

/* Is pw1 obviously equal to pw2?
 * That is, do they have obviously identical cells and obviously identical
 * elements on each cell?
 *
 * If "pw1" or "pw2" contain any NaNs, then they are considered
 * not to be the same.  A NaN is not equal to anything, not even
 * to another NaN.
 */
isl_bool FN(PW,plain_is_equal)(__isl_keep PW *pw1, __isl_keep PW *pw2)
{
	int i;
	isl_bool equal, has_nan;

	if (!pw1 || !pw2)
		return isl_bool_error;

	has_nan = FN(PW,involves_nan)(pw1);
	if (has_nan >= 0 && !has_nan)
		has_nan = FN(PW,involves_nan)(pw2);
	if (has_nan < 0 || has_nan)
		return isl_bool_not(has_nan);

	if (pw1 == pw2)
		return isl_bool_true;
	equal = FN(PW,has_equal_space)(pw1, pw2);
	if (equal < 0 || !equal)
		return equal;

	pw1 = FN(PW,copy)(pw1);
	pw2 = FN(PW,copy)(pw2);
	pw1 = FN(PW,normalize)(pw1);
	pw2 = FN(PW,normalize)(pw2);
	if (!pw1 || !pw2)
		goto error;

	equal = isl_bool_ok(pw1->n == pw2->n);
	for (i = 0; equal && i < pw1->n; ++i) {
		equal = isl_set_plain_is_equal(pw1->p[i].set, pw2->p[i].set);
		if (equal < 0)
			goto error;
		if (!equal)
			break;
		equal = FN(EL,plain_is_equal)(pw1->p[i].FIELD, pw2->p[i].FIELD);
		if (equal < 0)
			goto error;
	}

	FN(PW,free)(pw1);
	FN(PW,free)(pw2);
	return equal;
error:
	FN(PW,free)(pw1);
	FN(PW,free)(pw2);
	return isl_bool_error;
}

/* Does "pw" involve any NaNs?
 */
isl_bool FN(PW,involves_nan)(__isl_keep PW *pw)
{
	int i;

	if (!pw)
		return isl_bool_error;
	if (pw->n == 0)
		return isl_bool_false;

	for (i = 0; i < pw->n; ++i) {
		isl_bool has_nan = FN(EL,involves_nan)(pw->p[i].FIELD);
		if (has_nan < 0 || has_nan)
			return has_nan;
	}

	return isl_bool_false;
}
