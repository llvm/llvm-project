/*
 * Copyright 2010      INRIA Saclay
 * Copyright 2013      Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, INRIA Saclay - Ile-de-France,
 * Parc Club Orsay Universite, ZAC des vignes, 4 rue Jacques Monod,
 * 91893 Orsay, France 
 * and Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 */

#undef TYPE
#define TYPE	UNION
static
#include "has_single_reference_templ.c"

__isl_give UNION *FN(UNION,cow)(__isl_take UNION *u);

isl_ctx *FN(UNION,get_ctx)(__isl_keep UNION *u)
{
	return u ? u->space->ctx : NULL;
}

/* Return the space of "u".
 */
static __isl_keep isl_space *FN(UNION,peek_space)(__isl_keep UNION *u)
{
	if (!u)
		return NULL;
	return u->space;
}

/* Return a copy of the space of "u".
 */
__isl_give isl_space *FN(UNION,get_space)(__isl_keep UNION *u)
{
	return isl_space_copy(FN(UNION,peek_space)(u));
}

/* Return the number of parameters of "u", where "type"
 * is required to be set to isl_dim_param.
 */
isl_size FN(UNION,dim)(__isl_keep UNION *u, enum isl_dim_type type)
{
	if (!u)
		return isl_size_error;

	if (type != isl_dim_param)
		isl_die(FN(UNION,get_ctx)(u), isl_error_invalid,
			"can only reference parameters", return isl_size_error);

	return isl_space_dim(u->space, type);
}

/* Return the position of the parameter with the given name
 * in "u".
 * Return -1 if no such dimension can be found.
 */
int FN(UNION,find_dim_by_name)(__isl_keep UNION *u, enum isl_dim_type type,
	const char *name)
{
	if (!u)
		return -1;
	return isl_space_find_dim_by_name(u->space, type, name);
}

#include "opt_type.h"

static __isl_give UNION *FN(UNION,alloc)(__isl_take isl_space *space
	OPT_TYPE_PARAM, int size)
{
	UNION *u;

	space = isl_space_params(space);
	if (!space)
		return NULL;

	u = isl_calloc_type(space->ctx, UNION);
	if (!u)
		goto error;

	u->ref = 1;
	OPT_SET_TYPE(u->, type);
	u->space = space;
	if (isl_hash_table_init(space->ctx, &u->table, size) < 0)
		return FN(UNION,free)(u);

	return u;
error:
	isl_space_free(space);
	return NULL;
}

/* Create an empty/zero union without specifying any parameters.
 */
__isl_give UNION *FN(FN(UNION,ZERO),ctx)(isl_ctx *ctx OPT_TYPE_PARAM)
{
	isl_space *space;

	space = isl_space_unit(ctx);
	return FN(FN(UNION,ZERO),space)(space OPT_TYPE_ARG(NO_LOC));
}

__isl_give UNION *FN(FN(UNION,ZERO),space)(__isl_take isl_space *space
	OPT_TYPE_PARAM)
{
	return FN(UNION,alloc)(space OPT_TYPE_ARG(NO_LOC), 16);
}

/* This is an alternative name for the function above.
 */
__isl_give UNION *FN(UNION,ZERO)(__isl_take isl_space *space OPT_TYPE_PARAM)
{
	return FN(FN(UNION,ZERO),space)(space OPT_TYPE_ARG(NO_LOC));
}

__isl_give UNION *FN(UNION,copy)(__isl_keep UNION *u)
{
	if (!u)
		return NULL;

	u->ref++;
	return u;
}

/* Do the tuples of "space" correspond to those of the domain of "part"?
 * That is, is the domain space of "part" equal to "space", ignoring parameters?
 */
static isl_bool FN(PART,has_domain_space_tuples)(__isl_keep PART *part,
	__isl_keep isl_space *space)
{
	return isl_space_has_domain_tuples(space, FN(PART,peek_space)(part));
}

/* Extract the element of "u" living in "space" (ignoring parameters).
 *
 * Return the ZERO element if "u" does not contain any element
 * living in "space".
 */
__isl_give PART *FN(FN(UNION,extract),BASE)(__isl_keep UNION *u,
	__isl_take isl_space *space)
{
	struct isl_hash_table_entry *entry;

	entry = FN(UNION,find_part_entry)(u, space, 0);
	if (!entry)
		goto error;
	if (entry == isl_hash_table_entry_none)
		return FN(PART,ZERO)(space OPT_TYPE_ARG(u->));
	isl_space_free(space);
	return FN(PART,copy)(entry->data);
error:
	isl_space_free(space);
	return NULL;
}

/* Add "part" to "u".
 * If "disjoint" is set, then "u" is not allowed to already have
 * a part that is defined over a domain that overlaps with the domain
 * of "part".
 * Otherwise, compute the union sum of "part" and the part in "u"
 * defined on the same space.
 */
static __isl_give UNION *FN(UNION,add_part_generic)(__isl_take UNION *u,
	__isl_take PART *part, int disjoint)
{
	int empty;
	struct isl_hash_table_entry *entry;

	if (!part)
		goto error;

	empty = FN(PART,IS_ZERO)(part);
	if (empty < 0)
		goto error;
	if (empty) {
		FN(PART,free)(part);
		return u;
	}

	u = FN(UNION,align_params)(u, FN(PART,get_space)(part));
	part = FN(PART,align_params)(part, FN(UNION,get_space)(u));

	u = FN(UNION,cow)(u);

	if (!u)
		goto error;

	if (FN(UNION,check_disjoint_domain_other)(u, part) < 0)
		goto error;
	entry = FN(UNION,find_part_entry)(u, part->dim, 1);
	if (!entry)
		goto error;

	if (!entry->data)
		entry->data = part;
	else {
		if (disjoint &&
		    FN(UNION,check_disjoint_domain)(entry->data, part) < 0)
			goto error;
		entry->data = FN(PART,union_add_)(entry->data,
						FN(PART,copy)(part));
		if (!entry->data)
			goto error;
		empty = FN(PART,IS_ZERO)(part);
		if (empty < 0)
			goto error;
		if (empty)
			u = FN(UNION,remove_part_entry)(u, entry);
		FN(PART,free)(part);
	}

	return u;
error:
	FN(PART,free)(part);
	FN(UNION,free)(u);
	return NULL;
}

/* Add "part" to "u", where "u" is assumed not to already have
 * a part that is defined on the same space as "part".
 */
__isl_give UNION *FN(FN(UNION,add),BASE)(__isl_take UNION *u,
	__isl_take PART *part)
{
	return FN(UNION,add_part_generic)(u, part, 1);
}

/* Allocate a UNION with the same type (if any) and the same size as "u" and
 * with space "space".
 */
static __isl_give UNION *FN(UNION,alloc_same_size_on_space)(__isl_keep UNION *u,
	__isl_take isl_space *space)
{
	if (!u)
		goto error;
	return FN(UNION,alloc)(space OPT_TYPE_ARG(u->), u->table.n);
error:
	isl_space_free(space);
	return NULL;
}

/* Allocate a UNION with the same space, the same type (if any) and
 * the same size as "u".
 */
static __isl_give UNION *FN(UNION,alloc_same_size)(__isl_keep UNION *u)
{
	return FN(UNION,alloc_same_size_on_space)(u, FN(UNION,get_space)(u));
}

/* Data structure that specifies how isl_union_*_transform
 * should modify the base expressions in the union expression.
 *
 * If "inplace" is set, then the base expression in the input union
 * are modified in place.  This means that "fn" should not
 * change the meaning of the union or that the union only
 * has a single reference.
 * If "space" is not NULL, then a new union is created in this space.
 * If "filter" is not NULL, then only the base expressions that satisfy "filter"
 * are taken into account.
 * "filter_user" is passed as the second argument to "filter".
 * If "fn" it not NULL, then it is applied to each entry in the input.
 * "fn_user" is passed as the second argument to "fn".
 */
S(UNION,transform_control) {
	int inplace;
	isl_space *space;
	isl_bool (*filter)(__isl_keep PART *part, void *user);
	void *filter_user;
	__isl_give PART *(*fn)(__isl_take PART *part, void *user);
	void *fn_user;
};

/* Internal data structure for isl_union_*_transform_space.
 * "control" specifies how the base expressions should be modified.
 * "res" collects the results (if control->inplace is not set).
 */
S(UNION,transform_data)
{
	S(UNION,transform_control) *control;
	UNION *res;
};

/* Apply control->fn to "part" and add the result to data->res or
 * place it back into the input union if control->inplace is set.
 */
static isl_stat FN(UNION,transform_entry)(void **entry, void *user)
{
	S(UNION,transform_data) *data = (S(UNION,transform_data) *)user;
	S(UNION,transform_control) *control = data->control;
	PART *part = *entry;

	if (control->filter) {
		isl_bool handle;

		handle = control->filter(part, control->filter_user);
		if (handle < 0)
			return isl_stat_error;
		if (!handle)
			return isl_stat_ok;
	}

	if (!control->inplace)
		part = FN(PART,copy)(part);
	if (control->fn)
		part = control->fn(part, control->fn_user);
	if (control->inplace)
		*entry = part;
	else
		data->res = FN(FN(UNION,add),BASE)(data->res, part);
	if (!part || !data->res)
		return isl_stat_error;

	return isl_stat_ok;
}

/* Return a UNION that is obtained by modifying "u" according to "control".
 */
static __isl_give UNION *FN(UNION,transform)(__isl_take UNION *u,
	S(UNION,transform_control) *control)
{
	S(UNION,transform_data) data = { control };
	isl_space *space;

	if (control->inplace) {
		data.res = u;
	} else {
		if (control->space)
			space = isl_space_copy(control->space);
		else
			space = FN(UNION,get_space)(u);
		data.res = FN(UNION,alloc_same_size_on_space)(u, space);
	}
	if (FN(UNION,foreach_inplace)(u, &FN(UNION,transform_entry), &data) < 0)
		data.res = FN(UNION,free)(data.res);
	if (!control->inplace)
		FN(UNION,free)(u);
	return data.res;
}

/* Return a UNION living in "space" that is otherwise obtained by modifying "u"
 * according to "control".
 */
static __isl_give UNION *FN(UNION,transform_space)(__isl_take UNION *u,
	__isl_take isl_space *space, S(UNION,transform_control) *control)
{
	if (!space)
		return FN(UNION,free)(u);
	control->space = space;
	u = FN(UNION,transform)(u, control);
	isl_space_free(space);
	return u;
}

/* Update "u" by applying "fn" to each entry.
 * This operation is assumed not to change the number of entries nor
 * the spaces of the entries.
 *
 * If there is only one reference to "u", then change "u" inplace.
 * Otherwise, create a new UNION from "u" and discard the original.
 */
static __isl_give UNION *FN(UNION,transform_inplace)(__isl_take UNION *u,
	__isl_give PART *(*fn)(__isl_take PART *part, void *user), void *user)
{
	S(UNION,transform_control) control = { .fn = fn, .fn_user = user };
	isl_bool single_ref;

	single_ref = FN(UNION,has_single_reference)(u);
	if (single_ref < 0)
		return FN(UNION,free)(u);
	if (single_ref)
		control.inplace = 1;
	return FN(UNION,transform)(u, &control);
}

/* An isl_union_*_transform callback for use in isl_union_*_dup
 * that simply returns "part".
 */
static __isl_give PART *FN(UNION,copy_part)(__isl_take PART *part, void *user)
{
	return part;
}

__isl_give UNION *FN(UNION,dup)(__isl_keep UNION *u)
{
	S(UNION,transform_control) control = { .fn = &FN(UNION,copy_part) };

	u = FN(UNION,copy)(u);
	return FN(UNION,transform)(u, &control);
}

__isl_give UNION *FN(UNION,cow)(__isl_take UNION *u)
{
	if (!u)
		return NULL;

	if (u->ref == 1)
		return u;
	u->ref--;
	return FN(UNION,dup)(u);
}

__isl_null UNION *FN(UNION,free)(__isl_take UNION *u)
{
	if (!u)
		return NULL;

	if (--u->ref > 0)
		return NULL;

	isl_hash_table_foreach(u->space->ctx, &u->table,
				&FN(UNION,free_u_entry), NULL);
	isl_hash_table_clear(&u->table);
	isl_space_free(u->space);
	free(u);
	return NULL;
}

static __isl_give PART *FN(UNION,align_entry)(__isl_take PART *part, void *user)
{
	isl_reordering *exp = user;

	exp = isl_reordering_extend_space(isl_reordering_copy(exp),
				    FN(PART,get_domain_space)(part));
	return FN(PART,realign_domain)(part, exp);
}

/* Reorder the parameters of "u" according to the given reordering.
 */
static __isl_give UNION *FN(UNION,realign_domain)(__isl_take UNION *u,
	__isl_take isl_reordering *r)
{
	S(UNION,transform_control) control = {
		.fn = &FN(UNION,align_entry),
		.fn_user = r,
	};
	isl_space *space;

	if (!u || !r)
		goto error;

	space = isl_reordering_get_space(r);
	u = FN(UNION,transform_space)(u, space, &control);
	isl_reordering_free(r);
	return u;
error:
	FN(UNION,free)(u);
	isl_reordering_free(r);
	return NULL;
}

/* Align the parameters of "u" to those of "model".
 */
__isl_give UNION *FN(UNION,align_params)(__isl_take UNION *u,
	__isl_take isl_space *model)
{
	isl_bool equal_params;
	isl_reordering *r;

	if (!u || !model)
		goto error;

	equal_params = isl_space_has_equal_params(u->space, model);
	if (equal_params < 0)
		goto error;
	if (equal_params) {
		isl_space_free(model);
		return u;
	}

	r = isl_parameter_alignment_reordering(u->space, model);
	isl_space_free(model);

	return FN(UNION,realign_domain)(u, r);
error:
	isl_space_free(model);
	FN(UNION,free)(u);
	return NULL;
}

/* Add "part" to *u, taking the union sum if "u" already has
 * a part defined on the same space as "part".
 */
static isl_stat FN(UNION,union_add_part)(__isl_take PART *part, void *user)
{
	UNION **u = (UNION **)user;

	*u = FN(UNION,add_part_generic)(*u, part, 0);

	return isl_stat_ok;
}

/* Compute the sum of "u1" and "u2" on the union of their domains,
 * with the actual sum on the shared domain and
 * the defined expression on the symmetric difference of the domains.
 *
 * This is an internal function that is exposed under different
 * names depending on whether the base expressions have a zero default
 * value.
 * If they do, then this function is called "add".
 * Otherwise, it is called "union_add".
 */
static __isl_give UNION *FN(UNION,union_add_)(__isl_take UNION *u1,
	__isl_take UNION *u2)
{
	u1 = FN(UNION,align_params)(u1, FN(UNION,get_space)(u2));
	u2 = FN(UNION,align_params)(u2, FN(UNION,get_space)(u1));

	u1 = FN(UNION,cow)(u1);

	if (!u1 || !u2)
		goto error;

	if (FN(FN(UNION,foreach),BASE)(u2, &FN(UNION,union_add_part), &u1) < 0)
		goto error;

	FN(UNION,free)(u2);

	return u1;
error:
	FN(UNION,free)(u1);
	FN(UNION,free)(u2);
	return NULL;
}

__isl_give UNION *FN(FN(UNION,from),BASE)(__isl_take PART *part)
{
	isl_space *space;
	UNION *u;

	if (!part)
		return NULL;

	space = FN(PART,get_space)(part);
	space = isl_space_drop_dims(space, isl_dim_in, 0,
					isl_space_dim(space, isl_dim_in));
	space = isl_space_drop_dims(space, isl_dim_out, 0,
					isl_space_dim(space, isl_dim_out));
	u = FN(UNION,ZERO)(space OPT_TYPE_ARG(part->));
	u = FN(FN(UNION,add),BASE)(u, part);

	return u;
}

S(UNION,match_bin_data) {
	UNION *u2;
	UNION *res;
	__isl_give PART *(*fn)(__isl_take PART *, __isl_take PART *);
};

/* Check if data->u2 has an element living in the same space as "part".
 * If so, call data->fn on the two elements and add the result to
 * data->res.
 */
static isl_stat FN(UNION,match_bin_entry)(__isl_take PART *part, void *user)
{
	S(UNION,match_bin_data) *data = user;
	struct isl_hash_table_entry *entry2;
	isl_space *space;
	PART *part2;

	space = FN(PART,get_space)(part);
	entry2 = FN(UNION,find_part_entry)(data->u2, space, 0);
	isl_space_free(space);
	if (!entry2)
		goto error;
	if (entry2 == isl_hash_table_entry_none) {
		FN(PART,free)(part);
		return isl_stat_ok;
	}

	part2 = entry2->data;
	if (!isl_space_tuple_is_equal(part->dim, isl_dim_out,
					part2->dim, isl_dim_out))
		isl_die(FN(UNION,get_ctx)(data->u2), isl_error_invalid,
			"entries should have the same range space",
			goto error);

	part = data->fn(part, FN(PART, copy)(entry2->data));

	data->res = FN(FN(UNION,add),BASE)(data->res, part);
	if (!data->res)
		return isl_stat_error;

	return isl_stat_ok;
error:
	FN(PART,free)(part);
	return isl_stat_error;
}

/* This function is currently only used from isl_polynomial.c
 * and not from isl_fold.c.
 */
static __isl_give UNION *FN(UNION,match_bin_op)(__isl_take UNION *u1,
	__isl_take UNION *u2,
	__isl_give PART *(*fn)(__isl_take PART *, __isl_take PART *))
	__attribute__ ((unused));
/* For each pair of elements in "u1" and "u2" living in the same space,
 * call "fn" and collect the results.
 */
static __isl_give UNION *FN(UNION,match_bin_op)(__isl_take UNION *u1,
	__isl_take UNION *u2,
	__isl_give PART *(*fn)(__isl_take PART *, __isl_take PART *))
{
	S(UNION,match_bin_data) data = { NULL, NULL, fn };

	u1 = FN(UNION,align_params)(u1, FN(UNION,get_space)(u2));
	u2 = FN(UNION,align_params)(u2, FN(UNION,get_space)(u1));

	if (!u1 || !u2)
		goto error;

	data.u2 = u2;
	data.res = FN(UNION,alloc_same_size)(u1);
	if (FN(FN(UNION,foreach),BASE)(u1,
				    &FN(UNION,match_bin_entry), &data) < 0)
		goto error;

	FN(UNION,free)(u1);
	FN(UNION,free)(u2);
	return data.res;
error:
	FN(UNION,free)(u1);
	FN(UNION,free)(u2);
	FN(UNION,free)(data.res);
	return NULL;
}

/* Compute the sum of "u1" and "u2".
 *
 * If the base expressions have a default zero value, then the sum
 * is computed on the union of the domains of "u1" and "u2".
 * Otherwise, it is computed on their shared domains.
 */
__isl_give UNION *FN(UNION,add)(__isl_take UNION *u1, __isl_take UNION *u2)
{
#if DEFAULT_IS_ZERO
	return FN(UNION,union_add_)(u1, u2);
#else
	return FN(UNION,match_bin_op)(u1, u2, &FN(PART,add));
#endif
}

#ifndef NO_SUB
/* Subtract "u2" from "u1" and return the result.
 */
__isl_give UNION *FN(UNION,sub)(__isl_take UNION *u1, __isl_take UNION *u2)
{
	return FN(UNION,match_bin_op)(u1, u2, &FN(PART,sub));
}
#endif

S(UNION,any_set_data) {
	isl_set *set;
	__isl_give PW *(*fn)(__isl_take PW*, __isl_take isl_set*);
};

static __isl_give PART *FN(UNION,any_set_entry)(__isl_take PART *part,
	void *user)
{
	S(UNION,any_set_data) *data = user;

	return data->fn(part, isl_set_copy(data->set));
}

/* Update each element of "u" by calling "fn" on the element and "set".
 */
static __isl_give UNION *FN(UNION,any_set_op)(__isl_take UNION *u,
	__isl_take isl_set *set,
	__isl_give PW *(*fn)(__isl_take PW*, __isl_take isl_set*))
{
	S(UNION,any_set_data) data = { NULL, fn };
	S(UNION,transform_control) control = {
		.fn = &FN(UNION,any_set_entry),
		.fn_user = &data,
	};

	u = FN(UNION,align_params)(u, isl_set_get_space(set));
	set = isl_set_align_params(set, FN(UNION,get_space)(u));

	if (!u || !set)
		goto error;

	data.set = set;
	u = FN(UNION,transform)(u, &control);
	isl_set_free(set);
	return u;
error:
	FN(UNION,free)(u);
	isl_set_free(set);
	return NULL;
}

/* Intersect the domain of "u" with the parameter domain "context".
 */
__isl_give UNION *FN(UNION,intersect_params)(__isl_take UNION *u,
	__isl_take isl_set *set)
{
	return FN(UNION,any_set_op)(u, set, &FN(PW,intersect_params));
}

/* Compute the gist of the domain of "u" with respect to
 * the parameter domain "context".
 */
__isl_give UNION *FN(UNION,gist_params)(__isl_take UNION *u,
	__isl_take isl_set *set)
{
	return FN(UNION,any_set_op)(u, set, &FN(PW,gist_params));
}

/* Data structure that specifies how isl_union_*_match_domain_op
 * should combine its arguments.
 *
 * If "filter" is not NULL, then only parts that pass the given
 * filter are considered for matching.
 * "fn" is applied to each part in the union and each corresponding
 * set in the union set, i.e., such that the set lives in the same space
 * as the domain of the part.
 * If "match_space" is not NULL, then the set extracted from the union set
 * does not live in the same space as the domain of the part,
 * but rather in the space that results from calling "match_space"
 * on this domain space.
 */
S(UNION,match_domain_control) {
	isl_bool (*filter)(__isl_keep PART *part);
	__isl_give isl_space *(*match_space)(__isl_take isl_space *space);
	__isl_give PW *(*fn)(__isl_take PW*, __isl_take isl_set*);
};

S(UNION,match_domain_data) {
	isl_union_set *uset;
	UNION *res;
	S(UNION,match_domain_control) *control;
};

/* Find the set in data->uset that lives in the same space as the domain
 * of "part" (ignoring parameters), apply data->fn to *entry and this set
 * (if any), and add the result to data->res.
 */
static isl_stat FN(UNION,match_domain_entry)(__isl_take PART *part, void *user)
{
	S(UNION,match_domain_data) *data = user;
	struct isl_hash_table_entry *entry2;
	isl_space *space;

	if (data->control->filter) {
		isl_bool pass = data->control->filter(part);
		if (pass < 0 || !pass) {
			FN(PART,free)(part);
			return pass < 0 ? isl_stat_error : isl_stat_ok;
		}
	}

	space = FN(PART,get_domain_space)(part);
	if (data->control->match_space)
		space = data->control->match_space(space);
	entry2 = isl_union_set_find_entry(data->uset, space, 0);
	isl_space_free(space);
	if (!entry2 || entry2 == isl_hash_table_entry_none) {
		FN(PART,free)(part);
		return isl_stat_non_null(entry2);
	}

	part = data->control->fn(part, isl_set_copy(entry2->data));

	data->res = FN(FN(UNION,add),BASE)(data->res, part);
	if (!data->res)
		return isl_stat_error;

	return isl_stat_ok;
}

/* Combine "u" and "uset" according to "control"
 * and collect the results.
 */
static __isl_give UNION *FN(UNION,match_domain_op)(__isl_take UNION *u,
	__isl_take isl_union_set *uset, S(UNION,match_domain_control) *control)
{
	S(UNION,match_domain_data) data = { NULL, NULL, control };

	if (!u || !uset)
		goto error;

	data.uset = uset;
	data.res = FN(UNION,alloc_same_size)(u);
	if (FN(FN(UNION,foreach),BASE)(u,
				   &FN(UNION,match_domain_entry), &data) < 0)
		goto error;

	FN(UNION,free)(u);
	isl_union_set_free(uset);
	return data.res;
error:
	FN(UNION,free)(u);
	isl_union_set_free(uset);
	FN(UNION,free)(data.res);
	return NULL;
}

/* Intersect the domain of "u" with "uset".
 * If "uset" is a parameters domain, then intersect the parameter
 * domain of "u" with this set.
 */
__isl_give UNION *FN(UNION,intersect_domain_union_set)(__isl_take UNION *u,
	__isl_take isl_union_set *uset)
{
	S(UNION,match_domain_control) control = {
		.fn = &FN(PW,intersect_domain),
	};

	if (isl_union_set_is_params(uset))
		return FN(UNION,intersect_params)(u,
						isl_set_from_union_set(uset));
	return FN(UNION,match_domain_op)(u, uset, &control);
}

/* This is an alternative name for the function above.
 */
__isl_give UNION *FN(UNION,intersect_domain)(__isl_take UNION *u,
	__isl_take isl_union_set *uset)
{
	return FN(UNION,intersect_domain_union_set)(u, uset);
}

/* Return true if this part should be kept.
 *
 * In particular, it should be kept if its domain space
 * corresponds to "space".
 */
static isl_bool FN(UNION,select_entry)(__isl_keep PART *part, void *user)
{
	isl_space *space = user;

	return FN(PW,has_domain_space_tuples)(part, space);
}

/* Remove any not element in "space" from the domain of "u".
 *
 * In particular, select any part of the function defined
 * on this domain space.
 */
__isl_give UNION *FN(UNION,intersect_domain_space)(__isl_take UNION *u,
	__isl_take isl_space *space)
{
	S(UNION,transform_control) control = {
		.filter = &FN(UNION,select_entry),
		.filter_user = space,
	};

	u = FN(UNION,transform)(u, &control);
	isl_space_free(space);
	return u;
}

/* Is the domain of "pw" a wrapped relation?
 */
static isl_bool FN(PW,domain_is_wrapping)(__isl_keep PW *pw)
{
	return isl_space_domain_is_wrapping(FN(PW,peek_space)(pw));
}

/* Intersect the domain of the wrapped relation inside the domain of "u"
 * with "uset".
 */
__isl_give UNION *FN(UNION,intersect_domain_wrapped_domain)(__isl_take UNION *u,
	__isl_take isl_union_set *uset)
{
	S(UNION,match_domain_control) control = {
		.filter = &FN(PART,domain_is_wrapping),
		.match_space = &isl_space_factor_domain,
		.fn = &FN(PW,intersect_domain_wrapped_domain),
	};

	return FN(UNION,match_domain_op)(u, uset, &control);
}

/* Intersect the range of the wrapped relation inside the domain of "u"
 * with "uset".
 */
__isl_give UNION *FN(UNION,intersect_domain_wrapped_range)(__isl_take UNION *u,
	__isl_take isl_union_set *uset)
{
	S(UNION,match_domain_control) control = {
		.filter = &FN(PART,domain_is_wrapping),
		.match_space = &isl_space_factor_range,
		.fn = &FN(PW,intersect_domain_wrapped_range),
	};

	return FN(UNION,match_domain_op)(u, uset, &control);
}

/* Take the set (which may be empty) in data->uset that lives
 * in the same space as the domain of "pw", subtract it from the domain
 * of "part" and return the result.
 */
static __isl_give PART *FN(UNION,subtract_domain_entry)(__isl_take PART *part,
	void *user)
{
	isl_union_set *uset = user;
	isl_space *space;
	isl_set *set;

	space = FN(PART,get_domain_space)(part);
	set = isl_union_set_extract_set(uset, space);
	return FN(PART,subtract_domain)(part, set);
}

/* Subtract "uset" from the domain of "u".
 */
__isl_give UNION *FN(UNION,subtract_domain_union_set)(__isl_take UNION *u,
	__isl_take isl_union_set *uset)
{
	S(UNION,transform_control) control = {
		.fn = &FN(UNION,subtract_domain_entry),
		.fn_user = uset,
	};

	u = FN(UNION,transform)(u, &control);
	isl_union_set_free(uset);
	return u;
}

/* This is an alternative name for the function above.
 */
__isl_give UNION *FN(UNION,subtract_domain)(__isl_take UNION *u,
	__isl_take isl_union_set *uset)
{
	return FN(UNION,subtract_domain_union_set)(u, uset);
}

/* Return true if this part should be kept.
 *
 * In particular, it should be kept if its domain space
 * does not correspond to "space".
 */
static isl_bool FN(UNION,filter_out_entry)(__isl_keep PART *part, void *user)
{
	isl_space *space = user;

	return isl_bool_not(FN(PW,has_domain_space_tuples)(part, space));
}

/* Remove any element in "space" from the domain of "u".
 *
 * In particular, filter out any part of the function defined
 * on this domain space.
 */
__isl_give UNION *FN(UNION,subtract_domain_space)(__isl_take UNION *u,
	__isl_take isl_space *space)
{
	S(UNION,transform_control) control = {
		.filter = &FN(UNION,filter_out_entry),
		.filter_user = space,
	};

	u = FN(UNION,transform)(u, &control);
	isl_space_free(space);
	return u;
}

__isl_give UNION *FN(UNION,gist)(__isl_take UNION *u,
	__isl_take isl_union_set *uset)
{
	S(UNION,match_domain_control) control = {
		.fn = &FN(PW,gist),
	};

	if (isl_union_set_is_params(uset))
		return FN(UNION,gist_params)(u, isl_set_from_union_set(uset));
	return FN(UNION,match_domain_op)(u, uset, &control);
}

/* Coalesce an entry in a UNION.  Coalescing is performed in-place.
 * Since the UNION may have several references, the entry is only
 * replaced if the coalescing is successful.
 */
static isl_stat FN(UNION,coalesce_entry)(void **entry, void *user)
{
	PART **part_p = (PART **) entry;
	PART *part;

	part = FN(PART,copy)(*part_p);
	part = FN(PW,coalesce)(part);
	if (!part)
		return isl_stat_error;
	FN(PART,free)(*part_p);
	*part_p = part;

	return isl_stat_ok;
}

__isl_give UNION *FN(UNION,coalesce)(__isl_take UNION *u)
{
	if (FN(UNION,foreach_inplace)(u, &FN(UNION,coalesce_entry), NULL) < 0)
		goto error;

	return u;
error:
	FN(UNION,free)(u);
	return NULL;
}

static isl_stat FN(UNION,domain_entry)(__isl_take PART *part, void *user)
{
	isl_union_set **uset = (isl_union_set **)user;

	*uset = isl_union_set_add_set(*uset, FN(PART,domain)(part));

	return isl_stat_ok;
}

__isl_give isl_union_set *FN(UNION,domain)(__isl_take UNION *u)
{
	isl_union_set *uset;

	uset = isl_union_set_empty(FN(UNION,get_space)(u));
	if (FN(FN(UNION,foreach),BASE)(u, &FN(UNION,domain_entry), &uset) < 0)
		goto error;

	FN(UNION,free)(u);
	
	return uset;
error:
	isl_union_set_free(uset);
	FN(UNION,free)(u);
	return NULL;
}

#ifdef HAS_TYPE
/* Negate the type of "u".
 */
static __isl_give UNION *FN(UNION,negate_type)(__isl_take UNION *u)
{
	u = FN(UNION,cow)(u);
	if (!u)
		return NULL;
	u->type = isl_fold_type_negate(u->type);
	return u;
}
#else
/* Negate the type of "u".
 * Since "u" does not have a type, do nothing.
 */
static __isl_give UNION *FN(UNION,negate_type)(__isl_take UNION *u)
{
	return u;
}
#endif

/* Multiply "part" by the isl_val "user" and return the result.
 */
static __isl_give PART *FN(UNION,scale_val_entry)(__isl_take PART *part,
	void *user)
{
	isl_val *v = user;

	return FN(PART,scale_val)(part, isl_val_copy(v));
}

/* Multiply "u" by "v" and return the result.
 */
__isl_give UNION *FN(UNION,scale_val)(__isl_take UNION *u,
	__isl_take isl_val *v)
{
	if (!u || !v)
		goto error;
	if (isl_val_is_one(v)) {
		isl_val_free(v);
		return u;
	}

	if (DEFAULT_IS_ZERO && u && isl_val_is_zero(v)) {
		UNION *zero;
		isl_space *space = FN(UNION,get_space)(u);
		zero = FN(UNION,ZERO)(space OPT_TYPE_ARG(u->));
		FN(UNION,free)(u);
		isl_val_free(v);
		return zero;
	}

	if (!isl_val_is_rat(v))
		isl_die(isl_val_get_ctx(v), isl_error_invalid,
			"expecting rational factor", goto error);

	u = FN(UNION,transform_inplace)(u, &FN(UNION,scale_val_entry), v);
	if (isl_val_is_neg(v))
		u = FN(UNION,negate_type)(u);

	isl_val_free(v);
	return u;
error:
	isl_val_free(v);
	FN(UNION,free)(u);
	return NULL;
}

/* Divide "part" by the isl_val "user" and return the result.
 */
static __isl_give PART *FN(UNION,scale_down_val_entry)(__isl_take PART *part,
	void *user)
{
	isl_val *v = user;

	return FN(PART,scale_down_val)(part, isl_val_copy(v));
}

/* Divide "u" by "v" and return the result.
 */
__isl_give UNION *FN(UNION,scale_down_val)(__isl_take UNION *u,
	__isl_take isl_val *v)
{
	if (!u || !v)
		goto error;
	if (isl_val_is_one(v)) {
		isl_val_free(v);
		return u;
	}

	if (!isl_val_is_rat(v))
		isl_die(isl_val_get_ctx(v), isl_error_invalid,
			"expecting rational factor", goto error);
	if (isl_val_is_zero(v))
		isl_die(isl_val_get_ctx(v), isl_error_invalid,
			"cannot scale down by zero", goto error);

	u = FN(UNION,transform_inplace)(u, &FN(UNION,scale_down_val_entry), v);
	if (isl_val_is_neg(v))
		u = FN(UNION,negate_type)(u);

	isl_val_free(v);
	return u;
error:
	isl_val_free(v);
	FN(UNION,free)(u);
	return NULL;
}

/* Internal data structure for isl_union_*_every_*.
 *
 * "test" is the user-specified callback function.
 * "user" is the user-specified callback function argument.
 *
 * "res" is the final result, initialized to isl_bool_true.
 */
S(UNION,every_data) {
	isl_bool (*test)(__isl_keep PW *pw, void *user);
	void *user;

	isl_bool res;
};

/* Call data->test on the piecewise expression at *entry,
 * updating the result in data->res.
 * Abort if this result is no longer isl_bool_true.
 */
static isl_stat FN(UNION,every_entry)(void **entry, void *user)
{
	S(UNION,every_data) *data = user;
	PW *pw = *entry;

	data->res = data->test(pw, data->user);
	if (data->res < 0 || !data->res)
		return isl_stat_error;

	return isl_stat_ok;
}

/* Does "test" succeed on every piecewise expression in "u"?
 */
isl_bool FN(FN(UNION,every),BASE)(__isl_keep UNION *u,
	isl_bool (*test)(__isl_keep PW *pw, void *user), void *user)
{
	S(UNION,every_data) data = { test, user };

	data.res = isl_bool_true;
	if (FN(UNION,foreach_inplace)(u, &FN(UNION,every_entry), &data) < 0 &&
	    data.res == isl_bool_true)
		return isl_bool_error;

	return data.res;
}

S(UNION,plain_is_equal_data)
{
	UNION *u2;
};

static isl_bool FN(UNION,plain_is_equal_el)(__isl_keep PW *pw, void *user)
{
	S(UNION,plain_is_equal_data) *data = user;
	struct isl_hash_table_entry *entry;

	entry = FN(UNION,find_part_entry)(data->u2, pw->dim, 0);
	if (!entry)
		return isl_bool_error;
	if (entry == isl_hash_table_entry_none)
		return isl_bool_false;

	return FN(PW,plain_is_equal)(pw, entry->data);
}

isl_bool FN(UNION,plain_is_equal)(__isl_keep UNION *u1, __isl_keep UNION *u2)
{
	S(UNION,plain_is_equal_data) data;
	isl_size n1, n2;
	isl_bool is_equal;

	if (!u1 || !u2)
		return isl_bool_error;
	if (u1 == u2)
		return isl_bool_true;
	if (u1->table.n != u2->table.n)
		return isl_bool_false;
	n1 = FN(FN(UNION,n),BASE)(u1);
	n2 = FN(FN(UNION,n),BASE)(u2);
	if (n1 < 0 || n2 < 0)
		return isl_bool_error;
	if (n1 != n2)
		return isl_bool_false;

	u1 = FN(UNION,copy)(u1);
	u2 = FN(UNION,copy)(u2);
	u1 = FN(UNION,align_params)(u1, FN(UNION,get_space)(u2));
	u2 = FN(UNION,align_params)(u2, FN(UNION,get_space)(u1));
	if (!u1 || !u2)
		goto error;

	data.u2 = u2;
	is_equal = FN(FN(UNION,every),BASE)(u1,
					  &FN(UNION,plain_is_equal_el), &data);

	FN(UNION,free)(u1);
	FN(UNION,free)(u2);

	return is_equal;
error:
	FN(UNION,free)(u1);
	FN(UNION,free)(u2);
	return isl_bool_error;
}

/* An isl_union_*_every_* callback that checks whether "pw"
 * does not involve any NaNs.
 */
static isl_bool FN(UNION,no_nan_el)(__isl_keep PW *pw, void *user)
{
	return isl_bool_not(FN(PW,involves_nan)(pw));
}

/* Does "u" involve any NaNs?
 */
isl_bool FN(UNION,involves_nan)(__isl_keep UNION *u)
{
	isl_bool no_nan;

	no_nan = FN(FN(UNION,every),BASE)(u, &FN(UNION,no_nan_el), NULL);

	return isl_bool_not(no_nan);
}

/* Internal data structure for isl_union_*_drop_dims.
 * type, first and n are passed to isl_*_drop_dims.
 */
S(UNION,drop_dims_data) {
	enum isl_dim_type type;
	unsigned first;
	unsigned n;
};

/* Drop the parameters specified by "data" from "part" and return the result.
 */
static __isl_give PART *FN(UNION,drop_dims_entry)(__isl_take PART *part,
	void *user)
{
	S(UNION,drop_dims_data) *data = user;

	return FN(PART,drop_dims)(part, data->type, data->first, data->n);
}

/* Drop the specified parameters from "u".
 * That is, type is required to be isl_dim_param.
 */
__isl_give UNION *FN(UNION,drop_dims)( __isl_take UNION *u,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	isl_space *space;
	S(UNION,drop_dims_data) data = { type, first, n };
	S(UNION,transform_control) control = {
		.fn = &FN(UNION,drop_dims_entry),
		.fn_user = &data,
	};

	if (!u)
		return NULL;

	if (type != isl_dim_param)
		isl_die(FN(UNION,get_ctx)(u), isl_error_invalid,
			"can only project out parameters",
			return FN(UNION,free)(u));

	space = FN(UNION,get_space)(u);
	space = isl_space_drop_dims(space, type, first, n);
	return FN(UNION,transform_space)(u, space, &control);
}

/* Internal data structure for isl_union_*_set_dim_name.
 * pos is the position of the parameter that needs to be renamed.
 * s is the new name.
 */
S(UNION,set_dim_name_data) {
	unsigned pos;
	const char *s;
};

/* Change the name of the parameter at position data->pos of "part" to data->s
 * and return the result.
 */
static __isl_give PART *FN(UNION,set_dim_name_entry)(__isl_take PART *part,
	void *user)
{
	S(UNION,set_dim_name_data) *data = user;

	return FN(PART,set_dim_name)(part, isl_dim_param, data->pos, data->s);
}

/* Change the name of the parameter at position "pos" to "s".
 * That is, type is required to be isl_dim_param.
 */
__isl_give UNION *FN(UNION,set_dim_name)(__isl_take UNION *u,
	enum isl_dim_type type, unsigned pos, const char *s)
{
	S(UNION,set_dim_name_data) data = { pos, s };
	S(UNION,transform_control) control = {
		.fn = &FN(UNION,set_dim_name_entry),
		.fn_user = &data,
	};
	isl_space *space;

	if (!u)
		return NULL;

	if (type != isl_dim_param)
		isl_die(FN(UNION,get_ctx)(u), isl_error_invalid,
			"can only set parameter names",
			return FN(UNION,free)(u));

	space = FN(UNION,get_space)(u);
	space = isl_space_set_dim_name(space, type, pos, s);
	return FN(UNION,transform_space)(u, space, &control);
}

/* Reset the user pointer on all identifiers of parameters and tuples
 * of the space of "part" and return the result.
 */
static __isl_give PART *FN(UNION,reset_user_entry)(__isl_take PART *part,
	void *user)
{
	return FN(PART,reset_user)(part);
}

/* Reset the user pointer on all identifiers of parameters and tuples
 * of the spaces of "u".
 */
__isl_give UNION *FN(UNION,reset_user)(__isl_take UNION *u)
{
	S(UNION,transform_control) control = {
		.fn = &FN(UNION,reset_user_entry),
	};
	isl_space *space;

	space = FN(UNION,get_space)(u);
	space = isl_space_reset_user(space);
	return FN(UNION,transform_space)(u, space, &control);
}

/* Add the base expression held by "entry" to "list".
 */
static isl_stat FN(UNION,add_to_list)(void **entry, void *user)
{
	PW *pw = *entry;
	LIST(PART) **list = user;

	*list = FN(LIST(PART),add)(*list, FN(PART,copy)(pw));
	if (!*list)
		return isl_stat_error;

	return isl_stat_ok;
}

/* Return a list containing all the base expressions in "u".
 *
 * First construct a list of the appropriate size and
 * then add all the elements.
 */
__isl_give LIST(PART) *FN(FN(UNION,get),LIST(BASE))(__isl_keep UNION *u)
{
	isl_size n;
	LIST(PART) *list;

	if (!u)
		return NULL;
	n = FN(FN(UNION,n),BASE)(u);
	if (n < 0)
		return NULL;
	list = FN(LIST(PART),alloc)(FN(UNION,get_ctx(u)), n);
	if (FN(UNION,foreach_inplace)(u, &FN(UNION,add_to_list), &list) < 0)
		return FN(LIST(PART),free)(list);

	return list;
}
