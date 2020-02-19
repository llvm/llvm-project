/*
 * Copyright 2010      INRIA Saclay
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, INRIA Saclay - Ile-de-France,
 * Parc Club Orsay Universite, ZAC des vignes, 4 rue Jacques Monod,
 * 91893 Orsay, France
 */

#include <isl_ctx_private.h>
#include <isl/id.h>
#include <isl_space_private.h>
#include <isl_reordering.h>

__isl_give isl_reordering *isl_reordering_alloc(isl_ctx *ctx, int len)
{
	isl_reordering *exp;

	exp = isl_alloc(ctx, struct isl_reordering,
			sizeof(struct isl_reordering) + (len - 1) * sizeof(int));
	if (!exp)
		return NULL;

	exp->ref = 1;
	exp->len = len;
	exp->space = NULL;

	return exp;
}

__isl_give isl_reordering *isl_reordering_copy(__isl_keep isl_reordering *exp)
{
	if (!exp)
		return NULL;

	exp->ref++;
	return exp;
}

__isl_give isl_reordering *isl_reordering_dup(__isl_keep isl_reordering *r)
{
	int i;
	isl_reordering *dup;

	if (!r)
		return NULL;

	dup = isl_reordering_alloc(isl_reordering_get_ctx(r), r->len);
	if (!dup)
		return NULL;

	dup->space = isl_reordering_get_space(r);
	if (!dup->space)
		return isl_reordering_free(dup);
	for (i = 0; i < dup->len; ++i)
		dup->pos[i] = r->pos[i];

	return dup;
}

__isl_give isl_reordering *isl_reordering_cow(__isl_take isl_reordering *r)
{
	if (!r)
		return NULL;

	if (r->ref == 1)
		return r;
	r->ref--;
	return isl_reordering_dup(r);
}

__isl_null isl_reordering *isl_reordering_free(__isl_take isl_reordering *exp)
{
	if (!exp)
		return NULL;

	if (--exp->ref > 0)
		return NULL;

	isl_space_free(exp->space);
	free(exp);
	return NULL;
}

/* Return the isl_ctx to which "r" belongs.
 */
isl_ctx *isl_reordering_get_ctx(__isl_keep isl_reordering *r)
{
	return isl_space_get_ctx(isl_reordering_peek_space(r));
}

/* Return the space of "r".
 */
__isl_keep isl_space *isl_reordering_peek_space(__isl_keep isl_reordering *r)
{
	if (!r)
		return NULL;
	return r->space;
}

/* Return a copy of the space of "r".
 */
__isl_give isl_space *isl_reordering_get_space(__isl_keep isl_reordering *r)
{
	return isl_space_copy(isl_reordering_peek_space(r));
}

/* Construct a reordering that maps the parameters of "alignee"
 * to the corresponding parameters in a new dimension specification
 * that has the parameters of "aligner" first, followed by
 * any remaining parameters of "alignee" that do not occur in "aligner".
 */
__isl_give isl_reordering *isl_parameter_alignment_reordering(
	__isl_keep isl_space *alignee, __isl_keep isl_space *aligner)
{
	int i, j;
	isl_reordering *exp;

	if (!alignee || !aligner)
		return NULL;

	exp = isl_reordering_alloc(alignee->ctx, alignee->nparam);
	if (!exp)
		return NULL;

	exp->space = isl_space_params(isl_space_copy(aligner));

	for (i = 0; i < alignee->nparam; ++i) {
		isl_id *id_i;
		id_i = isl_space_get_dim_id(alignee, isl_dim_param, i);
		if (!id_i)
			isl_die(alignee->ctx, isl_error_invalid,
				"cannot align unnamed parameters", goto error);
		for (j = 0; j < aligner->nparam; ++j) {
			isl_id *id_j;
			id_j = isl_space_get_dim_id(aligner, isl_dim_param, j);
			isl_id_free(id_j);
			if (id_i == id_j)
				break;
		}
		if (j < aligner->nparam) {
			exp->pos[i] = j;
			isl_id_free(id_i);
		} else {
			isl_size pos;
			pos = isl_space_dim(exp->space, isl_dim_param);
			if (pos < 0)
				exp->space = isl_space_free(exp->space);
			exp->space = isl_space_add_dims(exp->space,
						isl_dim_param, 1);
			exp->space = isl_space_set_dim_id(exp->space,
						isl_dim_param, pos, id_i);
			exp->pos[i] = pos;
		}
	}

	if (!exp->space)
		return isl_reordering_free(exp);
	return exp;
error:
	isl_reordering_free(exp);
	return NULL;
}

/* Return a reordering that moves the parameters identified by
 * the elements of "tuple" to a domain tuple inserted into "space".
 * The parameters that remain, are moved from their original positions
 * in the list of parameters to their new positions in this list.
 * The parameters that get removed, are moved to the corresponding
 * positions in the new domain.  Note that these set dimensions
 * do not necessarily need to appear as parameters in "space".
 * Any other dimensions are shifted by the number of extra dimensions
 * introduced, i.e., the number of dimensions in the new domain
 * that did not appear as parameters in "space".
 */
__isl_give isl_reordering *isl_reordering_unbind_params_insert_domain(
	__isl_keep isl_space *space, __isl_keep isl_multi_id *tuple)
{
	int i, n;
	int offset, first;
	isl_ctx *ctx;
	isl_reordering *r;

	if (!space || !tuple)
		return NULL;

	ctx = isl_space_get_ctx(space);
	r = isl_reordering_alloc(ctx, isl_space_dim(space, isl_dim_all));
	if (!r)
		return NULL;

	r->space = isl_space_copy(space);
	r->space = isl_space_unbind_params_insert_domain(r->space, tuple);
	if (!r->space)
		return isl_reordering_free(r);

	n = isl_space_dim(r->space, isl_dim_param);
	for (i = 0; i < n; ++i) {
		int pos;
		isl_id *id;

		id = isl_space_get_dim_id(r->space, isl_dim_param, i);
		if (!id)
			return isl_reordering_free(r);
		pos = isl_space_find_dim_by_id(space, isl_dim_param, id);
		isl_id_free(id);
		r->pos[pos] = i;
	}

	offset = isl_space_dim(r->space, isl_dim_param);
	n = isl_multi_id_size(tuple);
	for (i = 0; i < n; ++i) {
		int pos;
		isl_id *id;

		id = isl_multi_id_get_id(tuple, i);
		if (!id)
			return isl_reordering_free(r);
		pos = isl_space_find_dim_by_id(space, isl_dim_param, id);
		isl_id_free(id);
		if (pos < 0)
			continue;
		r->pos[pos] = offset + i;
	}

	offset = isl_space_dim(r->space, isl_dim_all) - r->len;
	first = isl_space_dim(space, isl_dim_param);
	n = r->len - first;
	for (i = 0; i < n; ++i)
		r->pos[first + i] = first + offset + i;

	return r;
}

__isl_give isl_reordering *isl_reordering_extend(__isl_take isl_reordering *exp,
	unsigned extra)
{
	int i;
	isl_ctx *ctx;
	isl_space *space;
	isl_reordering *res;
	int offset;
	isl_size dim;

	if (!exp)
		return NULL;
	if (extra == 0)
		return exp;

	ctx = isl_reordering_get_ctx(exp);
	space = isl_reordering_peek_space(exp);
	dim = isl_space_dim(space, isl_dim_all);
	if (dim < 0)
		return isl_reordering_free(exp);
	offset = dim - exp->len;
	res = isl_reordering_alloc(ctx, exp->len + extra);
	if (!res)
		goto error;
	res->space = isl_reordering_get_space(exp);
	for (i = 0; i < exp->len; ++i)
		res->pos[i] = exp->pos[i];
	for (i = exp->len; i < res->len; ++i)
		res->pos[i] = offset + i;

	isl_reordering_free(exp);

	return res;
error:
	isl_reordering_free(exp);
	return NULL;
}

__isl_give isl_reordering *isl_reordering_extend_space(
	__isl_take isl_reordering *exp, __isl_take isl_space *space)
{
	isl_space *exp_space;
	isl_reordering *res;
	isl_size dim;

	dim = isl_space_dim(space, isl_dim_all);
	if (!exp || dim < 0)
		goto error;

	res = isl_reordering_extend(isl_reordering_copy(exp), dim - exp->len);
	res = isl_reordering_cow(res);
	if (!res)
		goto error;
	isl_space_free(res->space);
	exp_space = isl_reordering_peek_space(exp);
	res->space = isl_space_replace_params(space, exp_space);

	isl_reordering_free(exp);

	if (!res->space)
		return isl_reordering_free(res);

	return res;
error:
	isl_reordering_free(exp);
	isl_space_free(space);
	return NULL;
}

void isl_reordering_dump(__isl_keep isl_reordering *exp)
{
	int i;

	isl_space_dump(exp->space);
	for (i = 0; i < exp->len; ++i)
		fprintf(stderr, "%d -> %d; ", i, exp->pos[i]);
	fprintf(stderr, "\n");
}
