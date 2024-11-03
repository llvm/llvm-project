/*
 * Copyright 2010      INRIA Saclay
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, INRIA Saclay - Ile-de-France,
 * Parc Club Orsay Universite, ZAC des vignes, 4 rue Jacques Monod,
 * 91893 Orsay, France 
 */

#include <stdlib.h>
#include <isl_ctx_private.h>
#include <isl_map_private.h>
#include <isl_factorization.h>
#include <isl_lp_private.h>
#include <isl_seq.h>
#include <isl_union_map_private.h>
#include <isl_constraint_private.h>
#include <isl_polynomial_private.h>
#include <isl_point_private.h>
#include <isl_space_private.h>
#include <isl_mat_private.h>
#include <isl_vec_private.h>
#include <isl_range.h>
#include <isl_local.h>
#include <isl_local_space_private.h>
#include <isl_aff_private.h>
#include <isl_val_private.h>
#include <isl_config.h>

#undef EL_BASE
#define EL_BASE qpolynomial

#include <isl_list_templ.c>

#undef EL_BASE
#define EL_BASE pw_qpolynomial

#include <isl_list_templ.c>

static unsigned pos(__isl_keep isl_space *space, enum isl_dim_type type)
{
	switch (type) {
	case isl_dim_param:	return 0;
	case isl_dim_in:	return space->nparam;
	case isl_dim_out:	return space->nparam + space->n_in;
	default:		return 0;
	}
}

isl_bool isl_poly_is_cst(__isl_keep isl_poly *poly)
{
	if (!poly)
		return isl_bool_error;

	return isl_bool_ok(poly->var < 0);
}

__isl_keep isl_poly_cst *isl_poly_as_cst(__isl_keep isl_poly *poly)
{
	if (!poly)
		return NULL;

	isl_assert(poly->ctx, poly->var < 0, return NULL);

	return (isl_poly_cst *) poly;
}

__isl_keep isl_poly_rec *isl_poly_as_rec(__isl_keep isl_poly *poly)
{
	if (!poly)
		return NULL;

	isl_assert(poly->ctx, poly->var >= 0, return NULL);

	return (isl_poly_rec *) poly;
}

/* Compare two polynomials.
 *
 * Return -1 if "poly1" is "smaller" than "poly2", 1 if "poly1" is "greater"
 * than "poly2" and 0 if they are equal.
 */
static int isl_poly_plain_cmp(__isl_keep isl_poly *poly1,
	__isl_keep isl_poly *poly2)
{
	int i;
	isl_bool is_cst1;
	isl_poly_rec *rec1, *rec2;

	if (poly1 == poly2)
		return 0;
	is_cst1 = isl_poly_is_cst(poly1);
	if (is_cst1 < 0)
		return -1;
	if (!poly2)
		return 1;
	if (poly1->var != poly2->var)
		return poly1->var - poly2->var;

	if (is_cst1) {
		isl_poly_cst *cst1, *cst2;
		int cmp;

		cst1 = isl_poly_as_cst(poly1);
		cst2 = isl_poly_as_cst(poly2);
		if (!cst1 || !cst2)
			return 0;
		cmp = isl_int_cmp(cst1->n, cst2->n);
		if (cmp != 0)
			return cmp;
		return isl_int_cmp(cst1->d, cst2->d);
	}

	rec1 = isl_poly_as_rec(poly1);
	rec2 = isl_poly_as_rec(poly2);
	if (!rec1 || !rec2)
		return 0;

	if (rec1->n != rec2->n)
		return rec1->n - rec2->n;

	for (i = 0; i < rec1->n; ++i) {
		int cmp = isl_poly_plain_cmp(rec1->p[i], rec2->p[i]);
		if (cmp != 0)
			return cmp;
	}

	return 0;
}

isl_bool isl_poly_is_equal(__isl_keep isl_poly *poly1,
	__isl_keep isl_poly *poly2)
{
	int i;
	isl_bool is_cst1;
	isl_poly_rec *rec1, *rec2;

	is_cst1 = isl_poly_is_cst(poly1);
	if (is_cst1 < 0 || !poly2)
		return isl_bool_error;
	if (poly1 == poly2)
		return isl_bool_true;
	if (poly1->var != poly2->var)
		return isl_bool_false;
	if (is_cst1) {
		isl_poly_cst *cst1, *cst2;
		int r;
		cst1 = isl_poly_as_cst(poly1);
		cst2 = isl_poly_as_cst(poly2);
		if (!cst1 || !cst2)
			return isl_bool_error;
		r = isl_int_eq(cst1->n, cst2->n) &&
		    isl_int_eq(cst1->d, cst2->d);
		return isl_bool_ok(r);
	}

	rec1 = isl_poly_as_rec(poly1);
	rec2 = isl_poly_as_rec(poly2);
	if (!rec1 || !rec2)
		return isl_bool_error;

	if (rec1->n != rec2->n)
		return isl_bool_false;

	for (i = 0; i < rec1->n; ++i) {
		isl_bool eq = isl_poly_is_equal(rec1->p[i], rec2->p[i]);
		if (eq < 0 || !eq)
			return eq;
	}

	return isl_bool_true;
}

isl_bool isl_poly_is_zero(__isl_keep isl_poly *poly)
{
	isl_bool is_cst;
	isl_poly_cst *cst;

	is_cst = isl_poly_is_cst(poly);
	if (is_cst < 0 || !is_cst)
		return is_cst;

	cst = isl_poly_as_cst(poly);
	if (!cst)
		return isl_bool_error;

	return isl_bool_ok(isl_int_is_zero(cst->n) && isl_int_is_pos(cst->d));
}

int isl_poly_sgn(__isl_keep isl_poly *poly)
{
	isl_bool is_cst;
	isl_poly_cst *cst;

	is_cst = isl_poly_is_cst(poly);
	if (is_cst < 0 || !is_cst)
		return 0;

	cst = isl_poly_as_cst(poly);
	if (!cst)
		return 0;

	return isl_int_sgn(cst->n);
}

isl_bool isl_poly_is_nan(__isl_keep isl_poly *poly)
{
	isl_bool is_cst;
	isl_poly_cst *cst;

	is_cst = isl_poly_is_cst(poly);
	if (is_cst < 0 || !is_cst)
		return is_cst;

	cst = isl_poly_as_cst(poly);
	if (!cst)
		return isl_bool_error;

	return isl_bool_ok(isl_int_is_zero(cst->n) && isl_int_is_zero(cst->d));
}

isl_bool isl_poly_is_infty(__isl_keep isl_poly *poly)
{
	isl_bool is_cst;
	isl_poly_cst *cst;

	is_cst = isl_poly_is_cst(poly);
	if (is_cst < 0 || !is_cst)
		return is_cst;

	cst = isl_poly_as_cst(poly);
	if (!cst)
		return isl_bool_error;

	return isl_bool_ok(isl_int_is_pos(cst->n) && isl_int_is_zero(cst->d));
}

isl_bool isl_poly_is_neginfty(__isl_keep isl_poly *poly)
{
	isl_bool is_cst;
	isl_poly_cst *cst;

	is_cst = isl_poly_is_cst(poly);
	if (is_cst < 0 || !is_cst)
		return is_cst;

	cst = isl_poly_as_cst(poly);
	if (!cst)
		return isl_bool_error;

	return isl_bool_ok(isl_int_is_neg(cst->n) && isl_int_is_zero(cst->d));
}

isl_bool isl_poly_is_one(__isl_keep isl_poly *poly)
{
	isl_bool is_cst;
	isl_poly_cst *cst;
	int r;

	is_cst = isl_poly_is_cst(poly);
	if (is_cst < 0 || !is_cst)
		return is_cst;

	cst = isl_poly_as_cst(poly);
	if (!cst)
		return isl_bool_error;

	r = isl_int_eq(cst->n, cst->d) && isl_int_is_pos(cst->d);
	return isl_bool_ok(r);
}

isl_bool isl_poly_is_negone(__isl_keep isl_poly *poly)
{
	isl_bool is_cst;
	isl_poly_cst *cst;

	is_cst = isl_poly_is_cst(poly);
	if (is_cst < 0 || !is_cst)
		return is_cst;

	cst = isl_poly_as_cst(poly);
	if (!cst)
		return isl_bool_error;

	return isl_bool_ok(isl_int_is_negone(cst->n) && isl_int_is_one(cst->d));
}

__isl_give isl_poly_cst *isl_poly_cst_alloc(isl_ctx *ctx)
{
	isl_poly_cst *cst;

	cst = isl_alloc_type(ctx, struct isl_poly_cst);
	if (!cst)
		return NULL;

	cst->poly.ref = 1;
	cst->poly.ctx = ctx;
	isl_ctx_ref(ctx);
	cst->poly.var = -1;

	isl_int_init(cst->n);
	isl_int_init(cst->d);

	return cst;
}

__isl_give isl_poly *isl_poly_zero(isl_ctx *ctx)
{
	isl_poly_cst *cst;

	cst = isl_poly_cst_alloc(ctx);
	if (!cst)
		return NULL;

	isl_int_set_si(cst->n, 0);
	isl_int_set_si(cst->d, 1);

	return &cst->poly;
}

__isl_give isl_poly *isl_poly_one(isl_ctx *ctx)
{
	isl_poly_cst *cst;

	cst = isl_poly_cst_alloc(ctx);
	if (!cst)
		return NULL;

	isl_int_set_si(cst->n, 1);
	isl_int_set_si(cst->d, 1);

	return &cst->poly;
}

__isl_give isl_poly *isl_poly_infty(isl_ctx *ctx)
{
	isl_poly_cst *cst;

	cst = isl_poly_cst_alloc(ctx);
	if (!cst)
		return NULL;

	isl_int_set_si(cst->n, 1);
	isl_int_set_si(cst->d, 0);

	return &cst->poly;
}

__isl_give isl_poly *isl_poly_neginfty(isl_ctx *ctx)
{
	isl_poly_cst *cst;

	cst = isl_poly_cst_alloc(ctx);
	if (!cst)
		return NULL;

	isl_int_set_si(cst->n, -1);
	isl_int_set_si(cst->d, 0);

	return &cst->poly;
}

__isl_give isl_poly *isl_poly_nan(isl_ctx *ctx)
{
	isl_poly_cst *cst;

	cst = isl_poly_cst_alloc(ctx);
	if (!cst)
		return NULL;

	isl_int_set_si(cst->n, 0);
	isl_int_set_si(cst->d, 0);

	return &cst->poly;
}

__isl_give isl_poly *isl_poly_rat_cst(isl_ctx *ctx, isl_int n, isl_int d)
{
	isl_poly_cst *cst;

	cst = isl_poly_cst_alloc(ctx);
	if (!cst)
		return NULL;

	isl_int_set(cst->n, n);
	isl_int_set(cst->d, d);

	return &cst->poly;
}

__isl_give isl_poly_rec *isl_poly_alloc_rec(isl_ctx *ctx, int var, int size)
{
	isl_poly_rec *rec;

	isl_assert(ctx, var >= 0, return NULL);
	isl_assert(ctx, size >= 0, return NULL);
	rec = isl_calloc(ctx, struct isl_poly_rec,
			sizeof(struct isl_poly_rec) +
			size * sizeof(struct isl_poly *));
	if (!rec)
		return NULL;

	rec->poly.ref = 1;
	rec->poly.ctx = ctx;
	isl_ctx_ref(ctx);
	rec->poly.var = var;

	rec->n = 0;
	rec->size = size;

	return rec;
}

__isl_give isl_qpolynomial *isl_qpolynomial_reset_domain_space(
	__isl_take isl_qpolynomial *qp, __isl_take isl_space *space)
{
	qp = isl_qpolynomial_cow(qp);
	if (!qp || !space)
		goto error;

	isl_space_free(qp->dim);
	qp->dim = space;

	return qp;
error:
	isl_qpolynomial_free(qp);
	isl_space_free(space);
	return NULL;
}

/* Reset the space of "qp".  This function is called from isl_pw_templ.c
 * and doesn't know if the space of an element object is represented
 * directly or through its domain.  It therefore passes along both.
 */
__isl_give isl_qpolynomial *isl_qpolynomial_reset_space_and_domain(
	__isl_take isl_qpolynomial *qp, __isl_take isl_space *space,
	__isl_take isl_space *domain)
{
	isl_space_free(space);
	return isl_qpolynomial_reset_domain_space(qp, domain);
}

isl_ctx *isl_qpolynomial_get_ctx(__isl_keep isl_qpolynomial *qp)
{
	return qp ? qp->dim->ctx : NULL;
}

/* Return the domain space of "qp".
 */
static __isl_keep isl_space *isl_qpolynomial_peek_domain_space(
	__isl_keep isl_qpolynomial *qp)
{
	return qp ? qp->dim : NULL;
}

/* Return a copy of the domain space of "qp".
 */
__isl_give isl_space *isl_qpolynomial_get_domain_space(
	__isl_keep isl_qpolynomial *qp)
{
	return isl_space_copy(isl_qpolynomial_peek_domain_space(qp));
}

#undef TYPE
#define TYPE		isl_qpolynomial
#undef PEEK_SPACE
#define PEEK_SPACE	peek_domain_space

static
#include "isl_type_has_equal_space_bin_templ.c"
static
#include "isl_type_check_equal_space_templ.c"

#undef PEEK_SPACE

/* Return a copy of the local space on which "qp" is defined.
 */
static __isl_give isl_local_space *isl_qpolynomial_get_domain_local_space(
	__isl_keep isl_qpolynomial *qp)
{
	isl_space *space;

	if (!qp)
		return NULL;

	space = isl_qpolynomial_get_domain_space(qp);
	return isl_local_space_alloc_div(space, isl_mat_copy(qp->div));
}

__isl_give isl_space *isl_qpolynomial_get_space(__isl_keep isl_qpolynomial *qp)
{
	isl_space *space;
	if (!qp)
		return NULL;
	space = isl_space_copy(qp->dim);
	space = isl_space_from_domain(space);
	space = isl_space_add_dims(space, isl_dim_out, 1);
	return space;
}

/* Return the number of variables of the given type in the domain of "qp".
 */
isl_size isl_qpolynomial_domain_dim(__isl_keep isl_qpolynomial *qp,
	enum isl_dim_type type)
{
	isl_space *space;
	isl_size dim;

	space = isl_qpolynomial_peek_domain_space(qp);

	if (!space)
		return isl_size_error;
	if (type == isl_dim_div)
		return qp->div->n_row;
	dim = isl_space_dim(space, type);
	if (dim < 0)
		return isl_size_error;
	if (type == isl_dim_all) {
		isl_size n_div;

		n_div = isl_qpolynomial_domain_dim(qp, isl_dim_div);
		if (n_div < 0)
			return isl_size_error;
		dim += n_div;
	}
	return dim;
}

/* Given the type of a dimension of an isl_qpolynomial,
 * return the type of the corresponding dimension in its domain.
 * This function is only called for "type" equal to isl_dim_in or
 * isl_dim_param.
 */
static enum isl_dim_type domain_type(enum isl_dim_type type)
{
	return type == isl_dim_in ? isl_dim_set : type;
}

/* Externally, an isl_qpolynomial has a map space, but internally, the
 * ls field corresponds to the domain of that space.
 */
isl_size isl_qpolynomial_dim(__isl_keep isl_qpolynomial *qp,
	enum isl_dim_type type)
{
	if (!qp)
		return isl_size_error;
	if (type == isl_dim_out)
		return 1;
	type = domain_type(type);
	return isl_qpolynomial_domain_dim(qp, type);
}

/* Return the offset of the first variable of type "type" within
 * the variables of the domain of "qp".
 */
static isl_size isl_qpolynomial_domain_var_offset(
	__isl_keep isl_qpolynomial *qp, enum isl_dim_type type)
{
	isl_space *space;

	space = isl_qpolynomial_peek_domain_space(qp);
	if (!space)
		return isl_size_error;

	switch (type) {
	case isl_dim_param:
	case isl_dim_set:	return isl_space_offset(space, type);
	case isl_dim_div:	return isl_space_dim(space, isl_dim_all);
	case isl_dim_cst:
	default:
		isl_die(isl_qpolynomial_get_ctx(qp), isl_error_invalid,
			"invalid dimension type", return isl_size_error);
	}
}

/* Return the offset of the first coefficient of type "type" in
 * the domain of "qp".
 */
unsigned isl_qpolynomial_domain_offset(__isl_keep isl_qpolynomial *qp,
	enum isl_dim_type type)
{
	switch (type) {
	case isl_dim_cst:
		return 0;
	case isl_dim_param:
	case isl_dim_set:
	case isl_dim_div:
		return 1 + isl_qpolynomial_domain_var_offset(qp, type);
	default:
		return 0;
	}
}

isl_bool isl_qpolynomial_is_zero(__isl_keep isl_qpolynomial *qp)
{
	return qp ? isl_poly_is_zero(qp->poly) : isl_bool_error;
}

isl_bool isl_qpolynomial_is_one(__isl_keep isl_qpolynomial *qp)
{
	return qp ? isl_poly_is_one(qp->poly) : isl_bool_error;
}

isl_bool isl_qpolynomial_is_nan(__isl_keep isl_qpolynomial *qp)
{
	return qp ? isl_poly_is_nan(qp->poly) : isl_bool_error;
}

isl_bool isl_qpolynomial_is_infty(__isl_keep isl_qpolynomial *qp)
{
	return qp ? isl_poly_is_infty(qp->poly) : isl_bool_error;
}

isl_bool isl_qpolynomial_is_neginfty(__isl_keep isl_qpolynomial *qp)
{
	return qp ? isl_poly_is_neginfty(qp->poly) : isl_bool_error;
}

int isl_qpolynomial_sgn(__isl_keep isl_qpolynomial *qp)
{
	return qp ? isl_poly_sgn(qp->poly) : 0;
}

static void poly_free_cst(__isl_take isl_poly_cst *cst)
{
	isl_int_clear(cst->n);
	isl_int_clear(cst->d);
}

static void poly_free_rec(__isl_take isl_poly_rec *rec)
{
	int i;

	for (i = 0; i < rec->n; ++i)
		isl_poly_free(rec->p[i]);
}

__isl_give isl_poly *isl_poly_copy(__isl_keep isl_poly *poly)
{
	if (!poly)
		return NULL;

	poly->ref++;
	return poly;
}

__isl_give isl_poly *isl_poly_dup_cst(__isl_keep isl_poly *poly)
{
	isl_poly_cst *cst;
	isl_poly_cst *dup;

	cst = isl_poly_as_cst(poly);
	if (!cst)
		return NULL;

	dup = isl_poly_as_cst(isl_poly_zero(poly->ctx));
	if (!dup)
		return NULL;
	isl_int_set(dup->n, cst->n);
	isl_int_set(dup->d, cst->d);

	return &dup->poly;
}

__isl_give isl_poly *isl_poly_dup_rec(__isl_keep isl_poly *poly)
{
	int i;
	isl_poly_rec *rec;
	isl_poly_rec *dup;

	rec = isl_poly_as_rec(poly);
	if (!rec)
		return NULL;

	dup = isl_poly_alloc_rec(poly->ctx, poly->var, rec->n);
	if (!dup)
		return NULL;

	for (i = 0; i < rec->n; ++i) {
		dup->p[i] = isl_poly_copy(rec->p[i]);
		if (!dup->p[i])
			goto error;
		dup->n++;
	}

	return &dup->poly;
error:
	isl_poly_free(&dup->poly);
	return NULL;
}

__isl_give isl_poly *isl_poly_dup(__isl_keep isl_poly *poly)
{
	isl_bool is_cst;

	is_cst = isl_poly_is_cst(poly);
	if (is_cst < 0)
		return NULL;
	if (is_cst)
		return isl_poly_dup_cst(poly);
	else
		return isl_poly_dup_rec(poly);
}

__isl_give isl_poly *isl_poly_cow(__isl_take isl_poly *poly)
{
	if (!poly)
		return NULL;

	if (poly->ref == 1)
		return poly;
	poly->ref--;
	return isl_poly_dup(poly);
}

__isl_null isl_poly *isl_poly_free(__isl_take isl_poly *poly)
{
	if (!poly)
		return NULL;

	if (--poly->ref > 0)
		return NULL;

	if (poly->var < 0)
		poly_free_cst((isl_poly_cst *) poly);
	else
		poly_free_rec((isl_poly_rec *) poly);

	isl_ctx_deref(poly->ctx);
	free(poly);
	return NULL;
}

static void isl_poly_cst_reduce(__isl_keep isl_poly_cst *cst)
{
	isl_int gcd;

	isl_int_init(gcd);
	isl_int_gcd(gcd, cst->n, cst->d);
	if (!isl_int_is_zero(gcd) && !isl_int_is_one(gcd)) {
		isl_int_divexact(cst->n, cst->n, gcd);
		isl_int_divexact(cst->d, cst->d, gcd);
	}
	isl_int_clear(gcd);
}

__isl_give isl_poly *isl_poly_sum_cst(__isl_take isl_poly *poly1,
	__isl_take isl_poly *poly2)
{
	isl_poly_cst *cst1;
	isl_poly_cst *cst2;

	poly1 = isl_poly_cow(poly1);
	if (!poly1 || !poly2)
		goto error;

	cst1 = isl_poly_as_cst(poly1);
	cst2 = isl_poly_as_cst(poly2);

	if (isl_int_eq(cst1->d, cst2->d))
		isl_int_add(cst1->n, cst1->n, cst2->n);
	else {
		isl_int_mul(cst1->n, cst1->n, cst2->d);
		isl_int_addmul(cst1->n, cst2->n, cst1->d);
		isl_int_mul(cst1->d, cst1->d, cst2->d);
	}

	isl_poly_cst_reduce(cst1);

	isl_poly_free(poly2);
	return poly1;
error:
	isl_poly_free(poly1);
	isl_poly_free(poly2);
	return NULL;
}

static __isl_give isl_poly *replace_by_zero(__isl_take isl_poly *poly)
{
	struct isl_ctx *ctx;

	if (!poly)
		return NULL;
	ctx = poly->ctx;
	isl_poly_free(poly);
	return isl_poly_zero(ctx);
}

static __isl_give isl_poly *replace_by_constant_term(__isl_take isl_poly *poly)
{
	isl_poly_rec *rec;
	isl_poly *cst;

	if (!poly)
		return NULL;

	rec = isl_poly_as_rec(poly);
	if (!rec)
		goto error;
	cst = isl_poly_copy(rec->p[0]);
	isl_poly_free(poly);
	return cst;
error:
	isl_poly_free(poly);
	return NULL;
}

__isl_give isl_poly *isl_poly_sum(__isl_take isl_poly *poly1,
	__isl_take isl_poly *poly2)
{
	int i;
	isl_bool is_zero, is_nan, is_cst;
	isl_poly_rec *rec1, *rec2;

	if (!poly1 || !poly2)
		goto error;

	is_nan = isl_poly_is_nan(poly1);
	if (is_nan < 0)
		goto error;
	if (is_nan) {
		isl_poly_free(poly2);
		return poly1;
	}

	is_nan = isl_poly_is_nan(poly2);
	if (is_nan < 0)
		goto error;
	if (is_nan) {
		isl_poly_free(poly1);
		return poly2;
	}

	is_zero = isl_poly_is_zero(poly1);
	if (is_zero < 0)
		goto error;
	if (is_zero) {
		isl_poly_free(poly1);
		return poly2;
	}

	is_zero = isl_poly_is_zero(poly2);
	if (is_zero < 0)
		goto error;
	if (is_zero) {
		isl_poly_free(poly2);
		return poly1;
	}

	if (poly1->var < poly2->var)
		return isl_poly_sum(poly2, poly1);

	if (poly2->var < poly1->var) {
		isl_poly_rec *rec;
		isl_bool is_infty;

		is_infty = isl_poly_is_infty(poly2);
		if (is_infty >= 0 && !is_infty)
			is_infty = isl_poly_is_neginfty(poly2);
		if (is_infty < 0)
			goto error;
		if (is_infty) {
			isl_poly_free(poly1);
			return poly2;
		}
		poly1 = isl_poly_cow(poly1);
		rec = isl_poly_as_rec(poly1);
		if (!rec)
			goto error;
		rec->p[0] = isl_poly_sum(rec->p[0], poly2);
		if (rec->n == 1)
			poly1 = replace_by_constant_term(poly1);
		return poly1;
	}

	is_cst = isl_poly_is_cst(poly1);
	if (is_cst < 0)
		goto error;
	if (is_cst)
		return isl_poly_sum_cst(poly1, poly2);

	rec1 = isl_poly_as_rec(poly1);
	rec2 = isl_poly_as_rec(poly2);
	if (!rec1 || !rec2)
		goto error;

	if (rec1->n < rec2->n)
		return isl_poly_sum(poly2, poly1);

	poly1 = isl_poly_cow(poly1);
	rec1 = isl_poly_as_rec(poly1);
	if (!rec1)
		goto error;

	for (i = rec2->n - 1; i >= 0; --i) {
		isl_bool is_zero;

		rec1->p[i] = isl_poly_sum(rec1->p[i],
					    isl_poly_copy(rec2->p[i]));
		if (!rec1->p[i])
			goto error;
		if (i != rec1->n - 1)
			continue;
		is_zero = isl_poly_is_zero(rec1->p[i]);
		if (is_zero < 0)
			goto error;
		if (is_zero) {
			isl_poly_free(rec1->p[i]);
			rec1->n--;
		}
	}

	if (rec1->n == 0)
		poly1 = replace_by_zero(poly1);
	else if (rec1->n == 1)
		poly1 = replace_by_constant_term(poly1);

	isl_poly_free(poly2);

	return poly1;
error:
	isl_poly_free(poly1);
	isl_poly_free(poly2);
	return NULL;
}

__isl_give isl_poly *isl_poly_cst_add_isl_int(__isl_take isl_poly *poly,
	isl_int v)
{
	isl_poly_cst *cst;

	poly = isl_poly_cow(poly);
	if (!poly)
		return NULL;

	cst = isl_poly_as_cst(poly);

	isl_int_addmul(cst->n, cst->d, v);

	return poly;
}

__isl_give isl_poly *isl_poly_add_isl_int(__isl_take isl_poly *poly, isl_int v)
{
	isl_bool is_cst;
	isl_poly_rec *rec;

	is_cst = isl_poly_is_cst(poly);
	if (is_cst < 0)
		return isl_poly_free(poly);
	if (is_cst)
		return isl_poly_cst_add_isl_int(poly, v);

	poly = isl_poly_cow(poly);
	rec = isl_poly_as_rec(poly);
	if (!rec)
		goto error;

	rec->p[0] = isl_poly_add_isl_int(rec->p[0], v);
	if (!rec->p[0])
		goto error;

	return poly;
error:
	isl_poly_free(poly);
	return NULL;
}

__isl_give isl_poly *isl_poly_cst_mul_isl_int(__isl_take isl_poly *poly,
	isl_int v)
{
	isl_bool is_zero;
	isl_poly_cst *cst;

	is_zero = isl_poly_is_zero(poly);
	if (is_zero < 0)
		return isl_poly_free(poly);
	if (is_zero)
		return poly;

	poly = isl_poly_cow(poly);
	if (!poly)
		return NULL;

	cst = isl_poly_as_cst(poly);

	isl_int_mul(cst->n, cst->n, v);

	return poly;
}

__isl_give isl_poly *isl_poly_mul_isl_int(__isl_take isl_poly *poly, isl_int v)
{
	int i;
	isl_bool is_cst;
	isl_poly_rec *rec;

	is_cst = isl_poly_is_cst(poly);
	if (is_cst < 0)
		return isl_poly_free(poly);
	if (is_cst)
		return isl_poly_cst_mul_isl_int(poly, v);

	poly = isl_poly_cow(poly);
	rec = isl_poly_as_rec(poly);
	if (!rec)
		goto error;

	for (i = 0; i < rec->n; ++i) {
		rec->p[i] = isl_poly_mul_isl_int(rec->p[i], v);
		if (!rec->p[i])
			goto error;
	}

	return poly;
error:
	isl_poly_free(poly);
	return NULL;
}

/* Multiply the constant polynomial "poly" by "v".
 */
static __isl_give isl_poly *isl_poly_cst_scale_val(__isl_take isl_poly *poly,
	__isl_keep isl_val *v)
{
	isl_bool is_zero;
	isl_poly_cst *cst;

	is_zero = isl_poly_is_zero(poly);
	if (is_zero < 0)
		return isl_poly_free(poly);
	if (is_zero)
		return poly;

	poly = isl_poly_cow(poly);
	if (!poly)
		return NULL;

	cst = isl_poly_as_cst(poly);

	isl_int_mul(cst->n, cst->n, v->n);
	isl_int_mul(cst->d, cst->d, v->d);
	isl_poly_cst_reduce(cst);

	return poly;
}

/* Multiply the polynomial "poly" by "v".
 */
static __isl_give isl_poly *isl_poly_scale_val(__isl_take isl_poly *poly,
	__isl_keep isl_val *v)
{
	int i;
	isl_bool is_cst;
	isl_poly_rec *rec;

	is_cst = isl_poly_is_cst(poly);
	if (is_cst < 0)
		return isl_poly_free(poly);
	if (is_cst)
		return isl_poly_cst_scale_val(poly, v);

	poly = isl_poly_cow(poly);
	rec = isl_poly_as_rec(poly);
	if (!rec)
		goto error;

	for (i = 0; i < rec->n; ++i) {
		rec->p[i] = isl_poly_scale_val(rec->p[i], v);
		if (!rec->p[i])
			goto error;
	}

	return poly;
error:
	isl_poly_free(poly);
	return NULL;
}

__isl_give isl_poly *isl_poly_mul_cst(__isl_take isl_poly *poly1,
	__isl_take isl_poly *poly2)
{
	isl_poly_cst *cst1;
	isl_poly_cst *cst2;

	poly1 = isl_poly_cow(poly1);
	if (!poly1 || !poly2)
		goto error;

	cst1 = isl_poly_as_cst(poly1);
	cst2 = isl_poly_as_cst(poly2);

	isl_int_mul(cst1->n, cst1->n, cst2->n);
	isl_int_mul(cst1->d, cst1->d, cst2->d);

	isl_poly_cst_reduce(cst1);

	isl_poly_free(poly2);
	return poly1;
error:
	isl_poly_free(poly1);
	isl_poly_free(poly2);
	return NULL;
}

__isl_give isl_poly *isl_poly_mul_rec(__isl_take isl_poly *poly1,
	__isl_take isl_poly *poly2)
{
	isl_poly_rec *rec1;
	isl_poly_rec *rec2;
	isl_poly_rec *res = NULL;
	int i, j;
	int size;

	rec1 = isl_poly_as_rec(poly1);
	rec2 = isl_poly_as_rec(poly2);
	if (!rec1 || !rec2)
		goto error;
	size = rec1->n + rec2->n - 1;
	res = isl_poly_alloc_rec(poly1->ctx, poly1->var, size);
	if (!res)
		goto error;

	for (i = 0; i < rec1->n; ++i) {
		res->p[i] = isl_poly_mul(isl_poly_copy(rec2->p[0]),
					    isl_poly_copy(rec1->p[i]));
		if (!res->p[i])
			goto error;
		res->n++;
	}
	for (; i < size; ++i) {
		res->p[i] = isl_poly_zero(poly1->ctx);
		if (!res->p[i])
			goto error;
		res->n++;
	}
	for (i = 0; i < rec1->n; ++i) {
		for (j = 1; j < rec2->n; ++j) {
			isl_poly *poly;
			poly = isl_poly_mul(isl_poly_copy(rec2->p[j]),
					    isl_poly_copy(rec1->p[i]));
			res->p[i + j] = isl_poly_sum(res->p[i + j], poly);
			if (!res->p[i + j])
				goto error;
		}
	}

	isl_poly_free(poly1);
	isl_poly_free(poly2);

	return &res->poly;
error:
	isl_poly_free(poly1);
	isl_poly_free(poly2);
	isl_poly_free(&res->poly);
	return NULL;
}

__isl_give isl_poly *isl_poly_mul(__isl_take isl_poly *poly1,
	__isl_take isl_poly *poly2)
{
	isl_bool is_zero, is_nan, is_one, is_cst;

	if (!poly1 || !poly2)
		goto error;

	is_nan = isl_poly_is_nan(poly1);
	if (is_nan < 0)
		goto error;
	if (is_nan) {
		isl_poly_free(poly2);
		return poly1;
	}

	is_nan = isl_poly_is_nan(poly2);
	if (is_nan < 0)
		goto error;
	if (is_nan) {
		isl_poly_free(poly1);
		return poly2;
	}

	is_zero = isl_poly_is_zero(poly1);
	if (is_zero < 0)
		goto error;
	if (is_zero) {
		isl_poly_free(poly2);
		return poly1;
	}

	is_zero = isl_poly_is_zero(poly2);
	if (is_zero < 0)
		goto error;
	if (is_zero) {
		isl_poly_free(poly1);
		return poly2;
	}

	is_one = isl_poly_is_one(poly1);
	if (is_one < 0)
		goto error;
	if (is_one) {
		isl_poly_free(poly1);
		return poly2;
	}

	is_one = isl_poly_is_one(poly2);
	if (is_one < 0)
		goto error;
	if (is_one) {
		isl_poly_free(poly2);
		return poly1;
	}

	if (poly1->var < poly2->var)
		return isl_poly_mul(poly2, poly1);

	if (poly2->var < poly1->var) {
		int i;
		isl_poly_rec *rec;
		isl_bool is_infty;

		is_infty = isl_poly_is_infty(poly2);
		if (is_infty >= 0 && !is_infty)
			is_infty = isl_poly_is_neginfty(poly2);
		if (is_infty < 0)
			goto error;
		if (is_infty) {
			isl_ctx *ctx = poly1->ctx;
			isl_poly_free(poly1);
			isl_poly_free(poly2);
			return isl_poly_nan(ctx);
		}
		poly1 = isl_poly_cow(poly1);
		rec = isl_poly_as_rec(poly1);
		if (!rec)
			goto error;

		for (i = 0; i < rec->n; ++i) {
			rec->p[i] = isl_poly_mul(rec->p[i],
						isl_poly_copy(poly2));
			if (!rec->p[i])
				goto error;
		}
		isl_poly_free(poly2);
		return poly1;
	}

	is_cst = isl_poly_is_cst(poly1);
	if (is_cst < 0)
		goto error;
	if (is_cst)
		return isl_poly_mul_cst(poly1, poly2);

	return isl_poly_mul_rec(poly1, poly2);
error:
	isl_poly_free(poly1);
	isl_poly_free(poly2);
	return NULL;
}

__isl_give isl_poly *isl_poly_pow(__isl_take isl_poly *poly, unsigned power)
{
	isl_poly *res;

	if (!poly)
		return NULL;
	if (power == 1)
		return poly;

	if (power % 2)
		res = isl_poly_copy(poly);
	else
		res = isl_poly_one(poly->ctx);

	while (power >>= 1) {
		poly = isl_poly_mul(poly, isl_poly_copy(poly));
		if (power % 2)
			res = isl_poly_mul(res, isl_poly_copy(poly));
	}

	isl_poly_free(poly);
	return res;
}

__isl_give isl_qpolynomial *isl_qpolynomial_alloc(__isl_take isl_space *space,
	unsigned n_div, __isl_take isl_poly *poly)
{
	struct isl_qpolynomial *qp = NULL;
	isl_size total;

	total = isl_space_dim(space, isl_dim_all);
	if (total < 0 || !poly)
		goto error;

	if (!isl_space_is_set(space))
		isl_die(isl_space_get_ctx(space), isl_error_invalid,
			"domain of polynomial should be a set", goto error);

	qp = isl_calloc_type(space->ctx, struct isl_qpolynomial);
	if (!qp)
		goto error;

	qp->ref = 1;
	qp->div = isl_mat_alloc(space->ctx, n_div, 1 + 1 + total + n_div);
	if (!qp->div)
		goto error;

	qp->dim = space;
	qp->poly = poly;

	return qp;
error:
	isl_space_free(space);
	isl_poly_free(poly);
	isl_qpolynomial_free(qp);
	return NULL;
}

__isl_give isl_qpolynomial *isl_qpolynomial_copy(__isl_keep isl_qpolynomial *qp)
{
	if (!qp)
		return NULL;

	qp->ref++;
	return qp;
}

__isl_give isl_qpolynomial *isl_qpolynomial_dup(__isl_keep isl_qpolynomial *qp)
{
	struct isl_qpolynomial *dup;

	if (!qp)
		return NULL;

	dup = isl_qpolynomial_alloc(isl_space_copy(qp->dim), qp->div->n_row,
				    isl_poly_copy(qp->poly));
	if (!dup)
		return NULL;
	isl_mat_free(dup->div);
	dup->div = isl_mat_copy(qp->div);
	if (!dup->div)
		goto error;

	return dup;
error:
	isl_qpolynomial_free(dup);
	return NULL;
}

__isl_give isl_qpolynomial *isl_qpolynomial_cow(__isl_take isl_qpolynomial *qp)
{
	if (!qp)
		return NULL;

	if (qp->ref == 1)
		return qp;
	qp->ref--;
	return isl_qpolynomial_dup(qp);
}

__isl_null isl_qpolynomial *isl_qpolynomial_free(
	__isl_take isl_qpolynomial *qp)
{
	if (!qp)
		return NULL;

	if (--qp->ref > 0)
		return NULL;

	isl_space_free(qp->dim);
	isl_mat_free(qp->div);
	isl_poly_free(qp->poly);

	free(qp);
	return NULL;
}

__isl_give isl_poly *isl_poly_var_pow(isl_ctx *ctx, int pos, int power)
{
	int i;
	isl_poly_rec *rec;
	isl_poly_cst *cst;

	rec = isl_poly_alloc_rec(ctx, pos, 1 + power);
	if (!rec)
		return NULL;
	for (i = 0; i < 1 + power; ++i) {
		rec->p[i] = isl_poly_zero(ctx);
		if (!rec->p[i])
			goto error;
		rec->n++;
	}
	cst = isl_poly_as_cst(rec->p[power]);
	isl_int_set_si(cst->n, 1);

	return &rec->poly;
error:
	isl_poly_free(&rec->poly);
	return NULL;
}

/* r array maps original positions to new positions.
 */
static __isl_give isl_poly *reorder(__isl_take isl_poly *poly, int *r)
{
	int i;
	isl_bool is_cst;
	isl_poly_rec *rec;
	isl_poly *base;
	isl_poly *res;

	is_cst = isl_poly_is_cst(poly);
	if (is_cst < 0)
		return isl_poly_free(poly);
	if (is_cst)
		return poly;

	rec = isl_poly_as_rec(poly);
	if (!rec)
		goto error;

	isl_assert(poly->ctx, rec->n >= 1, goto error);

	base = isl_poly_var_pow(poly->ctx, r[poly->var], 1);
	res = reorder(isl_poly_copy(rec->p[rec->n - 1]), r);

	for (i = rec->n - 2; i >= 0; --i) {
		res = isl_poly_mul(res, isl_poly_copy(base));
		res = isl_poly_sum(res, reorder(isl_poly_copy(rec->p[i]), r));
	}

	isl_poly_free(base);
	isl_poly_free(poly);

	return res;
error:
	isl_poly_free(poly);
	return NULL;
}

static isl_bool compatible_divs(__isl_keep isl_mat *div1,
	__isl_keep isl_mat *div2)
{
	int n_row, n_col;
	isl_bool equal;

	isl_assert(div1->ctx, div1->n_row >= div2->n_row &&
				div1->n_col >= div2->n_col,
		    return isl_bool_error);

	if (div1->n_row == div2->n_row)
		return isl_mat_is_equal(div1, div2);

	n_row = div1->n_row;
	n_col = div1->n_col;
	div1->n_row = div2->n_row;
	div1->n_col = div2->n_col;

	equal = isl_mat_is_equal(div1, div2);

	div1->n_row = n_row;
	div1->n_col = n_col;

	return equal;
}

static int cmp_row(__isl_keep isl_mat *div, int i, int j)
{
	int li, lj;

	li = isl_seq_last_non_zero(div->row[i], div->n_col);
	lj = isl_seq_last_non_zero(div->row[j], div->n_col);

	if (li != lj)
		return li - lj;

	return isl_seq_cmp(div->row[i], div->row[j], div->n_col);
}

struct isl_div_sort_info {
	isl_mat	*div;
	int	 row;
};

static int div_sort_cmp(const void *p1, const void *p2)
{
	const struct isl_div_sort_info *i1, *i2;
	i1 = (const struct isl_div_sort_info *) p1;
	i2 = (const struct isl_div_sort_info *) p2;

	return cmp_row(i1->div, i1->row, i2->row);
}

/* Sort divs and remove duplicates.
 */
static __isl_give isl_qpolynomial *sort_divs(__isl_take isl_qpolynomial *qp)
{
	int i;
	int skip;
	int len;
	struct isl_div_sort_info *array = NULL;
	int *pos = NULL, *at = NULL;
	int *reordering = NULL;
	isl_size div_pos;

	if (!qp)
		return NULL;
	if (qp->div->n_row <= 1)
		return qp;

	div_pos = isl_qpolynomial_domain_var_offset(qp, isl_dim_div);
	if (div_pos < 0)
		return isl_qpolynomial_free(qp);

	array = isl_alloc_array(qp->div->ctx, struct isl_div_sort_info,
				qp->div->n_row);
	pos = isl_alloc_array(qp->div->ctx, int, qp->div->n_row);
	at = isl_alloc_array(qp->div->ctx, int, qp->div->n_row);
	len = qp->div->n_col - 2;
	reordering = isl_alloc_array(qp->div->ctx, int, len);
	if (!array || !pos || !at || !reordering)
		goto error;

	for (i = 0; i < qp->div->n_row; ++i) {
		array[i].div = qp->div;
		array[i].row = i;
		pos[i] = i;
		at[i] = i;
	}

	qsort(array, qp->div->n_row, sizeof(struct isl_div_sort_info),
		div_sort_cmp);

	for (i = 0; i < div_pos; ++i)
		reordering[i] = i;

	for (i = 0; i < qp->div->n_row; ++i) {
		if (pos[array[i].row] == i)
			continue;
		qp->div = isl_mat_swap_rows(qp->div, i, pos[array[i].row]);
		pos[at[i]] = pos[array[i].row];
		at[pos[array[i].row]] = at[i];
		at[i] = array[i].row;
		pos[array[i].row] = i;
	}

	skip = 0;
	for (i = 0; i < len - div_pos; ++i) {
		if (i > 0 &&
		    isl_seq_eq(qp->div->row[i - skip - 1],
			       qp->div->row[i - skip], qp->div->n_col)) {
			qp->div = isl_mat_drop_rows(qp->div, i - skip, 1);
			isl_mat_col_add(qp->div, 2 + div_pos + i - skip - 1,
						 2 + div_pos + i - skip);
			qp->div = isl_mat_drop_cols(qp->div,
						    2 + div_pos + i - skip, 1);
			skip++;
		}
		reordering[div_pos + array[i].row] = div_pos + i - skip;
	}

	qp->poly = reorder(qp->poly, reordering);

	if (!qp->poly || !qp->div)
		goto error;

	free(at);
	free(pos);
	free(array);
	free(reordering);

	return qp;
error:
	free(at);
	free(pos);
	free(array);
	free(reordering);
	isl_qpolynomial_free(qp);
	return NULL;
}

static __isl_give isl_poly *expand(__isl_take isl_poly *poly, int *exp,
	int first)
{
	int i;
	isl_bool is_cst;
	isl_poly_rec *rec;

	is_cst = isl_poly_is_cst(poly);
	if (is_cst < 0)
		return isl_poly_free(poly);
	if (is_cst)
		return poly;

	if (poly->var < first)
		return poly;

	if (exp[poly->var - first] == poly->var - first)
		return poly;

	poly = isl_poly_cow(poly);
	if (!poly)
		goto error;

	poly->var = exp[poly->var - first] + first;

	rec = isl_poly_as_rec(poly);
	if (!rec)
		goto error;

	for (i = 0; i < rec->n; ++i) {
		rec->p[i] = expand(rec->p[i], exp, first);
		if (!rec->p[i])
			goto error;
	}

	return poly;
error:
	isl_poly_free(poly);
	return NULL;
}

static __isl_give isl_qpolynomial *with_merged_divs(
	__isl_give isl_qpolynomial *(*fn)(__isl_take isl_qpolynomial *qp1,
					  __isl_take isl_qpolynomial *qp2),
	__isl_take isl_qpolynomial *qp1, __isl_take isl_qpolynomial *qp2)
{
	int *exp1 = NULL;
	int *exp2 = NULL;
	isl_mat *div = NULL;
	int n_div1, n_div2;

	qp1 = isl_qpolynomial_cow(qp1);
	qp2 = isl_qpolynomial_cow(qp2);

	if (!qp1 || !qp2)
		goto error;

	isl_assert(qp1->div->ctx, qp1->div->n_row >= qp2->div->n_row &&
				qp1->div->n_col >= qp2->div->n_col, goto error);

	n_div1 = qp1->div->n_row;
	n_div2 = qp2->div->n_row;
	exp1 = isl_alloc_array(qp1->div->ctx, int, n_div1);
	exp2 = isl_alloc_array(qp2->div->ctx, int, n_div2);
	if ((n_div1 && !exp1) || (n_div2 && !exp2))
		goto error;

	div = isl_merge_divs(qp1->div, qp2->div, exp1, exp2);
	if (!div)
		goto error;

	isl_mat_free(qp1->div);
	qp1->div = isl_mat_copy(div);
	isl_mat_free(qp2->div);
	qp2->div = isl_mat_copy(div);

	qp1->poly = expand(qp1->poly, exp1, div->n_col - div->n_row - 2);
	qp2->poly = expand(qp2->poly, exp2, div->n_col - div->n_row - 2);

	if (!qp1->poly || !qp2->poly)
		goto error;

	isl_mat_free(div);
	free(exp1);
	free(exp2);

	return fn(qp1, qp2);
error:
	isl_mat_free(div);
	free(exp1);
	free(exp2);
	isl_qpolynomial_free(qp1);
	isl_qpolynomial_free(qp2);
	return NULL;
}

__isl_give isl_qpolynomial *isl_qpolynomial_add(__isl_take isl_qpolynomial *qp1,
	__isl_take isl_qpolynomial *qp2)
{
	isl_bool compatible;

	qp1 = isl_qpolynomial_cow(qp1);

	if (isl_qpolynomial_check_equal_space(qp1, qp2) < 0)
		goto error;

	if (qp1->div->n_row < qp2->div->n_row)
		return isl_qpolynomial_add(qp2, qp1);

	compatible = compatible_divs(qp1->div, qp2->div);
	if (compatible < 0)
		goto error;
	if (!compatible)
		return with_merged_divs(isl_qpolynomial_add, qp1, qp2);

	qp1->poly = isl_poly_sum(qp1->poly, isl_poly_copy(qp2->poly));
	if (!qp1->poly)
		goto error;

	isl_qpolynomial_free(qp2);

	return qp1;
error:
	isl_qpolynomial_free(qp1);
	isl_qpolynomial_free(qp2);
	return NULL;
}

__isl_give isl_qpolynomial *isl_qpolynomial_add_on_domain(
	__isl_keep isl_set *dom,
	__isl_take isl_qpolynomial *qp1,
	__isl_take isl_qpolynomial *qp2)
{
	qp1 = isl_qpolynomial_add(qp1, qp2);
	qp1 = isl_qpolynomial_gist(qp1, isl_set_copy(dom));
	return qp1;
}

__isl_give isl_qpolynomial *isl_qpolynomial_sub(__isl_take isl_qpolynomial *qp1,
	__isl_take isl_qpolynomial *qp2)
{
	return isl_qpolynomial_add(qp1, isl_qpolynomial_neg(qp2));
}

__isl_give isl_qpolynomial *isl_qpolynomial_add_isl_int(
	__isl_take isl_qpolynomial *qp, isl_int v)
{
	if (isl_int_is_zero(v))
		return qp;

	qp = isl_qpolynomial_cow(qp);
	if (!qp)
		return NULL;

	qp->poly = isl_poly_add_isl_int(qp->poly, v);
	if (!qp->poly)
		goto error;

	return qp;
error:
	isl_qpolynomial_free(qp);
	return NULL;

}

__isl_give isl_qpolynomial *isl_qpolynomial_neg(__isl_take isl_qpolynomial *qp)
{
	if (!qp)
		return NULL;

	return isl_qpolynomial_mul_isl_int(qp, qp->dim->ctx->negone);
}

__isl_give isl_qpolynomial *isl_qpolynomial_mul_isl_int(
	__isl_take isl_qpolynomial *qp, isl_int v)
{
	if (isl_int_is_one(v))
		return qp;

	if (qp && isl_int_is_zero(v)) {
		isl_qpolynomial *zero;
		zero = isl_qpolynomial_zero_on_domain(isl_space_copy(qp->dim));
		isl_qpolynomial_free(qp);
		return zero;
	}
	
	qp = isl_qpolynomial_cow(qp);
	if (!qp)
		return NULL;

	qp->poly = isl_poly_mul_isl_int(qp->poly, v);
	if (!qp->poly)
		goto error;

	return qp;
error:
	isl_qpolynomial_free(qp);
	return NULL;
}

__isl_give isl_qpolynomial *isl_qpolynomial_scale(
	__isl_take isl_qpolynomial *qp, isl_int v)
{
	return isl_qpolynomial_mul_isl_int(qp, v);
}

/* Multiply "qp" by "v".
 */
__isl_give isl_qpolynomial *isl_qpolynomial_scale_val(
	__isl_take isl_qpolynomial *qp, __isl_take isl_val *v)
{
	if (!qp || !v)
		goto error;

	if (!isl_val_is_rat(v))
		isl_die(isl_qpolynomial_get_ctx(qp), isl_error_invalid,
			"expecting rational factor", goto error);

	if (isl_val_is_one(v)) {
		isl_val_free(v);
		return qp;
	}

	if (isl_val_is_zero(v)) {
		isl_space *space;

		space = isl_qpolynomial_get_domain_space(qp);
		isl_qpolynomial_free(qp);
		isl_val_free(v);
		return isl_qpolynomial_zero_on_domain(space);
	}

	qp = isl_qpolynomial_cow(qp);
	if (!qp)
		goto error;

	qp->poly = isl_poly_scale_val(qp->poly, v);
	if (!qp->poly)
		qp = isl_qpolynomial_free(qp);

	isl_val_free(v);
	return qp;
error:
	isl_val_free(v);
	isl_qpolynomial_free(qp);
	return NULL;
}

/* Divide "qp" by "v".
 */
__isl_give isl_qpolynomial *isl_qpolynomial_scale_down_val(
	__isl_take isl_qpolynomial *qp, __isl_take isl_val *v)
{
	if (!qp || !v)
		goto error;

	if (!isl_val_is_rat(v))
		isl_die(isl_qpolynomial_get_ctx(qp), isl_error_invalid,
			"expecting rational factor", goto error);
	if (isl_val_is_zero(v))
		isl_die(isl_val_get_ctx(v), isl_error_invalid,
			"cannot scale down by zero", goto error);

	return isl_qpolynomial_scale_val(qp, isl_val_inv(v));
error:
	isl_val_free(v);
	isl_qpolynomial_free(qp);
	return NULL;
}

__isl_give isl_qpolynomial *isl_qpolynomial_mul(__isl_take isl_qpolynomial *qp1,
	__isl_take isl_qpolynomial *qp2)
{
	isl_bool compatible;

	qp1 = isl_qpolynomial_cow(qp1);

	if (isl_qpolynomial_check_equal_space(qp1, qp2) < 0)
		goto error;

	if (qp1->div->n_row < qp2->div->n_row)
		return isl_qpolynomial_mul(qp2, qp1);

	compatible = compatible_divs(qp1->div, qp2->div);
	if (compatible < 0)
		goto error;
	if (!compatible)
		return with_merged_divs(isl_qpolynomial_mul, qp1, qp2);

	qp1->poly = isl_poly_mul(qp1->poly, isl_poly_copy(qp2->poly));
	if (!qp1->poly)
		goto error;

	isl_qpolynomial_free(qp2);

	return qp1;
error:
	isl_qpolynomial_free(qp1);
	isl_qpolynomial_free(qp2);
	return NULL;
}

__isl_give isl_qpolynomial *isl_qpolynomial_pow(__isl_take isl_qpolynomial *qp,
	unsigned power)
{
	qp = isl_qpolynomial_cow(qp);

	if (!qp)
		return NULL;

	qp->poly = isl_poly_pow(qp->poly, power);
	if (!qp->poly)
		goto error;

	return qp;
error:
	isl_qpolynomial_free(qp);
	return NULL;
}

__isl_give isl_pw_qpolynomial *isl_pw_qpolynomial_pow(
	__isl_take isl_pw_qpolynomial *pwqp, unsigned power)
{
	int i;

	if (power == 1)
		return pwqp;

	pwqp = isl_pw_qpolynomial_cow(pwqp);
	if (!pwqp)
		return NULL;

	for (i = 0; i < pwqp->n; ++i) {
		pwqp->p[i].qp = isl_qpolynomial_pow(pwqp->p[i].qp, power);
		if (!pwqp->p[i].qp)
			return isl_pw_qpolynomial_free(pwqp);
	}

	return pwqp;
}

__isl_give isl_qpolynomial *isl_qpolynomial_zero_on_domain(
	__isl_take isl_space *domain)
{
	if (!domain)
		return NULL;
	return isl_qpolynomial_alloc(domain, 0, isl_poly_zero(domain->ctx));
}

__isl_give isl_qpolynomial *isl_qpolynomial_one_on_domain(
	__isl_take isl_space *domain)
{
	if (!domain)
		return NULL;
	return isl_qpolynomial_alloc(domain, 0, isl_poly_one(domain->ctx));
}

__isl_give isl_qpolynomial *isl_qpolynomial_infty_on_domain(
	__isl_take isl_space *domain)
{
	if (!domain)
		return NULL;
	return isl_qpolynomial_alloc(domain, 0, isl_poly_infty(domain->ctx));
}

__isl_give isl_qpolynomial *isl_qpolynomial_neginfty_on_domain(
	__isl_take isl_space *domain)
{
	if (!domain)
		return NULL;
	return isl_qpolynomial_alloc(domain, 0, isl_poly_neginfty(domain->ctx));
}

__isl_give isl_qpolynomial *isl_qpolynomial_nan_on_domain(
	__isl_take isl_space *domain)
{
	if (!domain)
		return NULL;
	return isl_qpolynomial_alloc(domain, 0, isl_poly_nan(domain->ctx));
}

__isl_give isl_qpolynomial *isl_qpolynomial_cst_on_domain(
	__isl_take isl_space *domain,
	isl_int v)
{
	struct isl_qpolynomial *qp;
	isl_poly_cst *cst;

	qp = isl_qpolynomial_zero_on_domain(domain);
	if (!qp)
		return NULL;

	cst = isl_poly_as_cst(qp->poly);
	isl_int_set(cst->n, v);

	return qp;
}

isl_bool isl_qpolynomial_is_cst(__isl_keep isl_qpolynomial *qp,
	isl_int *n, isl_int *d)
{
	isl_bool is_cst;
	isl_poly_cst *cst;

	if (!qp)
		return isl_bool_error;

	is_cst = isl_poly_is_cst(qp->poly);
	if (is_cst < 0 || !is_cst)
		return is_cst;

	cst = isl_poly_as_cst(qp->poly);
	if (!cst)
		return isl_bool_error;

	if (n)
		isl_int_set(*n, cst->n);
	if (d)
		isl_int_set(*d, cst->d);

	return isl_bool_true;
}

/* Return the constant term of "poly".
 */
static __isl_give isl_val *isl_poly_get_constant_val(__isl_keep isl_poly *poly)
{
	isl_bool is_cst;
	isl_poly_cst *cst;

	if (!poly)
		return NULL;

	while ((is_cst = isl_poly_is_cst(poly)) == isl_bool_false) {
		isl_poly_rec *rec;

		rec = isl_poly_as_rec(poly);
		if (!rec)
			return NULL;
		poly = rec->p[0];
	}
	if (is_cst < 0)
		return NULL;

	cst = isl_poly_as_cst(poly);
	if (!cst)
		return NULL;
	return isl_val_rat_from_isl_int(cst->poly.ctx, cst->n, cst->d);
}

/* Return the constant term of "qp".
 */
__isl_give isl_val *isl_qpolynomial_get_constant_val(
	__isl_keep isl_qpolynomial *qp)
{
	if (!qp)
		return NULL;

	return isl_poly_get_constant_val(qp->poly);
}

isl_bool isl_poly_is_affine(__isl_keep isl_poly *poly)
{
	isl_bool is_cst;
	isl_poly_rec *rec;

	if (!poly)
		return isl_bool_error;

	if (poly->var < 0)
		return isl_bool_true;

	rec = isl_poly_as_rec(poly);
	if (!rec)
		return isl_bool_error;

	if (rec->n > 2)
		return isl_bool_false;

	isl_assert(poly->ctx, rec->n > 1, return isl_bool_error);

	is_cst = isl_poly_is_cst(rec->p[1]);
	if (is_cst < 0 || !is_cst)
		return is_cst;

	return isl_poly_is_affine(rec->p[0]);
}

isl_bool isl_qpolynomial_is_affine(__isl_keep isl_qpolynomial *qp)
{
	if (!qp)
		return isl_bool_error;

	if (qp->div->n_row > 0)
		return isl_bool_false;

	return isl_poly_is_affine(qp->poly);
}

static void update_coeff(__isl_keep isl_vec *aff,
	__isl_keep isl_poly_cst *cst, int pos)
{
	isl_int gcd;
	isl_int f;

	if (isl_int_is_zero(cst->n))
		return;

	isl_int_init(gcd);
	isl_int_init(f);
	isl_int_gcd(gcd, cst->d, aff->el[0]);
	isl_int_divexact(f, cst->d, gcd);
	isl_int_divexact(gcd, aff->el[0], gcd);
	isl_seq_scale(aff->el, aff->el, f, aff->size);
	isl_int_mul(aff->el[1 + pos], gcd, cst->n);
	isl_int_clear(gcd);
	isl_int_clear(f);
}

int isl_poly_update_affine(__isl_keep isl_poly *poly, __isl_keep isl_vec *aff)
{
	isl_poly_cst *cst;
	isl_poly_rec *rec;

	if (!poly || !aff)
		return -1;

	if (poly->var < 0) {
		isl_poly_cst *cst;

		cst = isl_poly_as_cst(poly);
		if (!cst)
			return -1;
		update_coeff(aff, cst, 0);
		return 0;
	}

	rec = isl_poly_as_rec(poly);
	if (!rec)
		return -1;
	isl_assert(poly->ctx, rec->n == 2, return -1);

	cst = isl_poly_as_cst(rec->p[1]);
	if (!cst)
		return -1;
	update_coeff(aff, cst, 1 + poly->var);

	return isl_poly_update_affine(rec->p[0], aff);
}

__isl_give isl_vec *isl_qpolynomial_extract_affine(
	__isl_keep isl_qpolynomial *qp)
{
	isl_vec *aff;
	isl_size d;

	d = isl_qpolynomial_domain_dim(qp, isl_dim_all);
	if (d < 0)
		return NULL;

	aff = isl_vec_alloc(qp->div->ctx, 2 + d);
	if (!aff)
		return NULL;

	isl_seq_clr(aff->el + 1, 1 + d);
	isl_int_set_si(aff->el[0], 1);

	if (isl_poly_update_affine(qp->poly, aff) < 0)
		goto error;

	return aff;
error:
	isl_vec_free(aff);
	return NULL;
}

/* Compare two quasi-polynomials.
 *
 * Return -1 if "qp1" is "smaller" than "qp2", 1 if "qp1" is "greater"
 * than "qp2" and 0 if they are equal.
 */
int isl_qpolynomial_plain_cmp(__isl_keep isl_qpolynomial *qp1,
	__isl_keep isl_qpolynomial *qp2)
{
	int cmp;

	if (qp1 == qp2)
		return 0;
	if (!qp1)
		return -1;
	if (!qp2)
		return 1;

	cmp = isl_space_cmp(qp1->dim, qp2->dim);
	if (cmp != 0)
		return cmp;

	cmp = isl_local_cmp(qp1->div, qp2->div);
	if (cmp != 0)
		return cmp;

	return isl_poly_plain_cmp(qp1->poly, qp2->poly);
}

/* Is "qp1" obviously equal to "qp2"?
 *
 * NaN is not equal to anything, not even to another NaN.
 */
isl_bool isl_qpolynomial_plain_is_equal(__isl_keep isl_qpolynomial *qp1,
	__isl_keep isl_qpolynomial *qp2)
{
	isl_bool equal;

	if (!qp1 || !qp2)
		return isl_bool_error;

	if (isl_qpolynomial_is_nan(qp1) || isl_qpolynomial_is_nan(qp2))
		return isl_bool_false;

	equal = isl_space_is_equal(qp1->dim, qp2->dim);
	if (equal < 0 || !equal)
		return equal;

	equal = isl_mat_is_equal(qp1->div, qp2->div);
	if (equal < 0 || !equal)
		return equal;

	return isl_poly_is_equal(qp1->poly, qp2->poly);
}

static isl_stat poly_update_den(__isl_keep isl_poly *poly, isl_int *d)
{
	int i;
	isl_bool is_cst;
	isl_poly_rec *rec;

	is_cst = isl_poly_is_cst(poly);
	if (is_cst < 0)
		return isl_stat_error;
	if (is_cst) {
		isl_poly_cst *cst;
		cst = isl_poly_as_cst(poly);
		if (!cst)
			return isl_stat_error;
		isl_int_lcm(*d, *d, cst->d);
		return isl_stat_ok;
	}

	rec = isl_poly_as_rec(poly);
	if (!rec)
		return isl_stat_error;

	for (i = 0; i < rec->n; ++i)
		poly_update_den(rec->p[i], d);

	return isl_stat_ok;
}

__isl_give isl_val *isl_qpolynomial_get_den(__isl_keep isl_qpolynomial *qp)
{
	isl_val *d;

	if (!qp)
		return NULL;
	d = isl_val_one(isl_qpolynomial_get_ctx(qp));
	if (!d)
		return NULL;
	if (poly_update_den(qp->poly, &d->n) < 0)
		return isl_val_free(d);
	return d;
}

__isl_give isl_qpolynomial *isl_qpolynomial_var_pow_on_domain(
	__isl_take isl_space *domain, int pos, int power)
{
	struct isl_ctx *ctx;

	if (!domain)
		return NULL;

	ctx = domain->ctx;

	return isl_qpolynomial_alloc(domain, 0,
					isl_poly_var_pow(ctx, pos, power));
}

__isl_give isl_qpolynomial *isl_qpolynomial_var_on_domain(
	__isl_take isl_space *domain, enum isl_dim_type type, unsigned pos)
{
	if (isl_space_check_is_set(domain ) < 0)
		goto error;
	if (isl_space_check_range(domain, type, pos, 1) < 0)
		goto error;

	pos += isl_space_offset(domain, type);

	return isl_qpolynomial_var_pow_on_domain(domain, pos, 1);
error:
	isl_space_free(domain);
	return NULL;
}

__isl_give isl_poly *isl_poly_subs(__isl_take isl_poly *poly,
	unsigned first, unsigned n, __isl_keep isl_poly **subs)
{
	int i;
	isl_bool is_cst;
	isl_poly_rec *rec;
	isl_poly *base, *res;

	is_cst = isl_poly_is_cst(poly);
	if (is_cst < 0)
		return isl_poly_free(poly);
	if (is_cst)
		return poly;

	if (poly->var < first)
		return poly;

	rec = isl_poly_as_rec(poly);
	if (!rec)
		goto error;

	isl_assert(poly->ctx, rec->n >= 1, goto error);

	if (poly->var >= first + n)
		base = isl_poly_var_pow(poly->ctx, poly->var, 1);
	else
		base = isl_poly_copy(subs[poly->var - first]);

	res = isl_poly_subs(isl_poly_copy(rec->p[rec->n - 1]), first, n, subs);
	for (i = rec->n - 2; i >= 0; --i) {
		isl_poly *t;
		t = isl_poly_subs(isl_poly_copy(rec->p[i]), first, n, subs);
		res = isl_poly_mul(res, isl_poly_copy(base));
		res = isl_poly_sum(res, t);
	}

	isl_poly_free(base);
	isl_poly_free(poly);
				
	return res;
error:
	isl_poly_free(poly);
	return NULL;
}	

__isl_give isl_poly *isl_poly_from_affine(isl_ctx *ctx, isl_int *f,
	isl_int denom, unsigned len)
{
	int i;
	isl_poly *poly;

	isl_assert(ctx, len >= 1, return NULL);

	poly = isl_poly_rat_cst(ctx, f[0], denom);
	for (i = 0; i < len - 1; ++i) {
		isl_poly *t;
		isl_poly *c;

		if (isl_int_is_zero(f[1 + i]))
			continue;

		c = isl_poly_rat_cst(ctx, f[1 + i], denom);
		t = isl_poly_var_pow(ctx, i, 1);
		t = isl_poly_mul(c, t);
		poly = isl_poly_sum(poly, t);
	}

	return poly;
}

/* Remove common factor of non-constant terms and denominator.
 */
static void normalize_div(__isl_keep isl_qpolynomial *qp, int div)
{
	isl_ctx *ctx = qp->div->ctx;
	unsigned total = qp->div->n_col - 2;

	isl_seq_gcd(qp->div->row[div] + 2, total, &ctx->normalize_gcd);
	isl_int_gcd(ctx->normalize_gcd,
		    ctx->normalize_gcd, qp->div->row[div][0]);
	if (isl_int_is_one(ctx->normalize_gcd))
		return;

	isl_seq_scale_down(qp->div->row[div] + 2, qp->div->row[div] + 2,
			    ctx->normalize_gcd, total);
	isl_int_divexact(qp->div->row[div][0], qp->div->row[div][0],
			    ctx->normalize_gcd);
	isl_int_fdiv_q(qp->div->row[div][1], qp->div->row[div][1],
			    ctx->normalize_gcd);
}

/* Replace the integer division identified by "div" by the polynomial "s".
 * The integer division is assumed not to appear in the definition
 * of any other integer divisions.
 */
static __isl_give isl_qpolynomial *substitute_div(
	__isl_take isl_qpolynomial *qp, int div, __isl_take isl_poly *s)
{
	int i;
	isl_size div_pos;
	int *reordering;
	isl_ctx *ctx;

	if (!qp || !s)
		goto error;

	qp = isl_qpolynomial_cow(qp);
	if (!qp)
		goto error;

	div_pos = isl_qpolynomial_domain_var_offset(qp, isl_dim_div);
	if (div_pos < 0)
		goto error;
	qp->poly = isl_poly_subs(qp->poly, div_pos + div, 1, &s);
	if (!qp->poly)
		goto error;

	ctx = isl_qpolynomial_get_ctx(qp);
	reordering = isl_alloc_array(ctx, int, div_pos + qp->div->n_row);
	if (!reordering)
		goto error;
	for (i = 0; i < div_pos + div; ++i)
		reordering[i] = i;
	for (i = div_pos + div + 1; i < div_pos + qp->div->n_row; ++i)
		reordering[i] = i - 1;
	qp->div = isl_mat_drop_rows(qp->div, div, 1);
	qp->div = isl_mat_drop_cols(qp->div, 2 + div_pos + div, 1);
	qp->poly = reorder(qp->poly, reordering);
	free(reordering);

	if (!qp->poly || !qp->div)
		goto error;

	isl_poly_free(s);
	return qp;
error:
	isl_qpolynomial_free(qp);
	isl_poly_free(s);
	return NULL;
}

/* Replace all integer divisions [e/d] that turn out to not actually be integer
 * divisions because d is equal to 1 by their definition, i.e., e.
 */
static __isl_give isl_qpolynomial *substitute_non_divs(
	__isl_take isl_qpolynomial *qp)
{
	int i, j;
	isl_size div_pos;
	isl_poly *s;

	div_pos = isl_qpolynomial_domain_var_offset(qp, isl_dim_div);
	if (div_pos < 0)
		return isl_qpolynomial_free(qp);

	for (i = 0; qp && i < qp->div->n_row; ++i) {
		if (!isl_int_is_one(qp->div->row[i][0]))
			continue;
		for (j = i + 1; j < qp->div->n_row; ++j) {
			if (isl_int_is_zero(qp->div->row[j][2 + div_pos + i]))
				continue;
			isl_seq_combine(qp->div->row[j] + 1,
				qp->div->ctx->one, qp->div->row[j] + 1,
				qp->div->row[j][2 + div_pos + i],
				qp->div->row[i] + 1, 1 + div_pos + i);
			isl_int_set_si(qp->div->row[j][2 + div_pos + i], 0);
			normalize_div(qp, j);
		}
		s = isl_poly_from_affine(qp->dim->ctx, qp->div->row[i] + 1,
					qp->div->row[i][0], qp->div->n_col - 1);
		qp = substitute_div(qp, i, s);
		--i;
	}

	return qp;
}

/* Reduce the coefficients of div "div" to lie in the interval [0, d-1],
 * with d the denominator.  When replacing the coefficient e of x by
 * d * frac(e/d) = e - d * floor(e/d), we are subtracting d * floor(e/d) * x
 * inside the division, so we need to add floor(e/d) * x outside.
 * That is, we replace q by q' + floor(e/d) * x and we therefore need
 * to adjust the coefficient of x in each later div that depends on the
 * current div "div" and also in the affine expressions in the rows of "mat"
 * (if they too depend on "div").
 */
static void reduce_div(__isl_keep isl_qpolynomial *qp, int div,
	__isl_keep isl_mat **mat)
{
	int i, j;
	isl_int v;
	unsigned total = qp->div->n_col - qp->div->n_row - 2;

	isl_int_init(v);
	for (i = 0; i < 1 + total + div; ++i) {
		if (isl_int_is_nonneg(qp->div->row[div][1 + i]) &&
		    isl_int_lt(qp->div->row[div][1 + i], qp->div->row[div][0]))
			continue;
		isl_int_fdiv_q(v, qp->div->row[div][1 + i], qp->div->row[div][0]);
		isl_int_fdiv_r(qp->div->row[div][1 + i],
				qp->div->row[div][1 + i], qp->div->row[div][0]);
		*mat = isl_mat_col_addmul(*mat, i, v, 1 + total + div);
		for (j = div + 1; j < qp->div->n_row; ++j) {
			if (isl_int_is_zero(qp->div->row[j][2 + total + div]))
				continue;
			isl_int_addmul(qp->div->row[j][1 + i],
					v, qp->div->row[j][2 + total + div]);
		}
	}
	isl_int_clear(v);
}

/* Check if the last non-zero coefficient is bigger that half of the
 * denominator.  If so, we will invert the div to further reduce the number
 * of distinct divs that may appear.
 * If the last non-zero coefficient is exactly half the denominator,
 * then we continue looking for earlier coefficients that are bigger
 * than half the denominator.
 */
static int needs_invert(__isl_keep isl_mat *div, int row)
{
	int i;
	int cmp;

	for (i = div->n_col - 1; i >= 1; --i) {
		if (isl_int_is_zero(div->row[row][i]))
			continue;
		isl_int_mul_ui(div->row[row][i], div->row[row][i], 2);
		cmp = isl_int_cmp(div->row[row][i], div->row[row][0]);
		isl_int_divexact_ui(div->row[row][i], div->row[row][i], 2);
		if (cmp)
			return cmp > 0;
		if (i == 1)
			return 1;
	}

	return 0;
}

/* Replace div "div" q = [e/d] by -[(-e+(d-1))/d].
 * We only invert the coefficients of e (and the coefficient of q in
 * later divs and in the rows of "mat").  After calling this function, the
 * coefficients of e should be reduced again.
 */
static void invert_div(__isl_keep isl_qpolynomial *qp, int div,
	__isl_keep isl_mat **mat)
{
	unsigned total = qp->div->n_col - qp->div->n_row - 2;

	isl_seq_neg(qp->div->row[div] + 1,
		    qp->div->row[div] + 1, qp->div->n_col - 1);
	isl_int_sub_ui(qp->div->row[div][1], qp->div->row[div][1], 1);
	isl_int_add(qp->div->row[div][1],
		    qp->div->row[div][1], qp->div->row[div][0]);
	*mat = isl_mat_col_neg(*mat, 1 + total + div);
	isl_mat_col_mul(qp->div, 2 + total + div,
			qp->div->ctx->negone, 2 + total + div);
}

/* Reduce all divs of "qp" to have coefficients
 * in the interval [0, d-1], with d the denominator and such that the
 * last non-zero coefficient that is not equal to d/2 is smaller than d/2.
 * The modifications to the integer divisions need to be reflected
 * in the factors of the polynomial that refer to the original
 * integer divisions.  To this end, the modifications are collected
 * as a set of affine expressions and then plugged into the polynomial.
 *
 * After the reduction, some divs may have become redundant or identical,
 * so we call substitute_non_divs and sort_divs.  If these functions
 * eliminate divs or merge two or more divs into one, the coefficients
 * of the enclosing divs may have to be reduced again, so we call
 * ourselves recursively if the number of divs decreases.
 */
static __isl_give isl_qpolynomial *reduce_divs(__isl_take isl_qpolynomial *qp)
{
	int i;
	isl_ctx *ctx;
	isl_mat *mat;
	isl_poly **s;
	unsigned o_div;
	isl_size n_div, total, new_n_div;

	total = isl_qpolynomial_domain_dim(qp, isl_dim_all);
	n_div = isl_qpolynomial_domain_dim(qp, isl_dim_div);
	o_div = isl_qpolynomial_domain_offset(qp, isl_dim_div);
	if (total < 0 || n_div < 0)
		return isl_qpolynomial_free(qp);
	ctx = isl_qpolynomial_get_ctx(qp);
	mat = isl_mat_zero(ctx, n_div, 1 + total);

	for (i = 0; i < n_div; ++i)
		mat = isl_mat_set_element_si(mat, i, o_div + i, 1);

	for (i = 0; i < qp->div->n_row; ++i) {
		normalize_div(qp, i);
		reduce_div(qp, i, &mat);
		if (needs_invert(qp->div, i)) {
			invert_div(qp, i, &mat);
			reduce_div(qp, i, &mat);
		}
	}
	if (!mat)
		goto error;

	s = isl_alloc_array(ctx, struct isl_poly *, n_div);
	if (n_div && !s)
		goto error;
	for (i = 0; i < n_div; ++i)
		s[i] = isl_poly_from_affine(ctx, mat->row[i], ctx->one,
					    1 + total);
	qp->poly = isl_poly_subs(qp->poly, o_div - 1, n_div, s);
	for (i = 0; i < n_div; ++i)
		isl_poly_free(s[i]);
	free(s);
	if (!qp->poly)
		goto error;

	isl_mat_free(mat);

	qp = substitute_non_divs(qp);
	qp = sort_divs(qp);
	new_n_div = isl_qpolynomial_domain_dim(qp, isl_dim_div);
	if (new_n_div < 0)
		return isl_qpolynomial_free(qp);
	if (new_n_div < n_div)
		return reduce_divs(qp);

	return qp;
error:
	isl_qpolynomial_free(qp);
	isl_mat_free(mat);
	return NULL;
}

__isl_give isl_qpolynomial *isl_qpolynomial_rat_cst_on_domain(
	__isl_take isl_space *domain, const isl_int n, const isl_int d)
{
	struct isl_qpolynomial *qp;
	isl_poly_cst *cst;

	qp = isl_qpolynomial_zero_on_domain(domain);
	if (!qp)
		return NULL;

	cst = isl_poly_as_cst(qp->poly);
	isl_int_set(cst->n, n);
	isl_int_set(cst->d, d);

	return qp;
}

/* Return an isl_qpolynomial that is equal to "val" on domain space "domain".
 */
__isl_give isl_qpolynomial *isl_qpolynomial_val_on_domain(
	__isl_take isl_space *domain, __isl_take isl_val *val)
{
	isl_qpolynomial *qp;
	isl_poly_cst *cst;

	qp = isl_qpolynomial_zero_on_domain(domain);
	if (!qp || !val)
		goto error;

	cst = isl_poly_as_cst(qp->poly);
	isl_int_set(cst->n, val->n);
	isl_int_set(cst->d, val->d);

	isl_val_free(val);
	return qp;
error:
	isl_val_free(val);
	isl_qpolynomial_free(qp);
	return NULL;
}

static isl_stat poly_set_active(__isl_keep isl_poly *poly, int *active, int d)
{
	isl_bool is_cst;
	isl_poly_rec *rec;
	int i;

	is_cst = isl_poly_is_cst(poly);
	if (is_cst < 0)
		return isl_stat_error;
	if (is_cst)
		return isl_stat_ok;

	if (poly->var < d)
		active[poly->var] = 1;

	rec = isl_poly_as_rec(poly);
	for (i = 0; i < rec->n; ++i)
		if (poly_set_active(rec->p[i], active, d) < 0)
			return isl_stat_error;

	return isl_stat_ok;
}

static isl_stat set_active(__isl_keep isl_qpolynomial *qp, int *active)
{
	int i, j;
	isl_size d;
	isl_space *space;

	space = isl_qpolynomial_peek_domain_space(qp);
	d = isl_space_dim(space, isl_dim_all);
	if (d < 0 || !active)
		return isl_stat_error;

	for (i = 0; i < d; ++i)
		for (j = 0; j < qp->div->n_row; ++j) {
			if (isl_int_is_zero(qp->div->row[j][2 + i]))
				continue;
			active[i] = 1;
			break;
		}

	return poly_set_active(qp->poly, active, d);
}

#undef TYPE
#define TYPE	isl_qpolynomial
static
#include "check_type_range_templ.c"

isl_bool isl_qpolynomial_involves_dims(__isl_keep isl_qpolynomial *qp,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	int i;
	int *active = NULL;
	isl_bool involves = isl_bool_false;
	isl_size offset;
	isl_size d;
	isl_space *space;

	if (!qp)
		return isl_bool_error;
	if (n == 0)
		return isl_bool_false;

	if (isl_qpolynomial_check_range(qp, type, first, n) < 0)
		return isl_bool_error;
	isl_assert(qp->dim->ctx, type == isl_dim_param ||
				 type == isl_dim_in, return isl_bool_error);

	space = isl_qpolynomial_peek_domain_space(qp);
	d = isl_space_dim(space, isl_dim_all);
	if (d < 0)
		return isl_bool_error;
	active = isl_calloc_array(qp->dim->ctx, int, d);
	if (set_active(qp, active) < 0)
		goto error;

	offset = isl_qpolynomial_domain_var_offset(qp, domain_type(type));
	if (offset < 0)
		goto error;
	first += offset;
	for (i = 0; i < n; ++i)
		if (active[first + i]) {
			involves = isl_bool_true;
			break;
		}

	free(active);

	return involves;
error:
	free(active);
	return isl_bool_error;
}

/* Remove divs that do not appear in the quasi-polynomial, nor in any
 * of the divs that do appear in the quasi-polynomial.
 */
static __isl_give isl_qpolynomial *remove_redundant_divs(
	__isl_take isl_qpolynomial *qp)
{
	int i, j;
	isl_size div_pos;
	int len;
	int skip;
	int *active = NULL;
	int *reordering = NULL;
	int redundant = 0;
	int n_div;
	isl_ctx *ctx;

	if (!qp)
		return NULL;
	if (qp->div->n_row == 0)
		return qp;

	div_pos = isl_qpolynomial_domain_var_offset(qp, isl_dim_div);
	if (div_pos < 0)
		return isl_qpolynomial_free(qp);
	len = qp->div->n_col - 2;
	ctx = isl_qpolynomial_get_ctx(qp);
	active = isl_calloc_array(ctx, int, len);
	if (!active)
		goto error;

	if (poly_set_active(qp->poly, active, len) < 0)
		goto error;

	for (i = qp->div->n_row - 1; i >= 0; --i) {
		if (!active[div_pos + i]) {
			redundant = 1;
			continue;
		}
		for (j = 0; j < i; ++j) {
			if (isl_int_is_zero(qp->div->row[i][2 + div_pos + j]))
				continue;
			active[div_pos + j] = 1;
			break;
		}
	}

	if (!redundant) {
		free(active);
		return qp;
	}

	reordering = isl_alloc_array(qp->div->ctx, int, len);
	if (!reordering)
		goto error;

	for (i = 0; i < div_pos; ++i)
		reordering[i] = i;

	skip = 0;
	n_div = qp->div->n_row;
	for (i = 0; i < n_div; ++i) {
		if (!active[div_pos + i]) {
			qp->div = isl_mat_drop_rows(qp->div, i - skip, 1);
			qp->div = isl_mat_drop_cols(qp->div,
						    2 + div_pos + i - skip, 1);
			skip++;
		}
		reordering[div_pos + i] = div_pos + i - skip;
	}

	qp->poly = reorder(qp->poly, reordering);

	if (!qp->poly || !qp->div)
		goto error;

	free(active);
	free(reordering);

	return qp;
error:
	free(active);
	free(reordering);
	isl_qpolynomial_free(qp);
	return NULL;
}

__isl_give isl_poly *isl_poly_drop(__isl_take isl_poly *poly,
	unsigned first, unsigned n)
{
	int i;
	isl_poly_rec *rec;

	if (!poly)
		return NULL;
	if (n == 0 || poly->var < 0 || poly->var < first)
		return poly;
	if (poly->var < first + n) {
		poly = replace_by_constant_term(poly);
		return isl_poly_drop(poly, first, n);
	}
	poly = isl_poly_cow(poly);
	if (!poly)
		return NULL;
	poly->var -= n;
	rec = isl_poly_as_rec(poly);
	if (!rec)
		goto error;

	for (i = 0; i < rec->n; ++i) {
		rec->p[i] = isl_poly_drop(rec->p[i], first, n);
		if (!rec->p[i])
			goto error;
	}

	return poly;
error:
	isl_poly_free(poly);
	return NULL;
}

__isl_give isl_qpolynomial *isl_qpolynomial_set_dim_name(
	__isl_take isl_qpolynomial *qp,
	enum isl_dim_type type, unsigned pos, const char *s)
{
	qp = isl_qpolynomial_cow(qp);
	if (!qp)
		return NULL;
	if (type == isl_dim_out)
		isl_die(isl_qpolynomial_get_ctx(qp), isl_error_invalid,
			"cannot set name of output/set dimension",
			return isl_qpolynomial_free(qp));
	type = domain_type(type);
	qp->dim = isl_space_set_dim_name(qp->dim, type, pos, s);
	if (!qp->dim)
		goto error;
	return qp;
error:
	isl_qpolynomial_free(qp);
	return NULL;
}

__isl_give isl_qpolynomial *isl_qpolynomial_drop_dims(
	__isl_take isl_qpolynomial *qp,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	isl_size offset;

	if (!qp)
		return NULL;
	if (type == isl_dim_out)
		isl_die(qp->dim->ctx, isl_error_invalid,
			"cannot drop output/set dimension",
			goto error);
	if (isl_qpolynomial_check_range(qp, type, first, n) < 0)
		return isl_qpolynomial_free(qp);
	type = domain_type(type);
	if (n == 0 && !isl_space_is_named_or_nested(qp->dim, type))
		return qp;

	qp = isl_qpolynomial_cow(qp);
	if (!qp)
		return NULL;

	isl_assert(qp->dim->ctx, type == isl_dim_param ||
				 type == isl_dim_set, goto error);

	qp->dim = isl_space_drop_dims(qp->dim, type, first, n);
	if (!qp->dim)
		goto error;

	offset = isl_qpolynomial_domain_var_offset(qp, type);
	if (offset < 0)
		goto error;
	first += offset;

	qp->div = isl_mat_drop_cols(qp->div, 2 + first, n);
	if (!qp->div)
		goto error;

	qp->poly = isl_poly_drop(qp->poly, first, n);
	if (!qp->poly)
		goto error;

	return qp;
error:
	isl_qpolynomial_free(qp);
	return NULL;
}

/* Project the domain of the quasi-polynomial onto its parameter space.
 * The quasi-polynomial may not involve any of the domain dimensions.
 */
__isl_give isl_qpolynomial *isl_qpolynomial_project_domain_on_params(
	__isl_take isl_qpolynomial *qp)
{
	isl_space *space;
	isl_size n;
	isl_bool involves;

	n = isl_qpolynomial_dim(qp, isl_dim_in);
	if (n < 0)
		return isl_qpolynomial_free(qp);
	involves = isl_qpolynomial_involves_dims(qp, isl_dim_in, 0, n);
	if (involves < 0)
		return isl_qpolynomial_free(qp);
	if (involves)
		isl_die(isl_qpolynomial_get_ctx(qp), isl_error_invalid,
			"polynomial involves some of the domain dimensions",
			return isl_qpolynomial_free(qp));
	qp = isl_qpolynomial_drop_dims(qp, isl_dim_in, 0, n);
	space = isl_qpolynomial_get_domain_space(qp);
	space = isl_space_params(space);
	qp = isl_qpolynomial_reset_domain_space(qp, space);
	return qp;
}

static __isl_give isl_qpolynomial *isl_qpolynomial_substitute_equalities_lifted(
	__isl_take isl_qpolynomial *qp, __isl_take isl_basic_set *eq)
{
	int i, j, k;
	isl_int denom;
	unsigned total;
	unsigned n_div;
	isl_poly *poly;

	if (!eq)
		goto error;
	if (eq->n_eq == 0) {
		isl_basic_set_free(eq);
		return qp;
	}

	qp = isl_qpolynomial_cow(qp);
	if (!qp)
		goto error;
	qp->div = isl_mat_cow(qp->div);
	if (!qp->div)
		goto error;

	total = isl_basic_set_offset(eq, isl_dim_div);
	n_div = eq->n_div;
	isl_int_init(denom);
	for (i = 0; i < eq->n_eq; ++i) {
		j = isl_seq_last_non_zero(eq->eq[i], total + n_div);
		if (j < 0 || j == 0 || j >= total)
			continue;

		for (k = 0; k < qp->div->n_row; ++k) {
			if (isl_int_is_zero(qp->div->row[k][1 + j]))
				continue;
			isl_seq_elim(qp->div->row[k] + 1, eq->eq[i], j, total,
					&qp->div->row[k][0]);
			normalize_div(qp, k);
		}

		if (isl_int_is_pos(eq->eq[i][j]))
			isl_seq_neg(eq->eq[i], eq->eq[i], total);
		isl_int_abs(denom, eq->eq[i][j]);
		isl_int_set_si(eq->eq[i][j], 0);

		poly = isl_poly_from_affine(qp->dim->ctx,
						   eq->eq[i], denom, total);
		qp->poly = isl_poly_subs(qp->poly, j - 1, 1, &poly);
		isl_poly_free(poly);
	}
	isl_int_clear(denom);

	if (!qp->poly)
		goto error;

	isl_basic_set_free(eq);

	qp = substitute_non_divs(qp);
	qp = sort_divs(qp);

	return qp;
error:
	isl_basic_set_free(eq);
	isl_qpolynomial_free(qp);
	return NULL;
}

/* Exploit the equalities in "eq" to simplify the quasi-polynomial.
 */
__isl_give isl_qpolynomial *isl_qpolynomial_substitute_equalities(
	__isl_take isl_qpolynomial *qp, __isl_take isl_basic_set *eq)
{
	if (!qp || !eq)
		goto error;
	if (qp->div->n_row > 0)
		eq = isl_basic_set_add_dims(eq, isl_dim_set, qp->div->n_row);
	return isl_qpolynomial_substitute_equalities_lifted(qp, eq);
error:
	isl_basic_set_free(eq);
	isl_qpolynomial_free(qp);
	return NULL;
}

/* Look for equalities among the variables shared by context and qp
 * and the integer divisions of qp, if any.
 * The equalities are then used to eliminate variables and/or integer
 * divisions from qp.
 */
__isl_give isl_qpolynomial *isl_qpolynomial_gist(
	__isl_take isl_qpolynomial *qp, __isl_take isl_set *context)
{
	isl_local_space *ls;
	isl_basic_set *aff;

	ls = isl_qpolynomial_get_domain_local_space(qp);
	context = isl_local_space_lift_set(ls, context);

	aff = isl_set_affine_hull(context);
	return isl_qpolynomial_substitute_equalities_lifted(qp, aff);
}

__isl_give isl_qpolynomial *isl_qpolynomial_gist_params(
	__isl_take isl_qpolynomial *qp, __isl_take isl_set *context)
{
	isl_space *space = isl_qpolynomial_get_domain_space(qp);
	isl_set *dom_context = isl_set_universe(space);
	dom_context = isl_set_intersect_params(dom_context, context);
	return isl_qpolynomial_gist(qp, dom_context);
}

/* Return a zero isl_qpolynomial in the given space.
 *
 * This is a helper function for isl_pw_*_as_* that ensures a uniform
 * interface over all piecewise types.
 */
static __isl_give isl_qpolynomial *isl_qpolynomial_zero_in_space(
	__isl_take isl_space *space)
{
	return isl_qpolynomial_zero_on_domain(isl_space_domain(space));
}

#define isl_qpolynomial_involves_nan isl_qpolynomial_is_nan

#undef PW
#define PW isl_pw_qpolynomial
#undef BASE
#define BASE qpolynomial
#undef EL_IS_ZERO
#define EL_IS_ZERO is_zero
#undef ZERO
#define ZERO zero
#undef IS_ZERO
#define IS_ZERO is_zero
#undef FIELD
#define FIELD qp
#undef DEFAULT_IS_ZERO
#define DEFAULT_IS_ZERO 1

#include <isl_pw_templ.c>
#include <isl_pw_un_op_templ.c>
#include <isl_pw_add_disjoint_templ.c>
#include <isl_pw_eval.c>
#include <isl_pw_fix_templ.c>
#include <isl_pw_from_range_templ.c>
#include <isl_pw_insert_dims_templ.c>
#include <isl_pw_lift_templ.c>
#include <isl_pw_morph_templ.c>
#include <isl_pw_move_dims_templ.c>
#include <isl_pw_neg_templ.c>
#include <isl_pw_opt_templ.c>
#include <isl_pw_split_dims_templ.c>
#include <isl_pw_sub_templ.c>

#undef BASE
#define BASE pw_qpolynomial

#include <isl_union_single.c>
#include <isl_union_eval.c>
#include <isl_union_neg.c>
#include <isl_union_sub_templ.c>

int isl_pw_qpolynomial_is_one(__isl_keep isl_pw_qpolynomial *pwqp)
{
	if (!pwqp)
		return -1;

	if (pwqp->n != -1)
		return 0;

	if (!isl_set_plain_is_universe(pwqp->p[0].set))
		return 0;

	return isl_qpolynomial_is_one(pwqp->p[0].qp);
}

__isl_give isl_pw_qpolynomial *isl_pw_qpolynomial_add(
	__isl_take isl_pw_qpolynomial *pwqp1,
	__isl_take isl_pw_qpolynomial *pwqp2)
{
	return isl_pw_qpolynomial_union_add_(pwqp1, pwqp2);
}

__isl_give isl_pw_qpolynomial *isl_pw_qpolynomial_mul(
	__isl_take isl_pw_qpolynomial *pwqp1,
	__isl_take isl_pw_qpolynomial *pwqp2)
{
	int i, j, n;
	struct isl_pw_qpolynomial *res;

	if (!pwqp1 || !pwqp2)
		goto error;

	isl_assert(pwqp1->dim->ctx, isl_space_is_equal(pwqp1->dim, pwqp2->dim),
			goto error);

	if (isl_pw_qpolynomial_is_zero(pwqp1)) {
		isl_pw_qpolynomial_free(pwqp2);
		return pwqp1;
	}

	if (isl_pw_qpolynomial_is_zero(pwqp2)) {
		isl_pw_qpolynomial_free(pwqp1);
		return pwqp2;
	}

	if (isl_pw_qpolynomial_is_one(pwqp1)) {
		isl_pw_qpolynomial_free(pwqp1);
		return pwqp2;
	}

	if (isl_pw_qpolynomial_is_one(pwqp2)) {
		isl_pw_qpolynomial_free(pwqp2);
		return pwqp1;
	}

	n = pwqp1->n * pwqp2->n;
	res = isl_pw_qpolynomial_alloc_size(isl_space_copy(pwqp1->dim), n);

	for (i = 0; i < pwqp1->n; ++i) {
		for (j = 0; j < pwqp2->n; ++j) {
			struct isl_set *common;
			struct isl_qpolynomial *prod;
			common = isl_set_intersect(isl_set_copy(pwqp1->p[i].set),
						isl_set_copy(pwqp2->p[j].set));
			if (isl_set_plain_is_empty(common)) {
				isl_set_free(common);
				continue;
			}

			prod = isl_qpolynomial_mul(
				isl_qpolynomial_copy(pwqp1->p[i].qp),
				isl_qpolynomial_copy(pwqp2->p[j].qp));

			res = isl_pw_qpolynomial_add_piece(res, common, prod);
		}
	}

	isl_pw_qpolynomial_free(pwqp1);
	isl_pw_qpolynomial_free(pwqp2);

	return res;
error:
	isl_pw_qpolynomial_free(pwqp1);
	isl_pw_qpolynomial_free(pwqp2);
	return NULL;
}

__isl_give isl_val *isl_poly_eval(__isl_take isl_poly *poly,
	__isl_take isl_vec *vec)
{
	int i;
	isl_bool is_cst;
	isl_poly_rec *rec;
	isl_val *res;
	isl_val *base;

	is_cst = isl_poly_is_cst(poly);
	if (is_cst < 0)
		goto error;
	if (is_cst) {
		isl_vec_free(vec);
		res = isl_poly_get_constant_val(poly);
		isl_poly_free(poly);
		return res;
	}

	rec = isl_poly_as_rec(poly);
	if (!rec || !vec)
		goto error;

	isl_assert(poly->ctx, rec->n >= 1, goto error);

	base = isl_val_rat_from_isl_int(poly->ctx,
					vec->el[1 + poly->var], vec->el[0]);

	res = isl_poly_eval(isl_poly_copy(rec->p[rec->n - 1]),
				isl_vec_copy(vec));

	for (i = rec->n - 2; i >= 0; --i) {
		res = isl_val_mul(res, isl_val_copy(base));
		res = isl_val_add(res, isl_poly_eval(isl_poly_copy(rec->p[i]),
							    isl_vec_copy(vec)));
	}

	isl_val_free(base);
	isl_poly_free(poly);
	isl_vec_free(vec);
	return res;
error:
	isl_poly_free(poly);
	isl_vec_free(vec);
	return NULL;
}

/* Evaluate "qp" in the void point "pnt".
 * In particular, return the value NaN.
 */
static __isl_give isl_val *eval_void(__isl_take isl_qpolynomial *qp,
	__isl_take isl_point *pnt)
{
	isl_ctx *ctx;

	ctx = isl_point_get_ctx(pnt);
	isl_qpolynomial_free(qp);
	isl_point_free(pnt);
	return isl_val_nan(ctx);
}

__isl_give isl_val *isl_qpolynomial_eval(__isl_take isl_qpolynomial *qp,
	__isl_take isl_point *pnt)
{
	isl_bool is_void;
	isl_vec *ext;
	isl_val *v;

	if (!qp || !pnt)
		goto error;
	isl_assert(pnt->dim->ctx, isl_space_is_equal(pnt->dim, qp->dim), goto error);
	is_void = isl_point_is_void(pnt);
	if (is_void < 0)
		goto error;
	if (is_void)
		return eval_void(qp, pnt);

	ext = isl_local_extend_point_vec(qp->div, isl_vec_copy(pnt->vec));

	v = isl_poly_eval(isl_poly_copy(qp->poly), ext);

	isl_qpolynomial_free(qp);
	isl_point_free(pnt);

	return v;
error:
	isl_qpolynomial_free(qp);
	isl_point_free(pnt);
	return NULL;
}

int isl_poly_cmp(__isl_keep isl_poly_cst *cst1, __isl_keep isl_poly_cst *cst2)
{
	int cmp;
	isl_int t;
	isl_int_init(t);
	isl_int_mul(t, cst1->n, cst2->d);
	isl_int_submul(t, cst2->n, cst1->d);
	cmp = isl_int_sgn(t);
	isl_int_clear(t);
	return cmp;
}

__isl_give isl_qpolynomial *isl_qpolynomial_insert_dims(
	__isl_take isl_qpolynomial *qp, enum isl_dim_type type,
	unsigned first, unsigned n)
{
	unsigned total;
	unsigned g_pos;
	int *exp;

	if (!qp)
		return NULL;
	if (type == isl_dim_out)
		isl_die(qp->div->ctx, isl_error_invalid,
			"cannot insert output/set dimensions",
			goto error);
	if (isl_qpolynomial_check_range(qp, type, first, 0) < 0)
		return isl_qpolynomial_free(qp);
	type = domain_type(type);
	if (n == 0 && !isl_space_is_named_or_nested(qp->dim, type))
		return qp;

	qp = isl_qpolynomial_cow(qp);
	if (!qp)
		return NULL;

	g_pos = pos(qp->dim, type) + first;

	qp->div = isl_mat_insert_zero_cols(qp->div, 2 + g_pos, n);
	if (!qp->div)
		goto error;

	total = qp->div->n_col - 2;
	if (total > g_pos) {
		int i;
		exp = isl_alloc_array(qp->div->ctx, int, total - g_pos);
		if (!exp)
			goto error;
		for (i = 0; i < total - g_pos; ++i)
			exp[i] = i + n;
		qp->poly = expand(qp->poly, exp, g_pos);
		free(exp);
		if (!qp->poly)
			goto error;
	}

	qp->dim = isl_space_insert_dims(qp->dim, type, first, n);
	if (!qp->dim)
		goto error;

	return qp;
error:
	isl_qpolynomial_free(qp);
	return NULL;
}

__isl_give isl_qpolynomial *isl_qpolynomial_add_dims(
	__isl_take isl_qpolynomial *qp, enum isl_dim_type type, unsigned n)
{
	isl_size pos;

	pos = isl_qpolynomial_dim(qp, type);
	if (pos < 0)
		return isl_qpolynomial_free(qp);

	return isl_qpolynomial_insert_dims(qp, type, pos, n);
}

static int *reordering_move(isl_ctx *ctx,
	unsigned len, unsigned dst, unsigned src, unsigned n)
{
	int i;
	int *reordering;

	reordering = isl_alloc_array(ctx, int, len);
	if (!reordering)
		return NULL;

	if (dst <= src) {
		for (i = 0; i < dst; ++i)
			reordering[i] = i;
		for (i = 0; i < n; ++i)
			reordering[src + i] = dst + i;
		for (i = 0; i < src - dst; ++i)
			reordering[dst + i] = dst + n + i;
		for (i = 0; i < len - src - n; ++i)
			reordering[src + n + i] = src + n + i;
	} else {
		for (i = 0; i < src; ++i)
			reordering[i] = i;
		for (i = 0; i < n; ++i)
			reordering[src + i] = dst + i;
		for (i = 0; i < dst - src; ++i)
			reordering[src + n + i] = src + i;
		for (i = 0; i < len - dst - n; ++i)
			reordering[dst + n + i] = dst + n + i;
	}

	return reordering;
}

__isl_give isl_qpolynomial *isl_qpolynomial_move_dims(
	__isl_take isl_qpolynomial *qp,
	enum isl_dim_type dst_type, unsigned dst_pos,
	enum isl_dim_type src_type, unsigned src_pos, unsigned n)
{
	unsigned g_dst_pos;
	unsigned g_src_pos;
	int *reordering;

	if (!qp)
		return NULL;

	if (dst_type == isl_dim_out || src_type == isl_dim_out)
		isl_die(qp->dim->ctx, isl_error_invalid,
			"cannot move output/set dimension",
			goto error);
	if (isl_qpolynomial_check_range(qp, src_type, src_pos, n) < 0)
		return isl_qpolynomial_free(qp);
	if (dst_type == isl_dim_in)
		dst_type = isl_dim_set;
	if (src_type == isl_dim_in)
		src_type = isl_dim_set;

	if (n == 0 &&
	    !isl_space_is_named_or_nested(qp->dim, src_type) &&
	    !isl_space_is_named_or_nested(qp->dim, dst_type))
		return qp;

	qp = isl_qpolynomial_cow(qp);
	if (!qp)
		return NULL;

	g_dst_pos = pos(qp->dim, dst_type) + dst_pos;
	g_src_pos = pos(qp->dim, src_type) + src_pos;
	if (dst_type > src_type)
		g_dst_pos -= n;

	qp->div = isl_mat_move_cols(qp->div, 2 + g_dst_pos, 2 + g_src_pos, n);
	if (!qp->div)
		goto error;
	qp = sort_divs(qp);
	if (!qp)
		goto error;

	reordering = reordering_move(qp->dim->ctx,
				qp->div->n_col - 2, g_dst_pos, g_src_pos, n);
	if (!reordering)
		goto error;

	qp->poly = reorder(qp->poly, reordering);
	free(reordering);
	if (!qp->poly)
		goto error;

	qp->dim = isl_space_move_dims(qp->dim, dst_type, dst_pos, src_type, src_pos, n);
	if (!qp->dim)
		goto error;

	return qp;
error:
	isl_qpolynomial_free(qp);
	return NULL;
}

__isl_give isl_qpolynomial *isl_qpolynomial_from_affine(
	__isl_take isl_space *space, isl_int *f, isl_int denom)
{
	isl_size d;
	isl_poly *poly;

	space = isl_space_domain(space);
	if (!space)
		return NULL;

	d = isl_space_dim(space, isl_dim_all);
	poly = d < 0 ? NULL : isl_poly_from_affine(space->ctx, f, denom, 1 + d);

	return isl_qpolynomial_alloc(space, 0, poly);
}

__isl_give isl_qpolynomial *isl_qpolynomial_from_aff(__isl_take isl_aff *aff)
{
	isl_ctx *ctx;
	isl_poly *poly;
	isl_qpolynomial *qp;

	if (!aff)
		return NULL;

	ctx = isl_aff_get_ctx(aff);
	poly = isl_poly_from_affine(ctx, aff->v->el + 1, aff->v->el[0],
				    aff->v->size - 1);

	qp = isl_qpolynomial_alloc(isl_aff_get_domain_space(aff),
				    aff->ls->div->n_row, poly);
	if (!qp)
		goto error;

	isl_mat_free(qp->div);
	qp->div = isl_mat_copy(aff->ls->div);
	qp->div = isl_mat_cow(qp->div);
	if (!qp->div)
		goto error;

	isl_aff_free(aff);
	qp = reduce_divs(qp);
	qp = remove_redundant_divs(qp);
	return qp;
error:
	isl_aff_free(aff);
	return isl_qpolynomial_free(qp);
}

__isl_give isl_pw_qpolynomial *isl_pw_qpolynomial_from_pw_aff(
	__isl_take isl_pw_aff *pwaff)
{
	int i;
	isl_pw_qpolynomial *pwqp;

	if (!pwaff)
		return NULL;

	pwqp = isl_pw_qpolynomial_alloc_size(isl_pw_aff_get_space(pwaff),
						pwaff->n);

	for (i = 0; i < pwaff->n; ++i) {
		isl_set *dom;
		isl_qpolynomial *qp;

		dom = isl_set_copy(pwaff->p[i].set);
		qp = isl_qpolynomial_from_aff(isl_aff_copy(pwaff->p[i].aff));
		pwqp = isl_pw_qpolynomial_add_piece(pwqp,  dom, qp);
	}

	isl_pw_aff_free(pwaff);
	return pwqp;
}

__isl_give isl_qpolynomial *isl_qpolynomial_from_constraint(
	__isl_take isl_constraint *c, enum isl_dim_type type, unsigned pos)
{
	isl_aff *aff;

	aff = isl_constraint_get_bound(c, type, pos);
	isl_constraint_free(c);
	return isl_qpolynomial_from_aff(aff);
}

/* For each 0 <= i < "n", replace variable "first" + i of type "type"
 * in "qp" by subs[i].
 */
__isl_give isl_qpolynomial *isl_qpolynomial_substitute(
	__isl_take isl_qpolynomial *qp,
	enum isl_dim_type type, unsigned first, unsigned n,
	__isl_keep isl_qpolynomial **subs)
{
	int i;
	isl_poly **polys;

	if (n == 0)
		return qp;

	qp = isl_qpolynomial_cow(qp);
	if (!qp)
		return NULL;

	if (type == isl_dim_out)
		isl_die(qp->dim->ctx, isl_error_invalid,
			"cannot substitute output/set dimension",
			goto error);
	if (isl_qpolynomial_check_range(qp, type, first, n) < 0)
		return isl_qpolynomial_free(qp);
	type = domain_type(type);

	for (i = 0; i < n; ++i)
		if (!subs[i])
			goto error;

	for (i = 0; i < n; ++i)
		if (isl_qpolynomial_check_equal_space(qp, subs[i]) < 0)
			goto error;

	isl_assert(qp->dim->ctx, qp->div->n_row == 0, goto error);
	for (i = 0; i < n; ++i)
		isl_assert(qp->dim->ctx, subs[i]->div->n_row == 0, goto error);

	first += pos(qp->dim, type);

	polys = isl_alloc_array(qp->dim->ctx, struct isl_poly *, n);
	if (!polys)
		goto error;
	for (i = 0; i < n; ++i)
		polys[i] = subs[i]->poly;

	qp->poly = isl_poly_subs(qp->poly, first, n, polys);

	free(polys);

	if (!qp->poly)
		goto error;

	return qp;
error:
	isl_qpolynomial_free(qp);
	return NULL;
}

/* Extend "bset" with extra set dimensions for each integer division
 * in "qp" and then call "fn" with the extended bset and the polynomial
 * that results from replacing each of the integer divisions by the
 * corresponding extra set dimension.
 */
isl_stat isl_qpolynomial_as_polynomial_on_domain(__isl_keep isl_qpolynomial *qp,
	__isl_keep isl_basic_set *bset,
	isl_stat (*fn)(__isl_take isl_basic_set *bset,
		  __isl_take isl_qpolynomial *poly, void *user), void *user)
{
	isl_space *space;
	isl_local_space *ls;
	isl_qpolynomial *poly;

	if (!qp || !bset)
		return isl_stat_error;
	if (qp->div->n_row == 0)
		return fn(isl_basic_set_copy(bset), isl_qpolynomial_copy(qp),
			  user);

	space = isl_space_copy(qp->dim);
	space = isl_space_add_dims(space, isl_dim_set, qp->div->n_row);
	poly = isl_qpolynomial_alloc(space, 0, isl_poly_copy(qp->poly));
	bset = isl_basic_set_copy(bset);
	ls = isl_qpolynomial_get_domain_local_space(qp);
	bset = isl_local_space_lift_basic_set(ls, bset);

	return fn(bset, poly, user);
}

/* Return total degree in variables first (inclusive) up to last (exclusive).
 */
int isl_poly_degree(__isl_keep isl_poly *poly, int first, int last)
{
	int deg = -1;
	int i;
	isl_bool is_zero, is_cst;
	isl_poly_rec *rec;

	is_zero = isl_poly_is_zero(poly);
	if (is_zero < 0)
		return -2;
	if (is_zero)
		return -1;
	is_cst = isl_poly_is_cst(poly);
	if (is_cst < 0)
		return -2;
	if (is_cst || poly->var < first)
		return 0;

	rec = isl_poly_as_rec(poly);
	if (!rec)
		return -2;

	for (i = 0; i < rec->n; ++i) {
		int d;

		is_zero = isl_poly_is_zero(rec->p[i]);
		if (is_zero < 0)
			return -2;
		if (is_zero)
			continue;
		d = isl_poly_degree(rec->p[i], first, last);
		if (poly->var < last)
			d += i;
		if (d > deg)
			deg = d;
	}

	return deg;
}

/* Return total degree in set variables.
 */
int isl_qpolynomial_degree(__isl_keep isl_qpolynomial *poly)
{
	unsigned ovar;
	isl_size nvar;

	if (!poly)
		return -2;

	ovar = isl_space_offset(poly->dim, isl_dim_set);
	nvar = isl_space_dim(poly->dim, isl_dim_set);
	if (nvar < 0)
		return -2;
	return isl_poly_degree(poly->poly, ovar, ovar + nvar);
}

__isl_give isl_poly *isl_poly_coeff(__isl_keep isl_poly *poly,
	unsigned pos, int deg)
{
	int i;
	isl_bool is_cst;
	isl_poly_rec *rec;

	is_cst = isl_poly_is_cst(poly);
	if (is_cst < 0)
		return NULL;
	if (is_cst || poly->var < pos) {
		if (deg == 0)
			return isl_poly_copy(poly);
		else
			return isl_poly_zero(poly->ctx);
	}

	rec = isl_poly_as_rec(poly);
	if (!rec)
		return NULL;

	if (poly->var == pos) {
		if (deg < rec->n)
			return isl_poly_copy(rec->p[deg]);
		else
			return isl_poly_zero(poly->ctx);
	}

	poly = isl_poly_copy(poly);
	poly = isl_poly_cow(poly);
	rec = isl_poly_as_rec(poly);
	if (!rec)
		goto error;

	for (i = 0; i < rec->n; ++i) {
		isl_poly *t;
		t = isl_poly_coeff(rec->p[i], pos, deg);
		if (!t)
			goto error;
		isl_poly_free(rec->p[i]);
		rec->p[i] = t;
	}

	return poly;
error:
	isl_poly_free(poly);
	return NULL;
}

/* Return coefficient of power "deg" of variable "t_pos" of type "type".
 */
__isl_give isl_qpolynomial *isl_qpolynomial_coeff(
	__isl_keep isl_qpolynomial *qp,
	enum isl_dim_type type, unsigned t_pos, int deg)
{
	unsigned g_pos;
	isl_poly *poly;
	isl_qpolynomial *c;

	if (!qp)
		return NULL;

	if (type == isl_dim_out)
		isl_die(qp->div->ctx, isl_error_invalid,
			"output/set dimension does not have a coefficient",
			return NULL);
	if (isl_qpolynomial_check_range(qp, type, t_pos, 1) < 0)
		return NULL;
	type = domain_type(type);

	g_pos = pos(qp->dim, type) + t_pos;
	poly = isl_poly_coeff(qp->poly, g_pos, deg);

	c = isl_qpolynomial_alloc(isl_space_copy(qp->dim),
				qp->div->n_row, poly);
	if (!c)
		return NULL;
	isl_mat_free(c->div);
	c->div = isl_mat_copy(qp->div);
	if (!c->div)
		goto error;
	return c;
error:
	isl_qpolynomial_free(c);
	return NULL;
}

/* Homogenize the polynomial in the variables first (inclusive) up to
 * last (exclusive) by inserting powers of variable first.
 * Variable first is assumed not to appear in the input.
 */
__isl_give isl_poly *isl_poly_homogenize(__isl_take isl_poly *poly, int deg,
	int target, int first, int last)
{
	int i;
	isl_bool is_zero, is_cst;
	isl_poly_rec *rec;

	is_zero = isl_poly_is_zero(poly);
	if (is_zero < 0)
		return isl_poly_free(poly);
	if (is_zero)
		return poly;
	if (deg == target)
		return poly;
	is_cst = isl_poly_is_cst(poly);
	if (is_cst < 0)
		return isl_poly_free(poly);
	if (is_cst || poly->var < first) {
		isl_poly *hom;

		hom = isl_poly_var_pow(poly->ctx, first, target - deg);
		if (!hom)
			goto error;
		rec = isl_poly_as_rec(hom);
		rec->p[target - deg] = isl_poly_mul(rec->p[target - deg], poly);

		return hom;
	}

	poly = isl_poly_cow(poly);
	rec = isl_poly_as_rec(poly);
	if (!rec)
		goto error;

	for (i = 0; i < rec->n; ++i) {
		is_zero = isl_poly_is_zero(rec->p[i]);
		if (is_zero < 0)
			return isl_poly_free(poly);
		if (is_zero)
			continue;
		rec->p[i] = isl_poly_homogenize(rec->p[i],
				poly->var < last ? deg + i : i, target,
				first, last);
		if (!rec->p[i])
			goto error;
	}

	return poly;
error:
	isl_poly_free(poly);
	return NULL;
}

/* Homogenize the polynomial in the set variables by introducing
 * powers of an extra set variable at position 0.
 */
__isl_give isl_qpolynomial *isl_qpolynomial_homogenize(
	__isl_take isl_qpolynomial *poly)
{
	unsigned ovar;
	isl_size nvar;
	int deg = isl_qpolynomial_degree(poly);

	if (deg < -1)
		goto error;

	poly = isl_qpolynomial_insert_dims(poly, isl_dim_in, 0, 1);
	poly = isl_qpolynomial_cow(poly);
	if (!poly)
		goto error;

	ovar = isl_space_offset(poly->dim, isl_dim_set);
	nvar = isl_space_dim(poly->dim, isl_dim_set);
	if (nvar < 0)
		return isl_qpolynomial_free(poly);
	poly->poly = isl_poly_homogenize(poly->poly, 0, deg, ovar, ovar + nvar);
	if (!poly->poly)
		goto error;

	return poly;
error:
	isl_qpolynomial_free(poly);
	return NULL;
}

__isl_give isl_term *isl_term_alloc(__isl_take isl_space *space,
	__isl_take isl_mat *div)
{
	isl_term *term;
	isl_size d;
	int n;

	d = isl_space_dim(space, isl_dim_all);
	if (d < 0 || !div)
		goto error;

	n = d + div->n_row;

	term = isl_calloc(space->ctx, struct isl_term,
			sizeof(struct isl_term) + (n - 1) * sizeof(int));
	if (!term)
		goto error;

	term->ref = 1;
	term->dim = space;
	term->div = div;
	isl_int_init(term->n);
	isl_int_init(term->d);
	
	return term;
error:
	isl_space_free(space);
	isl_mat_free(div);
	return NULL;
}

__isl_give isl_term *isl_term_copy(__isl_keep isl_term *term)
{
	if (!term)
		return NULL;

	term->ref++;
	return term;
}

__isl_give isl_term *isl_term_dup(__isl_keep isl_term *term)
{
	int i;
	isl_term *dup;
	isl_size total;

	total = isl_term_dim(term, isl_dim_all);
	if (total < 0)
		return NULL;

	dup = isl_term_alloc(isl_space_copy(term->dim), isl_mat_copy(term->div));
	if (!dup)
		return NULL;

	isl_int_set(dup->n, term->n);
	isl_int_set(dup->d, term->d);

	for (i = 0; i < total; ++i)
		dup->pow[i] = term->pow[i];

	return dup;
}

__isl_give isl_term *isl_term_cow(__isl_take isl_term *term)
{
	if (!term)
		return NULL;

	if (term->ref == 1)
		return term;
	term->ref--;
	return isl_term_dup(term);
}

__isl_null isl_term *isl_term_free(__isl_take isl_term *term)
{
	if (!term)
		return NULL;

	if (--term->ref > 0)
		return NULL;

	isl_space_free(term->dim);
	isl_mat_free(term->div);
	isl_int_clear(term->n);
	isl_int_clear(term->d);
	free(term);

	return NULL;
}

isl_size isl_term_dim(__isl_keep isl_term *term, enum isl_dim_type type)
{
	isl_size dim;

	if (!term)
		return isl_size_error;

	switch (type) {
	case isl_dim_param:
	case isl_dim_in:
	case isl_dim_out:	return isl_space_dim(term->dim, type);
	case isl_dim_div:	return term->div->n_row;
	case isl_dim_all:	dim = isl_space_dim(term->dim, isl_dim_all);
				if (dim < 0)
					return isl_size_error;
				return dim + term->div->n_row;
	default:		return isl_size_error;
	}
}

/* Return the space of "term".
 */
static __isl_keep isl_space *isl_term_peek_space(__isl_keep isl_term *term)
{
	return term ? term->dim : NULL;
}

/* Return the offset of the first variable of type "type" within
 * the variables of "term".
 */
static isl_size isl_term_offset(__isl_keep isl_term *term,
	enum isl_dim_type type)
{
	isl_space *space;

	space = isl_term_peek_space(term);
	if (!space)
		return isl_size_error;

	switch (type) {
	case isl_dim_param:
	case isl_dim_set:	return isl_space_offset(space, type);
	case isl_dim_div:	return isl_space_dim(space, isl_dim_all);
	default:
		isl_die(isl_term_get_ctx(term), isl_error_invalid,
			"invalid dimension type", return isl_size_error);
	}
}

isl_ctx *isl_term_get_ctx(__isl_keep isl_term *term)
{
	return term ? term->dim->ctx : NULL;
}

void isl_term_get_num(__isl_keep isl_term *term, isl_int *n)
{
	if (!term)
		return;
	isl_int_set(*n, term->n);
}

/* Return the coefficient of the term "term".
 */
__isl_give isl_val *isl_term_get_coefficient_val(__isl_keep isl_term *term)
{
	if (!term)
		return NULL;

	return isl_val_rat_from_isl_int(isl_term_get_ctx(term),
					term->n, term->d);
}

#undef TYPE
#define TYPE	isl_term
static
#include "check_type_range_templ.c"

isl_size isl_term_get_exp(__isl_keep isl_term *term,
	enum isl_dim_type type, unsigned pos)
{
	isl_size offset;

	if (isl_term_check_range(term, type, pos, 1) < 0)
		return isl_size_error;
	offset = isl_term_offset(term, type);
	if (offset < 0)
		return isl_size_error;

	return term->pow[offset + pos];
}

__isl_give isl_aff *isl_term_get_div(__isl_keep isl_term *term, unsigned pos)
{
	isl_local_space *ls;
	isl_aff *aff;

	if (isl_term_check_range(term, isl_dim_div, pos, 1) < 0)
		return NULL;

	ls = isl_local_space_alloc_div(isl_space_copy(term->dim),
					isl_mat_copy(term->div));
	aff = isl_aff_alloc(ls);
	if (!aff)
		return NULL;

	isl_seq_cpy(aff->v->el, term->div->row[pos], aff->v->size);

	aff = isl_aff_normalize(aff);

	return aff;
}

__isl_give isl_term *isl_poly_foreach_term(__isl_keep isl_poly *poly,
	isl_stat (*fn)(__isl_take isl_term *term, void *user),
	__isl_take isl_term *term, void *user)
{
	int i;
	isl_bool is_zero, is_bad, is_cst;
	isl_poly_rec *rec;

	is_zero = isl_poly_is_zero(poly);
	if (is_zero < 0 || !term)
		goto error;

	if (is_zero)
		return term;

	is_cst = isl_poly_is_cst(poly);
	is_bad = isl_poly_is_nan(poly);
	if (is_bad >= 0 && !is_bad)
		is_bad = isl_poly_is_infty(poly);
	if (is_bad >= 0 && !is_bad)
		is_bad = isl_poly_is_neginfty(poly);
	if (is_cst < 0 || is_bad < 0)
		return isl_term_free(term);
	if (is_bad)
		isl_die(isl_term_get_ctx(term), isl_error_invalid,
			"cannot handle NaN/infty polynomial",
			return isl_term_free(term));

	if (is_cst) {
		isl_poly_cst *cst;
		cst = isl_poly_as_cst(poly);
		if (!cst)
			goto error;
		term = isl_term_cow(term);
		if (!term)
			goto error;
		isl_int_set(term->n, cst->n);
		isl_int_set(term->d, cst->d);
		if (fn(isl_term_copy(term), user) < 0)
			goto error;
		return term;
	}

	rec = isl_poly_as_rec(poly);
	if (!rec)
		goto error;

	for (i = 0; i < rec->n; ++i) {
		term = isl_term_cow(term);
		if (!term)
			goto error;
		term->pow[poly->var] = i;
		term = isl_poly_foreach_term(rec->p[i], fn, term, user);
		if (!term)
			goto error;
	}
	term = isl_term_cow(term);
	if (!term)
		return NULL;
	term->pow[poly->var] = 0;

	return term;
error:
	isl_term_free(term);
	return NULL;
}

isl_stat isl_qpolynomial_foreach_term(__isl_keep isl_qpolynomial *qp,
	isl_stat (*fn)(__isl_take isl_term *term, void *user), void *user)
{
	isl_term *term;

	if (!qp)
		return isl_stat_error;

	term = isl_term_alloc(isl_space_copy(qp->dim), isl_mat_copy(qp->div));
	if (!term)
		return isl_stat_error;

	term = isl_poly_foreach_term(qp->poly, fn, term, user);

	isl_term_free(term);

	return term ? isl_stat_ok : isl_stat_error;
}

__isl_give isl_qpolynomial *isl_qpolynomial_from_term(__isl_take isl_term *term)
{
	isl_poly *poly;
	isl_qpolynomial *qp;
	int i;
	isl_size n;

	n = isl_term_dim(term, isl_dim_all);
	if (n < 0)
		term = isl_term_free(term);
	if (!term)
		return NULL;

	poly = isl_poly_rat_cst(term->dim->ctx, term->n, term->d);
	for (i = 0; i < n; ++i) {
		if (!term->pow[i])
			continue;
		poly = isl_poly_mul(poly,
			    isl_poly_var_pow(term->dim->ctx, i, term->pow[i]));
	}

	qp = isl_qpolynomial_alloc(isl_space_copy(term->dim),
				    term->div->n_row, poly);
	if (!qp)
		goto error;
	isl_mat_free(qp->div);
	qp->div = isl_mat_copy(term->div);
	if (!qp->div)
		goto error;

	isl_term_free(term);
	return qp;
error:
	isl_qpolynomial_free(qp);
	isl_term_free(term);
	return NULL;
}

__isl_give isl_qpolynomial *isl_qpolynomial_lift(__isl_take isl_qpolynomial *qp,
	__isl_take isl_space *space)
{
	int i;
	int extra;
	isl_size total, d_set, d_qp;

	if (!qp || !space)
		goto error;

	if (isl_space_is_equal(qp->dim, space)) {
		isl_space_free(space);
		return qp;
	}

	qp = isl_qpolynomial_cow(qp);
	if (!qp)
		goto error;

	d_set = isl_space_dim(space, isl_dim_set);
	d_qp = isl_qpolynomial_domain_dim(qp, isl_dim_set);
	extra = d_set - d_qp;
	total = isl_space_dim(qp->dim, isl_dim_all);
	if (d_set < 0 || d_qp < 0 || total < 0)
		goto error;
	if (qp->div->n_row) {
		int *exp;

		exp = isl_alloc_array(qp->div->ctx, int, qp->div->n_row);
		if (!exp)
			goto error;
		for (i = 0; i < qp->div->n_row; ++i)
			exp[i] = extra + i;
		qp->poly = expand(qp->poly, exp, total);
		free(exp);
		if (!qp->poly)
			goto error;
	}
	qp->div = isl_mat_insert_cols(qp->div, 2 + total, extra);
	if (!qp->div)
		goto error;
	for (i = 0; i < qp->div->n_row; ++i)
		isl_seq_clr(qp->div->row[i] + 2 + total, extra);

	isl_space_free(qp->dim);
	qp->dim = space;

	return qp;
error:
	isl_space_free(space);
	isl_qpolynomial_free(qp);
	return NULL;
}

/* For each parameter or variable that does not appear in qp,
 * first eliminate the variable from all constraints and then set it to zero.
 */
static __isl_give isl_set *fix_inactive(__isl_take isl_set *set,
	__isl_keep isl_qpolynomial *qp)
{
	int *active = NULL;
	int i;
	isl_size d;
	isl_size nparam;
	isl_size nvar;

	d = isl_set_dim(set, isl_dim_all);
	if (d < 0 || !qp)
		goto error;

	active = isl_calloc_array(set->ctx, int, d);
	if (set_active(qp, active) < 0)
		goto error;

	for (i = 0; i < d; ++i)
		if (!active[i])
			break;

	if (i == d) {
		free(active);
		return set;
	}

	nparam = isl_set_dim(set, isl_dim_param);
	nvar = isl_set_dim(set, isl_dim_set);
	if (nparam < 0 || nvar < 0)
		goto error;
	for (i = 0; i < nparam; ++i) {
		if (active[i])
			continue;
		set = isl_set_eliminate(set, isl_dim_param, i, 1);
		set = isl_set_fix_si(set, isl_dim_param, i, 0);
	}
	for (i = 0; i < nvar; ++i) {
		if (active[nparam + i])
			continue;
		set = isl_set_eliminate(set, isl_dim_set, i, 1);
		set = isl_set_fix_si(set, isl_dim_set, i, 0);
	}

	free(active);

	return set;
error:
	free(active);
	isl_set_free(set);
	return NULL;
}

struct isl_opt_data {
	isl_qpolynomial *qp;
	int first;
	isl_val *opt;
	int max;
};

static isl_stat opt_fn(__isl_take isl_point *pnt, void *user)
{
	struct isl_opt_data *data = (struct isl_opt_data *)user;
	isl_val *val;

	val = isl_qpolynomial_eval(isl_qpolynomial_copy(data->qp), pnt);
	if (data->first) {
		data->first = 0;
		data->opt = val;
	} else if (data->max) {
		data->opt = isl_val_max(data->opt, val);
	} else {
		data->opt = isl_val_min(data->opt, val);
	}

	return isl_stat_ok;
}

__isl_give isl_val *isl_qpolynomial_opt_on_domain(
	__isl_take isl_qpolynomial *qp, __isl_take isl_set *set, int max)
{
	struct isl_opt_data data = { NULL, 1, NULL, max };
	isl_bool is_cst;

	if (!set || !qp)
		goto error;

	is_cst = isl_poly_is_cst(qp->poly);
	if (is_cst < 0)
		goto error;
	if (is_cst) {
		isl_set_free(set);
		data.opt = isl_qpolynomial_get_constant_val(qp);
		isl_qpolynomial_free(qp);
		return data.opt;
	}

	set = fix_inactive(set, qp);

	data.qp = qp;
	if (isl_set_foreach_point(set, opt_fn, &data) < 0)
		goto error;

	if (data.first)
		data.opt = isl_val_zero(isl_set_get_ctx(set));

	isl_set_free(set);
	isl_qpolynomial_free(qp);
	return data.opt;
error:
	isl_set_free(set);
	isl_qpolynomial_free(qp);
	isl_val_free(data.opt);
	return NULL;
}

__isl_give isl_qpolynomial *isl_qpolynomial_morph_domain(
	__isl_take isl_qpolynomial *qp, __isl_take isl_morph *morph)
{
	int i;
	int n_sub;
	isl_ctx *ctx;
	isl_space *space;
	isl_poly **subs;
	isl_mat *mat, *diag;

	qp = isl_qpolynomial_cow(qp);

	space = isl_qpolynomial_peek_domain_space(qp);
	if (isl_morph_check_applies(morph, space) < 0)
		goto error;

	ctx = isl_qpolynomial_get_ctx(qp);
	n_sub = morph->inv->n_row - 1;
	if (morph->inv->n_row != morph->inv->n_col)
		n_sub += qp->div->n_row;
	subs = isl_calloc_array(ctx, struct isl_poly *, n_sub);
	if (n_sub && !subs)
		goto error;

	for (i = 0; 1 + i < morph->inv->n_row; ++i)
		subs[i] = isl_poly_from_affine(ctx, morph->inv->row[1 + i],
					morph->inv->row[0][0], morph->inv->n_col);
	if (morph->inv->n_row != morph->inv->n_col)
		for (i = 0; i < qp->div->n_row; ++i)
			subs[morph->inv->n_row - 1 + i] =
			    isl_poly_var_pow(ctx, morph->inv->n_col - 1 + i, 1);

	qp->poly = isl_poly_subs(qp->poly, 0, n_sub, subs);

	for (i = 0; i < n_sub; ++i)
		isl_poly_free(subs[i]);
	free(subs);

	diag = isl_mat_diag(ctx, 1, morph->inv->row[0][0]);
	mat = isl_mat_diagonal(diag, isl_mat_copy(morph->inv));
	diag = isl_mat_diag(ctx, qp->div->n_row, morph->inv->row[0][0]);
	mat = isl_mat_diagonal(mat, diag);
	qp->div = isl_mat_product(qp->div, mat);
	isl_space_free(qp->dim);
	qp->dim = isl_space_copy(morph->ran->dim);

	if (!qp->poly || !qp->div || !qp->dim)
		goto error;

	isl_morph_free(morph);

	return qp;
error:
	isl_qpolynomial_free(qp);
	isl_morph_free(morph);
	return NULL;
}

__isl_give isl_union_pw_qpolynomial *isl_union_pw_qpolynomial_mul(
	__isl_take isl_union_pw_qpolynomial *upwqp1,
	__isl_take isl_union_pw_qpolynomial *upwqp2)
{
	return isl_union_pw_qpolynomial_match_bin_op(upwqp1, upwqp2,
						&isl_pw_qpolynomial_mul);
}

/* Reorder the dimension of "qp" according to the given reordering.
 */
__isl_give isl_qpolynomial *isl_qpolynomial_realign_domain(
	__isl_take isl_qpolynomial *qp, __isl_take isl_reordering *r)
{
	isl_space *space;

	qp = isl_qpolynomial_cow(qp);
	if (!qp)
		goto error;

	r = isl_reordering_extend(r, qp->div->n_row);
	if (!r)
		goto error;

	qp->div = isl_local_reorder(qp->div, isl_reordering_copy(r));
	if (!qp->div)
		goto error;

	qp->poly = reorder(qp->poly, r->pos);
	if (!qp->poly)
		goto error;

	space = isl_reordering_get_space(r);
	qp = isl_qpolynomial_reset_domain_space(qp, space);

	isl_reordering_free(r);
	return qp;
error:
	isl_qpolynomial_free(qp);
	isl_reordering_free(r);
	return NULL;
}

__isl_give isl_qpolynomial *isl_qpolynomial_align_params(
	__isl_take isl_qpolynomial *qp, __isl_take isl_space *model)
{
	isl_space *domain_space;
	isl_bool equal_params;

	domain_space = isl_qpolynomial_peek_domain_space(qp);
	equal_params = isl_space_has_equal_params(domain_space, model);
	if (equal_params < 0)
		goto error;
	if (!equal_params) {
		isl_reordering *exp;

		exp = isl_parameter_alignment_reordering(domain_space, model);
		qp = isl_qpolynomial_realign_domain(qp, exp);
	}

	isl_space_free(model);
	return qp;
error:
	isl_space_free(model);
	isl_qpolynomial_free(qp);
	return NULL;
}

struct isl_split_periods_data {
	int max_periods;
	isl_pw_qpolynomial *res;
};

/* Create a slice where the integer division "div" has the fixed value "v".
 * In particular, if "div" refers to floor(f/m), then create a slice
 *
 *	m v <= f <= m v + (m - 1)
 *
 * or
 *
 *	f - m v >= 0
 *	-f + m v + (m - 1) >= 0
 */
static __isl_give isl_set *set_div_slice(__isl_take isl_space *space,
	__isl_keep isl_qpolynomial *qp, int div, isl_int v)
{
	isl_size total;
	isl_basic_set *bset = NULL;
	int k;

	total = isl_space_dim(space, isl_dim_all);
	if (total < 0 || !qp)
		goto error;

	bset = isl_basic_set_alloc_space(isl_space_copy(space), 0, 0, 2);

	k = isl_basic_set_alloc_inequality(bset);
	if (k < 0)
		goto error;
	isl_seq_cpy(bset->ineq[k], qp->div->row[div] + 1, 1 + total);
	isl_int_submul(bset->ineq[k][0], v, qp->div->row[div][0]);

	k = isl_basic_set_alloc_inequality(bset);
	if (k < 0)
		goto error;
	isl_seq_neg(bset->ineq[k], qp->div->row[div] + 1, 1 + total);
	isl_int_addmul(bset->ineq[k][0], v, qp->div->row[div][0]);
	isl_int_add(bset->ineq[k][0], bset->ineq[k][0], qp->div->row[div][0]);
	isl_int_sub_ui(bset->ineq[k][0], bset->ineq[k][0], 1);

	isl_space_free(space);
	return isl_set_from_basic_set(bset);
error:
	isl_basic_set_free(bset);
	isl_space_free(space);
	return NULL;
}

static isl_stat split_periods(__isl_take isl_set *set,
	__isl_take isl_qpolynomial *qp, void *user);

/* Create a slice of the domain "set" such that integer division "div"
 * has the fixed value "v" and add the results to data->res,
 * replacing the integer division by "v" in "qp".
 */
static isl_stat set_div(__isl_take isl_set *set,
	__isl_take isl_qpolynomial *qp, int div, isl_int v,
	struct isl_split_periods_data *data)
{
	int i;
	isl_size div_pos;
	isl_set *slice;
	isl_poly *cst;

	slice = set_div_slice(isl_set_get_space(set), qp, div, v);
	set = isl_set_intersect(set, slice);

	div_pos = isl_qpolynomial_domain_var_offset(qp, isl_dim_div);
	if (div_pos < 0)
		goto error;

	for (i = div + 1; i < qp->div->n_row; ++i) {
		if (isl_int_is_zero(qp->div->row[i][2 + div_pos + div]))
			continue;
		isl_int_addmul(qp->div->row[i][1],
				qp->div->row[i][2 + div_pos + div], v);
		isl_int_set_si(qp->div->row[i][2 + div_pos + div], 0);
	}

	cst = isl_poly_rat_cst(qp->dim->ctx, v, qp->dim->ctx->one);
	qp = substitute_div(qp, div, cst);

	return split_periods(set, qp, data);
error:
	isl_set_free(set);
	isl_qpolynomial_free(qp);
	return isl_stat_error;
}

/* Split the domain "set" such that integer division "div"
 * has a fixed value (ranging from "min" to "max") on each slice
 * and add the results to data->res.
 */
static isl_stat split_div(__isl_take isl_set *set,
	__isl_take isl_qpolynomial *qp, int div, isl_int min, isl_int max,
	struct isl_split_periods_data *data)
{
	for (; isl_int_le(min, max); isl_int_add_ui(min, min, 1)) {
		isl_set *set_i = isl_set_copy(set);
		isl_qpolynomial *qp_i = isl_qpolynomial_copy(qp);

		if (set_div(set_i, qp_i, div, min, data) < 0)
			goto error;
	}
	isl_set_free(set);
	isl_qpolynomial_free(qp);
	return isl_stat_ok;
error:
	isl_set_free(set);
	isl_qpolynomial_free(qp);
	return isl_stat_error;
}

/* If "qp" refers to any integer division
 * that can only attain "max_periods" distinct values on "set"
 * then split the domain along those distinct values.
 * Add the results (or the original if no splitting occurs)
 * to data->res.
 */
static isl_stat split_periods(__isl_take isl_set *set,
	__isl_take isl_qpolynomial *qp, void *user)
{
	int i;
	isl_pw_qpolynomial *pwqp;
	struct isl_split_periods_data *data;
	isl_int min, max;
	isl_size div_pos;
	isl_stat r = isl_stat_ok;

	data = (struct isl_split_periods_data *)user;

	if (!set || !qp)
		goto error;

	if (qp->div->n_row == 0) {
		pwqp = isl_pw_qpolynomial_alloc(set, qp);
		data->res = isl_pw_qpolynomial_add_disjoint(data->res, pwqp);
		return isl_stat_ok;
	}

	div_pos = isl_qpolynomial_domain_var_offset(qp, isl_dim_div);
	if (div_pos < 0)
		goto error;

	isl_int_init(min);
	isl_int_init(max);
	for (i = 0; i < qp->div->n_row; ++i) {
		enum isl_lp_result lp_res;

		if (isl_seq_first_non_zero(qp->div->row[i] + 2 + div_pos,
						qp->div->n_row) != -1)
			continue;

		lp_res = isl_set_solve_lp(set, 0, qp->div->row[i] + 1,
					  set->ctx->one, &min, NULL, NULL);
		if (lp_res == isl_lp_error)
			goto error2;
		if (lp_res == isl_lp_unbounded || lp_res == isl_lp_empty)
			continue;
		isl_int_fdiv_q(min, min, qp->div->row[i][0]);

		lp_res = isl_set_solve_lp(set, 1, qp->div->row[i] + 1,
					  set->ctx->one, &max, NULL, NULL);
		if (lp_res == isl_lp_error)
			goto error2;
		if (lp_res == isl_lp_unbounded || lp_res == isl_lp_empty)
			continue;
		isl_int_fdiv_q(max, max, qp->div->row[i][0]);

		isl_int_sub(max, max, min);
		if (isl_int_cmp_si(max, data->max_periods) < 0) {
			isl_int_add(max, max, min);
			break;
		}
	}

	if (i < qp->div->n_row) {
		r = split_div(set, qp, i, min, max, data);
	} else {
		pwqp = isl_pw_qpolynomial_alloc(set, qp);
		data->res = isl_pw_qpolynomial_add_disjoint(data->res, pwqp);
	}

	isl_int_clear(max);
	isl_int_clear(min);

	return r;
error2:
	isl_int_clear(max);
	isl_int_clear(min);
error:
	isl_set_free(set);
	isl_qpolynomial_free(qp);
	return isl_stat_error;
}

/* If any quasi-polynomial in pwqp refers to any integer division
 * that can only attain "max_periods" distinct values on its domain
 * then split the domain along those distinct values.
 */
__isl_give isl_pw_qpolynomial *isl_pw_qpolynomial_split_periods(
	__isl_take isl_pw_qpolynomial *pwqp, int max_periods)
{
	struct isl_split_periods_data data;

	data.max_periods = max_periods;
	data.res = isl_pw_qpolynomial_zero(isl_pw_qpolynomial_get_space(pwqp));

	if (isl_pw_qpolynomial_foreach_piece(pwqp, &split_periods, &data) < 0)
		goto error;

	isl_pw_qpolynomial_free(pwqp);

	return data.res;
error:
	isl_pw_qpolynomial_free(data.res);
	isl_pw_qpolynomial_free(pwqp);
	return NULL;
}

/* Construct a piecewise quasipolynomial that is constant on the given
 * domain.  In particular, it is
 *	0	if cst == 0
 *	1	if cst == 1
 *  infinity	if cst == -1
 *
 * If cst == -1, then explicitly check whether the domain is empty and,
 * if so, return 0 instead.
 */
static __isl_give isl_pw_qpolynomial *constant_on_domain(
	__isl_take isl_basic_set *bset, int cst)
{
	isl_space *space;
	isl_qpolynomial *qp;

	if (cst < 0 && isl_basic_set_is_empty(bset) == isl_bool_true)
		cst = 0;
	if (!bset)
		return NULL;

	bset = isl_basic_set_params(bset);
	space = isl_basic_set_get_space(bset);
	if (cst < 0)
		qp = isl_qpolynomial_infty_on_domain(space);
	else if (cst == 0)
		qp = isl_qpolynomial_zero_on_domain(space);
	else
		qp = isl_qpolynomial_one_on_domain(space);
	return isl_pw_qpolynomial_alloc(isl_set_from_basic_set(bset), qp);
}

/* Internal data structure for multiplicative_call_factor_pw_qpolynomial.
 * "fn" is the function that is called on each factor.
 * "pwpq" collects the results.
 */
struct isl_multiplicative_call_data_pw_qpolynomial {
	__isl_give isl_pw_qpolynomial *(*fn)(__isl_take isl_basic_set *bset);
	isl_pw_qpolynomial *pwqp;
};

/* Call "fn" on "bset" and return the result,
 * but first check if "bset" has any redundant constraints or
 * implicit equality constraints.
 * If so, there may be further opportunities for detecting factors or
 * removing equality constraints, so recursively call
 * the top-level isl_basic_set_multiplicative_call.
 */
static __isl_give isl_pw_qpolynomial *multiplicative_call_base(
	__isl_take isl_basic_set *bset,
	__isl_give isl_pw_qpolynomial *(*fn)(__isl_take isl_basic_set *bset))
{
	isl_size n1, n2, n_eq;

	n1 = isl_basic_set_n_constraint(bset);
	if (n1 < 0)
		bset = isl_basic_set_free(bset);
	bset = isl_basic_set_remove_redundancies(bset);
	bset = isl_basic_set_detect_equalities(bset);
	n2 = isl_basic_set_n_constraint(bset);
	n_eq = isl_basic_set_n_equality(bset);
	if (n2 < 0 || n_eq < 0)
		bset = isl_basic_set_free(bset);
	else if (n2 < n1 || n_eq > 0)
		return isl_basic_set_multiplicative_call(bset, fn);
	return fn(bset);
}

/* isl_factorizer_every_factor_basic_set callback that applies
 * data->fn to the factor "bset" and multiplies in the result
 * in data->pwqp.
 */
static isl_bool multiplicative_call_factor_pw_qpolynomial(
	__isl_keep isl_basic_set *bset, void *user)
{
	struct isl_multiplicative_call_data_pw_qpolynomial *data = user;
	isl_pw_qpolynomial *res;

	bset = isl_basic_set_copy(bset);
	res = multiplicative_call_base(bset, data->fn);
	data->pwqp = isl_pw_qpolynomial_mul(data->pwqp, res);
	if (!data->pwqp)
		return isl_bool_error;

	return isl_bool_true;
}

/* Factor bset, call fn on each of the factors and return the product.
 *
 * If no factors can be found, simply call fn on the input.
 * Otherwise, construct the factors based on the factorizer,
 * call fn on each factor and compute the product.
 */
static __isl_give isl_pw_qpolynomial *compressed_multiplicative_call(
	__isl_take isl_basic_set *bset,
	__isl_give isl_pw_qpolynomial *(*fn)(__isl_take isl_basic_set *bset))
{
	struct isl_multiplicative_call_data_pw_qpolynomial data = { fn };
	isl_space *space;
	isl_set *set;
	isl_factorizer *f;
	isl_qpolynomial *qp;
	isl_bool every;

	f = isl_basic_set_factorizer(bset);
	if (!f)
		goto error;
	if (f->n_group == 0) {
		isl_factorizer_free(f);
		return multiplicative_call_base(bset, fn);
	}

	space = isl_basic_set_get_space(bset);
	space = isl_space_params(space);
	set = isl_set_universe(isl_space_copy(space));
	qp = isl_qpolynomial_one_on_domain(space);
	data.pwqp = isl_pw_qpolynomial_alloc(set, qp);

	every = isl_factorizer_every_factor_basic_set(f,
			&multiplicative_call_factor_pw_qpolynomial, &data);
	if (every < 0)
		data.pwqp = isl_pw_qpolynomial_free(data.pwqp);

	isl_basic_set_free(bset);
	isl_factorizer_free(f);

	return data.pwqp;
error:
	isl_basic_set_free(bset);
	return NULL;
}

/* Factor bset, call fn on each of the factors and return the product.
 * The function is assumed to evaluate to zero on empty domains,
 * to one on zero-dimensional domains and to infinity on unbounded domains
 * and will not be called explicitly on zero-dimensional or unbounded domains.
 *
 * We first check for some special cases and remove all equalities.
 * Then we hand over control to compressed_multiplicative_call.
 */
__isl_give isl_pw_qpolynomial *isl_basic_set_multiplicative_call(
	__isl_take isl_basic_set *bset,
	__isl_give isl_pw_qpolynomial *(*fn)(__isl_take isl_basic_set *bset))
{
	isl_bool bounded;
	isl_size dim;
	isl_morph *morph;
	isl_pw_qpolynomial *pwqp;

	if (!bset)
		return NULL;

	if (isl_basic_set_plain_is_empty(bset))
		return constant_on_domain(bset, 0);

	dim = isl_basic_set_dim(bset, isl_dim_set);
	if (dim < 0)
		goto error;
	if (dim == 0)
		return constant_on_domain(bset, 1);

	bounded = isl_basic_set_is_bounded(bset);
	if (bounded < 0)
		goto error;
	if (!bounded)
		return constant_on_domain(bset, -1);

	if (bset->n_eq == 0)
		return compressed_multiplicative_call(bset, fn);

	morph = isl_basic_set_full_compression(bset);
	bset = isl_morph_basic_set(isl_morph_copy(morph), bset);

	pwqp = compressed_multiplicative_call(bset, fn);

	morph = isl_morph_dom_params(morph);
	morph = isl_morph_ran_params(morph);
	morph = isl_morph_inverse(morph);

	pwqp = isl_pw_qpolynomial_morph_domain(pwqp, morph);

	return pwqp;
error:
	isl_basic_set_free(bset);
	return NULL;
}

/* Drop all floors in "qp", turning each integer division [a/m] into
 * a rational division a/m.  If "down" is set, then the integer division
 * is replaced by (a-(m-1))/m instead.
 */
static __isl_give isl_qpolynomial *qp_drop_floors(
	__isl_take isl_qpolynomial *qp, int down)
{
	int i;
	isl_poly *s;

	if (!qp)
		return NULL;
	if (qp->div->n_row == 0)
		return qp;

	qp = isl_qpolynomial_cow(qp);
	if (!qp)
		return NULL;

	for (i = qp->div->n_row - 1; i >= 0; --i) {
		if (down) {
			isl_int_sub(qp->div->row[i][1],
				    qp->div->row[i][1], qp->div->row[i][0]);
			isl_int_add_ui(qp->div->row[i][1],
				       qp->div->row[i][1], 1);
		}
		s = isl_poly_from_affine(qp->dim->ctx, qp->div->row[i] + 1,
					qp->div->row[i][0], qp->div->n_col - 1);
		qp = substitute_div(qp, i, s);
		if (!qp)
			return NULL;
	}

	return qp;
}

/* Drop all floors in "pwqp", turning each integer division [a/m] into
 * a rational division a/m.
 */
static __isl_give isl_pw_qpolynomial *pwqp_drop_floors(
	__isl_take isl_pw_qpolynomial *pwqp)
{
	int i;

	if (!pwqp)
		return NULL;

	if (isl_pw_qpolynomial_is_zero(pwqp))
		return pwqp;

	pwqp = isl_pw_qpolynomial_cow(pwqp);
	if (!pwqp)
		return NULL;

	for (i = 0; i < pwqp->n; ++i) {
		pwqp->p[i].qp = qp_drop_floors(pwqp->p[i].qp, 0);
		if (!pwqp->p[i].qp)
			goto error;
	}

	return pwqp;
error:
	isl_pw_qpolynomial_free(pwqp);
	return NULL;
}

/* Adjust all the integer divisions in "qp" such that they are at least
 * one over the given orthant (identified by "signs").  This ensures
 * that they will still be non-negative even after subtracting (m-1)/m.
 *
 * In particular, f is replaced by f' + v, changing f = [a/m]
 * to f' = [(a - m v)/m].
 * If the constant term k in a is smaller than m,
 * the constant term of v is set to floor(k/m) - 1.
 * For any other term, if the coefficient c and the variable x have
 * the same sign, then no changes are needed.
 * Otherwise, if the variable is positive (and c is negative),
 * then the coefficient of x in v is set to floor(c/m).
 * If the variable is negative (and c is positive),
 * then the coefficient of x in v is set to ceil(c/m).
 */
static __isl_give isl_qpolynomial *make_divs_pos(__isl_take isl_qpolynomial *qp,
	int *signs)
{
	int i, j;
	isl_size div_pos;
	isl_vec *v = NULL;
	isl_poly *s;

	qp = isl_qpolynomial_cow(qp);
	div_pos = isl_qpolynomial_domain_var_offset(qp, isl_dim_div);
	if (div_pos < 0)
		return isl_qpolynomial_free(qp);
	qp->div = isl_mat_cow(qp->div);
	if (!qp->div)
		goto error;

	v = isl_vec_alloc(qp->div->ctx, qp->div->n_col - 1);

	for (i = 0; i < qp->div->n_row; ++i) {
		isl_int *row = qp->div->row[i];
		v = isl_vec_clr(v);
		if (!v)
			goto error;
		if (isl_int_lt(row[1], row[0])) {
			isl_int_fdiv_q(v->el[0], row[1], row[0]);
			isl_int_sub_ui(v->el[0], v->el[0], 1);
			isl_int_submul(row[1], row[0], v->el[0]);
		}
		for (j = 0; j < div_pos; ++j) {
			if (isl_int_sgn(row[2 + j]) * signs[j] >= 0)
				continue;
			if (signs[j] < 0)
				isl_int_cdiv_q(v->el[1 + j], row[2 + j], row[0]);
			else
				isl_int_fdiv_q(v->el[1 + j], row[2 + j], row[0]);
			isl_int_submul(row[2 + j], row[0], v->el[1 + j]);
		}
		for (j = 0; j < i; ++j) {
			if (isl_int_sgn(row[2 + div_pos + j]) >= 0)
				continue;
			isl_int_fdiv_q(v->el[1 + div_pos + j],
					row[2 + div_pos + j], row[0]);
			isl_int_submul(row[2 + div_pos + j],
					row[0], v->el[1 + div_pos + j]);
		}
		for (j = i + 1; j < qp->div->n_row; ++j) {
			if (isl_int_is_zero(qp->div->row[j][2 + div_pos + i]))
				continue;
			isl_seq_combine(qp->div->row[j] + 1,
				qp->div->ctx->one, qp->div->row[j] + 1,
				qp->div->row[j][2 + div_pos + i], v->el,
				v->size);
		}
		isl_int_set_si(v->el[1 + div_pos + i], 1);
		s = isl_poly_from_affine(qp->dim->ctx, v->el,
					qp->div->ctx->one, v->size);
		qp->poly = isl_poly_subs(qp->poly, div_pos + i, 1, &s);
		isl_poly_free(s);
		if (!qp->poly)
			goto error;
	}

	isl_vec_free(v);
	return qp;
error:
	isl_vec_free(v);
	isl_qpolynomial_free(qp);
	return NULL;
}

struct isl_to_poly_data {
	int sign;
	isl_pw_qpolynomial *res;
	isl_qpolynomial *qp;
};

/* Appoximate data->qp by a polynomial on the orthant identified by "signs".
 * We first make all integer divisions positive and then split the
 * quasipolynomials into terms with sign data->sign (the direction
 * of the requested approximation) and terms with the opposite sign.
 * In the first set of terms, each integer division [a/m] is
 * overapproximated by a/m, while in the second it is underapproximated
 * by (a-(m-1))/m.
 */
static isl_stat to_polynomial_on_orthant(__isl_take isl_set *orthant,
	int *signs, void *user)
{
	struct isl_to_poly_data *data = user;
	isl_pw_qpolynomial *t;
	isl_qpolynomial *qp, *up, *down;

	qp = isl_qpolynomial_copy(data->qp);
	qp = make_divs_pos(qp, signs);

	up = isl_qpolynomial_terms_of_sign(qp, signs, data->sign);
	up = qp_drop_floors(up, 0);
	down = isl_qpolynomial_terms_of_sign(qp, signs, -data->sign);
	down = qp_drop_floors(down, 1);

	isl_qpolynomial_free(qp);
	qp = isl_qpolynomial_add(up, down);

	t = isl_pw_qpolynomial_alloc(orthant, qp);
	data->res = isl_pw_qpolynomial_add_disjoint(data->res, t);

	return isl_stat_ok;
}

/* Approximate each quasipolynomial by a polynomial.  If "sign" is positive,
 * the polynomial will be an overapproximation.  If "sign" is negative,
 * it will be an underapproximation.  If "sign" is zero, the approximation
 * will lie somewhere in between.
 *
 * In particular, is sign == 0, we simply drop the floors, turning
 * the integer divisions into rational divisions.
 * Otherwise, we split the domains into orthants, make all integer divisions
 * positive and then approximate each [a/m] by either a/m or (a-(m-1))/m,
 * depending on the requested sign and the sign of the term in which
 * the integer division appears.
 */
__isl_give isl_pw_qpolynomial *isl_pw_qpolynomial_to_polynomial(
	__isl_take isl_pw_qpolynomial *pwqp, int sign)
{
	int i;
	struct isl_to_poly_data data;

	if (sign == 0)
		return pwqp_drop_floors(pwqp);

	if (!pwqp)
		return NULL;

	data.sign = sign;
	data.res = isl_pw_qpolynomial_zero(isl_pw_qpolynomial_get_space(pwqp));

	for (i = 0; i < pwqp->n; ++i) {
		if (pwqp->p[i].qp->div->n_row == 0) {
			isl_pw_qpolynomial *t;
			t = isl_pw_qpolynomial_alloc(
					isl_set_copy(pwqp->p[i].set),
					isl_qpolynomial_copy(pwqp->p[i].qp));
			data.res = isl_pw_qpolynomial_add_disjoint(data.res, t);
			continue;
		}
		data.qp = pwqp->p[i].qp;
		if (isl_set_foreach_orthant(pwqp->p[i].set,
					&to_polynomial_on_orthant, &data) < 0)
			goto error;
	}

	isl_pw_qpolynomial_free(pwqp);

	return data.res;
error:
	isl_pw_qpolynomial_free(pwqp);
	isl_pw_qpolynomial_free(data.res);
	return NULL;
}

static __isl_give isl_pw_qpolynomial *poly_entry(
	__isl_take isl_pw_qpolynomial *pwqp, void *user)
{
	int *sign = user;

	return isl_pw_qpolynomial_to_polynomial(pwqp, *sign);
}

__isl_give isl_union_pw_qpolynomial *isl_union_pw_qpolynomial_to_polynomial(
	__isl_take isl_union_pw_qpolynomial *upwqp, int sign)
{
	return isl_union_pw_qpolynomial_transform_inplace(upwqp,
				   &poly_entry, &sign);
}

__isl_give isl_basic_map *isl_basic_map_from_qpolynomial(
	__isl_take isl_qpolynomial *qp)
{
	int i, k;
	isl_space *space;
	isl_vec *aff = NULL;
	isl_basic_map *bmap = NULL;
	isl_bool is_affine;
	unsigned pos;
	unsigned n_div;

	if (!qp)
		return NULL;
	is_affine = isl_poly_is_affine(qp->poly);
	if (is_affine < 0)
		goto error;
	if (!is_affine)
		isl_die(qp->dim->ctx, isl_error_invalid,
			"input quasi-polynomial not affine", goto error);
	aff = isl_qpolynomial_extract_affine(qp);
	if (!aff)
		goto error;
	space = isl_qpolynomial_get_space(qp);
	pos = 1 + isl_space_offset(space, isl_dim_out);
	n_div = qp->div->n_row;
	bmap = isl_basic_map_alloc_space(space, n_div, 1, 2 * n_div);

	for (i = 0; i < n_div; ++i) {
		k = isl_basic_map_alloc_div(bmap);
		if (k < 0)
			goto error;
		isl_seq_cpy(bmap->div[k], qp->div->row[i], qp->div->n_col);
		isl_int_set_si(bmap->div[k][qp->div->n_col], 0);
		bmap = isl_basic_map_add_div_constraints(bmap, k);
	}
	k = isl_basic_map_alloc_equality(bmap);
	if (k < 0)
		goto error;
	isl_int_neg(bmap->eq[k][pos], aff->el[0]);
	isl_seq_cpy(bmap->eq[k], aff->el + 1, pos);
	isl_seq_cpy(bmap->eq[k] + pos + 1, aff->el + 1 + pos, n_div);

	isl_vec_free(aff);
	isl_qpolynomial_free(qp);
	bmap = isl_basic_map_finalize(bmap);
	return bmap;
error:
	isl_vec_free(aff);
	isl_qpolynomial_free(qp);
	isl_basic_map_free(bmap);
	return NULL;
}
