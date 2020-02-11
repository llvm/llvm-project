/*
 * Copyright 2010      INRIA Saclay
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, INRIA Saclay - Ile-de-France,
 * Parc Club Orsay Universite, ZAC des vignes, 4 rue Jacques Monod,
 * 91893 Orsay, France 
 */

#include <isl_map_private.h>
#include <isl/set.h>
#include <isl_space_private.h>
#include <isl_seq.h>

/*
 * Let C be a cone and define
 *
 *	C' := { y | forall x in C : y x >= 0 }
 *
 * C' contains the coefficients of all linear constraints
 * that are valid for C.
 * Furthermore, C'' = C.
 *
 * If C is defined as { x | A x >= 0 }
 * then any element in C' must be a non-negative combination
 * of the rows of A, i.e., y = t A with t >= 0.  That is,
 *
 *	C' = { y | exists t >= 0 : y = t A }
 *
 * If any of the rows in A actually represents an equality, then
 * also negative combinations of this row are allowed and so the
 * non-negativity constraint on the corresponding element of t
 * can be dropped.
 *
 * A polyhedron P = { x | b + A x >= 0 } can be represented
 * in homogeneous coordinates by the cone
 * C = { [z,x] | b z + A x >= and z >= 0 }
 * The valid linear constraints on C correspond to the valid affine
 * constraints on P.
 * This is essentially Farkas' lemma.
 *
 * Since
 *				  [ 1 0 ]
 *		[ w y ] = [t_0 t] [ b A ]
 *
 * we have
 *
 *	C' = { w, y | exists t_0, t >= 0 : y = t A and w = t_0 + t b }
 * or
 *
 *	C' = { w, y | exists t >= 0 : y = t A and w - t b >= 0 }
 *
 * In practice, we introduce an extra variable (w), shifting all
 * other variables to the right, and an extra inequality
 * (w - t b >= 0) corresponding to the positivity constraint on
 * the homogeneous coordinate.
 *
 * When going back from coefficients to solutions, we immediately
 * plug in 1 for z, which corresponds to shifting all variables
 * to the left, with the leftmost ending up in the constant position.
 */

/* Add the given prefix to all named isl_dim_set dimensions in "space".
 */
static __isl_give isl_space *isl_space_prefix(__isl_take isl_space *space,
	const char *prefix)
{
	int i;
	isl_ctx *ctx;
	isl_size nvar;
	size_t prefix_len = strlen(prefix);

	if (!space)
		return NULL;

	ctx = isl_space_get_ctx(space);
	nvar = isl_space_dim(space, isl_dim_set);
	if (nvar < 0)
		return isl_space_free(space);

	for (i = 0; i < nvar; ++i) {
		const char *name;
		char *prefix_name;

		name = isl_space_get_dim_name(space, isl_dim_set, i);
		if (!name)
			continue;

		prefix_name = isl_alloc_array(ctx, char,
					      prefix_len + strlen(name) + 1);
		if (!prefix_name)
			goto error;
		memcpy(prefix_name, prefix, prefix_len);
		strcpy(prefix_name + prefix_len, name);

		space = isl_space_set_dim_name(space,
						isl_dim_set, i, prefix_name);
		free(prefix_name);
	}

	return space;
error:
	isl_space_free(space);
	return NULL;
}

/* Given a dimension specification of the solutions space, construct
 * a dimension specification for the space of coefficients.
 *
 * In particular transform
 *
 *	[params] -> { S }
 *
 * to
 *
 *	{ coefficients[[cst, params] -> S] }
 *
 * and prefix each dimension name with "c_".
 */
static __isl_give isl_space *isl_space_coefficients(__isl_take isl_space *space)
{
	isl_space *space_param;
	isl_size nvar;
	isl_size nparam;

	nvar = isl_space_dim(space, isl_dim_set);
	nparam = isl_space_dim(space, isl_dim_param);
	if (nvar < 0 || nparam < 0)
		return isl_space_free(space);
	space_param = isl_space_copy(space);
	space_param = isl_space_drop_dims(space_param, isl_dim_set, 0, nvar);
	space_param = isl_space_move_dims(space_param, isl_dim_set, 0,
				 isl_dim_param, 0, nparam);
	space_param = isl_space_prefix(space_param, "c_");
	space_param = isl_space_insert_dims(space_param, isl_dim_set, 0, 1);
	space_param = isl_space_set_dim_name(space_param,
				isl_dim_set, 0, "c_cst");
	space = isl_space_drop_dims(space, isl_dim_param, 0, nparam);
	space = isl_space_prefix(space, "c_");
	space = isl_space_join(isl_space_from_domain(space_param),
			   isl_space_from_range(space));
	space = isl_space_wrap(space);
	space = isl_space_set_tuple_name(space, isl_dim_set, "coefficients");

	return space;
}

/* Drop the given prefix from all named dimensions of type "type" in "space".
 */
static __isl_give isl_space *isl_space_unprefix(__isl_take isl_space *space,
	enum isl_dim_type type, const char *prefix)
{
	int i;
	isl_size n;
	size_t prefix_len = strlen(prefix);

	n = isl_space_dim(space, type);
	if (n < 0)
		return isl_space_free(space);

	for (i = 0; i < n; ++i) {
		const char *name;

		name = isl_space_get_dim_name(space, type, i);
		if (!name)
			continue;
		if (strncmp(name, prefix, prefix_len))
			continue;

		space = isl_space_set_dim_name(space,
						type, i, name + prefix_len);
	}

	return space;
}

/* Given a dimension specification of the space of coefficients, construct
 * a dimension specification for the space of solutions.
 *
 * In particular transform
 *
 *	{ coefficients[[cst, params] -> S] }
 *
 * to
 *
 *	[params] -> { S }
 *
 * and drop the "c_" prefix from the dimension names.
 */
static __isl_give isl_space *isl_space_solutions(__isl_take isl_space *space)
{
	isl_size nparam;

	space = isl_space_unwrap(space);
	space = isl_space_drop_dims(space, isl_dim_in, 0, 1);
	space = isl_space_unprefix(space, isl_dim_in, "c_");
	space = isl_space_unprefix(space, isl_dim_out, "c_");
	nparam = isl_space_dim(space, isl_dim_in);
	if (nparam < 0)
		return isl_space_free(space);
	space = isl_space_move_dims(space,
				    isl_dim_param, 0, isl_dim_in, 0, nparam);
	space = isl_space_range(space);

	return space;
}

/* Return the rational universe basic set in the given space.
 */
static __isl_give isl_basic_set *rational_universe(__isl_take isl_space *space)
{
	isl_basic_set *bset;

	bset = isl_basic_set_universe(space);
	bset = isl_basic_set_set_rational(bset);

	return bset;
}

/* Compute the dual of "bset" by applying Farkas' lemma.
 * As explained above, we add an extra dimension to represent
 * the coefficient of the constant term when going from solutions
 * to coefficients (shift == 1) and we drop the extra dimension when going
 * in the opposite direction (shift == -1).  "space" is the space in which
 * the dual should be created.
 *
 * If "bset" is (obviously) empty, then the way this emptiness
 * is represented by the constraints does not allow for the application
 * of the standard farkas algorithm.  We therefore handle this case
 * specifically and return the universe basic set.
 */
static __isl_give isl_basic_set *farkas(__isl_take isl_space *space,
	__isl_take isl_basic_set *bset, int shift)
{
	int i, j, k;
	isl_basic_set *dual = NULL;
	isl_size total;

	if (isl_basic_set_plain_is_empty(bset)) {
		isl_basic_set_free(bset);
		return rational_universe(space);
	}

	total = isl_basic_set_dim(bset, isl_dim_all);
	if (total < 0)
		space = isl_space_free(space);

	dual = isl_basic_set_alloc_space(space, bset->n_eq + bset->n_ineq,
					total, bset->n_ineq + (shift > 0));
	dual = isl_basic_set_set_rational(dual);

	for (i = 0; i < bset->n_eq + bset->n_ineq; ++i) {
		k = isl_basic_set_alloc_div(dual);
		if (k < 0)
			goto error;
		isl_int_set_si(dual->div[k][0], 0);
	}

	for (i = 0; i < total; ++i) {
		k = isl_basic_set_alloc_equality(dual);
		if (k < 0)
			goto error;
		isl_seq_clr(dual->eq[k], 1 + shift + total);
		isl_int_set_si(dual->eq[k][1 + shift + i], -1);
		for (j = 0; j < bset->n_eq; ++j)
			isl_int_set(dual->eq[k][1 + shift + total + j],
				    bset->eq[j][1 + i]);
		for (j = 0; j < bset->n_ineq; ++j)
			isl_int_set(dual->eq[k][1 + shift + total + bset->n_eq + j],
				    bset->ineq[j][1 + i]);
	}

	for (i = 0; i < bset->n_ineq; ++i) {
		k = isl_basic_set_alloc_inequality(dual);
		if (k < 0)
			goto error;
		isl_seq_clr(dual->ineq[k],
			    1 + shift + total + bset->n_eq + bset->n_ineq);
		isl_int_set_si(dual->ineq[k][1 + shift + total + bset->n_eq + i], 1);
	}

	if (shift > 0) {
		k = isl_basic_set_alloc_inequality(dual);
		if (k < 0)
			goto error;
		isl_seq_clr(dual->ineq[k], 2 + total);
		isl_int_set_si(dual->ineq[k][1], 1);
		for (j = 0; j < bset->n_eq; ++j)
			isl_int_neg(dual->ineq[k][2 + total + j],
				    bset->eq[j][0]);
		for (j = 0; j < bset->n_ineq; ++j)
			isl_int_neg(dual->ineq[k][2 + total + bset->n_eq + j],
				    bset->ineq[j][0]);
	}

	dual = isl_basic_set_remove_divs(dual);
	dual = isl_basic_set_simplify(dual);
	dual = isl_basic_set_finalize(dual);

	isl_basic_set_free(bset);
	return dual;
error:
	isl_basic_set_free(bset);
	isl_basic_set_free(dual);
	return NULL;
}

/* Construct a basic set containing the tuples of coefficients of all
 * valid affine constraints on the given basic set.
 */
__isl_give isl_basic_set *isl_basic_set_coefficients(
	__isl_take isl_basic_set *bset)
{
	isl_space *space;

	if (!bset)
		return NULL;
	if (bset->n_div)
		isl_die(bset->ctx, isl_error_invalid,
			"input set not allowed to have local variables",
			goto error);

	space = isl_basic_set_get_space(bset);
	space = isl_space_coefficients(space);

	return farkas(space, bset, 1);
error:
	isl_basic_set_free(bset);
	return NULL;
}

/* Construct a basic set containing the elements that satisfy all
 * affine constraints whose coefficient tuples are
 * contained in the given basic set.
 */
__isl_give isl_basic_set *isl_basic_set_solutions(
	__isl_take isl_basic_set *bset)
{
	isl_space *space;

	if (!bset)
		return NULL;
	if (bset->n_div)
		isl_die(bset->ctx, isl_error_invalid,
			"input set not allowed to have local variables",
			goto error);

	space = isl_basic_set_get_space(bset);
	space = isl_space_solutions(space);

	return farkas(space, bset, -1);
error:
	isl_basic_set_free(bset);
	return NULL;
}

/* Construct a basic set containing the tuples of coefficients of all
 * valid affine constraints on the given set.
 */
__isl_give isl_basic_set *isl_set_coefficients(__isl_take isl_set *set)
{
	int i;
	isl_basic_set *coeff;

	if (!set)
		return NULL;
	if (set->n == 0) {
		isl_space *space = isl_set_get_space(set);
		space = isl_space_coefficients(space);
		isl_set_free(set);
		return rational_universe(space);
	}

	coeff = isl_basic_set_coefficients(isl_basic_set_copy(set->p[0]));

	for (i = 1; i < set->n; ++i) {
		isl_basic_set *bset, *coeff_i;
		bset = isl_basic_set_copy(set->p[i]);
		coeff_i = isl_basic_set_coefficients(bset);
		coeff = isl_basic_set_intersect(coeff, coeff_i);
	}

	isl_set_free(set);
	return coeff;
}

/* Wrapper around isl_basic_set_coefficients for use
 * as a isl_basic_set_list_map callback.
 */
static __isl_give isl_basic_set *coefficients_wrap(
	__isl_take isl_basic_set *bset, void *user)
{
	return isl_basic_set_coefficients(bset);
}

/* Replace the elements of "list" by the result of applying
 * isl_basic_set_coefficients to them.
 */
__isl_give isl_basic_set_list *isl_basic_set_list_coefficients(
	__isl_take isl_basic_set_list *list)
{
	return isl_basic_set_list_map(list, &coefficients_wrap, NULL);
}

/* Construct a basic set containing the elements that satisfy all
 * affine constraints whose coefficient tuples are
 * contained in the given set.
 */
__isl_give isl_basic_set *isl_set_solutions(__isl_take isl_set *set)
{
	int i;
	isl_basic_set *sol;

	if (!set)
		return NULL;
	if (set->n == 0) {
		isl_space *space = isl_set_get_space(set);
		space = isl_space_solutions(space);
		isl_set_free(set);
		return rational_universe(space);
	}

	sol = isl_basic_set_solutions(isl_basic_set_copy(set->p[0]));

	for (i = 1; i < set->n; ++i) {
		isl_basic_set *bset, *sol_i;
		bset = isl_basic_set_copy(set->p[i]);
		sol_i = isl_basic_set_solutions(bset);
		sol = isl_basic_set_intersect(sol, sol_i);
	}

	isl_set_free(set);
	return sol;
}
