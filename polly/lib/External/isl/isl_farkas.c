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
#include <isl_aff_private.h>
#include <isl_mat_private.h>
#include <isl_factorization.h>

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
 * in the opposite direction (shift == -1).
 * The dual can be created in an arbitrary space.
 * The caller is responsible for putting the result in the appropriate space.
 *
 * If "bset" is (obviously) empty, then the way this emptiness
 * is represented by the constraints does not allow for the application
 * of the standard farkas algorithm.  We therefore handle this case
 * specifically and return the universe basic set.
 */
static __isl_give isl_basic_set *farkas(__isl_take isl_basic_set *bset,
	int shift)
{
	int i, j, k;
	isl_ctx *ctx;
	isl_space *space;
	isl_basic_set *dual = NULL;
	isl_size total;

	total = isl_basic_set_dim(bset, isl_dim_all);
	if (total < 0)
		return isl_basic_set_free(bset);

	ctx = isl_basic_set_get_ctx(bset);
	space = isl_space_set_alloc(ctx, 0, total + shift);
	if (isl_basic_set_plain_is_empty(bset)) {
		isl_basic_set_free(bset);
		return rational_universe(space);
	}

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
 * valid affine constraints on the given basic set, ignoring
 * the space of input and output and without any further decomposition.
 */
static __isl_give isl_basic_set *isl_basic_set_coefficients_base(
	__isl_take isl_basic_set *bset)
{
	return farkas(bset, 1);
}

/* Return the inverse mapping of "morph".
 */
static __isl_give isl_mat *peek_inv(__isl_keep isl_morph *morph)
{
	return morph ? morph->inv : NULL;
}

/* Return a copy of the inverse mapping of "morph".
 */
static __isl_give isl_mat *get_inv(__isl_keep isl_morph *morph)
{
	return isl_mat_copy(peek_inv(morph));
}

/* Information about a single factor within isl_basic_set_coefficients_product.
 *
 * "start" is the position of the first coefficient (beyond
 * the one corresponding to the constant term) in this factor.
 * "dim" is the number of coefficients (other than
 * the one corresponding to the constant term) in this factor.
 * "n_line" is the number of lines in "coeff".
 * "n_ray" is the number of rays (other than lines) in "coeff".
 * "n_vertex" is the number of vertices in "coeff".
 *
 * While iterating over the vertices,
 * "pos" represents the inequality constraint corresponding
 * to the current vertex.
 */
struct isl_coefficients_factor_data {
	isl_basic_set *coeff;
	int start;
	int dim;
	int n_line;
	int n_ray;
	int n_vertex;
	int pos;
};

/* Internal data structure for isl_basic_set_coefficients_product.
 * "n" is the number of factors in the factorization.
 * "pos" is the next factor that will be considered.
 * "start_next" is the position of the first coefficient (beyond
 * the one corresponding to the constant term) in the next factor.
 * "factors" contains information about the individual "n" factors.
 */
struct isl_coefficients_product_data {
	int n;
	int pos;
	int start_next;
	struct isl_coefficients_factor_data *factors;
};

/* Initialize the internal data structure for
 * isl_basic_set_coefficients_product.
 */
static isl_stat isl_coefficients_product_data_init(isl_ctx *ctx,
	struct isl_coefficients_product_data *data, int n)
{
	data->n = n;
	data->pos = 0;
	data->start_next = 0;
	data->factors = isl_calloc_array(ctx,
					struct isl_coefficients_factor_data, n);
	if (!data->factors)
		return isl_stat_error;
	return isl_stat_ok;
}

/* Free all memory allocated in "data".
 */
static void isl_coefficients_product_data_clear(
	struct isl_coefficients_product_data *data)
{
	int i;

	if (data->factors) {
		for (i = 0; i < data->n; ++i) {
			isl_basic_set_free(data->factors[i].coeff);
		}
	}
	free(data->factors);
}

/* Does inequality "ineq" in the (dual) basic set "bset" represent a ray?
 * In particular, does it have a zero denominator
 * (i.e., a zero coefficient for the constant term)?
 */
static int is_ray(__isl_keep isl_basic_set *bset, int ineq)
{
	return isl_int_is_zero(bset->ineq[ineq][1]);
}

/* isl_factorizer_every_factor_basic_set callback that
 * constructs a basic set containing the tuples of coefficients of all
 * valid affine constraints on the factor "bset" and
 * extracts further information that will be used
 * when combining the results over the different factors.
 */
static isl_bool isl_basic_set_coefficients_factor(
	__isl_keep isl_basic_set *bset, void *user)
{
	struct isl_coefficients_product_data *data = user;
	isl_basic_set *coeff;
	isl_size n_eq, n_ineq, dim;
	int i, n_ray, n_vertex;

	coeff = isl_basic_set_coefficients_base(isl_basic_set_copy(bset));
	data->factors[data->pos].coeff = coeff;
	if (!coeff)
		return isl_bool_error;

	dim = isl_basic_set_dim(bset, isl_dim_set);
	n_eq = isl_basic_set_n_equality(coeff);
	n_ineq = isl_basic_set_n_inequality(coeff);
	if (dim < 0 || n_eq < 0 || n_ineq < 0)
		return isl_bool_error;
	n_ray = n_vertex = 0;
	for (i = 0; i < n_ineq; ++i) {
		if (is_ray(coeff, i))
			n_ray++;
		else
			n_vertex++;
	}
	data->factors[data->pos].start = data->start_next;
	data->factors[data->pos].dim = dim;
	data->factors[data->pos].n_line = n_eq;
	data->factors[data->pos].n_ray = n_ray;
	data->factors[data->pos].n_vertex = n_vertex;
	data->pos++;
	data->start_next += dim;

	return isl_bool_true;
}

/* Clear an entry in the product, given that there is a "total" number
 * of coefficients (other than that of the constant term).
 */
static void clear_entry(isl_int *entry, int total)
{
	isl_seq_clr(entry, 1 + 1 + total);
}

/* Set the part of the entry corresponding to factor "data",
 * from the factor coefficients in "src".
 */
static void set_factor(isl_int *entry, isl_int *src,
	struct isl_coefficients_factor_data *data)
{
	isl_seq_cpy(entry + 1 + 1 + data->start, src + 1 + 1, data->dim);
}

/* Set the part of the entry corresponding to factor "data",
 * from the factor coefficients in "src" multiplied by "f".
 */
static void scale_factor(isl_int *entry, isl_int *src, isl_int f,
	struct isl_coefficients_factor_data *data)
{
	isl_seq_scale(entry + 1 + 1 + data->start, src + 1 + 1, f, data->dim);
}

/* Add all lines from the given factor to "bset",
 * given that there is a "total" number of coefficients
 * (other than that of the constant term).
 */
static __isl_give isl_basic_set *add_lines(__isl_take isl_basic_set *bset,
	struct isl_coefficients_factor_data *factor, int total)
{
	int i;

	for (i = 0; i < factor->n_line; ++i) {
		int k;

		k = isl_basic_set_alloc_equality(bset);
		if (k < 0)
			return isl_basic_set_free(bset);
		clear_entry(bset->eq[k], total);
		set_factor(bset->eq[k], factor->coeff->eq[i], factor);
	}

	return bset;
}

/* Add all rays (other than lines) from the given factor to "bset",
 * given that there is a "total" number of coefficients
 * (other than that of the constant term).
 */
static __isl_give isl_basic_set *add_rays(__isl_take isl_basic_set *bset,
	struct isl_coefficients_factor_data *data, int total)
{
	int i;
	int n_ineq = data->n_ray + data->n_vertex;

	for (i = 0; i < n_ineq; ++i) {
		int k;

		if (!is_ray(data->coeff, i))
			continue;

		k = isl_basic_set_alloc_inequality(bset);
		if (k < 0)
			return isl_basic_set_free(bset);
		clear_entry(bset->ineq[k], total);
		set_factor(bset->ineq[k], data->coeff->ineq[i], data);
	}

	return bset;
}

/* Move to the first vertex of the given factor starting
 * at inequality constraint "start", setting factor->pos and
 * returning 1 if a vertex is found.
 */
static int factor_first_vertex(struct isl_coefficients_factor_data *factor,
	int start)
{
	int j;
	int n = factor->n_ray + factor->n_vertex;

	for (j = start; j < n; ++j) {
		if (is_ray(factor->coeff, j))
			continue;
		factor->pos = j;
		return 1;
	}

	return 0;
}

/* Move to the first constraint in each factor starting at "first"
 * that represents a vertex.
 * In particular, skip the initial constraints that correspond to rays.
 */
static void first_vertex(struct isl_coefficients_product_data *data, int first)
{
	int i;

	for (i = first; i < data->n; ++i)
		factor_first_vertex(&data->factors[i], 0);
}

/* Move to the next vertex in the product.
 * In particular, move to the next vertex of the last factor.
 * If all vertices of this last factor have already been considered,
 * then move to the next vertex of the previous factor(s)
 * until a factor is found that still has a next vertex.
 * Once such a next vertex has been found, the subsequent
 * factors are reset to the first vertex.
 * Return 1 if any next vertex was found.
 */
static int next_vertex(struct isl_coefficients_product_data *data)
{
	int i;

	for (i = data->n - 1; i >= 0; --i) {
		struct isl_coefficients_factor_data *factor = &data->factors[i];

		if (!factor_first_vertex(factor, factor->pos + 1))
			continue;
		first_vertex(data, i + 1);
		return 1;
	}

	return 0;
}

/* Add a vertex to the product "bset" combining the currently selected
 * vertices of the factors.
 *
 * In the dual representation, the constant term is always zero.
 * The vertex itself is the sum of the contributions of the factors
 * with a shared denominator in position 1.
 *
 * First compute the shared denominator (lcm) and
 * then scale the numerators to this shared denominator.
 */
static __isl_give isl_basic_set *add_vertex(__isl_take isl_basic_set *bset,
	struct isl_coefficients_product_data *data)
{
	int i;
	int k;
	isl_int lcm, f;

	k = isl_basic_set_alloc_inequality(bset);
	if (k < 0)
		return isl_basic_set_free(bset);

	isl_int_init(lcm);
	isl_int_init(f);
	isl_int_set_si(lcm, 1);
	for (i = 0; i < data->n; ++i) {
		struct isl_coefficients_factor_data *factor = &data->factors[i];
		isl_basic_set *coeff = factor->coeff;
		int pos = factor->pos;
		isl_int_lcm(lcm, lcm, coeff->ineq[pos][1]);
	}
	isl_int_set_si(bset->ineq[k][0], 0);
	isl_int_set(bset->ineq[k][1], lcm);

	for (i = 0; i < data->n; ++i) {
		struct isl_coefficients_factor_data *factor = &data->factors[i];
		isl_basic_set *coeff = factor->coeff;
		int pos = factor->pos;
		isl_int_divexact(f, lcm, coeff->ineq[pos][1]);
		scale_factor(bset->ineq[k], coeff->ineq[pos], f, factor);
	}

	isl_int_clear(f);
	isl_int_clear(lcm);

	return bset;
}

/* Combine the duals of the factors in the factorization of a basic set
 * to form the dual of the entire basic set.
 * The dual share the coefficient of the constant term.
 * All other coefficients are specific to a factor.
 * Any constraint not involving the coefficient of the constant term
 * can therefor simply be copied into the appropriate position.
 * This includes all equality constraints since the coefficient
 * of the constant term can always be increased and therefore
 * never appears in an equality constraint.
 * The inequality constraints involving the coefficient of
 * the constant term need to be combined across factors.
 * In particular, if this coefficient needs to be greater than or equal
 * to some linear combination of the other coefficients in each factor,
 * then it needs to be greater than or equal to the sum of
 * these linear combinations across the factors.
 *
 * Alternatively, the constraints of the dual can be seen
 * as the vertices, rays and lines of the original basic set.
 * Clearly, rays and lines can simply be copied,
 * while vertices needs to be combined across factors.
 * This means that the number of rays and lines in the product
 * is equal to the sum of the numbers in the factors,
 * while the number of vertices is the product
 * of the number of vertices in the factors.  Note that each
 * factor has at least one vertex.
 * The only exception is when the factor is the dual of an obviously empty set,
 * in which case a universe dual is created.
 * In this case, return a universe dual for the product as well.
 *
 * While constructing the vertices, look for the first combination
 * of inequality constraints that represent a vertex,
 * construct the corresponding vertex and then move on
 * to the next combination of inequality constraints until
 * all combinations have been considered.
 */
static __isl_give isl_basic_set *construct_product(isl_ctx *ctx,
	struct isl_coefficients_product_data *data)
{
	int i;
	int n_line, n_ray, n_vertex;
	int total;
	isl_space *space;
	isl_basic_set *product;

	if (!data->factors)
		return NULL;

	total = data->start_next;

	n_line = 0;
	n_ray = 0;
	n_vertex = 1;
	for (i = 0; i < data->n; ++i) {
		n_line += data->factors[i].n_line;
		n_ray += data->factors[i].n_ray;
		n_vertex *= data->factors[i].n_vertex;
	}

	space = isl_space_set_alloc(ctx, 0, 1 + total);
	if (n_vertex == 0)
		return rational_universe(space);
	product = isl_basic_set_alloc_space(space, 0, n_line, n_ray + n_vertex);
	product = isl_basic_set_set_rational(product);

	for (i = 0; i < data->n; ++i)
		product = add_lines(product, &data->factors[i], total);
	for (i = 0; i < data->n; ++i)
		product = add_rays(product, &data->factors[i], total);

	first_vertex(data, 0);
	do {
		product = add_vertex(product, data);
	} while (next_vertex(data));

	return product;
}

/* Given a factorization "f" of a basic set,
 * construct a basic set containing the tuples of coefficients of all
 * valid affine constraints on the product of the factors, ignoring
 * the space of input and output.
 * Note that this product may not be equal to the original basic set,
 * if a non-trivial transformation is involved.
 * This is handled by the caller.
 *
 * Compute the tuples of coefficients for each factor separately and
 * then combine the results.
 */
static __isl_give isl_basic_set *isl_basic_set_coefficients_product(
	__isl_take isl_factorizer *f)
{
	struct isl_coefficients_product_data data;
	isl_ctx *ctx;
	isl_basic_set *coeff;
	isl_bool every;

	ctx = isl_factorizer_get_ctx(f);
	if (isl_coefficients_product_data_init(ctx, &data, f->n_group) < 0)
		f = isl_factorizer_free(f);
	every = isl_factorizer_every_factor_basic_set(f,
			&isl_basic_set_coefficients_factor, &data);
	isl_factorizer_free(f);
	if (every >= 0)
		coeff = construct_product(ctx, &data);
	else
		coeff = NULL;
	isl_coefficients_product_data_clear(&data);

	return coeff;
}

/* Given a factorization "f" of a basic set,
 * construct a basic set containing the tuples of coefficients of all
 * valid affine constraints on the basic set, ignoring
 * the space of input and output.
 *
 * The factorization may involve a linear transformation of the basic set.
 * In particular, the transformed basic set is formulated
 * in terms of x' = U x, i.e., x = V x', with V = U^{-1}.
 * The dual is then computed in terms of y' with y'^t [z; x'] >= 0.
 * Plugging in y' = [1 0; 0 V^t] y yields
 * y^t [1 0; 0 V] [z; x'] >= 0, i.e., y^t [z; x] >= 0, which is
 * the desired set of coefficients y.
 * Note that this transformation to y' only needs to be applied
 * if U is not the identity matrix.
 */
static __isl_give isl_basic_set *isl_basic_set_coefficients_morphed_product(
	__isl_take isl_factorizer *f)
{
	isl_bool is_identity;
	isl_space *space;
	isl_mat *inv;
	isl_multi_aff *ma;
	isl_basic_set *coeff;

	if (!f)
		goto error;
	is_identity = isl_mat_is_scaled_identity(peek_inv(f->morph));
	if (is_identity < 0)
		goto error;
	if (is_identity)
		return isl_basic_set_coefficients_product(f);

	inv = get_inv(f->morph);
	inv = isl_mat_transpose(inv);
	inv = isl_mat_lin_to_aff(inv);

	coeff = isl_basic_set_coefficients_product(f);
	space = isl_space_map_from_set(isl_basic_set_get_space(coeff));
	ma = isl_multi_aff_from_aff_mat(space, inv);
	coeff = isl_basic_set_preimage_multi_aff(coeff, ma);

	return coeff;
error:
	isl_factorizer_free(f);
	return NULL;
}

/* Construct a basic set containing the tuples of coefficients of all
 * valid affine constraints on the given basic set, ignoring
 * the space of input and output.
 *
 * The caller has already checked that "bset" does not involve
 * any local variables.  It may have parameters, though.
 * Treat them as regular variables internally.
 * This is especially important for the factorization,
 * since the (original) parameters should be taken into account
 * explicitly in this factorization.
 *
 * Check if the basic set can be factorized.
 * If so, compute constraints on the coefficients of the factors
 * separately and combine the results.
 * Otherwise, compute the results for the input basic set as a whole.
 */
static __isl_give isl_basic_set *basic_set_coefficients(
	__isl_take isl_basic_set *bset)
{
	isl_factorizer *f;
	isl_size nparam;

	nparam = isl_basic_set_dim(bset, isl_dim_param);
	if (nparam < 0)
		return isl_basic_set_free(bset);
	bset = isl_basic_set_move_dims(bset, isl_dim_set, 0,
					    isl_dim_param, 0, nparam);

	f = isl_basic_set_factorizer(bset);
	if (!f)
		return isl_basic_set_free(bset);
	if (f->n_group > 0) {
		isl_basic_set_free(bset);
		return isl_basic_set_coefficients_morphed_product(f);
	}
	isl_factorizer_free(f);
	return isl_basic_set_coefficients_base(bset);
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

	bset = basic_set_coefficients(bset);
	bset = isl_basic_set_reset_space(bset, space);
	return bset;
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

	bset = farkas(bset, -1);
	bset = isl_basic_set_reset_space(bset, space);
	return bset;
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
