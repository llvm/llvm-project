/*
 * Copyright 2011      INRIA Saclay
 * Copyright 2014      Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, INRIA Saclay - Ile-de-France,
 * Parc Club Orsay Universite, ZAC des vignes, 4 rue Jacques Monod,
 * 91893 Orsay, France
 * and Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 */

#include <isl/space.h>
#include <isl_vec_private.h>
#include <isl_mat_private.h>
#include <isl_seq.h>
#include <isl_local.h>

/* Return the isl_ctx to which "local" belongs.
 */
isl_ctx *isl_local_get_ctx(__isl_keep isl_local *local)
{
	if (!local)
		return NULL;

	return isl_mat_get_ctx(local);
}

/* Return the number of local variables (isl_dim_div),
 * the number of other variables (isl_dim_set) or
 * the total number of variables (isl_dim_all) in "local".
 *
 * Other types do not have any meaning for an isl_local object.
 */
int isl_local_dim(__isl_keep isl_local *local, enum isl_dim_type type)
{
	isl_mat *mat = local;

	if (!local)
		return 0;
	if (type == isl_dim_div)
		return isl_mat_rows(mat);
	if (type == isl_dim_all)
		return isl_mat_cols(mat) - 2;
	if (type == isl_dim_set)
		return isl_local_dim(local, isl_dim_all) -
			isl_local_dim(local, isl_dim_div);
	isl_die(isl_local_get_ctx(local), isl_error_unsupported,
		"unsupported dimension type", return 0);
}

/* Check that "pos" is a valid position for a variable in "local".
 */
static isl_stat isl_local_check_pos(__isl_keep isl_local *local, int pos)
{
	if (!local)
		return isl_stat_error;
	if (pos < 0 || pos >= isl_local_dim(local, isl_dim_div))
		isl_die(isl_local_get_ctx(local), isl_error_invalid,
			"position out of bounds", return isl_stat_error);
	return isl_stat_ok;
}

/* Given local variables "local",
 * is the variable at position "pos" marked as not having
 * an explicit representation?
 * Note that even if this variable is not marked in this way and therefore
 * does have an explicit representation, this representation may still
 * depend (indirectly) on other local variables that do not
 * have an explicit representation.
 */
isl_bool isl_local_div_is_marked_unknown(__isl_keep isl_local *local, int pos)
{
	isl_mat *mat = local;

	if (isl_local_check_pos(local, pos) < 0)
		return isl_bool_error;
	return isl_int_is_zero(mat->row[pos][0]);
}

/* Given local variables "local",
 * does the variable at position "pos" have a complete explicit representation?
 * Having a complete explicit representation requires not only
 * an explicit representation, but also that all local variables
 * that appear in this explicit representation in turn have
 * a complete explicit representation.
 */
isl_bool isl_local_div_is_known(__isl_keep isl_local *local, int pos)
{
	isl_bool marked;
	int i, n, off;
	isl_mat *mat = local;

	if (isl_local_check_pos(local, pos) < 0)
		return isl_bool_error;

	marked = isl_local_div_is_marked_unknown(local, pos);
	if (marked < 0 || marked)
		return isl_bool_not(marked);

	n = isl_local_dim(local, isl_dim_div);
	off = isl_mat_cols(mat) - n;

	for (i = n - 1; i >= 0; --i) {
		isl_bool known;

		if (isl_int_is_zero(mat->row[pos][off + i]))
			continue;
		known = isl_local_div_is_known(local, i);
		if (known < 0 || !known)
			return known;
	}

	return isl_bool_true;
}

/* Does "local" have an explicit representation for all local variables?
 */
isl_bool isl_local_divs_known(__isl_keep isl_local *local)
{
	int i, n;

	if (!local)
		return isl_bool_error;

	n = isl_local_dim(local, isl_dim_div);
	for (i = 0; i < n; ++i) {
		isl_bool unknown = isl_local_div_is_marked_unknown(local, i);
		if (unknown < 0 || unknown)
			return isl_bool_not(unknown);
	}

	return isl_bool_true;
}

/* Compare two sets of local variables, defined over
 * the same space.
 *
 * Return -1 if "local1" is "smaller" than "local2", 1 if "local1" is "greater"
 * than "local2" and 0 if they are equal.
 *
 * The order is fairly arbitrary.  We do "prefer" divs that only involve
 * earlier dimensions in the sense that we consider matrices where
 * the first differing div involves earlier dimensions to be smaller.
 */
int isl_local_cmp(__isl_keep isl_local *local1, __isl_keep isl_local *local2)
{
	int i;
	int cmp;
	isl_bool unknown1, unknown2;
	int last1, last2;
	int n_col;
	isl_mat *mat1 = local1;
	isl_mat *mat2 = local2;

	if (local1 == local2)
		return 0;
	if (!local1)
		return -1;
	if (!local2)
		return 1;

	if (mat1->n_row != mat2->n_row)
		return mat1->n_row - mat2->n_row;

	n_col = isl_mat_cols(mat1);
	for (i = 0; i < mat1->n_row; ++i) {
		unknown1 = isl_local_div_is_marked_unknown(local1, i);
		unknown2 = isl_local_div_is_marked_unknown(local2, i);
		if (unknown1 && unknown2)
			continue;
		if (unknown1)
			return 1;
		if (unknown2)
			return -1;
		last1 = isl_seq_last_non_zero(mat1->row[i] + 1, n_col - 1);
		last2 = isl_seq_last_non_zero(mat2->row[i] + 1, n_col - 1);
		if (last1 != last2)
			return last1 - last2;
		cmp = isl_seq_cmp(mat1->row[i], mat2->row[i], n_col);
		if (cmp != 0)
			return cmp;
	}

	return 0;
}

/* Extend a vector "v" representing an integer point
 * in the domain space of "local"
 * to one that also includes values for the local variables.
 * All local variables are required to have an explicit representation.
 */
__isl_give isl_vec *isl_local_extend_point_vec(__isl_keep isl_local *local,
	__isl_take isl_vec *v)
{
	unsigned n_div;
	isl_bool known;
	isl_mat *mat = local;

	if (!local || !v)
		return isl_vec_free(v);
	known = isl_local_divs_known(local);
	if (known < 0)
		return isl_vec_free(v);
	if (!known)
		isl_die(isl_local_get_ctx(local), isl_error_invalid,
			"unknown local variables", return isl_vec_free(v));
	if (isl_vec_size(v) != 1 + isl_local_dim(local, isl_dim_set))
		isl_die(isl_local_get_ctx(local), isl_error_invalid,
			"incorrect size", return isl_vec_free(v));
	if (!isl_int_is_one(v->el[0]))
		isl_die(isl_local_get_ctx(local), isl_error_invalid,
			"expecting integer point", return isl_vec_free(v));
	n_div = isl_local_dim(local, isl_dim_div);
	if (n_div != 0) {
		int i;
		unsigned dim = isl_local_dim(local, isl_dim_set);
		v = isl_vec_add_els(v, n_div);
		if (!v)
			return NULL;

		for (i = 0; i < n_div; ++i) {
			isl_seq_inner_product(mat->row[i] + 1, v->el,
						1 + dim + i, &v->el[1+dim+i]);
			isl_int_fdiv_q(v->el[1+dim+i], v->el[1+dim+i],
					mat->row[i][0]);
		}
	}

	return v;
}
