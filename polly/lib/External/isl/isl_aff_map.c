/*
 * Copyright 2011      INRIA Saclay
 * Copyright 2012-2013 Ecole Normale Superieure
 * Copyright 2016      Sven Verdoolaege
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, INRIA Saclay - Ile-de-France,
 * Parc Club Orsay Universite, ZAC des vignes, 4 rue Jacques Monod,
 * 91893 Orsay, France
 * and Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 */

#include <isl/ctx.h>
#include <isl/space.h>
#include <isl/local_space.h>
#include <isl/union_map.h>
#include <isl_map_private.h>
#include <isl_aff_private.h>
#include <isl_vec_private.h>
#include <isl_seq.h>

#include <bset_from_bmap.c>
#include <set_from_map.c>

/* Check that the input living in "space" lives in a map space.
 * That is, check that "space" is a map space.
 */
static isl_stat check_input_is_map(__isl_keep isl_space *space)
{
	isl_bool is_set;

	is_set = isl_space_is_set(space);
	if (is_set < 0)
		return isl_stat_error;
	if (is_set)
		isl_die(isl_space_get_ctx(space), isl_error_invalid,
			"space of input is not a map", return isl_stat_error);
	return isl_stat_ok;
}

/* Check that the input living in "space" lives in a set space.
 * That is, check that "space" is a set space.
 */
static isl_stat check_input_is_set(__isl_keep isl_space *space)
{
	isl_bool is_set;

	is_set = isl_space_is_set(space);
	if (is_set < 0)
		return isl_stat_error;
	if (!is_set)
		isl_die(isl_space_get_ctx(space), isl_error_invalid,
			"space of input is not a set", return isl_stat_error);
	return isl_stat_ok;
}

/* Construct a basic map mapping the domain of the affine expression
 * to a one-dimensional range prescribed by the affine expression.
 * If "rational" is set, then construct a rational basic map.
 *
 * A NaN affine expression cannot be converted to a basic map.
 */
static __isl_give isl_basic_map *isl_basic_map_from_aff2(
	__isl_take isl_aff *aff, int rational)
{
	int k;
	int pos;
	isl_bool is_nan;
	isl_local_space *ls;
	isl_basic_map *bmap = NULL;

	if (!aff)
		return NULL;
	is_nan = isl_aff_is_nan(aff);
	if (is_nan < 0)
		goto error;
	if (is_nan)
		isl_die(isl_aff_get_ctx(aff), isl_error_invalid,
			"cannot convert NaN", goto error);

	ls = isl_aff_get_local_space(aff);
	bmap = isl_basic_map_from_local_space(ls);
	bmap = isl_basic_map_extend_constraints(bmap, 1, 0);
	k = isl_basic_map_alloc_equality(bmap);
	if (k < 0)
		goto error;

	pos = isl_basic_map_offset(bmap, isl_dim_out);
	isl_seq_cpy(bmap->eq[k], aff->v->el + 1, pos);
	isl_int_neg(bmap->eq[k][pos], aff->v->el[0]);
	isl_seq_cpy(bmap->eq[k] + pos + 1, aff->v->el + 1 + pos,
		    aff->v->size - (pos + 1));

	isl_aff_free(aff);
	if (rational)
		bmap = isl_basic_map_set_rational(bmap);
	bmap = isl_basic_map_gauss(bmap, NULL);
	bmap = isl_basic_map_finalize(bmap);
	return bmap;
error:
	isl_aff_free(aff);
	isl_basic_map_free(bmap);
	return NULL;
}

/* Construct a basic map mapping the domain of the affine expression
 * to a one-dimensional range prescribed by the affine expression.
 */
__isl_give isl_basic_map *isl_basic_map_from_aff(__isl_take isl_aff *aff)
{
	return isl_basic_map_from_aff2(aff, 0);
}

/* Construct a map mapping the domain of the affine expression
 * to a one-dimensional range prescribed by the affine expression.
 */
__isl_give isl_map *isl_map_from_aff(__isl_take isl_aff *aff)
{
	isl_basic_map *bmap;

	bmap = isl_basic_map_from_aff(aff);
	return isl_map_from_basic_map(bmap);
}

/* Construct a basic map mapping the domain of the multi-affine expression
 * to its range, with each dimension in the range equated to the
 * corresponding affine expression.
 * If "rational" is set, then construct a rational basic map.
 */
__isl_give isl_basic_map *isl_basic_map_from_multi_aff2(
	__isl_take isl_multi_aff *maff, int rational)
{
	int i;
	isl_size dim;
	isl_space *space;
	isl_basic_map *bmap;

	dim = isl_multi_aff_dim(maff, isl_dim_out);
	if (dim < 0)
		goto error;

	if (dim != maff->n)
		isl_die(isl_multi_aff_get_ctx(maff), isl_error_internal,
			"invalid space", goto error);

	space = isl_space_domain(isl_multi_aff_get_space(maff));
	bmap = isl_basic_map_universe(isl_space_from_domain(space));
	if (rational)
		bmap = isl_basic_map_set_rational(bmap);

	for (i = 0; i < maff->n; ++i) {
		isl_aff *aff;
		isl_basic_map *bmap_i;

		aff = isl_aff_copy(maff->u.p[i]);
		bmap_i = isl_basic_map_from_aff2(aff, rational);

		bmap = isl_basic_map_flat_range_product(bmap, bmap_i);
	}

	bmap = isl_basic_map_reset_space(bmap, isl_multi_aff_get_space(maff));

	isl_multi_aff_free(maff);
	return bmap;
error:
	isl_multi_aff_free(maff);
	return NULL;
}

/* Construct a basic map mapping the domain of the multi-affine expression
 * to its range, with each dimension in the range equated to the
 * corresponding affine expression.
 * If "ma" lives in a set space, then the result is actually a set.
 */
static __isl_give isl_basic_map *basic_map_from_multi_aff(
	__isl_take isl_multi_aff *ma)
{
	return isl_basic_map_from_multi_aff2(ma, 0);
}

/* Construct a basic map mapping the domain of the multi-affine expression
 * to its range, with each dimension in the range equated to the
 * corresponding affine expression.
 */
__isl_give isl_basic_map *isl_basic_map_from_multi_aff(
	__isl_take isl_multi_aff *ma)
{
	if (check_input_is_map(isl_multi_aff_peek_space(ma)) < 0)
		ma = isl_multi_aff_free(ma);
	return basic_map_from_multi_aff(ma);
}

/* Construct a basic set mapping the parameter domain
 * of the multi-affine expression to its space, with each dimension
 * in the space equated to the corresponding affine expression.
 */
__isl_give isl_basic_set *isl_basic_set_from_multi_aff(
	__isl_take isl_multi_aff *ma)
{
	if (check_input_is_set(isl_multi_aff_peek_space(ma)) < 0)
		ma = isl_multi_aff_free(ma);
	return bset_from_bmap(basic_map_from_multi_aff(ma));
}

/* Construct a map mapping the domain of the multi-affine expression
 * to its range, with each dimension in the range equated to the
 * corresponding affine expression.
 * If "maff" lives in a set space, then the result is actually a set.
 */
__isl_give isl_map *isl_map_from_multi_aff_internal(
	__isl_take isl_multi_aff *maff)
{
	isl_basic_map *bmap;

	bmap = basic_map_from_multi_aff(maff);
	return isl_map_from_basic_map(bmap);
}

/* Construct a map mapping the domain the multi-affine expression
 * to its range, with each dimension in the range equated to the
 * corresponding affine expression.
 */
__isl_give isl_map *isl_map_from_multi_aff(__isl_take isl_multi_aff *ma)
{
	if (check_input_is_map(isl_multi_aff_peek_space(ma)) < 0)
		ma = isl_multi_aff_free(ma);
	return isl_map_from_multi_aff_internal(ma);
}

/* Construct a set mapping the parameter domain the multi-affine expression
 * to its space, with each dimension in the space equated to the
 * corresponding affine expression.
 */
__isl_give isl_set *isl_set_from_multi_aff(__isl_take isl_multi_aff *ma)
{
	if (check_input_is_set(isl_multi_aff_peek_space(ma)) < 0)
		ma = isl_multi_aff_free(ma);
	return isl_map_from_multi_aff_internal(ma);
}

/* Construct a basic map mapping a domain in the given space to
 * to an n-dimensional range, with n the number of elements in the list,
 * where each coordinate in the range is prescribed by the
 * corresponding affine expression.
 * The domains of all affine expressions in the list are assumed to match
 * domain_space.
 */
__isl_give isl_basic_map *isl_basic_map_from_aff_list(
	__isl_take isl_space *domain_space, __isl_take isl_aff_list *list)
{
	int i;
	isl_space *space;
	isl_basic_map *bmap;

	if (!list)
		return NULL;

	space = isl_space_from_domain(domain_space);
	bmap = isl_basic_map_universe(space);

	for (i = 0; i < list->n; ++i) {
		isl_aff *aff;
		isl_basic_map *bmap_i;

		aff = isl_aff_copy(list->p[i]);
		bmap_i = isl_basic_map_from_aff(aff);

		bmap = isl_basic_map_flat_range_product(bmap, bmap_i);
	}

	isl_aff_list_free(list);
	return bmap;
}

/* Construct a map with as domain the domain of pwaff and
 * one-dimensional range corresponding to the affine expressions.
 * If "pwaff" lives in a set space, then the result is actually a set.
 */
__isl_give isl_map *isl_map_from_pw_aff_internal(__isl_take isl_pw_aff *pwaff)
{
	int i;
	isl_space *space;
	isl_map *map;

	if (!pwaff)
		return NULL;

	space = isl_pw_aff_get_space(pwaff);
	map = isl_map_empty(space);

	for (i = 0; i < pwaff->n; ++i) {
		isl_basic_map *bmap;
		isl_map *map_i;

		bmap = isl_basic_map_from_aff(isl_aff_copy(pwaff->p[i].aff));
		map_i = isl_map_from_basic_map(bmap);
		map_i = isl_map_intersect_domain(map_i,
						isl_set_copy(pwaff->p[i].set));
		map = isl_map_union_disjoint(map, map_i);
	}

	isl_pw_aff_free(pwaff);

	return map;
}

/* Construct a map with as domain the domain of pwaff and
 * one-dimensional range corresponding to the affine expressions.
 */
__isl_give isl_map *isl_map_from_pw_aff(__isl_take isl_pw_aff *pwaff)
{
	if (check_input_is_map(isl_pw_aff_peek_space(pwaff)) < 0)
		pwaff = isl_pw_aff_free(pwaff);
	return isl_map_from_pw_aff_internal(pwaff);
}

/* Construct a one-dimensional set with as parameter domain
 * the domain of pwaff and the single set dimension
 * corresponding to the affine expressions.
 */
__isl_give isl_set *isl_set_from_pw_aff(__isl_take isl_pw_aff *pwaff)
{
	if (check_input_is_set(isl_pw_aff_peek_space(pwaff)) < 0)
		pwaff = isl_pw_aff_free(pwaff);
	return set_from_map(isl_map_from_pw_aff_internal(pwaff));
}

/* Construct a map mapping the domain of the piecewise multi-affine expression
 * to its range, with each dimension in the range equated to the
 * corresponding affine expression on its cell.
 * If "pma" lives in a set space, then the result is actually a set.
 *
 * If the domain of "pma" is rational, then so is the constructed "map".
 */
__isl_give isl_map *isl_map_from_pw_multi_aff_internal(
	__isl_take isl_pw_multi_aff *pma)
{
	int i;
	isl_map *map;

	if (!pma)
		return NULL;

	map = isl_map_empty(isl_pw_multi_aff_get_space(pma));

	for (i = 0; i < pma->n; ++i) {
		isl_bool rational;
		isl_multi_aff *maff;
		isl_basic_map *bmap;
		isl_map *map_i;

		rational = isl_set_is_rational(pma->p[i].set);
		if (rational < 0)
			map = isl_map_free(map);
		maff = isl_multi_aff_copy(pma->p[i].maff);
		bmap = isl_basic_map_from_multi_aff2(maff, rational);
		map_i = isl_map_from_basic_map(bmap);
		map_i = isl_map_intersect_domain(map_i,
						isl_set_copy(pma->p[i].set));
		map = isl_map_union_disjoint(map, map_i);
	}

	isl_pw_multi_aff_free(pma);
	return map;
}

/* Construct a map mapping the domain of the piecewise multi-affine expression
 * to its range, with each dimension in the range equated to the
 * corresponding affine expression on its cell.
 */
__isl_give isl_map *isl_map_from_pw_multi_aff(__isl_take isl_pw_multi_aff *pma)
{
	if (check_input_is_map(isl_pw_multi_aff_peek_space(pma)) < 0)
		pma = isl_pw_multi_aff_free(pma);
	return isl_map_from_pw_multi_aff_internal(pma);
}

__isl_give isl_set *isl_set_from_pw_multi_aff(__isl_take isl_pw_multi_aff *pma)
{
	if (check_input_is_set(isl_pw_multi_aff_peek_space(pma)) < 0)
		pma = isl_pw_multi_aff_free(pma);
	return set_from_map(isl_map_from_pw_multi_aff_internal(pma));
}

/* Construct a set or map mapping the shared (parameter) domain
 * of the piecewise affine expressions to the range of "mpa"
 * with each dimension in the range equated to the
 * corresponding piecewise affine expression.
 */
static __isl_give isl_map *map_from_multi_pw_aff(
	__isl_take isl_multi_pw_aff *mpa)
{
	int i;
	isl_size dim;
	isl_space *space;
	isl_map *map;

	dim = isl_multi_pw_aff_dim(mpa, isl_dim_out);
	if (dim < 0)
		goto error;

	if (isl_space_dim(mpa->space, isl_dim_out) != mpa->n)
		isl_die(isl_multi_pw_aff_get_ctx(mpa), isl_error_internal,
			"invalid space", goto error);

	space = isl_multi_pw_aff_get_domain_space(mpa);
	map = isl_map_universe(isl_space_from_domain(space));

	for (i = 0; i < mpa->n; ++i) {
		isl_pw_aff *pa;
		isl_map *map_i;

		pa = isl_pw_aff_copy(mpa->u.p[i]);
		map_i = isl_map_from_pw_aff_internal(pa);

		map = isl_map_flat_range_product(map, map_i);
	}

	map = isl_map_reset_space(map, isl_multi_pw_aff_get_space(mpa));

	isl_multi_pw_aff_free(mpa);
	return map;
error:
	isl_multi_pw_aff_free(mpa);
	return NULL;
}

/* Construct a map mapping the shared domain
 * of the piecewise affine expressions to the range of "mpa"
 * with each dimension in the range equated to the
 * corresponding piecewise affine expression.
 */
__isl_give isl_map *isl_map_from_multi_pw_aff(__isl_take isl_multi_pw_aff *mpa)
{
	if (check_input_is_map(isl_multi_pw_aff_peek_space(mpa)) < 0)
		mpa = isl_multi_pw_aff_free(mpa);
	return map_from_multi_pw_aff(mpa);
}

/* Construct a set mapping the shared parameter domain
 * of the piecewise affine expressions to the space of "mpa"
 * with each dimension in the range equated to the
 * corresponding piecewise affine expression.
 */
__isl_give isl_set *isl_set_from_multi_pw_aff(__isl_take isl_multi_pw_aff *mpa)
{
	if (check_input_is_set(isl_multi_pw_aff_peek_space(mpa)) < 0)
		mpa = isl_multi_pw_aff_free(mpa);
	return set_from_map(map_from_multi_pw_aff(mpa));
}

/* Convert "pa" to an isl_map and add it to *umap.
 */
static isl_stat map_from_pw_aff_entry(__isl_take isl_pw_aff *pa, void *user)
{
	isl_union_map **umap = user;
	isl_map *map;

	map = isl_map_from_pw_aff(pa);
	*umap = isl_union_map_add_map(*umap, map);

	return *umap ? isl_stat_ok : isl_stat_error;
}

/* Construct a union map mapping the domain of the union
 * piecewise affine expression to its range, with the single output dimension
 * equated to the corresponding affine expressions on their cells.
 */
__isl_give isl_union_map *isl_union_map_from_union_pw_aff(
	__isl_take isl_union_pw_aff *upa)
{
	isl_space *space;
	isl_union_map *umap;

	if (!upa)
		return NULL;

	space = isl_union_pw_aff_get_space(upa);
	umap = isl_union_map_empty(space);

	if (isl_union_pw_aff_foreach_pw_aff(upa, &map_from_pw_aff_entry,
						&umap) < 0)
		umap = isl_union_map_free(umap);

	isl_union_pw_aff_free(upa);
	return umap;
}

/* Convert "pma" to an isl_map and add it to *umap.
 */
static isl_stat map_from_pw_multi_aff(__isl_take isl_pw_multi_aff *pma,
	void *user)
{
	isl_union_map **umap = user;
	isl_map *map;

	map = isl_map_from_pw_multi_aff(pma);
	*umap = isl_union_map_add_map(*umap, map);

	return isl_stat_ok;
}

/* Construct a union map mapping the domain of the union
 * piecewise multi-affine expression to its range, with each dimension
 * in the range equated to the corresponding affine expression on its cell.
 */
__isl_give isl_union_map *isl_union_map_from_union_pw_multi_aff(
	__isl_take isl_union_pw_multi_aff *upma)
{
	isl_space *space;
	isl_union_map *umap;

	if (!upma)
		return NULL;

	space = isl_union_pw_multi_aff_get_space(upma);
	umap = isl_union_map_empty(space);

	if (isl_union_pw_multi_aff_foreach_pw_multi_aff(upma,
					&map_from_pw_multi_aff, &umap) < 0)
		goto error;

	isl_union_pw_multi_aff_free(upma);
	return umap;
error:
	isl_union_pw_multi_aff_free(upma);
	isl_union_map_free(umap);
	return NULL;
}
