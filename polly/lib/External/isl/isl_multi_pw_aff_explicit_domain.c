/*
 * Copyright 2017      Sven Verdoolaege
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege.
 */

/* Initialize the explicit domain of "mpa".
 *
 * The explicit domain is initialized to a universe set
 * in the domain space.
 */
static __isl_give isl_multi_pw_aff *isl_multi_pw_aff_init_explicit_domain(
	__isl_take isl_multi_pw_aff *mpa)
{
	if (isl_multi_pw_aff_check_has_explicit_domain(mpa) < 0)
		return isl_multi_pw_aff_free(mpa);
	mpa->u.dom = isl_set_universe(isl_multi_pw_aff_get_domain_space(mpa));
	if (!mpa->u.dom)
		return isl_multi_pw_aff_free(mpa);
	return mpa;
}

/* Intersect the domain of "dst" with the domain product
 * of the explicit domains of "src1" and "src2".
 * This function is only called if at least one of "src1" or "src2"
 * has an explicit domain.
 */
static __isl_give isl_multi_pw_aff *
isl_multi_pw_aff_intersect_explicit_domain_product(
	__isl_take isl_multi_pw_aff *dst, __isl_keep isl_multi_pw_aff *src1,
	__isl_keep isl_multi_pw_aff *src2)
{
	isl_space *space;
	isl_set *dom;
	isl_map *map;

	if (!src1 || !src2)
		return FN(isl_multi_pw_aff,free)(dst);
	space = isl_multi_pw_aff_get_domain_space(dst);
	dom = isl_set_universe(space);
	map = isl_set_unwrap(dom);
	if (isl_multi_pw_aff_has_explicit_domain(src1)) {
		dom = isl_set_copy(src1->u.dom);
		map = isl_map_intersect_domain(map, dom);
	}
	if (isl_multi_pw_aff_has_explicit_domain(src2)) {
		dom = isl_set_copy(src2->u.dom);
		map = isl_map_intersect_range(map, dom);
	}
	dom = isl_map_wrap(map);
	dst = isl_multi_pw_aff_intersect_domain(dst, dom);
	return dst;
}

/* Check whether the explicit domain of "mpa" has non-zero coefficients
 * for any dimension in the given range or if any of these dimensions appear
 * with non-zero coefficients in any of the integer divisions involved.
 */
isl_bool isl_multi_pw_aff_involves_explicit_domain_dims(
	__isl_keep isl_multi_pw_aff *mpa,
	enum isl_dim_type type, unsigned pos, unsigned n)
{
	if (isl_multi_pw_aff_check_has_explicit_domain(mpa) < 0)
		return isl_bool_error;
	if (type == isl_dim_in)
		type = isl_dim_set;
	return isl_set_involves_dims(mpa->u.dom, type, pos, n);
}

/* Insert "n" dimensions of type "type" at position "pos"
 * of the explicit domain of "mpa".
 */
static __isl_give isl_multi_pw_aff *
isl_multi_pw_aff_insert_explicit_domain_dims(__isl_take isl_multi_pw_aff *mpa,
	enum isl_dim_type type, unsigned pos, unsigned n)
{
	if (isl_multi_pw_aff_check_has_explicit_domain(mpa) < 0)
		return isl_multi_pw_aff_free(mpa);
	mpa = isl_multi_pw_aff_cow(mpa);
	if (!mpa)
		return NULL;
	if (type == isl_dim_in)
		type = isl_dim_set;
	mpa->u.dom = isl_set_insert_dims(mpa->u.dom, type, pos, n);
	if (!mpa->u.dom)
		return isl_multi_pw_aff_free(mpa);
	return mpa;
}

/* Drop the "n" dimensions of type "type" starting at position "pos"
 * of the explicit domain of "mpa".
 */
static __isl_give isl_multi_pw_aff *
isl_multi_pw_aff_drop_explicit_domain_dims(__isl_take isl_multi_pw_aff *mpa,
	enum isl_dim_type type, unsigned pos, unsigned n)
{
	if (isl_multi_pw_aff_check_has_explicit_domain(mpa) < 0)
		return isl_multi_pw_aff_free(mpa);
	mpa = isl_multi_pw_aff_cow(mpa);
	if (!mpa)
		return NULL;
	if (type == isl_dim_in)
		type = isl_dim_set;
	mpa->u.dom = isl_set_drop(mpa->u.dom, type, pos, n);
	if (!mpa->u.dom)
		return isl_multi_pw_aff_free(mpa);
	return mpa;
}

/* Move the "n" dimensions of "src_type" starting at "src_pos" of
 * of the explicit domain of "mpa" to dimensions of "dst_type" at "dst_pos".
 */
static __isl_give isl_multi_pw_aff *isl_multi_pw_aff_move_explicit_domain_dims(
	__isl_take isl_multi_pw_aff *mpa,
	enum isl_dim_type dst_type, unsigned dst_pos,
	enum isl_dim_type src_type, unsigned src_pos, unsigned n)
{
	if (isl_multi_pw_aff_check_has_explicit_domain(mpa) < 0)
		return isl_multi_pw_aff_free(mpa);
	mpa = isl_multi_pw_aff_cow(mpa);
	if (!mpa)
		return NULL;
	if (dst_type == isl_dim_in)
		dst_type = isl_dim_set;
	if (src_type == isl_dim_in)
		src_type = isl_dim_set;
	mpa->u.dom = isl_set_move_dims(mpa->u.dom, dst_type, dst_pos,
				src_type, src_pos, n);
	if (!mpa->u.dom)
		return isl_multi_pw_aff_free(mpa);
	return mpa;
}
