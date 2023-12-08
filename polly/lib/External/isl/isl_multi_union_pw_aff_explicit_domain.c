/*
 * Copyright 2017      Sven Verdoolaege
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege.
 */

/* Initialize the explicit domain of "mupa".
 *
 * The explicit domain is initialized to a universe parameter set.
 * It may later be specialized with constraints on the parameter or
 * specific domain instances.
 */
static __isl_give isl_multi_union_pw_aff *
isl_multi_union_pw_aff_init_explicit_domain(
	__isl_take isl_multi_union_pw_aff *mupa)
{
	isl_space *space;

	if (isl_multi_union_pw_aff_check_has_explicit_domain(mupa) < 0)
		return isl_multi_union_pw_aff_free(mupa);
	space = isl_space_params(isl_multi_union_pw_aff_get_space(mupa));
	mupa->u.dom = isl_union_set_from_set(isl_set_universe(space));
	if (!mupa->u.dom)
		return isl_multi_union_pw_aff_free(mupa);
	return mupa;
}

/* Drop the "n" dimensions of type "type" starting at position "pos"
 * of the explicit domain of "mupa".
 */
static __isl_give isl_multi_union_pw_aff *
isl_multi_union_pw_aff_drop_explicit_domain_dims(
	__isl_take isl_multi_union_pw_aff *mupa,
	enum isl_dim_type type, unsigned pos, unsigned n)
{
	if (isl_multi_union_pw_aff_check_has_explicit_domain(mupa) < 0)
		return isl_multi_union_pw_aff_free(mupa);
	if (type != isl_dim_param)
		isl_die(isl_multi_union_pw_aff_get_ctx(mupa), isl_error_invalid,
			"can only drop parameters",
			return isl_multi_union_pw_aff_free(mupa));
	mupa = isl_multi_union_pw_aff_cow(mupa);
	if (!mupa)
		return NULL;
	mupa->u.dom = isl_union_set_project_out(mupa->u.dom, type, pos, n);
	if (!mupa->u.dom)
		return isl_multi_union_pw_aff_free(mupa);
	return mupa;
}
