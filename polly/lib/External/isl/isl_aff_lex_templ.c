/*
 * Copyright 2014      INRIA Rocquencourt
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Inria Paris - Rocquencourt, Domaine de Voluceau - Rocquencourt,
 * B.P. 105 - 78153 Le Chesnay, France
 */

#define xFN(TYPE,NAME) TYPE ## _ ## NAME
#define FN(TYPE,NAME) xFN(TYPE,NAME)

/* Return a map containing pairs of elements in the domains of "mpa1" and "mpa2"
 * where the function value of "mpa1" lexicographically compares as "ORDER"
 * to that of "mpa2".  "space" is the space of the result.
 * The parameters of "mpa1" and "mpa2" are assumed to have been aligned.
 *
 * "mpa1" is in the given lexicographic order compared to "mpa2"
 * if, for some i, the i-th element of "mpa1" is in that order compared to
 * the i-th element of "mpa2" while all previous elements are
 * pairwise equal, where the order needs to be strict (not-equal)
 * if i corresponds to anything but the last element.
 * The strict version of "ORDER" is defined by "STRICT_ORDER",
 * which is the same if "ORDER" itself is strict.
 */
static __isl_give isl_map *FN(FN(isl_multi_pw_aff_lex,ORDER),map_on_space)(
	__isl_keep isl_multi_pw_aff *mpa1, __isl_keep isl_multi_pw_aff *mpa2,
	__isl_take isl_space *space)
{
	return isl_multi_pw_aff_lex_map_on_space(mpa1, mpa2,
					&FN(FN(isl_pw_aff,STRICT_ORDER),map),
					&FN(FN(isl_pw_aff,ORDER),map), space);
}

/* Return a map containing pairs of elements in the domains of "mpa1" and "mpa2"
 * where the function value of "mpa1" lexicographically compares as "ORDER"
 * to that of "mpa2".
 */
__isl_give isl_map *FN(FN(isl_multi_pw_aff_lex,ORDER),map)(
	__isl_take isl_multi_pw_aff *mpa1, __isl_take isl_multi_pw_aff *mpa2)
{
	return isl_multi_pw_aff_order_map(mpa1, mpa2,
			&FN(FN(isl_multi_pw_aff_lex,ORDER),map_on_space));
}

/* Return the subset of "map" where the domain and the range
 * have "mpa" values that lexicographically compare as "ORDER".
 */
__isl_give isl_map *FN(FN(isl_map_lex,ORDER),at_multi_pw_aff)(
	__isl_take isl_map *map, __isl_take isl_multi_pw_aff *mpa)
{
	return isl_map_order_at_multi_pw_aff(map, mpa,
				&FN(FN(isl_multi_pw_aff_lex,ORDER),map));
}
