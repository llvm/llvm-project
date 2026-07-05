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

/* Return the subset of "umap" where the domain and the range
 * have "mupa" values that lexicographically compare as "ORDER".
 */
__isl_give isl_union_map *FN(FN(isl_union_map_lex,ORDER),at_multi_union_pw_aff)(
	__isl_take isl_union_map *umap,
	__isl_take isl_multi_union_pw_aff *mupa)
{
	return isl_union_map_order_at_multi_union_pw_aff(umap, mupa,
				&FN(FN(isl_multi_pw_aff_lex,ORDER),map));
}
