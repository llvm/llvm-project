/*
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, K.U.Leuven, Departement
 * Computerwetenschappen, Celestijnenlaan 200A, B-3001 Leuven, Belgium
 */

#ifndef ISL_ILP_H
#define ISL_ILP_H

#include <isl/aff_type.h>
#include <isl/set_type.h>
#include <isl/union_set_type.h>
#include <isl/val_type.h>
#include <isl/vec.h>

#if defined(__cplusplus)
extern "C" {
#endif

__isl_give isl_val *isl_basic_set_max_val(__isl_keep isl_basic_set *bset,
	__isl_keep isl_aff *obj);
__isl_export
__isl_give isl_val *isl_set_min_val(__isl_keep isl_set *set,
	__isl_keep isl_aff *obj);
__isl_export
__isl_give isl_val *isl_set_max_val(__isl_keep isl_set *set,
	__isl_keep isl_aff *obj);
__isl_give isl_multi_val *isl_union_set_min_multi_union_pw_aff(
	__isl_keep isl_union_set *uset, __isl_keep isl_multi_union_pw_aff *obj);

__isl_export
__isl_give isl_val *isl_pw_aff_min_val(__isl_take isl_pw_aff *pa);
__isl_export
__isl_give isl_val *isl_pw_aff_max_val(__isl_take isl_pw_aff *pa);
__isl_export
__isl_give isl_multi_val *isl_pw_multi_aff_min_multi_val(
	__isl_take isl_pw_multi_aff *pma);
__isl_export
__isl_give isl_multi_val *isl_pw_multi_aff_max_multi_val(
	__isl_take isl_pw_multi_aff *pma);
__isl_export
__isl_give isl_multi_val *isl_multi_pw_aff_min_multi_val(
	__isl_take isl_multi_pw_aff *mpa);
__isl_export
__isl_give isl_multi_val *isl_multi_pw_aff_max_multi_val(
	__isl_take isl_multi_pw_aff *mpa);

__isl_give isl_val *isl_union_pw_aff_min_val(__isl_take isl_union_pw_aff *upa);
__isl_give isl_val *isl_union_pw_aff_max_val(__isl_take isl_union_pw_aff *upa);

__isl_give isl_multi_val *isl_multi_union_pw_aff_min_multi_val(
	__isl_take isl_multi_union_pw_aff *mupa);
__isl_give isl_multi_val *isl_multi_union_pw_aff_max_multi_val(
	__isl_take isl_multi_union_pw_aff *mupa);

__isl_export
__isl_give isl_val *isl_basic_set_dim_max_val(__isl_take isl_basic_set *bset,
	int pos);
__isl_export
__isl_give isl_val *isl_set_dim_min_val(__isl_take isl_set *set, int pos);
__isl_export
__isl_give isl_val *isl_set_dim_max_val(__isl_take isl_set *set, int pos);

#if defined(__cplusplus)
}
#endif

#endif
