#ifndef ISL_LOCAL_H
#define ISL_LOCAL_H

#include <isl/mat.h>
#include <isl_reordering.h>

typedef isl_mat isl_local;

isl_bool isl_local_div_is_marked_unknown(__isl_keep isl_local *local, int pos);
isl_bool isl_local_div_is_known(__isl_keep isl_local *local, int pos);
isl_bool isl_local_divs_known(__isl_keep isl_local *local);

int isl_local_cmp(__isl_keep isl_local *local1, __isl_keep isl_local *local2);

__isl_give isl_local *isl_local_reorder(__isl_take isl_local *local,
	__isl_take isl_reordering *r);

__isl_give isl_vec *isl_local_extend_point_vec(__isl_keep isl_local *local,
	__isl_take isl_vec *v);

#endif
