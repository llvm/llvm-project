/*
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, K.U.Leuven, Departement
 * Computerwetenschappen, Celestijnenlaan 200A, B-3001 Leuven, Belgium
 */

#ifndef ISL_SCAN_H
#define ISL_SCAN_H

#include <isl/set.h>
#include <isl/vec.h>

struct isl_scan_callback {
	isl_stat (*add)(struct isl_scan_callback *cb,
		__isl_take isl_vec *sample);
};

isl_stat isl_basic_set_scan(__isl_take isl_basic_set *bset,
	struct isl_scan_callback *callback);
isl_stat isl_set_scan(__isl_take isl_set *set,
	struct isl_scan_callback *callback);

#endif
