//===---- Rfuns.cpp - OpenMP reduction functions ---- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains simple reduction functions used as function pointers to
// reduction helper functions for cross team reductions defined in Xteamr.cpp
//
//===----------------------------------------------------------------------===//

#pragma omp declare target

#define _RF_ATTR extern "C" __attribute__((flatten, always_inline)) void
#define _RF_LDS volatile __attribute__((address_space(3)))

_RF_ATTR __kmpc_rfun_sum_d(double *val, double otherval) { *val += otherval; }
_RF_ATTR __kmpc_rfun_sum_lds_d(_RF_LDS double *val, _RF_LDS double *otherval) {
  *val += *otherval;
}
_RF_ATTR __kmpc_rfun_sum_f(float *val, float otherval) { *val += otherval; }
_RF_ATTR __kmpc_rfun_sum_lds_f(_RF_LDS float *val, _RF_LDS float *otherval) {
  *val += *otherval;
}
_RF_ATTR __kmpc_rfun_sum_cd(double _Complex *val, double _Complex otherval) { *val += otherval; }
_RF_ATTR __kmpc_rfun_sum_lds_cd(_RF_LDS double _Complex *val, _RF_LDS double _Complex*otherval) {
  *val += *otherval;
}
_RF_ATTR __kmpc_rfun_sum_cf(float _Complex *val, float _Complex otherval) { *val += otherval; }
_RF_ATTR __kmpc_rfun_sum_lds_cf(_RF_LDS float _Complex *val, _RF_LDS float _Complex *otherval) {
  *val += *otherval;
}
_RF_ATTR __kmpc_rfun_sum_i(int *val, int otherval) { *val += otherval; }
_RF_ATTR __kmpc_rfun_sum_lds_i(_RF_LDS int *val, _RF_LDS int *otherval) {
  *val += *otherval;
}
_RF_ATTR __kmpc_rfun_sum_ui(unsigned int *val, unsigned int otherval) {
  *val += otherval;
}
_RF_ATTR __kmpc_rfun_sum_lds_ui(_RF_LDS unsigned int *val,
                                _RF_LDS unsigned int *otherval) {
  *val += *otherval;
}
_RF_ATTR __kmpc_rfun_sum_l(long *val, long otherval) { *val += otherval; }
_RF_ATTR __kmpc_rfun_sum_lds_l(_RF_LDS long *val, _RF_LDS long *otherval) {
  *val += *otherval;
}
_RF_ATTR __kmpc_rfun_sum_ul(unsigned long *val, unsigned long otherval) {
  *val += otherval;
}
_RF_ATTR __kmpc_rfun_sum_lds_ul(_RF_LDS unsigned long *val,
                                _RF_LDS unsigned long *otherval) {
  *val += *otherval;
}

_RF_ATTR __kmpc_rfun_min_d(double *val, double otherval) {
  *val = (otherval < *val) ? otherval : *val;
}
_RF_ATTR __kmpc_rfun_min_lds_d(_RF_LDS double *val, _RF_LDS double *otherval) {
  *val = (*otherval < *val) ? *otherval : *val;
}
_RF_ATTR __kmpc_rfun_min_f(float *val, float otherval) {
  *val = (otherval < *val) ? otherval : *val;
}
_RF_ATTR __kmpc_rfun_min_lds_f(_RF_LDS float *val, _RF_LDS float *otherval) {
  *val = (*otherval < *val) ? *otherval : *val;
}
_RF_ATTR __kmpc_rfun_min_i(int *val, int otherval) {
  *val = (otherval < *val) ? otherval : *val;
}
_RF_ATTR __kmpc_rfun_min_lds_i(_RF_LDS int *val, _RF_LDS int *otherval) {
  *val = (*otherval < *val) ? *otherval : *val;
}
_RF_ATTR __kmpc_rfun_min_ui(unsigned int *val, unsigned int otherval) {
  *val = (otherval < *val) ? otherval : *val;
}
_RF_ATTR __kmpc_rfun_min_lds_ui(_RF_LDS unsigned int *val,
                                _RF_LDS unsigned int *otherval) {
  *val = (*otherval < *val) ? *otherval : *val;
}
_RF_ATTR __kmpc_rfun_min_l(long *val, long otherval) {
  *val = (otherval < *val) ? otherval : *val;
}
_RF_ATTR __kmpc_rfun_min_lds_l(_RF_LDS long *val, _RF_LDS long *otherval) {
  *val = (*otherval < *val) ? *otherval : *val;
}
_RF_ATTR __kmpc_rfun_min_ul(unsigned long *val, unsigned long otherval) {
  *val = (otherval < *val) ? otherval : *val;
}
_RF_ATTR __kmpc_rfun_min_lds_ul(_RF_LDS unsigned long *val,
                                _RF_LDS unsigned long *otherval) {
  *val = (*otherval < *val) ? *otherval : *val;
}

_RF_ATTR __kmpc_rfun_max_d(double *val, double otherval) {
  *val = (otherval > *val) ? otherval : *val;
}
_RF_ATTR __kmpc_rfun_max_lds_d(_RF_LDS double *val, _RF_LDS double *otherval) {
  *val = (*otherval > *val) ? *otherval : *val;
}
_RF_ATTR __kmpc_rfun_max_f(float *val, float otherval) {
  *val = (otherval > *val) ? otherval : *val;
}
_RF_ATTR __kmpc_rfun_max_lds_f(_RF_LDS float *val, _RF_LDS float *otherval) {
  *val = (*otherval > *val) ? *otherval : *val;
}
_RF_ATTR __kmpc_rfun_max_i(int *val, int otherval) {
  *val = (otherval > *val) ? otherval : *val;
}
_RF_ATTR __kmpc_rfun_max_lds_i(_RF_LDS int *val, _RF_LDS int *otherval) {
  *val = (*otherval > *val) ? *otherval : *val;
}
_RF_ATTR __kmpc_rfun_max_ui(unsigned int *val, unsigned int otherval) {
  *val = (otherval > *val) ? otherval : *val;
}
_RF_ATTR __kmpc_rfun_max_lds_ui(_RF_LDS unsigned int *val,
                                _RF_LDS unsigned int *otherval) {
  *val = (*otherval > *val) ? *otherval : *val;
}
_RF_ATTR __kmpc_rfun_max_l(long *val, long otherval) {
  *val = (otherval > *val) ? otherval : *val;
}
_RF_ATTR __kmpc_rfun_max_lds_l(_RF_LDS long *val, _RF_LDS long *otherval) {
  *val = (*otherval > *val) ? *otherval : *val;
}
_RF_ATTR __kmpc_rfun_max_ul(unsigned long *val, unsigned long otherval) {
  *val = (otherval > *val) ? otherval : *val;
}
_RF_ATTR __kmpc_rfun_max_lds_ul(_RF_LDS unsigned long *val,
                                _RF_LDS unsigned long *otherval) {
  *val = (*otherval > *val) ? *otherval : *val;
}

#undef _RF_ATTR
#undef _RF_LDS

#pragma omp end declare target
