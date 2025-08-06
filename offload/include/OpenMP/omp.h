//===-- OpenMP/omp.h - Copies of OpenMP user facing types and APIs - C++ -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This copies some OpenMP user facing types and APIs for easy reach within the
// implementation.
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_OPENMP_OMP_H
#define OMPTARGET_OPENMP_OMP_H

#include <cstdint>

#if defined(_WIN32)
#define __KAI_KMPC_CONVENTION __cdecl
#ifndef __KMP_IMP
#define __KMP_IMP __declspec(dllimport)
#endif
#else
#define __KAI_KMPC_CONVENTION
#ifndef __KMP_IMP
#define __KMP_IMP
#endif
#endif

extern "C" {

/// Type declarations
///{

typedef void *omp_depend_t;

///}

/// API declarations
///{

int omp_get_default_device(void) __attribute__((weak));

///}

/// InteropAPI
///
///{

/// TODO: Include the `omp.h` of the current build
/* OpenMP 5.1 interop */
typedef intptr_t omp_intptr_t;

/* 0..omp_get_num_interop_properties()-1 are reserved for implementation-defined
 * properties */
typedef enum omp_interop_property {
  omp_ipr_fr_id = -1,
  omp_ipr_fr_name = -2,
  omp_ipr_vendor = -3,
  omp_ipr_vendor_name = -4,
  omp_ipr_device_num = -5,
  omp_ipr_platform = -6,
  omp_ipr_device = -7,
  omp_ipr_device_context = -8,
  omp_ipr_targetsync = -9,
  omp_ipr_first = -9
} omp_interop_property_t;

#define omp_interop_none 0

typedef enum omp_interop_rc {
  omp_irc_no_value = 1,
  omp_irc_success = 0,
  omp_irc_empty = -1,
  omp_irc_out_of_range = -2,
  omp_irc_type_int = -3,
  omp_irc_type_ptr = -4,
  omp_irc_type_str = -5,
  omp_irc_other = -6
} omp_interop_rc_t;

/* Foreign runtime values from OpenMP Additional Definitions document v2.1 */
typedef enum tgt_foreign_runtime_id_t {
  tgt_fr_none = 0,
  tgt_fr_cuda = 1,
  tgt_fr_cuda_driver = 2,
  tgt_fr_opencl = 3,
  tgt_fr_sycl = 4,
  tgt_fr_hip = 5,
  tgt_fr_level_zero = 6,
  tgt_fr_hsa = 7,
  tgt_fr_last = 8
} tgt_foreign_runtime_id_t;

typedef void *omp_interop_t;

/*!
 * The `omp_get_num_interop_properties` routine retrieves the number of
 * implementation-defined properties available for an `omp_interop_t` object.
 */
int __KAI_KMPC_CONVENTION omp_get_num_interop_properties(const omp_interop_t);
/*!
 * The `omp_get_interop_int` routine retrieves an integer property from an
 * `omp_interop_t` object.
 */
omp_intptr_t __KAI_KMPC_CONVENTION
omp_get_interop_int(const omp_interop_t, omp_interop_property_t, int *);
/*!
 * The `omp_get_interop_ptr` routine retrieves a pointer property from an
 * `omp_interop_t` object.
 */
void *__KAI_KMPC_CONVENTION omp_get_interop_ptr(const omp_interop_t,
                                                omp_interop_property_t, int *);
/*!
 * The `omp_get_interop_str` routine retrieves a string property from an
 * `omp_interop_t` object.
 */
const char *__KAI_KMPC_CONVENTION
omp_get_interop_str(const omp_interop_t, omp_interop_property_t, int *);
/*!
 * The `omp_get_interop_name` routine retrieves a property name from an
 * `omp_interop_t` object.
 */
const char *__KAI_KMPC_CONVENTION omp_get_interop_name(const omp_interop_t,
                                                       omp_interop_property_t);
/*!
 * The `omp_get_interop_type_desc` routine retrieves a description of the type
 * of a property associated with an `omp_interop_t` object.
 */
const char *__KAI_KMPC_CONVENTION
omp_get_interop_type_desc(const omp_interop_t, omp_interop_property_t);
/*!
 * The `omp_get_interop_rc_desc` routine retrieves a description of the return
 * code associated with an `omp_interop_t` object.
 */
extern const char *__KAI_KMPC_CONVENTION
omp_get_interop_rc_desc(const omp_interop_t, omp_interop_rc_t);

/* Vendor defined values from OpenMP Additional Definitions document v2.1*/
typedef enum omp_vendor_id {
  omp_vendor_unknown = 0,
  omp_vendor_amd = 1,
  omp_vendor_arm = 2,
  omp_vendor_bsc = 3,
  omp_vendor_fujitsu = 4,
  omp_vendor_gnu = 5,
  omp_vendor_hpe = 6,
  omp_vendor_ibm = 7,
  omp_vendor_intel = 8,
  omp_vendor_llvm = 9,
  omp_vendor_nec = 10,
  omp_vendor_nvidia = 11,
  omp_vendor_ti = 12,
  omp_vendor_last = 13
} omp_vendor_id_t;

///} InteropAPI

} // extern "C"

#endif // OMPTARGET_OPENMP_OMP_H
