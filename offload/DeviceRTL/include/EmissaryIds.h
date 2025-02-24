//===- offload/DeviceRTL/include/EmissaryIds.h enum & headers ----- C++ ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines Emissary API identifiers. This header is used by both host 
// and device compilations. 
//
//===----------------------------------------------------------------------===//

#ifndef OFFLOAD_EMISSARY_IDS_H
#define OFFLOAD_EMISSARY_IDS_H
/// The sets of emissary APIs under development
typedef enum {
  EMIS_ID_INVALID,
  EMIS_ID_FORTRT,
  EMIS_ID_PRINT,
  EMIS_ID_MPI,
  EMIS_ID_HDF5,
} offload_emis_id_t;

typedef enum {
  _print_INVALID,
  _printf_idx,
  _fprintf_idx,
  _ockl_asan_report_idx,
} offload_emis_print_t;

/// The vargs function used by emissary API device stubs
unsigned long long _emissary_exec(unsigned long long, ...);

#define _PACK_EMIS_IDS(x, y)                                                   \
  ((unsigned long long)x << 32) | ((unsigned long long)y)

typedef enum {
  _FortranAio_INVALID,
  _FortranAioBeginExternalListOutput_idx,
  _FortranAioOutputAscii_idx,
  _FortranAioOutputInteger32_idx,
  _FortranAioEndIoStatement_idx,
  _FortranAioOutputInteger8_idx,
  _FortranAioOutputInteger16_idx,
  _FortranAioOutputInteger64_idx,
  _FortranAioOutputReal32_idx,
  _FortranAioOutputReal64_idx,
  _FortranAioOutputComplex32_idx,
  _FortranAioOutputComplex64_idx,
  _FortranAioOutputLogical_idx,
  _FortranAAbort_idx,
  _FortranAStopStatementText_idx,
} offload_emis_fortrt_idx;

#endif // OFFLOAD_EMISSARY_IDS_H
