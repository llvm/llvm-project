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

typedef enum {
  _MPI_INVALID,
  _MPI_Send_idx,
  _MPI_Recv_idx,
} offload_emis_mpi_t;

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

//  mpi.h (needed for MPI types) will not compile while building DeviceRTL,
//  So emissary stubs for MPI functions can NOT be in libomptarget.bc.
//  These are skipped whild building DeviceRTL because compilation of DeviceRTL
//  does not have include mpi.h. The user will build these stubs on their
//  device pass when they include EmissaryIds.h.

#if defined(__NVPTX__) || defined(__AMDGCN__)
#if defined(__has_include)
#if __has_include("mpi.h")
#include "mpi.h"
extern "C" int MPI_Send(const void *buf, int count, MPI_Datatype datatype,
                        int dest, int tag, MPI_Comm comm) {
  return (int)_emissary_exec(_PACK_EMIS_IDS(EMIS_ID_MPI, _MPI_Send_idx), buf,
                             count, datatype, dest, tag, comm);
}
extern "C" int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source,
                        int tag, MPI_Comm comm, MPI_Status *st) {
  return (int)_emissary_exec(_PACK_EMIS_IDS(EMIS_ID_MPI, _MPI_Recv_idx), buf,
                             count, datatype, source, tag, comm, st);
}
#endif
#endif
#endif

#endif // OFFLOAD_EMISSARY_IDS_H
