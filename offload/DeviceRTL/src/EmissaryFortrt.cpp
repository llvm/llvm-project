//===- EmissaryFortrt.cpp - Fortran Runtime emissary API ----- ---- c++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Device stubs for Fortran Runtime emissary API
//
//===----------------------------------------------------------------------===//

#include "DeviceTypes.h"
#include "EmissaryIds.h"
#include "Shared/RPCOpcodes.h"
#include "shared/rpc.h"

unsigned long long _emissary_exec(unsigned long long, ...);

extern "C" {

// The clang compiler will generate calls to this only when a string length is
// not a compile time constant.
uint32_t __strlen_max(char *instr, uint32_t maxstrlen) {
  for (uint32_t i = 0; i < maxstrlen; i++)
    if (instr[i] == (char)0)
      return (uint32_t)(i + 1);
  return maxstrlen;
}

uint32_t omp_get_thread_num();
uint32_t omp_get_num_threads();
uint32_t omp_get_team_num();
uint32_t omp_get_num_teams();

#define _EXTRA_ARGS                                                            \
  omp_get_thread_num(), omp_get_num_threads(), omp_get_team_num(),             \
      omp_get_num_teams()

void *_FortranAioBeginExternalListOutput(uint32_t a1, const char *a2,
                                         uint32_t a3) {
  void *cookie = (void *)_emissary_exec(
      _PACK_EMIS_IDS(EMIS_ID_FORTRT, _FortranAioBeginExternalListOutput_idx),
      _EXTRA_ARGS, a1, a2, a3);
  return cookie;
}
bool _FortranAioOutputAscii(void *a1, char *a2, uint64_t a3) {
  // insert null terminating char so  _emissary_exec can correctly
  // calculate runtime str length
  a2[a3 - 1] = (char)0;
  return (bool)_emissary_exec(
      _PACK_EMIS_IDS(EMIS_ID_FORTRT, _FortranAioOutputAscii_idx), _EXTRA_ARGS,
      a1, a2, a3);
}
bool _FortranAioOutputInteger32(void *a1, uint32_t a2) {
  return (bool)_emissary_exec(
      _PACK_EMIS_IDS(EMIS_ID_FORTRT, _FortranAioOutputInteger32_idx),
      _EXTRA_ARGS, a1, a2);
}
uint32_t _FortranAioEndIoStatement(void *a1) {
  return (uint32_t)_emissary_exec(
      _PACK_EMIS_IDS(EMIS_ID_FORTRT, _FortranAioEndIoStatement_idx),
      _EXTRA_ARGS, a1);
}
bool _FortranAioOutputInteger8(void *cookie, int8_t n) {
  return (bool)_emissary_exec(
      _PACK_EMIS_IDS(EMIS_ID_FORTRT, _FortranAioOutputInteger8_idx),
      _EXTRA_ARGS, cookie, n);
}
bool _FortranAioOutputInteger16(void *cookie, int16_t n) {
  return (bool)_emissary_exec(
      _PACK_EMIS_IDS(EMIS_ID_FORTRT, _FortranAioOutputInteger16_idx),
      _EXTRA_ARGS, cookie, n);
}
bool _FortranAioOutputInteger64(void *cookie, int64_t n) {
  return (bool)_emissary_exec(
      _PACK_EMIS_IDS(EMIS_ID_FORTRT, _FortranAioOutputInteger64_idx),
      _EXTRA_ARGS, cookie, n);
}
bool _FortranAioOutputReal32(void *cookie, float x) {
  return (bool)_emissary_exec(
      _PACK_EMIS_IDS(EMIS_ID_FORTRT, _FortranAioOutputReal32_idx), _EXTRA_ARGS,
      cookie, x);
}
bool _FortranAioOutputReal64(void *cookie, double x) {
  return (bool)_emissary_exec(
      _PACK_EMIS_IDS(EMIS_ID_FORTRT, _FortranAioOutputReal64_idx), _EXTRA_ARGS,
      cookie, x);
}
bool _FortranAioOutputComplex32(void *cookie, float re, float im) {
  return (bool)_emissary_exec(
      _PACK_EMIS_IDS(EMIS_ID_FORTRT, _FortranAioOutputComplex32_idx),
      _EXTRA_ARGS, cookie, re, im);
}
bool _FortranAioOutputComplex64(void *cookie, double re, double im) {
  return (bool)_emissary_exec(
      _PACK_EMIS_IDS(EMIS_ID_FORTRT, _FortranAioOutputComplex64_idx),
      _EXTRA_ARGS, cookie, re, im);
}
bool _FortranAioOutputLogical(void *cookie, bool barg) {
  return (bool)_emissary_exec(
      _PACK_EMIS_IDS(EMIS_ID_FORTRT, _FortranAioOutputLogical_idx), _EXTRA_ARGS,
      cookie, barg);
}
void _FortranAAbort() {
  _emissary_exec(_PACK_EMIS_IDS(EMIS_ID_FORTRT, _FortranAAbort_idx));
  // When  host service _FortranAAbort finishes, we must die from the device.
  __builtin_trap();
}
void _FortranAStopStatementText(char *errmsg, int64_t a1, bool a2, bool a3) {
  // TODO: must use string length from a1 arg
  errmsg[a1 - 1] = (char)0;
  _emissary_exec(_PACK_EMIS_IDS(EMIS_ID_FORTRT, _FortranAStopStatementText_idx),
                 errmsg, a1, a2, a3);
  __builtin_trap();
}

} // end extern "C"
#undef _EXTRA_ARGS
