//===---- offload/plugins-nextgen/common/src/EmissaryFortrt.cpp  ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Host support for Fortran runtime Emissary API 
//
//===----------------------------------------------------------------------===//
#include "PluginInterface.h"
#include "RPC.h"
#include "Shared/Debug.h"
#include "Shared/RPCOpcodes.h"
#include "shared/rpc.h"
#include "shared/rpc_opcodes.h"
#include <assert.h>
#include <cstring>
#include <ctype.h>
#include <list>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tuple>
#include <vector>

#include "../../../DeviceRTL/include/EmissaryIds.h"
#include "Emissary.h"

static service_rc fortran_rt_service(uint32_t DeviceRuntime_idx, char *buf,
                                     size_t bufsz, uint64_t *return_value);

extern "C" emis_return_t _emissary_execute_fortrt(uint32_t emis_func_id,
                                                  void *data, uint32_t sz) {
  uint64_t return_value;
  service_rc rc =
      fortran_rt_service(emis_func_id, (char *)data, (size_t)sz, &return_value);
  return (emis_return_t)return_value;
}

// Make the vargs function call to the function pointer fnptr
// by casting fnptr to vfnptr. Return uint32_t
template <typename T, typename FT>
static uint32_t _s_call_fnptr(uint32_t NumArgs, void *fnptr,
                              uint64_t *a[MAXVARGS], T *rv) {
  FT *vfnptr = (FT *)fnptr;

  switch (NumArgs) {
  case 1:
    *rv = vfnptr(fnptr, a[0]);
    break;
  case 2:
    *rv = vfnptr(fnptr, a[0], a[1]);
    break;
  case 3:
    *rv = vfnptr(fnptr, a[0], a[1], a[2]);
    break;
  case 4:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3]);
    break;
  case 5:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4]);
    break;
  case 6:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5]);
    break;
  case 7:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
    break;
  case 8:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
    break;
  case 9:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
    break;
  case 10:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9]);
    break;
  case 11:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10]);
    break;
  case 12:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11]);
    break;
  case 13:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12]);
    break;
  case 14:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13]);
    break;
  case 15:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14]);
    break;
  case 16:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15]);
    break;
  case 17:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16]);
    break;
  case 18:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17]);
    break;
  case 19:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18]);
    break;
  case 20:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19]);
    break;
  case 21:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20]);
    break;
  case 22:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21]);
    break;
  case 23:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22]);
    break;
  case 24:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23]);
    break;
  case 25:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24]);
    break;
  case 26:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25]);
    break;
  case 27:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26]);
    break;
  case 28:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26],
                 a[27]);
    break;
  case 29:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26],
                 a[27], a[28]);
    break;
  case 30:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26],
                 a[27], a[28], a[29]);
    break;
  case 31:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26],
                 a[27], a[28], a[29], a[30]);
    break;
  case 32:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26],
                 a[27], a[28], a[29], a[30], a[31]);
    break;
  default:
    return _RC_EXCEED_MAXVARGS_ERROR;
  }
  return _RC_SUCCESS;
}

// Headers for Host Fortran Runtime API as built in flang/runtime
extern "C" {
void *_FortranAioBeginExternalListOutput(uint32_t a1, const char *a2,
                                         uint32_t a3);
bool _FortranAioOutputAscii(void *a1, char *a2, uint64_t a3);
bool _FortranAioOutputInteger32(void *a1, uint32_t a2);
uint32_t _FortranAioEndIoStatement(void *a1);
bool _FortranAioOutputInteger8(void *cookie, int8_t n);
bool _FortranAioOutputInteger16(void *cookie, int16_t n);
bool _FortranAioOutputInteger64(void *cookie, int64_t n);
bool _FortranAioOutputReal32(void *cookie, float x);
bool _FortranAioOutputReal64(void *cookie, double x);
bool _FortranAioOutputComplex32(void *cookie, float re, float im);
bool _FortranAioOutputComplex64(void *cookie, double re, double im);
bool _FortranAioOutputLogical(void *cookie, bool truth);
void _FortranAAbort();
void _FortranAStopStatementText(char *errmsg, int64_t a1, bool a2, bool a3);

//  Save the cookie because deferred functions have execution reordered.
static void *_list_started_cookie = nullptr;
extern void *V_FortranAioBeginExternalListOutput(void *fnptr, ...) {
  va_list args;
  va_start(args, fnptr);
  int32_t v0 = va_arg(args, int32_t);
  const char *v1 = va_arg(args, const char *);
  int32_t v2 = va_arg(args, int32_t);
  va_end(args);
  void *cookie = _FortranAioBeginExternalListOutput(v0, v1, v2);
  _list_started_cookie = cookie;
  return cookie;
}
extern bool V_FortranAioOutputAscii(void *fnptr, ...) {
  va_list args;
  va_start(args, fnptr);
  void *v0 = va_arg(args, void *);
  char *v1 = va_arg(args, char *);
  uint64_t v2 = va_arg(args, uint64_t);
  va_end(args);
  v0 = _list_started_cookie;
  return _FortranAioOutputAscii(v0, v1, v2);
}
extern bool V_FortranAioOutputInteger32(void *fnptr, ...) {
  va_list args;
  va_start(args, fnptr);
  void *v0 = va_arg(args, void *);
  uint32_t v1 = va_arg(args, uint32_t);
  va_end(args);
  v0 = _list_started_cookie;
  return _FortranAioOutputInteger32(v0, v1);
}
extern uint32_t V_FortranAioEndIoStatement(void *fnptr, ...) {
  va_list args;
  va_start(args, fnptr);
  void *v0 = va_arg(args, void *);
  va_end(args);
  v0 = _list_started_cookie;
  uint32_t rv = _FortranAioEndIoStatement(v0);
  return rv;
}
extern bool V_FortranAioOutputInteger8(void *fnptr, ...) {
  va_list args;
  va_start(args, fnptr);
  void *v0 = va_arg(args, void *);
  uint32_t v1 = va_arg(args, uint32_t);
  va_end(args);
  v0 = _list_started_cookie;
  return _FortranAioOutputInteger8(v0, v1);
}
extern bool V_FortranAioOutputInteger16(void *fnptr, ...) {
  va_list args;
  va_start(args, fnptr);
  void *v0 = va_arg(args, void *);
  uint32_t v1 = va_arg(args, uint32_t);
  va_end(args);
  v0 = _list_started_cookie;
  return _FortranAioOutputInteger16(v0, v1);
}
extern bool V_FortranAioOutputInteger64(void *fnptr, ...) {
  va_list args;
  va_start(args, fnptr);
  void *v0 = va_arg(args, void *);
  uint32_t v1 = va_arg(args, uint32_t);
  va_end(args);
  v0 = _list_started_cookie;
  return _FortranAioOutputInteger64(v0, v1);
}
extern bool V_FortranAioOutputReal32(void *fnptr, ...) {
  va_list args;
  va_start(args, fnptr);
  void *v0 = va_arg(args, void *);
  uint64_t v1 = va_arg(args, uint64_t);
  va_end(args);
  v0 = _list_started_cookie;
  double dv;
  memcpy(&dv, &v1, 8);
  return _FortranAioOutputReal32(v0, (float)dv);
}
extern bool V_FortranAioOutputReal64(void *fnptr, ...) {
  va_list args;
  va_start(args, fnptr);
  void *cookie = va_arg(args, void *);
  uint64_t v1 = va_arg(args, uint64_t);
  va_end(args);
  cookie = _list_started_cookie;
  double dv;
  memcpy(&dv, &v1, 8);
  return _FortranAioOutputReal64(cookie, dv);
}
extern bool V_FortranAioOutputComplex32(void *fnptr, ...) {
  va_list args;
  va_start(args, fnptr);
  void *v0 = va_arg(args, void *);
  uint64_t v1 = va_arg(args, uint64_t);
  uint64_t v2 = va_arg(args, uint64_t);
  va_end(args);
  v0 = _list_started_cookie;
  double dv1, dv2;
  memcpy(&dv1, &v1, 8);
  memcpy(&dv2, &v2, 8);
  return _FortranAioOutputComplex32(v0, (float)dv1, (float)dv2);
}
extern bool V_FortranAioOutputComplex64(void *fnptr, ...) {
  va_list args;
  va_start(args, fnptr);
  void *v0 = va_arg(args, void *);
  uint64_t v1 = va_arg(args, uint64_t);
  uint64_t v2 = va_arg(args, uint64_t);
  va_end(args);
  v0 = _list_started_cookie;
  double dv1, dv2;
  memcpy(&dv1, &v1, 8);
  memcpy(&dv2, &v2, 8);
  return _FortranAioOutputComplex64(v0, dv1, dv2);
}
extern bool V_FortranAioOutputLogical(void *fnptr, ...) {
  va_list args;
  va_start(args, fnptr);
  void *v0 = va_arg(args, void *);
  uint32_t v1 = va_arg(args, uint32_t);
  va_end(args);
  v0 = _list_started_cookie;
  return _FortranAioOutputLogical(v0, v1);
}
extern void V_FortranAAbort(void *fnptr, ...) {
  va_list args;
  va_start(args, fnptr);
  va_end(args);
  _FortranAAbort();
  // Now return to device to run abort from stub
}
extern void V_FortranAStopStatementText(void *fnptr, ...) {
  va_list args;
  va_start(args, fnptr);
  char *errmsg = va_arg(args, char *);
  int64_t a1 = va_arg(args, int64_t);
  uint32_t a2 = va_arg(args, uint32_t);
  uint32_t a3 = va_arg(args, uint32_t);
  va_end(args);
  bool b2 = (bool)a2;
  bool b3 = (bool)a3;
  _FortranAStopStatementText(errmsg, a1, b2, b3);
}
} // end extern "C"

// Static vars used to defer functions to reorder execution by thread and team.
static uint32_t _deferred_fn_count = 0;
static uint32_t _deferred_begin_statements = 0;
static uint32_t _deferred_end_statements = 0;
static uint64_t _max_num_threads = 0;
static uint64_t _max_num_teams = 0;

// structure for deferred functions
typedef struct {
  uint32_t NumArgs;    // The number of args in arg_array
  void *fnptr;         // The function pointer for this index
  uint64_t fn_idx;     // The function index, good for debug
  uint32_t dfnid;      // The dvoideferred function id, in order received
  uint64_t *arg_array; // ptr to malloced arg_array
  char *c_ptr;         // ptr to null terminated char string
  uint64_t thread_num;
  uint64_t num_threads;
  uint64_t team_num;
  uint64_t num_teams;
  uint64_t *return_value; // pointer to where return value is copied
} deferred_entry_t;

static std::vector<deferred_entry_t *> *_deferred_fns_ptr;
// static std::list<deferred_entry_t *> _deferred_fns;

static service_rc fortran_rt_service(uint32_t DeviceRuntime_idx, char *buf,
                                     size_t bufsz, uint64_t *return_value) {
  if (bufsz == 0)
    return _RC_SUCCESS;

  // Get 6 values needed to unpack the buffer
  int *datalen = (int *)buf;
  int NumArgs = *((int *)(buf + sizeof(int)));
  size_t data_not_used =
      (size_t)(*datalen) - ((size_t)(2 + NumArgs) * sizeof(int));
  char *keyptr = buf + (2 * sizeof(int));
  char *dataptr = keyptr + (NumArgs * sizeof(int));
  char *strptr = buf + (size_t)*datalen;

  // skip the function pointer arg including any align buffer
  if (((size_t)dataptr) % (size_t)8) {
    dataptr += 4;
    data_not_used -= 4;
  }
  void *fnptr = getfnptr(dataptr);
  NumArgs--;
  keyptr += 4;
  dataptr += 8;
  data_not_used -= 4;

  if (NumArgs <= 0)
    return _RC_ERROR_INVALID_REQUEST;

  uint64_t *a[MAXVARGS];
  if (_build_vargs_array(NumArgs, keyptr, dataptr, strptr, &data_not_used, a) !=
      _RC_SUCCESS)
    return _RC_ERROR_INVALID_REQUEST;

  // std::list<deferred_entry_t *> _deferred_fns;
  if (!_deferred_fns_ptr)
    _deferred_fns_ptr = new std::vector<deferred_entry_t *>;

  char *c_ptr = nullptr;
  bool defer_for_reorder = true;
  bool run_deferred_functions = false;
  switch (DeviceRuntime_idx) {
  case _FortranAioBeginExternalListOutput_idx: {
    _deferred_begin_statements++;
    fnptr = (void *)V_FortranAioBeginExternalListOutput;
    size_t slen = std::strlen((char *)a[5]) + 1;
    c_ptr = (char *)aligned_alloc(sizeof(uint64_t *), slen);
    if (!c_ptr)
      fprintf(stderr, "MALLOC FAILED for c_ptr size:%ld \n", slen);
    std::strncpy(c_ptr, (char *)a[5], slen - 1);
    c_ptr[slen - 1] = (char)0;
    a[5] = (uint64_t *)c_ptr;
    break;
  }
  case _FortranAioOutputAscii_idx: {
    fnptr = (void *)V_FortranAioOutputAscii;

    size_t slen = (size_t)a[6] + 1;
    c_ptr = (char *)aligned_alloc(sizeof(uint64_t *), slen);
    if (!c_ptr)
      fprintf(stderr, "MALLOC FAILED for c_ptr size:%ld \n", slen);
    std::strncpy(c_ptr, (char *)a[5], slen - 1);
    c_ptr[slen - 1] = (char)0;
    a[5] = (uint64_t *)c_ptr;

    break;
  }
  case _FortranAioOutputInteger32_idx: {
    fnptr = (void *)V_FortranAioOutputInteger32;
    break;
  }
  case _FortranAioEndIoStatement_idx: {
    _deferred_end_statements++;
    fnptr = (void *)V_FortranAioEndIoStatement;
    // We cannot use last tread and team number to trigger running deferred
    // functions because its warp could finish early (out of order). So, if
    // this is the last FortranAioEndIoStatement by count of begin statements,
    // then run the deferred functions ordered by team and thread number.
    if (_deferred_end_statements == _deferred_begin_statements)
      run_deferred_functions = true;
    break;
  }
  case _FortranAioOutputInteger8_idx: {
    fnptr = (void *)V_FortranAioOutputInteger8;
    break;
  }
  case _FortranAioOutputInteger16_idx: {
    fnptr = (void *)V_FortranAioOutputInteger16;
    break;
  }
  case _FortranAioOutputInteger64_idx: {
    fnptr = (void *)V_FortranAioOutputInteger64;
    break;
  }
  case _FortranAioOutputReal32_idx: {
    fnptr = (void *)V_FortranAioOutputReal32;
    break;
  }
  case _FortranAioOutputReal64_idx: {
    fnptr = (void *)V_FortranAioOutputReal64;
    break;
  }
  case _FortranAioOutputComplex32_idx: {
    fnptr = (void *)V_FortranAioOutputComplex32;
    break;
  }
  case _FortranAioOutputComplex64_idx: {
    fnptr = (void *)V_FortranAioOutputComplex64;
    break;
  }
  case _FortranAioOutputLogical_idx: {
    fnptr = (void *)V_FortranAioOutputLogical;
    break;
  }
  case _FortranAAbort_idx: {
    defer_for_reorder = false;
    fnptr = (void *)V_FortranAAbort;
    break;
  }
  case _FortranAStopStatementText_idx: {
    defer_for_reorder = false;
    fnptr = (void *)V_FortranAStopStatementText;
    break;
  }
  case _FortranAio_INVALID:
  default: {
    defer_for_reorder = false;
    break;
  }
  } // end of switch

  if (defer_for_reorder) {
    _deferred_fn_count++;
    deferred_entry_t *q = new deferred_entry_t;

    q->dfnid = _deferred_fn_count - 1;
    q->thread_num = (uint64_t)a[0];
    q->num_threads = (uint64_t)a[1];
    _max_num_threads =
        (q->num_threads > _max_num_threads) ? q->num_threads : _max_num_threads;
    q->team_num = (uint64_t)a[2];
    q->num_teams = (uint64_t)a[3];
    _max_num_teams =
        (q->num_teams > _max_num_teams) ? q->num_teams : _max_num_teams;
    q->NumArgs = NumArgs - 4;
    q->fnptr = fnptr;
    q->fn_idx = DeviceRuntime_idx;
    uint64_t *arg_array = (uint64_t *)aligned_alloc(
        sizeof(uint64_t), (NumArgs - 4) * sizeof(uint64_t));
    if (!arg_array)
      fprintf(stderr, " MALLOC FAILED for arg_array size:%ld \n",
              sizeof(uint64_t) * (NumArgs - 4));
    for (int32_t i = 0; i < NumArgs - 4; i++) {
      uint64_t val = (uint64_t)a[i + 4];
      arg_array[i] = val;
    }
    q->arg_array = arg_array;
    q->return_value = (uint64_t *)return_value;
    q->c_ptr = c_ptr;
    _deferred_fns_ptr->push_back(q);
  } else {
    // non deferred functions get a return_value
    if (_s_call_fnptr<uint64_t, emis_uint64_t>(NumArgs - 4, fnptr, &a[4],
                                               return_value) != _RC_SUCCESS)
      return _RC_ERROR_INVALID_REQUEST;
  }

  if (run_deferred_functions) {
    // This specific team and thread ordering does not reflect the
    // actual non-deterministic ordering.
    for (uint32_t team_num = 0; team_num < _max_num_teams; team_num++) {
      for (uint32_t thread_num = 0; thread_num < _max_num_threads;
           thread_num++) {
        for (auto q : *_deferred_fns_ptr) {
          if ((thread_num == q->thread_num) && (team_num == q->team_num)) {
            for (uint32_t i = 0; i < q->NumArgs; i++)
              a[i] = (uint64_t *)q->arg_array[i];
            uint32_t rc = _s_call_fnptr<uint64_t, emis_uint64_t>(
                q->NumArgs, q->fnptr, a, q->return_value);
            if (rc != _RC_SUCCESS) {
              fprintf(stderr, "    BAD RETURN FROM _call_fnptr %d\n", rc);
              return _RC_ERROR_INVALID_REQUEST;
            }
          }
          // Only the return value for the last end statement is returned.
          return_value = q->return_value;
        }
      }
    }

    //  Reset static deferred function counters and free memory
    for (auto q : *_deferred_fns_ptr) {
      if (q->c_ptr)
        free(q->c_ptr);
      free(q->arg_array);
      delete q;
    }
    _deferred_fns_ptr->clear();
    _deferred_fn_count = 0;
    _deferred_begin_statements = 0;
    _deferred_end_statements = 0;
    _max_num_threads = 0;
    _max_num_teams = 0;
    delete _deferred_fns_ptr;
  } // end run_deferred_functions

  return _RC_SUCCESS;
} // end fortran_rt_service
