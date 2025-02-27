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

// Headers for Host Fortran Runtime API as built in llvm/flang/runtime
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
  emis_return_t return_value;
} deferred_entry_t;

static std::vector<deferred_entry_t *> *_deferred_fns_ptr;
// static std::list<deferred_entry_t *> _deferred_fns;
//

extern "C" emis_return_t EmissaryFortrt(char *data, emisArgBuf_t *ab) {
  emis_return_t return_value = (emis_return_t)0;

  if (ab->DataLen == 0)
    return _RC_SUCCESS;

  void *fnptr;
  if (ab->NumArgs <= 0)
    return _RC_ERROR_INVALID_REQUEST;

  uint64_t *a[MAXVARGS];
  if (EmissaryBuildVargs(ab->NumArgs, ab->keyptr, ab->argptr, ab->strptr,
                         &ab->data_not_used, a) != _RC_SUCCESS)
    return _RC_ERROR_INVALID_REQUEST;

  // std::list<deferred_entry_t *> _deferred_fns;
  if (!_deferred_fns_ptr)
    _deferred_fns_ptr = new std::vector<deferred_entry_t *>;

  char *c_ptr = nullptr;
  bool defer_for_reorder = true;
  bool run_deferred_functions = false;
  switch (ab->emisfnid) {
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
    q->NumArgs = ab->NumArgs - 4;
    q->fnptr = fnptr;
    q->fn_idx = ab->emisfnid;
    uint64_t *arg_array = (uint64_t *)aligned_alloc(
        sizeof(uint64_t), (ab->NumArgs - 4) * sizeof(uint64_t));
    if (!arg_array)
      fprintf(stderr, " MALLOC FAILED for arg_array size:%ld \n",
              sizeof(uint64_t) * (ab->NumArgs - 4));
    for (uint32_t i = 0; i < ab->NumArgs - 4; i++) {
      uint64_t val = (uint64_t)a[i + 4];
      arg_array[i] = val;
    }
    q->arg_array = arg_array;
    q->return_value = (emis_return_t)0;
    q->c_ptr = c_ptr;
    _deferred_fns_ptr->push_back(q);
  } else {
    // execute a non deferred function
    return_value = EmissaryCallFnptr<emis_return_t, emisfn_t>(ab->NumArgs - 4,
                                                              fnptr, &a[4]);
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
            q->return_value = EmissaryCallFnptr<emis_return_t, emisfn_t>(
                q->NumArgs, q->fnptr, a);
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

  return return_value;
} // end EmissaryFortrt
