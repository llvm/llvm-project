// clang-format off
// RUN: %libomptarget-compilexx-generic -I %S/../../../libc
// RUN: env LIBOMPTARGET_INFO=16 \
// RUN:   %libomptarget-run-generic 2>&1 

// UNSUPPORTED: nvptx64-nvidia-cuda
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// REQUIRES: gpu
// XFAIL: intelgpu

// ---------- These are the functions in our host library ---------- 
#if (!defined(__NVPTX__) && !defined(__AMDGCN__))
extern "C" int foo(int x, double y, int *iarray) { return (int)y * iarray[0]; }
extern "C" double bar(int x, double y, double *darray) { return y * darray[0]; }
#endif

// ---------- Emissary API definition for foobar_openmp library  ---------- 
//  This 4-part definition file is typically in it's own file:
//
// 1. === Includes, always include EmissaryIds.h
#if __has_include("../../../clang/lib/Headers/EmissaryIds.h")
#include "../clang/lib/Headers/EmissaryIds.h"
#else
#include "EmissaryIds.h"
#endif
#include <stdarg.h>

// 2.===  Enum with index for each function provided by Emissary API.
typedef enum {
  _RESERVE_INVALID, // recommend 0 is INVALID
  _RESERVE_foo_idx,
  _RESERVE_bar_idx,
} offload_emis_rsrv_t;

// 3. === Device Stubs for each function in the API
//        This section ONLY for device compilation
#if (defined(__NVPTX__) || defined(__AMDGCN__))
extern "C" int foo(int x, double y, int *iarray) {
   return (int)_emissary_exec(
      _PACK_EMIS_IDS(EMIS_ID_RESERVE, _RESERVE_foo_idx, 0, 0), x, y, iarray); }
extern "C" double bar(int x, double y, double *darray) {
   return (double)_emissary_exec(
      _PACK_EMIS_IDS(EMIS_ID_RESERVE, _RESERVE_bar_idx, 0, 0), x, y, darray); }

#else // end device stub definitions
     
// 4. === Define host selector function for Emissary API reserve
//       Section 4 is only compiled on host pass
#include <cstdint>
#include <shared/rpc_server.h>
#include <shared/emissary_rpc_server.h>
#define _PTR_TO_64BIT_ (unsigned long long int)
// This is the EmissaryReserve selector function. It is called when the emissary
// runtime sees EMIS_ID_RESERVE as the API identifier. It is dispatched through
// the runtime registry (see the self-registration constructor below)
// This function invokes host function based on function index (emisfnid).
extern "C" EmissaryReturn_t EmissaryReserve(char *data, emisArgBuf_t *ab,
                                            emis_argptr_t *a[]) {
  switch (ab->emisfnid) {

  case _RESERVE_foo_idx: {
    return (EmissaryReturn_t) foo (
        (int)(_PTR_TO_64BIT_ a[0]),
        (double)(_PTR_TO_64BIT_ a[1]),
        (int *)(_PTR_TO_64BIT_ a[2]));
  }

  case _RESERVE_bar_idx: {
    return (EmissaryReturn_t) bar (
        (int)(_PTR_TO_64BIT_ a[0]),
        (double)(_PTR_TO_64BIT_ a[1]),
        (double *)(_PTR_TO_64BIT_ a[2]));
  }

  } // end switch statement
  return (EmissaryReturn_t)0;
} // end EmissaryReserve function selector

// Self-register the host selector fn at load time so the RPC server
// dispatches EMIS_ID_RESERVE through the runtime registry
extern "C" __attribute__((constructor)) void
emissary_reserve_self_register(void) {
  EmissaryRegister(EMIS_ID_RESERVE, &EmissaryReserve);
}

#undef _PTR_TO_64BIT_
#endif 
//== End section 4 and end of Emissary API definition for foobar_openmp library

// ---------- Demo app using foobar_openmp lib on host AND device  ---------- 
#define VSIZE 10
#include <stdio.h>
int main(int argc, char *argv[]) {
  double yfoo = 2.0;
  int iarray[2] = {4,42};
  int foo_rc = foo(-1, yfoo, iarray);

  double ybar = 3.0;
  double darray[2] = {4.0 , 42.0};;
  double bar_rc = bar(-2, ybar, darray);
  printf("MAIN foo_rc:%d  bar_rc:%f\n",foo_rc, bar_rc);
  foo_rc = 1; bar_rc=1;

  printf("PREREGION foo_rc:%d  bar_rc:%f yfoo:%f \n",foo_rc, bar_rc, yfoo);
#pragma omp target teams distribute parallel for map(to:yfoo,ybar) map(from: foo_rc,bar_rc) is_device_ptr(iarray, darray)
  for (int i = 0; i < VSIZE; i++) {
    foo_rc = foo(i, yfoo, iarray);
    bar_rc = bar(i, ybar, darray);
  }
  printf("POSTREGION foo_rc:%d  bar_rc:%f yfoo:%f \n",foo_rc, bar_rc, yfoo);
  int rc = 0;
  if (foo_rc != 8 )
    rc = 1;
  if (bar_rc != 12.0 )
    rc = 2;
  return rc;
}
