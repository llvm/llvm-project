//===----------- EmissaryPrint.cpp - Misc Emissary API ------------ c++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Device stubs for misc emissary API
//
//===----------------------------------------------------------------------===//


#include "Allocator.h"
#include "Configuration.h"
#include "DeviceTypes.h"
#include "Shared/RPCOpcodes.h"
#include "extra_allocators.h"
#include "shared/rpc.h"

#include "Debug.h"
#include "EmissaryIds.h"

extern "C" {

__attribute__((flatten, always_inline)) void f90print_(char *s) {
  _emissary_exec(_PACK_EMIS_IDS(EMIS_ID_PRINT, _printf_idx),
		  "%s\n", s);
}
__attribute__((flatten, always_inline)) void f90printi_(char *s, int *i) {
  _emissary_exec(_PACK_EMIS_IDS(EMIS_ID_PRINT, _printf_idx),
		  "%s $d\n", s, *i);
}
__attribute__((flatten, always_inline)) void f90printl_(char *s, long *i) {
  _emissary_exec(_PACK_EMIS_IDS(EMIS_ID_PRINT, _printf_idx),
		  "%s %ld\n", s, *i);
}
__attribute__((flatten, always_inline)) void f90printf_(char *s, float *f) {
  _emissary_exec(_PACK_EMIS_IDS(EMIS_ID_PRINT, _printf_idx),
		  "%s %f\n", s, *f);
}
__attribute__((flatten, always_inline)) void f90printd_(char *s, double *d) {
  _emissary_exec(_PACK_EMIS_IDS(EMIS_ID_PRINT, _printf_idx),
		  "%s %g\n", s, *d);
}

// This definition of __ockl_devmem_request and __ockl_sanitizer_report needs to
// override the weak symbol for __ockl_devmem_request and
// __ockl_sanitizer_report in rocm device lib ockl.bc because ockl uses
// hostcall but OpenMP uses rpc.
//
__attribute__((noinline)) void
__ockl_sanitizer_report(uint64_t addr, uint64_t pc, uint64_t wgidx,
                        uint64_t wgidy, uint64_t wgidz, uint64_t wave_id,
                        uint64_t is_read, uint64_t access_size) {
  unsigned long long rc =
      _emissary_exec(_PACK_EMIS_IDS(EMIS_ID_PRINT, _ockl_asan_report_idx), addr,
                     pc, wgidx, wgidy, wgidz, wave_id, is_read, access_size);
  return;
}
#if SANITIZER_AMDGPU
__attribute__((noinline)) uint64_t __asan_malloc_impl(uint64_t bufsz,
                                                      uint64_t pc);
__attribute__((noinline)) void __asan_free_impl(uint64_t ptr, uint64_t pc);
#endif

__attribute__((flatten, always_inline)) char *global_allocate(uint32_t bufsz) {
#if SANITIZER_AMDGPU
  return (char *)__asan_malloc_impl(bufsz,
                                    (uint64_t)__builtin_return_address(0));
#else
  return (char *)malloc((uint64_t)bufsz);
#endif
}
__attribute__((flatten, always_inline)) int global_free(void *ptr) {
#if SANITIZER_AMDGPU
  __asan_free_impl((uint64_t)ptr, (uint64_t)__builtin_return_address(0));
#else
  free(ptr);
#endif
  return 0;
}

} // end extern "C"
