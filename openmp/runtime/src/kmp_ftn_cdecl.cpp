/*
 * kmp_ftn_cdecl.cpp -- Fortran __cdecl linkage support for OpenMP.
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kmp.h"
#include "kmp_affinity.h"

#if KMP_OS_WINDOWS
#if defined KMP_WIN_CDECL || !KMP_DYNAMIC_LIB
#define KMP_FTN_ENTRIES KMP_FTN_UPPER
#endif
#elif KMP_OS_UNIX
#define KMP_FTN_ENTRIES KMP_FTN_PLAIN
#endif

// Note: This string is not printed when KMP_VERSION=1.
char const __kmp_version_ftncdecl[] =
    KMP_VERSION_PREFIX "Fortran __cdecl OMP support: "
#ifdef KMP_FTN_ENTRIES
                       "yes";
#define FTN_STDCALL /* no stdcall */
#include "kmp_ftn_os.h"
#include "kmp_ftn_entry.h"

// FIXME: this is a hack to get the UID functions working for C.
// It will be moved and also made available for Fortran in a follow-up patch.
extern "C" {
const char *FTN_STDCALL omp_get_uid_from_device(int device_num)
    KMP_WEAK_ATTRIBUTE_EXTERNAL;
const char *FTN_STDCALL omp_get_uid_from_device(int device_num) {
#if KMP_OS_DARWIN || KMP_OS_WASI || defined(KMP_STUB)
  return nullptr;
#else
  const char *(*fptr)(int);
  if ((*(void **)(&fptr) = KMP_DLSYM_NEXT("omp_get_uid_from_device")))
    return (*fptr)(device_num);
  // Returns the same string as used by libomptarget
  return "HOST";
#endif
}
int FTN_STDCALL omp_get_device_from_uid(const char *device_uid)
    KMP_WEAK_ATTRIBUTE_EXTERNAL;
int FTN_STDCALL omp_get_device_from_uid(const char *device_uid) {
#if KMP_OS_DARWIN || KMP_OS_WASI || defined(KMP_STUB)
  return omp_invalid_device;
#else
  int (*fptr)(const char *);
  if ((*(void **)(&fptr) = KMP_DLSYM_NEXT("omp_get_device_from_uid")))
    return (*fptr)(device_uid);
  return KMP_EXPAND_NAME(FTN_GET_INITIAL_DEVICE)();
#endif
}
}
#else
                       "no";
#endif /* KMP_FTN_ENTRIES */
