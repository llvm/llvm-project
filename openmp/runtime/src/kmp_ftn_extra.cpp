/*
 * kmp_ftn_extra.cpp -- Fortran 'extra' linkage support for OpenMP.
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
#define KMP_FTN_ENTRIES KMP_FTN_PLAIN
#elif KMP_OS_UNIX
#define KMP_FTN_ENTRIES KMP_FTN_APPEND
#endif

// Note: This string is not printed when KMP_VERSION=1.
char const __kmp_version_ftnextra[] =
    KMP_VERSION_PREFIX "Fortran \"extra\" OMP support: "
#ifdef KMP_FTN_ENTRIES
                       "yes";
#define FTN_STDCALL /* nothing to do */
#include "kmp_ftn_os.h"
#include "kmp_ftn_entry.h"

#if KMP_FTN_ENTRIES == KMP_FTN_PLAIN
#define FTN_KMP_GET_UID_FROM_DEVICE __kmp_get_uid_from_device
#define FTN_KMP_GET_DEVICE_FROM_UID __kmp_get_device_from_uid
#endif
#if KMP_FTN_ENTRIES == KMP_FTN_APPEND
#define FTN_KMP_GET_UID_FROM_DEVICE __kmp_get_uid_from_device_
#define FTN_KMP_GET_DEVICE_FROM_UID __kmp_get_device_from_uid_
#endif
#if KMP_FTN_ENTRIES == KMP_FTN_UPPER
#define FTN_KMP_GET_UID_FROM_DEVICE __KMP_GET_UID_FROM_DEVICE
#define FTN_KMP_GET_DEVICE_FROM_UID __KMP_GET_DEVICE_FROM_UID
#endif
#if KMP_FTN_ENTRIES == KMP_FTN_UAPPEND
#define FTN_KMP_GET_UID_FROM_DEVICE __KMP_GET_UID_FROM_DEVICE_
#define FTN_KMP_GET_DEVICE_FROM_UID __KMP_GET_DEVICE_FROM_UID_
#endif

extern "C" {
const char *FTN_STDCALL KMP_EXPAND_NAME(FTN_KMP_GET_UID_FROM_DEVICE)(
    int device_num) KMP_WEAK_ATTRIBUTE_EXTERNAL;
const char *FTN_STDCALL
KMP_EXPAND_NAME(FTN_KMP_GET_UID_FROM_DEVICE)(int device_num) {
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
int FTN_STDCALL KMP_EXPAND_NAME(FTN_KMP_GET_DEVICE_FROM_UID)(
    const char *device_uid) KMP_WEAK_ATTRIBUTE_EXTERNAL;
int FTN_STDCALL
KMP_EXPAND_NAME(FTN_KMP_GET_DEVICE_FROM_UID)(const char *device_uid) {
#if KMP_OS_DARWIN || KMP_OS_WASI || defined(KMP_STUB)
  return omp_invalid_device;
#else
  int (*fptr)(const char *);
  if ((*(void **)(&fptr) = KMP_DLSYM_NEXT("omp_get_device_from_uid")))
    return (*fptr)(device_uid);
  return KMP_EXPAND_NAME(FTN_GET_INITIAL_DEVICE)();
#endif
}

KMP_VERSION_SYMBOL(FTN_KMP_GET_UID_FROM_DEVICE, 60, "OMP_6.0");
KMP_VERSION_SYMBOL(FTN_KMP_GET_DEVICE_FROM_UID, 60, "OMP_6.0");
} // extern "C"
#else
                       "no";
#endif /* KMP_FTN_ENTRIES */
