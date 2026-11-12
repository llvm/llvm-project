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
#else
                       "no";
#endif /* KMP_FTN_ENTRIES */
