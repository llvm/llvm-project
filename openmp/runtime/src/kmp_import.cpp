/*
 * kmp_import.cpp
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/* Object generated from this source file is linked to Windows* OS DLL import
   library (libompmd.lib) only! It is not a part of regular static or dynamic
   OpenMP RTL. Any code that just needs to go in the libompmd.lib (but not in
   libompmt.lib and libompmd.dll) should be placed in this file. */

#ifdef __cplusplus
extern "C" {
#endif

/*These symbols are required for mutual exclusion with Microsoft OpenMP RTL
  (and compatibility with MS Compiler). */

int _You_must_link_with_exactly_one_OpenMP_library = 1;
int _You_must_link_with_Intel_OpenMP_library = 1;
int _You_must_link_with_Microsoft_OpenMP_library = 1;

#ifdef __cplusplus
}
#endif

// end of file //
