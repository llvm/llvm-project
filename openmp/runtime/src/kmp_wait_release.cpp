/*
 * kmp_wait_release.cpp -- Wait/Release implementation
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kmp_wait_release.h"

void __kmp_wait_64(kmp_info_t *this_thr, kmp_flag_64 *flag,
                   int final_spin USE_ITT_BUILD_ARG(void *itt_sync_obj)) {
  if (final_spin)
    __kmp_wait_template<kmp_flag_64, TRUE>(
        this_thr, flag USE_ITT_BUILD_ARG(itt_sync_obj));
  else
    __kmp_wait_template<kmp_flag_64, FALSE>(
        this_thr, flag USE_ITT_BUILD_ARG(itt_sync_obj));
}

void __kmp_release_64(kmp_flag_64 *flag) { __kmp_release_template(flag); }
