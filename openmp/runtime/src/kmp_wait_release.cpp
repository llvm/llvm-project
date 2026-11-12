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

void __kmp_wait_64(kmp_info_t *this_thr, kmp_flag_64<> *flag,
                   int final_spin USE_ITT_BUILD_ARG(void *itt_sync_obj)) {
  if (final_spin)
    __kmp_wait_template<kmp_flag_64<>, TRUE>(
        this_thr, flag USE_ITT_BUILD_ARG(itt_sync_obj));
  else
    __kmp_wait_template<kmp_flag_64<>, FALSE>(
        this_thr, flag USE_ITT_BUILD_ARG(itt_sync_obj));
}

void __kmp_release_64(kmp_flag_64<> *flag) { __kmp_release_template(flag); }

#if KMP_HAVE_MWAIT || KMP_HAVE_UMWAIT
template <bool C, bool S>
void __kmp_mwait_32(int th_gtid, kmp_flag_32<C, S> *flag) {
  __kmp_mwait_template(th_gtid, flag);
}
template <bool C, bool S>
void __kmp_mwait_64(int th_gtid, kmp_flag_64<C, S> *flag) {
  __kmp_mwait_template(th_gtid, flag);
}
template <bool C, bool S>
void __kmp_atomic_mwait_64(int th_gtid, kmp_atomic_flag_64<C, S> *flag) {
  __kmp_mwait_template(th_gtid, flag);
}
void __kmp_mwait_oncore(int th_gtid, kmp_flag_oncore *flag) {
  __kmp_mwait_template(th_gtid, flag);
}

template void __kmp_mwait_32<false, false>(int, kmp_flag_32<false, false> *);
template void __kmp_mwait_64<false, true>(int, kmp_flag_64<false, true> *);
template void __kmp_mwait_64<true, false>(int, kmp_flag_64<true, false> *);
template void
__kmp_atomic_mwait_64<false, true>(int, kmp_atomic_flag_64<false, true> *);
template void
__kmp_atomic_mwait_64<true, false>(int, kmp_atomic_flag_64<true, false> *);
#endif
