//===- LibcallLoweringInfo.cpp - Interface for runtime libcalls -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/LibcallLoweringInfo.h"

using namespace llvm;

LibcallLoweringInfo::LibcallLoweringInfo(
    const RTLIB::RuntimeLibcallsInfo &RTLCI)
    : RTLCI(RTLCI) {
  // TODO: This should be generated with lowering predicates, and assert the
  // call is available.
  for (RTLIB::LibcallImpl Impl : RTLIB::libcall_impls()) {
    if (RTLCI.isAvailable(Impl)) {
      RTLIB::Libcall LC = RTLIB::RuntimeLibcallsInfo::getLibcallFromImpl(Impl);
      // FIXME: Hack, assume the first available libcall wins.
      if (LibcallImpls[LC] == RTLIB::Unsupported)
        LibcallImpls[LC] = Impl;
    }
  }
}
