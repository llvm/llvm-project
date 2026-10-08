//===- RTTI.cpp -----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the implementation of APIs in the orc-rt/support/RTTI.h and
// orc-rt-c/support/RTTI.h headers.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/support/RTTI.h"

namespace orc_rt {

char RTTIRoot::ThisLibraryID = 0;
void RTTIRoot::anchor() noexcept {}

// --- C API Implementation ---

extern "C" {

const char *orc_rt_RTTIRoot_getTypeName(orc_rt_RTTIRootRef Obj) noexcept {
  return unwrap(Obj)->dynamicRTTIName();
}

} // extern "C"

} // namespace orc_rt
