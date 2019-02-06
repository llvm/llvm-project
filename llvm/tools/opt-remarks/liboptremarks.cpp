//===-liboptremarks.cpp - LLVM Opt-Remarks Shared Library -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provide a library to work with optimization remarks.
//
//===----------------------------------------------------------------------===//

#include "llvm-c/OptRemarks.h"

extern uint32_t LLVMOptRemarkVersion(void) {
  return OPT_REMARKS_API_VERSION;
}
