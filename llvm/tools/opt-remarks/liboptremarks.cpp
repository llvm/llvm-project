//===-liboptremarks.cpp - LLVM Opt-Remarks Shared Library -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
