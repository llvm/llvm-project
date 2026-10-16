//===- Debug.h --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBACCTARGET_DEBUG_H_
#define _LIBACCTARGET_DEBUG_H_

#include "Shared/Debug.h"

namespace llvm::acc::target::debug {

// Debug types to use in libacctarget
constexpr const char *ADT_Init = "ACCInit";
// The trace of the runtime API routines (acc_*).
constexpr const char *ADT_APITrace = "ACCAPITrace";

} // namespace llvm::acc::target::debug

/// Report that something is not yet implemented and exit.
#define ACC_NYI() REPORT_FATAL() << "not yet implemented: "

#endif // _LIBACCTARGET_DEBUG_H_
