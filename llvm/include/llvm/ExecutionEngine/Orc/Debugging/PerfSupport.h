//===-- PerfSupport.h - Utils for enabling debugger support --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for enabling perf support.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_PERFSUPPORT_H
#define LLVM_EXECUTIONENGINE_ORC_PERFSUPPORT_H

#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace orc {

class LLJIT;

LLVM_ABI Error enablePerfSupport(LLJIT &J);

} // namespace orc
} // namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_PERFSUPPORT_H
