//===- RegAllocCommon.h - Utilities shared between allocators ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_REGALLOCCOMMON_H
#define LLVM_CODEGEN_REGALLOCCOMMON_H

#include <functional>

namespace llvm {

class TargetRegisterClass;
class TargetRegisterInfo;

typedef std::function<bool(const TargetRegisterInfo &TRI,
                           const TargetRegisterClass &RC)> RegClassFilterFunc;

/// Default register class filter function for register allocation. All virtual
/// registers should be allocated.
static inline bool allocateAllRegClasses(const TargetRegisterInfo &,
                                         const TargetRegisterClass &) {
  return true;
}

}

#endif // LLVM_CODEGEN_REGALLOCCOMMON_H
