//===- RegAllocCommon.h - Utilities shared between allocators ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
