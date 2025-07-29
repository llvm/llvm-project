//===- EmitChangedFuncDebugInfo.h - Emit Additional Debug Info -*- C++ --*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Emit debug info for changed or new funcs.
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_EMITCHANGEDFUNCDEBUGINFO_H
#define LLVM_TRANSFORMS_UTILS_EMITCHANGEDFUNCDEBUGINFO_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class Module;

// Pass that emits late dwarf.
class EmitChangedFuncDebugInfoPass
    : public PassInfoMixin<EmitChangedFuncDebugInfoPass> {
public:
  EmitChangedFuncDebugInfoPass() = default;

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_EMITCHANGEDFUNCDEBUGINFO_H
