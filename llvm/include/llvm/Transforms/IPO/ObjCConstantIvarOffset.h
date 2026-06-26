//===- ObjCConstantIvarOffset.h - Constify ObjC ivar offsets ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// This pass promotes OBJC_IVAR_$_* offset globals to constants when the full
/// class hierarchy is visible, pre-sliding offsets to match the runtime
/// moveIvars().
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_OBJCCONSTANTIVAROFFSET_H
#define LLVM_TRANSFORMS_IPO_OBJCCONSTANTIVAROFFSET_H

#include "llvm/IR/PassManager.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class Module;
class ModuleSummaryIndex;

struct ObjCConstantIvarOffsetPass
    : OptionalPassInfoMixin<ObjCConstantIvarOffsetPass> {
  // Null for Full LTO
  const ModuleSummaryIndex *ImportSummary;

  ObjCConstantIvarOffsetPass(const ModuleSummaryIndex *IS)
      : ImportSummary(IS) {}
  LLVM_ABI PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_IPO_OBJCCONSTANTIVAROFFSET_H
