//===-- ForceFunctionAttrs.h - Force function attrs for debugging ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// Super simple passes to force specific function attrs from the commandline
/// into the IR for debugging purposes.
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_FORCEFUNCTIONATTRS_H
#define LLVM_TRANSFORMS_IPO_FORCEFUNCTIONATTRS_H

#include "llvm/IR/PassManager.h"

namespace llvm {
class Module;
class Pass;

/// Pass which forces specific function attributes into the IR, primarily as
/// a debugging tool.
struct ForceFunctionAttrsPass : PassInfoMixin<ForceFunctionAttrsPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};

/// Create a legacy pass manager instance of a pass to force function attrs.
Pass *createForceFunctionAttrsLegacyPass();

}

#endif // LLVM_TRANSFORMS_IPO_FORCEFUNCTIONATTRS_H
