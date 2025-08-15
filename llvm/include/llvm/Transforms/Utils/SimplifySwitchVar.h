//===-- SimplifySwitchVar.h - Simplify Switch Variables ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file defines a pass for switch variable simplification.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SIMPLIFYSWITCHVAR_H
#define LLVM_TRANSFORMS_SIMPLIFYSWITCHVAR_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class SimplifySwitchVarPass : public PassInfoMixin<SimplifySwitchVarPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_SIMPLIFYSWITCHVAR_H
