//===- FakePrivatize.h - Test-only alloca privatization
//--------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_FAKEPRIVATIZE_H
#define LLVM_TRANSFORMS_UTILS_FAKEPRIVATIZE_H

#include "llvm/IR/PassManager.h"

namespace llvm {

/// Replace the generated test's scalar alloca with an iteration-specific slot.
class FakePrivatizePass : public RequiredPassInfoMixin<FakePrivatizePass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_FAKEPRIVATIZE_H
