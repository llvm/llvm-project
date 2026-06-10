//===- llvm/CodeGen/TargetBitwiseImmRewrite.h ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the TargetBitwiseImmRewritePass class.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_TARGETBITWISEIMMREWRITE_H
#define LLVM_CODEGEN_TARGETBITWISEIMMREWRITE_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class TargetMachine;

class TargetBitwiseImmRewritePass
    : public PassInfoMixin<TargetBitwiseImmRewritePass> {
  const TargetMachine *TM;

public:
  explicit TargetBitwiseImmRewritePass(const TargetMachine &TM) : TM(&TM) {}
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);
};

} // end namespace llvm

#endif // LLVM_CODEGEN_TARGETBITWISEIMMREWRITE_H
