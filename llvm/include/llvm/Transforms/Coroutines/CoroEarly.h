//===---- CoroEarly.h - Lower early coroutine intrinsics --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file
// This file provides the interface to the early coroutine intrinsic lowering
// pass. This pass lowers coroutine intrinsics that hide the details of the
// exact calling convention for coroutine resume and destroy functions and
// details of the structure of the coroutine frame.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_COROUTINES_COROEARLY_H
#define LLVM_TRANSFORMS_COROUTINES_COROEARLY_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class Function;

struct CoroEarlyPass : PassInfoMixin<CoroEarlyPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};
} // end namespace llvm

#endif // LLVM_TRANSFORMS_COROUTINES_COROEARLY_H
