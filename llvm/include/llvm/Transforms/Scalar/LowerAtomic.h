//===- LowerAtomic.cpp - Lower atomic intrinsics ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
// This pass lowers atomic intrinsics to non-atomic form for use in a known
// non-preemptible environment.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_LOWERATOMIC_H
#define LLVM_TRANSFORMS_SCALAR_LOWERATOMIC_H

#include "llvm/IR/PassManager.h"

namespace llvm {

/// A pass that lowers atomic intrinsic into non-atomic intrinsics.
class LowerAtomicPass : public PassInfoMixin<LowerAtomicPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &);
  static bool isRequired() { return true; }
};

class AtomicRMWInst;
/// Convert the given RMWI into primitive load and stores,
/// assuming that doing so is legal. Return true if the lowering
/// succeeds.
bool lowerAtomicRMWInst(AtomicRMWInst *RMWI);
}

#endif // LLVM_TRANSFORMS_SCALAR_LOWERATOMIC_H
