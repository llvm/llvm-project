//===- IsolatePath.h - Path isolation for undefined behavior ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the interface for the IsolatePath pass.
//
// The pass identifies undefined behavior (UB) that is reachable via a PHI node
// that can select a null pointer. It then refactors the control-flow graph to
// isolate the UB-triggering path from the safe paths.
//
// Once isolated, the UB path is terminated, either with an 'unreachable'
// instruction or, optionally, with a 'trap' followed by 'unreachable'. This
// prevents the optimizer from making unsafe assumptions based on the presence
// of UB, which could otherwise lead to miscompilations.
//
// For example, a null pointer dereference is transformed from:
//
//   bb:
//     %phi = phi ptr [ %valid_ptr, %pred1 ], [ null, %pred2 ]
//     %val = load i32, ptr %phi
//
// To:
//
//   bb:
//     %phi = phi ptr [ %valid_ptr, %pred1 ]
//     %val = load i32, ptr %phi
//     ...
//
//   bb.ub.path:
//     %phi.ub = phi ptr [ null, %pred2 ]
//     unreachable
//
// Or to this with the optional trap-unreachable flag:
//
//   bb.ub.path:
//     %phi.ub = phi ptr [ null, %pred2 ]
//     %val.ub = load volatile i32, ptr %phi.ub ; Optional trap
//     call void @llvm.trap()
//     unreachable
//
// This ensures that the presence of the null path does not interfere with
// valid code paths.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_ISOLATEPATH_H
#define LLVM_TRANSFORMS_SCALAR_ISOLATEPATH_H

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class BasicBlock;
class DomTreeUpdater;
class Function;
class LoopInfo;

/// A pass that isolates paths with undefined behavior and converts the UB into
/// a trap or unreachable instruction.
class IsolatePathPass : public PassInfoMixin<IsolatePathPass> {
  SmallPtrSet<BasicBlock *, 4> SplitUBBlocks;

  bool ProcessPointerUndefinedBehavior(BasicBlock *BB, DomTreeUpdater *DTU,
                                       LoopInfo *LI);

public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_ISOLATEPATH_H
