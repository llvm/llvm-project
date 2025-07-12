//===- IsolatePath.cpp - Code to isolate paths with UB ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass isolates code paths with undefined behavior from paths without
// undefined behavior, and then add a trap instruction on that path. This
// prevents code generation where, after the UB instruction's eliminated, the
// code can wander off the end of a function.
//
// For example, a nullptr dereference:
//
//   foo:
//     %phi.val = phi ptr [ %arrayidx.i, %pred1 ], [ null, %pred2 ]
//     %load.val = load i32, ptr %phi.val, align 4
//
// is converted into:
//
//   foo.ub.path:
//     %load.val.ub = load volatile i32, ptr null, align 4
//     tail call void @llvm.trap()
//     unreachable
//
// Note: we allow the NULL dereference to actually occur so that code that
// wishes to catch the signal can do so.
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

/// This pass performs 'path isolation', which looks for undefined behavior and
/// isolates the path from non-undefined behavior code and converts the UB into
/// a trap call.
class IsolatePathPass : public PassInfoMixin<IsolatePathPass> {
  SmallPtrSet<BasicBlock *, 4> SplitUBBlocks;

  bool ProcessPointerUndefinedBehavior(BasicBlock *BB, DomTreeUpdater *DTU);

public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_ISOLATEPATH_H
