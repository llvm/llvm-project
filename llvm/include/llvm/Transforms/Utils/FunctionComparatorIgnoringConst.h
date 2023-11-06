//===- FunctionComparatorIgnoringConst.h - Function Comparator --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the FunctionComparatorIgnoringConst class which is used by
// the MergeFuncIgnoringConst pass for comparing functions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_FUNCTIONCOMPARATORIGNORINGCONST_H
#define LLVM_TRANSFORMS_UTILS_FUNCTIONCOMPARATORIGNORINGCONST_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/ValueMap.h"
#include "llvm/Support/AtomicOrdering.h"
#include "llvm/Support/Casting.h"
#include "llvm/Transforms/Utils/FunctionComparator.h"
#include <set>

namespace llvm {

/// FunctionComparatorIgnoringConst - Compares two functions to determine
/// whether or not they match when certain constants are ignored.
class FunctionComparatorIgnoringConst : public FunctionComparator {
public:
  FunctionComparatorIgnoringConst(const Function *F1, const Function *F2,
                                  GlobalNumberState *GN)
      : FunctionComparator(F1, F2, GN) {}

  int cmpOperandsIgnoringConsts(const Instruction *L, const Instruction *R,
                                unsigned opIdx);

  int cmpBasicBlocksIgnoringConsts(
      const BasicBlock *BBL, const BasicBlock *BBR,
      const std::set<std::pair<int, int>> *InstOpndIndex = nullptr);

  int compareIgnoringConsts(
      const std::set<std::pair<int, int>> *InstOpndIndex = nullptr);

  int compareConstants(const Constant *L, const Constant *R) const {
    return cmpConstants(L, R);
  }

private:
  /// Scratch index for instruction in order during cmpOperandsIgnoringConsts.
  int Index = 0;
};

} // end namespace llvm
#endif // LLVM_TRANSFORMS_UTILS_FUNCTIONCOMPARATORIGNORINGCONST_H
