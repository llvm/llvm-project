//===- ScalableToFixedVectors.h - Convert scalable to fixed vectors -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass converts IR operations on scalable vector types to fixed-length
// vectors when the effective length is known and is less than the minimum
// possible scaled vector length. For a scalable vector type with
// element count VF (known min elements), if minvscale * VF > VL, the we can
// convert to a fixed length vector of length VL.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_SCALABLETOFIXEDVECTORS_H
#define LLVM_TRANSFORMS_SCALAR_SCALABLETOFIXEDVECTORS_H

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>

namespace llvm {

class ScalableToFixedVectorsPass
    : public PassInfoMixin<ScalableToFixedVectorsPass> {
public:
  explicit ScalableToFixedVectorsPass() {}

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);

private:
  void reset();
  unsigned getMinimumVLOfInst(const Instruction *I) const;
  void convertToFixed(IRBuilder<> &Builder, Instruction *I, unsigned VL);
  bool isSupported(const Value *V) const;
  void transfer(Instruction *I);
  bool isSeedCandidate(const Instruction *I) const;
  bool allVectorOperandsConverted(Instruction *I) const;

  // Minimum VScale for the current function
  unsigned MinVScale;
  static constexpr unsigned MaxVL = std::numeric_limits<unsigned>::max();

  /// For a given instruction, records what elements of it are demanded by
  /// downstream users.
  MapVector<Instruction *, unsigned> DemandedVLs;
  /// Maps the original instructions to their fixed vector type version
  MapVector<Instruction *, Value *> ScaledToFixed;
  SetVector<Instruction *> Worklist;

  /// \returns all operands that are of scalable vector type
  auto vector_operands(Instruction *I) const {
    return make_filter_range(I->operands(), [this](Value *V) {
      return V->getType()->isScalableTy();
    });
  }

  /// \returns all users of the given instruction
  /// Returns as a Value* to be compatible with `vector_operands`
  auto vector_users(Instruction *I) const {
    return map_range(I->users(), [this](User *U) -> Value * { return U; });
  }
};
} // end namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_SCALABLETOFIXEDVECTORS_H
