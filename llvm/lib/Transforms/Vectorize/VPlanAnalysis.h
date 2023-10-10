//===- VPlanAnalysis.h - Various Analyses working on VPlan ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_VPLANANALYSIS_H
#define LLVM_TRANSFORMS_VECTORIZE_VPLANANALYSIS_H

#include "llvm/ADT/DenseMap.h"

namespace llvm {

class LLVMContext;
class VPValue;
class VPBlendRecipe;
class VPInterleaveRecipe;
class VPInstruction;
class VPReductionPHIRecipe;
class VPWidenRecipe;
class VPWidenCallRecipe;
class VPWidenCastRecipe;
class VPWidenIntOrFpInductionRecipe;
class VPWidenMemoryInstructionRecipe;
struct VPWidenSelectRecipe;
class VPReplicateRecipe;
class Type;

/// An analysis for type-inferrence for VPValues.
class VPTypeAnalysis {
  DenseMap<const VPValue *, Type *> CachedTypes;
  LLVMContext &Ctx;

  Type *inferType(const VPBlendRecipe *R);
  Type *inferType(const VPInstruction *R);
  Type *inferType(const VPInterleaveRecipe *R);
  Type *inferType(const VPWidenCallRecipe *R);
  Type *inferType(const VPReductionPHIRecipe *R);
  Type *inferType(const VPWidenRecipe *R);
  Type *inferType(const VPWidenIntOrFpInductionRecipe *R);
  Type *inferType(const VPWidenMemoryInstructionRecipe *R);
  Type *inferType(const VPWidenSelectRecipe *R);
  Type *inferType(const VPReplicateRecipe *R);

public:
  VPTypeAnalysis(LLVMContext &Ctx) : Ctx(Ctx) {}

  /// Infer the type of \p V. Returns the scalar type of \p V.
  Type *inferType(const VPValue *V);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_VPLANANALYSIS_H
