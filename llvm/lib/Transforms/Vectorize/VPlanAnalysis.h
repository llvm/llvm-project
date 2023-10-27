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

/// An analysis for type-inference for VPValues.
/// It infers the scalar type for a given VPValue by bottom-up traversing
/// through defining recipes until root nodes with known types are reached (e.g.
/// live-ins or load recipes). The types are then propagated top down through
/// operations.
/// Note that the analysis caches the inferred types. A new analysis object must
/// be constructed once a VPlan has been modified in a way that invalidates any
/// of the previously inferred types.
class VPTypeAnalysis {
  DenseMap<const VPValue *, Type *> CachedTypes;
  LLVMContext &Ctx;

  Type *inferScalarTypeForRecipe(const VPBlendRecipe *R);
  Type *inferScalarTypeForRecipe(const VPInstruction *R);
  Type *inferScalarTypeForRecipe(const VPWidenCallRecipe *R);
  Type *inferScalarTypeForRecipe(const VPWidenRecipe *R);
  Type *inferScalarTypeForRecipe(const VPWidenIntOrFpInductionRecipe *R);
  Type *inferScalarTypeForRecipe(const VPWidenMemoryInstructionRecipe *R);
  Type *inferScalarTypeForRecipe(const VPWidenSelectRecipe *R);
  Type *inferScalarTypeForRecipe(const VPReplicateRecipe *R);

public:
  VPTypeAnalysis(LLVMContext &Ctx) : Ctx(Ctx) {}

  /// Infer the type of \p V. Returns the scalar type of \p V.
  Type *inferScalarType(const VPValue *V);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_VPLANANALYSIS_H
