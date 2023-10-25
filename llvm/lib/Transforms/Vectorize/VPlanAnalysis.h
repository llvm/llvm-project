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
/// live-ins or memory recipes). The types are then propagated top down through
/// operations.
/// Note that the analysis caches the infered types. A new analysis object must
/// be constructed once a VPlan has been modified in a way that invalidates any
/// of the previously infered types.
class VPTypeAnalysis {
  DenseMap<const VPValue *, Type *> CachedTypes;
  LLVMContext &Ctx;

  Type *inferScalarType(const VPBlendRecipe *R);
  Type *inferScalarType(const VPInstruction *R);
  Type *inferScalarType(const VPWidenCallRecipe *R);
  Type *inferScalarType(const VPWidenRecipe *R);
  Type *inferScalarType(const VPWidenIntOrFpInductionRecipe *R);
  Type *inferScalarType(const VPWidenMemoryInstructionRecipe *R);
  Type *inferScalarType(const VPWidenSelectRecipe *R);
  Type *inferScalarType(const VPReplicateRecipe *R);

public:
  VPTypeAnalysis(LLVMContext &Ctx) : Ctx(Ctx) {}

  /// Infer the type of \p V. Returns the scalar type of \p V.
  Type *inferScalarType(const VPValue *V);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_VPLANANALYSIS_H
