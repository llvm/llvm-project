//===- SLPVPlanCodegen.h - VPlan-based codegen for SLP ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helpers building and executing the VPlan for an SLP tree that do not depend
// on BoUpSLP.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TRANSFORMS_VECTORIZE_SLPVPLANCODEGEN_H
#define LLVM_LIB_TRANSFORMS_VECTORIZE_SLPVPLANCODEGEN_H

#include "llvm/ADT/ArrayRef.h"

namespace llvm {
class Instruction;
class Value;
struct VPBuilderDefaultInserter;
template <typename InserterTy> class VPBuilderBase;
using VPBuilder = VPBuilderBase<VPBuilderDefaultInserter>;
class VPValue;
class VPlan;
struct VPTransformState;

namespace slpvectorizer {

/// Returns true if operand \p J of a bundle with main opcode \p Opcode is a
/// plan live-in rather than being defined by another tree entry.
bool isLiveInOperand(unsigned Opcode, unsigned J);

/// Returns true if VPlan-based codegen can emit a recipe for a bundle with
/// main opcode \p Opcode. Keep in sync with createRecipeForBundle().
bool isSupportedVPlanCodegenOpcode(unsigned Opcode);

/// Creates the recipe for the bundle \p Scalars with main instruction \p MainOp
/// and vector operands \p Ops. Returns its value, or nullptr for stores.
VPValue *createRecipeForBundle(VPlan &Plan, VPBuilder &VPB, Instruction *MainOp,
                               ArrayRef<Value *> Scalars,
                               ArrayRef<VPValue *> Ops);

/// Executes the recipes of \p Plan, which are already in execution order.
void executeSLPPlan(VPlan &Plan, VPTransformState &State);

} // namespace slpvectorizer
} // namespace llvm

#endif // LLVM_LIB_TRANSFORMS_VECTORIZE_SLPVPLANCODEGEN_H
