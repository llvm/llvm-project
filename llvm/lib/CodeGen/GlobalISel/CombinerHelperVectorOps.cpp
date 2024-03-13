//===- CombinerHelperVectorOps.cpp-----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements CombinerHelper for G_EXTRACT_VECTOR_ELT.
//
//===----------------------------------------------------------------------===//
#include "llvm/CodeGen/GlobalISel/CombinerHelper.h"
#include "llvm/CodeGen/GlobalISel/GenericMachineInstrs.h"
#include "llvm/CodeGen/GlobalISel/LegalizerHelper.h"
#include "llvm/CodeGen/GlobalISel/LegalizerInfo.h"
#include "llvm/CodeGen/GlobalISel/MIPatternMatch.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/LowLevelTypeUtils.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/Support/Casting.h"
#include <optional>

#define DEBUG_TYPE "gi-combiner"

using namespace llvm;
using namespace MIPatternMatch;

bool CombinerHelper::matchExtractVectorElement(MachineInstr &MI,
                                               BuildFnTy &MatchInfo) {

  GExtractVectorElement *Extract = cast<GExtractVectorElement>(&MI);

  Register Dst = Extract->getReg(0);
  Register Vector = Extract->getVectorReg();
  Register Index = Extract->getIndexReg();
  LLT DstTy = MRI.getType(Dst);
  LLT VectorTy = MRI.getType(Vector);

  // The vector register can be def'd by various ops that
  // have vector as its type. They can all be used for
  // constant folding, scalarizing, canonicalization, or
  // combining based on symmetry.
  //
  // vector like ops
  // * build vector
  // * build vector trunc
  // * shuffle vector
  // * splat vector
  // * concat vectors
  // * insert/extract vector element
  // * insert/extract subvector
  // * vector loads
  // * scalable vector loads
  //
  // compute like ops
  // * binary ops
  // * unary ops
  //  * exts and truncs
  //  * casts
  //  * fneg
  // * select
  // * phis
  // * cmps
  // * freeze
  // * bitcast
  // * undef

  // Fold extractVectorElement(undef, undef) -> undef
  if ((getOpcodeDef<GImplicitDef>(Vector, MRI) ||
       getOpcodeDef<GImplicitDef>(Index, MRI)) &&
      isLegalOrBeforeLegalizer({TargetOpcode::G_IMPLICIT_DEF, {DstTy}})) {
    // If the Vector register is undef, then we cannot extract an element from
    // it. An undef extract Index can be arbitrarily chosen to be an
    // out-of-range index value, which would result in the instruction being
    // poison.
    MatchInfo = [=](MachineIRBuilder &B) { B.buildUndef(Dst); };
    return true;
  }

  // We try to get the value of the Index register.
  std::optional<ValueAndVReg> MaybeIndex =
      getIConstantVRegValWithLookThrough(Index, MRI);
  std::optional<APInt> IndexC = std::nullopt;

  if (MaybeIndex)
    IndexC = MaybeIndex->Value;

  // Fold extractVectorElement(Vector, TOOLARGE) -> undef
  if (IndexC && VectorTy.isFixedVector() &&
      IndexC->uge(VectorTy.getNumElements()) &&
      isLegalOrBeforeLegalizer({TargetOpcode::G_IMPLICIT_DEF, {DstTy}})) {
    // For fixed-length vectors, it's invalid to extract out-of-range elements.
    MatchInfo = [=](MachineIRBuilder &B) { B.buildUndef(Dst); };
    return true;
  }

  // Fold extractVectorElement(freeze(FV), Index) ->
  //     freeze(extractVectorElement(FV, Index))
  if (auto *Freeze = getOpcodeDef<GFreeze>(Vector, MRI)) {
    if (MRI.hasOneNonDBGUse(Freeze->getReg(0)) &&
        isLegalOrBeforeLegalizer({TargetOpcode::G_FREEZE, {DstTy}})) {
      // For G_FREEZE, the input and the output types are identical.
      // Moving the freeze from the Vector into the front of the extract
      // preserves the freeze semantics. We check above that
      // the Index register is not undef.
      // Furthermore, the Vector register
      // becomes easier to analyze. A build vector
      // could have been hidden behind the freeze.
      MatchInfo = [=](MachineIRBuilder &B) {
        auto Extract =
            B.buildExtractVectorElement(DstTy, Freeze->getSourceReg(), Index);
        B.buildFreeze(Dst, Extract);
      };
      return true;
    }
  }

  // Fold extractVectorElement(insertVectorElement(_, Value, Index), Index) ->
  // Value
  if (auto *Insert = getOpcodeDef<GInsertVectorElement>(Vector, MRI)) {
    if (Insert->getIndexReg() == Index) {
      // There is no one-use check. We have to keep the insert.
      // We only check for equality of the Index registers.
      // The combine is independent of their constness.
      // We try to insert Value and then immediately extract
      // it from the same Index.
      MatchInfo = [=](MachineIRBuilder &B) {
        B.buildCopy(Dst, Insert->getElementReg());
      };
      return true;
    }
  }

  // Fold extractVectorElement(insertVectorElement(Vector, _, C1), C2),
  // where C1 != C2
  // -> extractVectorElement(Vector, C2)
  if (IndexC) {
    if (auto *Insert = getOpcodeDef<GInsertVectorElement>(Vector, MRI)) {
      std::optional<ValueAndVReg> MaybeIndex =
          getIConstantVRegValWithLookThrough(Insert->getIndexReg(), MRI);
      if (MaybeIndex && MaybeIndex->Value != *IndexC) {
        // There is no one-use check. We have to keep the insert.
        // When both Index registers are constants and not equal,
        // we can look into the Vector register of the insert.
        MatchInfo = [=](MachineIRBuilder &B) {
          B.buildExtractVectorElement(Dst, Insert->getVectorReg(), Index);
        };
        return true;
      }
    }
  }

  // Fold extractVectorElement(BuildVector(.., V, ...), IndexOfV) -> V
  if (IndexC) {
    if (auto *Build = getOpcodeDef<GBuildVector>(Vector, MRI)) {
      EVT Ty(getMVTForLLT(VectorTy));
      if (MRI.hasOneNonDBGUse(Build->getReg(0)) ||
          getTargetLowering().aggressivelyPreferBuildVectorSources(Ty)) {
        // There is a one-use check. There are more combines on build vectors.
        // If the Index is constant, then we can extract the element from the
        // given offset.
        MatchInfo = [=](MachineIRBuilder &B) {
          B.buildCopy(Dst, Build->getSourceReg(IndexC->getLimitedValue()));
        };
        return true;
      }
    }
  }

  return false;
}
