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
#include "llvm/CodeGen/MachineOperand.h"
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

  // The vector register can be def'd by various ops that have vector as its
  // type. They can all be used for constant folding, scalarizing,
  // canonicalization, or combining based on symmetry.
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

  // We try to get the value of the Index register.
  std::optional<ValueAndVReg> MaybeIndex =
      getIConstantVRegValWithLookThrough(Index, MRI);
  std::optional<APInt> IndexC = std::nullopt;

  if (MaybeIndex)
    IndexC = MaybeIndex->Value;

  // Fold extractVectorElement(Vector, TOOLARGE) -> undef
  if (IndexC && VectorTy.isFixedVector() &&
      IndexC->getZExtValue() >= VectorTy.getNumElements() &&
      isLegalOrBeforeLegalizer({TargetOpcode::G_IMPLICIT_DEF, {DstTy}})) {
    // For fixed-length vectors, it's invalid to extract out-of-range elements.
    MatchInfo = [=](MachineIRBuilder &B) { B.buildUndef(Dst); };
    return true;
  }

  return false;
}

bool CombinerHelper::matchExtractVectorElementWithDifferentIndices(
    const MachineOperand &MO, BuildFnTy &MatchInfo) {
  MachineInstr *Root = getDefIgnoringCopies(MO.getReg(), MRI);
  GExtractVectorElement *Extract = cast<GExtractVectorElement>(Root);

  //
  //  %idx1:_(s64) = G_CONSTANT i64 1
  //  %idx2:_(s64) = G_CONSTANT i64 2
  //  %insert:_(<2 x s32>) = G_INSERT_VECTOR_ELT_ELT %bv(<2 x s32>),
  //  %value(s32), %idx2(s64) %extract:_(s32) = G_EXTRACT_VECTOR_ELT %insert(<2
  //  x s32>), %idx1(s64)
  //
  //  -->
  //
  //  %insert:_(<2 x s32>) = G_INSERT_VECTOR_ELT_ELT %bv(<2 x s32>),
  //  %value(s32), %idx2(s64) %extract:_(s32) = G_EXTRACT_VECTOR_ELT %bv(<2 x
  //  s32>), %idx1(s64)
  //
  //

  Register Index = Extract->getIndexReg();

  // We try to get the value of the Index register.
  std::optional<ValueAndVReg> MaybeIndex =
      getIConstantVRegValWithLookThrough(Index, MRI);
  std::optional<APInt> IndexC = std::nullopt;

  if (!MaybeIndex)
    return false;
  else
    IndexC = MaybeIndex->Value;

  Register Vector = Extract->getVectorReg();

  GInsertVectorElement *Insert =
      getOpcodeDef<GInsertVectorElement>(Vector, MRI);
  if (!Insert)
    return false;

  Register Dst = Extract->getReg(0);

  std::optional<ValueAndVReg> MaybeInsertIndex =
      getIConstantVRegValWithLookThrough(Insert->getIndexReg(), MRI);

  if (MaybeInsertIndex && MaybeInsertIndex->Value != *IndexC) {
    // There is no one-use check. We have to keep the insert. When both Index
    // registers are constants and not equal, we can look into the Vector
    // register of the insert.
    MatchInfo = [=](MachineIRBuilder &B) {
      B.buildExtractVectorElement(Dst, Insert->getVectorReg(), Index);
    };
    return true;
  }

  return false;
}

bool CombinerHelper::matchExtractVectorElementWithFreeze(
    const MachineOperand &MO, BuildFnTy &MatchInfo) {
  MachineInstr *Root = getDefIgnoringCopies(MO.getReg(), MRI);
  GExtractVectorElement *Extract = cast<GExtractVectorElement>(Root);

  Register Vector = Extract->getVectorReg();

  //
  //  %bv:_(<2 x s32>) = G_BUILD_VECTOR %arg1(s32), %arg2(s32)
  //  %freeze:_(<2 x s32>) = G_FREEZE %bv(<2 x s32>)
  //  %extract:_(s32) = G_EXTRACT_VECTOR_ELT %bv(<2 x s32>), %opaque(s64)
  //
  //  -->
  //
  //  %bv:_(<2 x s32>) = G_BUILD_VECTOR %arg1(s32), %arg2(s32)
  //  %extract:_(s32) = G_EXTRACT_VECTOR_ELT %bv(<2 x s32>), %opaque(s64)
  //  %freeze:_(s32) = G_FREEZE %extract(s32)
  //
  //

  // For G_FREEZE, the input and the output types are identical. Moving the
  // freeze from the Vector into the front of the extract preserves the freeze
  // semantics. The result is still freeze'd. Furthermore, the Vector register
  // becomes easier to analyze. A build vector could have been hidden behind the
  // freeze.

  // We expect a freeze on the Vector register.
  GFreeze *Freeze = getOpcodeDef<GFreeze>(Vector, MRI);
  if (!Freeze)
    return false;

  Register Dst = Extract->getReg(0);
  LLT DstTy = MRI.getType(Dst);

  // We first have to check for one-use and legality of the freeze.
  // The type of the extractVectorElement did not change.
  if (!MRI.hasOneNonDBGUse(Freeze->getReg(0)) ||
      !isLegalOrBeforeLegalizer({TargetOpcode::G_FREEZE, {DstTy}}))
    return false;

  Register Index = Extract->getIndexReg();

  // We move the freeze from the Vector register in front of the
  // extractVectorElement.
  MatchInfo = [=](MachineIRBuilder &B) {
    auto Extract =
        B.buildExtractVectorElement(DstTy, Freeze->getSourceReg(), Index);
    B.buildFreeze(Dst, Extract);
  };

  return true;
}

bool CombinerHelper::matchExtractVectorElementWithBuildVector(
    const MachineOperand &MO, BuildFnTy &MatchInfo) {
  MachineInstr *Root = getDefIgnoringCopies(MO.getReg(), MRI);
  GExtractVectorElement *Extract = cast<GExtractVectorElement>(Root);

  //
  //  %zero:_(s64) = G_CONSTANT i64 0
  //  %bv:_(<2 x s32>) = G_BUILD_VECTOR %arg1(s32), %arg2(s32)
  //  %extract:_(s32) = G_EXTRACT_VECTOR_ELT %bv(<2 x s32>), %zero(s64)
  //
  //  -->
  //
  //  %extract:_(32) = COPY %arg1(s32)
  //
  //
  //
  //  %bv:_(<2 x s32>) = G_BUILD_VECTOR %arg1(s32), %arg2(s32)
  //  %extract:_(s32) = G_EXTRACT_VECTOR_ELT %bv(<2 x s32>), %opaque(s64)
  //
  //  -->
  //
  //  %bv:_(<2 x s32>) = G_BUILD_VECTOR %arg1(s32), %arg2(s32)
  //  %extract:_(s32) = G_EXTRACT_VECTOR_ELT %bv(<2 x s32>), %opaque(s64)
  //

  Register Vector = Extract->getVectorReg();

  // We expect a buildVector on the Vector register.
  GBuildVector *Build = getOpcodeDef<GBuildVector>(Vector, MRI);
  if (!Build)
    return false;

  LLT VectorTy = MRI.getType(Vector);

  // There is a one-use check. There are more combines on build vectors.
  EVT Ty(getMVTForLLT(VectorTy));
  if (!MRI.hasOneNonDBGUse(Build->getReg(0)) ||
      !getTargetLowering().aggressivelyPreferBuildVectorSources(Ty))
    return false;

  Register Index = Extract->getIndexReg();

  // If the Index is constant, then we can extract the element from the given
  // offset.
  std::optional<ValueAndVReg> MaybeIndex =
      getIConstantVRegValWithLookThrough(Index, MRI);
  if (!MaybeIndex)
    return false;

  // We now know that there is a buildVector def'd on the Vector register and
  // the index is const. The combine will succeed.

  Register Dst = Extract->getReg(0);

  MatchInfo = [=](MachineIRBuilder &B) {
    B.buildCopy(Dst, Build->getSourceReg(MaybeIndex->Value.getZExtValue()));
  };

  return true;
}

bool CombinerHelper::matchExtractVectorElementWithBuildVectorTrunc(
    const MachineOperand &MO, BuildFnTy &MatchInfo) {
  MachineInstr *Root = getDefIgnoringCopies(MO.getReg(), MRI);
  GExtractVectorElement *Extract = cast<GExtractVectorElement>(Root);

  //
  //  %zero:_(s64) = G_CONSTANT i64 0
  //  %bv:_(<2 x s32>) = G_BUILD_VECTOR_TRUNC %arg1(s64), %arg2(s64)
  //  %extract:_(s32) = G_EXTRACT_VECTOR_ELT %bv(<2 x s32>), %zero(s64)
  //
  //  -->
  //
  //  %extract:_(32) = G_TRUNC %arg1(s64)
  //
  //
  //
  //  %bv:_(<2 x s32>) = G_BUILD_VECTOR_TRUNC %arg1(s64), %arg2(s64)
  //  %extract:_(s32) = G_EXTRACT_VECTOR_ELT %bv(<2 x s32>), %opaque(s64)
  //
  //  -->
  //
  //  %bv:_(<2 x s32>) = G_BUILD_VECTOR_TRUNC %arg1(s64), %arg2(s64)
  //  %extract:_(s32) = G_EXTRACT_VECTOR_ELT %bv(<2 x s32>), %opaque(s64)
  //

  Register Vector = Extract->getVectorReg();

  // We expect a buildVectorTrunc on the Vector register.
  GBuildVectorTrunc *Build = getOpcodeDef<GBuildVectorTrunc>(Vector, MRI);
  if (!Build)
    return false;

  LLT VectorTy = MRI.getType(Vector);

  // There is a one-use check. There are more combines on build vectors.
  EVT Ty(getMVTForLLT(VectorTy));
  if (!MRI.hasOneNonDBGUse(Build->getReg(0)) ||
      !getTargetLowering().aggressivelyPreferBuildVectorSources(Ty))
    return false;

  Register Index = Extract->getIndexReg();

  // If the Index is constant, then we can extract the element from the given
  // offset.
  std::optional<ValueAndVReg> MaybeIndex =
      getIConstantVRegValWithLookThrough(Index, MRI);
  if (!MaybeIndex)
    return false;

  // We now know that there is a buildVectorTrunc def'd on the Vector register
  // and the index is const. The combine will succeed.

  Register Dst = Extract->getReg(0);
  LLT DstTy = MRI.getType(Dst);
  LLT SrcTy = MRI.getType(Build->getSourceReg(0));

  // For buildVectorTrunc, the inputs are truncated.
  if (!isLegalOrBeforeLegalizer({TargetOpcode::G_TRUNC, {DstTy, SrcTy}}))
    return false;

  MatchInfo = [=](MachineIRBuilder &B) {
    B.buildTrunc(Dst, Build->getSourceReg(MaybeIndex->Value.getZExtValue()));
  };

  return true;
}
