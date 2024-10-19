//===- CombinerHelperArtifacts.cpp-----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements CombinerHelper for legalization artifacts.
//
//===----------------------------------------------------------------------===//
//
// G_UNMERGE_VALUES
//
//===----------------------------------------------------------------------===//
#include "llvm/CodeGen/GlobalISel/CombinerHelper.h"
#include "llvm/CodeGen/GlobalISel/LegalizerHelper.h"
#include "llvm/CodeGen/GlobalISel/LegalizerInfo.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/LowLevelTypeUtils.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/Support/Casting.h"

#define DEBUG_TYPE "gi-combiner"

using namespace llvm;

bool CombinerHelper::matchUnmergeValuesAnyExtBuildVector(const MachineInstr &MI,
                                                         BuildFnTy &MatchInfo) {
  const GUnmerge *Unmerge = cast<GUnmerge>(&MI);

  if (!MRI.hasOneNonDBGUse(Unmerge->getSourceReg()))
    return false;

  const MachineInstr *Source = MRI.getVRegDef(Unmerge->getSourceReg());

  LLT DstTy = MRI.getType(Unmerge->getReg(0));

  // $bv:_(<8 x s8>) = G_BUILD_VECTOR ....
  // $any:_(<8 x s16>) = G_ANYEXT $bv
  // $uv:_(<4 x s16>), $uv1:_(<4 x s16>) = G_UNMERGE_VALUES $any
  //
  // ->
  //
  // $any:_(s16) = G_ANYEXT $bv[0]
  // $any1:_(s16) = G_ANYEXT $bv[1]
  // $any2:_(s16) = G_ANYEXT $bv[2]
  // $any3:_(s16) = G_ANYEXT $bv[3]
  // $any4:_(s16) = G_ANYEXT $bv[4]
  // $any5:_(s16) = G_ANYEXT $bv[5]
  // $any6:_(s16) = G_ANYEXT $bv[6]
  // $any7:_(s16) = G_ANYEXT $bv[7]
  // $uv:_(<4 x s16>) = G_BUILD_VECTOR $any, $any1, $any2, $any3
  // $uv1:_(<4 x s16>) = G_BUILD_VECTOR $any4, $any5, $any6, $any7

  // We want to unmerge into vectors.
  if (!DstTy.isFixedVector())
    return false;

  const GAnyExt *Any = dyn_cast<GAnyExt>(Source);
  if (!Any)
    return false;

  const MachineInstr *NextSource = MRI.getVRegDef(Any->getSrcReg());

  if (const GBuildVector *BV = dyn_cast<GBuildVector>(NextSource)) {
    // G_UNMERGE_VALUES G_ANYEXT G_BUILD_VECTOR

    if (!MRI.hasOneNonDBGUse(BV->getReg(0)))
      return false;

    // FIXME: check element types?
    if (BV->getNumSources() % Unmerge->getNumDefs() != 0)
      return false;

    LLT BigBvTy = MRI.getType(BV->getReg(0));
    LLT SmallBvTy = DstTy;
    LLT SmallBvElemenTy = SmallBvTy.getElementType();

    if (!isLegalOrBeforeLegalizer(
            {TargetOpcode::G_BUILD_VECTOR, {SmallBvTy, SmallBvElemenTy}}))
      return false;

    // We check the legality of scalar anyext.
    if (!isLegalOrBeforeLegalizer(
            {TargetOpcode::G_ANYEXT,
             {SmallBvElemenTy, BigBvTy.getElementType()}}))
      return false;

    MatchInfo = [=](MachineIRBuilder &B) {
      // Build into each G_UNMERGE_VALUES def
      // a small build vector with anyext from the source build vector.
      for (unsigned I = 0; I < Unmerge->getNumDefs(); ++I) {
        SmallVector<Register> Ops;
        for (unsigned J = 0; J < SmallBvTy.getNumElements(); ++J) {
          Register SourceArray =
              BV->getSourceReg(I * SmallBvTy.getNumElements() + J);
          auto AnyExt = B.buildAnyExt(SmallBvElemenTy, SourceArray);
          Ops.push_back(AnyExt.getReg(0));
        }
        B.buildBuildVector(Unmerge->getOperand(I).getReg(), Ops);
      };
    };
    return true;
  };

  return false;
}

bool CombinerHelper::matchUnmergeValuesOfScalarAndVector(const MachineInstr &MI,
                                                         BuildFnTy &MatchInfo) {

  constexpr unsigned MAX_NUM_DEFS_LIMIT = 8;

  //  %opaque:_(<2 x s64>) = G_OPAQUE
  //  %un1:_(s64), %un2:_(s64) = G_UNMERGE_VALUES %opaque(<2 x s64>)
  //
  //  ->
  //
  //  %zero:_(s64) = G_CONSTANT i64 0
  //  %one:_(s64) = G_CONSTANT i64 1
  //  %un1:_(s64) = G_EXTRACT_VECTOR_ELT %opaque, $zero
  //  %un2:_(s64) = G_EXTRACT_VECTOR_ELT %opaque, $one

  const GUnmerge *Unmerge = cast<GUnmerge>(&MI);

  if (Unmerge->getNumDefs() > MAX_NUM_DEFS_LIMIT)
    return false;

  LLT DstTy = MRI.getType(Unmerge->getReg(0));
  LLT SrcTy = MRI.getType(Unmerge->getSourceReg());

  // We want to unmerge a vector into scalars.
  if (!DstTy.isScalar() || !SrcTy.isFixedVector() || DstTy.getSizeInBits() > 64)
    return false;

  if (DstTy != SrcTy.getElementType())
    return false;

  // We want to unmerge from an opaque vector.
  const MachineInstr *Source = MRI.getVRegDef(Unmerge->getSourceReg());
  if (isa<GBuildVector>(Source))
    return false;

  unsigned PreferredVecIdxWidth =
      getTargetLowering().getVectorIdxTy(getDataLayout()).getSizeInBits();

  LLT IdxTy = LLT::scalar(PreferredVecIdxWidth);

  if (!isLegalOrBeforeLegalizer(
          {TargetOpcode::G_EXTRACT_VECTOR_ELT, {DstTy, SrcTy, IdxTy}}))
    return false;

  if (!isConstantLegalOrBeforeLegalizer(IdxTy))
    return false;

  MatchInfo = [=](MachineIRBuilder &B) {
    for (unsigned I = 0; I < Unmerge->getNumDefs(); ++I) {
      auto Index = B.buildConstant(IdxTy, I);
      B.buildExtractVectorElement(Unmerge->getOperand(I).getReg(),
                                  Unmerge->getSourceReg(), Index);
    }
  };

  return true;
}
