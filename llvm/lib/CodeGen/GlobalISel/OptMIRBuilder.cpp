//===-- llvm/CodeGen/GlobalISel/OptMIRBuilder.cpp -----------------*- C++-*-==//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements the OptMIRBuilder class which optimizes as it builds
/// instructions.
//===----------------------------------------------------------------------===//
//

#include "llvm/CodeGen/GlobalISel/OptMIRBuilder.h"
#include "llvm/CodeGen/GlobalISel/CSEMIRBuilder.h"
#include "llvm/CodeGen/GlobalISel/LegalizerInfo.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/TargetOpcodes.h"

using namespace llvm;

bool OptMIRBuilder::isPrelegalize() const { return IsPrelegalize; }

bool OptMIRBuilder::isLegal(const LegalityQuery &Query) const {
  assert(LI != nullptr && "legalizer info is not available");
  return LI->isLegal(Query);
}

bool OptMIRBuilder::isConstantLegal(LLT Ty) {
  if (Ty.isScalar())
    return isLegal({TargetOpcode::G_CONSTANT, {Ty}});

  LLT EltTy = Ty.getElementType();
  if (Ty.isFixedVector())
    return isLegal({TargetOpcode::G_BUILD_VECTOR, {Ty, EltTy}}) &&
           isLegal({TargetOpcode::G_CONSTANT, {EltTy}});

  // scalable vector
  assert(Ty.isScalableVector() && "Unexpected LLT");
  return isLegal({TargetOpcode::G_SPLAT_VECTOR, {Ty, EltTy}}) &&
         isLegal({TargetOpcode::G_CONSTANT, {EltTy}});
}

bool OptMIRBuilder::isLegalOrBeforeLegalizer(const LegalityQuery &Query) const {
  return isPrelegalize() || isLegal(Query);
}

bool OptMIRBuilder::isConstantLegalOrBeforeLegalizer(LLT Ty) {
  return isPrelegalize() || isConstantLegal(Ty);
}

bool OptMIRBuilder::isUndef(Register Reg) const {
  const MachineInstr *MI = getMRI()->getVRegDef(Reg);
  return MI->getOpcode() == TargetOpcode::G_IMPLICIT_DEF;
}

MachineInstrBuilder OptMIRBuilder::buildNegation(const DstOp &DstOp,
                                                 const SrcOp &SrcOp) {
  LLT DstTy = DstOp.getLLTTy(*getMRI());

  auto Zero = buildConstant(DstTy, 0);
  return buildSub(DstOp, Zero, SrcOp);
}

MachineInstrBuilder OptMIRBuilder::buildGIConstant(const DstOp &DstOp,
                                                   const GIConstant &Const) {
  LLT DstTy = DstOp.getLLTTy(*getMRI());

  switch (Const.getKind()) {
  case GIConstant::GIConstantKind::Scalar:
    return buildConstant(DstOp, Const.getScalarValue());
  case GIConstant::GIConstantKind::FixedVector:
    return buildBuildVectorConstant(DstOp, Const.getAsArrayRef());
  case GIConstant::GIConstantKind::ScalableVector: {
    auto Constant =
        buildConstant(DstTy.getElementType(), Const.getSplatValue());
    return buildSplatVector(DstOp, Constant);
  }
  }
}

MachineInstrBuilder OptMIRBuilder::optimizeAdd(unsigned Opc,
                                               ArrayRef<DstOp> DstOps,
                                               ArrayRef<SrcOp> SrcOps,
                                               std::optional<unsigned> Flag) {
  assert(SrcOps.size() == 2 && "Invalid sources");
  assert(DstOps.size() == 1 && "Invalid dsts");

  LLT DstTy = DstOps[0].getLLTTy(*getMRI());

  if (isUndef(SrcOps[1].getReg()) || isUndef(SrcOps[0].getReg()))
    return buildUndef(DstTy);

  std::optional<GIConstant> RHS =
      GIConstant::getConstant(SrcOps[1].getReg(), *getMRI());
  if (!RHS)
    return CSEMIRBuilder::buildInstr(Opc, DstOps, SrcOps, Flag);

  if (RHS->isZero())
    return buildCopy(DstOps[0], SrcOps[0]);

  if (!isConstantLegalOrBeforeLegalizer(DstTy))
    return CSEMIRBuilder::buildInstr(Opc, DstOps, SrcOps, Flag);

  std::optional<GIConstant> LHS =
      GIConstant::getConstant(SrcOps[0].getReg(), *getMRI());
  if (!LHS)
    return CSEMIRBuilder::buildInstr(Opc, DstOps, SrcOps, Flag);

  GIConstant Add = LHS->add(*RHS);

  return buildGIConstant(DstOps[0], Add);
}

MachineInstrBuilder OptMIRBuilder::optimizeSub(unsigned Opc,
                                               ArrayRef<DstOp> DstOps,
                                               ArrayRef<SrcOp> SrcOps,
                                               std::optional<unsigned> Flag) {
  assert(SrcOps.size() == 2 && "Invalid sources");
  assert(DstOps.size() == 1 && "Invalid dsts");

  LLT DstTy = DstOps[0].getLLTTy(*getMRI());

  if (isUndef(SrcOps[1].getReg()) || isUndef(SrcOps[0].getReg()))
    return buildUndef(DstTy);

  std::optional<GIConstant> RHS =
      GIConstant::getConstant(SrcOps[1].getReg(), *getMRI());
  if (!RHS)
    return CSEMIRBuilder::buildInstr(Opc, DstOps, SrcOps, Flag);

  if (RHS->isZero())
    return buildCopy(DstOps[0], SrcOps[0]);

  if (!isConstantLegalOrBeforeLegalizer(DstTy))
    return CSEMIRBuilder::buildInstr(Opc, DstOps, SrcOps, Flag);

  std::optional<GIConstant> LHS =
      GIConstant::getConstant(SrcOps[0].getReg(), *getMRI());
  if (!LHS)
    return CSEMIRBuilder::buildInstr(Opc, DstOps, SrcOps, Flag);

  GIConstant Sub = LHS->sub(*RHS);

  return buildGIConstant(DstOps[0], Sub);
}

MachineInstrBuilder OptMIRBuilder::optimizeMul(unsigned Opc,
                                               ArrayRef<DstOp> DstOps,
                                               ArrayRef<SrcOp> SrcOps,
                                               std::optional<unsigned> Flag) {
  assert(SrcOps.size() == 2 && "Invalid sources");
  assert(DstOps.size() == 1 && "Invalid dsts");

  LLT DstTy = DstOps[0].getLLTTy(*getMRI());

  if ((isUndef(SrcOps[1].getReg()) || isUndef(SrcOps[0].getReg())) &&
      isConstantLegalOrBeforeLegalizer(DstTy))
    return buildConstant(DstTy, 0);

  std::optional<GIConstant> RHS =
      GIConstant::getConstant(SrcOps[1].getReg(), *getMRI());
  if (!RHS)
    return CSEMIRBuilder::buildInstr(Opc, DstOps, SrcOps, Flag);

  if (RHS->isZero() && isConstantLegalOrBeforeLegalizer(DstTy))
    return buildConstant(DstTy, 0);

  if (RHS->isOne())
    return buildCopy(DstOps[0], SrcOps[0]);

  if (RHS->isAllOnes() && isConstantLegalOrBeforeLegalizer(DstTy) &&
      isLegalOrBeforeLegalizer({TargetOpcode::G_SUB, {DstTy}}))
    return buildNegation(DstOps[0], SrcOps[0]);

  if (RHS->isTwo() && isLegalOrBeforeLegalizer({TargetOpcode::G_ADD, {DstTy}}))
    return buildAdd(DstOps[0], SrcOps[0], SrcOps[0]);

  if (!isConstantLegalOrBeforeLegalizer(DstTy))
    return CSEMIRBuilder::buildInstr(Opc, DstOps, SrcOps, Flag);

  std::optional<GIConstant> LHS =
      GIConstant::getConstant(SrcOps[0].getReg(), *getMRI());
  if (!LHS)
    return CSEMIRBuilder::buildInstr(Opc, DstOps, SrcOps, Flag);

  GIConstant Mul = LHS->mul(*RHS);

  return buildGIConstant(DstOps[0], Mul);
}

MachineInstrBuilder OptMIRBuilder::buildInstr(unsigned Opc,
                                              ArrayRef<DstOp> DstOps,
                                              ArrayRef<SrcOp> SrcOps,
                                              std::optional<unsigned> Flag) {
  switch (Opc) {
  case TargetOpcode::G_ADD:
    return optimizeAdd(Opc, DstOps, SrcOps, Flag);
  case TargetOpcode::G_SUB:
    return optimizeSub(Opc, DstOps, SrcOps, Flag);
  case TargetOpcode::G_MUL:
    return optimizeMul(Opc, DstOps, SrcOps, Flag);
  default:
    return CSEMIRBuilder::buildInstr(Opc, DstOps, SrcOps, Flag);
  }
}
