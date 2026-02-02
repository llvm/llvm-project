//===-- SPIRVCombinerHelper.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SPIRVCombinerHelper.h"
#include "SPIRVGlobalRegistry.h"
#include "SPIRVUtils.h"
#include "llvm/CodeGen/GlobalISel/GenericMachineInstrs.h"
#include "llvm/CodeGen/GlobalISel/MIPatternMatch.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IntrinsicsSPIRV.h"
#include "llvm/IR/LLVMContext.h" // Explicitly include for LLVMContext
#include "llvm/Target/TargetMachine.h"

using namespace llvm;
using namespace MIPatternMatch;

SPIRVCombinerHelper::SPIRVCombinerHelper(
    GISelChangeObserver &Observer, MachineIRBuilder &B, bool IsPreLegalize,
    GISelValueTracking *VT, MachineDominatorTree *MDT, const LegalizerInfo *LI,
    const SPIRVSubtarget &STI)
    : CombinerHelper(Observer, B, IsPreLegalize, VT, MDT, LI), STI(STI) {}

/// This match is part of a combine that
/// rewrites length(X - Y) to distance(X, Y)
///   (f32 (g_intrinsic length
///           (g_fsub (vXf32 X) (vXf32 Y))))
/// ->
///   (f32 (g_intrinsic distance
///           (vXf32 X) (vXf32 Y)))
///
bool SPIRVCombinerHelper::matchLengthToDistance(MachineInstr &MI) const {
  if (MI.getOpcode() != TargetOpcode::G_INTRINSIC ||
      cast<GIntrinsic>(MI).getIntrinsicID() != Intrinsic::spv_length)
    return false;

  // First operand of MI is `G_INTRINSIC` so start at operand 2.
  Register SubReg = MI.getOperand(2).getReg();
  MachineInstr *SubInstr = MRI.getVRegDef(SubReg);
  if (SubInstr->getOpcode() != TargetOpcode::G_FSUB)
    return false;

  return true;
}

void SPIRVCombinerHelper::applySPIRVDistance(MachineInstr &MI) const {
  // Extract the operands for X and Y from the match criteria.
  Register SubDestReg = MI.getOperand(2).getReg();
  MachineInstr *SubInstr = MRI.getVRegDef(SubDestReg);
  Register SubOperand1 = SubInstr->getOperand(1).getReg();
  Register SubOperand2 = SubInstr->getOperand(2).getReg();
  Register ResultReg = MI.getOperand(0).getReg();

  Builder.setInstrAndDebugLoc(MI);
  Builder.buildIntrinsic(Intrinsic::spv_distance, ResultReg)
      .addUse(SubOperand1)
      .addUse(SubOperand2);

  MI.eraseFromParent();
}

/// This match is part of a combine that
/// rewrites select(fcmp(dot(I, Ng), 0), N, -N) to faceforward(N, I, Ng)
///   (vXf32 (g_select
///             (g_fcmp
///                (g_intrinsic dot(vXf32 I) (vXf32 Ng)
///                 0)
///             (vXf32 N)
///             (vXf32 g_fneg (vXf32 N))))
/// ->
///   (vXf32 (g_intrinsic faceforward
///             (vXf32 N) (vXf32 I) (vXf32 Ng)))
///
/// This only works for Vulkan shader targets.
///
bool SPIRVCombinerHelper::matchSelectToFaceForward(MachineInstr &MI) const {
  if (!STI.isShader())
    return false;

  // Match overall select pattern.
  Register CondReg, TrueReg, FalseReg;
  if (!mi_match(MI.getOperand(0).getReg(), MRI,
                m_GISelect(m_Reg(CondReg), m_Reg(TrueReg), m_Reg(FalseReg))))
    return false;

  // Match the FCMP condition.
  Register DotReg, CondZeroReg;
  CmpInst::Predicate Pred;
  if (!mi_match(CondReg, MRI,
                m_GFCmp(m_Pred(Pred), m_Reg(DotReg), m_Reg(CondZeroReg))) ||
      !(Pred == CmpInst::FCMP_OLT || Pred == CmpInst::FCMP_ULT)) {
    if (!(Pred == CmpInst::FCMP_OGT || Pred == CmpInst::FCMP_UGT))
      return false;
    std::swap(DotReg, CondZeroReg);
  }

  // Check if FCMP is a comparison between a dot product and 0.
  MachineInstr *DotInstr = MRI.getVRegDef(DotReg);
  if (DotInstr->getOpcode() != TargetOpcode::G_INTRINSIC ||
      cast<GIntrinsic>(DotInstr)->getIntrinsicID() != Intrinsic::spv_fdot) {
    Register DotOperand1, DotOperand2;
    // Check for scalar dot product.
    if (!mi_match(DotReg, MRI,
                  m_GFMul(m_Reg(DotOperand1), m_Reg(DotOperand2))) ||
        !MRI.getType(DotOperand1).isScalar() ||
        !MRI.getType(DotOperand2).isScalar())
      return false;
  }

  const ConstantFP *ZeroVal;
  if (!mi_match(CondZeroReg, MRI, m_GFCst(ZeroVal)) || !ZeroVal->isZero())
    return false;

  // Check if select's false operand is the negation of the true operand.
  auto AreNegatedConstantsOrSplats = [&](Register TrueReg, Register FalseReg) {
    std::optional<FPValueAndVReg> TrueVal, FalseVal;
    if (!mi_match(TrueReg, MRI, m_GFCstOrSplat(TrueVal)) ||
        !mi_match(FalseReg, MRI, m_GFCstOrSplat(FalseVal)))
      return false;
    APFloat TrueValNegated = TrueVal->Value;
    TrueValNegated.changeSign();
    return FalseVal->Value.compare(TrueValNegated) == APFloat::cmpEqual;
  };

  if (!mi_match(TrueReg, MRI, m_GFNeg(m_SpecificReg(FalseReg))) &&
      !mi_match(FalseReg, MRI, m_GFNeg(m_SpecificReg(TrueReg)))) {
    std::optional<FPValueAndVReg> MulConstant;
    MachineInstr *TrueInstr = MRI.getVRegDef(TrueReg);
    MachineInstr *FalseInstr = MRI.getVRegDef(FalseReg);
    if (TrueInstr->getOpcode() == TargetOpcode::G_BUILD_VECTOR &&
        FalseInstr->getOpcode() == TargetOpcode::G_BUILD_VECTOR &&
        TrueInstr->getNumOperands() == FalseInstr->getNumOperands()) {
      for (unsigned I = 1; I < TrueInstr->getNumOperands(); ++I)
        if (!AreNegatedConstantsOrSplats(TrueInstr->getOperand(I).getReg(),
                                         FalseInstr->getOperand(I).getReg()))
          return false;
    } else if (mi_match(TrueReg, MRI,
                        m_GFMul(m_SpecificReg(FalseReg),
                                m_GFCstOrSplat(MulConstant))) ||
               mi_match(FalseReg, MRI,
                        m_GFMul(m_SpecificReg(TrueReg),
                                m_GFCstOrSplat(MulConstant))) ||
               mi_match(TrueReg, MRI,
                        m_GFMul(m_GFCstOrSplat(MulConstant),
                                m_SpecificReg(FalseReg))) ||
               mi_match(FalseReg, MRI,
                        m_GFMul(m_GFCstOrSplat(MulConstant),
                                m_SpecificReg(TrueReg)))) {
      if (!MulConstant || !MulConstant->Value.isExactlyValue(-1.0))
        return false;
    } else if (!AreNegatedConstantsOrSplats(TrueReg, FalseReg))
      return false;
  }

  return true;
}

void SPIRVCombinerHelper::applySPIRVFaceForward(MachineInstr &MI) const {
  // Extract the operands for N, I, and Ng from the match criteria.
  Register CondReg = MI.getOperand(1).getReg();
  MachineInstr *CondInstr = MRI.getVRegDef(CondReg);
  Register DotReg = CondInstr->getOperand(2).getReg();
  CmpInst::Predicate Pred = cast<GFCmp>(CondInstr)->getCond();
  if (Pred == CmpInst::FCMP_OGT || Pred == CmpInst::FCMP_UGT)
    DotReg = CondInstr->getOperand(3).getReg();
  MachineInstr *DotInstr = MRI.getVRegDef(DotReg);
  Register DotOperand1, DotOperand2;
  if (DotInstr->getOpcode() == TargetOpcode::G_FMUL) {
    DotOperand1 = DotInstr->getOperand(1).getReg();
    DotOperand2 = DotInstr->getOperand(2).getReg();
  } else {
    DotOperand1 = DotInstr->getOperand(2).getReg();
    DotOperand2 = DotInstr->getOperand(3).getReg();
  }
  Register TrueReg = MI.getOperand(2).getReg();
  Register FalseReg = MI.getOperand(3).getReg();
  MachineInstr *TrueInstr = MRI.getVRegDef(TrueReg);
  if (TrueInstr->getOpcode() == TargetOpcode::G_FNEG ||
      TrueInstr->getOpcode() == TargetOpcode::G_FMUL)
    std::swap(TrueReg, FalseReg);
  MachineInstr *FalseInstr = MRI.getVRegDef(FalseReg);

  Register ResultReg = MI.getOperand(0).getReg();
  Builder.setInstrAndDebugLoc(MI);
  Builder.buildIntrinsic(Intrinsic::spv_faceforward, ResultReg)
      .addUse(TrueReg)      // N
      .addUse(DotOperand1)  // I
      .addUse(DotOperand2); // Ng

  SPIRVGlobalRegistry *GR =
      MI.getMF()->getSubtarget<SPIRVSubtarget>().getSPIRVGlobalRegistry();
  auto RemoveAllUses = [&](Register Reg) {
    SmallVector<MachineInstr *, 4> UsesToErase;
    for (auto &UseMI : MRI.use_instructions(Reg))
      UsesToErase.push_back(&UseMI);

    // calling eraseFromParent to early invalidates the iterator.
    for (auto *MIToErase : UsesToErase)
      MIToErase->eraseFromParent();
  };

  RemoveAllUses(CondReg); // remove all uses of FCMP Result
  GR->invalidateMachineInstr(CondInstr);
  CondInstr->eraseFromParent(); // remove FCMP instruction
  RemoveAllUses(DotReg);        // remove all uses of spv_fdot/G_FMUL Result
  GR->invalidateMachineInstr(DotInstr);
  DotInstr->eraseFromParent(); // remove spv_fdot/G_FMUL instruction
  RemoveAllUses(FalseReg);
  GR->invalidateMachineInstr(FalseInstr);
  FalseInstr->eraseFromParent();
}

bool SPIRVCombinerHelper::matchMatrixTranspose(MachineInstr &MI) const {
  return MI.getOpcode() == TargetOpcode::G_INTRINSIC &&
         cast<GIntrinsic>(MI).getIntrinsicID() == Intrinsic::matrix_transpose;
}

void SPIRVCombinerHelper::applyMatrixTranspose(MachineInstr &MI) const {
  Register ResReg = MI.getOperand(0).getReg();
  Register InReg = MI.getOperand(2).getReg();
  uint32_t Rows = MI.getOperand(3).getImm();
  uint32_t Cols = MI.getOperand(4).getImm();

  Builder.setInstrAndDebugLoc(MI);

  if (Rows == 1 && Cols == 1) {
    Builder.buildCopy(ResReg, InReg);
    MI.eraseFromParent();
    return;
  }

  SmallVector<int, 16> Mask;
  for (uint32_t K = 0; K < Rows * Cols; ++K) {
    uint32_t R = K / Cols;
    uint32_t C = K % Cols;
    Mask.push_back(C * Rows + R);
  }

  Builder.buildShuffleVector(ResReg, InReg, InReg, Mask);
  MI.eraseFromParent();
}

bool SPIRVCombinerHelper::matchMatrixMultiply(MachineInstr &MI) const {
  return MI.getOpcode() == TargetOpcode::G_INTRINSIC &&
         cast<GIntrinsic>(MI).getIntrinsicID() == Intrinsic::matrix_multiply;
}

SmallVector<Register, 4>
SPIRVCombinerHelper::extractColumns(Register MatrixReg, uint32_t NumberOfCols,
                                    SPIRVType *SpvColType,
                                    SPIRVGlobalRegistry *GR) const {
  // If the matrix is a single colunm, return that single column.
  if (NumberOfCols == 1)
    return {MatrixReg};

  SmallVector<Register, 4> Cols;
  LLT ColTy = GR->getRegType(SpvColType);
  for (uint32_t J = 0; J < NumberOfCols; ++J)
    Cols.push_back(MRI.createGenericVirtualRegister(ColTy));
  Builder.buildUnmerge(Cols, MatrixReg);
  for (Register R : Cols) {
    setRegClassType(R, SpvColType, GR, &MRI, Builder.getMF());
  }
  return Cols;
}

SmallVector<Register, 4>
SPIRVCombinerHelper::extractRows(Register MatrixReg, uint32_t NumRows,
                                 uint32_t NumCols, SPIRVType *SpvRowType,
                                 SPIRVGlobalRegistry *GR) const {
  SmallVector<Register, 4> Rows;
  LLT VecTy = GR->getRegType(SpvRowType);

  // If there is only one column, then each row is a scalar that needs
  // to be extracted.
  if (NumCols == 1) {
    assert(SpvRowType->getOpcode() != SPIRV::OpTypeVector);
    for (uint32_t I = 0; I < NumRows; ++I)
      Rows.push_back(MRI.createGenericVirtualRegister(VecTy));
    Builder.buildUnmerge(Rows, MatrixReg);
    for (Register R : Rows) {
      setRegClassType(R, SpvRowType, GR, &MRI, Builder.getMF());
    }
    return Rows;
  }

  // If the matrix is a single row return that row.
  if (NumRows == 1) {
    return {MatrixReg};
  }

  for (uint32_t I = 0; I < NumRows; ++I) {
    SmallVector<int, 4> Mask;
    for (uint32_t k = 0; k < NumCols; ++k)
      Mask.push_back(k * NumRows + I);
    Rows.push_back(Builder.buildShuffleVector(VecTy, MatrixReg, MatrixReg, Mask)
                       .getReg(0));
  }
  for (Register R : Rows) {
    setRegClassType(R, SpvRowType, GR, &MRI, Builder.getMF());
  }
  return Rows;
}

Register SPIRVCombinerHelper::computeDotProduct(Register RowA, Register ColB,
                                                SPIRVType *SpvVecType,
                                                SPIRVGlobalRegistry *GR) const {
  bool IsVectorOp = SpvVecType->getOpcode() == SPIRV::OpTypeVector;
  SPIRVType *SpvScalarType = GR->getScalarOrVectorComponentType(SpvVecType);
  bool IsFloatOp = SpvScalarType->getOpcode() == SPIRV::OpTypeFloat;
  LLT VecTy = GR->getRegType(SpvVecType);

  Register DotRes;
  if (IsVectorOp) {
    LLT ScalarTy = VecTy.getElementType();
    Intrinsic::SPVIntrinsics DotIntrinsic =
        (IsFloatOp ? Intrinsic::spv_fdot : Intrinsic::spv_udot);
    DotRes = Builder.buildIntrinsic(DotIntrinsic, {ScalarTy})
                 .addUse(RowA)
                 .addUse(ColB)
                 .getReg(0);
  } else {
    if (IsFloatOp)
      DotRes = Builder.buildFMul(VecTy, RowA, ColB).getReg(0);
    else
      DotRes = Builder.buildMul(VecTy, RowA, ColB).getReg(0);
  }
  setRegClassType(DotRes, SpvScalarType, GR, &MRI, Builder.getMF());
  return DotRes;
}

SmallVector<Register, 16>
SPIRVCombinerHelper::computeDotProducts(const SmallVector<Register, 4> &RowsA,
                                        const SmallVector<Register, 4> &ColsB,
                                        SPIRVType *SpvVecType,
                                        SPIRVGlobalRegistry *GR) const {
  SmallVector<Register, 16> ResultScalars;
  for (uint32_t J = 0; J < ColsB.size(); ++J) {
    for (uint32_t I = 0; I < RowsA.size(); ++I) {
      ResultScalars.push_back(
          computeDotProduct(RowsA[I], ColsB[J], SpvVecType, GR));
    }
  }
  return ResultScalars;
}

SPIRVType *
SPIRVCombinerHelper::getDotProductVectorType(Register ResReg, uint32_t K,
                                             SPIRVGlobalRegistry *GR) const {
  // Loop over all non debug uses of ResReg
  Type *ScalarResType = nullptr;
  for (auto &UseMI : MRI.use_instructions(ResReg)) {
    if (UseMI.getOpcode() != TargetOpcode::G_INTRINSIC_W_SIDE_EFFECTS)
      continue;

    if (!isSpvIntrinsic(UseMI, Intrinsic::spv_assign_type))
      continue;

    Type *Ty = getMDOperandAsType(UseMI.getOperand(2).getMetadata(), 0);
    if (Ty->isVectorTy())
      ScalarResType = cast<VectorType>(Ty)->getElementType();
    else
      ScalarResType = Ty;
    assert(ScalarResType->isIntegerTy() || ScalarResType->isFloatingPointTy());
    break;
  }
  if (!ScalarResType)
    llvm_unreachable("Could not determine scalar result type");
  Type *VecType =
      (K > 1 ? FixedVectorType::get(ScalarResType, K) : ScalarResType);
  return GR->getOrCreateSPIRVType(VecType, Builder,
                                  SPIRV::AccessQualifier::None, false);
}

void SPIRVCombinerHelper::applyMatrixMultiply(MachineInstr &MI) const {
  Register ResReg = MI.getOperand(0).getReg();
  Register AReg = MI.getOperand(2).getReg();
  Register BReg = MI.getOperand(3).getReg();
  uint32_t NumRowsA = MI.getOperand(4).getImm();
  uint32_t NumColsA = MI.getOperand(5).getImm();
  uint32_t NumColsB = MI.getOperand(6).getImm();

  Builder.setInstrAndDebugLoc(MI);

  SPIRVGlobalRegistry *GR =
      MI.getMF()->getSubtarget<SPIRVSubtarget>().getSPIRVGlobalRegistry();

  SPIRVType *SpvVecType = getDotProductVectorType(ResReg, NumColsA, GR);
  SmallVector<Register, 4> ColsB =
      extractColumns(BReg, NumColsB, SpvVecType, GR);
  SmallVector<Register, 4> RowsA =
      extractRows(AReg, NumRowsA, NumColsA, SpvVecType, GR);
  SmallVector<Register, 16> ResultScalars =
      computeDotProducts(RowsA, ColsB, SpvVecType, GR);

  Builder.buildBuildVector(ResReg, ResultScalars);
  MI.eraseFromParent();
}
