//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "X86ISelLowering.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "gtest/gtest.h"

namespace llvm {

class X86SelectionDAGTest : public testing::Test {
protected:
  const TargetSubtargetInfo *STI;

  static void SetUpTestCase() {
    LLVMInitializeX86TargetInfo();
    LLVMInitializeX86Target();
    LLVMInitializeX86TargetMC();
  }

  void SetUp() override {
    StringRef Assembly = "define void @f() { ret void }";

    Triple TargetTriple("x86_64--");
    std::string Error;
    const Target *T = TargetRegistry::lookupTarget("", TargetTriple, Error);

    TargetOptions Options;
    TM = std::unique_ptr<TargetMachine>(T->createTargetMachine(
        TargetTriple, "x86-64-v4", "", Options, std::nullopt, std::nullopt,
        CodeGenOptLevel::Aggressive));

    SMDiagnostic SMError;
    M = parseAssemblyString(Assembly, SMError, Context);
    if (!M)
      report_fatal_error(SMError.getMessage());
    M->setDataLayout(TM->createDataLayout());

    F = M->getFunction("f");
    if (!F)
      report_fatal_error("F?");

    MachineModuleInfo MMI(TM.get());

    STI = TM->getSubtargetImpl(*F);
    MF = std::make_unique<MachineFunction>(*F, *TM, *STI, MMI.getContext(), 0);

    DAG = std::make_unique<SelectionDAG>(*TM, CodeGenOptLevel::None);
    if (!DAG)
      report_fatal_error("DAG?");
    OptimizationRemarkEmitter ORE(F);
    DAG->init(*MF, ORE, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
              MMI, nullptr);
  }

  SDValue simplifyMultipleUseDemandedBits(SDValue Op, const APInt &DemandedBits,
                                          const APInt &DemandedElts) {
    return DAG->getTargetLoweringInfo().SimplifyMultipleUseDemandedBits(
        Op, DemandedBits, DemandedElts, *DAG);
  }

  SDValue getVector(MVT VT, unsigned Reg) {
    return DAG->getRegister(Register::index2VirtReg(Reg), VT);
  }

  SDValue getExtract(SDValue Vec, SDValue Idx, MVT ResultVT) {
    return DAG->getNode(ISD::EXTRACT_VECTOR_ELT, SDLoc(), ResultVT, Vec, Idx);
  }

  LLVMContext Context;
  std::unique_ptr<TargetMachine> TM;
  std::unique_ptr<Module> M;
  Function *F;
  std::unique_ptr<MachineFunction> MF;
  std::unique_ptr<SelectionDAG> DAG;
};

TEST_F(X86SelectionDAGTest, computeKnownBits_FAND) {
  SDLoc Loc;

  auto SrcF32 = DAG->getCopyFromReg(DAG->getEntryNode(), Loc,
                                    Register::index2VirtReg(1), MVT::f32);
  auto ZeroF32 = DAG->getConstantFP(+0.0, Loc, MVT::f32);
  auto OpF32 = DAG->getNode(X86ISD::FAND, Loc, MVT::f32, ZeroF32, SrcF32);
  KnownBits KnownF32 = DAG->computeKnownBits(OpF32);
  EXPECT_TRUE(KnownF32.isZero());

  auto Src2xF64 = DAG->getCopyFromReg(DAG->getEntryNode(), Loc,
                                      Register::index2VirtReg(2), MVT::v2f64);
  auto ZeroF64 = DAG->getConstantFP(+0.0, Loc, MVT::f64);
  auto SignBitF64 = DAG->getConstantFP(-0.0, Loc, MVT::f64);
  auto LoZeroHiSign2xF64 =
      DAG->getBuildVector(MVT::v2f64, Loc, {ZeroF64, SignBitF64});
  auto Op2xF64 =
      DAG->getNode(X86ISD::FAND, Loc, MVT::v2f64, LoZeroHiSign2xF64, Src2xF64);
  KnownBits KnownAll2xF64 = DAG->computeKnownBits(Op2xF64);
  KnownBits KnownLo2xF64 = DAG->computeKnownBits(Op2xF64, APInt(2, 1));
  KnownBits KnownHi2xF64 = DAG->computeKnownBits(Op2xF64, APInt(2, 2));
  EXPECT_FALSE(KnownAll2xF64.isZero());
  EXPECT_TRUE(KnownLo2xF64.isZero());
  EXPECT_FALSE(KnownHi2xF64.isZero());
}

TEST_F(X86SelectionDAGTest, computeKnownBits_FANDN) {
  SDLoc Loc;

  auto SrcF32 = DAG->getCopyFromReg(DAG->getEntryNode(), Loc,
                                    Register::index2VirtReg(1), MVT::f32);
  auto SignBitF32 = DAG->getConstantFP(-0.0f, Loc, MVT::f32);
  auto OpF32 = DAG->getNode(X86ISD::FANDN, Loc, MVT::f32, SignBitF32, SrcF32);
  KnownBits KnownF32 = DAG->computeKnownBits(OpF32);
  EXPECT_TRUE(KnownF32.isNonNegative());

  auto Src2xF64 = DAG->getCopyFromReg(DAG->getEntryNode(), Loc,
                                      Register::index2VirtReg(2), MVT::v2f64);
  auto ZeroF64 = DAG->getConstantFP(+0.0f, Loc, MVT::f64);
  auto SignBitF64 = DAG->getConstantFP(-0.0f, Loc, MVT::f64);
  auto HiSign2xF64 =
      DAG->getBuildVector(MVT::v2f64, Loc, {ZeroF64, SignBitF64});
  auto Op2xF64 =
      DAG->getNode(X86ISD::FANDN, Loc, MVT::v2f64, HiSign2xF64, Src2xF64);
  KnownBits KnownAll2xF64 = DAG->computeKnownBits(Op2xF64);
  KnownBits KnownLo2xF64 = DAG->computeKnownBits(Op2xF64, APInt(2, 1));
  KnownBits KnownHi2xF64 = DAG->computeKnownBits(Op2xF64, APInt(2, 2));
  EXPECT_FALSE(KnownAll2xF64.isNonNegative());
  EXPECT_FALSE(KnownLo2xF64.isNonNegative());
  EXPECT_TRUE(KnownHi2xF64.isNonNegative());
}

TEST_F(X86SelectionDAGTest, computeKnownBits_FXOR) {
  SDLoc Loc;

  auto SignBitF32 = DAG->getConstantFP(-0.0f, Loc, MVT::f32);
  auto OpF32 =
      DAG->getNode(X86ISD::FXOR, Loc, MVT::f32, SignBitF32, SignBitF32);
  KnownBits KnownF32 = DAG->computeKnownBits(OpF32);
  EXPECT_TRUE(KnownF32.isZero());

  auto ZeroF64 = DAG->getConstantFP(+0.0, Loc, MVT::f64);
  auto SignBitF64 = DAG->getConstantFP(-0.0, Loc, MVT::f64);
  auto NegNeg2xF64 =
      DAG->getBuildVector(MVT::v2f64, Loc, {SignBitF64, SignBitF64});
  auto NegZero2xF64 =
      DAG->getBuildVector(MVT::v2f64, Loc, {SignBitF64, ZeroF64});
  auto Op2xF64 =
      DAG->getNode(X86ISD::FXOR, Loc, MVT::v2f64, NegNeg2xF64, NegZero2xF64);
  KnownBits KnownAll2xF64 = DAG->computeKnownBits(Op2xF64);
  KnownBits KnownLo2xF64 = DAG->computeKnownBits(Op2xF64, APInt(2, 1));
  KnownBits KnownHi2xF64 = DAG->computeKnownBits(Op2xF64, APInt(2, 2));
  EXPECT_FALSE(KnownAll2xF64.isNonNegative());
  EXPECT_TRUE(KnownLo2xF64.isZero());
  EXPECT_FALSE(KnownHi2xF64.isNonNegative());
}

TEST_F(X86SelectionDAGTest, MultipleUseDemandedBitsScalarToVectorDirect) {
  SDLoc DL;
  SDValue Vec = getVector(MVT::v2i64, 1);
  SDValue Extract =
      getExtract(Vec, DAG->getConstant(0, DL, MVT::i64), MVT::i64);
  SDValue Op = DAG->getNode(ISD::SCALAR_TO_VECTOR, DL, MVT::v4i32, Extract);

  SDValue Result = simplifyMultipleUseDemandedBits(Op, APInt::getAllOnes(32),
                                                   APInt::getAllOnes(4));
  ASSERT_TRUE(Result);
  EXPECT_EQ(Result.getOpcode(), ISD::BITCAST);
  EXPECT_EQ(Result.getOperand(0), Vec);
}

TEST_F(X86SelectionDAGTest,
       MultipleUseDemandedBitsScalarToVectorTruncateShared) {
  SDLoc DL;
  SDValue Vec = getVector(MVT::v2i64, 1);
  SDValue Extract =
      getExtract(Vec, DAG->getConstant(0, DL, MVT::i64), MVT::i64);
  SDValue Trunc = DAG->getNode(ISD::TRUNCATE, DL, MVT::i32, Extract);
  SDValue Op = DAG->getNode(ISD::SCALAR_TO_VECTOR, DL, MVT::v4i32, Trunc);
  SDValue OtherUse = DAG->getNode(ISD::XOR, DL, MVT::i32, Trunc,
                                  DAG->getConstant(1, DL, MVT::i32));
  ASSERT_FALSE(Trunc.hasOneUse());

  SDValue Result = simplifyMultipleUseDemandedBits(Op, APInt::getAllOnes(32),
                                                   APInt::getAllOnes(4));
  ASSERT_TRUE(Result);
  EXPECT_EQ(Result.getOpcode(), ISD::BITCAST);
  EXPECT_EQ(Result.getOperand(0), Vec);
  EXPECT_EQ(OtherUse.getOperand(0), Trunc);
}

TEST_F(X86SelectionDAGTest, MultipleUseDemandedBitsScalarToVectorGuards) {
  SDLoc DL;
  SDValue Zero = DAG->getConstant(0, DL, MVT::i64);
  SDValue One = DAG->getConstant(1, DL, MVT::i64);
  SDValue Variable = DAG->getRegister(Register::index2VirtReg(1), MVT::i64);
  SDValue Vec = getVector(MVT::v4i32, 2);
  SDValue ExtractOne = getExtract(Vec, One, MVT::i32);
  SDValue ExtractVariable = getExtract(Vec, Variable, MVT::i32);
  SDValue ExtractTypeMismatch = getExtract(Vec, Zero, MVT::i64);

  SDValue SmallVec = getVector(MVT::v2i32, 3);
  SDValue SmallExtract = getExtract(SmallVec, Zero, MVT::i32);
  SDValue WideVec = getVector(MVT::v2i64, 4);
  SDValue WideExtract = getExtract(WideVec, Zero, MVT::i64);
  SDValue Trunc = DAG->getNode(ISD::TRUNCATE, DL, MVT::i32, WideExtract);

  auto ScalarToVector = [&](MVT VT, SDValue Scalar) {
    return DAG->getNode(ISD::SCALAR_TO_VECTOR, DL, VT, Scalar);
  };
  auto IsSimplified = [&](SDValue Op) {
    EVT VT = Op.getValueType();
    return bool(simplifyMultipleUseDemandedBits(
        Op, APInt::getAllOnes(VT.getScalarSizeInBits()),
        APInt::getAllOnes(VT.getVectorNumElements())));
  };

  EXPECT_FALSE(IsSimplified(ScalarToVector(MVT::v4i32, ExtractOne)));
  EXPECT_FALSE(IsSimplified(ScalarToVector(MVT::v4i32, ExtractVariable)));
  EXPECT_FALSE(IsSimplified(ScalarToVector(MVT::v4i32, ExtractTypeMismatch)));
  EXPECT_FALSE(IsSimplified(ScalarToVector(MVT::v4i32, SmallExtract)));
  EXPECT_FALSE(IsSimplified(ScalarToVector(MVT::v8i16, Trunc)));
}

TEST_F(X86SelectionDAGTest, MultipleUseDemandedBitsScalarToVectorZeroDemand) {
  SDLoc DL;
  SDValue Vec = getVector(MVT::v4i32, 1);
  SDValue Extract =
      getExtract(Vec, DAG->getConstant(0, DL, MVT::i64), MVT::i32);
  SDValue Op = DAG->getNode(ISD::SCALAR_TO_VECTOR, DL, MVT::v4i32, Extract);

  SDValue Result = simplifyMultipleUseDemandedBits(Op, APInt::getZero(32),
                                                   APInt::getAllOnes(4));
  ASSERT_TRUE(Result);
  EXPECT_TRUE(Result.isUndef());
}

} // end namespace llvm
