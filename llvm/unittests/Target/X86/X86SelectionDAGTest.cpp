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

  LLVMContext Context;
  std::unique_ptr<TargetMachine> TM;
  std::unique_ptr<Module> M;
  Function *F;
  std::unique_ptr<MachineFunction> MF;
  std::unique_ptr<SelectionDAG> DAG;
};

TEST_F(X86SelectionDAGTest, computeKnownBits_FAND) {
  SDLoc Loc;

  auto SrcF32 = DAG->getCopyFromReg(DAG->getEntryNode(), Loc, 1, MVT::f32);
  auto ZeroF32 = DAG->getConstantFP(+0.0, Loc, MVT::f32);
  auto OpF32 = DAG->getNode(X86ISD::FAND, Loc, MVT::f32, ZeroF32, SrcF32);
  KnownBits KnownF32 = DAG->computeKnownBits(OpF32);
  EXPECT_TRUE(KnownF32.isZero());

  auto Src2xF64 = DAG->getCopyFromReg(DAG->getEntryNode(), Loc, 1, MVT::v2f64);
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

  auto SrcF32 = DAG->getCopyFromReg(DAG->getEntryNode(), Loc, 1, MVT::f32);
  auto SignBitF32 = DAG->getConstantFP(-0.0f, Loc, MVT::f32);
  auto OpF32 = DAG->getNode(X86ISD::FANDN, Loc, MVT::f32, SignBitF32, SrcF32);
  KnownBits KnownF32 = DAG->computeKnownBits(OpF32);
  EXPECT_TRUE(KnownF32.isNonNegative());

  auto Src2xF64 = DAG->getCopyFromReg(DAG->getEntryNode(), Loc, 1, MVT::v2f64);
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

} // end namespace llvm
