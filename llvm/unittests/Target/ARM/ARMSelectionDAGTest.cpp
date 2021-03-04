//===- llvm/unittest/Target/ARM/ARMSelectionDAGTest.cpp -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ARMISelLowering.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "gtest/gtest.h"

namespace llvm {

class ARMSelectionDAGTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    InitializeAllTargets();
    InitializeAllTargetMCs();
  }

  void SetUp() override {
    StringRef Assembly = "define void @f() { ret void }";

    Triple TargetTriple("thumbv8.1m.main-none-eabi");
    std::string Error;
    const Target *T = TargetRegistry::lookupTarget("", TargetTriple, Error);
    if (!T)
      return;

    TargetOptions Options;
    TM = std::unique_ptr<LLVMTargetMachine>(static_cast<LLVMTargetMachine *>(
        T->createTargetMachine("ARM", "", "+mve.fp", Options, None, None,
                               CodeGenOpt::Aggressive)));
    if (!TM)
      return;

    SMDiagnostic SMError;
    M = parseAssemblyString(Assembly, SMError, Context);
    if (!M)
      report_fatal_error(SMError.getMessage());
    M->setDataLayout(TM->createDataLayout());

    F = M->getFunction("f");
    if (!F)
      report_fatal_error("F?");

    MachineModuleInfo MMI(TM.get());

    MF = std::make_unique<MachineFunction>(*F, *TM, *TM->getSubtargetImpl(*F),
                                           0, MMI);

    DAG = std::make_unique<SelectionDAG>(*TM, CodeGenOpt::None);
    if (!DAG)
      report_fatal_error("DAG?");
    OptimizationRemarkEmitter ORE(F);
    DAG->init(*MF, ORE, nullptr, nullptr, nullptr, nullptr, nullptr);
  }

  TargetLoweringBase::LegalizeTypeAction getTypeAction(EVT VT) {
    return DAG->getTargetLoweringInfo().getTypeAction(Context, VT);
  }

  EVT getTypeToTransformTo(EVT VT) {
    return DAG->getTargetLoweringInfo().getTypeToTransformTo(Context, VT);
  }

  LLVMContext Context;
  std::unique_ptr<LLVMTargetMachine> TM;
  std::unique_ptr<Module> M;
  Function *F;
  std::unique_ptr<MachineFunction> MF;
  std::unique_ptr<SelectionDAG> DAG;
};

TEST_F(ARMSelectionDAGTest, computeKnownBits_CSINC) {
  if (!TM)
    return;
  SDLoc DL;
  SDValue Zero = DAG->getConstant(0, DL, MVT::i32);
  SDValue One = DAG->getConstant(1, DL, MVT::i32);
  SDValue ARMcc = DAG->getConstant(8, DL, MVT::i32);
  SDValue Cmp = DAG->getNode(ARMISD::CMP, DL, MVT::Glue, Zero, One);

  SDValue Op0 =
      DAG->getNode(ARMISD::CSINC, DL, MVT::i32, Zero, Zero, ARMcc, Cmp);
  KnownBits Known = DAG->computeKnownBits(Op0);
  EXPECT_EQ(Known.Zero, 0xfffffffe);
  EXPECT_EQ(Known.One, 0x0);

  SDValue Op1 = DAG->getNode(ARMISD::CSINC, DL, MVT::i32, One, One, ARMcc, Cmp);
  Known = DAG->computeKnownBits(Op1);
  EXPECT_EQ(Known.Zero, 0xfffffffc);
  EXPECT_EQ(Known.One, 0x0);
}

TEST_F(ARMSelectionDAGTest, computeKnownBits_CSINV) {
  if (!TM)
    return;
  SDLoc DL;
  SDValue Zero = DAG->getConstant(0, DL, MVT::i32);
  SDValue One = DAG->getConstant(1, DL, MVT::i32);
  SDValue ARMcc = DAG->getConstant(8, DL, MVT::i32);
  SDValue Cmp = DAG->getNode(ARMISD::CMP, DL, MVT::Glue, Zero, One);

  SDValue Op0 =
      DAG->getNode(ARMISD::CSINV, DL, MVT::i32, Zero, Zero, ARMcc, Cmp);
  KnownBits Known = DAG->computeKnownBits(Op0);
  EXPECT_EQ(Known.Zero, 0x0);
  EXPECT_EQ(Known.One, 0x0);

  SDValue Op1 =
      DAG->getNode(ARMISD::CSINV, DL, MVT::i32, Zero, One, ARMcc, Cmp);
  Known = DAG->computeKnownBits(Op1);
  EXPECT_EQ(Known.Zero, 0x1);
  EXPECT_EQ(Known.One, 0x0);
}

TEST_F(ARMSelectionDAGTest, computeKnownBits_CSNEG) {
  if (!TM)
    return;
  SDLoc DL;
  SDValue Zero = DAG->getConstant(0, DL, MVT::i32);
  SDValue One = DAG->getConstant(1, DL, MVT::i32);
  SDValue ARMcc = DAG->getConstant(8, DL, MVT::i32);
  SDValue Cmp = DAG->getNode(ARMISD::CMP, DL, MVT::Glue, Zero, One);

  SDValue Op0 =
      DAG->getNode(ARMISD::CSNEG, DL, MVT::i32, Zero, Zero, ARMcc, Cmp);
  KnownBits Known = DAG->computeKnownBits(Op0);
  EXPECT_EQ(Known.Zero, 0xffffffff);
  EXPECT_EQ(Known.One, 0x0);

  SDValue Op1 =
      DAG->getNode(ARMISD::CSNEG, DL, MVT::i32, One, Zero, ARMcc, Cmp);
  Known = DAG->computeKnownBits(Op1);
  EXPECT_EQ(Known.Zero, 0xfffffffe);
  EXPECT_EQ(Known.One, 0x0);
}

} // end namespace llvm
