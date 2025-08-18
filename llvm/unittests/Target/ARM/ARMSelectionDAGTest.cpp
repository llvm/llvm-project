//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ARMISelLowering.h"
#include "MCTargetDesc/ARMAddressingModes.h"
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

class ARMSelectionDAGTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    LLVMInitializeARMTargetInfo();
    LLVMInitializeARMTarget();
    LLVMInitializeARMTargetMC();
  }

  void SetUp() override {
    StringRef Assembly = "define void @f() { ret void }";

    Triple TargetTriple("armv7-unknown-none-eabi");

    std::string Error;
    const Target *T = TargetRegistry::lookupTarget("", TargetTriple, Error);

    TargetOptions Options;
    TM = std::unique_ptr<TargetMachine>(
        T->createTargetMachine(TargetTriple,
                               /*CPU*/ "cortex-a9",
                               /*Features*/ "+neon", Options, std::nullopt,
                               std::nullopt, CodeGenOptLevel::Aggressive));

    SMDiagnostic SMError;
    M = parseAssemblyString(Assembly, SMError, Context);
    if (!M)
      report_fatal_error(SMError.getMessage());
    M->setDataLayout(TM->createDataLayout());

    F = M->getFunction("f");
    if (!F)
      report_fatal_error("Function 'f' not found");

    MachineModuleInfo MMI(TM.get());

    MF = std::make_unique<MachineFunction>(*F, *TM, *TM->getSubtargetImpl(*F),
                                           MMI.getContext(), /*FunctionNum*/ 0);

    DAG = std::make_unique<SelectionDAG>(*TM, CodeGenOptLevel::None);
    if (!DAG)
      report_fatal_error("SelectionDAG allocation failed");

    OptimizationRemarkEmitter ORE(F);
    DAG->init(*MF, ORE, /*LibInfo*/ nullptr, /*AA*/ nullptr,
              /*AC*/ nullptr, /*MDT*/ nullptr, /*MSDT*/ nullptr, MMI, nullptr);
  }

  TargetLoweringBase::LegalizeTypeAction getTypeAction(EVT VT) {
    return DAG->getTargetLoweringInfo().getTypeAction(Context, VT);
  }

  EVT getTypeToTransformTo(EVT VT) {
    return DAG->getTargetLoweringInfo().getTypeToTransformTo(Context, VT);
  }

  LLVMContext Context;
  std::unique_ptr<TargetMachine> TM;
  std::unique_ptr<Module> M;
  Function *F = nullptr;
  std::unique_ptr<MachineFunction> MF;
  std::unique_ptr<SelectionDAG> DAG;
};

TEST_F(ARMSelectionDAGTest, computeKnownBits_VORRIMM) {
  SDLoc DL;
  EVT VT = EVT::getVectorVT(Context, EVT::getIntegerVT(Context, 8), 2);
  SDValue LHS = DAG->getRegister(0, VT);

  unsigned Encoded = 0xAA;
  SDValue EncSD = DAG->getTargetConstant(ARM_AM::createVMOVModImm(0xe, Encoded),
                                         DL, MVT::i32);
  SDValue Op = DAG->getNode(ARMISD::VORRIMM, DL, VT, LHS, EncSD);

  // LHS     = ????????
  // Encoded = 10101010
  //  =>
  // Known.One  = 10101010 (0xAA)
  // Known.Zero = 00000000 (0x0)
  APInt DemandedElts = APInt::getAllOnes(2);
  KnownBits Known = DAG->computeKnownBits(Op, DemandedElts);
  EXPECT_EQ(Known.One, APInt(8, 0xAA));
  EXPECT_EQ(Known.Zero, APInt(8, 0x0));
}

TEST_F(ARMSelectionDAGTest, computeKnownBits_VBICIMM) {
  SDLoc DL;
  EVT VT = EVT::getVectorVT(Context, EVT::getIntegerVT(Context, 8), 2);

  SDValue LHS = DAG->getConstant(APInt(8, 0xCC), DL, VT);

  unsigned Encoded = 0xAA;
  SDValue EncSD = DAG->getTargetConstant(ARM_AM::createVMOVModImm(0xe, Encoded),
                                         DL, MVT::i32);
  SDValue Op = DAG->getNode(ARMISD::VBICIMM, DL, VT, LHS, EncSD);

  // LHS     = 11001100
  // Encoded = 10101010
  //  =>
  // Known.One  = 01000100 (0x44)
  // Known.Zero = 10111011 (0xBB)
  APInt DemandedElts = APInt::getAllOnes(2);
  KnownBits Known = DAG->computeKnownBits(Op, DemandedElts);
  EXPECT_EQ(Known.One, APInt(8, 0x44));
  EXPECT_EQ(Known.Zero, APInt(8, 0xBB));
}

} // end namespace llvm
