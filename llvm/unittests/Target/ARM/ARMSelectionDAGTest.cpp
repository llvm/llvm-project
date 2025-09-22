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

/// VORR (immediate): per-lane OR with 32-bit elements.
/// cmode=0x0 puts imm8 in byte0 => per-lane constant = 0x000000AA.
TEST_F(ARMSelectionDAGTest, computeKnownBits_VORRIMM) {
  SDLoc DL;
  EVT VT = MVT::v4i32;
  SDValue LHS = DAG->getRegister(0, VT);

  SDValue EncSD =
      DAG->getTargetConstant(ARM_AM::createVMOVModImm(0x0, 0xAA), DL, MVT::i32);
  SDValue Op = DAG->getNode(ARMISD::VORRIMM, DL, VT, LHS, EncSD);

  // LHS(per-lane)     = ???????? ???????? ???????? ????????
  // Encoded(per-lane) = 00000000 00000000 00000000 10101010  (0x000000AA)
  //  =>
  // Known.One  = 00000000 00000000 00000000 10101010  (0x000000AA)
  // Known.Zero = 00000000 00000000 00000000 00000000  (0x00000000)
  APInt DemandedElts = APInt::getAllOnes(4);
  KnownBits Known = DAG->computeKnownBits(Op, DemandedElts);
  EXPECT_EQ(Known.One, APInt(32, 0xAA));
  EXPECT_EQ(Known.Zero, APInt(32, 0x0));

  // LHS(per-lane)     = 00000000 00000000 00000000 00000000  (0x00000000)
  // Encoded(per-lane) = 00000000 00000000 00000000 10101010  (0x000000AA)
  //  =>
  // Known.One  = 00000000 00000000 00000000 10101010  (0x000000AA)
  // Known.Zero = 11111111 11111111 11111111 01010101  (0x00000000)
  SDValue Zero = DAG->getConstant(0, DL, MVT::i32);
  SDValue ZeroVec = DAG->getSplatBuildVector(VT, DL, Zero);
  Op = DAG->getNode(ARMISD::VORRIMM, DL, VT, ZeroVec, EncSD);
  SDValue FrVORRIMM = DAG->getFreeze(Op);
  Known = DAG->computeKnownBits(FrVORRIMM);
  EXPECT_EQ(Known.One, APInt(32, 0xAA));
  EXPECT_EQ(Known.Zero, APInt(32, 0xFFFFFF55));
}

/// VBIC (immediate): x & ~imm with 32-bit elements.
/// LHS(per-lane)=0xFFFFFFFF; imm per-lane = 0x000000AA => result = 0xFFFFFF55
TEST_F(ARMSelectionDAGTest, computeKnownBits_VBICIMM) {
  SDLoc DL;
  EVT VT = MVT::v4i32;

  SDValue LHS = DAG->getConstant(APInt(32, 0xFFFFFFFF), DL, VT);

  SDValue EncSD =
      DAG->getTargetConstant(ARM_AM::createVMOVModImm(0x0, 0xAA), DL, MVT::i32);
  SDValue Op = DAG->getNode(ARMISD::VBICIMM, DL, VT, LHS, EncSD);

  // LHS(per-lane)     = 11111111 11111111 11111111 11111111  (0xFFFFFFFF)
  // Encoded(per-lane) = 00000000 00000000 00000000 10101010  (0x000000AA)
  //  =>
  // Known.One  = 11111111 11111111 11111111 01010101  (0xFFFFFF55)
  // Known.Zero = 00000000 00000000 00000000 10101010  (0x000000AA)
  APInt DemandedElts = APInt::getAllOnes(4);
  KnownBits Known = DAG->computeKnownBits(Op, DemandedElts);
  EXPECT_EQ(Known.One, APInt(32, 0xFFFFFF55));
  EXPECT_EQ(Known.Zero, APInt(32, 0x000000AA));

  SDValue FrVBICIMM = DAG->getFreeze(Op);
  Known = DAG->computeKnownBits(FrVBICIMM);
  EXPECT_EQ(Known.One, APInt(32, 0xFFFFFF55));
  EXPECT_EQ(Known.Zero, APInt(32, 0x000000AA));
}

/// VORR (immediate): per-lane OR with 32-bit elements.
/// Encoded = 0x2AA (cmode=0x2, imm8=0xAA) => per-lane constant = 0x0000AA00.
TEST_F(ARMSelectionDAGTest, computeKnownBits_VORRIMM_cmode2) {
  SDLoc DL;
  EVT VT = MVT::v4i32;
  SDValue LHS = DAG->getRegister(0, VT);

  // Use the exact encoded immediate the reviewer asked for.
  SDValue EncSD =
      DAG->getTargetConstant(ARM_AM::createVMOVModImm(0x2, 0xAA), DL, MVT::i32);
  SDValue Op = DAG->getNode(ARMISD::VORRIMM, DL, VT, LHS, EncSD);

  // LHS (per-lane)     = ???????? ???????? ???????? ????????
  // Encoded (per-lane) = 00000000 00000000 10101010 00000000  (0x0000AA00)
  //  =>
  // Known.One          = 00000000 00000000 10101010 00000000  (0x0000AA00)
  // Known.Zero         = 00000000 00000000 00000000 00000000  (0x00000000)
  APInt DemandedElts = APInt::getAllOnes(4);
  KnownBits Known = DAG->computeKnownBits(Op, DemandedElts);
  EXPECT_EQ(Known.One, APInt(32, 0x0000AA00));
  EXPECT_EQ(Known.Zero, APInt(32, 0x00000000));
}

/// VBIC (immediate) with constant-all-ones LHS:
/// Encoded = 0x2AA => per-lane constant = 0x0000AA00; VBIC = A & ~Imm.
TEST_F(ARMSelectionDAGTest, computeKnownBits_VBICIMM_cmode2_lhs_ones) {
  SDLoc DL;
  EVT VT = MVT::v4i32;

  SDValue LHS = DAG->getConstant(APInt(32, 0xFFFFFFFF), DL, VT);
  SDValue EncSD =
      DAG->getTargetConstant(ARM_AM::createVMOVModImm(0x2, 0xAA), DL, MVT::i32);
  SDValue Op = DAG->getNode(ARMISD::VBICIMM, DL, VT, LHS, EncSD);

  // LHS (per-lane)     = 11111111 11111111 11111111 11111111
  // Encoded (per-lane) = 00000000 00000000 10101010 00000000  (0x0000AA00)
  //  =>
  // Known.One          = 11111111 11111111 01010101 11111111  (0xFFFF55FF)
  // Known.Zero         = 00000000 00000000 10101010 00000000  (0x0000AA00)
  APInt DemandedElts = APInt::getAllOnes(4);
  KnownBits Known = DAG->computeKnownBits(Op, DemandedElts);
  EXPECT_EQ(Known.One, APInt(32, 0xFFFF55FF));
  EXPECT_EQ(Known.Zero, APInt(32, 0x0000AA00));
}

} // end namespace llvm
