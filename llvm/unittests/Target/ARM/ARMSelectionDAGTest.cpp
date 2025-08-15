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
  EVT VT = EVT::getVectorVT(Context, EVT::getIntegerVT(Context, 32), 2);

  SDValue LHS = DAG->getConstant(0, DL, VT);

  unsigned Encoded = 0xF0;
  SDValue EncSD = DAG->getTargetConstant(Encoded, DL, MVT::i32);
  SDValue Op = DAG->getNode(ARMISD::VORRIMM, DL, VT, LHS, EncSD);

  APInt DemandedElts = APInt::getAllOnes(2);

  KnownBits Known = DAG->computeKnownBits(Op, DemandedElts);

  unsigned ElemBits = 32;
  uint64_t Decoded = ARM_AM::decodeVMOVModImm(Encoded, ElemBits);
  APInt Imm(32, Decoded);

  EXPECT_EQ(Known.One, Imm);
  EXPECT_EQ(Known.Zero, ~Imm);
}

TEST_F(ARMSelectionDAGTest, computeKnownBits_VBICIMM) {
  SDLoc DL;
  EVT VT = EVT::getVectorVT(Context, EVT::getIntegerVT(Context, 32), 2);

  APInt AllOnes = APInt::getAllOnes(32);
  SDValue LHS = DAG->getConstant(AllOnes, DL, VT);

  unsigned Encoded = 0xF0;
  SDValue EncSD = DAG->getTargetConstant(Encoded, DL, MVT::i32);
  SDValue Op = DAG->getNode(ARMISD::VBICIMM, DL, VT, LHS, EncSD);

  APInt DemandedElts = APInt::getAllOnes(2);

  KnownBits Known = DAG->computeKnownBits(Op, DemandedElts);

  unsigned ElemBits = 32;
  uint64_t Decoded = ARM_AM::decodeVMOVModImm(Encoded, ElemBits);
  APInt Imm(32, Decoded);

  EXPECT_EQ(Known.One, ~Imm);
  EXPECT_EQ(Known.Zero, Imm);
}

} // end namespace llvm
