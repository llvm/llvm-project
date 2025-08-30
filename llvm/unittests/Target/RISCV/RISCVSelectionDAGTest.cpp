//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RISCVISelLowering.h"
#include "RISCVSelectionDAGInfo.h"
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
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "gtest/gtest.h"

namespace llvm {

class RISCVSelectionDAGTest : public testing::Test {

protected:
  static void SetUpTestCase() {
    LLVMInitializeRISCVTargetInfo();
    LLVMInitializeRISCVTarget();
    LLVMInitializeRISCVTargetMC();
  }

  void SetUp() override {
    StringRef Assembly = "define void @f() { ret void }";

    Triple TargetTriple("riscv64", "unknown", "linux");

    std::string Error;
    const Target *T = TargetRegistry::lookupTarget("", TargetTriple, Error);

    TargetOptions Options;
    TM = std::unique_ptr<TargetMachine>(T->createTargetMachine(
        TargetTriple, "generic", "", Options, std::nullopt, std::nullopt,
        CodeGenOptLevel::Default));

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

  LLVMContext Context;
  std::unique_ptr<TargetMachine> TM;
  std::unique_ptr<Module> M;
  Function *F = nullptr;
  std::unique_ptr<MachineFunction> MF;
  std::unique_ptr<SelectionDAG> DAG;
};

/// SRLW: Logical Shift Right
TEST_F(RISCVSelectionDAGTest, computeKnownBits_SRLW) {
  // Following DAG is created from this IR snippet:
  //
  // define i64 @f(i32 %x, i32 %y) {
  //  %a = and i32 %x, 2147483647  ; zeros the MSB for %x
  //  %b = lshr i32 %a, %y
  //  %c = zext i32 %b to i64 ; makes the most significant 32 bits 0
  //  ret i64 %c
  // }
  SDLoc Loc;
  auto IntVT = EVT::getIntegerVT(Context, 32);
  auto Int64VT = EVT::getIntegerVT(Context, 64);
  auto Px = DAG->getRegister(0, IntVT);
  auto Py = DAG->getConstant(2147483647, Loc, IntVT);
  auto N1 = DAG->getNode(ISD::AND, Loc, IntVT, Px, Py);
  auto Qx = DAG->getRegister(0, Int64VT);
  auto N2 = DAG->getNode(RISCVISD::SRLW, Loc, Int64VT, N1, Qx);
  auto N3 = DAG->getNode(ISD::ZERO_EXTEND, Loc, Int64VT, N2);
  // N1 = 0???????????????????????????????
  // N2 = 0???????????????????????????????
  // N3 = 000000000000000000000000000000000???????????????????????????????
  // After zero extend, we expect 33 most significant zeros to be known:
  // 32 from sign extension and 1 from AND operation
  KnownBits Known = DAG->computeKnownBits(N3);
  EXPECT_EQ(Known.Zero, APInt(64, -2147483648));
  EXPECT_EQ(Known.One, APInt(64, 0));
}

} // end namespace llvm
