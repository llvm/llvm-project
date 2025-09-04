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
  // Given the following IR snippet:
  //  define i64 @f(i32 %x, i32 %y) {
  //   %a = and i32 %x, 2147483647  ; zeros the MSB for %x
  //   %b = lshr i32 %a, %y
  //   %c = zext i32 %b to i64 ; makes the most significant 32 bits 0
  //   ret i64 %c
  //  }
  // The Optimized SelectionDAG as show by llc -mtriple="riscv64"
  // -debug-only=isel-dump is:
  //      t0: ch,glue = EntryToken
  //          t2: i64,ch = CopyFromReg t0, Register:i64 %0
  //        t18: i64 = and t2, Constant:i64<2147483647>
  //        t4: i64,ch = CopyFromReg t0, Register:i64 %1
  //      t20: i64 = RISCVISD::SRLW t18, t4
  //    t22: i64 = and t20, Constant:i64<4294967295>
  //  t13: ch,glue = CopyToReg t0, Register:i64 $x10, t22
  //  t14: ch = RISCVISD::RET_GLUE t13, Register:i64 $x10, t13:1
  //
  // The DAG created below is derived from this
  SDLoc Loc;
  auto Int64VT = EVT::getIntegerVT(Context, 64);
  auto Px = DAG->getRegister(0, Int64VT);
  auto Py = DAG->getConstant(2147483647, Loc, Int64VT);
  auto N1 = DAG->getNode(ISD::AND, Loc, Int64VT, Px, Py);
  auto Qx = DAG->getRegister(0, Int64VT);
  auto N2 = DAG->getNode(RISCVISD::SRLW, Loc, Int64VT, N1, Qx);
  auto Py2 = DAG->getConstant(4294967295, Loc, Int64VT);
  auto N3 = DAG->getNode(ISD::AND, Loc, Int64VT, N2, Py2);
  // N1 = Px & 0x7FFFFFFF
  // The first AND ensures that the input to the shift has bit 31 cleared.
  // This means bits [63:31] of N1 are known to be zero.
  //
  // N2 = SRLW N1, Qx
  // SRLW performs a 32-bit logical right shift and then sign-extends the
  // 32-bit result to 64 bits. Because we know N1's bit 31 is 0, the
  // 32-bit result of the shift will also have its sign bit (bit 31) as 0.
  // Therefore, the sign-extension is guaranteed to be a zero-extension.
  //
  // N3 = N2 & 0xFFFFFFFF
  // This second AND is part of the canonical pattern to clear the upper
  // 32 bits, explicitly performing the zero-extension. From a KnownBits
  // perspective, it's redundant, as N2's upper bits are already known zero.
  //
  // As a result, for N3, we know the upper 32 bits are zero (from the effective
  // zero-extension) and we also know bit 31 is zero (from the initial AND).
  // This gives us 33 known most-significant zero bits.
  KnownBits Known = DAG->computeKnownBits(N3);
  EXPECT_EQ(Known.Zero, APInt(64, -2147483648));
  EXPECT_EQ(Known.One, APInt(64, 0));
}

} // end namespace llvm
