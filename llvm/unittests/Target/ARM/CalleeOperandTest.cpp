//===- llvm/unittests/Target/ARM/CalleeOperandTest.cpp
//---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ARMInstrInfo.h"
#include "ARMSubtarget.h"
#include "ARMTargetMachine.h"
#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

std::unique_ptr<TargetMachine> createTargetMachine(const std::string &TTStr) {
  Triple TT(TTStr);
  std::string Error;
  const Target *T = TargetRegistry::lookupTarget(TT, Error);
  if (!T)
    return nullptr;

  TargetOptions Options;
  return std::unique_ptr<TargetMachine>(
      T->createTargetMachine(TT, "generic", "", Options, std::nullopt,
                             std::nullopt, CodeGenOptLevel::Default));
}

void checkCalleeOperand(
    TargetMachine *TM, const StringRef MIRString, unsigned InstIndex,
    std::function<void(const ARMBaseInstrInfo &, const MachineInstr &)> Check) {
  LLVMContext Context;
  std::unique_ptr<MemoryBuffer> MBuffer = MemoryBuffer::getMemBuffer(MIRString);
  std::unique_ptr<MIRParser> MParser =
      createMIRParser(std::move(MBuffer), Context);
  ASSERT_TRUE(MParser);

  std::unique_ptr<Module> M = MParser->parseIRModule();
  ASSERT_TRUE(M);

  M->setTargetTriple(TM->getTargetTriple());
  M->setDataLayout(TM->createDataLayout());

  MachineModuleInfo MMI(TM);
  bool Res = MParser->parseMachineFunctions(*M, MMI);
  ASSERT_FALSE(Res);

  auto F = M->getFunction("test");
  ASSERT_TRUE(F != nullptr);
  auto &MF = MMI.getOrCreateMachineFunction(*F);
  const ARMSubtarget &ST = MF.getSubtarget<ARMSubtarget>();
  const ARMBaseInstrInfo *TII = ST.getInstrInfo();

  auto &MBB = *MF.begin();
  ASSERT_FALSE(MBB.empty());
  auto It = MBB.begin();
  for (unsigned i = 0; i < InstIndex && It != MBB.end(); ++i)
    ++It;
  ASSERT_NE(It, MBB.end());
  Check(*TII, *It);
}

} // anonymous namespace

TEST(ARMBaseInstrInfoTest, ThumbCalleeOperand) {
  LLVMInitializeARMTargetInfo();
  LLVMInitializeARMTarget();
  LLVMInitializeARMTargetMC();

  auto TM = createTargetMachine("thumbv7m-unknown-none-eabi");
  if (!TM)
    GTEST_SKIP();

  // Test tBL: direct call has callee at operand 2.
  checkCalleeOperand(TM.get(),
                     "--- |\n"
                     "  declare void @callee()\n"
                     "  define void @test() { ret void }\n"
                     "...\n"
                     "---\n"
                     "name: test\n"
                     "tracksRegLiveness: true\n"
                     "body: |\n"
                     "  bb.0:\n"
                     "    tBL 14, $noreg, @callee, csr_aapcs\n"
                     "    tBX_RET 14, $noreg\n"
                     "...\n",
                     0,
                     [](const ARMBaseInstrInfo &TII, const MachineInstr &MI) {
                       ASSERT_TRUE(MI.isCall());
                       const MachineOperand &Op = TII.getCalleeOperand(MI);
                       EXPECT_EQ(&Op, &MI.getOperand(2));
                       ASSERT_TRUE(Op.isGlobal());
                       EXPECT_EQ(Op.getGlobal()->getName(), "callee");
                     });

  // Test tBLXi: direct call with architecture switch has callee at operand 2.
  checkCalleeOperand(TM.get(),
                     "--- |\n"
                     "  declare void @callee()\n"
                     "  define void @test() { ret void }\n"
                     "...\n"
                     "---\n"
                     "name: test\n"
                     "tracksRegLiveness: true\n"
                     "body: |\n"
                     "  bb.0:\n"
                     "    tBLXi 14, $noreg, @callee, csr_aapcs\n"
                     "    tBX_RET 14, $noreg\n"
                     "...\n",
                     0,
                     [](const ARMBaseInstrInfo &TII, const MachineInstr &MI) {
                       ASSERT_TRUE(MI.isCall());
                       const MachineOperand &Op = TII.getCalleeOperand(MI);
                       EXPECT_EQ(&Op, &MI.getOperand(2));
                       ASSERT_TRUE(Op.isGlobal());
                       EXPECT_EQ(Op.getGlobal()->getName(), "callee");
                     });

  // Test tBLXr: indirect register call has callee at operand 2.
  checkCalleeOperand(TM.get(),
                     "--- |\n"
                     "  define void @test() { ret void }\n"
                     "...\n"
                     "---\n"
                     "name: test\n"
                     "tracksRegLiveness: true\n"
                     "body: |\n"
                     "  bb.0:\n"
                     "    liveins: $r0\n"
                     "    tBLXr 14, $noreg, $r0, csr_aapcs\n"
                     "    tBX_RET 14, $noreg\n"
                     "...\n",
                     0,
                     [](const ARMBaseInstrInfo &TII, const MachineInstr &MI) {
                       ASSERT_TRUE(MI.isCall());
                       const MachineOperand &Op = TII.getCalleeOperand(MI);
                       EXPECT_EQ(&Op, &MI.getOperand(2));
                       ASSERT_TRUE(Op.isReg());
                       EXPECT_EQ(Op.getReg(), ARM::R0);
                     });

  // Test tTAILJMPd: direct tail call has callee at operand 0.
  checkCalleeOperand(TM.get(),
                     "--- |\n"
                     "  declare void @callee()\n"
                     "  define void @test() { ret void }\n"
                     "...\n"
                     "---\n"
                     "name: test\n"
                     "tracksRegLiveness: true\n"
                     "body: |\n"
                     "  bb.0:\n"
                     "    tTAILJMPd @callee, 14, $noreg\n"
                     "...\n",
                     0,
                     [](const ARMBaseInstrInfo &TII, const MachineInstr &MI) {
                       ASSERT_TRUE(MI.isCall());
                       const MachineOperand &Op = TII.getCalleeOperand(MI);
                       EXPECT_EQ(&Op, &MI.getOperand(0));
                       ASSERT_TRUE(Op.isGlobal());
                       EXPECT_EQ(Op.getGlobal()->getName(), "callee");
                     });
}

TEST(ARMBaseInstrInfoTest, ARMCalleeOperand) {
  LLVMInitializeARMTargetInfo();
  LLVMInitializeARMTarget();
  LLVMInitializeARMTargetMC();

  auto TM = createTargetMachine("arm-unknown-linux");
  if (!TM)
    GTEST_SKIP();

  // Test BL: direct call has callee at operand 0.
  checkCalleeOperand(TM.get(),
                     "--- |\n"
                     "  declare void @callee()\n"
                     "  define void @test() { ret void }\n"
                     "...\n"
                     "---\n"
                     "name: test\n"
                     "tracksRegLiveness: true\n"
                     "body: |\n"
                     "  bb.0:\n"
                     "    BL @callee, csr_aapcs\n"
                     "    BX_RET 14, $noreg\n"
                     "...\n",
                     0,
                     [](const ARMBaseInstrInfo &TII, const MachineInstr &MI) {
                       ASSERT_TRUE(MI.isCall());
                       const MachineOperand &Op = TII.getCalleeOperand(MI);
                       EXPECT_EQ(&Op, &MI.getOperand(0));
                       ASSERT_TRUE(Op.isGlobal());
                       EXPECT_EQ(Op.getGlobal()->getName(), "callee");
                     });

  // Test BLX: indirect register call has callee at operand 0.
  checkCalleeOperand(TM.get(),
                     "--- |\n"
                     "  define void @test() { ret void }\n"
                     "...\n"
                     "---\n"
                     "name: test\n"
                     "tracksRegLiveness: true\n"
                     "body: |\n"
                     "  bb.0:\n"
                     "    liveins: $r0\n"
                     "    BLX $r0, csr_aapcs\n"
                     "    BX_RET 14, $noreg\n"
                     "...\n",
                     0,
                     [](const ARMBaseInstrInfo &TII, const MachineInstr &MI) {
                       ASSERT_TRUE(MI.isCall());
                       const MachineOperand &Op = TII.getCalleeOperand(MI);
                       EXPECT_EQ(&Op, &MI.getOperand(0));
                       ASSERT_TRUE(Op.isReg());
                       EXPECT_EQ(Op.getReg(), ARM::R0);
                     });

  // Test BL_pred: predicated direct call has callee at operand 0.
  checkCalleeOperand(
      TM.get(),
      "--- |\n"
      "  declare void @callee()\n"
      "  define void @test() { ret void }\n"
      "...\n"
      "---\n"
      "name: test\n"
      "tracksRegLiveness: true\n"
      "body: |\n"
      "  bb.0:\n"
      "    liveins: $r0\n"
      "    CMPri killed $r0, 0, 14, $noreg, implicit-def $cpsr\n"
      "    BL_pred @callee, 0, killed $cpsr, csr_aapcs, implicit $cpsr\n"
      "    BX_RET 14, $noreg\n"
      "...\n",
      1, [](const ARMBaseInstrInfo &TII, const MachineInstr &MI) {
        ASSERT_TRUE(MI.isCall());
        const MachineOperand &Op = TII.getCalleeOperand(MI);
        EXPECT_EQ(&Op, &MI.getOperand(0));
        ASSERT_TRUE(Op.isGlobal());
        EXPECT_EQ(Op.getGlobal()->getName(), "callee");
      });

  // Test TAILJMPd: direct tail call has callee at operand 0.
  checkCalleeOperand(TM.get(),
                     "--- |\n"
                     "  declare void @callee()\n"
                     "  define void @test() { ret void }\n"
                     "...\n"
                     "---\n"
                     "name: test\n"
                     "tracksRegLiveness: true\n"
                     "body: |\n"
                     "  bb.0:\n"
                     "    TAILJMPd @callee\n"
                     "...\n",
                     0,
                     [](const ARMBaseInstrInfo &TII, const MachineInstr &MI) {
                       ASSERT_TRUE(MI.isCall());
                       const MachineOperand &Op = TII.getCalleeOperand(MI);
                       EXPECT_EQ(&Op, &MI.getOperand(0));
                       ASSERT_TRUE(Op.isGlobal());
                       EXPECT_EQ(Op.getGlobal()->getName(), "callee");
                     });
}
