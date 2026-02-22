//===- MachineCopyPropagationTest.cpp - MCP unit tests --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class NoInitMachineModuleInfoWrapperPass : public MachineModuleInfoWrapperPass {
public:
  explicit NoInitMachineModuleInfoWrapperPass(const TargetMachine *TM)
      : MachineModuleInfoWrapperPass(TM) {}

  bool doInitialization(Module &M) override { return false; }
  bool doFinalization(Module &M) override { return false; }
};

class MachineCopyPropagationTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    InitializeAllTargets();
    InitializeAllTargetMCs();
  }

  void SetUp() override {
    Triple TargetTriple("x86_64-unknown-linux-gnu");
    std::string Error;
    const Target *T = TargetRegistry::lookupTarget("", TargetTriple, Error);
    if (!T)
      GTEST_SKIP() << Error;

    TargetOptions Options;
    TM = std::unique_ptr<TargetMachine>(T->createTargetMachine(
        TargetTriple, "", "", Options, std::nullopt));
    if (!TM)
      GTEST_SKIP();
  }

  bool parseMIR(StringRef MIRCode, MachineModuleInfo &MMI) {
    std::unique_ptr<MemoryBuffer> MBuffer = MemoryBuffer::getMemBuffer(MIRCode);
    MIR = createMIRParser(std::move(MBuffer), Context);
    if (!MIR)
      return false;

    M = MIR->parseIRModule();
    if (!M)
      return false;
    M->setDataLayout(TM->createDataLayout());

    return !MIR->parseMachineFunctions(*M, MMI);
  }

  static unsigned countCopies(const MachineBasicBlock &MBB) {
    return count_if(MBB, [](const MachineInstr &MI) { return MI.isCopy(); });
  }

  static size_t liveInCount(const MachineBasicBlock &MBB) {
    return static_cast<size_t>(std::distance(MBB.liveins().begin(),
                                             MBB.liveins().end()));
  }

  LLVMContext Context;
  std::unique_ptr<TargetMachine> TM;
  std::unique_ptr<Module> M;
  std::unique_ptr<MIRParser> MIR;
};

TEST_F(MachineCopyPropagationTest, KeepsCopyWhenSuccessorUsesRegBeforeDef) {
  StringRef MIRString = R"(
--- |
  define void @f() {
  entry:
    br label %bb1
  bb1:
    ret void
  }
...
---
name:            f
tracksRegLiveness: true
body:             |
  bb.0.entry:
    successors: %bb.1.bb1(0x80000000)
    liveins: $ecx, $edx

    $eax = COPY killed $ecx
    JMP_1 %bb.1.bb1

  bb.1.bb1:
    liveins: $eax, $edx

    $eax = ADD32rr killed $eax, killed $edx, implicit-def dead $eflags
    RET64
...
)";

  auto *MMIWP = new NoInitMachineModuleInfoWrapperPass(TM.get());
  ASSERT_TRUE(parseMIR(MIRString, MMIWP->getMMI()));

  Function *F = M->getFunction("f");
  ASSERT_NE(F, nullptr);
  MachineFunction *MF = MMIWP->getMMI().getMachineFunction(*F);
  ASSERT_NE(MF, nullptr);
  ASSERT_FALSE(MF->empty());

  MachineBasicBlock &EntryMBB = MF->front();
  ASSERT_EQ(EntryMBB.succ_size(), 1u);
  MachineBasicBlock &SuccMBB = **EntryMBB.succ_begin();

  EXPECT_EQ(countCopies(EntryMBB), 1u);
  ASSERT_EQ(liveInCount(SuccMBB), 2u);

  // simulate stale successor liveness metadata while keeping tracksLiveness on
  SuccMBB.clearLiveIns();
  EXPECT_EQ(liveInCount(SuccMBB), 0u);

  legacy::PassManager PM;
  PM.add(MMIWP);
  PM.add(createMachineCopyPropagationPass(false));
  PM.run(*M);

  EXPECT_EQ(countCopies(EntryMBB), 1u);
}

} // namespace
