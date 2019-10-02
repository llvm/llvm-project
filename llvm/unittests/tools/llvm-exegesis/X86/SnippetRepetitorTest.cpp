//===-- SnippetRepetitorTest.cpp --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../Common/AssemblerUtils.h"
#include "Latency.h"
#include "LlvmState.h"
#include "MCInstrDescView.h"
#include "RegisterAliasing.h"
#include "TestBase.h"
#include "Uops.h"
#include "X86InstrInfo.h"
#include "llvm/CodeGen/MachineBasicBlock.h"

namespace llvm {
namespace exegesis {

void InitializeX86ExegesisTarget();

namespace {

using testing::ElementsAre;
using testing::Eq;
using testing::Field;
using testing::Property;
using testing::UnorderedElementsAre;

class X86SnippetRepetitorTest : public X86TestBase {
protected:
  void SetUp() {
    TM = State.createTargetMachine();
    Context = std::make_unique<LLVMContext>();
    Mod =
        std::make_unique<Module>("X86SnippetRepetitorTest", *Context);
    Mod->setDataLayout(TM->createDataLayout());
    MMI = std::make_unique<MachineModuleInfo>(TM.get());
    MF = &createVoidVoidPtrMachineFunction("TestFn", Mod.get(), MMI.get());
  }

  void TestCommon(InstructionBenchmark::RepetitionModeE RepetitionMode) {
    const auto Repetitor = SnippetRepetitor::Create(RepetitionMode, State);
    const std::vector<MCInst> Instructions = {MCInstBuilder(X86::NOOP)};
    FunctionFiller Sink(*MF, {X86::EAX});
    const auto Fill = Repetitor->Repeat(Instructions, kMinInstructions);
    Fill(Sink);
  }

  static constexpr const unsigned kMinInstructions = 3;

  std::unique_ptr<LLVMTargetMachine> TM;
  std::unique_ptr<LLVMContext> Context;
  std::unique_ptr<Module> Mod;
  std::unique_ptr<MachineModuleInfo> MMI;
  MachineFunction *MF = nullptr;
};

static auto HasOpcode = [](unsigned Opcode) {
  return Property(&MachineInstr::getOpcode, Eq(Opcode));
};

static auto LiveReg = [](unsigned Reg) {
  return Field(&MachineBasicBlock::RegisterMaskPair::PhysReg, Eq(Reg));
};

TEST_F(X86SnippetRepetitorTest, Duplicate) {
  TestCommon(InstructionBenchmark::Duplicate);
  // Duplicating creates a single basic block that repeats the instructions.
  ASSERT_EQ(MF->getNumBlockIDs(), 1u);
  EXPECT_THAT(MF->getBlockNumbered(0)->instrs(),
              ElementsAre(HasOpcode(X86::NOOP), HasOpcode(X86::NOOP),
                          HasOpcode(X86::NOOP), HasOpcode(X86::RETQ)));
}

TEST_F(X86SnippetRepetitorTest, Loop) {
  TestCommon(InstructionBenchmark::Loop);
  // Duplicating creates an entry block, a loop body and a ret block.
  ASSERT_EQ(MF->getNumBlockIDs(), 3u);
  const auto &LoopBlock = *MF->getBlockNumbered(1);
  EXPECT_THAT(LoopBlock.instrs(),
              ElementsAre(HasOpcode(X86::NOOP), HasOpcode(X86::ADD64ri8),
                          HasOpcode(X86::JCC_1)));
  EXPECT_THAT(LoopBlock.liveins(),
              UnorderedElementsAre(
                  LiveReg(X86::EAX),
                  LiveReg(State.getExegesisTarget().getLoopCounterRegister(
                      State.getTargetMachine().getTargetTriple()))));
  EXPECT_THAT(MF->getBlockNumbered(2)->instrs(),
              ElementsAre(HasOpcode(X86::RETQ)));
}

} // namespace
} // namespace exegesis
} // namespace llvm
