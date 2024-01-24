//===-- SnippetRepetitorTest.cpp --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../Common/AssemblerUtils.h"
#include "LlvmState.h"
#include "MCInstrDescView.h"
#include "RegisterAliasing.h"
#include "TestBase.h"
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
  void SetUp() override {
    TM = State.createTargetMachine();
    Context = std::make_unique<LLVMContext>();
    Mod = std::make_unique<Module>("X86SnippetRepetitorTest", *Context);
    Mod->setDataLayout(TM->createDataLayout());
    MMI = std::make_unique<MachineModuleInfo>(TM.get());
    MF = &createVoidVoidPtrMachineFunction("TestFn", Mod.get(), MMI.get());
  }

  void TestCommon(Benchmark::RepetitionModeE RepetitionMode,
                  unsigned SnippetInstructions = 1) {
    const auto Repetitor = SnippetRepetitor::Create(RepetitionMode, State);
    const std::vector<MCInst> Instructions(SnippetInstructions,
                                           MCInstBuilder(X86::NOOP));
    FunctionFiller Sink(*MF, {X86::EAX});
    const auto Fill =
        Repetitor->Repeat(Instructions, kMinInstructions, kLoopBodySize, false);
    Fill(Sink);
  }

  static constexpr const unsigned kMinInstructions = 3;
  static constexpr const unsigned kLoopBodySize = 5;

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
  TestCommon(Benchmark::Duplicate);
  // Duplicating creates a single basic block that repeats the instructions.
  ASSERT_EQ(MF->getNumBlockIDs(), 1u);
  EXPECT_THAT(MF->getBlockNumbered(0)->instrs(),
              ElementsAre(HasOpcode(X86::NOOP), HasOpcode(X86::NOOP),
                          HasOpcode(X86::NOOP), HasOpcode(X86::RET64)));
}

TEST_F(X86SnippetRepetitorTest, DuplicateSnippetInstructionCount) {
  TestCommon(Benchmark::Duplicate, 2);
  // Duplicating a snippet of two instructions with the minimum number of
  // instructions set to three duplicates the snippet twice for a total of
  // four instructions.
  ASSERT_EQ(MF->getNumBlockIDs(), 1u);
  EXPECT_THAT(MF->getBlockNumbered(0)->instrs(),
              ElementsAre(HasOpcode(X86::NOOP), HasOpcode(X86::NOOP),
                          HasOpcode(X86::NOOP), HasOpcode(X86::NOOP),
                          HasOpcode(X86::RET64)));
}

TEST_F(X86SnippetRepetitorTest, Loop) {
  TestCommon(Benchmark::Loop);
  // Duplicating creates an entry block, a loop body and a ret block.
  ASSERT_EQ(MF->getNumBlockIDs(), 3u);
  const auto &LoopBlock = *MF->getBlockNumbered(1);
  EXPECT_THAT(LoopBlock.instrs(),
              ElementsAre(HasOpcode(X86::NOOP), HasOpcode(X86::NOOP),
                          HasOpcode(X86::NOOP), HasOpcode(X86::NOOP),
                          HasOpcode(X86::NOOP), HasOpcode(X86::ADD64ri8),
                          HasOpcode(X86::JCC_1)));
  EXPECT_THAT(LoopBlock.liveins(),
              UnorderedElementsAre(
                  LiveReg(X86::EAX),
                  LiveReg(State.getExegesisTarget().getLoopCounterRegister(
                      State.getTargetMachine().getTargetTriple()))));
  EXPECT_THAT(MF->getBlockNumbered(2)->instrs(),
              ElementsAre(HasOpcode(X86::RET64)));
}

} // namespace
} // namespace exegesis
} // namespace llvm
