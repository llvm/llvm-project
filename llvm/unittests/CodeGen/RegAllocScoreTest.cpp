//===- MachineInstrTest.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../lib/CodeGen/RegAllocScore.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/ModuleSlotTracker.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Triple.h"
#include "gtest/gtest.h"

using namespace llvm;
extern cl::opt<double> CopyWeight;
extern cl::opt<double> LoadWeight;
extern cl::opt<double> StoreWeight;
extern cl::opt<double> CheapRematWeight;
extern cl::opt<double> ExpensiveRematWeight;

namespace {
// Include helper functions to ease the manipulation of MachineFunctions.
#include "MFCommon.inc"

// MachineFunction::CreateMachineInstr doesn't copy the MCInstrDesc, it
// takes its address. So we want a bunch of pre-allocated mock MCInstrDescs.
#define MOCK_INSTR(MACRO)                                                      \
  MACRO(Copy, TargetOpcode::COPY, 0)                                           \
  MACRO(Load, 0, 1ULL << MCID::MayLoad)                                        \
  MACRO(Store, 0, 1ULL << MCID::MayStore)                                      \
  MACRO(LoadStore, 0, (1ULL << MCID::MayLoad) | (1ULL << MCID::MayStore))      \
  MACRO(CheapRemat, 0, 1ULL << MCID::CheapAsAMove)                             \
  MACRO(ExpensiveRemat, 0, 0)                                                  \
  MACRO(Dbg, TargetOpcode::DBG_LABEL,                                          \
        (1ULL << MCID::MayLoad) | (1ULL << MCID::MayStore))                    \
  MACRO(InlAsm, TargetOpcode::INLINEASM,                                       \
        (1ULL << MCID::MayLoad) | (1ULL << MCID::MayStore))                    \
  MACRO(Kill, TargetOpcode::KILL,                                              \
        (1ULL << MCID::MayLoad) | (1ULL << MCID::MayStore))

enum MockInstrId {
#define MOCK_INSTR_ID(ID, IGNORE, IGNORE2) ID,
  MOCK_INSTR(MOCK_INSTR_ID)
#undef MOCK_INSTR_ID
      TotalMockInstrs
};

const std::array<MCInstrDesc, MockInstrId::TotalMockInstrs> MockInstrDescs{{
#define MOCK_SPEC(IGNORE, OPCODE, FLAGS)                                       \
  {OPCODE, 0, 0, 0, 0, 0, 0, 0, 0, FLAGS, 0},
    MOCK_INSTR(MOCK_SPEC)
#undef MOCK_SPEC
}};

MachineInstr *createMockCopy(MachineFunction &MF) {
  return MF.CreateMachineInstr(MockInstrDescs[MockInstrId::Copy], DebugLoc());
}

MachineInstr *createMockLoad(MachineFunction &MF) {
  return MF.CreateMachineInstr(MockInstrDescs[MockInstrId::Load], DebugLoc());
}

MachineInstr *createMockStore(MachineFunction &MF) {
  return MF.CreateMachineInstr(MockInstrDescs[MockInstrId::Store], DebugLoc());
}

MachineInstr *createMockLoadStore(MachineFunction &MF) {
  return MF.CreateMachineInstr(MockInstrDescs[MockInstrId::LoadStore],
                               DebugLoc());
}

MachineInstr *createMockCheapRemat(MachineFunction &MF) {
  return MF.CreateMachineInstr(MockInstrDescs[MockInstrId::CheapRemat],
                               DebugLoc());
}

MachineInstr *createMockExpensiveRemat(MachineFunction &MF) {
  return MF.CreateMachineInstr(MockInstrDescs[MockInstrId::ExpensiveRemat],
                               DebugLoc());
}

MachineInstr *createMockDebug(MachineFunction &MF) {
  return MF.CreateMachineInstr(MockInstrDescs[MockInstrId::Dbg], DebugLoc());
}

MachineInstr *createMockKill(MachineFunction &MF) {
  return MF.CreateMachineInstr(MockInstrDescs[MockInstrId::Kill], DebugLoc());
}

MachineInstr *createMockInlineAsm(MachineFunction &MF) {
  return MF.CreateMachineInstr(MockInstrDescs[MockInstrId::InlAsm], DebugLoc());
}

TEST(RegAllocScoreTest, SkipDebugKillInlineAsm) {
  LLVMContext Ctx;
  Module Mod("Module", Ctx);
  auto MF = createMachineFunction(Ctx, Mod);

  auto *MBB = MF->CreateMachineBasicBlock();
  MF->insert(MF->end(), MBB);
  auto MBBFreqMock = [&](const MachineBasicBlock &_MBB) -> double {
    assert(&_MBB == MBB);
    return 0.5;
  };
  auto Next = MBB->end();
  Next = MBB->insertAfter(Next, createMockInlineAsm(*MF));
  Next = MBB->insertAfter(Next, createMockDebug(*MF));
  Next = MBB->insertAfter(Next, createMockKill(*MF));
  const auto Score = llvm::calculateRegAllocScore(
      *MF, MBBFreqMock, [](const MachineInstr &) { return false; });
  ASSERT_EQ(MF->size(), 1U);
  ASSERT_EQ(Score, RegAllocScore());
}

TEST(RegAllocScoreTest, Counts) {
  LLVMContext Ctx;
  Module Mod("Module", Ctx);
  auto MF = createMachineFunction(Ctx, Mod);

  auto *MBB1 = MF->CreateMachineBasicBlock();
  auto *MBB2 = MF->CreateMachineBasicBlock();
  MF->insert(MF->end(), MBB1);
  MF->insert(MF->end(), MBB2);
  const double Freq1 = 0.5;
  const double Freq2 = 10.0;
  auto MBBFreqMock = [&](const MachineBasicBlock &MBB) -> double {
    if (&MBB == MBB1)
      return Freq1;
    if (&MBB == MBB2)
      return Freq2;
    llvm_unreachable("We only created 2 basic blocks");
  };
  auto Next = MBB1->end();
  Next = MBB1->insertAfter(Next, createMockCopy(*MF));
  Next = MBB1->insertAfter(Next, createMockLoad(*MF));
  Next = MBB1->insertAfter(Next, createMockLoad(*MF));
  Next = MBB1->insertAfter(Next, createMockStore(*MF));
  auto *CheapRemat = createMockCheapRemat(*MF);
  MBB1->insertAfter(Next, CheapRemat);
  Next = MBB2->end();
  Next = MBB2->insertAfter(Next, createMockLoad(*MF));
  Next = MBB2->insertAfter(Next, createMockStore(*MF));
  Next = MBB2->insertAfter(Next, createMockLoadStore(*MF));
  auto *ExpensiveRemat = createMockExpensiveRemat(*MF);
  MBB2->insertAfter(Next, ExpensiveRemat);
  auto IsRemat = [&](const MachineInstr &MI) {
    return &MI == CheapRemat || &MI == ExpensiveRemat;
  };
  ASSERT_EQ(MF->size(), 2U);
  const auto TotalScore =
      llvm::calculateRegAllocScore(*MF, MBBFreqMock, IsRemat);
  ASSERT_EQ(Freq1, TotalScore.copyCounts());
  ASSERT_EQ(2.0 * Freq1 + Freq2, TotalScore.loadCounts());
  ASSERT_EQ(Freq1 + Freq2, TotalScore.storeCounts());
  ASSERT_EQ(Freq2, TotalScore.loadStoreCounts());
  ASSERT_EQ(Freq1, TotalScore.cheapRematCounts());
  ASSERT_EQ(Freq2, TotalScore.expensiveRematCounts());
  ASSERT_EQ(TotalScore.getScore(),
            TotalScore.copyCounts() * CopyWeight +
                TotalScore.loadCounts() * LoadWeight +
                TotalScore.storeCounts() * StoreWeight +
                TotalScore.loadStoreCounts() * (LoadWeight + StoreWeight) +
                TotalScore.cheapRematCounts() * CheapRematWeight +
                TotalScore.expensiveRematCounts() * ExpensiveRematWeight

  );
}
} // end namespace
