//===- MachineBasicBlockTest.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/CodeGenTargetMachineImpl.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {
// Include helper functions to ease the manipulation of MachineFunctions.
#include "MFCommon.inc"

TEST(FindDebugLocTest, DifferentIterators) {
  LLVMContext Ctx;
  Module Mod("Module", Ctx);
  auto MF = createMachineFunction(Ctx, Mod);
  auto &MBB = *MF->CreateMachineBasicBlock();

  // Create metadata: CU, subprogram, some blocks and an inline function
  // scope.
  DIBuilder DIB(Mod);
  DIFile *OurFile = DIB.createFile("foo.c", "/bar");
  DICompileUnit *OurCU = DIB.createCompileUnit(
      DISourceLanguageName(dwarf::DW_LANG_C99), OurFile, "", false, "", 0);
  auto OurSubT = DIB.createSubroutineType(DIB.getOrCreateTypeArray({}));
  DISubprogram *OurFunc =
      DIB.createFunction(OurCU, "bees", "", OurFile, 1, OurSubT, 1,
                         DINode::FlagZero, DISubprogram::SPFlagDefinition);

  DebugLoc DL0;
  DebugLoc DL1 = DILocation::get(Ctx, 1, 0, OurFunc);
  DebugLoc DL2 = DILocation::get(Ctx, 2, 0, OurFunc);
  DebugLoc DL3 = DILocation::get(Ctx, 3, 0, OurFunc);

  // Test using and empty MBB.
  EXPECT_EQ(DL0, MBB.findDebugLoc(MBB.instr_begin()));
  EXPECT_EQ(DL0, MBB.findDebugLoc(MBB.instr_end()));

  EXPECT_EQ(DL0, MBB.rfindDebugLoc(MBB.instr_rbegin()));
  EXPECT_EQ(DL0, MBB.rfindDebugLoc(MBB.instr_rend()));

  EXPECT_EQ(DL0, MBB.findPrevDebugLoc(MBB.instr_begin()));
  EXPECT_EQ(DL0, MBB.findPrevDebugLoc(MBB.instr_end()));

  EXPECT_EQ(DL0, MBB.rfindPrevDebugLoc(MBB.instr_rbegin()));
  EXPECT_EQ(DL0, MBB.rfindPrevDebugLoc(MBB.instr_rend()));

  // Insert two MIs with DebugLoc DL1 and DL3.
  // Also add a DBG_VALUE with a different DebugLoc in between.
  MCInstrDesc COPY = {TargetOpcode::COPY, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  MCInstrDesc DBG = {TargetOpcode::DBG_VALUE, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  auto MI3 = MF->CreateMachineInstr(COPY, DL3);
  MBB.insert(MBB.begin(), MI3);
  auto MI2 = MF->CreateMachineInstr(DBG, DL2);
  MBB.insert(MBB.begin(), MI2);
  auto MI1 = MF->CreateMachineInstr(COPY, DL1);
  MBB.insert(MBB.begin(), MI1);

  // Test using two MIs with a debug instruction in between.
  EXPECT_EQ(DL1, MBB.findDebugLoc(MBB.instr_begin()));
  EXPECT_EQ(DL1, MBB.findDebugLoc(MI1));
  EXPECT_EQ(DL3, MBB.findDebugLoc(MI2));
  EXPECT_EQ(DL3, MBB.findDebugLoc(MI3));
  EXPECT_EQ(DL0, MBB.findDebugLoc(MBB.instr_end()));

  EXPECT_EQ(DL1, MBB.rfindDebugLoc(MBB.instr_rend()));
  EXPECT_EQ(DL1, MBB.rfindDebugLoc(MI1));
  EXPECT_EQ(DL3, MBB.rfindDebugLoc(MI2));
  EXPECT_EQ(DL3, MBB.rfindDebugLoc(MI3));
  EXPECT_EQ(DL3, MBB.rfindDebugLoc(MBB.instr_rbegin()));

  EXPECT_EQ(DL0, MBB.findPrevDebugLoc(MBB.instr_begin()));
  EXPECT_EQ(DL0, MBB.findPrevDebugLoc(MI1));
  EXPECT_EQ(DL1, MBB.findPrevDebugLoc(MI2));
  EXPECT_EQ(DL1, MBB.findPrevDebugLoc(MI3));
  EXPECT_EQ(DL3, MBB.findPrevDebugLoc(MBB.instr_end()));

  EXPECT_EQ(DL0, MBB.rfindPrevDebugLoc(MBB.instr_rend()));
  EXPECT_EQ(DL0, MBB.rfindPrevDebugLoc(MI1));
  EXPECT_EQ(DL1, MBB.rfindPrevDebugLoc(MI2));
  EXPECT_EQ(DL1, MBB.rfindPrevDebugLoc(MI3));
  EXPECT_EQ(DL1, MBB.rfindPrevDebugLoc(MBB.instr_rbegin()));

  // Finalize DIBuilder to avoid memory leaks.
  DIB.finalize();
}

static MachineInstr *createMI(MachineFunction &MF, unsigned Opcode,
                              bool BBProlog = false) {
  MCInstrDesc Desc = {Opcode, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  MachineInstr *MI = MF.CreateMachineInstr(Desc, DebugLoc());
  if (BBProlog)
    MI->setFlag(MachineInstr::BBProlog);
  return MI;
}

// An instruction inserted in front of a block prolog instruction is in the
// prolog.
TEST(BBPrologTest, IsInBBProlog) {
  LLVMContext Ctx;
  Module Mod("Module", Ctx);
  auto MF = createMachineFunction(Ctx, Mod);
  auto &MBB = *MF->CreateMachineBasicBlock();

  MachineInstr *Prolog = createMI(*MF, TargetOpcode::COPY, true);
  MachineInstr *Body = createMI(*MF, TargetOpcode::COPY);
  MBB.push_back(Prolog);
  MBB.push_back(Body);

  EXPECT_TRUE(MBB.isInBBProlog(Prolog->getIterator()));
  EXPECT_FALSE(MBB.isInBBProlog(Body->getIterator()));
  EXPECT_FALSE(MBB.isInBBProlog(MBB.end()));
  EXPECT_EQ(MBB.getBBPrologFlag(Prolog->getIterator()), MachineInstr::BBProlog);
  EXPECT_EQ(MBB.getBBPrologFlag(Body->getIterator()), MachineInstr::NoFlags);
}

// Debug instructions, labels and pseudo probes in front of a prolog instruction
// are looked through.
TEST(BBPrologTest, IsInBBPrologLooksThroughDebugLabelsAndProbes) {
  LLVMContext Ctx;
  Module Mod("Module", Ctx);
  auto MF = createMachineFunction(Ctx, Mod);
  auto &MBB = *MF->CreateMachineBasicBlock();

  MachineInstr *Dbg = createMI(*MF, TargetOpcode::DBG_VALUE);
  MachineInstr *Label = createMI(*MF, TargetOpcode::EH_LABEL);
  MachineInstr *Probe = createMI(*MF, TargetOpcode::PSEUDO_PROBE);
  MachineInstr *Prolog = createMI(*MF, TargetOpcode::COPY, true);
  MachineInstr *Dbg2 = createMI(*MF, TargetOpcode::DBG_VALUE);
  MachineInstr *Body = createMI(*MF, TargetOpcode::COPY);
  for (MachineInstr *MI : {Dbg, Label, Probe, Prolog, Dbg2, Body})
    MBB.push_back(MI);

  EXPECT_TRUE(MBB.isInBBProlog(Dbg->getIterator()));
  EXPECT_TRUE(MBB.isInBBProlog(Label->getIterator()));
  EXPECT_TRUE(MBB.isInBBProlog(Probe->getIterator()));
  EXPECT_FALSE(MBB.isInBBProlog(Dbg2->getIterator()));
}

// setBBPrologFlag marks a range, except labels, debug instructions and pseudo
// probes.
TEST(BBPrologTest, SetBBPrologFlag) {
  LLVMContext Ctx;
  Module Mod("Module", Ctx);
  auto MF = createMachineFunction(Ctx, Mod);
  auto &MBB = *MF->CreateMachineBasicBlock();

  MachineInstr *A = createMI(*MF, TargetOpcode::COPY);
  MachineInstr *Dbg = createMI(*MF, TargetOpcode::DBG_VALUE);
  MachineInstr *Label = createMI(*MF, TargetOpcode::EH_LABEL);
  MachineInstr *Probe = createMI(*MF, TargetOpcode::PSEUDO_PROBE);
  MachineInstr *B = createMI(*MF, TargetOpcode::COPY);
  MachineInstr *Body = createMI(*MF, TargetOpcode::COPY);
  for (MachineInstr *MI : {A, Dbg, Label, Probe, B, Body})
    MBB.push_back(MI);

  MBB.setBBPrologFlag(A->getIterator(), Body->getIterator());
  EXPECT_TRUE(A->getFlag(MachineInstr::BBProlog));
  EXPECT_TRUE(B->getFlag(MachineInstr::BBProlog));
  EXPECT_FALSE(Dbg->getFlag(MachineInstr::BBProlog));
  EXPECT_FALSE(Label->getFlag(MachineInstr::BBProlog));
  EXPECT_FALSE(Probe->getFlag(MachineInstr::BBProlog));
  EXPECT_FALSE(Body->getFlag(MachineInstr::BBProlog));
}

// The block-entry walks stop after the prolog.
TEST(BBPrologTest, SkipPHIsSkipsProlog) {
  LLVMContext Ctx;
  Module Mod("Module", Ctx);
  auto MF = createMachineFunction(Ctx, Mod);
  auto &MBB = *MF->CreateMachineBasicBlock();

  MachineInstr *PHI = createMI(*MF, TargetOpcode::PHI);
  MachineInstr *Label = createMI(*MF, TargetOpcode::EH_LABEL);
  MachineInstr *Prolog = createMI(*MF, TargetOpcode::COPY, true);
  MachineInstr *Dbg = createMI(*MF, TargetOpcode::DBG_VALUE);
  MachineInstr *Prolog2 = createMI(*MF, TargetOpcode::COPY, true);
  MachineInstr *Body = createMI(*MF, TargetOpcode::COPY);
  for (MachineInstr *MI : {PHI, Label, Prolog, Dbg, Prolog2, Body})
    MBB.push_back(MI);

  EXPECT_EQ(&*MBB.SkipPHIsLabelsAndDebug(MBB.begin()), Body);
  MBB.erase(Dbg);
  EXPECT_EQ(&*MBB.SkipPHIsAndLabels(MBB.begin()), Body);

  Prolog->clearFlag(MachineInstr::BBProlog);
  Prolog2->clearFlag(MachineInstr::BBProlog);
  EXPECT_EQ(&*MBB.SkipPHIsLabelsAndDebug(MBB.begin()), Prolog);
  EXPECT_EQ(&*MBB.SkipPHIsAndLabels(MBB.begin()), Prolog);
}

} // end namespace
