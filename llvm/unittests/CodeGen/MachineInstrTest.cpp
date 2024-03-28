//===- MachineInstrTest.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/MemoryModelRelaxationAnnotations.h"
#include "llvm/IR/ModuleSlotTracker.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Triple.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {
// Include helper functions to ease the manipulation of MachineFunctions.
#include "MFCommon.inc"

std::unique_ptr<MCContext> createMCContext(MCAsmInfo *AsmInfo) {
  Triple TheTriple(/*ArchStr=*/"", /*VendorStr=*/"", /*OSStr=*/"",
                   /*EnvironmentStr=*/"elf");
  return std::make_unique<MCContext>(TheTriple, AsmInfo, nullptr, nullptr,
                                     nullptr, nullptr, false);
}

// This test makes sure that MachineInstr::isIdenticalTo handles Defs correctly
// for various combinations of IgnoreDefs, and also that it is symmetrical.
TEST(IsIdenticalToTest, DifferentDefs) {
  LLVMContext Ctx;
  Module Mod("Module", Ctx);
  auto MF = createMachineFunction(Ctx, Mod);

  unsigned short NumOps = 2;
  unsigned char NumDefs = 1;
  struct {
    MCInstrDesc MCID;
    MCOperandInfo OpInfo[2];
  } Table = {
      {0, NumOps, NumDefs, 0, 0, 0, 0, 0, 0, 1ULL << MCID::HasOptionalDef, 0},
      {{0, 0, MCOI::OPERAND_REGISTER, 0},
       {0, 1 << MCOI::OptionalDef, MCOI::OPERAND_REGISTER, 0}}};

  // Create two MIs with different virtual reg defs and the same uses.
  unsigned VirtualDef1 = -42; // The value doesn't matter, but the sign does.
  unsigned VirtualDef2 = -43;
  unsigned VirtualUse = -44;

  auto MI1 = MF->CreateMachineInstr(Table.MCID, DebugLoc());
  MI1->addOperand(*MF, MachineOperand::CreateReg(VirtualDef1, /*isDef*/ true));
  MI1->addOperand(*MF, MachineOperand::CreateReg(VirtualUse, /*isDef*/ false));

  auto MI2 = MF->CreateMachineInstr(Table.MCID, DebugLoc());
  MI2->addOperand(*MF, MachineOperand::CreateReg(VirtualDef2, /*isDef*/ true));
  MI2->addOperand(*MF, MachineOperand::CreateReg(VirtualUse, /*isDef*/ false));

  // Check that they are identical when we ignore virtual register defs, but not
  // when we check defs.
  ASSERT_FALSE(MI1->isIdenticalTo(*MI2, MachineInstr::CheckDefs));
  ASSERT_FALSE(MI2->isIdenticalTo(*MI1, MachineInstr::CheckDefs));

  ASSERT_TRUE(MI1->isIdenticalTo(*MI2, MachineInstr::IgnoreVRegDefs));
  ASSERT_TRUE(MI2->isIdenticalTo(*MI1, MachineInstr::IgnoreVRegDefs));

  // Create two MIs with different virtual reg defs, and a def or use of a
  // sentinel register.
  unsigned SentinelReg = 0;

  auto MI3 = MF->CreateMachineInstr(Table.MCID, DebugLoc());
  MI3->addOperand(*MF, MachineOperand::CreateReg(VirtualDef1, /*isDef*/ true));
  MI3->addOperand(*MF, MachineOperand::CreateReg(SentinelReg, /*isDef*/ true));

  auto MI4 = MF->CreateMachineInstr(Table.MCID, DebugLoc());
  MI4->addOperand(*MF, MachineOperand::CreateReg(VirtualDef2, /*isDef*/ true));
  MI4->addOperand(*MF, MachineOperand::CreateReg(SentinelReg, /*isDef*/ false));

  // Check that they are never identical.
  ASSERT_FALSE(MI3->isIdenticalTo(*MI4, MachineInstr::CheckDefs));
  ASSERT_FALSE(MI4->isIdenticalTo(*MI3, MachineInstr::CheckDefs));

  ASSERT_FALSE(MI3->isIdenticalTo(*MI4, MachineInstr::IgnoreVRegDefs));
  ASSERT_FALSE(MI4->isIdenticalTo(*MI3, MachineInstr::IgnoreVRegDefs));
}

// Check that MachineInstrExpressionTrait::isEqual is symmetric and in sync with
// MachineInstrExpressionTrait::getHashValue
void checkHashAndIsEqualMatch(MachineInstr *MI1, MachineInstr *MI2) {
  bool IsEqual1 = MachineInstrExpressionTrait::isEqual(MI1, MI2);
  bool IsEqual2 = MachineInstrExpressionTrait::isEqual(MI2, MI1);

  ASSERT_EQ(IsEqual1, IsEqual2);

  auto Hash1 = MachineInstrExpressionTrait::getHashValue(MI1);
  auto Hash2 = MachineInstrExpressionTrait::getHashValue(MI2);

  ASSERT_EQ(IsEqual1, Hash1 == Hash2);
}

// This test makes sure that MachineInstrExpressionTraits::isEqual is in sync
// with MachineInstrExpressionTraits::getHashValue.
TEST(MachineInstrExpressionTraitTest, IsEqualAgreesWithGetHashValue) {
  LLVMContext Ctx;
  Module Mod("Module", Ctx);
  auto MF = createMachineFunction(Ctx, Mod);

  unsigned short NumOps = 2;
  unsigned char NumDefs = 1;
  struct {
    MCInstrDesc MCID;
    MCOperandInfo OpInfo[2];
  } Table = {
      {0, NumOps, NumDefs, 0, 0, 0, 0, 0, 0, 1ULL << MCID::HasOptionalDef, 0},
      {{0, 0, MCOI::OPERAND_REGISTER, 0},
       {0, 1 << MCOI::OptionalDef, MCOI::OPERAND_REGISTER, 0}}};

  // Define a series of instructions with different kinds of operands and make
  // sure that the hash function is consistent with isEqual for various
  // combinations of them.
  unsigned VirtualDef1 = -42;
  unsigned VirtualDef2 = -43;
  unsigned VirtualReg = -44;
  unsigned SentinelReg = 0;
  unsigned PhysicalReg = 45;

  auto VD1VU = MF->CreateMachineInstr(Table.MCID, DebugLoc());
  VD1VU->addOperand(*MF,
                    MachineOperand::CreateReg(VirtualDef1, /*isDef*/ true));
  VD1VU->addOperand(*MF,
                    MachineOperand::CreateReg(VirtualReg, /*isDef*/ false));

  auto VD2VU = MF->CreateMachineInstr(Table.MCID, DebugLoc());
  VD2VU->addOperand(*MF,
                    MachineOperand::CreateReg(VirtualDef2, /*isDef*/ true));
  VD2VU->addOperand(*MF,
                    MachineOperand::CreateReg(VirtualReg, /*isDef*/ false));

  auto VD1SU = MF->CreateMachineInstr(Table.MCID, DebugLoc());
  VD1SU->addOperand(*MF,
                    MachineOperand::CreateReg(VirtualDef1, /*isDef*/ true));
  VD1SU->addOperand(*MF,
                    MachineOperand::CreateReg(SentinelReg, /*isDef*/ false));

  auto VD1SD = MF->CreateMachineInstr(Table.MCID, DebugLoc());
  VD1SD->addOperand(*MF,
                    MachineOperand::CreateReg(VirtualDef1, /*isDef*/ true));
  VD1SD->addOperand(*MF,
                    MachineOperand::CreateReg(SentinelReg, /*isDef*/ true));

  auto VD2PU = MF->CreateMachineInstr(Table.MCID, DebugLoc());
  VD2PU->addOperand(*MF,
                    MachineOperand::CreateReg(VirtualDef2, /*isDef*/ true));
  VD2PU->addOperand(*MF,
                    MachineOperand::CreateReg(PhysicalReg, /*isDef*/ false));

  auto VD2PD = MF->CreateMachineInstr(Table.MCID, DebugLoc());
  VD2PD->addOperand(*MF,
                    MachineOperand::CreateReg(VirtualDef2, /*isDef*/ true));
  VD2PD->addOperand(*MF,
                    MachineOperand::CreateReg(PhysicalReg, /*isDef*/ true));

  checkHashAndIsEqualMatch(VD1VU, VD2VU);
  checkHashAndIsEqualMatch(VD1VU, VD1SU);
  checkHashAndIsEqualMatch(VD1VU, VD1SD);
  checkHashAndIsEqualMatch(VD1VU, VD2PU);
  checkHashAndIsEqualMatch(VD1VU, VD2PD);

  checkHashAndIsEqualMatch(VD2VU, VD1SU);
  checkHashAndIsEqualMatch(VD2VU, VD1SD);
  checkHashAndIsEqualMatch(VD2VU, VD2PU);
  checkHashAndIsEqualMatch(VD2VU, VD2PD);

  checkHashAndIsEqualMatch(VD1SU, VD1SD);
  checkHashAndIsEqualMatch(VD1SU, VD2PU);
  checkHashAndIsEqualMatch(VD1SU, VD2PD);

  checkHashAndIsEqualMatch(VD1SD, VD2PU);
  checkHashAndIsEqualMatch(VD1SD, VD2PD);

  checkHashAndIsEqualMatch(VD2PU, VD2PD);
}

TEST(MachineInstrPrintingTest, DebugLocPrinting) {
  LLVMContext Ctx;
  Module Mod("Module", Ctx);
  auto MF = createMachineFunction(Ctx, Mod);

  struct {
    MCInstrDesc MCID;
    MCOperandInfo OpInfo;
  } Table = {{0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0},
             {0, 0, MCOI::OPERAND_REGISTER, 0}};

  DIFile *DIF = DIFile::getDistinct(Ctx, "filename", "");
  DISubprogram *DIS = DISubprogram::getDistinct(
      Ctx, nullptr, "", "", DIF, 0, nullptr, 0, nullptr, 0, 0, DINode::FlagZero,
      DISubprogram::SPFlagZero, nullptr);
  DILocation *DIL = DILocation::get(Ctx, 1, 5, DIS);
  DebugLoc DL(DIL);
  MachineInstr *MI = MF->CreateMachineInstr(Table.MCID, DL);
  MI->addOperand(*MF, MachineOperand::CreateReg(0, /*isDef*/ true));

  std::string str;
  raw_string_ostream OS(str);
  MI->print(OS, /*IsStandalone*/true, /*SkipOpers*/false, /*SkipDebugLoc*/false,
            /*AddNewLine*/false);
  ASSERT_TRUE(
      StringRef(OS.str()).starts_with("$noreg = UNKNOWN debug-location "));
  ASSERT_TRUE(StringRef(OS.str()).ends_with("filename:1:5"));
}

TEST(MachineInstrSpan, DistanceBegin) {
  LLVMContext Ctx;
  Module Mod("Module", Ctx);
  auto MF = createMachineFunction(Ctx, Mod);
  auto MBB = MF->CreateMachineBasicBlock();

  MCInstrDesc MCID = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  auto MII = MBB->begin();
  MachineInstrSpan MIS(MII, MBB);
  ASSERT_TRUE(MIS.empty());

  auto MI = MF->CreateMachineInstr(MCID, DebugLoc());
  MBB->insert(MII, MI);
  ASSERT_TRUE(std::distance(MIS.begin(), MII) == 1);
}

TEST(MachineInstrSpan, DistanceEnd) {
  LLVMContext Ctx;
  Module Mod("Module", Ctx);
  auto MF = createMachineFunction(Ctx, Mod);
  auto MBB = MF->CreateMachineBasicBlock();

  MCInstrDesc MCID = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  auto MII = MBB->end();
  MachineInstrSpan MIS(MII, MBB);
  ASSERT_TRUE(MIS.empty());

  auto MI = MF->CreateMachineInstr(MCID, DebugLoc());
  MBB->insert(MII, MI);
  ASSERT_TRUE(std::distance(MIS.begin(), MII) == 1);
}

TEST(MachineInstrExtraInfo, AddExtraInfo) {
  LLVMContext Ctx;
  Module Mod("Module", Ctx);
  auto MF = createMachineFunction(Ctx, Mod);
  MCInstrDesc MCID = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  auto MI = MF->CreateMachineInstr(MCID, DebugLoc());
  auto MAI = MCAsmInfo();
  auto MC = createMCContext(&MAI);
  auto MMO = MF->getMachineMemOperand(MachinePointerInfo(),
                                      MachineMemOperand::MOLoad, 8, Align(8));
  SmallVector<MachineMemOperand *, 2> MMOs;
  MMOs.push_back(MMO);
  MCSymbol *Sym1 = MC->createTempSymbol("pre_label", false);
  MCSymbol *Sym2 = MC->createTempSymbol("post_label", false);
  MDNode *HAM = MDNode::getDistinct(Ctx, std::nullopt);
  MDNode *PCS = MDNode::getDistinct(Ctx, std::nullopt);
  MDNode *MMRA = MMRAMetadata().addTag("foo", "bar").getAsMD(Ctx);

  ASSERT_TRUE(MI->memoperands_empty());
  ASSERT_FALSE(MI->getPreInstrSymbol());
  ASSERT_FALSE(MI->getPostInstrSymbol());
  ASSERT_FALSE(MI->getHeapAllocMarker());
  ASSERT_FALSE(MI->getPCSections());
  ASSERT_FALSE(MI->getMMRAMetadata());

  MI->setMemRefs(*MF, MMOs);
  ASSERT_TRUE(MI->memoperands().size() == 1);
  ASSERT_FALSE(MI->getPreInstrSymbol());
  ASSERT_FALSE(MI->getPostInstrSymbol());
  ASSERT_FALSE(MI->getHeapAllocMarker());
  ASSERT_FALSE(MI->getPCSections());
  ASSERT_FALSE(MI->getMMRAMetadata());

  MI->setPreInstrSymbol(*MF, Sym1);
  ASSERT_TRUE(MI->memoperands().size() == 1);
  ASSERT_TRUE(MI->getPreInstrSymbol() == Sym1);
  ASSERT_FALSE(MI->getPostInstrSymbol());
  ASSERT_FALSE(MI->getHeapAllocMarker());
  ASSERT_FALSE(MI->getPCSections());
  ASSERT_FALSE(MI->getMMRAMetadata());

  MI->setPostInstrSymbol(*MF, Sym2);
  ASSERT_TRUE(MI->memoperands().size() == 1);
  ASSERT_TRUE(MI->getPreInstrSymbol() == Sym1);
  ASSERT_TRUE(MI->getPostInstrSymbol() == Sym2);
  ASSERT_FALSE(MI->getHeapAllocMarker());
  ASSERT_FALSE(MI->getPCSections());
  ASSERT_FALSE(MI->getMMRAMetadata());

  MI->setHeapAllocMarker(*MF, HAM);
  ASSERT_TRUE(MI->memoperands().size() == 1);
  ASSERT_TRUE(MI->getPreInstrSymbol() == Sym1);
  ASSERT_TRUE(MI->getPostInstrSymbol() == Sym2);
  ASSERT_TRUE(MI->getHeapAllocMarker() == HAM);
  ASSERT_FALSE(MI->getPCSections());
  ASSERT_FALSE(MI->getMMRAMetadata());

  MI->setPCSections(*MF, PCS);
  ASSERT_TRUE(MI->memoperands().size() == 1);
  ASSERT_TRUE(MI->getPreInstrSymbol() == Sym1);
  ASSERT_TRUE(MI->getPostInstrSymbol() == Sym2);
  ASSERT_TRUE(MI->getHeapAllocMarker() == HAM);
  ASSERT_TRUE(MI->getPCSections() == PCS);
  ASSERT_FALSE(MI->getMMRAMetadata());

  MI->setMMRAMetadata(*MF, MMRA);
  ASSERT_TRUE(MI->memoperands().size() == 1);
  ASSERT_TRUE(MI->getPreInstrSymbol() == Sym1);
  ASSERT_TRUE(MI->getPostInstrSymbol() == Sym2);
  ASSERT_TRUE(MI->getHeapAllocMarker() == HAM);
  ASSERT_TRUE(MI->getPCSections() == PCS);
  ASSERT_TRUE(MI->getMMRAMetadata() == MMRA);

  // Check with nothing but MMRAs.
  MachineInstr *MMRAMI = MF->CreateMachineInstr(MCID, DebugLoc());
  ASSERT_FALSE(MMRAMI->getMMRAMetadata());
  MMRAMI->setMMRAMetadata(*MF, MMRA);
  ASSERT_TRUE(MMRAMI->getMMRAMetadata() == MMRA);
}

TEST(MachineInstrExtraInfo, ChangeExtraInfo) {
  LLVMContext Ctx;
  Module Mod("Module", Ctx);
  auto MF = createMachineFunction(Ctx, Mod);
  MCInstrDesc MCID = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  auto MI = MF->CreateMachineInstr(MCID, DebugLoc());
  auto MAI = MCAsmInfo();
  auto MC = createMCContext(&MAI);
  auto MMO = MF->getMachineMemOperand(MachinePointerInfo(),
                                      MachineMemOperand::MOLoad, 8, Align(8));
  SmallVector<MachineMemOperand *, 2> MMOs;
  MMOs.push_back(MMO);
  MCSymbol *Sym1 = MC->createTempSymbol("pre_label", false);
  MCSymbol *Sym2 = MC->createTempSymbol("post_label", false);
  MDNode *HAM = MDNode::getDistinct(Ctx, std::nullopt);
  MDNode *PCS = MDNode::getDistinct(Ctx, std::nullopt);

  MDNode *MMRA1 = MMRAMetadata().addTag("foo", "bar").getAsMD(Ctx);
  MDNode *MMRA2 = MMRAMetadata().addTag("bar", "bux").getAsMD(Ctx);

  MI->setMemRefs(*MF, MMOs);
  MI->setPreInstrSymbol(*MF, Sym1);
  MI->setPostInstrSymbol(*MF, Sym2);
  MI->setHeapAllocMarker(*MF, HAM);
  MI->setPCSections(*MF, PCS);
  MI->setMMRAMetadata(*MF, MMRA1);

  MMOs.push_back(MMO);

  MI->setMemRefs(*MF, MMOs);
  ASSERT_TRUE(MI->memoperands().size() == 2);
  ASSERT_TRUE(MI->getPreInstrSymbol() == Sym1);
  ASSERT_TRUE(MI->getPostInstrSymbol() == Sym2);
  ASSERT_TRUE(MI->getHeapAllocMarker() == HAM);
  ASSERT_TRUE(MI->getPCSections() == PCS);
  ASSERT_TRUE(MI->getMMRAMetadata() == MMRA1);

  MI->setPostInstrSymbol(*MF, Sym1);
  ASSERT_TRUE(MI->memoperands().size() == 2);
  ASSERT_TRUE(MI->getPreInstrSymbol() == Sym1);
  ASSERT_TRUE(MI->getPostInstrSymbol() == Sym1);
  ASSERT_TRUE(MI->getHeapAllocMarker() == HAM);
  ASSERT_TRUE(MI->getPCSections() == PCS);
  ASSERT_TRUE(MI->getMMRAMetadata() == MMRA1);

  MI->setMMRAMetadata(*MF, MMRA2);
  ASSERT_TRUE(MI->memoperands().size() == 2);
  ASSERT_TRUE(MI->getPreInstrSymbol() == Sym1);
  ASSERT_TRUE(MI->getPostInstrSymbol() == Sym1);
  ASSERT_TRUE(MI->getHeapAllocMarker() == HAM);
  ASSERT_TRUE(MI->getPCSections() == PCS);
  ASSERT_TRUE(MI->getMMRAMetadata() == MMRA2);
}

TEST(MachineInstrExtraInfo, RemoveExtraInfo) {
  LLVMContext Ctx;
  Module Mod("Module", Ctx);
  auto MF = createMachineFunction(Ctx, Mod);
  MCInstrDesc MCID = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  auto MI = MF->CreateMachineInstr(MCID, DebugLoc());
  auto MAI = MCAsmInfo();
  auto MC = createMCContext(&MAI);
  auto MMO = MF->getMachineMemOperand(MachinePointerInfo(),
                                      MachineMemOperand::MOLoad, 8, Align(8));
  SmallVector<MachineMemOperand *, 2> MMOs;
  MMOs.push_back(MMO);
  MMOs.push_back(MMO);
  MCSymbol *Sym1 = MC->createTempSymbol("pre_label", false);
  MCSymbol *Sym2 = MC->createTempSymbol("post_label", false);
  MDNode *HAM = MDNode::getDistinct(Ctx, std::nullopt);
  MDNode *PCS = MDNode::getDistinct(Ctx, std::nullopt);

  MDNode *MMRA = MMRAMetadata().getAsMD(Ctx);

  MI->setMemRefs(*MF, MMOs);
  MI->setPreInstrSymbol(*MF, Sym1);
  MI->setPostInstrSymbol(*MF, Sym2);
  MI->setHeapAllocMarker(*MF, HAM);
  MI->setPCSections(*MF, PCS);
  MI->setMMRAMetadata(*MF, MMRA);

  MI->setPostInstrSymbol(*MF, nullptr);
  ASSERT_TRUE(MI->memoperands().size() == 2);
  ASSERT_TRUE(MI->getPreInstrSymbol() == Sym1);
  ASSERT_FALSE(MI->getPostInstrSymbol());
  ASSERT_TRUE(MI->getHeapAllocMarker() == HAM);
  ASSERT_TRUE(MI->getPCSections() == PCS);
  ASSERT_TRUE(MI->getMMRAMetadata() == MMRA);

  MI->setHeapAllocMarker(*MF, nullptr);
  ASSERT_TRUE(MI->memoperands().size() == 2);
  ASSERT_TRUE(MI->getPreInstrSymbol() == Sym1);
  ASSERT_FALSE(MI->getPostInstrSymbol());
  ASSERT_FALSE(MI->getHeapAllocMarker());
  ASSERT_TRUE(MI->getPCSections() == PCS);
  ASSERT_TRUE(MI->getMMRAMetadata() == MMRA);

  MI->setPCSections(*MF, nullptr);
  ASSERT_TRUE(MI->memoperands().size() == 2);
  ASSERT_TRUE(MI->getPreInstrSymbol() == Sym1);
  ASSERT_FALSE(MI->getPostInstrSymbol());
  ASSERT_FALSE(MI->getHeapAllocMarker());
  ASSERT_FALSE(MI->getPCSections());
  ASSERT_TRUE(MI->getMMRAMetadata() == MMRA);

  MI->setPreInstrSymbol(*MF, nullptr);
  ASSERT_TRUE(MI->memoperands().size() == 2);
  ASSERT_FALSE(MI->getPreInstrSymbol());
  ASSERT_FALSE(MI->getPostInstrSymbol());
  ASSERT_FALSE(MI->getHeapAllocMarker());
  ASSERT_FALSE(MI->getPCSections());
  ASSERT_TRUE(MI->getMMRAMetadata() == MMRA);

  MI->setMemRefs(*MF, {});
  ASSERT_TRUE(MI->memoperands_empty());
  ASSERT_FALSE(MI->getPreInstrSymbol());
  ASSERT_FALSE(MI->getPostInstrSymbol());
  ASSERT_FALSE(MI->getHeapAllocMarker());
  ASSERT_FALSE(MI->getPCSections());
  ASSERT_TRUE(MI->getMMRAMetadata() == MMRA);

  MI->setMMRAMetadata(*MF, nullptr);
  ASSERT_TRUE(MI->memoperands_empty());
  ASSERT_FALSE(MI->getPreInstrSymbol());
  ASSERT_FALSE(MI->getPostInstrSymbol());
  ASSERT_FALSE(MI->getHeapAllocMarker());
  ASSERT_FALSE(MI->getPCSections());
  ASSERT_FALSE(MI->getMMRAMetadata());
}

TEST(MachineInstrDebugValue, AddDebugValueOperand) {
  LLVMContext Ctx;
  Module Mod("Module", Ctx);
  auto MF = createMachineFunction(Ctx, Mod);

  for (const unsigned short Opcode :
       {TargetOpcode::DBG_VALUE, TargetOpcode::DBG_VALUE_LIST,
        TargetOpcode::DBG_INSTR_REF, TargetOpcode::DBG_PHI,
        TargetOpcode::DBG_LABEL}) {
    const MCInstrDesc MCID = {
        Opcode, 0, 0, 0, 0,
        0,      0, 0, 0, (1ULL << MCID::Pseudo) | (1ULL << MCID::Variadic),
        0};

    auto *MI = MF->CreateMachineInstr(MCID, DebugLoc());
    MI->addOperand(*MF, MachineOperand::CreateReg(0, /*isDef*/ false));

    MI->addOperand(*MF, MachineOperand::CreateImm(0));
    MI->getOperand(1).ChangeToRegister(0, false);

    ASSERT_TRUE(MI->getOperand(0).isDebug());
    ASSERT_TRUE(MI->getOperand(1).isDebug());
  }
}

MATCHER_P(HasMIMetadata, MIMD, "") {
  return arg->getDebugLoc() == MIMD.getDL() &&
         arg->getPCSections() == MIMD.getPCSections();
}

TEST(MachineInstrBuilder, BuildMI) {
  LLVMContext Ctx;
  MDNode *PCS = MDNode::getDistinct(Ctx, std::nullopt);
  MDNode *DI = MDNode::getDistinct(Ctx, std::nullopt);
  DebugLoc DL(DI);
  MIMetadata MIMD(DL, PCS);
  EXPECT_EQ(MIMD.getDL(), DL);
  EXPECT_EQ(MIMD.getPCSections(), PCS);
  // Check common BuildMI() overloads propagate MIMetadata.
  Module Mod("Module", Ctx);
  auto MF = createMachineFunction(Ctx, Mod);
  auto MBB = MF->CreateMachineBasicBlock();
  MCInstrDesc MCID = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  EXPECT_THAT(BuildMI(*MF, MIMD, MCID), HasMIMetadata(MIMD));
  EXPECT_THAT(BuildMI(*MF, MIMD, MCID), HasMIMetadata(MIMD));
  EXPECT_THAT(BuildMI(*MBB, MBB->end(), MIMD, MCID), HasMIMetadata(MIMD));
  EXPECT_THAT(BuildMI(*MBB, MBB->end(), MIMD, MCID), HasMIMetadata(MIMD));
  EXPECT_THAT(BuildMI(*MBB, MBB->instr_end(), MIMD, MCID), HasMIMetadata(MIMD));
  EXPECT_THAT(BuildMI(*MBB, *MBB->begin(), MIMD, MCID), HasMIMetadata(MIMD));
  EXPECT_THAT(BuildMI(*MBB, &*MBB->begin(), MIMD, MCID), HasMIMetadata(MIMD));
  EXPECT_THAT(BuildMI(MBB, MIMD, MCID), HasMIMetadata(MIMD));
}

static_assert(std::is_trivially_copyable_v<MCOperand>, "trivially copyable");

TEST(MachineInstrTest, SpliceOperands) {
  LLVMContext Ctx;
  Module Mod("Module", Ctx);
  std::unique_ptr<MachineFunction> MF = createMachineFunction(Ctx, Mod);
  MachineBasicBlock *MBB = MF->CreateMachineBasicBlock();
  MCInstrDesc MCID = {TargetOpcode::INLINEASM,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      (1ULL << MCID::Pseudo) | (1ULL << MCID::Variadic),
                      0};
  MachineInstr *MI = MF->CreateMachineInstr(MCID, DebugLoc());
  MBB->insert(MBB->begin(), MI);
  MI->addOperand(MachineOperand::CreateImm(0));
  MI->addOperand(MachineOperand::CreateImm(1));
  MI->addOperand(MachineOperand::CreateImm(2));
  MI->addOperand(MachineOperand::CreateImm(3));
  MI->addOperand(MachineOperand::CreateImm(4));

  MI->removeOperand(1);
  EXPECT_EQ(MI->getOperand(1).getImm(), MachineOperand::CreateImm(2).getImm());
  EXPECT_EQ(MI->getNumOperands(), 4U);

  MachineOperand Ops[] = {
      MachineOperand::CreateImm(42),   MachineOperand::CreateImm(1024),
      MachineOperand::CreateImm(2048), MachineOperand::CreateImm(4096),
      MachineOperand::CreateImm(8192),
  };
  auto *It = MI->operands_begin();
  ++It;
  MI->insert(It, Ops);

  EXPECT_EQ(MI->getNumOperands(), 9U);
  EXPECT_EQ(MI->getOperand(0).getImm(), MachineOperand::CreateImm(0).getImm());
  EXPECT_EQ(MI->getOperand(1).getImm(), MachineOperand::CreateImm(42).getImm());
  EXPECT_EQ(MI->getOperand(2).getImm(),
            MachineOperand::CreateImm(1024).getImm());
  EXPECT_EQ(MI->getOperand(3).getImm(),
            MachineOperand::CreateImm(2048).getImm());
  EXPECT_EQ(MI->getOperand(4).getImm(),
            MachineOperand::CreateImm(4096).getImm());
  EXPECT_EQ(MI->getOperand(5).getImm(),
            MachineOperand::CreateImm(8192).getImm());
  EXPECT_EQ(MI->getOperand(6).getImm(), MachineOperand::CreateImm(2).getImm());
  EXPECT_EQ(MI->getOperand(7).getImm(), MachineOperand::CreateImm(3).getImm());
  EXPECT_EQ(MI->getOperand(8).getImm(), MachineOperand::CreateImm(4).getImm());

  // test tied operands
  MCRegisterClass MRC{
      0, 0, 0, 0, 0, 0, 0, 0, /*Allocatable=*/true, /*BaseClass=*/true};
  TargetRegisterClass RC{&MRC, 0, 0, {}, 0, 0, 0, 0, 0, 0, 0};
  // MachineRegisterInfo will be very upset if these registers aren't
  // allocatable.
  assert(RC.isAllocatable() && "unusable TargetRegisterClass");
  MachineRegisterInfo &MRI = MF->getRegInfo();
  Register A = MRI.createVirtualRegister(&RC);
  Register B = MRI.createVirtualRegister(&RC);
  MI->getOperand(0).ChangeToRegister(A, /*isDef=*/true);
  MI->getOperand(1).ChangeToRegister(B, /*isDef=*/false);
  MI->tieOperands(0, 1);
  EXPECT_TRUE(MI->getOperand(0).isTied());
  EXPECT_TRUE(MI->getOperand(1).isTied());
  EXPECT_EQ(MI->findTiedOperandIdx(0), 1U);
  EXPECT_EQ(MI->findTiedOperandIdx(1), 0U);
  MI->insert(&MI->getOperand(1), {MachineOperand::CreateImm(7)});
  EXPECT_TRUE(MI->getOperand(0).isTied());
  EXPECT_TRUE(MI->getOperand(1).isImm());
  EXPECT_TRUE(MI->getOperand(2).isTied());
  EXPECT_EQ(MI->findTiedOperandIdx(0), 2U);
  EXPECT_EQ(MI->findTiedOperandIdx(2), 0U);
  EXPECT_EQ(MI->getOperand(0).getReg(), A);
  EXPECT_EQ(MI->getOperand(2).getReg(), B);

  // bad inputs
  EXPECT_EQ(MI->getNumOperands(), 10U);
  MI->insert(MI->operands_begin(), {});
  EXPECT_EQ(MI->getNumOperands(), 10U);
}

} // end namespace
