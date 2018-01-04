//===- MachineInstrTest.cpp -----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {
// Add a few Bogus backend classes so we can create MachineInstrs without
// depending on a real target.
class BogusTargetLowering : public TargetLowering {
public:
  BogusTargetLowering(TargetMachine &TM) : TargetLowering(TM) {}
};

class BogusFrameLowering : public TargetFrameLowering {
public:
  BogusFrameLowering()
      : TargetFrameLowering(TargetFrameLowering::StackGrowsDown, 4, 4) {}

  void emitPrologue(MachineFunction &MF,
                    MachineBasicBlock &MBB) const override {}
  void emitEpilogue(MachineFunction &MF,
                    MachineBasicBlock &MBB) const override {}
  bool hasFP(const MachineFunction &MF) const override { return false; }
};

class BogusSubtarget : public TargetSubtargetInfo {
public:
  BogusSubtarget(TargetMachine &TM)
      : TargetSubtargetInfo(Triple(""), "", "", {}, {}, nullptr, nullptr,
                            nullptr, nullptr, nullptr, nullptr, nullptr),
        FL(), TL(TM) {}
  ~BogusSubtarget() override {}

  const TargetFrameLowering *getFrameLowering() const override { return &FL; }

  const TargetLowering *getTargetLowering() const override { return &TL; }

  const TargetInstrInfo *getInstrInfo() const override { return &TII; }

private:
  BogusFrameLowering FL;
  BogusTargetLowering TL;
  TargetInstrInfo TII;
};

class BogusTargetMachine : public LLVMTargetMachine {
public:
  BogusTargetMachine()
      : LLVMTargetMachine(Target(), "", Triple(""), "", "", TargetOptions(),
                          Reloc::Static, CodeModel::Small, CodeGenOpt::Default),
        ST(*this) {}
  ~BogusTargetMachine() override {}

  const TargetSubtargetInfo *getSubtargetImpl(const Function &) const override {
    return &ST;
  }

private:
  BogusSubtarget ST;
};

std::unique_ptr<BogusTargetMachine> createTargetMachine() {
  return llvm::make_unique<BogusTargetMachine>();
}

std::unique_ptr<MachineFunction> createMachineFunction() {
  LLVMContext Ctx;
  Module M("Module", Ctx);
  auto Type = FunctionType::get(Type::getVoidTy(Ctx), false);
  auto F = Function::Create(Type, GlobalValue::ExternalLinkage, "Test", &M);

  auto TM = createTargetMachine();
  unsigned FunctionNum = 42;
  MachineModuleInfo MMI(TM.get());
  const TargetSubtargetInfo &STI = *TM->getSubtargetImpl(*F);

  return llvm::make_unique<MachineFunction>(*F, *TM, STI, FunctionNum, MMI);
}

// This test makes sure that MachineInstr::isIdenticalTo handles Defs correctly
// for various combinations of IgnoreDefs, and also that it is symmetrical.
TEST(IsIdenticalToTest, DifferentDefs) {
  auto MF = createMachineFunction();

  unsigned short NumOps = 2;
  unsigned char NumDefs = 1;
  MCOperandInfo OpInfo[] = {
      {0, 0, MCOI::OPERAND_REGISTER, 0},
      {0, 1 << MCOI::OptionalDef, MCOI::OPERAND_REGISTER, 0}};
  MCInstrDesc MCID = {
      0, NumOps,  NumDefs, 0,      0, 1ULL << MCID::HasOptionalDef,
      0, nullptr, nullptr, OpInfo, 0, nullptr};

  // Create two MIs with different virtual reg defs and the same uses.
  unsigned VirtualDef1 = -42; // The value doesn't matter, but the sign does.
  unsigned VirtualDef2 = -43;
  unsigned VirtualUse = -44;

  auto MI1 = MF->CreateMachineInstr(MCID, DebugLoc());
  MI1->addOperand(*MF, MachineOperand::CreateReg(VirtualDef1, /*isDef*/ true));
  MI1->addOperand(*MF, MachineOperand::CreateReg(VirtualUse, /*isDef*/ false));

  auto MI2 = MF->CreateMachineInstr(MCID, DebugLoc());
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

  auto MI3 = MF->CreateMachineInstr(MCID, DebugLoc());
  MI3->addOperand(*MF, MachineOperand::CreateReg(VirtualDef1, /*isDef*/ true));
  MI3->addOperand(*MF, MachineOperand::CreateReg(SentinelReg, /*isDef*/ true));

  auto MI4 = MF->CreateMachineInstr(MCID, DebugLoc());
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
  auto MF = createMachineFunction();

  unsigned short NumOps = 2;
  unsigned char NumDefs = 1;
  MCOperandInfo OpInfo[] = {
      {0, 0, MCOI::OPERAND_REGISTER, 0},
      {0, 1 << MCOI::OptionalDef, MCOI::OPERAND_REGISTER, 0}};
  MCInstrDesc MCID = {
      0, NumOps,  NumDefs, 0,      0, 1ULL << MCID::HasOptionalDef,
      0, nullptr, nullptr, OpInfo, 0, nullptr};

  // Define a series of instructions with different kinds of operands and make
  // sure that the hash function is consistent with isEqual for various
  // combinations of them.
  unsigned VirtualDef1 = -42;
  unsigned VirtualDef2 = -43;
  unsigned VirtualReg = -44;
  unsigned SentinelReg = 0;
  unsigned PhysicalReg = 45;

  auto VD1VU = MF->CreateMachineInstr(MCID, DebugLoc());
  VD1VU->addOperand(*MF,
                    MachineOperand::CreateReg(VirtualDef1, /*isDef*/ true));
  VD1VU->addOperand(*MF,
                    MachineOperand::CreateReg(VirtualReg, /*isDef*/ false));

  auto VD2VU = MF->CreateMachineInstr(MCID, DebugLoc());
  VD2VU->addOperand(*MF,
                    MachineOperand::CreateReg(VirtualDef2, /*isDef*/ true));
  VD2VU->addOperand(*MF,
                    MachineOperand::CreateReg(VirtualReg, /*isDef*/ false));

  auto VD1SU = MF->CreateMachineInstr(MCID, DebugLoc());
  VD1SU->addOperand(*MF,
                    MachineOperand::CreateReg(VirtualDef1, /*isDef*/ true));
  VD1SU->addOperand(*MF,
                    MachineOperand::CreateReg(SentinelReg, /*isDef*/ false));

  auto VD1SD = MF->CreateMachineInstr(MCID, DebugLoc());
  VD1SD->addOperand(*MF,
                    MachineOperand::CreateReg(VirtualDef1, /*isDef*/ true));
  VD1SD->addOperand(*MF,
                    MachineOperand::CreateReg(SentinelReg, /*isDef*/ true));

  auto VD2PU = MF->CreateMachineInstr(MCID, DebugLoc());
  VD2PU->addOperand(*MF,
                    MachineOperand::CreateReg(VirtualDef2, /*isDef*/ true));
  VD2PU->addOperand(*MF,
                    MachineOperand::CreateReg(PhysicalReg, /*isDef*/ false));

  auto VD2PD = MF->CreateMachineInstr(MCID, DebugLoc());
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

TEST(MachineBasicBlockTest, PhiRange) {
  auto MF = createMachineFunction();

  // Create the main block.
  auto BB = MF->CreateMachineBasicBlock();

  // Create some predecessors of it.
  auto BB1 = MF->CreateMachineBasicBlock();
  BB1->addSuccessor(BB);
  auto BB2 = MF->CreateMachineBasicBlock();
  BB2->addSuccessor(BB);

  // Make sure this doesn't crash if there are no phis.
  for (auto &PN : BB->phis()) {
    (void)PN;
    ASSERT_TRUE(false) << "empty block should have no phis";
  }

  // Make it a cycle.
  BB->addSuccessor(BB);

  // Now insert some PHI nodes.
  MCOperandInfo OpInfo[] = { { -1, 0, MCOI::OPERAND_UNKNOWN, 0} };
  MCInstrDesc PHIMCID = {
      TargetOpcode::PHI, 1, 1, 0, 0,
      (1ULL << MCID::Pseudo) | (1ULL << MCID::Variadic), 0,
      nullptr, nullptr, OpInfo, -1, nullptr};
  auto P1 = BuildMI(*BB, BB->end(), DebugLoc(), PHIMCID, -101);
  auto P2 = BuildMI(*BB, BB->end(), DebugLoc(), PHIMCID, -102);
  auto P3 = BuildMI(*BB, BB->end(), DebugLoc(), PHIMCID, -103);

  // A non-PHI node.
  MCInstrDesc ImpDefMCID = {
      TargetOpcode::IMPLICIT_DEF, 1, 1, 0, 0,
      (1ULL << MCID::Pseudo), 0,
      nullptr, nullptr, OpInfo, -1, nullptr};
  BuildMI(*BB, BB->end(), DebugLoc(), ImpDefMCID, -104);

  // Now wire up the incoming values that are interesting.
  P1.addReg(-102).addMBB(BB);
  P2.addReg(-101).addMBB(BB);
  P3.addReg(-104).addMBB(BB);

  // Finally, let's iterate them, which is the thing we're trying to test.
  // We'll use this to wire up the rest of the incoming values.
  for (auto &PN : BB->phis()) {
    EXPECT_TRUE(PN.isPHI());
    PN.addOperand(*MF, MachineOperand::CreateReg(-100, /*isDef*/ false));
    PN.addOperand(*MF, MachineOperand::CreateMBB(BB1));
    PN.addOperand(*MF, MachineOperand::CreateReg(-100, /*isDef*/ false));
    PN.addOperand(*MF, MachineOperand::CreateMBB(BB2));
  }

  // Test that we can use const iterators and generally that the iterators
  // behave like iterators.
  MachineBasicBlock::const_iterator CI;
  CI = BB->phis().begin();
  EXPECT_NE(CI, BB->phis().end());

  // And iterate a const range.
  for (const auto &PN : const_cast<const MachineBasicBlock *>(BB)->phis()) {
    EXPECT_EQ(BB, PN.getOperand(2).getMBB());
    EXPECT_EQ(BB1, PN.getOperand(4).getMBB());
    EXPECT_EQ(BB2, PN.getOperand(6).getMBB());
  }
}
} // end namespace
