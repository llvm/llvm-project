//===- AArch64ExpandPseudoInsts.cpp - Expand pseudo instructions ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass that expands pseudo instructions into target
// instructions to allow proper scheduling and other late optimizations.  This
// pass should be run after register allocation but before the post-regalloc
// scheduling pass.
//
//===----------------------------------------------------------------------===//

#include "AArch64ExpandImm.h"
#include "AArch64InstrInfo.h"
#include "AArch64MachineFunctionInfo.h"
#include "AArch64Subtarget.h"
#include "MCTargetDesc/AArch64AddressingModes.h"
#include "Utils/AArch64BaseInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Triple.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/Pass.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Target/TargetMachine.h"
#include <cassert>
#include <cstdint>
#include <iterator>
#include <limits>
#include <utility>

using namespace llvm;

#define AARCH64_EXPAND_PSEUDO_NAME "AArch64 pseudo instruction expansion pass"

namespace {

class AArch64ExpandPseudo : public MachineFunctionPass {
public:
  const AArch64InstrInfo *TII;

  static char ID;

  AArch64ExpandPseudo() : MachineFunctionPass(ID) {
    initializeAArch64ExpandPseudoPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &Fn) override;

  StringRef getPassName() const override { return AARCH64_EXPAND_PSEUDO_NAME; }

private:
  bool expandMBB(MachineBasicBlock &MBB);
  bool expandMI(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                MachineBasicBlock::iterator &NextMBBI);
  bool expandMOVImm(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                    unsigned BitSize);

  bool expandCMP_SWAP(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                      unsigned LdarOp, unsigned StlrOp, unsigned CmpOp,
                      unsigned ExtendImm, unsigned ZeroReg,
                      MachineBasicBlock::iterator &NextMBBI);
  bool expandCMP_SWAP_128(MachineBasicBlock &MBB,
                          MachineBasicBlock::iterator MBBI,
                          MachineBasicBlock::iterator &NextMBBI);
  bool expandSetTagLoop(MachineBasicBlock &MBB,
                        MachineBasicBlock::iterator MBBI,
                        MachineBasicBlock::iterator &NextMBBI);
};

} // end anonymous namespace

char AArch64ExpandPseudo::ID = 0;

INITIALIZE_PASS(AArch64ExpandPseudo, "aarch64-expand-pseudo",
                AARCH64_EXPAND_PSEUDO_NAME, false, false)

/// Transfer implicit operands on the pseudo instruction to the
/// instructions created from the expansion.
static void transferImpOps(MachineInstr &OldMI, MachineInstrBuilder &UseMI,
                           MachineInstrBuilder &DefMI) {
  const MCInstrDesc &Desc = OldMI.getDesc();
  for (unsigned i = Desc.getNumOperands(), e = OldMI.getNumOperands(); i != e;
       ++i) {
    const MachineOperand &MO = OldMI.getOperand(i);
    assert(MO.isReg() && MO.getReg());
    if (MO.isUse())
      UseMI.add(MO);
    else
      DefMI.add(MO);
  }
}

/// Expand a MOVi32imm or MOVi64imm pseudo instruction to one or more
/// real move-immediate instructions to synthesize the immediate.
bool AArch64ExpandPseudo::expandMOVImm(MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator MBBI,
                                       unsigned BitSize) {
  MachineInstr &MI = *MBBI;
  Register DstReg = MI.getOperand(0).getReg();
  uint64_t RenamableState =
      MI.getOperand(0).isRenamable() ? RegState::Renamable : 0;
  uint64_t Imm = MI.getOperand(1).getImm();

  if (DstReg == AArch64::XZR || DstReg == AArch64::WZR) {
    // Useless def, and we don't want to risk creating an invalid ORR (which
    // would really write to sp).
    MI.eraseFromParent();
    return true;
  }

  SmallVector<AArch64_IMM::ImmInsnModel, 4> Insn;
  AArch64_IMM::expandMOVImm(Imm, BitSize, Insn);
  assert(Insn.size() != 0);

  SmallVector<MachineInstrBuilder, 4> MIBS;
  for (auto I = Insn.begin(), E = Insn.end(); I != E; ++I) {
    bool LastItem = std::next(I) == E;
    switch (I->Opcode)
    {
    default: llvm_unreachable("unhandled!"); break;

    case AArch64::ORRWri:
    case AArch64::ORRXri:
      MIBS.push_back(BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(I->Opcode))
        .add(MI.getOperand(0))
        .addReg(BitSize == 32 ? AArch64::WZR : AArch64::XZR)
        .addImm(I->Op2));
      break;
    case AArch64::MOVNWi:
    case AArch64::MOVNXi:
    case AArch64::MOVZWi:
    case AArch64::MOVZXi: {
      bool DstIsDead = MI.getOperand(0).isDead();
      MIBS.push_back(BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(I->Opcode))
        .addReg(DstReg, RegState::Define |
                getDeadRegState(DstIsDead && LastItem) |
                RenamableState)
        .addImm(I->Op1)
        .addImm(I->Op2));
      } break;
    case AArch64::MOVKWi:
    case AArch64::MOVKXi: {
      Register DstReg = MI.getOperand(0).getReg();
      bool DstIsDead = MI.getOperand(0).isDead();
      MIBS.push_back(BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(I->Opcode))
        .addReg(DstReg,
                RegState::Define |
                getDeadRegState(DstIsDead && LastItem) |
                RenamableState)
        .addReg(DstReg)
        .addImm(I->Op1)
        .addImm(I->Op2));
      } break;
    }
  }
  transferImpOps(MI, MIBS.front(), MIBS.back());
  MI.eraseFromParent();
  return true;
}

bool AArch64ExpandPseudo::expandCMP_SWAP(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI, unsigned LdarOp,
    unsigned StlrOp, unsigned CmpOp, unsigned ExtendImm, unsigned ZeroReg,
    MachineBasicBlock::iterator &NextMBBI) {
  MachineInstr &MI = *MBBI;
  DebugLoc DL = MI.getDebugLoc();
  const MachineOperand &Dest = MI.getOperand(0);
  Register StatusReg = MI.getOperand(1).getReg();
  bool StatusDead = MI.getOperand(1).isDead();
  // Duplicating undef operands into 2 instructions does not guarantee the same
  // value on both; However undef should be replaced by xzr anyway.
  assert(!MI.getOperand(2).isUndef() && "cannot handle undef");
  Register AddrReg = MI.getOperand(2).getReg();
  Register DesiredReg = MI.getOperand(3).getReg();
  Register NewReg = MI.getOperand(4).getReg();

  MachineFunction *MF = MBB.getParent();
  auto LoadCmpBB = MF->CreateMachineBasicBlock(MBB.getBasicBlock());
  auto StoreBB = MF->CreateMachineBasicBlock(MBB.getBasicBlock());
  auto DoneBB = MF->CreateMachineBasicBlock(MBB.getBasicBlock());

  MF->insert(++MBB.getIterator(), LoadCmpBB);
  MF->insert(++LoadCmpBB->getIterator(), StoreBB);
  MF->insert(++StoreBB->getIterator(), DoneBB);

  // .Lloadcmp:
  //     mov wStatus, 0
  //     ldaxr xDest, [xAddr]
  //     cmp xDest, xDesired
  //     b.ne .Ldone
  if (!StatusDead)
    BuildMI(LoadCmpBB, DL, TII->get(AArch64::MOVZWi), StatusReg)
      .addImm(0).addImm(0);
  BuildMI(LoadCmpBB, DL, TII->get(LdarOp), Dest.getReg())
      .addReg(AddrReg);
  BuildMI(LoadCmpBB, DL, TII->get(CmpOp), ZeroReg)
      .addReg(Dest.getReg(), getKillRegState(Dest.isDead()))
      .addReg(DesiredReg)
      .addImm(ExtendImm);
  BuildMI(LoadCmpBB, DL, TII->get(AArch64::Bcc))
      .addImm(AArch64CC::NE)
      .addMBB(DoneBB)
      .addReg(AArch64::NZCV, RegState::Implicit | RegState::Kill);
  LoadCmpBB->addSuccessor(DoneBB);
  LoadCmpBB->addSuccessor(StoreBB);

  // .Lstore:
  //     stlxr wStatus, xNew, [xAddr]
  //     cbnz wStatus, .Lloadcmp
  BuildMI(StoreBB, DL, TII->get(StlrOp), StatusReg)
      .addReg(NewReg)
      .addReg(AddrReg);
  BuildMI(StoreBB, DL, TII->get(AArch64::CBNZW))
      .addReg(StatusReg, getKillRegState(StatusDead))
      .addMBB(LoadCmpBB);
  StoreBB->addSuccessor(LoadCmpBB);
  StoreBB->addSuccessor(DoneBB);

  DoneBB->splice(DoneBB->end(), &MBB, MI, MBB.end());
  DoneBB->transferSuccessors(&MBB);

  MBB.addSuccessor(LoadCmpBB);

  NextMBBI = MBB.end();
  MI.eraseFromParent();

  // Recompute livein lists.
  LivePhysRegs LiveRegs;
  computeAndAddLiveIns(LiveRegs, *DoneBB);
  computeAndAddLiveIns(LiveRegs, *StoreBB);
  computeAndAddLiveIns(LiveRegs, *LoadCmpBB);
  // Do an extra pass around the loop to get loop carried registers right.
  StoreBB->clearLiveIns();
  computeAndAddLiveIns(LiveRegs, *StoreBB);
  LoadCmpBB->clearLiveIns();
  computeAndAddLiveIns(LiveRegs, *LoadCmpBB);

  return true;
}

bool AArch64ExpandPseudo::expandCMP_SWAP_128(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    MachineBasicBlock::iterator &NextMBBI) {
  MachineInstr &MI = *MBBI;
  DebugLoc DL = MI.getDebugLoc();
  MachineOperand &DestLo = MI.getOperand(0);
  MachineOperand &DestHi = MI.getOperand(1);
  Register StatusReg = MI.getOperand(2).getReg();
  bool StatusDead = MI.getOperand(2).isDead();
  // Duplicating undef operands into 2 instructions does not guarantee the same
  // value on both; However undef should be replaced by xzr anyway.
  assert(!MI.getOperand(3).isUndef() && "cannot handle undef");
  Register AddrReg = MI.getOperand(3).getReg();
  Register DesiredLoReg = MI.getOperand(4).getReg();
  Register DesiredHiReg = MI.getOperand(5).getReg();
  Register NewLoReg = MI.getOperand(6).getReg();
  Register NewHiReg = MI.getOperand(7).getReg();

  MachineFunction *MF = MBB.getParent();
  auto LoadCmpBB = MF->CreateMachineBasicBlock(MBB.getBasicBlock());
  auto StoreBB = MF->CreateMachineBasicBlock(MBB.getBasicBlock());
  auto DoneBB = MF->CreateMachineBasicBlock(MBB.getBasicBlock());

  MF->insert(++MBB.getIterator(), LoadCmpBB);
  MF->insert(++LoadCmpBB->getIterator(), StoreBB);
  MF->insert(++StoreBB->getIterator(), DoneBB);

  // .Lloadcmp:
  //     ldaxp xDestLo, xDestHi, [xAddr]
  //     cmp xDestLo, xDesiredLo
  //     sbcs xDestHi, xDesiredHi
  //     b.ne .Ldone
  BuildMI(LoadCmpBB, DL, TII->get(AArch64::LDAXPX))
      .addReg(DestLo.getReg(), RegState::Define)
      .addReg(DestHi.getReg(), RegState::Define)
      .addReg(AddrReg);
  BuildMI(LoadCmpBB, DL, TII->get(AArch64::SUBSXrs), AArch64::XZR)
      .addReg(DestLo.getReg(), getKillRegState(DestLo.isDead()))
      .addReg(DesiredLoReg)
      .addImm(0);
  BuildMI(LoadCmpBB, DL, TII->get(AArch64::CSINCWr), StatusReg)
    .addUse(AArch64::WZR)
    .addUse(AArch64::WZR)
    .addImm(AArch64CC::EQ);
  BuildMI(LoadCmpBB, DL, TII->get(AArch64::SUBSXrs), AArch64::XZR)
      .addReg(DestHi.getReg(), getKillRegState(DestHi.isDead()))
      .addReg(DesiredHiReg)
      .addImm(0);
  BuildMI(LoadCmpBB, DL, TII->get(AArch64::CSINCWr), StatusReg)
      .addUse(StatusReg, RegState::Kill)
      .addUse(StatusReg, RegState::Kill)
      .addImm(AArch64CC::EQ);
  BuildMI(LoadCmpBB, DL, TII->get(AArch64::CBNZW))
      .addUse(StatusReg, getKillRegState(StatusDead))
      .addMBB(DoneBB);
  LoadCmpBB->addSuccessor(DoneBB);
  LoadCmpBB->addSuccessor(StoreBB);

  // .Lstore:
  //     stlxp wStatus, xNewLo, xNewHi, [xAddr]
  //     cbnz wStatus, .Lloadcmp
  BuildMI(StoreBB, DL, TII->get(AArch64::STLXPX), StatusReg)
      .addReg(NewLoReg)
      .addReg(NewHiReg)
      .addReg(AddrReg);
  BuildMI(StoreBB, DL, TII->get(AArch64::CBNZW))
      .addReg(StatusReg, getKillRegState(StatusDead))
      .addMBB(LoadCmpBB);
  StoreBB->addSuccessor(LoadCmpBB);
  StoreBB->addSuccessor(DoneBB);

  DoneBB->splice(DoneBB->end(), &MBB, MI, MBB.end());
  DoneBB->transferSuccessors(&MBB);

  MBB.addSuccessor(LoadCmpBB);

  NextMBBI = MBB.end();
  MI.eraseFromParent();

  // Recompute liveness bottom up.
  LivePhysRegs LiveRegs;
  computeAndAddLiveIns(LiveRegs, *DoneBB);
  computeAndAddLiveIns(LiveRegs, *StoreBB);
  computeAndAddLiveIns(LiveRegs, *LoadCmpBB);
  // Do an extra pass in the loop to get the loop carried dependencies right.
  StoreBB->clearLiveIns();
  computeAndAddLiveIns(LiveRegs, *StoreBB);
  LoadCmpBB->clearLiveIns();
  computeAndAddLiveIns(LiveRegs, *LoadCmpBB);

  return true;
}

bool AArch64ExpandPseudo::expandSetTagLoop(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    MachineBasicBlock::iterator &NextMBBI) {
  MachineInstr &MI = *MBBI;
  DebugLoc DL = MI.getDebugLoc();
  Register SizeReg = MI.getOperand(2).getReg();
  Register AddressReg = MI.getOperand(3).getReg();

  MachineFunction *MF = MBB.getParent();

  bool ZeroData = MI.getOpcode() == AArch64::STZGloop;
  const unsigned OpCode =
      ZeroData ? AArch64::STZ2GPostIndex : AArch64::ST2GPostIndex;

  auto LoopBB = MF->CreateMachineBasicBlock(MBB.getBasicBlock());
  auto DoneBB = MF->CreateMachineBasicBlock(MBB.getBasicBlock());

  MF->insert(++MBB.getIterator(), LoopBB);
  MF->insert(++LoopBB->getIterator(), DoneBB);

  BuildMI(LoopBB, DL, TII->get(OpCode))
      .addDef(AddressReg)
      .addReg(AddressReg)
      .addReg(AddressReg)
      .addImm(2)
      .cloneMemRefs(MI)
      .setMIFlags(MI.getFlags());
  BuildMI(LoopBB, DL, TII->get(AArch64::SUBXri))
      .addDef(SizeReg)
      .addReg(SizeReg)
      .addImm(16 * 2)
      .addImm(0);
  BuildMI(LoopBB, DL, TII->get(AArch64::CBNZX)).addUse(SizeReg).addMBB(LoopBB);

  LoopBB->addSuccessor(LoopBB);
  LoopBB->addSuccessor(DoneBB);

  DoneBB->splice(DoneBB->end(), &MBB, MI, MBB.end());
  DoneBB->transferSuccessors(&MBB);

  MBB.addSuccessor(LoopBB);

  NextMBBI = MBB.end();
  MI.eraseFromParent();
  // Recompute liveness bottom up.
  LivePhysRegs LiveRegs;
  computeAndAddLiveIns(LiveRegs, *DoneBB);
  computeAndAddLiveIns(LiveRegs, *LoopBB);
  // Do an extra pass in the loop to get the loop carried dependencies right.
  // FIXME: is this necessary?
  LoopBB->clearLiveIns();
  computeAndAddLiveIns(LiveRegs, *LoopBB);
  DoneBB->clearLiveIns();
  computeAndAddLiveIns(LiveRegs, *DoneBB);

  return true;
}

/// If MBBI references a pseudo instruction that should be expanded here,
/// do the expansion and return true.  Otherwise return false.
bool AArch64ExpandPseudo::expandMI(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MBBI,
                                   MachineBasicBlock::iterator &NextMBBI) {
  MachineInstr &MI = *MBBI;
  unsigned Opcode = MI.getOpcode();
  switch (Opcode) {
  default:
    break;

  case AArch64::ADDWrr:
  case AArch64::SUBWrr:
  case AArch64::ADDXrr:
  case AArch64::SUBXrr:
  case AArch64::ADDSWrr:
  case AArch64::SUBSWrr:
  case AArch64::ADDSXrr:
  case AArch64::SUBSXrr:
  case AArch64::ANDWrr:
  case AArch64::ANDXrr:
  case AArch64::BICWrr:
  case AArch64::BICXrr:
  case AArch64::ANDSWrr:
  case AArch64::ANDSXrr:
  case AArch64::BICSWrr:
  case AArch64::BICSXrr:
  case AArch64::EONWrr:
  case AArch64::EONXrr:
  case AArch64::EORWrr:
  case AArch64::EORXrr:
  case AArch64::ORNWrr:
  case AArch64::ORNXrr:
  case AArch64::ORRWrr:
  case AArch64::ORRXrr: {
    unsigned Opcode;
    switch (MI.getOpcode()) {
    default:
      return false;
    case AArch64::ADDWrr:      Opcode = AArch64::ADDWrs; break;
    case AArch64::SUBWrr:      Opcode = AArch64::SUBWrs; break;
    case AArch64::ADDXrr:      Opcode = AArch64::ADDXrs; break;
    case AArch64::SUBXrr:      Opcode = AArch64::SUBXrs; break;
    case AArch64::ADDSWrr:     Opcode = AArch64::ADDSWrs; break;
    case AArch64::SUBSWrr:     Opcode = AArch64::SUBSWrs; break;
    case AArch64::ADDSXrr:     Opcode = AArch64::ADDSXrs; break;
    case AArch64::SUBSXrr:     Opcode = AArch64::SUBSXrs; break;
    case AArch64::ANDWrr:      Opcode = AArch64::ANDWrs; break;
    case AArch64::ANDXrr:      Opcode = AArch64::ANDXrs; break;
    case AArch64::BICWrr:      Opcode = AArch64::BICWrs; break;
    case AArch64::BICXrr:      Opcode = AArch64::BICXrs; break;
    case AArch64::ANDSWrr:     Opcode = AArch64::ANDSWrs; break;
    case AArch64::ANDSXrr:     Opcode = AArch64::ANDSXrs; break;
    case AArch64::BICSWrr:     Opcode = AArch64::BICSWrs; break;
    case AArch64::BICSXrr:     Opcode = AArch64::BICSXrs; break;
    case AArch64::EONWrr:      Opcode = AArch64::EONWrs; break;
    case AArch64::EONXrr:      Opcode = AArch64::EONXrs; break;
    case AArch64::EORWrr:      Opcode = AArch64::EORWrs; break;
    case AArch64::EORXrr:      Opcode = AArch64::EORXrs; break;
    case AArch64::ORNWrr:      Opcode = AArch64::ORNWrs; break;
    case AArch64::ORNXrr:      Opcode = AArch64::ORNXrs; break;
    case AArch64::ORRWrr:      Opcode = AArch64::ORRWrs; break;
    case AArch64::ORRXrr:      Opcode = AArch64::ORRXrs; break;
    }
    MachineInstrBuilder MIB1 =
        BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(Opcode),
                MI.getOperand(0).getReg())
            .add(MI.getOperand(1))
            .add(MI.getOperand(2))
            .addImm(AArch64_AM::getShifterImm(AArch64_AM::LSL, 0));
    transferImpOps(MI, MIB1, MIB1);
    MI.eraseFromParent();
    return true;
  }

  case AArch64::LOADgot: {
    MachineFunction *MF = MBB.getParent();
    Register DstReg = MI.getOperand(0).getReg();
    const MachineOperand &MO1 = MI.getOperand(1);
    unsigned Flags = MO1.getTargetFlags();

    if (MF->getTarget().getCodeModel() == CodeModel::Tiny) {
      // Tiny codemodel expand to LDR
      MachineInstrBuilder MIB = BuildMI(MBB, MBBI, MI.getDebugLoc(),
                                        TII->get(AArch64::LDRXl), DstReg);

      if (MO1.isGlobal()) {
        MIB.addGlobalAddress(MO1.getGlobal(), 0, Flags);
      } else if (MO1.isSymbol()) {
        MIB.addExternalSymbol(MO1.getSymbolName(), Flags);
      } else {
        assert(MO1.isCPI() &&
               "Only expect globals, externalsymbols, or constant pools");
        MIB.addConstantPoolIndex(MO1.getIndex(), MO1.getOffset(), Flags);
      }
    } else {
      // Small codemodel expand into ADRP + LDR.
      MachineFunction &MF = *MI.getParent()->getParent();
      DebugLoc DL = MI.getDebugLoc();
      MachineInstrBuilder MIB1 =
          BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(AArch64::ADRP), DstReg);

      MachineInstrBuilder MIB2;
      if (MF.getSubtarget<AArch64Subtarget>().isTargetILP32()) {
        auto TRI = MBB.getParent()->getSubtarget().getRegisterInfo();
        unsigned Reg32 = TRI->getSubReg(DstReg, AArch64::sub_32);
        unsigned DstFlags = MI.getOperand(0).getTargetFlags();
        MIB2 = BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(AArch64::LDRWui))
                   .addDef(Reg32)
                   .addReg(DstReg, RegState::Kill)
                   .addReg(DstReg, DstFlags | RegState::Implicit);
      } else {
        unsigned DstReg = MI.getOperand(0).getReg();
        MIB2 = BuildMI(MBB, MBBI, DL, TII->get(AArch64::LDRXui))
                   .add(MI.getOperand(0))
                   .addUse(DstReg, RegState::Kill);
      }

      if (MO1.isGlobal()) {
        MIB1.addGlobalAddress(MO1.getGlobal(), 0, Flags | AArch64II::MO_PAGE);
        MIB2.addGlobalAddress(MO1.getGlobal(), 0,
                              Flags | AArch64II::MO_PAGEOFF | AArch64II::MO_NC);
      } else if (MO1.isSymbol()) {
        MIB1.addExternalSymbol(MO1.getSymbolName(), Flags | AArch64II::MO_PAGE);
        MIB2.addExternalSymbol(MO1.getSymbolName(), Flags |
                                                        AArch64II::MO_PAGEOFF |
                                                        AArch64II::MO_NC);
      } else {
        assert(MO1.isCPI() &&
               "Only expect globals, externalsymbols, or constant pools");
        MIB1.addConstantPoolIndex(MO1.getIndex(), MO1.getOffset(),
                                  Flags | AArch64II::MO_PAGE);
        MIB2.addConstantPoolIndex(MO1.getIndex(), MO1.getOffset(),
                                  Flags | AArch64II::MO_PAGEOFF |
                                      AArch64II::MO_NC);
      }

      transferImpOps(MI, MIB1, MIB2);
    }
    MI.eraseFromParent();
    return true;
  }

  case AArch64::MOVaddr:
  case AArch64::MOVaddrJT:
  case AArch64::MOVaddrCP:
  case AArch64::MOVaddrBA:
  case AArch64::MOVaddrTLS:
  case AArch64::MOVaddrEXT: {
    // Expand into ADRP + ADD.
    Register DstReg = MI.getOperand(0).getReg();
    MachineInstrBuilder MIB1 =
        BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(AArch64::ADRP), DstReg)
            .add(MI.getOperand(1));

    if (MI.getOperand(1).getTargetFlags() & AArch64II::MO_TAGGED) {
      // MO_TAGGED on the page indicates a tagged address. Set the tag now.
      // We do so by creating a MOVK that sets bits 48-63 of the register to
      // (global address + 0x100000000 - PC) >> 48. This assumes that we're in
      // the small code model so we can assume a binary size of <= 4GB, which
      // makes the untagged PC relative offset positive. The binary must also be
      // loaded into address range [0, 2^48). Both of these properties need to
      // be ensured at runtime when using tagged addresses.
      auto Tag = MI.getOperand(1);
      Tag.setTargetFlags(AArch64II::MO_PREL | AArch64II::MO_G3);
      Tag.setOffset(0x100000000);
      BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(AArch64::MOVKXi), DstReg)
          .addReg(DstReg)
          .add(Tag)
          .addImm(48);
    }

    MachineInstrBuilder MIB2 =
        BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(AArch64::ADDXri))
            .add(MI.getOperand(0))
            .addReg(DstReg)
            .add(MI.getOperand(2))
            .addImm(0);

    transferImpOps(MI, MIB1, MIB2);
    MI.eraseFromParent();
    return true;
  }
  case AArch64::ADDlowTLS:
    // Produce a plain ADD
    BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(AArch64::ADDXri))
        .add(MI.getOperand(0))
        .add(MI.getOperand(1))
        .add(MI.getOperand(2))
        .addImm(0);
    MI.eraseFromParent();
    return true;

  case AArch64::MOVbaseTLS: {
    Register DstReg = MI.getOperand(0).getReg();
    auto SysReg = AArch64SysReg::TPIDR_EL0;
    MachineFunction *MF = MBB.getParent();
    if (MF->getTarget().getTargetTriple().isOSFuchsia() &&
        MF->getTarget().getCodeModel() == CodeModel::Kernel)
      SysReg = AArch64SysReg::TPIDR_EL1;
    else if (MF->getSubtarget<AArch64Subtarget>().useEL3ForTP())
      SysReg = AArch64SysReg::TPIDR_EL3;
    else if (MF->getSubtarget<AArch64Subtarget>().useEL2ForTP())
      SysReg = AArch64SysReg::TPIDR_EL2;
    else if (MF->getSubtarget<AArch64Subtarget>().useEL1ForTP())
      SysReg = AArch64SysReg::TPIDR_EL1;
    BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(AArch64::MRS), DstReg)
        .addImm(SysReg);
    MI.eraseFromParent();
    return true;
  }

  case AArch64::MOVi32imm:
    return expandMOVImm(MBB, MBBI, 32);
  case AArch64::MOVi64imm:
    return expandMOVImm(MBB, MBBI, 64);
  case AArch64::RET_ReallyLR: {
    // Hiding the LR use with RET_ReallyLR may lead to extra kills in the
    // function and missing live-ins. We are fine in practice because callee
    // saved register handling ensures the register value is restored before
    // RET, but we need the undef flag here to appease the MachineVerifier
    // liveness checks.
    MachineInstrBuilder MIB =
        BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(AArch64::RET))
          .addReg(AArch64::LR, RegState::Undef);
    transferImpOps(MI, MIB, MIB);
    MI.eraseFromParent();
    return true;
  }
  case AArch64::CMP_SWAP_8:
    return expandCMP_SWAP(MBB, MBBI, AArch64::LDAXRB, AArch64::STLXRB,
                          AArch64::SUBSWrx,
                          AArch64_AM::getArithExtendImm(AArch64_AM::UXTB, 0),
                          AArch64::WZR, NextMBBI);
  case AArch64::CMP_SWAP_16:
    return expandCMP_SWAP(MBB, MBBI, AArch64::LDAXRH, AArch64::STLXRH,
                          AArch64::SUBSWrx,
                          AArch64_AM::getArithExtendImm(AArch64_AM::UXTH, 0),
                          AArch64::WZR, NextMBBI);
  case AArch64::CMP_SWAP_32:
    return expandCMP_SWAP(MBB, MBBI, AArch64::LDAXRW, AArch64::STLXRW,
                          AArch64::SUBSWrs,
                          AArch64_AM::getShifterImm(AArch64_AM::LSL, 0),
                          AArch64::WZR, NextMBBI);
  case AArch64::CMP_SWAP_64:
    return expandCMP_SWAP(MBB, MBBI,
                          AArch64::LDAXRX, AArch64::STLXRX, AArch64::SUBSXrs,
                          AArch64_AM::getShifterImm(AArch64_AM::LSL, 0),
                          AArch64::XZR, NextMBBI);
  case AArch64::CMP_SWAP_128:
    return expandCMP_SWAP_128(MBB, MBBI, NextMBBI);

  case AArch64::AESMCrrTied:
  case AArch64::AESIMCrrTied: {
    MachineInstrBuilder MIB =
    BuildMI(MBB, MBBI, MI.getDebugLoc(),
            TII->get(Opcode == AArch64::AESMCrrTied ? AArch64::AESMCrr :
                                                      AArch64::AESIMCrr))
      .add(MI.getOperand(0))
      .add(MI.getOperand(1));
    transferImpOps(MI, MIB, MIB);
    MI.eraseFromParent();
    return true;
   }
   case AArch64::IRGstack: {
     MachineFunction &MF = *MBB.getParent();
     const AArch64FunctionInfo *AFI = MF.getInfo<AArch64FunctionInfo>();
     const AArch64FrameLowering *TFI =
         MF.getSubtarget<AArch64Subtarget>().getFrameLowering();

     // IRG does not allow immediate offset. getTaggedBasePointerOffset should
     // almost always point to SP-after-prologue; if not, emit a longer
     // instruction sequence.
     int BaseOffset = -AFI->getTaggedBasePointerOffset();
     unsigned FrameReg;
     StackOffset FrameRegOffset = TFI->resolveFrameOffsetReference(
         MF, BaseOffset, false /*isFixed*/, false /*isSVE*/, FrameReg,
         /*PreferFP=*/false,
         /*ForSimm=*/true);
     Register SrcReg = FrameReg;
     if (FrameRegOffset) {
       // Use output register as temporary.
       SrcReg = MI.getOperand(0).getReg();
       emitFrameOffset(MBB, &MI, MI.getDebugLoc(), SrcReg, FrameReg,
                       FrameRegOffset, TII);
     }
     BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(AArch64::IRG))
         .add(MI.getOperand(0))
         .addUse(SrcReg)
         .add(MI.getOperand(2));
     MI.eraseFromParent();
     return true;
   }
   case AArch64::TAGPstack: {
     int64_t Offset = MI.getOperand(2).getImm();
     BuildMI(MBB, MBBI, MI.getDebugLoc(),
             TII->get(Offset >= 0 ? AArch64::ADDG : AArch64::SUBG))
         .add(MI.getOperand(0))
         .add(MI.getOperand(1))
         .addImm(std::abs(Offset))
         .add(MI.getOperand(4));
     MI.eraseFromParent();
     return true;
   }
   case AArch64::STGloop:
   case AArch64::STZGloop:
     return expandSetTagLoop(MBB, MBBI, NextMBBI);
  }
  return false;
}

/// Iterate over the instructions in basic block MBB and expand any
/// pseudo instructions.  Return true if anything was modified.
bool AArch64ExpandPseudo::expandMBB(MachineBasicBlock &MBB) {
  bool Modified = false;

  MachineBasicBlock::iterator MBBI = MBB.begin(), E = MBB.end();
  while (MBBI != E) {
    MachineBasicBlock::iterator NMBBI = std::next(MBBI);
    Modified |= expandMI(MBB, MBBI, NMBBI);
    MBBI = NMBBI;
  }

  return Modified;
}

bool AArch64ExpandPseudo::runOnMachineFunction(MachineFunction &MF) {
  TII = static_cast<const AArch64InstrInfo *>(MF.getSubtarget().getInstrInfo());

  bool Modified = false;
  for (auto &MBB : MF)
    Modified |= expandMBB(MBB);
  return Modified;
}

/// Returns an instance of the pseudo instruction expansion pass.
FunctionPass *llvm::createAArch64ExpandPseudoPass() {
  return new AArch64ExpandPseudo();
}
