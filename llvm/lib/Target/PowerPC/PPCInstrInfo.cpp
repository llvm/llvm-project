//===-- PPCInstrInfo.cpp - PowerPC Instruction Information ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the PowerPC implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "PPCInstrInfo.h"
#include "PPC.h"
#include "PPCInstrBuilder.h"
#include "PPCMachineFunctionInfo.h"
#include "PPCTargetMachine.h"
#include "PPCHazardRecognizers.h"
#include "MCTargetDesc/PPCPredicates.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/STLExtras.h"

#define GET_INSTRINFO_CTOR
#include "PPCGenInstrInfo.inc"

namespace llvm {
extern cl::opt<bool> DisablePPC32RS;
extern cl::opt<bool> DisablePPC64RS;
}

using namespace llvm;

static cl::
opt<bool> DisableCTRLoopAnal("disable-ppc-ctrloop-analysis", cl::Hidden,
            cl::desc("Disable analysis for CTR loops"));

PPCInstrInfo::PPCInstrInfo(PPCTargetMachine &tm)
  : PPCGenInstrInfo(PPC::ADJCALLSTACKDOWN, PPC::ADJCALLSTACKUP),
    TM(tm), RI(*TM.getSubtargetImpl(), *this) {}

/// CreateTargetHazardRecognizer - Return the hazard recognizer to use for
/// this target when scheduling the DAG.
ScheduleHazardRecognizer *PPCInstrInfo::CreateTargetHazardRecognizer(
  const TargetMachine *TM,
  const ScheduleDAG *DAG) const {
  unsigned Directive = TM->getSubtarget<PPCSubtarget>().getDarwinDirective();
  if (Directive == PPC::DIR_440 || Directive == PPC::DIR_A2 ||
      Directive == PPC::DIR_E500mc || Directive == PPC::DIR_E5500) {
    const InstrItineraryData *II = TM->getInstrItineraryData();
    return new PPCScoreboardHazardRecognizer(II, DAG);
  }

  return TargetInstrInfoImpl::CreateTargetHazardRecognizer(TM, DAG);
}

/// CreateTargetPostRAHazardRecognizer - Return the postRA hazard recognizer
/// to use for this target when scheduling the DAG.
ScheduleHazardRecognizer *PPCInstrInfo::CreateTargetPostRAHazardRecognizer(
  const InstrItineraryData *II,
  const ScheduleDAG *DAG) const {
  unsigned Directive = TM.getSubtarget<PPCSubtarget>().getDarwinDirective();

  // Most subtargets use a PPC970 recognizer.
  if (Directive != PPC::DIR_440 && Directive != PPC::DIR_A2 &&
      Directive != PPC::DIR_E500mc && Directive != PPC::DIR_E5500) {
    const TargetInstrInfo *TII = TM.getInstrInfo();
    assert(TII && "No InstrInfo?");

    return new PPCHazardRecognizer970(*TII);
  }

  return new PPCScoreboardHazardRecognizer(II, DAG);
}

// Detect 32 -> 64-bit extensions where we may reuse the low sub-register.
bool PPCInstrInfo::isCoalescableExtInstr(const MachineInstr &MI,
                                         unsigned &SrcReg, unsigned &DstReg,
                                         unsigned &SubIdx) const {
  switch (MI.getOpcode()) {
  default: return false;
  case PPC::EXTSW:
  case PPC::EXTSW_32_64:
    SrcReg = MI.getOperand(1).getReg();
    DstReg = MI.getOperand(0).getReg();
    SubIdx = PPC::sub_32;
    return true;
  }
}

unsigned PPCInstrInfo::isLoadFromStackSlot(const MachineInstr *MI,
                                           int &FrameIndex) const {
  switch (MI->getOpcode()) {
  default: break;
  case PPC::LD:
  case PPC::LWZ:
  case PPC::LFS:
  case PPC::LFD:
    if (MI->getOperand(1).isImm() && !MI->getOperand(1).getImm() &&
        MI->getOperand(2).isFI()) {
      FrameIndex = MI->getOperand(2).getIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  }
  return 0;
}

unsigned PPCInstrInfo::isStoreToStackSlot(const MachineInstr *MI,
                                          int &FrameIndex) const {
  switch (MI->getOpcode()) {
  default: break;
  case PPC::STD:
  case PPC::STW:
  case PPC::STFS:
  case PPC::STFD:
    if (MI->getOperand(1).isImm() && !MI->getOperand(1).getImm() &&
        MI->getOperand(2).isFI()) {
      FrameIndex = MI->getOperand(2).getIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  }
  return 0;
}

// commuteInstruction - We can commute rlwimi instructions, but only if the
// rotate amt is zero.  We also have to munge the immediates a bit.
MachineInstr *
PPCInstrInfo::commuteInstruction(MachineInstr *MI, bool NewMI) const {
  MachineFunction &MF = *MI->getParent()->getParent();

  // Normal instructions can be commuted the obvious way.
  if (MI->getOpcode() != PPC::RLWIMI)
    return TargetInstrInfoImpl::commuteInstruction(MI, NewMI);

  // Cannot commute if it has a non-zero rotate count.
  if (MI->getOperand(3).getImm() != 0)
    return 0;

  // If we have a zero rotate count, we have:
  //   M = mask(MB,ME)
  //   Op0 = (Op1 & ~M) | (Op2 & M)
  // Change this to:
  //   M = mask((ME+1)&31, (MB-1)&31)
  //   Op0 = (Op2 & ~M) | (Op1 & M)

  // Swap op1/op2
  unsigned Reg0 = MI->getOperand(0).getReg();
  unsigned Reg1 = MI->getOperand(1).getReg();
  unsigned Reg2 = MI->getOperand(2).getReg();
  bool Reg1IsKill = MI->getOperand(1).isKill();
  bool Reg2IsKill = MI->getOperand(2).isKill();
  bool ChangeReg0 = false;
  // If machine instrs are no longer in two-address forms, update
  // destination register as well.
  if (Reg0 == Reg1) {
    // Must be two address instruction!
    assert(MI->getDesc().getOperandConstraint(0, MCOI::TIED_TO) &&
           "Expecting a two-address instruction!");
    Reg2IsKill = false;
    ChangeReg0 = true;
  }

  // Masks.
  unsigned MB = MI->getOperand(4).getImm();
  unsigned ME = MI->getOperand(5).getImm();

  if (NewMI) {
    // Create a new instruction.
    unsigned Reg0 = ChangeReg0 ? Reg2 : MI->getOperand(0).getReg();
    bool Reg0IsDead = MI->getOperand(0).isDead();
    return BuildMI(MF, MI->getDebugLoc(), MI->getDesc())
      .addReg(Reg0, RegState::Define | getDeadRegState(Reg0IsDead))
      .addReg(Reg2, getKillRegState(Reg2IsKill))
      .addReg(Reg1, getKillRegState(Reg1IsKill))
      .addImm((ME+1) & 31)
      .addImm((MB-1) & 31);
  }

  if (ChangeReg0)
    MI->getOperand(0).setReg(Reg2);
  MI->getOperand(2).setReg(Reg1);
  MI->getOperand(1).setReg(Reg2);
  MI->getOperand(2).setIsKill(Reg1IsKill);
  MI->getOperand(1).setIsKill(Reg2IsKill);

  // Swap the mask around.
  MI->getOperand(4).setImm((ME+1) & 31);
  MI->getOperand(5).setImm((MB-1) & 31);
  return MI;
}

void PPCInstrInfo::insertNoop(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator MI) const {
  DebugLoc DL;
  BuildMI(MBB, MI, DL, get(PPC::NOP));
}


// Branch analysis.
// Note: If the condition register is set to CTR or CTR8 then this is a
// BDNZ (imm == 1) or BDZ (imm == 0) branch.
bool PPCInstrInfo::AnalyzeBranch(MachineBasicBlock &MBB,MachineBasicBlock *&TBB,
                                 MachineBasicBlock *&FBB,
                                 SmallVectorImpl<MachineOperand> &Cond,
                                 bool AllowModify) const {
  bool isPPC64 = TM.getSubtargetImpl()->isPPC64();

  // If the block has no terminators, it just falls into the block after it.
  MachineBasicBlock::iterator I = MBB.end();
  if (I == MBB.begin())
    return false;
  --I;
  while (I->isDebugValue()) {
    if (I == MBB.begin())
      return false;
    --I;
  }
  if (!isUnpredicatedTerminator(I))
    return false;

  // Get the last instruction in the block.
  MachineInstr *LastInst = I;

  // If there is only one terminator instruction, process it.
  if (I == MBB.begin() || !isUnpredicatedTerminator(--I)) {
    if (LastInst->getOpcode() == PPC::B) {
      if (!LastInst->getOperand(0).isMBB())
        return true;
      TBB = LastInst->getOperand(0).getMBB();
      return false;
    } else if (LastInst->getOpcode() == PPC::BCC) {
      if (!LastInst->getOperand(2).isMBB())
        return true;
      // Block ends with fall-through condbranch.
      TBB = LastInst->getOperand(2).getMBB();
      Cond.push_back(LastInst->getOperand(0));
      Cond.push_back(LastInst->getOperand(1));
      return false;
    } else if (LastInst->getOpcode() == PPC::BDNZ8 ||
               LastInst->getOpcode() == PPC::BDNZ) {
      if (!LastInst->getOperand(0).isMBB())
        return true;
      if (DisableCTRLoopAnal)
        return true;
      TBB = LastInst->getOperand(0).getMBB();
      Cond.push_back(MachineOperand::CreateImm(1));
      Cond.push_back(MachineOperand::CreateReg(isPPC64 ? PPC::CTR8 : PPC::CTR,
                                               true));
      return false;
    } else if (LastInst->getOpcode() == PPC::BDZ8 ||
               LastInst->getOpcode() == PPC::BDZ) {
      if (!LastInst->getOperand(0).isMBB())
        return true;
      if (DisableCTRLoopAnal)
        return true;
      TBB = LastInst->getOperand(0).getMBB();
      Cond.push_back(MachineOperand::CreateImm(0));
      Cond.push_back(MachineOperand::CreateReg(isPPC64 ? PPC::CTR8 : PPC::CTR,
                                               true));
      return false;
    }

    // Otherwise, don't know what this is.
    return true;
  }

  // Get the instruction before it if it's a terminator.
  MachineInstr *SecondLastInst = I;

  // If there are three terminators, we don't know what sort of block this is.
  if (SecondLastInst && I != MBB.begin() &&
      isUnpredicatedTerminator(--I))
    return true;

  // If the block ends with PPC::B and PPC:BCC, handle it.
  if (SecondLastInst->getOpcode() == PPC::BCC &&
      LastInst->getOpcode() == PPC::B) {
    if (!SecondLastInst->getOperand(2).isMBB() ||
        !LastInst->getOperand(0).isMBB())
      return true;
    TBB =  SecondLastInst->getOperand(2).getMBB();
    Cond.push_back(SecondLastInst->getOperand(0));
    Cond.push_back(SecondLastInst->getOperand(1));
    FBB = LastInst->getOperand(0).getMBB();
    return false;
  } else if ((SecondLastInst->getOpcode() == PPC::BDNZ8 ||
              SecondLastInst->getOpcode() == PPC::BDNZ) &&
      LastInst->getOpcode() == PPC::B) {
    if (!SecondLastInst->getOperand(0).isMBB() ||
        !LastInst->getOperand(0).isMBB())
      return true;
    if (DisableCTRLoopAnal)
      return true;
    TBB = SecondLastInst->getOperand(0).getMBB();
    Cond.push_back(MachineOperand::CreateImm(1));
    Cond.push_back(MachineOperand::CreateReg(isPPC64 ? PPC::CTR8 : PPC::CTR,
                                             true));
    FBB = LastInst->getOperand(0).getMBB();
    return false;
  } else if ((SecondLastInst->getOpcode() == PPC::BDZ8 ||
              SecondLastInst->getOpcode() == PPC::BDZ) &&
      LastInst->getOpcode() == PPC::B) {
    if (!SecondLastInst->getOperand(0).isMBB() ||
        !LastInst->getOperand(0).isMBB())
      return true;
    if (DisableCTRLoopAnal)
      return true;
    TBB = SecondLastInst->getOperand(0).getMBB();
    Cond.push_back(MachineOperand::CreateImm(0));
    Cond.push_back(MachineOperand::CreateReg(isPPC64 ? PPC::CTR8 : PPC::CTR,
                                             true));
    FBB = LastInst->getOperand(0).getMBB();
    return false;
  }

  // If the block ends with two PPC:Bs, handle it.  The second one is not
  // executed, so remove it.
  if (SecondLastInst->getOpcode() == PPC::B &&
      LastInst->getOpcode() == PPC::B) {
    if (!SecondLastInst->getOperand(0).isMBB())
      return true;
    TBB = SecondLastInst->getOperand(0).getMBB();
    I = LastInst;
    if (AllowModify)
      I->eraseFromParent();
    return false;
  }

  // Otherwise, can't handle this.
  return true;
}

unsigned PPCInstrInfo::RemoveBranch(MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator I = MBB.end();
  if (I == MBB.begin()) return 0;
  --I;
  while (I->isDebugValue()) {
    if (I == MBB.begin())
      return 0;
    --I;
  }
  if (I->getOpcode() != PPC::B && I->getOpcode() != PPC::BCC &&
      I->getOpcode() != PPC::BDNZ8 && I->getOpcode() != PPC::BDNZ &&
      I->getOpcode() != PPC::BDZ8  && I->getOpcode() != PPC::BDZ)
    return 0;

  // Remove the branch.
  I->eraseFromParent();

  I = MBB.end();

  if (I == MBB.begin()) return 1;
  --I;
  if (I->getOpcode() != PPC::BCC &&
      I->getOpcode() != PPC::BDNZ8 && I->getOpcode() != PPC::BDNZ &&
      I->getOpcode() != PPC::BDZ8  && I->getOpcode() != PPC::BDZ)
    return 1;

  // Remove the branch.
  I->eraseFromParent();
  return 2;
}

unsigned
PPCInstrInfo::InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                           MachineBasicBlock *FBB,
                           const SmallVectorImpl<MachineOperand> &Cond,
                           DebugLoc DL) const {
  // Shouldn't be a fall through.
  assert(TBB && "InsertBranch must not be told to insert a fallthrough");
  assert((Cond.size() == 2 || Cond.size() == 0) &&
         "PPC branch conditions have two components!");

  bool isPPC64 = TM.getSubtargetImpl()->isPPC64();

  // One-way branch.
  if (FBB == 0) {
    if (Cond.empty())   // Unconditional branch
      BuildMI(&MBB, DL, get(PPC::B)).addMBB(TBB);
    else if (Cond[1].getReg() == PPC::CTR || Cond[1].getReg() == PPC::CTR8)
      BuildMI(&MBB, DL, get(Cond[0].getImm() ?
                              (isPPC64 ? PPC::BDNZ8 : PPC::BDNZ) :
                              (isPPC64 ? PPC::BDZ8  : PPC::BDZ))).addMBB(TBB);
    else                // Conditional branch
      BuildMI(&MBB, DL, get(PPC::BCC))
        .addImm(Cond[0].getImm()).addReg(Cond[1].getReg()).addMBB(TBB);
    return 1;
  }

  // Two-way Conditional Branch.
  if (Cond[1].getReg() == PPC::CTR || Cond[1].getReg() == PPC::CTR8)
    BuildMI(&MBB, DL, get(Cond[0].getImm() ?
                            (isPPC64 ? PPC::BDNZ8 : PPC::BDNZ) :
                            (isPPC64 ? PPC::BDZ8  : PPC::BDZ))).addMBB(TBB);
  else
    BuildMI(&MBB, DL, get(PPC::BCC))
      .addImm(Cond[0].getImm()).addReg(Cond[1].getReg()).addMBB(TBB);
  BuildMI(&MBB, DL, get(PPC::B)).addMBB(FBB);
  return 2;
}

void PPCInstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                               MachineBasicBlock::iterator I, DebugLoc DL,
                               unsigned DestReg, unsigned SrcReg,
                               bool KillSrc) const {
  unsigned Opc;
  if (PPC::GPRCRegClass.contains(DestReg, SrcReg))
    Opc = PPC::OR;
  else if (PPC::G8RCRegClass.contains(DestReg, SrcReg))
    Opc = PPC::OR8;
  else if (PPC::F4RCRegClass.contains(DestReg, SrcReg))
    Opc = PPC::FMR;
  else if (PPC::CRRCRegClass.contains(DestReg, SrcReg))
    Opc = PPC::MCRF;
  else if (PPC::VRRCRegClass.contains(DestReg, SrcReg))
    Opc = PPC::VOR;
  else if (PPC::CRBITRCRegClass.contains(DestReg, SrcReg))
    Opc = PPC::CROR;
  else
    llvm_unreachable("Impossible reg-to-reg copy");

  const MCInstrDesc &MCID = get(Opc);
  if (MCID.getNumOperands() == 3)
    BuildMI(MBB, I, DL, MCID, DestReg)
      .addReg(SrcReg).addReg(SrcReg, getKillRegState(KillSrc));
  else
    BuildMI(MBB, I, DL, MCID, DestReg).addReg(SrcReg, getKillRegState(KillSrc));
}

// This function returns true if a CR spill is necessary and false otherwise.
bool
PPCInstrInfo::StoreRegToStackSlot(MachineFunction &MF,
                                  unsigned SrcReg, bool isKill,
                                  int FrameIdx,
                                  const TargetRegisterClass *RC,
                                  SmallVectorImpl<MachineInstr*> &NewMIs) const{
  DebugLoc DL;
  if (PPC::GPRCRegClass.hasSubClassEq(RC)) {
    if (SrcReg != PPC::LR) {
      NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::STW))
                                         .addReg(SrcReg,
                                                 getKillRegState(isKill)),
                                         FrameIdx));
    } else {
      // FIXME: this spills LR immediately to memory in one step.  To do this,
      // we use R11, which we know cannot be used in the prolog/epilog.  This is
      // a hack.
      NewMIs.push_back(BuildMI(MF, DL, get(PPC::MFLR), PPC::R11));
      NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::STW))
                                         .addReg(PPC::R11,
                                                 getKillRegState(isKill)),
                                         FrameIdx));
    }
  } else if (PPC::G8RCRegClass.hasSubClassEq(RC)) {
    if (SrcReg != PPC::LR8) {
      NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::STD))
                                         .addReg(SrcReg,
                                                 getKillRegState(isKill)),
                                         FrameIdx));
    } else {
      // FIXME: this spills LR immediately to memory in one step.  To do this,
      // we use X11, which we know cannot be used in the prolog/epilog.  This is
      // a hack.
      NewMIs.push_back(BuildMI(MF, DL, get(PPC::MFLR8), PPC::X11));
      NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::STD))
                                         .addReg(PPC::X11,
                                                 getKillRegState(isKill)),
                                         FrameIdx));
    }
  } else if (PPC::F8RCRegClass.hasSubClassEq(RC)) {
    NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::STFD))
                                       .addReg(SrcReg,
                                               getKillRegState(isKill)),
                                       FrameIdx));
  } else if (PPC::F4RCRegClass.hasSubClassEq(RC)) {
    NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::STFS))
                                       .addReg(SrcReg,
                                               getKillRegState(isKill)),
                                       FrameIdx));
  } else if (PPC::CRRCRegClass.hasSubClassEq(RC)) {
    if ((!DisablePPC32RS && !TM.getSubtargetImpl()->isPPC64()) ||
        (!DisablePPC64RS && TM.getSubtargetImpl()->isPPC64())) {
      NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::SPILL_CR))
                                         .addReg(SrcReg,
                                                 getKillRegState(isKill)),
                                         FrameIdx));
      return true;
    } else {
      // FIXME: We need a scatch reg here.  The trouble with using R0 is that
      // it's possible for the stack frame to be so big the save location is
      // out of range of immediate offsets, necessitating another register.
      // We hack this on Darwin by reserving R2.  It's probably broken on Linux
      // at the moment.

      bool is64Bit = TM.getSubtargetImpl()->isPPC64();
      // We need to store the CR in the low 4-bits of the saved value.  First,
      // issue a MFCR to save all of the CRBits.
      unsigned ScratchReg = TM.getSubtargetImpl()->isDarwinABI() ?
                              (is64Bit ? PPC::X2 : PPC::R2) :
                              (is64Bit ? PPC::X0 : PPC::R0);
      NewMIs.push_back(BuildMI(MF, DL, get(is64Bit ? PPC::MFCR8pseud :
                                             PPC::MFCRpseud), ScratchReg)
                               .addReg(SrcReg, getKillRegState(isKill)));

      // If the saved register wasn't CR0, shift the bits left so that they are
      // in CR0's slot.
      if (SrcReg != PPC::CR0) {
        unsigned ShiftBits = getPPCRegisterNumbering(SrcReg)*4;
        // rlwinm scratch, scratch, ShiftBits, 0, 31.
        NewMIs.push_back(BuildMI(MF, DL, get(is64Bit ? PPC::RLWINM8 :
                           PPC::RLWINM), ScratchReg)
                       .addReg(ScratchReg).addImm(ShiftBits)
                       .addImm(0).addImm(31));
      }

      NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(is64Bit ?
                                           PPC::STW8 : PPC::STW))
                                         .addReg(ScratchReg,
                                                 getKillRegState(isKill)),
                                         FrameIdx));
    }
  } else if (PPC::CRBITRCRegClass.hasSubClassEq(RC)) {
    // FIXME: We use CRi here because there is no mtcrf on a bit. Since the
    // backend currently only uses CR1EQ as an individual bit, this should
    // not cause any bug. If we need other uses of CR bits, the following
    // code may be invalid.
    unsigned Reg = 0;
    if (SrcReg == PPC::CR0LT || SrcReg == PPC::CR0GT ||
        SrcReg == PPC::CR0EQ || SrcReg == PPC::CR0UN)
      Reg = PPC::CR0;
    else if (SrcReg == PPC::CR1LT || SrcReg == PPC::CR1GT ||
             SrcReg == PPC::CR1EQ || SrcReg == PPC::CR1UN)
      Reg = PPC::CR1;
    else if (SrcReg == PPC::CR2LT || SrcReg == PPC::CR2GT ||
             SrcReg == PPC::CR2EQ || SrcReg == PPC::CR2UN)
      Reg = PPC::CR2;
    else if (SrcReg == PPC::CR3LT || SrcReg == PPC::CR3GT ||
             SrcReg == PPC::CR3EQ || SrcReg == PPC::CR3UN)
      Reg = PPC::CR3;
    else if (SrcReg == PPC::CR4LT || SrcReg == PPC::CR4GT ||
             SrcReg == PPC::CR4EQ || SrcReg == PPC::CR4UN)
      Reg = PPC::CR4;
    else if (SrcReg == PPC::CR5LT || SrcReg == PPC::CR5GT ||
             SrcReg == PPC::CR5EQ || SrcReg == PPC::CR5UN)
      Reg = PPC::CR5;
    else if (SrcReg == PPC::CR6LT || SrcReg == PPC::CR6GT ||
             SrcReg == PPC::CR6EQ || SrcReg == PPC::CR6UN)
      Reg = PPC::CR6;
    else if (SrcReg == PPC::CR7LT || SrcReg == PPC::CR7GT ||
             SrcReg == PPC::CR7EQ || SrcReg == PPC::CR7UN)
      Reg = PPC::CR7;

    return StoreRegToStackSlot(MF, Reg, isKill, FrameIdx,
                               &PPC::CRRCRegClass, NewMIs);

  } else if (PPC::VRRCRegClass.hasSubClassEq(RC)) {
    // We don't have indexed addressing for vector loads.  Emit:
    // R0 = ADDI FI#
    // STVX VAL, 0, R0
    //
    // FIXME: We use R0 here, because it isn't available for RA.
    bool Is64Bit = TM.getSubtargetImpl()->isPPC64();
    unsigned Instr = Is64Bit ? PPC::ADDI8 : PPC::ADDI;
    unsigned GPR0  = Is64Bit ? PPC::X0    : PPC::R0;
    NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(Instr), GPR0),
                                       FrameIdx, 0, 0));
    NewMIs.push_back(BuildMI(MF, DL, get(PPC::STVX))
                     .addReg(SrcReg, getKillRegState(isKill))
                     .addReg(GPR0)
                     .addReg(GPR0));
  } else {
    llvm_unreachable("Unknown regclass!");
  }

  return false;
}

void
PPCInstrInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                  MachineBasicBlock::iterator MI,
                                  unsigned SrcReg, bool isKill, int FrameIdx,
                                  const TargetRegisterClass *RC,
                                  const TargetRegisterInfo *TRI) const {
  MachineFunction &MF = *MBB.getParent();
  SmallVector<MachineInstr*, 4> NewMIs;

  if (StoreRegToStackSlot(MF, SrcReg, isKill, FrameIdx, RC, NewMIs)) {
    PPCFunctionInfo *FuncInfo = MF.getInfo<PPCFunctionInfo>();
    FuncInfo->setSpillsCR();
  }

  for (unsigned i = 0, e = NewMIs.size(); i != e; ++i)
    MBB.insert(MI, NewMIs[i]);

  const MachineFrameInfo &MFI = *MF.getFrameInfo();
  MachineMemOperand *MMO =
    MF.getMachineMemOperand(MachinePointerInfo::getFixedStack(FrameIdx),
                            MachineMemOperand::MOStore,
                            MFI.getObjectSize(FrameIdx),
                            MFI.getObjectAlignment(FrameIdx));
  NewMIs.back()->addMemOperand(MF, MMO);
}

bool
PPCInstrInfo::LoadRegFromStackSlot(MachineFunction &MF, DebugLoc DL,
                                   unsigned DestReg, int FrameIdx,
                                   const TargetRegisterClass *RC,
                                   SmallVectorImpl<MachineInstr*> &NewMIs)const{
  if (PPC::GPRCRegClass.hasSubClassEq(RC)) {
    if (DestReg != PPC::LR) {
      NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::LWZ),
                                                 DestReg), FrameIdx));
    } else {
      NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::LWZ),
                                                 PPC::R11), FrameIdx));
      NewMIs.push_back(BuildMI(MF, DL, get(PPC::MTLR)).addReg(PPC::R11));
    }
  } else if (PPC::G8RCRegClass.hasSubClassEq(RC)) {
    if (DestReg != PPC::LR8) {
      NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::LD), DestReg),
                                         FrameIdx));
    } else {
      NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::LD),
                                                 PPC::X11), FrameIdx));
      NewMIs.push_back(BuildMI(MF, DL, get(PPC::MTLR8)).addReg(PPC::X11));
    }
  } else if (PPC::F8RCRegClass.hasSubClassEq(RC)) {
    NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::LFD), DestReg),
                                       FrameIdx));
  } else if (PPC::F4RCRegClass.hasSubClassEq(RC)) {
    NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::LFS), DestReg),
                                       FrameIdx));
  } else if (PPC::CRRCRegClass.hasSubClassEq(RC)) {
    if ((!DisablePPC32RS && !TM.getSubtargetImpl()->isPPC64()) ||
        (!DisablePPC64RS && TM.getSubtargetImpl()->isPPC64())) {
      NewMIs.push_back(addFrameReference(BuildMI(MF, DL,
                                                 get(PPC::RESTORE_CR), DestReg)
                                         , FrameIdx));
      return true;
    } else {
      // FIXME: We need a scatch reg here.  The trouble with using R0 is that
      // it's possible for the stack frame to be so big the save location is
      // out of range of immediate offsets, necessitating another register.
      // We hack this on Darwin by reserving R2.  It's probably broken on Linux
      // at the moment.
      unsigned ScratchReg = TM.getSubtargetImpl()->isDarwinABI() ?
                                                            PPC::R2 : PPC::R0;
      NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::LWZ),
                                         ScratchReg), FrameIdx));
  
      // If the reloaded register isn't CR0, shift the bits right so that they are
      // in the right CR's slot.
      if (DestReg != PPC::CR0) {
        unsigned ShiftBits = getPPCRegisterNumbering(DestReg)*4;
        // rlwinm r11, r11, 32-ShiftBits, 0, 31.
        NewMIs.push_back(BuildMI(MF, DL, get(PPC::RLWINM), ScratchReg)
                      .addReg(ScratchReg).addImm(32-ShiftBits).addImm(0)
                      .addImm(31));
      }
  
      NewMIs.push_back(BuildMI(MF, DL, get(TM.getSubtargetImpl()->isPPC64() ?
                         PPC::MTCRF8 : PPC::MTCRF), DestReg)
                       .addReg(ScratchReg));
    }
  } else if (PPC::CRBITRCRegClass.hasSubClassEq(RC)) {

    unsigned Reg = 0;
    if (DestReg == PPC::CR0LT || DestReg == PPC::CR0GT ||
        DestReg == PPC::CR0EQ || DestReg == PPC::CR0UN)
      Reg = PPC::CR0;
    else if (DestReg == PPC::CR1LT || DestReg == PPC::CR1GT ||
             DestReg == PPC::CR1EQ || DestReg == PPC::CR1UN)
      Reg = PPC::CR1;
    else if (DestReg == PPC::CR2LT || DestReg == PPC::CR2GT ||
             DestReg == PPC::CR2EQ || DestReg == PPC::CR2UN)
      Reg = PPC::CR2;
    else if (DestReg == PPC::CR3LT || DestReg == PPC::CR3GT ||
             DestReg == PPC::CR3EQ || DestReg == PPC::CR3UN)
      Reg = PPC::CR3;
    else if (DestReg == PPC::CR4LT || DestReg == PPC::CR4GT ||
             DestReg == PPC::CR4EQ || DestReg == PPC::CR4UN)
      Reg = PPC::CR4;
    else if (DestReg == PPC::CR5LT || DestReg == PPC::CR5GT ||
             DestReg == PPC::CR5EQ || DestReg == PPC::CR5UN)
      Reg = PPC::CR5;
    else if (DestReg == PPC::CR6LT || DestReg == PPC::CR6GT ||
             DestReg == PPC::CR6EQ || DestReg == PPC::CR6UN)
      Reg = PPC::CR6;
    else if (DestReg == PPC::CR7LT || DestReg == PPC::CR7GT ||
             DestReg == PPC::CR7EQ || DestReg == PPC::CR7UN)
      Reg = PPC::CR7;

    return LoadRegFromStackSlot(MF, DL, Reg, FrameIdx,
                                &PPC::CRRCRegClass, NewMIs);

  } else if (PPC::VRRCRegClass.hasSubClassEq(RC)) {
    // We don't have indexed addressing for vector loads.  Emit:
    // R0 = ADDI FI#
    // Dest = LVX 0, R0
    //
    // FIXME: We use R0 here, because it isn't available for RA.
    bool Is64Bit = TM.getSubtargetImpl()->isPPC64();
    unsigned Instr = Is64Bit ? PPC::ADDI8 : PPC::ADDI;
    unsigned GPR0  = Is64Bit ? PPC::X0    : PPC::R0;
    NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(Instr), GPR0),
                                       FrameIdx, 0, 0));
    NewMIs.push_back(BuildMI(MF, DL, get(PPC::LVX),DestReg).addReg(GPR0)
                     .addReg(GPR0));
  } else {
    llvm_unreachable("Unknown regclass!");
  }

  return false;
}

void
PPCInstrInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MI,
                                   unsigned DestReg, int FrameIdx,
                                   const TargetRegisterClass *RC,
                                   const TargetRegisterInfo *TRI) const {
  MachineFunction &MF = *MBB.getParent();
  SmallVector<MachineInstr*, 4> NewMIs;
  DebugLoc DL;
  if (MI != MBB.end()) DL = MI->getDebugLoc();
  if (LoadRegFromStackSlot(MF, DL, DestReg, FrameIdx, RC, NewMIs)) {
    PPCFunctionInfo *FuncInfo = MF.getInfo<PPCFunctionInfo>();
    FuncInfo->setSpillsCR();
  }
  for (unsigned i = 0, e = NewMIs.size(); i != e; ++i)
    MBB.insert(MI, NewMIs[i]);

  const MachineFrameInfo &MFI = *MF.getFrameInfo();
  MachineMemOperand *MMO =
    MF.getMachineMemOperand(MachinePointerInfo::getFixedStack(FrameIdx),
                            MachineMemOperand::MOLoad,
                            MFI.getObjectSize(FrameIdx),
                            MFI.getObjectAlignment(FrameIdx));
  NewMIs.back()->addMemOperand(MF, MMO);
}

MachineInstr*
PPCInstrInfo::emitFrameIndexDebugValue(MachineFunction &MF,
                                       int FrameIx, uint64_t Offset,
                                       const MDNode *MDPtr,
                                       DebugLoc DL) const {
  MachineInstrBuilder MIB = BuildMI(MF, DL, get(PPC::DBG_VALUE));
  addFrameReference(MIB, FrameIx, 0, false).addImm(Offset).addMetadata(MDPtr);
  return &*MIB;
}

bool PPCInstrInfo::
ReverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const {
  assert(Cond.size() == 2 && "Invalid PPC branch opcode!");
  if (Cond[1].getReg() == PPC::CTR8 || Cond[1].getReg() == PPC::CTR)
    Cond[0].setImm(Cond[0].getImm() == 0 ? 1 : 0);
  else
    // Leave the CR# the same, but invert the condition.
    Cond[0].setImm(PPC::InvertPredicate((PPC::Predicate)Cond[0].getImm()));
  return false;
}

/// GetInstSize - Return the number of bytes of code the specified
/// instruction may be.  This returns the maximum number of bytes.
///
unsigned PPCInstrInfo::GetInstSizeInBytes(const MachineInstr *MI) const {
  switch (MI->getOpcode()) {
  case PPC::INLINEASM: {       // Inline Asm: Variable size.
    const MachineFunction *MF = MI->getParent()->getParent();
    const char *AsmStr = MI->getOperand(0).getSymbolName();
    return getInlineAsmLength(AsmStr, *MF->getTarget().getMCAsmInfo());
  }
  case PPC::PROLOG_LABEL:
  case PPC::EH_LABEL:
  case PPC::GC_LABEL:
  case PPC::DBG_VALUE:
    return 0;
  case PPC::BL8_NOP_ELF:
  case PPC::BLA8_NOP_ELF:
    return 8;
  default:
    return 4; // PowerPC instructions are all 4 bytes
  }
}
