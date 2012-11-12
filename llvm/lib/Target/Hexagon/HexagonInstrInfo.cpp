//===-- HexagonInstrInfo.cpp - Hexagon Instruction Information ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Hexagon implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "HexagonInstrInfo.h"
#include "HexagonRegisterInfo.h"
#include "HexagonSubtarget.h"
#include "Hexagon.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/DFAPacketizer.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/Support/MathExtras.h"
#define GET_INSTRINFO_CTOR
#define GET_INSTRMAP_INFO
#include "HexagonGenInstrInfo.inc"
#include "HexagonGenDFAPacketizer.inc"

using namespace llvm;

///
/// Constants for Hexagon instructions.
///
const int Hexagon_MEMW_OFFSET_MAX = 4095;
const int Hexagon_MEMW_OFFSET_MIN = -4096;
const int Hexagon_MEMD_OFFSET_MAX = 8191;
const int Hexagon_MEMD_OFFSET_MIN = -8192;
const int Hexagon_MEMH_OFFSET_MAX = 2047;
const int Hexagon_MEMH_OFFSET_MIN = -2048;
const int Hexagon_MEMB_OFFSET_MAX = 1023;
const int Hexagon_MEMB_OFFSET_MIN = -1024;
const int Hexagon_ADDI_OFFSET_MAX = 32767;
const int Hexagon_ADDI_OFFSET_MIN = -32768;
const int Hexagon_MEMD_AUTOINC_MAX = 56;
const int Hexagon_MEMD_AUTOINC_MIN = -64;
const int Hexagon_MEMW_AUTOINC_MAX = 28;
const int Hexagon_MEMW_AUTOINC_MIN = -32;
const int Hexagon_MEMH_AUTOINC_MAX = 14;
const int Hexagon_MEMH_AUTOINC_MIN = -16;
const int Hexagon_MEMB_AUTOINC_MAX = 7;
const int Hexagon_MEMB_AUTOINC_MIN = -8;


HexagonInstrInfo::HexagonInstrInfo(HexagonSubtarget &ST)
  : HexagonGenInstrInfo(Hexagon::ADJCALLSTACKDOWN, Hexagon::ADJCALLSTACKUP),
    RI(ST, *this), Subtarget(ST) {
}


/// isLoadFromStackSlot - If the specified machine instruction is a direct
/// load from a stack slot, return the virtual or physical register number of
/// the destination along with the FrameIndex of the loaded stack slot.  If
/// not, return 0.  This predicate must return 0 if the instruction has
/// any side effects other than loading from the stack slot.
unsigned HexagonInstrInfo::isLoadFromStackSlot(const MachineInstr *MI,
                                             int &FrameIndex) const {


  switch (MI->getOpcode()) {
  default: break;
  case Hexagon::LDriw:
  case Hexagon::LDrid:
  case Hexagon::LDrih:
  case Hexagon::LDrib:
  case Hexagon::LDriub:
    if (MI->getOperand(2).isFI() &&
        MI->getOperand(1).isImm() && (MI->getOperand(1).getImm() == 0)) {
      FrameIndex = MI->getOperand(2).getIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  }
  return 0;
}


/// isStoreToStackSlot - If the specified machine instruction is a direct
/// store to a stack slot, return the virtual or physical register number of
/// the source reg along with the FrameIndex of the loaded stack slot.  If
/// not, return 0.  This predicate must return 0 if the instruction has
/// any side effects other than storing to the stack slot.
unsigned HexagonInstrInfo::isStoreToStackSlot(const MachineInstr *MI,
                                            int &FrameIndex) const {
  switch (MI->getOpcode()) {
  default: break;
  case Hexagon::STriw:
  case Hexagon::STrid:
  case Hexagon::STrih:
  case Hexagon::STrib:
    if (MI->getOperand(2).isFI() &&
        MI->getOperand(1).isImm() && (MI->getOperand(1).getImm() == 0)) {
      FrameIndex = MI->getOperand(0).getIndex();
      return MI->getOperand(2).getReg();
    }
    break;
  }
  return 0;
}


unsigned
HexagonInstrInfo::InsertBranch(MachineBasicBlock &MBB,MachineBasicBlock *TBB,
                             MachineBasicBlock *FBB,
                             const SmallVectorImpl<MachineOperand> &Cond,
                             DebugLoc DL) const{

    int BOpc   = Hexagon::JMP;
    int BccOpc = Hexagon::JMP_c;

    assert(TBB && "InsertBranch must not be told to insert a fallthrough");

    int regPos = 0;
    // Check if ReverseBranchCondition has asked to reverse this branch
    // If we want to reverse the branch an odd number of times, we want
    // JMP_cNot.
    if (!Cond.empty() && Cond[0].isImm() && Cond[0].getImm() == 0) {
      BccOpc = Hexagon::JMP_cNot;
      regPos = 1;
    }

    if (FBB == 0) {
      if (Cond.empty()) {
        // Due to a bug in TailMerging/CFG Optimization, we need to add a
        // special case handling of a predicated jump followed by an
        // unconditional jump. If not, Tail Merging and CFG Optimization go
        // into an infinite loop.
        MachineBasicBlock *NewTBB, *NewFBB;
        SmallVector<MachineOperand, 4> Cond;
        MachineInstr *Term = MBB.getFirstTerminator();
        if (isPredicated(Term) && !AnalyzeBranch(MBB, NewTBB, NewFBB, Cond,
                                                 false)) {
          MachineBasicBlock *NextBB =
            llvm::next(MachineFunction::iterator(&MBB));
          if (NewTBB == NextBB) {
            ReverseBranchCondition(Cond);
            RemoveBranch(MBB);
            return InsertBranch(MBB, TBB, 0, Cond, DL);
          }
        }
        BuildMI(&MBB, DL, get(BOpc)).addMBB(TBB);
      } else {
        BuildMI(&MBB, DL,
                get(BccOpc)).addReg(Cond[regPos].getReg()).addMBB(TBB);
      }
      return 1;
    }

    BuildMI(&MBB, DL, get(BccOpc)).addReg(Cond[regPos].getReg()).addMBB(TBB);
    BuildMI(&MBB, DL, get(BOpc)).addMBB(FBB);

    return 2;
}


bool HexagonInstrInfo::AnalyzeBranch(MachineBasicBlock &MBB,
                                     MachineBasicBlock *&TBB,
                                 MachineBasicBlock *&FBB,
                                 SmallVectorImpl<MachineOperand> &Cond,
                                 bool AllowModify) const {
  TBB = NULL;
  FBB = NULL;

  // If the block has no terminators, it just falls into the block after it.
  MachineBasicBlock::iterator I = MBB.end();
  if (I == MBB.begin())
    return false;

  // A basic block may looks like this:
  //
  //  [   insn
  //     EH_LABEL
  //      insn
  //      insn
  //      insn
  //     EH_LABEL
  //      insn     ]
  //
  // It has two succs but does not have a terminator
  // Don't know how to handle it.
  do {
    --I;
    if (I->isEHLabel())
      return true;
  } while (I != MBB.begin());

  I = MBB.end();
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
    if (LastInst->getOpcode() == Hexagon::JMP) {
      TBB = LastInst->getOperand(0).getMBB();
      return false;
    }
    if (LastInst->getOpcode() == Hexagon::JMP_c) {
      // Block ends with fall-through true condbranch.
      TBB = LastInst->getOperand(1).getMBB();
      Cond.push_back(LastInst->getOperand(0));
      return false;
    }
    if (LastInst->getOpcode() == Hexagon::JMP_cNot) {
      // Block ends with fall-through false condbranch.
      TBB = LastInst->getOperand(1).getMBB();
      Cond.push_back(MachineOperand::CreateImm(0));
      Cond.push_back(LastInst->getOperand(0));
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

  // If the block ends with Hexagon::BRCOND and Hexagon:JMP, handle it.
  if (((SecondLastInst->getOpcode() == Hexagon::BRCOND) ||
      (SecondLastInst->getOpcode() == Hexagon::JMP_c)) &&
      LastInst->getOpcode() == Hexagon::JMP) {
    TBB =  SecondLastInst->getOperand(1).getMBB();
    Cond.push_back(SecondLastInst->getOperand(0));
    FBB = LastInst->getOperand(0).getMBB();
    return false;
  }

  // If the block ends with Hexagon::JMP_cNot and Hexagon:JMP, handle it.
  if ((SecondLastInst->getOpcode() == Hexagon::JMP_cNot) &&
      LastInst->getOpcode() == Hexagon::JMP) {
    TBB =  SecondLastInst->getOperand(1).getMBB();
    Cond.push_back(MachineOperand::CreateImm(0));
    Cond.push_back(SecondLastInst->getOperand(0));
    FBB = LastInst->getOperand(0).getMBB();
    return false;
  }

  // If the block ends with two Hexagon:JMPs, handle it.  The second one is not
  // executed, so remove it.
  if (SecondLastInst->getOpcode() == Hexagon::JMP &&
      LastInst->getOpcode() == Hexagon::JMP) {
    TBB = SecondLastInst->getOperand(0).getMBB();
    I = LastInst;
    if (AllowModify)
      I->eraseFromParent();
    return false;
  }

  // Otherwise, can't handle this.
  return true;
}


unsigned HexagonInstrInfo::RemoveBranch(MachineBasicBlock &MBB) const {
  int BOpc   = Hexagon::JMP;
  int BccOpc = Hexagon::JMP_c;
  int BccOpcNot = Hexagon::JMP_cNot;

  MachineBasicBlock::iterator I = MBB.end();
  if (I == MBB.begin()) return 0;
  --I;
  if (I->getOpcode() != BOpc && I->getOpcode() != BccOpc &&
      I->getOpcode() != BccOpcNot)
    return 0;

  // Remove the branch.
  I->eraseFromParent();

  I = MBB.end();

  if (I == MBB.begin()) return 1;
  --I;
  if (I->getOpcode() != BccOpc && I->getOpcode() != BccOpcNot)
    return 1;

  // Remove the branch.
  I->eraseFromParent();
  return 2;
}


void HexagonInstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator I, DebugLoc DL,
                                 unsigned DestReg, unsigned SrcReg,
                                 bool KillSrc) const {
  if (Hexagon::IntRegsRegClass.contains(SrcReg, DestReg)) {
    BuildMI(MBB, I, DL, get(Hexagon::TFR), DestReg).addReg(SrcReg);
    return;
  }
  if (Hexagon::DoubleRegsRegClass.contains(SrcReg, DestReg)) {
    BuildMI(MBB, I, DL, get(Hexagon::TFR_64), DestReg).addReg(SrcReg);
    return;
  }
  if (Hexagon::PredRegsRegClass.contains(SrcReg, DestReg)) {
    // Map Pd = Ps to Pd = or(Ps, Ps).
    BuildMI(MBB, I, DL, get(Hexagon::OR_pp),
            DestReg).addReg(SrcReg).addReg(SrcReg);
    return;
  }
  if (Hexagon::DoubleRegsRegClass.contains(DestReg) &&
      Hexagon::IntRegsRegClass.contains(SrcReg)) {
    // We can have an overlap between single and double reg: r1:0 = r0.
    if(SrcReg == RI.getSubReg(DestReg, Hexagon::subreg_loreg)) {
        // r1:0 = r0
        BuildMI(MBB, I, DL, get(Hexagon::TFRI), (RI.getSubReg(DestReg,
                Hexagon::subreg_hireg))).addImm(0);
    } else {
        // r1:0 = r1 or no overlap.
        BuildMI(MBB, I, DL, get(Hexagon::TFR), (RI.getSubReg(DestReg,
                Hexagon::subreg_loreg))).addReg(SrcReg);
        BuildMI(MBB, I, DL, get(Hexagon::TFRI), (RI.getSubReg(DestReg,
                Hexagon::subreg_hireg))).addImm(0);
    }
    return;
  }
  if (Hexagon::CRRegsRegClass.contains(DestReg) &&
      Hexagon::IntRegsRegClass.contains(SrcReg)) {
    BuildMI(MBB, I, DL, get(Hexagon::TFCR), DestReg).addReg(SrcReg);
    return;
  }

  llvm_unreachable("Unimplemented");
}


void HexagonInstrInfo::
storeRegToStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                    unsigned SrcReg, bool isKill, int FI,
                    const TargetRegisterClass *RC,
                    const TargetRegisterInfo *TRI) const {

  DebugLoc DL = MBB.findDebugLoc(I);
  MachineFunction &MF = *MBB.getParent();
  MachineFrameInfo &MFI = *MF.getFrameInfo();
  unsigned Align = MFI.getObjectAlignment(FI);

  MachineMemOperand *MMO =
      MF.getMachineMemOperand(
                      MachinePointerInfo(PseudoSourceValue::getFixedStack(FI)),
                      MachineMemOperand::MOStore,
                      MFI.getObjectSize(FI),
                      Align);

  if (Hexagon::IntRegsRegClass.hasSubClassEq(RC)) {
    BuildMI(MBB, I, DL, get(Hexagon::STriw))
          .addFrameIndex(FI).addImm(0)
          .addReg(SrcReg, getKillRegState(isKill)).addMemOperand(MMO);
  } else if (Hexagon::DoubleRegsRegClass.hasSubClassEq(RC)) {
    BuildMI(MBB, I, DL, get(Hexagon::STrid))
          .addFrameIndex(FI).addImm(0)
          .addReg(SrcReg, getKillRegState(isKill)).addMemOperand(MMO);
  } else if (Hexagon::PredRegsRegClass.hasSubClassEq(RC)) {
    BuildMI(MBB, I, DL, get(Hexagon::STriw_pred))
          .addFrameIndex(FI).addImm(0)
          .addReg(SrcReg, getKillRegState(isKill)).addMemOperand(MMO);
  } else {
    llvm_unreachable("Unimplemented");
  }
}


void HexagonInstrInfo::storeRegToAddr(
                                 MachineFunction &MF, unsigned SrcReg,
                                 bool isKill,
                                 SmallVectorImpl<MachineOperand> &Addr,
                                 const TargetRegisterClass *RC,
                                 SmallVectorImpl<MachineInstr*> &NewMIs) const
{
  llvm_unreachable("Unimplemented");
}


void HexagonInstrInfo::
loadRegFromStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                     unsigned DestReg, int FI,
                     const TargetRegisterClass *RC,
                     const TargetRegisterInfo *TRI) const {
  DebugLoc DL = MBB.findDebugLoc(I);
  MachineFunction &MF = *MBB.getParent();
  MachineFrameInfo &MFI = *MF.getFrameInfo();
  unsigned Align = MFI.getObjectAlignment(FI);

  MachineMemOperand *MMO =
      MF.getMachineMemOperand(
                      MachinePointerInfo(PseudoSourceValue::getFixedStack(FI)),
                      MachineMemOperand::MOLoad,
                      MFI.getObjectSize(FI),
                      Align);
  if (RC == &Hexagon::IntRegsRegClass) {
    BuildMI(MBB, I, DL, get(Hexagon::LDriw), DestReg)
          .addFrameIndex(FI).addImm(0).addMemOperand(MMO);
  } else if (RC == &Hexagon::DoubleRegsRegClass) {
    BuildMI(MBB, I, DL, get(Hexagon::LDrid), DestReg)
          .addFrameIndex(FI).addImm(0).addMemOperand(MMO);
  } else if (RC == &Hexagon::PredRegsRegClass) {
    BuildMI(MBB, I, DL, get(Hexagon::LDriw_pred), DestReg)
          .addFrameIndex(FI).addImm(0).addMemOperand(MMO);
  } else {
    llvm_unreachable("Can't store this register to stack slot");
  }
}


void HexagonInstrInfo::loadRegFromAddr(MachineFunction &MF, unsigned DestReg,
                                        SmallVectorImpl<MachineOperand> &Addr,
                                        const TargetRegisterClass *RC,
                                 SmallVectorImpl<MachineInstr*> &NewMIs) const {
  llvm_unreachable("Unimplemented");
}


MachineInstr *HexagonInstrInfo::foldMemoryOperandImpl(MachineFunction &MF,
                                                    MachineInstr* MI,
                                          const SmallVectorImpl<unsigned> &Ops,
                                                    int FI) const {
  // Hexagon_TODO: Implement.
  return(0);
}


unsigned HexagonInstrInfo::createVR(MachineFunction* MF, MVT VT) const {

  MachineRegisterInfo &RegInfo = MF->getRegInfo();
  const TargetRegisterClass *TRC;
  if (VT == MVT::i1) {
    TRC = &Hexagon::PredRegsRegClass;
  } else if (VT == MVT::i32 || VT == MVT::f32) {
    TRC = &Hexagon::IntRegsRegClass;
  } else if (VT == MVT::i64 || VT == MVT::f64) {
    TRC = &Hexagon::DoubleRegsRegClass;
  } else {
    llvm_unreachable("Cannot handle this register class");
  }

  unsigned NewReg = RegInfo.createVirtualRegister(TRC);
  return NewReg;
}

bool HexagonInstrInfo::isExtendable(const MachineInstr *MI) const {
  switch(MI->getOpcode()) {
    default: return false;
    // JMP_EQri
    case Hexagon::JMP_EQriPt_nv_V4:
    case Hexagon::JMP_EQriPnt_nv_V4:
    case Hexagon::JMP_EQriNotPt_nv_V4:
    case Hexagon::JMP_EQriNotPnt_nv_V4:

    // JMP_EQri - with -1
    case Hexagon::JMP_EQriPtneg_nv_V4:
    case Hexagon::JMP_EQriPntneg_nv_V4:
    case Hexagon::JMP_EQriNotPtneg_nv_V4:
    case Hexagon::JMP_EQriNotPntneg_nv_V4:

    // JMP_EQrr
    case Hexagon::JMP_EQrrPt_nv_V4:
    case Hexagon::JMP_EQrrPnt_nv_V4:
    case Hexagon::JMP_EQrrNotPt_nv_V4:
    case Hexagon::JMP_EQrrNotPnt_nv_V4:

    // JMP_GTri
    case Hexagon::JMP_GTriPt_nv_V4:
    case Hexagon::JMP_GTriPnt_nv_V4:
    case Hexagon::JMP_GTriNotPt_nv_V4:
    case Hexagon::JMP_GTriNotPnt_nv_V4:

    // JMP_GTri - with -1
    case Hexagon::JMP_GTriPtneg_nv_V4:
    case Hexagon::JMP_GTriPntneg_nv_V4:
    case Hexagon::JMP_GTriNotPtneg_nv_V4:
    case Hexagon::JMP_GTriNotPntneg_nv_V4:

    // JMP_GTrr
    case Hexagon::JMP_GTrrPt_nv_V4:
    case Hexagon::JMP_GTrrPnt_nv_V4:
    case Hexagon::JMP_GTrrNotPt_nv_V4:
    case Hexagon::JMP_GTrrNotPnt_nv_V4:

    // JMP_GTrrdn
    case Hexagon::JMP_GTrrdnPt_nv_V4:
    case Hexagon::JMP_GTrrdnPnt_nv_V4:
    case Hexagon::JMP_GTrrdnNotPt_nv_V4:
    case Hexagon::JMP_GTrrdnNotPnt_nv_V4:

    // JMP_GTUri
    case Hexagon::JMP_GTUriPt_nv_V4:
    case Hexagon::JMP_GTUriPnt_nv_V4:
    case Hexagon::JMP_GTUriNotPt_nv_V4:
    case Hexagon::JMP_GTUriNotPnt_nv_V4:

    // JMP_GTUrr
    case Hexagon::JMP_GTUrrPt_nv_V4:
    case Hexagon::JMP_GTUrrPnt_nv_V4:
    case Hexagon::JMP_GTUrrNotPt_nv_V4:
    case Hexagon::JMP_GTUrrNotPnt_nv_V4:

    // JMP_GTUrrdn
    case Hexagon::JMP_GTUrrdnPt_nv_V4:
    case Hexagon::JMP_GTUrrdnPnt_nv_V4:
    case Hexagon::JMP_GTUrrdnNotPt_nv_V4:
    case Hexagon::JMP_GTUrrdnNotPnt_nv_V4:

    // TFR_FI
    case Hexagon::TFR_FI:
      return true;
  }
}

bool HexagonInstrInfo::isExtended(const MachineInstr *MI) const {
  switch(MI->getOpcode()) {
    default: return false;
    // JMP_EQri
    case Hexagon::JMP_EQriPt_ie_nv_V4:
    case Hexagon::JMP_EQriPnt_ie_nv_V4:
    case Hexagon::JMP_EQriNotPt_ie_nv_V4:
    case Hexagon::JMP_EQriNotPnt_ie_nv_V4:

    // JMP_EQri - with -1
    case Hexagon::JMP_EQriPtneg_ie_nv_V4:
    case Hexagon::JMP_EQriPntneg_ie_nv_V4:
    case Hexagon::JMP_EQriNotPtneg_ie_nv_V4:
    case Hexagon::JMP_EQriNotPntneg_ie_nv_V4:

    // JMP_EQrr
    case Hexagon::JMP_EQrrPt_ie_nv_V4:
    case Hexagon::JMP_EQrrPnt_ie_nv_V4:
    case Hexagon::JMP_EQrrNotPt_ie_nv_V4:
    case Hexagon::JMP_EQrrNotPnt_ie_nv_V4:

    // JMP_GTri
    case Hexagon::JMP_GTriPt_ie_nv_V4:
    case Hexagon::JMP_GTriPnt_ie_nv_V4:
    case Hexagon::JMP_GTriNotPt_ie_nv_V4:
    case Hexagon::JMP_GTriNotPnt_ie_nv_V4:

    // JMP_GTri - with -1
    case Hexagon::JMP_GTriPtneg_ie_nv_V4:
    case Hexagon::JMP_GTriPntneg_ie_nv_V4:
    case Hexagon::JMP_GTriNotPtneg_ie_nv_V4:
    case Hexagon::JMP_GTriNotPntneg_ie_nv_V4:

    // JMP_GTrr
    case Hexagon::JMP_GTrrPt_ie_nv_V4:
    case Hexagon::JMP_GTrrPnt_ie_nv_V4:
    case Hexagon::JMP_GTrrNotPt_ie_nv_V4:
    case Hexagon::JMP_GTrrNotPnt_ie_nv_V4:

    // JMP_GTrrdn
    case Hexagon::JMP_GTrrdnPt_ie_nv_V4:
    case Hexagon::JMP_GTrrdnPnt_ie_nv_V4:
    case Hexagon::JMP_GTrrdnNotPt_ie_nv_V4:
    case Hexagon::JMP_GTrrdnNotPnt_ie_nv_V4:

    // JMP_GTUri
    case Hexagon::JMP_GTUriPt_ie_nv_V4:
    case Hexagon::JMP_GTUriPnt_ie_nv_V4:
    case Hexagon::JMP_GTUriNotPt_ie_nv_V4:
    case Hexagon::JMP_GTUriNotPnt_ie_nv_V4:

    // JMP_GTUrr
    case Hexagon::JMP_GTUrrPt_ie_nv_V4:
    case Hexagon::JMP_GTUrrPnt_ie_nv_V4:
    case Hexagon::JMP_GTUrrNotPt_ie_nv_V4:
    case Hexagon::JMP_GTUrrNotPnt_ie_nv_V4:

    // JMP_GTUrrdn
    case Hexagon::JMP_GTUrrdnPt_ie_nv_V4:
    case Hexagon::JMP_GTUrrdnPnt_ie_nv_V4:
    case Hexagon::JMP_GTUrrdnNotPt_ie_nv_V4:
    case Hexagon::JMP_GTUrrdnNotPnt_ie_nv_V4:

    // V4 absolute set addressing.
    case Hexagon::LDrid_abs_setimm_V4:
    case Hexagon::LDriw_abs_setimm_V4:
    case Hexagon::LDrih_abs_setimm_V4:
    case Hexagon::LDrib_abs_setimm_V4:
    case Hexagon::LDriuh_abs_setimm_V4:
    case Hexagon::LDriub_abs_setimm_V4:

    case Hexagon::STrid_abs_setimm_V4:
    case Hexagon::STrib_abs_setimm_V4:
    case Hexagon::STrih_abs_setimm_V4:
    case Hexagon::STriw_abs_setimm_V4:

    // V4 global address load.
    case Hexagon::LDrid_GP_cPt_V4 :
    case Hexagon::LDrid_GP_cNotPt_V4 :
    case Hexagon::LDrid_GP_cdnPt_V4 :
    case Hexagon::LDrid_GP_cdnNotPt_V4 :
    case Hexagon::LDrib_GP_cPt_V4 :
    case Hexagon::LDrib_GP_cNotPt_V4 :
    case Hexagon::LDrib_GP_cdnPt_V4 :
    case Hexagon::LDrib_GP_cdnNotPt_V4 :
    case Hexagon::LDriub_GP_cPt_V4 :
    case Hexagon::LDriub_GP_cNotPt_V4 :
    case Hexagon::LDriub_GP_cdnPt_V4 :
    case Hexagon::LDriub_GP_cdnNotPt_V4 :
    case Hexagon::LDrih_GP_cPt_V4 :
    case Hexagon::LDrih_GP_cNotPt_V4 :
    case Hexagon::LDrih_GP_cdnPt_V4 :
    case Hexagon::LDrih_GP_cdnNotPt_V4 :
    case Hexagon::LDriuh_GP_cPt_V4 :
    case Hexagon::LDriuh_GP_cNotPt_V4 :
    case Hexagon::LDriuh_GP_cdnPt_V4 :
    case Hexagon::LDriuh_GP_cdnNotPt_V4 :
    case Hexagon::LDriw_GP_cPt_V4 :
    case Hexagon::LDriw_GP_cNotPt_V4 :
    case Hexagon::LDriw_GP_cdnPt_V4 :
    case Hexagon::LDriw_GP_cdnNotPt_V4 :
    case Hexagon::LDd_GP_cPt_V4 :
    case Hexagon::LDd_GP_cNotPt_V4 :
    case Hexagon::LDd_GP_cdnPt_V4 :
    case Hexagon::LDd_GP_cdnNotPt_V4 :
    case Hexagon::LDb_GP_cPt_V4 :
    case Hexagon::LDb_GP_cNotPt_V4 :
    case Hexagon::LDb_GP_cdnPt_V4 :
    case Hexagon::LDb_GP_cdnNotPt_V4 :
    case Hexagon::LDub_GP_cPt_V4 :
    case Hexagon::LDub_GP_cNotPt_V4 :
    case Hexagon::LDub_GP_cdnPt_V4 :
    case Hexagon::LDub_GP_cdnNotPt_V4 :
    case Hexagon::LDh_GP_cPt_V4 :
    case Hexagon::LDh_GP_cNotPt_V4 :
    case Hexagon::LDh_GP_cdnPt_V4 :
    case Hexagon::LDh_GP_cdnNotPt_V4 :
    case Hexagon::LDuh_GP_cPt_V4 :
    case Hexagon::LDuh_GP_cNotPt_V4 :
    case Hexagon::LDuh_GP_cdnPt_V4 :
    case Hexagon::LDuh_GP_cdnNotPt_V4 :
    case Hexagon::LDw_GP_cPt_V4 :
    case Hexagon::LDw_GP_cNotPt_V4 :
    case Hexagon::LDw_GP_cdnPt_V4 :
    case Hexagon::LDw_GP_cdnNotPt_V4 :

    // V4 global address store.
    case Hexagon::STrid_GP_cPt_V4 :
    case Hexagon::STrid_GP_cNotPt_V4 :
    case Hexagon::STrid_GP_cdnPt_V4 :
    case Hexagon::STrid_GP_cdnNotPt_V4 :
    case Hexagon::STrib_GP_cPt_V4 :
    case Hexagon::STrib_GP_cNotPt_V4 :
    case Hexagon::STrib_GP_cdnPt_V4 :
    case Hexagon::STrib_GP_cdnNotPt_V4 :
    case Hexagon::STrih_GP_cPt_V4 :
    case Hexagon::STrih_GP_cNotPt_V4 :
    case Hexagon::STrih_GP_cdnPt_V4 :
    case Hexagon::STrih_GP_cdnNotPt_V4 :
    case Hexagon::STriw_GP_cPt_V4 :
    case Hexagon::STriw_GP_cNotPt_V4 :
    case Hexagon::STriw_GP_cdnPt_V4 :
    case Hexagon::STriw_GP_cdnNotPt_V4 :
    case Hexagon::STd_GP_cPt_V4 :
    case Hexagon::STd_GP_cNotPt_V4 :
    case Hexagon::STd_GP_cdnPt_V4 :
    case Hexagon::STd_GP_cdnNotPt_V4 :
    case Hexagon::STb_GP_cPt_V4 :
    case Hexagon::STb_GP_cNotPt_V4 :
    case Hexagon::STb_GP_cdnPt_V4 :
    case Hexagon::STb_GP_cdnNotPt_V4 :
    case Hexagon::STh_GP_cPt_V4 :
    case Hexagon::STh_GP_cNotPt_V4 :
    case Hexagon::STh_GP_cdnPt_V4 :
    case Hexagon::STh_GP_cdnNotPt_V4 :
    case Hexagon::STw_GP_cPt_V4 :
    case Hexagon::STw_GP_cNotPt_V4 :
    case Hexagon::STw_GP_cdnPt_V4 :
    case Hexagon::STw_GP_cdnNotPt_V4 :

    // V4 predicated global address new value store.
    case Hexagon::STrib_GP_cPt_nv_V4 :
    case Hexagon::STrib_GP_cNotPt_nv_V4 :
    case Hexagon::STrib_GP_cdnPt_nv_V4 :
    case Hexagon::STrib_GP_cdnNotPt_nv_V4 :
    case Hexagon::STrih_GP_cPt_nv_V4 :
    case Hexagon::STrih_GP_cNotPt_nv_V4 :
    case Hexagon::STrih_GP_cdnPt_nv_V4 :
    case Hexagon::STrih_GP_cdnNotPt_nv_V4 :
    case Hexagon::STriw_GP_cPt_nv_V4 :
    case Hexagon::STriw_GP_cNotPt_nv_V4 :
    case Hexagon::STriw_GP_cdnPt_nv_V4 :
    case Hexagon::STriw_GP_cdnNotPt_nv_V4 :
    case Hexagon::STb_GP_cPt_nv_V4 :
    case Hexagon::STb_GP_cNotPt_nv_V4 :
    case Hexagon::STb_GP_cdnPt_nv_V4 :
    case Hexagon::STb_GP_cdnNotPt_nv_V4 :
    case Hexagon::STh_GP_cPt_nv_V4 :
    case Hexagon::STh_GP_cNotPt_nv_V4 :
    case Hexagon::STh_GP_cdnPt_nv_V4 :
    case Hexagon::STh_GP_cdnNotPt_nv_V4 :
    case Hexagon::STw_GP_cPt_nv_V4 :
    case Hexagon::STw_GP_cNotPt_nv_V4 :
    case Hexagon::STw_GP_cdnPt_nv_V4 :
    case Hexagon::STw_GP_cdnNotPt_nv_V4 :

    // TFR_FI
    case Hexagon::TFR_FI_immext_V4:

    // TFRI_F
    case Hexagon::TFRI_f:
    case Hexagon::TFRI_cPt_f:
    case Hexagon::TFRI_cNotPt_f:
    case Hexagon::CONST64_Float_Real:
      return true;
  }
}

bool HexagonInstrInfo::isNewValueJump(const MachineInstr *MI) const {
  switch (MI->getOpcode()) {
    default: return false;
    // JMP_EQri
    case Hexagon::JMP_EQriPt_nv_V4:
    case Hexagon::JMP_EQriPnt_nv_V4:
    case Hexagon::JMP_EQriNotPt_nv_V4:
    case Hexagon::JMP_EQriNotPnt_nv_V4:
    case Hexagon::JMP_EQriPt_ie_nv_V4:
    case Hexagon::JMP_EQriPnt_ie_nv_V4:
    case Hexagon::JMP_EQriNotPt_ie_nv_V4:
    case Hexagon::JMP_EQriNotPnt_ie_nv_V4:

    // JMP_EQri - with -1
    case Hexagon::JMP_EQriPtneg_nv_V4:
    case Hexagon::JMP_EQriPntneg_nv_V4:
    case Hexagon::JMP_EQriNotPtneg_nv_V4:
    case Hexagon::JMP_EQriNotPntneg_nv_V4:
    case Hexagon::JMP_EQriPtneg_ie_nv_V4:
    case Hexagon::JMP_EQriPntneg_ie_nv_V4:
    case Hexagon::JMP_EQriNotPtneg_ie_nv_V4:
    case Hexagon::JMP_EQriNotPntneg_ie_nv_V4:

    // JMP_EQrr
    case Hexagon::JMP_EQrrPt_nv_V4:
    case Hexagon::JMP_EQrrPnt_nv_V4:
    case Hexagon::JMP_EQrrNotPt_nv_V4:
    case Hexagon::JMP_EQrrNotPnt_nv_V4:
    case Hexagon::JMP_EQrrPt_ie_nv_V4:
    case Hexagon::JMP_EQrrPnt_ie_nv_V4:
    case Hexagon::JMP_EQrrNotPt_ie_nv_V4:
    case Hexagon::JMP_EQrrNotPnt_ie_nv_V4:

    // JMP_GTri
    case Hexagon::JMP_GTriPt_nv_V4:
    case Hexagon::JMP_GTriPnt_nv_V4:
    case Hexagon::JMP_GTriNotPt_nv_V4:
    case Hexagon::JMP_GTriNotPnt_nv_V4:
    case Hexagon::JMP_GTriPt_ie_nv_V4:
    case Hexagon::JMP_GTriPnt_ie_nv_V4:
    case Hexagon::JMP_GTriNotPt_ie_nv_V4:
    case Hexagon::JMP_GTriNotPnt_ie_nv_V4:

    // JMP_GTri - with -1
    case Hexagon::JMP_GTriPtneg_nv_V4:
    case Hexagon::JMP_GTriPntneg_nv_V4:
    case Hexagon::JMP_GTriNotPtneg_nv_V4:
    case Hexagon::JMP_GTriNotPntneg_nv_V4:
    case Hexagon::JMP_GTriPtneg_ie_nv_V4:
    case Hexagon::JMP_GTriPntneg_ie_nv_V4:
    case Hexagon::JMP_GTriNotPtneg_ie_nv_V4:
    case Hexagon::JMP_GTriNotPntneg_ie_nv_V4:

    // JMP_GTrr
    case Hexagon::JMP_GTrrPt_nv_V4:
    case Hexagon::JMP_GTrrPnt_nv_V4:
    case Hexagon::JMP_GTrrNotPt_nv_V4:
    case Hexagon::JMP_GTrrNotPnt_nv_V4:
    case Hexagon::JMP_GTrrPt_ie_nv_V4:
    case Hexagon::JMP_GTrrPnt_ie_nv_V4:
    case Hexagon::JMP_GTrrNotPt_ie_nv_V4:
    case Hexagon::JMP_GTrrNotPnt_ie_nv_V4:

    // JMP_GTrrdn
    case Hexagon::JMP_GTrrdnPt_nv_V4:
    case Hexagon::JMP_GTrrdnPnt_nv_V4:
    case Hexagon::JMP_GTrrdnNotPt_nv_V4:
    case Hexagon::JMP_GTrrdnNotPnt_nv_V4:
    case Hexagon::JMP_GTrrdnPt_ie_nv_V4:
    case Hexagon::JMP_GTrrdnPnt_ie_nv_V4:
    case Hexagon::JMP_GTrrdnNotPt_ie_nv_V4:
    case Hexagon::JMP_GTrrdnNotPnt_ie_nv_V4:

    // JMP_GTUri
    case Hexagon::JMP_GTUriPt_nv_V4:
    case Hexagon::JMP_GTUriPnt_nv_V4:
    case Hexagon::JMP_GTUriNotPt_nv_V4:
    case Hexagon::JMP_GTUriNotPnt_nv_V4:
    case Hexagon::JMP_GTUriPt_ie_nv_V4:
    case Hexagon::JMP_GTUriPnt_ie_nv_V4:
    case Hexagon::JMP_GTUriNotPt_ie_nv_V4:
    case Hexagon::JMP_GTUriNotPnt_ie_nv_V4:

    // JMP_GTUrr
    case Hexagon::JMP_GTUrrPt_nv_V4:
    case Hexagon::JMP_GTUrrPnt_nv_V4:
    case Hexagon::JMP_GTUrrNotPt_nv_V4:
    case Hexagon::JMP_GTUrrNotPnt_nv_V4:
    case Hexagon::JMP_GTUrrPt_ie_nv_V4:
    case Hexagon::JMP_GTUrrPnt_ie_nv_V4:
    case Hexagon::JMP_GTUrrNotPt_ie_nv_V4:
    case Hexagon::JMP_GTUrrNotPnt_ie_nv_V4:

    // JMP_GTUrrdn
    case Hexagon::JMP_GTUrrdnPt_nv_V4:
    case Hexagon::JMP_GTUrrdnPnt_nv_V4:
    case Hexagon::JMP_GTUrrdnNotPt_nv_V4:
    case Hexagon::JMP_GTUrrdnNotPnt_nv_V4:
    case Hexagon::JMP_GTUrrdnPt_ie_nv_V4:
    case Hexagon::JMP_GTUrrdnPnt_ie_nv_V4:
    case Hexagon::JMP_GTUrrdnNotPt_ie_nv_V4:
    case Hexagon::JMP_GTUrrdnNotPnt_ie_nv_V4:
      return true;
  }
}

unsigned HexagonInstrInfo::getImmExtForm(const MachineInstr* MI) const {
  switch(MI->getOpcode()) {
    default: llvm_unreachable("Unknown type of instruction.");
    // JMP_EQri
    case Hexagon::JMP_EQriPt_nv_V4:
      return Hexagon::JMP_EQriPt_ie_nv_V4;
    case Hexagon::JMP_EQriNotPt_nv_V4:
      return Hexagon::JMP_EQriNotPt_ie_nv_V4;
    case Hexagon::JMP_EQriPnt_nv_V4:
      return Hexagon::JMP_EQriPnt_ie_nv_V4;
    case Hexagon::JMP_EQriNotPnt_nv_V4:
      return Hexagon::JMP_EQriNotPnt_ie_nv_V4;

    // JMP_EQri -- with -1
    case Hexagon::JMP_EQriPtneg_nv_V4:
      return Hexagon::JMP_EQriPtneg_ie_nv_V4;
    case Hexagon::JMP_EQriNotPtneg_nv_V4:
      return Hexagon::JMP_EQriNotPtneg_ie_nv_V4;
    case Hexagon::JMP_EQriPntneg_nv_V4:
      return Hexagon::JMP_EQriPntneg_ie_nv_V4;
    case Hexagon::JMP_EQriNotPntneg_nv_V4:
      return Hexagon::JMP_EQriNotPntneg_ie_nv_V4;

    // JMP_EQrr
    case Hexagon::JMP_EQrrPt_nv_V4:
      return Hexagon::JMP_EQrrPt_ie_nv_V4;
    case Hexagon::JMP_EQrrNotPt_nv_V4:
      return Hexagon::JMP_EQrrNotPt_ie_nv_V4;
    case Hexagon::JMP_EQrrPnt_nv_V4:
      return Hexagon::JMP_EQrrPnt_ie_nv_V4;
    case Hexagon::JMP_EQrrNotPnt_nv_V4:
      return Hexagon::JMP_EQrrNotPnt_ie_nv_V4;

    // JMP_GTri
    case Hexagon::JMP_GTriPt_nv_V4:
      return Hexagon::JMP_GTriPt_ie_nv_V4;
    case Hexagon::JMP_GTriNotPt_nv_V4:
      return Hexagon::JMP_GTriNotPt_ie_nv_V4;
    case Hexagon::JMP_GTriPnt_nv_V4:
      return Hexagon::JMP_GTriPnt_ie_nv_V4;
    case Hexagon::JMP_GTriNotPnt_nv_V4:
      return Hexagon::JMP_GTriNotPnt_ie_nv_V4;

    // JMP_GTri -- with -1
    case Hexagon::JMP_GTriPtneg_nv_V4:
      return Hexagon::JMP_GTriPtneg_ie_nv_V4;
    case Hexagon::JMP_GTriNotPtneg_nv_V4:
      return Hexagon::JMP_GTriNotPtneg_ie_nv_V4;
    case Hexagon::JMP_GTriPntneg_nv_V4:
      return Hexagon::JMP_GTriPntneg_ie_nv_V4;
    case Hexagon::JMP_GTriNotPntneg_nv_V4:
      return Hexagon::JMP_GTriNotPntneg_ie_nv_V4;

    // JMP_GTrr
    case Hexagon::JMP_GTrrPt_nv_V4:
      return Hexagon::JMP_GTrrPt_ie_nv_V4;
    case Hexagon::JMP_GTrrNotPt_nv_V4:
      return Hexagon::JMP_GTrrNotPt_ie_nv_V4;
    case Hexagon::JMP_GTrrPnt_nv_V4:
      return Hexagon::JMP_GTrrPnt_ie_nv_V4;
    case Hexagon::JMP_GTrrNotPnt_nv_V4:
      return Hexagon::JMP_GTrrNotPnt_ie_nv_V4;

    // JMP_GTrrdn
    case Hexagon::JMP_GTrrdnPt_nv_V4:
      return Hexagon::JMP_GTrrdnPt_ie_nv_V4;
    case Hexagon::JMP_GTrrdnNotPt_nv_V4:
      return Hexagon::JMP_GTrrdnNotPt_ie_nv_V4;
    case Hexagon::JMP_GTrrdnPnt_nv_V4:
      return Hexagon::JMP_GTrrdnPnt_ie_nv_V4;
    case Hexagon::JMP_GTrrdnNotPnt_nv_V4:
      return Hexagon::JMP_GTrrdnNotPnt_ie_nv_V4;

    // JMP_GTUri
    case Hexagon::JMP_GTUriPt_nv_V4:
      return Hexagon::JMP_GTUriPt_ie_nv_V4;
    case Hexagon::JMP_GTUriNotPt_nv_V4:
      return Hexagon::JMP_GTUriNotPt_ie_nv_V4;
    case Hexagon::JMP_GTUriPnt_nv_V4:
      return Hexagon::JMP_GTUriPnt_ie_nv_V4;
    case Hexagon::JMP_GTUriNotPnt_nv_V4:
      return Hexagon::JMP_GTUriNotPnt_ie_nv_V4;

    // JMP_GTUrr
    case Hexagon::JMP_GTUrrPt_nv_V4:
      return Hexagon::JMP_GTUrrPt_ie_nv_V4;
    case Hexagon::JMP_GTUrrNotPt_nv_V4:
      return Hexagon::JMP_GTUrrNotPt_ie_nv_V4;
    case Hexagon::JMP_GTUrrPnt_nv_V4:
      return Hexagon::JMP_GTUrrPnt_ie_nv_V4;
    case Hexagon::JMP_GTUrrNotPnt_nv_V4:
      return Hexagon::JMP_GTUrrNotPnt_ie_nv_V4;

    // JMP_GTUrrdn
    case Hexagon::JMP_GTUrrdnPt_nv_V4:
      return Hexagon::JMP_GTUrrdnPt_ie_nv_V4;
    case Hexagon::JMP_GTUrrdnNotPt_nv_V4:
      return Hexagon::JMP_GTUrrdnNotPt_ie_nv_V4;
    case Hexagon::JMP_GTUrrdnPnt_nv_V4:
      return Hexagon::JMP_GTUrrdnPnt_ie_nv_V4;
    case Hexagon::JMP_GTUrrdnNotPnt_nv_V4:
      return Hexagon::JMP_GTUrrdnNotPnt_ie_nv_V4;

    case Hexagon::TFR_FI:
        return Hexagon::TFR_FI_immext_V4;

    case Hexagon::MEMw_ADDSUBi_indexed_MEM_V4 :
    case Hexagon::MEMw_ADDi_indexed_MEM_V4 :
    case Hexagon::MEMw_SUBi_indexed_MEM_V4 :
    case Hexagon::MEMw_ADDr_indexed_MEM_V4 :
    case Hexagon::MEMw_SUBr_indexed_MEM_V4 :
    case Hexagon::MEMw_ANDr_indexed_MEM_V4 :
    case Hexagon::MEMw_ORr_indexed_MEM_V4 :
    case Hexagon::MEMw_ADDSUBi_MEM_V4 :
    case Hexagon::MEMw_ADDi_MEM_V4 :
    case Hexagon::MEMw_SUBi_MEM_V4 :
    case Hexagon::MEMw_ADDr_MEM_V4 :
    case Hexagon::MEMw_SUBr_MEM_V4 :
    case Hexagon::MEMw_ANDr_MEM_V4 :
    case Hexagon::MEMw_ORr_MEM_V4 :
    case Hexagon::MEMh_ADDSUBi_indexed_MEM_V4 :
    case Hexagon::MEMh_ADDi_indexed_MEM_V4 :
    case Hexagon::MEMh_SUBi_indexed_MEM_V4 :
    case Hexagon::MEMh_ADDr_indexed_MEM_V4 :
    case Hexagon::MEMh_SUBr_indexed_MEM_V4 :
    case Hexagon::MEMh_ANDr_indexed_MEM_V4 :
    case Hexagon::MEMh_ORr_indexed_MEM_V4 :
    case Hexagon::MEMh_ADDSUBi_MEM_V4 :
    case Hexagon::MEMh_ADDi_MEM_V4 :
    case Hexagon::MEMh_SUBi_MEM_V4 :
    case Hexagon::MEMh_ADDr_MEM_V4 :
    case Hexagon::MEMh_SUBr_MEM_V4 :
    case Hexagon::MEMh_ANDr_MEM_V4 :
    case Hexagon::MEMh_ORr_MEM_V4 :
    case Hexagon::MEMb_ADDSUBi_indexed_MEM_V4 :
    case Hexagon::MEMb_ADDi_indexed_MEM_V4 :
    case Hexagon::MEMb_SUBi_indexed_MEM_V4 :
    case Hexagon::MEMb_ADDr_indexed_MEM_V4 :
    case Hexagon::MEMb_SUBr_indexed_MEM_V4 :
    case Hexagon::MEMb_ANDr_indexed_MEM_V4 :
    case Hexagon::MEMb_ORr_indexed_MEM_V4 :
    case Hexagon::MEMb_ADDSUBi_MEM_V4 :
    case Hexagon::MEMb_ADDi_MEM_V4 :
    case Hexagon::MEMb_SUBi_MEM_V4 :
    case Hexagon::MEMb_ADDr_MEM_V4 :
    case Hexagon::MEMb_SUBr_MEM_V4 :
    case Hexagon::MEMb_ANDr_MEM_V4 :
    case Hexagon::MEMb_ORr_MEM_V4 :
      llvm_unreachable("Needs implementing.");
  }
}

unsigned HexagonInstrInfo::getNormalBranchForm(const MachineInstr* MI) const {
  switch(MI->getOpcode()) {
    default: llvm_unreachable("Unknown type of jump instruction.");
    // JMP_EQri
    case Hexagon::JMP_EQriPt_ie_nv_V4:
      return Hexagon::JMP_EQriPt_nv_V4;
    case Hexagon::JMP_EQriNotPt_ie_nv_V4:
      return Hexagon::JMP_EQriNotPt_nv_V4;
    case Hexagon::JMP_EQriPnt_ie_nv_V4:
      return Hexagon::JMP_EQriPnt_nv_V4;
    case Hexagon::JMP_EQriNotPnt_ie_nv_V4:
      return Hexagon::JMP_EQriNotPnt_nv_V4;

    // JMP_EQri -- with -1
    case Hexagon::JMP_EQriPtneg_ie_nv_V4:
      return Hexagon::JMP_EQriPtneg_nv_V4;
    case Hexagon::JMP_EQriNotPtneg_ie_nv_V4:
      return Hexagon::JMP_EQriNotPtneg_nv_V4;
    case Hexagon::JMP_EQriPntneg_ie_nv_V4:
      return Hexagon::JMP_EQriPntneg_nv_V4;
    case Hexagon::JMP_EQriNotPntneg_ie_nv_V4:
      return Hexagon::JMP_EQriNotPntneg_nv_V4;

    // JMP_EQrr
    case Hexagon::JMP_EQrrPt_ie_nv_V4:
      return Hexagon::JMP_EQrrPt_nv_V4;
    case Hexagon::JMP_EQrrNotPt_ie_nv_V4:
      return Hexagon::JMP_EQrrNotPt_nv_V4;
    case Hexagon::JMP_EQrrPnt_ie_nv_V4:
      return Hexagon::JMP_EQrrPnt_nv_V4;
    case Hexagon::JMP_EQrrNotPnt_ie_nv_V4:
      return Hexagon::JMP_EQrrNotPnt_nv_V4;

    // JMP_GTri
    case Hexagon::JMP_GTriPt_ie_nv_V4:
      return Hexagon::JMP_GTriPt_nv_V4;
    case Hexagon::JMP_GTriNotPt_ie_nv_V4:
      return Hexagon::JMP_GTriNotPt_nv_V4;
    case Hexagon::JMP_GTriPnt_ie_nv_V4:
      return Hexagon::JMP_GTriPnt_nv_V4;
    case Hexagon::JMP_GTriNotPnt_ie_nv_V4:
      return Hexagon::JMP_GTriNotPnt_nv_V4;

    // JMP_GTri -- with -1
    case Hexagon::JMP_GTriPtneg_ie_nv_V4:
      return Hexagon::JMP_GTriPtneg_nv_V4;
    case Hexagon::JMP_GTriNotPtneg_ie_nv_V4:
      return Hexagon::JMP_GTriNotPtneg_nv_V4;
    case Hexagon::JMP_GTriPntneg_ie_nv_V4:
      return Hexagon::JMP_GTriPntneg_nv_V4;
    case Hexagon::JMP_GTriNotPntneg_ie_nv_V4:
      return Hexagon::JMP_GTriNotPntneg_nv_V4;

    // JMP_GTrr
    case Hexagon::JMP_GTrrPt_ie_nv_V4:
      return Hexagon::JMP_GTrrPt_nv_V4;
    case Hexagon::JMP_GTrrNotPt_ie_nv_V4:
      return Hexagon::JMP_GTrrNotPt_nv_V4;
    case Hexagon::JMP_GTrrPnt_ie_nv_V4:
      return Hexagon::JMP_GTrrPnt_nv_V4;
    case Hexagon::JMP_GTrrNotPnt_ie_nv_V4:
      return Hexagon::JMP_GTrrNotPnt_nv_V4;

    // JMP_GTrrdn
    case Hexagon::JMP_GTrrdnPt_ie_nv_V4:
      return Hexagon::JMP_GTrrdnPt_nv_V4;
    case Hexagon::JMP_GTrrdnNotPt_ie_nv_V4:
      return Hexagon::JMP_GTrrdnNotPt_nv_V4;
    case Hexagon::JMP_GTrrdnPnt_ie_nv_V4:
      return Hexagon::JMP_GTrrdnPnt_nv_V4;
    case Hexagon::JMP_GTrrdnNotPnt_ie_nv_V4:
      return Hexagon::JMP_GTrrdnNotPnt_nv_V4;

    // JMP_GTUri
    case Hexagon::JMP_GTUriPt_ie_nv_V4:
      return Hexagon::JMP_GTUriPt_nv_V4;
    case Hexagon::JMP_GTUriNotPt_ie_nv_V4:
      return Hexagon::JMP_GTUriNotPt_nv_V4;
    case Hexagon::JMP_GTUriPnt_ie_nv_V4:
      return Hexagon::JMP_GTUriPnt_nv_V4;
    case Hexagon::JMP_GTUriNotPnt_ie_nv_V4:
      return Hexagon::JMP_GTUriNotPnt_nv_V4;

    // JMP_GTUrr
    case Hexagon::JMP_GTUrrPt_ie_nv_V4:
      return Hexagon::JMP_GTUrrPt_nv_V4;
    case Hexagon::JMP_GTUrrNotPt_ie_nv_V4:
      return Hexagon::JMP_GTUrrNotPt_nv_V4;
    case Hexagon::JMP_GTUrrPnt_ie_nv_V4:
      return Hexagon::JMP_GTUrrPnt_nv_V4;
    case Hexagon::JMP_GTUrrNotPnt_ie_nv_V4:
      return Hexagon::JMP_GTUrrNotPnt_nv_V4;

    // JMP_GTUrrdn
    case Hexagon::JMP_GTUrrdnPt_ie_nv_V4:
      return Hexagon::JMP_GTUrrdnPt_nv_V4;
    case Hexagon::JMP_GTUrrdnNotPt_ie_nv_V4:
      return Hexagon::JMP_GTUrrdnNotPt_nv_V4;
    case Hexagon::JMP_GTUrrdnPnt_ie_nv_V4:
      return Hexagon::JMP_GTUrrdnPnt_nv_V4;
    case Hexagon::JMP_GTUrrdnNotPnt_ie_nv_V4:
      return Hexagon::JMP_GTUrrdnNotPnt_nv_V4;
  }
}


bool HexagonInstrInfo::isNewValueStore(const MachineInstr *MI) const {
  switch (MI->getOpcode()) {
    default: return false;
    // Store Byte
    case Hexagon::STrib_nv_V4:
    case Hexagon::STrib_indexed_nv_V4:
    case Hexagon::STrib_indexed_shl_nv_V4:
    case Hexagon::STrib_shl_nv_V4:
    case Hexagon::STrib_GP_nv_V4:
    case Hexagon::STb_GP_nv_V4:
    case Hexagon::POST_STbri_nv_V4:
    case Hexagon::STrib_cPt_nv_V4:
    case Hexagon::STrib_cdnPt_nv_V4:
    case Hexagon::STrib_cNotPt_nv_V4:
    case Hexagon::STrib_cdnNotPt_nv_V4:
    case Hexagon::STrib_indexed_cPt_nv_V4:
    case Hexagon::STrib_indexed_cdnPt_nv_V4:
    case Hexagon::STrib_indexed_cNotPt_nv_V4:
    case Hexagon::STrib_indexed_cdnNotPt_nv_V4:
    case Hexagon::STrib_indexed_shl_cPt_nv_V4:
    case Hexagon::STrib_indexed_shl_cdnPt_nv_V4:
    case Hexagon::STrib_indexed_shl_cNotPt_nv_V4:
    case Hexagon::STrib_indexed_shl_cdnNotPt_nv_V4:
    case Hexagon::POST_STbri_cPt_nv_V4:
    case Hexagon::POST_STbri_cdnPt_nv_V4:
    case Hexagon::POST_STbri_cNotPt_nv_V4:
    case Hexagon::POST_STbri_cdnNotPt_nv_V4:
    case Hexagon::STb_GP_cPt_nv_V4:
    case Hexagon::STb_GP_cNotPt_nv_V4:
    case Hexagon::STb_GP_cdnPt_nv_V4:
    case Hexagon::STb_GP_cdnNotPt_nv_V4:
    case Hexagon::STrib_GP_cPt_nv_V4:
    case Hexagon::STrib_GP_cNotPt_nv_V4:
    case Hexagon::STrib_GP_cdnPt_nv_V4:
    case Hexagon::STrib_GP_cdnNotPt_nv_V4:
    case Hexagon::STrib_abs_nv_V4:
    case Hexagon::STrib_abs_cPt_nv_V4:
    case Hexagon::STrib_abs_cdnPt_nv_V4:
    case Hexagon::STrib_abs_cNotPt_nv_V4:
    case Hexagon::STrib_abs_cdnNotPt_nv_V4:
    case Hexagon::STrib_imm_abs_nv_V4:
    case Hexagon::STrib_imm_abs_cPt_nv_V4:
    case Hexagon::STrib_imm_abs_cdnPt_nv_V4:
    case Hexagon::STrib_imm_abs_cNotPt_nv_V4:
    case Hexagon::STrib_imm_abs_cdnNotPt_nv_V4:

    // Store Halfword
    case Hexagon::STrih_nv_V4:
    case Hexagon::STrih_indexed_nv_V4:
    case Hexagon::STrih_indexed_shl_nv_V4:
    case Hexagon::STrih_shl_nv_V4:
    case Hexagon::STrih_GP_nv_V4:
    case Hexagon::STh_GP_nv_V4:
    case Hexagon::POST_SThri_nv_V4:
    case Hexagon::STrih_cPt_nv_V4:
    case Hexagon::STrih_cdnPt_nv_V4:
    case Hexagon::STrih_cNotPt_nv_V4:
    case Hexagon::STrih_cdnNotPt_nv_V4:
    case Hexagon::STrih_indexed_cPt_nv_V4:
    case Hexagon::STrih_indexed_cdnPt_nv_V4:
    case Hexagon::STrih_indexed_cNotPt_nv_V4:
    case Hexagon::STrih_indexed_cdnNotPt_nv_V4:
    case Hexagon::STrih_indexed_shl_cPt_nv_V4:
    case Hexagon::STrih_indexed_shl_cdnPt_nv_V4:
    case Hexagon::STrih_indexed_shl_cNotPt_nv_V4:
    case Hexagon::STrih_indexed_shl_cdnNotPt_nv_V4:
    case Hexagon::POST_SThri_cPt_nv_V4:
    case Hexagon::POST_SThri_cdnPt_nv_V4:
    case Hexagon::POST_SThri_cNotPt_nv_V4:
    case Hexagon::POST_SThri_cdnNotPt_nv_V4:
    case Hexagon::STh_GP_cPt_nv_V4:
    case Hexagon::STh_GP_cNotPt_nv_V4:
    case Hexagon::STh_GP_cdnPt_nv_V4:
    case Hexagon::STh_GP_cdnNotPt_nv_V4:
    case Hexagon::STrih_GP_cPt_nv_V4:
    case Hexagon::STrih_GP_cNotPt_nv_V4:
    case Hexagon::STrih_GP_cdnPt_nv_V4:
    case Hexagon::STrih_GP_cdnNotPt_nv_V4:
    case Hexagon::STrih_abs_nv_V4:
    case Hexagon::STrih_abs_cPt_nv_V4:
    case Hexagon::STrih_abs_cdnPt_nv_V4:
    case Hexagon::STrih_abs_cNotPt_nv_V4:
    case Hexagon::STrih_abs_cdnNotPt_nv_V4:
    case Hexagon::STrih_imm_abs_nv_V4:
    case Hexagon::STrih_imm_abs_cPt_nv_V4:
    case Hexagon::STrih_imm_abs_cdnPt_nv_V4:
    case Hexagon::STrih_imm_abs_cNotPt_nv_V4:
    case Hexagon::STrih_imm_abs_cdnNotPt_nv_V4:

    // Store Word
    case Hexagon::STriw_nv_V4:
    case Hexagon::STriw_indexed_nv_V4:
    case Hexagon::STriw_indexed_shl_nv_V4:
    case Hexagon::STriw_shl_nv_V4:
    case Hexagon::STriw_GP_nv_V4:
    case Hexagon::STw_GP_nv_V4:
    case Hexagon::POST_STwri_nv_V4:
    case Hexagon::STriw_cPt_nv_V4:
    case Hexagon::STriw_cdnPt_nv_V4:
    case Hexagon::STriw_cNotPt_nv_V4:
    case Hexagon::STriw_cdnNotPt_nv_V4:
    case Hexagon::STriw_indexed_cPt_nv_V4:
    case Hexagon::STriw_indexed_cdnPt_nv_V4:
    case Hexagon::STriw_indexed_cNotPt_nv_V4:
    case Hexagon::STriw_indexed_cdnNotPt_nv_V4:
    case Hexagon::STriw_indexed_shl_cPt_nv_V4:
    case Hexagon::STriw_indexed_shl_cdnPt_nv_V4:
    case Hexagon::STriw_indexed_shl_cNotPt_nv_V4:
    case Hexagon::STriw_indexed_shl_cdnNotPt_nv_V4:
    case Hexagon::POST_STwri_cPt_nv_V4:
    case Hexagon::POST_STwri_cdnPt_nv_V4:
    case Hexagon::POST_STwri_cNotPt_nv_V4:
    case Hexagon::POST_STwri_cdnNotPt_nv_V4:
    case Hexagon::STw_GP_cPt_nv_V4:
    case Hexagon::STw_GP_cNotPt_nv_V4:
    case Hexagon::STw_GP_cdnPt_nv_V4:
    case Hexagon::STw_GP_cdnNotPt_nv_V4:
    case Hexagon::STriw_GP_cPt_nv_V4:
    case Hexagon::STriw_GP_cNotPt_nv_V4:
    case Hexagon::STriw_GP_cdnPt_nv_V4:
    case Hexagon::STriw_GP_cdnNotPt_nv_V4:
    case Hexagon::STriw_abs_nv_V4:
    case Hexagon::STriw_abs_cPt_nv_V4:
    case Hexagon::STriw_abs_cdnPt_nv_V4:
    case Hexagon::STriw_abs_cNotPt_nv_V4:
    case Hexagon::STriw_abs_cdnNotPt_nv_V4:
    case Hexagon::STriw_imm_abs_nv_V4:
    case Hexagon::STriw_imm_abs_cPt_nv_V4:
    case Hexagon::STriw_imm_abs_cdnPt_nv_V4:
    case Hexagon::STriw_imm_abs_cNotPt_nv_V4:
    case Hexagon::STriw_imm_abs_cdnNotPt_nv_V4:
      return true;
  }
}

bool HexagonInstrInfo::isPostIncrement (const MachineInstr* MI) const {
  switch (MI->getOpcode())
  {
    default: return false;
    // Load Byte
    case Hexagon::POST_LDrib:
    case Hexagon::POST_LDrib_cPt:
    case Hexagon::POST_LDrib_cNotPt:
    case Hexagon::POST_LDrib_cdnPt_V4:
    case Hexagon::POST_LDrib_cdnNotPt_V4:

    // Load unsigned byte
    case Hexagon::POST_LDriub:
    case Hexagon::POST_LDriub_cPt:
    case Hexagon::POST_LDriub_cNotPt:
    case Hexagon::POST_LDriub_cdnPt_V4:
    case Hexagon::POST_LDriub_cdnNotPt_V4:

    // Load halfword
    case Hexagon::POST_LDrih:
    case Hexagon::POST_LDrih_cPt:
    case Hexagon::POST_LDrih_cNotPt:
    case Hexagon::POST_LDrih_cdnPt_V4:
    case Hexagon::POST_LDrih_cdnNotPt_V4:

    // Load unsigned halfword
    case Hexagon::POST_LDriuh:
    case Hexagon::POST_LDriuh_cPt:
    case Hexagon::POST_LDriuh_cNotPt:
    case Hexagon::POST_LDriuh_cdnPt_V4:
    case Hexagon::POST_LDriuh_cdnNotPt_V4:

    // Load word
    case Hexagon::POST_LDriw:
    case Hexagon::POST_LDriw_cPt:
    case Hexagon::POST_LDriw_cNotPt:
    case Hexagon::POST_LDriw_cdnPt_V4:
    case Hexagon::POST_LDriw_cdnNotPt_V4:

    // Load double word
    case Hexagon::POST_LDrid:
    case Hexagon::POST_LDrid_cPt:
    case Hexagon::POST_LDrid_cNotPt:
    case Hexagon::POST_LDrid_cdnPt_V4:
    case Hexagon::POST_LDrid_cdnNotPt_V4:

    // Store byte
    case Hexagon::POST_STbri:
    case Hexagon::POST_STbri_cPt:
    case Hexagon::POST_STbri_cNotPt:
    case Hexagon::POST_STbri_cdnPt_V4:
    case Hexagon::POST_STbri_cdnNotPt_V4:

    // Store halfword
    case Hexagon::POST_SThri:
    case Hexagon::POST_SThri_cPt:
    case Hexagon::POST_SThri_cNotPt:
    case Hexagon::POST_SThri_cdnPt_V4:
    case Hexagon::POST_SThri_cdnNotPt_V4:

    // Store word
    case Hexagon::POST_STwri:
    case Hexagon::POST_STwri_cPt:
    case Hexagon::POST_STwri_cNotPt:
    case Hexagon::POST_STwri_cdnPt_V4:
    case Hexagon::POST_STwri_cdnNotPt_V4:

    // Store double word
    case Hexagon::POST_STdri:
    case Hexagon::POST_STdri_cPt:
    case Hexagon::POST_STdri_cNotPt:
    case Hexagon::POST_STdri_cdnPt_V4:
    case Hexagon::POST_STdri_cdnNotPt_V4:
      return true;
  }
}

bool HexagonInstrInfo::isSaveCalleeSavedRegsCall(const MachineInstr *MI) const {
  return MI->getOpcode() == Hexagon::SAVE_REGISTERS_CALL_V4;
}

bool HexagonInstrInfo::isPredicable(MachineInstr *MI) const {
  bool isPred = MI->getDesc().isPredicable();

  if (!isPred)
    return false;

  const int Opc = MI->getOpcode();

  switch(Opc) {
  case Hexagon::TFRI:
    return isInt<12>(MI->getOperand(1).getImm());

  case Hexagon::STrid:
  case Hexagon::STrid_indexed:
    return isShiftedUInt<6,3>(MI->getOperand(1).getImm());

  case Hexagon::STriw:
  case Hexagon::STriw_indexed:
  case Hexagon::STriw_nv_V4:
    return isShiftedUInt<6,2>(MI->getOperand(1).getImm());

  case Hexagon::STrih:
  case Hexagon::STrih_indexed:
  case Hexagon::STrih_nv_V4:
    return isShiftedUInt<6,1>(MI->getOperand(1).getImm());

  case Hexagon::STrib:
  case Hexagon::STrib_indexed:
  case Hexagon::STrib_nv_V4:
    return isUInt<6>(MI->getOperand(1).getImm());

  case Hexagon::LDrid:
  case Hexagon::LDrid_indexed:
    return isShiftedUInt<6,3>(MI->getOperand(2).getImm());

  case Hexagon::LDriw:
  case Hexagon::LDriw_indexed:
    return isShiftedUInt<6,2>(MI->getOperand(2).getImm());

  case Hexagon::LDrih:
  case Hexagon::LDriuh:
  case Hexagon::LDrih_indexed:
  case Hexagon::LDriuh_indexed:
    return isShiftedUInt<6,1>(MI->getOperand(2).getImm());

  case Hexagon::LDrib:
  case Hexagon::LDriub:
  case Hexagon::LDrib_indexed:
  case Hexagon::LDriub_indexed:
    return isUInt<6>(MI->getOperand(2).getImm());

  case Hexagon::POST_LDrid:
    return isShiftedInt<4,3>(MI->getOperand(3).getImm());

  case Hexagon::POST_LDriw:
    return isShiftedInt<4,2>(MI->getOperand(3).getImm());

  case Hexagon::POST_LDrih:
  case Hexagon::POST_LDriuh:
    return isShiftedInt<4,1>(MI->getOperand(3).getImm());

  case Hexagon::POST_LDrib:
  case Hexagon::POST_LDriub:
    return isInt<4>(MI->getOperand(3).getImm());

  case Hexagon::STrib_imm_V4:
  case Hexagon::STrih_imm_V4:
  case Hexagon::STriw_imm_V4:
    return (isUInt<6>(MI->getOperand(1).getImm()) &&
            isInt<6>(MI->getOperand(2).getImm()));

  case Hexagon::ADD_ri:
    return isInt<8>(MI->getOperand(2).getImm());

  case Hexagon::ASLH:
  case Hexagon::ASRH:
  case Hexagon::SXTB:
  case Hexagon::SXTH:
  case Hexagon::ZXTB:
  case Hexagon::ZXTH:
    return Subtarget.hasV4TOps();

  case Hexagon::JMPR:
    return false;
  }

  return true;
}

// This function performs the following inversiones:
//
//  cPt    ---> cNotPt
//  cNotPt ---> cPt
//
// however, these inversiones are NOT included:
//
//  cdnPt      -X-> cdnNotPt
//  cdnNotPt   -X-> cdnPt
//  cPt_nv     -X-> cNotPt_nv (new value stores)
//  cNotPt_nv  -X-> cPt_nv    (new value stores)
//
// because only the following transformations are allowed:
//
//  cNotPt  ---> cdnNotPt
//  cPt     ---> cdnPt
//  cNotPt  ---> cNotPt_nv
//  cPt     ---> cPt_nv
unsigned HexagonInstrInfo::getInvertedPredicatedOpcode(const int Opc) const {
  switch(Opc) {
    default: llvm_unreachable("Unexpected predicated instruction");
    case Hexagon::TFR_cPt:
      return Hexagon::TFR_cNotPt;
    case Hexagon::TFR_cNotPt:
      return Hexagon::TFR_cPt;

    case Hexagon::TFRI_cPt:
      return Hexagon::TFRI_cNotPt;
    case Hexagon::TFRI_cNotPt:
      return Hexagon::TFRI_cPt;

    case Hexagon::JMP_c:
      return Hexagon::JMP_cNot;
    case Hexagon::JMP_cNot:
      return Hexagon::JMP_c;

    case Hexagon::ADD_ri_cPt:
      return Hexagon::ADD_ri_cNotPt;
    case Hexagon::ADD_ri_cNotPt:
      return Hexagon::ADD_ri_cPt;

    case Hexagon::ADD_rr_cPt:
      return Hexagon::ADD_rr_cNotPt;
    case Hexagon::ADD_rr_cNotPt:
      return Hexagon::ADD_rr_cPt;

    case Hexagon::XOR_rr_cPt:
      return Hexagon::XOR_rr_cNotPt;
    case Hexagon::XOR_rr_cNotPt:
      return Hexagon::XOR_rr_cPt;

    case Hexagon::AND_rr_cPt:
      return Hexagon::AND_rr_cNotPt;
    case Hexagon::AND_rr_cNotPt:
      return Hexagon::AND_rr_cPt;

    case Hexagon::OR_rr_cPt:
      return Hexagon::OR_rr_cNotPt;
    case Hexagon::OR_rr_cNotPt:
      return Hexagon::OR_rr_cPt;

    case Hexagon::SUB_rr_cPt:
      return Hexagon::SUB_rr_cNotPt;
    case Hexagon::SUB_rr_cNotPt:
      return Hexagon::SUB_rr_cPt;

    case Hexagon::COMBINE_rr_cPt:
      return Hexagon::COMBINE_rr_cNotPt;
    case Hexagon::COMBINE_rr_cNotPt:
      return Hexagon::COMBINE_rr_cPt;

    case Hexagon::ASLH_cPt_V4:
      return Hexagon::ASLH_cNotPt_V4;
    case Hexagon::ASLH_cNotPt_V4:
      return Hexagon::ASLH_cPt_V4;

    case Hexagon::ASRH_cPt_V4:
      return Hexagon::ASRH_cNotPt_V4;
    case Hexagon::ASRH_cNotPt_V4:
      return Hexagon::ASRH_cPt_V4;

    case Hexagon::SXTB_cPt_V4:
      return Hexagon::SXTB_cNotPt_V4;
    case Hexagon::SXTB_cNotPt_V4:
      return Hexagon::SXTB_cPt_V4;

    case Hexagon::SXTH_cPt_V4:
      return Hexagon::SXTH_cNotPt_V4;
    case Hexagon::SXTH_cNotPt_V4:
      return Hexagon::SXTH_cPt_V4;

    case Hexagon::ZXTB_cPt_V4:
      return Hexagon::ZXTB_cNotPt_V4;
    case Hexagon::ZXTB_cNotPt_V4:
      return Hexagon::ZXTB_cPt_V4;

    case Hexagon::ZXTH_cPt_V4:
      return Hexagon::ZXTH_cNotPt_V4;
    case Hexagon::ZXTH_cNotPt_V4:
      return Hexagon::ZXTH_cPt_V4;


    case Hexagon::JMPR_cPt:
      return Hexagon::JMPR_cNotPt;
    case Hexagon::JMPR_cNotPt:
      return Hexagon::JMPR_cPt;

  // V4 indexed+scaled load.
    case Hexagon::LDrid_indexed_cPt_V4:
      return Hexagon::LDrid_indexed_cNotPt_V4;
    case Hexagon::LDrid_indexed_cNotPt_V4:
      return Hexagon::LDrid_indexed_cPt_V4;

    case Hexagon::LDrid_indexed_shl_cPt_V4:
      return Hexagon::LDrid_indexed_shl_cNotPt_V4;
    case Hexagon::LDrid_indexed_shl_cNotPt_V4:
      return Hexagon::LDrid_indexed_shl_cPt_V4;

    case Hexagon::LDrib_indexed_cPt_V4:
      return Hexagon::LDrib_indexed_cNotPt_V4;
    case Hexagon::LDrib_indexed_cNotPt_V4:
      return Hexagon::LDrib_indexed_cPt_V4;

    case Hexagon::LDriub_indexed_cPt_V4:
      return Hexagon::LDriub_indexed_cNotPt_V4;
    case Hexagon::LDriub_indexed_cNotPt_V4:
      return Hexagon::LDriub_indexed_cPt_V4;

    case Hexagon::LDrib_indexed_shl_cPt_V4:
      return Hexagon::LDrib_indexed_shl_cNotPt_V4;
    case Hexagon::LDrib_indexed_shl_cNotPt_V4:
      return Hexagon::LDrib_indexed_shl_cPt_V4;

    case Hexagon::LDriub_indexed_shl_cPt_V4:
      return Hexagon::LDriub_indexed_shl_cNotPt_V4;
    case Hexagon::LDriub_indexed_shl_cNotPt_V4:
      return Hexagon::LDriub_indexed_shl_cPt_V4;

    case Hexagon::LDrih_indexed_cPt_V4:
      return Hexagon::LDrih_indexed_cNotPt_V4;
    case Hexagon::LDrih_indexed_cNotPt_V4:
      return Hexagon::LDrih_indexed_cPt_V4;

    case Hexagon::LDriuh_indexed_cPt_V4:
      return Hexagon::LDriuh_indexed_cNotPt_V4;
    case Hexagon::LDriuh_indexed_cNotPt_V4:
      return Hexagon::LDriuh_indexed_cPt_V4;

    case Hexagon::LDrih_indexed_shl_cPt_V4:
      return Hexagon::LDrih_indexed_shl_cNotPt_V4;
    case Hexagon::LDrih_indexed_shl_cNotPt_V4:
      return Hexagon::LDrih_indexed_shl_cPt_V4;

    case Hexagon::LDriuh_indexed_shl_cPt_V4:
      return Hexagon::LDriuh_indexed_shl_cNotPt_V4;
    case Hexagon::LDriuh_indexed_shl_cNotPt_V4:
      return Hexagon::LDriuh_indexed_shl_cPt_V4;

    case Hexagon::LDriw_indexed_cPt_V4:
      return Hexagon::LDriw_indexed_cNotPt_V4;
    case Hexagon::LDriw_indexed_cNotPt_V4:
      return Hexagon::LDriw_indexed_cPt_V4;

    case Hexagon::LDriw_indexed_shl_cPt_V4:
      return Hexagon::LDriw_indexed_shl_cNotPt_V4;
    case Hexagon::LDriw_indexed_shl_cNotPt_V4:
      return Hexagon::LDriw_indexed_shl_cPt_V4;

    // Byte.
    case Hexagon::POST_STbri_cPt:
      return Hexagon::POST_STbri_cNotPt;
    case Hexagon::POST_STbri_cNotPt:
      return Hexagon::POST_STbri_cPt;

    case Hexagon::STrib_cPt:
      return Hexagon::STrib_cNotPt;
    case Hexagon::STrib_cNotPt:
      return Hexagon::STrib_cPt;

    case Hexagon::STrib_indexed_cPt:
      return Hexagon::STrib_indexed_cNotPt;
    case Hexagon::STrib_indexed_cNotPt:
      return Hexagon::STrib_indexed_cPt;

    case Hexagon::STrib_imm_cPt_V4:
      return Hexagon::STrib_imm_cNotPt_V4;
    case Hexagon::STrib_imm_cNotPt_V4:
      return Hexagon::STrib_imm_cPt_V4;

    case Hexagon::STrib_indexed_shl_cPt_V4:
      return Hexagon::STrib_indexed_shl_cNotPt_V4;
    case Hexagon::STrib_indexed_shl_cNotPt_V4:
      return Hexagon::STrib_indexed_shl_cPt_V4;

  // Halfword.
    case Hexagon::POST_SThri_cPt:
      return Hexagon::POST_SThri_cNotPt;
    case Hexagon::POST_SThri_cNotPt:
      return Hexagon::POST_SThri_cPt;

    case Hexagon::STrih_cPt:
      return Hexagon::STrih_cNotPt;
    case Hexagon::STrih_cNotPt:
      return Hexagon::STrih_cPt;

    case Hexagon::STrih_indexed_cPt:
      return Hexagon::STrih_indexed_cNotPt;
    case Hexagon::STrih_indexed_cNotPt:
      return Hexagon::STrih_indexed_cPt;

    case Hexagon::STrih_imm_cPt_V4:
      return Hexagon::STrih_imm_cNotPt_V4;
    case Hexagon::STrih_imm_cNotPt_V4:
      return Hexagon::STrih_imm_cPt_V4;

    case Hexagon::STrih_indexed_shl_cPt_V4:
      return Hexagon::STrih_indexed_shl_cNotPt_V4;
    case Hexagon::STrih_indexed_shl_cNotPt_V4:
      return Hexagon::STrih_indexed_shl_cPt_V4;

  // Word.
    case Hexagon::POST_STwri_cPt:
      return Hexagon::POST_STwri_cNotPt;
    case Hexagon::POST_STwri_cNotPt:
      return Hexagon::POST_STwri_cPt;

    case Hexagon::STriw_cPt:
      return Hexagon::STriw_cNotPt;
    case Hexagon::STriw_cNotPt:
      return Hexagon::STriw_cPt;

    case Hexagon::STriw_indexed_cPt:
      return Hexagon::STriw_indexed_cNotPt;
    case Hexagon::STriw_indexed_cNotPt:
      return Hexagon::STriw_indexed_cPt;

    case Hexagon::STriw_indexed_shl_cPt_V4:
      return Hexagon::STriw_indexed_shl_cNotPt_V4;
    case Hexagon::STriw_indexed_shl_cNotPt_V4:
      return Hexagon::STriw_indexed_shl_cPt_V4;

    case Hexagon::STriw_imm_cPt_V4:
      return Hexagon::STriw_imm_cNotPt_V4;
    case Hexagon::STriw_imm_cNotPt_V4:
      return Hexagon::STriw_imm_cPt_V4;

  // Double word.
    case Hexagon::POST_STdri_cPt:
      return Hexagon::POST_STdri_cNotPt;
    case Hexagon::POST_STdri_cNotPt:
      return Hexagon::POST_STdri_cPt;

    case Hexagon::STrid_cPt:
      return Hexagon::STrid_cNotPt;
    case Hexagon::STrid_cNotPt:
      return Hexagon::STrid_cPt;

    case Hexagon::STrid_indexed_cPt:
      return Hexagon::STrid_indexed_cNotPt;
    case Hexagon::STrid_indexed_cNotPt:
      return Hexagon::STrid_indexed_cPt;

    case Hexagon::STrid_indexed_shl_cPt_V4:
      return Hexagon::STrid_indexed_shl_cNotPt_V4;
    case Hexagon::STrid_indexed_shl_cNotPt_V4:
      return Hexagon::STrid_indexed_shl_cPt_V4;

    // V4 Store to global address.
    case Hexagon::STd_GP_cPt_V4:
      return Hexagon::STd_GP_cNotPt_V4;
    case Hexagon::STd_GP_cNotPt_V4:
      return Hexagon::STd_GP_cPt_V4;

    case Hexagon::STb_GP_cPt_V4:
      return Hexagon::STb_GP_cNotPt_V4;
    case Hexagon::STb_GP_cNotPt_V4:
      return Hexagon::STb_GP_cPt_V4;

    case Hexagon::STh_GP_cPt_V4:
      return Hexagon::STh_GP_cNotPt_V4;
    case Hexagon::STh_GP_cNotPt_V4:
      return Hexagon::STh_GP_cPt_V4;

    case Hexagon::STw_GP_cPt_V4:
      return Hexagon::STw_GP_cNotPt_V4;
    case Hexagon::STw_GP_cNotPt_V4:
      return Hexagon::STw_GP_cPt_V4;

    case Hexagon::STrid_GP_cPt_V4:
      return Hexagon::STrid_GP_cNotPt_V4;
    case Hexagon::STrid_GP_cNotPt_V4:
      return Hexagon::STrid_GP_cPt_V4;

    case Hexagon::STrib_GP_cPt_V4:
      return Hexagon::STrib_GP_cNotPt_V4;
    case Hexagon::STrib_GP_cNotPt_V4:
      return Hexagon::STrib_GP_cPt_V4;

    case Hexagon::STrih_GP_cPt_V4:
      return Hexagon::STrih_GP_cNotPt_V4;
    case Hexagon::STrih_GP_cNotPt_V4:
      return Hexagon::STrih_GP_cPt_V4;

    case Hexagon::STriw_GP_cPt_V4:
      return Hexagon::STriw_GP_cNotPt_V4;
    case Hexagon::STriw_GP_cNotPt_V4:
      return Hexagon::STriw_GP_cPt_V4;

  // Load.
    case Hexagon::LDrid_cPt:
      return Hexagon::LDrid_cNotPt;
    case Hexagon::LDrid_cNotPt:
      return Hexagon::LDrid_cPt;

    case Hexagon::LDriw_cPt:
      return Hexagon::LDriw_cNotPt;
    case Hexagon::LDriw_cNotPt:
      return Hexagon::LDriw_cPt;

    case Hexagon::LDrih_cPt:
      return Hexagon::LDrih_cNotPt;
    case Hexagon::LDrih_cNotPt:
      return Hexagon::LDrih_cPt;

    case Hexagon::LDriuh_cPt:
      return Hexagon::LDriuh_cNotPt;
    case Hexagon::LDriuh_cNotPt:
      return Hexagon::LDriuh_cPt;

    case Hexagon::LDrib_cPt:
      return Hexagon::LDrib_cNotPt;
    case Hexagon::LDrib_cNotPt:
      return Hexagon::LDrib_cPt;

    case Hexagon::LDriub_cPt:
      return Hexagon::LDriub_cNotPt;
    case Hexagon::LDriub_cNotPt:
      return Hexagon::LDriub_cPt;

 // Load Indexed.
    case Hexagon::LDrid_indexed_cPt:
      return Hexagon::LDrid_indexed_cNotPt;
    case Hexagon::LDrid_indexed_cNotPt:
      return Hexagon::LDrid_indexed_cPt;

    case Hexagon::LDriw_indexed_cPt:
      return Hexagon::LDriw_indexed_cNotPt;
    case Hexagon::LDriw_indexed_cNotPt:
      return Hexagon::LDriw_indexed_cPt;

    case Hexagon::LDrih_indexed_cPt:
      return Hexagon::LDrih_indexed_cNotPt;
    case Hexagon::LDrih_indexed_cNotPt:
      return Hexagon::LDrih_indexed_cPt;

    case Hexagon::LDriuh_indexed_cPt:
      return Hexagon::LDriuh_indexed_cNotPt;
    case Hexagon::LDriuh_indexed_cNotPt:
      return Hexagon::LDriuh_indexed_cPt;

    case Hexagon::LDrib_indexed_cPt:
      return Hexagon::LDrib_indexed_cNotPt;
    case Hexagon::LDrib_indexed_cNotPt:
      return Hexagon::LDrib_indexed_cPt;

    case Hexagon::LDriub_indexed_cPt:
      return Hexagon::LDriub_indexed_cNotPt;
    case Hexagon::LDriub_indexed_cNotPt:
      return Hexagon::LDriub_indexed_cPt;

  // Post Inc Load.
    case Hexagon::POST_LDrid_cPt:
      return Hexagon::POST_LDrid_cNotPt;
    case Hexagon::POST_LDriw_cNotPt:
      return Hexagon::POST_LDriw_cPt;

    case Hexagon::POST_LDrih_cPt:
      return Hexagon::POST_LDrih_cNotPt;
    case Hexagon::POST_LDrih_cNotPt:
      return Hexagon::POST_LDrih_cPt;

    case Hexagon::POST_LDriuh_cPt:
      return Hexagon::POST_LDriuh_cNotPt;
    case Hexagon::POST_LDriuh_cNotPt:
      return Hexagon::POST_LDriuh_cPt;

    case Hexagon::POST_LDrib_cPt:
      return Hexagon::POST_LDrib_cNotPt;
    case Hexagon::POST_LDrib_cNotPt:
      return Hexagon::POST_LDrib_cPt;

    case Hexagon::POST_LDriub_cPt:
      return Hexagon::POST_LDriub_cNotPt;
    case Hexagon::POST_LDriub_cNotPt:
      return Hexagon::POST_LDriub_cPt;

  // Dealloc_return.
    case Hexagon::DEALLOC_RET_cPt_V4:
      return Hexagon::DEALLOC_RET_cNotPt_V4;
    case Hexagon::DEALLOC_RET_cNotPt_V4:
      return Hexagon::DEALLOC_RET_cPt_V4;

   // New Value Jump.
   // JMPEQ_ri - with -1.
    case Hexagon::JMP_EQriPtneg_nv_V4:
      return Hexagon::JMP_EQriNotPtneg_nv_V4;
    case Hexagon::JMP_EQriNotPtneg_nv_V4:
      return Hexagon::JMP_EQriPtneg_nv_V4;

    case Hexagon::JMP_EQriPntneg_nv_V4:
      return Hexagon::JMP_EQriNotPntneg_nv_V4;
    case Hexagon::JMP_EQriNotPntneg_nv_V4:
      return Hexagon::JMP_EQriPntneg_nv_V4;

   // JMPEQ_ri.
     case Hexagon::JMP_EQriPt_nv_V4:
      return Hexagon::JMP_EQriNotPt_nv_V4;
    case Hexagon::JMP_EQriNotPt_nv_V4:
      return Hexagon::JMP_EQriPt_nv_V4;

     case Hexagon::JMP_EQriPnt_nv_V4:
      return Hexagon::JMP_EQriNotPnt_nv_V4;
    case Hexagon::JMP_EQriNotPnt_nv_V4:
      return Hexagon::JMP_EQriPnt_nv_V4;

   // JMPEQ_rr.
     case Hexagon::JMP_EQrrPt_nv_V4:
      return Hexagon::JMP_EQrrNotPt_nv_V4;
    case Hexagon::JMP_EQrrNotPt_nv_V4:
      return Hexagon::JMP_EQrrPt_nv_V4;

     case Hexagon::JMP_EQrrPnt_nv_V4:
      return Hexagon::JMP_EQrrNotPnt_nv_V4;
    case Hexagon::JMP_EQrrNotPnt_nv_V4:
      return Hexagon::JMP_EQrrPnt_nv_V4;

   // JMPGT_ri - with -1.
    case Hexagon::JMP_GTriPtneg_nv_V4:
      return Hexagon::JMP_GTriNotPtneg_nv_V4;
    case Hexagon::JMP_GTriNotPtneg_nv_V4:
      return Hexagon::JMP_GTriPtneg_nv_V4;

    case Hexagon::JMP_GTriPntneg_nv_V4:
      return Hexagon::JMP_GTriNotPntneg_nv_V4;
    case Hexagon::JMP_GTriNotPntneg_nv_V4:
      return Hexagon::JMP_GTriPntneg_nv_V4;

   // JMPGT_ri.
     case Hexagon::JMP_GTriPt_nv_V4:
      return Hexagon::JMP_GTriNotPt_nv_V4;
    case Hexagon::JMP_GTriNotPt_nv_V4:
      return Hexagon::JMP_GTriPt_nv_V4;

     case Hexagon::JMP_GTriPnt_nv_V4:
      return Hexagon::JMP_GTriNotPnt_nv_V4;
    case Hexagon::JMP_GTriNotPnt_nv_V4:
      return Hexagon::JMP_GTriPnt_nv_V4;

   // JMPGT_rr.
     case Hexagon::JMP_GTrrPt_nv_V4:
      return Hexagon::JMP_GTrrNotPt_nv_V4;
    case Hexagon::JMP_GTrrNotPt_nv_V4:
      return Hexagon::JMP_GTrrPt_nv_V4;

     case Hexagon::JMP_GTrrPnt_nv_V4:
      return Hexagon::JMP_GTrrNotPnt_nv_V4;
    case Hexagon::JMP_GTrrNotPnt_nv_V4:
      return Hexagon::JMP_GTrrPnt_nv_V4;

   // JMPGT_rrdn.
     case Hexagon::JMP_GTrrdnPt_nv_V4:
      return Hexagon::JMP_GTrrdnNotPt_nv_V4;
    case Hexagon::JMP_GTrrdnNotPt_nv_V4:
      return Hexagon::JMP_GTrrdnPt_nv_V4;

     case Hexagon::JMP_GTrrdnPnt_nv_V4:
      return Hexagon::JMP_GTrrdnNotPnt_nv_V4;
    case Hexagon::JMP_GTrrdnNotPnt_nv_V4:
      return Hexagon::JMP_GTrrdnPnt_nv_V4;

   // JMPGTU_ri.
     case Hexagon::JMP_GTUriPt_nv_V4:
      return Hexagon::JMP_GTUriNotPt_nv_V4;
    case Hexagon::JMP_GTUriNotPt_nv_V4:
      return Hexagon::JMP_GTUriPt_nv_V4;

     case Hexagon::JMP_GTUriPnt_nv_V4:
      return Hexagon::JMP_GTUriNotPnt_nv_V4;
    case Hexagon::JMP_GTUriNotPnt_nv_V4:
      return Hexagon::JMP_GTUriPnt_nv_V4;

   // JMPGTU_rr.
     case Hexagon::JMP_GTUrrPt_nv_V4:
      return Hexagon::JMP_GTUrrNotPt_nv_V4;
    case Hexagon::JMP_GTUrrNotPt_nv_V4:
      return Hexagon::JMP_GTUrrPt_nv_V4;

     case Hexagon::JMP_GTUrrPnt_nv_V4:
      return Hexagon::JMP_GTUrrNotPnt_nv_V4;
    case Hexagon::JMP_GTUrrNotPnt_nv_V4:
      return Hexagon::JMP_GTUrrPnt_nv_V4;

   // JMPGTU_rrdn.
     case Hexagon::JMP_GTUrrdnPt_nv_V4:
      return Hexagon::JMP_GTUrrdnNotPt_nv_V4;
    case Hexagon::JMP_GTUrrdnNotPt_nv_V4:
      return Hexagon::JMP_GTUrrdnPt_nv_V4;

     case Hexagon::JMP_GTUrrdnPnt_nv_V4:
      return Hexagon::JMP_GTUrrdnNotPnt_nv_V4;
    case Hexagon::JMP_GTUrrdnNotPnt_nv_V4:
      return Hexagon::JMP_GTUrrdnPnt_nv_V4;
  }
}


int HexagonInstrInfo::
getMatchingCondBranchOpcode(int Opc, bool invertPredicate) const {
  enum Hexagon::PredSense inPredSense;
  inPredSense = invertPredicate ? Hexagon::PredSense_false :
                                  Hexagon::PredSense_true;
  int CondOpcode = Hexagon::getPredOpcode(Opc, inPredSense);
  if (CondOpcode >= 0) // Valid Conditional opcode/instruction
    return CondOpcode;

  // This switch case will be removed once all the instructions have been
  // modified to use relation maps.
  switch(Opc) {
  case Hexagon::TFR:
    return !invertPredicate ? Hexagon::TFR_cPt :
                              Hexagon::TFR_cNotPt;
  case Hexagon::TFRI_f:
    return !invertPredicate ? Hexagon::TFRI_cPt_f :
                              Hexagon::TFRI_cNotPt_f;
  case Hexagon::TFRI:
    return !invertPredicate ? Hexagon::TFRI_cPt :
                              Hexagon::TFRI_cNotPt;
  case Hexagon::JMP:
    return !invertPredicate ? Hexagon::JMP_c :
                              Hexagon::JMP_cNot;
  case Hexagon::JMP_EQrrPt_nv_V4:
    return !invertPredicate ? Hexagon::JMP_EQrrPt_nv_V4 :
                              Hexagon::JMP_EQrrNotPt_nv_V4;
  case Hexagon::JMP_EQriPt_nv_V4:
    return !invertPredicate ? Hexagon::JMP_EQriPt_nv_V4 :
                              Hexagon::JMP_EQriNotPt_nv_V4;
  case Hexagon::COMBINE_rr:
    return !invertPredicate ? Hexagon::COMBINE_rr_cPt :
                              Hexagon::COMBINE_rr_cNotPt;
  case Hexagon::ASLH:
    return !invertPredicate ? Hexagon::ASLH_cPt_V4 :
                              Hexagon::ASLH_cNotPt_V4;
  case Hexagon::ASRH:
    return !invertPredicate ? Hexagon::ASRH_cPt_V4 :
                              Hexagon::ASRH_cNotPt_V4;
  case Hexagon::SXTB:
    return !invertPredicate ? Hexagon::SXTB_cPt_V4 :
                              Hexagon::SXTB_cNotPt_V4;
  case Hexagon::SXTH:
    return !invertPredicate ? Hexagon::SXTH_cPt_V4 :
                              Hexagon::SXTH_cNotPt_V4;
  case Hexagon::ZXTB:
    return !invertPredicate ? Hexagon::ZXTB_cPt_V4 :
                              Hexagon::ZXTB_cNotPt_V4;
  case Hexagon::ZXTH:
    return !invertPredicate ? Hexagon::ZXTH_cPt_V4 :
                              Hexagon::ZXTH_cNotPt_V4;

  case Hexagon::JMPR:
    return !invertPredicate ? Hexagon::JMPR_cPt :
                              Hexagon::JMPR_cNotPt;

  // V4 indexed+scaled load.
  case Hexagon::LDrid_indexed_V4:
    return !invertPredicate ? Hexagon::LDrid_indexed_cPt_V4 :
                              Hexagon::LDrid_indexed_cNotPt_V4;
  case Hexagon::LDrid_indexed_shl_V4:
    return !invertPredicate ? Hexagon::LDrid_indexed_shl_cPt_V4 :
                              Hexagon::LDrid_indexed_shl_cNotPt_V4;
  case Hexagon::LDrib_indexed_V4:
    return !invertPredicate ? Hexagon::LDrib_indexed_cPt_V4 :
                              Hexagon::LDrib_indexed_cNotPt_V4;
  case Hexagon::LDriub_indexed_V4:
    return !invertPredicate ? Hexagon::LDriub_indexed_cPt_V4 :
                              Hexagon::LDriub_indexed_cNotPt_V4;
  case Hexagon::LDriub_ae_indexed_V4:
    return !invertPredicate ? Hexagon::LDriub_indexed_cPt_V4 :
                              Hexagon::LDriub_indexed_cNotPt_V4;
  case Hexagon::LDrib_indexed_shl_V4:
    return !invertPredicate ? Hexagon::LDrib_indexed_shl_cPt_V4 :
                              Hexagon::LDrib_indexed_shl_cNotPt_V4;
  case Hexagon::LDriub_indexed_shl_V4:
    return !invertPredicate ? Hexagon::LDriub_indexed_shl_cPt_V4 :
                              Hexagon::LDriub_indexed_shl_cNotPt_V4;
  case Hexagon::LDriub_ae_indexed_shl_V4:
    return !invertPredicate ? Hexagon::LDriub_indexed_shl_cPt_V4 :
                              Hexagon::LDriub_indexed_shl_cNotPt_V4;
  case Hexagon::LDrih_indexed_V4:
    return !invertPredicate ? Hexagon::LDrih_indexed_cPt_V4 :
                              Hexagon::LDrih_indexed_cNotPt_V4;
  case Hexagon::LDriuh_indexed_V4:
    return !invertPredicate ? Hexagon::LDriuh_indexed_cPt_V4 :
                              Hexagon::LDriuh_indexed_cNotPt_V4;
  case Hexagon::LDriuh_ae_indexed_V4:
    return !invertPredicate ? Hexagon::LDriuh_indexed_cPt_V4 :
                              Hexagon::LDriuh_indexed_cNotPt_V4;
  case Hexagon::LDrih_indexed_shl_V4:
    return !invertPredicate ? Hexagon::LDrih_indexed_shl_cPt_V4 :
                              Hexagon::LDrih_indexed_shl_cNotPt_V4;
  case Hexagon::LDriuh_indexed_shl_V4:
    return !invertPredicate ? Hexagon::LDriuh_indexed_shl_cPt_V4 :
                              Hexagon::LDriuh_indexed_shl_cNotPt_V4;
  case Hexagon::LDriuh_ae_indexed_shl_V4:
    return !invertPredicate ? Hexagon::LDriuh_indexed_shl_cPt_V4 :
                              Hexagon::LDriuh_indexed_shl_cNotPt_V4;
  case Hexagon::LDriw_indexed_V4:
    return !invertPredicate ? Hexagon::LDriw_indexed_cPt_V4 :
                              Hexagon::LDriw_indexed_cNotPt_V4;
  case Hexagon::LDriw_indexed_shl_V4:
    return !invertPredicate ? Hexagon::LDriw_indexed_shl_cPt_V4 :
                              Hexagon::LDriw_indexed_shl_cNotPt_V4;

  // V4 Load from global address
  case Hexagon::LDrid_GP_V4:
    return !invertPredicate ? Hexagon::LDrid_GP_cPt_V4 :
                              Hexagon::LDrid_GP_cNotPt_V4;
  case Hexagon::LDrib_GP_V4:
    return !invertPredicate ? Hexagon::LDrib_GP_cPt_V4 :
                              Hexagon::LDrib_GP_cNotPt_V4;
  case Hexagon::LDriub_GP_V4:
    return !invertPredicate ? Hexagon::LDriub_GP_cPt_V4 :
                              Hexagon::LDriub_GP_cNotPt_V4;
  case Hexagon::LDrih_GP_V4:
    return !invertPredicate ? Hexagon::LDrih_GP_cPt_V4 :
                              Hexagon::LDrih_GP_cNotPt_V4;
  case Hexagon::LDriuh_GP_V4:
    return !invertPredicate ? Hexagon::LDriuh_GP_cPt_V4 :
                              Hexagon::LDriuh_GP_cNotPt_V4;
  case Hexagon::LDriw_GP_V4:
    return !invertPredicate ? Hexagon::LDriw_GP_cPt_V4 :
                              Hexagon::LDriw_GP_cNotPt_V4;

  case Hexagon::LDd_GP_V4:
    return !invertPredicate ? Hexagon::LDd_GP_cPt_V4 :
                              Hexagon::LDd_GP_cNotPt_V4;
  case Hexagon::LDb_GP_V4:
    return !invertPredicate ? Hexagon::LDb_GP_cPt_V4 :
                              Hexagon::LDb_GP_cNotPt_V4;
  case Hexagon::LDub_GP_V4:
    return !invertPredicate ? Hexagon::LDub_GP_cPt_V4 :
                              Hexagon::LDub_GP_cNotPt_V4;
  case Hexagon::LDh_GP_V4:
    return !invertPredicate ? Hexagon::LDh_GP_cPt_V4 :
                              Hexagon::LDh_GP_cNotPt_V4;
  case Hexagon::LDuh_GP_V4:
    return !invertPredicate ? Hexagon::LDuh_GP_cPt_V4 :
                              Hexagon::LDuh_GP_cNotPt_V4;
  case Hexagon::LDw_GP_V4:
    return !invertPredicate ? Hexagon::LDw_GP_cPt_V4 :
                              Hexagon::LDw_GP_cNotPt_V4;

    // Byte.
  case Hexagon::POST_STbri:
    return !invertPredicate ? Hexagon::POST_STbri_cPt :
                              Hexagon::POST_STbri_cNotPt;
  case Hexagon::STrib:
    return !invertPredicate ? Hexagon::STrib_cPt :
                              Hexagon::STrib_cNotPt;
  case Hexagon::STrib_indexed:
    return !invertPredicate ? Hexagon::STrib_indexed_cPt :
                              Hexagon::STrib_indexed_cNotPt;
  case Hexagon::STrib_imm_V4:
    return !invertPredicate ? Hexagon::STrib_imm_cPt_V4 :
                              Hexagon::STrib_imm_cNotPt_V4;
  case Hexagon::STrib_indexed_shl_V4:
    return !invertPredicate ? Hexagon::STrib_indexed_shl_cPt_V4 :
                              Hexagon::STrib_indexed_shl_cNotPt_V4;
  // Halfword.
  case Hexagon::POST_SThri:
    return !invertPredicate ? Hexagon::POST_SThri_cPt :
                              Hexagon::POST_SThri_cNotPt;
  case Hexagon::STrih:
    return !invertPredicate ? Hexagon::STrih_cPt :
                              Hexagon::STrih_cNotPt;
  case Hexagon::STrih_indexed:
    return !invertPredicate ? Hexagon::STrih_indexed_cPt :
                              Hexagon::STrih_indexed_cNotPt;
  case Hexagon::STrih_imm_V4:
    return !invertPredicate ? Hexagon::STrih_imm_cPt_V4 :
                              Hexagon::STrih_imm_cNotPt_V4;
  case Hexagon::STrih_indexed_shl_V4:
    return !invertPredicate ? Hexagon::STrih_indexed_shl_cPt_V4 :
                              Hexagon::STrih_indexed_shl_cNotPt_V4;
  // Word.
  case Hexagon::POST_STwri:
    return !invertPredicate ? Hexagon::POST_STwri_cPt :
                              Hexagon::POST_STwri_cNotPt;
  case Hexagon::STriw:
    return !invertPredicate ? Hexagon::STriw_cPt :
                              Hexagon::STriw_cNotPt;
  case Hexagon::STriw_indexed:
    return !invertPredicate ? Hexagon::STriw_indexed_cPt :
                              Hexagon::STriw_indexed_cNotPt;
  case Hexagon::STriw_indexed_shl_V4:
    return !invertPredicate ? Hexagon::STriw_indexed_shl_cPt_V4 :
                              Hexagon::STriw_indexed_shl_cNotPt_V4;
  case Hexagon::STriw_imm_V4:
    return !invertPredicate ? Hexagon::STriw_imm_cPt_V4 :
                              Hexagon::STriw_imm_cNotPt_V4;
  // Double word.
  case Hexagon::POST_STdri:
    return !invertPredicate ? Hexagon::POST_STdri_cPt :
                              Hexagon::POST_STdri_cNotPt;
  case Hexagon::STrid:
    return !invertPredicate ? Hexagon::STrid_cPt :
                              Hexagon::STrid_cNotPt;
  case Hexagon::STrid_indexed:
    return !invertPredicate ? Hexagon::STrid_indexed_cPt :
                              Hexagon::STrid_indexed_cNotPt;
  case Hexagon::STrid_indexed_shl_V4:
    return !invertPredicate ? Hexagon::STrid_indexed_shl_cPt_V4 :
                              Hexagon::STrid_indexed_shl_cNotPt_V4;

  // V4 Store to global address
  case Hexagon::STrid_GP_V4:
    return !invertPredicate ? Hexagon::STrid_GP_cPt_V4 :
                              Hexagon::STrid_GP_cNotPt_V4;
  case Hexagon::STrib_GP_V4:
    return !invertPredicate ? Hexagon::STrib_GP_cPt_V4 :
                              Hexagon::STrib_GP_cNotPt_V4;
  case Hexagon::STrih_GP_V4:
    return !invertPredicate ? Hexagon::STrih_GP_cPt_V4 :
                              Hexagon::STrih_GP_cNotPt_V4;
  case Hexagon::STriw_GP_V4:
    return !invertPredicate ? Hexagon::STriw_GP_cPt_V4 :
                              Hexagon::STriw_GP_cNotPt_V4;

  case Hexagon::STd_GP_V4:
    return !invertPredicate ? Hexagon::STd_GP_cPt_V4 :
                              Hexagon::STd_GP_cNotPt_V4;
  case Hexagon::STb_GP_V4:
    return !invertPredicate ? Hexagon::STb_GP_cPt_V4 :
                              Hexagon::STb_GP_cNotPt_V4;
  case Hexagon::STh_GP_V4:
    return !invertPredicate ? Hexagon::STh_GP_cPt_V4 :
                              Hexagon::STh_GP_cNotPt_V4;
  case Hexagon::STw_GP_V4:
    return !invertPredicate ? Hexagon::STw_GP_cPt_V4 :
                              Hexagon::STw_GP_cNotPt_V4;

  // Load.
  case Hexagon::LDrid:
    return !invertPredicate ? Hexagon::LDrid_cPt :
                              Hexagon::LDrid_cNotPt;
  case Hexagon::LDriw:
    return !invertPredicate ? Hexagon::LDriw_cPt :
                              Hexagon::LDriw_cNotPt;
  case Hexagon::LDrih:
    return !invertPredicate ? Hexagon::LDrih_cPt :
                              Hexagon::LDrih_cNotPt;
  case Hexagon::LDriuh:
    return !invertPredicate ? Hexagon::LDriuh_cPt :
                              Hexagon::LDriuh_cNotPt;
  case Hexagon::LDrib:
    return !invertPredicate ? Hexagon::LDrib_cPt :
                              Hexagon::LDrib_cNotPt;
  case Hexagon::LDriub:
    return !invertPredicate ? Hexagon::LDriub_cPt :
                              Hexagon::LDriub_cNotPt;
 // Load Indexed.
  case Hexagon::LDrid_indexed:
    return !invertPredicate ? Hexagon::LDrid_indexed_cPt :
                              Hexagon::LDrid_indexed_cNotPt;
  case Hexagon::LDriw_indexed:
    return !invertPredicate ? Hexagon::LDriw_indexed_cPt :
                              Hexagon::LDriw_indexed_cNotPt;
  case Hexagon::LDrih_indexed:
    return !invertPredicate ? Hexagon::LDrih_indexed_cPt :
                              Hexagon::LDrih_indexed_cNotPt;
  case Hexagon::LDriuh_indexed:
    return !invertPredicate ? Hexagon::LDriuh_indexed_cPt :
                              Hexagon::LDriuh_indexed_cNotPt;
  case Hexagon::LDrib_indexed:
    return !invertPredicate ? Hexagon::LDrib_indexed_cPt :
                              Hexagon::LDrib_indexed_cNotPt;
  case Hexagon::LDriub_indexed:
    return !invertPredicate ? Hexagon::LDriub_indexed_cPt :
                              Hexagon::LDriub_indexed_cNotPt;
  // Post Increment Load.
  case Hexagon::POST_LDrid:
    return !invertPredicate ? Hexagon::POST_LDrid_cPt :
                              Hexagon::POST_LDrid_cNotPt;
  case Hexagon::POST_LDriw:
    return !invertPredicate ? Hexagon::POST_LDriw_cPt :
                              Hexagon::POST_LDriw_cNotPt;
  case Hexagon::POST_LDrih:
    return !invertPredicate ? Hexagon::POST_LDrih_cPt :
                              Hexagon::POST_LDrih_cNotPt;
  case Hexagon::POST_LDriuh:
    return !invertPredicate ? Hexagon::POST_LDriuh_cPt :
                              Hexagon::POST_LDriuh_cNotPt;
  case Hexagon::POST_LDrib:
    return !invertPredicate ? Hexagon::POST_LDrib_cPt :
                              Hexagon::POST_LDrib_cNotPt;
  case Hexagon::POST_LDriub:
    return !invertPredicate ? Hexagon::POST_LDriub_cPt :
                              Hexagon::POST_LDriub_cNotPt;
  // DEALLOC_RETURN.
  case Hexagon::DEALLOC_RET_V4:
    return !invertPredicate ? Hexagon::DEALLOC_RET_cPt_V4 :
                              Hexagon::DEALLOC_RET_cNotPt_V4;
  }
  llvm_unreachable("Unexpected predicable instruction");
}


bool HexagonInstrInfo::
PredicateInstruction(MachineInstr *MI,
                     const SmallVectorImpl<MachineOperand> &Cond) const {
  int Opc = MI->getOpcode();
  assert (isPredicable(MI) && "Expected predicable instruction");
  bool invertJump = (!Cond.empty() && Cond[0].isImm() &&
                     (Cond[0].getImm() == 0));
  MI->setDesc(get(getMatchingCondBranchOpcode(Opc, invertJump)));
  //
  // This assumes that the predicate is always the first operand
  // in the set of inputs.
  //
  MI->addOperand(MI->getOperand(MI->getNumOperands()-1));
  int oper;
  for (oper = MI->getNumOperands() - 3; oper >= 0; --oper) {
    MachineOperand MO = MI->getOperand(oper);
    if ((MO.isReg() && !MO.isUse() && !MO.isImplicit())) {
      break;
    }

    if (MO.isReg()) {
      MI->getOperand(oper+1).ChangeToRegister(MO.getReg(), MO.isDef(),
                                              MO.isImplicit(), MO.isKill(),
                                              MO.isDead(), MO.isUndef(),
                                              MO.isDebug());
    } else if (MO.isImm()) {
      MI->getOperand(oper+1).ChangeToImmediate(MO.getImm());
    } else {
      llvm_unreachable("Unexpected operand type");
    }
  }

  int regPos = invertJump ? 1 : 0;
  MachineOperand PredMO = Cond[regPos];
  MI->getOperand(oper+1).ChangeToRegister(PredMO.getReg(), PredMO.isDef(),
                                          PredMO.isImplicit(), PredMO.isKill(),
                                          PredMO.isDead(), PredMO.isUndef(),
                                          PredMO.isDebug());

  return true;
}


bool
HexagonInstrInfo::
isProfitableToIfCvt(MachineBasicBlock &MBB,
                    unsigned NumCycles,
                    unsigned ExtraPredCycles,
                    const BranchProbability &Probability) const {
  return true;
}


bool
HexagonInstrInfo::
isProfitableToIfCvt(MachineBasicBlock &TMBB,
                    unsigned NumTCycles,
                    unsigned ExtraTCycles,
                    MachineBasicBlock &FMBB,
                    unsigned NumFCycles,
                    unsigned ExtraFCycles,
                    const BranchProbability &Probability) const {
  return true;
}


bool HexagonInstrInfo::isPredicated(const MachineInstr *MI) const {
  const uint64_t F = MI->getDesc().TSFlags;

  return ((F >> HexagonII::PredicatedPos) & HexagonII::PredicatedMask);
}

bool
HexagonInstrInfo::DefinesPredicate(MachineInstr *MI,
                                   std::vector<MachineOperand> &Pred) const {
  for (unsigned oper = 0; oper < MI->getNumOperands(); ++oper) {
    MachineOperand MO = MI->getOperand(oper);
    if (MO.isReg() && MO.isDef()) {
      const TargetRegisterClass* RC = RI.getMinimalPhysRegClass(MO.getReg());
      if (RC == &Hexagon::PredRegsRegClass) {
        Pred.push_back(MO);
        return true;
      }
    }
  }
  return false;
}


bool
HexagonInstrInfo::
SubsumesPredicate(const SmallVectorImpl<MachineOperand> &Pred1,
                  const SmallVectorImpl<MachineOperand> &Pred2) const {
  // TODO: Fix this
  return false;
}


//
// We indicate that we want to reverse the branch by
// inserting a 0 at the beginning of the Cond vector.
//
bool HexagonInstrInfo::
ReverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const {
  if (!Cond.empty() && Cond[0].isImm() && Cond[0].getImm() == 0) {
    Cond.erase(Cond.begin());
  } else {
    Cond.insert(Cond.begin(), MachineOperand::CreateImm(0));
  }
  return false;
}


bool HexagonInstrInfo::
isProfitableToDupForIfCvt(MachineBasicBlock &MBB,unsigned NumInstrs,
                          const BranchProbability &Probability) const {
  return (NumInstrs <= 4);
}

bool HexagonInstrInfo::isDeallocRet(const MachineInstr *MI) const {
  switch (MI->getOpcode()) {
  default: return false;
  case Hexagon::DEALLOC_RET_V4 :
  case Hexagon::DEALLOC_RET_cPt_V4 :
  case Hexagon::DEALLOC_RET_cNotPt_V4 :
  case Hexagon::DEALLOC_RET_cdnPnt_V4 :
  case Hexagon::DEALLOC_RET_cNotdnPnt_V4 :
  case Hexagon::DEALLOC_RET_cdnPt_V4 :
  case Hexagon::DEALLOC_RET_cNotdnPt_V4 :
   return true;
  }
}


bool HexagonInstrInfo::
isValidOffset(const int Opcode, const int Offset) const {
  // This function is to check whether the "Offset" is in the correct range of
  // the given "Opcode". If "Offset" is not in the correct range, "ADD_ri" is
  // inserted to calculate the final address. Due to this reason, the function
  // assumes that the "Offset" has correct alignment.

  switch(Opcode) {

  case Hexagon::LDriw:
  case Hexagon::LDriw_f:
  case Hexagon::STriw:
  case Hexagon::STriw_f:
    assert((Offset % 4 == 0) && "Offset has incorrect alignment");
    return (Offset >= Hexagon_MEMW_OFFSET_MIN) &&
      (Offset <= Hexagon_MEMW_OFFSET_MAX);

  case Hexagon::LDrid:
  case Hexagon::LDrid_f:
  case Hexagon::STrid:
  case Hexagon::STrid_f:
    assert((Offset % 8 == 0) && "Offset has incorrect alignment");
    return (Offset >= Hexagon_MEMD_OFFSET_MIN) &&
      (Offset <= Hexagon_MEMD_OFFSET_MAX);

  case Hexagon::LDrih:
  case Hexagon::LDriuh:
  case Hexagon::STrih:
    assert((Offset % 2 == 0) && "Offset has incorrect alignment");
    return (Offset >= Hexagon_MEMH_OFFSET_MIN) &&
      (Offset <= Hexagon_MEMH_OFFSET_MAX);

  case Hexagon::LDrib:
  case Hexagon::STrib:
  case Hexagon::LDriub:
    return (Offset >= Hexagon_MEMB_OFFSET_MIN) &&
      (Offset <= Hexagon_MEMB_OFFSET_MAX);

  case Hexagon::ADD_ri:
  case Hexagon::TFR_FI:
    return (Offset >= Hexagon_ADDI_OFFSET_MIN) &&
      (Offset <= Hexagon_ADDI_OFFSET_MAX);

  case Hexagon::MEMw_ADDSUBi_indexed_MEM_V4 :
  case Hexagon::MEMw_ADDi_indexed_MEM_V4 :
  case Hexagon::MEMw_SUBi_indexed_MEM_V4 :
  case Hexagon::MEMw_ADDr_indexed_MEM_V4 :
  case Hexagon::MEMw_SUBr_indexed_MEM_V4 :
  case Hexagon::MEMw_ANDr_indexed_MEM_V4 :
  case Hexagon::MEMw_ORr_indexed_MEM_V4 :
  case Hexagon::MEMw_ADDSUBi_MEM_V4 :
  case Hexagon::MEMw_ADDi_MEM_V4 :
  case Hexagon::MEMw_SUBi_MEM_V4 :
  case Hexagon::MEMw_ADDr_MEM_V4 :
  case Hexagon::MEMw_SUBr_MEM_V4 :
  case Hexagon::MEMw_ANDr_MEM_V4 :
  case Hexagon::MEMw_ORr_MEM_V4 :
    assert ((Offset % 4) == 0 && "MEMOPw offset is not aligned correctly." );
    return (0 <= Offset && Offset <= 255);

  case Hexagon::MEMh_ADDSUBi_indexed_MEM_V4 :
  case Hexagon::MEMh_ADDi_indexed_MEM_V4 :
  case Hexagon::MEMh_SUBi_indexed_MEM_V4 :
  case Hexagon::MEMh_ADDr_indexed_MEM_V4 :
  case Hexagon::MEMh_SUBr_indexed_MEM_V4 :
  case Hexagon::MEMh_ANDr_indexed_MEM_V4 :
  case Hexagon::MEMh_ORr_indexed_MEM_V4 :
  case Hexagon::MEMh_ADDSUBi_MEM_V4 :
  case Hexagon::MEMh_ADDi_MEM_V4 :
  case Hexagon::MEMh_SUBi_MEM_V4 :
  case Hexagon::MEMh_ADDr_MEM_V4 :
  case Hexagon::MEMh_SUBr_MEM_V4 :
  case Hexagon::MEMh_ANDr_MEM_V4 :
  case Hexagon::MEMh_ORr_MEM_V4 :
    assert ((Offset % 2) == 0 && "MEMOPh offset is not aligned correctly." );
    return (0 <= Offset && Offset <= 127);

  case Hexagon::MEMb_ADDSUBi_indexed_MEM_V4 :
  case Hexagon::MEMb_ADDi_indexed_MEM_V4 :
  case Hexagon::MEMb_SUBi_indexed_MEM_V4 :
  case Hexagon::MEMb_ADDr_indexed_MEM_V4 :
  case Hexagon::MEMb_SUBr_indexed_MEM_V4 :
  case Hexagon::MEMb_ANDr_indexed_MEM_V4 :
  case Hexagon::MEMb_ORr_indexed_MEM_V4 :
  case Hexagon::MEMb_ADDSUBi_MEM_V4 :
  case Hexagon::MEMb_ADDi_MEM_V4 :
  case Hexagon::MEMb_SUBi_MEM_V4 :
  case Hexagon::MEMb_ADDr_MEM_V4 :
  case Hexagon::MEMb_SUBr_MEM_V4 :
  case Hexagon::MEMb_ANDr_MEM_V4 :
  case Hexagon::MEMb_ORr_MEM_V4 :
    return (0 <= Offset && Offset <= 63);

  // LDri_pred and STriw_pred are pseudo operations, so it has to take offset of
  // any size. Later pass knows how to handle it.
  case Hexagon::STriw_pred:
  case Hexagon::LDriw_pred:
    return true;

  // INLINEASM is very special.
  case Hexagon::INLINEASM:
    return true;
  }

  llvm_unreachable("No offset range is defined for this opcode. "
                   "Please define it in the above switch statement!");
}


//
// Check if the Offset is a valid auto-inc imm by Load/Store Type.
//
bool HexagonInstrInfo::
isValidAutoIncImm(const EVT VT, const int Offset) const {

  if (VT == MVT::i64) {
      return (Offset >= Hexagon_MEMD_AUTOINC_MIN &&
              Offset <= Hexagon_MEMD_AUTOINC_MAX &&
              (Offset & 0x7) == 0);
  }
  if (VT == MVT::i32) {
      return (Offset >= Hexagon_MEMW_AUTOINC_MIN &&
              Offset <= Hexagon_MEMW_AUTOINC_MAX &&
              (Offset & 0x3) == 0);
  }
  if (VT == MVT::i16) {
      return (Offset >= Hexagon_MEMH_AUTOINC_MIN &&
              Offset <= Hexagon_MEMH_AUTOINC_MAX &&
              (Offset & 0x1) == 0);
  }
  if (VT == MVT::i8) {
      return (Offset >= Hexagon_MEMB_AUTOINC_MIN &&
              Offset <= Hexagon_MEMB_AUTOINC_MAX);
  }
  llvm_unreachable("Not an auto-inc opc!");
}


bool HexagonInstrInfo::
isMemOp(const MachineInstr *MI) const {
  switch (MI->getOpcode())
  {
    default: return false;
    case Hexagon::MEMw_ADDSUBi_indexed_MEM_V4 :
    case Hexagon::MEMw_ADDi_indexed_MEM_V4 :
    case Hexagon::MEMw_SUBi_indexed_MEM_V4 :
    case Hexagon::MEMw_ADDr_indexed_MEM_V4 :
    case Hexagon::MEMw_SUBr_indexed_MEM_V4 :
    case Hexagon::MEMw_ANDr_indexed_MEM_V4 :
    case Hexagon::MEMw_ORr_indexed_MEM_V4 :
    case Hexagon::MEMw_ADDSUBi_MEM_V4 :
    case Hexagon::MEMw_ADDi_MEM_V4 :
    case Hexagon::MEMw_SUBi_MEM_V4 :
    case Hexagon::MEMw_ADDr_MEM_V4 :
    case Hexagon::MEMw_SUBr_MEM_V4 :
    case Hexagon::MEMw_ANDr_MEM_V4 :
    case Hexagon::MEMw_ORr_MEM_V4 :
    case Hexagon::MEMh_ADDSUBi_indexed_MEM_V4 :
    case Hexagon::MEMh_ADDi_indexed_MEM_V4 :
    case Hexagon::MEMh_SUBi_indexed_MEM_V4 :
    case Hexagon::MEMh_ADDr_indexed_MEM_V4 :
    case Hexagon::MEMh_SUBr_indexed_MEM_V4 :
    case Hexagon::MEMh_ANDr_indexed_MEM_V4 :
    case Hexagon::MEMh_ORr_indexed_MEM_V4 :
    case Hexagon::MEMh_ADDSUBi_MEM_V4 :
    case Hexagon::MEMh_ADDi_MEM_V4 :
    case Hexagon::MEMh_SUBi_MEM_V4 :
    case Hexagon::MEMh_ADDr_MEM_V4 :
    case Hexagon::MEMh_SUBr_MEM_V4 :
    case Hexagon::MEMh_ANDr_MEM_V4 :
    case Hexagon::MEMh_ORr_MEM_V4 :
    case Hexagon::MEMb_ADDSUBi_indexed_MEM_V4 :
    case Hexagon::MEMb_ADDi_indexed_MEM_V4 :
    case Hexagon::MEMb_SUBi_indexed_MEM_V4 :
    case Hexagon::MEMb_ADDr_indexed_MEM_V4 :
    case Hexagon::MEMb_SUBr_indexed_MEM_V4 :
    case Hexagon::MEMb_ANDr_indexed_MEM_V4 :
    case Hexagon::MEMb_ORr_indexed_MEM_V4 :
    case Hexagon::MEMb_ADDSUBi_MEM_V4 :
    case Hexagon::MEMb_ADDi_MEM_V4 :
    case Hexagon::MEMb_SUBi_MEM_V4 :
    case Hexagon::MEMb_ADDr_MEM_V4 :
    case Hexagon::MEMb_SUBr_MEM_V4 :
    case Hexagon::MEMb_ANDr_MEM_V4 :
    case Hexagon::MEMb_ORr_MEM_V4 :
      return true;
  }
}


bool HexagonInstrInfo::
isSpillPredRegOp(const MachineInstr *MI) const {
  switch (MI->getOpcode()) {
    default: return false;
    case Hexagon::STriw_pred :
    case Hexagon::LDriw_pred :
      return true;
  }
}

bool HexagonInstrInfo::isNewValueJumpCandidate(const MachineInstr *MI) const {
  switch (MI->getOpcode()) {
    default: return false;
    case Hexagon::CMPEQrr:
    case Hexagon::CMPEQri:
    case Hexagon::CMPLTrr:
    case Hexagon::CMPGTrr:
    case Hexagon::CMPGTri:
    case Hexagon::CMPLTUrr:
    case Hexagon::CMPGTUrr:
    case Hexagon::CMPGTUri:
    case Hexagon::CMPGEri:
    case Hexagon::CMPGEUri:
      return true;
  }
}

bool HexagonInstrInfo::
isConditionalTransfer (const MachineInstr *MI) const {
  switch (MI->getOpcode()) {
    default: return false;
    case Hexagon::TFR_cPt:
    case Hexagon::TFR_cNotPt:
    case Hexagon::TFRI_cPt:
    case Hexagon::TFRI_cNotPt:
    case Hexagon::TFR_cdnPt:
    case Hexagon::TFR_cdnNotPt:
    case Hexagon::TFRI_cdnPt:
    case Hexagon::TFRI_cdnNotPt:
      return true;
  }
}

bool HexagonInstrInfo::isConditionalALU32 (const MachineInstr* MI) const {
  const HexagonRegisterInfo& QRI = getRegisterInfo();
  switch (MI->getOpcode())
  {
    default: return false;
    case Hexagon::ADD_ri_cPt:
    case Hexagon::ADD_ri_cNotPt:
    case Hexagon::ADD_rr_cPt:
    case Hexagon::ADD_rr_cNotPt:
    case Hexagon::XOR_rr_cPt:
    case Hexagon::XOR_rr_cNotPt:
    case Hexagon::AND_rr_cPt:
    case Hexagon::AND_rr_cNotPt:
    case Hexagon::OR_rr_cPt:
    case Hexagon::OR_rr_cNotPt:
    case Hexagon::SUB_rr_cPt:
    case Hexagon::SUB_rr_cNotPt:
    case Hexagon::COMBINE_rr_cPt:
    case Hexagon::COMBINE_rr_cNotPt:
      return true;
    case Hexagon::ASLH_cPt_V4:
    case Hexagon::ASLH_cNotPt_V4:
    case Hexagon::ASRH_cPt_V4:
    case Hexagon::ASRH_cNotPt_V4:
    case Hexagon::SXTB_cPt_V4:
    case Hexagon::SXTB_cNotPt_V4:
    case Hexagon::SXTH_cPt_V4:
    case Hexagon::SXTH_cNotPt_V4:
    case Hexagon::ZXTB_cPt_V4:
    case Hexagon::ZXTB_cNotPt_V4:
    case Hexagon::ZXTH_cPt_V4:
    case Hexagon::ZXTH_cNotPt_V4:
      return QRI.Subtarget.hasV4TOps();
  }
}

bool HexagonInstrInfo::
isConditionalLoad (const MachineInstr* MI) const {
  const HexagonRegisterInfo& QRI = getRegisterInfo();
  switch (MI->getOpcode())
  {
    default: return false;
    case Hexagon::LDrid_cPt :
    case Hexagon::LDrid_cNotPt :
    case Hexagon::LDrid_indexed_cPt :
    case Hexagon::LDrid_indexed_cNotPt :
    case Hexagon::LDriw_cPt :
    case Hexagon::LDriw_cNotPt :
    case Hexagon::LDriw_indexed_cPt :
    case Hexagon::LDriw_indexed_cNotPt :
    case Hexagon::LDrih_cPt :
    case Hexagon::LDrih_cNotPt :
    case Hexagon::LDrih_indexed_cPt :
    case Hexagon::LDrih_indexed_cNotPt :
    case Hexagon::LDrib_cPt :
    case Hexagon::LDrib_cNotPt :
    case Hexagon::LDrib_indexed_cPt :
    case Hexagon::LDrib_indexed_cNotPt :
    case Hexagon::LDriuh_cPt :
    case Hexagon::LDriuh_cNotPt :
    case Hexagon::LDriuh_indexed_cPt :
    case Hexagon::LDriuh_indexed_cNotPt :
    case Hexagon::LDriub_cPt :
    case Hexagon::LDriub_cNotPt :
    case Hexagon::LDriub_indexed_cPt :
    case Hexagon::LDriub_indexed_cNotPt :
      return true;
    case Hexagon::POST_LDrid_cPt :
    case Hexagon::POST_LDrid_cNotPt :
    case Hexagon::POST_LDriw_cPt :
    case Hexagon::POST_LDriw_cNotPt :
    case Hexagon::POST_LDrih_cPt :
    case Hexagon::POST_LDrih_cNotPt :
    case Hexagon::POST_LDrib_cPt :
    case Hexagon::POST_LDrib_cNotPt :
    case Hexagon::POST_LDriuh_cPt :
    case Hexagon::POST_LDriuh_cNotPt :
    case Hexagon::POST_LDriub_cPt :
    case Hexagon::POST_LDriub_cNotPt :
      return QRI.Subtarget.hasV4TOps();
    case Hexagon::LDrid_indexed_cPt_V4 :
    case Hexagon::LDrid_indexed_cNotPt_V4 :
    case Hexagon::LDrid_indexed_shl_cPt_V4 :
    case Hexagon::LDrid_indexed_shl_cNotPt_V4 :
    case Hexagon::LDrib_indexed_cPt_V4 :
    case Hexagon::LDrib_indexed_cNotPt_V4 :
    case Hexagon::LDrib_indexed_shl_cPt_V4 :
    case Hexagon::LDrib_indexed_shl_cNotPt_V4 :
    case Hexagon::LDriub_indexed_cPt_V4 :
    case Hexagon::LDriub_indexed_cNotPt_V4 :
    case Hexagon::LDriub_indexed_shl_cPt_V4 :
    case Hexagon::LDriub_indexed_shl_cNotPt_V4 :
    case Hexagon::LDrih_indexed_cPt_V4 :
    case Hexagon::LDrih_indexed_cNotPt_V4 :
    case Hexagon::LDrih_indexed_shl_cPt_V4 :
    case Hexagon::LDrih_indexed_shl_cNotPt_V4 :
    case Hexagon::LDriuh_indexed_cPt_V4 :
    case Hexagon::LDriuh_indexed_cNotPt_V4 :
    case Hexagon::LDriuh_indexed_shl_cPt_V4 :
    case Hexagon::LDriuh_indexed_shl_cNotPt_V4 :
    case Hexagon::LDriw_indexed_cPt_V4 :
    case Hexagon::LDriw_indexed_cNotPt_V4 :
    case Hexagon::LDriw_indexed_shl_cPt_V4 :
    case Hexagon::LDriw_indexed_shl_cNotPt_V4 :
      return QRI.Subtarget.hasV4TOps();
  }
}

// Returns true if an instruction is a conditional store.
//
// Note: It doesn't include conditional new-value stores as they can't be
// converted to .new predicate.
//
//               p.new NV store [ if(p0.new)memw(R0+#0)=R2.new ]
//                ^           ^
//               /             \ (not OK. it will cause new-value store to be
//              /               X conditional on p0.new while R2 producer is
//             /                 \ on p0)
//            /                   \.
//     p.new store                 p.old NV store
// [if(p0.new)memw(R0+#0)=R2]    [if(p0)memw(R0+#0)=R2.new]
//            ^                  ^
//             \                /
//              \              /
//               \            /
//                 p.old store
//             [if (p0)memw(R0+#0)=R2]
//
// The above diagram shows the steps involoved in the conversion of a predicated
// store instruction to its .new predicated new-value form.
//
// The following set of instructions further explains the scenario where
// conditional new-value store becomes invalid when promoted to .new predicate
// form.
//
// { 1) if (p0) r0 = add(r1, r2)
//   2) p0 = cmp.eq(r3, #0) }
//
//   3) if (p0) memb(r1+#0) = r0  --> this instruction can't be grouped with
// the first two instructions because in instr 1, r0 is conditional on old value
// of p0 but its use in instr 3 is conditional on p0 modified by instr 2 which
// is not valid for new-value stores.
bool HexagonInstrInfo::
isConditionalStore (const MachineInstr* MI) const {
  const HexagonRegisterInfo& QRI = getRegisterInfo();
  switch (MI->getOpcode())
  {
    default: return false;
    case Hexagon::STrib_imm_cPt_V4 :
    case Hexagon::STrib_imm_cNotPt_V4 :
    case Hexagon::STrib_indexed_shl_cPt_V4 :
    case Hexagon::STrib_indexed_shl_cNotPt_V4 :
    case Hexagon::STrib_cPt :
    case Hexagon::STrib_cNotPt :
    case Hexagon::POST_STbri_cPt :
    case Hexagon::POST_STbri_cNotPt :
    case Hexagon::STrid_indexed_cPt :
    case Hexagon::STrid_indexed_cNotPt :
    case Hexagon::STrid_indexed_shl_cPt_V4 :
    case Hexagon::POST_STdri_cPt :
    case Hexagon::POST_STdri_cNotPt :
    case Hexagon::STrih_cPt :
    case Hexagon::STrih_cNotPt :
    case Hexagon::STrih_indexed_cPt :
    case Hexagon::STrih_indexed_cNotPt :
    case Hexagon::STrih_imm_cPt_V4 :
    case Hexagon::STrih_imm_cNotPt_V4 :
    case Hexagon::STrih_indexed_shl_cPt_V4 :
    case Hexagon::STrih_indexed_shl_cNotPt_V4 :
    case Hexagon::POST_SThri_cPt :
    case Hexagon::POST_SThri_cNotPt :
    case Hexagon::STriw_cPt :
    case Hexagon::STriw_cNotPt :
    case Hexagon::STriw_indexed_cPt :
    case Hexagon::STriw_indexed_cNotPt :
    case Hexagon::STriw_imm_cPt_V4 :
    case Hexagon::STriw_imm_cNotPt_V4 :
    case Hexagon::STriw_indexed_shl_cPt_V4 :
    case Hexagon::STriw_indexed_shl_cNotPt_V4 :
    case Hexagon::POST_STwri_cPt :
    case Hexagon::POST_STwri_cNotPt :
      return QRI.Subtarget.hasV4TOps();

    // V4 global address store before promoting to dot new.
    case Hexagon::STrid_GP_cPt_V4 :
    case Hexagon::STrid_GP_cNotPt_V4 :
    case Hexagon::STrib_GP_cPt_V4 :
    case Hexagon::STrib_GP_cNotPt_V4 :
    case Hexagon::STrih_GP_cPt_V4 :
    case Hexagon::STrih_GP_cNotPt_V4 :
    case Hexagon::STriw_GP_cPt_V4 :
    case Hexagon::STriw_GP_cNotPt_V4 :
    case Hexagon::STd_GP_cPt_V4 :
    case Hexagon::STd_GP_cNotPt_V4 :
    case Hexagon::STb_GP_cPt_V4 :
    case Hexagon::STb_GP_cNotPt_V4 :
    case Hexagon::STh_GP_cPt_V4 :
    case Hexagon::STh_GP_cNotPt_V4 :
    case Hexagon::STw_GP_cPt_V4 :
    case Hexagon::STw_GP_cNotPt_V4 :
      return QRI.Subtarget.hasV4TOps();

    // Predicated new value stores (i.e. if (p0) memw(..)=r0.new) are excluded
    // from the "Conditional Store" list. Because a predicated new value store
    // would NOT be promoted to a double dot new store. See diagram below:
    // This function returns yes for those stores that are predicated but not
    // yet promoted to predicate dot new instructions.
    //
    //                          +---------------------+
    //                    /-----| if (p0) memw(..)=r0 |---------\~
    //                   ||     +---------------------+         ||
    //          promote  ||       /\       /\                   ||  promote
    //                   ||      /||\     /||\                  ||
    //                  \||/    demote     ||                  \||/
    //                   \/       ||       ||                   \/
    //       +-------------------------+   ||   +-------------------------+
    //       | if (p0.new) memw(..)=r0 |   ||   | if (p0) memw(..)=r0.new |
    //       +-------------------------+   ||   +-------------------------+
    //                        ||           ||         ||
    //                        ||         demote      \||/
    //                      promote        ||         \/ NOT possible
    //                        ||           ||         /\~
    //                       \||/          ||        /||\~
    //                        \/           ||         ||
    //                      +-----------------------------+
    //                      | if (p0.new) memw(..)=r0.new |
    //                      +-----------------------------+
    //                           Double Dot New Store
    //
  }
}



DFAPacketizer *HexagonInstrInfo::
CreateTargetScheduleState(const TargetMachine *TM,
                           const ScheduleDAG *DAG) const {
  const InstrItineraryData *II = TM->getInstrItineraryData();
  return TM->getSubtarget<HexagonGenSubtargetInfo>().createDFAPacketizer(II);
}

bool HexagonInstrInfo::isSchedulingBoundary(const MachineInstr *MI,
                                            const MachineBasicBlock *MBB,
                                            const MachineFunction &MF) const {
  // Debug info is never a scheduling boundary. It's necessary to be explicit
  // due to the special treatment of IT instructions below, otherwise a
  // dbg_value followed by an IT will result in the IT instruction being
  // considered a scheduling hazard, which is wrong. It should be the actual
  // instruction preceding the dbg_value instruction(s), just like it is
  // when debug info is not present.
  if (MI->isDebugValue())
    return false;

  // Terminators and labels can't be scheduled around.
  if (MI->getDesc().isTerminator() || MI->isLabel() || MI->isInlineAsm())
    return true;

  return false;
}
