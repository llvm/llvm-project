//===- IA64InstrInfo.cpp - IA64 Instruction Information -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the IA64 implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "IA64InstrInfo.h"
#include "IA64.h"
#include "IA64InstrBuilder.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/ADT/SmallVector.h"
#include "IA64GenInstrInfo.inc"
using namespace llvm;

IA64InstrInfo::IA64InstrInfo()
  : TargetInstrInfoImpl(IA64Insts, sizeof(IA64Insts)/sizeof(IA64Insts[0])),
    RI(*this) {
}


bool IA64InstrInfo::isMoveInstr(const MachineInstr& MI,
                                unsigned& sourceReg,
                                unsigned& destReg,
                                unsigned& SrcSR, unsigned& DstSR) const {
  SrcSR = DstSR = 0;  // No sub-registers.

  unsigned oc = MI.getOpcode();
  if (oc == IA64::MOV || oc == IA64::FMOV) {
  // TODO: this doesn't detect predicate moves
     assert(MI.getNumOperands() >= 2 &&
             /* MI.getOperand(0).isReg() &&
             MI.getOperand(1).isReg() && */
             "invalid register-register move instruction");
     if (MI.getOperand(0).isReg() &&
         MI.getOperand(1).isReg()) {
       // if both operands of the MOV/FMOV are registers, then
       // yes, this is a move instruction
       sourceReg = MI.getOperand(1).getReg();
       destReg = MI.getOperand(0).getReg();
       return true;
     }
  }
  return false; // we don't consider e.g. %regN = MOV <FrameIndex #x> a
                // move instruction
}

unsigned
IA64InstrInfo::InsertBranch(MachineBasicBlock &MBB,MachineBasicBlock *TBB,
                            MachineBasicBlock *FBB,
                            const SmallVectorImpl<MachineOperand> &Cond)const {
  // Can only insert uncond branches so far.
  assert(Cond.empty() && !FBB && TBB && "Can only handle uncond branches!");
  BuildMI(&MBB, get(IA64::BRL_NOTCALL)).addMBB(TBB);
  return 1;
}

bool IA64InstrInfo::copyRegToReg(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MI,
                                   unsigned DestReg, unsigned SrcReg,
                                   const TargetRegisterClass *DestRC,
                                   const TargetRegisterClass *SrcRC) const {
  if (DestRC != SrcRC) {
    // Not yet supported!
    return false;
  }

  if(DestRC == IA64::PRRegisterClass ) // if a bool, we use pseudocode
    // (SrcReg) DestReg = cmp.eq.unc(r0, r0)
    BuildMI(MBB, MI, get(IA64::PCMPEQUNC), DestReg)
      .addReg(IA64::r0).addReg(IA64::r0).addReg(SrcReg);
  else // otherwise, MOV works (for both gen. regs and FP regs)
    BuildMI(MBB, MI, get(IA64::MOV), DestReg).addReg(SrcReg);
  
  return true;
}

void IA64InstrInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                           MachineBasicBlock::iterator MI,
                                           unsigned SrcReg, bool isKill,
                                           int FrameIdx,
                                           const TargetRegisterClass *RC) const{

  if (RC == IA64::FPRegisterClass) {
    BuildMI(MBB, MI, get(IA64::STF_SPILL)).addFrameIndex(FrameIdx)
      .addReg(SrcReg, false, false, isKill);
  } else if (RC == IA64::GRRegisterClass) {
    BuildMI(MBB, MI, get(IA64::ST8)).addFrameIndex(FrameIdx)
      .addReg(SrcReg, false, false, isKill);
  } else if (RC == IA64::PRRegisterClass) {
    /* we use IA64::r2 as a temporary register for doing this hackery. */
    // first we load 0:
    BuildMI(MBB, MI, get(IA64::MOV), IA64::r2).addReg(IA64::r0);
    // then conditionally add 1:
    BuildMI(MBB, MI, get(IA64::CADDIMM22), IA64::r2).addReg(IA64::r2)
      .addImm(1).addReg(SrcReg, false, false, isKill);
    // and then store it to the stack
    BuildMI(MBB, MI, get(IA64::ST8)).addFrameIndex(FrameIdx).addReg(IA64::r2);
  } else assert(0 &&
      "sorry, I don't know how to store this sort of reg in the stack\n");
}

void IA64InstrInfo::storeRegToAddr(MachineFunction &MF, unsigned SrcReg,
                                      bool isKill,
                                      SmallVectorImpl<MachineOperand> &Addr,
                                      const TargetRegisterClass *RC,
                                 SmallVectorImpl<MachineInstr*> &NewMIs) const {
  unsigned Opc = 0;
  if (RC == IA64::FPRegisterClass) {
    Opc = IA64::STF8;
  } else if (RC == IA64::GRRegisterClass) {
    Opc = IA64::ST8;
  } else if (RC == IA64::PRRegisterClass) {
    Opc = IA64::ST1;
  } else {
    assert(0 &&
      "sorry, I don't know how to store this sort of reg\n");
  }

  MachineInstrBuilder MIB = BuildMI(MF, get(Opc));
  for (unsigned i = 0, e = Addr.size(); i != e; ++i) {
    MachineOperand &MO = Addr[i];
    if (MO.isReg())
      MIB.addReg(MO.getReg());
    else if (MO.isImm())
      MIB.addImm(MO.getImm());
    else
      MIB.addFrameIndex(MO.getIndex());
  }
  MIB.addReg(SrcReg, false, false, isKill);
  NewMIs.push_back(MIB);
  return;

}

void IA64InstrInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                            MachineBasicBlock::iterator MI,
                                            unsigned DestReg, int FrameIdx,
                                            const TargetRegisterClass *RC)const{

  if (RC == IA64::FPRegisterClass) {
    BuildMI(MBB, MI, get(IA64::LDF_FILL), DestReg).addFrameIndex(FrameIdx);
  } else if (RC == IA64::GRRegisterClass) {
    BuildMI(MBB, MI, get(IA64::LD8), DestReg).addFrameIndex(FrameIdx);
 } else if (RC == IA64::PRRegisterClass) {
   // first we load a byte from the stack into r2, our 'predicate hackery'
   // scratch reg
   BuildMI(MBB, MI, get(IA64::LD8), IA64::r2).addFrameIndex(FrameIdx);
   // then we compare it to zero. If it _is_ zero, compare-not-equal to
   // r0 gives us 0, which is what we want, so that's nice.
   BuildMI(MBB, MI, get(IA64::CMPNE), DestReg).addReg(IA64::r2).addReg(IA64::r0);
 } else assert(0 &&
     "sorry, I don't know how to load this sort of reg from the stack\n");
}

void IA64InstrInfo::loadRegFromAddr(MachineFunction &MF, unsigned DestReg,
                                       SmallVectorImpl<MachineOperand> &Addr,
                                       const TargetRegisterClass *RC,
                                 SmallVectorImpl<MachineInstr*> &NewMIs) const {
  unsigned Opc = 0;
  if (RC == IA64::FPRegisterClass) {
    Opc = IA64::LDF8;
  } else if (RC == IA64::GRRegisterClass) {
    Opc = IA64::LD8;
  } else if (RC == IA64::PRRegisterClass) {
    Opc = IA64::LD1;
  } else {
    assert(0 &&
      "sorry, I don't know how to store this sort of reg\n");
  }

  MachineInstrBuilder MIB = BuildMI(MF, get(Opc), DestReg);
  for (unsigned i = 0, e = Addr.size(); i != e; ++i) {
    MachineOperand &MO = Addr[i];
    if (MO.isReg())
      MIB.addReg(MO.getReg());
    else if (MO.isImm())
      MIB.addImm(MO.getImm());
    else
      MIB.addFrameIndex(MO.getIndex());
  }
  NewMIs.push_back(MIB);
  return;
}
