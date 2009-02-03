//===- IA64RegisterInfo.cpp - IA64 Register Information ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the IA64 implementation of the TargetRegisterInfo class.
// This file is responsible for the frame pointer elimination optimization
// on IA64.
//
//===----------------------------------------------------------------------===//

#include "IA64.h"
#include "IA64RegisterInfo.h"
#include "IA64InstrBuilder.h"
#include "IA64MachineFunctionInfo.h"
#include "llvm/Constants.h"
#include "llvm/Type.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineLocation.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
using namespace llvm;

IA64RegisterInfo::IA64RegisterInfo(const TargetInstrInfo &tii)
  : IA64GenRegisterInfo(IA64::ADJUSTCALLSTACKDOWN, IA64::ADJUSTCALLSTACKUP),
    TII(tii) {}

const unsigned* IA64RegisterInfo::getCalleeSavedRegs(const MachineFunction *MF)
                                                                         const {
  static const unsigned CalleeSavedRegs[] = {
    IA64::r5,  0
  };
  return CalleeSavedRegs;
}

const TargetRegisterClass* const*
IA64RegisterInfo::getCalleeSavedRegClasses(const MachineFunction *MF) const {
  static const TargetRegisterClass * const CalleeSavedRegClasses[] = {
    &IA64::GRRegClass,  0
  };
  return CalleeSavedRegClasses;
}

BitVector IA64RegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());
  Reserved.set(IA64::r0);
  Reserved.set(IA64::r1);
  Reserved.set(IA64::r2);
  Reserved.set(IA64::r5);
  Reserved.set(IA64::r12);
  Reserved.set(IA64::r13);
  Reserved.set(IA64::r22);
  Reserved.set(IA64::rp);
  return Reserved;
}

//===----------------------------------------------------------------------===//
// Stack Frame Processing methods
//===----------------------------------------------------------------------===//

// hasFP - Return true if the specified function should have a dedicated frame
// pointer register.  This is true if the function has variable sized allocas or
// if frame pointer elimination is disabled.
//
bool IA64RegisterInfo::hasFP(const MachineFunction &MF) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  return NoFramePointerElim || MFI->hasVarSizedObjects();
}

void IA64RegisterInfo::
eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I) const {
  if (hasFP(MF)) {
    // If we have a frame pointer, turn the adjcallstackup instruction into a
    // 'sub SP, <amt>' and the adjcallstackdown instruction into 'add SP,
    // <amt>'
    MachineInstr *Old = I;
    unsigned Amount = Old->getOperand(0).getImm();
    if (Amount != 0) {
      // We need to keep the stack aligned properly.  To do this, we round the
      // amount of space needed for the outgoing arguments up to the next
      // alignment boundary.
      unsigned Align = MF.getTarget().getFrameInfo()->getStackAlignment();
      Amount = (Amount+Align-1)/Align*Align;

      // Replace the pseudo instruction with a new instruction...
      if (Old->getOpcode() == IA64::ADJUSTCALLSTACKDOWN) {
        BuildMI(MBB, I, TII.get(IA64::ADDIMM22), IA64::r12).addReg(IA64::r12)
          .addImm(-Amount);
      } else {
        assert(Old->getOpcode() == IA64::ADJUSTCALLSTACKUP);
        BuildMI(MBB, I, TII.get(IA64::ADDIMM22), IA64::r12).addReg(IA64::r12)
          .addImm(Amount);
      }
    }
  }

  MBB.erase(I);
}

void IA64RegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                           int SPAdj, RegScavenger *RS)const{
  assert(SPAdj == 0 && "Unexpected");

  unsigned i = 0;
  MachineInstr &MI = *II;
  MachineBasicBlock &MBB = *MI.getParent();
  MachineFunction &MF = *MBB.getParent();

  bool FP = hasFP(MF);

  while (!MI.getOperand(i).isFI()) {
    ++i;
    assert(i < MI.getNumOperands() && "Instr doesn't have FrameIndex operand!");
  }

  int FrameIndex = MI.getOperand(i).getIndex();

  // choose a base register: ( hasFP? framepointer : stack pointer )
  unsigned BaseRegister = FP ? IA64::r5 : IA64::r12;
  // Add the base register
  MI.getOperand(i).ChangeToRegister(BaseRegister, false);

  // Now add the frame object offset to the offset from r1.
  int Offset = MF.getFrameInfo()->getObjectOffset(FrameIndex);

  // If we're not using a Frame Pointer that has been set to the value of the
  // SP before having the stack size subtracted from it, then add the stack size
  // to Offset to get the correct offset.
  Offset += MF.getFrameInfo()->getStackSize();

  // XXX: we use 'r22' as another hack+slash temporary register here :(
  if (Offset <= 8191 && Offset >= -8192) { // smallish offset
    // Fix up the old:
    MI.getOperand(i).ChangeToRegister(IA64::r22, false);
    //insert the new
    BuildMI(MBB, II, TII.get(IA64::ADDIMM22), IA64::r22)
      .addReg(BaseRegister).addImm(Offset);
  } else { // it's big
    //fix up the old:
    MI.getOperand(i).ChangeToRegister(IA64::r22, false);
    BuildMI(MBB, II, TII.get(IA64::MOVLIMM64), IA64::r22).addImm(Offset);
    BuildMI(MBB, II, TII.get(IA64::ADD), IA64::r22).addReg(BaseRegister)
      .addReg(IA64::r22);
  }

}

void IA64RegisterInfo::emitPrologue(MachineFunction &MF) const {
  MachineBasicBlock &MBB = MF.front();   // Prolog goes in entry BB
  MachineBasicBlock::iterator MBBI = MBB.begin();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  bool FP = hasFP(MF);

  // first, we handle the 'alloc' instruction, that should be right up the
  // top of any function
  static const unsigned RegsInOrder[96] = { // there are 96 GPRs the
                                            // RSE worries about
        IA64::r32, IA64::r33, IA64::r34, IA64::r35,
        IA64::r36, IA64::r37, IA64::r38, IA64::r39, IA64::r40, IA64::r41,
        IA64::r42, IA64::r43, IA64::r44, IA64::r45, IA64::r46, IA64::r47,
        IA64::r48, IA64::r49, IA64::r50, IA64::r51, IA64::r52, IA64::r53,
        IA64::r54, IA64::r55, IA64::r56, IA64::r57, IA64::r58, IA64::r59,
        IA64::r60, IA64::r61, IA64::r62, IA64::r63, IA64::r64, IA64::r65,
        IA64::r66, IA64::r67, IA64::r68, IA64::r69, IA64::r70, IA64::r71,
        IA64::r72, IA64::r73, IA64::r74, IA64::r75, IA64::r76, IA64::r77,
        IA64::r78, IA64::r79, IA64::r80, IA64::r81, IA64::r82, IA64::r83,
        IA64::r84, IA64::r85, IA64::r86, IA64::r87, IA64::r88, IA64::r89,
        IA64::r90, IA64::r91, IA64::r92, IA64::r93, IA64::r94, IA64::r95,
        IA64::r96, IA64::r97, IA64::r98, IA64::r99, IA64::r100, IA64::r101,
        IA64::r102, IA64::r103, IA64::r104, IA64::r105, IA64::r106, IA64::r107,
        IA64::r108, IA64::r109, IA64::r110, IA64::r111, IA64::r112, IA64::r113,
        IA64::r114, IA64::r115, IA64::r116, IA64::r117, IA64::r118, IA64::r119,
        IA64::r120, IA64::r121, IA64::r122, IA64::r123, IA64::r124, IA64::r125,
        IA64::r126, IA64::r127 };

  unsigned numStackedGPRsUsed=0;
  for (int i=0; i != 96; i++) {
    if (MF.getRegInfo().isPhysRegUsed(RegsInOrder[i]))
      numStackedGPRsUsed=i+1; // (i+1 and not ++ - consider fn(fp, fp, int)
  }

  unsigned numOutRegsUsed=MF.getInfo<IA64FunctionInfo>()->outRegsUsed;

  // XXX FIXME : this code should be a bit more reliable (in case there _isn't_
  // a pseudo_alloc in the MBB)
  unsigned dstRegOfPseudoAlloc;
  for(MBBI = MBB.begin(); /*MBBI->getOpcode() != IA64::PSEUDO_ALLOC*/; ++MBBI) {
    assert(MBBI != MBB.end());
    if(MBBI->getOpcode() == IA64::PSEUDO_ALLOC) {
      dstRegOfPseudoAlloc=MBBI->getOperand(0).getReg();
      break;
    }
  }

  BuildMI(MBB, MBBI, TII.get(IA64::ALLOC)).
     addReg(dstRegOfPseudoAlloc).addImm(0).
     addImm(numStackedGPRsUsed).addImm(numOutRegsUsed).addImm(0);

  // Get the number of bytes to allocate from the FrameInfo
  unsigned NumBytes = MFI->getStackSize();

  if(FP)
    NumBytes += 8; // reserve space for the old FP

  // Do we need to allocate space on the stack?
  if (NumBytes == 0)
    return;

  // Add 16 bytes at the bottom of the stack (scratch area)
  // and round the size to a multiple of the alignment.
  unsigned Align = MF.getTarget().getFrameInfo()->getStackAlignment();
  unsigned Size = 16 + (FP ? 8 : 0);
  NumBytes = (NumBytes+Size+Align-1)/Align*Align;

  // Update frame info to pretend that this is part of the stack...
  MFI->setStackSize(NumBytes);

  // adjust stack pointer: r12 -= numbytes
  if (NumBytes <= 8191) {
    BuildMI(MBB, MBBI, TII.get(IA64::ADDIMM22),IA64::r12).addReg(IA64::r12).
      addImm(-NumBytes);
  } else { // we use r22 as a scratch register here
    // first load the decrement into r22
    BuildMI(MBB, MBBI, TII.get(IA64::MOVLIMM64), IA64::r22).addImm(-NumBytes);
    // FIXME: MOVLSI32 expects a _u_32imm
    // then add (subtract) it to r12 (stack ptr)
    BuildMI(MBB, MBBI, TII.get(IA64::ADD), IA64::r12)
      .addReg(IA64::r12).addReg(IA64::r22);
    
  }

  // now if we need to, save the old FP and set the new
  if (FP) {
    BuildMI(MBB, MBBI, TII.get(IA64::ST8)).addReg(IA64::r12).addReg(IA64::r5);
    // this must be the last instr in the prolog ?  (XXX: why??)
    BuildMI(MBB, MBBI, TII.get(IA64::MOV), IA64::r5).addReg(IA64::r12);
  }

}

void IA64RegisterInfo::emitEpilogue(MachineFunction &MF,
                                   MachineBasicBlock &MBB) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  MachineBasicBlock::iterator MBBI = prior(MBB.end());
  assert(MBBI->getOpcode() == IA64::RET &&
         "Can only insert epilog into returning blocks");

  bool FP = hasFP(MF);

  // Get the number of bytes allocated from the FrameInfo...
  unsigned NumBytes = MFI->getStackSize();

  //now if we need to, restore the old FP
  if (FP)
  {
    //copy the FP into the SP (discards allocas)
    BuildMI(MBB, MBBI, TII.get(IA64::MOV), IA64::r12).addReg(IA64::r5);
    //restore the FP
    BuildMI(MBB, MBBI, TII.get(IA64::LD8), IA64::r5).addReg(IA64::r5);
  }

  if (NumBytes != 0)
  {
    if (NumBytes <= 8191) {
      BuildMI(MBB, MBBI, TII.get(IA64::ADDIMM22),IA64::r12).addReg(IA64::r12).
        addImm(NumBytes);
    } else {
      BuildMI(MBB, MBBI, TII.get(IA64::MOVLIMM64), IA64::r22).
        addImm(NumBytes);
      BuildMI(MBB, MBBI, TII.get(IA64::ADD), IA64::r12).addReg(IA64::r12).
        addReg(IA64::r22);
    }
  }

}

unsigned IA64RegisterInfo::getRARegister() const {
  assert(0 && "What is the return address register");
  return 0;
}

unsigned IA64RegisterInfo::getFrameRegister(MachineFunction &MF) const {
  return hasFP(MF) ? IA64::r5 : IA64::r12;
}

unsigned IA64RegisterInfo::getEHExceptionRegister() const {
  assert(0 && "What is the exception register");
  return 0;
}

unsigned IA64RegisterInfo::getEHHandlerRegister() const {
  assert(0 && "What is the exception handler register");
  return 0;
}

int IA64RegisterInfo::getDwarfRegNum(unsigned RegNum, bool isEH) const {
  assert(0 && "What is the dwarf register number");
  return -1;
}

#include "IA64GenRegisterInfo.inc"

