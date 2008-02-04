//===- X86RegisterInfo.cpp - X86 Register Information -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the X86 implementation of the MRegisterInfo class.  This
// file is responsible for the frame pointer elimination optimization on X86.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86RegisterInfo.h"
#include "X86InstrBuilder.h"
#include "X86MachineFunctionInfo.h"
#include "X86Subtarget.h"
#include "X86TargetMachine.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/Type.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineLocation.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
using namespace llvm;

X86RegisterInfo::X86RegisterInfo(X86TargetMachine &tm,
                                 const TargetInstrInfo &tii)
  : X86GenRegisterInfo(X86::ADJCALLSTACKDOWN, X86::ADJCALLSTACKUP),
    TM(tm), TII(tii) {
  // Cache some information.
  const X86Subtarget *Subtarget = &TM.getSubtarget<X86Subtarget>();
  Is64Bit = Subtarget->is64Bit();
  StackAlign = TM.getFrameInfo()->getStackAlignment();
  if (Is64Bit) {
    SlotSize = 8;
    StackPtr = X86::RSP;
    FramePtr = X86::RBP;
  } else {
    SlotSize = 4;
    StackPtr = X86::ESP;
    FramePtr = X86::EBP;
  }
}

// getDwarfRegNum - This function maps LLVM register identifiers to the
// Dwarf specific numbering, used in debug info and exception tables.

int X86RegisterInfo::getDwarfRegNum(unsigned RegNo, bool isEH) const {
  const X86Subtarget *Subtarget = &TM.getSubtarget<X86Subtarget>();
  unsigned Flavour = DWARFFlavour::X86_64;
  if (!Subtarget->is64Bit()) {
    if (Subtarget->isTargetDarwin()) {
      if (isEH)
        Flavour = DWARFFlavour::X86_32_DarwinEH;
      else
        Flavour = DWARFFlavour::X86_32_Generic;
    } else if (Subtarget->isTargetCygMing()) {
      // Unsupported by now, just quick fallback
      Flavour = DWARFFlavour::X86_32_Generic;
    } else {
      Flavour = DWARFFlavour::X86_32_Generic;
    }
  }

  return X86GenRegisterInfo::getDwarfRegNumFull(RegNo, Flavour);
}

// getX86RegNum - This function maps LLVM register identifiers to their X86
// specific numbering, which is used in various places encoding instructions.
//
unsigned X86RegisterInfo::getX86RegNum(unsigned RegNo) {
  switch(RegNo) {
  case X86::RAX: case X86::EAX: case X86::AX: case X86::AL: return N86::EAX;
  case X86::RCX: case X86::ECX: case X86::CX: case X86::CL: return N86::ECX;
  case X86::RDX: case X86::EDX: case X86::DX: case X86::DL: return N86::EDX;
  case X86::RBX: case X86::EBX: case X86::BX: case X86::BL: return N86::EBX;
  case X86::RSP: case X86::ESP: case X86::SP: case X86::SPL: case X86::AH:
    return N86::ESP;
  case X86::RBP: case X86::EBP: case X86::BP: case X86::BPL: case X86::CH:
    return N86::EBP;
  case X86::RSI: case X86::ESI: case X86::SI: case X86::SIL: case X86::DH:
    return N86::ESI;
  case X86::RDI: case X86::EDI: case X86::DI: case X86::DIL: case X86::BH:
    return N86::EDI;

  case X86::R8:  case X86::R8D:  case X86::R8W:  case X86::R8B:
    return N86::EAX;
  case X86::R9:  case X86::R9D:  case X86::R9W:  case X86::R9B:
    return N86::ECX;
  case X86::R10: case X86::R10D: case X86::R10W: case X86::R10B:
    return N86::EDX;
  case X86::R11: case X86::R11D: case X86::R11W: case X86::R11B:
    return N86::EBX;
  case X86::R12: case X86::R12D: case X86::R12W: case X86::R12B:
    return N86::ESP;
  case X86::R13: case X86::R13D: case X86::R13W: case X86::R13B:
    return N86::EBP;
  case X86::R14: case X86::R14D: case X86::R14W: case X86::R14B:
    return N86::ESI;
  case X86::R15: case X86::R15D: case X86::R15W: case X86::R15B:
    return N86::EDI;

  case X86::ST0: case X86::ST1: case X86::ST2: case X86::ST3:
  case X86::ST4: case X86::ST5: case X86::ST6: case X86::ST7:
    return RegNo-X86::ST0;

  case X86::XMM0: case X86::XMM8: case X86::MM0:
    return 0;
  case X86::XMM1: case X86::XMM9: case X86::MM1:
    return 1;
  case X86::XMM2: case X86::XMM10: case X86::MM2:
    return 2;
  case X86::XMM3: case X86::XMM11: case X86::MM3:
    return 3;
  case X86::XMM4: case X86::XMM12: case X86::MM4:
    return 4;
  case X86::XMM5: case X86::XMM13: case X86::MM5:
    return 5;
  case X86::XMM6: case X86::XMM14: case X86::MM6:
    return 6;
  case X86::XMM7: case X86::XMM15: case X86::MM7:
    return 7;

  default:
    assert(isVirtualRegister(RegNo) && "Unknown physical register!");
    assert(0 && "Register allocator hasn't allocated reg correctly yet!");
    return 0;
  }
}

const TargetRegisterClass *
X86RegisterInfo::getCrossCopyRegClass(const TargetRegisterClass *RC) const {
  if (RC == &X86::CCRRegClass)
    if (Is64Bit)
      return &X86::GR64RegClass;
    else
      return &X86::GR32RegClass;
  return NULL;
}

void X86RegisterInfo::reMaterialize(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator I,
                                    unsigned DestReg,
                                    const MachineInstr *Orig) const {
  // MOV32r0 etc. are implemented with xor which clobbers condition code.
  // Re-materialize them as movri instructions to avoid side effects.
  switch (Orig->getOpcode()) {
  case X86::MOV8r0:
    BuildMI(MBB, I, TII.get(X86::MOV8ri), DestReg).addImm(0);
    break;
  case X86::MOV16r0:
    BuildMI(MBB, I, TII.get(X86::MOV16ri), DestReg).addImm(0);
    break;
  case X86::MOV32r0:
    BuildMI(MBB, I, TII.get(X86::MOV32ri), DestReg).addImm(0);
    break;
  case X86::MOV64r0:
    BuildMI(MBB, I, TII.get(X86::MOV64ri32), DestReg).addImm(0);
    break;
  default: {
    MachineInstr *MI = Orig->clone();
    MI->getOperand(0).setReg(DestReg);
    MBB.insert(I, MI);
    break;
  }
  }
}

const unsigned *
X86RegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  static const unsigned CalleeSavedRegs32Bit[] = {
    X86::ESI, X86::EDI, X86::EBX, X86::EBP,  0
  };

  static const unsigned CalleeSavedRegs32EHRet[] = {
    X86::EAX, X86::EDX, X86::ESI, X86::EDI, X86::EBX, X86::EBP,  0
  };

  static const unsigned CalleeSavedRegs64Bit[] = {
    X86::RBX, X86::R12, X86::R13, X86::R14, X86::R15, X86::RBP, 0
  };

  if (Is64Bit)
    return CalleeSavedRegs64Bit;
  else {
    if (MF) {
        MachineFrameInfo *MFI = MF->getFrameInfo();
        MachineModuleInfo *MMI = MFI->getMachineModuleInfo();
        if (MMI && MMI->callsEHReturn())
          return CalleeSavedRegs32EHRet;
    }
    return CalleeSavedRegs32Bit;
  }
}

const TargetRegisterClass* const*
X86RegisterInfo::getCalleeSavedRegClasses(const MachineFunction *MF) const {
  static const TargetRegisterClass * const CalleeSavedRegClasses32Bit[] = {
    &X86::GR32RegClass, &X86::GR32RegClass,
    &X86::GR32RegClass, &X86::GR32RegClass,  0
  };
  static const TargetRegisterClass * const CalleeSavedRegClasses32EHRet[] = {
    &X86::GR32RegClass, &X86::GR32RegClass,
    &X86::GR32RegClass, &X86::GR32RegClass,
    &X86::GR32RegClass, &X86::GR32RegClass,  0
  };
  static const TargetRegisterClass * const CalleeSavedRegClasses64Bit[] = {
    &X86::GR64RegClass, &X86::GR64RegClass,
    &X86::GR64RegClass, &X86::GR64RegClass,
    &X86::GR64RegClass, &X86::GR64RegClass, 0
  };

  if (Is64Bit)
    return CalleeSavedRegClasses64Bit;
  else {
    if (MF) {
        MachineFrameInfo *MFI = MF->getFrameInfo();
        MachineModuleInfo *MMI = MFI->getMachineModuleInfo();
        if (MMI && MMI->callsEHReturn())
          return CalleeSavedRegClasses32EHRet;
    }
    return CalleeSavedRegClasses32Bit;
  }

}

BitVector X86RegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());
  Reserved.set(X86::RSP);
  Reserved.set(X86::ESP);
  Reserved.set(X86::SP);
  Reserved.set(X86::SPL);
  if (hasFP(MF)) {
    Reserved.set(X86::RBP);
    Reserved.set(X86::EBP);
    Reserved.set(X86::BP);
    Reserved.set(X86::BPL);
  }
  return Reserved;
}

//===----------------------------------------------------------------------===//
// Stack Frame Processing methods
//===----------------------------------------------------------------------===//

// hasFP - Return true if the specified function should have a dedicated frame
// pointer register.  This is true if the function has variable sized allocas or
// if frame pointer elimination is disabled.
//
bool X86RegisterInfo::hasFP(const MachineFunction &MF) const {
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MachineModuleInfo *MMI = MFI->getMachineModuleInfo();

  return (NoFramePointerElim || 
          MFI->hasVarSizedObjects() ||
          MF.getInfo<X86MachineFunctionInfo>()->getForceFramePointer() ||
          (MMI && MMI->callsUnwindInit()));
}

bool X86RegisterInfo::hasReservedCallFrame(MachineFunction &MF) const {
  return !MF.getFrameInfo()->hasVarSizedObjects();
}

void X86RegisterInfo::
eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I) const {
  if (!hasReservedCallFrame(MF)) {
    // If the stack pointer can be changed after prologue, turn the
    // adjcallstackup instruction into a 'sub ESP, <amt>' and the
    // adjcallstackdown instruction into 'add ESP, <amt>'
    // TODO: consider using push / pop instead of sub + store / add
    MachineInstr *Old = I;
    uint64_t Amount = Old->getOperand(0).getImm();
    if (Amount != 0) {
      // We need to keep the stack aligned properly.  To do this, we round the
      // amount of space needed for the outgoing arguments up to the next
      // alignment boundary.
      Amount = (Amount+StackAlign-1)/StackAlign*StackAlign;

      MachineInstr *New = 0;
      if (Old->getOpcode() == X86::ADJCALLSTACKDOWN) {
        New=BuildMI(TII.get(Is64Bit ? X86::SUB64ri32 : X86::SUB32ri), StackPtr)
          .addReg(StackPtr).addImm(Amount);
      } else {
        assert(Old->getOpcode() == X86::ADJCALLSTACKUP);
        // factor out the amount the callee already popped.
        uint64_t CalleeAmt = Old->getOperand(1).getImm();
        Amount -= CalleeAmt;
        if (Amount) {
          unsigned Opc = (Amount < 128) ?
            (Is64Bit ? X86::ADD64ri8 : X86::ADD32ri8) :
            (Is64Bit ? X86::ADD64ri32 : X86::ADD32ri);
          New = BuildMI(TII.get(Opc), StackPtr).addReg(StackPtr).addImm(Amount);
        }
      }

      // Replace the pseudo instruction with a new instruction...
      if (New) MBB.insert(I, New);
    }
  } else if (I->getOpcode() == X86::ADJCALLSTACKUP) {
    // If we are performing frame pointer elimination and if the callee pops
    // something off the stack pointer, add it back.  We do this until we have
    // more advanced stack pointer tracking ability.
    if (uint64_t CalleeAmt = I->getOperand(1).getImm()) {
      unsigned Opc = (CalleeAmt < 128) ?
        (Is64Bit ? X86::SUB64ri8 : X86::SUB32ri8) :
        (Is64Bit ? X86::SUB64ri32 : X86::SUB32ri);
      MachineInstr *New =
        BuildMI(TII.get(Opc), StackPtr).addReg(StackPtr).addImm(CalleeAmt);
      MBB.insert(I, New);
    }
  }

  MBB.erase(I);
}

void X86RegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                          int SPAdj, RegScavenger *RS) const{
  assert(SPAdj == 0 && "Unexpected");

  unsigned i = 0;
  MachineInstr &MI = *II;
  MachineFunction &MF = *MI.getParent()->getParent();
  while (!MI.getOperand(i).isFrameIndex()) {
    ++i;
    assert(i < MI.getNumOperands() && "Instr doesn't have FrameIndex operand!");
  }

  int FrameIndex = MI.getOperand(i).getIndex();
  // This must be part of a four operand memory reference.  Replace the
  // FrameIndex with base register with EBP.  Add an offset to the offset.
  MI.getOperand(i).ChangeToRegister(hasFP(MF) ? FramePtr : StackPtr, false);

  // Now add the frame object offset to the offset from EBP.
  int64_t Offset = MF.getFrameInfo()->getObjectOffset(FrameIndex) +
                   MI.getOperand(i+3).getImm()+SlotSize;

  if (!hasFP(MF))
    Offset += MF.getFrameInfo()->getStackSize();
  else {
    Offset += SlotSize;  // Skip the saved EBP
    // Skip the RETADDR move area
    X86MachineFunctionInfo *X86FI = MF.getInfo<X86MachineFunctionInfo>();
    int TailCallReturnAddrDelta = X86FI->getTCReturnAddrDelta();
    if (TailCallReturnAddrDelta < 0) Offset -= TailCallReturnAddrDelta;
  }
  
  MI.getOperand(i+3).ChangeToImmediate(Offset);
}

void
X86RegisterInfo::processFunctionBeforeFrameFinalized(MachineFunction &MF) const{
  X86MachineFunctionInfo *X86FI = MF.getInfo<X86MachineFunctionInfo>();
  int32_t TailCallReturnAddrDelta = X86FI->getTCReturnAddrDelta();
  if (TailCallReturnAddrDelta < 0) {
    // create RETURNADDR area
    //   arg
    //   arg
    //   RETADDR
    //   { ...
    //     RETADDR area
    //     ...
    //   }
    //   [EBP]
    MF.getFrameInfo()->
      CreateFixedObject(-TailCallReturnAddrDelta,
                        (-1*SlotSize)+TailCallReturnAddrDelta);
  }
  if (hasFP(MF)) {
    assert((TailCallReturnAddrDelta <= 0) &&
           "The Delta should always be zero or negative");
    // Create a frame entry for the EBP register that must be saved.
    int FrameIdx = MF.getFrameInfo()->CreateFixedObject(SlotSize,
                                                        (int)SlotSize * -2+
                                                       TailCallReturnAddrDelta);
    assert(FrameIdx == MF.getFrameInfo()->getObjectIndexBegin() &&
           "Slot for EBP register must be last in order to be found!");
  }
}

/// emitSPUpdate - Emit a series of instructions to increment / decrement the
/// stack pointer by a constant value.
static
void emitSPUpdate(MachineBasicBlock &MBB, MachineBasicBlock::iterator &MBBI,
                  unsigned StackPtr, int64_t NumBytes, bool Is64Bit,
                  const TargetInstrInfo &TII) {
  bool isSub = NumBytes < 0;
  uint64_t Offset = isSub ? -NumBytes : NumBytes;
  unsigned Opc = isSub
    ? ((Offset < 128) ?
       (Is64Bit ? X86::SUB64ri8 : X86::SUB32ri8) :
       (Is64Bit ? X86::SUB64ri32 : X86::SUB32ri))
    : ((Offset < 128) ?
       (Is64Bit ? X86::ADD64ri8 : X86::ADD32ri8) :
       (Is64Bit ? X86::ADD64ri32 : X86::ADD32ri));
  uint64_t Chunk = (1LL << 31) - 1;

  while (Offset) {
    uint64_t ThisVal = (Offset > Chunk) ? Chunk : Offset;
    BuildMI(MBB, MBBI, TII.get(Opc), StackPtr).addReg(StackPtr).addImm(ThisVal);
    Offset -= ThisVal;
  }
}

// mergeSPUpdatesUp - Merge two stack-manipulating instructions upper iterator.
static
void mergeSPUpdatesUp(MachineBasicBlock &MBB, MachineBasicBlock::iterator &MBBI,
                      unsigned StackPtr, uint64_t *NumBytes = NULL) {
  if (MBBI == MBB.begin()) return;
  
  MachineBasicBlock::iterator PI = prior(MBBI);
  unsigned Opc = PI->getOpcode();
  if ((Opc == X86::ADD64ri32 || Opc == X86::ADD64ri8 ||
       Opc == X86::ADD32ri || Opc == X86::ADD32ri8) &&
      PI->getOperand(0).getReg() == StackPtr) {
    if (NumBytes)
      *NumBytes += PI->getOperand(2).getImm();
    MBB.erase(PI);
  } else if ((Opc == X86::SUB64ri32 || Opc == X86::SUB64ri8 ||
              Opc == X86::SUB32ri || Opc == X86::SUB32ri8) &&
             PI->getOperand(0).getReg() == StackPtr) {
    if (NumBytes)
      *NumBytes -= PI->getOperand(2).getImm();
    MBB.erase(PI);
  }
}

// mergeSPUpdatesUp - Merge two stack-manipulating instructions lower iterator.
static
void mergeSPUpdatesDown(MachineBasicBlock &MBB,
                        MachineBasicBlock::iterator &MBBI,
                        unsigned StackPtr, uint64_t *NumBytes = NULL) {
  return;
  
  if (MBBI == MBB.end()) return;
  
  MachineBasicBlock::iterator NI = next(MBBI);
  if (NI == MBB.end()) return;
  
  unsigned Opc = NI->getOpcode();
  if ((Opc == X86::ADD64ri32 || Opc == X86::ADD64ri8 ||
       Opc == X86::ADD32ri || Opc == X86::ADD32ri8) &&
      NI->getOperand(0).getReg() == StackPtr) {
    if (NumBytes)
      *NumBytes -= NI->getOperand(2).getImm();
    MBB.erase(NI);
    MBBI = NI;
  } else if ((Opc == X86::SUB64ri32 || Opc == X86::SUB64ri8 ||
              Opc == X86::SUB32ri || Opc == X86::SUB32ri8) &&
             NI->getOperand(0).getReg() == StackPtr) {
    if (NumBytes)
      *NumBytes += NI->getOperand(2).getImm();
    MBB.erase(NI);
    MBBI = NI;
  }
}

/// mergeSPUpdates - Checks the instruction before/after the passed
/// instruction. If it is an ADD/SUB instruction it is deleted 
/// argument and the stack adjustment is returned as a positive value for ADD
/// and a negative for SUB. 
static int mergeSPUpdates(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator &MBBI,
                           unsigned StackPtr,                     
                           bool doMergeWithPrevious) {

  if ((doMergeWithPrevious && MBBI == MBB.begin()) ||
      (!doMergeWithPrevious && MBBI == MBB.end()))
    return 0;

  int Offset = 0;

  MachineBasicBlock::iterator PI = doMergeWithPrevious ? prior(MBBI) : MBBI;
  MachineBasicBlock::iterator NI = doMergeWithPrevious ? 0 : next(MBBI);
  unsigned Opc = PI->getOpcode();
  if ((Opc == X86::ADD64ri32 || Opc == X86::ADD64ri8 ||
       Opc == X86::ADD32ri || Opc == X86::ADD32ri8) &&
      PI->getOperand(0).getReg() == StackPtr){
    Offset += PI->getOperand(2).getImm();
    MBB.erase(PI);
    if (!doMergeWithPrevious) MBBI = NI;
  } else if ((Opc == X86::SUB64ri32 || Opc == X86::SUB64ri8 ||
              Opc == X86::SUB32ri || Opc == X86::SUB32ri8) &&
             PI->getOperand(0).getReg() == StackPtr) {
    Offset -= PI->getOperand(2).getImm();
    MBB.erase(PI);
    if (!doMergeWithPrevious) MBBI = NI;
  }   

  return Offset;
}

void X86RegisterInfo::emitPrologue(MachineFunction &MF) const {
  MachineBasicBlock &MBB = MF.front();   // Prolog goes in entry BB
  MachineFrameInfo *MFI = MF.getFrameInfo();
  const Function* Fn = MF.getFunction();
  const X86Subtarget* Subtarget = &MF.getTarget().getSubtarget<X86Subtarget>();
  MachineModuleInfo *MMI = MFI->getMachineModuleInfo();
  X86MachineFunctionInfo *X86FI = MF.getInfo<X86MachineFunctionInfo>();
  MachineBasicBlock::iterator MBBI = MBB.begin();
  
  // Prepare for frame info.
  unsigned FrameLabelId = 0;
  
  // Get the number of bytes to allocate from the FrameInfo.
  uint64_t StackSize = MFI->getStackSize();
  // Add RETADDR move area to callee saved frame size.
  int TailCallReturnAddrDelta = X86FI->getTCReturnAddrDelta();
  if (TailCallReturnAddrDelta < 0)  
    X86FI->setCalleeSavedFrameSize(
          X86FI->getCalleeSavedFrameSize() +(-TailCallReturnAddrDelta));
  uint64_t NumBytes = StackSize - X86FI->getCalleeSavedFrameSize();

  // Insert stack pointer adjustment for later moving of return addr.  Only
  // applies to tail call optimized functions where the callee argument stack
  // size is bigger than the callers.
  if (TailCallReturnAddrDelta < 0) {
    BuildMI(MBB, MBBI, TII.get(Is64Bit? X86::SUB64ri32 : X86::SUB32ri), 
            StackPtr).addReg(StackPtr).addImm(-TailCallReturnAddrDelta);
  }

  if (hasFP(MF)) {
    // Get the offset of the stack slot for the EBP register... which is
    // guaranteed to be the last slot by processFunctionBeforeFrameFinalized.
    // Update the frame offset adjustment.
    MFI->setOffsetAdjustment(SlotSize-NumBytes);

    // Save EBP into the appropriate stack slot...
    BuildMI(MBB, MBBI, TII.get(Is64Bit ? X86::PUSH64r : X86::PUSH32r))
      .addReg(FramePtr);
    NumBytes -= SlotSize;

    if (MMI && MMI->needsFrameInfo()) {
      // Mark effective beginning of when frame pointer becomes valid.
      FrameLabelId = MMI->NextLabelID();
      BuildMI(MBB, MBBI, TII.get(X86::LABEL)).addImm(FrameLabelId);
    }

    // Update EBP with the new base value...
    BuildMI(MBB, MBBI, TII.get(Is64Bit ? X86::MOV64rr : X86::MOV32rr), FramePtr)
      .addReg(StackPtr);
  }
  
  unsigned ReadyLabelId = 0;
  if (MMI && MMI->needsFrameInfo()) {
    // Mark effective beginning of when frame pointer is ready.
    ReadyLabelId = MMI->NextLabelID();
    BuildMI(MBB, MBBI, TII.get(X86::LABEL)).addImm(ReadyLabelId);
  }

  // Skip the callee-saved push instructions.
  while (MBBI != MBB.end() &&
         (MBBI->getOpcode() == X86::PUSH32r ||
          MBBI->getOpcode() == X86::PUSH64r))
    ++MBBI;

  if (NumBytes) {   // adjust stack pointer: ESP -= numbytes
    if (NumBytes >= 4096 && Subtarget->isTargetCygMing()) {
      // Check, whether EAX is livein for this function
      bool isEAXAlive = false;
      for (MachineRegisterInfo::livein_iterator
           II = MF.getRegInfo().livein_begin(),
           EE = MF.getRegInfo().livein_end(); (II != EE) && !isEAXAlive; ++II) {
        unsigned Reg = II->first;
        isEAXAlive = (Reg == X86::EAX || Reg == X86::AX ||
                      Reg == X86::AH || Reg == X86::AL);
      }

      // Function prologue calls _alloca to probe the stack when allocating  
      // more than 4k bytes in one go. Touching the stack at 4K increments is  
      // necessary to ensure that the guard pages used by the OS virtual memory
      // manager are allocated in correct sequence.
      if (!isEAXAlive) {
        BuildMI(MBB, MBBI, TII.get(X86::MOV32ri), X86::EAX).addImm(NumBytes);
        BuildMI(MBB, MBBI, TII.get(X86::CALLpcrel32))
          .addExternalSymbol("_alloca");
      } else {
        // Save EAX
        BuildMI(MBB, MBBI, TII.get(X86::PUSH32r), X86::EAX);
        // Allocate NumBytes-4 bytes on stack. We'll also use 4 already
        // allocated bytes for EAX.
        BuildMI(MBB, MBBI, TII.get(X86::MOV32ri), X86::EAX).addImm(NumBytes-4);
        BuildMI(MBB, MBBI, TII.get(X86::CALLpcrel32))
          .addExternalSymbol("_alloca");
        // Restore EAX
        MachineInstr *MI = addRegOffset(BuildMI(TII.get(X86::MOV32rm),X86::EAX),
                                        StackPtr, NumBytes-4);
        MBB.insert(MBBI, MI);
      }
    } else {
      // If there is an SUB32ri of ESP immediately before this instruction,
      // merge the two. This can be the case when tail call elimination is
      // enabled and the callee has more arguments then the caller.
      NumBytes -= mergeSPUpdates(MBB, MBBI, StackPtr, true);
      // If there is an ADD32ri or SUB32ri of ESP immediately after this
      // instruction, merge the two instructions.
      mergeSPUpdatesDown(MBB, MBBI, StackPtr, &NumBytes);
      
      if (NumBytes)
        emitSPUpdate(MBB, MBBI, StackPtr, -(int64_t)NumBytes, Is64Bit, TII);
    }
  }

  if (MMI && MMI->needsFrameInfo()) {
    std::vector<MachineMove> &Moves = MMI->getFrameMoves();
    const TargetData *TD = MF.getTarget().getTargetData();

    // Calculate amount of bytes used for return address storing
    int stackGrowth =
      (MF.getTarget().getFrameInfo()->getStackGrowthDirection() ==
       TargetFrameInfo::StackGrowsUp ?
       TD->getPointerSize() : -TD->getPointerSize());

    if (StackSize) {
      // Show update of SP.
      if (hasFP(MF)) {
        // Adjust SP
        MachineLocation SPDst(MachineLocation::VirtualFP);
        MachineLocation SPSrc(MachineLocation::VirtualFP, 2*stackGrowth);
        Moves.push_back(MachineMove(FrameLabelId, SPDst, SPSrc));
      } else {
        MachineLocation SPDst(MachineLocation::VirtualFP);
        MachineLocation SPSrc(MachineLocation::VirtualFP,
                              -StackSize+stackGrowth);
        Moves.push_back(MachineMove(FrameLabelId, SPDst, SPSrc));
      }
    } else {
      //FIXME: Verify & implement for FP
      MachineLocation SPDst(StackPtr);
      MachineLocation SPSrc(StackPtr, stackGrowth);
      Moves.push_back(MachineMove(FrameLabelId, SPDst, SPSrc));
    }
            
    // Add callee saved registers to move list.
    const std::vector<CalleeSavedInfo> &CSI = MFI->getCalleeSavedInfo();

    // FIXME: This is dirty hack. The code itself is pretty mess right now.
    // It should be rewritten from scratch and generalized sometimes.
    
    // Determine maximum offset (minumum due to stack growth)
    int64_t MaxOffset = 0;
    for (unsigned I = 0, E = CSI.size(); I!=E; ++I)
      MaxOffset = std::min(MaxOffset,
                           MFI->getObjectOffset(CSI[I].getFrameIdx()));

    // Calculate offsets
    int64_t saveAreaOffset = (hasFP(MF) ? 3 : 2)*stackGrowth;
    for (unsigned I = 0, E = CSI.size(); I!=E; ++I) {
      int64_t Offset = MFI->getObjectOffset(CSI[I].getFrameIdx());
      unsigned Reg = CSI[I].getReg();
      Offset = (MaxOffset-Offset+saveAreaOffset);
      MachineLocation CSDst(MachineLocation::VirtualFP, Offset);
      MachineLocation CSSrc(Reg);
      Moves.push_back(MachineMove(FrameLabelId, CSDst, CSSrc));
    }
    
    if (hasFP(MF)) {
      // Save FP
      MachineLocation FPDst(MachineLocation::VirtualFP, 2*stackGrowth);
      MachineLocation FPSrc(FramePtr);
      Moves.push_back(MachineMove(ReadyLabelId, FPDst, FPSrc));
    }
    
    MachineLocation FPDst(hasFP(MF) ? FramePtr : StackPtr);
    MachineLocation FPSrc(MachineLocation::VirtualFP);
    Moves.push_back(MachineMove(ReadyLabelId, FPDst, FPSrc));
  }

  // If it's main() on Cygwin\Mingw32 we should align stack as well
  if (Fn->hasExternalLinkage() && Fn->getName() == "main" &&
      Subtarget->isTargetCygMing()) {
    BuildMI(MBB, MBBI, TII.get(X86::AND32ri), X86::ESP)
                .addReg(X86::ESP).addImm(-StackAlign);

    // Probe the stack
    BuildMI(MBB, MBBI, TII.get(X86::MOV32ri), X86::EAX).addImm(StackAlign);
    BuildMI(MBB, MBBI, TII.get(X86::CALLpcrel32)).addExternalSymbol("_alloca");
  }
}

void X86RegisterInfo::emitEpilogue(MachineFunction &MF,
                                   MachineBasicBlock &MBB) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  const Function* Fn = MF.getFunction();
  X86MachineFunctionInfo *X86FI = MF.getInfo<X86MachineFunctionInfo>();
  const X86Subtarget* Subtarget = &MF.getTarget().getSubtarget<X86Subtarget>();
  MachineBasicBlock::iterator MBBI = prior(MBB.end());
  unsigned RetOpcode = MBBI->getOpcode();

  switch (RetOpcode) {
  case X86::RET:
  case X86::RETI:
  case X86::TCRETURNdi:
  case X86::TCRETURNri:
  case X86::TCRETURNri64:
  case X86::TCRETURNdi64:
  case X86::EH_RETURN:
  case X86::TAILJMPd:
  case X86::TAILJMPr:
  case X86::TAILJMPm: break;  // These are ok
  default:
    assert(0 && "Can only insert epilog into returning blocks");
  }

  // Get the number of bytes to allocate from the FrameInfo
  uint64_t StackSize = MFI->getStackSize();
  unsigned CSSize = X86FI->getCalleeSavedFrameSize();
  uint64_t NumBytes = StackSize - CSSize;

  if (hasFP(MF)) {
    // pop EBP.
    BuildMI(MBB, MBBI, TII.get(Is64Bit ? X86::POP64r : X86::POP32r), FramePtr);
    NumBytes -= SlotSize;
  }

  // Skip the callee-saved pop instructions.
  while (MBBI != MBB.begin()) {
    MachineBasicBlock::iterator PI = prior(MBBI);
    unsigned Opc = PI->getOpcode();
    if (Opc != X86::POP32r && Opc != X86::POP64r &&
        !PI->getDesc().isTerminator())
      break;
    --MBBI;
  }

  // If there is an ADD32ri or SUB32ri of ESP immediately before this
  // instruction, merge the two instructions.
  if (NumBytes || MFI->hasVarSizedObjects())
    mergeSPUpdatesUp(MBB, MBBI, StackPtr, &NumBytes);

  // If dynamic alloca is used, then reset esp to point to the last callee-saved
  // slot before popping them off!  Also, if it's main() on Cygwin/Mingw32 we
  // aligned stack in the prologue, - revert stack changes back. Note: we're
  // assuming, that frame pointer was forced for main()
  if (MFI->hasVarSizedObjects() ||
      (Fn->hasExternalLinkage() && Fn->getName() == "main" &&
       Subtarget->isTargetCygMing())) {
    unsigned Opc = Is64Bit ? X86::LEA64r : X86::LEA32r;
    if (CSSize) {
      MachineInstr *MI = addRegOffset(BuildMI(TII.get(Opc), StackPtr),
                                      FramePtr, -CSSize);
      MBB.insert(MBBI, MI);
    } else
      BuildMI(MBB, MBBI, TII.get(Is64Bit ? X86::MOV64rr : X86::MOV32rr),StackPtr).
        addReg(FramePtr);

    NumBytes = 0;
  }

  // adjust stack pointer back: ESP += numbytes
  if (NumBytes)
    emitSPUpdate(MBB, MBBI, StackPtr, NumBytes, Is64Bit, TII);

  // We're returning from function via eh_return.
  if (RetOpcode == X86::EH_RETURN) {
    MBBI = prior(MBB.end());
    MachineOperand &DestAddr  = MBBI->getOperand(0);
    assert(DestAddr.isRegister() && "Offset should be in register!");
    BuildMI(MBB, MBBI, TII.get(Is64Bit ? X86::MOV64rr : X86::MOV32rr),StackPtr).
      addReg(DestAddr.getReg()); 
  // Tail call return: adjust the stack pointer and jump to callee
  } else if (RetOpcode == X86::TCRETURNri || RetOpcode == X86::TCRETURNdi ||
             RetOpcode== X86::TCRETURNri64 || RetOpcode == X86::TCRETURNdi64) {
    MBBI = prior(MBB.end());
    MachineOperand &JumpTarget = MBBI->getOperand(0);
    MachineOperand &StackAdjust = MBBI->getOperand(1);
    assert( StackAdjust.isImmediate() && "Expecting immediate value.");
    
    // Adjust stack pointer.
    int StackAdj = StackAdjust.getImm();
    int MaxTCDelta = X86FI->getTCReturnAddrDelta();
    int Offset = 0;
    assert(MaxTCDelta <= 0 && "MaxTCDelta should never be positive");
    // Incoporate the retaddr area.
    Offset = StackAdj-MaxTCDelta;
    assert(Offset >= 0 && "Offset should never be negative");
    if (Offset) {
      // Check for possible merge with preceeding ADD instruction.
      Offset += mergeSPUpdates(MBB, MBBI, StackPtr, true);
      emitSPUpdate(MBB, MBBI, StackPtr, Offset, Is64Bit, TII);
    } 
    // Jump to label or value in register.
    if (RetOpcode == X86::TCRETURNdi|| RetOpcode == X86::TCRETURNdi64)
      BuildMI(MBB, MBBI, TII.get(X86::TAILJMPd)).
        addGlobalAddress(JumpTarget.getGlobal(), JumpTarget.getOffset());
    else if (RetOpcode== X86::TCRETURNri64) {
      BuildMI(MBB, MBBI, TII.get(X86::TAILJMPr64), JumpTarget.getReg());
    } else
       BuildMI(MBB, MBBI, TII.get(X86::TAILJMPr), JumpTarget.getReg());
    // Delete the pseudo instruction TCRETURN.
    MBB.erase(MBBI);
  } else if ((RetOpcode == X86::RET || RetOpcode == X86::RETI) && 
             (X86FI->getTCReturnAddrDelta() < 0)) {
    // Add the return addr area delta back since we are not tail calling.
    int delta = -1*X86FI->getTCReturnAddrDelta();
    MBBI = prior(MBB.end());
    // Check for possible merge with preceeding ADD instruction.
    delta += mergeSPUpdates(MBB, MBBI, StackPtr, true);
    emitSPUpdate(MBB, MBBI, StackPtr, delta, Is64Bit, TII);
  }
}

unsigned X86RegisterInfo::getRARegister() const {
  if (Is64Bit)
    return X86::RIP;  // Should have dwarf #16
  else
    return X86::EIP;  // Should have dwarf #8
}

unsigned X86RegisterInfo::getFrameRegister(MachineFunction &MF) const {
  return hasFP(MF) ? FramePtr : StackPtr;
}

void X86RegisterInfo::getInitialFrameState(std::vector<MachineMove> &Moves)
                                                                         const {
  // Calculate amount of bytes used for return address storing
  int stackGrowth = (Is64Bit ? -8 : -4);

  // Initial state of the frame pointer is esp+4.
  MachineLocation Dst(MachineLocation::VirtualFP);
  MachineLocation Src(StackPtr, stackGrowth);
  Moves.push_back(MachineMove(0, Dst, Src));

  // Add return address to move list
  MachineLocation CSDst(StackPtr, stackGrowth);
  MachineLocation CSSrc(getRARegister());
  Moves.push_back(MachineMove(0, CSDst, CSSrc));
}

unsigned X86RegisterInfo::getEHExceptionRegister() const {
  assert(0 && "What is the exception register");
  return 0;
}

unsigned X86RegisterInfo::getEHHandlerRegister() const {
  assert(0 && "What is the exception handler register");
  return 0;
}

namespace llvm {
unsigned getX86SubSuperRegister(unsigned Reg, MVT::ValueType VT, bool High) {
  switch (VT) {
  default: return Reg;
  case MVT::i8:
    if (High) {
      switch (Reg) {
      default: return 0;
      case X86::AH: case X86::AL: case X86::AX: case X86::EAX: case X86::RAX:
        return X86::AH;
      case X86::DH: case X86::DL: case X86::DX: case X86::EDX: case X86::RDX:
        return X86::DH;
      case X86::CH: case X86::CL: case X86::CX: case X86::ECX: case X86::RCX:
        return X86::CH;
      case X86::BH: case X86::BL: case X86::BX: case X86::EBX: case X86::RBX:
        return X86::BH;
      }
    } else {
      switch (Reg) {
      default: return 0;
      case X86::AH: case X86::AL: case X86::AX: case X86::EAX: case X86::RAX:
        return X86::AL;
      case X86::DH: case X86::DL: case X86::DX: case X86::EDX: case X86::RDX:
        return X86::DL;
      case X86::CH: case X86::CL: case X86::CX: case X86::ECX: case X86::RCX:
        return X86::CL;
      case X86::BH: case X86::BL: case X86::BX: case X86::EBX: case X86::RBX:
        return X86::BL;
      case X86::SIL: case X86::SI: case X86::ESI: case X86::RSI:
        return X86::SIL;
      case X86::DIL: case X86::DI: case X86::EDI: case X86::RDI:
        return X86::DIL;
      case X86::BPL: case X86::BP: case X86::EBP: case X86::RBP:
        return X86::BPL;
      case X86::SPL: case X86::SP: case X86::ESP: case X86::RSP:
        return X86::SPL;
      case X86::R8B: case X86::R8W: case X86::R8D: case X86::R8:
        return X86::R8B;
      case X86::R9B: case X86::R9W: case X86::R9D: case X86::R9:
        return X86::R9B;
      case X86::R10B: case X86::R10W: case X86::R10D: case X86::R10:
        return X86::R10B;
      case X86::R11B: case X86::R11W: case X86::R11D: case X86::R11:
        return X86::R11B;
      case X86::R12B: case X86::R12W: case X86::R12D: case X86::R12:
        return X86::R12B;
      case X86::R13B: case X86::R13W: case X86::R13D: case X86::R13:
        return X86::R13B;
      case X86::R14B: case X86::R14W: case X86::R14D: case X86::R14:
        return X86::R14B;
      case X86::R15B: case X86::R15W: case X86::R15D: case X86::R15:
        return X86::R15B;
      }
    }
  case MVT::i16:
    switch (Reg) {
    default: return Reg;
    case X86::AH: case X86::AL: case X86::AX: case X86::EAX: case X86::RAX:
      return X86::AX;
    case X86::DH: case X86::DL: case X86::DX: case X86::EDX: case X86::RDX:
      return X86::DX;
    case X86::CH: case X86::CL: case X86::CX: case X86::ECX: case X86::RCX:
      return X86::CX;
    case X86::BH: case X86::BL: case X86::BX: case X86::EBX: case X86::RBX:
      return X86::BX;
    case X86::SIL: case X86::SI: case X86::ESI: case X86::RSI:
      return X86::SI;
    case X86::DIL: case X86::DI: case X86::EDI: case X86::RDI:
      return X86::DI;
    case X86::BPL: case X86::BP: case X86::EBP: case X86::RBP:
      return X86::BP;
    case X86::SPL: case X86::SP: case X86::ESP: case X86::RSP:
      return X86::SP;
    case X86::R8B: case X86::R8W: case X86::R8D: case X86::R8:
      return X86::R8W;
    case X86::R9B: case X86::R9W: case X86::R9D: case X86::R9:
      return X86::R9W;
    case X86::R10B: case X86::R10W: case X86::R10D: case X86::R10:
      return X86::R10W;
    case X86::R11B: case X86::R11W: case X86::R11D: case X86::R11:
      return X86::R11W;
    case X86::R12B: case X86::R12W: case X86::R12D: case X86::R12:
      return X86::R12W;
    case X86::R13B: case X86::R13W: case X86::R13D: case X86::R13:
      return X86::R13W;
    case X86::R14B: case X86::R14W: case X86::R14D: case X86::R14:
      return X86::R14W;
    case X86::R15B: case X86::R15W: case X86::R15D: case X86::R15:
      return X86::R15W;
    }
  case MVT::i32:
    switch (Reg) {
    default: return Reg;
    case X86::AH: case X86::AL: case X86::AX: case X86::EAX: case X86::RAX:
      return X86::EAX;
    case X86::DH: case X86::DL: case X86::DX: case X86::EDX: case X86::RDX:
      return X86::EDX;
    case X86::CH: case X86::CL: case X86::CX: case X86::ECX: case X86::RCX:
      return X86::ECX;
    case X86::BH: case X86::BL: case X86::BX: case X86::EBX: case X86::RBX:
      return X86::EBX;
    case X86::SIL: case X86::SI: case X86::ESI: case X86::RSI:
      return X86::ESI;
    case X86::DIL: case X86::DI: case X86::EDI: case X86::RDI:
      return X86::EDI;
    case X86::BPL: case X86::BP: case X86::EBP: case X86::RBP:
      return X86::EBP;
    case X86::SPL: case X86::SP: case X86::ESP: case X86::RSP:
      return X86::ESP;
    case X86::R8B: case X86::R8W: case X86::R8D: case X86::R8:
      return X86::R8D;
    case X86::R9B: case X86::R9W: case X86::R9D: case X86::R9:
      return X86::R9D;
    case X86::R10B: case X86::R10W: case X86::R10D: case X86::R10:
      return X86::R10D;
    case X86::R11B: case X86::R11W: case X86::R11D: case X86::R11:
      return X86::R11D;
    case X86::R12B: case X86::R12W: case X86::R12D: case X86::R12:
      return X86::R12D;
    case X86::R13B: case X86::R13W: case X86::R13D: case X86::R13:
      return X86::R13D;
    case X86::R14B: case X86::R14W: case X86::R14D: case X86::R14:
      return X86::R14D;
    case X86::R15B: case X86::R15W: case X86::R15D: case X86::R15:
      return X86::R15D;
    }
  case MVT::i64:
    switch (Reg) {
    default: return Reg;
    case X86::AH: case X86::AL: case X86::AX: case X86::EAX: case X86::RAX:
      return X86::RAX;
    case X86::DH: case X86::DL: case X86::DX: case X86::EDX: case X86::RDX:
      return X86::RDX;
    case X86::CH: case X86::CL: case X86::CX: case X86::ECX: case X86::RCX:
      return X86::RCX;
    case X86::BH: case X86::BL: case X86::BX: case X86::EBX: case X86::RBX:
      return X86::RBX;
    case X86::SIL: case X86::SI: case X86::ESI: case X86::RSI:
      return X86::RSI;
    case X86::DIL: case X86::DI: case X86::EDI: case X86::RDI:
      return X86::RDI;
    case X86::BPL: case X86::BP: case X86::EBP: case X86::RBP:
      return X86::RBP;
    case X86::SPL: case X86::SP: case X86::ESP: case X86::RSP:
      return X86::RSP;
    case X86::R8B: case X86::R8W: case X86::R8D: case X86::R8:
      return X86::R8;
    case X86::R9B: case X86::R9W: case X86::R9D: case X86::R9:
      return X86::R9;
    case X86::R10B: case X86::R10W: case X86::R10D: case X86::R10:
      return X86::R10;
    case X86::R11B: case X86::R11W: case X86::R11D: case X86::R11:
      return X86::R11;
    case X86::R12B: case X86::R12W: case X86::R12D: case X86::R12:
      return X86::R12;
    case X86::R13B: case X86::R13W: case X86::R13D: case X86::R13:
      return X86::R13;
    case X86::R14B: case X86::R14W: case X86::R14D: case X86::R14:
      return X86::R14;
    case X86::R15B: case X86::R15W: case X86::R15D: case X86::R15:
      return X86::R15;
    }
  }

  return Reg;
}
}

#include "X86GenRegisterInfo.inc"

