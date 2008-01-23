//===- MipsRegisterInfo.cpp - MIPS Register Information -== -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the MIPS implementation of the MRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "mips-reg-info"

#include "Mips.h"
#include "MipsRegisterInfo.h"
#include "MipsMachineFunction.h"
#include "llvm/Constants.h"
#include "llvm/Type.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineLocation.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
//#include "MipsSubtarget.h"

using namespace llvm;

// TODO: add subtarget support
MipsRegisterInfo::MipsRegisterInfo(const TargetInstrInfo &tii)
  : MipsGenRegisterInfo(Mips::ADJCALLSTACKDOWN, Mips::ADJCALLSTACKUP),
  TII(tii) {}

/// getRegisterNumbering - Given the enum value for some register, e.g.
/// Mips::RA, return the number that it corresponds to (e.g. 31).
unsigned MipsRegisterInfo::
getRegisterNumbering(unsigned RegEnum) 
{
  switch (RegEnum) {
    case Mips::ZERO : return 0;
    case Mips::AT   : return 1;
    case Mips::V0   : return 2;
    case Mips::V1   : return 3;
    case Mips::A0   : return 4;
    case Mips::A1   : return 5;
    case Mips::A2   : return 6;
    case Mips::A3   : return 7;
    case Mips::T0   : return 8;
    case Mips::T1   : return 9;
    case Mips::T2   : return 10;
    case Mips::T3   : return 11;
    case Mips::T4   : return 12;
    case Mips::T5   : return 13;
    case Mips::T6   : return 14;
    case Mips::T7   : return 15;
    case Mips::T8   : return 16;
    case Mips::T9   : return 17;
    case Mips::S0   : return 18;
    case Mips::S1   : return 19;
    case Mips::S2   : return 20;
    case Mips::S3   : return 21;
    case Mips::S4   : return 22;
    case Mips::S5   : return 23;
    case Mips::S6   : return 24;
    case Mips::S7   : return 25;
    case Mips::K0   : return 26;
    case Mips::K1   : return 27;
    case Mips::GP   : return 28;
    case Mips::SP   : return 29;
    case Mips::FP   : return 30;
    case Mips::RA   : return 31;
    default: assert(0 && "Unknown register number!");
  }    
}

void MipsRegisterInfo::reMaterialize(MachineBasicBlock &MBB, 
                                      MachineBasicBlock::iterator I,
                                      unsigned DestReg, 
                                      const MachineInstr *Orig) const 
{
    MachineInstr *MI = Orig->clone();
    MI->getOperand(0).setReg(DestReg);
    MBB.insert(I, MI);
}

//===----------------------------------------------------------------------===//
//
// Callee Saved Registers methods 
//
//===----------------------------------------------------------------------===//

/// Mips Callee Saved Registers
const unsigned* MipsRegisterInfo::
getCalleeSavedRegs(const MachineFunction *MF) const 
{
  // Mips calle-save register range is $16-$26(s0-s7)
  static const unsigned CalleeSavedRegs[] = {  
    Mips::S0, Mips::S1, Mips::S2, Mips::S3, 
    Mips::S4, Mips::S5, Mips::S6, Mips::S7, 0
  };
  return CalleeSavedRegs;
}

/// Mips Callee Saved Register Classes
const TargetRegisterClass* const* 
MipsRegisterInfo::getCalleeSavedRegClasses(const MachineFunction *MF) const 
{
  static const TargetRegisterClass * const CalleeSavedRegClasses[] = {
    &Mips::CPURegsRegClass, &Mips::CPURegsRegClass,
    &Mips::CPURegsRegClass, &Mips::CPURegsRegClass,
    &Mips::CPURegsRegClass, &Mips::CPURegsRegClass,
    &Mips::CPURegsRegClass, &Mips::CPURegsRegClass, 0 
  };
  return CalleeSavedRegClasses;
}

BitVector MipsRegisterInfo::
getReservedRegs(const MachineFunction &MF) const
{
  BitVector Reserved(getNumRegs());
  Reserved.set(Mips::ZERO);
  Reserved.set(Mips::AT);
  Reserved.set(Mips::K0);
  Reserved.set(Mips::K1);
  Reserved.set(Mips::GP);
  Reserved.set(Mips::SP);
  Reserved.set(Mips::FP);
  Reserved.set(Mips::RA);
  return Reserved;
}

//===----------------------------------------------------------------------===//
//
// Stack Frame Processing methods
// +----------------------------+
//
// The stack is allocated decrementing the stack pointer on
// the first instruction of a function prologue. Once decremented,
// all stack referencesare are done thought a positive offset
// from the stack/frame pointer, so the stack is considering
// to grow up! Otherwise terrible hacks would have to be made
// to get this stack ABI compliant :)
//
//  The stack frame required by the ABI:
//  Offset
//
//  0                 ----------
//  4                 Args to pass
//  .                 saved $GP  (used in PIC - not supported yet)
//  .                 Local Area
//  .                 saved "Callee Saved" Registers
//  .                 saved FP
//  .                 saved RA
//  StackSize         -----------
//
// Offset - offset from sp after stack allocation on function prologue
//
// The sp is the stack pointer subtracted/added from the stack size
// at the Prologue/Epilogue
//
// References to the previous stack (to obtain arguments) are done
// with offsets that exceeds the stack size: (stacksize+(4*(num_arg-1))
//
// Examples:
// - reference to the actual stack frame
//   for any local area var there is smt like : FI >= 0, StackOffset: 4
//     sw REGX, 4(SP)
//
// - reference to previous stack frame
//   suppose there's a load to the 5th arguments : FI < 0, StackOffset: 16.
//   The emitted instruction will be something like:
//     lw REGX, 16+StackSize(SP)
//
// Since the total stack size is unknown on LowerFORMAL_ARGUMENTS, all
// stack references (ObjectOffset) created to reference the function 
// arguments, are negative numbers. This way, on eliminateFrameIndex it's
// possible to detect those references and the offsets are adjusted to
// their real location.
//
//
//
//===----------------------------------------------------------------------===//

// hasFP - Return true if the specified function should have a dedicated frame
// pointer register.  This is true if the function has variable sized allocas or
// if frame pointer elimination is disabled.
bool MipsRegisterInfo::
hasFP(const MachineFunction &MF) const {
  return (NoFramePointerElim || MF.getFrameInfo()->hasVarSizedObjects());
}

// This function eliminate ADJCALLSTACKDOWN, 
// ADJCALLSTACKUP pseudo instructions
void MipsRegisterInfo::
eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I) const {
  // Simply discard ADJCALLSTACKDOWN, ADJCALLSTACKUP instructions.
  MBB.erase(I);
}

// FrameIndex represent objects inside a abstract stack.
// We must replace FrameIndex with an stack/frame pointer
// direct reference.
void MipsRegisterInfo::
eliminateFrameIndex(MachineBasicBlock::iterator II, int SPAdj, 
                    RegScavenger *RS) const 
{
  MachineInstr &MI = *II;
  MachineFunction &MF = *MI.getParent()->getParent();

  unsigned i = 0;
  while (!MI.getOperand(i).isFrameIndex()) {
    ++i;
    assert(i < MI.getNumOperands() && 
           "Instr doesn't have FrameIndex operand!");
  }

  int FrameIndex = MI.getOperand(i).getIndex();
  int stackSize  = MF.getFrameInfo()->getStackSize();
  int spOffset   = MF.getFrameInfo()->getObjectOffset(FrameIndex);

  #ifndef NDEBUG
  DOUT << "\nFunction : " << MF.getFunction()->getName() << "\n";
  DOUT << "<--------->\n";
  MI.print(DOUT);
  DOUT << "FrameIndex : " << FrameIndex << "\n";
  DOUT << "spOffset   : " << spOffset << "\n";
  DOUT << "stackSize  : " << stackSize << "\n";
  #endif

  // as explained on LowerFORMAL_ARGUMENTS, detect negative offsets 
  // and adjust SPOffsets considering the final stack size.
  int Offset = ((spOffset < 0) ? (stackSize + (-(spOffset+4))) : (spOffset));
  Offset    += MI.getOperand(i-1).getImm();

  #ifndef NDEBUG
  DOUT << "Offset     : " << Offset << "\n";
  DOUT << "<--------->\n";
  #endif

  MI.getOperand(i-1).ChangeToImmediate(Offset);
  MI.getOperand(i).ChangeToRegister(getFrameRegister(MF), false);
}

void MipsRegisterInfo::
emitPrologue(MachineFunction &MF) const 
{
  MachineBasicBlock &MBB   = MF.front();
  MachineFrameInfo *MFI    = MF.getFrameInfo();
  MipsFunctionInfo *MipsFI = MF.getInfo<MipsFunctionInfo>();
  MachineBasicBlock::iterator MBBI = MBB.begin();
  bool isPIC = (MF.getTarget().getRelocationModel() == Reloc::PIC_);

  // Replace the dummy '0' SPOffset by the negative 
  // offsets, as explained on LowerFORMAL_ARGUMENTS
  MipsFI->adjustLoadArgsFI(MFI);
  MipsFI->adjustStoreVarArgsFI(MFI); 

  // Get the number of bytes to allocate from the FrameInfo.
  int NumBytes = (int) MFI->getStackSize();

  #ifndef NDEBUG
  DOUT << "\n<--- EMIT PROLOGUE --->\n";
  DOUT << "Actual Stack size :" << NumBytes << "\n";
  #endif

  // No need to allocate space on the stack.
  if (NumBytes == 0) return;

  int FPOffset, RAOffset;
  
  // Allocate space for saved RA and FP when needed 
  if ((hasFP(MF)) && (MFI->hasCalls())) {
    FPOffset = NumBytes;
    RAOffset = (NumBytes+4);
    NumBytes += 8;
  } else if ((!hasFP(MF)) && (MFI->hasCalls())) {
    FPOffset = 0;
    RAOffset = NumBytes;
    NumBytes += 4;
  } else if ((hasFP(MF)) && (!MFI->hasCalls())) {
    FPOffset = NumBytes;
    RAOffset = 0;
    NumBytes += 4;
  } else {
    // No calls and no fp.
    RAOffset = FPOffset = 0;
  }

  MFI->setObjectOffset(MFI->CreateStackObject(4,4), FPOffset);
  MFI->setObjectOffset(MFI->CreateStackObject(4,4), RAOffset);
  MipsFI->setFPStackOffset(FPOffset);
  MipsFI->setRAStackOffset(RAOffset);

  // Align stack. 
  unsigned Align = MF.getTarget().getFrameInfo()->getStackAlignment();
  NumBytes = ((NumBytes+Align-1)/Align*Align);

  #ifndef NDEBUG
  DOUT << "FPOffset :" << FPOffset << "\n";
  DOUT << "RAOffset :" << RAOffset << "\n";
  DOUT << "New stack size :" << NumBytes << "\n\n";
  #endif

  // Update frame info
  MFI->setStackSize(NumBytes);

  // PIC speficic function prologue
  if (isPIC)
    BuildMI(MBB, MBBI, TII.get(Mips::CPLOAD)).addReg(Mips::T9);

  // Adjust stack : addi sp, sp, (-imm)
  BuildMI(MBB, MBBI, TII.get(Mips::ADDiu), Mips::SP)
      .addReg(Mips::SP).addImm(-NumBytes);

  // Save the return address only if the function isnt a leaf one.
  // sw  $ra, stack_loc($sp)
  if (MFI->hasCalls()) { 
    BuildMI(MBB, MBBI, TII.get(Mips::SW))
        .addReg(Mips::RA).addImm(RAOffset).addReg(Mips::SP);
  }

  // if framepointer enabled, save it and set it
  // to point to the stack pointer
  if (hasFP(MF)) {
    // sw  $fp,stack_loc($sp)
    BuildMI(MBB, MBBI, TII.get(Mips::SW))
      .addReg(Mips::FP).addImm(FPOffset).addReg(Mips::SP);

    // move $fp, $sp
    BuildMI(MBB, MBBI, TII.get(Mips::ADDu), Mips::FP)
      .addReg(Mips::SP).addReg(Mips::ZERO);
  }

  // PIC speficic function prologue
  if ((isPIC) && (MFI->hasCalls()))
    BuildMI(MBB, MBBI, TII.get(Mips::CPRESTORE))
      .addImm(MipsFI->getGPStackOffset());
}

void MipsRegisterInfo::
emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const 
{
  MachineBasicBlock::iterator MBBI = prior(MBB.end());
  MachineFrameInfo *MFI            = MF.getFrameInfo();
  MipsFunctionInfo *MipsFI         = MF.getInfo<MipsFunctionInfo>();

  // Get the number of bytes from FrameInfo
  int NumBytes = (int) MFI->getStackSize();

  // Get the FI's where RA and FP are saved.
  int FPOffset = MipsFI->getFPStackOffset();
  int RAOffset = MipsFI->getRAStackOffset();

  // if framepointer enabled, restore it and restore the
  // stack pointer
  if (hasFP(MF)) {
    // move $sp, $fp
    BuildMI(MBB, MBBI, TII.get(Mips::ADDu), Mips::SP)
      .addReg(Mips::FP).addReg(Mips::ZERO);

    // lw  $fp,stack_loc($sp)
    BuildMI(MBB, MBBI, TII.get(Mips::LW))
      .addReg(Mips::FP).addImm(FPOffset).addReg(Mips::SP);
  }

  // Restore the return address only if the function isnt a leaf one.
  // lw  $ra, stack_loc($sp)
  if (MFI->hasCalls()) { 
    BuildMI(MBB, MBBI, TII.get(Mips::LW))
        .addReg(Mips::RA).addImm(RAOffset).addReg(Mips::SP);
  }

  // adjust stack  : insert addi sp, sp, (imm)
  if (NumBytes) {
    BuildMI(MBB, MBBI, TII.get(Mips::ADDiu), Mips::SP)
      .addReg(Mips::SP).addImm(NumBytes);
  }
}

void MipsRegisterInfo::
processFunctionBeforeFrameFinalized(MachineFunction &MF) const {
  // Set the SPOffset on the FI where GP must be saved/loaded.
  MachineFrameInfo *MFI = MF.getFrameInfo();
  if (MFI->hasCalls()) { 
    MipsFunctionInfo *MipsFI = MF.getInfo<MipsFunctionInfo>();
    #ifndef NDEBUG
    DOUT << "processFunctionBeforeFrameFinalized\n";
    DOUT << "GPOffset :" << MipsFI->getGPStackOffset() << "\n";
    DOUT << "FI :" << MipsFI->getGPFI() << "\n";
    #endif
    MFI->setObjectOffset(MipsFI->getGPFI(), MipsFI->getGPStackOffset());
  }    
}

unsigned MipsRegisterInfo::
getRARegister() const {
  return Mips::RA;
}

unsigned MipsRegisterInfo::
getFrameRegister(MachineFunction &MF) const {
  return hasFP(MF) ? Mips::FP : Mips::SP;
}

unsigned MipsRegisterInfo::
getEHExceptionRegister() const {
  assert(0 && "What is the exception register");
  return 0;
}

unsigned MipsRegisterInfo::
getEHHandlerRegister() const {
  assert(0 && "What is the exception handler register");
  return 0;
}

int MipsRegisterInfo::
getDwarfRegNum(unsigned RegNum, bool isEH) const {
  assert(0 && "What is the dwarf register number");
  return -1;
}

#include "MipsGenRegisterInfo.inc"

