//===- MipsRegisterInfo.cpp - MIPS Register Information -== -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the MIPS implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "mips-reg-info"

#include "Mips.h"
#include "MipsSubtarget.h"
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

using namespace llvm;

MipsRegisterInfo::MipsRegisterInfo(const MipsSubtarget &ST, 
                                   const TargetInstrInfo &tii)
  : MipsGenRegisterInfo(Mips::ADJCALLSTACKDOWN, Mips::ADJCALLSTACKUP),
    Subtarget(ST), TII(tii) {}

/// getRegisterNumbering - Given the enum value for some register, e.g.
/// Mips::RA, return the number that it corresponds to (e.g. 31).
unsigned MipsRegisterInfo::
getRegisterNumbering(unsigned RegEnum) 
{
  switch (RegEnum) {
    case Mips::ZERO : case Mips::F0 : case Mips::D0 : return 0;
    case Mips::AT   : case Mips::F1 : return 1;
    case Mips::V0   : case Mips::F2 : case Mips::D1 : return 2;
    case Mips::V1   : case Mips::F3 : return 3;
    case Mips::A0   : case Mips::F4 : case Mips::D2 : return 4;
    case Mips::A1   : case Mips::F5 : return 5;
    case Mips::A2   : case Mips::F6 : case Mips::D3 : return 6;
    case Mips::A3   : case Mips::F7 : return 7;
    case Mips::T0   : case Mips::F8 : case Mips::D4 : return 8;
    case Mips::T1   : case Mips::F9 : return 9;
    case Mips::T2   : case Mips::F10: case Mips::D5: return 10;
    case Mips::T3   : case Mips::F11: return 11;
    case Mips::T4   : case Mips::F12: case Mips::D6: return 12;
    case Mips::T5   : case Mips::F13: return 13;
    case Mips::T6   : case Mips::F14: case Mips::D7: return 14;
    case Mips::T7   : case Mips::F15: return 15;
    case Mips::T8   : case Mips::F16: case Mips::D8: return 16;
    case Mips::T9   : case Mips::F17: return 17;
    case Mips::S0   : case Mips::F18: case Mips::D9: return 18;
    case Mips::S1   : case Mips::F19: return 19;
    case Mips::S2   : case Mips::F20: case Mips::D10: return 20;
    case Mips::S3   : case Mips::F21: return 21;
    case Mips::S4   : case Mips::F22: case Mips::D11: return 22;
    case Mips::S5   : case Mips::F23: return 23;
    case Mips::S6   : case Mips::F24: case Mips::D12: return 24;
    case Mips::S7   : case Mips::F25: return 25;
    case Mips::K0   : case Mips::F26: case Mips::D13: return 26;
    case Mips::K1   : case Mips::F27: return 27;
    case Mips::GP   : case Mips::F28: case Mips::D14: return 28;
    case Mips::SP   : case Mips::F29: return 29;
    case Mips::FP   : case Mips::F30: case Mips::D15: return 30;
    case Mips::RA   : case Mips::F31: return 31;
    default: assert(0 && "Unknown register number!");
  }    
  return 0; // Not reached
}

unsigned MipsRegisterInfo::getPICCallReg(void) { return Mips::T9; }

//===----------------------------------------------------------------------===//
// Callee Saved Registers methods 
//===----------------------------------------------------------------------===//

/// Mips Callee Saved Registers
const unsigned* MipsRegisterInfo::
getCalleeSavedRegs(const MachineFunction *MF) const 
{
  // Mips callee-save register range is $16-$23, $f20-$f30
  static const unsigned SingleFloatOnlyCalleeSavedRegs[] = {
    Mips::S0, Mips::S1, Mips::S2, Mips::S3, 
    Mips::S4, Mips::S5, Mips::S6, Mips::S7,
    Mips::F20, Mips::F21, Mips::F22, Mips::F23, Mips::F24, Mips::F25, 
    Mips::F26, Mips::F27, Mips::F28, Mips::F29, Mips::F30, 0
  };

  static const unsigned BitMode32CalleeSavedRegs[] = {
    Mips::S0, Mips::S1, Mips::S2, Mips::S3, 
    Mips::S4, Mips::S5, Mips::S6, Mips::S7,
    Mips::F20, Mips::F22, Mips::F24, Mips::F26, Mips::F28, Mips::F30, 
    Mips::D10, Mips::D11, Mips::D12, Mips::D13, Mips::D14, Mips::D15,0
  };

  if (Subtarget.isSingleFloat())
    return SingleFloatOnlyCalleeSavedRegs;
  else
    return BitMode32CalleeSavedRegs;
}

/// Mips Callee Saved Register Classes
const TargetRegisterClass* const* 
MipsRegisterInfo::getCalleeSavedRegClasses(const MachineFunction *MF) const 
{
  static const TargetRegisterClass * const SingleFloatOnlyCalleeSavedRC[] = {
    &Mips::CPURegsRegClass, &Mips::CPURegsRegClass, &Mips::CPURegsRegClass, 
    &Mips::CPURegsRegClass, &Mips::CPURegsRegClass, &Mips::CPURegsRegClass,
    &Mips::CPURegsRegClass, &Mips::CPURegsRegClass,
    &Mips::FGR32RegClass, &Mips::FGR32RegClass, &Mips::FGR32RegClass, 
    &Mips::FGR32RegClass, &Mips::FGR32RegClass, &Mips::FGR32RegClass, 
    &Mips::FGR32RegClass, &Mips::FGR32RegClass, &Mips::FGR32RegClass, 
    &Mips::FGR32RegClass, &Mips::FGR32RegClass, 0
  };

  static const TargetRegisterClass * const BitMode32CalleeSavedRC[] = {
    &Mips::CPURegsRegClass, &Mips::CPURegsRegClass, &Mips::CPURegsRegClass, 
    &Mips::CPURegsRegClass, &Mips::CPURegsRegClass, &Mips::CPURegsRegClass,
    &Mips::CPURegsRegClass, &Mips::CPURegsRegClass,
    &Mips::AFGR32RegClass, &Mips::AFGR32RegClass, &Mips::AFGR32RegClass, 
    &Mips::AFGR32RegClass, &Mips::AFGR32RegClass, &Mips::AFGR32RegClass,
    &Mips::AFGR64RegClass, &Mips::AFGR64RegClass, &Mips::AFGR64RegClass, 
    &Mips::AFGR64RegClass, &Mips::AFGR64RegClass, &Mips::AFGR64RegClass, 0
  };

  if (Subtarget.isSingleFloat())
    return SingleFloatOnlyCalleeSavedRC;
  else
    return BitMode32CalleeSavedRC;
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
//  The stack frame required by the ABI (after call):
//  Offset
//
//  0                 ----------
//  4                 Args to pass 
//  .                 saved $GP  (used in PIC)
//  .                 Alloca allocations
//  .                 Local Area
//  .                 CPU "Callee Saved" Registers
//  .                 saved FP
//  .                 saved RA
//  .                 FPU "Callee Saved" Registers
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
//===----------------------------------------------------------------------===//

void MipsRegisterInfo::adjustMipsStackFrame(MachineFunction &MF) const
{
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MipsFunctionInfo *MipsFI = MF.getInfo<MipsFunctionInfo>();
  const std::vector<CalleeSavedInfo> &CSI = MFI->getCalleeSavedInfo();
  unsigned StackAlign = MF.getTarget().getFrameInfo()->getStackAlignment();

  // Min and Max CSI FrameIndex.
  int MinCSFI = -1, MaxCSFI = -1; 

  // See the description at MipsMachineFunction.h
  int TopCPUSavedRegOff = -1, TopFPUSavedRegOff = -1;

  // Replace the dummy '0' SPOffset by the negative offsets, as explained on 
  // LowerFORMAL_ARGUMENTS. Leaving '0' for while is necessary to avoid 
  // the approach done by calculateFrameObjectOffsets to the stack frame.
  MipsFI->adjustLoadArgsFI(MFI);
  MipsFI->adjustStoreVarArgsFI(MFI); 

  // It happens that the default stack frame allocation order does not directly 
  // map to the convention used for mips. So we must fix it. We move the callee 
  // save register slots after the local variables area, as described in the
  // stack frame above.
  unsigned CalleeSavedAreaSize = 0;
  if (!CSI.empty()) {
    MinCSFI = CSI[0].getFrameIdx();
    MaxCSFI = CSI[CSI.size()-1].getFrameIdx();
  }
  for (unsigned i = 0, e = CSI.size(); i != e; ++i)
    CalleeSavedAreaSize += MFI->getObjectAlignment(CSI[i].getFrameIdx());

  // Adjust local variables. They should come on the stack right
  // after the arguments.
  int LastOffsetFI = -1;
  for (int i = 0, e = MFI->getObjectIndexEnd(); i != e; ++i) {
    if (i >= MinCSFI && i <= MaxCSFI)
      continue;
    if (MFI->isDeadObjectIndex(i))
      continue;
    unsigned Offset = MFI->getObjectOffset(i) - CalleeSavedAreaSize;
    if (LastOffsetFI == -1)
      LastOffsetFI = i;
    if (Offset > MFI->getObjectOffset(LastOffsetFI))
      LastOffsetFI = i;
    MFI->setObjectOffset(i, Offset);
  }

  // Adjust CPU Callee Saved Registers Area. Registers RA and FP must
  // be saved in this CPU Area there is the need. This whole Area must 
  // be aligned to the default Stack Alignment requirements.
  unsigned StackOffset = 0;
  unsigned RegSize = Subtarget.isGP32bit() ? 4 : 8;

  if (LastOffsetFI >= 0)
    StackOffset = MFI->getObjectOffset(LastOffsetFI)+ 
                  MFI->getObjectAlignment(LastOffsetFI);
  StackOffset = ((StackOffset+StackAlign-1)/StackAlign*StackAlign);

  for (unsigned i = 0, e = CSI.size(); i != e ; ++i) {
    if (CSI[i].getRegClass() != Mips::CPURegsRegisterClass)
      break;
    MFI->setObjectOffset(CSI[i].getFrameIdx(), StackOffset);
    TopCPUSavedRegOff = StackOffset;
    StackOffset += MFI->getObjectAlignment(CSI[i].getFrameIdx());
  }

  if (hasFP(MF)) {
    MFI->setObjectOffset(MFI->CreateStackObject(RegSize, RegSize), 
                         StackOffset);
    MipsFI->setFPStackOffset(StackOffset);
    TopCPUSavedRegOff = StackOffset;
    StackOffset += RegSize;
  }

  if (MFI->hasCalls()) {
    MFI->setObjectOffset(MFI->CreateStackObject(RegSize, RegSize), 
                         StackOffset);
    MipsFI->setRAStackOffset(StackOffset);
    TopCPUSavedRegOff = StackOffset;
    StackOffset += RegSize;
  }
  StackOffset = ((StackOffset+StackAlign-1)/StackAlign*StackAlign);
  
  // Adjust FPU Callee Saved Registers Area. This Area must be 
  // aligned to the default Stack Alignment requirements.
  for (unsigned i = 0, e = CSI.size(); i != e; ++i) {
    if (CSI[i].getRegClass() == Mips::CPURegsRegisterClass)
      continue;
    MFI->setObjectOffset(CSI[i].getFrameIdx(), StackOffset);
    TopFPUSavedRegOff = StackOffset;
    StackOffset += MFI->getObjectAlignment(CSI[i].getFrameIdx());
  }
  StackOffset = ((StackOffset+StackAlign-1)/StackAlign*StackAlign);

  // Update frame info
  MFI->setStackSize(StackOffset);

  // Recalculate the final tops offset. The final values must be '0'
  // if there isn't a callee saved register for CPU or FPU, otherwise
  // a negative offset is needed.
  if (TopCPUSavedRegOff >= 0)
    MipsFI->setCPUTopSavedRegOff(TopCPUSavedRegOff-StackOffset);

  if (TopFPUSavedRegOff >= 0)
    MipsFI->setFPUTopSavedRegOff(TopFPUSavedRegOff-StackOffset);
}

// hasFP - Return true if the specified function should have a dedicated frame
// pointer register.  This is true if the function has variable sized allocas or
// if frame pointer elimination is disabled.
bool MipsRegisterInfo::
hasFP(const MachineFunction &MF) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  return NoFramePointerElim || MFI->hasVarSizedObjects();
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
  while (!MI.getOperand(i).isFI()) {
    ++i;
    assert(i < MI.getNumOperands() && 
           "Instr doesn't have FrameIndex operand!");
  }

  #ifndef NDEBUG
  DOUT << "\nFunction : " << MF.getFunction()->getName() << "\n";
  DOUT << "<--------->\n";
  MI.print(DOUT);
  #endif

  int FrameIndex = MI.getOperand(i).getIndex();
  int stackSize  = MF.getFrameInfo()->getStackSize();
  int spOffset   = MF.getFrameInfo()->getObjectOffset(FrameIndex);

  #ifndef NDEBUG
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

  // Get the right frame order for Mips.
  adjustMipsStackFrame(MF);

  // Get the number of bytes to allocate from the FrameInfo.
  unsigned StackSize = MFI->getStackSize();

  // No need to allocate space on the stack.
  if (StackSize == 0 && !MFI->hasCalls()) return;

  int FPOffset = MipsFI->getFPStackOffset();
  int RAOffset = MipsFI->getRAStackOffset();

  BuildMI(MBB, MBBI, TII.get(Mips::NOREORDER));
  
  // TODO: check need from GP here.
  if (isPIC && Subtarget.isABI_O32()) 
    BuildMI(MBB, MBBI, TII.get(Mips::CPLOAD)).addReg(getPICCallReg());
  BuildMI(MBB, MBBI, TII.get(Mips::NOMACRO));

  // Adjust stack : addi sp, sp, (-imm)
  BuildMI(MBB, MBBI, TII.get(Mips::ADDiu), Mips::SP)
      .addReg(Mips::SP).addImm(-StackSize);

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
  if ((isPIC) && (MFI->hasCalls())) {
    BuildMI(MBB, MBBI, TII.get(Mips::CPRESTORE))
      .addImm(MipsFI->getGPStackOffset());
  }
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
  bool isPIC = (MF.getTarget().getRelocationModel() == Reloc::PIC_);
  if (MFI->hasCalls() && isPIC) { 
    MipsFunctionInfo *MipsFI = MF.getInfo<MipsFunctionInfo>();
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

