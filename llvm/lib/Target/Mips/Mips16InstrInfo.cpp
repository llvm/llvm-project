//===-- Mips16InstrInfo.cpp - Mips16 Instruction Information --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Mips16 implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "Mips16InstrInfo.h"
#include "MipsTargetMachine.h"
#include "MipsMachineFunction.h"
#include "InstPrinter/MipsInstPrinter.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

using namespace llvm;

Mips16InstrInfo::Mips16InstrInfo(MipsTargetMachine &tm)
  : MipsInstrInfo(tm, Mips::BimmX16),
    RI(*tm.getSubtargetImpl(), *this) {}

const MipsRegisterInfo &Mips16InstrInfo::getRegisterInfo() const {
  return RI;
}

/// isLoadFromStackSlot - If the specified machine instruction is a direct
/// load from a stack slot, return the virtual or physical register number of
/// the destination along with the FrameIndex of the loaded stack slot.  If
/// not, return 0.  This predicate must return 0 if the instruction has
/// any side effects other than loading from the stack slot.
unsigned Mips16InstrInfo::
isLoadFromStackSlot(const MachineInstr *MI, int &FrameIndex) const
{
  return 0;
}

/// isStoreToStackSlot - If the specified machine instruction is a direct
/// store to a stack slot, return the virtual or physical register number of
/// the source reg along with the FrameIndex of the loaded stack slot.  If
/// not, return 0.  This predicate must return 0 if the instruction has
/// any side effects other than storing to the stack slot.
unsigned Mips16InstrInfo::
isStoreToStackSlot(const MachineInstr *MI, int &FrameIndex) const
{
  return 0;
}

void Mips16InstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                                  MachineBasicBlock::iterator I, DebugLoc DL,
                                  unsigned DestReg, unsigned SrcReg,
                                  bool KillSrc) const {
  unsigned Opc = 0;

  if (Mips::CPU16RegsRegClass.contains(DestReg) &&
      Mips::CPURegsRegClass.contains(SrcReg))
    Opc = Mips::MoveR3216;
  else if (Mips::CPURegsRegClass.contains(DestReg) &&
           Mips::CPU16RegsRegClass.contains(SrcReg))
    Opc = Mips::Move32R16;
  else if ((SrcReg == Mips::HI) &&
           (Mips::CPU16RegsRegClass.contains(DestReg)))
    Opc = Mips::Mfhi16, SrcReg = 0;

  else if ((SrcReg == Mips::LO) &&
           (Mips::CPU16RegsRegClass.contains(DestReg)))
    Opc = Mips::Mflo16, SrcReg = 0;


  assert(Opc && "Cannot copy registers");

  MachineInstrBuilder MIB = BuildMI(MBB, I, DL, get(Opc));

  if (DestReg)
    MIB.addReg(DestReg, RegState::Define);

  if (SrcReg)
    MIB.addReg(SrcReg, getKillRegState(KillSrc));
}

void Mips16InstrInfo::
storeRegToStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                    unsigned SrcReg, bool isKill, int FI,
                    const TargetRegisterClass *RC,
                    const TargetRegisterInfo *TRI) const {
  DebugLoc DL;
  if (I != MBB.end()) DL = I->getDebugLoc();
  MachineMemOperand *MMO = GetMemOperand(MBB, FI, MachineMemOperand::MOStore);
  unsigned Opc = 0;
  if (Mips::CPU16RegsRegClass.hasSubClassEq(RC))
    Opc = Mips::SwRxSpImmX16;
  assert(Opc && "Register class not handled!");
  BuildMI(MBB, I, DL, get(Opc)).addReg(SrcReg, getKillRegState(isKill))
    .addFrameIndex(FI).addImm(0).addMemOperand(MMO);
}

void Mips16InstrInfo::
loadRegFromStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                     unsigned DestReg, int FI,
                     const TargetRegisterClass *RC,
                     const TargetRegisterInfo *TRI) const {
  DebugLoc DL;
  if (I != MBB.end()) DL = I->getDebugLoc();
  MachineMemOperand *MMO = GetMemOperand(MBB, FI, MachineMemOperand::MOLoad);
  unsigned Opc = 0;

  if (Mips::CPU16RegsRegClass.hasSubClassEq(RC))
    Opc = Mips::LwRxSpImmX16;
  assert(Opc && "Register class not handled!");
  BuildMI(MBB, I, DL, get(Opc), DestReg).addFrameIndex(FI).addImm(0)
    .addMemOperand(MMO);
}

bool Mips16InstrInfo::expandPostRAPseudo(MachineBasicBlock::iterator MI) const {
  MachineBasicBlock &MBB = *MI->getParent();

  switch(MI->getDesc().getOpcode()) {
  default:
    return false;
  case Mips::RetRA16:
    ExpandRetRA16(MBB, MI, Mips::JrcRa16);
    break;
  }

  MBB.erase(MI);
  return true;
}

/// GetOppositeBranchOpc - Return the inverse of the specified
/// opcode, e.g. turning BEQ to BNE.
unsigned Mips16InstrInfo::GetOppositeBranchOpc(unsigned Opc) const {
  switch (Opc) {
  default:  llvm_unreachable("Illegal opcode!");
  case Mips::BeqzRxImmX16: return Mips::BnezRxImmX16;
  case Mips::BnezRxImmX16: return Mips::BeqzRxImmX16;
  case Mips::BteqzT8CmpX16: return Mips::BtnezT8CmpX16;
  case Mips::BteqzT8SltX16: return Mips::BtnezT8SltX16;
  case Mips::BteqzT8SltiX16: return Mips::BtnezT8SltiX16;
  case Mips::BtnezX16: return Mips::BteqzX16;
  case Mips::BtnezT8CmpiX16: return Mips::BteqzT8CmpiX16;
  case Mips::BtnezT8SltuX16: return Mips::BteqzT8SltuX16;
  case Mips::BtnezT8SltiuX16: return Mips::BteqzT8SltiuX16;
  case Mips::BteqzX16: return Mips::BtnezX16;
  case Mips::BteqzT8CmpiX16: return Mips::BtnezT8CmpiX16;
  case Mips::BteqzT8SltuX16: return Mips::BtnezT8SltuX16;
  case Mips::BteqzT8SltiuX16: return Mips::BtnezT8SltiuX16;
  case Mips::BtnezT8CmpX16: return Mips::BteqzT8CmpX16;
  case Mips::BtnezT8SltX16: return Mips::BteqzT8SltX16;
  case Mips::BtnezT8SltiX16: return Mips::BteqzT8SltiX16;
  }
  assert(false && "Implement this function.");
  return 0;
}

/// Adjust SP by Amount bytes.
void Mips16InstrInfo::adjustStackPtr(unsigned SP, int64_t Amount,
                                     MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator I) const {
  DebugLoc DL = I != MBB.end() ? I->getDebugLoc() : DebugLoc();
  if (isInt<16>(Amount)) {
    if (Amount < 0)
      BuildMI(MBB, I, DL, get(Mips::SaveDecSpF16)). addImm(-Amount);
    else if (Amount > 0)
      BuildMI(MBB, I, DL, get(Mips::RestoreIncSpF16)).addImm(Amount);
  }
  else
    // not implemented for large values yet
    assert(false && "adjust stack pointer amount exceeded");
}

unsigned Mips16InstrInfo::GetAnalyzableBrOpc(unsigned Opc) const {
  return (Opc == Mips::BeqzRxImmX16   || Opc == Mips::BimmX16  ||
          Opc == Mips::BnezRxImmX16   || Opc == Mips::BteqzX16 ||
          Opc == Mips::BteqzT8CmpX16  || Opc == Mips::BteqzT8CmpiX16 ||
          Opc == Mips::BteqzT8SltX16  || Opc == Mips::BteqzT8SltuX16  ||
          Opc == Mips::BteqzT8SltiX16 || Opc == Mips::BteqzT8SltiuX16 ||
          Opc == Mips::BtnezX16       || Opc == Mips::BtnezT8CmpX16 ||
          Opc == Mips::BtnezT8CmpiX16 || Opc == Mips::BtnezT8SltX16 ||
          Opc == Mips::BtnezT8SltuX16 || Opc == Mips::BtnezT8SltiX16 ||
          Opc == Mips::BtnezT8SltiuX16 ) ? Opc : 0;
}

void Mips16InstrInfo::ExpandRetRA16(MachineBasicBlock &MBB,
                                  MachineBasicBlock::iterator I,
                                  unsigned Opc) const {
  BuildMI(MBB, I, I->getDebugLoc(), get(Opc));
}

const MipsInstrInfo *llvm::createMips16InstrInfo(MipsTargetMachine &TM) {
  return new Mips16InstrInfo(TM);
}
