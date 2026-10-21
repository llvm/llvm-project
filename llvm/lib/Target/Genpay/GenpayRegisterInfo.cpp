#include "GenpayRegisterInfo.h"
#include "GenpaySubtarget.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/ADT/BitVector.h"

using namespace llvm;

// Definitions of the tablegen-generated base class ctor and of the
// TargetRegisterClass objects (Genpay::GPRRegClass, ...).
#define GET_REGINFO_TARGET_DESC
#include "GenpayGenRegisterInfo.inc"

GenpayRegisterInfo::GenpayRegisterInfo() : GenpayGenRegisterInfo(Genpay::SP) {}

const MCPhysReg *
GenpayRegisterInfo::getCalleeSavedRegs(const MachineFunction *) const {
  static const MCPhysReg CalleeSavedRegs[] = {0};
  return CalleeSavedRegs;
}

BitVector
GenpayRegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());
  Reserved.set(Genpay::SP);
  return Reserved;
}

bool GenpayRegisterInfo::eliminateFrameIndex(
    MachineBasicBlock::iterator II, int SPAdj, unsigned FIOperandNum,
    RegScavenger *) const {
  llvm_unreachable("Genpay: frame indices are not supported yet");
  return false;
}

Register GenpayRegisterInfo::getFrameRegister(const MachineFunction &) const {
  return Genpay::SP;
}
