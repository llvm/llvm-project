//===-- Toy64RegisterInfo.cpp - Toy64 Register Information ----------------===//

#include "Toy64RegisterInfo.h"
#include "Toy64FrameLowering.h"
#include "MCTargetDesc/Toy64MCTargetDesc.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/Support/ErrorHandling.h"

#define GET_REGINFO_TARGET_DESC
#include "Toy64GenRegisterInfo.inc"

using namespace llvm;

Toy64RegisterInfo::Toy64RegisterInfo()
    : Toy64GenRegisterInfo(Toy64::X1) {}

// We have not defined the Toy64 calling convention yet.
//
// Therefore no registers are considered callee-saved at this stage.
// A register-save list is terminated by register 0.
const uint16_t *
Toy64RegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  static const uint16_t CalleeSavedRegs[] = {0};
  return CalleeSavedRegs;
}

BitVector
Toy64RegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());

  // x0 is hard-wired to zero.
  Reserved.set(Toy64::X0);

  // x1 is currently treated as the return-address register.
  Reserved.set(Toy64::X1);

  // x2 is the stack pointer.
  Reserved.set(Toy64::X2);

  // Reserve x8 as our future frame pointer during initial bring-up.
  Reserved.set(Toy64::X8);

  return Reserved;
}

bool Toy64RegisterInfo::eliminateFrameIndex(
    MachineBasicBlock::iterator II,
    int SPAdj,
    unsigned FIOperandNum,
    RegScavenger *RS) const {
  llvm_unreachable(
      "Toy64 frame-index elimination is not implemented yet");
}

Register
Toy64RegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  // Until frame-pointer support is implemented, use SP.
  return Toy64::X2;
}