//===-- Toy64RegisterInfo.cpp - Toy64 Register Information ----------------===//

#include "Toy64RegisterInfo.h"
#include "MCTargetDesc/Toy64MCTargetDesc.h"
#include "Toy64FrameLowering.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"

#define GET_REGINFO_TARGET_DESC
#include "Toy64GenRegisterInfo.inc"

using namespace llvm;

Toy64RegisterInfo::Toy64RegisterInfo() : Toy64GenRegisterInfo(Toy64::X1) {}

// We have not defined the Toy64 calling convention yet.
//
// Therefore no registers are considered callee-saved at this stage.
// A register-save list is terminated by register 0.
const uint16_t *
Toy64RegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  static const uint16_t CalleeSavedRegs[] = {0};
  return CalleeSavedRegs;
}

BitVector Toy64RegisterInfo::getReservedRegs(const MachineFunction &MF) const {
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

bool Toy64RegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                            int SPAdj, unsigned FIOperandNum,
                                            RegScavenger *RS) const {

  assert(SPAdj == 0 && "Toy64 does not support dynamic SP adjustments yet");

  MachineInstr &MI = *II;
  MachineFunction &MF = *MI.getParent()->getParent();

  int FrameIndex = MI.getOperand(FIOperandNum).getIndex();

  Register FrameReg;

  int64_t Offset = getFrameLowering(MF)
                       ->getFrameIndexReference(MF, FrameIndex, FrameReg)
                       .getFixed();

  // LD/SD also contain their own immediate displacement.
  Offset += MI.getOperand(FIOperandNum + 1).getImm();

  if (!isInt<12>(Offset))
    report_fatal_error("Toy64 stack offset does not fit in 12 bits");

  // Replace:
  //
  //   FrameIndex
  //
  // with:
  //
  //   X2
  //
  MI.getOperand(FIOperandNum).ChangeToRegister(FrameReg, false);

  // Replace the displacement with the final SP-relative offset.
  MI.getOperand(FIOperandNum + 1).ChangeToImmediate(Offset);

  return false;
}

Register Toy64RegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  // Until frame-pointer support is implemented, use SP.
  return Toy64::X2;
}