//===-- Toy64FrameLowering.cpp - Toy64 Frame Lowering ---------------------===//

#include "Toy64FrameLowering.h"

#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

void Toy64FrameLowering::emitPrologue(
    MachineFunction &MF,
    MachineBasicBlock &MBB) const {

  // We do not support real stack frames yet.
  //
  // For functions requiring no stack space, there is nothing to emit.
  if (MF.getFrameInfo().getStackSize() == 0)
    return;

  llvm_unreachable(
      "Toy64 stack-frame prologue is not implemented yet");
}

void Toy64FrameLowering::emitEpilogue(
    MachineFunction &MF,
    MachineBasicBlock &MBB) const {

  // We do not support real stack frames yet.
  if (MF.getFrameInfo().getStackSize() == 0)
    return;

  llvm_unreachable(
      "Toy64 stack-frame epilogue is not implemented yet");
}