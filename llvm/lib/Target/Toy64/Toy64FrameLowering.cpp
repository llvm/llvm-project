//===-- Toy64FrameLowering.cpp - Toy64 Frame Lowering ---------------------===//

#include "Toy64FrameLowering.h"

#include "MCTargetDesc/Toy64MCTargetDesc.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Support/ErrorHandling.h"

#include "Toy64Subtarget.h"

#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

void Toy64FrameLowering::emitPrologue(MachineFunction &MF,
                                      MachineBasicBlock &MBB) const {

  uint64_t StackSize = MF.getFrameInfo().getStackSize();

  if (StackSize == 0)
    return;

  // ADDI's positive and negative immediates must both be encodable.
  if (StackSize > 2047)
    report_fatal_error("Toy64 stack frames larger than 2047 bytes "
                       "are not implemented");

  const Toy64InstrInfo *TII = MF.getSubtarget<Toy64Subtarget>().getInstrInfo();

  MachineBasicBlock::iterator I = MBB.begin();

  BuildMI(MBB, I, DebugLoc(), TII->get(Toy64::ADDI), Toy64::X2)
      .addReg(Toy64::X2)
      .addImm(-static_cast<int64_t>(StackSize))
      .setMIFlag(MachineInstr::FrameSetup);
}

void Toy64FrameLowering::emitEpilogue(MachineFunction &MF,
                                      MachineBasicBlock &MBB) const {

  uint64_t StackSize = MF.getFrameInfo().getStackSize();

  if (StackSize == 0)
    return;

  if (StackSize > 2047)
    report_fatal_error("Toy64 stack frames larger than 2047 bytes "
                       "are not implemented");

  const Toy64InstrInfo *TII = MF.getSubtarget<Toy64Subtarget>().getInstrInfo();

  MachineBasicBlock::iterator I = MBB.getFirstTerminator();

  BuildMI(MBB, I, DebugLoc(), TII->get(Toy64::ADDI), Toy64::X2)
      .addReg(Toy64::X2)
      .addImm(static_cast<int64_t>(StackSize))
      .setMIFlag(MachineInstr::FrameDestroy);
}

StackOffset
Toy64FrameLowering::getFrameIndexReference(const MachineFunction &MF, int FI,
                                           Register &FrameReg) const {

  const MachineFrameInfo &MFI = MF.getFrameInfo();

  FrameReg = Toy64::X2;

  int64_t Offset = MFI.getObjectOffset(FI) + MFI.getStackSize();

  return StackOffset::getFixed(Offset);
}