//===--------------------- SIFrameLowering.h --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_SIFRAMELOWERING_H
#define LLVM_LIB_TARGET_AMDGPU_SIFRAMELOWERING_H

#include "AMDGPUFrameLowering.h"

namespace llvm {

class SIInstrInfo;
class SIMachineFunctionInfo;
class SIRegisterInfo;
class SISubtarget;

class SIFrameLowering final : public AMDGPUFrameLowering {
public:
  SIFrameLowering(StackDirection D, unsigned StackAl, int LAO,
                  unsigned TransAl = 1) :
    AMDGPUFrameLowering(D, StackAl, LAO, TransAl) {}
  ~SIFrameLowering() override = default;

  void emitEntryFunctionPrologue(MachineFunction &MF,
                                 MachineBasicBlock &MBB) const;
  void emitPrologue(MachineFunction &MF,
                    MachineBasicBlock &MBB) const override;
  void emitEpilogue(MachineFunction &MF,
                    MachineBasicBlock &MBB) const override;
  int getFrameIndexReference(const MachineFunction &MF, int FI,
                             unsigned &FrameReg) const override;

  void processFunctionBeforeFrameFinalized(
    MachineFunction &MF,
    RegScavenger *RS = nullptr) const override;

private:
  void emitFlatScratchInit(const SISubtarget &ST,
                           MachineFunction &MF,
                           MachineBasicBlock &MBB) const;

  unsigned getReservedPrivateSegmentBufferReg(
    const SISubtarget &ST,
    const SIInstrInfo *TII,
    const SIRegisterInfo *TRI,
    SIMachineFunctionInfo *MFI,
    MachineFunction &MF) const;

  std::pair<unsigned, unsigned> getReservedPrivateSegmentWaveByteOffsetReg(
    const SISubtarget &ST,
    const SIInstrInfo *TII,
    const SIRegisterInfo *TRI,
    SIMachineFunctionInfo *MFI,
    MachineFunction &MF) const;

  /// \brief Emits debugger prologue.
  void emitDebuggerPrologue(MachineFunction &MF, MachineBasicBlock &MBB) const;
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_SIFRAMELOWERING_H
