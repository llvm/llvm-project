//===--------------------- AMDGPUFrameLowering.h ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief Interface to describe a layout of a stack frame on an AMDGPU target.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUFRAMELOWERING_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUFRAMELOWERING_H

#include "llvm/Target/TargetFrameLowering.h"

namespace llvm {

/// \brief Information about the stack frame layout on the AMDGPU targets.
///
/// It holds the direction of the stack growth, the known stack alignment on
/// entry to each function, and the offset to the locals area.
/// See TargetFrameInfo for more comments.
class AMDGPUFrameLowering : public TargetFrameLowering {
public:
  AMDGPUFrameLowering(StackDirection D, unsigned StackAl, int LAO,
                      unsigned TransAl = 1);
  virtual ~AMDGPUFrameLowering();

  /// \returns The number of 32-bit sub-registers that are used when storing
  /// values to the stack.
  unsigned getStackWidth(const MachineFunction &MF) const;
  int getFrameIndexReference(const MachineFunction &MF, int FI,
                             unsigned &FrameReg) const override;
  const SpillSlot *
    getCalleeSavedSpillSlots(unsigned &NumEntries) const override;
  void emitPrologue(MachineFunction &MF, MachineBasicBlock &MBB) const override;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const override;
  bool hasFP(const MachineFunction &MF) const override;
};
} // namespace llvm
#endif
