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

class SIFrameLowering final : public AMDGPUFrameLowering {
public:
  SIFrameLowering(StackDirection D, unsigned StackAl, int LAO,
                  unsigned TransAl = 1) :
    AMDGPUFrameLowering(D, StackAl, LAO, TransAl) {}
  ~SIFrameLowering() override {}

  void emitPrologue(MachineFunction &MF,
                    MachineBasicBlock &MBB) const override;

  void processFunctionBeforeFrameFinalized(
    MachineFunction &MF,
    RegScavenger *RS = nullptr) const override;
};

}

#endif
