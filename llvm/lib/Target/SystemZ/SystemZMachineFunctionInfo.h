//=== SystemZMachineFunctionInfo.h - SystemZ machine function info -*- C++ -*-//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SYSTEMZ_SYSTEMZMACHINEFUNCTIONINFO_H
#define LLVM_LIB_TARGET_SYSTEMZ_SYSTEMZMACHINEFUNCTIONINFO_H

#include "llvm/CodeGen/MachineFunction.h"

namespace llvm {

namespace SystemZ {
// A struct to hold the low and high GPR registers to be saved/restored as
// well as the offset into the register save area of the low register.
struct GPRRegs {
  unsigned LowGPR;
  unsigned HighGPR;
  unsigned GPROffset;
  GPRRegs() : LowGPR(0), HighGPR(0), GPROffset(0) {}
  };
}

class SystemZMachineFunctionInfo : public MachineFunctionInfo {
  virtual void anchor();

  SystemZ::GPRRegs SpillGPRRegs;
  SystemZ::GPRRegs RestoreGPRRegs;
  Register VarArgsFirstGPR;
  Register VarArgsFirstFPR;
  unsigned VarArgsFrameIndex;
  unsigned RegSaveFrameIndex;
  int FramePointerSaveIndex;
  unsigned NumLocalDynamics;

public:
  explicit SystemZMachineFunctionInfo(MachineFunction &MF)
    : VarArgsFirstGPR(0), VarArgsFirstFPR(0), VarArgsFrameIndex(0),
      RegSaveFrameIndex(0), FramePointerSaveIndex(0), NumLocalDynamics(0) {}

  MachineFunctionInfo *
  clone(BumpPtrAllocator &Allocator, MachineFunction &DestMF,
        const DenseMap<MachineBasicBlock *, MachineBasicBlock *> &Src2DstMBB)
      const override;

  // Get and set the first and last call-saved GPR that should be saved by
  // this function and the SP offset for the STMG.  These are 0 if no GPRs
  // need to be saved or restored.
  SystemZ::GPRRegs getSpillGPRRegs() const { return SpillGPRRegs; }
  void setSpillGPRRegs(Register Low, Register High, unsigned Offs) {
    SpillGPRRegs.LowGPR = Low;
    SpillGPRRegs.HighGPR = High;
    SpillGPRRegs.GPROffset = Offs;
  }

  // Get and set the first and last call-saved GPR that should be restored by
  // this function and the SP offset for the LMG.  These are 0 if no GPRs
  // need to be saved or restored.
  SystemZ::GPRRegs getRestoreGPRRegs() const { return RestoreGPRRegs; }
  void setRestoreGPRRegs(Register Low, Register High, unsigned Offs) {
    RestoreGPRRegs.LowGPR = Low;
    RestoreGPRRegs.HighGPR = High;
    RestoreGPRRegs.GPROffset = Offs;
  }

  // Get and set the number of fixed (as opposed to variable) arguments
  // that are passed in GPRs to this function.
  Register getVarArgsFirstGPR() const { return VarArgsFirstGPR; }
  void setVarArgsFirstGPR(Register GPR) { VarArgsFirstGPR = GPR; }

  // Likewise FPRs.
  Register getVarArgsFirstFPR() const { return VarArgsFirstFPR; }
  void setVarArgsFirstFPR(Register FPR) { VarArgsFirstFPR = FPR; }

  // Get and set the frame index of the first stack vararg.
  unsigned getVarArgsFrameIndex() const { return VarArgsFrameIndex; }
  void setVarArgsFrameIndex(unsigned FI) { VarArgsFrameIndex = FI; }

  // Get and set the frame index of the register save area
  // (i.e. the incoming stack pointer).
  unsigned getRegSaveFrameIndex() const { return RegSaveFrameIndex; }
  void setRegSaveFrameIndex(unsigned FI) { RegSaveFrameIndex = FI; }

  // Get and set the frame index of where the old frame pointer is stored.
  int getFramePointerSaveIndex() const { return FramePointerSaveIndex; }
  void setFramePointerSaveIndex(int Idx) { FramePointerSaveIndex = Idx; }

  // Count number of local-dynamic TLS symbols used.
  unsigned getNumLocalDynamicTLSAccesses() const { return NumLocalDynamics; }
  void incNumLocalDynamicTLSAccesses() { ++NumLocalDynamics; }
};

} // end namespace llvm

#endif
