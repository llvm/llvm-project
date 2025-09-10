//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the AArch64PrologueEmitter class,
/// which is is used to emit the prologue on AArch64.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AARCH64_AARCH64PROLOGUEEPILOGUE_H
#define LLVM_LIB_TARGET_AARCH64_AARCH64PROLOGUEEPILOGUE_H

#include "AArch64RegisterInfo.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFunction.h"

namespace llvm {

class AArch64Subtarget;
class AArch64FunctionInfo;
class AArch64FrameLowering;

/// A helper class for emitting the prologue. Substantial new functionality
/// should be factored into a new method. Where possible "emit*" methods should
/// be const, and any flags that change how the prologue is emitted should be
/// set in the constructor.
class AArch64PrologueEmitter {
public:
  AArch64PrologueEmitter(MachineFunction &MF, MachineBasicBlock &MBB,
                         const AArch64FrameLowering &AFL);

  /// Emit the prologue.
  void emitPrologue();

  ~AArch64PrologueEmitter() {
    MF.setHasWinCFI(HasWinCFI);
#ifndef NDEBUG
    verifyPrologueClobbers();
#endif
  }

private:
  void emitShadowCallStackPrologue(MachineBasicBlock::iterator MBBI,
                                   const DebugLoc &DL) const;

  void emitSwiftAsyncContextFramePointer(MachineBasicBlock::iterator MBBI,
                                         const DebugLoc &DL) const;

  void emitEmptyStackFramePrologue(int64_t NumBytes,
                                   MachineBasicBlock::iterator MBBI,
                                   const DebugLoc &DL) const;

  void emitFramePointerSetup(MachineBasicBlock::iterator MBBI,
                             const DebugLoc &DL, unsigned FixedObject);

  void emitDefineCFAWithFP(MachineBasicBlock::iterator MBBI,
                           unsigned FixedObject) const;

  void emitWindowsStackProbe(MachineBasicBlock::iterator MBBI,
                             const DebugLoc &DL, int64_t &NumBytes,
                             int64_t RealignmentPadding) const;

  void emitCalleeSavedGPRLocations(MachineBasicBlock::iterator MBBI) const;
  void emitCalleeSavedSVELocations(MachineBasicBlock::iterator MBBI) const;

  void determineLocalsStackSize(uint64_t StackSize, uint64_t PrologueSaveSize);

  MachineFunction &MF;
  MachineBasicBlock &MBB;

  const Function &F;
  const MachineFrameInfo &MFI;
  const AArch64Subtarget &Subtarget;
  const AArch64FrameLowering &AFL;
  const AArch64RegisterInfo &RegInfo;

#ifndef NDEBUG
  mutable LivePhysRegs LiveRegs{RegInfo};
  MachineBasicBlock::iterator PrologueEndI;

  void collectBlockLiveins();
  void verifyPrologueClobbers() const;
#endif

  // Prologue flags. These generally should not change outside of the
  // constructor. Two exceptions are "CombineSPBump" which is set in
  // determineLocalsStackSize, and "NeedsWinCFI" which is set in
  // emitFramePointerSetup.
  bool EmitCFI = false;
  bool EmitAsyncCFI = false;
  bool HasFP = false;
  bool IsFunclet = false;
  bool CombineSPBump = false;
  bool HomPrologEpilog = false;
  bool NeedsWinCFI = false;

  // Note: "HasWinCFI" is mutable as it can change in any "emit" function.
  mutable bool HasWinCFI = false;

  const TargetInstrInfo *TII = nullptr;
  AArch64FunctionInfo *AFI = nullptr;
};

} // namespace llvm

#endif
