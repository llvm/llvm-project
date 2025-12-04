//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the AArch64PrologueEmitter and
/// AArch64EpilogueEmitter classes, which are is used to emit the prologue and
/// epilogue on AArch64.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AARCH64_AARCH64PROLOGUEEPILOGUE_H
#define LLVM_LIB_TARGET_AARCH64_AARCH64PROLOGUEEPILOGUE_H

#include "AArch64RegisterInfo.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFunction.h"

namespace llvm {

class TargetLowering;
class AArch64Subtarget;
class AArch64FunctionInfo;
class AArch64FrameLowering;

struct SVEFrameSizes {
  struct {
    StackOffset CalleeSavesSize, LocalsSize;
  } PPR, ZPR;
};

struct SVEStackAllocations {
  StackOffset BeforePPRs, AfterPPRs, AfterZPRs;
  StackOffset totalSize() const { return BeforePPRs + AfterPPRs + AfterZPRs; }
};

class AArch64PrologueEpilogueCommon {
public:
  AArch64PrologueEpilogueCommon(MachineFunction &MF, MachineBasicBlock &MBB,
                                const AArch64FrameLowering &AFL);

  enum class SVEStackLayout {
    Default,
    Split,
    CalleeSavesAboveFrameRecord,
  };

protected:
  bool requiresGetVGCall() const;

  bool isVGInstruction(MachineBasicBlock::iterator MBBI,
                       const TargetLowering &TLI) const;

  // Convert callee-save register save/restore instruction to do stack pointer
  // decrement/increment to allocate/deallocate the callee-save stack area by
  // converting store/load to use pre/post increment version.
  MachineBasicBlock::iterator convertCalleeSaveRestoreToSPPrePostIncDec(
      MachineBasicBlock::iterator MBBI, const DebugLoc &DL, int CSStackSizeInc,
      bool EmitCFI, MachineInstr::MIFlag FrameFlag = MachineInstr::FrameSetup,
      int CFAOffset = 0) const;

  // Fixup callee-save register save/restore instructions to take into account
  // combined SP bump by adding the local stack size to the stack offsets.
  void fixupCalleeSaveRestoreStackOffset(MachineInstr &MI,
                                         uint64_t LocalStackSize) const;

  bool shouldCombineCSRLocalStackBump(uint64_t StackBumpBytes) const;

  SVEFrameSizes getSVEStackFrameSizes() const;
  SVEStackAllocations getSVEStackAllocations(SVEFrameSizes const &);

  MachineFunction &MF;
  MachineBasicBlock &MBB;

  const MachineFrameInfo &MFI;
  const AArch64Subtarget &Subtarget;
  const AArch64FrameLowering &AFL;
  const AArch64RegisterInfo &RegInfo;

  // Common flags. These generally should not change outside of the (possibly
  // derived) constructor.
  bool HasFP = false;
  bool EmitCFI = false;     // Note: Set in derived constructors.
  bool IsFunclet = false;   // Note: Set in derived constructors.
  bool NeedsWinCFI = false; // Note: Can be changed in emitFramePointerSetup.
  bool HomPrologEpilog = false; // Note: Set in derived constructors.
  SVEStackLayout SVELayout = SVEStackLayout::Default;

  // Note: "HasWinCFI" is mutable as it can change in any "emit" function.
  mutable bool HasWinCFI = false;

  const TargetInstrInfo *TII = nullptr;
  AArch64FunctionInfo *AFI = nullptr;
};

/// A helper class for emitting the prologue. Substantial new functionality
/// should be factored into a new method. Where possible "emit*" methods should
/// be const, and any flags that change how the prologue is emitted should be
/// set in the constructor.
class AArch64PrologueEmitter final : public AArch64PrologueEpilogueCommon {
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
  void allocateStackSpace(MachineBasicBlock::iterator MBBI,
                          int64_t RealignmentPadding, StackOffset AllocSize,
                          bool EmitCFI, StackOffset InitialOffset,
                          bool FollowupAllocs);

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

  const Function &F;

#ifndef NDEBUG
  mutable LivePhysRegs LiveRegs{RegInfo};
  MachineBasicBlock::iterator PrologueEndI;

  void collectBlockLiveins();
  void verifyPrologueClobbers() const;
#endif

  // Prologue flags. These generally should not change outside of the
  // constructor.
  bool EmitAsyncCFI = false;
  bool CombineSPBump = false; // Note: This is set in determineLocalsStackSize.
};

/// A helper class for emitting the epilogue. Substantial new functionality
/// should be factored into a new method. Where possible "emit*" methods should
/// be const, and any flags that change how the epilogue is emitted should be
/// set in the constructor.
class AArch64EpilogueEmitter final : public AArch64PrologueEpilogueCommon {
public:
  AArch64EpilogueEmitter(MachineFunction &MF, MachineBasicBlock &MBB,
                         const AArch64FrameLowering &AFL);

  /// Emit the epilogue.
  void emitEpilogue();

  ~AArch64EpilogueEmitter() { finalizeEpilogue(); }

private:
  bool shouldCombineCSRLocalStackBump(uint64_t StackBumpBytes) const;

  /// A helper for moving the SP to a negative offset from the FP, without
  /// deallocating any stack in the range FP to FP + Offset.
  void moveSPBelowFP(MachineBasicBlock::iterator MBBI, StackOffset Offset);

  void emitSwiftAsyncContextFramePointer(MachineBasicBlock::iterator MBBI,
                                         const DebugLoc &DL) const;

  void emitShadowCallStackEpilogue(MachineBasicBlock::iterator MBBI,
                                   const DebugLoc &DL) const;

  void emitCalleeSavedRestores(MachineBasicBlock::iterator MBBI,
                               bool SVE) const;

  void emitCalleeSavedGPRRestores(MachineBasicBlock::iterator MBBI) const {
    emitCalleeSavedRestores(MBBI, /*SVE=*/false);
  }

  void emitCalleeSavedSVERestores(MachineBasicBlock::iterator MBBI) const {
    emitCalleeSavedRestores(MBBI, /*SVE=*/true);
  }

  void finalizeEpilogue() const;

  MachineBasicBlock::iterator SEHEpilogueStartI;
  DebugLoc DL;
};

} // namespace llvm

#endif
