//==-- AArch64FrameLowering.h - TargetFrameLowering for AArch64 --*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AARCH64_AARCH64FRAMELOWERING_H
#define LLVM_LIB_TARGET_AARCH64_AARCH64FRAMELOWERING_H

#include "llvm/CodeGen/MachineOptimizationRemarkEmitter.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/Support/TypeSize.h"

namespace llvm {

class TargetLowering;
class AArch64FunctionInfo;
class AArch64PrologueEmitter;
class AArch64EpilogueEmitter;

class AArch64FrameLowering : public TargetFrameLowering {
public:
  explicit AArch64FrameLowering()
      : TargetFrameLowering(StackGrowsDown, Align(16), 0, Align(16),
                            true /*StackRealignable*/) {}

  void resetCFIToInitialState(MachineBasicBlock &MBB) const override;

  MachineBasicBlock::iterator
  eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator I) const override;

  /// emitProlog/emitEpilog - These methods insert prolog and epilog code into
  /// the function.
  void emitPrologue(MachineFunction &MF, MachineBasicBlock &MBB) const override;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const override;

  /// Harden the entire function with pac-ret.
  ///
  /// If pac-ret+leaf is requested, we want to harden as much code as possible.
  /// This function inserts pac-ret hardening at the points where prologue and
  /// epilogue are traditionally inserted, ignoring possible shrink-wrapping
  /// optimization.
  void emitPacRetPlusLeafHardening(MachineFunction &MF) const;

  bool enableCFIFixup(const MachineFunction &MF) const override;

  bool enableFullCFIFixup(const MachineFunction &MF) const override;

  bool canUseAsPrologue(const MachineBasicBlock &MBB) const override;

  StackOffset getFrameIndexReference(const MachineFunction &MF, int FI,
                                     Register &FrameReg) const override;
  StackOffset getFrameIndexReferenceFromSP(const MachineFunction &MF,
                                           int FI) const override;
  StackOffset resolveFrameIndexReference(const MachineFunction &MF, int FI,
                                         Register &FrameReg, bool PreferFP,
                                         bool ForSimm) const;
  StackOffset resolveFrameOffsetReference(const MachineFunction &MF,
                                          int64_t ObjectOffset, bool isFixed,
                                          bool isSVE, Register &FrameReg,
                                          bool PreferFP, bool ForSimm) const;
  bool spillCalleeSavedRegisters(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator MI,
                                 ArrayRef<CalleeSavedInfo> CSI,
                                 const TargetRegisterInfo *TRI) const override;

  bool
  restoreCalleeSavedRegisters(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator MI,
                              MutableArrayRef<CalleeSavedInfo> CSI,
                              const TargetRegisterInfo *TRI) const override;

  /// Can this function use the red zone for local allocations.
  bool canUseRedZone(const MachineFunction &MF) const;

  bool hasReservedCallFrame(const MachineFunction &MF) const override;

  bool assignCalleeSavedSpillSlots(MachineFunction &MF,
                                   const TargetRegisterInfo *TRI,
                                   std::vector<CalleeSavedInfo> &CSI,
                                   unsigned &MinCSFrameIndex,
                                   unsigned &MaxCSFrameIndex) const override;

  void determineCalleeSaves(MachineFunction &MF, BitVector &SavedRegs,
                            RegScavenger *RS) const override;

  /// Returns true if the target will correctly handle shrink wrapping.
  bool enableShrinkWrapping(const MachineFunction &MF) const override {
    return true;
  }

  bool enableStackSlotScavenging(const MachineFunction &MF) const override;
  TargetStackID::Value getStackIDForScalableVectors() const override;

  void processFunctionBeforeFrameFinalized(MachineFunction &MF,
                                           RegScavenger *RS) const override;

  void
  processFunctionBeforeFrameIndicesReplaced(MachineFunction &MF,
                                            RegScavenger *RS) const override;

  unsigned getWinEHParentFrameOffset(const MachineFunction &MF) const override;

  unsigned getWinEHFuncletFrameSize(const MachineFunction &MF) const;

  StackOffset
  getFrameIndexReferencePreferSP(const MachineFunction &MF, int FI,
                                 Register &FrameReg,
                                 bool IgnoreSPUpdates) const override;
  StackOffset getNonLocalFrameIndexReference(const MachineFunction &MF,
                                             int FI) const override;
  int getSEHFrameIndexOffset(const MachineFunction &MF, int FI) const;

  bool isSupportedStackID(TargetStackID::Value ID) const override {
    switch (ID) {
    default:
      return false;
    case TargetStackID::Default:
    case TargetStackID::ScalableVector:
    case TargetStackID::ScalablePredicateVector:
    case TargetStackID::NoAlloc:
      return true;
    }
  }

  bool isStackIdSafeForLocalArea(unsigned StackId) const override {
    // We don't support putting SVE objects into the pre-allocated local
    // frame block at the moment.
    return (StackId != TargetStackID::ScalableVector &&
            StackId != TargetStackID::ScalablePredicateVector);
  }

  void
  orderFrameObjects(const MachineFunction &MF,
                    SmallVectorImpl<int> &ObjectsToAllocate) const override;

  bool isFPReserved(const MachineFunction &MF) const;

  bool needsWinCFI(const MachineFunction &MF) const;

  bool requiresSaveVG(const MachineFunction &MF) const;

  StackOffset getSVEStackSize(const MachineFunction &MF) const;

  friend class AArch64PrologueEpilogueCommon;
  friend class AArch64PrologueEmitter;
  friend class AArch64EpilogueEmitter;

protected:
  bool hasFPImpl(const MachineFunction &MF) const override;

private:
  /// Returns true if a homogeneous prolog or epilog code can be emitted
  /// for the size optimization. If so, HOM_Prolog/HOM_Epilog pseudo
  /// instructions are emitted in place. When Exit block is given, this check is
  /// for epilog.
  bool homogeneousPrologEpilog(MachineFunction &MF,
                               MachineBasicBlock *Exit = nullptr) const;

  /// Returns true if CSRs should be paired.
  bool producePairRegisters(MachineFunction &MF) const;

  int64_t estimateSVEStackObjectOffsets(MachineFrameInfo &MF) const;
  int64_t assignSVEStackObjectOffsets(MachineFrameInfo &MF,
                                      int &MinCSFrameIndex,
                                      int &MaxCSFrameIndex) const;
  /// Make a determination whether a Hazard slot is used and create it if
  /// needed.
  void determineStackHazardSlot(MachineFunction &MF,
                                BitVector &SavedRegs) const;

  /// Emit target zero call-used regs.
  void emitZeroCallUsedRegs(BitVector RegsToZero,
                            MachineBasicBlock &MBB) const override;

  /// Replace a StackProbe stub (if any) with the actual probe code inline
  void inlineStackProbe(MachineFunction &MF,
                        MachineBasicBlock &PrologueMBB) const override;

  void inlineStackProbeFixed(MachineBasicBlock::iterator MBBI,
                             Register ScratchReg, int64_t FrameSize,
                             StackOffset CFAOffset) const;

  MachineBasicBlock::iterator
  inlineStackProbeLoopExactMultiple(MachineBasicBlock::iterator MBBI,
                                    int64_t NegProbeSize,
                                    Register TargetReg) const;

  void emitRemarks(const MachineFunction &MF,
                   MachineOptimizationRemarkEmitter *ORE) const override;

  bool windowsRequiresStackProbe(const MachineFunction &MF,
                                 uint64_t StackSizeInBytes) const;

  bool shouldSignReturnAddressEverywhere(const MachineFunction &MF) const;

  StackOffset getFPOffset(const MachineFunction &MF,
                          int64_t ObjectOffset) const;

  StackOffset getStackOffset(const MachineFunction &MF,
                             int64_t ObjectOffset) const;

  // Given a load or a store instruction, generate an appropriate unwinding SEH
  // code on Windows.
  MachineBasicBlock::iterator insertSEH(MachineBasicBlock::iterator MBBI,
                                        const TargetInstrInfo &TII,
                                        MachineInstr::MIFlag Flag) const;

  /// Returns how much of the incoming argument stack area (in bytes) we should
  /// clean up in an epilogue. For the C calling convention this will be 0, for
  /// guaranteed tail call conventions it can be positive (a normal return or a
  /// tail call to a function that uses less stack space for arguments) or
  /// negative (for a tail call to a function that needs more stack space than
  /// us for arguments).
  int64_t getArgumentStackToRestore(MachineFunction &MF,
                                    MachineBasicBlock &MBB) const;

  // Find a scratch register that we can use at the start of the prologue to
  // re-align the stack pointer.  We avoid using callee-save registers since
  // they may appear to be free when this is called from canUseAsPrologue
  // (during shrink wrapping), but then no longer be free when this is called
  // from emitPrologue.
  //
  // FIXME: This is a bit conservative, since in the above case we could use one
  // of the callee-save registers as a scratch temp to re-align the stack
  // pointer, but we would then have to make sure that we were in fact saving at
  // least one callee-save register in the prologue, which is additional
  // complexity that doesn't seem worth the benefit.
  Register findScratchNonCalleeSaveRegister(MachineBasicBlock *MBB,
                                            bool HasCall = false) const;

  /// Returns the size of the fixed object area (allocated next to sp on entry)
  /// On Win64 this may include a var args area and an UnwindHelp object for EH.
  unsigned getFixedObjectSize(const MachineFunction &MF,
                              const AArch64FunctionInfo *AFI, bool IsWin64,
                              bool IsFunclet) const;
};

} // End llvm namespace

#endif
