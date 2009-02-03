//====- ARMMachineFuctionInfo.h - ARM machine function info -----*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file declares ARM-specific per-machine-function information.
//
//===----------------------------------------------------------------------===//

#ifndef ARMMACHINEFUNCTIONINFO_H
#define ARMMACHINEFUNCTIONINFO_H

#include "ARMSubtarget.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/ADT/BitVector.h"

namespace llvm {

/// ARMFunctionInfo - This class is derived from MachineFunction private
/// ARM target-specific information for each MachineFunction.
class ARMFunctionInfo : public MachineFunctionInfo {

  /// isThumb - True if this function is compiled under Thumb mode.
  /// Used to initialized Align, so must precede it.
  bool isThumb;

  /// Align - required alignment.  ARM functions and Thumb functions with
  /// constant pools require 4-byte alignment; other Thumb functions
  /// require only 2-byte alignment.
  unsigned Align;

  /// VarArgsRegSaveSize - Size of the register save area for vararg functions.
  ///
  unsigned VarArgsRegSaveSize;

  /// HasStackFrame - True if this function has a stack frame. Set by
  /// processFunctionBeforeCalleeSavedScan().
  bool HasStackFrame;

  /// LRSpilledForFarJump - True if the LR register has been for spilled to
  /// enable far jump.
  bool LRSpilledForFarJump;

  /// R3IsLiveIn - True if R3 is live in to this function.
  /// FIXME: Remove when register scavenger for Thumb is done.
  bool R3IsLiveIn;

  /// FramePtrSpillOffset - If HasStackFrame, this records the frame pointer
  /// spill stack offset.
  unsigned FramePtrSpillOffset;

  /// GPRCS1Offset, GPRCS2Offset, DPRCSOffset - Starting offset of callee saved
  /// register spills areas. For Mac OS X:
  ///
  /// GPR callee-saved (1) : r4, r5, r6, r7, lr
  /// --------------------------------------------
  /// GPR callee-saved (2) : r8, r10, r11
  /// --------------------------------------------
  /// DPR callee-saved : d8 - d15
  unsigned GPRCS1Offset;
  unsigned GPRCS2Offset;
  unsigned DPRCSOffset;

  /// GPRCS1Size, GPRCS2Size, DPRCSSize - Sizes of callee saved register spills
  /// areas.
  unsigned GPRCS1Size;
  unsigned GPRCS2Size;
  unsigned DPRCSSize;

  /// GPRCS1Frames, GPRCS2Frames, DPRCSFrames - Keeps track of frame indices
  /// which belong to these spill areas.
  BitVector GPRCS1Frames;
  BitVector GPRCS2Frames;
  BitVector DPRCSFrames;

  /// SpilledCSRegs - A BitVector mask of all spilled callee-saved registers.
  ///
  BitVector SpilledCSRegs;

  /// JumpTableUId - Unique id for jumptables.
  ///
  unsigned JumpTableUId;

  unsigned ConstPoolEntryUId;

public:
  ARMFunctionInfo() :
    isThumb(false), 
    Align(2U),
    VarArgsRegSaveSize(0), HasStackFrame(false),
    LRSpilledForFarJump(false), R3IsLiveIn(false),
    FramePtrSpillOffset(0), GPRCS1Offset(0), GPRCS2Offset(0), DPRCSOffset(0),
    GPRCS1Size(0), GPRCS2Size(0), DPRCSSize(0),
    GPRCS1Frames(0), GPRCS2Frames(0), DPRCSFrames(0),
    JumpTableUId(0), ConstPoolEntryUId(0) {}

  ARMFunctionInfo(MachineFunction &MF) :
    isThumb(MF.getTarget().getSubtarget<ARMSubtarget>().isThumb()),
    Align(isThumb ? 1U : 2U),
    VarArgsRegSaveSize(0), HasStackFrame(false),
    LRSpilledForFarJump(false), R3IsLiveIn(false),
    FramePtrSpillOffset(0), GPRCS1Offset(0), GPRCS2Offset(0), DPRCSOffset(0),
    GPRCS1Size(0), GPRCS2Size(0), DPRCSSize(0),
    GPRCS1Frames(32), GPRCS2Frames(32), DPRCSFrames(32),
    SpilledCSRegs(MF.getTarget().getRegisterInfo()->getNumRegs()),
    JumpTableUId(0), ConstPoolEntryUId(0) {}

  bool isThumbFunction() const { return isThumb; }

  unsigned getAlign() const { return Align; }
  void setAlign(unsigned a) { Align = a; }

  unsigned getVarArgsRegSaveSize() const { return VarArgsRegSaveSize; }
  void setVarArgsRegSaveSize(unsigned s) { VarArgsRegSaveSize = s; }

  bool hasStackFrame() const { return HasStackFrame; }
  void setHasStackFrame(bool s) { HasStackFrame = s; }

  bool isLRSpilledForFarJump() const { return LRSpilledForFarJump; }
  void setLRIsSpilledForFarJump(bool s) { LRSpilledForFarJump = s; }

  // FIXME: Remove when register scavenger for Thumb is done.
  bool isR3LiveIn() const { return R3IsLiveIn; }
  void setR3IsLiveIn(bool l) { R3IsLiveIn = l; }

  unsigned getFramePtrSpillOffset() const { return FramePtrSpillOffset; }
  void setFramePtrSpillOffset(unsigned o) { FramePtrSpillOffset = o; }
  
  unsigned getGPRCalleeSavedArea1Offset() const { return GPRCS1Offset; }
  unsigned getGPRCalleeSavedArea2Offset() const { return GPRCS2Offset; }
  unsigned getDPRCalleeSavedAreaOffset()  const { return DPRCSOffset; }

  void setGPRCalleeSavedArea1Offset(unsigned o) { GPRCS1Offset = o; }
  void setGPRCalleeSavedArea2Offset(unsigned o) { GPRCS2Offset = o; }
  void setDPRCalleeSavedAreaOffset(unsigned o)  { DPRCSOffset = o; }

  unsigned getGPRCalleeSavedArea1Size() const { return GPRCS1Size; }
  unsigned getGPRCalleeSavedArea2Size() const { return GPRCS2Size; }
  unsigned getDPRCalleeSavedAreaSize()  const { return DPRCSSize; }

  void setGPRCalleeSavedArea1Size(unsigned s) { GPRCS1Size = s; }
  void setGPRCalleeSavedArea2Size(unsigned s) { GPRCS2Size = s; }
  void setDPRCalleeSavedAreaSize(unsigned s)  { DPRCSSize = s; }

  bool isGPRCalleeSavedArea1Frame(int fi) const {
    if (fi < 0 || fi >= (int)GPRCS1Frames.size())
      return false;
    return GPRCS1Frames[fi];
  }
  bool isGPRCalleeSavedArea2Frame(int fi) const {
    if (fi < 0 || fi >= (int)GPRCS2Frames.size())
      return false;
    return GPRCS2Frames[fi];
  }
  bool isDPRCalleeSavedAreaFrame(int fi) const {
    if (fi < 0 || fi >= (int)DPRCSFrames.size())
      return false;
    return DPRCSFrames[fi];
  }

  void addGPRCalleeSavedArea1Frame(int fi) {
    if (fi >= 0) {
      int Size = GPRCS1Frames.size();
      if (fi >= Size) {
        Size *= 2;
        if (fi >= Size)
          Size = fi+1;
        GPRCS1Frames.resize(Size);
      }
      GPRCS1Frames[fi] = true;
    }
  }
  void addGPRCalleeSavedArea2Frame(int fi) {
    if (fi >= 0) {
      int Size = GPRCS2Frames.size();
      if (fi >= Size) {
        Size *= 2;
        if (fi >= Size)
          Size = fi+1;
        GPRCS2Frames.resize(Size);
      }
      GPRCS2Frames[fi] = true;
    }
  }
  void addDPRCalleeSavedAreaFrame(int fi) {
    if (fi >= 0) {
      int Size = DPRCSFrames.size();
      if (fi >= Size) {
        Size *= 2;
        if (fi >= Size)
          Size = fi+1;
        DPRCSFrames.resize(Size);
      }
      DPRCSFrames[fi] = true;
    }
  }

  void setCSRegisterIsSpilled(unsigned Reg) {
    SpilledCSRegs.set(Reg);
  }

  bool isCSRegisterSpilled(unsigned Reg) const {
    return SpilledCSRegs[Reg];
  }

  const BitVector &getSpilledCSRegisters() const {
    return SpilledCSRegs;
  }

  unsigned createJumpTableUId() {
    return JumpTableUId++;
  }

  unsigned getNumJumpTables() const {
    return JumpTableUId;
  }

  void initConstPoolEntryUId(unsigned UId) {
    ConstPoolEntryUId = UId;
  }

  unsigned getNumConstPoolEntries() const {
    return ConstPoolEntryUId;
  }

  unsigned createConstPoolEntryUId() {
    return ConstPoolEntryUId++;
  }
};
} // End llvm namespace

#endif // ARMMACHINEFUNCTIONINFO_H
