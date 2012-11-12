//===-- MBlazeMachineFunctionInfo.h - Private data --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the MBlaze specific subclass of MachineFunctionInfo.
//
//===----------------------------------------------------------------------===//

#ifndef MBLAZE_MACHINE_FUNCTION_INFO_H
#define MBLAZE_MACHINE_FUNCTION_INFO_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"

namespace llvm {

/// MBlazeFunctionInfo - This class is derived from MachineFunction private
/// MBlaze target-specific information for each MachineFunction.
class MBlazeFunctionInfo : public MachineFunctionInfo {
  virtual void anchor();

  /// Holds for each function where on the stack the Frame Pointer must be
  /// saved. This is used on Prologue and Epilogue to emit FP save/restore
  int FPStackOffset;

  /// Holds for each function where on the stack the Return Address must be
  /// saved. This is used on Prologue and Epilogue to emit RA save/restore
  int RAStackOffset;

  /// MBlazeFIHolder - Holds a FrameIndex and it's Stack Pointer Offset
  struct MBlazeFIHolder {

    int FI;
    int SPOffset;

    MBlazeFIHolder(int FrameIndex, int StackPointerOffset)
      : FI(FrameIndex), SPOffset(StackPointerOffset) {}
  };

  /// When PIC is used the GP must be saved on the stack on the function
  /// prologue and must be reloaded from this stack location after every
  /// call. A reference to its stack location and frame index must be kept
  /// to be used on emitPrologue and processFunctionBeforeFrameFinalized.
  MBlazeFIHolder GPHolder;

  /// On LowerFormalArguments the stack size is unknown, so the Stack
  /// Pointer Offset calculation of "not in register arguments" must be
  /// postponed to emitPrologue.
  SmallVector<MBlazeFIHolder, 16> FnLoadArgs;
  bool HasLoadArgs;

  // When VarArgs, we must write registers back to caller stack, preserving
  // on register arguments. Since the stack size is unknown on
  // LowerFormalArguments, the Stack Pointer Offset calculation must be
  // postponed to emitPrologue.
  SmallVector<MBlazeFIHolder, 4> FnStoreVarArgs;
  bool HasStoreVarArgs;

  // When determining the final stack layout some of the frame indexes may
  // be replaced by new frame indexes that reside in the caller's stack
  // frame. The replacements are recorded in this structure.
  DenseMap<int,int> FIReplacements;

  /// SRetReturnReg - Some subtargets require that sret lowering includes
  /// returning the value of the returned struct in a register. This field
  /// holds the virtual register into which the sret argument is passed.
  unsigned SRetReturnReg;

  /// GlobalBaseReg - keeps track of the virtual register initialized for
  /// use as the global base register. This is used for PIC in some PIC
  /// relocation models.
  unsigned GlobalBaseReg;

  // VarArgsFrameIndex - FrameIndex for start of varargs area.
  int VarArgsFrameIndex;

  /// LiveInFI - keeps track of the frame indexes in a callers stack
  /// frame that are live into a function.
  SmallVector<int, 16> LiveInFI;

public:
  MBlazeFunctionInfo(MachineFunction& MF)
  : FPStackOffset(0), RAStackOffset(0), GPHolder(-1,-1), HasLoadArgs(false),
    HasStoreVarArgs(false), SRetReturnReg(0), GlobalBaseReg(0),
    VarArgsFrameIndex(0), LiveInFI()
  {}

  int getFPStackOffset() const { return FPStackOffset; }
  void setFPStackOffset(int Off) { FPStackOffset = Off; }

  int getRAStackOffset() const { return RAStackOffset; }
  void setRAStackOffset(int Off) { RAStackOffset = Off; }

  int getGPStackOffset() const { return GPHolder.SPOffset; }
  int getGPFI() const { return GPHolder.FI; }
  void setGPStackOffset(int Off) { GPHolder.SPOffset = Off; }
  void setGPFI(int FI) { GPHolder.FI = FI; }
  bool needGPSaveRestore() const { return GPHolder.SPOffset != -1; }

  bool hasLoadArgs() const { return HasLoadArgs; }
  bool hasStoreVarArgs() const { return HasStoreVarArgs; }

  void recordLiveIn(int FI) {
    LiveInFI.push_back(FI);
  }

  bool isLiveIn(int FI) {
    for (unsigned i = 0, e = LiveInFI.size(); i < e; ++i)
      if (FI == LiveInFI[i]) return true;

    return false;
  }

  const SmallVector<int, 16>& getLiveIn() const { return LiveInFI; }

  void recordReplacement(int OFI, int NFI) {
    FIReplacements.insert(std::make_pair(OFI,NFI));
  }

  bool hasReplacement(int OFI) const {
    return FIReplacements.find(OFI) != FIReplacements.end();
  }

  int getReplacement(int OFI) const {
    return FIReplacements.lookup(OFI);
  }

  void recordLoadArgsFI(int FI, int SPOffset) {
    if (!HasLoadArgs) HasLoadArgs=true;
    FnLoadArgs.push_back(MBlazeFIHolder(FI, SPOffset));
  }

  void recordStoreVarArgsFI(int FI, int SPOffset) {
    if (!HasStoreVarArgs) HasStoreVarArgs=true;
    FnStoreVarArgs.push_back(MBlazeFIHolder(FI, SPOffset));
  }

  void adjustLoadArgsFI(MachineFrameInfo *MFI) const {
    if (!hasLoadArgs()) return;
    for (unsigned i = 0, e = FnLoadArgs.size(); i != e; ++i)
      MFI->setObjectOffset(FnLoadArgs[i].FI, FnLoadArgs[i].SPOffset);
  }

  void adjustStoreVarArgsFI(MachineFrameInfo *MFI) const {
    if (!hasStoreVarArgs()) return;
    for (unsigned i = 0, e = FnStoreVarArgs.size(); i != e; ++i)
      MFI->setObjectOffset(FnStoreVarArgs[i].FI, FnStoreVarArgs[i].SPOffset);
  }

  unsigned getSRetReturnReg() const { return SRetReturnReg; }
  void setSRetReturnReg(unsigned Reg) { SRetReturnReg = Reg; }

  unsigned getGlobalBaseReg() const { return GlobalBaseReg; }
  void setGlobalBaseReg(unsigned Reg) { GlobalBaseReg = Reg; }

  int getVarArgsFrameIndex() const { return VarArgsFrameIndex; }
  void setVarArgsFrameIndex(int Index) { VarArgsFrameIndex = Index; }
};

} // end of namespace llvm

#endif // MBLAZE_MACHINE_FUNCTION_INFO_H
