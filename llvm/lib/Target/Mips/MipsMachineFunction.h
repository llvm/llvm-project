//===-- MipsMachineFunctionInfo.h - Private data used for Mips ----*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the Mips specific subclass of MachineFunctionInfo.
//
//===----------------------------------------------------------------------===//

#ifndef MIPS_MACHINE_FUNCTION_INFO_H
#define MIPS_MACHINE_FUNCTION_INFO_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/VectorExtras.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"

namespace llvm {

/// MipsFunctionInfo - This class is derived from MachineFunction private
/// Mips target-specific information for each MachineFunction.
class MipsFunctionInfo : public MachineFunctionInfo {

private:
  /// Holds for each function where on the stack the Frame Pointer must be 
  /// saved. This is used on Prologue and Epilogue to emit FP save/restore
  int FPStackOffset;

  /// Holds for each function where on the stack the Return Address must be 
  /// saved. This is used on Prologue and Epilogue to emit RA save/restore
  int RAStackOffset;

  /// At each function entry, two special bitmask directives must be emitted
  /// to help debugging, for CPU and FPU callee saved registers. Both need
  /// the negative offset from the final stack size and its higher registers
  /// location on the stack.
  int CPUTopSavedRegOff;
  int FPUTopSavedRegOff;

  /// MipsFIHolder - Holds a FrameIndex and it's Stack Pointer Offset
  struct MipsFIHolder {

    int FI;
    int SPOffset;

    MipsFIHolder(int FrameIndex, int StackPointerOffset)
      : FI(FrameIndex), SPOffset(StackPointerOffset) {}
  };

  /// When PIC is used the GP must be saved on the stack on the function 
  /// prologue and must be reloaded from this stack location after every 
  /// call. A reference to its stack location and frame index must be kept 
  /// to be used on emitPrologue and processFunctionBeforeFrameFinalized.
  MipsFIHolder GPHolder;

  /// On LowerFORMAL_ARGUMENTS the stack size is unknown, so the Stack 
  /// Pointer Offset calculation of "not in register arguments" must be 
  /// postponed to emitPrologue. 
  SmallVector<MipsFIHolder, 16> FnLoadArgs;
  bool HasLoadArgs;

  // When VarArgs, we must write registers back to caller stack, preserving 
  // on register arguments. Since the stack size is unknown on 
  // LowerFORMAL_ARGUMENTS, the Stack Pointer Offset calculation must be
  // postponed to emitPrologue. 
  SmallVector<MipsFIHolder, 4> FnStoreVarArgs;
  bool HasStoreVarArgs;

  /// SRetReturnReg - Some subtargets require that sret lowering includes
  /// returning the value of the returned struct in a register. This field
  /// holds the virtual register into which the sret argument is passed.
  unsigned SRetReturnReg;

public:
  MipsFunctionInfo(MachineFunction& MF) 
  : FPStackOffset(0), RAStackOffset(0), CPUTopSavedRegOff(0), 
    FPUTopSavedRegOff(0), GPHolder(-1,-1), HasLoadArgs(false), 
    HasStoreVarArgs(false), SRetReturnReg(0)
  {}

  int getFPStackOffset() const { return FPStackOffset; }
  void setFPStackOffset(int Off) { FPStackOffset = Off; }

  int getRAStackOffset() const { return RAStackOffset; }
  void setRAStackOffset(int Off) { RAStackOffset = Off; }

  int getCPUTopSavedRegOff() const { return CPUTopSavedRegOff; }
  void setCPUTopSavedRegOff(int Off) { CPUTopSavedRegOff = Off; }

  int getFPUTopSavedRegOff() const { return FPUTopSavedRegOff; }
  void setFPUTopSavedRegOff(int Off) { FPUTopSavedRegOff = Off; }

  int getGPStackOffset() const { return GPHolder.SPOffset; }
  int getGPFI() const { return GPHolder.FI; }
  void setGPStackOffset(int Off) { GPHolder.SPOffset = Off; }
  void setGPFI(int FI) { GPHolder.FI = FI; }

  bool hasLoadArgs() const { return HasLoadArgs; }
  bool hasStoreVarArgs() const { return HasStoreVarArgs; } 

  void recordLoadArgsFI(int FI, int SPOffset) {
    if (!HasLoadArgs) HasLoadArgs=true;
    FnLoadArgs.push_back(MipsFIHolder(FI, SPOffset));
  }
  void recordStoreVarArgsFI(int FI, int SPOffset) {
    if (!HasStoreVarArgs) HasStoreVarArgs=true;
    FnStoreVarArgs.push_back(MipsFIHolder(FI, SPOffset));
  }

  void adjustLoadArgsFI(MachineFrameInfo *MFI) const {
    if (!hasLoadArgs()) return;
    for (unsigned i = 0, e = FnLoadArgs.size(); i != e; ++i) 
      MFI->setObjectOffset( FnLoadArgs[i].FI, FnLoadArgs[i].SPOffset );
  }
  void adjustStoreVarArgsFI(MachineFrameInfo *MFI) const {
    if (!hasStoreVarArgs()) return; 
    for (unsigned i = 0, e = FnStoreVarArgs.size(); i != e; ++i) 
      MFI->setObjectOffset( FnStoreVarArgs[i].FI, FnStoreVarArgs[i].SPOffset );
  }

  unsigned getSRetReturnReg() const { return SRetReturnReg; }
  void setSRetReturnReg(unsigned Reg) { SRetReturnReg = Reg; }
};

} // end of namespace llvm

#endif // MIPS_MACHINE_FUNCTION_INFO_H
