//=- RISCVMachineFunctionInfo.h - RISCV machine function info -----*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares RISCV-specific per-machine-function information.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_RISCVMACHINEFUNCTIONINFO_H
#define LLVM_LIB_TARGET_RISCV_RISCVMACHINEFUNCTIONINFO_H

#include "RISCVSubtarget.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"

namespace llvm {

/// RISCVMachineFunctionInfo - This class is derived from MachineFunctionInfo
/// and contains private RISCV-specific information for each MachineFunction.
class RISCVMachineFunctionInfo : public MachineFunctionInfo {
private:
  MachineFunction &MF;
  /// FrameIndex for start of varargs area
  int VarArgsFrameIndex = 0;
  /// Size of the save area used for varargs
  int VarArgsSaveSize = 0;
  /// FrameIndex used for transferring values between 64-bit FPRs and a pair
  /// of 32-bit GPRs via the stack.
  int MoveF64FrameIndex = -1;
  /// Size of any opaque stack adjustment due to save/restore libcalls.
  unsigned LibCallStackSize = 0;

public:
  RISCVMachineFunctionInfo(MachineFunction &MF) : MF(MF) {}

  int getVarArgsFrameIndex() const { return VarArgsFrameIndex; }
  void setVarArgsFrameIndex(int Index) { VarArgsFrameIndex = Index; }

  unsigned getVarArgsSaveSize() const { return VarArgsSaveSize; }
  void setVarArgsSaveSize(int Size) { VarArgsSaveSize = Size; }

  int getMoveF64FrameIndex() {
    if (MoveF64FrameIndex == -1)
      MoveF64FrameIndex = MF.getFrameInfo().CreateStackObject(8, 8, false);
    return MoveF64FrameIndex;
  }

  unsigned getLibCallStackSize() const { return LibCallStackSize; }
  void setLibCallStackSize(unsigned Size) { LibCallStackSize = Size; }

  bool useSaveRestoreLibCalls() const {
    // We cannot use fixed locations for the callee saved spill slots if the
    // function uses a varargs save area.
    return MF.getSubtarget<RISCVSubtarget>().enableSaveRestore() &&
           VarArgsSaveSize == 0 && !MF.getFrameInfo().hasTailCall();
  }
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_RISCV_RISCVMACHINEFUNCTIONINFO_H
