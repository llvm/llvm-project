//===-- WebAssemblyUtilities.cpp - WebAssembly Utility Functions ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file implements several utility functions for WebAssembly.
///
//===----------------------------------------------------------------------===//

#include "WebAssemblyUtilities.h"
#include "WebAssemblyMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
using namespace llvm;

bool WebAssembly::isArgument(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case WebAssembly::ARGUMENT_I32:
  case WebAssembly::ARGUMENT_I64:
  case WebAssembly::ARGUMENT_F32:
  case WebAssembly::ARGUMENT_F64:
  case WebAssembly::ARGUMENT_v16i8:
  case WebAssembly::ARGUMENT_v8i16:
  case WebAssembly::ARGUMENT_v4i32:
  case WebAssembly::ARGUMENT_v4f32:
    return true;
  default:
    return false;
  }
}

bool WebAssembly::isCopy(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case WebAssembly::COPY_I32:
  case WebAssembly::COPY_I64:
  case WebAssembly::COPY_F32:
  case WebAssembly::COPY_F64:
    return true;
  default:
    return false;
  }
}

bool WebAssembly::isTee(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case WebAssembly::TEE_I32:
  case WebAssembly::TEE_I64:
  case WebAssembly::TEE_F32:
  case WebAssembly::TEE_F64:
    return true;
  default:
    return false;
  }
}

/// Test whether MI is a child of some other node in an expression tree.
bool WebAssembly::isChild(const MachineInstr &MI,
                          const WebAssemblyFunctionInfo &MFI) {
  if (MI.getNumOperands() == 0)
    return false;
  const MachineOperand &MO = MI.getOperand(0);
  if (!MO.isReg() || MO.isImplicit() || !MO.isDef())
    return false;
  unsigned Reg = MO.getReg();
  return TargetRegisterInfo::isVirtualRegister(Reg) &&
         MFI.isVRegStackified(Reg);
}

bool WebAssembly::isCallIndirect(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case WebAssembly::CALL_INDIRECT_VOID:
  case WebAssembly::CALL_INDIRECT_I32:
  case WebAssembly::CALL_INDIRECT_I64:
  case WebAssembly::CALL_INDIRECT_F32:
  case WebAssembly::CALL_INDIRECT_F64:
  case WebAssembly::CALL_INDIRECT_v16i8:
  case WebAssembly::CALL_INDIRECT_v8i16:
  case WebAssembly::CALL_INDIRECT_v4i32:
  case WebAssembly::CALL_INDIRECT_v4f32:
    return true;
  default:
    return false;
  }
}
