//===-- WebAssemblyUtilities.cpp - WebAssembly Utility Functions ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements several utility functions for WebAssembly.
///
//===----------------------------------------------------------------------===//

#include "WebAssemblyUtilities.h"
#include "WebAssemblyMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
using namespace llvm;

const char *const WebAssembly::ClangCallTerminateFn = "__clang_call_terminate";
const char *const WebAssembly::CxaBeginCatchFn = "__cxa_begin_catch";
const char *const WebAssembly::CxaRethrowFn = "__cxa_rethrow";
const char *const WebAssembly::StdTerminateFn = "_ZSt9terminatev";
const char *const WebAssembly::PersonalityWrapperFn =
    "_Unwind_Wasm_CallPersonality";

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

bool WebAssembly::mayThrow(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case WebAssembly::THROW:
  case WebAssembly::THROW_S:
  case WebAssembly::RETHROW:
  case WebAssembly::RETHROW_S:
    return true;
  }
  if (isCallIndirect(MI.getOpcode()))
    return true;
  if (!MI.isCall())
    return false;

  const MachineOperand &MO = MI.getOperand(getCalleeOpNo(MI.getOpcode()));
  assert(MO.isGlobal());
  const auto *F = dyn_cast<Function>(MO.getGlobal());
  if (!F)
    return true;
  if (F->doesNotThrow())
    return false;
  // These functions never throw
  if (F->getName() == CxaBeginCatchFn || F->getName() == PersonalityWrapperFn ||
      F->getName() == ClangCallTerminateFn || F->getName() == StdTerminateFn)
    return false;

  // TODO Can we exclude call instructions that are marked as 'nounwind' in the
  // original LLVm IR? (Even when the callee may throw)
  return true;
}
