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
#include "llvm/MC/MCContext.h"
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
  Register Reg = MO.getReg();
  return Register::isVirtualRegister(Reg) && MFI.isVRegStackified(Reg);
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

  const MachineOperand &MO = getCalleeOp(MI);
  assert(MO.isGlobal() || MO.isSymbol());

  if (MO.isSymbol()) {
    // Some intrinsics are lowered to calls to external symbols, which are then
    // lowered to calls to library functions. Most of libcalls don't throw, but
    // we only list some of them here now.
    // TODO Consider adding 'nounwind' info in TargetLowering::CallLoweringInfo
    // instead for more accurate info.
    const char *Name = MO.getSymbolName();
    if (strcmp(Name, "memcpy") == 0 || strcmp(Name, "memmove") == 0 ||
        strcmp(Name, "memset") == 0)
      return false;
    return true;
  }

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

const MachineOperand &WebAssembly::getCalleeOp(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case WebAssembly::CALL:
  case WebAssembly::CALL_S:
  case WebAssembly::RET_CALL:
  case WebAssembly::RET_CALL_S:
    return MI.getOperand(MI.getNumExplicitDefs());
  case WebAssembly::CALL_INDIRECT:
  case WebAssembly::CALL_INDIRECT_S:
  case WebAssembly::RET_CALL_INDIRECT:
  case WebAssembly::RET_CALL_INDIRECT_S:
    return MI.getOperand(MI.getNumOperands() - 1);
  default:
    llvm_unreachable("Not a call instruction");
  }
}

MCSymbolWasm *WebAssembly::getOrCreateFunctionTableSymbol(
    MCContext &Ctx, const WebAssemblySubtarget *Subtarget) {
  StringRef Name = "__indirect_function_table";
  MCSymbolWasm *Sym = cast_or_null<MCSymbolWasm>(Ctx.lookupSymbol(Name));
  if (Sym) {
    if (!Sym->isFunctionTable())
      Ctx.reportError(SMLoc(), "symbol is not a wasm funcref table");
  } else {
    Sym = cast<MCSymbolWasm>(Ctx.getOrCreateSymbol(Name));
    Sym->setFunctionTable();
    // The default function table is synthesized by the linker.
    Sym->setUndefined();
  }
  // MVP object files can't have symtab entries for tables.
  if (!(Subtarget && Subtarget->hasReferenceTypes()))
    Sym->setOmitFromLinkingSection();
  return Sym;
}

// Find a catch instruction from an EH pad.
MachineInstr *WebAssembly::findCatch(MachineBasicBlock *EHPad) {
  assert(EHPad->isEHPad());
  auto Pos = EHPad->begin();
  // Skip any label or debug instructions. Also skip 'end' marker instructions
  // that may exist after marker placement in CFGStackify.
  while (Pos != EHPad->end() &&
         (Pos->isLabel() || Pos->isDebugInstr() || isMarker(Pos->getOpcode())))
    Pos++;
  if (Pos != EHPad->end() && WebAssembly::isCatch(Pos->getOpcode()))
    return &*Pos;
  return nullptr;
}
