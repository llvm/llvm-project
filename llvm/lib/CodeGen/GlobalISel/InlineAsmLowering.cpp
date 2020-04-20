//===-- lib/CodeGen/GlobalISel/InlineAsmLowering.cpp ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the lowering from LLVM IR inline asm to MIR INLINEASM
///
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GlobalISel/InlineAsmLowering.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

#define DEBUG_TYPE "inline-asm-lowering"

using namespace llvm;

void InlineAsmLowering::anchor() {}

bool InlineAsmLowering::lowerInlineAsm(MachineIRBuilder &MIRBuilder,
                                       const CallBase &Call) const {

  const InlineAsm *IA = cast<InlineAsm>(Call.getCalledValue());
  StringRef ConstraintStr = IA->getConstraintString();

  bool HasOnlyMemoryClobber = false;
  if (!ConstraintStr.empty()) {
    // Until we have full inline assembly support, we just try to handle the
    // very simple case of just "~{memory}" to avoid falling back so often.
    if (ConstraintStr != "~{memory}")
      return false;
    HasOnlyMemoryClobber = true;
  }

  unsigned ExtraInfo = 0;
  if (IA->hasSideEffects())
    ExtraInfo |= InlineAsm::Extra_HasSideEffects;
  if (IA->getDialect() == InlineAsm::AD_Intel)
    ExtraInfo |= InlineAsm::Extra_AsmDialect;

  // HACK: special casing for ~memory.
  if (HasOnlyMemoryClobber)
    ExtraInfo |= (InlineAsm::Extra_MayLoad | InlineAsm::Extra_MayStore);

  auto Inst = MIRBuilder.buildInstr(TargetOpcode::INLINEASM)
                  .addExternalSymbol(IA->getAsmString().c_str())
                  .addImm(ExtraInfo);
  if (const MDNode *SrcLoc = Call.getMetadata("srcloc"))
    Inst.addMetadata(SrcLoc);

  return true;
}
