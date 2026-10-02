//===-- AMDGPUDwarfDebug.cpp - AMDGPU DwarfDebug Implementation -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUDwarfDebug.h"
#include "SIMachineFunctionInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/MC/MCDwarf.h"

using namespace llvm;

void AMDGPUDwarfDebug::beginInstruction(const MachineInstr *MI) {
  recordFusedSourceLine(*MI);
  DwarfDebug::beginInstruction(MI);
}

void AMDGPUDwarfDebug::recordFusedSourceLine(const MachineInstr &MI) {
  if (!Asm->hasDebugInfo() || MI.getFlag(MachineInstr::FrameSetup))
    return;

  const MachineFunction &MF = *MI.getMF();
  const DISubprogram *SP = MF.getFunction().getSubprogram();
  if (!SP || SP->getUnit()->getEmissionKind() == DICompileUnit::NoDebug)
    return;

  DebugLoc DL = MF.getInfo<SIMachineFunctionInfo>()->getFusedDebugLoc(MI);
  if (!DL || !AMDGPU::isVOPD(MI.getOpcode()))
    return;

  // Same rule as the non-Key-Instructions case of
  // DwarfDebug::beginInstruction: a new line is a new statement.
  // FIXME: With Key Instructions, is_stmt is decided per instruction, and the
  // instruction fused into MI was erased before the key instructions were
  // computed. If it was the key instruction of its atom, that atom gets no
  // is_stmt.
  unsigned Flags = 0;
  if (!DL->getScope()->getSubprogram()->getKeyInstructionsEnabled() &&
      (!PrevInstLoc || PrevInstLoc.getLine() != DL.getLine()))
    Flags |= DWARF2_FLAG_IS_STMT;

  recordTargetSourceLine(DL, Flags);

  // DwarfDebug::beginInstruction compares against this, so the instruction's
  // own location is emitted after this row even if it matches the location of
  // the previous instruction.
  PrevInstLoc = DL;
}
