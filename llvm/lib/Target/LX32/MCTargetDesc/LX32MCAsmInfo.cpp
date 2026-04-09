//===-- LX32MCAsmInfo.cpp - LX32 MC Assembler Information ----------------===//
//
// Part of the LX32 Project
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// This file defines LX32 MCAsmInfo defaults used by assembly/object emission.
// It is organized into the following sections:
//
//   Section 0 — Class anchor
//   Section 1 — Constructor defaults and ABI choices
//
//===----------------------------------------------------------------------===//

#include "LX32MCAsmInfo.h"
#include "llvm/TargetParser/Triple.h"

void LX32MCAsmInfo::anchor() {}

//===----------------------------------------------------------------------===//
// Section 1 — Constructor defaults and ABI choices
//===----------------------------------------------------------------------===//

LX32MCAsmInfo::LX32MCAsmInfo(const llvm::Triple &TT) {
  // LX32 is little-endian and uses 32-bit pointers/stack slots.
  IsLittleEndian = true;
  CodePointerSize = 4;
  CalleeSaveStackSlotSize = 4;

  // GNU-style assembler comments.
  CommentString = "#";
  AlignmentIsInBytes = false;
  SupportsDebugInformation = true;
  // Must match Triple default; CodeGenTargetMachineImpl::initAsmInfo asserts
  // if MCAsmInfo and Triple disagree.
  ExceptionsType = TT.getDefaultExceptionHandling();
  Data16bitsDirective = "\t.half\t";
  Data32bitsDirective = "\t.word\t";
}