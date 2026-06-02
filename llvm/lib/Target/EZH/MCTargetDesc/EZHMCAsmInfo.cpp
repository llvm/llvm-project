//===-- EZHMCAsmInfo.cpp - EZH asm properties -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the EZHMCAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "EZHMCAsmInfo.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

void EZHMCAsmInfo::anchor() {}

EZHMCAsmInfo::EZHMCAsmInfo(const Triple & /*TheTriple*/,
                           const MCTargetOptions &Options)
    : MCAsmInfoELF(Options) {
  IsLittleEndian = true;
  InternalSymbolPrefix = ".L";
  WeakRefDirective = "\t.weak\t";
  ExceptionsType = ExceptionHandling::SjLj;

  // EZH assembly requires ".section" before ".bss"
  UsesELFSectionDirectiveForBSS = true;

  // Use ';' as comment string.
  CommentString = ";";

  // Target supports emission of debugging information.
  SupportsDebugInformation = true;

  // Set the instruction alignment.
  MinInstAlignment = 4;

  // DWARF initial frame state: CFA = SP (reg 12) + 0
  addInitialFrameState(MCCFIInstruction::cfiDefCfa(nullptr, 12, 0));
}
