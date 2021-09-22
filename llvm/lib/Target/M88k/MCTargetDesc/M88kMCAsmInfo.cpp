//===-- M88kMCAsmInfo.cpp - M88k asm properties ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "M88kMCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSectionELF.h"

using namespace llvm;

M88kMCAsmInfo::M88kMCAsmInfo(const Triple &TT) {
  // TODO: Check!
  CodePointerSize = 4;
  CalleeSaveStackSlotSize = 4;
  IsLittleEndian = false;
  UseDotAlignForAlignment = true;
  MinInstAlignment = 4;

  CommentString = "|"; // # as comment delimiter is only allowed at first column
  ZeroDirective = "\t.space\t";
  Data64bitsDirective = "\t.quad\t";
  UsesELFSectionDirectiveForBSS = true;
  SupportsDebugInformation = false;
  ExceptionsType = ExceptionHandling::SjLj;
}
