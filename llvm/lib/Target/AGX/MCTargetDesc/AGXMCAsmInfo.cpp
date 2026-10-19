//===-- AGXMCAsmInfo.cpp - AGX asm properties -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AGXMCAsmInfo.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

void AGXMCAsmInfo::anchor() {}

AGXMCAsmInfo::AGXMCAsmInfo(const Triple &TT, const MCTargetOptions &Options)
    : MCAsmInfoELF(Options) {
  CodePointerSize = 8;
  CalleeSaveStackSlotSize = 4;
  IsLittleEndian = true;
  CommentString = "//";
  InternalSymbolPrefix = ".L";
  // Instructions are 2 to 12 bytes long, in units of 2 bytes.
  MinInstAlignment = 2;
  MaxInstLength = 12;
  SupportsDebugInformation = true;
  UsesELFSectionDirectiveForBSS = true;
  HasDotTypeDotSizeDirective = true;
  HasSingleParameterDotFile = true;
}
