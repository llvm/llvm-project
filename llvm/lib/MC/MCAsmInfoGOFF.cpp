//===- MCAsmInfoGOFF.cpp - MCGOFFAsmInfo properties -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines certain target specific asm properties for GOFF (z/OS)
/// based targets.
///
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCAsmInfoGOFF.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

MCAsmInfoGOFF::MCAsmInfoGOFF(const MCTargetOptions &Options)
    : MCAsmInfo(Options) {
  Data64bitsDirective = "\t.quad\t";
  WeakRefDirective = "WXTRN";
  InternalSymbolPrefix = "L#";
  ZeroDirective = "\t.space\t";
}

void MCAsmInfoGOFF::printSwitchToSection(const MCSection &, uint32_t,
                                         const Triple &, raw_ostream &) const {
  llvm_unreachable("GOFF section switching is handled by the HLASM streamer");
}
