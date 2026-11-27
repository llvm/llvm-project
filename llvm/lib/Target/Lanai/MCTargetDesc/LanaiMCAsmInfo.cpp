//===-- LanaiMCAsmInfo.cpp - Lanai asm properties -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the LanaiMCAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "LanaiMCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

void LanaiMCAsmInfo::anchor() {}

LanaiMCAsmInfo::LanaiMCAsmInfo(const Triple &T, const MCTargetOptions &Options)
    : MCAsmInfoELF(T, Options) {
  IsLittleEndian = false;
  InternalSymbolPrefix = ".L";
  WeakRefDirective = "\t.weak\t";

  // Use '!' as comment string to correspond with old toolchain.
  CommentString = "!";

  // Target supports emission of debugging information.
  SupportsDebugInformation = true;

  // Set the instruction alignment. Currently used only for address adjustment
  // in dwarf generation.
  MinInstAlignment = 4;

  SpecifierExprPrintFn = [](const MCAsmInfo &MAI, raw_ostream &OS,
                            const MCSpecifierExpr &Expr) {
    if (Expr.getSpecifier() == 0) {
      MAI.printExpr(OS, *Expr.getSubExpr());
      return;
    }

    switch (Expr.getSpecifier()) {
    default:
      llvm_unreachable("Invalid kind!");
    case Lanai::S_ABS_HI:
      OS << "hi";
      break;
    case Lanai::S_ABS_LO:
      OS << "lo";
      break;
    }

    OS << '(';
    MAI.printExpr(OS, *Expr.getSubExpr());
    OS << ')';
  };
}
