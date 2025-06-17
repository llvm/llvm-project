//===-- MipsMCAsmInfo.cpp - Mips Asm Properties ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the MipsMCAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "MipsMCAsmInfo.h"
#include "MipsABIInfo.h"
#include "llvm/Support/Casting.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

void MipsELFMCAsmInfo::anchor() {}

MipsELFMCAsmInfo::MipsELFMCAsmInfo(const Triple &TheTriple,
                                   const MCTargetOptions &Options) {
  IsLittleEndian = TheTriple.isLittleEndian();

  MipsABIInfo ABI = MipsABIInfo::computeTargetABI(TheTriple, "", Options);

  if (TheTriple.isMIPS64() && !ABI.IsN32())
    CodePointerSize = CalleeSaveStackSlotSize = 8;

  if (ABI.IsO32())
    PrivateGlobalPrefix = "$";
  else if (ABI.IsN32() || ABI.IsN64())
    PrivateGlobalPrefix = ".L";
  PrivateLabelPrefix = PrivateGlobalPrefix;

  AlignmentIsInBytes          = false;
  Data16bitsDirective         = "\t.2byte\t";
  Data32bitsDirective         = "\t.4byte\t";
  Data64bitsDirective         = "\t.8byte\t";
  CommentString               = "#";
  ZeroDirective               = "\t.space\t";
  UseAssignmentForEHBegin = true;
  SupportsDebugInformation = true;
  ExceptionsType = ExceptionHandling::DwarfCFI;
  DwarfRegNumForCFI = true;
}

void MipsCOFFMCAsmInfo::anchor() {}

MipsCOFFMCAsmInfo::MipsCOFFMCAsmInfo() {
  HasSingleParameterDotFile = true;
  WinEHEncodingType = WinEH::EncodingType::Itanium;

  ExceptionsType = ExceptionHandling::WinEH;

  PrivateGlobalPrefix = ".L";
  PrivateLabelPrefix = ".L";
  AllowAtInName = true;
}

static void printImpl(const MCAsmInfo &MAI, raw_ostream &OS,
                      const MCSpecifierExpr &Expr) {
  int64_t AbsVal;

  switch (Expr.getSpecifier()) {
  case Mips::S_None:
  case Mips::S_Special:
    llvm_unreachable("Mips::S_None and MEK_Special are invalid");
    break;
  case Mips::S_DTPREL:
    // Mips::S_DTPREL is used for marking TLS DIEExpr only
    // and contains a regular sub-expression.
    MAI.printExpr(OS, *Expr.getSubExpr());
    return;
  case Mips::S_CALL_HI16:
    OS << "%call_hi";
    break;
  case Mips::S_CALL_LO16:
    OS << "%call_lo";
    break;
  case Mips::S_DTPREL_HI:
    OS << "%dtprel_hi";
    break;
  case Mips::S_DTPREL_LO:
    OS << "%dtprel_lo";
    break;
  case Mips::S_GOT:
    OS << "%got";
    break;
  case Mips::S_GOTTPREL:
    OS << "%gottprel";
    break;
  case Mips::S_GOT_CALL:
    OS << "%call16";
    break;
  case Mips::S_GOT_DISP:
    OS << "%got_disp";
    break;
  case Mips::S_GOT_HI16:
    OS << "%got_hi";
    break;
  case Mips::S_GOT_LO16:
    OS << "%got_lo";
    break;
  case Mips::S_GOT_PAGE:
    OS << "%got_page";
    break;
  case Mips::S_GOT_OFST:
    OS << "%got_ofst";
    break;
  case Mips::S_GPREL:
    OS << "%gp_rel";
    break;
  case Mips::S_HI:
    OS << "%hi";
    break;
  case Mips::S_HIGHER:
    OS << "%higher";
    break;
  case Mips::S_HIGHEST:
    OS << "%highest";
    break;
  case Mips::S_LO:
    OS << "%lo";
    break;
  case Mips::S_NEG:
    OS << "%neg";
    break;
  case Mips::S_PCREL_HI16:
    OS << "%pcrel_hi";
    break;
  case Mips::S_PCREL_LO16:
    OS << "%pcrel_lo";
    break;
  case Mips::S_TLSGD:
    OS << "%tlsgd";
    break;
  case Mips::S_TLSLDM:
    OS << "%tlsldm";
    break;
  case Mips::S_TPREL_HI:
    OS << "%tprel_hi";
    break;
  case Mips::S_TPREL_LO:
    OS << "%tprel_lo";
    break;
  }

  OS << '(';
  if (Expr.evaluateAsAbsolute(AbsVal))
    OS << AbsVal;
  else
    MAI.printExpr(OS, *Expr.getSubExpr());
  OS << ')';
}

bool Mips::isGpOff(const MCSpecifierExpr &E) {
  if (E.getSpecifier() == Mips::S_HI || E.getSpecifier() == Mips::S_LO) {
    if (const auto *S1 = dyn_cast<const MCSpecifierExpr>(E.getSubExpr())) {
      if (const auto *S2 = dyn_cast<const MCSpecifierExpr>(S1->getSubExpr())) {
        if (S1->getSpecifier() == Mips::S_NEG &&
            S2->getSpecifier() == Mips::S_GPREL) {
          // S = E.getSpecifier();
          return true;
        }
      }
    }
  }
  return false;
}

static bool evaluate(const MCSpecifierExpr &Expr, MCValue &Res,
                     const MCAssembler *Asm) {
  // Look for the %hi(%neg(%gp_rel(X))) and %lo(%neg(%gp_rel(X)))
  // special cases.
  if (Mips::isGpOff(Expr)) {
    const MCExpr *SubExpr =
        cast<MCSpecifierExpr>(
            cast<MCSpecifierExpr>(Expr.getSubExpr())->getSubExpr())
            ->getSubExpr();
    if (!SubExpr->evaluateAsRelocatable(Res, Asm))
      return false;

    Res.setSpecifier(Mips::S_Special);
    return true;
  }

  if (!Expr.getSubExpr()->evaluateAsRelocatable(Res, Asm))
    return false;
  Res.setSpecifier(Expr.getSpecifier());
  return !Res.getSubSym();
}

void MipsELFMCAsmInfo::printSpecifierExpr(raw_ostream &OS,
                                          const MCSpecifierExpr &Expr) const {
  printImpl(*this, OS, Expr);
}

bool MipsELFMCAsmInfo::evaluateAsRelocatableImpl(const MCSpecifierExpr &Expr,
                                                 MCValue &Res,
                                                 const MCAssembler *Asm) const {
  return evaluate(Expr, Res, Asm);
}

void MipsCOFFMCAsmInfo::printSpecifierExpr(raw_ostream &OS,
                                           const MCSpecifierExpr &Expr) const {
  printImpl(*this, OS, Expr);
}

bool MipsCOFFMCAsmInfo::evaluateAsRelocatableImpl(
    const MCSpecifierExpr &Expr, MCValue &Res, const MCAssembler *Asm) const {
  return evaluate(Expr, Res, Asm);
}
