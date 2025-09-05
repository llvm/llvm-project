//===-- ARMMCAsmInfo.cpp - ARM asm properties -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the ARMMCAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "ARMMCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

const MCAsmInfo::AtSpecifier atSpecifiers[] = {
    {ARM::S_GOT_PREL, "GOT_PREL"},
    {ARM::S_ARM_NONE, "none"},
    {ARM::S_PREL31, "prel31"},
    {ARM::S_SBREL, "sbrel"},
    {ARM::S_TARGET1, "target1"},
    {ARM::S_TARGET2, "target2"},
    {ARM::S_TLSLDO, "TLSLDO"},
    {MCSymbolRefExpr::VK_COFF_IMGREL32, "imgrel"},
    {ARM::S_FUNCDESC, "FUNCDESC"},
    {ARM::S_GOT, "GOT"},
    {ARM::S_GOTFUNCDESC, "GOTFUNCDESC"},
    {ARM::S_GOTOFF, "GOTOFF"},
    {ARM::S_GOTOFFFUNCDESC, "GOTOFFFUNCDESC"},
    {ARM::S_GOTTPOFF, "GOTTPOFF"},
    {ARM::S_GOTTPOFF_FDPIC, "gottpoff_fdpic"},
    {ARM::S_PLT, "PLT"},
    {ARM::S_COFF_SECREL, "SECREL32"},
    {ARM::S_TLSCALL, "tlscall"},
    {ARM::S_TLSDESC, "tlsdesc"},
    {ARM::S_TLSGD, "TLSGD"},
    {ARM::S_TLSGD_FDPIC, "tlsgd_fdpic"},
    {ARM::S_TLSLDM, "TLSLDM"},
    {ARM::S_TLSLDM_FDPIC, "tlsldm_fdpic"},
    {ARM::S_TPOFF, "TPOFF"},
};

void ARMMCAsmInfoDarwin::anchor() { }

ARMMCAsmInfoDarwin::ARMMCAsmInfoDarwin(const Triple &TheTriple) {
  if ((TheTriple.getArch() == Triple::armeb) ||
      (TheTriple.getArch() == Triple::thumbeb))
    IsLittleEndian = false;

  Data64bitsDirective = nullptr;
  CommentString = "@";
  AllowDollarAtStartOfIdentifier = false;
  UseDataRegionDirectives = true;

  SupportsDebugInformation = true;

  // Conditional Thumb 4-byte instructions can have an implicit IT.
  MaxInstLength = 6;

  // Exceptions handling
  ExceptionsType = (TheTriple.isOSDarwin() && !TheTriple.isWatchABI())
                       ? ExceptionHandling::SjLj
                       : ExceptionHandling::DwarfCFI;

  initializeAtSpecifiers(atSpecifiers);
}

void ARMELFMCAsmInfo::anchor() { }

ARMELFMCAsmInfo::ARMELFMCAsmInfo(const Triple &TheTriple) {
  if ((TheTriple.getArch() == Triple::armeb) ||
      (TheTriple.getArch() == Triple::thumbeb))
    IsLittleEndian = false;

  // ".comm align is in bytes but .align is pow-2."
  AlignmentIsInBytes = false;

  Data64bitsDirective = nullptr;
  CommentString = "@";
  AllowDollarAtStartOfIdentifier = false;

  SupportsDebugInformation = true;

  // Conditional Thumb 4-byte instructions can have an implicit IT.
  MaxInstLength = 6;

  // Exceptions handling
  switch (TheTriple.getOS()) {
  case Triple::NetBSD:
    ExceptionsType = ExceptionHandling::DwarfCFI;
    break;
  default:
    ExceptionsType = ExceptionHandling::ARM;
    break;
  }

  // foo(plt) instead of foo@plt
  UseAtForSpecifier = false;
  UseParensForSpecifier = true;

  initializeAtSpecifiers(atSpecifiers);
}

void ARMELFMCAsmInfo::setUseIntegratedAssembler(bool Value) {
  UseIntegratedAssembler = Value;
  if (!UseIntegratedAssembler) {
    // gas doesn't handle VFP register names in cfi directives,
    // so don't use register names with external assembler.
    // See https://sourceware.org/bugzilla/show_bug.cgi?id=16694
    DwarfRegNumForCFI = true;
  }
}

void ARMCOFFMCAsmInfoMicrosoft::anchor() { }

ARMCOFFMCAsmInfoMicrosoft::ARMCOFFMCAsmInfoMicrosoft() {
  AlignmentIsInBytes = false;
  SupportsDebugInformation = true;
  ExceptionsType = ExceptionHandling::WinEH;
  WinEHEncodingType = WinEH::EncodingType::Itanium;
  PrivateGlobalPrefix = "$M";
  PrivateLabelPrefix = "$M";
  CommentString = "@";

  // Conditional Thumb 4-byte instructions can have an implicit IT.
  MaxInstLength = 6;

  initializeAtSpecifiers(atSpecifiers);
}

void ARMCOFFMCAsmInfoGNU::anchor() { }

ARMCOFFMCAsmInfoGNU::ARMCOFFMCAsmInfoGNU() {
  AlignmentIsInBytes = false;
  HasSingleParameterDotFile = true;

  CommentString = "@";
  AllowDollarAtStartOfIdentifier = false;
  PrivateGlobalPrefix = ".L";
  PrivateLabelPrefix = ".L";

  SupportsDebugInformation = true;
  ExceptionsType = ExceptionHandling::WinEH;
  WinEHEncodingType = WinEH::EncodingType::Itanium;
  UseAtForSpecifier = false;
  UseParensForSpecifier = true;

  DwarfRegNumForCFI = false;

  // Conditional Thumb 4-byte instructions can have an implicit IT.
  MaxInstLength = 6;

  initializeAtSpecifiers(atSpecifiers);
}

void ARM::printSpecifierExpr(const MCAsmInfo &MAI, raw_ostream &OS,
                             const MCSpecifierExpr &Expr) {
  switch (Expr.getSpecifier()) {
  default:
    llvm_unreachable("Invalid kind!");
  case ARM::S_HI16:
    OS << ":upper16:";
    break;
  case ARM::S_LO16:
    OS << ":lower16:";
    break;
  case ARM::S_HI_8_15:
    OS << ":upper8_15:";
    break;
  case ARM::S_HI_0_7:
    OS << ":upper0_7:";
    break;
  case ARM::S_LO_8_15:
    OS << ":lower8_15:";
    break;
  case ARM::S_LO_0_7:
    OS << ":lower0_7:";
    break;
  }

  const MCExpr *Sub = Expr.getSubExpr();
  if (Sub->getKind() != MCExpr::SymbolRef)
    OS << '(';
  MAI.printExpr(OS, *Sub);
  if (Sub->getKind() != MCExpr::SymbolRef)
    OS << ')';
}

const MCSpecifierExpr *ARM::createUpper16(const MCExpr *Expr, MCContext &Ctx) {
  return MCSpecifierExpr::create(Expr, ARM::S_HI16, Ctx);
}

const MCSpecifierExpr *ARM::createLower16(const MCExpr *Expr, MCContext &Ctx) {
  return MCSpecifierExpr::create(Expr, ARM::S_LO16, Ctx);
}

const MCSpecifierExpr *ARM::createUpper8_15(const MCExpr *Expr,
                                            MCContext &Ctx) {
  return MCSpecifierExpr::create(Expr, ARM::S_HI_8_15, Ctx);
}

const MCSpecifierExpr *ARM::createUpper0_7(const MCExpr *Expr, MCContext &Ctx) {
  return MCSpecifierExpr::create(Expr, ARM::S_HI_0_7, Ctx);
}

const MCSpecifierExpr *ARM::createLower8_15(const MCExpr *Expr,
                                            MCContext &Ctx) {
  return MCSpecifierExpr::create(Expr, ARM::S_LO_8_15, Ctx);
}

const MCSpecifierExpr *ARM::createLower0_7(const MCExpr *Expr, MCContext &Ctx) {
  return MCSpecifierExpr::create(Expr, ARM::S_LO_0_7, Ctx);
}
