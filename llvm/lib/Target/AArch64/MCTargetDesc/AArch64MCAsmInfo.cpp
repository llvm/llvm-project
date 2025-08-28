//===-- AArch64MCAsmInfo.cpp - AArch64 asm properties ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the AArch64MCAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "AArch64MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/TargetParser/Triple.h"
using namespace llvm;

enum AsmWriterVariantTy {
  Default = -1,
  Generic = 0,
  Apple = 1
};

static cl::opt<AsmWriterVariantTy> AsmWriterVariant(
    "aarch64-neon-syntax", cl::init(Default),
    cl::desc("Choose style of NEON code to emit from AArch64 backend:"),
    cl::values(clEnumValN(Generic, "generic", "Emit generic NEON assembly"),
               clEnumValN(Apple, "apple", "Emit Apple-style NEON assembly")));

const MCAsmInfo::AtSpecifier COFFAtSpecifiers[] = {
    {MCSymbolRefExpr::VK_COFF_IMGREL32, "IMGREL"},
    {AArch64::S_MACHO_PAGEOFF, "PAGEOFF"},
};

const MCAsmInfo::AtSpecifier ELFAtSpecifiers[] = {
    {AArch64::S_GOT, "GOT"},
    {AArch64::S_GOTPCREL, "GOTPCREL"},
    {AArch64::S_PLT, "PLT"},
};

const MCAsmInfo::AtSpecifier MachOAtSpecifiers[] = {
    {AArch64::S_MACHO_GOT, "GOT"},
    {AArch64::S_MACHO_GOTPAGE, "GOTPAGE"},
    {AArch64::S_MACHO_GOTPAGEOFF, "GOTPAGEOFF"},
    {AArch64::S_MACHO_PAGE, "PAGE"},
    {AArch64::S_MACHO_PAGEOFF, "PAGEOFF"},
    {AArch64::S_MACHO_TLVP, "TLVP"},
    {AArch64::S_MACHO_TLVPPAGE, "TLVPPAGE"},
    {AArch64::S_MACHO_TLVPPAGEOFF, "TLVPPAGEOFF"},
};

StringRef AArch64::getSpecifierName(const MCSpecifierExpr &Expr) {
  // clang-format off
  switch (static_cast<uint32_t>(Expr.getSpecifier())) {
  case AArch64::S_CALL:                return "";
  case AArch64::S_LO12:                return ":lo12:";
  case AArch64::S_ABS_G3:              return ":abs_g3:";
  case AArch64::S_ABS_G2:              return ":abs_g2:";
  case AArch64::S_ABS_G2_S:            return ":abs_g2_s:";
  case AArch64::S_ABS_G2_NC:           return ":abs_g2_nc:";
  case AArch64::S_ABS_G1:              return ":abs_g1:";
  case AArch64::S_ABS_G1_S:            return ":abs_g1_s:";
  case AArch64::S_ABS_G1_NC:           return ":abs_g1_nc:";
  case AArch64::S_ABS_G0:              return ":abs_g0:";
  case AArch64::S_ABS_G0_S:            return ":abs_g0_s:";
  case AArch64::S_ABS_G0_NC:           return ":abs_g0_nc:";
  case AArch64::S_PREL_G3:             return ":prel_g3:";
  case AArch64::S_PREL_G2:             return ":prel_g2:";
  case AArch64::S_PREL_G2_NC:          return ":prel_g2_nc:";
  case AArch64::S_PREL_G1:             return ":prel_g1:";
  case AArch64::S_PREL_G1_NC:          return ":prel_g1_nc:";
  case AArch64::S_PREL_G0:             return ":prel_g0:";
  case AArch64::S_PREL_G0_NC:          return ":prel_g0_nc:";
  case AArch64::S_DTPREL_G2:           return ":dtprel_g2:";
  case AArch64::S_DTPREL_G1:           return ":dtprel_g1:";
  case AArch64::S_DTPREL_G1_NC:        return ":dtprel_g1_nc:";
  case AArch64::S_DTPREL_G0:           return ":dtprel_g0:";
  case AArch64::S_DTPREL_G0_NC:        return ":dtprel_g0_nc:";
  case AArch64::S_DTPREL_HI12:         return ":dtprel_hi12:";
  case AArch64::S_DTPREL_LO12:         return ":dtprel_lo12:";
  case AArch64::S_DTPREL_LO12_NC:      return ":dtprel_lo12_nc:";
  case AArch64::S_TPREL_G2:            return ":tprel_g2:";
  case AArch64::S_TPREL_G1:            return ":tprel_g1:";
  case AArch64::S_TPREL_G1_NC:         return ":tprel_g1_nc:";
  case AArch64::S_TPREL_G0:            return ":tprel_g0:";
  case AArch64::S_TPREL_G0_NC:         return ":tprel_g0_nc:";
  case AArch64::S_TPREL_HI12:          return ":tprel_hi12:";
  case AArch64::S_TPREL_LO12:          return ":tprel_lo12:";
  case AArch64::S_TPREL_LO12_NC:       return ":tprel_lo12_nc:";
  case AArch64::S_TLSDESC_LO12:        return ":tlsdesc_lo12:";
  case AArch64::S_TLSDESC_AUTH_LO12:   return ":tlsdesc_auth_lo12:";
  case AArch64::S_ABS_PAGE:            return "";
  case AArch64::S_ABS_PAGE_NC:         return ":pg_hi21_nc:";
  case AArch64::S_GOT:                 return ":got:";
  case AArch64::S_GOT_PAGE:            return ":got:";
  case AArch64::S_GOT_PAGE_LO15:       return ":gotpage_lo15:";
  case AArch64::S_GOT_LO12:            return ":got_lo12:";
  case AArch64::S_GOTTPREL:            return ":gottprel:";
  case AArch64::S_GOTTPREL_PAGE:       return ":gottprel:";
  case AArch64::S_GOTTPREL_LO12_NC:    return ":gottprel_lo12:";
  case AArch64::S_GOTTPREL_G1:         return ":gottprel_g1:";
  case AArch64::S_GOTTPREL_G0_NC:      return ":gottprel_g0_nc:";
  case AArch64::S_TLSDESC:             return "";
  case AArch64::S_TLSDESC_PAGE:        return ":tlsdesc:";
  case AArch64::S_TLSDESC_AUTH:        return "";
  case AArch64::S_TLSDESC_AUTH_PAGE:   return ":tlsdesc_auth:";
  case AArch64::S_SECREL_LO12:         return ":secrel_lo12:";
  case AArch64::S_SECREL_HI12:         return ":secrel_hi12:";
  case AArch64::S_GOT_AUTH:            return ":got_auth:";
  case AArch64::S_GOT_AUTH_PAGE:       return ":got_auth:";
  case AArch64::S_GOT_AUTH_LO12:       return ":got_auth_lo12:";
  default:
    llvm_unreachable("Invalid relocation specifier");
  }
  // clang-format on
}

static bool evaluate(const MCSpecifierExpr &Expr, MCValue &Res,
                     const MCAssembler *Asm) {
  if (!Expr.getSubExpr()->evaluateAsRelocatable(Res, Asm))
    return false;
  Res.setSpecifier(Expr.getSpecifier());
  return !Res.getSubSym();
}

AArch64MCAsmInfoDarwin::AArch64MCAsmInfoDarwin(bool IsILP32) {
  // We prefer NEON instructions to be printed in the short, Apple-specific
  // form when targeting Darwin.
  AssemblerDialect = AsmWriterVariant == Default ? Apple : AsmWriterVariant;

  PrivateGlobalPrefix = "L";
  PrivateLabelPrefix = "L";
  SeparatorString = "%%";
  CommentString = ";";
  CalleeSaveStackSlotSize = 8;
  CodePointerSize = IsILP32 ? 4 : 8;

  AlignmentIsInBytes = false;
  UsesELFSectionDirectiveForBSS = true;
  SupportsDebugInformation = true;
  UseDataRegionDirectives = true;
  UseAtForSpecifier = false;

  ExceptionsType = ExceptionHandling::DwarfCFI;

  initializeAtSpecifiers(MachOAtSpecifiers);
}

const MCExpr *AArch64MCAsmInfoDarwin::getExprForPersonalitySymbol(
    const MCSymbol *Sym, unsigned Encoding, MCStreamer &Streamer) const {
  // On Darwin, we can reference dwarf symbols with foo@GOT-., which
  // is an indirect pc-relative reference. The default implementation
  // won't reference using the GOT, so we need this target-specific
  // version.
  MCContext &Context = Streamer.getContext();
  const MCExpr *Res =
      MCSymbolRefExpr::create(Sym, AArch64::S_MACHO_GOT, Context);
  MCSymbol *PCSym = Context.createTempSymbol();
  Streamer.emitLabel(PCSym);
  const MCExpr *PC = MCSymbolRefExpr::create(PCSym, Context);
  return MCBinaryExpr::createSub(Res, PC, Context);
}

void AArch64AuthMCExpr::print(raw_ostream &OS, const MCAsmInfo *MAI) const {
  bool WrapSubExprInParens = !isa<MCSymbolRefExpr>(getSubExpr());
  if (WrapSubExprInParens)
    OS << '(';
  MAI->printExpr(OS, *getSubExpr());
  if (WrapSubExprInParens)
    OS << ')';

  OS << "@AUTH(" << AArch64PACKeyIDToString(Key) << ',' << Discriminator;
  if (hasAddressDiversity())
    OS << ",addr";
  OS << ')';
}

void AArch64MCAsmInfoDarwin::printSpecifierExpr(
    raw_ostream &OS, const MCSpecifierExpr &Expr) const {
  if (auto *AE = dyn_cast<AArch64AuthMCExpr>(&Expr))
    return AE->print(OS, this);
  OS << AArch64::getSpecifierName(Expr);
  printExpr(OS, *Expr.getSubExpr());
}

bool AArch64MCAsmInfoDarwin::evaluateAsRelocatableImpl(
    const MCSpecifierExpr &Expr, MCValue &Res, const MCAssembler *Asm) const {
  return evaluate(Expr, Res, Asm);
}

AArch64MCAsmInfoELF::AArch64MCAsmInfoELF(const Triple &T) {
  if (T.getArch() == Triple::aarch64_be)
    IsLittleEndian = false;

  // We prefer NEON instructions to be printed in the generic form when
  // targeting ELF.
  AssemblerDialect = AsmWriterVariant == Default ? Generic : AsmWriterVariant;

  CodePointerSize = T.getEnvironment() == Triple::GNUILP32 ? 4 : 8;

  // ".comm align is in bytes but .align is pow-2."
  AlignmentIsInBytes = false;

  CommentString = "//";
  PrivateGlobalPrefix = ".L";
  PrivateLabelPrefix = ".L";

  Data16bitsDirective = "\t.hword\t";
  Data32bitsDirective = "\t.word\t";
  Data64bitsDirective = "\t.xword\t";

  UseDataRegionDirectives = false;
  UseAtForSpecifier = false;

  WeakRefDirective = "\t.weak\t";

  SupportsDebugInformation = true;

  // Exceptions handling
  ExceptionsType = ExceptionHandling::DwarfCFI;

  HasIdentDirective = true;

  initializeAtSpecifiers(ELFAtSpecifiers);
}

void AArch64MCAsmInfoELF::printSpecifierExpr(
    raw_ostream &OS, const MCSpecifierExpr &Expr) const {
  if (auto *AE = dyn_cast<AArch64AuthMCExpr>(&Expr))
    return AE->print(OS, this);
  OS << AArch64::getSpecifierName(Expr);
  printExpr(OS, *Expr.getSubExpr());
}

bool AArch64MCAsmInfoELF::evaluateAsRelocatableImpl(
    const MCSpecifierExpr &Expr, MCValue &Res, const MCAssembler *Asm) const {
  return evaluate(Expr, Res, Asm);
}

AArch64MCAsmInfoMicrosoftCOFF::AArch64MCAsmInfoMicrosoftCOFF() {
  PrivateGlobalPrefix = ".L";
  PrivateLabelPrefix = ".L";

  Data16bitsDirective = "\t.hword\t";
  Data32bitsDirective = "\t.word\t";
  Data64bitsDirective = "\t.xword\t";

  AlignmentIsInBytes = false;
  SupportsDebugInformation = true;
  CodePointerSize = 8;

  CommentString = "//";
  ExceptionsType = ExceptionHandling::WinEH;
  WinEHEncodingType = WinEH::EncodingType::Itanium;

  initializeAtSpecifiers(COFFAtSpecifiers);
}

void AArch64MCAsmInfoMicrosoftCOFF::printSpecifierExpr(
    raw_ostream &OS, const MCSpecifierExpr &Expr) const {
  OS << AArch64::getSpecifierName(Expr);
  printExpr(OS, *Expr.getSubExpr());
}

bool AArch64MCAsmInfoMicrosoftCOFF::evaluateAsRelocatableImpl(
    const MCSpecifierExpr &Expr, MCValue &Res, const MCAssembler *Asm) const {
  return evaluate(Expr, Res, Asm);
}

AArch64MCAsmInfoGNUCOFF::AArch64MCAsmInfoGNUCOFF() {
  PrivateGlobalPrefix = ".L";
  PrivateLabelPrefix = ".L";

  Data16bitsDirective = "\t.hword\t";
  Data32bitsDirective = "\t.word\t";
  Data64bitsDirective = "\t.xword\t";

  AlignmentIsInBytes = false;
  SupportsDebugInformation = true;
  CodePointerSize = 8;

  CommentString = "//";
  ExceptionsType = ExceptionHandling::WinEH;
  WinEHEncodingType = WinEH::EncodingType::Itanium;

  initializeAtSpecifiers(COFFAtSpecifiers);
}

void AArch64MCAsmInfoGNUCOFF::printSpecifierExpr(
    raw_ostream &OS, const MCSpecifierExpr &Expr) const {
  OS << AArch64::getSpecifierName(Expr);
  printExpr(OS, *Expr.getSubExpr());
}

bool AArch64MCAsmInfoGNUCOFF::evaluateAsRelocatableImpl(
    const MCSpecifierExpr &Expr, MCValue &Res, const MCAssembler *Asm) const {
  return evaluate(Expr, Res, Asm);
}
