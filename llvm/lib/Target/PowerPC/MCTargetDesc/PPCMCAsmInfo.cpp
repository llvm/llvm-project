//===-- PPCMCAsmInfo.cpp - PPC asm properties -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the MCAsmInfoDarwin properties.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/PPCMCAsmInfo.h"
#include "llvm/ADT/Enum.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

void PPCELFMCAsmInfo::anchor() { }

constexpr EnumStringDef<MCAsmInfo::AtSpecifierKind> ELFAtSpecifierDefs[] = {
    {{"DTPREL"}, PPC::S_DTPREL},
    {{"GOT"}, PPC::S_GOT},
    {{"got@ha"}, PPC::S_GOT_HA},
    {{"got@h"}, PPC::S_GOT_HI},
    {{"got@l"}, PPC::S_GOT_LO},
    {{"ha"}, PPC::S_HA},
    {{"h"}, PPC::S_HI},
    {{"high"}, PPC::S_HIGH},
    {{"higha"}, PPC::S_HIGHA},
    {{"higher"}, PPC::S_HIGHER},
    {{"highera"}, PPC::S_HIGHERA},
    {{"highest"}, PPC::S_HIGHEST},
    {{"highesta"}, PPC::S_HIGHESTA},
    {{"l"}, PPC::S_LO},
    {{"PCREL"}, PPC::S_PCREL},
    {{"PLT"}, PPC::S_PLT},
    {{"tlsgd"}, PPC::S_TLSGD},
    {{"tlsld"}, PPC::S_TLSLD},
    {{"toc"}, PPC::S_TOC},
    {{"tocbase"}, PPC::S_TOCBASE},
    {{"toc@ha"}, PPC::S_TOC_HA},
    {{"toc@h"}, PPC::S_TOC_HI},
    {{"toc@l"}, PPC::S_TOC_LO},
    {{"TPREL"}, PPC::S_TPREL},
    {{"gd"}, PPC::S_AIX_TLSGD},
    {{"m"}, PPC::S_AIX_TLSGDM},
    {{"ie"}, PPC::S_AIX_TLSIE},
    {{"ld"}, PPC::S_AIX_TLSLD},
    {{"le"}, PPC::S_AIX_TLSLE},
    {{"ml"}, PPC::S_AIX_TLSML},
    {{"dtpmod"}, PPC::S_DTPMOD},
    {{"dtprel@ha"}, PPC::S_DTPREL_HA},
    {{"dtprel@h"}, PPC::S_DTPREL_HI},
    {{"dtprel@high"}, PPC::S_DTPREL_HIGH},
    {{"dtprel@higha"}, PPC::S_DTPREL_HIGHA},
    {{"dtprel@higher"}, PPC::S_DTPREL_HIGHER},
    {{"dtprel@highera"}, PPC::S_DTPREL_HIGHERA},
    {{"dtprel@highest"}, PPC::S_DTPREL_HIGHEST},
    {{"dtprel@highesta"}, PPC::S_DTPREL_HIGHESTA},
    {{"dtprel@l"}, PPC::S_DTPREL_LO},
    {{"got@dtprel"}, PPC::S_GOT_DTPREL},
    {{"got@dtprel@ha"}, PPC::S_GOT_DTPREL_HA},
    {{"got@dtprel@h"}, PPC::S_GOT_DTPREL_HI},
    {{"got@dtprel@l"}, PPC::S_GOT_DTPREL_LO},
    {{"got@pcrel"}, PPC::S_GOT_PCREL},
    {{"got@tlsgd"}, PPC::S_GOT_TLSGD},
    {{"got@tlsgd@ha"}, PPC::S_GOT_TLSGD_HA},
    {{"got@tlsgd@h"}, PPC::S_GOT_TLSGD_HI},
    {{"got@tlsgd@l"}, PPC::S_GOT_TLSGD_LO},
    {{"got@tlsgd@pcrel"}, PPC::S_GOT_TLSGD_PCREL},
    {{"got@tlsld"}, PPC::S_GOT_TLSLD},
    {{"got@tlsld@ha"}, PPC::S_GOT_TLSLD_HA},
    {{"got@tlsld@h"}, PPC::S_GOT_TLSLD_HI},
    {{"got@tlsld@l"}, PPC::S_GOT_TLSLD_LO},
    {{"got@tlsld@pcrel"}, PPC::S_GOT_TLSLD_PCREL},
    {{"got@tprel"}, PPC::S_GOT_TPREL},
    {{"got@tprel@ha"}, PPC::S_GOT_TPREL_HA},
    {{"got@tprel@h"}, PPC::S_GOT_TPREL_HI},
    {{"got@tprel@l"}, PPC::S_GOT_TPREL_LO},
    {{"got@tprel@pcrel"}, PPC::S_GOT_TPREL_PCREL},
    {{"local"}, PPC::S_LOCAL},
    {{"notoc"}, PPC::S_NOTOC},
    {{"<<invalid>>"}, PPC::S_PCREL_OPT},
    {{"tls"}, PPC::S_TLS},
    {{"tls@pcrel"}, PPC::S_TLS_PCREL},
    {{"tprel@ha"}, PPC::S_TPREL_HA},
    {{"tprel@h"}, PPC::S_TPREL_HI},
    {{"tprel@high"}, PPC::S_TPREL_HIGH},
    {{"tprel@higha"}, PPC::S_TPREL_HIGHA},
    {{"tprel@higher"}, PPC::S_TPREL_HIGHER},
    {{"tprel@highera"}, PPC::S_TPREL_HIGHERA},
    {{"tprel@highest"}, PPC::S_TPREL_HIGHEST},
    {{"tprel@highesta"}, PPC::S_TPREL_HIGHESTA},
    {{"tprel@l"}, PPC::S_TPREL_LO},
};
constexpr auto elfAtSpecifiers = BUILD_ENUM_STRINGS(ELFAtSpecifierDefs);

constexpr EnumStringDef<MCAsmInfo::AtSpecifierKind> XCOFFAtSpecifierDefs[] = {
    // clang-format off
    {{"gd"}, PPC::S_AIX_TLSGD},
    {{"m"}, PPC::S_AIX_TLSGDM},
    {{"ie"}, PPC::S_AIX_TLSIE},
    {{"ld"}, PPC::S_AIX_TLSLD},
    {{"le"}, PPC::S_AIX_TLSLE},
    {{"ml"}, PPC::S_AIX_TLSML},
    {{"l"}, PPC::S_L},
    {{"u"}, PPC::S_U},
    // clang-format on
};
constexpr auto xcoffAtSpecifiers = BUILD_ENUM_STRINGS(XCOFFAtSpecifierDefs);

static std::optional<int64_t> evaluateAsInt64(uint16_t specifier,
                                              int64_t Value) {
  switch (specifier) {
  case PPC::S_LO:
    return Value & 0xffff;
  case PPC::S_HI:
    return (Value >> 16) & 0xffff;
  case PPC::S_HA:
    return ((Value + 0x8000) >> 16) & 0xffff;
  case PPC::S_HIGH:
    return (Value >> 16) & 0xffff;
  case PPC::S_HIGHA:
    return ((Value + 0x8000) >> 16) & 0xffff;
  case PPC::S_HIGHER:
    return (Value >> 32) & 0xffff;
  case PPC::S_HIGHERA:
    return ((Value + 0x8000) >> 32) & 0xffff;
  case PPC::S_HIGHEST:
    return (Value >> 48) & 0xffff;
  case PPC::S_HIGHESTA:
    return ((Value + 0x8000) >> 48) & 0xffff;
  default:
    return {};
  }
}

bool PPC::evaluateAsConstant(const MCSpecifierExpr &Expr, int64_t &Res) {
  MCValue Value;

  if (!Expr.getSubExpr()->evaluateAsRelocatable(Value, nullptr))
    return false;

  if (!Value.isAbsolute())
    return false;
  auto Tmp = evaluateAsInt64(Expr.getSpecifier(), Value.getConstant());
  if (!Tmp)
    return false;
  Res = *Tmp;
  return true;
}

static bool evaluateAsRelocatable(const MCSpecifierExpr &Expr, MCValue &Res,
                                  const MCAssembler *Asm) {
  if (!Expr.getSubExpr()->evaluateAsRelocatable(Res, Asm))
    return false;

  // The signedness of the result is dependent on the instruction operand. E.g.
  // in addis 3,3,65535@l, 65535@l is signed. In the absence of information at
  // parse time (!Asm), disable the folding.
  std::optional<int64_t> MaybeInt =
      evaluateAsInt64(Expr.getSpecifier(), Res.getConstant());
  if (Res.isAbsolute() && MaybeInt) {
    Res = MCValue::get(*MaybeInt);
  } else {
    Res.setSpecifier(Expr.getSpecifier());
  }

  return true;
}

PPCELFMCAsmInfo::PPCELFMCAsmInfo(bool is64Bit, const Triple &T,
                                 const MCTargetOptions &Options)
    : MCAsmInfoELF(Options) {
  // FIXME: This is not always needed. For example, it is not needed in the
  // v2 abi.
  NeedsLocalForSize = true;

  if (is64Bit) {
    CodePointerSize = CalleeSaveStackSlotSize = 8;
  }
  IsLittleEndian =
      T.getArch() == Triple::ppc64le || T.getArch() == Triple::ppcle;

  // ".comm align is in bytes but .align is pow-2."
  AlignmentIsInBytes = false;

  CommentString = "#";

  // Uses '.section' before '.bss' directive
  UsesELFSectionDirectiveForBSS = true;

  // Debug Information
  SupportsDebugInformation = true;

  DollarIsPC = true;

  // Set up DWARF directives
  MinInstAlignment = 4;

  // Exceptions handling
  ExceptionsType = ExceptionHandling::DwarfCFI;

  ZeroDirective = "\t.space\t";
  Data64bitsDirective = is64Bit ? "\t.quad\t" : nullptr;
  AssemblerDialect = 1;           // New-Style mnemonics.
  LCOMMDirectiveAlignmentType = LCOMM::ByteAlignment;

  initializeAtSpecifiers(elfAtSpecifiers);
}

void PPCELFMCAsmInfo::printSpecifierExpr(raw_ostream &OS,
                                         const MCSpecifierExpr &Expr) const {
  printExpr(OS, *Expr.getSubExpr());
  OS << '@' << getSpecifierName(Expr.getSpecifier());
}

bool PPCELFMCAsmInfo::evaluateAsRelocatableImpl(const MCSpecifierExpr &Expr,
                                                MCValue &Res,
                                                const MCAssembler *Asm) const {
  return evaluateAsRelocatable(Expr, Res, Asm);
}

PPCXCOFFMCAsmInfo::PPCXCOFFMCAsmInfo(bool Is64Bit, const Triple &T,
                                     const MCTargetOptions &Options)
    : MCAsmInfoXCOFF(Options) {
  if (T.getArch() == Triple::ppc64le || T.getArch() == Triple::ppcle)
    report_fatal_error("XCOFF is not supported for little-endian targets");
  CodePointerSize = CalleeSaveStackSlotSize = Is64Bit ? 8 : 4;

  // A size of 8 is only supported by the assembler under 64-bit.
  Data64bitsDirective = Is64Bit ? "\t.vbyte\t8, " : nullptr;

  // Debug Information
  SupportsDebugInformation = true;

  // Set up DWARF directives
  MinInstAlignment = 4;

  // Support $ as PC in inline asm
  DollarIsPC = true;

  UsesSetToEquateSymbol = true;

  initializeAtSpecifiers(xcoffAtSpecifiers);
}

void PPCXCOFFMCAsmInfo::printSpecifierExpr(raw_ostream &OS,
                                           const MCSpecifierExpr &Expr) const {
  printExpr(OS, *Expr.getSubExpr());
  OS << '@' << getSpecifierName(Expr.getSpecifier());
}

bool PPCXCOFFMCAsmInfo::evaluateAsRelocatableImpl(
    const MCSpecifierExpr &Expr, MCValue &Res, const MCAssembler *Asm) const {
  return evaluateAsRelocatable(Expr, Res, Asm);
}
