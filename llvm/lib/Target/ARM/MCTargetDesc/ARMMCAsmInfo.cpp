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
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

const MCAsmInfo::VariantKindDesc variantKindDescs[] = {
    {MCSymbolRefExpr::VK_ARM_GOT_PREL, "GOT_PREL"},
    {MCSymbolRefExpr::VK_ARM_NONE, "none"},
    {MCSymbolRefExpr::VK_ARM_PREL31, "prel31"},
    {MCSymbolRefExpr::VK_ARM_SBREL, "sbrel"},
    {MCSymbolRefExpr::VK_ARM_TARGET1, "target1"},
    {MCSymbolRefExpr::VK_ARM_TARGET2, "target2"},
    {MCSymbolRefExpr::VK_ARM_TLSLDO, "TLSLDO"},
    {MCSymbolRefExpr::VK_COFF_IMGREL32, "imgrel"},
    {MCSymbolRefExpr::VK_FUNCDESC, "FUNCDESC"},
    {MCSymbolRefExpr::VK_GOT, "GOT"},
    {MCSymbolRefExpr::VK_GOTFUNCDESC, "GOTFUNCDESC"},
    {MCSymbolRefExpr::VK_GOTOFF, "GOTOFF"},
    {MCSymbolRefExpr::VK_GOTOFFFUNCDESC, "GOTOFFFUNCDESC"},
    {MCSymbolRefExpr::VK_GOTTPOFF, "GOTTPOFF"},
    {MCSymbolRefExpr::VK_GOTTPOFF_FDPIC, "gottpoff_fdpic"},
    {MCSymbolRefExpr::VK_PLT, "PLT"},
    {MCSymbolRefExpr::VK_SECREL, "SECREL32"},
    {MCSymbolRefExpr::VK_TLSCALL, "tlscall"},
    {MCSymbolRefExpr::VK_TLSDESC, "tlsdesc"},
    {MCSymbolRefExpr::VK_TLSGD, "TLSGD"},
    {MCSymbolRefExpr::VK_TLSGD_FDPIC, "tlsgd_fdpic"},
    {MCSymbolRefExpr::VK_TLSLD, "TLSLD"},
    {MCSymbolRefExpr::VK_TLSLDM, "TLSLDM"},
    {MCSymbolRefExpr::VK_TLSLDM_FDPIC, "tlsldm_fdpic"},
    {MCSymbolRefExpr::VK_TPOFF, "TPOFF"},
};

void ARMMCAsmInfoDarwin::anchor() { }

ARMMCAsmInfoDarwin::ARMMCAsmInfoDarwin(const Triple &TheTriple) {
  if ((TheTriple.getArch() == Triple::armeb) ||
      (TheTriple.getArch() == Triple::thumbeb))
    IsLittleEndian = false;

  Data64bitsDirective = nullptr;
  CommentString = "@";
  Code16Directive = ".code\t16";
  Code32Directive = ".code\t32";
  UseDataRegionDirectives = true;

  SupportsDebugInformation = true;

  // Conditional Thumb 4-byte instructions can have an implicit IT.
  MaxInstLength = 6;

  // Exceptions handling
  ExceptionsType = (TheTriple.isOSDarwin() && !TheTriple.isWatchABI())
                       ? ExceptionHandling::SjLj
                       : ExceptionHandling::DwarfCFI;

  initializeVariantKinds(variantKindDescs);
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
  Code16Directive = ".code\t16";
  Code32Directive = ".code\t32";

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
  UseParensForSymbolVariant = true;

  initializeVariantKinds(variantKindDescs);
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

  initializeVariantKinds(variantKindDescs);
}

void ARMCOFFMCAsmInfoGNU::anchor() { }

ARMCOFFMCAsmInfoGNU::ARMCOFFMCAsmInfoGNU() {
  AlignmentIsInBytes = false;
  HasSingleParameterDotFile = true;

  CommentString = "@";
  Code16Directive = ".code\t16";
  Code32Directive = ".code\t32";
  PrivateGlobalPrefix = ".L";
  PrivateLabelPrefix = ".L";

  SupportsDebugInformation = true;
  ExceptionsType = ExceptionHandling::WinEH;
  WinEHEncodingType = WinEH::EncodingType::Itanium;
  UseParensForSymbolVariant = true;

  DwarfRegNumForCFI = false;

  // Conditional Thumb 4-byte instructions can have an implicit IT.
  MaxInstLength = 6;

  initializeVariantKinds(variantKindDescs);
}
