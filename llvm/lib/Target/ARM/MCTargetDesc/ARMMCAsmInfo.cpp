//===-- ARMMCAsmInfo.cpp - ARM asm properties -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the ARMMCAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "ARMMCAsmInfo.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

cl::opt<bool>
EnableARMEHABI("arm-enable-ehabi", cl::Hidden,
  cl::desc("Generate ARM EHABI tables"),
  cl::init(false));


void ARMMCAsmInfoDarwin::anchor() { }

ARMMCAsmInfoDarwin::ARMMCAsmInfoDarwin() {
  Data64bitsDirective = 0;
  CommentString = "@";
  Code16Directive = ".code\t16";
  Code32Directive = ".code\t32";
  UseDataRegionDirectives = true;

  SupportsDebugInformation = true;

  // Exceptions handling
  ExceptionsType = ExceptionHandling::SjLj;
}

void ARMELFMCAsmInfo::anchor() { }

ARMELFMCAsmInfo::ARMELFMCAsmInfo() {
  // ".comm align is in bytes but .align is pow-2."
  AlignmentIsInBytes = false;

  Data64bitsDirective = 0;
  CommentString = "@";
  PrivateGlobalPrefix = ".L";
  Code16Directive = ".code\t16";
  Code32Directive = ".code\t32";

  WeakRefDirective = "\t.weak\t";

  HasLEB128 = true;
  SupportsDebugInformation = true;

  // Exceptions handling
  if (EnableARMEHABI)
    ExceptionsType = ExceptionHandling::ARM;
}
