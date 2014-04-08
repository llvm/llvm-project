//===-- MCAsmInfoCOFF.cpp - COFF asm properties -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines target asm properties related what form asm statements
// should take in general on COFF-based targets
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCAsmInfoCOFF.h"
using namespace llvm;

void MCAsmInfoCOFF::anchor() { }

MCAsmInfoCOFF::MCAsmInfoCOFF() {
  GlobalPrefix = "_";
  // MingW 4.5 and later support .comm with log2 alignment, but .lcomm uses byte
  // alignment.
  COMMDirectiveAlignmentIsInBytes = false;
  LCOMMDirectiveAlignmentType = LCOMM::ByteAlignment;
  HasDotTypeDotSizeDirective = false;
  HasSingleParameterDotFile = false;
  PrivateGlobalPrefix = "L";  // Prefix for private global symbols
  WeakRefDirective = "\t.weak\t";
  HasLinkOnceDirective = true;

  // Doesn't support visibility:
  HiddenVisibilityAttr = HiddenDeclarationVisibilityAttr = MCSA_Invalid;
  ProtectedVisibilityAttr = MCSA_Invalid;

  // Set up DWARF directives
  HasLEB128 = true;  // Target asm supports leb128 directives (little-endian)
  SupportsDebugInformation = true;
  HasMicrosoftFastStdCallMangling = true;
  NeedsDwarfSectionOffsetDirective = true;
}

void MCAsmInfoMicrosoft::anchor() { }

MCAsmInfoMicrosoft::MCAsmInfoMicrosoft() {
}

void MCAsmInfoGNUCOFF::anchor() { }

MCAsmInfoGNUCOFF::MCAsmInfoGNUCOFF() {

}
