//===- MCAsmInfoCOFF.cpp - COFF asm properties ----------------------------===//
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
#include "llvm/MC/MCDirectives.h"

using namespace llvm;

void MCAsmInfoCOFF::anchor() {}

MCAsmInfoCOFF::MCAsmInfoCOFF() {
  // MingW 4.5 and later support .comm with log2 alignment, but .lcomm uses byte
  // alignment.
  COMMDirectiveAlignmentIsInBytes = false;
  LCOMMDirectiveAlignmentType = LCOMM::ByteAlignment;
  HasDotTypeDotSizeDirective = false;
  HasSingleParameterDotFile = false;
  WeakRefDirective = "\t.weak\t";
  HasLinkOnceDirective = true;

  // Doesn't support visibility:
  HiddenVisibilityAttr = HiddenDeclarationVisibilityAttr = MCSA_Invalid;
  ProtectedVisibilityAttr = MCSA_Invalid;

  // Set up DWARF directives
  SupportsDebugInformation = true;
  NeedsDwarfSectionOffsetDirective = true;

  UseIntegratedAssembler = true;

  // At least MSVC inline-asm does AShr.
  UseLogicalShr = false;
}

void MCAsmInfoMicrosoft::anchor() {}

MCAsmInfoMicrosoft::MCAsmInfoMicrosoft() = default;

void MCAsmInfoGNUCOFF::anchor() {}

MCAsmInfoGNUCOFF::MCAsmInfoGNUCOFF() = default;
