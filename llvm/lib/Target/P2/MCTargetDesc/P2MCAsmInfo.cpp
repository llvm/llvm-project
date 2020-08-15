//===-- P2MCAsmInfo.cpp - P2 Asm Properties ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the P2MCAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "P2MCAsmInfo.h"

#include "llvm/ADT/Triple.h"

using namespace llvm;

void P2MCAsmInfo::anchor() { }

P2MCAsmInfo::P2MCAsmInfo(const Triple &TheTriple, const MCTargetOptions &Options) {
    CodePointerSize = 4;
    CalleeSaveStackSlotSize = 4;
    StackGrowsUp = true;

    IsLittleEndian = true;
    CommentString = "\'\'";

    UsesELFSectionDirectiveForBSS = true;
    AllowAtInName = true;
    HiddenVisibilityAttr = MCSA_Invalid;
    HiddenDeclarationVisibilityAttr = MCSA_Invalid;
    ProtectedVisibilityAttr = MCSA_Invalid;
    UseIntegratedAssembler = true;
    PreserveAsmComments = true;
}