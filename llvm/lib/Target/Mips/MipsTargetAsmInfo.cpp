//===-- MipsTargetAsmInfo.cpp - Mips asm properties -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the MipsTargetAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "MipsTargetAsmInfo.h"
#include "MipsTargetMachine.h"
#include "llvm/GlobalVariable.h"

using namespace llvm;

MipsTargetAsmInfo::MipsTargetAsmInfo(const MipsTargetMachine &TM):
  ELFTargetAsmInfo(TM) {

  Subtarget = &TM.getSubtarget<MipsSubtarget>();

  AlignmentIsInBytes          = false;
  COMMDirectiveTakesAlignment = true;
  Data16bitsDirective         = "\t.half\t";
  Data32bitsDirective         = "\t.word\t";
  Data64bitsDirective         = NULL;
  PrivateGlobalPrefix         = "$";
  JumpTableDataSection        = "\t.rdata";
  CommentString               = "#";
  ZeroDirective               = "\t.space\t";
  BSSSection                  = "\t.section\t.bss";
  CStringSection              = ".rodata.str";

  if (!Subtarget->hasABICall()) {
    JumpTableDirective = "\t.word\t";
    SmallDataSection = getNamedSection("\t.sdata", SectionFlags::Writeable);
    SmallBSSSection = getNamedSection("\t.sbss",
                                      SectionFlags::Writeable |
                                      SectionFlags::BSS);
  } else
    JumpTableDirective = "\t.gpword\t";

}

unsigned MipsTargetAsmInfo::
SectionFlagsForGlobal(const GlobalValue *GV, const char* Name) const {
  unsigned Flags = ELFTargetAsmInfo::SectionFlagsForGlobal(GV, Name);
  // Mask out Small Section flag bit, Mips doesnt support 's' section symbol
  // for its small sections.
  return (Flags & (~SectionFlags::Small));
}

SectionKind::Kind MipsTargetAsmInfo::
SectionKindForGlobal(const GlobalValue *GV) const {
  SectionKind::Kind K = ELFTargetAsmInfo::SectionKindForGlobal(GV);

  if (Subtarget->hasABICall())
    return K;

  if (K != SectionKind::Data && K != SectionKind::BSS &&
      K != SectionKind::RODataMergeConst)
    return K;

  if (isa<GlobalVariable>(GV)) {
    const TargetData *TD = TM.getTargetData();
    unsigned Size = TD->getTypePaddedSize(GV->getType()->getElementType());
    unsigned Threshold = Subtarget->getSSectionThreshold();

    if (Size > 0 && Size <= Threshold) {
      if (K == SectionKind::BSS)
        return SectionKind::SmallBSS;
      else
        return SectionKind::SmallData;
    }
  }

  return K;
}

const Section* MipsTargetAsmInfo::
SelectSectionForGlobal(const GlobalValue *GV) const {
  SectionKind::Kind K = SectionKindForGlobal(GV);
  const GlobalVariable *GVA = dyn_cast<GlobalVariable>(GV);

  if (GVA && (!GVA->mayBeOverridden()))
    switch (K) {
      case SectionKind::SmallData:
        return getSmallDataSection();
      case SectionKind::SmallBSS:
        return getSmallBSSSection();
      default: break;
    }

  return ELFTargetAsmInfo::SelectSectionForGlobal(GV);
}
