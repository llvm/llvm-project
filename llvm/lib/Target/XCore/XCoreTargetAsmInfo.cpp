//===-- XCoreTargetAsmInfo.cpp - XCore asm properties -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the XCoreTargetAsmInfo properties.
// We use the small section flag for the CP relative and DP relative
// flags. If a section is small and writable then it is DP relative. If a
// section is small and not writable then it is CP relative.
//
//===----------------------------------------------------------------------===//

#include "XCoreTargetAsmInfo.h"
#include "XCoreTargetMachine.h"
#include "llvm/GlobalVariable.h"
#include "llvm/ADT/StringExtras.h"

using namespace llvm;

XCoreTargetAsmInfo::XCoreTargetAsmInfo(const XCoreTargetMachine &TM)
  : ELFTargetAsmInfo(TM),
    Subtarget(TM.getSubtargetImpl()) {
  TextSection = getUnnamedSection("\t.text", SectionFlags::Code);
  DataSection = getNamedSection("\t.dp.data", SectionFlags::Writeable |
                                SectionFlags::Small);
  BSSSection_  = getNamedSection("\t.dp.bss", SectionFlags::Writeable |
                                 SectionFlags::BSS | SectionFlags::Small);
  if (Subtarget->isXS1A()) {
    ReadOnlySection = getNamedSection("\t.dp.rodata", SectionFlags::None |
                                      SectionFlags::Writeable |
                                      SectionFlags::Small);
  } else {
    ReadOnlySection = getNamedSection("\t.cp.rodata", SectionFlags::None |
                                      SectionFlags::Small);
  }
  Data16bitsDirective = "\t.short\t";
  Data32bitsDirective = "\t.long\t";
  Data64bitsDirective = 0;
  ZeroDirective = "\t.space\t";
  CommentString = "#";
  ConstantPoolSection = "\t.section\t.cp.rodata,\"ac\",@progbits";
  JumpTableDataSection = "\t.section\t.dp.data,\"awd\",@progbits";
  PrivateGlobalPrefix = ".L";
  AscizDirective = ".asciiz";
  WeakDefDirective = "\t.weak\t";
  WeakRefDirective = "\t.weak\t";
  SetDirective = "\t.set\t";

  // Debug
  HasLEB128 = true;
  AbsoluteDebugSectionOffsets = true;
  
  DwarfAbbrevSection = "\t.section\t.debug_abbrev,\"\",@progbits";
  DwarfInfoSection = "\t.section\t.debug_info,\"\",@progbits";
  DwarfLineSection = "\t.section\t.debug_line,\"\",@progbits";
  DwarfFrameSection = "\t.section\t.debug_frame,\"\",@progbits";
  DwarfPubNamesSection = "\t.section\t.debug_pubnames,\"\",@progbits";
  DwarfPubTypesSection = "\t.section\t.debug_pubtypes,\"\",@progbits";
  DwarfStrSection = "\t.section\t.debug_str,\"\",@progbits";
  DwarfLocSection = "\t.section\t.debug_loc,\"\",@progbits";
  DwarfARangesSection = "\t.section\t.debug_aranges,\"\",@progbits";
  DwarfRangesSection = "\t.section\t.debug_ranges,\"\",@progbits";
  DwarfMacInfoSection = "\t.section\t.debug_macinfo,\"\",@progbits";
}

const Section*
XCoreTargetAsmInfo::SelectSectionForGlobal(const GlobalValue *GV) const {
  SectionKind::Kind Kind = SectionKindForGlobal(GV);

  if (const GlobalVariable *GVar = dyn_cast<GlobalVariable>(GV))
  {
    if (!GVar->mayBeOverridden()) {
      switch (Kind) {
      case SectionKind::RODataMergeStr:
        return MergeableStringSection(GVar);
      case SectionKind::RODataMergeConst:
        return getReadOnlySection();
      case SectionKind::ThreadData:
        return DataSection;
      case SectionKind::ThreadBSS:
        return getBSSSection_();
      default:
        break;
      }
    }
  }
  return ELFTargetAsmInfo::SelectSectionForGlobal(GV);
}

const Section*
XCoreTargetAsmInfo::SelectSectionForMachineConst(const Type *Ty) const {
  return MergeableConstSection(Ty);
}

const Section*
XCoreTargetAsmInfo::MergeableConstSection(const GlobalVariable *GV) const {
  Constant *C = GV->getInitializer();
  return MergeableConstSection(C->getType());
}

inline const Section*
XCoreTargetAsmInfo::MergeableConstSection(const Type *Ty) const {
  const TargetData *TD = TM.getTargetData();

  unsigned Size = TD->getTypePaddedSize(Ty);
  if (Size == 4 || Size == 8 || Size == 16) {
    std::string Name =  ".cp.const" + utostr(Size);

    return getNamedSection(Name.c_str(),
                           SectionFlags::setEntitySize(SectionFlags::Mergeable |
                                                       SectionFlags::Small,
                                                       Size));
  }

  return getReadOnlySection();
}

const Section* XCoreTargetAsmInfo::
MergeableStringSection(const GlobalVariable *GV) const {
  // FIXME insert in correct mergable section
  return getReadOnlySection();
}

unsigned XCoreTargetAsmInfo::
SectionFlagsForGlobal(const GlobalValue *GV,
                                     const char* Name) const {
  unsigned Flags = ELFTargetAsmInfo::SectionFlagsForGlobal(GV, Name);
  // Mask out unsupported flags
  Flags &= ~(SectionFlags::Small | SectionFlags::TLS);

  // Set CP / DP relative flags
  if (GV) {
    SectionKind::Kind Kind = SectionKindForGlobal(GV);
    switch (Kind) {
    case SectionKind::ThreadData:
    case SectionKind::ThreadBSS:
    case SectionKind::Data:
    case SectionKind::BSS:
    case SectionKind::SmallData:
    case SectionKind::SmallBSS:
      Flags |= SectionFlags::Small;
      break;
    case SectionKind::ROData:
    case SectionKind::RODataMergeStr:
    case SectionKind::SmallROData:
      if (Subtarget->isXS1A()) {
        Flags |= SectionFlags::Writeable;
      }
      Flags |=SectionFlags::Small;
      break;
    case SectionKind::RODataMergeConst:
      Flags |=SectionFlags::Small;
    default:
      break;
    }
  }

  return Flags;
}

std::string XCoreTargetAsmInfo::
printSectionFlags(unsigned flags) const {
  std::string Flags = ",\"";

  if (!(flags & SectionFlags::Debug))
    Flags += 'a';
  if (flags & SectionFlags::Code)
    Flags += 'x';
  if (flags & SectionFlags::Writeable)
    Flags += 'w';
  if (flags & SectionFlags::Mergeable)
    Flags += 'M';
  if (flags & SectionFlags::Strings)
    Flags += 'S';
  if (flags & SectionFlags::TLS)
    Flags += 'T';
  if (flags & SectionFlags::Small) {
    if (flags & SectionFlags::Writeable)
      Flags += 'd'; // DP relative
    else
      Flags += 'c'; // CP relative
  }

  Flags += "\",";
  
  Flags += '@';

  if (flags & SectionFlags::BSS)
    Flags += "nobits";
  else
    Flags += "progbits";

  if (unsigned entitySize = SectionFlags::getEntitySize(flags))
    Flags += "," + utostr(entitySize);

  return Flags;
}
