//===-- SPUTargetAsmInfo.cpp - Cell SPU asm properties ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the SPUTargetAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "SPUTargetAsmInfo.h"
#include "SPUTargetMachine.h"
#include "llvm/Function.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Dwarf.h"

using namespace llvm;
using namespace llvm::dwarf;

SPULinuxTargetAsmInfo::SPULinuxTargetAsmInfo(const SPUTargetMachine &TM) :
    SPUTargetAsmInfo<ELFTargetAsmInfo>(TM) {
  PCSymbol = ".";
  CommentString = "#";
  GlobalPrefix = "";
  PrivateGlobalPrefix = ".L";
  // This corresponds to what the gcc SPU compiler emits, for consistency.
  CStringSection = ".rodata.str";

  // Has leb128, .loc and .file
  HasLEB128 = true;
  HasDotLocAndDotFile = true;

  // BSS section needs to be emitted as ".section"
  BSSSection = "\t.section\t.bss";
  BSSSection_ = getUnnamedSection("\t.section\t.bss",
                                  SectionFlags::Writeable | SectionFlags::BSS,
                                  true);

  SupportsDebugInformation = true;
  NeedsSet = true;
  SupportsMacInfoSection = false;
  DwarfAbbrevSection =  "\t.section        .debug_abbrev,\"\",@progbits";
  DwarfInfoSection =    "\t.section        .debug_info,\"\",@progbits";
  DwarfLineSection =    "\t.section        .debug_line,\"\",@progbits";
  DwarfFrameSection =   "\t.section        .debug_frame,\"\",@progbits";
  DwarfPubNamesSection = "\t.section        .debug_pubnames,\"\",@progbits";
  DwarfPubTypesSection = "\t.section        .debug_pubtypes,\"\",progbits";
  DwarfStrSection =     "\t.section        .debug_str,\"MS\",@progbits,1";
  DwarfLocSection =     "\t.section        .debug_loc,\"\",@progbits";
  DwarfARangesSection = "\t.section        .debug_aranges,\"\",@progbits";
  DwarfRangesSection =  "\t.section        .debug_ranges,\"\",@progbits";
  DwarfMacInfoSection = "\t.section        .debug_macinfo,\"\",progbits";

  // Exception handling is not supported on CellSPU (think about it: you only
  // have 256K for code+data. Would you support exception handling?)
  SupportsExceptionHandling = false;
}

/// PreferredEHDataFormat - This hook allows the target to select data
/// format used for encoding pointers in exception handling data. Reason is
/// 0 for data, 1 for code labels, 2 for function pointers. Global is true
/// if the symbol can be relocated.
unsigned
SPULinuxTargetAsmInfo::PreferredEHDataFormat(DwarfEncoding::Target Reason,
                                             bool Global) const {
  // We really need to write something here.
  return TargetAsmInfo::PreferredEHDataFormat(Reason, Global);
}

// Instantiate default implementation.
TEMPLATE_INSTANTIATION(class SPUTargetAsmInfo<TargetAsmInfo>);
