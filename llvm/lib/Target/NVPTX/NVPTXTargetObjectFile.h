//===-- NVPTXTargetObjectFile.h - NVPTX Object Info -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_NVPTX_TARGETOBJECTFILE_H
#define LLVM_TARGET_NVPTX_TARGETOBJECTFILE_H

#include "NVPTXSection.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include <string>

namespace llvm {
class GlobalVariable;
class Module;

class NVPTXTargetObjectFile : public TargetLoweringObjectFile {

public:
  NVPTXTargetObjectFile() {}
  ~NVPTXTargetObjectFile() {
    delete TextSection;
    delete DataSection;
    delete BSSSection;
    delete ReadOnlySection;

    delete StaticCtorSection;
    delete StaticDtorSection;
    delete LSDASection;
    delete EHFrameSection;
    delete DwarfAbbrevSection;
    delete DwarfInfoSection;
    delete DwarfLineSection;
    delete DwarfFrameSection;
    delete DwarfPubTypesSection;
    delete DwarfDebugInlineSection;
    delete DwarfStrSection;
    delete DwarfLocSection;
    delete DwarfARangesSection;
    delete DwarfRangesSection;
    delete DwarfMacroInfoSection;
  }

  virtual void Initialize(MCContext &ctx, const TargetMachine &TM) {
    TextSection = new NVPTXSection(MCSection::SV_ELF,
                                   SectionKind::getText());
    DataSection = new NVPTXSection(MCSection::SV_ELF,
                                   SectionKind::getDataRel());
    BSSSection = new NVPTXSection(MCSection::SV_ELF,
                                  SectionKind::getBSS());
    ReadOnlySection = new NVPTXSection(MCSection::SV_ELF,
                                       SectionKind::getReadOnly());

    StaticCtorSection = new NVPTXSection(MCSection::SV_ELF,
                                         SectionKind::getMetadata());
    StaticDtorSection = new NVPTXSection(MCSection::SV_ELF,
                                         SectionKind::getMetadata());
    LSDASection = new NVPTXSection(MCSection::SV_ELF,
                                   SectionKind::getMetadata());
    EHFrameSection = new NVPTXSection(MCSection::SV_ELF,
                                      SectionKind::getMetadata());
    DwarfAbbrevSection = new NVPTXSection(MCSection::SV_ELF,
                                          SectionKind::getMetadata());
    DwarfInfoSection = new NVPTXSection(MCSection::SV_ELF,
                                        SectionKind::getMetadata());
    DwarfLineSection = new NVPTXSection(MCSection::SV_ELF,
                                        SectionKind::getMetadata());
    DwarfFrameSection = new NVPTXSection(MCSection::SV_ELF,
                                         SectionKind::getMetadata());
    DwarfPubTypesSection = new NVPTXSection(MCSection::SV_ELF,
                                            SectionKind::getMetadata());
    DwarfDebugInlineSection = new NVPTXSection(MCSection::SV_ELF,
                                               SectionKind::getMetadata());
    DwarfStrSection = new NVPTXSection(MCSection::SV_ELF,
                                       SectionKind::getMetadata());
    DwarfLocSection = new NVPTXSection(MCSection::SV_ELF,
                                       SectionKind::getMetadata());
    DwarfARangesSection = new NVPTXSection(MCSection::SV_ELF,
                                           SectionKind::getMetadata());
    DwarfRangesSection = new NVPTXSection(MCSection::SV_ELF,
                                          SectionKind::getMetadata());
    DwarfMacroInfoSection = new NVPTXSection(MCSection::SV_ELF,
                                             SectionKind::getMetadata());
  }

  virtual const MCSection *getSectionForConstant(SectionKind Kind) const {
    return ReadOnlySection;
  }

  virtual const MCSection *
  getExplicitSectionGlobal(const GlobalValue *GV, SectionKind Kind,
                           Mangler *Mang,
                           const TargetMachine &TM) const {
    return DataSection;
  }

};

} // end namespace llvm

#endif
