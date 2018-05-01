//===-- NVPTXTargetObjectFile.h - NVPTX Object Info -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NVPTX_NVPTXTARGETOBJECTFILE_H
#define LLVM_LIB_TARGET_NVPTX_NVPTXTARGETOBJECTFILE_H

#include "NVPTXSection.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/SectionKind.h"
#include "llvm/Target/TargetLoweringObjectFile.h"

namespace llvm {

class NVPTXTargetObjectFile : public TargetLoweringObjectFile {
public:
  NVPTXTargetObjectFile() {
    TextSection = nullptr;
    DataSection = nullptr;
    BSSSection = nullptr;
    ReadOnlySection = nullptr;

    StaticCtorSection = nullptr;
    StaticDtorSection = nullptr;
    LSDASection = nullptr;
    EHFrameSection = nullptr;
    DwarfAbbrevSection = nullptr;
    DwarfInfoSection = nullptr;
    DwarfLineSection = nullptr;
    DwarfFrameSection = nullptr;
    DwarfPubTypesSection = nullptr;
    DwarfDebugInlineSection = nullptr;
    DwarfStrSection = nullptr;
    DwarfLocSection = nullptr;
    DwarfARangesSection = nullptr;
    DwarfRangesSection = nullptr;
    DwarfMacinfoSection = nullptr;
  }

  ~NVPTXTargetObjectFile() override;

  void Initialize(MCContext &ctx, const TargetMachine &TM) override {
    TargetLoweringObjectFile::Initialize(ctx, TM);
    TextSection = new NVPTXSection(MCSection::SV_ELF, SectionKind::getText());
    DataSection = new NVPTXSection(MCSection::SV_ELF, SectionKind::getData());
    BSSSection = new NVPTXSection(MCSection::SV_ELF, SectionKind::getBSS());
    ReadOnlySection =
        new NVPTXSection(MCSection::SV_ELF, SectionKind::getReadOnly());
    StaticCtorSection =
        new NVPTXSection(MCSection::SV_ELF, SectionKind::getMetadata());
    StaticDtorSection =
        new NVPTXSection(MCSection::SV_ELF, SectionKind::getMetadata());
    LSDASection =
        new NVPTXSection(MCSection::SV_ELF, SectionKind::getMetadata());
    EHFrameSection =
        new NVPTXSection(MCSection::SV_ELF, SectionKind::getMetadata());
    DwarfAbbrevSection =
        new NVPTXSection(MCSection::SV_ELF, SectionKind::getMetadata());
    DwarfInfoSection =
        new NVPTXSection(MCSection::SV_ELF, SectionKind::getMetadata());
    DwarfLineSection =
        new NVPTXSection(MCSection::SV_ELF, SectionKind::getMetadata());
    DwarfFrameSection =
        new NVPTXSection(MCSection::SV_ELF, SectionKind::getMetadata());
    DwarfPubTypesSection =
        new NVPTXSection(MCSection::SV_ELF, SectionKind::getMetadata());
    DwarfDebugInlineSection =
        new NVPTXSection(MCSection::SV_ELF, SectionKind::getMetadata());
    DwarfStrSection =
        new NVPTXSection(MCSection::SV_ELF, SectionKind::getMetadata());
    DwarfLocSection =
        new NVPTXSection(MCSection::SV_ELF, SectionKind::getMetadata());
    DwarfARangesSection =
        new NVPTXSection(MCSection::SV_ELF, SectionKind::getMetadata());
    DwarfRangesSection =
        new NVPTXSection(MCSection::SV_ELF, SectionKind::getMetadata());
    DwarfMacinfoSection =
        new NVPTXSection(MCSection::SV_ELF, SectionKind::getMetadata());
  }

  MCSection *getSectionForConstant(const DataLayout &DL, SectionKind Kind,
                                   const Constant *C,
                                   unsigned &Align) const override {
    return ReadOnlySection;
  }

  MCSection *getExplicitSectionGlobal(const GlobalObject *GO, SectionKind Kind,
                                      const TargetMachine &TM) const override {
    return DataSection;
  }

  MCSection *SelectSectionForGlobal(const GlobalObject *GO, SectionKind Kind,
                                    const TargetMachine &TM) const override;
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_NVPTX_NVPTXTARGETOBJECTFILE_H
