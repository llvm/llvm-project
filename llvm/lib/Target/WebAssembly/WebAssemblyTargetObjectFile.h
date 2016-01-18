//===-- WebAssemblyTargetObjectFile.h - WebAssembly Object Info -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file declares the WebAssembly-specific subclass of
/// TargetLoweringObjectFile.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_WEBASSEMBLY_WEBASSEMBLYTARGETOBJECTFILE_H
#define LLVM_LIB_TARGET_WEBASSEMBLY_WEBASSEMBLYTARGETOBJECTFILE_H

#include "llvm/Target/TargetLoweringObjectFile.h"

namespace llvm {

class GlobalVariable;

class WebAssemblyTargetObjectFile final : public TargetLoweringObjectFile {
public:
  WebAssemblyTargetObjectFile() {
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
  }

  MCSection *getSectionForConstant(const DataLayout &DL, SectionKind Kind,
                                   const Constant *C) const override {
    return ReadOnlySection;
  }

  MCSection *getExplicitSectionGlobal(const GlobalValue *GV, SectionKind Kind,
                                      Mangler &Mang,
                                      const TargetMachine &TM) const override {
    return DataSection;
  }

  MCSection *SelectSectionForGlobal(const GlobalValue *GV, SectionKind Kind,
                                    Mangler &Mang,
                                    const TargetMachine &TM) const override;
};

} // end namespace llvm

#endif
