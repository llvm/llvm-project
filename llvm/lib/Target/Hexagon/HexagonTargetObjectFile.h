//===-- HexagonTargetAsmInfo.h - Hexagon asm properties --------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef HexagonTARGETOBJECTFILE_H
#define HexagonTARGETOBJECTFILE_H

#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/MC/MCSectionELF.h"

namespace llvm {

  class HexagonTargetObjectFile : public TargetLoweringObjectFileELF {
    const MCSectionELF *SmallDataSection;
    const MCSectionELF *SmallBSSSection;
  public:
    virtual void Initialize(MCContext &Ctx, const TargetMachine &TM);

    /// IsGlobalInSmallSection - Return true if this global address should be
    /// placed into small data/bss section.
    bool IsGlobalInSmallSection(const GlobalValue *GV,
                                const TargetMachine &TM,
                                SectionKind Kind) const;
    bool IsGlobalInSmallSection(const GlobalValue *GV,
                                const TargetMachine &TM) const;

    const MCSection* SelectSectionForGlobal(const GlobalValue *GV,
                                            SectionKind Kind,
                                            Mangler *Mang,
                                            const TargetMachine &TM) const;
  };

} // namespace llvm

#endif
