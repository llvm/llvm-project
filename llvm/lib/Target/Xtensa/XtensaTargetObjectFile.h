//===-- XtensaTargetObjectFile.h - Xtensa Object Info -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_XTENSA_XTENSATARGETOBJECTFILE_H
#define LLVM_TARGET_XTENSA_XTENSATARGETOBJECTFILE_H

#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"

namespace llvm {
class XtensaTargetMachine;
class XtensaTargetObjectFile : public TargetLoweringObjectFileELF {
  MCSection *LiteralSection;
  MCSection *SmallDataSection;
  MCSection *SmallBSSSection;
  const XtensaTargetMachine *TM;

public:
  void Initialize(MCContext &Ctx, const TargetMachine &TM) override;

  /// Return true if this global address should be placed into small data/bss
  /// section.
  bool IsGlobalInSmallSection(const GlobalObject *GO, const TargetMachine &TM,
                              SectionKind Kind) const;
  bool IsGlobalInSmallSection(const GlobalObject *GO,
                              const TargetMachine &TM) const;
  bool IsGlobalInSmallSectionImpl(const GlobalObject *GO,
                                  const TargetMachine &TM) const;

  MCSection *SelectSectionForGlobal(const GlobalObject *GO, SectionKind Kind,
                                    const TargetMachine &TM) const override;

  /// Return true if this constant should be placed into small data section.
  bool IsConstantInSmallSection(const DataLayout &DL, const Constant *CN,
                                const TargetMachine &TM) const;

  MCSection *getSectionForConstant(const DataLayout &DL, SectionKind Kind,
                                   const Constant *C,
                                   unsigned &Align) const override;
};
} // end namespace llvm

#endif /* LLVM_TARGET_XTENSA_XTENSATARGETOBJECTFILE_H */
