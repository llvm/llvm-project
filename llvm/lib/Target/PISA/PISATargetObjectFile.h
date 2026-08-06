//===-- PISATargetObjectFile.h - PISA Object Info -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_PISA_PISATARGETOBJECTFILE_H
#define LLVM_LIB_TARGET_PISA_PISATARGETOBJECTFILE_H

#include "llvm/IR/GlobalValue.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/SectionKind.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

class PISATargetObjectFile : public TargetLoweringObjectFile {
public:
  ~PISATargetObjectFile() override;

  void Initialize(MCContext &Ctx, const TargetMachine &TM) override {
    TargetLoweringObjectFile::Initialize(Ctx, TM);
  }
  // All words in a PISA module (excepting the first 5 ones) are a linear
  // sequence of instructions in a specific order. We put all the instructions
  // in the single text section.
  MCSection *getSectionForConstant(const DataLayout &DL, SectionKind Kind,
                                   const Constant *C, Align &Alignment,
                                   const Function *F) const override {
    return TextSection;
  }
  MCSection *getExplicitSectionGlobal(const GlobalObject *GO, SectionKind Kind,
                                      const TargetMachine &TM) const override {
    return TextSection;
  }
  MCSection *SelectSectionForGlobal(const GlobalObject *GO, SectionKind Kind,
                                    const TargetMachine &TM) const override {
    return TextSection;
  }
  // PISA doesn't want '\01' suppression mangling to strip the leading '\01'.
  // A named variable should just be passed along as is.
  MCSymbol *getTargetSymbol(const GlobalValue *GV,
                            const TargetMachine &TM) const override {
    SmallString<128> Name;
    if (GV->hasName())
      Name = GV->getName();
    else
      TM.getNameWithPrefix(Name, GV, getMangler());
    return getContext().getOrCreateSymbol(Name);
  }
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_PISA_PISATARGETOBJECTFILE_H
