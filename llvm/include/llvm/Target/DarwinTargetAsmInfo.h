//===---- DarwinTargetAsmInfo.h - Darwin asm properties ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines target asm properties related what form asm statements
// should take in general on Darwin-based targets
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DARWIN_TARGET_ASM_INFO_H
#define LLVM_DARWIN_TARGET_ASM_INFO_H

#include "llvm/Target/TargetAsmInfo.h"

namespace llvm {
  class GlobalValue;
  class GlobalVariable;
  class Type;
  class Mangler;

  struct DarwinTargetAsmInfo: public TargetAsmInfo {
    const Section* TextCoalSection;
    const Section* ConstTextCoalSection;
    const Section* ConstDataCoalSection;
    const Section* ConstDataSection;
    const Section* DataCoalSection;
    const Section* FourByteConstantSection;
    const Section* EightByteConstantSection;
    const Section* SixteenByteConstantSection;

    explicit DarwinTargetAsmInfo(const TargetMachine &TM);
    virtual const Section* SelectSectionForGlobal(const GlobalValue *GV) const;
    virtual std::string UniqueSectionForGlobal(const GlobalValue* GV,
                                               SectionKind::Kind kind) const;
    virtual bool emitUsedDirectiveFor(const GlobalValue *GV,
                                      Mangler *Mang) const;
    const Section* MergeableConstSection(const GlobalVariable *GV) const;
    const Section* MergeableConstSection(const Type *Ty) const;
    const Section* MergeableStringSection(const GlobalVariable *GV) const;
    const Section* SelectSectionForMachineConst(const Type *Ty) const;
  };
}


#endif // LLVM_DARWIN_TARGET_ASM_INFO_H
