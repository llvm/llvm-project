//===---- ELFTargetAsmInfo.h - ELF asm properties ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines target asm properties related what form asm statements
// should take in general on ELF-based targets
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ELF_TARGET_ASM_INFO_H
#define LLVM_ELF_TARGET_ASM_INFO_H

#include "llvm/Target/TargetAsmInfo.h"

namespace llvm {
  class GlobalValue;
  class GlobalVariable;
  class Type;

  struct ELFTargetAsmInfo: public TargetAsmInfo {
    explicit ELFTargetAsmInfo(const TargetMachine &TM);

    virtual const Section* SelectSectionForGlobal(const GlobalValue *GV) const;
    virtual std::string printSectionFlags(unsigned flags) const;
    const Section* MergeableConstSection(const GlobalVariable *GV) const;
    inline const Section* MergeableConstSection(const Type *Ty) const;
    const Section* MergeableStringSection(const GlobalVariable *GV) const;
    virtual const Section*
    SelectSectionForMachineConst(const Type *Ty) const;
  };
}


#endif // LLVM_ELF_TARGET_ASM_INFO_H
