//=====-- MipsTargetAsmInfo.h - Mips asm properties -----------*- C++ -*--====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the MipsTargetAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef MIPSTARGETASMINFO_H
#define MIPSTARGETASMINFO_H

#include "MipsSubtarget.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/ELFTargetAsmInfo.h"

namespace llvm {

  // Forward declaration.
  class GlobalValue;
  class MipsTargetMachine;

  struct MipsTargetAsmInfo : public ELFTargetAsmInfo {
    explicit MipsTargetAsmInfo(const MipsTargetMachine &TM);

    /// SectionKindForGlobal - This hook allows the target to select proper
    /// section kind used for global emission.
    virtual SectionKind::Kind
    SectionKindForGlobal(const GlobalValue *GV) const;

    /// SectionFlagsForGlobal - This hook allows the target to select proper
    /// section flags either for given global or for section.
    virtual unsigned
    SectionFlagsForGlobal(const GlobalValue *GV = NULL,
                          const char* name = NULL) const;

    virtual const Section* SelectSectionForGlobal(const GlobalValue *GV) const;

    private:
      const MipsSubtarget *Subtarget;
  };

} // namespace llvm

#endif
