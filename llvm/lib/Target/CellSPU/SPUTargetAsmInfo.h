//===-- SPUTargetAsmInfo.h - Cell SPU asm properties -----------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the SPUTargetAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef SPUTARGETASMINFO_H
#define SPUTARGETASMINFO_H

#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Target/ELFTargetAsmInfo.h"
#include "SPUTargetMachine.h"
#include "SPUSubtarget.h"

namespace llvm {

  // Forward declaration.
  class SPUTargetMachine;
  
  template <class BaseTAI>
  struct SPUTargetAsmInfo : public BaseTAI {
    explicit SPUTargetAsmInfo(const SPUTargetMachine &TM):
      BaseTAI(TM) {
      /* (unused today)
       * const SPUSubtarget *Subtarget = &TM.getSubtarget<SPUSubtarget>(); */

      BaseTAI::ZeroDirective = "\t.space\t";
      BaseTAI::SetDirective = "\t.set";
      BaseTAI::Data64bitsDirective = "\t.quad\t";
      BaseTAI::AlignmentIsInBytes = false;
      BaseTAI::LCOMMDirective = "\t.lcomm\t";
      BaseTAI::InlineAsmStart = "# InlineAsm Start";
      BaseTAI::InlineAsmEnd = "# InlineAsm End";
    }
  };
  
  struct SPULinuxTargetAsmInfo : public SPUTargetAsmInfo<ELFTargetAsmInfo> {
    explicit SPULinuxTargetAsmInfo(const SPUTargetMachine &TM);
    virtual unsigned PreferredEHDataFormat(DwarfEncoding::Target Reason,
                                           bool Global) const;
  };
} // namespace llvm

#endif /* SPUTARGETASMINFO_H */
