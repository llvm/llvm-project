//=====-- ARMTargetAsmInfo.h - ARM asm properties -------------*- C++ -*--====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the ARMTargetAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef ARMTARGETASMINFO_H
#define ARMTARGETASMINFO_H

#include "ARMTargetMachine.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Target/ELFTargetAsmInfo.h"
#include "llvm/Target/DarwinTargetAsmInfo.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

  extern const char *const arm_asm_table[];

  template <class BaseTAI>
  struct ARMTargetAsmInfo : public BaseTAI {
    explicit ARMTargetAsmInfo(const ARMTargetMachine &TM):
      BaseTAI(TM) {
      BaseTAI::AsmTransCBE = arm_asm_table;

      BaseTAI::AlignmentIsInBytes = false;
      BaseTAI::Data64bitsDirective = 0;
      BaseTAI::CommentString = "@";
      BaseTAI::ConstantPoolSection = "\t.text\n";
      BaseTAI::COMMDirectiveTakesAlignment = false;
      BaseTAI::InlineAsmStart = "@ InlineAsm Start";
      BaseTAI::InlineAsmEnd = "@ InlineAsm End";
      BaseTAI::LCOMMDirective = "\t.lcomm\t";
    }

    const ARMSubtarget *Subtarget;

    virtual unsigned getInlineAsmLength(const char *Str) const;
    unsigned countArguments(const char *p) const;
    unsigned countString(const char *p) const;
  };

  typedef ARMTargetAsmInfo<TargetAsmInfo> ARMGenericTargetAsmInfo;

  EXTERN_TEMPLATE_INSTANTIATION(class ARMTargetAsmInfo<TargetAsmInfo>);

  struct ARMDarwinTargetAsmInfo : public ARMTargetAsmInfo<DarwinTargetAsmInfo> {
    explicit ARMDarwinTargetAsmInfo(const ARMTargetMachine &TM);
  };

  struct ARMELFTargetAsmInfo : public ARMTargetAsmInfo<ELFTargetAsmInfo> {
    explicit ARMELFTargetAsmInfo(const ARMTargetMachine &TM);
  };

} // namespace llvm

#endif
