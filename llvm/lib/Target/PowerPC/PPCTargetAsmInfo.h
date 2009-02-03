//=====-- PPCTargetAsmInfo.h - PPC asm properties -------------*- C++ -*--====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the DarwinTargetAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef PPCTARGETASMINFO_H
#define PPCTARGETASMINFO_H

#include "PPCTargetMachine.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Target/DarwinTargetAsmInfo.h"
#include "llvm/Target/ELFTargetAsmInfo.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

  template <class BaseTAI>
  struct PPCTargetAsmInfo : public BaseTAI {
    explicit PPCTargetAsmInfo(const PPCTargetMachine &TM):
      BaseTAI(TM) {
      const PPCSubtarget *Subtarget = &TM.getSubtarget<PPCSubtarget>();
      bool isPPC64 = Subtarget->isPPC64();

      BaseTAI::ZeroDirective = "\t.space\t";
      BaseTAI::SetDirective = "\t.set";
      BaseTAI::Data64bitsDirective = isPPC64 ? "\t.quad\t" : 0;
      BaseTAI::AlignmentIsInBytes = false;
      BaseTAI::LCOMMDirective = "\t.lcomm\t";
      BaseTAI::InlineAsmStart = "# InlineAsm Start";
      BaseTAI::InlineAsmEnd = "# InlineAsm End";
      BaseTAI::AssemblerDialect = Subtarget->getAsmFlavor();
    }
  };

  typedef PPCTargetAsmInfo<TargetAsmInfo> PPCGenericTargetAsmInfo;

  EXTERN_TEMPLATE_INSTANTIATION(class PPCTargetAsmInfo<TargetAsmInfo>);

  struct PPCDarwinTargetAsmInfo : public PPCTargetAsmInfo<DarwinTargetAsmInfo> {
    explicit PPCDarwinTargetAsmInfo(const PPCTargetMachine &TM);
    virtual unsigned PreferredEHDataFormat(DwarfEncoding::Target Reason,
                                           bool Global) const;
    virtual const char *getEHGlobalPrefix() const;
  };

  struct PPCLinuxTargetAsmInfo : public PPCTargetAsmInfo<ELFTargetAsmInfo> {
    explicit PPCLinuxTargetAsmInfo(const PPCTargetMachine &TM);
    virtual unsigned PreferredEHDataFormat(DwarfEncoding::Target Reason,
                                           bool Global) const;
  };

} // namespace llvm

#endif
