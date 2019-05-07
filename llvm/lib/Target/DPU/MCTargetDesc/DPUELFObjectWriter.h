//===-- DPUELFObjectWriter.cpp - DPU ELF Writer ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the DPUELFObjectWriter class.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCELFObjectWriter.h"

#ifndef LLVM_LIB_TARGET_DPU_MCTARGETDESC_DPUELFOBJECTWRITER_H
#define LLVM_LIB_TARGET_DPU_MCTARGETDESC_DPUELFOBJECTWRITER_H

namespace llvm {
class DPUELFObjectWriter : public MCELFObjectTargetWriter {

public:
  explicit DPUELFObjectWriter()
      : MCELFObjectTargetWriter(/*Is64Bit_=*/false, /*OSABI_*/ 0, ELF::EM_DPU,
                                /*HasRelocationAddend=*/true) {}
  ~DPUELFObjectWriter() override = default;

  unsigned int getRelocType(MCContext &Ctx, const MCValue &Target,
                            const MCFixup &Fixup, bool IsPCRel) const override;

  bool needsRelocateWithSymbol(const MCSymbol &Sym,
                               unsigned Type) const override;
};
} // namespace llvm

#endif // LLVM_LIB_TARGET_DPU_MCTARGETDESC_DPUELFOBJECTWRITER_H
