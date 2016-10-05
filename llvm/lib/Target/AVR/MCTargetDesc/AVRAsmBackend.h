//===-- AVRAsmBackend.h - AVR Asm Backend  --------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// \file The AVR assembly backend implementation.
//
//===----------------------------------------------------------------------===//
//

#ifndef LLVM_AVR_ASM_BACKEND_H
#define LLVM_AVR_ASM_BACKEND_H

#include "MCTargetDesc/AVRFixupKinds.h"

#include "llvm/ADT/Triple.h"
#include "llvm/MC/MCAsmBackend.h"

namespace llvm {

class MCAssembler;
class MCObjectWriter;
class Target;

struct MCFixupKindInfo;

/// Utilities for manipulating generated AVR machine code.
class AVRAsmBackend : public MCAsmBackend {
public:

  AVRAsmBackend(Triple::OSType OSType)
      : MCAsmBackend(), OSType(OSType) {}

  void adjustFixupValue(const MCFixup &Fixup, uint64_t &Value,
                        MCContext *Ctx = nullptr) const;

  MCObjectWriter *createObjectWriter(raw_pwrite_stream &OS) const override;

  void applyFixup(const MCFixup &Fixup, char *Data, unsigned DataSize,
                  uint64_t Value, bool IsPCRel) const override;

  const MCFixupKindInfo &getFixupKindInfo(MCFixupKind Kind) const override;

  unsigned getNumFixupKinds() const override {
    return AVR::NumTargetFixupKinds;
  }

  bool mayNeedRelaxation(const MCInst &Inst) const override { return false; }

  bool fixupNeedsRelaxation(const MCFixup &Fixup, uint64_t Value,
                            const MCRelaxableFragment *DF,
                            const MCAsmLayout &Layout) const override {
    llvm_unreachable("RelaxInstruction() unimplemented");
    return false;
  }

  void relaxInstruction(const MCInst &Inst, const MCSubtargetInfo &STI,
                        MCInst &Res) const override {}

  bool writeNopData(uint64_t Count, MCObjectWriter *OW) const override;

  void processFixupValue(const MCAssembler &Asm, const MCAsmLayout &Layout,
                         const MCFixup &Fixup, const MCFragment *DF,
                         const MCValue &Target, uint64_t &Value,
                         bool &IsResolved) override;

private:
  Triple::OSType OSType;
};

} // end namespace llvm

#endif // LLVM_AVR_ASM_BACKEND_H

