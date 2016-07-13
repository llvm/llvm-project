//===-- ARMAsmBackend.h - ARM Assembler Backend -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_ARM_ARMASMBACKEND_H
#define LLVM_LIB_TARGET_ARM_ARMASMBACKEND_H

#include "MCTargetDesc/ARMFixupKinds.h"
#include "MCTargetDesc/ARMMCTargetDesc.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/TargetRegistry.h"

namespace llvm {

class ARMAsmBackend : public MCAsmBackend {
  const MCSubtargetInfo *STI;
  bool isThumbMode;    // Currently emitting Thumb code.
  bool IsLittleEndian; // Big or little endian.
public:
  ARMAsmBackend(const Target &T, const Triple &TT, bool IsLittle)
      : MCAsmBackend(), STI(ARM_MC::createARMMCSubtargetInfo(TT, "", "")),
        isThumbMode(TT.getArchName().startswith("thumb")),
        IsLittleEndian(IsLittle) {}

  ~ARMAsmBackend() override { delete STI; }

  unsigned getNumFixupKinds() const override {
    return ARM::NumTargetFixupKinds;
  }

  bool hasNOP() const { return STI->getFeatureBits()[ARM::HasV6T2Ops]; }

  const MCFixupKindInfo &getFixupKindInfo(MCFixupKind Kind) const override;

  /// processFixupValue - Target hook to process the literal value of a fixup
  /// if necessary.
  void processFixupValue(const MCAssembler &Asm, const MCAsmLayout &Layout,
                         const MCFixup &Fixup, const MCFragment *DF,
                         const MCValue &Target, uint64_t &Value,
                         bool &IsResolved) override;

  unsigned adjustFixupValue(const MCFixup &Fixup, uint64_t Value, bool IsPCRel,
                            MCContext *Ctx, bool IsLittleEndian,
                            bool IsResolved) const;

  void applyFixup(const MCFixup &Fixup, char *Data, unsigned DataSize,
                  uint64_t Value, bool IsPCRel) const override;

  unsigned getRelaxedOpcode(unsigned Op) const;

  bool mayNeedRelaxation(const MCInst &Inst) const override;

  const char *reasonForFixupRelaxation(const MCFixup &Fixup,
                                       uint64_t Value) const;

  bool fixupNeedsRelaxation(const MCFixup &Fixup, uint64_t Value,
                            const MCRelaxableFragment *DF,
                            const MCAsmLayout &Layout) const override;

  void relaxInstruction(const MCInst &Inst, const MCSubtargetInfo &STI,
                        MCInst &Res) const override;

  bool writeNopData(uint64_t Count, MCObjectWriter *OW) const override;

  void handleAssemblerFlag(MCAssemblerFlag Flag) override;

  unsigned getPointerSize() const { return 4; }
  bool isThumb() const { return isThumbMode; }
  void setIsThumb(bool it) { isThumbMode = it; }
  bool isLittle() const { return IsLittleEndian; }
};
} // end namespace llvm

#endif
