//===-- SuperHAsmBackend.h - SuperH Assembler Backend ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SUPERH_MCTARGETDESC_SUPERHASMBACKEND_H
#define LLVM_LIB_TARGET_SUPERH_MCTARGETDESC_SUPERHASMBACKEND_H

#include "MCTargetDesc/SuperHMCTargetDesc.h"
#include "MCTargetDesc/SuperHMCAsmInfo.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCSubtargetInfo.h"


namespace llvm {
class MCAssembler;
class MCObjectTargetWriter;
class raw_ostream;

class SuperHAsmBackend : public MCAsmBackend {
protected:
  const MCSubtargetInfo &STI;
  uint8_t OSABI;
public:
  SuperHAsmBackend(const MCSubtargetInfo &STI, uint8_t OSABI);
  ~SuperHAsmBackend() override = default;

  std::optional<MCFixupKind> getFixupKind(StringRef Name) const override;
  MCFixupKindInfo getFixupKindInfo(MCFixupKind Kind) const override;

  void applyFixup(const MCFragment &, const MCFixup &, const MCValue &Target,
                  uint8_t *Data, uint64_t Value, bool IsResolved) override;
  std::optional<bool> evaluateFixup(const MCFragment &, MCFixup &, MCValue &,
                                    uint64_t &) override;
	
	std::unique_ptr<MCObjectTargetWriter>
  createObjectTargetWriter() const override;
  
  bool writeNopData(raw_ostream &OS, uint64_t Count,
                    const MCSubtargetInfo *STI) const override;
  

  bool tryAddReloc(const MCFragment &F, const MCFixup &Fixup,
                               const MCValue &Target, uint64_t &FixedValue,
                               bool IsResolved);

  unsigned adjustFixupValue(const MCAssembler &Asm, const MCFixup &Fixup,
                            const MCValue &Target, uint64_t Value,
                            bool IsResolved, MCContext &Ctx,
                            const MCSubtargetInfo *STI) const;
};
} // namespace llvm

#endif