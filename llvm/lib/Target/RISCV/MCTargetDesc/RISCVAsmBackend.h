//===-- RISCVAsmBackend.h - RISC-V Assembler Backend ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_MCTARGETDESC_RISCVASMBACKEND_H
#define LLVM_LIB_TARGET_RISCV_MCTARGETDESC_RISCVASMBACKEND_H

#include "MCTargetDesc/RISCVBaseInfo.h"
#include "MCTargetDesc/RISCVFixupKinds.h"
#include "MCTargetDesc/RISCVMCTargetDesc.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCSubtargetInfo.h"

namespace llvm {
class MCAssembler;
class MCObjectTargetWriter;
class raw_ostream;

class RISCVAsmBackend : public MCAsmBackend {
  const MCSubtargetInfo &STI;
  uint8_t OSABI;
  bool Is64Bit;
  const MCTargetOptions &TargetOptions;
  // Temporary symbol used to check whether a PC-relative fixup is resolved.
  MCSymbol *PCRelTemp = nullptr;

  bool isPCRelFixupResolved(const MCSymbol *SymA, const MCFragment &F);

  StringMap<MCSymbol *> VendorSymbols;

public:
  RISCVAsmBackend(const MCSubtargetInfo &STI, uint8_t OSABI, bool Is64Bit,
                  const MCTargetOptions &Options);
  ~RISCVAsmBackend() override = default;

  std::optional<bool> evaluateFixup(const MCFragment &, MCFixup &, MCValue &,
                                    uint64_t &) override;
  bool addReloc(const MCFragment &, const MCFixup &, const MCValue &,
                uint64_t &FixedValue, bool IsResolved);

  void maybeAddVendorReloc(const MCFragment &, const MCFixup &);

  void applyFixup(const MCFragment &, const MCFixup &, const MCValue &Target,
                  MutableArrayRef<char> Data, uint64_t Value,
                  bool IsResolved) override;

  std::unique_ptr<MCObjectTargetWriter>
  createObjectTargetWriter() const override;

  bool fixupNeedsRelaxationAdvanced(const MCFragment &, const MCFixup &,
                                    const MCValue &, uint64_t,
                                    bool) const override;

  std::optional<MCFixupKind> getFixupKind(StringRef Name) const override;

  MCFixupKindInfo getFixupKindInfo(MCFixupKind Kind) const override;

  bool mayNeedRelaxation(unsigned Opcode, ArrayRef<MCOperand> Operands,
                         const MCSubtargetInfo &STI) const override;
  void relaxInstruction(MCInst &Inst,
                        const MCSubtargetInfo &STI) const override;

  bool relaxAlign(MCFragment &F, unsigned &Size) override;
  bool relaxDwarfLineAddr(MCFragment &F, bool &WasRelaxed) const override;
  bool relaxDwarfCFA(MCFragment &F, bool &WasRelaxed) const override;
  std::pair<bool, bool> relaxLEB128(MCFragment &LF,
                                    int64_t &Value) const override;

  bool writeNopData(raw_ostream &OS, uint64_t Count,
                    const MCSubtargetInfo *STI) const override;

  const MCTargetOptions &getTargetOptions() const { return TargetOptions; }
};
}

#endif
