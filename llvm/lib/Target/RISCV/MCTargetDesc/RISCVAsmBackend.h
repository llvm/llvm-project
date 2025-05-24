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
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCFixupKindInfo.h"
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

  bool isPCRelFixupResolved(const MCAssembler &Asm, const MCSymbol *SymA,
                            const MCFragment &F);

public:
  RISCVAsmBackend(const MCSubtargetInfo &STI, uint8_t OSABI, bool Is64Bit,
                  const MCTargetOptions &Options);
  ~RISCVAsmBackend() override = default;

  // Return Size with extra Nop Bytes for alignment directive in code section.
  bool shouldInsertExtraNopBytesForCodeAlign(const MCAlignFragment &AF,
                                             unsigned &Size) override;

  // Insert target specific fixup type for alignment directive in code section.
  bool shouldInsertFixupForCodeAlign(MCAssembler &Asm,
                                     MCAlignFragment &AF) override;

  bool evaluateTargetFixup(const MCAssembler &Asm, const MCFixup &Fixup,
                           const MCFragment *DF, const MCValue &Target,
                           const MCSubtargetInfo *STI,
                           uint64_t &Value) override;

  bool addReloc(MCAssembler &Asm, const MCFragment &F, const MCFixup &Fixup,
                const MCValue &Target, uint64_t &FixedValue, bool IsResolved,
                const MCSubtargetInfo *) override;

  void applyFixup(const MCAssembler &Asm, const MCFixup &Fixup,
                  const MCValue &Target, MutableArrayRef<char> Data,
                  uint64_t Value, bool IsResolved,
                  const MCSubtargetInfo *STI) const override;

  std::unique_ptr<MCObjectTargetWriter>
  createObjectTargetWriter() const override;

  bool fixupNeedsRelaxationAdvanced(const MCAssembler &,
                                    const MCFixup &, const MCValue &, uint64_t,
                                    bool) const override;

  std::optional<MCFixupKind> getFixupKind(StringRef Name) const override;

  MCFixupKindInfo getFixupKindInfo(MCFixupKind Kind) const override;

  bool mayNeedRelaxation(const MCInst &Inst,
                         const MCSubtargetInfo &STI) const override;

  void relaxInstruction(MCInst &Inst,
                        const MCSubtargetInfo &STI) const override;

  bool relaxDwarfLineAddr(const MCAssembler &Asm, MCDwarfLineAddrFragment &DF,
                          bool &WasRelaxed) const override;
  bool relaxDwarfCFA(const MCAssembler &Asm, MCDwarfCallFrameFragment &DF,
                     bool &WasRelaxed) const override;
  std::pair<bool, bool> relaxLEB128(const MCAssembler &Asm, MCLEBFragment &LF,
                                    int64_t &Value) const override;

  bool writeNopData(raw_ostream &OS, uint64_t Count,
                    const MCSubtargetInfo *STI) const override;

  const MCTargetOptions &getTargetOptions() const { return TargetOptions; }
};
}

#endif
