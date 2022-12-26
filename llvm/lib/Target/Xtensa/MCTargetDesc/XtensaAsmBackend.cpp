//===-- XtensaMCAsmBackend.cpp - Xtensa assembler backend -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/XtensaMCTargetDesc.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCFixupKindInfo.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace llvm {
class MCObjectTargetWriter;
class XtensaMCAsmBackend : public MCAsmBackend {
  uint8_t OSABI;
  bool IsLittleEndian;

public:
  XtensaMCAsmBackend(uint8_t osABI, bool isLE)
      : MCAsmBackend(support::little), OSABI(osABI), IsLittleEndian(isLE) {}

  unsigned getNumFixupKinds() const override { return 1; }
  const MCFixupKindInfo &getFixupKindInfo(MCFixupKind Kind) const override;
  void applyFixup(const MCAssembler &Asm, const MCFixup &Fixup,
                  const MCValue &Target, MutableArrayRef<char> Data,
                  uint64_t Value, bool IsResolved,
                  const MCSubtargetInfo *STI) const override;
  bool mayNeedRelaxation(const MCInst &Inst,
                         const MCSubtargetInfo &STI) const override;
  bool fixupNeedsRelaxation(const MCFixup &Fixup, uint64_t Value,
                            const MCRelaxableFragment *Fragment,
                            const MCAsmLayout &Layout) const override;
  void relaxInstruction(MCInst &Inst,
                        const MCSubtargetInfo &STI) const override;
  bool writeNopData(raw_ostream &OS, uint64_t Count,
                    const MCSubtargetInfo *STI) const override;

  std::unique_ptr<MCObjectTargetWriter> createObjectTargetWriter() const override {
    return createXtensaObjectWriter(OSABI, IsLittleEndian);
  }
};
} // namespace llvm

const MCFixupKindInfo &
XtensaMCAsmBackend::getFixupKindInfo(MCFixupKind Kind) const {
  return MCAsmBackend::getFixupKindInfo(MCFixupKind::FK_NONE);
}
void XtensaMCAsmBackend::applyFixup(const MCAssembler &Asm,
                                    const MCFixup &Fixup, const MCValue &Target,
                                    MutableArrayRef<char> Data, uint64_t Value,
                                    bool IsResolved,
                                    const MCSubtargetInfo *STI) const {}

bool XtensaMCAsmBackend::mayNeedRelaxation(const MCInst &Inst,
                                           const MCSubtargetInfo &STI) const {
  return false;
}

bool XtensaMCAsmBackend::fixupNeedsRelaxation(
    const MCFixup &Fixup, uint64_t Value, const MCRelaxableFragment *Fragment,
    const MCAsmLayout &Layout) const {
  return false;
}

void XtensaMCAsmBackend::relaxInstruction(MCInst &Inst,
                                          const MCSubtargetInfo &STI) const {}

bool XtensaMCAsmBackend::writeNopData(raw_ostream &OS, uint64_t Count,
                                      const MCSubtargetInfo *STI) const {
  uint64_t NumNops24b = Count / 3;

  for (uint64_t i = 0; i != NumNops24b; ++i) {
    // Currently just little-endian machine supported,
    // but probably big-endian will be also implemented in future
    if (IsLittleEndian) {
      OS.write("\xf0", 1);
      OS.write("\x20", 1);
      OS.write("\0x00", 1);
    } else {
      report_fatal_error("Big-endian mode currently is not supported!");
    }
    Count -= 3;
  }

  // TODO maybe function should return error if (Count > 0)
  switch (Count) {
  default:
    break;
  case 1:
    OS.write("\0", 1);
    break;
  case 2:
    // NOP.N instruction
    OS.write("\x3d", 1);
    OS.write("\xf0", 1);
    break;
  }

  return true;
}

MCAsmBackend *llvm::createXtensaMCAsmBackend(const Target &T,
                                             const MCSubtargetInfo &STI,
                                             const MCRegisterInfo &MRI,
                                             const MCTargetOptions &Options) {
  uint8_t OSABI =
      MCELFObjectTargetWriter::getOSABI(STI.getTargetTriple().getOS());
  return new llvm::XtensaMCAsmBackend(OSABI, true);
}
