//===-- Next32AsmBackend.cpp - Next32 Assembler Backend -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/Next32FixupKinds.h"
#include "MCTargetDesc/Next32MCTargetDesc.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCFixupKindInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/EndianStream.h"
#include <cassert>
#include <cstdint>

using namespace llvm;

namespace {

class Next32AsmBackend : public MCAsmBackend {
public:
  Next32AsmBackend() : MCAsmBackend(llvm::endianness::little) {}
  ~Next32AsmBackend() override = default;

  void applyFixup(const MCAssembler &Asm, const MCFixup &Fixup,
                  const MCValue &Target, MutableArrayRef<char> Data,
                  uint64_t Value, bool IsResolved,
                  const MCSubtargetInfo *STI) const override;

  std::unique_ptr<MCObjectTargetWriter>
  createObjectTargetWriter() const override;

  // No instruction requires relaxation
  bool fixupNeedsRelaxation(const MCFixup &Fixup,
                            uint64_t Value) const override {
    return false;
  }

  unsigned getNumFixupKinds() const override {
    return Next32::NumTargetFixupKinds;
  }

  bool mayNeedRelaxation(const MCInst &Inst,
                         const MCSubtargetInfo &STI) const override {
    return false;
  }

  bool writeNopData(raw_ostream &OS, uint64_t Count,
                    const MCSubtargetInfo *STI) const override;

  const MCFixupKindInfo &getFixupKindInfo(MCFixupKind Kind) const override;
};

} // end anonymous namespace

bool Next32AsmBackend::writeNopData(raw_ostream &OS, uint64_t Count,
                                    const MCSubtargetInfo *STI) const {
  if ((Count % 8) != 0)
    return false;

  // dup r0, r0
  const uint64_t Nop = 0;
  for (uint64_t i = 0; i < Count; i += 8)
    support::endian::write(OS, Nop, llvm::endianness::big);

  return true;
}

void Next32AsmBackend::applyFixup(const MCAssembler &Asm, const MCFixup &Fixup,
                                  const MCValue &Target,
                                  MutableArrayRef<char> Data, uint64_t Value,
                                  bool IsResolved,
                                  const MCSubtargetInfo *STI) const {
  if (!Value)
    return; // Doesn't change encoding.

  MCFixupKindInfo Info = getFixupKindInfo(Fixup.getKind());
  unsigned Offset = (Info.TargetOffset / 8) + Fixup.getOffset();
  unsigned NumBytes = Info.TargetSize / 8;

  // For each byte of the fragment that the fixup touches, mask in the bits from
  // the fixup value. The Value has been "split up" into the appropriate
  // bitfields above.
  for (unsigned i = 0; i != NumBytes; ++i)
    Data[Offset + i] = uint8_t((Value >> (i * 8)) & 0xff);
}

std::unique_ptr<MCObjectTargetWriter>
Next32AsmBackend::createObjectTargetWriter() const {
  return createNext32ELFObjectWriter(/*OSABI=*/0);
}

MCAsmBackend *llvm::createNext32AsmBackend(const Target &T,
                                           const MCSubtargetInfo &STI,
                                           const MCRegisterInfo &MRI,
                                           const MCTargetOptions &Options) {
  return new Next32AsmBackend();
}

const MCFixupKindInfo &
Next32AsmBackend::getFixupKindInfo(MCFixupKind Kind) const {
  const static MCFixupKindInfo Info[Next32::NumTargetFixupKinds] = {
      // This table *must* be in same the order of the Fixup enum kinds in
      // Next32FixupKinds.h.
      // name, offset, bits, flags
      {"fixup_Next32_reloc_4byte_mem_high", 0, 32, 0},
      {"fixup_Next32_reloc_4byte_mem_low", 0, 32, 0},
      {"fixup_Next32_reloc_4byte_sym_bb_imm", 0, 32, 0},
      {"fixup_Next32_reloc_4byte_sym_function", 0, 32, 0},
      {"fixup_Next32_reloc_4byte_func_high", 0, 32, 0},
      {"fixup_Next32_reloc_4byte_func_low", 0, 32, 0},
  };

  if (Kind < FirstTargetFixupKind)
    return MCAsmBackend::getFixupKindInfo(Kind);

  assert(unsigned(Kind - FirstTargetFixupKind) < getNumFixupKinds() &&
         "Invalid kind!");

  return Info[Kind - FirstTargetFixupKind];
}