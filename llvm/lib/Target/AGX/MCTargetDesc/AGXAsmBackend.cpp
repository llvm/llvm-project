//===-- AGXAsmBackend.cpp - AGX assembler backend -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/AGXFixupKinds.h"
#include "MCTargetDesc/AGXMCTargetDesc.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/EndianStream.h"

using namespace llvm;

static cl::opt<bool> CallRelocations(
    "agx-call-relocations", cl::init(false), cl::Hidden,
    cl::desc("Calls to functions as relocations (R_AGX_CALL32) instead of "
             "offsets resolved in the object, for a linker that places the "
             "functions (ZLUDA's object merger, which shares one copy of "
             "the functions several kernels call)"));

namespace {

class AGXAsmBackend : public MCAsmBackend {
  uint8_t OSABI;

public:
  explicit AGXAsmBackend(uint8_t OSABI)
      : MCAsmBackend(llvm::endianness::little), OSABI(OSABI) {}

  void applyFixup(const MCFragment &F, const MCFixup &Fixup,
                  const MCValue &Target, uint8_t *Data, uint64_t Value,
                  bool IsResolved) override;

  std::unique_ptr<MCObjectTargetWriter>
  createObjectTargetWriter() const override {
    return createAGXELFObjectWriter(OSABI);
  }

  MCFixupKindInfo getFixupKindInfo(MCFixupKind Kind) const override;

  std::optional<bool> evaluateFixup(const MCFragment &, MCFixup &Fixup,
                                    MCValue &Target, uint64_t &Value) override {
    // A call (a branch to a function's symbol; branches within a function go
    // to temporary labels) left to a relocation, its field 0.
    const MCSymbol *Add = Target.getAddSym();
    if (!CallRelocations || Fixup.getKind() != AGX::fixup_agx_branch32 ||
        !Add || Add->isTemporary() || Target.getSubSym())
      return std::nullopt;
    Value = uint64_t(-2);
    return false;
  }

  bool writeNopData(raw_ostream &OS, uint64_t Count,
                    const MCSubtargetInfo *STI) const override;
};

} // namespace

void AGXAsmBackend::applyFixup(const MCFragment &F, const MCFixup &Fixup,
                               const MCValue &Target, uint8_t *Data,
                               uint64_t Value, bool IsResolved) {
  if (!IsResolved)
    Asm->getWriter().recordRelocation(F, Fixup, Target, Value);

  switch (Fixup.getKind()) {
  case AGX::fixup_agx_branch32: {
    // The offset is relative to the start of the branch, two bytes before the
    // fixup.
    int64_t Offset = int64_t(Value) + 2;
    if (!isInt<32>(Offset))
      getContext().reportError(Fixup.getLoc(), "branch target out of range");
    support::endian::write32le(Data, uint32_t(Offset));
    return;
  }
  case FK_Data_1:
    Data[0] = uint8_t(Value);
    return;
  case FK_Data_2:
    support::endian::write16le(Data, uint16_t(Value));
    return;
  case FK_Data_4:
    support::endian::write32le(Data, uint32_t(Value));
    return;
  case FK_Data_8:
    support::endian::write64le(Data, Value);
    return;
  default:
    llvm_unreachable("unknown fixup kind");
  }
}

MCFixupKindInfo AGXAsmBackend::getFixupKindInfo(MCFixupKind Kind) const {
  static const MCFixupKindInfo Infos[AGX::NumTargetFixupKinds] = {
      // name                 offset bits flags
      {"fixup_agx_branch32", 0, 32, 0},
  };
  if (Kind < FirstTargetFixupKind)
    return MCAsmBackend::getFixupKindInfo(Kind);
  assert(unsigned(Kind - FirstTargetFixupKind) < AGX::NumTargetFixupKinds &&
         "invalid kind");
  return Infos[Kind - FirstTargetFixupKind];
}

bool AGXAsmBackend::writeNopData(raw_ostream &OS, uint64_t Count,
                                 const MCSubtargetInfo *STI) const {
  // There is no documented no-op. Pad with trap, which is never executed.
  if (Count % 2)
    return false;
  for (uint64_t I = 0; I < Count; I += 2)
    support::endian::write<uint16_t>(OS, 0x0008, llvm::endianness::little);
  return true;
}

MCAsmBackend *llvm::createAGXAsmBackend(const Target &T,
                                        const MCSubtargetInfo &STI,
                                        const MCRegisterInfo &MRI,
                                        const MCTargetOptions &Options) {
  uint8_t OSABI =
      MCELFObjectTargetWriter::getOSABI(STI.getTargetTriple().getOS());
  return new AGXAsmBackend(OSABI);
}
