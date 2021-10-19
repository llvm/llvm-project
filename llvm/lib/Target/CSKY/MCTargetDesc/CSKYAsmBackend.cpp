//===-- CSKYAsmBackend.cpp - CSKY Assembler Backend -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CSKYAsmBackend.h"
#include "MCTargetDesc/CSKYMCTargetDesc.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/MC/MCAsmLayout.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCFixupKindInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "csky-asmbackend"

using namespace llvm;

std::unique_ptr<MCObjectTargetWriter>
CSKYAsmBackend::createObjectTargetWriter() const {
  return createCSKYELFObjectWriter();
}

const MCFixupKindInfo &
CSKYAsmBackend::getFixupKindInfo(MCFixupKind Kind) const {

  static llvm::DenseMap<unsigned, MCFixupKindInfo> Infos = {
      {CSKY::Fixups::fixup_csky_addr32, {"fixup_csky_addr32", 0, 32, 0}},
      {CSKY::Fixups::fixup_csky_pcrel_imm16_scale2,
       {"fixup_csky_pcrel_imm16_scale2", 0, 32, MCFixupKindInfo::FKF_IsPCRel}},
      {CSKY::Fixups::fixup_csky_pcrel_uimm16_scale4,
       {"fixup_csky_pcrel_uimm16_scale4", 0, 32, MCFixupKindInfo::FKF_IsPCRel}},
      {CSKY::Fixups::fixup_csky_pcrel_imm26_scale2,
       {"fixup_csky_pcrel_imm26_scale2", 0, 32, MCFixupKindInfo::FKF_IsPCRel}},
      {CSKY::Fixups::fixup_csky_pcrel_imm18_scale2,
       {"fixup_csky_pcrel_imm18_scale2", 0, 32, MCFixupKindInfo::FKF_IsPCRel}}};
  assert(Infos.size() == CSKY::NumTargetFixupKinds &&
         "Not all fixup kinds added to Infos array");

  assert(unsigned(Kind - FirstTargetFixupKind) < getNumFixupKinds() &&
         "Invalid kind!");
  if (FirstTargetFixupKind <= Kind && Kind < FirstLiteralRelocationKind)
    return Infos[Kind];
  else if (Kind < FirstTargetFixupKind)
    return MCAsmBackend::getFixupKindInfo(Kind);
  else
    return MCAsmBackend::getFixupKindInfo(FK_NONE);
}

static uint64_t adjustFixupValue(const MCFixup &Fixup, uint64_t Value,
                                 MCContext &Ctx) {
  switch (Fixup.getTargetKind()) {
  default:
    llvm_unreachable("Unknown fixup kind!");
  case FK_Data_1:
  case FK_Data_2:
  case FK_Data_4:
  case FK_Data_8:
    return Value;
  case CSKY::fixup_csky_addr32:
    return Value & 0xffffffff;
  case CSKY::fixup_csky_pcrel_imm16_scale2:
    if (!isIntN(17, Value))
      Ctx.reportError(Fixup.getLoc(), "out of range pc-relative fixup value.");
    if (Value & 0x1)
      Ctx.reportError(Fixup.getLoc(), "fixup value must be 2-byte aligned.");

    return (Value >> 1) & 0xffff;
  case CSKY::fixup_csky_pcrel_uimm16_scale4:
    if (!isUIntN(18, Value))
      Ctx.reportError(Fixup.getLoc(), "out of range pc-relative fixup value.");
    if (Value & 0x3)
      Ctx.reportError(Fixup.getLoc(), "fixup value must be 4-byte aligned.");

    return (Value >> 2) & 0xffff;
  case CSKY::fixup_csky_pcrel_imm26_scale2:
    if (!isIntN(27, Value))
      Ctx.reportError(Fixup.getLoc(), "out of range pc-relative fixup value.");
    if (Value & 0x1)
      Ctx.reportError(Fixup.getLoc(), "fixup value must be 2-byte aligned.");

    return (Value >> 1) & 0x3ffffff;
  case CSKY::fixup_csky_pcrel_imm18_scale2:
    if (!isIntN(19, Value))
      Ctx.reportError(Fixup.getLoc(), "out of range pc-relative fixup value.");
    if (Value & 0x1)
      Ctx.reportError(Fixup.getLoc(), "fixup value must be 2-byte aligned.");

    return (Value >> 1) & 0x3ffff;
  }
}

void CSKYAsmBackend::applyFixup(const MCAssembler &Asm, const MCFixup &Fixup,
                                const MCValue &Target,
                                MutableArrayRef<char> Data, uint64_t Value,
                                bool IsResolved,
                                const MCSubtargetInfo *STI) const {
  MCFixupKind Kind = Fixup.getKind();
  if (Kind >= FirstLiteralRelocationKind)
    return;
  MCContext &Ctx = Asm.getContext();
  MCFixupKindInfo Info = getFixupKindInfo(Kind);
  if (!Value)
    return; // Doesn't change encoding.
  // Apply any target-specific value adjustments.
  Value = adjustFixupValue(Fixup, Value, Ctx);

  // Shift the value into position.
  Value <<= Info.TargetOffset;

  unsigned Offset = Fixup.getOffset();
  unsigned NumBytes = alignTo(Info.TargetSize + Info.TargetOffset, 8) / 8;

  assert(Offset + NumBytes <= Data.size() && "Invalid fixup offset!");

  // For each byte of the fragment that the fixup touches, mask in the
  // bits from the fixup value.
  bool IsLittleEndian = (Endian == support::little);

  if (IsLittleEndian && (NumBytes == 4)) {
    Data[Offset + 0] |= uint8_t((Value >> 16) & 0xff);
    Data[Offset + 1] |= uint8_t((Value >> 24) & 0xff);
    Data[Offset + 2] |= uint8_t(Value & 0xff);
    Data[Offset + 3] |= uint8_t((Value >> 8) & 0xff);
  } else {
    for (unsigned I = 0; I != NumBytes; I++) {
      unsigned Idx = IsLittleEndian ? I : (NumBytes - 1 - I);
      Data[Offset + Idx] |= uint8_t((Value >> (I * 8)) & 0xff);
    }
  }
}

bool CSKYAsmBackend::fixupNeedsRelaxation(const MCFixup &Fixup, uint64_t Value,
                                          const MCRelaxableFragment *DF,
                                          const MCAsmLayout &Layout) const {
  return false;
}

void CSKYAsmBackend::relaxInstruction(MCInst &Inst,
                                      const MCSubtargetInfo &STI) const {
  llvm_unreachable("CSKYAsmBackend::relaxInstruction() unimplemented");
}

bool CSKYAsmBackend::writeNopData(raw_ostream &OS, uint64_t Count,
                                  const MCSubtargetInfo *STI) const {
  if (Count % 2)
    return false;

  // MOV32 r0, r0
  while (Count >= 4) {
    OS.write("\xc4\x00\x48\x20", 4);
    Count -= 4;
  }
  // MOV16 r0, r0
  if (Count)
    OS.write("\x6c\x03", 2);

  return true;
}

MCAsmBackend *llvm::createCSKYAsmBackend(const Target &T,
                                         const MCSubtargetInfo &STI,
                                         const MCRegisterInfo &MRI,
                                         const MCTargetOptions &Options) {
  return new CSKYAsmBackend(STI, Options);
}
