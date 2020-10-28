//===-- AMDGPUAsmBackend.cpp - AMDGPU Assembler Backend -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
/// \file
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/AMDGPUFixupKinds.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCFixupKindInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/TargetRegistry.h"
#include "Utils/AMDGPUBaseInfo.h"

using namespace llvm;
using namespace llvm::AMDGPU;

namespace {

class AMDGPUAsmBackend : public MCAsmBackend {
public:
  AMDGPUAsmBackend(const Target &T) : MCAsmBackend(support::little) {}

  unsigned getNumFixupKinds() const override { return AMDGPU::NumTargetFixupKinds; };

  void applyFixup(const MCAssembler &Asm, const MCFixup &Fixup,
                  const MCValue &Target, MutableArrayRef<char> Data,
                  uint64_t Value, bool IsResolved,
                  const MCSubtargetInfo *STI) const override;
  bool fixupNeedsRelaxation(const MCFixup &Fixup, uint64_t Value,
                            const MCRelaxableFragment *DF,
                            const MCAsmLayout &Layout) const override;

  void relaxInstruction(MCInst &Inst,
                        const MCSubtargetInfo &STI) const override;

  bool mayNeedRelaxation(const MCInst &Inst,
                         const MCSubtargetInfo &STI) const override;

  unsigned getMinimumNopSize() const override;
  bool writeNopData(raw_ostream &OS, uint64_t Count) const override;

  const MCFixupKindInfo &getFixupKindInfo(MCFixupKind Kind) const override;
};

} //End anonymous namespace

void AMDGPUAsmBackend::relaxInstruction(MCInst &Inst,
                                        const MCSubtargetInfo &STI) const {
  MCInst Res;
  unsigned RelaxedOpcode = AMDGPU::getSOPPWithRelaxation(Inst.getOpcode());
  Res.setOpcode(RelaxedOpcode);
  Res.addOperand(Inst.getOperand(0));
  Inst = std::move(Res);
  return;
}

bool AMDGPUAsmBackend::fixupNeedsRelaxation(const MCFixup &Fixup,
                                            uint64_t Value,
                                            const MCRelaxableFragment *DF,
                                            const MCAsmLayout &Layout) const {
  // if the branch target has an offset of x3f this needs to be relaxed to
  // add a s_nop 0 immediately after branch to effectively increment offset
  // for hardware workaround in gfx1010
  return (((int64_t(Value)/4)-1) == 0x3f);
}

bool AMDGPUAsmBackend::mayNeedRelaxation(const MCInst &Inst,
                       const MCSubtargetInfo &STI) const {
  if (!STI.getFeatureBits()[AMDGPU::FeatureOffset3fBug])
    return false;

  if (AMDGPU::getSOPPWithRelaxation(Inst.getOpcode()) >= 0)
    return true;

  return false;
}

static unsigned getFixupKindNumBytes(unsigned Kind) {
  switch (Kind) {
  case AMDGPU::fixup_si_sopp_br:
    return 2;
  case FK_SecRel_1:
  case FK_Data_1:
    return 1;
  case FK_SecRel_2:
  case FK_Data_2:
    return 2;
  case FK_SecRel_4:
  case FK_Data_4:
  case FK_PCRel_4:
    return 4;
  case FK_SecRel_8:
  case FK_Data_8:
    return 8;
  default:
    llvm_unreachable("Unknown fixup kind!");
  }
}

static uint64_t adjustFixupValue(const MCFixup &Fixup, uint64_t Value,
                                 MCContext *Ctx) {
  int64_t SignedValue = static_cast<int64_t>(Value);

  switch (Fixup.getTargetKind()) {
  case AMDGPU::fixup_si_sopp_br: {
    int64_t BrImm = (SignedValue - 4) / 4;

    if (Ctx && !isInt<16>(BrImm))
      Ctx->reportError(Fixup.getLoc(), "branch size exceeds simm16");

    return BrImm;
  }
  case FK_Data_1:
  case FK_Data_2:
  case FK_Data_4:
  case FK_Data_8:
  case FK_PCRel_4:
  case FK_SecRel_4:
    return Value;
  default:
    llvm_unreachable("unhandled fixup kind");
  }
}

void AMDGPUAsmBackend::applyFixup(const MCAssembler &Asm, const MCFixup &Fixup,
                                  const MCValue &Target,
                                  MutableArrayRef<char> Data, uint64_t Value,
                                  bool IsResolved,
                                  const MCSubtargetInfo *STI) const {
  Value = adjustFixupValue(Fixup, Value, &Asm.getContext());
  if (!Value)
    return; // Doesn't change encoding.

  MCFixupKindInfo Info = getFixupKindInfo(Fixup.getKind());

  // Shift the value into position.
  Value <<= Info.TargetOffset;

  unsigned NumBytes = getFixupKindNumBytes(Fixup.getKind());
  uint32_t Offset = Fixup.getOffset();
  assert(Offset + NumBytes <= Data.size() && "Invalid fixup offset!");

  // For each byte of the fragment that the fixup touches, mask in the bits from
  // the fixup value.
  for (unsigned i = 0; i != NumBytes; ++i)
    Data[Offset + i] |= static_cast<uint8_t>((Value >> (i * 8)) & 0xff);
}

const MCFixupKindInfo &AMDGPUAsmBackend::getFixupKindInfo(
                                                       MCFixupKind Kind) const {
  const static MCFixupKindInfo Infos[AMDGPU::NumTargetFixupKinds] = {
    // name                   offset bits  flags
    { "fixup_si_sopp_br",     0,     16,   MCFixupKindInfo::FKF_IsPCRel },
  };

  if (Kind < FirstTargetFixupKind)
    return MCAsmBackend::getFixupKindInfo(Kind);

  return Infos[Kind - FirstTargetFixupKind];
}

unsigned AMDGPUAsmBackend::getMinimumNopSize() const {
  return 4;
}

bool AMDGPUAsmBackend::writeNopData(raw_ostream &OS, uint64_t Count) const {
  // If the count is not 4-byte aligned, we must be writing data into the text
  // section (otherwise we have unaligned instructions, and thus have far
  // bigger problems), so just write zeros instead.
  OS.write_zeros(Count % 4);

  // We are properly aligned, so write NOPs as requested.
  Count /= 4;

  // FIXME: R600 support.
  // s_nop 0
  const uint32_t Encoded_S_NOP_0 = 0xbf800000;

  for (uint64_t I = 0; I != Count; ++I)
    support::endian::write<uint32_t>(OS, Encoded_S_NOP_0, Endian);

  return true;
}

//===----------------------------------------------------------------------===//
// ELFAMDGPUAsmBackend class
//===----------------------------------------------------------------------===//

namespace {

class ELFAMDGPUAsmBackend : public AMDGPUAsmBackend {
  bool Is64Bit;
  bool HasRelocationAddend;
  uint8_t OSABI = ELF::ELFOSABI_NONE;
  uint8_t ABIVersion = 0;

public:
  ELFAMDGPUAsmBackend(const Target &T, const Triple &TT, uint8_t ABIVersion) :
      AMDGPUAsmBackend(T), Is64Bit(TT.getArch() == Triple::amdgcn),
      HasRelocationAddend(TT.getOS() == Triple::AMDHSA),
      ABIVersion(ABIVersion) {
    switch (TT.getOS()) {
    case Triple::AMDHSA:
      OSABI = ELF::ELFOSABI_AMDGPU_HSA;
      break;
    case Triple::AMDPAL:
      OSABI = ELF::ELFOSABI_AMDGPU_PAL;
      break;
    case Triple::Mesa3D:
      OSABI = ELF::ELFOSABI_AMDGPU_MESA3D;
      break;
    default:
      break;
    }
  }

  std::unique_ptr<MCObjectTargetWriter>
  createObjectTargetWriter() const override {
    return createAMDGPUELFObjectWriter(Is64Bit, OSABI, HasRelocationAddend,
                                       ABIVersion);
  }
};

} // end anonymous namespace

MCAsmBackend *llvm::createAMDGPUAsmBackend(const Target &T,
                                           const MCSubtargetInfo &STI,
                                           const MCRegisterInfo &MRI,
                                           const MCTargetOptions &Options) {
  return new ELFAMDGPUAsmBackend(T, STI.getTargetTriple(),
                                 getHsaAbiVersion(&STI).getValueOr(0));
}
