//===-- AMDGPUAsmBackend.cpp - AMDGPU Assembler Backend -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
/// \file
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "MCTargetDesc/AMDGPUFixupKinds.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCFixupKindInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

namespace {

class AMDGPUAsmBackend : public MCAsmBackend {
public:
  AMDGPUAsmBackend(const Target &T)
    : MCAsmBackend() {}

  unsigned getNumFixupKinds() const override { return AMDGPU::NumTargetFixupKinds; };

  void processFixupValue(const MCAssembler &Asm,
                         const MCAsmLayout &Layout,
                         const MCFixup &Fixup, const MCFragment *DF,
                         const MCValue &Target, uint64_t &Value,
                         bool &IsResolved) override;

  void applyFixup(const MCFixup &Fixup, char *Data, unsigned DataSize,
                  uint64_t Value, bool IsPCRel, MCContext &Ctx) const override;
  bool fixupNeedsRelaxation(const MCFixup &Fixup, uint64_t Value,
                            const MCRelaxableFragment *DF,
                            const MCAsmLayout &Layout) const override {
    return false;
  }
  void relaxInstruction(const MCInst &Inst, const MCSubtargetInfo &STI,
                        MCInst &Res) const override {
    llvm_unreachable("Not implemented");
  }
  bool mayNeedRelaxation(const MCInst &Inst) const override { return false; }
  bool writeNopData(uint64_t Count, MCObjectWriter *OW) const override;

  const MCFixupKindInfo &getFixupKindInfo(MCFixupKind Kind) const override;
};

} //End anonymous namespace

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

  switch (Fixup.getKind()) {
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

void AMDGPUAsmBackend::processFixupValue(const MCAssembler &Asm,
                                         const MCAsmLayout &Layout,
                                         const MCFixup &Fixup, const MCFragment *DF,
                                         const MCValue &Target, uint64_t &Value,
                                         bool &IsResolved) {
  MCValue Res;

  // When we have complex expressions like: BB0_1 + (BB0_2 - 4), which are
  // used for long branches, this function will be called with
  // IsResolved = false and Value set to some pre-computed value.  In
  // the example above, the value would be:
  // (BB0_1 + (BB0_2 - 4)) - CurrentOffsetFromStartOfFunction.
  // This is not what we want.  We just want the expression computation
  // only.  The reason the MC layer subtracts the current offset from the
  // expression is because the fixup is of kind FK_PCRel_4.
  // For these scenarios, evaluateAsValue gives us the computation that we
  // want.
  if (!IsResolved && Fixup.getValue()->evaluateAsValue(Res, Layout) &&
      Res.isAbsolute()) {
    Value = Res.getConstant();
    IsResolved = true;

  }
  if (IsResolved)
    Value = adjustFixupValue(Fixup, Value, &Asm.getContext());
}

void AMDGPUAsmBackend::applyFixup(const MCFixup &Fixup, char *Data,
                                  unsigned DataSize, uint64_t Value,
                                  bool IsPCRel, MCContext &Ctx) const {
  if (!Value)
    return; // Doesn't change encoding.

  MCFixupKindInfo Info = getFixupKindInfo(Fixup.getKind());

  // Shift the value into position.
  Value <<= Info.TargetOffset;

  unsigned NumBytes = getFixupKindNumBytes(Fixup.getKind());
  uint32_t Offset = Fixup.getOffset();
  assert(Offset + NumBytes <= DataSize && "Invalid fixup offset!");

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

bool AMDGPUAsmBackend::writeNopData(uint64_t Count, MCObjectWriter *OW) const {
  // If the count is not 4-byte aligned, we must be writing data into the text
  // section (otherwise we have unaligned instructions, and thus have far
  // bigger problems), so just write zeros instead.
  OW->WriteZeros(Count % 4);

  // We are properly aligned, so write NOPs as requested.
  Count /= 4;

  // FIXME: R600 support.
  // s_nop 0
  const uint32_t Encoded_S_NOP_0 = 0xbf800000;

  for (uint64_t I = 0; I != Count; ++I)
    OW->write32(Encoded_S_NOP_0);

  return true;
}

//===----------------------------------------------------------------------===//
// ELFAMDGPUAsmBackend class
//===----------------------------------------------------------------------===//

namespace {

class ELFAMDGPUAsmBackend : public AMDGPUAsmBackend {
  bool Is64Bit;
  bool HasRelocationAddend;

public:
  ELFAMDGPUAsmBackend(const Target &T, const Triple &TT) :
      AMDGPUAsmBackend(T), Is64Bit(TT.getArch() == Triple::amdgcn),
      HasRelocationAddend(TT.getOS() == Triple::AMDHSA) { }

  MCObjectWriter *createObjectWriter(raw_pwrite_stream &OS) const override {
    return createAMDGPUELFObjectWriter(Is64Bit, HasRelocationAddend, OS);
  }
};

} // end anonymous namespace

MCAsmBackend *llvm::createAMDGPUAsmBackend(const Target &T,
                                           const MCRegisterInfo &MRI,
                                           const Triple &TT, StringRef CPU,
                                           const MCTargetOptions &Options) {
  // Use 64-bit ELF for amdgcn
  return new ELFAMDGPUAsmBackend(T, TT);
}
