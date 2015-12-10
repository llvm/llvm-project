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
#include "llvm/MC/MCFixupKindInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

namespace {

class AMDGPUMCObjectWriter : public MCObjectWriter {
public:
  AMDGPUMCObjectWriter(raw_pwrite_stream &OS) : MCObjectWriter(OS, true) {}
  void executePostLayoutBinding(MCAssembler &Asm,
                                const MCAsmLayout &Layout) override {
    //XXX: Implement if necessary.
  }
  void recordRelocation(MCAssembler &Asm, const MCAsmLayout &Layout,
                        const MCFragment *Fragment, const MCFixup &Fixup,
                        MCValue Target, bool &IsPCRel,
                        uint64_t &FixedValue) override {
    assert(!"Not implemented");
  }

  void writeObject(MCAssembler &Asm, const MCAsmLayout &Layout) override;

};

class AMDGPUAsmBackend : public MCAsmBackend {
public:
  AMDGPUAsmBackend(const Target &T)
    : MCAsmBackend() {}

  unsigned getNumFixupKinds() const override { return AMDGPU::NumTargetFixupKinds; };
  void applyFixup(const MCFixup &Fixup, char *Data, unsigned DataSize,
                  uint64_t Value, bool IsPCRel) const override;
  bool fixupNeedsRelaxation(const MCFixup &Fixup, uint64_t Value,
                            const MCRelaxableFragment *DF,
                            const MCAsmLayout &Layout) const override {
    return false;
  }
  void relaxInstruction(const MCInst &Inst, MCInst &Res) const override {
    assert(!"Not implemented");
  }
  bool mayNeedRelaxation(const MCInst &Inst) const override { return false; }
  bool writeNopData(uint64_t Count, MCObjectWriter *OW) const override;

  const MCFixupKindInfo &getFixupKindInfo(MCFixupKind Kind) const override;
};

} //End anonymous namespace

void AMDGPUMCObjectWriter::writeObject(MCAssembler &Asm,
                                       const MCAsmLayout &Layout) {
  for (MCAssembler::iterator I = Asm.begin(), E = Asm.end(); I != E; ++I) {
    Asm.writeSectionData(&*I, Layout);
  }
}

static unsigned getFixupKindNumBytes(unsigned Kind) {
  switch (Kind) {
  case FK_Data_1:
    return 1;
  case FK_Data_2:
    return 2;
  case FK_Data_4:
    return 4;
  case FK_Data_8:
    return 8;
  default:
    llvm_unreachable("Unknown fixup kind!");
  }
}

void AMDGPUAsmBackend::applyFixup(const MCFixup &Fixup, char *Data,
                                  unsigned DataSize, uint64_t Value,
                                  bool IsPCRel) const {

  switch ((unsigned)Fixup.getKind()) {
    case AMDGPU::fixup_si_sopp_br: {
      uint16_t *Dst = (uint16_t*)(Data + Fixup.getOffset());
      *Dst = (Value - 4) / 4;
      break;
    }

    case AMDGPU::fixup_si_rodata: {
      uint32_t *Dst = (uint32_t*)(Data + Fixup.getOffset());
      // We emit constant data at the end of the text section and generate its
      // address using the following code sequence:
      // s_getpc_b64 s[0:1]
      // s_add_u32 s0, s0, $symbol
      // s_addc_u32 s1, s1, 0
      //
      // s_getpc_b64 returns the address of the s_add_u32 instruction and then
      // the fixup replaces $symbol with a literal constant, which is a
      // pc-relative  offset from the encoding of the $symbol operand to the
      // constant data.
      //
      // What we want here is an offset from the start of the s_add_u32
      // instruction to the constant data, but since the encoding of $symbol
      // starts 4 bytes after the start of the add instruction, we end up
      // with an offset that is 4 bytes too small.  This requires us to
      // add 4 to the fixup value before applying it.
      *Dst = Value + 4;
      break;
    }
    default: {
      // FIXME: Copied from AArch64
      unsigned NumBytes = getFixupKindNumBytes(Fixup.getKind());
      if (!Value)
        return; // Doesn't change encoding.
      MCFixupKindInfo Info = getFixupKindInfo(Fixup.getKind());

      // Shift the value into position.
      Value <<= Info.TargetOffset;

      unsigned Offset = Fixup.getOffset();
      assert(Offset + NumBytes <= DataSize && "Invalid fixup offset!");

      // For each byte of the fragment that the fixup touches, mask in the
      // bits from the fixup value.
      for (unsigned i = 0; i != NumBytes; ++i)
        Data[Offset + i] |= uint8_t((Value >> (i * 8)) & 0xff);
    }
  }
}

const MCFixupKindInfo &AMDGPUAsmBackend::getFixupKindInfo(
                                                       MCFixupKind Kind) const {
  const static MCFixupKindInfo Infos[AMDGPU::NumTargetFixupKinds] = {
    // name                   offset bits  flags
    { "fixup_si_sopp_br",     0,     16,   MCFixupKindInfo::FKF_IsPCRel },
    { "fixup_si_rodata",      0,     32,   MCFixupKindInfo::FKF_IsPCRel }
  };

  if (Kind < FirstTargetFixupKind)
    return MCAsmBackend::getFixupKindInfo(Kind);

  return Infos[Kind - FirstTargetFixupKind];
}

bool AMDGPUAsmBackend::writeNopData(uint64_t Count, MCObjectWriter *OW) const {
  OW->WriteZeros(Count);

  return true;
}

//===----------------------------------------------------------------------===//
// ELFAMDGPUAsmBackend class
//===----------------------------------------------------------------------===//

namespace {

class ELFAMDGPUAsmBackend : public AMDGPUAsmBackend {
  bool Is64Bit;

public:
  ELFAMDGPUAsmBackend(const Target &T, bool Is64Bit) :
      AMDGPUAsmBackend(T), Is64Bit(Is64Bit) { }

  MCObjectWriter *createObjectWriter(raw_pwrite_stream &OS) const override {
    return createAMDGPUELFObjectWriter(Is64Bit, OS);
  }
};

} // end anonymous namespace

MCAsmBackend *llvm::createAMDGPUAsmBackend(const Target &T,
                                           const MCRegisterInfo &MRI,
                                           const Triple &TT, StringRef CPU) {
  Triple TargetTriple(TT);

  // Use 64-bit ELF for amdgcn
  return new ELFAMDGPUAsmBackend(T, TargetTriple.getArch() == Triple::amdgcn);
}
