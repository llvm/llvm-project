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
  AMDGPUMCObjectWriter(raw_ostream &OS) : MCObjectWriter(OS, true) { }
  void ExecutePostLayoutBinding(MCAssembler &Asm,
                                const MCAsmLayout &Layout) override {
    //XXX: Implement if necessary.
  }
  void RecordRelocation(const MCAssembler &Asm, const MCAsmLayout &Layout,
                        const MCFragment *Fragment, const MCFixup &Fixup,
                        MCValue Target, bool &IsPCRel,
                        uint64_t &FixedValue) override {
    assert(!"Not implemented");
  }

  void WriteObject(MCAssembler &Asm, const MCAsmLayout &Layout) override;

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

void AMDGPUMCObjectWriter::WriteObject(MCAssembler &Asm,
                                       const MCAsmLayout &Layout) {
  for (MCAssembler::iterator I = Asm.begin(), E = Asm.end(); I != E; ++I) {
    Asm.writeSectionData(I, Layout);
  }
}

void AMDGPUAsmBackend::applyFixup(const MCFixup &Fixup, char *Data,
                                  unsigned DataSize, uint64_t Value,
                                  bool IsPCRel) const {

  switch ((unsigned)Fixup.getKind()) {
    default: llvm_unreachable("Unknown fixup kind");
    case AMDGPU::fixup_si_sopp_br: {
      uint16_t *Dst = (uint16_t*)(Data + Fixup.getOffset());
      *Dst = (Value - 4) / 4;
      break;
    }

    case AMDGPU::fixup_si_rodata: {
      uint32_t *Dst = (uint32_t*)(Data + Fixup.getOffset());
      *Dst = Value;
      break;
    }

    case AMDGPU::fixup_si_end_of_text: {
      uint32_t *Dst = (uint32_t*)(Data + Fixup.getOffset());
      // The value points to the last instruction in the text section, so we
      // need to add 4 bytes to get to the start of the constants.
      *Dst = Value + 4;
      break;
    }
  }
}

const MCFixupKindInfo &AMDGPUAsmBackend::getFixupKindInfo(
                                                       MCFixupKind Kind) const {
  const static MCFixupKindInfo Infos[AMDGPU::NumTargetFixupKinds] = {
    // name                   offset bits  flags
    { "fixup_si_sopp_br",     0,     16,   MCFixupKindInfo::FKF_IsPCRel },
    { "fixup_si_rodata",      0,     32,   0 },
    { "fixup_si_end_of_text", 0,     32,   MCFixupKindInfo::FKF_IsPCRel }
  };

  if (Kind < FirstTargetFixupKind)
    return MCAsmBackend::getFixupKindInfo(Kind);

  return Infos[Kind - FirstTargetFixupKind];
}

bool AMDGPUAsmBackend::writeNopData(uint64_t Count, MCObjectWriter *OW) const {
  for (unsigned i = 0; i < Count; ++i)
    OW->Write8(0);

  return true;
}

//===----------------------------------------------------------------------===//
// ELFAMDGPUAsmBackend class
//===----------------------------------------------------------------------===//

namespace {

class ELFAMDGPUAsmBackend : public AMDGPUAsmBackend {
public:
  ELFAMDGPUAsmBackend(const Target &T) : AMDGPUAsmBackend(T) { }

  MCObjectWriter *createObjectWriter(raw_ostream &OS) const override {
    return createAMDGPUELFObjectWriter(OS);
  }
};

} // end anonymous namespace

MCAsmBackend *llvm::createAMDGPUAsmBackend(const Target &T,
                                           const MCRegisterInfo &MRI,
                                           StringRef TT,
                                           StringRef CPU) {
  return new ELFAMDGPUAsmBackend(T);
}
