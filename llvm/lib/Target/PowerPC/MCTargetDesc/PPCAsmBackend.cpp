//===-- PPCAsmBackend.cpp - PPC Assembler Backend -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/PPCFixupKinds.h"
#include "MCTargetDesc/PPCMCTargetDesc.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCFixupKindInfo.h"
#include "llvm/MC/MCMachObjectWriter.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCSymbolELF.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"
using namespace llvm;

static uint64_t adjustFixupValue(unsigned Kind, uint64_t Value) {
  switch (Kind) {
  default:
    llvm_unreachable("Unknown fixup kind!");
  case FK_Data_1:
  case FK_Data_2:
  case FK_Data_4:
  case FK_Data_8:
  case PPC::fixup_ppc_nofixup:
    return Value;
  case PPC::fixup_ppc_brcond14:
  case PPC::fixup_ppc_brcond14abs:
    return Value & 0xfffc;
  case PPC::fixup_ppc_br24:
  case PPC::fixup_ppc_br24abs:
    return Value & 0x3fffffc;
  case PPC::fixup_ppc_half16:
    return Value & 0xffff;
  case PPC::fixup_ppc_half16ds:
    return Value & 0xfffc;
  }
}

static unsigned getFixupKindNumBytes(unsigned Kind) {
  switch (Kind) {
  default:
    llvm_unreachable("Unknown fixup kind!");
  case FK_Data_1:
    return 1;
  case FK_Data_2:
  case PPC::fixup_ppc_half16:
  case PPC::fixup_ppc_half16ds:
    return 2;
  case FK_Data_4:
  case PPC::fixup_ppc_brcond14:
  case PPC::fixup_ppc_brcond14abs:
  case PPC::fixup_ppc_br24:
  case PPC::fixup_ppc_br24abs:
    return 4;
  case FK_Data_8:
    return 8;
  case PPC::fixup_ppc_nofixup:
    return 0;
  }
}

namespace {

class PPCAsmBackend : public MCAsmBackend {
  const Target &TheTarget;
  bool IsLittleEndian;
public:
  PPCAsmBackend(const Target &T, bool isLittle) : MCAsmBackend(), TheTarget(T),
    IsLittleEndian(isLittle) {}

  unsigned getNumFixupKinds() const override {
    return PPC::NumTargetFixupKinds;
  }

  const MCFixupKindInfo &getFixupKindInfo(MCFixupKind Kind) const override {
    const static MCFixupKindInfo InfosBE[PPC::NumTargetFixupKinds] = {
      // name                    offset  bits  flags
      { "fixup_ppc_br24",        6,      24,   MCFixupKindInfo::FKF_IsPCRel },
      { "fixup_ppc_brcond14",    16,     14,   MCFixupKindInfo::FKF_IsPCRel },
      { "fixup_ppc_br24abs",     6,      24,   0 },
      { "fixup_ppc_brcond14abs", 16,     14,   0 },
      { "fixup_ppc_half16",       0,     16,   0 },
      { "fixup_ppc_half16ds",     0,     14,   0 },
      { "fixup_ppc_nofixup",      0,      0,   0 }
    };
    const static MCFixupKindInfo InfosLE[PPC::NumTargetFixupKinds] = {
      // name                    offset  bits  flags
      { "fixup_ppc_br24",        2,      24,   MCFixupKindInfo::FKF_IsPCRel },
      { "fixup_ppc_brcond14",    2,      14,   MCFixupKindInfo::FKF_IsPCRel },
      { "fixup_ppc_br24abs",     2,      24,   0 },
      { "fixup_ppc_brcond14abs", 2,      14,   0 },
      { "fixup_ppc_half16",      0,      16,   0 },
      { "fixup_ppc_half16ds",    2,      14,   0 },
      { "fixup_ppc_nofixup",     0,       0,   0 }
    };

    if (Kind < FirstTargetFixupKind)
      return MCAsmBackend::getFixupKindInfo(Kind);

    assert(unsigned(Kind - FirstTargetFixupKind) < getNumFixupKinds() &&
           "Invalid kind!");
    return (IsLittleEndian? InfosLE : InfosBE)[Kind - FirstTargetFixupKind];
  }

  void applyFixup(const MCFixup &Fixup, char *Data, unsigned DataSize,
                  uint64_t Value, bool IsPCRel, MCContext &Ctx) const override {
    Value = adjustFixupValue(Fixup.getKind(), Value);
    if (!Value) return;           // Doesn't change encoding.

    unsigned Offset = Fixup.getOffset();
    unsigned NumBytes = getFixupKindNumBytes(Fixup.getKind());

    // For each byte of the fragment that the fixup touches, mask in the bits
    // from the fixup value. The Value has been "split up" into the appropriate
    // bitfields above.
    for (unsigned i = 0; i != NumBytes; ++i) {
      unsigned Idx = IsLittleEndian ? i : (NumBytes - 1 - i);
      Data[Offset + i] |= uint8_t((Value >> (Idx * 8)) & 0xff);
    }
  }

  void processFixupValue(const MCAssembler &Asm, const MCAsmLayout &Layout,
                         const MCFixup &Fixup, const MCFragment *DF,
                         const MCValue &Target, uint64_t &Value,
                         bool &IsResolved) override {
    switch ((PPC::Fixups)Fixup.getKind()) {
    default: break;
    case PPC::fixup_ppc_br24:
    case PPC::fixup_ppc_br24abs:
      // If the target symbol has a local entry point we must not attempt
      // to resolve the fixup directly.  Emit a relocation and leave
      // resolution of the final target address to the linker.
      if (const MCSymbolRefExpr *A = Target.getSymA()) {
        if (const auto *S = dyn_cast<MCSymbolELF>(&A->getSymbol())) {
          // The "other" values are stored in the last 6 bits of the second
          // byte. The traditional defines for STO values assume the full byte
          // and thus the shift to pack it.
          unsigned Other = S->getOther() << 2;
          if ((Other & ELF::STO_PPC64_LOCAL_MASK) != 0)
            IsResolved = false;
        }
      }
      break;
    }
  }

  bool mayNeedRelaxation(const MCInst &Inst) const override {
    // FIXME.
    return false;
  }

  bool fixupNeedsRelaxation(const MCFixup &Fixup,
                            uint64_t Value,
                            const MCRelaxableFragment *DF,
                            const MCAsmLayout &Layout) const override {
    // FIXME.
    llvm_unreachable("relaxInstruction() unimplemented");
  }

  void relaxInstruction(const MCInst &Inst, const MCSubtargetInfo &STI,
                        MCInst &Res) const override {
    // FIXME.
    llvm_unreachable("relaxInstruction() unimplemented");
  }

  bool writeNopData(uint64_t Count, MCObjectWriter *OW) const override {
    uint64_t NumNops = Count / 4;
    for (uint64_t i = 0; i != NumNops; ++i)
      OW->write32(0x60000000);

    OW->WriteZeros(Count % 4);

    return true;
  }

  unsigned getPointerSize() const {
    StringRef Name = TheTarget.getName();
    if (Name == "ppc64" || Name == "ppc64le") return 8;
    assert(Name == "ppc32" && "Unknown target name!");
    return 4;
  }

  bool isLittleEndian() const {
    return IsLittleEndian;
  }
};
} // end anonymous namespace


// FIXME: This should be in a separate file.
namespace {
  class DarwinPPCAsmBackend : public PPCAsmBackend {
  public:
    DarwinPPCAsmBackend(const Target &T) : PPCAsmBackend(T, false) { }

    MCObjectWriter *createObjectWriter(raw_pwrite_stream &OS) const override {
      bool is64 = getPointerSize() == 8;
      return createPPCMachObjectWriter(
          OS,
          /*Is64Bit=*/is64,
          (is64 ? MachO::CPU_TYPE_POWERPC64 : MachO::CPU_TYPE_POWERPC),
          MachO::CPU_SUBTYPE_POWERPC_ALL);
    }
  };

  class ELFPPCAsmBackend : public PPCAsmBackend {
    uint8_t OSABI;
  public:
    ELFPPCAsmBackend(const Target &T, bool IsLittleEndian, uint8_t OSABI) :
      PPCAsmBackend(T, IsLittleEndian), OSABI(OSABI) { }

    MCObjectWriter *createObjectWriter(raw_pwrite_stream &OS) const override {
      bool is64 = getPointerSize() == 8;
      return createPPCELFObjectWriter(OS, is64, isLittleEndian(), OSABI);
    }
  };

} // end anonymous namespace

MCAsmBackend *llvm::createPPCAsmBackend(const Target &T,
                                        const MCRegisterInfo &MRI,
                                        const Triple &TT, StringRef CPU,
                                        const MCTargetOptions &Options) {
  if (TT.isOSDarwin())
    return new DarwinPPCAsmBackend(T);

  uint8_t OSABI = MCELFObjectTargetWriter::getOSABI(TT.getOS());
  bool IsLittleEndian = TT.getArch() == Triple::ppc64le;
  return new ELFPPCAsmBackend(T, IsLittleEndian, OSABI);
}
