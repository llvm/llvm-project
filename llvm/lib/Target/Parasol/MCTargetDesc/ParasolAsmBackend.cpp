//===-- ParasolAsmBackend.cpp - Parasol Assembler Backend -----------------===//
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/ParasolMCTargetDesc.h"
#include "ParasolFixupKinds.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCFixupKindInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCValue.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/EndianStream.h"

using namespace llvm;

namespace {
class ParasolAsmBackend : public MCAsmBackend {
protected:
  const Target &TheTarget;
  bool Is64Bit;

public:
  ParasolAsmBackend(const Target &T)
      : MCAsmBackend(StringRef(T.getName()) == "parasol"
                         ? llvm::endianness::little
                         : llvm::endianness::big),
        TheTarget(T) {}

  unsigned getNumFixupKinds() const override {
    return Parasol::NumTargetFixupKinds;
  }

  std::optional<MCFixupKind> getFixupKind(StringRef Name) const override {
    unsigned Type;
    Type = llvm::StringSwitch<unsigned>(Name)
#define ELF_RELOC(X, Y) .Case(#X, Y)
#include "llvm/BinaryFormat/ELFRelocs/Parasol.def"
#undef ELF_RELOC
               .Default(-1u);
    if (Type == -1u)
      return std::nullopt;
    return static_cast<MCFixupKind>(FirstLiteralRelocationKind + Type);
  }

  const MCFixupKindInfo &getFixupKindInfo(MCFixupKind Kind) const override {
    static const MCFixupKindInfo Infos[Parasol::NumTargetFixupKinds] = {
        {"fixup_br_one_reg_imm", 14, 32, MCFixupKindInfo::FKF_IsPCRel},
        {"fixup_br_imm", 8, 32, MCFixupKindInfo::FKF_IsPCRel},
    };

    if (Kind < FirstTargetFixupKind)
      return MCAsmBackend::getFixupKindInfo(Kind);

    assert(unsigned(Kind - FirstTargetFixupKind) < getNumFixupKinds() &&
           "Invalid kind!");
    return Infos[Kind - FirstTargetFixupKind];
  }

  bool shouldForceRelocation(const MCAssembler &Asm, const MCFixup &Fixup,
                             const MCValue &Target,
                             const MCSubtargetInfo *STI) override {
    return false;
  }

  /// fixupNeedsRelaxation - Target specific predicate for whether a given
  /// fixup requires the associated instruction to be relaxed.
  bool fixupNeedsRelaxation(const MCFixup &Fixup, uint64_t Value,
                            const MCRelaxableFragment *DF,
                            const MCAsmLayout &Layout) const override {
    // FIXME.
    llvm_unreachable("fixupNeedsRelaxation() unimplemented");
    return false;
  }
  void relaxInstruction(MCInst &Inst,
                        const MCSubtargetInfo &STI) const override {
    // FIXME.
    llvm_unreachable("relaxInstruction() unimplemented");
  }

  bool writeNopData(raw_ostream &OS, uint64_t Count,
                    const MCSubtargetInfo *STI) const override {
    // Cannot emit NOP with size not multiple of 32 bits.
    if (Count % 8 != 0)
      return false;

    uint64_t NumNops = Count / 8;
    for (uint64_t i = 0; i != NumNops; ++i)
      support::endian::write<uint64_t>(OS, 0x01000000, Endian);

    return true;
  }
};

class ELFParasolAsmBackend : public ParasolAsmBackend {
  Triple::OSType OSType;

public:
  ELFParasolAsmBackend(const Target &T, Triple::OSType OSType)
      : ParasolAsmBackend(T), OSType(OSType) {}

  void applyFixup(const MCAssembler &Asm, const MCFixup &Fixup,
                  const MCValue &Target, MutableArrayRef<char> Data,
                  uint64_t Value, bool IsResolved,
                  const MCSubtargetInfo *STI) const override {

    if (Fixup.getKind() >= FirstLiteralRelocationKind)
      return;

    auto fixup_info = getFixupKindInfo(Fixup.getKind());
    Value = (Value & 0xFFFFFFFF) << fixup_info.TargetOffset;
    if (!Value)
      return; // Doesn't change encoding.

    unsigned NumBytes =
        (fixup_info.TargetOffset + fixup_info.TargetSize + 7) / 8;
    unsigned Offset = Fixup.getOffset();
    // For each byte of the fragment that the fixup touches, mask in the bits
    // from the fixup value. The Value has been "split up" into the
    // appropriate bitfields above.
    for (unsigned i = 0; i != NumBytes; ++i) {
      unsigned Idx =
          Endian == llvm::endianness::little ? i : (NumBytes - 1) - i;
      Data[Offset + Idx] |= uint8_t((Value >> (i * 8)) & 0xff);
    }
  }

  std::unique_ptr<MCObjectTargetWriter>
  createObjectTargetWriter() const override {
    uint8_t OSABI = MCELFObjectTargetWriter::getOSABI(OSType);
    return createParasolELFObjectWriter(Is64Bit, OSABI);
  }
};

} // end anonymous namespace

MCAsmBackend *llvm::createParasolAsmBackend(const Target &T,
                                            const MCSubtargetInfo &STI,
                                            const MCRegisterInfo &MRI,
                                            const MCTargetOptions &Options) {
  return new ELFParasolAsmBackend(T, STI.getTargetTriple().getOS());
}
