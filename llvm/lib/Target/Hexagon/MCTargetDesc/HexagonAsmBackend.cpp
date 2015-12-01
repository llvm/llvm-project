//===-- HexagonAsmBackend.cpp - Hexagon Assembler Backend -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Hexagon.h"
#include "HexagonFixupKinds.h"
#include "HexagonMCTargetDesc.h"
#include "MCTargetDesc/HexagonBaseInfo.h"
#include "MCTargetDesc/HexagonMCInstrInfo.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmLayout.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCFixupKindInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;
using namespace Hexagon;

#define DEBUG_TYPE "hexagon-asm-backend"

namespace {

class HexagonAsmBackend : public MCAsmBackend {
  uint8_t OSABI;
  StringRef CPU;
  mutable uint64_t relaxedCnt;
  std::unique_ptr <MCInstrInfo> MCII;
  std::unique_ptr <MCInst *> RelaxTarget;
  MCInst * Extender;
public:
  HexagonAsmBackend(Target const &T,  uint8_t OSABI, StringRef CPU) :
    OSABI(OSABI), MCII (T.createMCInstrInfo()), RelaxTarget(new MCInst *),
    Extender(nullptr) {}

  MCObjectWriter *createObjectWriter(raw_pwrite_stream &OS) const override {
    return createHexagonELFObjectWriter(OS, OSABI, CPU);
  }

  void setExtender(MCContext &Context) const {
    if (Extender == nullptr)
      const_cast<HexagonAsmBackend *>(this)->Extender = new (Context) MCInst;
  }

  MCInst *takeExtender() const {
    assert(Extender != nullptr);
    MCInst * Result = Extender;
    const_cast<HexagonAsmBackend *>(this)->Extender = nullptr;
    return Result;
  }

  unsigned getNumFixupKinds() const override {
    return Hexagon::NumTargetFixupKinds;
  }

  const MCFixupKindInfo &getFixupKindInfo(MCFixupKind Kind) const override {
    const static MCFixupKindInfo Infos[Hexagon::NumTargetFixupKinds] = {
        // This table *must* be in same the order of fixup_* kinds in
        // HexagonFixupKinds.h.
        //
        // namei                          offset  bits    flags
        {"fixup_Hexagon_B22_PCREL", 0, 32, MCFixupKindInfo::FKF_IsPCRel},
        {"fixup_Hexagon_B15_PCREL", 0, 32, MCFixupKindInfo::FKF_IsPCRel},
        {"fixup_Hexagon_B7_PCREL", 0, 32, MCFixupKindInfo::FKF_IsPCRel},
        {"fixup_Hexagon_LO16", 0, 32, 0},
        {"fixup_Hexagon_HI16", 0, 32, 0},
        {"fixup_Hexagon_32", 0, 32, 0},
        {"fixup_Hexagon_16", 0, 32, 0},
        {"fixup_Hexagon_8", 0, 32, 0},
        {"fixup_Hexagon_GPREL16_0", 0, 32, 0},
        {"fixup_Hexagon_GPREL16_1", 0, 32, 0},
        {"fixup_Hexagon_GPREL16_2", 0, 32, 0},
        {"fixup_Hexagon_GPREL16_3", 0, 32, 0},
        {"fixup_Hexagon_HL16", 0, 32, 0},
        {"fixup_Hexagon_B13_PCREL", 0, 32, MCFixupKindInfo::FKF_IsPCRel},
        {"fixup_Hexagon_B9_PCREL", 0, 32, MCFixupKindInfo::FKF_IsPCRel},
        {"fixup_Hexagon_B32_PCREL_X", 0, 32, MCFixupKindInfo::FKF_IsPCRel},
        {"fixup_Hexagon_32_6_X", 0, 32, 0},
        {"fixup_Hexagon_B22_PCREL_X", 0, 32, MCFixupKindInfo::FKF_IsPCRel},
        {"fixup_Hexagon_B15_PCREL_X", 0, 32, MCFixupKindInfo::FKF_IsPCRel},
        {"fixup_Hexagon_B13_PCREL_X", 0, 32, MCFixupKindInfo::FKF_IsPCRel},
        {"fixup_Hexagon_B9_PCREL_X", 0, 32, MCFixupKindInfo::FKF_IsPCRel},
        {"fixup_Hexagon_B7_PCREL_X", 0, 32, MCFixupKindInfo::FKF_IsPCRel},
        {"fixup_Hexagon_16_X", 0, 32, 0},
        {"fixup_Hexagon_12_X", 0, 32, 0},
        {"fixup_Hexagon_11_X", 0, 32, 0},
        {"fixup_Hexagon_10_X", 0, 32, 0},
        {"fixup_Hexagon_9_X", 0, 32, 0},
        {"fixup_Hexagon_8_X", 0, 32, 0},
        {"fixup_Hexagon_7_X", 0, 32, 0},
        {"fixup_Hexagon_6_X", 0, 32, 0},
        {"fixup_Hexagon_32_PCREL", 0, 32, MCFixupKindInfo::FKF_IsPCRel},
        {"fixup_Hexagon_COPY", 0, 32, 0},
        {"fixup_Hexagon_GLOB_DAT", 0, 32, 0},
        {"fixup_Hexagon_JMP_SLOT", 0, 32, 0},
        {"fixup_Hexagon_RELATIVE", 0, 32, 0},
        {"fixup_Hexagon_PLT_B22_PCREL", 0, 32, MCFixupKindInfo::FKF_IsPCRel},
        {"fixup_Hexagon_GOTREL_LO16", 0, 32, 0},
        {"fixup_Hexagon_GOTREL_HI16", 0, 32, 0},
        {"fixup_Hexagon_GOTREL_32", 0, 32, 0},
        {"fixup_Hexagon_GOT_LO16", 0, 32, 0},
        {"fixup_Hexagon_GOT_HI16", 0, 32, 0},
        {"fixup_Hexagon_GOT_32", 0, 32, 0},
        {"fixup_Hexagon_GOT_16", 0, 32, 0},
        {"fixup_Hexagon_DTPMOD_32", 0, 32, 0},
        {"fixup_Hexagon_DTPREL_LO16", 0, 32, 0},
        {"fixup_Hexagon_DTPREL_HI16", 0, 32, 0},
        {"fixup_Hexagon_DTPREL_32", 0, 32, 0},
        {"fixup_Hexagon_DTPREL_16", 0, 32, 0},
        {"fixup_Hexagon_GD_PLT_B22_PCREL", 0, 32, MCFixupKindInfo::FKF_IsPCRel},
        {"fixup_Hexagon_LD_PLT_B22_PCREL", 0, 32, MCFixupKindInfo::FKF_IsPCRel},
        {"fixup_Hexagon_GD_GOT_LO16", 0, 32, 0},
        {"fixup_Hexagon_GD_GOT_HI16", 0, 32, 0},
        {"fixup_Hexagon_GD_GOT_32", 0, 32, 0},
        {"fixup_Hexagon_GD_GOT_16", 0, 32, 0},
        {"fixup_Hexagon_LD_GOT_LO16", 0, 32, 0},
        {"fixup_Hexagon_LD_GOT_HI16", 0, 32, 0},
        {"fixup_Hexagon_LD_GOT_32", 0, 32, 0},
        {"fixup_Hexagon_LD_GOT_16", 0, 32, 0},
        {"fixup_Hexagon_IE_LO16", 0, 32, 0},
        {"fixup_Hexagon_IE_HI16", 0, 32, 0},
        {"fixup_Hexagon_IE_32", 0, 32, 0},
        {"fixup_Hexagon_IE_16", 0, 32, 0},
        {"fixup_Hexagon_IE_GOT_LO16", 0, 32, 0},
        {"fixup_Hexagon_IE_GOT_HI16", 0, 32, 0},
        {"fixup_Hexagon_IE_GOT_32", 0, 32, 0},
        {"fixup_Hexagon_IE_GOT_16", 0, 32, 0},
        {"fixup_Hexagon_TPREL_LO16", 0, 32, 0},
        {"fixup_Hexagon_TPREL_HI16", 0, 32, 0},
        {"fixup_Hexagon_TPREL_32", 0, 32, 0},
        {"fixup_Hexagon_TPREL_16", 0, 32, 0},
        {"fixup_Hexagon_6_PCREL_X", 0, 32, MCFixupKindInfo::FKF_IsPCRel},
        {"fixup_Hexagon_GOTREL_32_6_X", 0, 32, 0},
        {"fixup_Hexagon_GOTREL_16_X", 0, 32, 0},
        {"fixup_Hexagon_GOTREL_11_X", 0, 32, 0},
        {"fixup_Hexagon_GOT_32_6_X", 0, 32, 0},
        {"fixup_Hexagon_GOT_16_X", 0, 32, 0},
        {"fixup_Hexagon_GOT_11_X", 0, 32, 0},
        {"fixup_Hexagon_DTPREL_32_6_X", 0, 32, 0},
        {"fixup_Hexagon_DTPREL_16_X", 0, 32, 0},
        {"fixup_Hexagon_DTPREL_11_X", 0, 32, 0},
        {"fixup_Hexagon_GD_GOT_32_6_X", 0, 32, 0},
        {"fixup_Hexagon_GD_GOT_16_X", 0, 32, 0},
        {"fixup_Hexagon_GD_GOT_11_X", 0, 32, 0},
        {"fixup_Hexagon_LD_GOT_32_6_X", 0, 32, 0},
        {"fixup_Hexagon_LD_GOT_16_X", 0, 32, 0},
        {"fixup_Hexagon_LD_GOT_11_X", 0, 32, 0},
        {"fixup_Hexagon_IE_32_6_X", 0, 32, 0},
        {"fixup_Hexagon_IE_16_X", 0, 32, 0},
        {"fixup_Hexagon_IE_GOT_32_6_X", 0, 32, 0},
        {"fixup_Hexagon_IE_GOT_16_X", 0, 32, 0},
        {"fixup_Hexagon_IE_GOT_11_X", 0, 32, 0},
        {"fixup_Hexagon_TPREL_32_6_X", 0, 32, 0},
        {"fixup_Hexagon_TPREL_16_X", 0, 32, 0},
        {"fixup_Hexagon_TPREL_11_X", 0, 32, 0}};

    if (Kind < FirstTargetFixupKind) {
      return MCAsmBackend::getFixupKindInfo(Kind);
    }

    assert(unsigned(Kind - FirstTargetFixupKind) < getNumFixupKinds() &&
           "Invalid kind!");
    return Infos[Kind - FirstTargetFixupKind];
  }

  void applyFixup(MCFixup const & /*Fixup*/, char * /*Data*/,
                  unsigned /*DataSize*/, uint64_t /*Value*/,
                  bool /*IsPCRel*/) const override {
    return;
  }

  bool isInstRelaxable(MCInst const &HMI) const {
    const MCInstrDesc &MCID = HexagonMCInstrInfo::getDesc(*MCII, HMI);
    bool Relaxable = false;
    // Branches and loop-setup insns are handled as necessary by relaxation.
    if (llvm::HexagonMCInstrInfo::getType(*MCII, HMI) == HexagonII::TypeJ ||
        (llvm::HexagonMCInstrInfo::getType(*MCII, HMI) == HexagonII::TypeNV &&
         MCID.isBranch()) ||
        (llvm::HexagonMCInstrInfo::getType(*MCII, HMI) == HexagonII::TypeCR &&
         HMI.getOpcode() != Hexagon::C4_addipc))
      if (HexagonMCInstrInfo::isExtendable(*MCII, HMI))
        Relaxable = true;

    return Relaxable;
  }

  /// MayNeedRelaxation - Check whether the given instruction may need
  /// relaxation.
  ///
  /// \param Inst - The instruction to test.
  bool mayNeedRelaxation(MCInst const &Inst) const override {
    assert(HexagonMCInstrInfo::isBundle(Inst));
    bool PreviousIsExtender = false;
    for (auto const &I : HexagonMCInstrInfo::bundleInstructions(Inst)) {
      auto const &Inst = *I.getInst();
      if (!PreviousIsExtender) {
        if (isInstRelaxable(Inst))
          return true;
      }
      PreviousIsExtender = HexagonMCInstrInfo::isImmext(Inst);
    }
    return false;
  }

  /// fixupNeedsRelaxation - Target specific predicate for whether a given
  /// fixup requires the associated instruction to be relaxed.
  bool fixupNeedsRelaxationAdvanced(const MCFixup &Fixup, bool Resolved,
                                    uint64_t Value,
                                    const MCRelaxableFragment *DF,
                                    const MCAsmLayout &Layout) const override {
    MCInst const &MCB = DF->getInst();
    assert(HexagonMCInstrInfo::isBundle(MCB));

    *RelaxTarget = nullptr;
    MCInst &MCI = const_cast<MCInst &>(HexagonMCInstrInfo::instruction(
        MCB, Fixup.getOffset() / HEXAGON_INSTR_SIZE));
    // If we cannot resolve the fixup value, it requires relaxation.
    if (!Resolved) {
      switch ((unsigned)Fixup.getKind()) {
      case fixup_Hexagon_B22_PCREL:
      // GetFixupCount assumes B22 won't relax
      // Fallthrough
      default:
        return false;
        break;
      case fixup_Hexagon_B13_PCREL:
      case fixup_Hexagon_B15_PCREL:
      case fixup_Hexagon_B9_PCREL:
      case fixup_Hexagon_B7_PCREL: {
        if (HexagonMCInstrInfo::bundleSize(MCB) < HEXAGON_PACKET_SIZE) {
          ++relaxedCnt;
          *RelaxTarget = &MCI;
          setExtender(Layout.getAssembler().getContext());
          return true;
        } else {
          return false;
        }
        break;
      }
      }
    }
    bool Relaxable = isInstRelaxable(MCI);
    if (Relaxable == false)
      return false;

    MCFixupKind Kind = Fixup.getKind();
    int64_t sValue = Value;
    int64_t maxValue;

    switch ((unsigned)Kind) {
    case fixup_Hexagon_B7_PCREL:
      maxValue = 1 << 8;
      break;
    case fixup_Hexagon_B9_PCREL:
      maxValue = 1 << 10;
      break;
    case fixup_Hexagon_B15_PCREL:
      maxValue = 1 << 16;
      break;
    case fixup_Hexagon_B22_PCREL:
      maxValue = 1 << 23;
      break;
    default:
      maxValue = INT64_MAX;
      break;
    }

    bool isFarAway = -maxValue > sValue || sValue > maxValue - 1;

    if (isFarAway) {
      if (HexagonMCInstrInfo::bundleSize(MCB) < HEXAGON_PACKET_SIZE) {
        ++relaxedCnt;
        *RelaxTarget = &MCI;
        setExtender(Layout.getAssembler().getContext());
        return true;
      }
    }

    return false;
  }

  /// Simple predicate for targets where !Resolved implies requiring relaxation
  bool fixupNeedsRelaxation(const MCFixup &Fixup, uint64_t Value,
                            const MCRelaxableFragment *DF,
                            const MCAsmLayout &Layout) const override {
    llvm_unreachable("Handled by fixupNeedsRelaxationAdvanced");
  }

  void relaxInstruction(MCInst const & Inst,
                        MCInst & Res) const override {
    assert(HexagonMCInstrInfo::isBundle(Inst) &&
           "Hexagon relaxInstruction only works on bundles");

    Res = HexagonMCInstrInfo::createBundle();
    // Copy the results into the bundle.
    bool Update = false;
    for (auto &I : HexagonMCInstrInfo::bundleInstructions(Inst)) {
      MCInst &CrntHMI = const_cast<MCInst &>(*I.getInst());

      // if immediate extender needed, add it in
      if (*RelaxTarget == &CrntHMI) {
        Update = true;
        assert((HexagonMCInstrInfo::bundleSize(Res) < HEXAGON_PACKET_SIZE) &&
               "No room to insert extender for relaxation");

        MCInst *HMIx = takeExtender();
        *HMIx = HexagonMCInstrInfo::deriveExtender(
                *MCII, CrntHMI,
                HexagonMCInstrInfo::getExtendableOperand(*MCII, CrntHMI));
        Res.addOperand(MCOperand::createInst(HMIx));
        *RelaxTarget = nullptr;
      }
      // now copy over the original instruction(the one we may have extended)
      Res.addOperand(MCOperand::createInst(I.getInst()));
    }
    (void)Update;
    assert(Update && "Didn't find relaxation target");
  }

  bool writeNopData(uint64_t Count,
                    MCObjectWriter * OW) const override {
    static const uint32_t Nopcode  = 0x7f000000, // Hard-coded NOP.
                          ParseIn  = 0x00004000, // In packet parse-bits.
                          ParseEnd = 0x0000c000; // End of packet parse-bits.

    while(Count % HEXAGON_INSTR_SIZE) {
      DEBUG(dbgs() << "Alignment not a multiple of the instruction size:" <<
          Count % HEXAGON_INSTR_SIZE << "/" << HEXAGON_INSTR_SIZE << "\n");
      --Count;
      OW->write8(0);
    }

    while(Count) {
      Count -= HEXAGON_INSTR_SIZE;
      // Close the packet whenever a multiple of the maximum packet size remains
      uint32_t ParseBits = (Count % (HEXAGON_PACKET_SIZE * HEXAGON_INSTR_SIZE))?
                           ParseIn: ParseEnd;
      OW->write32(Nopcode | ParseBits);
    }
    return true;
  }
};
} // end anonymous namespace

namespace llvm {
MCAsmBackend *createHexagonAsmBackend(Target const &T,
                                      MCRegisterInfo const & /*MRI*/,
                                      const Triple &TT, StringRef CPU) {
  uint8_t OSABI = MCELFObjectTargetWriter::getOSABI(TT.getOS());
  return new HexagonAsmBackend(T, OSABI, CPU);
}
}
