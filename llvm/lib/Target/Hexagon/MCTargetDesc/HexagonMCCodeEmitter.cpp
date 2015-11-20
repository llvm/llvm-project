//===-- HexagonMCCodeEmitter.cpp - Hexagon Target Descriptions ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Hexagon.h"
#include "MCTargetDesc/HexagonBaseInfo.h"
#include "MCTargetDesc/HexagonFixupKinds.h"
#include "MCTargetDesc/HexagonMCCodeEmitter.h"
#include "MCTargetDesc/HexagonMCInstrInfo.h"
#include "MCTargetDesc/HexagonMCTargetDesc.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "mccodeemitter"

using namespace llvm;
using namespace Hexagon;

STATISTIC(MCNumEmitted, "Number of MC instructions emitted");

HexagonMCCodeEmitter::HexagonMCCodeEmitter(MCInstrInfo const &aMII,
                                           MCContext &aMCT)
    : MCT(aMCT), MCII(aMII), Addend(new unsigned(0)),
      Extended(new bool(false)), CurrentBundle(new MCInst const *) {}

uint32_t HexagonMCCodeEmitter::parseBits(size_t Instruction, size_t Last,
                                         MCInst const &MCB,
                                         MCInst const &MCI) const {
  bool Duplex = HexagonMCInstrInfo::isDuplex(MCII, MCI);
  if (Instruction == 0) {
    if (HexagonMCInstrInfo::isInnerLoop(MCB)) {
      assert(!Duplex);
      assert(Instruction != Last);
      return HexagonII::INST_PARSE_LOOP_END;
    }
  }
  if (Instruction == 1) {
    if (HexagonMCInstrInfo::isOuterLoop(MCB)) {
      assert(!Duplex);
      assert(Instruction != Last);
      return HexagonII::INST_PARSE_LOOP_END;
    }
  }
  if (Duplex) {
    assert(Instruction == Last);
    return HexagonII::INST_PARSE_DUPLEX;
  }
  if(Instruction == Last)
    return HexagonII::INST_PARSE_PACKET_END;
  return HexagonII::INST_PARSE_NOT_END;
}

void HexagonMCCodeEmitter::encodeInstruction(MCInst const &MI, raw_ostream &OS,
                                             SmallVectorImpl<MCFixup> &Fixups,
                                             MCSubtargetInfo const &STI) const {
  MCInst &HMB = const_cast<MCInst &>(MI);

  assert(HexagonMCInstrInfo::isBundle(HMB));
  DEBUG(dbgs() << "Encoding bundle\n";);
  *Addend = 0;
  *Extended = false;
  *CurrentBundle = &MI;
  size_t Instruction = 0;
  size_t Last = HexagonMCInstrInfo::bundleSize(HMB) - 1;
  for (auto &I : HexagonMCInstrInfo::bundleInstructions(HMB)) {
    MCInst &HMI = const_cast<MCInst &>(*I.getInst());
    EncodeSingleInstruction(HMI, OS, Fixups, STI,
                            parseBits(Instruction, Last, HMB, HMI),
                            Instruction);
    *Extended = HexagonMCInstrInfo::isImmext(HMI);
    *Addend += HEXAGON_INSTR_SIZE;
    ++Instruction;
  }
  return;
}

/// EncodeSingleInstruction - Emit a single
void HexagonMCCodeEmitter::EncodeSingleInstruction(
    const MCInst &MI, raw_ostream &OS, SmallVectorImpl<MCFixup> &Fixups,
    const MCSubtargetInfo &STI, uint32_t Parse, size_t Index) const {
  MCInst HMB = MI;
  assert(!HexagonMCInstrInfo::isBundle(HMB));
  uint64_t Binary;

  // Compound instructions are limited to using registers 0-7 and 16-23
  // and here we make a map 16-23 to 8-15 so they can be correctly encoded.
  static unsigned RegMap[8] = {Hexagon::R8,  Hexagon::R9,  Hexagon::R10,
                               Hexagon::R11, Hexagon::R12, Hexagon::R13,
                               Hexagon::R14, Hexagon::R15};

  // Pseudo instructions don't get encoded and shouldn't be here
  // in the first place!
  assert(!HexagonMCInstrInfo::getDesc(MCII, HMB).isPseudo() &&
         "pseudo-instruction found");
  DEBUG(dbgs() << "Encoding insn"
                  " `" << HexagonMCInstrInfo::getName(MCII, HMB) << "'"
                                                                    "\n");

  if (llvm::HexagonMCInstrInfo::getType(MCII, HMB) == HexagonII::TypeCOMPOUND) {
    for (unsigned i = 0; i < HMB.getNumOperands(); ++i)
      if (HMB.getOperand(i).isReg()) {
        unsigned Reg =
            MCT.getRegisterInfo()->getEncodingValue(HMB.getOperand(i).getReg());
        if ((Reg <= 23) && (Reg >= 16))
          HMB.getOperand(i).setReg(RegMap[Reg - 16]);
      }
  }

  if (HexagonMCInstrInfo::isNewValue(MCII, HMB)) {
    // Calculate the new value distance to the associated producer
    MCOperand &MCO =
        HMB.getOperand(HexagonMCInstrInfo::getNewValueOp(MCII, HMB));
    unsigned SOffset = 0;
    unsigned Register = MCO.getReg();
    unsigned Register1;
    auto Instructions = HexagonMCInstrInfo::bundleInstructions(**CurrentBundle);
    auto i = Instructions.begin() + Index - 1;
    for (;; --i) {
      assert(i != Instructions.begin() - 1 && "Couldn't find producer");
      MCInst const &Inst = *i->getInst();
      if (HexagonMCInstrInfo::isImmext(Inst))
        continue;
      ++SOffset;
      Register1 =
          HexagonMCInstrInfo::hasNewValue(MCII, Inst)
              ? HexagonMCInstrInfo::getNewValueOperand(MCII, Inst).getReg()
              : static_cast<unsigned>(Hexagon::NoRegister);
      if (Register != Register1)
        // This isn't the register we're looking for
        continue;
      if (!HexagonMCInstrInfo::isPredicated(MCII, Inst))
        // Producer is unpredicated
        break;
      assert(HexagonMCInstrInfo::isPredicated(MCII, HMB) &&
             "Unpredicated consumer depending on predicated producer");
      if (HexagonMCInstrInfo::isPredicatedTrue(MCII, Inst) ==
          HexagonMCInstrInfo::isPredicatedTrue(MCII, HMB))
        // Producer predicate sense matched ours
        break;
    }
    // Hexagon PRM 10.11 Construct Nt from distance
    unsigned Offset = SOffset;
    Offset <<= 1;
    MCO.setReg(Offset + Hexagon::R0);
  }

  Binary = getBinaryCodeForInstr(HMB, Fixups, STI);
  // Check for unimplemented instructions. Immediate extenders
  // are encoded as zero, so they need to be accounted for.
  if ((!Binary) &&
      ((HMB.getOpcode() != DuplexIClass0) && (HMB.getOpcode() != A4_ext) &&
       (HMB.getOpcode() != A4_ext_b) && (HMB.getOpcode() != A4_ext_c) &&
       (HMB.getOpcode() != A4_ext_g))) {
    // Use a A2_nop for unimplemented instructions.
    DEBUG(dbgs() << "Unimplemented inst: "
                    " `" << HexagonMCInstrInfo::getName(MCII, HMB) << "'"
                                                                      "\n");
    llvm_unreachable("Unimplemented Instruction");
  }
  Binary |= Parse;

  // if we need to emit a duplexed instruction
  if (HMB.getOpcode() >= Hexagon::DuplexIClass0 &&
      HMB.getOpcode() <= Hexagon::DuplexIClassF) {
    assert(Parse == HexagonII::INST_PARSE_DUPLEX &&
           "Emitting duplex without duplex parse bits");
    unsigned dupIClass;
    switch (HMB.getOpcode()) {
    case Hexagon::DuplexIClass0:
      dupIClass = 0;
      break;
    case Hexagon::DuplexIClass1:
      dupIClass = 1;
      break;
    case Hexagon::DuplexIClass2:
      dupIClass = 2;
      break;
    case Hexagon::DuplexIClass3:
      dupIClass = 3;
      break;
    case Hexagon::DuplexIClass4:
      dupIClass = 4;
      break;
    case Hexagon::DuplexIClass5:
      dupIClass = 5;
      break;
    case Hexagon::DuplexIClass6:
      dupIClass = 6;
      break;
    case Hexagon::DuplexIClass7:
      dupIClass = 7;
      break;
    case Hexagon::DuplexIClass8:
      dupIClass = 8;
      break;
    case Hexagon::DuplexIClass9:
      dupIClass = 9;
      break;
    case Hexagon::DuplexIClassA:
      dupIClass = 10;
      break;
    case Hexagon::DuplexIClassB:
      dupIClass = 11;
      break;
    case Hexagon::DuplexIClassC:
      dupIClass = 12;
      break;
    case Hexagon::DuplexIClassD:
      dupIClass = 13;
      break;
    case Hexagon::DuplexIClassE:
      dupIClass = 14;
      break;
    case Hexagon::DuplexIClassF:
      dupIClass = 15;
      break;
    default:
      llvm_unreachable("Unimplemented DuplexIClass");
      break;
    }
    // 29 is the bit position.
    // 0b1110 =0xE bits are masked off and down shifted by 1 bit.
    // Last bit is moved to bit position 13
    Binary = ((dupIClass & 0xE) << (29 - 1)) | ((dupIClass & 0x1) << 13);

    const MCInst *subInst0 = HMB.getOperand(0).getInst();
    const MCInst *subInst1 = HMB.getOperand(1).getInst();

    // get subinstruction slot 0
    unsigned subInstSlot0Bits = getBinaryCodeForInstr(*subInst0, Fixups, STI);
    // get subinstruction slot 1
    unsigned subInstSlot1Bits = getBinaryCodeForInstr(*subInst1, Fixups, STI);

    Binary |= subInstSlot0Bits | (subInstSlot1Bits << 16);
  }
  support::endian::Writer<support::little>(OS).write<uint32_t>(Binary);
  ++MCNumEmitted;
}

static Hexagon::Fixups getFixupNoBits(MCInstrInfo const &MCII, const MCInst &MI,
                                      const MCOperand &MO,
                                      const MCSymbolRefExpr::VariantKind kind) {
  const MCInstrDesc &MCID = HexagonMCInstrInfo::getDesc(MCII, MI);
  unsigned insnType = llvm::HexagonMCInstrInfo::getType(MCII, MI);

  if (insnType == HexagonII::TypePREFIX) {
    switch (kind) {
    case llvm::MCSymbolRefExpr::VK_GOTOFF:
      return Hexagon::fixup_Hexagon_GOTREL_32_6_X;
    case llvm::MCSymbolRefExpr::VK_GOT:
      return Hexagon::fixup_Hexagon_GOT_32_6_X;
    case llvm::MCSymbolRefExpr::VK_TPREL:
      return Hexagon::fixup_Hexagon_TPREL_32_6_X;
    case llvm::MCSymbolRefExpr::VK_DTPREL:
      return Hexagon::fixup_Hexagon_DTPREL_32_6_X;
    case llvm::MCSymbolRefExpr::VK_Hexagon_GD_GOT:
      return Hexagon::fixup_Hexagon_GD_GOT_32_6_X;
    case llvm::MCSymbolRefExpr::VK_Hexagon_LD_GOT:
      return Hexagon::fixup_Hexagon_LD_GOT_32_6_X;
    case llvm::MCSymbolRefExpr::VK_Hexagon_IE:
      return Hexagon::fixup_Hexagon_IE_32_6_X;
    case llvm::MCSymbolRefExpr::VK_Hexagon_IE_GOT:
      return Hexagon::fixup_Hexagon_IE_GOT_32_6_X;
    default:
      if (MCID.isBranch())
        return Hexagon::fixup_Hexagon_B32_PCREL_X;
      else
        return Hexagon::fixup_Hexagon_32_6_X;
    }
  } else if (MCID.isBranch())
    return (Hexagon::fixup_Hexagon_B13_PCREL);

  switch (MCID.getOpcode()) {
  case Hexagon::HI:
  case Hexagon::A2_tfrih:
    switch (kind) {
    case llvm::MCSymbolRefExpr::VK_GOT:
      return Hexagon::fixup_Hexagon_GOT_HI16;
    case llvm::MCSymbolRefExpr::VK_GOTOFF:
      return Hexagon::fixup_Hexagon_GOTREL_HI16;
    case llvm::MCSymbolRefExpr::VK_Hexagon_GD_GOT:
      return Hexagon::fixup_Hexagon_GD_GOT_HI16;
    case llvm::MCSymbolRefExpr::VK_Hexagon_LD_GOT:
      return Hexagon::fixup_Hexagon_LD_GOT_HI16;
    case llvm::MCSymbolRefExpr::VK_Hexagon_IE:
      return Hexagon::fixup_Hexagon_IE_HI16;
    case llvm::MCSymbolRefExpr::VK_Hexagon_IE_GOT:
      return Hexagon::fixup_Hexagon_IE_GOT_HI16;
    case llvm::MCSymbolRefExpr::VK_TPREL:
      return Hexagon::fixup_Hexagon_TPREL_HI16;
    case llvm::MCSymbolRefExpr::VK_DTPREL:
      return Hexagon::fixup_Hexagon_DTPREL_HI16;
    default:
      return Hexagon::fixup_Hexagon_HI16;
    }

  case Hexagon::LO:
  case Hexagon::A2_tfril:
    switch (kind) {
    case llvm::MCSymbolRefExpr::VK_GOT:
      return Hexagon::fixup_Hexagon_GOT_LO16;
    case llvm::MCSymbolRefExpr::VK_GOTOFF:
      return Hexagon::fixup_Hexagon_GOTREL_LO16;
    case llvm::MCSymbolRefExpr::VK_Hexagon_GD_GOT:
      return Hexagon::fixup_Hexagon_GD_GOT_LO16;
    case llvm::MCSymbolRefExpr::VK_Hexagon_LD_GOT:
      return Hexagon::fixup_Hexagon_LD_GOT_LO16;
    case llvm::MCSymbolRefExpr::VK_Hexagon_IE:
      return Hexagon::fixup_Hexagon_IE_LO16;
    case llvm::MCSymbolRefExpr::VK_Hexagon_IE_GOT:
      return Hexagon::fixup_Hexagon_IE_GOT_LO16;
    case llvm::MCSymbolRefExpr::VK_TPREL:
      return Hexagon::fixup_Hexagon_TPREL_LO16;
    case llvm::MCSymbolRefExpr::VK_DTPREL:
      return Hexagon::fixup_Hexagon_DTPREL_LO16;
    default:
      return Hexagon::fixup_Hexagon_LO16;
    }

  // The only relocs left should be GP relative:
  default:
    if (MCID.mayStore() || MCID.mayLoad()) {
      for (const uint16_t *ImpUses = MCID.getImplicitUses(); *ImpUses;
           ++ImpUses) {
        if (*ImpUses == Hexagon::GP) {
          switch (HexagonMCInstrInfo::getAccessSize(MCII, MI)) {
          case HexagonII::MemAccessSize::ByteAccess:
            return fixup_Hexagon_GPREL16_0;
          case HexagonII::MemAccessSize::HalfWordAccess:
            return fixup_Hexagon_GPREL16_1;
          case HexagonII::MemAccessSize::WordAccess:
            return fixup_Hexagon_GPREL16_2;
          case HexagonII::MemAccessSize::DoubleWordAccess:
            return fixup_Hexagon_GPREL16_3;
          default:
            llvm_unreachable("unhandled fixup");
          }
        }
      }
    } else
      llvm_unreachable("unhandled fixup");
  }

  return LastTargetFixupKind;
}

namespace llvm {
extern const MCInstrDesc HexagonInsts[];
}

namespace {
  bool isPCRel (unsigned Kind) {
    switch(Kind){
    case fixup_Hexagon_B22_PCREL:
    case fixup_Hexagon_B15_PCREL:
    case fixup_Hexagon_B7_PCREL:
    case fixup_Hexagon_B13_PCREL:
    case fixup_Hexagon_B9_PCREL:
    case fixup_Hexagon_B32_PCREL_X:
    case fixup_Hexagon_B22_PCREL_X:
    case fixup_Hexagon_B15_PCREL_X:
    case fixup_Hexagon_B13_PCREL_X:
    case fixup_Hexagon_B9_PCREL_X:
    case fixup_Hexagon_B7_PCREL_X:
    case fixup_Hexagon_32_PCREL:
    case fixup_Hexagon_PLT_B22_PCREL:
    case fixup_Hexagon_GD_PLT_B22_PCREL:
    case fixup_Hexagon_LD_PLT_B22_PCREL:
    case fixup_Hexagon_6_PCREL_X:
      return true;
    default:
      return false;
    }
  }
}

unsigned HexagonMCCodeEmitter::getExprOpValue(const MCInst &MI,
                                              const MCOperand &MO,
                                              const MCExpr *ME,
                                              SmallVectorImpl<MCFixup> &Fixups,
                                              const MCSubtargetInfo &STI) const

{
  int64_t Res;

  if (ME->evaluateAsAbsolute(Res))
    return Res;

  MCExpr::ExprKind MK = ME->getKind();
  if (MK == MCExpr::Constant) {
    return cast<MCConstantExpr>(ME)->getValue();
  }
  if (MK == MCExpr::Binary) {
    getExprOpValue(MI, MO, cast<MCBinaryExpr>(ME)->getLHS(), Fixups, STI);
    getExprOpValue(MI, MO, cast<MCBinaryExpr>(ME)->getRHS(), Fixups, STI);
    return 0;
  }

  assert(MK == MCExpr::SymbolRef);

  Hexagon::Fixups FixupKind =
      Hexagon::Fixups(Hexagon::fixup_Hexagon_TPREL_LO16);
  const MCSymbolRefExpr *MCSRE = static_cast<const MCSymbolRefExpr *>(ME);
  const MCInstrDesc &MCID = HexagonMCInstrInfo::getDesc(MCII, MI);
  unsigned bits = HexagonMCInstrInfo::getExtentBits(MCII, MI) -
                  HexagonMCInstrInfo::getExtentAlignment(MCII, MI);
  const MCSymbolRefExpr::VariantKind kind = MCSRE->getKind();

  DEBUG(dbgs() << "----------------------------------------\n");
  DEBUG(dbgs() << "Opcode Name: " << HexagonMCInstrInfo::getName(MCII, MI)
               << "\n");
  DEBUG(dbgs() << "Opcode: " << MCID.getOpcode() << "\n");
  DEBUG(dbgs() << "Relocation bits: " << bits << "\n");
  DEBUG(dbgs() << "Addend: " << *Addend << "\n");
  DEBUG(dbgs() << "----------------------------------------\n");

  switch (bits) {
  default:
    DEBUG(dbgs() << "unrecognized bit count of " << bits << '\n');
    break;

  case 32:
    switch (kind) {
    case llvm::MCSymbolRefExpr::VK_Hexagon_PCREL:
      FixupKind = Hexagon::fixup_Hexagon_32_PCREL;
      break;
    case llvm::MCSymbolRefExpr::VK_GOT:
      FixupKind = *Extended ? Hexagon::fixup_Hexagon_GOT_32_6_X
                            : Hexagon::fixup_Hexagon_GOT_32;
      break;
    case llvm::MCSymbolRefExpr::VK_GOTOFF:
      FixupKind = *Extended ? Hexagon::fixup_Hexagon_GOTREL_32_6_X
                            : Hexagon::fixup_Hexagon_GOTREL_32;
      break;
    case llvm::MCSymbolRefExpr::VK_Hexagon_GD_GOT:
      FixupKind = *Extended ? Hexagon::fixup_Hexagon_GD_GOT_32_6_X
                            : Hexagon::fixup_Hexagon_GD_GOT_32;
      break;
    case llvm::MCSymbolRefExpr::VK_Hexagon_LD_GOT:
      FixupKind = *Extended ? Hexagon::fixup_Hexagon_LD_GOT_32_6_X
                            : Hexagon::fixup_Hexagon_LD_GOT_32;
      break;
    case llvm::MCSymbolRefExpr::VK_Hexagon_IE:
      FixupKind = *Extended ? Hexagon::fixup_Hexagon_IE_32_6_X
                            : Hexagon::fixup_Hexagon_IE_32;
      break;
    case llvm::MCSymbolRefExpr::VK_Hexagon_IE_GOT:
      FixupKind = *Extended ? Hexagon::fixup_Hexagon_IE_GOT_32_6_X
                            : Hexagon::fixup_Hexagon_IE_GOT_32;
      break;
    case llvm::MCSymbolRefExpr::VK_TPREL:
      FixupKind = *Extended ? Hexagon::fixup_Hexagon_TPREL_32_6_X
                            : Hexagon::fixup_Hexagon_TPREL_32;
      break;
    case llvm::MCSymbolRefExpr::VK_DTPREL:
      FixupKind = *Extended ? Hexagon::fixup_Hexagon_DTPREL_32_6_X
                            : Hexagon::fixup_Hexagon_DTPREL_32;
      break;
    default:
      FixupKind =
          *Extended ? Hexagon::fixup_Hexagon_32_6_X : Hexagon::fixup_Hexagon_32;
      break;
    }
    break;

  case 22:
    switch (kind) {
    case llvm::MCSymbolRefExpr::VK_Hexagon_GD_PLT:
      FixupKind = Hexagon::fixup_Hexagon_GD_PLT_B22_PCREL;
      break;
    case llvm::MCSymbolRefExpr::VK_Hexagon_LD_PLT:
      FixupKind = Hexagon::fixup_Hexagon_LD_PLT_B22_PCREL;
      break;
    default:
      if (MCID.isBranch() || MCID.isCall()) {
        FixupKind = *Extended ? Hexagon::fixup_Hexagon_B22_PCREL_X
                              : Hexagon::fixup_Hexagon_B22_PCREL;
      } else {
        errs() << "unrecognized relocation, bits: " << bits << "\n";
        errs() << "name = " << HexagonMCInstrInfo::getName(MCII, MI) << "\n";
      }
      break;
    }
    break;

  case 16:
    if (*Extended) {
      switch (kind) {
      default:
        FixupKind = Hexagon::fixup_Hexagon_16_X;
        break;
      case llvm::MCSymbolRefExpr::VK_GOT:
        FixupKind = Hexagon::fixup_Hexagon_GOT_16_X;
        break;
      case llvm::MCSymbolRefExpr::VK_GOTOFF:
        FixupKind = Hexagon::fixup_Hexagon_GOTREL_16_X;
        break;
      case llvm::MCSymbolRefExpr::VK_Hexagon_GD_GOT:
        FixupKind = Hexagon::fixup_Hexagon_GD_GOT_16_X;
        break;
      case llvm::MCSymbolRefExpr::VK_Hexagon_LD_GOT:
        FixupKind = Hexagon::fixup_Hexagon_LD_GOT_16_X;
        break;
      case llvm::MCSymbolRefExpr::VK_Hexagon_IE:
        FixupKind = Hexagon::fixup_Hexagon_IE_16_X;
        break;
      case llvm::MCSymbolRefExpr::VK_Hexagon_IE_GOT:
        FixupKind = Hexagon::fixup_Hexagon_IE_GOT_16_X;
        break;
      case llvm::MCSymbolRefExpr::VK_TPREL:
        FixupKind = Hexagon::fixup_Hexagon_TPREL_16_X;
        break;
      case llvm::MCSymbolRefExpr::VK_DTPREL:
        FixupKind = Hexagon::fixup_Hexagon_DTPREL_16_X;
        break;
      }
    } else
      switch (kind) {
      default:
        errs() << "unrecognized relocation, bits " << bits << "\n";
        errs() << "name = " << HexagonMCInstrInfo::getName(MCII, MI) << "\n";
        break;
      case llvm::MCSymbolRefExpr::VK_GOTOFF:
        if ((MCID.getOpcode() == Hexagon::HI) ||
            (MCID.getOpcode() == Hexagon::LO_H))
          FixupKind = Hexagon::fixup_Hexagon_GOTREL_HI16;
        else
          FixupKind = Hexagon::fixup_Hexagon_GOTREL_LO16;
        break;
      case llvm::MCSymbolRefExpr::VK_Hexagon_GPREL:
        FixupKind = Hexagon::fixup_Hexagon_GPREL16_0;
        break;
      case llvm::MCSymbolRefExpr::VK_Hexagon_LO16:
        FixupKind = Hexagon::fixup_Hexagon_LO16;
        break;
      case llvm::MCSymbolRefExpr::VK_Hexagon_HI16:
        FixupKind = Hexagon::fixup_Hexagon_HI16;
        break;
      case llvm::MCSymbolRefExpr::VK_Hexagon_GD_GOT:
        FixupKind = Hexagon::fixup_Hexagon_GD_GOT_16;
        break;
      case llvm::MCSymbolRefExpr::VK_Hexagon_LD_GOT:
        FixupKind = Hexagon::fixup_Hexagon_LD_GOT_16;
        break;
      case llvm::MCSymbolRefExpr::VK_Hexagon_IE_GOT:
        FixupKind = Hexagon::fixup_Hexagon_IE_GOT_16;
        break;
      case llvm::MCSymbolRefExpr::VK_TPREL:
        FixupKind = Hexagon::fixup_Hexagon_TPREL_16;
        break;
      case llvm::MCSymbolRefExpr::VK_DTPREL:
        FixupKind = Hexagon::fixup_Hexagon_DTPREL_16;
        break;
      }
    break;

  case 15:
    if (MCID.isBranch() || MCID.isCall())
      FixupKind = *Extended ? Hexagon::fixup_Hexagon_B15_PCREL_X
                            : Hexagon::fixup_Hexagon_B15_PCREL;
    break;

  case 13:
    if (MCID.isBranch())
      FixupKind = Hexagon::fixup_Hexagon_B13_PCREL;
    else {
      errs() << "unrecognized relocation, bits " << bits << "\n";
      errs() << "name = " << HexagonMCInstrInfo::getName(MCII, MI) << "\n";
    }
    break;

  case 12:
    if (*Extended)
      switch (kind) {
      default:
        FixupKind = Hexagon::fixup_Hexagon_12_X;
        break;
      // There isn't a GOT_12_X, both 11_X and 16_X resolve to 6/26
      case llvm::MCSymbolRefExpr::VK_GOT:
        FixupKind = Hexagon::fixup_Hexagon_GOT_16_X;
        break;
      case llvm::MCSymbolRefExpr::VK_GOTOFF:
        FixupKind = Hexagon::fixup_Hexagon_GOTREL_16_X;
        break;
      }
    else {
      errs() << "unrecognized relocation, bits " << bits << "\n";
      errs() << "name = " << HexagonMCInstrInfo::getName(MCII, MI) << "\n";
    }
    break;

  case 11:
    if (*Extended)
      switch (kind) {
      default:
        FixupKind = Hexagon::fixup_Hexagon_11_X;
        break;
      case llvm::MCSymbolRefExpr::VK_GOT:
        FixupKind = Hexagon::fixup_Hexagon_GOT_11_X;
        break;
      case llvm::MCSymbolRefExpr::VK_GOTOFF:
        FixupKind = Hexagon::fixup_Hexagon_GOTREL_11_X;
        break;
      case llvm::MCSymbolRefExpr::VK_Hexagon_GD_GOT:
        FixupKind = Hexagon::fixup_Hexagon_GD_GOT_11_X;
        break;
      case llvm::MCSymbolRefExpr::VK_Hexagon_LD_GOT:
        FixupKind = Hexagon::fixup_Hexagon_LD_GOT_11_X;
        break;
      case llvm::MCSymbolRefExpr::VK_Hexagon_IE_GOT:
        FixupKind = Hexagon::fixup_Hexagon_IE_GOT_11_X;
        break;
      case llvm::MCSymbolRefExpr::VK_TPREL:
        FixupKind = Hexagon::fixup_Hexagon_TPREL_11_X;
        break;
      case llvm::MCSymbolRefExpr::VK_DTPREL:
        FixupKind = Hexagon::fixup_Hexagon_DTPREL_11_X;
        break;
      }
    else {
      errs() << "unrecognized relocation, bits " << bits << "\n";
      errs() << "name = " << HexagonMCInstrInfo::getName(MCII, MI) << "\n";
    }
    break;

  case 10:
    if (*Extended)
      FixupKind = Hexagon::fixup_Hexagon_10_X;
    break;

  case 9:
    if (MCID.isBranch() ||
        (llvm::HexagonMCInstrInfo::getType(MCII, MI) == HexagonII::TypeCR))
      FixupKind = *Extended ? Hexagon::fixup_Hexagon_B9_PCREL_X
                            : Hexagon::fixup_Hexagon_B9_PCREL;
    else if (*Extended)
      FixupKind = Hexagon::fixup_Hexagon_9_X;
    else {
      errs() << "unrecognized relocation, bits " << bits << "\n";
      errs() << "name = " << HexagonMCInstrInfo::getName(MCII, MI) << "\n";
    }
    break;

  case 8:
    if (*Extended)
      FixupKind = Hexagon::fixup_Hexagon_8_X;
    else {
      errs() << "unrecognized relocation, bits " << bits << "\n";
      errs() << "name = " << HexagonMCInstrInfo::getName(MCII, MI) << "\n";
    }
    break;

  case 7:
    if (MCID.isBranch() ||
        (llvm::HexagonMCInstrInfo::getType(MCII, MI) == HexagonII::TypeCR))
      FixupKind = *Extended ? Hexagon::fixup_Hexagon_B7_PCREL_X
                            : Hexagon::fixup_Hexagon_B7_PCREL;
    else if (*Extended)
      FixupKind = Hexagon::fixup_Hexagon_7_X;
    else {
      errs() << "unrecognized relocation, bits " << bits << "\n";
      errs() << "name = " << HexagonMCInstrInfo::getName(MCII, MI) << "\n";
    }
    break;

  case 6:
    if (*Extended) {
      switch (kind) {
      default:
        FixupKind = Hexagon::fixup_Hexagon_6_X;
        break;
      case llvm::MCSymbolRefExpr::VK_Hexagon_PCREL:
        FixupKind = Hexagon::fixup_Hexagon_6_PCREL_X;
        break;
      // This is part of an extender, GOT_11 is a
      // Word32_U6 unsigned/truncated reloc.
      case llvm::MCSymbolRefExpr::VK_GOT:
        FixupKind = Hexagon::fixup_Hexagon_GOT_11_X;
        break;
      case llvm::MCSymbolRefExpr::VK_GOTOFF:
        FixupKind = Hexagon::fixup_Hexagon_GOTREL_11_X;
        break;
      }
    } else {
      errs() << "unrecognized relocation, bits " << bits << "\n";
      errs() << "name = " << HexagonMCInstrInfo::getName(MCII, MI) << "\n";
    }
    break;

  case 0:
    FixupKind = getFixupNoBits(MCII, MI, MO, kind);
    break;
  }

  MCExpr const *FixupExpression = (*Addend > 0 && isPCRel(FixupKind)) ?
    MCBinaryExpr::createAdd(MO.getExpr(),
                            MCConstantExpr::create(*Addend, MCT), MCT) :
    MO.getExpr();

  MCFixup fixup = MCFixup::create(*Addend, FixupExpression, 
                                  MCFixupKind(FixupKind), MI.getLoc());
  Fixups.push_back(fixup);
  // All of the information is in the fixup.
  return (0);
}

unsigned
HexagonMCCodeEmitter::getMachineOpValue(MCInst const &MI, MCOperand const &MO,
                                        SmallVectorImpl<MCFixup> &Fixups,
                                        MCSubtargetInfo const &STI) const {
  if (MO.isReg())
    return MCT.getRegisterInfo()->getEncodingValue(MO.getReg());
  if (MO.isImm())
    return static_cast<unsigned>(MO.getImm());

  // MO must be an ME.
  assert(MO.isExpr());
  return getExprOpValue(MI, MO, MO.getExpr(), Fixups, STI);
}

MCCodeEmitter *llvm::createHexagonMCCodeEmitter(MCInstrInfo const &MII,
                                                MCRegisterInfo const &MRI,
                                                MCContext &MCT) {
  return new HexagonMCCodeEmitter(MII, MCT);
}

#include "HexagonGenMCCodeEmitter.inc"
