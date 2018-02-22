//===-- RISCVMCCodeEmitter.cpp - Convert RISCV code to machine code -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the RISCVMCCodeEmitter class.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/RISCVBaseInfo.h"
#include "MCTargetDesc/RISCVFixupKinds.h"
#include "MCTargetDesc/RISCVMCExpr.h"
#include "MCTargetDesc/RISCVMCTargetDesc.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "mccodeemitter"

STATISTIC(MCNumEmitted, "Number of MC instructions emitted");
STATISTIC(MCNumFixups, "Number of MC fixups created");

namespace {
class RISCVMCCodeEmitter : public MCCodeEmitter {
  RISCVMCCodeEmitter(const RISCVMCCodeEmitter &) = delete;
  void operator=(const RISCVMCCodeEmitter &) = delete;
  MCContext &Ctx;
  MCInstrInfo const &MCII;

public:
  RISCVMCCodeEmitter(MCContext &ctx, MCInstrInfo const &MCII)
      : Ctx(ctx), MCII(MCII) {}

  ~RISCVMCCodeEmitter() override {}

  void encodeInstruction(const MCInst &MI, raw_ostream &OS,
                         SmallVectorImpl<MCFixup> &Fixups,
                         const MCSubtargetInfo &STI) const override;

  /// TableGen'erated function for getting the binary encoding for an
  /// instruction.
  uint64_t getBinaryCodeForInstr(const MCInst &MI,
                                 SmallVectorImpl<MCFixup> &Fixups,
                                 const MCSubtargetInfo &STI) const;

  /// Return binary encoding of operand. If the machine operand requires
  /// relocation, record the relocation and return zero.
  unsigned getMachineOpValue(const MCInst &MI, const MCOperand &MO,
                             SmallVectorImpl<MCFixup> &Fixups,
                             const MCSubtargetInfo &STI) const;

  unsigned getImmOpValueAsr1(const MCInst &MI, unsigned OpNo,
                             SmallVectorImpl<MCFixup> &Fixups,
                             const MCSubtargetInfo &STI) const;

  unsigned getImmOpValue(const MCInst &MI, unsigned OpNo,
                         SmallVectorImpl<MCFixup> &Fixups,
                         const MCSubtargetInfo &STI) const;
};
} // end anonymous namespace

MCCodeEmitter *llvm::createRISCVMCCodeEmitter(const MCInstrInfo &MCII,
                                              const MCRegisterInfo &MRI,
                                              MCContext &Ctx) {
  return new RISCVMCCodeEmitter(Ctx, MCII);
}

void RISCVMCCodeEmitter::encodeInstruction(const MCInst &MI, raw_ostream &OS,
                                           SmallVectorImpl<MCFixup> &Fixups,
                                           const MCSubtargetInfo &STI) const {
  const MCInstrDesc &Desc = MCII.get(MI.getOpcode());
  // Get byte count of instruction.
  unsigned Size = Desc.getSize();

  switch (Size) {
  default:
    llvm_unreachable("Unhandled encodeInstruction length!");
  case 2: {
    uint16_t Bits = getBinaryCodeForInstr(MI, Fixups, STI);
    support::endian::Writer<support::little>(OS).write<uint16_t>(Bits);
    break;
  }
  case 4: {
    uint32_t Bits = getBinaryCodeForInstr(MI, Fixups, STI);
    support::endian::Writer<support::little>(OS).write(Bits);
    break;
  }
  }

  ++MCNumEmitted; // Keep track of the # of mi's emitted.
}

unsigned
RISCVMCCodeEmitter::getMachineOpValue(const MCInst &MI, const MCOperand &MO,
                                      SmallVectorImpl<MCFixup> &Fixups,
                                      const MCSubtargetInfo &STI) const {

  if (MO.isReg())
    return Ctx.getRegisterInfo()->getEncodingValue(MO.getReg());

  if (MO.isImm())
    return static_cast<unsigned>(MO.getImm());

  llvm_unreachable("Unhandled expression!");
  return 0;
}

unsigned
RISCVMCCodeEmitter::getImmOpValueAsr1(const MCInst &MI, unsigned OpNo,
                                      SmallVectorImpl<MCFixup> &Fixups,
                                      const MCSubtargetInfo &STI) const {
  const MCOperand &MO = MI.getOperand(OpNo);

  if (MO.isImm()) {
    unsigned Res = MO.getImm();
    assert((Res & 1) == 0 && "LSB is non-zero");
    return Res >> 1;
  }

  return getImmOpValue(MI, OpNo, Fixups, STI);
}

unsigned RISCVMCCodeEmitter::getImmOpValue(const MCInst &MI, unsigned OpNo,
                                           SmallVectorImpl<MCFixup> &Fixups,
                                           const MCSubtargetInfo &STI) const {

  const MCOperand &MO = MI.getOperand(OpNo);

  MCInstrDesc const &Desc = MCII.get(MI.getOpcode());
  unsigned MIFrm = Desc.TSFlags & RISCVII::InstFormatMask;

  // If the destination is an immediate, there is nothing to do
  if (MO.isImm())
    return MO.getImm();

  assert(MO.isExpr() &&
         "getImmOpValue expects only expressions or immediates");
  const MCExpr *Expr = MO.getExpr();
  MCExpr::ExprKind Kind = Expr->getKind();
  RISCV::Fixups FixupKind = RISCV::fixup_riscv_invalid;
  if (Kind == MCExpr::Target) {
    const RISCVMCExpr *RVExpr = cast<RISCVMCExpr>(Expr);

    switch (RVExpr->getKind()) {
    case RISCVMCExpr::VK_RISCV_None:
    case RISCVMCExpr::VK_RISCV_Invalid:
      llvm_unreachable("Unhandled fixup kind!");
    case RISCVMCExpr::VK_RISCV_LO:
      if (MIFrm == RISCVII::InstFormatI)
        FixupKind = RISCV::fixup_riscv_lo12_i;
      else if (MIFrm == RISCVII::InstFormatS)
        FixupKind = RISCV::fixup_riscv_lo12_s;
      else
        llvm_unreachable("VK_RISCV_LO used with unexpected instruction format");
      break;
    case RISCVMCExpr::VK_RISCV_HI:
      FixupKind = RISCV::fixup_riscv_hi20;
      break;
    case RISCVMCExpr::VK_RISCV_PCREL_LO:
      if (MIFrm == RISCVII::InstFormatI)
        FixupKind = RISCV::fixup_riscv_pcrel_lo12_i;
      else if (MIFrm == RISCVII::InstFormatS)
        FixupKind = RISCV::fixup_riscv_pcrel_lo12_s;
      else
        llvm_unreachable(
            "VK_RISCV_PCREL_LO used with unexpected instruction format");
      break;
    case RISCVMCExpr::VK_RISCV_PCREL_HI:
      FixupKind = RISCV::fixup_riscv_pcrel_hi20;
      break;
    }
  } else if (Kind == MCExpr::SymbolRef &&
             cast<MCSymbolRefExpr>(Expr)->getKind() == MCSymbolRefExpr::VK_None) {
    if (Desc.getOpcode() == RISCV::JAL) {
      FixupKind = RISCV::fixup_riscv_jal;
    } else if (MIFrm == RISCVII::InstFormatB) {
      FixupKind = RISCV::fixup_riscv_branch;
    } else if (MIFrm == RISCVII::InstFormatCJ) {
      FixupKind = RISCV::fixup_riscv_rvc_jump;
    } else if (MIFrm == RISCVII::InstFormatCB) {
      FixupKind = RISCV::fixup_riscv_rvc_branch;
    }
  }

  assert(FixupKind != RISCV::fixup_riscv_invalid && "Unhandled expression!");

  Fixups.push_back(
      MCFixup::create(0, Expr, MCFixupKind(FixupKind), MI.getLoc()));
  ++MCNumFixups;

  return 0;
}

#include "RISCVGenMCCodeEmitter.inc"
