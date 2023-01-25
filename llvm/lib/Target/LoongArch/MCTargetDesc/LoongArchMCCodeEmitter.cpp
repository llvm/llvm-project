//=- LoongArchMCCodeEmitter.cpp - Convert LoongArch code to machine code --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the LoongArchMCCodeEmitter class.
//
//===----------------------------------------------------------------------===//

#include "LoongArchFixupKinds.h"
#include "MCTargetDesc/LoongArchBaseInfo.h"
#include "MCTargetDesc/LoongArchMCExpr.h"
#include "MCTargetDesc/LoongArchMCTargetDesc.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/EndianStream.h"

using namespace llvm;

#define DEBUG_TYPE "mccodeemitter"

namespace {
class LoongArchMCCodeEmitter : public MCCodeEmitter {
  LoongArchMCCodeEmitter(const LoongArchMCCodeEmitter &) = delete;
  void operator=(const LoongArchMCCodeEmitter &) = delete;
  MCContext &Ctx;
  MCInstrInfo const &MCII;

public:
  LoongArchMCCodeEmitter(MCContext &ctx, MCInstrInfo const &MCII)
      : Ctx(ctx), MCII(MCII) {}

  ~LoongArchMCCodeEmitter() override {}

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

  /// Return binary encoding of an immediate operand specified by OpNo.
  /// The value returned is the value of the immediate minus 1.
  /// Note that this function is dedicated to specific immediate types,
  /// e.g. uimm2_plus1.
  unsigned getImmOpValueSub1(const MCInst &MI, unsigned OpNo,
                             SmallVectorImpl<MCFixup> &Fixups,
                             const MCSubtargetInfo &STI) const;

  /// Return binary encoding of an immediate operand specified by OpNo.
  /// The value returned is the value of the immediate shifted right
  //  arithmetically by 2.
  /// Note that this function is dedicated to specific immediate types,
  /// e.g. simm14_lsl2, simm16_lsl2, simm21_lsl2 and simm26_lsl2.
  unsigned getImmOpValueAsr2(const MCInst &MI, unsigned OpNo,
                             SmallVectorImpl<MCFixup> &Fixups,
                             const MCSubtargetInfo &STI) const;

  unsigned getExprOpValue(const MCInst &MI, const MCOperand &MO,
                          SmallVectorImpl<MCFixup> &Fixups,
                          const MCSubtargetInfo &STI) const;
};
} // end namespace

unsigned
LoongArchMCCodeEmitter::getMachineOpValue(const MCInst &MI, const MCOperand &MO,
                                          SmallVectorImpl<MCFixup> &Fixups,
                                          const MCSubtargetInfo &STI) const {

  if (MO.isReg())
    return Ctx.getRegisterInfo()->getEncodingValue(MO.getReg());

  if (MO.isImm())
    return static_cast<unsigned>(MO.getImm());

  // MO must be an Expr.
  assert(MO.isExpr());
  return getExprOpValue(MI, MO, Fixups, STI);
}

unsigned
LoongArchMCCodeEmitter::getImmOpValueSub1(const MCInst &MI, unsigned OpNo,
                                          SmallVectorImpl<MCFixup> &Fixups,
                                          const MCSubtargetInfo &STI) const {
  return MI.getOperand(OpNo).getImm() - 1;
}

unsigned
LoongArchMCCodeEmitter::getImmOpValueAsr2(const MCInst &MI, unsigned OpNo,
                                          SmallVectorImpl<MCFixup> &Fixups,
                                          const MCSubtargetInfo &STI) const {
  const MCOperand &MO = MI.getOperand(OpNo);

  if (MO.isImm()) {
    unsigned Res = MI.getOperand(OpNo).getImm();
    assert((Res & 3) == 0 && "lowest 2 bits are non-zero");
    return Res >> 2;
  }

  return getExprOpValue(MI, MO, Fixups, STI);
}

unsigned
LoongArchMCCodeEmitter::getExprOpValue(const MCInst &MI, const MCOperand &MO,
                                       SmallVectorImpl<MCFixup> &Fixups,
                                       const MCSubtargetInfo &STI) const {
  assert(MO.isExpr() && "getExprOpValue expects only expressions");
  const MCExpr *Expr = MO.getExpr();
  MCExpr::ExprKind Kind = Expr->getKind();
  LoongArch::Fixups FixupKind = LoongArch::fixup_loongarch_invalid;
  if (Kind == MCExpr::Target) {
    const LoongArchMCExpr *LAExpr = cast<LoongArchMCExpr>(Expr);

    switch (LAExpr->getKind()) {
    case LoongArchMCExpr::VK_LoongArch_None:
    case LoongArchMCExpr::VK_LoongArch_Invalid:
      llvm_unreachable("Unhandled fixup kind!");
    case LoongArchMCExpr::VK_LoongArch_B16:
      FixupKind = LoongArch::fixup_loongarch_b16;
      break;
    case LoongArchMCExpr::VK_LoongArch_B21:
      FixupKind = LoongArch::fixup_loongarch_b21;
      break;
    case LoongArchMCExpr::VK_LoongArch_B26:
    case LoongArchMCExpr::VK_LoongArch_CALL:
    case LoongArchMCExpr::VK_LoongArch_CALL_PLT:
      FixupKind = LoongArch::fixup_loongarch_b26;
      break;
    case LoongArchMCExpr::VK_LoongArch_ABS_HI20:
      FixupKind = LoongArch::fixup_loongarch_abs_hi20;
      break;
    case LoongArchMCExpr::VK_LoongArch_ABS_LO12:
      FixupKind = LoongArch::fixup_loongarch_abs_lo12;
      break;
    case LoongArchMCExpr::VK_LoongArch_ABS64_LO20:
      FixupKind = LoongArch::fixup_loongarch_abs64_lo20;
      break;
    case LoongArchMCExpr::VK_LoongArch_ABS64_HI12:
      FixupKind = LoongArch::fixup_loongarch_abs64_hi12;
      break;
    case LoongArchMCExpr::VK_LoongArch_PCALA_HI20:
      FixupKind = LoongArch::fixup_loongarch_pcala_hi20;
      break;
    case LoongArchMCExpr::VK_LoongArch_PCALA_LO12:
      FixupKind = LoongArch::fixup_loongarch_pcala_lo12;
      break;
    case LoongArchMCExpr::VK_LoongArch_PCALA64_LO20:
      FixupKind = LoongArch::fixup_loongarch_pcala64_lo20;
      break;
    case LoongArchMCExpr::VK_LoongArch_PCALA64_HI12:
      FixupKind = LoongArch::fixup_loongarch_pcala64_hi12;
      break;
    case LoongArchMCExpr::VK_LoongArch_GOT_PC_HI20:
      FixupKind = LoongArch::fixup_loongarch_got_pc_hi20;
      break;
    case LoongArchMCExpr::VK_LoongArch_GOT_PC_LO12:
      FixupKind = LoongArch::fixup_loongarch_got_pc_lo12;
      break;
    case LoongArchMCExpr::VK_LoongArch_GOT64_PC_LO20:
      FixupKind = LoongArch::fixup_loongarch_got64_pc_lo20;
      break;
    case LoongArchMCExpr::VK_LoongArch_GOT64_PC_HI12:
      FixupKind = LoongArch::fixup_loongarch_got64_pc_hi12;
      break;
    case LoongArchMCExpr::VK_LoongArch_GOT_HI20:
      FixupKind = LoongArch::fixup_loongarch_got_hi20;
      break;
    case LoongArchMCExpr::VK_LoongArch_GOT_LO12:
      FixupKind = LoongArch::fixup_loongarch_got_lo12;
      break;
    case LoongArchMCExpr::VK_LoongArch_GOT64_LO20:
      FixupKind = LoongArch::fixup_loongarch_got64_lo20;
      break;
    case LoongArchMCExpr::VK_LoongArch_GOT64_HI12:
      FixupKind = LoongArch::fixup_loongarch_got64_hi12;
      break;
    case LoongArchMCExpr::VK_LoongArch_TLS_LE_HI20:
      FixupKind = LoongArch::fixup_loongarch_tls_le_hi20;
      break;
    case LoongArchMCExpr::VK_LoongArch_TLS_LE_LO12:
      FixupKind = LoongArch::fixup_loongarch_tls_le_lo12;
      break;
    case LoongArchMCExpr::VK_LoongArch_TLS_LE64_LO20:
      FixupKind = LoongArch::fixup_loongarch_tls_le64_lo20;
      break;
    case LoongArchMCExpr::VK_LoongArch_TLS_LE64_HI12:
      FixupKind = LoongArch::fixup_loongarch_tls_le64_hi12;
      break;
    case LoongArchMCExpr::VK_LoongArch_TLS_IE_PC_HI20:
      FixupKind = LoongArch::fixup_loongarch_tls_ie_pc_hi20;
      break;
    case LoongArchMCExpr::VK_LoongArch_TLS_IE_PC_LO12:
      FixupKind = LoongArch::fixup_loongarch_tls_ie_pc_lo12;
      break;
    case LoongArchMCExpr::VK_LoongArch_TLS_IE64_PC_LO20:
      FixupKind = LoongArch::fixup_loongarch_tls_ie64_pc_lo20;
      break;
    case LoongArchMCExpr::VK_LoongArch_TLS_IE64_PC_HI12:
      FixupKind = LoongArch::fixup_loongarch_tls_ie64_pc_hi12;
      break;
    case LoongArchMCExpr::VK_LoongArch_TLS_IE_HI20:
      FixupKind = LoongArch::fixup_loongarch_tls_ie_hi20;
      break;
    case LoongArchMCExpr::VK_LoongArch_TLS_IE_LO12:
      FixupKind = LoongArch::fixup_loongarch_tls_ie_lo12;
      break;
    case LoongArchMCExpr::VK_LoongArch_TLS_IE64_LO20:
      FixupKind = LoongArch::fixup_loongarch_tls_ie64_lo20;
      break;
    case LoongArchMCExpr::VK_LoongArch_TLS_IE64_HI12:
      FixupKind = LoongArch::fixup_loongarch_tls_ie64_hi12;
      break;
    case LoongArchMCExpr::VK_LoongArch_TLS_LD_PC_HI20:
      FixupKind = LoongArch::fixup_loongarch_tls_ld_pc_hi20;
      break;
    case LoongArchMCExpr::VK_LoongArch_TLS_LD_HI20:
      FixupKind = LoongArch::fixup_loongarch_tls_ld_hi20;
      break;
    case LoongArchMCExpr::VK_LoongArch_TLS_GD_PC_HI20:
      FixupKind = LoongArch::fixup_loongarch_tls_gd_pc_hi20;
      break;
    case LoongArchMCExpr::VK_LoongArch_TLS_GD_HI20:
      FixupKind = LoongArch::fixup_loongarch_tls_gd_hi20;
      break;
    }
  } else if (Kind == MCExpr::SymbolRef &&
             cast<MCSymbolRefExpr>(Expr)->getKind() ==
                 MCSymbolRefExpr::VK_None) {
    switch (MI.getOpcode()) {
    default:
      break;
    case LoongArch::BEQ:
    case LoongArch::BNE:
    case LoongArch::BLT:
    case LoongArch::BGE:
    case LoongArch::BLTU:
    case LoongArch::BGEU:
      FixupKind = LoongArch::fixup_loongarch_b16;
      break;
    case LoongArch::BEQZ:
    case LoongArch::BNEZ:
    case LoongArch::BCEQZ:
    case LoongArch::BCNEZ:
      FixupKind = LoongArch::fixup_loongarch_b21;
      break;
    case LoongArch::B:
      FixupKind = LoongArch::fixup_loongarch_b26;
      break;
    }
  }

  assert(FixupKind != LoongArch::fixup_loongarch_invalid &&
         "Unhandled expression!");

  Fixups.push_back(
      MCFixup::create(0, Expr, MCFixupKind(FixupKind), MI.getLoc()));
  return 0;
}

void LoongArchMCCodeEmitter::encodeInstruction(
    const MCInst &MI, raw_ostream &OS, SmallVectorImpl<MCFixup> &Fixups,
    const MCSubtargetInfo &STI) const {
  const MCInstrDesc &Desc = MCII.get(MI.getOpcode());
  // Get byte count of instruction.
  unsigned Size = Desc.getSize();

  switch (Size) {
  default:
    llvm_unreachable("Unhandled encodeInstruction length!");
  case 4: {
    uint32_t Bits = getBinaryCodeForInstr(MI, Fixups, STI);
    support::endian::write(OS, Bits, support::little);
    break;
  }
  }
}

MCCodeEmitter *llvm::createLoongArchMCCodeEmitter(const MCInstrInfo &MCII,
                                                  MCContext &Ctx) {
  return new LoongArchMCCodeEmitter(Ctx, MCII);
}

#include "LoongArchGenMCCodeEmitter.inc"
