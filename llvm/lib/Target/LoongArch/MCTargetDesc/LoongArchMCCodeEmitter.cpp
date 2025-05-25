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
#include "MCTargetDesc/LoongArchMCExpr.h"
#include "MCTargetDesc/LoongArchMCTargetDesc.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
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

  void encodeInstruction(const MCInst &MI, SmallVectorImpl<char> &CB,
                         SmallVectorImpl<MCFixup> &Fixups,
                         const MCSubtargetInfo &STI) const override;

  template <unsigned Opc>
  void expandToVectorLDI(const MCInst &MI, SmallVectorImpl<char> &CB,
                         SmallVectorImpl<MCFixup> &Fixups,
                         const MCSubtargetInfo &STI) const;

  void expandAddTPRel(const MCInst &MI, SmallVectorImpl<char> &CB,
                      SmallVectorImpl<MCFixup> &Fixups,
                      const MCSubtargetInfo &STI) const;

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
  //  arithmetically by N.
  /// Note that this function is dedicated to specific immediate types,
  /// e.g. simm14_lsl2, simm16_lsl2, simm21_lsl2 and simm26_lsl2.
  template <unsigned N>
  unsigned getImmOpValueAsr(const MCInst &MI, unsigned OpNo,
                            SmallVectorImpl<MCFixup> &Fixups,
                            const MCSubtargetInfo &STI) const {
    const MCOperand &MO = MI.getOperand(OpNo);
    if (MO.isImm()) {
      unsigned Res = MI.getOperand(OpNo).getImm();
      assert((Res & ((1U << N) - 1U)) == 0 && "lowest N bits are non-zero");
      return Res >> N;
    }
    return getExprOpValue(MI, MO, Fixups, STI);
  }

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
LoongArchMCCodeEmitter::getExprOpValue(const MCInst &MI, const MCOperand &MO,
                                       SmallVectorImpl<MCFixup> &Fixups,
                                       const MCSubtargetInfo &STI) const {
  assert(MO.isExpr() && "getExprOpValue expects only expressions");
  bool RelaxCandidate = false;
  bool EnableRelax = STI.hasFeature(LoongArch::FeatureRelax);
  const MCExpr *Expr = MO.getExpr();
  MCExpr::ExprKind Kind = Expr->getKind();
  unsigned FixupKind = LoongArch::fixup_loongarch_invalid;
  if (Kind == MCExpr::Target) {
    const LoongArchMCExpr *LAExpr = cast<LoongArchMCExpr>(Expr);
    FixupKind = LAExpr->getSpecifier();
    RelaxCandidate = LAExpr->getRelaxHint();
    switch (uint16_t(LAExpr->getSpecifier())) {
    case LoongArchMCExpr::VK_None:
      llvm_unreachable("Unhandled fixup kind!");
    case ELF::R_LARCH_TLS_LE_ADD_R:
      llvm_unreachable("ELF::R_LARCH_TLS_LE_ADD_R should not represent an "
                       "instruction operand");
    case ELF::R_LARCH_B16:
      FixupKind = LoongArch::fixup_loongarch_b16;
      break;
    case ELF::R_LARCH_B21:
      FixupKind = LoongArch::fixup_loongarch_b21;
      break;
    case ELF::R_LARCH_B26:
      FixupKind = LoongArch::fixup_loongarch_b26;
      break;
    case ELF::R_LARCH_ABS_HI20:
      FixupKind = LoongArch::fixup_loongarch_abs_hi20;
      break;
    case ELF::R_LARCH_ABS_LO12:
      FixupKind = LoongArch::fixup_loongarch_abs_lo12;
      break;
    case ELF::R_LARCH_ABS64_LO20:
      FixupKind = LoongArch::fixup_loongarch_abs64_lo20;
      break;
    case ELF::R_LARCH_ABS64_HI12:
      FixupKind = LoongArch::fixup_loongarch_abs64_hi12;
      break;
    case ELF::R_LARCH_CALL36:
    case ELF::R_LARCH_TLS_LE_HI20_R:
    case ELF::R_LARCH_TLS_LE_LO12_R:
      RelaxCandidate = true;
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
    case LoongArch::BL:
      FixupKind = LoongArch::fixup_loongarch_b26;
      break;
    }
  }

  assert(FixupKind != LoongArch::fixup_loongarch_invalid &&
         "Unhandled expression!");

  Fixups.push_back(
      MCFixup::create(0, Expr, MCFixupKind(FixupKind), MI.getLoc()));
  // If linker relaxation is enabled and supported by this relocation, set
  // a bit so that if fixup is unresolved, a R_LARCH_RELAX relocation will be
  // appended.
  if (EnableRelax && RelaxCandidate)
    Fixups.back().setLinkerRelaxable();

  return 0;
}

template <unsigned Opc>
void LoongArchMCCodeEmitter::expandToVectorLDI(
    const MCInst &MI, SmallVectorImpl<char> &CB,
    SmallVectorImpl<MCFixup> &Fixups, const MCSubtargetInfo &STI) const {
  int64_t Imm = MI.getOperand(1).getImm() & 0x3FF;
  switch (MI.getOpcode()) {
  case LoongArch::PseudoVREPLI_B:
  case LoongArch::PseudoXVREPLI_B:
    break;
  case LoongArch::PseudoVREPLI_H:
  case LoongArch::PseudoXVREPLI_H:
    Imm |= 0x400;
    break;
  case LoongArch::PseudoVREPLI_W:
  case LoongArch::PseudoXVREPLI_W:
    Imm |= 0x800;
    break;
  case LoongArch::PseudoVREPLI_D:
  case LoongArch::PseudoXVREPLI_D:
    Imm |= 0xC00;
    break;
  }
  MCInst TmpInst = MCInstBuilder(Opc).addOperand(MI.getOperand(0)).addImm(Imm);
  uint32_t Binary = getBinaryCodeForInstr(TmpInst, Fixups, STI);
  support::endian::write(CB, Binary, llvm::endianness::little);
}

void LoongArchMCCodeEmitter::expandAddTPRel(const MCInst &MI,
                                            SmallVectorImpl<char> &CB,
                                            SmallVectorImpl<MCFixup> &Fixups,
                                            const MCSubtargetInfo &STI) const {
  MCOperand Rd = MI.getOperand(0);
  MCOperand Rj = MI.getOperand(1);
  MCOperand Rk = MI.getOperand(2);
  MCOperand Symbol = MI.getOperand(3);
  assert(Symbol.isExpr() &&
         "Expected expression as third input to TP-relative add");

  const LoongArchMCExpr *Expr = dyn_cast<LoongArchMCExpr>(Symbol.getExpr());
  assert(Expr && Expr->getSpecifier() == ELF::R_LARCH_TLS_LE_ADD_R &&
         "Expected %le_add_r relocation on TP-relative symbol");

  // Emit the correct %le_add_r relocation for the symbol.
  Fixups.push_back(
      MCFixup::create(0, Expr, ELF::R_LARCH_TLS_LE_ADD_R, MI.getLoc()));
  if (STI.hasFeature(LoongArch::FeatureRelax))
    Fixups.back().setLinkerRelaxable();

  // Emit a normal ADD instruction with the given operands.
  unsigned ADD = MI.getOpcode() == LoongArch::PseudoAddTPRel_D
                     ? LoongArch::ADD_D
                     : LoongArch::ADD_W;
  MCInst TmpInst =
      MCInstBuilder(ADD).addOperand(Rd).addOperand(Rj).addOperand(Rk);
  uint32_t Binary = getBinaryCodeForInstr(TmpInst, Fixups, STI);
  support::endian::write(CB, Binary, llvm::endianness::little);
}

void LoongArchMCCodeEmitter::encodeInstruction(
    const MCInst &MI, SmallVectorImpl<char> &CB,
    SmallVectorImpl<MCFixup> &Fixups, const MCSubtargetInfo &STI) const {
  const MCInstrDesc &Desc = MCII.get(MI.getOpcode());
  // Get byte count of instruction.
  unsigned Size = Desc.getSize();

  switch (MI.getOpcode()) {
  default:
    break;
  case LoongArch::PseudoVREPLI_B:
  case LoongArch::PseudoVREPLI_H:
  case LoongArch::PseudoVREPLI_W:
  case LoongArch::PseudoVREPLI_D:
    return expandToVectorLDI<LoongArch::VLDI>(MI, CB, Fixups, STI);
  case LoongArch::PseudoXVREPLI_B:
  case LoongArch::PseudoXVREPLI_H:
  case LoongArch::PseudoXVREPLI_W:
  case LoongArch::PseudoXVREPLI_D:
    return expandToVectorLDI<LoongArch::XVLDI>(MI, CB, Fixups, STI);
  case LoongArch::PseudoAddTPRel_W:
  case LoongArch::PseudoAddTPRel_D:
    return expandAddTPRel(MI, CB, Fixups, STI);
  }

  switch (Size) {
  default:
    llvm_unreachable("Unhandled encodeInstruction length!");
  case 4: {
    uint32_t Bits = getBinaryCodeForInstr(MI, Fixups, STI);
    support::endian::write(CB, Bits, llvm::endianness::little);
    break;
  }
  }
}

MCCodeEmitter *llvm::createLoongArchMCCodeEmitter(const MCInstrInfo &MCII,
                                                  MCContext &Ctx) {
  return new LoongArchMCCodeEmitter(Ctx, MCII);
}

#include "LoongArchGenMCCodeEmitter.inc"
