//===-- RISCVMCCodeEmitter.cpp - Convert RISC-V code to machine code ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the RISCVMCCodeEmitter class.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/RISCVBaseInfo.h"
#include "MCTargetDesc/RISCVFixupKinds.h"
#include "MCTargetDesc/RISCVMCAsmInfo.h"
#include "MCTargetDesc/RISCVMCTargetDesc.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/EndianStream.h"

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

  ~RISCVMCCodeEmitter() override = default;

  void encodeInstruction(const MCInst &MI, SmallVectorImpl<char> &CB,
                         SmallVectorImpl<MCFixup> &Fixups,
                         const MCSubtargetInfo &STI) const override;

  void expandFunctionCall(const MCInst &MI, SmallVectorImpl<char> &CB,
                          SmallVectorImpl<MCFixup> &Fixups,
                          const MCSubtargetInfo &STI) const;

  void expandTLSDESCCall(const MCInst &MI, SmallVectorImpl<char> &CB,
                         SmallVectorImpl<MCFixup> &Fixups,
                         const MCSubtargetInfo &STI) const;

  void expandAddTPRel(const MCInst &MI, SmallVectorImpl<char> &CB,
                      SmallVectorImpl<MCFixup> &Fixups,
                      const MCSubtargetInfo &STI) const;

  void expandLongCondBr(const MCInst &MI, SmallVectorImpl<char> &CB,
                        SmallVectorImpl<MCFixup> &Fixups,
                        const MCSubtargetInfo &STI) const;

  void expandQCLongCondBrImm(const MCInst &MI, SmallVectorImpl<char> &CB,
                             SmallVectorImpl<MCFixup> &Fixups,
                             const MCSubtargetInfo &STI, unsigned Size) const;

  /// TableGen'erated function for getting the binary encoding for an
  /// instruction.
  uint64_t getBinaryCodeForInstr(const MCInst &MI,
                                 SmallVectorImpl<MCFixup> &Fixups,
                                 const MCSubtargetInfo &STI) const;

  /// Return binary encoding of operand. If the machine operand requires
  /// relocation, record the relocation and return zero.
  uint64_t getMachineOpValue(const MCInst &MI, const MCOperand &MO,
                             SmallVectorImpl<MCFixup> &Fixups,
                             const MCSubtargetInfo &STI) const;

  uint64_t getImmOpValueMinus1(const MCInst &MI, unsigned OpNo,
                               SmallVectorImpl<MCFixup> &Fixups,
                               const MCSubtargetInfo &STI) const;

  uint64_t getImmOpValueSlist(const MCInst &MI, unsigned OpNo,
                              SmallVectorImpl<MCFixup> &Fixups,
                              const MCSubtargetInfo &STI) const;

  template <unsigned N>
  unsigned getImmOpValueAsrN(const MCInst &MI, unsigned OpNo,
                             SmallVectorImpl<MCFixup> &Fixups,
                             const MCSubtargetInfo &STI) const;

  uint64_t getImmOpValueZibi(const MCInst &MI, unsigned OpNo,
                             SmallVectorImpl<MCFixup> &Fixups,
                             const MCSubtargetInfo &STI) const;

  uint64_t getImmOpValue(const MCInst &MI, unsigned OpNo,
                         SmallVectorImpl<MCFixup> &Fixups,
                         const MCSubtargetInfo &STI) const;

  unsigned getVMaskReg(const MCInst &MI, unsigned OpNo,
                       SmallVectorImpl<MCFixup> &Fixups,
                       const MCSubtargetInfo &STI) const;

  unsigned getRlistOpValue(const MCInst &MI, unsigned OpNo,
                           SmallVectorImpl<MCFixup> &Fixups,
                           const MCSubtargetInfo &STI) const;

  unsigned getRlistS0OpValue(const MCInst &MI, unsigned OpNo,
                             SmallVectorImpl<MCFixup> &Fixups,
                             const MCSubtargetInfo &STI) const;
};
} // end anonymous namespace

MCCodeEmitter *llvm::createRISCVMCCodeEmitter(const MCInstrInfo &MCII,
                                              MCContext &Ctx) {
  return new RISCVMCCodeEmitter(Ctx, MCII);
}

static void addFixup(SmallVectorImpl<MCFixup> &Fixups, uint32_t Offset,
                     const MCExpr *Value, uint16_t Kind) {
  bool PCRel = false;
  switch (Kind) {
  case ELF::R_RISCV_CALL_PLT:
  case RISCV::fixup_riscv_pcrel_hi20:
  case RISCV::fixup_riscv_pcrel_lo12_i:
  case RISCV::fixup_riscv_pcrel_lo12_s:
  case RISCV::fixup_riscv_jal:
  case RISCV::fixup_riscv_branch:
  case RISCV::fixup_riscv_rvc_jump:
  case RISCV::fixup_riscv_rvc_branch:
  case RISCV::fixup_riscv_call:
  case RISCV::fixup_riscv_call_plt:
  case RISCV::fixup_riscv_qc_e_branch:
  case RISCV::fixup_riscv_qc_e_call_plt:
  case RISCV::fixup_riscv_nds_branch_10:
    PCRel = true;
  }
  Fixups.push_back(MCFixup::create(Offset, Value, Kind, PCRel));
}

// Expand PseudoCALL(Reg), PseudoTAIL and PseudoJump to AUIPC and JALR with
// relocation types. We expand those pseudo-instructions while encoding them,
// meaning AUIPC and JALR won't go through RISC-V MC to MC compressed
// instruction transformation. This is acceptable because AUIPC has no 16-bit
// form and C_JALR has no immediate operand field.  We let linker relaxation
// deal with it. When linker relaxation is enabled, AUIPC and JALR have a
// chance to relax to JAL.
// If the C extension is enabled, JAL has a chance relax to C_JAL.
void RISCVMCCodeEmitter::expandFunctionCall(const MCInst &MI,
                                            SmallVectorImpl<char> &CB,
                                            SmallVectorImpl<MCFixup> &Fixups,
                                            const MCSubtargetInfo &STI) const {
  MCInst TmpInst;
  MCOperand Func;
  MCRegister Ra;
  if (MI.getOpcode() == RISCV::PseudoTAIL) {
    Func = MI.getOperand(0);
    Ra = RISCVII::getTailExpandUseRegNo(STI.getFeatureBits());
  } else if (MI.getOpcode() == RISCV::PseudoCALLReg) {
    Func = MI.getOperand(1);
    Ra = MI.getOperand(0).getReg();
  } else if (MI.getOpcode() == RISCV::PseudoCALL) {
    Func = MI.getOperand(0);
    Ra = RISCV::X1;
  } else if (MI.getOpcode() == RISCV::PseudoJump) {
    Func = MI.getOperand(1);
    Ra = MI.getOperand(0).getReg();
  }
  uint32_t Binary;

  assert(Func.isExpr() && "Expected expression");

  const MCExpr *CallExpr = Func.getExpr();

  // Emit AUIPC Ra, Func with R_RISCV_CALL relocation type.
  TmpInst = MCInstBuilder(RISCV::AUIPC).addReg(Ra).addExpr(CallExpr);
  Binary = getBinaryCodeForInstr(TmpInst, Fixups, STI);
  support::endian::write(CB, Binary, llvm::endianness::little);

  if (MI.getOpcode() == RISCV::PseudoTAIL ||
      MI.getOpcode() == RISCV::PseudoJump)
    // Emit JALR X0, Ra, 0
    TmpInst = MCInstBuilder(RISCV::JALR).addReg(RISCV::X0).addReg(Ra).addImm(0);
  else
    // Emit JALR Ra, Ra, 0
    TmpInst = MCInstBuilder(RISCV::JALR).addReg(Ra).addReg(Ra).addImm(0);
  Binary = getBinaryCodeForInstr(TmpInst, Fixups, STI);
  support::endian::write(CB, Binary, llvm::endianness::little);
}

void RISCVMCCodeEmitter::expandTLSDESCCall(const MCInst &MI,
                                           SmallVectorImpl<char> &CB,
                                           SmallVectorImpl<MCFixup> &Fixups,
                                           const MCSubtargetInfo &STI) const {
  MCOperand SrcSymbol = MI.getOperand(3);
  assert(SrcSymbol.isExpr() &&
         "Expected expression as first input to TLSDESCCALL");
  const auto *Expr = dyn_cast<MCSpecifierExpr>(SrcSymbol.getExpr());
  MCRegister Link = MI.getOperand(0).getReg();
  MCRegister Dest = MI.getOperand(1).getReg();
  int64_t Imm = MI.getOperand(2).getImm();
  addFixup(Fixups, 0, Expr, ELF::R_RISCV_TLSDESC_CALL);
  MCInst Call =
      MCInstBuilder(RISCV::JALR).addReg(Link).addReg(Dest).addImm(Imm);

  uint32_t Binary = getBinaryCodeForInstr(Call, Fixups, STI);
  support::endian::write(CB, Binary, llvm::endianness::little);
}

// Expand PseudoAddTPRel to a simple ADD with the correct relocation.
void RISCVMCCodeEmitter::expandAddTPRel(const MCInst &MI,
                                        SmallVectorImpl<char> &CB,
                                        SmallVectorImpl<MCFixup> &Fixups,
                                        const MCSubtargetInfo &STI) const {
  MCOperand DestReg = MI.getOperand(0);
  MCOperand SrcReg = MI.getOperand(1);
  MCOperand TPReg = MI.getOperand(2);
  assert(TPReg.isReg() && TPReg.getReg() == RISCV::X4 &&
         "Expected thread pointer as second input to TP-relative add");

  MCOperand SrcSymbol = MI.getOperand(3);
  assert(SrcSymbol.isExpr() &&
         "Expected expression as third input to TP-relative add");

  const auto *Expr = dyn_cast<MCSpecifierExpr>(SrcSymbol.getExpr());
  assert(Expr && Expr->getSpecifier() == ELF::R_RISCV_TPREL_ADD &&
         "Expected tprel_add relocation on TP-relative symbol");

  addFixup(Fixups, 0, Expr, ELF::R_RISCV_TPREL_ADD);
  if (STI.hasFeature(RISCV::FeatureRelax))
    Fixups.back().setLinkerRelaxable();

  // Emit a normal ADD instruction with the given operands.
  MCInst TmpInst = MCInstBuilder(RISCV::ADD)
                       .addOperand(DestReg)
                       .addOperand(SrcReg)
                       .addOperand(TPReg);
  uint32_t Binary = getBinaryCodeForInstr(TmpInst, Fixups, STI);
  support::endian::write(CB, Binary, llvm::endianness::little);
}

static unsigned getInvertedBranchOp(unsigned BrOp) {
  switch (BrOp) {
  default:
    llvm_unreachable("Unexpected branch opcode!");
  case RISCV::PseudoLongBEQ:
    return RISCV::BNE;
  case RISCV::PseudoLongBNE:
    return RISCV::BEQ;
  case RISCV::PseudoLongBLT:
    return RISCV::BGE;
  case RISCV::PseudoLongBGE:
    return RISCV::BLT;
  case RISCV::PseudoLongBLTU:
    return RISCV::BGEU;
  case RISCV::PseudoLongBGEU:
    return RISCV::BLTU;
  case RISCV::PseudoLongQC_BEQI:
    return RISCV::QC_BNEI;
  case RISCV::PseudoLongQC_BNEI:
    return RISCV::QC_BEQI;
  case RISCV::PseudoLongQC_BLTI:
    return RISCV::QC_BGEI;
  case RISCV::PseudoLongQC_BGEI:
    return RISCV::QC_BLTI;
  case RISCV::PseudoLongQC_BLTUI:
    return RISCV::QC_BGEUI;
  case RISCV::PseudoLongQC_BGEUI:
    return RISCV::QC_BLTUI;
  case RISCV::PseudoLongQC_E_BEQI:
    return RISCV::QC_E_BNEI;
  case RISCV::PseudoLongQC_E_BNEI:
    return RISCV::QC_E_BEQI;
  case RISCV::PseudoLongQC_E_BLTI:
    return RISCV::QC_E_BGEI;
  case RISCV::PseudoLongQC_E_BGEI:
    return RISCV::QC_E_BLTI;
  case RISCV::PseudoLongQC_E_BLTUI:
    return RISCV::QC_E_BGEUI;
  case RISCV::PseudoLongQC_E_BGEUI:
    return RISCV::QC_E_BLTUI;
  }
}

// Expand PseudoLongBxx to an inverted conditional branch and an unconditional
// jump.
void RISCVMCCodeEmitter::expandLongCondBr(const MCInst &MI,
                                          SmallVectorImpl<char> &CB,
                                          SmallVectorImpl<MCFixup> &Fixups,
                                          const MCSubtargetInfo &STI) const {
  MCRegister SrcReg1 = MI.getOperand(0).getReg();
  MCRegister SrcReg2 = MI.getOperand(1).getReg();
  MCOperand SrcSymbol = MI.getOperand(2);
  unsigned Opcode = MI.getOpcode();
  bool IsEqTest =
      Opcode == RISCV::PseudoLongBNE || Opcode == RISCV::PseudoLongBEQ;

  bool UseCompressedBr = false;
  if (IsEqTest && STI.hasFeature(RISCV::FeatureStdExtZca)) {
    if (RISCV::X8 <= SrcReg1.id() && SrcReg1.id() <= RISCV::X15 &&
        SrcReg2.id() == RISCV::X0) {
      UseCompressedBr = true;
    } else if (RISCV::X8 <= SrcReg2.id() && SrcReg2.id() <= RISCV::X15 &&
               SrcReg1.id() == RISCV::X0) {
      std::swap(SrcReg1, SrcReg2);
      UseCompressedBr = true;
    }
  }

  uint32_t Offset;
  if (UseCompressedBr) {
    unsigned InvOpc =
        Opcode == RISCV::PseudoLongBNE ? RISCV::C_BEQZ : RISCV::C_BNEZ;
    MCInst TmpInst = MCInstBuilder(InvOpc).addReg(SrcReg1).addImm(6);
    uint16_t Binary = getBinaryCodeForInstr(TmpInst, Fixups, STI);
    support::endian::write<uint16_t>(CB, Binary, llvm::endianness::little);
    Offset = 2;
  } else {
    unsigned InvOpc = getInvertedBranchOp(Opcode);
    MCInst TmpInst =
        MCInstBuilder(InvOpc).addReg(SrcReg1).addReg(SrcReg2).addImm(8);
    uint32_t Binary = getBinaryCodeForInstr(TmpInst, Fixups, STI);
    support::endian::write(CB, Binary, llvm::endianness::little);
    Offset = 4;
  }

  // Save the number fixups.
  size_t FixupStartIndex = Fixups.size();

  // Emit an unconditional jump to the destination.
  MCInst TmpInst =
      MCInstBuilder(RISCV::JAL).addReg(RISCV::X0).addOperand(SrcSymbol);
  uint32_t Binary = getBinaryCodeForInstr(TmpInst, Fixups, STI);
  support::endian::write(CB, Binary, llvm::endianness::little);

  // Drop any fixup added so we can add the correct one.
  Fixups.resize(FixupStartIndex);

  if (SrcSymbol.isExpr())
    addFixup(Fixups, Offset, SrcSymbol.getExpr(), RISCV::fixup_riscv_jal);
}

// Expand PseudoLongQC_(E_)Bxxx to an inverted conditional branch and an
// unconditional jump.
void RISCVMCCodeEmitter::expandQCLongCondBrImm(const MCInst &MI,
                                               SmallVectorImpl<char> &CB,
                                               SmallVectorImpl<MCFixup> &Fixups,
                                               const MCSubtargetInfo &STI,
                                               unsigned Size) const {
  MCRegister SrcReg1 = MI.getOperand(0).getReg();
  auto BrImm = MI.getOperand(1).getImm();
  MCOperand SrcSymbol = MI.getOperand(2);
  unsigned Opcode = MI.getOpcode();
  uint32_t Offset;
  unsigned InvOpc = getInvertedBranchOp(Opcode);
  // Emit inverted conditional branch with offset:
  // 8 (QC.BXXX(4) + JAL(4))
  // or
  // 10 (QC.E.BXXX(6) + JAL(4)).
  if (Size == 4) {
    MCInst TmpBr =
        MCInstBuilder(InvOpc).addReg(SrcReg1).addImm(BrImm).addImm(8);
    uint32_t BrBinary = getBinaryCodeForInstr(TmpBr, Fixups, STI);
    support::endian::write(CB, BrBinary, llvm::endianness::little);
  } else {
    MCInst TmpBr =
        MCInstBuilder(InvOpc).addReg(SrcReg1).addImm(BrImm).addImm(10);
    uint64_t BrBinary =
        getBinaryCodeForInstr(TmpBr, Fixups, STI) & 0xffff'ffff'ffffu;
    SmallVector<char, 8> Encoding;
    support::endian::write(Encoding, BrBinary, llvm::endianness::little);
    assert(Encoding[6] == 0 && Encoding[7] == 0 &&
           "Unexpected encoding for 48-bit instruction");
    Encoding.truncate(6);
    CB.append(Encoding);
  }
  Offset = Size;
  // Save the number fixups.
  size_t FixupStartIndex = Fixups.size();
  // Emit an unconditional jump to the destination.
  MCInst TmpJ =
      MCInstBuilder(RISCV::JAL).addReg(RISCV::X0).addOperand(SrcSymbol);
  uint32_t JBinary = getBinaryCodeForInstr(TmpJ, Fixups, STI);
  support::endian::write(CB, JBinary, llvm::endianness::little);
  // Drop any fixup added so we can add the correct one.
  Fixups.resize(FixupStartIndex);
  if (SrcSymbol.isExpr())
    addFixup(Fixups, Offset, SrcSymbol.getExpr(), RISCV::fixup_riscv_jal);
}

void RISCVMCCodeEmitter::encodeInstruction(const MCInst &MI,
                                           SmallVectorImpl<char> &CB,
                                           SmallVectorImpl<MCFixup> &Fixups,
                                           const MCSubtargetInfo &STI) const {
  const MCInstrDesc &Desc = MCII.get(MI.getOpcode());
  // Get byte count of instruction.
  unsigned Size = Desc.getSize();

  // RISCVInstrInfo::getInstSizeInBytes expects that the total size of the
  // expanded instructions for each pseudo is correct in the Size field of the
  // tablegen definition for the pseudo.
  switch (MI.getOpcode()) {
  default:
    break;
  case RISCV::PseudoCALLReg:
  case RISCV::PseudoCALL:
  case RISCV::PseudoTAIL:
  case RISCV::PseudoJump:
    expandFunctionCall(MI, CB, Fixups, STI);
    MCNumEmitted += 2;
    return;
  case RISCV::PseudoAddTPRel:
    expandAddTPRel(MI, CB, Fixups, STI);
    MCNumEmitted += 1;
    return;
  case RISCV::PseudoLongBEQ:
  case RISCV::PseudoLongBNE:
  case RISCV::PseudoLongBLT:
  case RISCV::PseudoLongBGE:
  case RISCV::PseudoLongBLTU:
  case RISCV::PseudoLongBGEU:
    expandLongCondBr(MI, CB, Fixups, STI);
    MCNumEmitted += 2;
    return;
  case RISCV::PseudoLongQC_BEQI:
  case RISCV::PseudoLongQC_BNEI:
  case RISCV::PseudoLongQC_BLTI:
  case RISCV::PseudoLongQC_BGEI:
  case RISCV::PseudoLongQC_BLTUI:
  case RISCV::PseudoLongQC_BGEUI:
    expandQCLongCondBrImm(MI, CB, Fixups, STI, 4);
    MCNumEmitted += 2;
    return;
  case RISCV::PseudoLongQC_E_BEQI:
  case RISCV::PseudoLongQC_E_BNEI:
  case RISCV::PseudoLongQC_E_BLTI:
  case RISCV::PseudoLongQC_E_BGEI:
  case RISCV::PseudoLongQC_E_BLTUI:
  case RISCV::PseudoLongQC_E_BGEUI:
    expandQCLongCondBrImm(MI, CB, Fixups, STI, 6);
    MCNumEmitted += 2;
    return;
  case RISCV::PseudoTLSDESCCall:
    expandTLSDESCCall(MI, CB, Fixups, STI);
    MCNumEmitted += 1;
    return;
  }

  switch (Size) {
  default:
    llvm_unreachable("Unhandled encodeInstruction length!");
  case 2: {
    uint16_t Bits = getBinaryCodeForInstr(MI, Fixups, STI);
    support::endian::write<uint16_t>(CB, Bits, llvm::endianness::little);
    break;
  }
  case 4: {
    uint32_t Bits = getBinaryCodeForInstr(MI, Fixups, STI);
    support::endian::write(CB, Bits, llvm::endianness::little);
    break;
  }
  case 6: {
    uint64_t Bits = getBinaryCodeForInstr(MI, Fixups, STI) & 0xffff'ffff'ffffu;
    SmallVector<char, 8> Encoding;
    support::endian::write(Encoding, Bits, llvm::endianness::little);
    assert(Encoding[6] == 0 && Encoding[7] == 0 &&
           "Unexpected encoding for 48-bit instruction");
    Encoding.truncate(6);
    CB.append(Encoding);
    break;
  }
  case 8: {
    uint64_t Bits = getBinaryCodeForInstr(MI, Fixups, STI);
    support::endian::write(CB, Bits, llvm::endianness::little);
    break;
  }
  }

  ++MCNumEmitted; // Keep track of the # of mi's emitted.
}

uint64_t
RISCVMCCodeEmitter::getMachineOpValue(const MCInst &MI, const MCOperand &MO,
                                      SmallVectorImpl<MCFixup> &Fixups,
                                      const MCSubtargetInfo &STI) const {

  if (MO.isReg())
    return Ctx.getRegisterInfo()->getEncodingValue(MO.getReg());

  if (MO.isImm())
    return MO.getImm();

  llvm_unreachable("Unhandled expression!");
  return 0;
}

uint64_t
RISCVMCCodeEmitter::getImmOpValueMinus1(const MCInst &MI, unsigned OpNo,
                                        SmallVectorImpl<MCFixup> &Fixups,
                                        const MCSubtargetInfo &STI) const {
  const MCOperand &MO = MI.getOperand(OpNo);

  if (MO.isImm()) {
    uint64_t Res = MO.getImm();
    return (Res - 1);
  }

  llvm_unreachable("Unhandled expression!");
  return 0;
}

uint64_t
RISCVMCCodeEmitter::getImmOpValueSlist(const MCInst &MI, unsigned OpNo,
                                       SmallVectorImpl<MCFixup> &Fixups,
                                       const MCSubtargetInfo &STI) const {
  const MCOperand &MO = MI.getOperand(OpNo);
  assert(MO.isImm() && "Slist operand must be immediate");

  uint64_t Res = MO.getImm();
  switch (Res) {
  case 0:
    return 0;
  case 1:
    return 1;
  case 2:
    return 2;
  case 4:
    return 3;
  case 8:
    return 4;
  case 16:
    return 5;
  case 15:
    return 6;
  case 31:
    return 7;
  default:
    llvm_unreachable("Unhandled Slist value!");
  }
}

template <unsigned N>
unsigned
RISCVMCCodeEmitter::getImmOpValueAsrN(const MCInst &MI, unsigned OpNo,
                                      SmallVectorImpl<MCFixup> &Fixups,
                                      const MCSubtargetInfo &STI) const {
  const MCOperand &MO = MI.getOperand(OpNo);

  if (MO.isImm()) {
    uint64_t Res = MO.getImm();
    assert((Res & ((1 << N) - 1)) == 0 && "LSB is non-zero");
    return Res >> N;
  }

  return getImmOpValue(MI, OpNo, Fixups, STI);
}

uint64_t
RISCVMCCodeEmitter::getImmOpValueZibi(const MCInst &MI, unsigned OpNo,
                                      SmallVectorImpl<MCFixup> &Fixups,
                                      const MCSubtargetInfo &STI) const {
  const MCOperand &MO = MI.getOperand(OpNo);
  assert(MO.isImm() && "Zibi operand must be an immediate");
  int64_t Res = MO.getImm();
  if (Res == -1)
    return 0;

  return Res;
}

uint64_t RISCVMCCodeEmitter::getImmOpValue(const MCInst &MI, unsigned OpNo,
                                           SmallVectorImpl<MCFixup> &Fixups,
                                           const MCSubtargetInfo &STI) const {
  bool EnableRelax = STI.hasFeature(RISCV::FeatureRelax);
  const MCOperand &MO = MI.getOperand(OpNo);

  MCInstrDesc const &Desc = MCII.get(MI.getOpcode());
  unsigned MIFrm = RISCVII::getFormat(Desc.TSFlags);

  // If the destination is an immediate, there is nothing to do.
  if (MO.isImm())
    return MO.getImm();

  assert(MO.isExpr() &&
         "getImmOpValue expects only expressions or immediates");
  const MCExpr *Expr = MO.getExpr();
  MCExpr::ExprKind Kind = Expr->getKind();

  // `RelaxCandidate` must be set to `true` in two cases:
  // - The fixup's relocation gets a R_RISCV_RELAX relocation
  // - The underlying instruction may be relaxed to an instruction that gets a
  //   `R_RISCV_RELAX` relocation.
  //
  // The actual emission of `R_RISCV_RELAX` will be handled in
  // `RISCVAsmBackend::applyFixup`.
  bool RelaxCandidate = false;
  auto AsmRelaxToLinkerRelaxableWithFeature = [&](unsigned Feature) -> void {
    if (!STI.hasFeature(RISCV::FeatureExactAssembly) && STI.hasFeature(Feature))
      RelaxCandidate = true;
  };

  unsigned FixupKind = RISCV::fixup_riscv_invalid;
  if (Kind == MCExpr::Specifier) {
    const auto *RVExpr = cast<MCSpecifierExpr>(Expr);
    FixupKind = RVExpr->getSpecifier();
    switch (RVExpr->getSpecifier()) {
    default:
      assert(FixupKind && FixupKind < FirstTargetFixupKind &&
             "invalid specifier");
      break;
    case ELF::R_RISCV_TPREL_ADD:
      // tprel_add is only used to indicate that a relocation should be emitted
      // for an add instruction used in TP-relative addressing. It should not be
      // expanded as if representing an actual instruction operand and so to
      // encounter it here is an error.
      llvm_unreachable(
          "ELF::R_RISCV_TPREL_ADD should not represent an instruction operand");
    case RISCV::S_LO:
      if (MIFrm == RISCVII::InstFormatI)
        FixupKind = RISCV::fixup_riscv_lo12_i;
      else if (MIFrm == RISCVII::InstFormatS)
        FixupKind = RISCV::fixup_riscv_lo12_s;
      else
        llvm_unreachable("VK_LO used with unexpected instruction format");
      RelaxCandidate = true;
      break;
    case ELF::R_RISCV_HI20:
      FixupKind = RISCV::fixup_riscv_hi20;
      RelaxCandidate = true;
      break;
    case RISCV::S_PCREL_LO:
      if (MIFrm == RISCVII::InstFormatI)
        FixupKind = RISCV::fixup_riscv_pcrel_lo12_i;
      else if (MIFrm == RISCVII::InstFormatS)
        FixupKind = RISCV::fixup_riscv_pcrel_lo12_s;
      else
        llvm_unreachable("VK_PCREL_LO used with unexpected instruction format");
      RelaxCandidate = true;
      break;
    case ELF::R_RISCV_PCREL_HI20:
      FixupKind = RISCV::fixup_riscv_pcrel_hi20;
      RelaxCandidate = true;
      break;
    case RISCV::S_TPREL_LO:
      if (MIFrm == RISCVII::InstFormatI)
        FixupKind = ELF::R_RISCV_TPREL_LO12_I;
      else if (MIFrm == RISCVII::InstFormatS)
        FixupKind = ELF::R_RISCV_TPREL_LO12_S;
      else
        llvm_unreachable("VK_TPREL_LO used with unexpected instruction format");
      RelaxCandidate = true;
      break;
    case ELF::R_RISCV_TPREL_HI20:
      RelaxCandidate = true;
      break;
    case ELF::R_RISCV_CALL_PLT:
      FixupKind = RISCV::fixup_riscv_call_plt;
      RelaxCandidate = true;
      break;
    case RISCV::S_QC_ABS20:
      FixupKind = RISCV::fixup_riscv_qc_abs20_u;
      RelaxCandidate = true;
      break;
    }
  } else if (Kind == MCExpr::SymbolRef || Kind == MCExpr::Binary) {
    // FIXME: Sub kind binary exprs have chance of underflow.
    if (MIFrm == RISCVII::InstFormatJ) {
      FixupKind = RISCV::fixup_riscv_jal;
      AsmRelaxToLinkerRelaxableWithFeature(RISCV::FeatureVendorXqcilb);
    } else if (MIFrm == RISCVII::InstFormatB) {
      FixupKind = RISCV::fixup_riscv_branch;
      // This might be assembler relaxed to `b<cc>; jal` but we cannot relax
      // the `jal` again in the assembler.
    } else if (MIFrm == RISCVII::InstFormatCJ) {
      FixupKind = RISCV::fixup_riscv_rvc_jump;
      AsmRelaxToLinkerRelaxableWithFeature(RISCV::FeatureVendorXqcilb);
    } else if (MIFrm == RISCVII::InstFormatCB) {
      FixupKind = RISCV::fixup_riscv_rvc_branch;
      // This might be assembler relaxed to `b<cc>; jal` but we cannot relax
      // the `jal` again in the assembler.
    } else if (MIFrm == RISCVII::InstFormatCI) {
      FixupKind = RISCV::fixup_riscv_rvc_imm;
    } else if (MIFrm == RISCVII::InstFormatI) {
      FixupKind = RISCV::fixup_riscv_12_i;
    } else if (MIFrm == RISCVII::InstFormatQC_EB) {
      FixupKind = RISCV::fixup_riscv_qc_e_branch;
      // This might be assembler relaxed to `qc.e.b<cc>; jal` but we cannot
      // relax the `jal` again in the assembler.
    } else if (MIFrm == RISCVII::InstFormatQC_EAI) {
      FixupKind = RISCV::fixup_riscv_qc_e_32;
      RelaxCandidate = true;
    } else if (MIFrm == RISCVII::InstFormatQC_EJ) {
      FixupKind = RISCV::fixup_riscv_qc_e_call_plt;
      RelaxCandidate = true;
    } else if (MIFrm == RISCVII::InstFormatNDS_BRANCH_10) {
      FixupKind = RISCV::fixup_riscv_nds_branch_10;
    }
  }

  assert(FixupKind != RISCV::fixup_riscv_invalid && "Unhandled expression!");

  addFixup(Fixups, 0, Expr, FixupKind);
  // If linker relaxation is enabled and supported by this relocation, set a bit
  // so that the assembler knows the size of the instruction is not fixed/known,
  // and the relocation will need a R_RISCV_RELAX relocation.
  if (EnableRelax && RelaxCandidate)
    Fixups.back().setLinkerRelaxable();
  ++MCNumFixups;

  return 0;
}

unsigned RISCVMCCodeEmitter::getVMaskReg(const MCInst &MI, unsigned OpNo,
                                         SmallVectorImpl<MCFixup> &Fixups,
                                         const MCSubtargetInfo &STI) const {
  MCOperand MO = MI.getOperand(OpNo);
  assert(MO.isReg() && "Expected a register.");

  switch (MO.getReg()) {
  default:
    llvm_unreachable("Invalid mask register.");
  case RISCV::V0:
    return 0;
  case RISCV::NoRegister:
    return 1;
  }
}

unsigned RISCVMCCodeEmitter::getRlistOpValue(const MCInst &MI, unsigned OpNo,
                                             SmallVectorImpl<MCFixup> &Fixups,
                                             const MCSubtargetInfo &STI) const {
  const MCOperand &MO = MI.getOperand(OpNo);
  assert(MO.isImm() && "Rlist operand must be immediate");
  auto Imm = MO.getImm();
  assert(Imm >= 4 && "EABI is currently not implemented");
  return Imm;
}
unsigned
RISCVMCCodeEmitter::getRlistS0OpValue(const MCInst &MI, unsigned OpNo,
                                      SmallVectorImpl<MCFixup> &Fixups,
                                      const MCSubtargetInfo &STI) const {
  const MCOperand &MO = MI.getOperand(OpNo);
  assert(MO.isImm() && "Rlist operand must be immediate");
  auto Imm = MO.getImm();
  assert(Imm >= 4 && "EABI is currently not implemented");
  assert(Imm != RISCVZC::RA && "Rlist operand must include s0");
  return Imm;
}

#include "RISCVGenMCCodeEmitter.inc"
