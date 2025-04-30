//===-- Next32MCCodeEmitter.cpp - Convert Next32 code to machine code
//-----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Next32MCCodeEmitter class.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/Next32MCTargetDesc.h"
#include "Next32MCExpr.h"
#include "TargetInfo/Next32BaseInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/EndianStream.h"
#include <cassert>
#include <cstdint>

using namespace llvm;

#define DEBUG_TYPE "mccodeemitter"

namespace {

class Next32MCCodeEmitter : public MCCodeEmitter {
  const MCInstrInfo &MCII;
  MCContext &Ctx;

public:
  Next32MCCodeEmitter(const MCInstrInfo &mcii, MCContext &Ctx)
      : MCII(mcii), Ctx(Ctx) {}
  Next32MCCodeEmitter(const Next32MCCodeEmitter &) = delete;
  void operator=(const Next32MCCodeEmitter &) = delete;
  ~Next32MCCodeEmitter() override = default;

  // getBinaryCodeForInstr - TableGen'erated function for getting the
  // binary encoding for an instruction.
  uint64_t getBinaryCodeForInstr(const MCInst &MI,
                                 SmallVectorImpl<MCFixup> &Fixups,
                                 const MCSubtargetInfo &STI) const;

  // getMachineOpValue - Return binary encoding of operand. If the machin
  // operand requires relocation, record the relocation and return zero.
  unsigned getMachineOpValue(const MCInst &MI, const MCOperand &MO,
                             SmallVectorImpl<MCFixup> &Fixups,
                             const MCSubtargetInfo &STI) const;

  void encodeInstruction(const MCInst &MI, SmallVectorImpl<char> &CB,
                         SmallVectorImpl<MCFixup> &Fixups,
                         const MCSubtargetInfo &STI) const override;

private:
  unsigned int HandleExpr(const MCExpr *Expr,
                          SmallVectorImpl<MCFixup> &Fixups) const;
};

} // end anonymous namespace

MCCodeEmitter *llvm::createNext32MCCodeEmitter(const MCInstrInfo &MCII,
                                               MCContext &Ctx) {
  return new Next32MCCodeEmitter(MCII, Ctx);
}

unsigned
Next32MCCodeEmitter::getMachineOpValue(const MCInst &MI, const MCOperand &MO,
                                       SmallVectorImpl<MCFixup> &Fixups,
                                       const MCSubtargetInfo &STI) const {
  if (MO.isReg())
    return Ctx.getRegisterInfo()->getEncodingValue(MO.getReg());

  if (MO.isImm())
    return static_cast<unsigned>(MO.getImm());

  if (MO.isExpr())
    return HandleExpr(MO.getExpr(), Fixups);

  assert(false && "Invalid machine operand value");
  return 0;
}

unsigned int
Next32MCCodeEmitter::HandleExpr(const MCExpr *Expr,
                                SmallVectorImpl<MCFixup> &Fixups) const {
  if (const MCSymbolRefExpr *SymExpr = dyn_cast<const MCSymbolRefExpr>(Expr)) {
    auto FixupExpr = MCSymbolRefExpr::create(&SymExpr->getSymbol(), Ctx);
    Fixups.push_back(MCFixup::create(
        4, FixupExpr, (MCFixupKind)Next32::reloc_4byte_sym_bb_imm));
    return 0x11AD11AD;
  }

  if (const Next32MCExpr *Next32Expr = dyn_cast<const Next32MCExpr>(Expr)) {
    auto FixupExpr = MCSymbolRefExpr::create(Next32Expr->getSymbol(), Ctx);
    Fixups.push_back(
        MCFixup::create(4, FixupExpr, (MCFixupKind)Next32Expr->getKind()));
    return 0x11AD11AD;
  }
  llvm_unreachable("Unsupported MCExpr");
}

void Next32MCCodeEmitter::encodeInstruction(const MCInst &MI,
                                            SmallVectorImpl<char> &CB,
                                            SmallVectorImpl<MCFixup> &Fixups,
                                            const MCSubtargetInfo &STI) const {

  uint64_t InstrFirstPart = getBinaryCodeForInstr(MI, Fixups, STI);
  if (MI.getOpcode() != Next32::MOVL && MI.getOpcode() != Next32::ADC &&
      MI.getOpcode() != Next32::SBB && MI.getOpcode() != Next32::LEAINDEX &&
      MI.getOpcode() != Next32::PREFETCH &&
      (InstrFirstPart & 0xF000000000000000) == 0) {
    InstrFirstPart ^= InstrFirstPart & 0xFFFF00000000;
  }

  if (MI.getOpcode() == Next32::FEEDER || MI.getOpcode() == Next32::FEEDERP) {
    InstrFirstPart ^= InstrFirstPart & 0xFFFF0000;
  }

  support::endian::write<uint64_t>(CB, InstrFirstPart, llvm::endianness::big);

  const MCInstrDesc &Desc = MCII.get(MI.getOpcode());
  if (!(Desc.TSFlags & llvm::Next32II::Is128BitRRRRInstruction))
    return;

  // This code assumes that the SrcReg1 and SrcReg2 are specified last in the
  // opcode's 'ins' and 'outs' list (according to whether they're in/out/inout),
  // and so we can consume operands backwards to obtain the correct MCOperands
  // corresponding to the registers that were allocated and that we must now
  // encode into the second uint64_t of the instruction.

  unsigned int DefsIndex = Desc.getNumDefs() - 1;
  unsigned int OpsIndex = Desc.getNumOperands() - 1;
  const MCOperand *SrcReg1Operand = nullptr;
  const MCOperand *SrcReg2Operand = nullptr;
  unsigned int Count = 0;
  unsigned int AddrSpace = 0;

  if (Desc.TSFlags & Next32II::is128BitRRRRInstructionWithAddrSpace)
    AddrSpace = MI.getOperand(OpsIndex--).getImm();

  if (Desc.TSFlags & Next32II::is128BitRRRRInstructionWithCount)
    Count = MI.getOperand(OpsIndex--).getImm();

  if (Desc.TSFlags & Next32II::Is128BitRRRRInstructionSrcReg2Out)
    SrcReg2Operand = &MI.getOperand(DefsIndex--);
  else if (Desc.TSFlags & Next32II::Is128BitRRRRInstructionSrcReg2In)
    SrcReg2Operand = &MI.getOperand(OpsIndex--);

  if (Desc.TSFlags & Next32II::Is128BitRRRRInstructionSrcReg1Out)
    SrcReg1Operand = &MI.getOperand(DefsIndex--);
  else if (Desc.TSFlags & Next32II::Is128BitRRRRInstructionSrcReg1In)
    SrcReg1Operand = &MI.getOperand(OpsIndex--);

  const unsigned int SrcReg1 =
      SrcReg1Operand ? getMachineOpValue(MI, *SrcReg1Operand, Fixups, STI) : 0;
  const unsigned int SrcReg2 =
      SrcReg2Operand ? getMachineOpValue(MI, *SrcReg2Operand, Fixups, STI) : 0;
  const uint64_t InstrSecondPart =
      (static_cast<uint64_t>(SrcReg1 & 0xFFFF) << 48) |
      (static_cast<uint64_t>(SrcReg2 & 0xFFFF) << 32) |
      (static_cast<uint64_t>(AddrSpace & 0x7) << 3) |
      (static_cast<uint64_t>(Count & 0x7));
  support::endian::write<uint64_t>(CB, InstrSecondPart, llvm::endianness::big);
}

#define ENABLE_INSTR_PREDICATE_VERIFIER
#include "Next32GenMCCodeEmitter.inc"
