//===-- Parasol/ParasolMCCodeEmitter.cpp - Convert Parasol code to machine code
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//
//
// This file implements the ParasolMCCodeEmitter class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "mccodeemitter"
#include "MCTargetDesc/ParasolMCTargetDesc.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/SubtargetFeature.h"

using namespace llvm;

STATISTIC(MCNumEmitted, "Number of MC instructions emitted.");

namespace {
class ParasolMCCodeEmitter : public MCCodeEmitter {
  MCContext &Ctx;

public:
  ParasolMCCodeEmitter(const MCInstrInfo &, MCContext &ctx) : Ctx(ctx) {}

  ParasolMCCodeEmitter(const ParasolMCCodeEmitter &) = delete;
  ParasolMCCodeEmitter &operator=(const ParasolMCCodeEmitter &) = delete;

  ~ParasolMCCodeEmitter() override = default;

  void encodeInstruction(const MCInst &MI, SmallVectorImpl<char> &CB,
                         SmallVectorImpl<MCFixup> &Fixups,
                         const MCSubtargetInfo &STI) const override;

  // getBinaryCodeForInstr - TableGen'erated function for getting the
  // binary encoding for an instruction.
  uint64_t getBinaryCodeForInstr(const MCInst &MI,
                                 SmallVectorImpl<MCFixup> &Fixups,
                                 const MCSubtargetInfo &STI) const;

  /// getMachineOpValue - Return binary encoding of operand. If the machine
  /// operand requires relocation, record the relocation and return zero.
  unsigned getMachineOpValue(const MCInst &MI, const MCOperand &MO,
                             SmallVectorImpl<MCFixup> &Fixups,
                             const MCSubtargetInfo &STI) const;
  unsigned getCallTargetOpValue(const MCInst &MI, unsigned OpNo,
                                SmallVectorImpl<MCFixup> &Fixups,
                                const MCSubtargetInfo &STI) const;
  unsigned getBranchTargetOpValue(const MCInst &MI, unsigned OpNo,
                                  SmallVectorImpl<MCFixup> &Fixups,
                                  const MCSubtargetInfo &STI) const;
};

} // end anonymous namespace

void ParasolMCCodeEmitter::encodeInstruction(const MCInst &MI,
                                             SmallVectorImpl<char> &CB,
                                             SmallVectorImpl<MCFixup> &Fixups,
                                             const MCSubtargetInfo &STI) const {
  uint64_t Bits = getBinaryCodeForInstr(MI, Fixups, STI);
  support::endian::write<uint64_t>(CB, Bits,
                                   Ctx.getAsmInfo()->isLittleEndian()
                                       ? llvm::endianness::little
                                       : llvm::endianness::big);

  ++MCNumEmitted; // Keep track of the # of mi's emitted.
}

unsigned
ParasolMCCodeEmitter::getMachineOpValue(const MCInst &MI, const MCOperand &MO,
                                        SmallVectorImpl<MCFixup> &Fixups,
                                        const MCSubtargetInfo &STI) const {
  if (MO.isReg())
    return Ctx.getRegisterInfo()->getEncodingValue(MO.getReg());

  if (MO.isImm())
    return MO.getImm();

  assert(MO.isExpr());
  const MCExpr *Expr = MO.getExpr();
  // SUNSCREEN TODO: Reenable?
  //   if (const ParasolMCExpr *SExpr = dyn_cast<ParasolMCExpr>(Expr)) {
  //     MCFixupKind Kind = (MCFixupKind)SExpr->getFixupKind();
  //     Fixups.push_back(MCFixup::create(0, Expr, Kind));
  //     return 0;
  //   }

  int64_t Res;
  if (Expr->evaluateAsAbsolute(Res))
    return Res;

  llvm_unreachable("Unhandled expression!");
  return 0;
}

unsigned
ParasolMCCodeEmitter::getCallTargetOpValue(const MCInst &MI, unsigned OpNo,
                                           SmallVectorImpl<MCFixup> &Fixups,
                                           const MCSubtargetInfo &STI) const {
  const MCOperand &MO = MI.getOperand(OpNo);
  const MCExpr *Expr = MO.getExpr();

  return 0;
}

unsigned
ParasolMCCodeEmitter::getBranchTargetOpValue(const MCInst &MI, unsigned OpNo,
                                             SmallVectorImpl<MCFixup> &Fixups,
                                             const MCSubtargetInfo &STI) const {
  const MCOperand &MO = MI.getOperand(OpNo);
  if (MO.isReg() || MO.isImm())
    return getMachineOpValue(MI, MO, Fixups, STI);

  return 0;
}

#include "ParasolGenMCCodeEmitter.inc"

MCCodeEmitter *llvm::createParasolMCCodeEmitter(const MCInstrInfo &MCII,
                                                MCContext &Ctx) {
  return new ParasolMCCodeEmitter(MCII, Ctx);
}
