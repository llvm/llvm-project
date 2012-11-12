//===-- MBlazeMCCodeEmitter.cpp - Convert MBlaze code to machine code -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MBlazeMCCodeEmitter class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "mccodeemitter"
#include "MCTargetDesc/MBlazeBaseInfo.h"
#include "MCTargetDesc/MBlazeMCTargetDesc.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

STATISTIC(MCNumEmitted, "Number of MC instructions emitted");

namespace {
class MBlazeMCCodeEmitter : public MCCodeEmitter {
  MBlazeMCCodeEmitter(const MBlazeMCCodeEmitter &) LLVM_DELETED_FUNCTION;
  void operator=(const MBlazeMCCodeEmitter &) LLVM_DELETED_FUNCTION;
  const MCInstrInfo &MCII;

public:
  MBlazeMCCodeEmitter(const MCInstrInfo &mcii, const MCSubtargetInfo &sti,
                      MCContext &ctx)
    : MCII(mcii) {
  }

  ~MBlazeMCCodeEmitter() {}

  // getBinaryCodeForInstr - TableGen'erated function for getting the
  // binary encoding for an instruction.
  uint64_t getBinaryCodeForInstr(const MCInst &MI) const;

  /// getMachineOpValue - Return binary encoding of operand. If the machine
  /// operand requires relocation, record the relocation and return zero.
  unsigned getMachineOpValue(const MCInst &MI,const MCOperand &MO) const;
  unsigned getMachineOpValue(const MCInst &MI, unsigned OpIdx) const {
    return getMachineOpValue(MI, MI.getOperand(OpIdx));
  }

  static unsigned GetMBlazeRegNum(const MCOperand &MO) {
    // FIXME: getMBlazeRegisterNumbering() is sufficient?
    llvm_unreachable("MBlazeMCCodeEmitter::GetMBlazeRegNum() not yet "
                     "implemented.");
  }

  void EmitByte(unsigned char C, unsigned &CurByte, raw_ostream &OS) const {
    // The MicroBlaze uses a bit reversed format so we need to reverse the
    // order of the bits. Taken from:
    // http://graphics.stanford.edu/~seander/bithacks.html
    C = ((C * 0x80200802ULL) & 0x0884422110ULL) * 0x0101010101ULL >> 32;

    OS << (char)C;
    ++CurByte;
  }

  void EmitRawByte(unsigned char C, unsigned &CurByte, raw_ostream &OS) const {
    OS << (char)C;
    ++CurByte;
  }

  void EmitConstant(uint64_t Val, unsigned Size, unsigned &CurByte,
                    raw_ostream &OS) const {
    assert(Size <= 8 && "size too big in emit constant");

    for (unsigned i = 0; i != Size; ++i) {
      EmitByte(Val & 255, CurByte, OS);
      Val >>= 8;
    }
  }

  void EmitIMM(const MCOperand &imm, unsigned &CurByte, raw_ostream &OS) const;
  void EmitIMM(const MCInst &MI, unsigned &CurByte, raw_ostream &OS) const;

  void EmitImmediate(const MCInst &MI, unsigned opNo, bool pcrel,
                     unsigned &CurByte, raw_ostream &OS,
                     SmallVectorImpl<MCFixup> &Fixups) const;

  void EncodeInstruction(const MCInst &MI, raw_ostream &OS,
                         SmallVectorImpl<MCFixup> &Fixups) const;
};

} // end anonymous namespace


MCCodeEmitter *llvm::createMBlazeMCCodeEmitter(const MCInstrInfo &MCII,
                                               const MCRegisterInfo &MRI,
                                               const MCSubtargetInfo &STI,
                                               MCContext &Ctx) {
  return new MBlazeMCCodeEmitter(MCII, STI, Ctx);
}

/// getMachineOpValue - Return binary encoding of operand. If the machine
/// operand requires relocation, record the relocation and return zero.
unsigned MBlazeMCCodeEmitter::getMachineOpValue(const MCInst &MI,
                                             const MCOperand &MO) const {
  if (MO.isReg())
    return getMBlazeRegisterNumbering(MO.getReg());
  if (MO.isImm())
    return static_cast<unsigned>(MO.getImm());
  if (MO.isExpr())
    return 0; // The relocation has already been recorded at this point.
#ifndef NDEBUG
  errs() << MO;
#endif
  llvm_unreachable(0);
}

void MBlazeMCCodeEmitter::
EmitIMM(const MCOperand &imm, unsigned &CurByte, raw_ostream &OS) const {
  int32_t val = (int32_t)imm.getImm();
  if (val > 32767 || val < -32768) {
    EmitByte(0x0D, CurByte, OS);
    EmitByte(0x00, CurByte, OS);
    EmitRawByte((val >> 24) & 0xFF, CurByte, OS);
    EmitRawByte((val >> 16) & 0xFF, CurByte, OS);
  }
}

void MBlazeMCCodeEmitter::
EmitIMM(const MCInst &MI, unsigned &CurByte,raw_ostream &OS) const {
  switch (MI.getOpcode()) {
  default: break;

  case MBlaze::ADDIK32:
  case MBlaze::ORI32:
  case MBlaze::BRLID32:
    EmitByte(0x0D, CurByte, OS);
    EmitByte(0x00, CurByte, OS);
    EmitRawByte(0, CurByte, OS);
    EmitRawByte(0, CurByte, OS);
  }
}

void MBlazeMCCodeEmitter::
EmitImmediate(const MCInst &MI, unsigned opNo, bool pcrel, unsigned &CurByte,
              raw_ostream &OS, SmallVectorImpl<MCFixup> &Fixups) const {
  assert(MI.getNumOperands()>opNo && "Not enought operands for instruction");

  MCOperand oper = MI.getOperand(opNo);

  if (oper.isImm()) {
    EmitIMM(oper, CurByte, OS);
  } else if (oper.isExpr()) {
    MCFixupKind FixupKind;
    switch (MI.getOpcode()) {
    default:
      FixupKind = pcrel ? FK_PCRel_2 : FK_Data_2;
      Fixups.push_back(MCFixup::Create(0,oper.getExpr(),FixupKind));
      break;
    case MBlaze::ORI32:
    case MBlaze::ADDIK32:
    case MBlaze::BRLID32:
      FixupKind = pcrel ? FK_PCRel_4 : FK_Data_4;
      Fixups.push_back(MCFixup::Create(0,oper.getExpr(),FixupKind));
      break;
    }
  }
}



void MBlazeMCCodeEmitter::
EncodeInstruction(const MCInst &MI, raw_ostream &OS,
                  SmallVectorImpl<MCFixup> &Fixups) const {
  unsigned Opcode = MI.getOpcode();
  const MCInstrDesc &Desc = MCII.get(Opcode);
  uint64_t TSFlags = Desc.TSFlags;
  // Keep track of the current byte being emitted.
  unsigned CurByte = 0;

  // Emit an IMM instruction if the instruction we are encoding requires it
  EmitIMM(MI,CurByte,OS);

  switch ((TSFlags & MBlazeII::FormMask)) {
  default: break;
  case MBlazeII::FPseudo:
    // Pseudo instructions don't get encoded.
    return;
  case MBlazeII::FRRI:
    EmitImmediate(MI, 2, false, CurByte, OS, Fixups);
    break;
  case MBlazeII::FRIR:
    EmitImmediate(MI, 1, false, CurByte, OS, Fixups);
    break;
  case MBlazeII::FCRI:
    EmitImmediate(MI, 1, true, CurByte, OS, Fixups);
    break;
  case MBlazeII::FRCI:
    EmitImmediate(MI, 1, true, CurByte, OS, Fixups);
  case MBlazeII::FCCI:
    EmitImmediate(MI, 0, true, CurByte, OS, Fixups);
    break;
  }

  ++MCNumEmitted;  // Keep track of the # of mi's emitted
  unsigned Value = getBinaryCodeForInstr(MI);
  EmitConstant(Value, 4, CurByte, OS);
}

// FIXME: These #defines shouldn't be necessary. Instead, tblgen should
// be able to generate code emitter helpers for either variant, like it
// does for the AsmWriter.
#define MBlazeCodeEmitter MBlazeMCCodeEmitter
#define MachineInstr MCInst
#include "MBlazeGenCodeEmitter.inc"
#undef MBlazeCodeEmitter
#undef MachineInstr
