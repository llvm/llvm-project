//===-- MipsMCCodeEmitter.cpp - Convert Mips Code to Machine Code ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MipsMCCodeEmitter class.
//
//===----------------------------------------------------------------------===//
//
#define DEBUG_TYPE "mccodeemitter"
#include "MCTargetDesc/MipsBaseInfo.h"
#include "MCTargetDesc/MipsDirectObjLower.h"
#include "MCTargetDesc/MipsFixupKinds.h"
#include "MCTargetDesc/MipsMCTargetDesc.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
class MipsMCCodeEmitter : public MCCodeEmitter {
  MipsMCCodeEmitter(const MipsMCCodeEmitter &) LLVM_DELETED_FUNCTION;
  void operator=(const MipsMCCodeEmitter &) LLVM_DELETED_FUNCTION;
  const MCInstrInfo &MCII;
  bool IsLittleEndian;

public:
  MipsMCCodeEmitter(const MCInstrInfo &mcii, bool IsLittle) :
            MCII(mcii), IsLittleEndian(IsLittle) {}

  ~MipsMCCodeEmitter() {}

  void EmitByte(unsigned char C, raw_ostream &OS) const {
    OS << (char)C;
  }

  void EmitInstruction(uint64_t Val, unsigned Size, raw_ostream &OS) const {
    // Output the instruction encoding in little endian byte order.
    for (unsigned i = 0; i < Size; ++i) {
      unsigned Shift = IsLittleEndian ? i * 8 : (Size - 1 - i) * 8;
      EmitByte((Val >> Shift) & 0xff, OS);
    }
  }

  void EncodeInstruction(const MCInst &MI, raw_ostream &OS,
                         SmallVectorImpl<MCFixup> &Fixups) const;

  // getBinaryCodeForInstr - TableGen'erated function for getting the
  // binary encoding for an instruction.
  uint64_t getBinaryCodeForInstr(const MCInst &MI,
                                 SmallVectorImpl<MCFixup> &Fixups) const;

  // getBranchJumpOpValue - Return binary encoding of the jump
  // target operand. If the machine operand requires relocation,
  // record the relocation and return zero.
   unsigned getJumpTargetOpValue(const MCInst &MI, unsigned OpNo,
                                 SmallVectorImpl<MCFixup> &Fixups) const;

   // getBranchTargetOpValue - Return binary encoding of the branch
   // target operand. If the machine operand requires relocation,
   // record the relocation and return zero.
  unsigned getBranchTargetOpValue(const MCInst &MI, unsigned OpNo,
                                  SmallVectorImpl<MCFixup> &Fixups) const;

   // getMachineOpValue - Return binary encoding of operand. If the machin
   // operand requires relocation, record the relocation and return zero.
  unsigned getMachineOpValue(const MCInst &MI,const MCOperand &MO,
                             SmallVectorImpl<MCFixup> &Fixups) const;

  unsigned getMemEncoding(const MCInst &MI, unsigned OpNo,
                          SmallVectorImpl<MCFixup> &Fixups) const;
  unsigned getSizeExtEncoding(const MCInst &MI, unsigned OpNo,
                              SmallVectorImpl<MCFixup> &Fixups) const;
  unsigned getSizeInsEncoding(const MCInst &MI, unsigned OpNo,
                              SmallVectorImpl<MCFixup> &Fixups) const;

}; // class MipsMCCodeEmitter
}  // namespace

MCCodeEmitter *llvm::createMipsMCCodeEmitterEB(const MCInstrInfo &MCII,
                                               const MCRegisterInfo &MRI,
                                               const MCSubtargetInfo &STI,
                                               MCContext &Ctx)
{
  return new MipsMCCodeEmitter(MCII, false);
}

MCCodeEmitter *llvm::createMipsMCCodeEmitterEL(const MCInstrInfo &MCII,
                                               const MCRegisterInfo &MRI,
                                               const MCSubtargetInfo &STI,
                                               MCContext &Ctx)
{
  return new MipsMCCodeEmitter(MCII, true);
}

/// EncodeInstruction - Emit the instruction.
/// Size the instruction (currently only 4 bytes
void MipsMCCodeEmitter::
EncodeInstruction(const MCInst &MI, raw_ostream &OS,
                  SmallVectorImpl<MCFixup> &Fixups) const
{

  // Non-pseudo instructions that get changed for direct object
  // only based on operand values.
  // If this list of instructions get much longer we will move
  // the check to a function call. Until then, this is more efficient.
  MCInst TmpInst = MI;
  switch (MI.getOpcode()) {
  // If shift amount is >= 32 it the inst needs to be lowered further
  case Mips::DSLL:
  case Mips::DSRL:
  case Mips::DSRA:
    Mips::LowerLargeShift(TmpInst);
    break;
    // Double extract instruction is chosen by pos and size operands
  case Mips::DEXT:
  case Mips::DINS:
    Mips::LowerDextDins(TmpInst);
  }

  uint32_t Binary = getBinaryCodeForInstr(TmpInst, Fixups);

  // Check for unimplemented opcodes.
  // Unfortunately in MIPS both NOP and SLL will come in with Binary == 0
  // so we have to special check for them.
  unsigned Opcode = TmpInst.getOpcode();
  if ((Opcode != Mips::NOP) && (Opcode != Mips::SLL) && !Binary)
    llvm_unreachable("unimplemented opcode in EncodeInstruction()");

  const MCInstrDesc &Desc = MCII.get(TmpInst.getOpcode());
  uint64_t TSFlags = Desc.TSFlags;

  // Pseudo instructions don't get encoded and shouldn't be here
  // in the first place!
  if ((TSFlags & MipsII::FormMask) == MipsII::Pseudo)
    llvm_unreachable("Pseudo opcode found in EncodeInstruction()");

  // Get byte count of instruction
  unsigned Size = Desc.getSize();
  if (!Size)
    llvm_unreachable("Desc.getSize() returns 0");

  EmitInstruction(Binary, Size, OS);
}

/// getBranchTargetOpValue - Return binary encoding of the branch
/// target operand. If the machine operand requires relocation,
/// record the relocation and return zero.
unsigned MipsMCCodeEmitter::
getBranchTargetOpValue(const MCInst &MI, unsigned OpNo,
                       SmallVectorImpl<MCFixup> &Fixups) const {

  const MCOperand &MO = MI.getOperand(OpNo);

  // If the destination is an immediate, we have nothing to do.
  if (MO.isImm()) return MO.getImm();
  assert(MO.isExpr() &&
         "getBranchTargetOpValue expects only expressions or immediates");

  const MCExpr *Expr = MO.getExpr();
  Fixups.push_back(MCFixup::Create(0, Expr,
                                   MCFixupKind(Mips::fixup_Mips_PC16)));
  return 0;
}

/// getJumpTargetOpValue - Return binary encoding of the jump
/// target operand. If the machine operand requires relocation,
/// record the relocation and return zero.
unsigned MipsMCCodeEmitter::
getJumpTargetOpValue(const MCInst &MI, unsigned OpNo,
                     SmallVectorImpl<MCFixup> &Fixups) const {

  const MCOperand &MO = MI.getOperand(OpNo);
  // If the destination is an immediate, we have nothing to do.
  if (MO.isImm()) return MO.getImm();
  assert(MO.isExpr() &&
         "getJumpTargetOpValue expects only expressions or an immediate");

  const MCExpr *Expr = MO.getExpr();
  Fixups.push_back(MCFixup::Create(0, Expr,
                                   MCFixupKind(Mips::fixup_Mips_26)));
  return 0;
}

/// getMachineOpValue - Return binary encoding of operand. If the machine
/// operand requires relocation, record the relocation and return zero.
unsigned MipsMCCodeEmitter::
getMachineOpValue(const MCInst &MI, const MCOperand &MO,
                  SmallVectorImpl<MCFixup> &Fixups) const {
  if (MO.isReg()) {
    unsigned Reg = MO.getReg();
    unsigned RegNo = getMipsRegisterNumbering(Reg);
    return RegNo;
  } else if (MO.isImm()) {
    return static_cast<unsigned>(MO.getImm());
  } else if (MO.isFPImm()) {
    return static_cast<unsigned>(APFloat(MO.getFPImm())
        .bitcastToAPInt().getHiBits(32).getLimitedValue());
  }

  // MO must be an Expr.
  assert(MO.isExpr());

  const MCExpr *Expr = MO.getExpr();
  MCExpr::ExprKind Kind = Expr->getKind();

  if (Kind == MCExpr::Binary) {
    Expr = static_cast<const MCBinaryExpr*>(Expr)->getLHS();
    Kind = Expr->getKind();
  }

  assert (Kind == MCExpr::SymbolRef);

  Mips::Fixups FixupKind = Mips::Fixups(0);

  switch(cast<MCSymbolRefExpr>(Expr)->getKind()) {
  default: llvm_unreachable("Unknown fixup kind!");
    break;
  case MCSymbolRefExpr::VK_Mips_GPOFF_HI :
    FixupKind = Mips::fixup_Mips_GPOFF_HI;
    break;
  case MCSymbolRefExpr::VK_Mips_GPOFF_LO :
    FixupKind = Mips::fixup_Mips_GPOFF_LO;
    break;
  case MCSymbolRefExpr::VK_Mips_GOT_PAGE :
    FixupKind = Mips::fixup_Mips_GOT_PAGE;
    break;
  case MCSymbolRefExpr::VK_Mips_GOT_OFST :
    FixupKind = Mips::fixup_Mips_GOT_OFST;
    break;
  case MCSymbolRefExpr::VK_Mips_GOT_DISP :
    FixupKind = Mips::fixup_Mips_GOT_DISP;
    break;
  case MCSymbolRefExpr::VK_Mips_GPREL:
    FixupKind = Mips::fixup_Mips_GPREL16;
    break;
  case MCSymbolRefExpr::VK_Mips_GOT_CALL:
    FixupKind = Mips::fixup_Mips_CALL16;
    break;
  case MCSymbolRefExpr::VK_Mips_GOT16:
    FixupKind = Mips::fixup_Mips_GOT_Global;
    break;
  case MCSymbolRefExpr::VK_Mips_GOT:
    FixupKind = Mips::fixup_Mips_GOT_Local;
    break;
  case MCSymbolRefExpr::VK_Mips_ABS_HI:
    FixupKind = Mips::fixup_Mips_HI16;
    break;
  case MCSymbolRefExpr::VK_Mips_ABS_LO:
    FixupKind = Mips::fixup_Mips_LO16;
    break;
  case MCSymbolRefExpr::VK_Mips_TLSGD:
    FixupKind = Mips::fixup_Mips_TLSGD;
    break;
  case MCSymbolRefExpr::VK_Mips_TLSLDM:
    FixupKind = Mips::fixup_Mips_TLSLDM;
    break;
  case MCSymbolRefExpr::VK_Mips_DTPREL_HI:
    FixupKind = Mips::fixup_Mips_DTPREL_HI;
    break;
  case MCSymbolRefExpr::VK_Mips_DTPREL_LO:
    FixupKind = Mips::fixup_Mips_DTPREL_LO;
    break;
  case MCSymbolRefExpr::VK_Mips_GOTTPREL:
    FixupKind = Mips::fixup_Mips_GOTTPREL;
    break;
  case MCSymbolRefExpr::VK_Mips_TPREL_HI:
    FixupKind = Mips::fixup_Mips_TPREL_HI;
    break;
  case MCSymbolRefExpr::VK_Mips_TPREL_LO:
    FixupKind = Mips::fixup_Mips_TPREL_LO;
    break;
  case MCSymbolRefExpr::VK_Mips_HIGHER:
    FixupKind = Mips::fixup_Mips_HIGHER;
    break;
  case MCSymbolRefExpr::VK_Mips_HIGHEST:
    FixupKind = Mips::fixup_Mips_HIGHEST;
    break;
  } // switch

  Fixups.push_back(MCFixup::Create(0, MO.getExpr(), MCFixupKind(FixupKind)));

  // All of the information is in the fixup.
  return 0;
}

/// getMemEncoding - Return binary encoding of memory related operand.
/// If the offset operand requires relocation, record the relocation.
unsigned
MipsMCCodeEmitter::getMemEncoding(const MCInst &MI, unsigned OpNo,
                                  SmallVectorImpl<MCFixup> &Fixups) const {
  // Base register is encoded in bits 20-16, offset is encoded in bits 15-0.
  assert(MI.getOperand(OpNo).isReg());
  unsigned RegBits = getMachineOpValue(MI, MI.getOperand(OpNo),Fixups) << 16;
  unsigned OffBits = getMachineOpValue(MI, MI.getOperand(OpNo+1), Fixups);

  return (OffBits & 0xFFFF) | RegBits;
}

unsigned
MipsMCCodeEmitter::getSizeExtEncoding(const MCInst &MI, unsigned OpNo,
                                      SmallVectorImpl<MCFixup> &Fixups) const {
  assert(MI.getOperand(OpNo).isImm());
  unsigned SizeEncoding = getMachineOpValue(MI, MI.getOperand(OpNo), Fixups);
  return SizeEncoding - 1;
}

// FIXME: should be called getMSBEncoding
//
unsigned
MipsMCCodeEmitter::getSizeInsEncoding(const MCInst &MI, unsigned OpNo,
                                      SmallVectorImpl<MCFixup> &Fixups) const {
  assert(MI.getOperand(OpNo-1).isImm());
  assert(MI.getOperand(OpNo).isImm());
  unsigned Position = getMachineOpValue(MI, MI.getOperand(OpNo-1), Fixups);
  unsigned Size = getMachineOpValue(MI, MI.getOperand(OpNo), Fixups);

  return Position + Size - 1;
}

#include "MipsGenMCCodeEmitter.inc"

