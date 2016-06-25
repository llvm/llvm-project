//===-- llvm/CodeGen/DwarfExpression.cpp - Dwarf Debug Framework ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing dwarf debug info into asm files.
//
//===----------------------------------------------------------------------===//

#include "DwarfExpression.h"
#include "DwarfDebug.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"

using namespace llvm;

void DwarfExpression::AddReg(int DwarfReg, const char *Comment) {
  assert(DwarfReg >= 0 && "invalid negative dwarf register number");
  if (DwarfReg < 32) {
    EmitOp(dwarf::DW_OP_reg0 + DwarfReg, Comment);
  } else {
    EmitOp(dwarf::DW_OP_regx, Comment);
    EmitUnsigned(DwarfReg);
  }
}

void DwarfExpression::AddRegIndirect(int DwarfReg, int Offset, bool Deref) {
  assert(DwarfReg >= 0 && "invalid negative dwarf register number");
  if (DwarfReg < 32) {
    EmitOp(dwarf::DW_OP_breg0 + DwarfReg);
  } else {
    EmitOp(dwarf::DW_OP_bregx);
    EmitUnsigned(DwarfReg);
  }
  EmitSigned(Offset);
  if (Deref)
    EmitOp(dwarf::DW_OP_deref);
}

void DwarfExpression::AddOpPiece(unsigned SizeInBits, unsigned OffsetInBits) {
  assert(SizeInBits > 0 && "piece has size zero");
  const unsigned SizeOfByte = 8;
  if (OffsetInBits > 0 || SizeInBits % SizeOfByte) {
    EmitOp(dwarf::DW_OP_bit_piece);
    EmitUnsigned(SizeInBits);
    EmitUnsigned(OffsetInBits);
  } else {
    EmitOp(dwarf::DW_OP_piece);
    unsigned ByteSize = SizeInBits / SizeOfByte;
    EmitUnsigned(ByteSize);
  }
}

void DwarfExpression::AddShr(unsigned ShiftBy) {
  EmitOp(dwarf::DW_OP_constu);
  EmitUnsigned(ShiftBy);
  EmitOp(dwarf::DW_OP_shr);
}

bool DwarfExpression::AddMachineRegIndirect(const TargetRegisterInfo &TRI,
                                            unsigned MachineReg, int Offset) {
  if (isFrameRegister(TRI, MachineReg)) {
    // If variable offset is based in frame register then use fbreg.
    EmitOp(dwarf::DW_OP_fbreg);
    EmitSigned(Offset);
    return true;
  }

  int DwarfReg = TRI.getDwarfRegNum(MachineReg, false);
  if (DwarfReg < 0)
    return false;

  AddRegIndirect(DwarfReg, Offset);
  return true;
}

bool DwarfExpression::AddMachineRegPiece(const TargetRegisterInfo &TRI,
                                         unsigned MachineReg,
                                         unsigned PieceSizeInBits,
                                         unsigned PieceOffsetInBits) {
  if (!TRI.isPhysicalRegister(MachineReg))
    return false;

  int Reg = TRI.getDwarfRegNum(MachineReg, false);

  // If this is a valid register number, emit it.
  if (Reg >= 0) {
    AddReg(Reg);
    if (PieceSizeInBits)
      AddOpPiece(PieceSizeInBits, PieceOffsetInBits);
    return true;
  }

  // Walk up the super-register chain until we find a valid number.
  // For example, EAX on x86_64 is a 32-bit piece of RAX with offset 0.
  for (MCSuperRegIterator SR(MachineReg, &TRI); SR.isValid(); ++SR) {
    Reg = TRI.getDwarfRegNum(*SR, false);
    if (Reg >= 0) {
      unsigned Idx = TRI.getSubRegIndex(*SR, MachineReg);
      unsigned Size = TRI.getSubRegIdxSize(Idx);
      unsigned RegOffset = TRI.getSubRegIdxOffset(Idx);
      AddReg(Reg, "super-register");
      if (PieceOffsetInBits == RegOffset) {
        AddOpPiece(Size, RegOffset);
      } else {
        // If this is part of a variable in a sub-register at a
        // non-zero offset, we need to manually shift the value into
        // place, since the DW_OP_piece describes the part of the
        // variable, not the position of the subregister.
        if (RegOffset)
          AddShr(RegOffset);
        AddOpPiece(Size, PieceOffsetInBits);
      }
      return true;
    }
  }

  // Otherwise, attempt to find a covering set of sub-register numbers.
  // For example, Q0 on ARM is a composition of D0+D1.
  //
  // Keep track of the current position so we can emit the more
  // efficient DW_OP_piece.
  unsigned CurPos = PieceOffsetInBits;
  // The size of the register in bits, assuming 8 bits per byte.
  unsigned RegSize = TRI.getMinimalPhysRegClass(MachineReg)->getSize() * 8;
  // Keep track of the bits in the register we already emitted, so we
  // can avoid emitting redundant aliasing subregs.
  SmallBitVector Coverage(RegSize, false);
  for (MCSubRegIterator SR(MachineReg, &TRI); SR.isValid(); ++SR) {
    unsigned Idx = TRI.getSubRegIndex(MachineReg, *SR);
    unsigned Size = TRI.getSubRegIdxSize(Idx);
    unsigned Offset = TRI.getSubRegIdxOffset(Idx);
    Reg = TRI.getDwarfRegNum(*SR, false);

    // Intersection between the bits we already emitted and the bits
    // covered by this subregister.
    SmallBitVector Intersection(RegSize, false);
    Intersection.set(Offset, Offset + Size);
    Intersection ^= Coverage;

    // If this sub-register has a DWARF number and we haven't covered
    // its range, emit a DWARF piece for it.
    if (Reg >= 0 && Intersection.any()) {
      AddReg(Reg, "sub-register");
      AddOpPiece(Size, Offset == CurPos ? 0 : Offset);
      CurPos = Offset + Size;

      // Mark it as emitted.
      Coverage.set(Offset, Offset + Size);
    }
  }

  return CurPos > PieceOffsetInBits;
}

void DwarfExpression::AddStackValue() {
  if (DwarfVersion >= 4)
    EmitOp(dwarf::DW_OP_stack_value);
}

void DwarfExpression::AddSignedConstant(int64_t Value) {
  EmitOp(dwarf::DW_OP_consts);
  EmitSigned(Value);
  AddStackValue();
}

void DwarfExpression::AddUnsignedConstant(uint64_t Value) {
  EmitOp(dwarf::DW_OP_constu);
  EmitUnsigned(Value);
  AddStackValue();
}

void DwarfExpression::AddUnsignedConstant(const APInt &Value) {
  unsigned Size = Value.getBitWidth();
  const uint64_t *Data = Value.getRawData();

  // Chop it up into 64-bit pieces, because that's the maximum that
  // AddUnsignedConstant takes.
  unsigned Offset = 0;
  while (Offset < Size) {
    AddUnsignedConstant(*Data++);
    if (Offset == 0 && Size <= 64)
      break;
    AddOpPiece(std::min(Size-Offset, 64u), Offset);
    Offset += 64;
  }
}

static unsigned getOffsetOrZero(unsigned OffsetInBits,
                                unsigned PieceOffsetInBits) {
  if (OffsetInBits == PieceOffsetInBits)
    return 0;
  assert(OffsetInBits >= PieceOffsetInBits && "overlapping pieces");
  return OffsetInBits;
}

bool DwarfExpression::AddMachineRegExpression(const TargetRegisterInfo &TRI,
                                              const DIExpression *Expr,
                                              unsigned MachineReg,
                                              unsigned PieceOffsetInBits) {
  auto I = Expr->expr_op_begin();
  auto E = Expr->expr_op_end();
  if (I == E)
    return AddMachineRegPiece(TRI, MachineReg);

  // Pattern-match combinations for which more efficient representations exist
  // first.
  bool ValidReg = false;
  switch (I->getOp()) {
  case dwarf::DW_OP_bit_piece: {
    unsigned OffsetInBits = I->getArg(0);
    unsigned SizeInBits   = I->getArg(1);
    // Piece always comes at the end of the expression.
    return AddMachineRegPiece(TRI, MachineReg, SizeInBits,
               getOffsetOrZero(OffsetInBits, PieceOffsetInBits));
  }
  case dwarf::DW_OP_plus:
  case dwarf::DW_OP_minus: {
    // [DW_OP_reg,Offset,DW_OP_plus, DW_OP_deref] --> [DW_OP_breg, Offset].
    // [DW_OP_reg,Offset,DW_OP_minus,DW_OP_deref] --> [DW_OP_breg,-Offset].
    auto N = I.getNext();
    if (N != E && N->getOp() == dwarf::DW_OP_deref) {
      unsigned Offset = I->getArg(0);
      ValidReg = AddMachineRegIndirect(
          TRI, MachineReg, I->getOp() == dwarf::DW_OP_plus ? Offset : -Offset);
      std::advance(I, 2);
      break;
    } else
      ValidReg = AddMachineRegPiece(TRI, MachineReg);
  }
  case dwarf::DW_OP_deref: {
      // [DW_OP_reg,DW_OP_deref] --> [DW_OP_breg].
      ValidReg = AddMachineRegIndirect(TRI, MachineReg);
      ++I;
      break;
  }
  default:
    llvm_unreachable("unsupported operand");
  }

  if (!ValidReg)
    return false;

  // Emit remaining elements of the expression.
  AddExpression(I, E, PieceOffsetInBits);
  return true;
}

void DwarfExpression::AddExpression(DIExpression::expr_op_iterator I,
                                    DIExpression::expr_op_iterator E,
                                    unsigned PieceOffsetInBits) {
  for (; I != E; ++I) {
    switch (I->getOp()) {
    case dwarf::DW_OP_bit_piece: {
      unsigned OffsetInBits = I->getArg(0);
      unsigned SizeInBits   = I->getArg(1);
      AddOpPiece(SizeInBits, getOffsetOrZero(OffsetInBits, PieceOffsetInBits));
      break;
    }
    case dwarf::DW_OP_plus:
      EmitOp(dwarf::DW_OP_plus_uconst);
      EmitUnsigned(I->getArg(0));
      break;
    case dwarf::DW_OP_minus:
      // There is no OP_minus_uconst.
      EmitOp(dwarf::DW_OP_constu);
      EmitUnsigned(I->getArg(0));
      EmitOp(dwarf::DW_OP_minus);
      break;
    case dwarf::DW_OP_deref:
      EmitOp(dwarf::DW_OP_deref);
      break;
    default:
      llvm_unreachable("unhandled opcode found in expression");
    }
  }
}
