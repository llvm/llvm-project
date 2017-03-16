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

void DwarfExpression::addReg(int DwarfReg, const char *Comment) {
  assert(DwarfReg >= 0 && "invalid negative dwarf register number");
  if (DwarfReg < 32) {
    emitOp(dwarf::DW_OP_reg0 + DwarfReg, Comment);
  } else {
    emitOp(dwarf::DW_OP_regx, Comment);
    emitUnsigned(DwarfReg);
  }
}

void DwarfExpression::addRegIndirect(int DwarfReg, int Offset) {
  assert(DwarfReg >= 0 && "invalid negative dwarf register number");
  if (DwarfReg < 32) {
    emitOp(dwarf::DW_OP_breg0 + DwarfReg);
  } else {
    emitOp(dwarf::DW_OP_bregx);
    emitUnsigned(DwarfReg);
  }
  emitSigned(Offset);
}

void DwarfExpression::addOpPiece(unsigned SizeInBits, unsigned OffsetInBits) {
  if (!SizeInBits)
    return;

  const unsigned SizeOfByte = 8;
  if (OffsetInBits > 0 || SizeInBits % SizeOfByte) {
    emitOp(dwarf::DW_OP_bit_piece);
    emitUnsigned(SizeInBits);
    emitUnsigned(OffsetInBits);
  } else {
    emitOp(dwarf::DW_OP_piece);
    unsigned ByteSize = SizeInBits / SizeOfByte;
    emitUnsigned(ByteSize);
  }
  this->OffsetInBits += SizeInBits;
}

void DwarfExpression::addShr(unsigned ShiftBy) {
  emitOp(dwarf::DW_OP_constu);
  emitUnsigned(ShiftBy);
  emitOp(dwarf::DW_OP_shr);
}

void DwarfExpression::addAnd(unsigned Mask) {
  emitOp(dwarf::DW_OP_constu);
  emitUnsigned(Mask);
  emitOp(dwarf::DW_OP_and);
}

bool DwarfExpression::addMachineRegIndirect(const TargetRegisterInfo &TRI,
                                            unsigned MachineReg, int Offset) {
  if (isFrameRegister(TRI, MachineReg)) {
    // If variable offset is based in frame register then use fbreg.
    emitOp(dwarf::DW_OP_fbreg);
    emitSigned(Offset);
    return true;
  }

  int DwarfReg = TRI.getDwarfRegNum(MachineReg, false);
  if (DwarfReg < 0)
    return false;

  addRegIndirect(DwarfReg, Offset);
  return true;
}

bool DwarfExpression::addMachineReg(const TargetRegisterInfo &TRI,
                                    unsigned MachineReg, unsigned MaxSize) {
  if (!TRI.isPhysicalRegister(MachineReg))
    return false;

  int Reg = TRI.getDwarfRegNum(MachineReg, false);

  // If this is a valid register number, emit it.
  if (Reg >= 0) {
    addReg(Reg);
    return true;
  }

  // Walk up the super-register chain until we find a valid number.
  // For example, EAX on x86_64 is a 32-bit fragment of RAX with offset 0.
  for (MCSuperRegIterator SR(MachineReg, &TRI); SR.isValid(); ++SR) {
    Reg = TRI.getDwarfRegNum(*SR, false);
    if (Reg >= 0) {
      unsigned Idx = TRI.getSubRegIndex(*SR, MachineReg);
      unsigned Size = TRI.getSubRegIdxSize(Idx);
      unsigned RegOffset = TRI.getSubRegIdxOffset(Idx);
      addReg(Reg, "super-register");
      // Use a DW_OP_bit_piece to describe the sub-register.
      setSubRegisterPiece(Size, RegOffset);
      return true;
    }
  }

  // Otherwise, attempt to find a covering set of sub-register numbers.
  // For example, Q0 on ARM is a composition of D0+D1.
  unsigned CurPos = 0;
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
      addReg(Reg, "sub-register");
      if (Offset >= MaxSize)
	break;
      // emit a piece for the any gap in the coverage.
      if (Offset > CurPos)
        addOpPiece(Offset - CurPos);
      addOpPiece(std::min<unsigned>(Size, MaxSize - Offset));
      CurPos = Offset + Size;

      // Mark it as emitted.
      Coverage.set(Offset, Offset + Size);
    }
  }

  return CurPos;
}

void DwarfExpression::addStackValue() {
  if (DwarfVersion >= 4)
    emitOp(dwarf::DW_OP_stack_value);
}

void DwarfExpression::addSignedConstant(int64_t Value) {
  emitOp(dwarf::DW_OP_consts);
  emitSigned(Value);
  addStackValue();
}

void DwarfExpression::addUnsignedConstant(uint64_t Value) {
  emitOp(dwarf::DW_OP_constu);
  emitUnsigned(Value);
  addStackValue();
}

void DwarfExpression::addUnsignedConstant(const APInt &Value) {
  unsigned Size = Value.getBitWidth();
  const uint64_t *Data = Value.getRawData();

  // Chop it up into 64-bit pieces, because that's the maximum that
  // addUnsignedConstant takes.
  unsigned Offset = 0;
  while (Offset < Size) {
    addUnsignedConstant(*Data++);
    if (Offset == 0 && Size <= 64)
      break;
    addOpPiece(std::min(Size-Offset, 64u), Offset);
    Offset += 64;
  }
}

bool DwarfExpression::addMachineRegExpression(const TargetRegisterInfo &TRI,
                                              DIExpressionCursor &ExprCursor,
                                              unsigned MachineReg,
                                              unsigned FragmentOffsetInBits) {
  if (!ExprCursor)
    return addMachineReg(TRI, MachineReg);

  // Pattern-match combinations for which more efficient representations exist
  // first.
  bool ValidReg = false;
  auto Op = ExprCursor.peek();
  switch (Op->getOp()) {
  default: {
    auto Fragment = ExprCursor.getFragmentInfo();
    ValidReg = addMachineReg(TRI, MachineReg,
			     Fragment ? Fragment->SizeInBits : ~1U);
    break;
  }
  case dwarf::DW_OP_plus:
  case dwarf::DW_OP_minus: {
    // [DW_OP_reg,Offset,DW_OP_plus, DW_OP_deref] --> [DW_OP_breg, Offset].
    // [DW_OP_reg,Offset,DW_OP_minus,DW_OP_deref] --> [DW_OP_breg,-Offset].
    auto N = ExprCursor.peekNext();
    if (N && N->getOp() == dwarf::DW_OP_deref) {
      unsigned Offset = Op->getArg(0);
      ValidReg = addMachineRegIndirect(
          TRI, MachineReg, Op->getOp() == dwarf::DW_OP_plus ? Offset : -Offset);
      ExprCursor.consume(2);
    } else
      ValidReg = addMachineReg(TRI, MachineReg);
    break;
  }
  case dwarf::DW_OP_deref:
    // [DW_OP_reg,DW_OP_deref] --> [DW_OP_breg].
    ValidReg = addMachineRegIndirect(TRI, MachineReg);
    ExprCursor.take();
    break;
  }

  return ValidReg;
}

void DwarfExpression::addExpression(DIExpressionCursor &&ExprCursor,
                                    unsigned FragmentOffsetInBits) {
  while (ExprCursor) {
    auto Op = ExprCursor.take();

    // If we need to mask out a subregister, do it now, unless the next
    // operation would emit an OpPiece anyway.
    if (SubRegisterSizeInBits && Op->getOp() != dwarf::DW_OP_LLVM_fragment)
      maskSubRegister();

    switch (Op->getOp()) {
    case dwarf::DW_OP_LLVM_fragment: {
      unsigned SizeInBits = Op->getArg(1);
      unsigned FragmentOffset = Op->getArg(0);
      // The fragment offset must have already been adjusted by emitting an
      // empty DW_OP_piece / DW_OP_bit_piece before we emitted the base
      // location.
      assert(OffsetInBits >= FragmentOffset && "fragment offset not added?");

      // If \a addMachineReg already emitted DW_OP_piece operations to represent
      // a super-register by splicing together sub-registers, subtract the size
      // of the pieces that was already emitted.
      SizeInBits -= OffsetInBits - FragmentOffset;

      // If \a addMachineReg requested a DW_OP_bit_piece to stencil out a
      // sub-register that is smaller than the current fragment's size, use it.
      if (SubRegisterSizeInBits)
        SizeInBits = std::min<unsigned>(SizeInBits, SubRegisterSizeInBits);
      
      addOpPiece(SizeInBits, SubRegisterOffsetInBits);
      setSubRegisterPiece(0, 0);
      break;
    }
    case dwarf::DW_OP_plus:
      emitOp(dwarf::DW_OP_plus_uconst);
      emitUnsigned(Op->getArg(0));
      break;
    case dwarf::DW_OP_minus:
      // There is no OP_minus_uconst.
      emitOp(dwarf::DW_OP_constu);
      emitUnsigned(Op->getArg(0));
      emitOp(dwarf::DW_OP_minus);
      break;
    case dwarf::DW_OP_deref:
      emitOp(dwarf::DW_OP_deref);
      break;
    case dwarf::DW_OP_constu:
      emitOp(dwarf::DW_OP_constu);
      emitUnsigned(Op->getArg(0));
      break;
    case dwarf::DW_OP_stack_value:
      addStackValue();
      break;
    case dwarf::DW_OP_swap:
      emitOp(dwarf::DW_OP_swap);
      break;
    case dwarf::DW_OP_xderef:
      emitOp(dwarf::DW_OP_xderef);
      break;
    default:
      llvm_unreachable("unhandled opcode found in expression");
    }
  }
}

/// add masking operations to stencil out a subregister.
void DwarfExpression::maskSubRegister() {
  assert(SubRegisterSizeInBits && "no subregister was registered");
  if (SubRegisterOffsetInBits > 0)
    addShr(SubRegisterOffsetInBits);
  uint64_t Mask = (1UL << SubRegisterSizeInBits) - 1;
  addAnd(Mask);
}


void DwarfExpression::finalize() {
  // Emit any outstanding DW_OP_piece operations to mask out subregisters.
  if (SubRegisterSizeInBits == 0)
    return;
  // Don't emit a DW_OP_piece for a subregister at offset 0.
  if (SubRegisterOffsetInBits == 0)
    return;
  addOpPiece(SubRegisterSizeInBits, SubRegisterOffsetInBits);
}

void DwarfExpression::addFragmentOffset(const DIExpression *Expr) {
  if (!Expr || !Expr->isFragment())
    return;

  uint64_t FragmentOffset = Expr->getFragmentInfo()->OffsetInBits;
  assert(FragmentOffset >= OffsetInBits &&
         "overlapping or duplicate fragments");
  if (FragmentOffset > OffsetInBits)
    addOpPiece(FragmentOffset - OffsetInBits);
  OffsetInBits = FragmentOffset;
}
