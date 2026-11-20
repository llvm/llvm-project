//===-- AMDGPUInlineAsmLowering.cpp - Inline asm lowering -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUInlineAsmLowering.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"

using namespace llvm;

/// Build a readfirstlane of the same type as \p Src.
static Register buildReadFirstLaneOfSameType(MachineIRBuilder &MIRBuilder,
                                             Register Src) {
  LLT Ty = MIRBuilder.getMRI()->getType(Src);
  return MIRBuilder.buildIntrinsic(Intrinsic::amdgcn_readfirstlane, {Ty})
      .addReg(Src)
      .getReg(0);
}

// Copying a divergent value into an SGPR is illegal, so an SGPR-constrained
// inline asm operand has to go through v_readfirstlane. There is no divergence
// information yet, so insert it speculatively; the divergence lowering drops
// the ones that turn out uniform.
Register AMDGPUInlineAsmLowering::lowerAsmInputValue(
    Register Dst, Register Src, MachineIRBuilder &MIRBuilder) const {
  MachineRegisterInfo *MRI = MIRBuilder.getMRI();
  const TargetRegisterInfo *TRI =
      MIRBuilder.getMF().getSubtarget().getRegisterInfo();

  // Dst is virtual for class constraints ("s") and physical for ${s[8:11]}.
  const TargetRegisterClass *DstRC =
      Dst.isVirtual() ? MRI->getRegClass(Dst) : TRI->getPhysRegBaseClass(Dst);
  if (!DstRC || !SIRegisterInfo::isSGPRClass(DstRC))
    return Src;

  LLT SrcTy = MRI->getType(Src);
  if (!SrcTy.isValid())
    return Src;

  // A constant is uniform already; do not create a readfirstlane for it.
  if (MachineInstr *Def = Src.isVirtual() ? MRI->getVRegDef(Src) : nullptr;
      Def && (Def->getOpcode() == TargetOpcode::G_CONSTANT ||
              Def->getOpcode() == TargetOpcode::G_FCONSTANT))
    return Src;

  // The operand only carries bits, so read pointers as integers.
  if (SrcTy.isPointer() ||
      (SrcTy.isVector() && SrcTy.getElementType().isPointer())) {
    LLT IntTy = SrcTy.isVector()
                    ? LLT::fixed_vector(
                          SrcTy.getNumElements(),
                          LLT::integer(SrcTy.getElementType().getSizeInBits()))
                    : LLT::integer(SrcTy.getSizeInBits());
    Src = MIRBuilder.buildPtrToInt(IntTy, Src).getReg(0);
    SrcTy = IntTy;
  }
  const LLT ValueTy = SrcTy;
  const unsigned Size = SrcTy.getSizeInBits();

  // v_readfirstlane_b32 reads one dword, extend narrower values first.
  if (Size < 32) {
    if (!SrcTy.isInteger())
      Src = MIRBuilder.buildBitcast(LLT::integer(Size), Src).getReg(0);
    Src = MIRBuilder.buildAnyExt(LLT::integer(32), Src).getReg(0);
    return buildReadFirstLaneOfSameType(MIRBuilder, Src);
  }
  if (Size == 32)
    return buildReadFirstLaneOfSameType(MIRBuilder, Src);

  // The constraint lookup rejects operands that are not whole dwords.
  assert(Size % 32 == 0 && "unexpected inline asm operand size");

  // Read one dword at a time and reassemble.
  const unsigned NumPieces = Size / 32;
  LLT ElemTy = SrcTy.isVector() ? SrcTy.getElementType() : SrcTy;
  LLT PieceTy = ElemTy.getSizeInBits() == 32 ? ElemTy : LLT::integer(32);
  LLT PiecesTy = LLT::fixed_vector(NumPieces, PieceTy);
  if (SrcTy != PiecesTy)
    Src = MIRBuilder.buildBitcast(PiecesTy, Src).getReg(0);

  SmallVector<Register, 8> Pieces;
  Pieces.reserve(NumPieces);
  for (unsigned I = 0; I != NumPieces; ++I)
    Pieces.push_back(MRI->createGenericVirtualRegister(PieceTy));
  MIRBuilder.buildUnmerge(Pieces, Src);

  for (Register &Piece : Pieces)
    Piece = buildReadFirstLaneOfSameType(MIRBuilder, Piece);

  Register Vec = MIRBuilder.buildBuildVector(PiecesTy, Pieces).getReg(0);
  if (MRI->getType(Vec) == ValueTy)
    return Vec;
  return MIRBuilder.buildBitcast(ValueTy, Vec).getReg(0);
}
