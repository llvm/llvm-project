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
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "amdgpu-inline-asm-lowering"

using namespace llvm;

/// Build a readfirstlane of the same type as \p Src and return its result.
/// This is not AMDGPU::buildReadFirstLane, that one takes the SGPR class
/// destination register and the register bank info, and both are chosen here by
/// the inline asm operand rather than by the caller.
static Register buildReadFirstLaneOfSameType(MachineIRBuilder &MIRBuilder,
                                             Register Src) {
  LLT Ty = MIRBuilder.getMRI()->getType(Src);
  return MIRBuilder.buildIntrinsic(Intrinsic::amdgcn_readfirstlane, {Ty})
      .addReg(Src)
      .getReg(0);
}

// Copying a divergent value into an SGPR is illegal, so an inline asm operand
// with an SGPR constraint has to go through v_readfirstlane. This mirrors
// SITargetLowering::LowerINLINEASM, which inserts the intrinsic for the
// divergent values it knows about before the copy into the constraint register.
// The lowering here runs before there is any divergence information, so the
// readfirstlane is inserted unconditionally; AMDGPUGlobalISelDivergenceLowering
// drops the ones whose operand is proven uniform.
Register AMDGPUInlineAsmLowering::lowerAsmInputValue(
    Register Dst, Register Src, MachineIRBuilder &MIRBuilder) const {
  MachineRegisterInfo *MRI = MIRBuilder.getMRI();
  const TargetRegisterInfo *TRI =
      MIRBuilder.getMF().getSubtarget().getRegisterInfo();

  // A constraint like ${s[8:11]} selects physical registers, Dst is virtual
  // only when the constraint named a register class (the "s" case).
  const TargetRegisterClass *DstRC =
      Dst.isVirtual() ? MRI->getRegClass(Dst) : TRI->getPhysRegBaseClass(Dst);
  if (!DstRC || !SIRegisterInfo::isSGPRClass(DstRC))
    return Src;

  LLT SrcTy = MRI->getType(Src);
  if (!SrcTy.isValid())
    return Src;

  // Constants are uniform already; the divergence lowering would fold the
  // readfirstlane away again, so do not build it.
  if (MachineInstr *Def = Src.isVirtual() ? MRI->getVRegDef(Src) : nullptr;
      Def && (Def->getOpcode() == TargetOpcode::G_CONSTANT ||
              Def->getOpcode() == TargetOpcode::G_FCONSTANT))
    return Src;

  // A vector of pointers is a bag of bits like any other operand, read it as a
  // vector of integers of the element size.
  if (SrcTy.isVector() && SrcTy.getElementType().isPointer()) {
    LLT IntTy =
        LLT::fixed_vector(SrcTy.getNumElements(),
                          LLT::integer(SrcTy.getElementType().getSizeInBits()));
    SrcTy = IntTy;
    Src = MIRBuilder.buildPtrToInt(IntTy, Src).getReg(0);
  }

  // The operand only carries bits, so a pointer is read as an integer of the
  // same size.
  if (SrcTy.isPointer()) {
    SrcTy = LLT::integer(SrcTy.getSizeInBits());
    Src = MIRBuilder.buildPtrToInt(SrcTy, Src).getReg(0);
  }
  const LLT ValueTy = SrcTy;

  const unsigned Size = SrcTy.getSizeInBits();
  // v_readfirstlane_b32 reads a single dword, narrower values are extended to
  // one first.
  if (Size < 32) {
    if (!SrcTy.isInteger())
      Src = MIRBuilder.buildBitcast(LLT::integer(Size), Src).getReg(0);
    Src = MIRBuilder.buildAnyExt(LLT::integer(32), Src).getReg(0);
    return buildReadFirstLaneOfSameType(MIRBuilder, Src);
  }

  if (Size == 32)
    return buildReadFirstLaneOfSameType(MIRBuilder, Src);

  // Types of an odd number of dwords cannot be read with a dword readfirstlane
  // and there is no vector of them. The inline asm constraint lookup rejects
  // such an operand before reaching here, so this is only a safety net.
  if (Size % 32 != 0) {
    LLVM_DEBUG(dbgs() << "Unhandled inline asm operand size " << Size << '\n');
    return Src;
  }

  const unsigned NumPieces = Size / 32;
  LLT ElemTy = SrcTy.isVector() ? SrcTy.getElementType() : SrcTy;
  // Keep dword elements as they are, so the common vector cases need no
  // bitcasts; otherwise read the raw dword sized bits.
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
