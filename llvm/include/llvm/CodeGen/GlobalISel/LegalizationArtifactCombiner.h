//===-- llvm/CodeGen/GlobalISel/LegalizationArtifactCombiner.h -----*- C++ -*-//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file contains some helper functions which try to cleanup artifacts
// such as G_TRUNCs/G_[ZSA]EXTENDS that were created during legalization to make
// the types match. This file also contains some combines of merges that happens
// at the end of the legalization.
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GlobalISel/Legalizer.h"
#include "llvm/CodeGen/GlobalISel/LegalizerInfo.h"
#include "llvm/CodeGen/GlobalISel/MIPatternMatch.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "legalizer"
using namespace llvm::MIPatternMatch;

namespace llvm {
class LegalizationArtifactCombiner {
  MachineIRBuilder &Builder;
  MachineRegisterInfo &MRI;
  const LegalizerInfo &LI;

  static bool isArtifactCast(unsigned Opc) {
    switch (Opc) {
    case TargetOpcode::G_TRUNC:
    case TargetOpcode::G_SEXT:
    case TargetOpcode::G_ZEXT:
    case TargetOpcode::G_ANYEXT:
      return true;
    default:
      return false;
    }
  }

public:
  LegalizationArtifactCombiner(MachineIRBuilder &B, MachineRegisterInfo &MRI,
                    const LegalizerInfo &LI)
      : Builder(B), MRI(MRI), LI(LI) {}

  bool tryCombineAnyExt(MachineInstr &MI,
                        SmallVectorImpl<MachineInstr *> &DeadInsts,
                        SmallVectorImpl<Register> &UpdatedDefs) {
    assert(MI.getOpcode() == TargetOpcode::G_ANYEXT);

    Builder.setInstrAndDebugLoc(MI);
    Register DstReg = MI.getOperand(0).getReg();
    Register SrcReg = lookThroughCopyInstrs(MI.getOperand(1).getReg());

    // aext(trunc x) - > aext/copy/trunc x
    Register TruncSrc;
    if (mi_match(SrcReg, MRI, m_GTrunc(m_Reg(TruncSrc)))) {
      LLVM_DEBUG(dbgs() << ".. Combine MI: " << MI;);
      Builder.buildAnyExtOrTrunc(DstReg, TruncSrc);
      UpdatedDefs.push_back(DstReg);
      markInstAndDefDead(MI, *MRI.getVRegDef(SrcReg), DeadInsts);
      return true;
    }

    // aext([asz]ext x) -> [asz]ext x
    Register ExtSrc;
    MachineInstr *ExtMI;
    if (mi_match(SrcReg, MRI,
                 m_all_of(m_MInstr(ExtMI), m_any_of(m_GAnyExt(m_Reg(ExtSrc)),
                                                    m_GSExt(m_Reg(ExtSrc)),
                                                    m_GZExt(m_Reg(ExtSrc)))))) {
      Builder.buildInstr(ExtMI->getOpcode(), {DstReg}, {ExtSrc});
      UpdatedDefs.push_back(DstReg);
      markInstAndDefDead(MI, *ExtMI, DeadInsts);
      return true;
    }

    // Try to fold aext(g_constant) when the larger constant type is legal.
    // Can't use MIPattern because we don't have a specific constant in mind.
    auto *SrcMI = MRI.getVRegDef(SrcReg);
    if (SrcMI->getOpcode() == TargetOpcode::G_CONSTANT) {
      const LLT DstTy = MRI.getType(DstReg);
      if (isInstLegal({TargetOpcode::G_CONSTANT, {DstTy}})) {
        auto &CstVal = SrcMI->getOperand(1);
        Builder.buildConstant(
            DstReg, CstVal.getCImm()->getValue().sext(DstTy.getSizeInBits()));
        UpdatedDefs.push_back(DstReg);
        markInstAndDefDead(MI, *SrcMI, DeadInsts);
        return true;
      }
    }
    return tryFoldImplicitDef(MI, DeadInsts, UpdatedDefs);
  }

  bool tryCombineZExt(MachineInstr &MI,
                      SmallVectorImpl<MachineInstr *> &DeadInsts,
                      SmallVectorImpl<Register> &UpdatedDefs,
                      GISelObserverWrapper &Observer) {
    assert(MI.getOpcode() == TargetOpcode::G_ZEXT);

    Builder.setInstrAndDebugLoc(MI);
    Register DstReg = MI.getOperand(0).getReg();
    Register SrcReg = lookThroughCopyInstrs(MI.getOperand(1).getReg());

    // zext(trunc x) - > and (aext/copy/trunc x), mask
    // zext(sext x) -> and (sext x), mask
    Register TruncSrc;
    Register SextSrc;
    if (mi_match(SrcReg, MRI, m_GTrunc(m_Reg(TruncSrc))) ||
        mi_match(SrcReg, MRI, m_GSExt(m_Reg(SextSrc)))) {
      LLT DstTy = MRI.getType(DstReg);
      if (isInstUnsupported({TargetOpcode::G_AND, {DstTy}}) ||
          isConstantUnsupported(DstTy))
        return false;
      LLVM_DEBUG(dbgs() << ".. Combine MI: " << MI;);
      LLT SrcTy = MRI.getType(SrcReg);
      APInt MaskVal = APInt::getAllOnesValue(SrcTy.getScalarSizeInBits());
      auto Mask = Builder.buildConstant(
        DstTy, MaskVal.zext(DstTy.getScalarSizeInBits()));
      auto Extended = SextSrc ? Builder.buildSExtOrTrunc(DstTy, SextSrc) :
                                Builder.buildAnyExtOrTrunc(DstTy, TruncSrc);
      Builder.buildAnd(DstReg, Extended, Mask);
      markInstAndDefDead(MI, *MRI.getVRegDef(SrcReg), DeadInsts);
      return true;
    }

    // zext(zext x) -> (zext x)
    Register ZextSrc;
    if (mi_match(SrcReg, MRI, m_GZExt(m_Reg(ZextSrc)))) {
      LLVM_DEBUG(dbgs() << ".. Combine MI: " << MI);
      Observer.changingInstr(MI);
      MI.getOperand(1).setReg(ZextSrc);
      Observer.changedInstr(MI);
      UpdatedDefs.push_back(DstReg);
      markDefDead(MI, *MRI.getVRegDef(SrcReg), DeadInsts);
      return true;
    }

    // Try to fold zext(g_constant) when the larger constant type is legal.
    // Can't use MIPattern because we don't have a specific constant in mind.
    auto *SrcMI = MRI.getVRegDef(SrcReg);
    if (SrcMI->getOpcode() == TargetOpcode::G_CONSTANT) {
      const LLT DstTy = MRI.getType(DstReg);
      if (isInstLegal({TargetOpcode::G_CONSTANT, {DstTy}})) {
        auto &CstVal = SrcMI->getOperand(1);
        Builder.buildConstant(
            DstReg, CstVal.getCImm()->getValue().zext(DstTy.getSizeInBits()));
        UpdatedDefs.push_back(DstReg);
        markInstAndDefDead(MI, *SrcMI, DeadInsts);
        return true;
      }
    }
    return tryFoldImplicitDef(MI, DeadInsts, UpdatedDefs);
  }

  bool tryCombineSExt(MachineInstr &MI,
                      SmallVectorImpl<MachineInstr *> &DeadInsts,
                      SmallVectorImpl<Register> &UpdatedDefs) {
    assert(MI.getOpcode() == TargetOpcode::G_SEXT);

    Builder.setInstrAndDebugLoc(MI);
    Register DstReg = MI.getOperand(0).getReg();
    Register SrcReg = lookThroughCopyInstrs(MI.getOperand(1).getReg());

    // sext(trunc x) - > (sext_inreg (aext/copy/trunc x), c)
    Register TruncSrc;
    if (mi_match(SrcReg, MRI, m_GTrunc(m_Reg(TruncSrc)))) {
      LLT DstTy = MRI.getType(DstReg);
      if (isInstUnsupported({TargetOpcode::G_SEXT_INREG, {DstTy}}))
        return false;
      LLVM_DEBUG(dbgs() << ".. Combine MI: " << MI;);
      LLT SrcTy = MRI.getType(SrcReg);
      uint64_t SizeInBits = SrcTy.getScalarSizeInBits();
      Builder.buildInstr(
          TargetOpcode::G_SEXT_INREG, {DstReg},
          {Builder.buildAnyExtOrTrunc(DstTy, TruncSrc), SizeInBits});
      markInstAndDefDead(MI, *MRI.getVRegDef(SrcReg), DeadInsts);
      return true;
    }

    // sext(zext x) -> (zext x)
    // sext(sext x) -> (sext x)
    Register ExtSrc;
    MachineInstr *ExtMI;
    if (mi_match(SrcReg, MRI,
                 m_all_of(m_MInstr(ExtMI), m_any_of(m_GZExt(m_Reg(ExtSrc)),
                                                    m_GSExt(m_Reg(ExtSrc)))))) {
      LLVM_DEBUG(dbgs() << ".. Combine MI: " << MI);
      Builder.buildInstr(ExtMI->getOpcode(), {DstReg}, {ExtSrc});
      UpdatedDefs.push_back(DstReg);
      markInstAndDefDead(MI, *MRI.getVRegDef(SrcReg), DeadInsts);
      return true;
    }

    return tryFoldImplicitDef(MI, DeadInsts, UpdatedDefs);
  }

  bool tryCombineTrunc(MachineInstr &MI,
                       SmallVectorImpl<MachineInstr *> &DeadInsts,
                       SmallVectorImpl<Register> &UpdatedDefs,
                       GISelObserverWrapper &Observer) {
    assert(MI.getOpcode() == TargetOpcode::G_TRUNC);

    Builder.setInstr(MI);
    Register DstReg = MI.getOperand(0).getReg();
    Register SrcReg = lookThroughCopyInstrs(MI.getOperand(1).getReg());

    // Try to fold trunc(g_constant) when the smaller constant type is legal.
    // Can't use MIPattern because we don't have a specific constant in mind.
    auto *SrcMI = MRI.getVRegDef(SrcReg);
    if (SrcMI->getOpcode() == TargetOpcode::G_CONSTANT) {
      const LLT DstTy = MRI.getType(DstReg);
      if (isInstLegal({TargetOpcode::G_CONSTANT, {DstTy}})) {
        auto &CstVal = SrcMI->getOperand(1);
        Builder.buildConstant(
            DstReg, CstVal.getCImm()->getValue().trunc(DstTy.getSizeInBits()));
        UpdatedDefs.push_back(DstReg);
        markInstAndDefDead(MI, *SrcMI, DeadInsts);
        return true;
      }
    }

    // Try to fold trunc(merge) to directly use the source of the merge.
    // This gets rid of large, difficult to legalize, merges
    if (SrcMI->getOpcode() == TargetOpcode::G_MERGE_VALUES) {
      const Register MergeSrcReg = SrcMI->getOperand(1).getReg();
      const LLT MergeSrcTy = MRI.getType(MergeSrcReg);
      const LLT DstTy = MRI.getType(DstReg);

      // We can only fold if the types are scalar
      const unsigned DstSize = DstTy.getSizeInBits();
      const unsigned MergeSrcSize = MergeSrcTy.getSizeInBits();
      if (!DstTy.isScalar() || !MergeSrcTy.isScalar())
        return false;

      if (DstSize < MergeSrcSize) {
        // When the merge source is larger than the destination, we can just
        // truncate the merge source directly
        if (isInstUnsupported({TargetOpcode::G_TRUNC, {DstTy, MergeSrcTy}}))
          return false;

        LLVM_DEBUG(dbgs() << "Combining G_TRUNC(G_MERGE_VALUES) to G_TRUNC: "
                          << MI);

        Builder.buildTrunc(DstReg, MergeSrcReg);
        UpdatedDefs.push_back(DstReg);
      } else if (DstSize == MergeSrcSize) {
        // If the sizes match we can simply try to replace the register
        LLVM_DEBUG(
            dbgs() << "Replacing G_TRUNC(G_MERGE_VALUES) with merge input: "
                   << MI);
        replaceRegOrBuildCopy(DstReg, MergeSrcReg, MRI, Builder, UpdatedDefs,
                              Observer);
      } else if (DstSize % MergeSrcSize == 0) {
        // If the trunc size is a multiple of the merge source size we can use
        // a smaller merge instead
        if (isInstUnsupported(
                {TargetOpcode::G_MERGE_VALUES, {DstTy, MergeSrcTy}}))
          return false;

        LLVM_DEBUG(
            dbgs() << "Combining G_TRUNC(G_MERGE_VALUES) to G_MERGE_VALUES: "
                   << MI);

        const unsigned NumSrcs = DstSize / MergeSrcSize;
        assert(NumSrcs < SrcMI->getNumOperands() - 1 &&
               "trunc(merge) should require less inputs than merge");
        SmallVector<Register, 8> SrcRegs(NumSrcs);
        for (unsigned i = 0; i < NumSrcs; ++i)
          SrcRegs[i] = SrcMI->getOperand(i + 1).getReg();

        Builder.buildMerge(DstReg, SrcRegs);
        UpdatedDefs.push_back(DstReg);
      } else {
        // Unable to combine
        return false;
      }

      markInstAndDefDead(MI, *SrcMI, DeadInsts);
      return true;
    }

    // trunc(trunc) -> trunc
    Register TruncSrc;
    if (mi_match(SrcReg, MRI, m_GTrunc(m_Reg(TruncSrc)))) {
      // Always combine trunc(trunc) since the eventual resulting trunc must be
      // legal anyway as it must be legal for all outputs of the consumer type
      // set.
      LLVM_DEBUG(dbgs() << ".. Combine G_TRUNC(G_TRUNC): " << MI);

      Builder.buildTrunc(DstReg, TruncSrc);
      UpdatedDefs.push_back(DstReg);
      markInstAndDefDead(MI, *MRI.getVRegDef(TruncSrc), DeadInsts);
      return true;
    }

    return false;
  }

  /// Try to fold G_[ASZ]EXT (G_IMPLICIT_DEF).
  bool tryFoldImplicitDef(MachineInstr &MI,
                          SmallVectorImpl<MachineInstr *> &DeadInsts,
                          SmallVectorImpl<Register> &UpdatedDefs) {
    unsigned Opcode = MI.getOpcode();
    assert(Opcode == TargetOpcode::G_ANYEXT || Opcode == TargetOpcode::G_ZEXT ||
           Opcode == TargetOpcode::G_SEXT);

    if (MachineInstr *DefMI = getOpcodeDef(TargetOpcode::G_IMPLICIT_DEF,
                                           MI.getOperand(1).getReg(), MRI)) {
      Builder.setInstr(MI);
      Register DstReg = MI.getOperand(0).getReg();
      LLT DstTy = MRI.getType(DstReg);

      if (Opcode == TargetOpcode::G_ANYEXT) {
        // G_ANYEXT (G_IMPLICIT_DEF) -> G_IMPLICIT_DEF
        if (!isInstLegal({TargetOpcode::G_IMPLICIT_DEF, {DstTy}}))
          return false;
        LLVM_DEBUG(dbgs() << ".. Combine G_ANYEXT(G_IMPLICIT_DEF): " << MI;);
        Builder.buildInstr(TargetOpcode::G_IMPLICIT_DEF, {DstReg}, {});
        UpdatedDefs.push_back(DstReg);
      } else {
        // G_[SZ]EXT (G_IMPLICIT_DEF) -> G_CONSTANT 0 because the top
        // bits will be 0 for G_ZEXT and 0/1 for the G_SEXT.
        if (isConstantUnsupported(DstTy))
          return false;
        LLVM_DEBUG(dbgs() << ".. Combine G_[SZ]EXT(G_IMPLICIT_DEF): " << MI;);
        Builder.buildConstant(DstReg, 0);
        UpdatedDefs.push_back(DstReg);
      }

      markInstAndDefDead(MI, *DefMI, DeadInsts);
      return true;
    }
    return false;
  }

  bool tryFoldUnmergeCast(MachineInstr &MI, MachineInstr &CastMI,
                          SmallVectorImpl<MachineInstr *> &DeadInsts,
                          SmallVectorImpl<Register> &UpdatedDefs) {

    assert(MI.getOpcode() == TargetOpcode::G_UNMERGE_VALUES);

    const unsigned CastOpc = CastMI.getOpcode();

    if (!isArtifactCast(CastOpc))
      return false;

    const unsigned NumDefs = MI.getNumOperands() - 1;

    const Register CastSrcReg = CastMI.getOperand(1).getReg();
    const LLT CastSrcTy = MRI.getType(CastSrcReg);
    const LLT DestTy = MRI.getType(MI.getOperand(0).getReg());
    const LLT SrcTy = MRI.getType(MI.getOperand(NumDefs).getReg());

    const unsigned CastSrcSize = CastSrcTy.getSizeInBits();
    const unsigned DestSize = DestTy.getSizeInBits();

    if (CastOpc == TargetOpcode::G_TRUNC) {
      if (SrcTy.isVector() && SrcTy.getScalarType() == DestTy.getScalarType()) {
        //  %1:_(<4 x s8>) = G_TRUNC %0(<4 x s32>)
        //  %2:_(s8), %3:_(s8), %4:_(s8), %5:_(s8) = G_UNMERGE_VALUES %1
        // =>
        //  %6:_(s32), %7:_(s32), %8:_(s32), %9:_(s32) = G_UNMERGE_VALUES %0
        //  %2:_(s8) = G_TRUNC %6
        //  %3:_(s8) = G_TRUNC %7
        //  %4:_(s8) = G_TRUNC %8
        //  %5:_(s8) = G_TRUNC %9

        unsigned UnmergeNumElts =
            DestTy.isVector() ? CastSrcTy.getNumElements() / NumDefs : 1;
        LLT UnmergeTy = CastSrcTy.changeNumElements(UnmergeNumElts);

        if (isInstUnsupported(
                {TargetOpcode::G_UNMERGE_VALUES, {UnmergeTy, CastSrcTy}}))
          return false;

        Builder.setInstr(MI);
        auto NewUnmerge = Builder.buildUnmerge(UnmergeTy, CastSrcReg);

        for (unsigned I = 0; I != NumDefs; ++I) {
          Register DefReg = MI.getOperand(I).getReg();
          UpdatedDefs.push_back(DefReg);
          Builder.buildTrunc(DefReg, NewUnmerge.getReg(I));
        }

        markInstAndDefDead(MI, CastMI, DeadInsts);
        return true;
      }

      if (CastSrcTy.isScalar() && SrcTy.isScalar() && !DestTy.isVector()) {
        //  %1:_(s16) = G_TRUNC %0(s32)
        //  %2:_(s8), %3:_(s8) = G_UNMERGE_VALUES %1
        // =>
        //  %2:_(s8), %3:_(s8), %4:_(s8), %5:_(s8) = G_UNMERGE_VALUES %0

        // Unmerge(trunc) can be combined if the trunc source size is a multiple
        // of the unmerge destination size
        if (CastSrcSize % DestSize != 0)
          return false;

        // Check if the new unmerge is supported
        if (isInstUnsupported(
                {TargetOpcode::G_UNMERGE_VALUES, {DestTy, CastSrcTy}}))
          return false;

        // Gather the original destination registers and create new ones for the
        // unused bits
        const unsigned NewNumDefs = CastSrcSize / DestSize;
        SmallVector<Register, 8> DstRegs(NewNumDefs);
        for (unsigned Idx = 0; Idx < NewNumDefs; ++Idx) {
          if (Idx < NumDefs)
            DstRegs[Idx] = MI.getOperand(Idx).getReg();
          else
            DstRegs[Idx] = MRI.createGenericVirtualRegister(DestTy);
        }

        // Build new unmerge
        Builder.setInstr(MI);
        Builder.buildUnmerge(DstRegs, CastSrcReg);
        UpdatedDefs.append(DstRegs.begin(), DstRegs.begin() + NewNumDefs);
        markInstAndDefDead(MI, CastMI, DeadInsts);
        return true;
      }
    }

    // TODO: support combines with other casts as well
    return false;
  }

  static bool canFoldMergeOpcode(unsigned MergeOp, unsigned ConvertOp,
                                 LLT OpTy, LLT DestTy) {
    // Check if we found a definition that is like G_MERGE_VALUES.
    switch (MergeOp) {
    default:
      return false;
    case TargetOpcode::G_BUILD_VECTOR:
    case TargetOpcode::G_MERGE_VALUES:
      // The convert operation that we will need to insert is
      // going to convert the input of that type of instruction (scalar)
      // to the destination type (DestTy).
      // The conversion needs to stay in the same domain (scalar to scalar
      // and vector to vector), so if we were to allow to fold the merge
      // we would need to insert some bitcasts.
      // E.g.,
      // <2 x s16> = build_vector s16, s16
      // <2 x s32> = zext <2 x s16>
      // <2 x s16>, <2 x s16> = unmerge <2 x s32>
      //
      // As is the folding would produce:
      // <2 x s16> = zext s16  <-- scalar to vector
      // <2 x s16> = zext s16  <-- scalar to vector
      // Which is invalid.
      // Instead we would want to generate:
      // s32 = zext s16
      // <2 x s16> = bitcast s32
      // s32 = zext s16
      // <2 x s16> = bitcast s32
      //
      // That is not done yet.
      if (ConvertOp == 0)
        return true;
      return !DestTy.isVector() && OpTy.isVector();
    case TargetOpcode::G_CONCAT_VECTORS: {
      if (ConvertOp == 0)
        return true;
      if (!DestTy.isVector())
        return false;

      const unsigned OpEltSize = OpTy.getElementType().getSizeInBits();

      // Don't handle scalarization with a cast that isn't in the same
      // direction as the vector cast. This could be handled, but it would
      // require more intermediate unmerges.
      if (ConvertOp == TargetOpcode::G_TRUNC)
        return DestTy.getSizeInBits() <= OpEltSize;
      return DestTy.getSizeInBits() >= OpEltSize;
    }
    }
  }

  /// Try to replace DstReg with SrcReg or build a COPY instruction
  /// depending on the register constraints.
  static void replaceRegOrBuildCopy(Register DstReg, Register SrcReg,
                                    MachineRegisterInfo &MRI,
                                    MachineIRBuilder &Builder,
                                    SmallVectorImpl<Register> &UpdatedDefs,
                                    GISelObserverWrapper &Observer) {
    if (!llvm::canReplaceReg(DstReg, SrcReg, MRI)) {
      Builder.buildCopy(DstReg, SrcReg);
      UpdatedDefs.push_back(DstReg);
      return;
    }
    SmallVector<MachineInstr *, 4> UseMIs;
    // Get the users and notify the observer before replacing.
    for (auto &UseMI : MRI.use_instructions(DstReg)) {
      UseMIs.push_back(&UseMI);
      Observer.changingInstr(UseMI);
    }
    // Replace the registers.
    MRI.replaceRegWith(DstReg, SrcReg);
    UpdatedDefs.push_back(SrcReg);
    // Notify the observer that we changed the instructions.
    for (auto *UseMI : UseMIs)
      Observer.changedInstr(*UseMI);
  }

  /// Return the operand index in \p MI that defines \p Def
  static unsigned getDefIndex(const MachineInstr &MI, Register SearchDef) {
    unsigned DefIdx = 0;
    for (const MachineOperand &Def : MI.defs()) {
      if (Def.getReg() == SearchDef)
        break;
      ++DefIdx;
    }

    return DefIdx;
  }

  bool tryCombineUnmergeValues(MachineInstr &MI,
                               SmallVectorImpl<MachineInstr *> &DeadInsts,
                               SmallVectorImpl<Register> &UpdatedDefs,
                               GISelObserverWrapper &Observer) {
    assert(MI.getOpcode() == TargetOpcode::G_UNMERGE_VALUES);

    unsigned NumDefs = MI.getNumOperands() - 1;
    Register SrcReg = MI.getOperand(NumDefs).getReg();
    MachineInstr *SrcDef = getDefIgnoringCopies(SrcReg, MRI);
    if (!SrcDef)
      return false;

    LLT OpTy = MRI.getType(MI.getOperand(NumDefs).getReg());
    LLT DestTy = MRI.getType(MI.getOperand(0).getReg());

    if (SrcDef->getOpcode() == TargetOpcode::G_UNMERGE_VALUES) {
      // %0:_(<4 x s16>) = G_FOO
      // %1:_(<2 x s16>), %2:_(<2 x s16>) = G_UNMERGE_VALUES %0
      // %3:_(s16), %4:_(s16) = G_UNMERGE_VALUES %1
      //
      // %3:_(s16), %4:_(s16), %5:_(s16), %6:_(s16) = G_UNMERGE_VALUES %0
      const unsigned NumSrcOps = SrcDef->getNumOperands();
      Register SrcUnmergeSrc = SrcDef->getOperand(NumSrcOps - 1).getReg();
      LLT SrcUnmergeSrcTy = MRI.getType(SrcUnmergeSrc);

      // If we need to decrease the number of vector elements in the result type
      // of an unmerge, this would involve the creation of an equivalent unmerge
      // to copy back to the original result registers.
      LegalizeActionStep ActionStep = LI.getAction(
          {TargetOpcode::G_UNMERGE_VALUES, {OpTy, SrcUnmergeSrcTy}});
      switch (ActionStep.Action) {
      case LegalizeActions::Lower:
      case LegalizeActions::Unsupported:
        break;
      case LegalizeActions::FewerElements:
      case LegalizeActions::NarrowScalar:
        if (ActionStep.TypeIdx == 1)
          return false;
        break;
      default:
        return false;
      }

      Builder.setInstrAndDebugLoc(MI);
      auto NewUnmerge = Builder.buildUnmerge(DestTy, SrcUnmergeSrc);

      // TODO: Should we try to process out the other defs now? If the other
      // defs of the source unmerge are also unmerged, we end up with a separate
      // unmerge for each one.
      unsigned SrcDefIdx = getDefIndex(*SrcDef, SrcReg);
      for (unsigned I = 0; I != NumDefs; ++I) {
        Register Def = MI.getOperand(I).getReg();
        replaceRegOrBuildCopy(Def, NewUnmerge.getReg(SrcDefIdx * NumDefs + I),
                              MRI, Builder, UpdatedDefs, Observer);
      }

      markInstAndDefDead(MI, *SrcDef, DeadInsts, SrcDefIdx);
      return true;
    }

    MachineInstr *MergeI = SrcDef;
    unsigned ConvertOp = 0;

    // Handle intermediate conversions
    unsigned SrcOp = SrcDef->getOpcode();
    if (isArtifactCast(SrcOp)) {
      ConvertOp = SrcOp;
      MergeI = getDefIgnoringCopies(SrcDef->getOperand(1).getReg(), MRI);
    }

    if (!MergeI || !canFoldMergeOpcode(MergeI->getOpcode(),
                                       ConvertOp, OpTy, DestTy)) {
      // We might have a chance to combine later by trying to combine
      // unmerge(cast) first
      return tryFoldUnmergeCast(MI, *SrcDef, DeadInsts, UpdatedDefs);
    }

    const unsigned NumMergeRegs = MergeI->getNumOperands() - 1;

    if (NumMergeRegs < NumDefs) {
      if (NumDefs % NumMergeRegs != 0)
        return false;

      Builder.setInstr(MI);
      // Transform to UNMERGEs, for example
      //   %1 = G_MERGE_VALUES %4, %5
      //   %9, %10, %11, %12 = G_UNMERGE_VALUES %1
      // to
      //   %9, %10 = G_UNMERGE_VALUES %4
      //   %11, %12 = G_UNMERGE_VALUES %5

      const unsigned NewNumDefs = NumDefs / NumMergeRegs;
      for (unsigned Idx = 0; Idx < NumMergeRegs; ++Idx) {
        SmallVector<Register, 8> DstRegs;
        for (unsigned j = 0, DefIdx = Idx * NewNumDefs; j < NewNumDefs;
             ++j, ++DefIdx)
          DstRegs.push_back(MI.getOperand(DefIdx).getReg());

        if (ConvertOp) {
          LLT MergeSrcTy = MRI.getType(MergeI->getOperand(1).getReg());

          // This is a vector that is being split and casted. Extract to the
          // element type, and do the conversion on the scalars (or smaller
          // vectors).
          LLT MergeEltTy = MergeSrcTy.divide(NewNumDefs);

          // Handle split to smaller vectors, with conversions.
          // %2(<8 x s8>) = G_CONCAT_VECTORS %0(<4 x s8>), %1(<4 x s8>)
          // %3(<8 x s16>) = G_SEXT %2
          // %4(<2 x s16>), %5(<2 x s16>), %6(<2 x s16>), %7(<2 x s16>) = G_UNMERGE_VALUES %3
          //
          // =>
          //
          // %8(<2 x s8>), %9(<2 x s8>) = G_UNMERGE_VALUES %0
          // %10(<2 x s8>), %11(<2 x s8>) = G_UNMERGE_VALUES %1
          // %4(<2 x s16>) = G_SEXT %8
          // %5(<2 x s16>) = G_SEXT %9
          // %6(<2 x s16>) = G_SEXT %10
          // %7(<2 x s16>)= G_SEXT %11

          SmallVector<Register, 4> TmpRegs(NewNumDefs);
          for (unsigned k = 0; k < NewNumDefs; ++k)
            TmpRegs[k] = MRI.createGenericVirtualRegister(MergeEltTy);

          Builder.buildUnmerge(TmpRegs, MergeI->getOperand(Idx + 1).getReg());

          for (unsigned k = 0; k < NewNumDefs; ++k)
            Builder.buildInstr(ConvertOp, {DstRegs[k]}, {TmpRegs[k]});
        } else {
          Builder.buildUnmerge(DstRegs, MergeI->getOperand(Idx + 1).getReg());
        }
        UpdatedDefs.append(DstRegs.begin(), DstRegs.end());
      }

    } else if (NumMergeRegs > NumDefs) {
      if (ConvertOp != 0 || NumMergeRegs % NumDefs != 0)
        return false;

      Builder.setInstr(MI);
      // Transform to MERGEs
      //   %6 = G_MERGE_VALUES %17, %18, %19, %20
      //   %7, %8 = G_UNMERGE_VALUES %6
      // to
      //   %7 = G_MERGE_VALUES %17, %18
      //   %8 = G_MERGE_VALUES %19, %20

      const unsigned NumRegs = NumMergeRegs / NumDefs;
      for (unsigned DefIdx = 0; DefIdx < NumDefs; ++DefIdx) {
        SmallVector<Register, 8> Regs;
        for (unsigned j = 0, Idx = NumRegs * DefIdx + 1; j < NumRegs;
             ++j, ++Idx)
          Regs.push_back(MergeI->getOperand(Idx).getReg());

        Register DefReg = MI.getOperand(DefIdx).getReg();
        Builder.buildMerge(DefReg, Regs);
        UpdatedDefs.push_back(DefReg);
      }

    } else {
      LLT MergeSrcTy = MRI.getType(MergeI->getOperand(1).getReg());

      if (!ConvertOp && DestTy != MergeSrcTy)
        ConvertOp = TargetOpcode::G_BITCAST;

      if (ConvertOp) {
        Builder.setInstr(MI);

        for (unsigned Idx = 0; Idx < NumDefs; ++Idx) {
          Register MergeSrc = MergeI->getOperand(Idx + 1).getReg();
          Register DefReg = MI.getOperand(Idx).getReg();
          Builder.buildInstr(ConvertOp, {DefReg}, {MergeSrc});
          UpdatedDefs.push_back(DefReg);
        }

        markInstAndDefDead(MI, *MergeI, DeadInsts);
        return true;
      }

      assert(DestTy == MergeSrcTy &&
             "Bitcast and the other kinds of conversions should "
             "have happened earlier");

      Builder.setInstr(MI);
      for (unsigned Idx = 0; Idx < NumDefs; ++Idx) {
        Register DstReg = MI.getOperand(Idx).getReg();
        Register SrcReg = MergeI->getOperand(Idx + 1).getReg();
        replaceRegOrBuildCopy(DstReg, SrcReg, MRI, Builder, UpdatedDefs,
                              Observer);
      }
    }

    markInstAndDefDead(MI, *MergeI, DeadInsts);
    return true;
  }

  static bool isMergeLikeOpcode(unsigned Opc) {
    switch (Opc) {
    case TargetOpcode::G_MERGE_VALUES:
    case TargetOpcode::G_BUILD_VECTOR:
    case TargetOpcode::G_CONCAT_VECTORS:
      return true;
    default:
      return false;
    }
  }

  bool tryCombineExtract(MachineInstr &MI,
                         SmallVectorImpl<MachineInstr *> &DeadInsts,
                         SmallVectorImpl<Register> &UpdatedDefs) {
    assert(MI.getOpcode() == TargetOpcode::G_EXTRACT);

    // Try to use the source registers from a G_MERGE_VALUES
    //
    // %2 = G_MERGE_VALUES %0, %1
    // %3 = G_EXTRACT %2, N
    // =>
    //
    // for N < %2.getSizeInBits() / 2
    //     %3 = G_EXTRACT %0, N
    //
    // for N >= %2.getSizeInBits() / 2
    //    %3 = G_EXTRACT %1, (N - %0.getSizeInBits()

    Register SrcReg = lookThroughCopyInstrs(MI.getOperand(1).getReg());
    MachineInstr *MergeI = MRI.getVRegDef(SrcReg);
    if (!MergeI || !isMergeLikeOpcode(MergeI->getOpcode()))
      return false;

    Register DstReg = MI.getOperand(0).getReg();
    LLT DstTy = MRI.getType(DstReg);
    LLT SrcTy = MRI.getType(SrcReg);

    // TODO: Do we need to check if the resulting extract is supported?
    unsigned ExtractDstSize = DstTy.getSizeInBits();
    unsigned Offset = MI.getOperand(2).getImm();
    unsigned NumMergeSrcs = MergeI->getNumOperands() - 1;
    unsigned MergeSrcSize = SrcTy.getSizeInBits() / NumMergeSrcs;
    unsigned MergeSrcIdx = Offset / MergeSrcSize;

    // Compute the offset of the last bit the extract needs.
    unsigned EndMergeSrcIdx = (Offset + ExtractDstSize - 1) / MergeSrcSize;

    // Can't handle the case where the extract spans multiple inputs.
    if (MergeSrcIdx != EndMergeSrcIdx)
      return false;

    // TODO: We could modify MI in place in most cases.
    Builder.setInstr(MI);
    Builder.buildExtract(DstReg, MergeI->getOperand(MergeSrcIdx + 1).getReg(),
                         Offset - MergeSrcIdx * MergeSrcSize);
    UpdatedDefs.push_back(DstReg);
    markInstAndDefDead(MI, *MergeI, DeadInsts);
    return true;
  }

  /// Try to combine away MI.
  /// Returns true if it combined away the MI.
  /// Adds instructions that are dead as a result of the combine
  /// into DeadInsts, which can include MI.
  bool tryCombineInstruction(MachineInstr &MI,
                             SmallVectorImpl<MachineInstr *> &DeadInsts,
                             GISelObserverWrapper &WrapperObserver) {
    // This might be a recursive call, and we might have DeadInsts already
    // populated. To avoid bad things happening later with multiple vreg defs
    // etc, process the dead instructions now if any.
    if (!DeadInsts.empty())
      deleteMarkedDeadInsts(DeadInsts, WrapperObserver);

    // Put here every vreg that was redefined in such a way that it's at least
    // possible that one (or more) of its users (immediate or COPY-separated)
    // could become artifact combinable with the new definition (or the
    // instruction reachable from it through a chain of copies if any).
    SmallVector<Register, 4> UpdatedDefs;
    bool Changed = false;
    switch (MI.getOpcode()) {
    default:
      return false;
    case TargetOpcode::G_ANYEXT:
      Changed = tryCombineAnyExt(MI, DeadInsts, UpdatedDefs);
      break;
    case TargetOpcode::G_ZEXT:
      Changed = tryCombineZExt(MI, DeadInsts, UpdatedDefs, WrapperObserver);
      break;
    case TargetOpcode::G_SEXT:
      Changed = tryCombineSExt(MI, DeadInsts, UpdatedDefs);
      break;
    case TargetOpcode::G_UNMERGE_VALUES:
      Changed =
          tryCombineUnmergeValues(MI, DeadInsts, UpdatedDefs, WrapperObserver);
      break;
    case TargetOpcode::G_MERGE_VALUES:
    case TargetOpcode::G_BUILD_VECTOR:
    case TargetOpcode::G_CONCAT_VECTORS:
      // If any of the users of this merge are an unmerge, then add them to the
      // artifact worklist in case there's folding that can be done looking up.
      for (MachineInstr &U : MRI.use_instructions(MI.getOperand(0).getReg())) {
        if (U.getOpcode() == TargetOpcode::G_UNMERGE_VALUES ||
            U.getOpcode() == TargetOpcode::G_TRUNC) {
          UpdatedDefs.push_back(MI.getOperand(0).getReg());
          break;
        }
      }
      break;
    case TargetOpcode::G_EXTRACT:
      Changed = tryCombineExtract(MI, DeadInsts, UpdatedDefs);
      break;
    case TargetOpcode::G_TRUNC:
      Changed = tryCombineTrunc(MI, DeadInsts, UpdatedDefs, WrapperObserver);
      if (!Changed) {
        // Try to combine truncates away even if they are legal. As all artifact
        // combines at the moment look only "up" the def-use chains, we achieve
        // that by throwing truncates' users (with look through copies) into the
        // ArtifactList again.
        UpdatedDefs.push_back(MI.getOperand(0).getReg());
      }
      break;
    }
    // If the main loop through the ArtifactList found at least one combinable
    // pair of artifacts, not only combine it away (as done above), but also
    // follow the def-use chain from there to combine everything that can be
    // combined within this def-use chain of artifacts.
    while (!UpdatedDefs.empty()) {
      Register NewDef = UpdatedDefs.pop_back_val();
      assert(NewDef.isVirtual() && "Unexpected redefinition of a physreg");
      for (MachineInstr &Use : MRI.use_instructions(NewDef)) {
        switch (Use.getOpcode()) {
        // Keep this list in sync with the list of all artifact combines.
        case TargetOpcode::G_ANYEXT:
        case TargetOpcode::G_ZEXT:
        case TargetOpcode::G_SEXT:
        case TargetOpcode::G_UNMERGE_VALUES:
        case TargetOpcode::G_EXTRACT:
        case TargetOpcode::G_TRUNC:
          // Adding Use to ArtifactList.
          WrapperObserver.changedInstr(Use);
          break;
        case TargetOpcode::COPY: {
          Register Copy = Use.getOperand(0).getReg();
          if (Copy.isVirtual())
            UpdatedDefs.push_back(Copy);
          break;
        }
        default:
          // If we do not have an artifact combine for the opcode, there is no
          // point in adding it to the ArtifactList as nothing interesting will
          // be done to it anyway.
          break;
        }
      }
    }
    return Changed;
  }

private:
  static Register getArtifactSrcReg(const MachineInstr &MI) {
    switch (MI.getOpcode()) {
    case TargetOpcode::COPY:
    case TargetOpcode::G_TRUNC:
    case TargetOpcode::G_ZEXT:
    case TargetOpcode::G_ANYEXT:
    case TargetOpcode::G_SEXT:
    case TargetOpcode::G_EXTRACT:
      return MI.getOperand(1).getReg();
    case TargetOpcode::G_UNMERGE_VALUES:
      return MI.getOperand(MI.getNumOperands() - 1).getReg();
    default:
      llvm_unreachable("Not a legalization artifact happen");
    }
  }

  /// Mark a def of one of MI's original operands, DefMI, as dead if changing MI
  /// (either by killing it or changing operands) results in DefMI being dead
  /// too. In-between COPYs or artifact-casts are also collected if they are
  /// dead.
  /// MI is not marked dead.
  void markDefDead(MachineInstr &MI, MachineInstr &DefMI,
                   SmallVectorImpl<MachineInstr *> &DeadInsts,
                   unsigned DefIdx = 0) {
    // Collect all the copy instructions that are made dead, due to deleting
    // this instruction. Collect all of them until the Trunc(DefMI).
    // Eg,
    // %1(s1) = G_TRUNC %0(s32)
    // %2(s1) = COPY %1(s1)
    // %3(s1) = COPY %2(s1)
    // %4(s32) = G_ANYEXT %3(s1)
    // In this case, we would have replaced %4 with a copy of %0,
    // and as a result, %3, %2, %1 are dead.
    MachineInstr *PrevMI = &MI;
    while (PrevMI != &DefMI) {
      Register PrevRegSrc = getArtifactSrcReg(*PrevMI);

      MachineInstr *TmpDef = MRI.getVRegDef(PrevRegSrc);
      if (MRI.hasOneUse(PrevRegSrc)) {
        if (TmpDef != &DefMI) {
          assert((TmpDef->getOpcode() == TargetOpcode::COPY ||
                  isArtifactCast(TmpDef->getOpcode())) &&
                 "Expecting copy or artifact cast here");

          DeadInsts.push_back(TmpDef);
        }
      } else
        break;
      PrevMI = TmpDef;
    }

    if (PrevMI == &DefMI) {
      unsigned I = 0;
      bool IsDead = true;
      for (MachineOperand &Def : DefMI.defs()) {
        if (I != DefIdx) {
          if (!MRI.use_empty(Def.getReg())) {
            IsDead = false;
            break;
          }
        } else {
          if (!MRI.hasOneUse(DefMI.getOperand(DefIdx).getReg()))
            break;
        }

        ++I;
      }

      if (IsDead)
        DeadInsts.push_back(&DefMI);
    }
  }

  /// Mark MI as dead. If a def of one of MI's operands, DefMI, would also be
  /// dead due to MI being killed, then mark DefMI as dead too.
  /// Some of the combines (extends(trunc)), try to walk through redundant
  /// copies in between the extends and the truncs, and this attempts to collect
  /// the in between copies if they're dead.
  void markInstAndDefDead(MachineInstr &MI, MachineInstr &DefMI,
                          SmallVectorImpl<MachineInstr *> &DeadInsts,
                          unsigned DefIdx = 0) {
    DeadInsts.push_back(&MI);
    markDefDead(MI, DefMI, DeadInsts, DefIdx);
  }

  /// Erase the dead instructions in the list and call the observer hooks.
  /// Normally the Legalizer will deal with erasing instructions that have been
  /// marked dead. However, for the trunc(ext(x)) cases we can end up trying to
  /// process instructions which have been marked dead, but otherwise break the
  /// MIR by introducing multiple vreg defs. For those cases, allow the combines
  /// to explicitly delete the instructions before we run into trouble.
  void deleteMarkedDeadInsts(SmallVectorImpl<MachineInstr *> &DeadInsts,
                             GISelObserverWrapper &WrapperObserver) {
    for (auto *DeadMI : DeadInsts) {
      LLVM_DEBUG(dbgs() << *DeadMI << "Is dead, eagerly deleting\n");
      WrapperObserver.erasingInstr(*DeadMI);
      DeadMI->eraseFromParentAndMarkDBGValuesForRemoval();
    }
    DeadInsts.clear();
  }

  /// Checks if the target legalizer info has specified anything about the
  /// instruction, or if unsupported.
  bool isInstUnsupported(const LegalityQuery &Query) const {
    using namespace LegalizeActions;
    auto Step = LI.getAction(Query);
    return Step.Action == Unsupported || Step.Action == NotFound;
  }

  bool isInstLegal(const LegalityQuery &Query) const {
    return LI.getAction(Query).Action == LegalizeActions::Legal;
  }

  bool isConstantUnsupported(LLT Ty) const {
    if (!Ty.isVector())
      return isInstUnsupported({TargetOpcode::G_CONSTANT, {Ty}});

    LLT EltTy = Ty.getElementType();
    return isInstUnsupported({TargetOpcode::G_CONSTANT, {EltTy}}) ||
           isInstUnsupported({TargetOpcode::G_BUILD_VECTOR, {Ty, EltTy}});
  }

  /// Looks through copy instructions and returns the actual
  /// source register.
  Register lookThroughCopyInstrs(Register Reg) {
    Register TmpReg;
    while (mi_match(Reg, MRI, m_Copy(m_Reg(TmpReg)))) {
      if (MRI.getType(TmpReg).isValid())
        Reg = TmpReg;
      else
        break;
    }
    return Reg;
  }
};

} // namespace llvm
