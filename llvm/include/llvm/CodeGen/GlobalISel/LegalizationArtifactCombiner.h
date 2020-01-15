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

    Builder.setInstr(MI);
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
      const LLT &DstTy = MRI.getType(DstReg);
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
                      SmallVectorImpl<Register> &UpdatedDefs) {
    assert(MI.getOpcode() == TargetOpcode::G_ZEXT);

    Builder.setInstr(MI);
    Register DstReg = MI.getOperand(0).getReg();
    Register SrcReg = lookThroughCopyInstrs(MI.getOperand(1).getReg());

    // zext(trunc x) - > and (aext/copy/trunc x), mask
    Register TruncSrc;
    if (mi_match(SrcReg, MRI, m_GTrunc(m_Reg(TruncSrc)))) {
      LLT DstTy = MRI.getType(DstReg);
      if (isInstUnsupported({TargetOpcode::G_AND, {DstTy}}) ||
          isConstantUnsupported(DstTy))
        return false;
      LLVM_DEBUG(dbgs() << ".. Combine MI: " << MI;);
      LLT SrcTy = MRI.getType(SrcReg);
      APInt Mask = APInt::getAllOnesValue(SrcTy.getScalarSizeInBits());
      auto MIBMask = Builder.buildConstant(
        DstTy, Mask.zext(DstTy.getScalarSizeInBits()));
      Builder.buildAnd(DstReg, Builder.buildAnyExtOrTrunc(DstTy, TruncSrc),
                       MIBMask);
      markInstAndDefDead(MI, *MRI.getVRegDef(SrcReg), DeadInsts);
      return true;
    }

    // Try to fold zext(g_constant) when the larger constant type is legal.
    // Can't use MIPattern because we don't have a specific constant in mind.
    auto *SrcMI = MRI.getVRegDef(SrcReg);
    if (SrcMI->getOpcode() == TargetOpcode::G_CONSTANT) {
      const LLT &DstTy = MRI.getType(DstReg);
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

    Builder.setInstr(MI);
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
    return tryFoldImplicitDef(MI, DeadInsts, UpdatedDefs);
  }

  bool tryCombineTrunc(MachineInstr &MI,
                       SmallVectorImpl<MachineInstr *> &DeadInsts,
                       SmallVectorImpl<Register> &UpdatedDefs) {
    assert(MI.getOpcode() == TargetOpcode::G_TRUNC);

    Builder.setInstr(MI);
    Register DstReg = MI.getOperand(0).getReg();
    Register SrcReg = lookThroughCopyInstrs(MI.getOperand(1).getReg());

    // Try to fold trunc(g_constant) when the smaller constant type is legal.
    // Can't use MIPattern because we don't have a specific constant in mind.
    auto *SrcMI = MRI.getVRegDef(SrcReg);
    if (SrcMI->getOpcode() == TargetOpcode::G_CONSTANT) {
      const LLT &DstTy = MRI.getType(DstReg);
      if (isInstLegal({TargetOpcode::G_CONSTANT, {DstTy}})) {
        auto &CstVal = SrcMI->getOperand(1);
        Builder.buildConstant(
            DstReg, CstVal.getCImm()->getValue().trunc(DstTy.getSizeInBits()));
        UpdatedDefs.push_back(DstReg);
        markInstAndDefDead(MI, *SrcMI, DeadInsts);
        return true;
      }
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
        if (isInstUnsupported({TargetOpcode::G_IMPLICIT_DEF, {DstTy}}))
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
      return !DestTy.isVector();
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

  bool tryCombineMerges(MachineInstr &MI,
                        SmallVectorImpl<MachineInstr *> &DeadInsts,
                        SmallVectorImpl<Register> &UpdatedDefs) {
    assert(MI.getOpcode() == TargetOpcode::G_UNMERGE_VALUES);

    unsigned NumDefs = MI.getNumOperands() - 1;
    MachineInstr *SrcDef =
        getDefIgnoringCopies(MI.getOperand(NumDefs).getReg(), MRI);
    if (!SrcDef)
      return false;

    LLT OpTy = MRI.getType(MI.getOperand(NumDefs).getReg());
    LLT DestTy = MRI.getType(MI.getOperand(0).getReg());
    MachineInstr *MergeI = SrcDef;
    unsigned ConvertOp = 0;

    // Handle intermediate conversions
    unsigned SrcOp = SrcDef->getOpcode();
    if (isArtifactCast(SrcOp)) {
      ConvertOp = SrcOp;
      MergeI = getDefIgnoringCopies(SrcDef->getOperand(1).getReg(), MRI);
    }

    if (!MergeI || !canFoldMergeOpcode(MergeI->getOpcode(),
                                       ConvertOp, OpTy, DestTy))
      return false;

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
        SmallVector<Register, 2> DstRegs;
        for (unsigned j = 0, DefIdx = Idx * NewNumDefs; j < NewNumDefs;
             ++j, ++DefIdx)
          DstRegs.push_back(MI.getOperand(DefIdx).getReg());

        if (ConvertOp) {
          SmallVector<Register, 2> TmpRegs;
          // This is a vector that is being scalarized and casted. Extract to
          // the element type, and do the conversion on the scalars.
          LLT MergeEltTy =
              MRI.getType(MergeI->getOperand(0).getReg()).getElementType();
          for (unsigned j = 0; j < NumMergeRegs; ++j)
            TmpRegs.push_back(MRI.createGenericVirtualRegister(MergeEltTy));

          Builder.buildUnmerge(TmpRegs, MergeI->getOperand(Idx + 1).getReg());

          for (unsigned j = 0; j < NumMergeRegs; ++j)
            Builder.buildInstr(ConvertOp, {DstRegs[j]}, {TmpRegs[j]});
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
        SmallVector<Register, 2> Regs;
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

      for (unsigned Idx = 0; Idx < NumDefs; ++Idx) {
        Register NewDef = MergeI->getOperand(Idx + 1).getReg();
        MRI.replaceRegWith(MI.getOperand(Idx).getReg(), NewDef);
        UpdatedDefs.push_back(NewDef);
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
      Changed = tryCombineZExt(MI, DeadInsts, UpdatedDefs);
      break;
    case TargetOpcode::G_SEXT:
      Changed = tryCombineSExt(MI, DeadInsts, UpdatedDefs);
      break;
    case TargetOpcode::G_UNMERGE_VALUES:
      Changed = tryCombineMerges(MI, DeadInsts, UpdatedDefs);
      break;
    case TargetOpcode::G_EXTRACT:
      Changed = tryCombineExtract(MI, DeadInsts, UpdatedDefs);
      break;
    case TargetOpcode::G_TRUNC:
      Changed = tryCombineTrunc(MI, DeadInsts, UpdatedDefs);
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
  static unsigned getArtifactSrcReg(const MachineInstr &MI) {
    switch (MI.getOpcode()) {
    case TargetOpcode::COPY:
    case TargetOpcode::G_TRUNC:
    case TargetOpcode::G_ZEXT:
    case TargetOpcode::G_ANYEXT:
    case TargetOpcode::G_SEXT:
    case TargetOpcode::G_UNMERGE_VALUES:
      return MI.getOperand(MI.getNumOperands() - 1).getReg();
    case TargetOpcode::G_EXTRACT:
      return MI.getOperand(1).getReg();
    default:
      llvm_unreachable("Not a legalization artifact happen");
    }
  }

  /// Mark MI as dead. If a def of one of MI's operands, DefMI, would also be
  /// dead due to MI being killed, then mark DefMI as dead too.
  /// Some of the combines (extends(trunc)), try to walk through redundant
  /// copies in between the extends and the truncs, and this attempts to collect
  /// the in between copies if they're dead.
  void markInstAndDefDead(MachineInstr &MI, MachineInstr &DefMI,
                          SmallVectorImpl<MachineInstr *> &DeadInsts) {
    DeadInsts.push_back(&MI);

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
      unsigned PrevRegSrc = getArtifactSrcReg(*PrevMI);

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
    if (PrevMI == &DefMI && MRI.hasOneUse(DefMI.getOperand(0).getReg()))
      DeadInsts.push_back(&DefMI);
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
  unsigned lookThroughCopyInstrs(Register Reg) {
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
