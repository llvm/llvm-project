//===-- llvm/CodeGen/GlobalISel/LegalizerCombiner.h --===========//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This file contains some helper functions which try to cleanup artifacts
// such as G_TRUNCs/G_[ZSA]EXTENDS that were created during legalization to make
// the types match. This file also contains some combines of merges that happens
// at the end of the legalization.
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GlobalISel/Legalizer.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "legalizer"

namespace llvm {
class LegalizerCombiner {
  MachineIRBuilder &Builder;
  MachineRegisterInfo &MRI;

public:
  LegalizerCombiner(MachineIRBuilder &B, MachineRegisterInfo &MRI)
      : Builder(B), MRI(MRI) {}

  bool tryCombineAnyExt(MachineInstr &MI,
                        SmallVectorImpl<MachineInstr *> &DeadInsts) {
    if (MI.getOpcode() != TargetOpcode::G_ANYEXT)
      return false;
    MachineInstr *DefMI = MRI.getVRegDef(MI.getOperand(1).getReg());
    if (DefMI->getOpcode() == TargetOpcode::G_TRUNC) {
      DEBUG(dbgs() << ".. Combine MI: " << MI;);
      unsigned DstReg = MI.getOperand(0).getReg();
      unsigned SrcReg = DefMI->getOperand(1).getReg();
      Builder.setInstr(MI);
      // We get a copy/trunc/extend depending on the sizes
      Builder.buildAnyExtOrTrunc(DstReg, SrcReg);
      MI.eraseFromParent();
      if (MRI.use_empty(DefMI->getOperand(0).getReg()))
        DeadInsts.push_back(DefMI);
      return true;
    }
    return false;
  }

  bool tryCombineZExt(MachineInstr &MI,
                      SmallVectorImpl<MachineInstr *> &DeadInsts) {

    if (MI.getOpcode() != TargetOpcode::G_ZEXT)
      return false;
    MachineInstr *DefMI = MRI.getVRegDef(MI.getOperand(1).getReg());
    if (DefMI->getOpcode() == TargetOpcode::G_TRUNC) {
      DEBUG(dbgs() << ".. Combine MI: " << MI;);
      Builder.setInstr(MI);
      unsigned DstReg = MI.getOperand(0).getReg();
      unsigned ZExtSrc = MI.getOperand(1).getReg();
      LLT ZExtSrcTy = MRI.getType(ZExtSrc);
      LLT DstTy = MRI.getType(DstReg);
      APInt Mask = APInt::getAllOnesValue(ZExtSrcTy.getSizeInBits());
      auto MaskCstMIB = Builder.buildConstant(DstTy, Mask.getZExtValue());
      unsigned TruncSrc = DefMI->getOperand(1).getReg();
      // We get a copy/trunc/extend depending on the sizes
      auto SrcCopyOrTrunc = Builder.buildAnyExtOrTrunc(DstTy, TruncSrc);
      Builder.buildAnd(DstReg, SrcCopyOrTrunc, MaskCstMIB);
      MI.eraseFromParent();
      if (MRI.use_empty(DefMI->getOperand(0).getReg()))
        DeadInsts.push_back(DefMI);
      return true;
    }
    return false;
  }

  bool tryCombineSExt(MachineInstr &MI,
                      SmallVectorImpl<MachineInstr *> &DeadInsts) {

    if (MI.getOpcode() != TargetOpcode::G_SEXT)
      return false;
    MachineInstr *DefMI = MRI.getVRegDef(MI.getOperand(1).getReg());
    if (DefMI->getOpcode() == TargetOpcode::G_TRUNC) {
      DEBUG(dbgs() << ".. Combine MI: " << MI;);
      Builder.setInstr(MI);
      unsigned DstReg = MI.getOperand(0).getReg();
      LLT DstTy = MRI.getType(DstReg);
      unsigned SExtSrc = MI.getOperand(1).getReg();
      LLT SExtSrcTy = MRI.getType(SExtSrc);
      unsigned SizeDiff = DstTy.getSizeInBits() - SExtSrcTy.getSizeInBits();
      auto SizeDiffMIB = Builder.buildConstant(DstTy, SizeDiff);
      unsigned TruncSrcReg = DefMI->getOperand(1).getReg();
      // We get a copy/trunc/extend depending on the sizes
      auto SrcCopyExtOrTrunc = Builder.buildAnyExtOrTrunc(DstTy, TruncSrcReg);
      auto ShlMIB = Builder.buildInstr(TargetOpcode::G_SHL, DstTy,
                                       SrcCopyExtOrTrunc, SizeDiffMIB);
      Builder.buildInstr(TargetOpcode::G_ASHR, DstReg, ShlMIB, SizeDiffMIB);
      MI.eraseFromParent();
      if (MRI.use_empty(DefMI->getOperand(0).getReg()))
        DeadInsts.push_back(DefMI);
      return true;
    }
    return false;
  }

  bool tryCombineMerges(MachineInstr &MI,
                        SmallVectorImpl<MachineInstr *> &DeadInsts) {

    if (MI.getOpcode() != TargetOpcode::G_UNMERGE_VALUES)
      return false;

    unsigned NumDefs = MI.getNumOperands() - 1;
    unsigned SrcReg = MI.getOperand(NumDefs).getReg();
    MachineInstr *MergeI = MRI.getVRegDef(SrcReg);
    if (!MergeI || (MergeI->getOpcode() != TargetOpcode::G_MERGE_VALUES))
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
        SmallVector<unsigned, 2> DstRegs;
        for (unsigned j = 0, DefIdx = Idx * NewNumDefs; j < NewNumDefs;
             ++j, ++DefIdx)
          DstRegs.push_back(MI.getOperand(DefIdx).getReg());

        Builder.buildUnmerge(DstRegs, MergeI->getOperand(Idx + 1).getReg());
      }

    } else if (NumMergeRegs > NumDefs) {
      if (NumMergeRegs % NumDefs != 0)
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
        SmallVector<unsigned, 2> Regs;
        for (unsigned j = 0, Idx = NumRegs * DefIdx + 1; j < NumRegs;
             ++j, ++Idx)
          Regs.push_back(MergeI->getOperand(Idx).getReg());

        Builder.buildMerge(MI.getOperand(DefIdx).getReg(), Regs);
      }

    } else {
      // FIXME: is a COPY appropriate if the types mismatch? We know both
      // registers are allocatable by now.
      if (MRI.getType(MI.getOperand(0).getReg()) !=
          MRI.getType(MergeI->getOperand(1).getReg()))
        return false;

      for (unsigned Idx = 0; Idx < NumDefs; ++Idx)
        MRI.replaceRegWith(MI.getOperand(Idx).getReg(),
                           MergeI->getOperand(Idx + 1).getReg());
    }

    MI.eraseFromParent();
    if (MRI.use_empty(MergeI->getOperand(0).getReg()))
      DeadInsts.push_back(MergeI);
    return true;
  }

  /// Try to combine away MI.
  /// Returns true if it combined away the MI.
  /// Caller should not rely in MI existing as it may be deleted.
  /// Adds instructions that are dead as a result of the combine
  // into DeadInsts
  bool tryCombineInstruction(MachineInstr &MI,
                             SmallVectorImpl<MachineInstr *> &DeadInsts) {
    switch (MI.getOpcode()) {
    default:
      return false;
    case TargetOpcode::G_ANYEXT:
      return tryCombineAnyExt(MI, DeadInsts);
    case TargetOpcode::G_ZEXT:
      return tryCombineZExt(MI, DeadInsts);
    case TargetOpcode::G_SEXT:
      return tryCombineSExt(MI, DeadInsts);
    case TargetOpcode::G_UNMERGE_VALUES:
      return tryCombineMerges(MI, DeadInsts);
    }
  }
};

} // namespace llvm
