//===-- lib/CodeGen/GlobalISel/GICombinerHelper.cpp -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "llvm/CodeGen/GlobalISel/CombinerHelper.h"
#include "llvm/CodeGen/GlobalISel/Combiner.h"
#include "llvm/CodeGen/GlobalISel/GISelChangeObserver.h"
#include "llvm/CodeGen/GlobalISel/GISelKnownBits.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/Target/TargetMachine.h"

#define DEBUG_TYPE "gi-combiner"

using namespace llvm;

// Option to allow testing of the combiner while no targets know about indexed
// addressing.
static cl::opt<bool>
    ForceLegalIndexing("force-legal-indexing", cl::Hidden, cl::init(false),
                       cl::desc("Force all indexed operations to be "
                                "legal for the GlobalISel combiner"));


CombinerHelper::CombinerHelper(GISelChangeObserver &Observer,
                               MachineIRBuilder &B, GISelKnownBits *KB,
                               MachineDominatorTree *MDT)
    : Builder(B), MRI(Builder.getMF().getRegInfo()), Observer(Observer),
      KB(KB), MDT(MDT) {
  (void)this->KB;
}

void CombinerHelper::replaceRegWith(MachineRegisterInfo &MRI, Register FromReg,
                                    Register ToReg) const {
  Observer.changingAllUsesOfReg(MRI, FromReg);

  if (MRI.constrainRegAttrs(ToReg, FromReg))
    MRI.replaceRegWith(FromReg, ToReg);
  else
    Builder.buildCopy(ToReg, FromReg);

  Observer.finishedChangingAllUsesOfReg();
}

void CombinerHelper::replaceRegOpWith(MachineRegisterInfo &MRI,
                                      MachineOperand &FromRegOp,
                                      Register ToReg) const {
  assert(FromRegOp.getParent() && "Expected an operand in an MI");
  Observer.changingInstr(*FromRegOp.getParent());

  FromRegOp.setReg(ToReg);

  Observer.changedInstr(*FromRegOp.getParent());
}

bool CombinerHelper::tryCombineCopy(MachineInstr &MI) {
  if (matchCombineCopy(MI)) {
    applyCombineCopy(MI);
    return true;
  }
  return false;
}
bool CombinerHelper::matchCombineCopy(MachineInstr &MI) {
  if (MI.getOpcode() != TargetOpcode::COPY)
    return false;
  Register DstReg = MI.getOperand(0).getReg();
  Register SrcReg = MI.getOperand(1).getReg();

  // Give up if either DstReg or SrcReg  is a physical register.
  if (Register::isPhysicalRegister(DstReg) ||
      Register::isPhysicalRegister(SrcReg))
    return false;

  // Give up the types don't match.
  LLT DstTy = MRI.getType(DstReg);
  LLT SrcTy = MRI.getType(SrcReg);
  // Give up if one has a valid LLT, but the other doesn't.
  if (DstTy.isValid() != SrcTy.isValid())
    return false;
  // Give up if the types don't match.
  if (DstTy.isValid() && SrcTy.isValid() && DstTy != SrcTy)
    return false;

  // Get the register banks and classes.
  const RegisterBank *DstBank = MRI.getRegBankOrNull(DstReg);
  const RegisterBank *SrcBank = MRI.getRegBankOrNull(SrcReg);
  const TargetRegisterClass *DstRC = MRI.getRegClassOrNull(DstReg);
  const TargetRegisterClass *SrcRC = MRI.getRegClassOrNull(SrcReg);

  // Replace if the register constraints match.
  if ((SrcRC == DstRC) && (SrcBank == DstBank))
    return true;
  // Replace if DstReg has no constraints.
  if (!DstBank && !DstRC)
    return true;

  return false;
}
void CombinerHelper::applyCombineCopy(MachineInstr &MI) {
  Register DstReg = MI.getOperand(0).getReg();
  Register SrcReg = MI.getOperand(1).getReg();
  MI.eraseFromParent();
  replaceRegWith(MRI, DstReg, SrcReg);
}

bool CombinerHelper::tryCombineConcatVectors(MachineInstr &MI) {
  bool IsUndef = false;
  SmallVector<Register, 4> Ops;
  if (matchCombineConcatVectors(MI, IsUndef, Ops)) {
    applyCombineConcatVectors(MI, IsUndef, Ops);
    return true;
  }
  return false;
}

bool CombinerHelper::matchCombineConcatVectors(MachineInstr &MI, bool &IsUndef,
                                               SmallVectorImpl<Register> &Ops) {
  assert(MI.getOpcode() == TargetOpcode::G_CONCAT_VECTORS &&
         "Invalid instruction");
  IsUndef = true;
  MachineInstr *Undef = nullptr;

  // Walk over all the operands of concat vectors and check if they are
  // build_vector themselves or undef.
  // Then collect their operands in Ops.
  for (const MachineOperand &MO : MI.uses()) {
    Register Reg = MO.getReg();
    MachineInstr *Def = MRI.getVRegDef(Reg);
    assert(Def && "Operand not defined");
    switch (Def->getOpcode()) {
    case TargetOpcode::G_BUILD_VECTOR:
      IsUndef = false;
      // Remember the operands of the build_vector to fold
      // them into the yet-to-build flattened concat vectors.
      for (const MachineOperand &BuildVecMO : Def->uses())
        Ops.push_back(BuildVecMO.getReg());
      break;
    case TargetOpcode::G_IMPLICIT_DEF: {
      LLT OpType = MRI.getType(Reg);
      // Keep one undef value for all the undef operands.
      if (!Undef) {
        Builder.setInsertPt(*MI.getParent(), MI);
        Undef = Builder.buildUndef(OpType.getScalarType());
      }
      assert(MRI.getType(Undef->getOperand(0).getReg()) ==
                 OpType.getScalarType() &&
             "All undefs should have the same type");
      // Break the undef vector in as many scalar elements as needed
      // for the flattening.
      for (unsigned EltIdx = 0, EltEnd = OpType.getNumElements();
           EltIdx != EltEnd; ++EltIdx)
        Ops.push_back(Undef->getOperand(0).getReg());
      break;
    }
    default:
      return false;
    }
  }
  return true;
}
void CombinerHelper::applyCombineConcatVectors(
    MachineInstr &MI, bool IsUndef, const ArrayRef<Register> Ops) {
  // We determined that the concat_vectors can be flatten.
  // Generate the flattened build_vector.
  Register DstReg = MI.getOperand(0).getReg();
  Builder.setInsertPt(*MI.getParent(), MI);
  Register NewDstReg = MRI.cloneVirtualRegister(DstReg);

  // Note: IsUndef is sort of redundant. We could have determine it by
  // checking that at all Ops are undef.  Alternatively, we could have
  // generate a build_vector of undefs and rely on another combine to
  // clean that up.  For now, given we already gather this information
  // in tryCombineConcatVectors, just save compile time and issue the
  // right thing.
  if (IsUndef)
    Builder.buildUndef(NewDstReg);
  else
    Builder.buildBuildVector(NewDstReg, Ops);
  MI.eraseFromParent();
  replaceRegWith(MRI, DstReg, NewDstReg);
}

bool CombinerHelper::tryCombineShuffleVector(MachineInstr &MI) {
  SmallVector<Register, 4> Ops;
  if (matchCombineShuffleVector(MI, Ops)) {
    applyCombineShuffleVector(MI, Ops);
    return true;
  }
  return false;
}

bool CombinerHelper::matchCombineShuffleVector(MachineInstr &MI,
                                               SmallVectorImpl<Register> &Ops) {
  assert(MI.getOpcode() == TargetOpcode::G_SHUFFLE_VECTOR &&
         "Invalid instruction kind");
  LLT DstType = MRI.getType(MI.getOperand(0).getReg());
  Register Src1 = MI.getOperand(1).getReg();
  LLT SrcType = MRI.getType(Src1);
  // As bizarre as it may look, shuffle vector can actually produce
  // scalar! This is because at the IR level a <1 x ty> shuffle
  // vector is perfectly valid.
  unsigned DstNumElts = DstType.isVector() ? DstType.getNumElements() : 1;
  unsigned SrcNumElts = SrcType.isVector() ? SrcType.getNumElements() : 1;

  // If the resulting vector is smaller than the size of the source
  // vectors being concatenated, we won't be able to replace the
  // shuffle vector into a concat_vectors.
  //
  // Note: We may still be able to produce a concat_vectors fed by
  //       extract_vector_elt and so on. It is less clear that would
  //       be better though, so don't bother for now.
  //
  // If the destination is a scalar, the size of the sources doesn't
  // matter. we will lower the shuffle to a plain copy. This will
  // work only if the source and destination have the same size. But
  // that's covered by the next condition.
  //
  // TODO: If the size between the source and destination don't match
  //       we could still emit an extract vector element in that case.
  if (DstNumElts < 2 * SrcNumElts && DstNumElts != 1)
    return false;

  // Check that the shuffle mask can be broken evenly between the
  // different sources.
  if (DstNumElts % SrcNumElts != 0)
    return false;

  // Mask length is a multiple of the source vector length.
  // Check if the shuffle is some kind of concatenation of the input
  // vectors.
  unsigned NumConcat = DstNumElts / SrcNumElts;
  SmallVector<int, 8> ConcatSrcs(NumConcat, -1);
  SmallVector<int, 8> Mask;
  ShuffleVectorInst::getShuffleMask(MI.getOperand(3).getShuffleMask(), Mask);
  for (unsigned i = 0; i != DstNumElts; ++i) {
    int Idx = Mask[i];
    // Undef value.
    if (Idx < 0)
      continue;
    // Ensure the indices in each SrcType sized piece are sequential and that
    // the same source is used for the whole piece.
    if ((Idx % SrcNumElts != (i % SrcNumElts)) ||
        (ConcatSrcs[i / SrcNumElts] >= 0 &&
         ConcatSrcs[i / SrcNumElts] != (int)(Idx / SrcNumElts)))
      return false;
    // Remember which source this index came from.
    ConcatSrcs[i / SrcNumElts] = Idx / SrcNumElts;
  }

  // The shuffle is concatenating multiple vectors together.
  // Collect the different operands for that.
  Register UndefReg;
  Register Src2 = MI.getOperand(2).getReg();
  for (auto Src : ConcatSrcs) {
    if (Src < 0) {
      if (!UndefReg) {
        Builder.setInsertPt(*MI.getParent(), MI);
        UndefReg = Builder.buildUndef(SrcType).getReg(0);
      }
      Ops.push_back(UndefReg);
    } else if (Src == 0)
      Ops.push_back(Src1);
    else
      Ops.push_back(Src2);
  }
  return true;
}

void CombinerHelper::applyCombineShuffleVector(MachineInstr &MI,
                                               const ArrayRef<Register> Ops) {
  Register DstReg = MI.getOperand(0).getReg();
  Builder.setInsertPt(*MI.getParent(), MI);
  Register NewDstReg = MRI.cloneVirtualRegister(DstReg);

  if (Ops.size() == 1)
    Builder.buildCopy(NewDstReg, Ops[0]);
  else
    Builder.buildMerge(NewDstReg, Ops);

  MI.eraseFromParent();
  replaceRegWith(MRI, DstReg, NewDstReg);
}

namespace {

/// Select a preference between two uses. CurrentUse is the current preference
/// while *ForCandidate is attributes of the candidate under consideration.
PreferredTuple ChoosePreferredUse(PreferredTuple &CurrentUse,
                                  const LLT &TyForCandidate,
                                  unsigned OpcodeForCandidate,
                                  MachineInstr *MIForCandidate) {
  if (!CurrentUse.Ty.isValid()) {
    if (CurrentUse.ExtendOpcode == OpcodeForCandidate ||
        CurrentUse.ExtendOpcode == TargetOpcode::G_ANYEXT)
      return {TyForCandidate, OpcodeForCandidate, MIForCandidate};
    return CurrentUse;
  }

  // We permit the extend to hoist through basic blocks but this is only
  // sensible if the target has extending loads. If you end up lowering back
  // into a load and extend during the legalizer then the end result is
  // hoisting the extend up to the load.

  // Prefer defined extensions to undefined extensions as these are more
  // likely to reduce the number of instructions.
  if (OpcodeForCandidate == TargetOpcode::G_ANYEXT &&
      CurrentUse.ExtendOpcode != TargetOpcode::G_ANYEXT)
    return CurrentUse;
  else if (CurrentUse.ExtendOpcode == TargetOpcode::G_ANYEXT &&
           OpcodeForCandidate != TargetOpcode::G_ANYEXT)
    return {TyForCandidate, OpcodeForCandidate, MIForCandidate};

  // Prefer sign extensions to zero extensions as sign-extensions tend to be
  // more expensive.
  if (CurrentUse.Ty == TyForCandidate) {
    if (CurrentUse.ExtendOpcode == TargetOpcode::G_SEXT &&
        OpcodeForCandidate == TargetOpcode::G_ZEXT)
      return CurrentUse;
    else if (CurrentUse.ExtendOpcode == TargetOpcode::G_ZEXT &&
             OpcodeForCandidate == TargetOpcode::G_SEXT)
      return {TyForCandidate, OpcodeForCandidate, MIForCandidate};
  }

  // This is potentially target specific. We've chosen the largest type
  // because G_TRUNC is usually free. One potential catch with this is that
  // some targets have a reduced number of larger registers than smaller
  // registers and this choice potentially increases the live-range for the
  // larger value.
  if (TyForCandidate.getSizeInBits() > CurrentUse.Ty.getSizeInBits()) {
    return {TyForCandidate, OpcodeForCandidate, MIForCandidate};
  }
  return CurrentUse;
}

/// Find a suitable place to insert some instructions and insert them. This
/// function accounts for special cases like inserting before a PHI node.
/// The current strategy for inserting before PHI's is to duplicate the
/// instructions for each predecessor. However, while that's ok for G_TRUNC
/// on most targets since it generally requires no code, other targets/cases may
/// want to try harder to find a dominating block.
static void InsertInsnsWithoutSideEffectsBeforeUse(
    MachineIRBuilder &Builder, MachineInstr &DefMI, MachineOperand &UseMO,
    std::function<void(MachineBasicBlock *, MachineBasicBlock::iterator,
                       MachineOperand &UseMO)>
        Inserter) {
  MachineInstr &UseMI = *UseMO.getParent();

  MachineBasicBlock *InsertBB = UseMI.getParent();

  // If the use is a PHI then we want the predecessor block instead.
  if (UseMI.isPHI()) {
    MachineOperand *PredBB = std::next(&UseMO);
    InsertBB = PredBB->getMBB();
  }

  // If the block is the same block as the def then we want to insert just after
  // the def instead of at the start of the block.
  if (InsertBB == DefMI.getParent()) {
    MachineBasicBlock::iterator InsertPt = &DefMI;
    Inserter(InsertBB, std::next(InsertPt), UseMO);
    return;
  }

  // Otherwise we want the start of the BB
  Inserter(InsertBB, InsertBB->getFirstNonPHI(), UseMO);
}
} // end anonymous namespace

bool CombinerHelper::tryCombineExtendingLoads(MachineInstr &MI) {
  PreferredTuple Preferred;
  if (matchCombineExtendingLoads(MI, Preferred)) {
    applyCombineExtendingLoads(MI, Preferred);
    return true;
  }
  return false;
}

bool CombinerHelper::matchCombineExtendingLoads(MachineInstr &MI,
                                                PreferredTuple &Preferred) {
  // We match the loads and follow the uses to the extend instead of matching
  // the extends and following the def to the load. This is because the load
  // must remain in the same position for correctness (unless we also add code
  // to find a safe place to sink it) whereas the extend is freely movable.
  // It also prevents us from duplicating the load for the volatile case or just
  // for performance.

  if (MI.getOpcode() != TargetOpcode::G_LOAD &&
      MI.getOpcode() != TargetOpcode::G_SEXTLOAD &&
      MI.getOpcode() != TargetOpcode::G_ZEXTLOAD)
    return false;

  auto &LoadValue = MI.getOperand(0);
  assert(LoadValue.isReg() && "Result wasn't a register?");

  LLT LoadValueTy = MRI.getType(LoadValue.getReg());
  if (!LoadValueTy.isScalar())
    return false;

  // Most architectures are going to legalize <s8 loads into at least a 1 byte
  // load, and the MMOs can only describe memory accesses in multiples of bytes.
  // If we try to perform extload combining on those, we can end up with
  // %a(s8) = extload %ptr (load 1 byte from %ptr)
  // ... which is an illegal extload instruction.
  if (LoadValueTy.getSizeInBits() < 8)
    return false;

  // For non power-of-2 types, they will very likely be legalized into multiple
  // loads. Don't bother trying to match them into extending loads.
  if (!isPowerOf2_32(LoadValueTy.getSizeInBits()))
    return false;

  // Find the preferred type aside from the any-extends (unless it's the only
  // one) and non-extending ops. We'll emit an extending load to that type and
  // and emit a variant of (extend (trunc X)) for the others according to the
  // relative type sizes. At the same time, pick an extend to use based on the
  // extend involved in the chosen type.
  unsigned PreferredOpcode = MI.getOpcode() == TargetOpcode::G_LOAD
                                 ? TargetOpcode::G_ANYEXT
                                 : MI.getOpcode() == TargetOpcode::G_SEXTLOAD
                                       ? TargetOpcode::G_SEXT
                                       : TargetOpcode::G_ZEXT;
  Preferred = {LLT(), PreferredOpcode, nullptr};
  for (auto &UseMI : MRI.use_instructions(LoadValue.getReg())) {
    if (UseMI.getOpcode() == TargetOpcode::G_SEXT ||
        UseMI.getOpcode() == TargetOpcode::G_ZEXT ||
        UseMI.getOpcode() == TargetOpcode::G_ANYEXT) {
      Preferred = ChoosePreferredUse(Preferred,
                                     MRI.getType(UseMI.getOperand(0).getReg()),
                                     UseMI.getOpcode(), &UseMI);
    }
  }

  // There were no extends
  if (!Preferred.MI)
    return false;
  // It should be impossible to chose an extend without selecting a different
  // type since by definition the result of an extend is larger.
  assert(Preferred.Ty != LoadValueTy && "Extending to same type?");

  LLVM_DEBUG(dbgs() << "Preferred use is: " << *Preferred.MI);
  return true;
}

void CombinerHelper::applyCombineExtendingLoads(MachineInstr &MI,
                                                PreferredTuple &Preferred) {
  // Rewrite the load to the chosen extending load.
  Register ChosenDstReg = Preferred.MI->getOperand(0).getReg();

  // Inserter to insert a truncate back to the original type at a given point
  // with some basic CSE to limit truncate duplication to one per BB.
  DenseMap<MachineBasicBlock *, MachineInstr *> EmittedInsns;
  auto InsertTruncAt = [&](MachineBasicBlock *InsertIntoBB,
                           MachineBasicBlock::iterator InsertBefore,
                           MachineOperand &UseMO) {
    MachineInstr *PreviouslyEmitted = EmittedInsns.lookup(InsertIntoBB);
    if (PreviouslyEmitted) {
      Observer.changingInstr(*UseMO.getParent());
      UseMO.setReg(PreviouslyEmitted->getOperand(0).getReg());
      Observer.changedInstr(*UseMO.getParent());
      return;
    }

    Builder.setInsertPt(*InsertIntoBB, InsertBefore);
    Register NewDstReg = MRI.cloneVirtualRegister(MI.getOperand(0).getReg());
    MachineInstr *NewMI = Builder.buildTrunc(NewDstReg, ChosenDstReg);
    EmittedInsns[InsertIntoBB] = NewMI;
    replaceRegOpWith(MRI, UseMO, NewDstReg);
  };

  Observer.changingInstr(MI);
  MI.setDesc(
      Builder.getTII().get(Preferred.ExtendOpcode == TargetOpcode::G_SEXT
                               ? TargetOpcode::G_SEXTLOAD
                               : Preferred.ExtendOpcode == TargetOpcode::G_ZEXT
                                     ? TargetOpcode::G_ZEXTLOAD
                                     : TargetOpcode::G_LOAD));

  // Rewrite all the uses to fix up the types.
  auto &LoadValue = MI.getOperand(0);
  SmallVector<MachineOperand *, 4> Uses;
  for (auto &UseMO : MRI.use_operands(LoadValue.getReg()))
    Uses.push_back(&UseMO);

  for (auto *UseMO : Uses) {
    MachineInstr *UseMI = UseMO->getParent();

    // If the extend is compatible with the preferred extend then we should fix
    // up the type and extend so that it uses the preferred use.
    if (UseMI->getOpcode() == Preferred.ExtendOpcode ||
        UseMI->getOpcode() == TargetOpcode::G_ANYEXT) {
      Register UseDstReg = UseMI->getOperand(0).getReg();
      MachineOperand &UseSrcMO = UseMI->getOperand(1);
      const LLT &UseDstTy = MRI.getType(UseDstReg);
      if (UseDstReg != ChosenDstReg) {
        if (Preferred.Ty == UseDstTy) {
          // If the use has the same type as the preferred use, then merge
          // the vregs and erase the extend. For example:
          //    %1:_(s8) = G_LOAD ...
          //    %2:_(s32) = G_SEXT %1(s8)
          //    %3:_(s32) = G_ANYEXT %1(s8)
          //    ... = ... %3(s32)
          // rewrites to:
          //    %2:_(s32) = G_SEXTLOAD ...
          //    ... = ... %2(s32)
          replaceRegWith(MRI, UseDstReg, ChosenDstReg);
          Observer.erasingInstr(*UseMO->getParent());
          UseMO->getParent()->eraseFromParent();
        } else if (Preferred.Ty.getSizeInBits() < UseDstTy.getSizeInBits()) {
          // If the preferred size is smaller, then keep the extend but extend
          // from the result of the extending load. For example:
          //    %1:_(s8) = G_LOAD ...
          //    %2:_(s32) = G_SEXT %1(s8)
          //    %3:_(s64) = G_ANYEXT %1(s8)
          //    ... = ... %3(s64)
          /// rewrites to:
          //    %2:_(s32) = G_SEXTLOAD ...
          //    %3:_(s64) = G_ANYEXT %2:_(s32)
          //    ... = ... %3(s64)
          replaceRegOpWith(MRI, UseSrcMO, ChosenDstReg);
        } else {
          // If the preferred size is large, then insert a truncate. For
          // example:
          //    %1:_(s8) = G_LOAD ...
          //    %2:_(s64) = G_SEXT %1(s8)
          //    %3:_(s32) = G_ZEXT %1(s8)
          //    ... = ... %3(s32)
          /// rewrites to:
          //    %2:_(s64) = G_SEXTLOAD ...
          //    %4:_(s8) = G_TRUNC %2:_(s32)
          //    %3:_(s64) = G_ZEXT %2:_(s8)
          //    ... = ... %3(s64)
          InsertInsnsWithoutSideEffectsBeforeUse(Builder, MI, *UseMO,
                                                 InsertTruncAt);
        }
        continue;
      }
      // The use is (one of) the uses of the preferred use we chose earlier.
      // We're going to update the load to def this value later so just erase
      // the old extend.
      Observer.erasingInstr(*UseMO->getParent());
      UseMO->getParent()->eraseFromParent();
      continue;
    }

    // The use isn't an extend. Truncate back to the type we originally loaded.
    // This is free on many targets.
    InsertInsnsWithoutSideEffectsBeforeUse(Builder, MI, *UseMO, InsertTruncAt);
  }

  MI.getOperand(0).setReg(ChosenDstReg);
  Observer.changedInstr(MI);
}

bool CombinerHelper::isPredecessor(MachineInstr &DefMI, MachineInstr &UseMI) {
  assert(DefMI.getParent() == UseMI.getParent());
  if (&DefMI == &UseMI)
    return false;

  // Loop through the basic block until we find one of the instructions.
  MachineBasicBlock::const_iterator I = DefMI.getParent()->begin();
  for (; &*I != &DefMI && &*I != &UseMI; ++I)
    return &*I == &DefMI;

  llvm_unreachable("Block must contain instructions");
}

bool CombinerHelper::dominates(MachineInstr &DefMI, MachineInstr &UseMI) {
  if (MDT)
    return MDT->dominates(&DefMI, &UseMI);
  else if (DefMI.getParent() != UseMI.getParent())
    return false;

  return isPredecessor(DefMI, UseMI);
}

bool CombinerHelper::findPostIndexCandidate(MachineInstr &MI, Register &Addr,
                                            Register &Base, Register &Offset) {
  auto &MF = *MI.getParent()->getParent();
  const auto &TLI = *MF.getSubtarget().getTargetLowering();

#ifndef NDEBUG
  unsigned Opcode = MI.getOpcode();
  assert(Opcode == TargetOpcode::G_LOAD || Opcode == TargetOpcode::G_SEXTLOAD ||
         Opcode == TargetOpcode::G_ZEXTLOAD || Opcode == TargetOpcode::G_STORE);
#endif

  Base = MI.getOperand(1).getReg();
  MachineInstr *BaseDef = MRI.getUniqueVRegDef(Base);
  if (BaseDef && BaseDef->getOpcode() == TargetOpcode::G_FRAME_INDEX)
    return false;

  LLVM_DEBUG(dbgs() << "Searching for post-indexing opportunity for: " << MI);

  for (auto &Use : MRI.use_instructions(Base)) {
    if (Use.getOpcode() != TargetOpcode::G_PTR_ADD)
      continue;

    Offset = Use.getOperand(2).getReg();
    if (!ForceLegalIndexing &&
        !TLI.isIndexingLegal(MI, Base, Offset, /*IsPre*/ false, MRI)) {
      LLVM_DEBUG(dbgs() << "    Ignoring candidate with illegal addrmode: "
                        << Use);
      continue;
    }

    // Make sure the offset calculation is before the potentially indexed op.
    // FIXME: we really care about dependency here. The offset calculation might
    // be movable.
    MachineInstr *OffsetDef = MRI.getUniqueVRegDef(Offset);
    if (!OffsetDef || !dominates(*OffsetDef, MI)) {
      LLVM_DEBUG(dbgs() << "    Ignoring candidate with offset after mem-op: "
                        << Use);
      continue;
    }

    // FIXME: check whether all uses of Base are load/store with foldable
    // addressing modes. If so, using the normal addr-modes is better than
    // forming an indexed one.

    bool MemOpDominatesAddrUses = true;
    for (auto &PtrAddUse : MRI.use_instructions(Use.getOperand(0).getReg())) {
      if (!dominates(MI, PtrAddUse)) {
        MemOpDominatesAddrUses = false;
        break;
      }
    }

    if (!MemOpDominatesAddrUses) {
      LLVM_DEBUG(
          dbgs() << "    Ignoring candidate as memop does not dominate uses: "
                 << Use);
      continue;
    }

    LLVM_DEBUG(dbgs() << "    Found match: " << Use);
    Addr = Use.getOperand(0).getReg();
    return true;
  }

  return false;
}

bool CombinerHelper::findPreIndexCandidate(MachineInstr &MI, Register &Addr,
                                           Register &Base, Register &Offset) {
  auto &MF = *MI.getParent()->getParent();
  const auto &TLI = *MF.getSubtarget().getTargetLowering();

#ifndef NDEBUG
  unsigned Opcode = MI.getOpcode();
  assert(Opcode == TargetOpcode::G_LOAD || Opcode == TargetOpcode::G_SEXTLOAD ||
         Opcode == TargetOpcode::G_ZEXTLOAD || Opcode == TargetOpcode::G_STORE);
#endif

  Addr = MI.getOperand(1).getReg();
  MachineInstr *AddrDef = getOpcodeDef(TargetOpcode::G_PTR_ADD, Addr, MRI);
  if (!AddrDef || MRI.hasOneUse(Addr))
    return false;

  Base = AddrDef->getOperand(1).getReg();
  Offset = AddrDef->getOperand(2).getReg();

  LLVM_DEBUG(dbgs() << "Found potential pre-indexed load_store: " << MI);

  if (!ForceLegalIndexing &&
      !TLI.isIndexingLegal(MI, Base, Offset, /*IsPre*/ true, MRI)) {
    LLVM_DEBUG(dbgs() << "    Skipping, not legal for target");
    return false;
  }

  MachineInstr *BaseDef = getDefIgnoringCopies(Base, MRI);
  if (BaseDef->getOpcode() == TargetOpcode::G_FRAME_INDEX) {
    LLVM_DEBUG(dbgs() << "    Skipping, frame index would need copy anyway.");
    return false;
  }

  if (MI.getOpcode() == TargetOpcode::G_STORE) {
    // Would require a copy.
    if (Base == MI.getOperand(0).getReg()) {
      LLVM_DEBUG(dbgs() << "    Skipping, storing base so need copy anyway.");
      return false;
    }

    // We're expecting one use of Addr in MI, but it could also be the
    // value stored, which isn't actually dominated by the instruction.
    if (MI.getOperand(0).getReg() == Addr) {
      LLVM_DEBUG(dbgs() << "    Skipping, does not dominate all addr uses");
      return false;
    }
  }

  // FIXME: check whether all uses of the base pointer are constant PtrAdds.
  // That might allow us to end base's liveness here by adjusting the constant.

  for (auto &UseMI : MRI.use_instructions(Addr)) {
    if (!dominates(MI, UseMI)) {
      LLVM_DEBUG(dbgs() << "    Skipping, does not dominate all addr uses.");
      return false;
    }
  }

  return true;
}

bool CombinerHelper::tryCombineIndexedLoadStore(MachineInstr &MI) {
  IndexedLoadStoreMatchInfo MatchInfo;
  if (matchCombineIndexedLoadStore(MI, MatchInfo)) {
    applyCombineIndexedLoadStore(MI, MatchInfo);
    return true;
  }
  return false;
}

bool CombinerHelper::matchCombineIndexedLoadStore(MachineInstr &MI, IndexedLoadStoreMatchInfo &MatchInfo) {
  unsigned Opcode = MI.getOpcode();
  if (Opcode != TargetOpcode::G_LOAD && Opcode != TargetOpcode::G_SEXTLOAD &&
      Opcode != TargetOpcode::G_ZEXTLOAD && Opcode != TargetOpcode::G_STORE)
    return false;

  MatchInfo.IsPre = findPreIndexCandidate(MI, MatchInfo.Addr, MatchInfo.Base,
                                          MatchInfo.Offset);
  if (!MatchInfo.IsPre &&
      !findPostIndexCandidate(MI, MatchInfo.Addr, MatchInfo.Base,
                              MatchInfo.Offset))
    return false;

  return true;
}

void CombinerHelper::applyCombineIndexedLoadStore(
    MachineInstr &MI, IndexedLoadStoreMatchInfo &MatchInfo) {
  MachineInstr &AddrDef = *MRI.getUniqueVRegDef(MatchInfo.Addr);
  MachineIRBuilder MIRBuilder(MI);
  unsigned Opcode = MI.getOpcode();
  bool IsStore = Opcode == TargetOpcode::G_STORE;
  unsigned NewOpcode;
  switch (Opcode) {
  case TargetOpcode::G_LOAD:
    NewOpcode = TargetOpcode::G_INDEXED_LOAD;
    break;
  case TargetOpcode::G_SEXTLOAD:
    NewOpcode = TargetOpcode::G_INDEXED_SEXTLOAD;
    break;
  case TargetOpcode::G_ZEXTLOAD:
    NewOpcode = TargetOpcode::G_INDEXED_ZEXTLOAD;
    break;
  case TargetOpcode::G_STORE:
    NewOpcode = TargetOpcode::G_INDEXED_STORE;
    break;
  default:
    llvm_unreachable("Unknown load/store opcode");
  }

  auto MIB = MIRBuilder.buildInstr(NewOpcode);
  if (IsStore) {
    MIB.addDef(MatchInfo.Addr);
    MIB.addUse(MI.getOperand(0).getReg());
  } else {
    MIB.addDef(MI.getOperand(0).getReg());
    MIB.addDef(MatchInfo.Addr);
  }

  MIB.addUse(MatchInfo.Base);
  MIB.addUse(MatchInfo.Offset);
  MIB.addImm(MatchInfo.IsPre);
  MI.eraseFromParent();
  AddrDef.eraseFromParent();

  LLVM_DEBUG(dbgs() << "    Combinined to indexed operation");
}

bool CombinerHelper::matchElideBrByInvertingCond(MachineInstr &MI) {
  if (MI.getOpcode() != TargetOpcode::G_BR)
    return false;

  // Try to match the following:
  // bb1:
  //   %c(s32) = G_ICMP pred, %a, %b
  //   %c1(s1) = G_TRUNC %c(s32)
  //   G_BRCOND %c1, %bb2
  //   G_BR %bb3
  // bb2:
  // ...
  // bb3:

  // The above pattern does not have a fall through to the successor bb2, always
  // resulting in a branch no matter which path is taken. Here we try to find
  // and replace that pattern with conditional branch to bb3 and otherwise
  // fallthrough to bb2.

  MachineBasicBlock *MBB = MI.getParent();
  MachineBasicBlock::iterator BrIt(MI);
  if (BrIt == MBB->begin())
    return false;
  assert(std::next(BrIt) == MBB->end() && "expected G_BR to be a terminator");

  MachineInstr *BrCond = &*std::prev(BrIt);
  if (BrCond->getOpcode() != TargetOpcode::G_BRCOND)
    return false;

  // Check that the next block is the conditional branch target.
  if (!MBB->isLayoutSuccessor(BrCond->getOperand(1).getMBB()))
    return false;

  MachineInstr *CmpMI = MRI.getVRegDef(BrCond->getOperand(0).getReg());
  if (!CmpMI || CmpMI->getOpcode() != TargetOpcode::G_ICMP ||
      !MRI.hasOneUse(CmpMI->getOperand(0).getReg()))
    return false;
  return true;
}

bool CombinerHelper::tryElideBrByInvertingCond(MachineInstr &MI) {
  if (!matchElideBrByInvertingCond(MI))
    return false;
  applyElideBrByInvertingCond(MI);
  return true;
}

void CombinerHelper::applyElideBrByInvertingCond(MachineInstr &MI) {
  MachineBasicBlock *BrTarget = MI.getOperand(0).getMBB();
  MachineBasicBlock::iterator BrIt(MI);
  MachineInstr *BrCond = &*std::prev(BrIt);
  MachineInstr *CmpMI = MRI.getVRegDef(BrCond->getOperand(0).getReg());

  CmpInst::Predicate InversePred = CmpInst::getInversePredicate(
      (CmpInst::Predicate)CmpMI->getOperand(1).getPredicate());

  // Invert the G_ICMP condition.
  Observer.changingInstr(*CmpMI);
  CmpMI->getOperand(1).setPredicate(InversePred);
  Observer.changedInstr(*CmpMI);

  // Change the conditional branch target.
  Observer.changingInstr(*BrCond);
  BrCond->getOperand(1).setMBB(BrTarget);
  Observer.changedInstr(*BrCond);
  MI.eraseFromParent();
}

static bool shouldLowerMemFuncForSize(const MachineFunction &MF) {
  // On Darwin, -Os means optimize for size without hurting performance, so
  // only really optimize for size when -Oz (MinSize) is used.
  if (MF.getTarget().getTargetTriple().isOSDarwin())
    return MF.getFunction().hasMinSize();
  return MF.getFunction().hasOptSize();
}

// Returns a list of types to use for memory op lowering in MemOps. A partial
// port of findOptimalMemOpLowering in TargetLowering.
static bool findGISelOptimalMemOpLowering(
    std::vector<LLT> &MemOps, unsigned Limit, uint64_t Size, unsigned DstAlign,
    unsigned SrcAlign, bool IsMemset, bool ZeroMemset, bool MemcpyStrSrc,
    bool AllowOverlap, unsigned DstAS, unsigned SrcAS,
    const AttributeList &FuncAttributes, const TargetLowering &TLI) {
  // If 'SrcAlign' is zero, that means the memory operation does not need to
  // load the value, i.e. memset or memcpy from constant string. Otherwise,
  // it's the inferred alignment of the source. 'DstAlign', on the other hand,
  // is the specified alignment of the memory operation. If it is zero, that
  // means it's possible to change the alignment of the destination.
  // 'MemcpyStrSrc' indicates whether the memcpy source is constant so it does
  // not need to be loaded.
  if (SrcAlign != 0 && SrcAlign < DstAlign)
    return false;

  LLT Ty = TLI.getOptimalMemOpLLT(Size, DstAlign, SrcAlign, IsMemset,
                                  ZeroMemset, MemcpyStrSrc, FuncAttributes);

  if (Ty == LLT()) {
    // Use the largest scalar type whose alignment constraints are satisfied.
    // We only need to check DstAlign here as SrcAlign is always greater or
    // equal to DstAlign (or zero).
    Ty = LLT::scalar(64);
    while (DstAlign && DstAlign < Ty.getSizeInBytes() &&
           !TLI.allowsMisalignedMemoryAccesses(Ty, DstAS, DstAlign))
      Ty = LLT::scalar(Ty.getSizeInBytes());
    assert(Ty.getSizeInBits() > 0 && "Could not find valid type");
    // FIXME: check for the largest legal type we can load/store to.
  }

  unsigned NumMemOps = 0;
  while (Size != 0) {
    unsigned TySize = Ty.getSizeInBytes();
    while (TySize > Size) {
      // For now, only use non-vector load / store's for the left-over pieces.
      LLT NewTy = Ty;
      // FIXME: check for mem op safety and legality of the types. Not all of
      // SDAGisms map cleanly to GISel concepts.
      if (NewTy.isVector())
        NewTy = NewTy.getSizeInBits() > 64 ? LLT::scalar(64) : LLT::scalar(32);
      NewTy = LLT::scalar(PowerOf2Floor(NewTy.getSizeInBits() - 1));
      unsigned NewTySize = NewTy.getSizeInBytes();
      assert(NewTySize > 0 && "Could not find appropriate type");

      // If the new LLT cannot cover all of the remaining bits, then consider
      // issuing a (or a pair of) unaligned and overlapping load / store.
      bool Fast;
      // Need to get a VT equivalent for allowMisalignedMemoryAccesses().
      MVT VT = getMVTForLLT(Ty);
      if (NumMemOps && AllowOverlap && NewTySize < Size &&
          TLI.allowsMisalignedMemoryAccesses(
              VT, DstAS, DstAlign, MachineMemOperand::MONone, &Fast) &&
          Fast)
        TySize = Size;
      else {
        Ty = NewTy;
        TySize = NewTySize;
      }
    }

    if (++NumMemOps > Limit)
      return false;

    MemOps.push_back(Ty);
    Size -= TySize;
  }

  return true;
}

static Type *getTypeForLLT(LLT Ty, LLVMContext &C) {
  if (Ty.isVector())
    return VectorType::get(IntegerType::get(C, Ty.getScalarSizeInBits()),
                           Ty.getNumElements());
  return IntegerType::get(C, Ty.getSizeInBits());
}

// Get a vectorized representation of the memset value operand, GISel edition.
static Register getMemsetValue(Register Val, LLT Ty, MachineIRBuilder &MIB) {
  MachineRegisterInfo &MRI = *MIB.getMRI();
  unsigned NumBits = Ty.getScalarSizeInBits();
  auto ValVRegAndVal = getConstantVRegValWithLookThrough(Val, MRI);
  if (!Ty.isVector() && ValVRegAndVal) {
    unsigned KnownVal = ValVRegAndVal->Value;
    APInt Scalar = APInt(8, KnownVal);
    APInt SplatVal = APInt::getSplat(NumBits, Scalar);
    return MIB.buildConstant(Ty, SplatVal).getReg(0);
  }
  // FIXME: for vector types create a G_BUILD_VECTOR.
  if (Ty.isVector())
    return Register();

  // Extend the byte value to the larger type, and then multiply by a magic
  // value 0x010101... in order to replicate it across every byte.
  LLT ExtType = Ty.getScalarType();
  auto ZExt = MIB.buildZExtOrTrunc(ExtType, Val);
  if (NumBits > 8) {
    APInt Magic = APInt::getSplat(NumBits, APInt(8, 0x01));
    auto MagicMI = MIB.buildConstant(ExtType, Magic);
    Val = MIB.buildMul(ExtType, ZExt, MagicMI).getReg(0);
  }

  assert(ExtType == Ty && "Vector memset value type not supported yet");
  return Val;
}

bool CombinerHelper::optimizeMemset(MachineInstr &MI, Register Dst, Register Val,
                                    unsigned KnownLen, unsigned Align,
                                    bool IsVolatile) {
  auto &MF = *MI.getParent()->getParent();
  const auto &TLI = *MF.getSubtarget().getTargetLowering();
  auto &DL = MF.getDataLayout();
  LLVMContext &C = MF.getFunction().getContext();

  assert(KnownLen != 0 && "Have a zero length memset length!");

  bool DstAlignCanChange = false;
  MachineFrameInfo &MFI = MF.getFrameInfo();
  bool OptSize = shouldLowerMemFuncForSize(MF);

  MachineInstr *FIDef = getOpcodeDef(TargetOpcode::G_FRAME_INDEX, Dst, MRI);
  if (FIDef && !MFI.isFixedObjectIndex(FIDef->getOperand(1).getIndex()))
    DstAlignCanChange = true;

  unsigned Limit = TLI.getMaxStoresPerMemset(OptSize);
  std::vector<LLT> MemOps;

  const auto &DstMMO = **MI.memoperands_begin();
  MachinePointerInfo DstPtrInfo = DstMMO.getPointerInfo();

  auto ValVRegAndVal = getConstantVRegValWithLookThrough(Val, MRI);
  bool IsZeroVal = ValVRegAndVal && ValVRegAndVal->Value == 0;

  if (!findGISelOptimalMemOpLowering(
          MemOps, Limit, KnownLen, (DstAlignCanChange ? 0 : Align), 0,
          /*IsMemset=*/true,
          /*ZeroMemset=*/IsZeroVal, /*MemcpyStrSrc=*/false,
          /*AllowOverlap=*/!IsVolatile, DstPtrInfo.getAddrSpace(), ~0u,
          MF.getFunction().getAttributes(), TLI))
    return false;

  if (DstAlignCanChange) {
    // Get an estimate of the type from the LLT.
    Type *IRTy = getTypeForLLT(MemOps[0], C);
    unsigned NewAlign = (unsigned)DL.getABITypeAlignment(IRTy);
    if (NewAlign > Align) {
      Align = NewAlign;
      unsigned FI = FIDef->getOperand(1).getIndex();
      // Give the stack frame object a larger alignment if needed.
      if (MFI.getObjectAlignment(FI) < Align)
        MFI.setObjectAlignment(FI, Align);
    }
  }

  MachineIRBuilder MIB(MI);
  // Find the largest store and generate the bit pattern for it.
  LLT LargestTy = MemOps[0];
  for (unsigned i = 1; i < MemOps.size(); i++)
    if (MemOps[i].getSizeInBits() > LargestTy.getSizeInBits())
      LargestTy = MemOps[i];

  // The memset stored value is always defined as an s8, so in order to make it
  // work with larger store types we need to repeat the bit pattern across the
  // wider type.
  Register MemSetValue = getMemsetValue(Val, LargestTy, MIB);

  if (!MemSetValue)
    return false;

  // Generate the stores. For each store type in the list, we generate the
  // matching store of that type to the destination address.
  LLT PtrTy = MRI.getType(Dst);
  unsigned DstOff = 0;
  unsigned Size = KnownLen;
  for (unsigned I = 0; I < MemOps.size(); I++) {
    LLT Ty = MemOps[I];
    unsigned TySize = Ty.getSizeInBytes();
    if (TySize > Size) {
      // Issuing an unaligned load / store pair that overlaps with the previous
      // pair. Adjust the offset accordingly.
      assert(I == MemOps.size() - 1 && I != 0);
      DstOff -= TySize - Size;
    }

    // If this store is smaller than the largest store see whether we can get
    // the smaller value for free with a truncate.
    Register Value = MemSetValue;
    if (Ty.getSizeInBits() < LargestTy.getSizeInBits()) {
      MVT VT = getMVTForLLT(Ty);
      MVT LargestVT = getMVTForLLT(LargestTy);
      if (!LargestTy.isVector() && !Ty.isVector() &&
          TLI.isTruncateFree(LargestVT, VT))
        Value = MIB.buildTrunc(Ty, MemSetValue).getReg(0);
      else
        Value = getMemsetValue(Val, Ty, MIB);
      if (!Value)
        return false;
    }

    auto *StoreMMO =
        MF.getMachineMemOperand(&DstMMO, DstOff, Ty.getSizeInBytes());

    Register Ptr = Dst;
    if (DstOff != 0) {
      auto Offset =
          MIB.buildConstant(LLT::scalar(PtrTy.getSizeInBits()), DstOff);
      Ptr = MIB.buildPtrAdd(PtrTy, Dst, Offset).getReg(0);
    }

    MIB.buildStore(Value, Ptr, *StoreMMO);
    DstOff += Ty.getSizeInBytes();
    Size -= TySize;
  }

  MI.eraseFromParent();
  return true;
}


bool CombinerHelper::optimizeMemcpy(MachineInstr &MI, Register Dst,
                                    Register Src, unsigned KnownLen,
                                    unsigned DstAlign, unsigned SrcAlign,
                                    bool IsVolatile) {
  auto &MF = *MI.getParent()->getParent();
  const auto &TLI = *MF.getSubtarget().getTargetLowering();
  auto &DL = MF.getDataLayout();
  LLVMContext &C = MF.getFunction().getContext();

  assert(KnownLen != 0 && "Have a zero length memcpy length!");

  bool DstAlignCanChange = false;
  MachineFrameInfo &MFI = MF.getFrameInfo();
  bool OptSize = shouldLowerMemFuncForSize(MF);
  unsigned Alignment = MinAlign(DstAlign, SrcAlign);

  MachineInstr *FIDef = getOpcodeDef(TargetOpcode::G_FRAME_INDEX, Dst, MRI);
  if (FIDef && !MFI.isFixedObjectIndex(FIDef->getOperand(1).getIndex()))
    DstAlignCanChange = true;

  // FIXME: infer better src pointer alignment like SelectionDAG does here.
  // FIXME: also use the equivalent of isMemSrcFromConstant and alwaysinlining
  // if the memcpy is in a tail call position.

  unsigned Limit = TLI.getMaxStoresPerMemcpy(OptSize);
  std::vector<LLT> MemOps;

  const auto &DstMMO = **MI.memoperands_begin();
  const auto &SrcMMO = **std::next(MI.memoperands_begin());
  MachinePointerInfo DstPtrInfo = DstMMO.getPointerInfo();
  MachinePointerInfo SrcPtrInfo = SrcMMO.getPointerInfo();

  if (!findGISelOptimalMemOpLowering(
          MemOps, Limit, KnownLen, (DstAlignCanChange ? 0 : Alignment),
          SrcAlign,
          /*IsMemset=*/false,
          /*ZeroMemset=*/false, /*MemcpyStrSrc=*/false,
          /*AllowOverlap=*/!IsVolatile, DstPtrInfo.getAddrSpace(),
          SrcPtrInfo.getAddrSpace(), MF.getFunction().getAttributes(), TLI))
    return false;

  if (DstAlignCanChange) {
    // Get an estimate of the type from the LLT.
    Type *IRTy = getTypeForLLT(MemOps[0], C);
    unsigned NewAlign = (unsigned)DL.getABITypeAlignment(IRTy);

    // Don't promote to an alignment that would require dynamic stack
    // realignment.
    const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
    if (!TRI->needsStackRealignment(MF))
      while (NewAlign > Alignment &&
             DL.exceedsNaturalStackAlignment(Align(NewAlign)))
        NewAlign /= 2;

    if (NewAlign > Alignment) {
      Alignment = NewAlign;
      unsigned FI = FIDef->getOperand(1).getIndex();
      // Give the stack frame object a larger alignment if needed.
      if (MFI.getObjectAlignment(FI) < Alignment)
        MFI.setObjectAlignment(FI, Alignment);
    }
  }

  LLVM_DEBUG(dbgs() << "Inlining memcpy: " << MI << " into loads & stores\n");

  MachineIRBuilder MIB(MI);
  // Now we need to emit a pair of load and stores for each of the types we've
  // collected. I.e. for each type, generate a load from the source pointer of
  // that type width, and then generate a corresponding store to the dest buffer
  // of that value loaded. This can result in a sequence of loads and stores
  // mixed types, depending on what the target specifies as good types to use.
  unsigned CurrOffset = 0;
  LLT PtrTy = MRI.getType(Src);
  unsigned Size = KnownLen;
  for (auto CopyTy : MemOps) {
    // Issuing an unaligned load / store pair  that overlaps with the previous
    // pair. Adjust the offset accordingly.
    if (CopyTy.getSizeInBytes() > Size)
      CurrOffset -= CopyTy.getSizeInBytes() - Size;

    // Construct MMOs for the accesses.
    auto *LoadMMO =
        MF.getMachineMemOperand(&SrcMMO, CurrOffset, CopyTy.getSizeInBytes());
    auto *StoreMMO = 
        MF.getMachineMemOperand(&DstMMO, CurrOffset, CopyTy.getSizeInBytes());

    // Create the load.
    Register LoadPtr = Src;
    Register Offset;
    if (CurrOffset != 0) {
      Offset = MIB.buildConstant(LLT::scalar(PtrTy.getSizeInBits()), CurrOffset)
                   .getReg(0);
      LoadPtr = MIB.buildPtrAdd(PtrTy, Src, Offset).getReg(0);
    }
    auto LdVal = MIB.buildLoad(CopyTy, LoadPtr, *LoadMMO);

    // Create the store.
    Register StorePtr =
        CurrOffset == 0 ? Dst : MIB.buildPtrAdd(PtrTy, Dst, Offset).getReg(0);
    MIB.buildStore(LdVal, StorePtr, *StoreMMO);
    CurrOffset += CopyTy.getSizeInBytes();
    Size -= CopyTy.getSizeInBytes();
  }

  MI.eraseFromParent();
  return true;
}

bool CombinerHelper::optimizeMemmove(MachineInstr &MI, Register Dst,
                                    Register Src, unsigned KnownLen,
                                    unsigned DstAlign, unsigned SrcAlign,
                                    bool IsVolatile) {
  auto &MF = *MI.getParent()->getParent();
  const auto &TLI = *MF.getSubtarget().getTargetLowering();
  auto &DL = MF.getDataLayout();
  LLVMContext &C = MF.getFunction().getContext();

  assert(KnownLen != 0 && "Have a zero length memmove length!");

  bool DstAlignCanChange = false;
  MachineFrameInfo &MFI = MF.getFrameInfo();
  bool OptSize = shouldLowerMemFuncForSize(MF);
  unsigned Alignment = MinAlign(DstAlign, SrcAlign);

  MachineInstr *FIDef = getOpcodeDef(TargetOpcode::G_FRAME_INDEX, Dst, MRI);
  if (FIDef && !MFI.isFixedObjectIndex(FIDef->getOperand(1).getIndex()))
    DstAlignCanChange = true;

  unsigned Limit = TLI.getMaxStoresPerMemmove(OptSize);
  std::vector<LLT> MemOps;

  const auto &DstMMO = **MI.memoperands_begin();
  const auto &SrcMMO = **std::next(MI.memoperands_begin());
  MachinePointerInfo DstPtrInfo = DstMMO.getPointerInfo();
  MachinePointerInfo SrcPtrInfo = SrcMMO.getPointerInfo();

  // FIXME: SelectionDAG always passes false for 'AllowOverlap', apparently due
  // to a bug in it's findOptimalMemOpLowering implementation. For now do the
  // same thing here.
  if (!findGISelOptimalMemOpLowering(
          MemOps, Limit, KnownLen, (DstAlignCanChange ? 0 : Alignment),
          SrcAlign,
          /*IsMemset=*/false,
          /*ZeroMemset=*/false, /*MemcpyStrSrc=*/false,
          /*AllowOverlap=*/false, DstPtrInfo.getAddrSpace(),
          SrcPtrInfo.getAddrSpace(), MF.getFunction().getAttributes(), TLI))
    return false;

  if (DstAlignCanChange) {
    // Get an estimate of the type from the LLT.
    Type *IRTy = getTypeForLLT(MemOps[0], C);
    unsigned NewAlign = (unsigned)DL.getABITypeAlignment(IRTy);

    // Don't promote to an alignment that would require dynamic stack
    // realignment.
    const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
    if (!TRI->needsStackRealignment(MF))
      while (NewAlign > Alignment &&
             DL.exceedsNaturalStackAlignment(Align(NewAlign)))
        NewAlign /= 2;

    if (NewAlign > Alignment) {
      Alignment = NewAlign;
      unsigned FI = FIDef->getOperand(1).getIndex();
      // Give the stack frame object a larger alignment if needed.
      if (MFI.getObjectAlignment(FI) < Alignment)
        MFI.setObjectAlignment(FI, Alignment);
    }
  }

  LLVM_DEBUG(dbgs() << "Inlining memmove: " << MI << " into loads & stores\n");

  MachineIRBuilder MIB(MI);
  // Memmove requires that we perform the loads first before issuing the stores.
  // Apart from that, this loop is pretty much doing the same thing as the
  // memcpy codegen function.
  unsigned CurrOffset = 0;
  LLT PtrTy = MRI.getType(Src);
  SmallVector<Register, 16> LoadVals;
  for (auto CopyTy : MemOps) {
    // Construct MMO for the load.
    auto *LoadMMO =
        MF.getMachineMemOperand(&SrcMMO, CurrOffset, CopyTy.getSizeInBytes());

    // Create the load.
    Register LoadPtr = Src;
    if (CurrOffset != 0) {
      auto Offset =
          MIB.buildConstant(LLT::scalar(PtrTy.getSizeInBits()), CurrOffset);
      LoadPtr = MIB.buildPtrAdd(PtrTy, Src, Offset).getReg(0);
    }
    LoadVals.push_back(MIB.buildLoad(CopyTy, LoadPtr, *LoadMMO).getReg(0));
    CurrOffset += CopyTy.getSizeInBytes();
  }

  CurrOffset = 0;
  for (unsigned I = 0; I < MemOps.size(); ++I) {
    LLT CopyTy = MemOps[I];
    // Now store the values loaded.
    auto *StoreMMO =
        MF.getMachineMemOperand(&DstMMO, CurrOffset, CopyTy.getSizeInBytes());

    Register StorePtr = Dst;
    if (CurrOffset != 0) {
      auto Offset =
          MIB.buildConstant(LLT::scalar(PtrTy.getSizeInBits()), CurrOffset);
      StorePtr = MIB.buildPtrAdd(PtrTy, Dst, Offset).getReg(0);
    }
    MIB.buildStore(LoadVals[I], StorePtr, *StoreMMO);
    CurrOffset += CopyTy.getSizeInBytes();
  }
  MI.eraseFromParent();
  return true;
}

bool CombinerHelper::tryCombineMemCpyFamily(MachineInstr &MI, unsigned MaxLen) {
  // This combine is fairly complex so it's not written with a separate
  // matcher function.
  assert(MI.getOpcode() == TargetOpcode::G_INTRINSIC_W_SIDE_EFFECTS);
  Intrinsic::ID ID = (Intrinsic::ID)MI.getIntrinsicID();
  assert((ID == Intrinsic::memcpy || ID == Intrinsic::memmove ||
          ID == Intrinsic::memset) &&
         "Expected a memcpy like intrinsic");

  auto MMOIt = MI.memoperands_begin();
  const MachineMemOperand *MemOp = *MMOIt;
  bool IsVolatile = MemOp->isVolatile();
  // Don't try to optimize volatile.
  if (IsVolatile)
    return false;

  unsigned DstAlign = MemOp->getBaseAlignment();
  unsigned SrcAlign = 0;
  Register Dst = MI.getOperand(1).getReg();
  Register Src = MI.getOperand(2).getReg();
  Register Len = MI.getOperand(3).getReg();

  if (ID != Intrinsic::memset) {
    assert(MMOIt != MI.memoperands_end() && "Expected a second MMO on MI");
    MemOp = *(++MMOIt);
    SrcAlign = MemOp->getBaseAlignment();
  }

  // See if this is a constant length copy
  auto LenVRegAndVal = getConstantVRegValWithLookThrough(Len, MRI);
  if (!LenVRegAndVal)
    return false; // Leave it to the legalizer to lower it to a libcall.
  unsigned KnownLen = LenVRegAndVal->Value;

  if (KnownLen == 0) {
    MI.eraseFromParent();
    return true;
  }

  if (MaxLen && KnownLen > MaxLen)
    return false;

  if (ID == Intrinsic::memcpy)
    return optimizeMemcpy(MI, Dst, Src, KnownLen, DstAlign, SrcAlign, IsVolatile);
  if (ID == Intrinsic::memmove)
    return optimizeMemmove(MI, Dst, Src, KnownLen, DstAlign, SrcAlign, IsVolatile);
  if (ID == Intrinsic::memset)
    return optimizeMemset(MI, Dst, Src, KnownLen, DstAlign, IsVolatile);
  return false;
}

bool CombinerHelper::tryCombine(MachineInstr &MI) {
  if (tryCombineCopy(MI))
    return true;
  if (tryCombineExtendingLoads(MI))
    return true;
  if (tryCombineIndexedLoadStore(MI))
    return true;
  return false;
}
