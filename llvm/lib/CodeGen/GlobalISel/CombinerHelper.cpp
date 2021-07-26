//===-- lib/CodeGen/GlobalISel/GICombinerHelper.cpp -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "llvm/CodeGen/GlobalISel/CombinerHelper.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/CodeGen/GlobalISel/Combiner.h"
#include "llvm/CodeGen/GlobalISel/GISelChangeObserver.h"
#include "llvm/CodeGen/GlobalISel/GISelKnownBits.h"
#include "llvm/CodeGen/GlobalISel/GenericMachineInstrs.h"
#include "llvm/CodeGen/GlobalISel/LegalizerInfo.h"
#include "llvm/CodeGen/GlobalISel/MIPatternMatch.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/LowLevelType.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Target/TargetMachine.h"
#include <tuple>

#define DEBUG_TYPE "gi-combiner"

using namespace llvm;
using namespace MIPatternMatch;

// Option to allow testing of the combiner while no targets know about indexed
// addressing.
static cl::opt<bool>
    ForceLegalIndexing("force-legal-indexing", cl::Hidden, cl::init(false),
                       cl::desc("Force all indexed operations to be "
                                "legal for the GlobalISel combiner"));

CombinerHelper::CombinerHelper(GISelChangeObserver &Observer,
                               MachineIRBuilder &B, GISelKnownBits *KB,
                               MachineDominatorTree *MDT,
                               const LegalizerInfo *LI)
    : Builder(B), MRI(Builder.getMF().getRegInfo()), Observer(Observer),
      KB(KB), MDT(MDT), LI(LI) {
  (void)this->KB;
}

const TargetLowering &CombinerHelper::getTargetLowering() const {
  return *Builder.getMF().getSubtarget().getTargetLowering();
}

/// \returns The little endian in-memory byte position of byte \p I in a
/// \p ByteWidth bytes wide type.
///
/// E.g. Given a 4-byte type x, x[0] -> byte 0
static unsigned littleEndianByteAt(const unsigned ByteWidth, const unsigned I) {
  assert(I < ByteWidth && "I must be in [0, ByteWidth)");
  return I;
}

/// \returns The big endian in-memory byte position of byte \p I in a
/// \p ByteWidth bytes wide type.
///
/// E.g. Given a 4-byte type x, x[0] -> byte 3
static unsigned bigEndianByteAt(const unsigned ByteWidth, const unsigned I) {
  assert(I < ByteWidth && "I must be in [0, ByteWidth)");
  return ByteWidth - I - 1;
}

/// Given a map from byte offsets in memory to indices in a load/store,
/// determine if that map corresponds to a little or big endian byte pattern.
///
/// \param MemOffset2Idx maps memory offsets to address offsets.
/// \param LowestIdx is the lowest index in \p MemOffset2Idx.
///
/// \returns true if the map corresponds to a big endian byte pattern, false
/// if it corresponds to a little endian byte pattern, and None otherwise.
///
/// E.g. given a 32-bit type x, and x[AddrOffset], the in-memory byte patterns
/// are as follows:
///
/// AddrOffset   Little endian    Big endian
/// 0            0                3
/// 1            1                2
/// 2            2                1
/// 3            3                0
static Optional<bool>
isBigEndian(const SmallDenseMap<int64_t, int64_t, 8> &MemOffset2Idx,
            int64_t LowestIdx) {
  // Need at least two byte positions to decide on endianness.
  unsigned Width = MemOffset2Idx.size();
  if (Width < 2)
    return None;
  bool BigEndian = true, LittleEndian = true;
  for (unsigned MemOffset = 0; MemOffset < Width; ++ MemOffset) {
    auto MemOffsetAndIdx = MemOffset2Idx.find(MemOffset);
    if (MemOffsetAndIdx == MemOffset2Idx.end())
      return None;
    const int64_t Idx = MemOffsetAndIdx->second - LowestIdx;
    assert(Idx >= 0 && "Expected non-negative byte offset?");
    LittleEndian &= Idx == littleEndianByteAt(Width, MemOffset);
    BigEndian &= Idx == bigEndianByteAt(Width, MemOffset);
    if (!BigEndian && !LittleEndian)
      return None;
  }

  assert((BigEndian != LittleEndian) &&
         "Pattern cannot be both big and little endian!");
  return BigEndian;
}

bool CombinerHelper::isLegalOrBeforeLegalizer(
    const LegalityQuery &Query) const {
  return !LI || LI->getAction(Query).Action == LegalizeActions::Legal;
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
  return canReplaceReg(DstReg, SrcReg, MRI);
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
  ArrayRef<int> Mask = MI.getOperand(3).getShuffleMask();
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
                                  const LLT TyForCandidate,
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
  GAnyLoad *LoadMI = dyn_cast<GAnyLoad>(&MI);
  if (!LoadMI)
    return false;

  Register LoadReg = LoadMI->getDstReg();

  LLT LoadValueTy = MRI.getType(LoadReg);
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
  unsigned PreferredOpcode =
      isa<GLoad>(&MI)
          ? TargetOpcode::G_ANYEXT
          : isa<GSExtLoad>(&MI) ? TargetOpcode::G_SEXT : TargetOpcode::G_ZEXT;
  Preferred = {LLT(), PreferredOpcode, nullptr};
  for (auto &UseMI : MRI.use_nodbg_instructions(LoadReg)) {
    if (UseMI.getOpcode() == TargetOpcode::G_SEXT ||
        UseMI.getOpcode() == TargetOpcode::G_ZEXT ||
        (UseMI.getOpcode() == TargetOpcode::G_ANYEXT)) {
      const auto &MMO = LoadMI->getMMO();
      // For atomics, only form anyextending loads.
      if (MMO.isAtomic() && UseMI.getOpcode() != TargetOpcode::G_ANYEXT)
        continue;
      // Check for legality.
      if (LI) {
        LegalityQuery::MemDesc MMDesc;
        MMDesc.MemoryTy = MMO.getMemoryType();
        MMDesc.AlignInBits = MMO.getAlign().value() * 8;
        MMDesc.Ordering = MMO.getSuccessOrdering();
        LLT UseTy = MRI.getType(UseMI.getOperand(0).getReg());
        LLT SrcTy = MRI.getType(LoadMI->getPointerReg());
        if (LI->getAction({LoadMI->getOpcode(), {UseTy, SrcTy}, {MMDesc}})
                .Action != LegalizeActions::Legal)
          continue;
      }
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
      const LLT UseDstTy = MRI.getType(UseDstReg);
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

bool CombinerHelper::isPredecessor(const MachineInstr &DefMI,
                                   const MachineInstr &UseMI) {
  assert(!DefMI.isDebugInstr() && !UseMI.isDebugInstr() &&
         "shouldn't consider debug uses");
  assert(DefMI.getParent() == UseMI.getParent());
  if (&DefMI == &UseMI)
    return false;
  const MachineBasicBlock &MBB = *DefMI.getParent();
  auto DefOrUse = find_if(MBB, [&DefMI, &UseMI](const MachineInstr &MI) {
    return &MI == &DefMI || &MI == &UseMI;
  });
  if (DefOrUse == MBB.end())
    llvm_unreachable("Block must contain both DefMI and UseMI!");
  return &*DefOrUse == &DefMI;
}

bool CombinerHelper::dominates(const MachineInstr &DefMI,
                               const MachineInstr &UseMI) {
  assert(!DefMI.isDebugInstr() && !UseMI.isDebugInstr() &&
         "shouldn't consider debug uses");
  if (MDT)
    return MDT->dominates(&DefMI, &UseMI);
  else if (DefMI.getParent() != UseMI.getParent())
    return false;

  return isPredecessor(DefMI, UseMI);
}

bool CombinerHelper::matchSextTruncSextLoad(MachineInstr &MI) {
  assert(MI.getOpcode() == TargetOpcode::G_SEXT_INREG);
  Register SrcReg = MI.getOperand(1).getReg();
  Register LoadUser = SrcReg;

  if (MRI.getType(SrcReg).isVector())
    return false;

  Register TruncSrc;
  if (mi_match(SrcReg, MRI, m_GTrunc(m_Reg(TruncSrc))))
    LoadUser = TruncSrc;

  uint64_t SizeInBits = MI.getOperand(2).getImm();
  // If the source is a G_SEXTLOAD from the same bit width, then we don't
  // need any extend at all, just a truncate.
  if (auto *LoadMI = getOpcodeDef<GSExtLoad>(LoadUser, MRI)) {
    // If truncating more than the original extended value, abort.
    auto LoadSizeBits = LoadMI->getMemSizeInBits();
    if (TruncSrc && MRI.getType(TruncSrc).getSizeInBits() < LoadSizeBits)
      return false;
    if (LoadSizeBits == SizeInBits)
      return true;
  }
  return false;
}

void CombinerHelper::applySextTruncSextLoad(MachineInstr &MI) {
  assert(MI.getOpcode() == TargetOpcode::G_SEXT_INREG);
  Builder.setInstrAndDebugLoc(MI);
  Builder.buildCopy(MI.getOperand(0).getReg(), MI.getOperand(1).getReg());
  MI.eraseFromParent();
}

bool CombinerHelper::matchSextInRegOfLoad(
    MachineInstr &MI, std::tuple<Register, unsigned> &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_SEXT_INREG);

  // Only supports scalars for now.
  if (MRI.getType(MI.getOperand(0).getReg()).isVector())
    return false;

  Register SrcReg = MI.getOperand(1).getReg();
  auto *LoadDef = getOpcodeDef<GLoad>(SrcReg, MRI);
  if (!LoadDef || !MRI.hasOneNonDBGUse(LoadDef->getOperand(0).getReg()) ||
      !LoadDef->isSimple())
    return false;

  // If the sign extend extends from a narrower width than the load's width,
  // then we can narrow the load width when we combine to a G_SEXTLOAD.
  // Avoid widening the load at all.
  unsigned NewSizeBits = std::min((uint64_t)MI.getOperand(2).getImm(),
                                  LoadDef->getMemSizeInBits());

  // Don't generate G_SEXTLOADs with a < 1 byte width.
  if (NewSizeBits < 8)
    return false;
  // Don't bother creating a non-power-2 sextload, it will likely be broken up
  // anyway for most targets.
  if (!isPowerOf2_32(NewSizeBits))
    return false;
  MatchInfo = std::make_tuple(LoadDef->getDstReg(), NewSizeBits);
  return true;
}

void CombinerHelper::applySextInRegOfLoad(
    MachineInstr &MI, std::tuple<Register, unsigned> &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_SEXT_INREG);
  Register LoadReg;
  unsigned ScalarSizeBits;
  std::tie(LoadReg, ScalarSizeBits) = MatchInfo;
  GLoad *LoadDef = cast<GLoad>(MRI.getVRegDef(LoadReg));

  // If we have the following:
  // %ld = G_LOAD %ptr, (load 2)
  // %ext = G_SEXT_INREG %ld, 8
  //    ==>
  // %ld = G_SEXTLOAD %ptr (load 1)

  auto &MMO = LoadDef->getMMO();
  Builder.setInstrAndDebugLoc(*LoadDef);
  auto &MF = Builder.getMF();
  auto PtrInfo = MMO.getPointerInfo();
  auto *NewMMO = MF.getMachineMemOperand(&MMO, PtrInfo, ScalarSizeBits / 8);
  Builder.buildLoadInstr(TargetOpcode::G_SEXTLOAD, MI.getOperand(0).getReg(),
                         LoadDef->getPointerReg(), *NewMMO);
  MI.eraseFromParent();
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
  // FIXME: The following use traversal needs a bail out for patholigical cases.
  for (auto &Use : MRI.use_nodbg_instructions(Base)) {
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
    for (auto &PtrAddUse :
         MRI.use_nodbg_instructions(Use.getOperand(0).getReg())) {
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
  if (!AddrDef || MRI.hasOneNonDBGUse(Addr))
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

  for (auto &UseMI : MRI.use_nodbg_instructions(Addr)) {
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

  // For now, no targets actually support these opcodes so don't waste time
  // running these unless we're forced to for testing.
  if (!ForceLegalIndexing)
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

bool CombinerHelper::matchCombineDivRem(MachineInstr &MI,
                                        MachineInstr *&OtherMI) {
  unsigned Opcode = MI.getOpcode();
  bool IsDiv, IsSigned;

  switch (Opcode) {
  default:
    llvm_unreachable("Unexpected opcode!");
  case TargetOpcode::G_SDIV:
  case TargetOpcode::G_UDIV: {
    IsDiv = true;
    IsSigned = Opcode == TargetOpcode::G_SDIV;
    break;
  }
  case TargetOpcode::G_SREM:
  case TargetOpcode::G_UREM: {
    IsDiv = false;
    IsSigned = Opcode == TargetOpcode::G_SREM;
    break;
  }
  }

  Register Src1 = MI.getOperand(1).getReg();
  unsigned DivOpcode, RemOpcode, DivremOpcode;
  if (IsSigned) {
    DivOpcode = TargetOpcode::G_SDIV;
    RemOpcode = TargetOpcode::G_SREM;
    DivremOpcode = TargetOpcode::G_SDIVREM;
  } else {
    DivOpcode = TargetOpcode::G_UDIV;
    RemOpcode = TargetOpcode::G_UREM;
    DivremOpcode = TargetOpcode::G_UDIVREM;
  }

  if (!isLegalOrBeforeLegalizer({DivremOpcode, {MRI.getType(Src1)}}))
    return false;

  // Combine:
  //   %div:_ = G_[SU]DIV %src1:_, %src2:_
  //   %rem:_ = G_[SU]REM %src1:_, %src2:_
  // into:
  //  %div:_, %rem:_ = G_[SU]DIVREM %src1:_, %src2:_

  // Combine:
  //   %rem:_ = G_[SU]REM %src1:_, %src2:_
  //   %div:_ = G_[SU]DIV %src1:_, %src2:_
  // into:
  //  %div:_, %rem:_ = G_[SU]DIVREM %src1:_, %src2:_

  for (auto &UseMI : MRI.use_nodbg_instructions(Src1)) {
    if (MI.getParent() == UseMI.getParent() &&
        ((IsDiv && UseMI.getOpcode() == RemOpcode) ||
         (!IsDiv && UseMI.getOpcode() == DivOpcode)) &&
        matchEqualDefs(MI.getOperand(2), UseMI.getOperand(2))) {
      OtherMI = &UseMI;
      return true;
    }
  }

  return false;
}

void CombinerHelper::applyCombineDivRem(MachineInstr &MI,
                                        MachineInstr *&OtherMI) {
  unsigned Opcode = MI.getOpcode();
  assert(OtherMI && "OtherMI shouldn't be empty.");

  Register DestDivReg, DestRemReg;
  if (Opcode == TargetOpcode::G_SDIV || Opcode == TargetOpcode::G_UDIV) {
    DestDivReg = MI.getOperand(0).getReg();
    DestRemReg = OtherMI->getOperand(0).getReg();
  } else {
    DestDivReg = OtherMI->getOperand(0).getReg();
    DestRemReg = MI.getOperand(0).getReg();
  }

  bool IsSigned =
      Opcode == TargetOpcode::G_SDIV || Opcode == TargetOpcode::G_SREM;

  // Check which instruction is first in the block so we don't break def-use
  // deps by "moving" the instruction incorrectly.
  if (dominates(MI, *OtherMI))
    Builder.setInstrAndDebugLoc(MI);
  else
    Builder.setInstrAndDebugLoc(*OtherMI);

  Builder.buildInstr(IsSigned ? TargetOpcode::G_SDIVREM
                              : TargetOpcode::G_UDIVREM,
                     {DestDivReg, DestRemReg},
                     {MI.getOperand(1).getReg(), MI.getOperand(2).getReg()});
  MI.eraseFromParent();
  OtherMI->eraseFromParent();
}

bool CombinerHelper::matchOptBrCondByInvertingCond(MachineInstr &MI,
                                                   MachineInstr *&BrCond) {
  assert(MI.getOpcode() == TargetOpcode::G_BR);

  // Try to match the following:
  // bb1:
  //   G_BRCOND %c1, %bb2
  //   G_BR %bb3
  // bb2:
  // ...
  // bb3:

  // The above pattern does not have a fall through to the successor bb2, always
  // resulting in a branch no matter which path is taken. Here we try to find
  // and replace that pattern with conditional branch to bb3 and otherwise
  // fallthrough to bb2. This is generally better for branch predictors.

  MachineBasicBlock *MBB = MI.getParent();
  MachineBasicBlock::iterator BrIt(MI);
  if (BrIt == MBB->begin())
    return false;
  assert(std::next(BrIt) == MBB->end() && "expected G_BR to be a terminator");

  BrCond = &*std::prev(BrIt);
  if (BrCond->getOpcode() != TargetOpcode::G_BRCOND)
    return false;

  // Check that the next block is the conditional branch target. Also make sure
  // that it isn't the same as the G_BR's target (otherwise, this will loop.)
  MachineBasicBlock *BrCondTarget = BrCond->getOperand(1).getMBB();
  return BrCondTarget != MI.getOperand(0).getMBB() &&
         MBB->isLayoutSuccessor(BrCondTarget);
}

void CombinerHelper::applyOptBrCondByInvertingCond(MachineInstr &MI,
                                                   MachineInstr *&BrCond) {
  MachineBasicBlock *BrTarget = MI.getOperand(0).getMBB();
  Builder.setInstrAndDebugLoc(*BrCond);
  LLT Ty = MRI.getType(BrCond->getOperand(0).getReg());
  // FIXME: Does int/fp matter for this? If so, we might need to restrict
  // this to i1 only since we might not know for sure what kind of
  // compare generated the condition value.
  auto True = Builder.buildConstant(
      Ty, getICmpTrueVal(getTargetLowering(), false, false));
  auto Xor = Builder.buildXor(Ty, BrCond->getOperand(0), True);

  auto *FallthroughBB = BrCond->getOperand(1).getMBB();
  Observer.changingInstr(MI);
  MI.getOperand(0).setMBB(FallthroughBB);
  Observer.changedInstr(MI);

  // Change the conditional branch to use the inverted condition and
  // new target block.
  Observer.changingInstr(*BrCond);
  BrCond->getOperand(0).setReg(Xor.getReg(0));
  BrCond->getOperand(1).setMBB(BrTarget);
  Observer.changedInstr(*BrCond);
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
static bool findGISelOptimalMemOpLowering(std::vector<LLT> &MemOps,
                                          unsigned Limit, const MemOp &Op,
                                          unsigned DstAS, unsigned SrcAS,
                                          const AttributeList &FuncAttributes,
                                          const TargetLowering &TLI) {
  if (Op.isMemcpyWithFixedDstAlign() && Op.getSrcAlign() < Op.getDstAlign())
    return false;

  LLT Ty = TLI.getOptimalMemOpLLT(Op, FuncAttributes);

  if (Ty == LLT()) {
    // Use the largest scalar type whose alignment constraints are satisfied.
    // We only need to check DstAlign here as SrcAlign is always greater or
    // equal to DstAlign (or zero).
    Ty = LLT::scalar(64);
    if (Op.isFixedDstAlign())
      while (Op.getDstAlign() < Ty.getSizeInBytes() &&
             !TLI.allowsMisalignedMemoryAccesses(Ty, DstAS, Op.getDstAlign()))
        Ty = LLT::scalar(Ty.getSizeInBytes());
    assert(Ty.getSizeInBits() > 0 && "Could not find valid type");
    // FIXME: check for the largest legal type we can load/store to.
  }

  unsigned NumMemOps = 0;
  uint64_t Size = Op.size();
  while (Size) {
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
      if (NumMemOps && Op.allowOverlap() && NewTySize < Size &&
          TLI.allowsMisalignedMemoryAccesses(
              VT, DstAS, Op.isFixedDstAlign() ? Op.getDstAlign() : Align(1),
              MachineMemOperand::MONone, &Fast) &&
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
    return FixedVectorType::get(IntegerType::get(C, Ty.getScalarSizeInBits()),
                                Ty.getNumElements());
  return IntegerType::get(C, Ty.getSizeInBits());
}

// Get a vectorized representation of the memset value operand, GISel edition.
static Register getMemsetValue(Register Val, LLT Ty, MachineIRBuilder &MIB) {
  MachineRegisterInfo &MRI = *MIB.getMRI();
  unsigned NumBits = Ty.getScalarSizeInBits();
  auto ValVRegAndVal = getConstantVRegValWithLookThrough(Val, MRI);
  if (!Ty.isVector() && ValVRegAndVal) {
    APInt Scalar = ValVRegAndVal->Value.truncOrSelf(8);
    APInt SplatVal = APInt::getSplat(NumBits, Scalar);
    return MIB.buildConstant(Ty, SplatVal).getReg(0);
  }

  // Extend the byte value to the larger type, and then multiply by a magic
  // value 0x010101... in order to replicate it across every byte.
  // Unless it's zero, in which case just emit a larger G_CONSTANT 0.
  if (ValVRegAndVal && ValVRegAndVal->Value == 0) {
    return MIB.buildConstant(Ty, 0).getReg(0);
  }

  LLT ExtType = Ty.getScalarType();
  auto ZExt = MIB.buildZExtOrTrunc(ExtType, Val);
  if (NumBits > 8) {
    APInt Magic = APInt::getSplat(NumBits, APInt(8, 0x01));
    auto MagicMI = MIB.buildConstant(ExtType, Magic);
    Val = MIB.buildMul(ExtType, ZExt, MagicMI).getReg(0);
  }

  // For vector types create a G_BUILD_VECTOR.
  if (Ty.isVector())
    Val = MIB.buildSplatVector(Ty, Val).getReg(0);

  return Val;
}

bool CombinerHelper::optimizeMemset(MachineInstr &MI, Register Dst,
                                    Register Val, uint64_t KnownLen,
                                    Align Alignment, bool IsVolatile) {
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

  if (!findGISelOptimalMemOpLowering(MemOps, Limit,
                                     MemOp::Set(KnownLen, DstAlignCanChange,
                                                Alignment,
                                                /*IsZeroMemset=*/IsZeroVal,
                                                /*IsVolatile=*/IsVolatile),
                                     DstPtrInfo.getAddrSpace(), ~0u,
                                     MF.getFunction().getAttributes(), TLI))
    return false;

  if (DstAlignCanChange) {
    // Get an estimate of the type from the LLT.
    Type *IRTy = getTypeForLLT(MemOps[0], C);
    Align NewAlign = DL.getABITypeAlign(IRTy);
    if (NewAlign > Alignment) {
      Alignment = NewAlign;
      unsigned FI = FIDef->getOperand(1).getIndex();
      // Give the stack frame object a larger alignment if needed.
      if (MFI.getObjectAlign(FI) < Alignment)
        MFI.setObjectAlignment(FI, Alignment);
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
        MF.getMachineMemOperand(&DstMMO, DstOff, Ty);

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

bool CombinerHelper::tryEmitMemcpyInline(MachineInstr &MI) {
  assert(MI.getOpcode() == TargetOpcode::G_MEMCPY_INLINE);

  Register Dst = MI.getOperand(0).getReg();
  Register Src = MI.getOperand(1).getReg();
  Register Len = MI.getOperand(2).getReg();

  const auto *MMOIt = MI.memoperands_begin();
  const MachineMemOperand *MemOp = *MMOIt;
  bool IsVolatile = MemOp->isVolatile();

  // See if this is a constant length copy
  auto LenVRegAndVal = getConstantVRegValWithLookThrough(Len, MRI);
  // FIXME: support dynamically sized G_MEMCPY_INLINE
  assert(LenVRegAndVal.hasValue() &&
         "inline memcpy with dynamic size is not yet supported");
  uint64_t KnownLen = LenVRegAndVal->Value.getZExtValue();
  if (KnownLen == 0) {
    MI.eraseFromParent();
    return true;
  }

  const auto &DstMMO = **MI.memoperands_begin();
  const auto &SrcMMO = **std::next(MI.memoperands_begin());
  Align DstAlign = DstMMO.getBaseAlign();
  Align SrcAlign = SrcMMO.getBaseAlign();

  return tryEmitMemcpyInline(MI, Dst, Src, KnownLen, DstAlign, SrcAlign,
                             IsVolatile);
}

bool CombinerHelper::tryEmitMemcpyInline(MachineInstr &MI, Register Dst,
                                         Register Src, uint64_t KnownLen,
                                         Align DstAlign, Align SrcAlign,
                                         bool IsVolatile) {
  assert(MI.getOpcode() == TargetOpcode::G_MEMCPY_INLINE);
  return optimizeMemcpy(MI, Dst, Src, KnownLen,
                        std::numeric_limits<uint64_t>::max(), DstAlign,
                        SrcAlign, IsVolatile);
}

bool CombinerHelper::optimizeMemcpy(MachineInstr &MI, Register Dst,
                                    Register Src, uint64_t KnownLen,
                                    uint64_t Limit, Align DstAlign,
                                    Align SrcAlign, bool IsVolatile) {
  auto &MF = *MI.getParent()->getParent();
  const auto &TLI = *MF.getSubtarget().getTargetLowering();
  auto &DL = MF.getDataLayout();
  LLVMContext &C = MF.getFunction().getContext();

  assert(KnownLen != 0 && "Have a zero length memcpy length!");

  bool DstAlignCanChange = false;
  MachineFrameInfo &MFI = MF.getFrameInfo();
  Align Alignment = commonAlignment(DstAlign, SrcAlign);

  MachineInstr *FIDef = getOpcodeDef(TargetOpcode::G_FRAME_INDEX, Dst, MRI);
  if (FIDef && !MFI.isFixedObjectIndex(FIDef->getOperand(1).getIndex()))
    DstAlignCanChange = true;

  // FIXME: infer better src pointer alignment like SelectionDAG does here.
  // FIXME: also use the equivalent of isMemSrcFromConstant and alwaysinlining
  // if the memcpy is in a tail call position.

  std::vector<LLT> MemOps;

  const auto &DstMMO = **MI.memoperands_begin();
  const auto &SrcMMO = **std::next(MI.memoperands_begin());
  MachinePointerInfo DstPtrInfo = DstMMO.getPointerInfo();
  MachinePointerInfo SrcPtrInfo = SrcMMO.getPointerInfo();

  if (!findGISelOptimalMemOpLowering(
          MemOps, Limit,
          MemOp::Copy(KnownLen, DstAlignCanChange, Alignment, SrcAlign,
                      IsVolatile),
          DstPtrInfo.getAddrSpace(), SrcPtrInfo.getAddrSpace(),
          MF.getFunction().getAttributes(), TLI))
    return false;

  if (DstAlignCanChange) {
    // Get an estimate of the type from the LLT.
    Type *IRTy = getTypeForLLT(MemOps[0], C);
    Align NewAlign = DL.getABITypeAlign(IRTy);

    // Don't promote to an alignment that would require dynamic stack
    // realignment.
    const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
    if (!TRI->hasStackRealignment(MF))
      while (NewAlign > Alignment && DL.exceedsNaturalStackAlignment(NewAlign))
        NewAlign = NewAlign / 2;

    if (NewAlign > Alignment) {
      Alignment = NewAlign;
      unsigned FI = FIDef->getOperand(1).getIndex();
      // Give the stack frame object a larger alignment if needed.
      if (MFI.getObjectAlign(FI) < Alignment)
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
                                     Register Src, uint64_t KnownLen,
                                     Align DstAlign, Align SrcAlign,
                                     bool IsVolatile) {
  auto &MF = *MI.getParent()->getParent();
  const auto &TLI = *MF.getSubtarget().getTargetLowering();
  auto &DL = MF.getDataLayout();
  LLVMContext &C = MF.getFunction().getContext();

  assert(KnownLen != 0 && "Have a zero length memmove length!");

  bool DstAlignCanChange = false;
  MachineFrameInfo &MFI = MF.getFrameInfo();
  bool OptSize = shouldLowerMemFuncForSize(MF);
  Align Alignment = commonAlignment(DstAlign, SrcAlign);

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
          MemOps, Limit,
          MemOp::Copy(KnownLen, DstAlignCanChange, Alignment, SrcAlign,
                      /*IsVolatile*/ true),
          DstPtrInfo.getAddrSpace(), SrcPtrInfo.getAddrSpace(),
          MF.getFunction().getAttributes(), TLI))
    return false;

  if (DstAlignCanChange) {
    // Get an estimate of the type from the LLT.
    Type *IRTy = getTypeForLLT(MemOps[0], C);
    Align NewAlign = DL.getABITypeAlign(IRTy);

    // Don't promote to an alignment that would require dynamic stack
    // realignment.
    const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
    if (!TRI->hasStackRealignment(MF))
      while (NewAlign > Alignment && DL.exceedsNaturalStackAlignment(NewAlign))
        NewAlign = NewAlign / 2;

    if (NewAlign > Alignment) {
      Alignment = NewAlign;
      unsigned FI = FIDef->getOperand(1).getIndex();
      // Give the stack frame object a larger alignment if needed.
      if (MFI.getObjectAlign(FI) < Alignment)
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
  const unsigned Opc = MI.getOpcode();
  // This combine is fairly complex so it's not written with a separate
  // matcher function.
  assert((Opc == TargetOpcode::G_MEMCPY || Opc == TargetOpcode::G_MEMMOVE ||
          Opc == TargetOpcode::G_MEMSET) && "Expected memcpy like instruction");

  auto MMOIt = MI.memoperands_begin();
  const MachineMemOperand *MemOp = *MMOIt;

  Align DstAlign = MemOp->getBaseAlign();
  Align SrcAlign;
  Register Dst = MI.getOperand(0).getReg();
  Register Src = MI.getOperand(1).getReg();
  Register Len = MI.getOperand(2).getReg();

  if (Opc != TargetOpcode::G_MEMSET) {
    assert(MMOIt != MI.memoperands_end() && "Expected a second MMO on MI");
    MemOp = *(++MMOIt);
    SrcAlign = MemOp->getBaseAlign();
  }

  // See if this is a constant length copy
  auto LenVRegAndVal = getConstantVRegValWithLookThrough(Len, MRI);
  if (!LenVRegAndVal)
    return false; // Leave it to the legalizer to lower it to a libcall.
  uint64_t KnownLen = LenVRegAndVal->Value.getZExtValue();

  if (KnownLen == 0) {
    MI.eraseFromParent();
    return true;
  }

  bool IsVolatile = MemOp->isVolatile();
  if (Opc == TargetOpcode::G_MEMCPY_INLINE)
    return tryEmitMemcpyInline(MI, Dst, Src, KnownLen, DstAlign, SrcAlign,
                               IsVolatile);

  // Don't try to optimize volatile.
  if (IsVolatile)
    return false;

  if (MaxLen && KnownLen > MaxLen)
    return false;

  if (Opc == TargetOpcode::G_MEMCPY) {
    auto &MF = *MI.getParent()->getParent();
    const auto &TLI = *MF.getSubtarget().getTargetLowering();
    bool OptSize = shouldLowerMemFuncForSize(MF);
    uint64_t Limit = TLI.getMaxStoresPerMemcpy(OptSize);
    return optimizeMemcpy(MI, Dst, Src, KnownLen, Limit, DstAlign, SrcAlign,
                          IsVolatile);
  }
  if (Opc == TargetOpcode::G_MEMMOVE)
    return optimizeMemmove(MI, Dst, Src, KnownLen, DstAlign, SrcAlign, IsVolatile);
  if (Opc == TargetOpcode::G_MEMSET)
    return optimizeMemset(MI, Dst, Src, KnownLen, DstAlign, IsVolatile);
  return false;
}

static Optional<APFloat> constantFoldFpUnary(unsigned Opcode, LLT DstTy,
                                             const Register Op,
                                             const MachineRegisterInfo &MRI) {
  const ConstantFP *MaybeCst = getConstantFPVRegVal(Op, MRI);
  if (!MaybeCst)
    return None;

  APFloat V = MaybeCst->getValueAPF();
  switch (Opcode) {
  default:
    llvm_unreachable("Unexpected opcode!");
  case TargetOpcode::G_FNEG: {
    V.changeSign();
    return V;
  }
  case TargetOpcode::G_FABS: {
    V.clearSign();
    return V;
  }
  case TargetOpcode::G_FPTRUNC:
    break;
  case TargetOpcode::G_FSQRT: {
    bool Unused;
    V.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven, &Unused);
    V = APFloat(sqrt(V.convertToDouble()));
    break;
  }
  case TargetOpcode::G_FLOG2: {
    bool Unused;
    V.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven, &Unused);
    V = APFloat(log2(V.convertToDouble()));
    break;
  }
  }
  // Convert `APFloat` to appropriate IEEE type depending on `DstTy`. Otherwise,
  // `buildFConstant` will assert on size mismatch. Only `G_FPTRUNC`, `G_FSQRT`,
  // and `G_FLOG2` reach here.
  bool Unused;
  V.convert(getFltSemanticForLLT(DstTy), APFloat::rmNearestTiesToEven, &Unused);
  return V;
}

bool CombinerHelper::matchCombineConstantFoldFpUnary(MachineInstr &MI,
                                                     Optional<APFloat> &Cst) {
  Register DstReg = MI.getOperand(0).getReg();
  Register SrcReg = MI.getOperand(1).getReg();
  LLT DstTy = MRI.getType(DstReg);
  Cst = constantFoldFpUnary(MI.getOpcode(), DstTy, SrcReg, MRI);
  return Cst.hasValue();
}

void CombinerHelper::applyCombineConstantFoldFpUnary(MachineInstr &MI,
                                                     Optional<APFloat> &Cst) {
  assert(Cst.hasValue() && "Optional is unexpectedly empty!");
  Builder.setInstrAndDebugLoc(MI);
  MachineFunction &MF = Builder.getMF();
  auto *FPVal = ConstantFP::get(MF.getFunction().getContext(), *Cst);
  Register DstReg = MI.getOperand(0).getReg();
  Builder.buildFConstant(DstReg, *FPVal);
  MI.eraseFromParent();
}

bool CombinerHelper::matchPtrAddImmedChain(MachineInstr &MI,
                                           PtrAddChain &MatchInfo) {
  // We're trying to match the following pattern:
  //   %t1 = G_PTR_ADD %base, G_CONSTANT imm1
  //   %root = G_PTR_ADD %t1, G_CONSTANT imm2
  // -->
  //   %root = G_PTR_ADD %base, G_CONSTANT (imm1 + imm2)

  if (MI.getOpcode() != TargetOpcode::G_PTR_ADD)
    return false;

  Register Add2 = MI.getOperand(1).getReg();
  Register Imm1 = MI.getOperand(2).getReg();
  auto MaybeImmVal = getConstantVRegValWithLookThrough(Imm1, MRI);
  if (!MaybeImmVal)
    return false;

  // Don't do this combine if there multiple uses of the first PTR_ADD,
  // since we may be able to compute the second PTR_ADD as an immediate
  // offset anyway. Folding the first offset into the second may cause us
  // to go beyond the bounds of our legal addressing modes.
  if (!MRI.hasOneNonDBGUse(Add2))
    return false;

  MachineInstr *Add2Def = MRI.getUniqueVRegDef(Add2);
  if (!Add2Def || Add2Def->getOpcode() != TargetOpcode::G_PTR_ADD)
    return false;

  Register Base = Add2Def->getOperand(1).getReg();
  Register Imm2 = Add2Def->getOperand(2).getReg();
  auto MaybeImm2Val = getConstantVRegValWithLookThrough(Imm2, MRI);
  if (!MaybeImm2Val)
    return false;

  // Pass the combined immediate to the apply function.
  MatchInfo.Imm = (MaybeImmVal->Value + MaybeImm2Val->Value).getSExtValue();
  MatchInfo.Base = Base;
  return true;
}

void CombinerHelper::applyPtrAddImmedChain(MachineInstr &MI,
                                           PtrAddChain &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_PTR_ADD && "Expected G_PTR_ADD");
  MachineIRBuilder MIB(MI);
  LLT OffsetTy = MRI.getType(MI.getOperand(2).getReg());
  auto NewOffset = MIB.buildConstant(OffsetTy, MatchInfo.Imm);
  Observer.changingInstr(MI);
  MI.getOperand(1).setReg(MatchInfo.Base);
  MI.getOperand(2).setReg(NewOffset.getReg(0));
  Observer.changedInstr(MI);
}

bool CombinerHelper::matchShiftImmedChain(MachineInstr &MI,
                                          RegisterImmPair &MatchInfo) {
  // We're trying to match the following pattern with any of
  // G_SHL/G_ASHR/G_LSHR/G_SSHLSAT/G_USHLSAT shift instructions:
  //   %t1 = SHIFT %base, G_CONSTANT imm1
  //   %root = SHIFT %t1, G_CONSTANT imm2
  // -->
  //   %root = SHIFT %base, G_CONSTANT (imm1 + imm2)

  unsigned Opcode = MI.getOpcode();
  assert((Opcode == TargetOpcode::G_SHL || Opcode == TargetOpcode::G_ASHR ||
          Opcode == TargetOpcode::G_LSHR || Opcode == TargetOpcode::G_SSHLSAT ||
          Opcode == TargetOpcode::G_USHLSAT) &&
         "Expected G_SHL, G_ASHR, G_LSHR, G_SSHLSAT or G_USHLSAT");

  Register Shl2 = MI.getOperand(1).getReg();
  Register Imm1 = MI.getOperand(2).getReg();
  auto MaybeImmVal = getConstantVRegValWithLookThrough(Imm1, MRI);
  if (!MaybeImmVal)
    return false;

  MachineInstr *Shl2Def = MRI.getUniqueVRegDef(Shl2);
  if (Shl2Def->getOpcode() != Opcode)
    return false;

  Register Base = Shl2Def->getOperand(1).getReg();
  Register Imm2 = Shl2Def->getOperand(2).getReg();
  auto MaybeImm2Val = getConstantVRegValWithLookThrough(Imm2, MRI);
  if (!MaybeImm2Val)
    return false;

  // Pass the combined immediate to the apply function.
  MatchInfo.Imm =
      (MaybeImmVal->Value.getSExtValue() + MaybeImm2Val->Value).getSExtValue();
  MatchInfo.Reg = Base;

  // There is no simple replacement for a saturating unsigned left shift that
  // exceeds the scalar size.
  if (Opcode == TargetOpcode::G_USHLSAT &&
      MatchInfo.Imm >= MRI.getType(Shl2).getScalarSizeInBits())
    return false;

  return true;
}

void CombinerHelper::applyShiftImmedChain(MachineInstr &MI,
                                          RegisterImmPair &MatchInfo) {
  unsigned Opcode = MI.getOpcode();
  assert((Opcode == TargetOpcode::G_SHL || Opcode == TargetOpcode::G_ASHR ||
          Opcode == TargetOpcode::G_LSHR || Opcode == TargetOpcode::G_SSHLSAT ||
          Opcode == TargetOpcode::G_USHLSAT) &&
         "Expected G_SHL, G_ASHR, G_LSHR, G_SSHLSAT or G_USHLSAT");

  Builder.setInstrAndDebugLoc(MI);
  LLT Ty = MRI.getType(MI.getOperand(1).getReg());
  unsigned const ScalarSizeInBits = Ty.getScalarSizeInBits();
  auto Imm = MatchInfo.Imm;

  if (Imm >= ScalarSizeInBits) {
    // Any logical shift that exceeds scalar size will produce zero.
    if (Opcode == TargetOpcode::G_SHL || Opcode == TargetOpcode::G_LSHR) {
      Builder.buildConstant(MI.getOperand(0), 0);
      MI.eraseFromParent();
      return;
    }
    // Arithmetic shift and saturating signed left shift have no effect beyond
    // scalar size.
    Imm = ScalarSizeInBits - 1;
  }

  LLT ImmTy = MRI.getType(MI.getOperand(2).getReg());
  Register NewImm = Builder.buildConstant(ImmTy, Imm).getReg(0);
  Observer.changingInstr(MI);
  MI.getOperand(1).setReg(MatchInfo.Reg);
  MI.getOperand(2).setReg(NewImm);
  Observer.changedInstr(MI);
}

bool CombinerHelper::matchShiftOfShiftedLogic(MachineInstr &MI,
                                              ShiftOfShiftedLogic &MatchInfo) {
  // We're trying to match the following pattern with any of
  // G_SHL/G_ASHR/G_LSHR/G_USHLSAT/G_SSHLSAT shift instructions in combination
  // with any of G_AND/G_OR/G_XOR logic instructions.
  //   %t1 = SHIFT %X, G_CONSTANT C0
  //   %t2 = LOGIC %t1, %Y
  //   %root = SHIFT %t2, G_CONSTANT C1
  // -->
  //   %t3 = SHIFT %X, G_CONSTANT (C0+C1)
  //   %t4 = SHIFT %Y, G_CONSTANT C1
  //   %root = LOGIC %t3, %t4
  unsigned ShiftOpcode = MI.getOpcode();
  assert((ShiftOpcode == TargetOpcode::G_SHL ||
          ShiftOpcode == TargetOpcode::G_ASHR ||
          ShiftOpcode == TargetOpcode::G_LSHR ||
          ShiftOpcode == TargetOpcode::G_USHLSAT ||
          ShiftOpcode == TargetOpcode::G_SSHLSAT) &&
         "Expected G_SHL, G_ASHR, G_LSHR, G_USHLSAT and G_SSHLSAT");

  // Match a one-use bitwise logic op.
  Register LogicDest = MI.getOperand(1).getReg();
  if (!MRI.hasOneNonDBGUse(LogicDest))
    return false;

  MachineInstr *LogicMI = MRI.getUniqueVRegDef(LogicDest);
  unsigned LogicOpcode = LogicMI->getOpcode();
  if (LogicOpcode != TargetOpcode::G_AND && LogicOpcode != TargetOpcode::G_OR &&
      LogicOpcode != TargetOpcode::G_XOR)
    return false;

  // Find a matching one-use shift by constant.
  const Register C1 = MI.getOperand(2).getReg();
  auto MaybeImmVal = getConstantVRegValWithLookThrough(C1, MRI);
  if (!MaybeImmVal)
    return false;

  const uint64_t C1Val = MaybeImmVal->Value.getZExtValue();

  auto matchFirstShift = [&](const MachineInstr *MI, uint64_t &ShiftVal) {
    // Shift should match previous one and should be a one-use.
    if (MI->getOpcode() != ShiftOpcode ||
        !MRI.hasOneNonDBGUse(MI->getOperand(0).getReg()))
      return false;

    // Must be a constant.
    auto MaybeImmVal =
        getConstantVRegValWithLookThrough(MI->getOperand(2).getReg(), MRI);
    if (!MaybeImmVal)
      return false;

    ShiftVal = MaybeImmVal->Value.getSExtValue();
    return true;
  };

  // Logic ops are commutative, so check each operand for a match.
  Register LogicMIReg1 = LogicMI->getOperand(1).getReg();
  MachineInstr *LogicMIOp1 = MRI.getUniqueVRegDef(LogicMIReg1);
  Register LogicMIReg2 = LogicMI->getOperand(2).getReg();
  MachineInstr *LogicMIOp2 = MRI.getUniqueVRegDef(LogicMIReg2);
  uint64_t C0Val;

  if (matchFirstShift(LogicMIOp1, C0Val)) {
    MatchInfo.LogicNonShiftReg = LogicMIReg2;
    MatchInfo.Shift2 = LogicMIOp1;
  } else if (matchFirstShift(LogicMIOp2, C0Val)) {
    MatchInfo.LogicNonShiftReg = LogicMIReg1;
    MatchInfo.Shift2 = LogicMIOp2;
  } else
    return false;

  MatchInfo.ValSum = C0Val + C1Val;

  // The fold is not valid if the sum of the shift values exceeds bitwidth.
  if (MatchInfo.ValSum >= MRI.getType(LogicDest).getScalarSizeInBits())
    return false;

  MatchInfo.Logic = LogicMI;
  return true;
}

void CombinerHelper::applyShiftOfShiftedLogic(MachineInstr &MI,
                                              ShiftOfShiftedLogic &MatchInfo) {
  unsigned Opcode = MI.getOpcode();
  assert((Opcode == TargetOpcode::G_SHL || Opcode == TargetOpcode::G_ASHR ||
          Opcode == TargetOpcode::G_LSHR || Opcode == TargetOpcode::G_USHLSAT ||
          Opcode == TargetOpcode::G_SSHLSAT) &&
         "Expected G_SHL, G_ASHR, G_LSHR, G_USHLSAT and G_SSHLSAT");

  LLT ShlType = MRI.getType(MI.getOperand(2).getReg());
  LLT DestType = MRI.getType(MI.getOperand(0).getReg());
  Builder.setInstrAndDebugLoc(MI);

  Register Const = Builder.buildConstant(ShlType, MatchInfo.ValSum).getReg(0);

  Register Shift1Base = MatchInfo.Shift2->getOperand(1).getReg();
  Register Shift1 =
      Builder.buildInstr(Opcode, {DestType}, {Shift1Base, Const}).getReg(0);

  Register Shift2Const = MI.getOperand(2).getReg();
  Register Shift2 = Builder
                        .buildInstr(Opcode, {DestType},
                                    {MatchInfo.LogicNonShiftReg, Shift2Const})
                        .getReg(0);

  Register Dest = MI.getOperand(0).getReg();
  Builder.buildInstr(MatchInfo.Logic->getOpcode(), {Dest}, {Shift1, Shift2});

  // These were one use so it's safe to remove them.
  MatchInfo.Shift2->eraseFromParent();
  MatchInfo.Logic->eraseFromParent();

  MI.eraseFromParent();
}

bool CombinerHelper::matchCombineMulToShl(MachineInstr &MI,
                                          unsigned &ShiftVal) {
  assert(MI.getOpcode() == TargetOpcode::G_MUL && "Expected a G_MUL");
  auto MaybeImmVal =
      getConstantVRegValWithLookThrough(MI.getOperand(2).getReg(), MRI);
  if (!MaybeImmVal)
    return false;

  ShiftVal = MaybeImmVal->Value.exactLogBase2();
  return (static_cast<int32_t>(ShiftVal) != -1);
}

void CombinerHelper::applyCombineMulToShl(MachineInstr &MI,
                                          unsigned &ShiftVal) {
  assert(MI.getOpcode() == TargetOpcode::G_MUL && "Expected a G_MUL");
  MachineIRBuilder MIB(MI);
  LLT ShiftTy = MRI.getType(MI.getOperand(0).getReg());
  auto ShiftCst = MIB.buildConstant(ShiftTy, ShiftVal);
  Observer.changingInstr(MI);
  MI.setDesc(MIB.getTII().get(TargetOpcode::G_SHL));
  MI.getOperand(2).setReg(ShiftCst.getReg(0));
  Observer.changedInstr(MI);
}

// shl ([sza]ext x), y => zext (shl x, y), if shift does not overflow source
bool CombinerHelper::matchCombineShlOfExtend(MachineInstr &MI,
                                             RegisterImmPair &MatchData) {
  assert(MI.getOpcode() == TargetOpcode::G_SHL && KB);

  Register LHS = MI.getOperand(1).getReg();

  Register ExtSrc;
  if (!mi_match(LHS, MRI, m_GAnyExt(m_Reg(ExtSrc))) &&
      !mi_match(LHS, MRI, m_GZExt(m_Reg(ExtSrc))) &&
      !mi_match(LHS, MRI, m_GSExt(m_Reg(ExtSrc))))
    return false;

  // TODO: Should handle vector splat.
  Register RHS = MI.getOperand(2).getReg();
  auto MaybeShiftAmtVal = getConstantVRegValWithLookThrough(RHS, MRI);
  if (!MaybeShiftAmtVal)
    return false;

  if (LI) {
    LLT SrcTy = MRI.getType(ExtSrc);

    // We only really care about the legality with the shifted value. We can
    // pick any type the constant shift amount, so ask the target what to
    // use. Otherwise we would have to guess and hope it is reported as legal.
    LLT ShiftAmtTy = getTargetLowering().getPreferredShiftAmountTy(SrcTy);
    if (!isLegalOrBeforeLegalizer({TargetOpcode::G_SHL, {SrcTy, ShiftAmtTy}}))
      return false;
  }

  int64_t ShiftAmt = MaybeShiftAmtVal->Value.getSExtValue();
  MatchData.Reg = ExtSrc;
  MatchData.Imm = ShiftAmt;

  unsigned MinLeadingZeros = KB->getKnownZeroes(ExtSrc).countLeadingOnes();
  return MinLeadingZeros >= ShiftAmt;
}

void CombinerHelper::applyCombineShlOfExtend(MachineInstr &MI,
                                             const RegisterImmPair &MatchData) {
  Register ExtSrcReg = MatchData.Reg;
  int64_t ShiftAmtVal = MatchData.Imm;

  LLT ExtSrcTy = MRI.getType(ExtSrcReg);
  Builder.setInstrAndDebugLoc(MI);
  auto ShiftAmt = Builder.buildConstant(ExtSrcTy, ShiftAmtVal);
  auto NarrowShift =
      Builder.buildShl(ExtSrcTy, ExtSrcReg, ShiftAmt, MI.getFlags());
  Builder.buildZExt(MI.getOperand(0), NarrowShift);
  MI.eraseFromParent();
}

bool CombinerHelper::matchCombineMergeUnmerge(MachineInstr &MI,
                                              Register &MatchInfo) {
  GMerge &Merge = cast<GMerge>(MI);
  SmallVector<Register, 16> MergedValues;
  for (unsigned I = 0; I < Merge.getNumSources(); ++I)
    MergedValues.emplace_back(Merge.getSourceReg(I));

  auto *Unmerge = getOpcodeDef<GUnmerge>(MergedValues[0], MRI);
  if (!Unmerge || Unmerge->getNumDefs() != Merge.getNumSources())
    return false;

  for (unsigned I = 0; I < MergedValues.size(); ++I)
    if (MergedValues[I] != Unmerge->getReg(I))
      return false;

  MatchInfo = Unmerge->getSourceReg();
  return true;
}

static Register peekThroughBitcast(Register Reg,
                                   const MachineRegisterInfo &MRI) {
  while (mi_match(Reg, MRI, m_GBitcast(m_Reg(Reg))))
    ;

  return Reg;
}

bool CombinerHelper::matchCombineUnmergeMergeToPlainValues(
    MachineInstr &MI, SmallVectorImpl<Register> &Operands) {
  assert(MI.getOpcode() == TargetOpcode::G_UNMERGE_VALUES &&
         "Expected an unmerge");
  Register SrcReg =
      peekThroughBitcast(MI.getOperand(MI.getNumOperands() - 1).getReg(), MRI);

  MachineInstr *SrcInstr = MRI.getVRegDef(SrcReg);
  if (SrcInstr->getOpcode() != TargetOpcode::G_MERGE_VALUES &&
      SrcInstr->getOpcode() != TargetOpcode::G_BUILD_VECTOR &&
      SrcInstr->getOpcode() != TargetOpcode::G_CONCAT_VECTORS)
    return false;

  // Check the source type of the merge.
  LLT SrcMergeTy = MRI.getType(SrcInstr->getOperand(1).getReg());
  LLT Dst0Ty = MRI.getType(MI.getOperand(0).getReg());
  bool SameSize = Dst0Ty.getSizeInBits() == SrcMergeTy.getSizeInBits();
  if (SrcMergeTy != Dst0Ty && !SameSize)
    return false;
  // They are the same now (modulo a bitcast).
  // We can collect all the src registers.
  for (unsigned Idx = 1, EndIdx = SrcInstr->getNumOperands(); Idx != EndIdx;
       ++Idx)
    Operands.push_back(SrcInstr->getOperand(Idx).getReg());
  return true;
}

void CombinerHelper::applyCombineUnmergeMergeToPlainValues(
    MachineInstr &MI, SmallVectorImpl<Register> &Operands) {
  assert(MI.getOpcode() == TargetOpcode::G_UNMERGE_VALUES &&
         "Expected an unmerge");
  assert((MI.getNumOperands() - 1 == Operands.size()) &&
         "Not enough operands to replace all defs");
  unsigned NumElems = MI.getNumOperands() - 1;

  LLT SrcTy = MRI.getType(Operands[0]);
  LLT DstTy = MRI.getType(MI.getOperand(0).getReg());
  bool CanReuseInputDirectly = DstTy == SrcTy;
  Builder.setInstrAndDebugLoc(MI);
  for (unsigned Idx = 0; Idx < NumElems; ++Idx) {
    Register DstReg = MI.getOperand(Idx).getReg();
    Register SrcReg = Operands[Idx];
    if (CanReuseInputDirectly)
      replaceRegWith(MRI, DstReg, SrcReg);
    else
      Builder.buildCast(DstReg, SrcReg);
  }
  MI.eraseFromParent();
}

bool CombinerHelper::matchCombineUnmergeConstant(MachineInstr &MI,
                                                 SmallVectorImpl<APInt> &Csts) {
  unsigned SrcIdx = MI.getNumOperands() - 1;
  Register SrcReg = MI.getOperand(SrcIdx).getReg();
  MachineInstr *SrcInstr = MRI.getVRegDef(SrcReg);
  if (SrcInstr->getOpcode() != TargetOpcode::G_CONSTANT &&
      SrcInstr->getOpcode() != TargetOpcode::G_FCONSTANT)
    return false;
  // Break down the big constant in smaller ones.
  const MachineOperand &CstVal = SrcInstr->getOperand(1);
  APInt Val = SrcInstr->getOpcode() == TargetOpcode::G_CONSTANT
                  ? CstVal.getCImm()->getValue()
                  : CstVal.getFPImm()->getValueAPF().bitcastToAPInt();

  LLT Dst0Ty = MRI.getType(MI.getOperand(0).getReg());
  unsigned ShiftAmt = Dst0Ty.getSizeInBits();
  // Unmerge a constant.
  for (unsigned Idx = 0; Idx != SrcIdx; ++Idx) {
    Csts.emplace_back(Val.trunc(ShiftAmt));
    Val = Val.lshr(ShiftAmt);
  }

  return true;
}

void CombinerHelper::applyCombineUnmergeConstant(MachineInstr &MI,
                                                 SmallVectorImpl<APInt> &Csts) {
  assert(MI.getOpcode() == TargetOpcode::G_UNMERGE_VALUES &&
         "Expected an unmerge");
  assert((MI.getNumOperands() - 1 == Csts.size()) &&
         "Not enough operands to replace all defs");
  unsigned NumElems = MI.getNumOperands() - 1;
  Builder.setInstrAndDebugLoc(MI);
  for (unsigned Idx = 0; Idx < NumElems; ++Idx) {
    Register DstReg = MI.getOperand(Idx).getReg();
    Builder.buildConstant(DstReg, Csts[Idx]);
  }

  MI.eraseFromParent();
}

bool CombinerHelper::matchCombineUnmergeWithDeadLanesToTrunc(MachineInstr &MI) {
  assert(MI.getOpcode() == TargetOpcode::G_UNMERGE_VALUES &&
         "Expected an unmerge");
  // Check that all the lanes are dead except the first one.
  for (unsigned Idx = 1, EndIdx = MI.getNumDefs(); Idx != EndIdx; ++Idx) {
    if (!MRI.use_nodbg_empty(MI.getOperand(Idx).getReg()))
      return false;
  }
  return true;
}

void CombinerHelper::applyCombineUnmergeWithDeadLanesToTrunc(MachineInstr &MI) {
  Builder.setInstrAndDebugLoc(MI);
  Register SrcReg = MI.getOperand(MI.getNumDefs()).getReg();
  // Truncating a vector is going to truncate every single lane,
  // whereas we want the full lowbits.
  // Do the operation on a scalar instead.
  LLT SrcTy = MRI.getType(SrcReg);
  if (SrcTy.isVector())
    SrcReg =
        Builder.buildCast(LLT::scalar(SrcTy.getSizeInBits()), SrcReg).getReg(0);

  Register Dst0Reg = MI.getOperand(0).getReg();
  LLT Dst0Ty = MRI.getType(Dst0Reg);
  if (Dst0Ty.isVector()) {
    auto MIB = Builder.buildTrunc(LLT::scalar(Dst0Ty.getSizeInBits()), SrcReg);
    Builder.buildCast(Dst0Reg, MIB);
  } else
    Builder.buildTrunc(Dst0Reg, SrcReg);
  MI.eraseFromParent();
}

bool CombinerHelper::matchCombineUnmergeZExtToZExt(MachineInstr &MI) {
  assert(MI.getOpcode() == TargetOpcode::G_UNMERGE_VALUES &&
         "Expected an unmerge");
  Register Dst0Reg = MI.getOperand(0).getReg();
  LLT Dst0Ty = MRI.getType(Dst0Reg);
  // G_ZEXT on vector applies to each lane, so it will
  // affect all destinations. Therefore we won't be able
  // to simplify the unmerge to just the first definition.
  if (Dst0Ty.isVector())
    return false;
  Register SrcReg = MI.getOperand(MI.getNumDefs()).getReg();
  LLT SrcTy = MRI.getType(SrcReg);
  if (SrcTy.isVector())
    return false;

  Register ZExtSrcReg;
  if (!mi_match(SrcReg, MRI, m_GZExt(m_Reg(ZExtSrcReg))))
    return false;

  // Finally we can replace the first definition with
  // a zext of the source if the definition is big enough to hold
  // all of ZExtSrc bits.
  LLT ZExtSrcTy = MRI.getType(ZExtSrcReg);
  return ZExtSrcTy.getSizeInBits() <= Dst0Ty.getSizeInBits();
}

void CombinerHelper::applyCombineUnmergeZExtToZExt(MachineInstr &MI) {
  assert(MI.getOpcode() == TargetOpcode::G_UNMERGE_VALUES &&
         "Expected an unmerge");

  Register Dst0Reg = MI.getOperand(0).getReg();

  MachineInstr *ZExtInstr =
      MRI.getVRegDef(MI.getOperand(MI.getNumDefs()).getReg());
  assert(ZExtInstr && ZExtInstr->getOpcode() == TargetOpcode::G_ZEXT &&
         "Expecting a G_ZEXT");

  Register ZExtSrcReg = ZExtInstr->getOperand(1).getReg();
  LLT Dst0Ty = MRI.getType(Dst0Reg);
  LLT ZExtSrcTy = MRI.getType(ZExtSrcReg);

  Builder.setInstrAndDebugLoc(MI);

  if (Dst0Ty.getSizeInBits() > ZExtSrcTy.getSizeInBits()) {
    Builder.buildZExt(Dst0Reg, ZExtSrcReg);
  } else {
    assert(Dst0Ty.getSizeInBits() == ZExtSrcTy.getSizeInBits() &&
           "ZExt src doesn't fit in destination");
    replaceRegWith(MRI, Dst0Reg, ZExtSrcReg);
  }

  Register ZeroReg;
  for (unsigned Idx = 1, EndIdx = MI.getNumDefs(); Idx != EndIdx; ++Idx) {
    if (!ZeroReg)
      ZeroReg = Builder.buildConstant(Dst0Ty, 0).getReg(0);
    replaceRegWith(MRI, MI.getOperand(Idx).getReg(), ZeroReg);
  }
  MI.eraseFromParent();
}

bool CombinerHelper::matchCombineShiftToUnmerge(MachineInstr &MI,
                                                unsigned TargetShiftSize,
                                                unsigned &ShiftVal) {
  assert((MI.getOpcode() == TargetOpcode::G_SHL ||
          MI.getOpcode() == TargetOpcode::G_LSHR ||
          MI.getOpcode() == TargetOpcode::G_ASHR) && "Expected a shift");

  LLT Ty = MRI.getType(MI.getOperand(0).getReg());
  if (Ty.isVector()) // TODO:
    return false;

  // Don't narrow further than the requested size.
  unsigned Size = Ty.getSizeInBits();
  if (Size <= TargetShiftSize)
    return false;

  auto MaybeImmVal =
    getConstantVRegValWithLookThrough(MI.getOperand(2).getReg(), MRI);
  if (!MaybeImmVal)
    return false;

  ShiftVal = MaybeImmVal->Value.getSExtValue();
  return ShiftVal >= Size / 2 && ShiftVal < Size;
}

void CombinerHelper::applyCombineShiftToUnmerge(MachineInstr &MI,
                                                const unsigned &ShiftVal) {
  Register DstReg = MI.getOperand(0).getReg();
  Register SrcReg = MI.getOperand(1).getReg();
  LLT Ty = MRI.getType(SrcReg);
  unsigned Size = Ty.getSizeInBits();
  unsigned HalfSize = Size / 2;
  assert(ShiftVal >= HalfSize);

  LLT HalfTy = LLT::scalar(HalfSize);

  Builder.setInstr(MI);
  auto Unmerge = Builder.buildUnmerge(HalfTy, SrcReg);
  unsigned NarrowShiftAmt = ShiftVal - HalfSize;

  if (MI.getOpcode() == TargetOpcode::G_LSHR) {
    Register Narrowed = Unmerge.getReg(1);

    //  dst = G_LSHR s64:x, C for C >= 32
    // =>
    //   lo, hi = G_UNMERGE_VALUES x
    //   dst = G_MERGE_VALUES (G_LSHR hi, C - 32), 0

    if (NarrowShiftAmt != 0) {
      Narrowed = Builder.buildLShr(HalfTy, Narrowed,
        Builder.buildConstant(HalfTy, NarrowShiftAmt)).getReg(0);
    }

    auto Zero = Builder.buildConstant(HalfTy, 0);
    Builder.buildMerge(DstReg, { Narrowed, Zero });
  } else if (MI.getOpcode() == TargetOpcode::G_SHL) {
    Register Narrowed = Unmerge.getReg(0);
    //  dst = G_SHL s64:x, C for C >= 32
    // =>
    //   lo, hi = G_UNMERGE_VALUES x
    //   dst = G_MERGE_VALUES 0, (G_SHL hi, C - 32)
    if (NarrowShiftAmt != 0) {
      Narrowed = Builder.buildShl(HalfTy, Narrowed,
        Builder.buildConstant(HalfTy, NarrowShiftAmt)).getReg(0);
    }

    auto Zero = Builder.buildConstant(HalfTy, 0);
    Builder.buildMerge(DstReg, { Zero, Narrowed });
  } else {
    assert(MI.getOpcode() == TargetOpcode::G_ASHR);
    auto Hi = Builder.buildAShr(
      HalfTy, Unmerge.getReg(1),
      Builder.buildConstant(HalfTy, HalfSize - 1));

    if (ShiftVal == HalfSize) {
      // (G_ASHR i64:x, 32) ->
      //   G_MERGE_VALUES hi_32(x), (G_ASHR hi_32(x), 31)
      Builder.buildMerge(DstReg, { Unmerge.getReg(1), Hi });
    } else if (ShiftVal == Size - 1) {
      // Don't need a second shift.
      // (G_ASHR i64:x, 63) ->
      //   %narrowed = (G_ASHR hi_32(x), 31)
      //   G_MERGE_VALUES %narrowed, %narrowed
      Builder.buildMerge(DstReg, { Hi, Hi });
    } else {
      auto Lo = Builder.buildAShr(
        HalfTy, Unmerge.getReg(1),
        Builder.buildConstant(HalfTy, ShiftVal - HalfSize));

      // (G_ASHR i64:x, C) ->, for C >= 32
      //   G_MERGE_VALUES (G_ASHR hi_32(x), C - 32), (G_ASHR hi_32(x), 31)
      Builder.buildMerge(DstReg, { Lo, Hi });
    }
  }

  MI.eraseFromParent();
}

bool CombinerHelper::tryCombineShiftToUnmerge(MachineInstr &MI,
                                              unsigned TargetShiftAmount) {
  unsigned ShiftAmt;
  if (matchCombineShiftToUnmerge(MI, TargetShiftAmount, ShiftAmt)) {
    applyCombineShiftToUnmerge(MI, ShiftAmt);
    return true;
  }

  return false;
}

bool CombinerHelper::matchCombineI2PToP2I(MachineInstr &MI, Register &Reg) {
  assert(MI.getOpcode() == TargetOpcode::G_INTTOPTR && "Expected a G_INTTOPTR");
  Register DstReg = MI.getOperand(0).getReg();
  LLT DstTy = MRI.getType(DstReg);
  Register SrcReg = MI.getOperand(1).getReg();
  return mi_match(SrcReg, MRI,
                  m_GPtrToInt(m_all_of(m_SpecificType(DstTy), m_Reg(Reg))));
}

void CombinerHelper::applyCombineI2PToP2I(MachineInstr &MI, Register &Reg) {
  assert(MI.getOpcode() == TargetOpcode::G_INTTOPTR && "Expected a G_INTTOPTR");
  Register DstReg = MI.getOperand(0).getReg();
  Builder.setInstr(MI);
  Builder.buildCopy(DstReg, Reg);
  MI.eraseFromParent();
}

bool CombinerHelper::matchCombineP2IToI2P(MachineInstr &MI, Register &Reg) {
  assert(MI.getOpcode() == TargetOpcode::G_PTRTOINT && "Expected a G_PTRTOINT");
  Register SrcReg = MI.getOperand(1).getReg();
  return mi_match(SrcReg, MRI, m_GIntToPtr(m_Reg(Reg)));
}

void CombinerHelper::applyCombineP2IToI2P(MachineInstr &MI, Register &Reg) {
  assert(MI.getOpcode() == TargetOpcode::G_PTRTOINT && "Expected a G_PTRTOINT");
  Register DstReg = MI.getOperand(0).getReg();
  Builder.setInstr(MI);
  Builder.buildZExtOrTrunc(DstReg, Reg);
  MI.eraseFromParent();
}

bool CombinerHelper::matchCombineAddP2IToPtrAdd(
    MachineInstr &MI, std::pair<Register, bool> &PtrReg) {
  assert(MI.getOpcode() == TargetOpcode::G_ADD);
  Register LHS = MI.getOperand(1).getReg();
  Register RHS = MI.getOperand(2).getReg();
  LLT IntTy = MRI.getType(LHS);

  // G_PTR_ADD always has the pointer in the LHS, so we may need to commute the
  // instruction.
  PtrReg.second = false;
  for (Register SrcReg : {LHS, RHS}) {
    if (mi_match(SrcReg, MRI, m_GPtrToInt(m_Reg(PtrReg.first)))) {
      // Don't handle cases where the integer is implicitly converted to the
      // pointer width.
      LLT PtrTy = MRI.getType(PtrReg.first);
      if (PtrTy.getScalarSizeInBits() == IntTy.getScalarSizeInBits())
        return true;
    }

    PtrReg.second = true;
  }

  return false;
}

void CombinerHelper::applyCombineAddP2IToPtrAdd(
    MachineInstr &MI, std::pair<Register, bool> &PtrReg) {
  Register Dst = MI.getOperand(0).getReg();
  Register LHS = MI.getOperand(1).getReg();
  Register RHS = MI.getOperand(2).getReg();

  const bool DoCommute = PtrReg.second;
  if (DoCommute)
    std::swap(LHS, RHS);
  LHS = PtrReg.first;

  LLT PtrTy = MRI.getType(LHS);

  Builder.setInstrAndDebugLoc(MI);
  auto PtrAdd = Builder.buildPtrAdd(PtrTy, LHS, RHS);
  Builder.buildPtrToInt(Dst, PtrAdd);
  MI.eraseFromParent();
}

bool CombinerHelper::matchCombineConstPtrAddToI2P(MachineInstr &MI,
                                                  int64_t &NewCst) {
  assert(MI.getOpcode() == TargetOpcode::G_PTR_ADD && "Expected a G_PTR_ADD");
  Register LHS = MI.getOperand(1).getReg();
  Register RHS = MI.getOperand(2).getReg();
  MachineRegisterInfo &MRI = Builder.getMF().getRegInfo();

  if (auto RHSCst = getConstantVRegSExtVal(RHS, MRI)) {
    int64_t Cst;
    if (mi_match(LHS, MRI, m_GIntToPtr(m_ICst(Cst)))) {
      NewCst = Cst + *RHSCst;
      return true;
    }
  }

  return false;
}

void CombinerHelper::applyCombineConstPtrAddToI2P(MachineInstr &MI,
                                                  int64_t &NewCst) {
  assert(MI.getOpcode() == TargetOpcode::G_PTR_ADD && "Expected a G_PTR_ADD");
  Register Dst = MI.getOperand(0).getReg();

  Builder.setInstrAndDebugLoc(MI);
  Builder.buildConstant(Dst, NewCst);
  MI.eraseFromParent();
}

bool CombinerHelper::matchCombineAnyExtTrunc(MachineInstr &MI, Register &Reg) {
  assert(MI.getOpcode() == TargetOpcode::G_ANYEXT && "Expected a G_ANYEXT");
  Register DstReg = MI.getOperand(0).getReg();
  Register SrcReg = MI.getOperand(1).getReg();
  LLT DstTy = MRI.getType(DstReg);
  return mi_match(SrcReg, MRI,
                  m_GTrunc(m_all_of(m_Reg(Reg), m_SpecificType(DstTy))));
}

bool CombinerHelper::matchCombineZextTrunc(MachineInstr &MI, Register &Reg) {
  assert(MI.getOpcode() == TargetOpcode::G_ZEXT && "Expected a G_ZEXT");
  Register DstReg = MI.getOperand(0).getReg();
  Register SrcReg = MI.getOperand(1).getReg();
  LLT DstTy = MRI.getType(DstReg);
  if (mi_match(SrcReg, MRI,
               m_GTrunc(m_all_of(m_Reg(Reg), m_SpecificType(DstTy))))) {
    unsigned DstSize = DstTy.getScalarSizeInBits();
    unsigned SrcSize = MRI.getType(SrcReg).getScalarSizeInBits();
    return KB->getKnownBits(Reg).countMinLeadingZeros() >= DstSize - SrcSize;
  }
  return false;
}

bool CombinerHelper::matchCombineExtOfExt(
    MachineInstr &MI, std::tuple<Register, unsigned> &MatchInfo) {
  assert((MI.getOpcode() == TargetOpcode::G_ANYEXT ||
          MI.getOpcode() == TargetOpcode::G_SEXT ||
          MI.getOpcode() == TargetOpcode::G_ZEXT) &&
         "Expected a G_[ASZ]EXT");
  Register SrcReg = MI.getOperand(1).getReg();
  MachineInstr *SrcMI = MRI.getVRegDef(SrcReg);
  // Match exts with the same opcode, anyext([sz]ext) and sext(zext).
  unsigned Opc = MI.getOpcode();
  unsigned SrcOpc = SrcMI->getOpcode();
  if (Opc == SrcOpc ||
      (Opc == TargetOpcode::G_ANYEXT &&
       (SrcOpc == TargetOpcode::G_SEXT || SrcOpc == TargetOpcode::G_ZEXT)) ||
      (Opc == TargetOpcode::G_SEXT && SrcOpc == TargetOpcode::G_ZEXT)) {
    MatchInfo = std::make_tuple(SrcMI->getOperand(1).getReg(), SrcOpc);
    return true;
  }
  return false;
}

void CombinerHelper::applyCombineExtOfExt(
    MachineInstr &MI, std::tuple<Register, unsigned> &MatchInfo) {
  assert((MI.getOpcode() == TargetOpcode::G_ANYEXT ||
          MI.getOpcode() == TargetOpcode::G_SEXT ||
          MI.getOpcode() == TargetOpcode::G_ZEXT) &&
         "Expected a G_[ASZ]EXT");

  Register Reg = std::get<0>(MatchInfo);
  unsigned SrcExtOp = std::get<1>(MatchInfo);

  // Combine exts with the same opcode.
  if (MI.getOpcode() == SrcExtOp) {
    Observer.changingInstr(MI);
    MI.getOperand(1).setReg(Reg);
    Observer.changedInstr(MI);
    return;
  }

  // Combine:
  // - anyext([sz]ext x) to [sz]ext x
  // - sext(zext x) to zext x
  if (MI.getOpcode() == TargetOpcode::G_ANYEXT ||
      (MI.getOpcode() == TargetOpcode::G_SEXT &&
       SrcExtOp == TargetOpcode::G_ZEXT)) {
    Register DstReg = MI.getOperand(0).getReg();
    Builder.setInstrAndDebugLoc(MI);
    Builder.buildInstr(SrcExtOp, {DstReg}, {Reg});
    MI.eraseFromParent();
  }
}

void CombinerHelper::applyCombineMulByNegativeOne(MachineInstr &MI) {
  assert(MI.getOpcode() == TargetOpcode::G_MUL && "Expected a G_MUL");
  Register DstReg = MI.getOperand(0).getReg();
  Register SrcReg = MI.getOperand(1).getReg();
  LLT DstTy = MRI.getType(DstReg);

  Builder.setInstrAndDebugLoc(MI);
  Builder.buildSub(DstReg, Builder.buildConstant(DstTy, 0), SrcReg,
                   MI.getFlags());
  MI.eraseFromParent();
}

bool CombinerHelper::matchCombineFNegOfFNeg(MachineInstr &MI, Register &Reg) {
  assert(MI.getOpcode() == TargetOpcode::G_FNEG && "Expected a G_FNEG");
  Register SrcReg = MI.getOperand(1).getReg();
  return mi_match(SrcReg, MRI, m_GFNeg(m_Reg(Reg)));
}

bool CombinerHelper::matchCombineFAbsOfFAbs(MachineInstr &MI, Register &Src) {
  assert(MI.getOpcode() == TargetOpcode::G_FABS && "Expected a G_FABS");
  Src = MI.getOperand(1).getReg();
  Register AbsSrc;
  return mi_match(Src, MRI, m_GFabs(m_Reg(AbsSrc)));
}

bool CombinerHelper::matchCombineTruncOfExt(
    MachineInstr &MI, std::pair<Register, unsigned> &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_TRUNC && "Expected a G_TRUNC");
  Register SrcReg = MI.getOperand(1).getReg();
  MachineInstr *SrcMI = MRI.getVRegDef(SrcReg);
  unsigned SrcOpc = SrcMI->getOpcode();
  if (SrcOpc == TargetOpcode::G_ANYEXT || SrcOpc == TargetOpcode::G_SEXT ||
      SrcOpc == TargetOpcode::G_ZEXT) {
    MatchInfo = std::make_pair(SrcMI->getOperand(1).getReg(), SrcOpc);
    return true;
  }
  return false;
}

void CombinerHelper::applyCombineTruncOfExt(
    MachineInstr &MI, std::pair<Register, unsigned> &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_TRUNC && "Expected a G_TRUNC");
  Register SrcReg = MatchInfo.first;
  unsigned SrcExtOp = MatchInfo.second;
  Register DstReg = MI.getOperand(0).getReg();
  LLT SrcTy = MRI.getType(SrcReg);
  LLT DstTy = MRI.getType(DstReg);
  if (SrcTy == DstTy) {
    MI.eraseFromParent();
    replaceRegWith(MRI, DstReg, SrcReg);
    return;
  }
  Builder.setInstrAndDebugLoc(MI);
  if (SrcTy.getSizeInBits() < DstTy.getSizeInBits())
    Builder.buildInstr(SrcExtOp, {DstReg}, {SrcReg});
  else
    Builder.buildTrunc(DstReg, SrcReg);
  MI.eraseFromParent();
}

bool CombinerHelper::matchCombineTruncOfShl(
    MachineInstr &MI, std::pair<Register, Register> &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_TRUNC && "Expected a G_TRUNC");
  Register DstReg = MI.getOperand(0).getReg();
  Register SrcReg = MI.getOperand(1).getReg();
  LLT DstTy = MRI.getType(DstReg);
  Register ShiftSrc;
  Register ShiftAmt;

  if (MRI.hasOneNonDBGUse(SrcReg) &&
      mi_match(SrcReg, MRI, m_GShl(m_Reg(ShiftSrc), m_Reg(ShiftAmt))) &&
      isLegalOrBeforeLegalizer(
          {TargetOpcode::G_SHL,
           {DstTy, getTargetLowering().getPreferredShiftAmountTy(DstTy)}})) {
    KnownBits Known = KB->getKnownBits(ShiftAmt);
    unsigned Size = DstTy.getSizeInBits();
    if (Known.getBitWidth() - Known.countMinLeadingZeros() <= Log2_32(Size)) {
      MatchInfo = std::make_pair(ShiftSrc, ShiftAmt);
      return true;
    }
  }
  return false;
}

void CombinerHelper::applyCombineTruncOfShl(
    MachineInstr &MI, std::pair<Register, Register> &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_TRUNC && "Expected a G_TRUNC");
  Register DstReg = MI.getOperand(0).getReg();
  Register SrcReg = MI.getOperand(1).getReg();
  LLT DstTy = MRI.getType(DstReg);
  MachineInstr *SrcMI = MRI.getVRegDef(SrcReg);

  Register ShiftSrc = MatchInfo.first;
  Register ShiftAmt = MatchInfo.second;
  Builder.setInstrAndDebugLoc(MI);
  auto TruncShiftSrc = Builder.buildTrunc(DstTy, ShiftSrc);
  Builder.buildShl(DstReg, TruncShiftSrc, ShiftAmt, SrcMI->getFlags());
  MI.eraseFromParent();
}

bool CombinerHelper::matchAnyExplicitUseIsUndef(MachineInstr &MI) {
  return any_of(MI.explicit_uses(), [this](const MachineOperand &MO) {
    return MO.isReg() &&
           getOpcodeDef(TargetOpcode::G_IMPLICIT_DEF, MO.getReg(), MRI);
  });
}

bool CombinerHelper::matchAllExplicitUsesAreUndef(MachineInstr &MI) {
  return all_of(MI.explicit_uses(), [this](const MachineOperand &MO) {
    return !MO.isReg() ||
           getOpcodeDef(TargetOpcode::G_IMPLICIT_DEF, MO.getReg(), MRI);
  });
}

bool CombinerHelper::matchUndefShuffleVectorMask(MachineInstr &MI) {
  assert(MI.getOpcode() == TargetOpcode::G_SHUFFLE_VECTOR);
  ArrayRef<int> Mask = MI.getOperand(3).getShuffleMask();
  return all_of(Mask, [](int Elt) { return Elt < 0; });
}

bool CombinerHelper::matchUndefStore(MachineInstr &MI) {
  assert(MI.getOpcode() == TargetOpcode::G_STORE);
  return getOpcodeDef(TargetOpcode::G_IMPLICIT_DEF, MI.getOperand(0).getReg(),
                      MRI);
}

bool CombinerHelper::matchUndefSelectCmp(MachineInstr &MI) {
  assert(MI.getOpcode() == TargetOpcode::G_SELECT);
  return getOpcodeDef(TargetOpcode::G_IMPLICIT_DEF, MI.getOperand(1).getReg(),
                      MRI);
}

bool CombinerHelper::matchConstantSelectCmp(MachineInstr &MI, unsigned &OpIdx) {
  assert(MI.getOpcode() == TargetOpcode::G_SELECT);
  if (auto MaybeCstCmp =
          getConstantVRegValWithLookThrough(MI.getOperand(1).getReg(), MRI)) {
    OpIdx = MaybeCstCmp->Value.isNullValue() ? 3 : 2;
    return true;
  }
  return false;
}

bool CombinerHelper::eraseInst(MachineInstr &MI) {
  MI.eraseFromParent();
  return true;
}

bool CombinerHelper::matchEqualDefs(const MachineOperand &MOP1,
                                    const MachineOperand &MOP2) {
  if (!MOP1.isReg() || !MOP2.isReg())
    return false;
  MachineInstr *I1 = getDefIgnoringCopies(MOP1.getReg(), MRI);
  if (!I1)
    return false;
  MachineInstr *I2 = getDefIgnoringCopies(MOP2.getReg(), MRI);
  if (!I2)
    return false;

  // Handle a case like this:
  //
  // %0:_(s64), %1:_(s64) = G_UNMERGE_VALUES %2:_(<2 x s64>)
  //
  // Even though %0 and %1 are produced by the same instruction they are not
  // the same values.
  if (I1 == I2)
    return MOP1.getReg() == MOP2.getReg();

  // If we have an instruction which loads or stores, we can't guarantee that
  // it is identical.
  //
  // For example, we may have
  //
  // %x1 = G_LOAD %addr (load N from @somewhere)
  // ...
  // call @foo
  // ...
  // %x2 = G_LOAD %addr (load N from @somewhere)
  // ...
  // %or = G_OR %x1, %x2
  //
  // It's possible that @foo will modify whatever lives at the address we're
  // loading from. To be safe, let's just assume that all loads and stores
  // are different (unless we have something which is guaranteed to not
  // change.)
  if (I1->mayLoadOrStore() && !I1->isDereferenceableInvariantLoad(nullptr))
    return false;

  // Check for physical registers on the instructions first to avoid cases
  // like this:
  //
  // %a = COPY $physreg
  // ...
  // SOMETHING implicit-def $physreg
  // ...
  // %b = COPY $physreg
  //
  // These copies are not equivalent.
  if (any_of(I1->uses(), [](const MachineOperand &MO) {
        return MO.isReg() && MO.getReg().isPhysical();
      })) {
    // Check if we have a case like this:
    //
    // %a = COPY $physreg
    // %b = COPY %a
    //
    // In this case, I1 and I2 will both be equal to %a = COPY $physreg.
    // From that, we know that they must have the same value, since they must
    // have come from the same COPY.
    return I1->isIdenticalTo(*I2);
  }

  // We don't have any physical registers, so we don't necessarily need the
  // same vreg defs.
  //
  // On the off-chance that there's some target instruction feeding into the
  // instruction, let's use produceSameValue instead of isIdenticalTo.
  return Builder.getTII().produceSameValue(*I1, *I2, &MRI);
}

bool CombinerHelper::matchConstantOp(const MachineOperand &MOP, int64_t C) {
  if (!MOP.isReg())
    return false;
  // MIPatternMatch doesn't let us look through G_ZEXT etc.
  auto ValAndVReg = getConstantVRegValWithLookThrough(MOP.getReg(), MRI);
  return ValAndVReg && ValAndVReg->Value == C;
}

bool CombinerHelper::replaceSingleDefInstWithOperand(MachineInstr &MI,
                                                     unsigned OpIdx) {
  assert(MI.getNumExplicitDefs() == 1 && "Expected one explicit def?");
  Register OldReg = MI.getOperand(0).getReg();
  Register Replacement = MI.getOperand(OpIdx).getReg();
  assert(canReplaceReg(OldReg, Replacement, MRI) && "Cannot replace register?");
  MI.eraseFromParent();
  replaceRegWith(MRI, OldReg, Replacement);
  return true;
}

bool CombinerHelper::replaceSingleDefInstWithReg(MachineInstr &MI,
                                                 Register Replacement) {
  assert(MI.getNumExplicitDefs() == 1 && "Expected one explicit def?");
  Register OldReg = MI.getOperand(0).getReg();
  assert(canReplaceReg(OldReg, Replacement, MRI) && "Cannot replace register?");
  MI.eraseFromParent();
  replaceRegWith(MRI, OldReg, Replacement);
  return true;
}

bool CombinerHelper::matchSelectSameVal(MachineInstr &MI) {
  assert(MI.getOpcode() == TargetOpcode::G_SELECT);
  // Match (cond ? x : x)
  return matchEqualDefs(MI.getOperand(2), MI.getOperand(3)) &&
         canReplaceReg(MI.getOperand(0).getReg(), MI.getOperand(2).getReg(),
                       MRI);
}

bool CombinerHelper::matchBinOpSameVal(MachineInstr &MI) {
  return matchEqualDefs(MI.getOperand(1), MI.getOperand(2)) &&
         canReplaceReg(MI.getOperand(0).getReg(), MI.getOperand(1).getReg(),
                       MRI);
}

bool CombinerHelper::matchOperandIsZero(MachineInstr &MI, unsigned OpIdx) {
  return matchConstantOp(MI.getOperand(OpIdx), 0) &&
         canReplaceReg(MI.getOperand(0).getReg(), MI.getOperand(OpIdx).getReg(),
                       MRI);
}

bool CombinerHelper::matchOperandIsUndef(MachineInstr &MI, unsigned OpIdx) {
  MachineOperand &MO = MI.getOperand(OpIdx);
  return MO.isReg() &&
         getOpcodeDef(TargetOpcode::G_IMPLICIT_DEF, MO.getReg(), MRI);
}

bool CombinerHelper::matchOperandIsKnownToBeAPowerOfTwo(MachineInstr &MI,
                                                        unsigned OpIdx) {
  MachineOperand &MO = MI.getOperand(OpIdx);
  return isKnownToBeAPowerOfTwo(MO.getReg(), MRI, KB);
}

bool CombinerHelper::replaceInstWithFConstant(MachineInstr &MI, double C) {
  assert(MI.getNumDefs() == 1 && "Expected only one def?");
  Builder.setInstr(MI);
  Builder.buildFConstant(MI.getOperand(0), C);
  MI.eraseFromParent();
  return true;
}

bool CombinerHelper::replaceInstWithConstant(MachineInstr &MI, int64_t C) {
  assert(MI.getNumDefs() == 1 && "Expected only one def?");
  Builder.setInstr(MI);
  Builder.buildConstant(MI.getOperand(0), C);
  MI.eraseFromParent();
  return true;
}

bool CombinerHelper::replaceInstWithConstant(MachineInstr &MI, APInt C) {
  assert(MI.getNumDefs() == 1 && "Expected only one def?");
  Builder.setInstr(MI);
  Builder.buildConstant(MI.getOperand(0), C);
  MI.eraseFromParent();
  return true;
}

bool CombinerHelper::replaceInstWithUndef(MachineInstr &MI) {
  assert(MI.getNumDefs() == 1 && "Expected only one def?");
  Builder.setInstr(MI);
  Builder.buildUndef(MI.getOperand(0));
  MI.eraseFromParent();
  return true;
}

bool CombinerHelper::matchSimplifyAddToSub(
    MachineInstr &MI, std::tuple<Register, Register> &MatchInfo) {
  Register LHS = MI.getOperand(1).getReg();
  Register RHS = MI.getOperand(2).getReg();
  Register &NewLHS = std::get<0>(MatchInfo);
  Register &NewRHS = std::get<1>(MatchInfo);

  // Helper lambda to check for opportunities for
  // ((0-A) + B) -> B - A
  // (A + (0-B)) -> A - B
  auto CheckFold = [&](Register &MaybeSub, Register &MaybeNewLHS) {
    if (!mi_match(MaybeSub, MRI, m_Neg(m_Reg(NewRHS))))
      return false;
    NewLHS = MaybeNewLHS;
    return true;
  };

  return CheckFold(LHS, RHS) || CheckFold(RHS, LHS);
}

bool CombinerHelper::matchCombineInsertVecElts(
    MachineInstr &MI, SmallVectorImpl<Register> &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_INSERT_VECTOR_ELT &&
         "Invalid opcode");
  Register DstReg = MI.getOperand(0).getReg();
  LLT DstTy = MRI.getType(DstReg);
  assert(DstTy.isVector() && "Invalid G_INSERT_VECTOR_ELT?");
  unsigned NumElts = DstTy.getNumElements();
  // If this MI is part of a sequence of insert_vec_elts, then
  // don't do the combine in the middle of the sequence.
  if (MRI.hasOneUse(DstReg) && MRI.use_instr_begin(DstReg)->getOpcode() ==
                                   TargetOpcode::G_INSERT_VECTOR_ELT)
    return false;
  MachineInstr *CurrInst = &MI;
  MachineInstr *TmpInst;
  int64_t IntImm;
  Register TmpReg;
  MatchInfo.resize(NumElts);
  while (mi_match(
      CurrInst->getOperand(0).getReg(), MRI,
      m_GInsertVecElt(m_MInstr(TmpInst), m_Reg(TmpReg), m_ICst(IntImm)))) {
    if (IntImm >= NumElts)
      return false;
    if (!MatchInfo[IntImm])
      MatchInfo[IntImm] = TmpReg;
    CurrInst = TmpInst;
  }
  // Variable index.
  if (CurrInst->getOpcode() == TargetOpcode::G_INSERT_VECTOR_ELT)
    return false;
  if (TmpInst->getOpcode() == TargetOpcode::G_BUILD_VECTOR) {
    for (unsigned I = 1; I < TmpInst->getNumOperands(); ++I) {
      if (!MatchInfo[I - 1].isValid())
        MatchInfo[I - 1] = TmpInst->getOperand(I).getReg();
    }
    return true;
  }
  // If we didn't end in a G_IMPLICIT_DEF, bail out.
  return TmpInst->getOpcode() == TargetOpcode::G_IMPLICIT_DEF;
}

void CombinerHelper::applyCombineInsertVecElts(
    MachineInstr &MI, SmallVectorImpl<Register> &MatchInfo) {
  Builder.setInstr(MI);
  Register UndefReg;
  auto GetUndef = [&]() {
    if (UndefReg)
      return UndefReg;
    LLT DstTy = MRI.getType(MI.getOperand(0).getReg());
    UndefReg = Builder.buildUndef(DstTy.getScalarType()).getReg(0);
    return UndefReg;
  };
  for (unsigned I = 0; I < MatchInfo.size(); ++I) {
    if (!MatchInfo[I])
      MatchInfo[I] = GetUndef();
  }
  Builder.buildBuildVector(MI.getOperand(0).getReg(), MatchInfo);
  MI.eraseFromParent();
}

void CombinerHelper::applySimplifyAddToSub(
    MachineInstr &MI, std::tuple<Register, Register> &MatchInfo) {
  Builder.setInstr(MI);
  Register SubLHS, SubRHS;
  std::tie(SubLHS, SubRHS) = MatchInfo;
  Builder.buildSub(MI.getOperand(0).getReg(), SubLHS, SubRHS);
  MI.eraseFromParent();
}

bool CombinerHelper::matchHoistLogicOpWithSameOpcodeHands(
    MachineInstr &MI, InstructionStepsMatchInfo &MatchInfo) {
  // Matches: logic (hand x, ...), (hand y, ...) -> hand (logic x, y), ...
  //
  // Creates the new hand + logic instruction (but does not insert them.)
  //
  // On success, MatchInfo is populated with the new instructions. These are
  // inserted in applyHoistLogicOpWithSameOpcodeHands.
  unsigned LogicOpcode = MI.getOpcode();
  assert(LogicOpcode == TargetOpcode::G_AND ||
         LogicOpcode == TargetOpcode::G_OR ||
         LogicOpcode == TargetOpcode::G_XOR);
  MachineIRBuilder MIB(MI);
  Register Dst = MI.getOperand(0).getReg();
  Register LHSReg = MI.getOperand(1).getReg();
  Register RHSReg = MI.getOperand(2).getReg();

  // Don't recompute anything.
  if (!MRI.hasOneNonDBGUse(LHSReg) || !MRI.hasOneNonDBGUse(RHSReg))
    return false;

  // Make sure we have (hand x, ...), (hand y, ...)
  MachineInstr *LeftHandInst = getDefIgnoringCopies(LHSReg, MRI);
  MachineInstr *RightHandInst = getDefIgnoringCopies(RHSReg, MRI);
  if (!LeftHandInst || !RightHandInst)
    return false;
  unsigned HandOpcode = LeftHandInst->getOpcode();
  if (HandOpcode != RightHandInst->getOpcode())
    return false;
  if (!LeftHandInst->getOperand(1).isReg() ||
      !RightHandInst->getOperand(1).isReg())
    return false;

  // Make sure the types match up, and if we're doing this post-legalization,
  // we end up with legal types.
  Register X = LeftHandInst->getOperand(1).getReg();
  Register Y = RightHandInst->getOperand(1).getReg();
  LLT XTy = MRI.getType(X);
  LLT YTy = MRI.getType(Y);
  if (XTy != YTy)
    return false;
  if (!isLegalOrBeforeLegalizer({LogicOpcode, {XTy, YTy}}))
    return false;

  // Optional extra source register.
  Register ExtraHandOpSrcReg;
  switch (HandOpcode) {
  default:
    return false;
  case TargetOpcode::G_ANYEXT:
  case TargetOpcode::G_SEXT:
  case TargetOpcode::G_ZEXT: {
    // Match: logic (ext X), (ext Y) --> ext (logic X, Y)
    break;
  }
  case TargetOpcode::G_AND:
  case TargetOpcode::G_ASHR:
  case TargetOpcode::G_LSHR:
  case TargetOpcode::G_SHL: {
    // Match: logic (binop x, z), (binop y, z) -> binop (logic x, y), z
    MachineOperand &ZOp = LeftHandInst->getOperand(2);
    if (!matchEqualDefs(ZOp, RightHandInst->getOperand(2)))
      return false;
    ExtraHandOpSrcReg = ZOp.getReg();
    break;
  }
  }

  // Record the steps to build the new instructions.
  //
  // Steps to build (logic x, y)
  auto NewLogicDst = MRI.createGenericVirtualRegister(XTy);
  OperandBuildSteps LogicBuildSteps = {
      [=](MachineInstrBuilder &MIB) { MIB.addDef(NewLogicDst); },
      [=](MachineInstrBuilder &MIB) { MIB.addReg(X); },
      [=](MachineInstrBuilder &MIB) { MIB.addReg(Y); }};
  InstructionBuildSteps LogicSteps(LogicOpcode, LogicBuildSteps);

  // Steps to build hand (logic x, y), ...z
  OperandBuildSteps HandBuildSteps = {
      [=](MachineInstrBuilder &MIB) { MIB.addDef(Dst); },
      [=](MachineInstrBuilder &MIB) { MIB.addReg(NewLogicDst); }};
  if (ExtraHandOpSrcReg.isValid())
    HandBuildSteps.push_back(
        [=](MachineInstrBuilder &MIB) { MIB.addReg(ExtraHandOpSrcReg); });
  InstructionBuildSteps HandSteps(HandOpcode, HandBuildSteps);

  MatchInfo = InstructionStepsMatchInfo({LogicSteps, HandSteps});
  return true;
}

void CombinerHelper::applyBuildInstructionSteps(
    MachineInstr &MI, InstructionStepsMatchInfo &MatchInfo) {
  assert(MatchInfo.InstrsToBuild.size() &&
         "Expected at least one instr to build?");
  Builder.setInstr(MI);
  for (auto &InstrToBuild : MatchInfo.InstrsToBuild) {
    assert(InstrToBuild.Opcode && "Expected a valid opcode?");
    assert(InstrToBuild.OperandFns.size() && "Expected at least one operand?");
    MachineInstrBuilder Instr = Builder.buildInstr(InstrToBuild.Opcode);
    for (auto &OperandFn : InstrToBuild.OperandFns)
      OperandFn(Instr);
  }
  MI.eraseFromParent();
}

bool CombinerHelper::matchAshrShlToSextInreg(
    MachineInstr &MI, std::tuple<Register, int64_t> &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_ASHR);
  int64_t ShlCst, AshrCst;
  Register Src;
  // FIXME: detect splat constant vectors.
  if (!mi_match(MI.getOperand(0).getReg(), MRI,
                m_GAShr(m_GShl(m_Reg(Src), m_ICst(ShlCst)), m_ICst(AshrCst))))
    return false;
  if (ShlCst != AshrCst)
    return false;
  if (!isLegalOrBeforeLegalizer(
          {TargetOpcode::G_SEXT_INREG, {MRI.getType(Src)}}))
    return false;
  MatchInfo = std::make_tuple(Src, ShlCst);
  return true;
}

void CombinerHelper::applyAshShlToSextInreg(
    MachineInstr &MI, std::tuple<Register, int64_t> &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_ASHR);
  Register Src;
  int64_t ShiftAmt;
  std::tie(Src, ShiftAmt) = MatchInfo;
  unsigned Size = MRI.getType(Src).getScalarSizeInBits();
  Builder.setInstrAndDebugLoc(MI);
  Builder.buildSExtInReg(MI.getOperand(0).getReg(), Src, Size - ShiftAmt);
  MI.eraseFromParent();
}

/// and(and(x, C1), C2) -> C1&C2 ? and(x, C1&C2) : 0
bool CombinerHelper::matchOverlappingAnd(
    MachineInstr &MI, std::function<void(MachineIRBuilder &)> &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_AND);

  Register Dst = MI.getOperand(0).getReg();
  LLT Ty = MRI.getType(Dst);

  Register R;
  int64_t C1;
  int64_t C2;
  if (!mi_match(
          Dst, MRI,
          m_GAnd(m_GAnd(m_Reg(R), m_ICst(C1)), m_ICst(C2))))
    return false;

  MatchInfo = [=](MachineIRBuilder &B) {
    if (C1 & C2) {
      B.buildAnd(Dst, R, B.buildConstant(Ty, C1 & C2));
      return;
    }
    auto Zero = B.buildConstant(Ty, 0);
    replaceRegWith(MRI, Dst, Zero->getOperand(0).getReg());
  };
  return true;
}

bool CombinerHelper::matchRedundantAnd(MachineInstr &MI,
                                       Register &Replacement) {
  // Given
  //
  // %y:_(sN) = G_SOMETHING
  // %x:_(sN) = G_SOMETHING
  // %res:_(sN) = G_AND %x, %y
  //
  // Eliminate the G_AND when it is known that x & y == x or x & y == y.
  //
  // Patterns like this can appear as a result of legalization. E.g.
  //
  // %cmp:_(s32) = G_ICMP intpred(pred), %x(s32), %y
  // %one:_(s32) = G_CONSTANT i32 1
  // %and:_(s32) = G_AND %cmp, %one
  //
  // In this case, G_ICMP only produces a single bit, so x & 1 == x.
  assert(MI.getOpcode() == TargetOpcode::G_AND);
  if (!KB)
    return false;

  Register AndDst = MI.getOperand(0).getReg();
  LLT DstTy = MRI.getType(AndDst);

  // FIXME: This should be removed once GISelKnownBits supports vectors.
  if (DstTy.isVector())
    return false;

  Register LHS = MI.getOperand(1).getReg();
  Register RHS = MI.getOperand(2).getReg();
  KnownBits LHSBits = KB->getKnownBits(LHS);
  KnownBits RHSBits = KB->getKnownBits(RHS);

  // Check that x & Mask == x.
  // x & 1 == x, always
  // x & 0 == x, only if x is also 0
  // Meaning Mask has no effect if every bit is either one in Mask or zero in x.
  //
  // Check if we can replace AndDst with the LHS of the G_AND
  if (canReplaceReg(AndDst, LHS, MRI) &&
      (LHSBits.Zero | RHSBits.One).isAllOnesValue()) {
    Replacement = LHS;
    return true;
  }

  // Check if we can replace AndDst with the RHS of the G_AND
  if (canReplaceReg(AndDst, RHS, MRI) &&
      (LHSBits.One | RHSBits.Zero).isAllOnesValue()) {
    Replacement = RHS;
    return true;
  }

  return false;
}

bool CombinerHelper::matchRedundantOr(MachineInstr &MI, Register &Replacement) {
  // Given
  //
  // %y:_(sN) = G_SOMETHING
  // %x:_(sN) = G_SOMETHING
  // %res:_(sN) = G_OR %x, %y
  //
  // Eliminate the G_OR when it is known that x | y == x or x | y == y.
  assert(MI.getOpcode() == TargetOpcode::G_OR);
  if (!KB)
    return false;

  Register OrDst = MI.getOperand(0).getReg();
  LLT DstTy = MRI.getType(OrDst);

  // FIXME: This should be removed once GISelKnownBits supports vectors.
  if (DstTy.isVector())
    return false;

  Register LHS = MI.getOperand(1).getReg();
  Register RHS = MI.getOperand(2).getReg();
  KnownBits LHSBits = KB->getKnownBits(LHS);
  KnownBits RHSBits = KB->getKnownBits(RHS);

  // Check that x | Mask == x.
  // x | 0 == x, always
  // x | 1 == x, only if x is also 1
  // Meaning Mask has no effect if every bit is either zero in Mask or one in x.
  //
  // Check if we can replace OrDst with the LHS of the G_OR
  if (canReplaceReg(OrDst, LHS, MRI) &&
      (LHSBits.One | RHSBits.Zero).isAllOnesValue()) {
    Replacement = LHS;
    return true;
  }

  // Check if we can replace OrDst with the RHS of the G_OR
  if (canReplaceReg(OrDst, RHS, MRI) &&
      (LHSBits.Zero | RHSBits.One).isAllOnesValue()) {
    Replacement = RHS;
    return true;
  }

  return false;
}

bool CombinerHelper::matchRedundantSExtInReg(MachineInstr &MI) {
  // If the input is already sign extended, just drop the extension.
  Register Src = MI.getOperand(1).getReg();
  unsigned ExtBits = MI.getOperand(2).getImm();
  unsigned TypeSize = MRI.getType(Src).getScalarSizeInBits();
  return KB->computeNumSignBits(Src) >= (TypeSize - ExtBits + 1);
}

static bool isConstValidTrue(const TargetLowering &TLI, unsigned ScalarSizeBits,
                             int64_t Cst, bool IsVector, bool IsFP) {
  // For i1, Cst will always be -1 regardless of boolean contents.
  return (ScalarSizeBits == 1 && Cst == -1) ||
         isConstTrueVal(TLI, Cst, IsVector, IsFP);
}

bool CombinerHelper::matchNotCmp(MachineInstr &MI,
                                 SmallVectorImpl<Register> &RegsToNegate) {
  assert(MI.getOpcode() == TargetOpcode::G_XOR);
  LLT Ty = MRI.getType(MI.getOperand(0).getReg());
  const auto &TLI = *Builder.getMF().getSubtarget().getTargetLowering();
  Register XorSrc;
  Register CstReg;
  // We match xor(src, true) here.
  if (!mi_match(MI.getOperand(0).getReg(), MRI,
                m_GXor(m_Reg(XorSrc), m_Reg(CstReg))))
    return false;

  if (!MRI.hasOneNonDBGUse(XorSrc))
    return false;

  // Check that XorSrc is the root of a tree of comparisons combined with ANDs
  // and ORs. The suffix of RegsToNegate starting from index I is used a work
  // list of tree nodes to visit.
  RegsToNegate.push_back(XorSrc);
  // Remember whether the comparisons are all integer or all floating point.
  bool IsInt = false;
  bool IsFP = false;
  for (unsigned I = 0; I < RegsToNegate.size(); ++I) {
    Register Reg = RegsToNegate[I];
    if (!MRI.hasOneNonDBGUse(Reg))
      return false;
    MachineInstr *Def = MRI.getVRegDef(Reg);
    switch (Def->getOpcode()) {
    default:
      // Don't match if the tree contains anything other than ANDs, ORs and
      // comparisons.
      return false;
    case TargetOpcode::G_ICMP:
      if (IsFP)
        return false;
      IsInt = true;
      // When we apply the combine we will invert the predicate.
      break;
    case TargetOpcode::G_FCMP:
      if (IsInt)
        return false;
      IsFP = true;
      // When we apply the combine we will invert the predicate.
      break;
    case TargetOpcode::G_AND:
    case TargetOpcode::G_OR:
      // Implement De Morgan's laws:
      // ~(x & y) -> ~x | ~y
      // ~(x | y) -> ~x & ~y
      // When we apply the combine we will change the opcode and recursively
      // negate the operands.
      RegsToNegate.push_back(Def->getOperand(1).getReg());
      RegsToNegate.push_back(Def->getOperand(2).getReg());
      break;
    }
  }

  // Now we know whether the comparisons are integer or floating point, check
  // the constant in the xor.
  int64_t Cst;
  if (Ty.isVector()) {
    MachineInstr *CstDef = MRI.getVRegDef(CstReg);
    auto MaybeCst = getBuildVectorConstantSplat(*CstDef, MRI);
    if (!MaybeCst)
      return false;
    if (!isConstValidTrue(TLI, Ty.getScalarSizeInBits(), *MaybeCst, true, IsFP))
      return false;
  } else {
    if (!mi_match(CstReg, MRI, m_ICst(Cst)))
      return false;
    if (!isConstValidTrue(TLI, Ty.getSizeInBits(), Cst, false, IsFP))
      return false;
  }

  return true;
}

void CombinerHelper::applyNotCmp(MachineInstr &MI,
                                 SmallVectorImpl<Register> &RegsToNegate) {
  for (Register Reg : RegsToNegate) {
    MachineInstr *Def = MRI.getVRegDef(Reg);
    Observer.changingInstr(*Def);
    // For each comparison, invert the opcode. For each AND and OR, change the
    // opcode.
    switch (Def->getOpcode()) {
    default:
      llvm_unreachable("Unexpected opcode");
    case TargetOpcode::G_ICMP:
    case TargetOpcode::G_FCMP: {
      MachineOperand &PredOp = Def->getOperand(1);
      CmpInst::Predicate NewP = CmpInst::getInversePredicate(
          (CmpInst::Predicate)PredOp.getPredicate());
      PredOp.setPredicate(NewP);
      break;
    }
    case TargetOpcode::G_AND:
      Def->setDesc(Builder.getTII().get(TargetOpcode::G_OR));
      break;
    case TargetOpcode::G_OR:
      Def->setDesc(Builder.getTII().get(TargetOpcode::G_AND));
      break;
    }
    Observer.changedInstr(*Def);
  }

  replaceRegWith(MRI, MI.getOperand(0).getReg(), MI.getOperand(1).getReg());
  MI.eraseFromParent();
}

bool CombinerHelper::matchXorOfAndWithSameReg(
    MachineInstr &MI, std::pair<Register, Register> &MatchInfo) {
  // Match (xor (and x, y), y) (or any of its commuted cases)
  assert(MI.getOpcode() == TargetOpcode::G_XOR);
  Register &X = MatchInfo.first;
  Register &Y = MatchInfo.second;
  Register AndReg = MI.getOperand(1).getReg();
  Register SharedReg = MI.getOperand(2).getReg();

  // Find a G_AND on either side of the G_XOR.
  // Look for one of
  //
  // (xor (and x, y), SharedReg)
  // (xor SharedReg, (and x, y))
  if (!mi_match(AndReg, MRI, m_GAnd(m_Reg(X), m_Reg(Y)))) {
    std::swap(AndReg, SharedReg);
    if (!mi_match(AndReg, MRI, m_GAnd(m_Reg(X), m_Reg(Y))))
      return false;
  }

  // Only do this if we'll eliminate the G_AND.
  if (!MRI.hasOneNonDBGUse(AndReg))
    return false;

  // We can combine if SharedReg is the same as either the LHS or RHS of the
  // G_AND.
  if (Y != SharedReg)
    std::swap(X, Y);
  return Y == SharedReg;
}

void CombinerHelper::applyXorOfAndWithSameReg(
    MachineInstr &MI, std::pair<Register, Register> &MatchInfo) {
  // Fold (xor (and x, y), y) -> (and (not x), y)
  Builder.setInstrAndDebugLoc(MI);
  Register X, Y;
  std::tie(X, Y) = MatchInfo;
  auto Not = Builder.buildNot(MRI.getType(X), X);
  Observer.changingInstr(MI);
  MI.setDesc(Builder.getTII().get(TargetOpcode::G_AND));
  MI.getOperand(1).setReg(Not->getOperand(0).getReg());
  MI.getOperand(2).setReg(Y);
  Observer.changedInstr(MI);
}

bool CombinerHelper::matchPtrAddZero(MachineInstr &MI) {
  Register DstReg = MI.getOperand(0).getReg();
  LLT Ty = MRI.getType(DstReg);
  const DataLayout &DL = Builder.getMF().getDataLayout();

  if (DL.isNonIntegralAddressSpace(Ty.getScalarType().getAddressSpace()))
    return false;

  if (Ty.isPointer()) {
    auto ConstVal = getConstantVRegVal(MI.getOperand(1).getReg(), MRI);
    return ConstVal && *ConstVal == 0;
  }

  assert(Ty.isVector() && "Expecting a vector type");
  const MachineInstr *VecMI = MRI.getVRegDef(MI.getOperand(1).getReg());
  return isBuildVectorAllZeros(*VecMI, MRI);
}

void CombinerHelper::applyPtrAddZero(MachineInstr &MI) {
  assert(MI.getOpcode() == TargetOpcode::G_PTR_ADD);
  Builder.setInstrAndDebugLoc(MI);
  Builder.buildIntToPtr(MI.getOperand(0), MI.getOperand(2));
  MI.eraseFromParent();
}

/// The second source operand is known to be a power of 2.
void CombinerHelper::applySimplifyURemByPow2(MachineInstr &MI) {
  Register DstReg = MI.getOperand(0).getReg();
  Register Src0 = MI.getOperand(1).getReg();
  Register Pow2Src1 = MI.getOperand(2).getReg();
  LLT Ty = MRI.getType(DstReg);
  Builder.setInstrAndDebugLoc(MI);

  // Fold (urem x, pow2) -> (and x, pow2-1)
  auto NegOne = Builder.buildConstant(Ty, -1);
  auto Add = Builder.buildAdd(Ty, Pow2Src1, NegOne);
  Builder.buildAnd(DstReg, Src0, Add);
  MI.eraseFromParent();
}

Optional<SmallVector<Register, 8>>
CombinerHelper::findCandidatesForLoadOrCombine(const MachineInstr *Root) const {
  assert(Root->getOpcode() == TargetOpcode::G_OR && "Expected G_OR only!");
  // We want to detect if Root is part of a tree which represents a bunch
  // of loads being merged into a larger load. We'll try to recognize patterns
  // like, for example:
  //
  //  Reg   Reg
  //   \    /
  //    OR_1   Reg
  //     \    /
  //      OR_2
  //        \     Reg
  //         .. /
  //        Root
  //
  //  Reg   Reg   Reg   Reg
  //     \ /       \   /
  //     OR_1      OR_2
  //       \       /
  //        \    /
  //         ...
  //         Root
  //
  // Each "Reg" may have been produced by a load + some arithmetic. This
  // function will save each of them.
  SmallVector<Register, 8> RegsToVisit;
  SmallVector<const MachineInstr *, 7> Ors = {Root};

  // In the "worst" case, we're dealing with a load for each byte. So, there
  // are at most #bytes - 1 ORs.
  const unsigned MaxIter =
      MRI.getType(Root->getOperand(0).getReg()).getSizeInBytes() - 1;
  for (unsigned Iter = 0; Iter < MaxIter; ++Iter) {
    if (Ors.empty())
      break;
    const MachineInstr *Curr = Ors.pop_back_val();
    Register OrLHS = Curr->getOperand(1).getReg();
    Register OrRHS = Curr->getOperand(2).getReg();

    // In the combine, we want to elimate the entire tree.
    if (!MRI.hasOneNonDBGUse(OrLHS) || !MRI.hasOneNonDBGUse(OrRHS))
      return None;

    // If it's a G_OR, save it and continue to walk. If it's not, then it's
    // something that may be a load + arithmetic.
    if (const MachineInstr *Or = getOpcodeDef(TargetOpcode::G_OR, OrLHS, MRI))
      Ors.push_back(Or);
    else
      RegsToVisit.push_back(OrLHS);
    if (const MachineInstr *Or = getOpcodeDef(TargetOpcode::G_OR, OrRHS, MRI))
      Ors.push_back(Or);
    else
      RegsToVisit.push_back(OrRHS);
  }

  // We're going to try and merge each register into a wider power-of-2 type,
  // so we ought to have an even number of registers.
  if (RegsToVisit.empty() || RegsToVisit.size() % 2 != 0)
    return None;
  return RegsToVisit;
}

/// Helper function for findLoadOffsetsForLoadOrCombine.
///
/// Check if \p Reg is the result of loading a \p MemSizeInBits wide value,
/// and then moving that value into a specific byte offset.
///
/// e.g. x[i] << 24
///
/// \returns The load instruction and the byte offset it is moved into.
static Optional<std::pair<GZExtLoad *, int64_t>>
matchLoadAndBytePosition(Register Reg, unsigned MemSizeInBits,
                         const MachineRegisterInfo &MRI) {
  assert(MRI.hasOneNonDBGUse(Reg) &&
         "Expected Reg to only have one non-debug use?");
  Register MaybeLoad;
  int64_t Shift;
  if (!mi_match(Reg, MRI,
                m_OneNonDBGUse(m_GShl(m_Reg(MaybeLoad), m_ICst(Shift))))) {
    Shift = 0;
    MaybeLoad = Reg;
  }

  if (Shift % MemSizeInBits != 0)
    return None;

  // TODO: Handle other types of loads.
  auto *Load = getOpcodeDef<GZExtLoad>(MaybeLoad, MRI);
  if (!Load)
    return None;

  if (!Load->isUnordered() || Load->getMemSizeInBits() != MemSizeInBits)
    return None;

  return std::make_pair(Load, Shift / MemSizeInBits);
}

Optional<std::tuple<GZExtLoad *, int64_t, GZExtLoad *>>
CombinerHelper::findLoadOffsetsForLoadOrCombine(
    SmallDenseMap<int64_t, int64_t, 8> &MemOffset2Idx,
    const SmallVector<Register, 8> &RegsToVisit, const unsigned MemSizeInBits) {

  // Each load found for the pattern. There should be one for each RegsToVisit.
  SmallSetVector<const MachineInstr *, 8> Loads;

  // The lowest index used in any load. (The lowest "i" for each x[i].)
  int64_t LowestIdx = INT64_MAX;

  // The load which uses the lowest index.
  GZExtLoad *LowestIdxLoad = nullptr;

  // Keeps track of the load indices we see. We shouldn't see any indices twice.
  SmallSet<int64_t, 8> SeenIdx;

  // Ensure each load is in the same MBB.
  // TODO: Support multiple MachineBasicBlocks.
  MachineBasicBlock *MBB = nullptr;
  const MachineMemOperand *MMO = nullptr;

  // Earliest instruction-order load in the pattern.
  GZExtLoad *EarliestLoad = nullptr;

  // Latest instruction-order load in the pattern.
  GZExtLoad *LatestLoad = nullptr;

  // Base pointer which every load should share.
  Register BasePtr;

  // We want to find a load for each register. Each load should have some
  // appropriate bit twiddling arithmetic. During this loop, we will also keep
  // track of the load which uses the lowest index. Later, we will check if we
  // can use its pointer in the final, combined load.
  for (auto Reg : RegsToVisit) {
    // Find the load, and find the position that it will end up in (e.g. a
    // shifted) value.
    auto LoadAndPos = matchLoadAndBytePosition(Reg, MemSizeInBits, MRI);
    if (!LoadAndPos)
      return None;
    GZExtLoad *Load;
    int64_t DstPos;
    std::tie(Load, DstPos) = *LoadAndPos;

    // TODO: Handle multiple MachineBasicBlocks. Currently not handled because
    // it is difficult to check for stores/calls/etc between loads.
    MachineBasicBlock *LoadMBB = Load->getParent();
    if (!MBB)
      MBB = LoadMBB;
    if (LoadMBB != MBB)
      return None;

    // Make sure that the MachineMemOperands of every seen load are compatible.
    auto &LoadMMO = Load->getMMO();
    if (!MMO)
      MMO = &LoadMMO;
    if (MMO->getAddrSpace() != LoadMMO.getAddrSpace())
      return None;

    // Find out what the base pointer and index for the load is.
    Register LoadPtr;
    int64_t Idx;
    if (!mi_match(Load->getOperand(1).getReg(), MRI,
                  m_GPtrAdd(m_Reg(LoadPtr), m_ICst(Idx)))) {
      LoadPtr = Load->getOperand(1).getReg();
      Idx = 0;
    }

    // Don't combine things like a[i], a[i] -> a bigger load.
    if (!SeenIdx.insert(Idx).second)
      return None;

    // Every load must share the same base pointer; don't combine things like:
    //
    // a[i], b[i + 1] -> a bigger load.
    if (!BasePtr.isValid())
      BasePtr = LoadPtr;
    if (BasePtr != LoadPtr)
      return None;

    if (Idx < LowestIdx) {
      LowestIdx = Idx;
      LowestIdxLoad = Load;
    }

    // Keep track of the byte offset that this load ends up at. If we have seen
    // the byte offset, then stop here. We do not want to combine:
    //
    // a[i] << 16, a[i + k] << 16 -> a bigger load.
    if (!MemOffset2Idx.try_emplace(DstPos, Idx).second)
      return None;
    Loads.insert(Load);

    // Keep track of the position of the earliest/latest loads in the pattern.
    // We will check that there are no load fold barriers between them later
    // on.
    //
    // FIXME: Is there a better way to check for load fold barriers?
    if (!EarliestLoad || dominates(*Load, *EarliestLoad))
      EarliestLoad = Load;
    if (!LatestLoad || dominates(*LatestLoad, *Load))
      LatestLoad = Load;
  }

  // We found a load for each register. Let's check if each load satisfies the
  // pattern.
  assert(Loads.size() == RegsToVisit.size() &&
         "Expected to find a load for each register?");
  assert(EarliestLoad != LatestLoad && EarliestLoad &&
         LatestLoad && "Expected at least two loads?");

  // Check if there are any stores, calls, etc. between any of the loads. If
  // there are, then we can't safely perform the combine.
  //
  // MaxIter is chosen based off the (worst case) number of iterations it
  // typically takes to succeed in the LLVM test suite plus some padding.
  //
  // FIXME: Is there a better way to check for load fold barriers?
  const unsigned MaxIter = 20;
  unsigned Iter = 0;
  for (const auto &MI : instructionsWithoutDebug(EarliestLoad->getIterator(),
                                                 LatestLoad->getIterator())) {
    if (Loads.count(&MI))
      continue;
    if (MI.isLoadFoldBarrier())
      return None;
    if (Iter++ == MaxIter)
      return None;
  }

  return std::make_tuple(LowestIdxLoad, LowestIdx, LatestLoad);
}

bool CombinerHelper::matchLoadOrCombine(
    MachineInstr &MI, std::function<void(MachineIRBuilder &)> &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_OR);
  MachineFunction &MF = *MI.getMF();
  // Assuming a little-endian target, transform:
  //  s8 *a = ...
  //  s32 val = a[0] | (a[1] << 8) | (a[2] << 16) | (a[3] << 24)
  // =>
  //  s32 val = *((i32)a)
  //
  //  s8 *a = ...
  //  s32 val = (a[0] << 24) | (a[1] << 16) | (a[2] << 8) | a[3]
  // =>
  //  s32 val = BSWAP(*((s32)a))
  Register Dst = MI.getOperand(0).getReg();
  LLT Ty = MRI.getType(Dst);
  if (Ty.isVector())
    return false;

  // We need to combine at least two loads into this type. Since the smallest
  // possible load is into a byte, we need at least a 16-bit wide type.
  const unsigned WideMemSizeInBits = Ty.getSizeInBits();
  if (WideMemSizeInBits < 16 || WideMemSizeInBits % 8 != 0)
    return false;

  // Match a collection of non-OR instructions in the pattern.
  auto RegsToVisit = findCandidatesForLoadOrCombine(&MI);
  if (!RegsToVisit)
    return false;

  // We have a collection of non-OR instructions. Figure out how wide each of
  // the small loads should be based off of the number of potential loads we
  // found.
  const unsigned NarrowMemSizeInBits = WideMemSizeInBits / RegsToVisit->size();
  if (NarrowMemSizeInBits % 8 != 0)
    return false;

  // Check if each register feeding into each OR is a load from the same
  // base pointer + some arithmetic.
  //
  // e.g. a[0], a[1] << 8, a[2] << 16, etc.
  //
  // Also verify that each of these ends up putting a[i] into the same memory
  // offset as a load into a wide type would.
  SmallDenseMap<int64_t, int64_t, 8> MemOffset2Idx;
  GZExtLoad *LowestIdxLoad, *LatestLoad;
  int64_t LowestIdx;
  auto MaybeLoadInfo = findLoadOffsetsForLoadOrCombine(
      MemOffset2Idx, *RegsToVisit, NarrowMemSizeInBits);
  if (!MaybeLoadInfo)
    return false;
  std::tie(LowestIdxLoad, LowestIdx, LatestLoad) = *MaybeLoadInfo;

  // We have a bunch of loads being OR'd together. Using the addresses + offsets
  // we found before, check if this corresponds to a big or little endian byte
  // pattern. If it does, then we can represent it using a load + possibly a
  // BSWAP.
  bool IsBigEndianTarget = MF.getDataLayout().isBigEndian();
  Optional<bool> IsBigEndian = isBigEndian(MemOffset2Idx, LowestIdx);
  if (!IsBigEndian.hasValue())
    return false;
  bool NeedsBSwap = IsBigEndianTarget != *IsBigEndian;
  if (NeedsBSwap && !isLegalOrBeforeLegalizer({TargetOpcode::G_BSWAP, {Ty}}))
    return false;

  // Make sure that the load from the lowest index produces offset 0 in the
  // final value.
  //
  // This ensures that we won't combine something like this:
  //
  // load x[i] -> byte 2
  // load x[i+1] -> byte 0 ---> wide_load x[i]
  // load x[i+2] -> byte 1
  const unsigned NumLoadsInTy = WideMemSizeInBits / NarrowMemSizeInBits;
  const unsigned ZeroByteOffset =
      *IsBigEndian
          ? bigEndianByteAt(NumLoadsInTy, 0)
          : littleEndianByteAt(NumLoadsInTy, 0);
  auto ZeroOffsetIdx = MemOffset2Idx.find(ZeroByteOffset);
  if (ZeroOffsetIdx == MemOffset2Idx.end() ||
      ZeroOffsetIdx->second != LowestIdx)
    return false;

  // We wil reuse the pointer from the load which ends up at byte offset 0. It
  // may not use index 0.
  Register Ptr = LowestIdxLoad->getPointerReg();
  const MachineMemOperand &MMO = LowestIdxLoad->getMMO();
  LegalityQuery::MemDesc MMDesc;
  MMDesc.MemoryTy = Ty;
  MMDesc.AlignInBits = MMO.getAlign().value() * 8;
  MMDesc.Ordering = MMO.getSuccessOrdering();
  if (!isLegalOrBeforeLegalizer(
          {TargetOpcode::G_LOAD, {Ty, MRI.getType(Ptr)}, {MMDesc}}))
    return false;
  auto PtrInfo = MMO.getPointerInfo();
  auto *NewMMO = MF.getMachineMemOperand(&MMO, PtrInfo, WideMemSizeInBits / 8);

  // Load must be allowed and fast on the target.
  LLVMContext &C = MF.getFunction().getContext();
  auto &DL = MF.getDataLayout();
  bool Fast = false;
  if (!getTargetLowering().allowsMemoryAccess(C, DL, Ty, *NewMMO, &Fast) ||
      !Fast)
    return false;

  MatchInfo = [=](MachineIRBuilder &MIB) {
    MIB.setInstrAndDebugLoc(*LatestLoad);
    Register LoadDst = NeedsBSwap ? MRI.cloneVirtualRegister(Dst) : Dst;
    MIB.buildLoad(LoadDst, Ptr, *NewMMO);
    if (NeedsBSwap)
      MIB.buildBSwap(Dst, LoadDst);
  };
  return true;
}

bool CombinerHelper::matchExtendThroughPhis(MachineInstr &MI,
                                            MachineInstr *&ExtMI) {
  assert(MI.getOpcode() == TargetOpcode::G_PHI);

  Register DstReg = MI.getOperand(0).getReg();

  // TODO: Extending a vector may be expensive, don't do this until heuristics
  // are better.
  if (MRI.getType(DstReg).isVector())
    return false;

  // Try to match a phi, whose only use is an extend.
  if (!MRI.hasOneNonDBGUse(DstReg))
    return false;
  ExtMI = &*MRI.use_instr_nodbg_begin(DstReg);
  switch (ExtMI->getOpcode()) {
  case TargetOpcode::G_ANYEXT:
    return true; // G_ANYEXT is usually free.
  case TargetOpcode::G_ZEXT:
  case TargetOpcode::G_SEXT:
    break;
  default:
    return false;
  }

  // If the target is likely to fold this extend away, don't propagate.
  if (Builder.getTII().isExtendLikelyToBeFolded(*ExtMI, MRI))
    return false;

  // We don't want to propagate the extends unless there's a good chance that
  // they'll be optimized in some way.
  // Collect the unique incoming values.
  SmallPtrSet<MachineInstr *, 4> InSrcs;
  for (unsigned Idx = 1; Idx < MI.getNumOperands(); Idx += 2) {
    auto *DefMI = getDefIgnoringCopies(MI.getOperand(Idx).getReg(), MRI);
    switch (DefMI->getOpcode()) {
    case TargetOpcode::G_LOAD:
    case TargetOpcode::G_TRUNC:
    case TargetOpcode::G_SEXT:
    case TargetOpcode::G_ZEXT:
    case TargetOpcode::G_ANYEXT:
    case TargetOpcode::G_CONSTANT:
      InSrcs.insert(getDefIgnoringCopies(MI.getOperand(Idx).getReg(), MRI));
      // Don't try to propagate if there are too many places to create new
      // extends, chances are it'll increase code size.
      if (InSrcs.size() > 2)
        return false;
      break;
    default:
      return false;
    }
  }
  return true;
}

void CombinerHelper::applyExtendThroughPhis(MachineInstr &MI,
                                            MachineInstr *&ExtMI) {
  assert(MI.getOpcode() == TargetOpcode::G_PHI);
  Register DstReg = ExtMI->getOperand(0).getReg();
  LLT ExtTy = MRI.getType(DstReg);

  // Propagate the extension into the block of each incoming reg's block.
  // Use a SetVector here because PHIs can have duplicate edges, and we want
  // deterministic iteration order.
  SmallSetVector<MachineInstr *, 8> SrcMIs;
  SmallDenseMap<MachineInstr *, MachineInstr *, 8> OldToNewSrcMap;
  for (unsigned SrcIdx = 1; SrcIdx < MI.getNumOperands(); SrcIdx += 2) {
    auto *SrcMI = MRI.getVRegDef(MI.getOperand(SrcIdx).getReg());
    if (!SrcMIs.insert(SrcMI))
      continue;

    // Build an extend after each src inst.
    auto *MBB = SrcMI->getParent();
    MachineBasicBlock::iterator InsertPt = ++SrcMI->getIterator();
    if (InsertPt != MBB->end() && InsertPt->isPHI())
      InsertPt = MBB->getFirstNonPHI();

    Builder.setInsertPt(*SrcMI->getParent(), InsertPt);
    Builder.setDebugLoc(MI.getDebugLoc());
    auto NewExt = Builder.buildExtOrTrunc(ExtMI->getOpcode(), ExtTy,
                                          SrcMI->getOperand(0).getReg());
    OldToNewSrcMap[SrcMI] = NewExt;
  }

  // Create a new phi with the extended inputs.
  Builder.setInstrAndDebugLoc(MI);
  auto NewPhi = Builder.buildInstrNoInsert(TargetOpcode::G_PHI);
  NewPhi.addDef(DstReg);
  for (unsigned SrcIdx = 1; SrcIdx < MI.getNumOperands(); ++SrcIdx) {
    auto &MO = MI.getOperand(SrcIdx);
    if (!MO.isReg()) {
      NewPhi.addMBB(MO.getMBB());
      continue;
    }
    auto *NewSrc = OldToNewSrcMap[MRI.getVRegDef(MO.getReg())];
    NewPhi.addUse(NewSrc->getOperand(0).getReg());
  }
  Builder.insertInstr(NewPhi);
  ExtMI->eraseFromParent();
}

bool CombinerHelper::matchExtractVecEltBuildVec(MachineInstr &MI,
                                                Register &Reg) {
  assert(MI.getOpcode() == TargetOpcode::G_EXTRACT_VECTOR_ELT);
  // If we have a constant index, look for a G_BUILD_VECTOR source
  // and find the source register that the index maps to.
  Register SrcVec = MI.getOperand(1).getReg();
  LLT SrcTy = MRI.getType(SrcVec);
  if (!isLegalOrBeforeLegalizer(
          {TargetOpcode::G_BUILD_VECTOR, {SrcTy, SrcTy.getElementType()}}))
    return false;

  auto Cst = getConstantVRegValWithLookThrough(MI.getOperand(2).getReg(), MRI);
  if (!Cst || Cst->Value.getZExtValue() >= SrcTy.getNumElements())
    return false;

  unsigned VecIdx = Cst->Value.getZExtValue();
  MachineInstr *BuildVecMI =
      getOpcodeDef(TargetOpcode::G_BUILD_VECTOR, SrcVec, MRI);
  if (!BuildVecMI) {
    BuildVecMI = getOpcodeDef(TargetOpcode::G_BUILD_VECTOR_TRUNC, SrcVec, MRI);
    if (!BuildVecMI)
      return false;
    LLT ScalarTy = MRI.getType(BuildVecMI->getOperand(1).getReg());
    if (!isLegalOrBeforeLegalizer(
            {TargetOpcode::G_BUILD_VECTOR_TRUNC, {SrcTy, ScalarTy}}))
      return false;
  }

  EVT Ty(getMVTForLLT(SrcTy));
  if (!MRI.hasOneNonDBGUse(SrcVec) &&
      !getTargetLowering().aggressivelyPreferBuildVectorSources(Ty))
    return false;

  Reg = BuildVecMI->getOperand(VecIdx + 1).getReg();
  return true;
}

void CombinerHelper::applyExtractVecEltBuildVec(MachineInstr &MI,
                                                Register &Reg) {
  // Check the type of the register, since it may have come from a
  // G_BUILD_VECTOR_TRUNC.
  LLT ScalarTy = MRI.getType(Reg);
  Register DstReg = MI.getOperand(0).getReg();
  LLT DstTy = MRI.getType(DstReg);

  Builder.setInstrAndDebugLoc(MI);
  if (ScalarTy != DstTy) {
    assert(ScalarTy.getSizeInBits() > DstTy.getSizeInBits());
    Builder.buildTrunc(DstReg, Reg);
    MI.eraseFromParent();
    return;
  }
  replaceSingleDefInstWithReg(MI, Reg);
}

bool CombinerHelper::matchExtractAllEltsFromBuildVector(
    MachineInstr &MI,
    SmallVectorImpl<std::pair<Register, MachineInstr *>> &SrcDstPairs) {
  assert(MI.getOpcode() == TargetOpcode::G_BUILD_VECTOR);
  // This combine tries to find build_vector's which have every source element
  // extracted using G_EXTRACT_VECTOR_ELT. This can happen when transforms like
  // the masked load scalarization is run late in the pipeline. There's already
  // a combine for a similar pattern starting from the extract, but that
  // doesn't attempt to do it if there are multiple uses of the build_vector,
  // which in this case is true. Starting the combine from the build_vector
  // feels more natural than trying to find sibling nodes of extracts.
  // E.g.
  //  %vec(<4 x s32>) = G_BUILD_VECTOR %s1(s32), %s2, %s3, %s4
  //  %ext1 = G_EXTRACT_VECTOR_ELT %vec, 0
  //  %ext2 = G_EXTRACT_VECTOR_ELT %vec, 1
  //  %ext3 = G_EXTRACT_VECTOR_ELT %vec, 2
  //  %ext4 = G_EXTRACT_VECTOR_ELT %vec, 3
  // ==>
  // replace ext{1,2,3,4} with %s{1,2,3,4}

  Register DstReg = MI.getOperand(0).getReg();
  LLT DstTy = MRI.getType(DstReg);
  unsigned NumElts = DstTy.getNumElements();

  SmallBitVector ExtractedElts(NumElts);
  for (auto &II : make_range(MRI.use_instr_nodbg_begin(DstReg),
                             MRI.use_instr_nodbg_end())) {
    if (II.getOpcode() != TargetOpcode::G_EXTRACT_VECTOR_ELT)
      return false;
    auto Cst = getConstantVRegVal(II.getOperand(2).getReg(), MRI);
    if (!Cst)
      return false;
    unsigned Idx = Cst.getValue().getZExtValue();
    if (Idx >= NumElts)
      return false; // Out of range.
    ExtractedElts.set(Idx);
    SrcDstPairs.emplace_back(
        std::make_pair(MI.getOperand(Idx + 1).getReg(), &II));
  }
  // Match if every element was extracted.
  return ExtractedElts.all();
}

void CombinerHelper::applyExtractAllEltsFromBuildVector(
    MachineInstr &MI,
    SmallVectorImpl<std::pair<Register, MachineInstr *>> &SrcDstPairs) {
  assert(MI.getOpcode() == TargetOpcode::G_BUILD_VECTOR);
  for (auto &Pair : SrcDstPairs) {
    auto *ExtMI = Pair.second;
    replaceRegWith(MRI, ExtMI->getOperand(0).getReg(), Pair.first);
    ExtMI->eraseFromParent();
  }
  MI.eraseFromParent();
}

void CombinerHelper::applyBuildFn(
    MachineInstr &MI, std::function<void(MachineIRBuilder &)> &MatchInfo) {
  Builder.setInstrAndDebugLoc(MI);
  MatchInfo(Builder);
  MI.eraseFromParent();
}

void CombinerHelper::applyBuildFnNoErase(
    MachineInstr &MI, std::function<void(MachineIRBuilder &)> &MatchInfo) {
  Builder.setInstrAndDebugLoc(MI);
  MatchInfo(Builder);
}

/// Match an FSHL or FSHR that can be combined to a ROTR or ROTL rotate.
bool CombinerHelper::matchFunnelShiftToRotate(MachineInstr &MI) {
  unsigned Opc = MI.getOpcode();
  assert(Opc == TargetOpcode::G_FSHL || Opc == TargetOpcode::G_FSHR);
  Register X = MI.getOperand(1).getReg();
  Register Y = MI.getOperand(2).getReg();
  if (X != Y)
    return false;
  unsigned RotateOpc =
      Opc == TargetOpcode::G_FSHL ? TargetOpcode::G_ROTL : TargetOpcode::G_ROTR;
  return isLegalOrBeforeLegalizer({RotateOpc, {MRI.getType(X), MRI.getType(Y)}});
}

void CombinerHelper::applyFunnelShiftToRotate(MachineInstr &MI) {
  unsigned Opc = MI.getOpcode();
  assert(Opc == TargetOpcode::G_FSHL || Opc == TargetOpcode::G_FSHR);
  bool IsFSHL = Opc == TargetOpcode::G_FSHL;
  Observer.changingInstr(MI);
  MI.setDesc(Builder.getTII().get(IsFSHL ? TargetOpcode::G_ROTL
                                         : TargetOpcode::G_ROTR));
  MI.RemoveOperand(2);
  Observer.changedInstr(MI);
}

// Fold (rot x, c) -> (rot x, c % BitSize)
bool CombinerHelper::matchRotateOutOfRange(MachineInstr &MI) {
  assert(MI.getOpcode() == TargetOpcode::G_ROTL ||
         MI.getOpcode() == TargetOpcode::G_ROTR);
  unsigned Bitsize =
      MRI.getType(MI.getOperand(0).getReg()).getScalarSizeInBits();
  Register AmtReg = MI.getOperand(2).getReg();
  bool OutOfRange = false;
  auto MatchOutOfRange = [Bitsize, &OutOfRange](const Constant *C) {
    if (auto *CI = dyn_cast<ConstantInt>(C))
      OutOfRange |= CI->getValue().uge(Bitsize);
    return true;
  };
  return matchUnaryPredicate(MRI, AmtReg, MatchOutOfRange) && OutOfRange;
}

void CombinerHelper::applyRotateOutOfRange(MachineInstr &MI) {
  assert(MI.getOpcode() == TargetOpcode::G_ROTL ||
         MI.getOpcode() == TargetOpcode::G_ROTR);
  unsigned Bitsize =
      MRI.getType(MI.getOperand(0).getReg()).getScalarSizeInBits();
  Builder.setInstrAndDebugLoc(MI);
  Register Amt = MI.getOperand(2).getReg();
  LLT AmtTy = MRI.getType(Amt);
  auto Bits = Builder.buildConstant(AmtTy, Bitsize);
  Amt = Builder.buildURem(AmtTy, MI.getOperand(2).getReg(), Bits).getReg(0);
  Observer.changingInstr(MI);
  MI.getOperand(2).setReg(Amt);
  Observer.changedInstr(MI);
}

bool CombinerHelper::matchICmpToTrueFalseKnownBits(MachineInstr &MI,
                                                   int64_t &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_ICMP);
  auto Pred = static_cast<CmpInst::Predicate>(MI.getOperand(1).getPredicate());
  auto KnownLHS = KB->getKnownBits(MI.getOperand(2).getReg());
  auto KnownRHS = KB->getKnownBits(MI.getOperand(3).getReg());
  Optional<bool> KnownVal;
  switch (Pred) {
  default:
    llvm_unreachable("Unexpected G_ICMP predicate?");
  case CmpInst::ICMP_EQ:
    KnownVal = KnownBits::eq(KnownLHS, KnownRHS);
    break;
  case CmpInst::ICMP_NE:
    KnownVal = KnownBits::ne(KnownLHS, KnownRHS);
    break;
  case CmpInst::ICMP_SGE:
    KnownVal = KnownBits::sge(KnownLHS, KnownRHS);
    break;
  case CmpInst::ICMP_SGT:
    KnownVal = KnownBits::sgt(KnownLHS, KnownRHS);
    break;
  case CmpInst::ICMP_SLE:
    KnownVal = KnownBits::sle(KnownLHS, KnownRHS);
    break;
  case CmpInst::ICMP_SLT:
    KnownVal = KnownBits::slt(KnownLHS, KnownRHS);
    break;
  case CmpInst::ICMP_UGE:
    KnownVal = KnownBits::uge(KnownLHS, KnownRHS);
    break;
  case CmpInst::ICMP_UGT:
    KnownVal = KnownBits::ugt(KnownLHS, KnownRHS);
    break;
  case CmpInst::ICMP_ULE:
    KnownVal = KnownBits::ule(KnownLHS, KnownRHS);
    break;
  case CmpInst::ICMP_ULT:
    KnownVal = KnownBits::ult(KnownLHS, KnownRHS);
    break;
  }
  if (!KnownVal)
    return false;
  MatchInfo =
      *KnownVal
          ? getICmpTrueVal(getTargetLowering(),
                           /*IsVector = */
                           MRI.getType(MI.getOperand(0).getReg()).isVector(),
                           /* IsFP = */ false)
          : 0;
  return true;
}

/// Form a G_SBFX from a G_SEXT_INREG fed by a right shift.
bool CombinerHelper::matchBitfieldExtractFromSExtInReg(
    MachineInstr &MI, std::function<void(MachineIRBuilder &)> &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_SEXT_INREG);
  Register Dst = MI.getOperand(0).getReg();
  Register Src = MI.getOperand(1).getReg();
  LLT Ty = MRI.getType(Src);
  LLT ExtractTy = getTargetLowering().getPreferredShiftAmountTy(Ty);
  if (!LI || !LI->isLegalOrCustom({TargetOpcode::G_SBFX, {Ty, ExtractTy}}))
    return false;
  int64_t Width = MI.getOperand(2).getImm();
  Register ShiftSrc;
  int64_t ShiftImm;
  if (!mi_match(
          Src, MRI,
          m_OneNonDBGUse(m_any_of(m_GAShr(m_Reg(ShiftSrc), m_ICst(ShiftImm)),
                                  m_GLShr(m_Reg(ShiftSrc), m_ICst(ShiftImm))))))
    return false;
  if (ShiftImm < 0 || ShiftImm + Width > Ty.getScalarSizeInBits())
    return false;

  MatchInfo = [=](MachineIRBuilder &B) {
    auto Cst1 = B.buildConstant(ExtractTy, ShiftImm);
    auto Cst2 = B.buildConstant(ExtractTy, Width);
    B.buildSbfx(Dst, ShiftSrc, Cst1, Cst2);
  };
  return true;
}

/// Form a G_UBFX from "(a srl b) & mask", where b and mask are constants.
bool CombinerHelper::matchBitfieldExtractFromAnd(
    MachineInstr &MI, std::function<void(MachineIRBuilder &)> &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_AND);
  Register Dst = MI.getOperand(0).getReg();
  LLT Ty = MRI.getType(Dst);
  if (!getTargetLowering().isConstantUnsignedBitfieldExtactLegal(
          TargetOpcode::G_UBFX, Ty, Ty))
    return false;

  int64_t AndImm, LSBImm;
  Register ShiftSrc;
  const unsigned Size = Ty.getScalarSizeInBits();
  if (!mi_match(MI.getOperand(0).getReg(), MRI,
                m_GAnd(m_OneNonDBGUse(m_GLShr(m_Reg(ShiftSrc), m_ICst(LSBImm))),
                       m_ICst(AndImm))))
    return false;

  // The mask is a mask of the low bits iff imm & (imm+1) == 0.
  auto MaybeMask = static_cast<uint64_t>(AndImm);
  if (MaybeMask & (MaybeMask + 1))
    return false;

  // LSB must fit within the register.
  if (static_cast<uint64_t>(LSBImm) >= Size)
    return false;

  LLT ExtractTy = getTargetLowering().getPreferredShiftAmountTy(Ty);
  uint64_t Width = APInt(Size, AndImm).countTrailingOnes();
  MatchInfo = [=](MachineIRBuilder &B) {
    auto WidthCst = B.buildConstant(ExtractTy, Width);
    auto LSBCst = B.buildConstant(ExtractTy, LSBImm);
    B.buildInstr(TargetOpcode::G_UBFX, {Dst}, {ShiftSrc, LSBCst, WidthCst});
  };
  return true;
}

bool CombinerHelper::reassociationCanBreakAddressingModePattern(
    MachineInstr &PtrAdd) {
  assert(PtrAdd.getOpcode() == TargetOpcode::G_PTR_ADD);

  Register Src1Reg = PtrAdd.getOperand(1).getReg();
  MachineInstr *Src1Def = getOpcodeDef(TargetOpcode::G_PTR_ADD, Src1Reg, MRI);
  if (!Src1Def)
    return false;

  Register Src2Reg = PtrAdd.getOperand(2).getReg();

  if (MRI.hasOneNonDBGUse(Src1Reg))
    return false;

  auto C1 = getConstantVRegVal(Src1Def->getOperand(2).getReg(), MRI);
  if (!C1)
    return false;
  auto C2 = getConstantVRegVal(Src2Reg, MRI);
  if (!C2)
    return false;

  const APInt &C1APIntVal = *C1;
  const APInt &C2APIntVal = *C2;
  const int64_t CombinedValue = (C1APIntVal + C2APIntVal).getSExtValue();

  for (auto &UseMI : MRI.use_nodbg_instructions(Src1Reg)) {
    // This combine may end up running before ptrtoint/inttoptr combines
    // manage to eliminate redundant conversions, so try to look through them.
    MachineInstr *ConvUseMI = &UseMI;
    unsigned ConvUseOpc = ConvUseMI->getOpcode();
    while (ConvUseOpc == TargetOpcode::G_INTTOPTR ||
           ConvUseOpc == TargetOpcode::G_PTRTOINT) {
      Register DefReg = ConvUseMI->getOperand(0).getReg();
      if (!MRI.hasOneNonDBGUse(DefReg))
        break;
      ConvUseMI = &*MRI.use_instr_nodbg_begin(DefReg);
      ConvUseOpc = ConvUseMI->getOpcode();
    }
    auto LoadStore = ConvUseOpc == TargetOpcode::G_LOAD ||
                     ConvUseOpc == TargetOpcode::G_STORE;
    if (!LoadStore)
      continue;
    // Is x[offset2] already not a legal addressing mode? If so then
    // reassociating the constants breaks nothing (we test offset2 because
    // that's the one we hope to fold into the load or store).
    TargetLoweringBase::AddrMode AM;
    AM.HasBaseReg = true;
    AM.BaseOffs = C2APIntVal.getSExtValue();
    unsigned AS =
        MRI.getType(ConvUseMI->getOperand(1).getReg()).getAddressSpace();
    Type *AccessTy =
        getTypeForLLT(MRI.getType(ConvUseMI->getOperand(0).getReg()),
                      PtrAdd.getMF()->getFunction().getContext());
    const auto &TLI = *PtrAdd.getMF()->getSubtarget().getTargetLowering();
    if (!TLI.isLegalAddressingMode(PtrAdd.getMF()->getDataLayout(), AM,
                                   AccessTy, AS))
      continue;

    // Would x[offset1+offset2] still be a legal addressing mode?
    AM.BaseOffs = CombinedValue;
    if (!TLI.isLegalAddressingMode(PtrAdd.getMF()->getDataLayout(), AM,
                                   AccessTy, AS))
      return true;
  }

  return false;
}

bool CombinerHelper::matchReassocPtrAdd(
    MachineInstr &MI, std::function<void(MachineIRBuilder &)> &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_PTR_ADD);
  // We're trying to match a few pointer computation patterns here for
  // re-association opportunities.
  // 1) Isolating a constant operand to be on the RHS, e.g.:
  // G_PTR_ADD(BASE, G_ADD(X, C)) -> G_PTR_ADD(G_PTR_ADD(BASE, X), C)
  //
  // 2) Folding two constants in each sub-tree as long as such folding
  // doesn't break a legal addressing mode.
  // G_PTR_ADD(G_PTR_ADD(BASE, C1), C2) -> G_PTR_ADD(BASE, C1+C2)
  Register Src1Reg = MI.getOperand(1).getReg();
  Register Src2Reg = MI.getOperand(2).getReg();
  MachineInstr *LHS = MRI.getVRegDef(Src1Reg);
  MachineInstr *RHS = MRI.getVRegDef(Src2Reg);

  if (LHS->getOpcode() != TargetOpcode::G_PTR_ADD) {
    // Try to match example 1).
    if (RHS->getOpcode() != TargetOpcode::G_ADD)
      return false;
    auto C2 = getConstantVRegVal(RHS->getOperand(2).getReg(), MRI);
    if (!C2)
      return false;

    MatchInfo = [=,&MI](MachineIRBuilder &B) {
      LLT PtrTy = MRI.getType(MI.getOperand(0).getReg());

      auto NewBase =
          Builder.buildPtrAdd(PtrTy, Src1Reg, RHS->getOperand(1).getReg());
      Observer.changingInstr(MI);
      MI.getOperand(1).setReg(NewBase.getReg(0));
      MI.getOperand(2).setReg(RHS->getOperand(2).getReg());
      Observer.changedInstr(MI);
    };
  } else {
    // Try to match example 2.
    Register LHSSrc1 = LHS->getOperand(1).getReg();
    Register LHSSrc2 = LHS->getOperand(2).getReg();
    auto C1 = getConstantVRegVal(LHSSrc2, MRI);
    if (!C1)
      return false;
    auto C2 = getConstantVRegVal(Src2Reg, MRI);
    if (!C2)
      return false;

    MatchInfo = [=, &MI](MachineIRBuilder &B) {
      auto NewCst = B.buildConstant(MRI.getType(Src2Reg), *C1 + *C2);
      Observer.changingInstr(MI);
      MI.getOperand(1).setReg(LHSSrc1);
      MI.getOperand(2).setReg(NewCst.getReg(0));
      Observer.changedInstr(MI);
    };
  }
  return !reassociationCanBreakAddressingModePattern(MI);
}

bool CombinerHelper::matchConstantFold(MachineInstr &MI, APInt &MatchInfo) {
  Register Op1 = MI.getOperand(1).getReg();
  Register Op2 = MI.getOperand(2).getReg();
  auto MaybeCst = ConstantFoldBinOp(MI.getOpcode(), Op1, Op2, MRI);
  if (!MaybeCst)
    return false;
  MatchInfo = *MaybeCst;
  return true;
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
