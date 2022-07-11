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
#include "llvm/CodeGen/GlobalISel/GISelChangeObserver.h"
#include "llvm/CodeGen/GlobalISel/GISelKnownBits.h"
#include "llvm/CodeGen/GlobalISel/GenericMachineInstrs.h"
#include "llvm/CodeGen/GlobalISel/LegalizerHelper.h"
#include "llvm/CodeGen/GlobalISel/LegalizerInfo.h"
#include "llvm/CodeGen/GlobalISel/MIPatternMatch.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/LowLevelType.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterBankInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/DivisionByConstantInfo.h"
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
    : Builder(B), MRI(Builder.getMF().getRegInfo()), Observer(Observer), KB(KB),
      MDT(MDT), LI(LI), RBI(Builder.getMF().getSubtarget().getRegBankInfo()),
      TRI(Builder.getMF().getSubtarget().getRegisterInfo()) {
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

/// Determines the LogBase2 value for a non-null input value using the
/// transform: LogBase2(V) = (EltBits - 1) - ctlz(V).
static Register buildLogBase2(Register V, MachineIRBuilder &MIB) {
  auto &MRI = *MIB.getMRI();
  LLT Ty = MRI.getType(V);
  auto Ctlz = MIB.buildCTLZ(Ty, V);
  auto Base = MIB.buildConstant(Ty, Ty.getScalarSizeInBits() - 1);
  return MIB.buildSub(Ty, Base, Ctlz).getReg(0);
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

bool CombinerHelper::isPreLegalize() const { return !LI; }

bool CombinerHelper::isLegal(const LegalityQuery &Query) const {
  assert(LI && "Must have LegalizerInfo to query isLegal!");
  return LI->getAction(Query).Action == LegalizeActions::Legal;
}

bool CombinerHelper::isLegalOrBeforeLegalizer(
    const LegalityQuery &Query) const {
  return isPreLegalize() || isLegal(Query);
}

bool CombinerHelper::isConstantLegalOrBeforeLegalizer(const LLT Ty) const {
  if (!Ty.isVector())
    return isLegalOrBeforeLegalizer({TargetOpcode::G_CONSTANT, {Ty}});
  // Vector constants are represented as a G_BUILD_VECTOR of scalar G_CONSTANTs.
  if (isPreLegalize())
    return true;
  LLT EltTy = Ty.getElementType();
  return isLegal({TargetOpcode::G_BUILD_VECTOR, {Ty, EltTy}}) &&
         isLegal({TargetOpcode::G_CONSTANT, {EltTy}});
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

void CombinerHelper::replaceOpcodeWith(MachineInstr &FromMI,
                                       unsigned ToOpcode) const {
  Observer.changingInstr(FromMI);

  FromMI.setDesc(Builder.getTII().get(ToOpcode));

  Observer.changedInstr(FromMI);
}

const RegisterBank *CombinerHelper::getRegBank(Register Reg) const {
  return RBI->getRegBank(Reg, MRI, *TRI);
}

void CombinerHelper::setRegBank(Register Reg, const RegisterBank *RegBank) {
  if (RegBank)
    MRI.setRegBank(Reg, *RegBank);
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
        LegalityQuery::MemDesc MMDesc(MMO);
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

bool CombinerHelper::matchCombineLoadWithAndMask(MachineInstr &MI,
                                                 BuildFnTy &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_AND);

  // If we have the following code:
  //  %mask = G_CONSTANT 255
  //  %ld   = G_LOAD %ptr, (load s16)
  //  %and  = G_AND %ld, %mask
  //
  // Try to fold it into
  //   %ld = G_ZEXTLOAD %ptr, (load s8)

  Register Dst = MI.getOperand(0).getReg();
  if (MRI.getType(Dst).isVector())
    return false;

  auto MaybeMask =
      getIConstantVRegValWithLookThrough(MI.getOperand(2).getReg(), MRI);
  if (!MaybeMask)
    return false;

  APInt MaskVal = MaybeMask->Value;

  if (!MaskVal.isMask())
    return false;

  Register SrcReg = MI.getOperand(1).getReg();
  GAnyLoad *LoadMI = getOpcodeDef<GAnyLoad>(SrcReg, MRI);
  if (!LoadMI || !MRI.hasOneNonDBGUse(LoadMI->getDstReg()))
    return false;

  Register LoadReg = LoadMI->getDstReg();
  LLT RegTy = MRI.getType(LoadReg);
  Register PtrReg = LoadMI->getPointerReg();
  unsigned RegSize = RegTy.getSizeInBits();
  uint64_t LoadSizeBits = LoadMI->getMemSizeInBits();
  unsigned MaskSizeBits = MaskVal.countTrailingOnes();

  // The mask may not be larger than the in-memory type, as it might cover sign
  // extended bits
  if (MaskSizeBits > LoadSizeBits)
    return false;

  // If the mask covers the whole destination register, there's nothing to
  // extend
  if (MaskSizeBits >= RegSize)
    return false;

  // Most targets cannot deal with loads of size < 8 and need to re-legalize to
  // at least byte loads. Avoid creating such loads here
  if (MaskSizeBits < 8 || !isPowerOf2_32(MaskSizeBits))
    return false;

  const MachineMemOperand &MMO = LoadMI->getMMO();
  LegalityQuery::MemDesc MemDesc(MMO);

  // Don't modify the memory access size if this is atomic/volatile, but we can
  // still adjust the opcode to indicate the high bit behavior.
  if (LoadMI->isSimple())
    MemDesc.MemoryTy = LLT::scalar(MaskSizeBits);
  else if (LoadSizeBits > MaskSizeBits || LoadSizeBits == RegSize)
    return false;

  // TODO: Could check if it's legal with the reduced or original memory size.
  if (!isLegalOrBeforeLegalizer(
          {TargetOpcode::G_ZEXTLOAD, {RegTy, MRI.getType(PtrReg)}, {MemDesc}}))
    return false;

  MatchInfo = [=](MachineIRBuilder &B) {
    B.setInstrAndDebugLoc(*LoadMI);
    auto &MF = B.getMF();
    auto PtrInfo = MMO.getPointerInfo();
    auto *NewMMO = MF.getMachineMemOperand(&MMO, PtrInfo, MemDesc.MemoryTy);
    B.buildLoadInstr(TargetOpcode::G_ZEXTLOAD, Dst, PtrReg, *NewMMO);
    LoadMI->eraseFromParent();
  };
  return true;
}

bool CombinerHelper::isPredecessor(const MachineInstr &DefMI,
                                   const MachineInstr &UseMI) {
  assert(!DefMI.isDebugInstr() && !UseMI.isDebugInstr() &&
         "shouldn't consider debug uses");
  assert(DefMI.getParent() == UseMI.getParent());
  if (&DefMI == &UseMI)
    return true;
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

  Register DstReg = MI.getOperand(0).getReg();
  LLT RegTy = MRI.getType(DstReg);

  // Only supports scalars for now.
  if (RegTy.isVector())
    return false;

  Register SrcReg = MI.getOperand(1).getReg();
  auto *LoadDef = getOpcodeDef<GLoad>(SrcReg, MRI);
  if (!LoadDef || !MRI.hasOneNonDBGUse(DstReg))
    return false;

  uint64_t MemBits = LoadDef->getMemSizeInBits();

  // If the sign extend extends from a narrower width than the load's width,
  // then we can narrow the load width when we combine to a G_SEXTLOAD.
  // Avoid widening the load at all.
  unsigned NewSizeBits = std::min((uint64_t)MI.getOperand(2).getImm(), MemBits);

  // Don't generate G_SEXTLOADs with a < 1 byte width.
  if (NewSizeBits < 8)
    return false;
  // Don't bother creating a non-power-2 sextload, it will likely be broken up
  // anyway for most targets.
  if (!isPowerOf2_32(NewSizeBits))
    return false;

  const MachineMemOperand &MMO = LoadDef->getMMO();
  LegalityQuery::MemDesc MMDesc(MMO);

  // Don't modify the memory access size if this is atomic/volatile, but we can
  // still adjust the opcode to indicate the high bit behavior.
  if (LoadDef->isSimple())
    MMDesc.MemoryTy = LLT::scalar(NewSizeBits);
  else if (MemBits > NewSizeBits || MemBits == RegTy.getSizeInBits())
    return false;

  // TODO: Could check if it's legal with the reduced or original memory size.
  if (!isLegalOrBeforeLegalizer({TargetOpcode::G_SEXTLOAD,
                                 {MRI.getType(LoadDef->getDstReg()),
                                  MRI.getType(LoadDef->getPointerReg())},
                                 {MMDesc}}))
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

static Type *getTypeForLLT(LLT Ty, LLVMContext &C) {
  if (Ty.isVector())
    return FixedVectorType::get(IntegerType::get(C, Ty.getScalarSizeInBits()),
                                Ty.getNumElements());
  return IntegerType::get(C, Ty.getSizeInBits());
}

bool CombinerHelper::tryEmitMemcpyInline(MachineInstr &MI) {
  MachineIRBuilder HelperBuilder(MI);
  GISelObserverWrapper DummyObserver;
  LegalizerHelper Helper(HelperBuilder.getMF(), DummyObserver, HelperBuilder);
  return Helper.lowerMemcpyInline(MI) ==
         LegalizerHelper::LegalizeResult::Legalized;
}

bool CombinerHelper::tryCombineMemCpyFamily(MachineInstr &MI, unsigned MaxLen) {
  MachineIRBuilder HelperBuilder(MI);
  GISelObserverWrapper DummyObserver;
  LegalizerHelper Helper(HelperBuilder.getMF(), DummyObserver, HelperBuilder);
  return Helper.lowerMemCpyFamily(MI, MaxLen) ==
         LegalizerHelper::LegalizeResult::Legalized;
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
  return Cst.has_value();
}

void CombinerHelper::applyCombineConstantFoldFpUnary(MachineInstr &MI,
                                                     Optional<APFloat> &Cst) {
  assert(Cst && "Optional is unexpectedly empty!");
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
  auto MaybeImmVal = getIConstantVRegValWithLookThrough(Imm1, MRI);
  if (!MaybeImmVal)
    return false;

  MachineInstr *Add2Def = MRI.getVRegDef(Add2);
  if (!Add2Def || Add2Def->getOpcode() != TargetOpcode::G_PTR_ADD)
    return false;

  Register Base = Add2Def->getOperand(1).getReg();
  Register Imm2 = Add2Def->getOperand(2).getReg();
  auto MaybeImm2Val = getIConstantVRegValWithLookThrough(Imm2, MRI);
  if (!MaybeImm2Val)
    return false;

  // Check if the new combined immediate forms an illegal addressing mode.
  // Do not combine if it was legal before but would get illegal.
  // To do so, we need to find a load/store user of the pointer to get
  // the access type.
  Type *AccessTy = nullptr;
  auto &MF = *MI.getMF();
  for (auto &UseMI : MRI.use_nodbg_instructions(MI.getOperand(0).getReg())) {
    if (auto *LdSt = dyn_cast<GLoadStore>(&UseMI)) {
      AccessTy = getTypeForLLT(MRI.getType(LdSt->getReg(0)),
                               MF.getFunction().getContext());
      break;
    }
  }
  TargetLoweringBase::AddrMode AMNew;
  APInt CombinedImm = MaybeImmVal->Value + MaybeImm2Val->Value;
  AMNew.BaseOffs = CombinedImm.getSExtValue();
  if (AccessTy) {
    AMNew.HasBaseReg = true;
    TargetLoweringBase::AddrMode AMOld;
    AMOld.BaseOffs = MaybeImm2Val->Value.getSExtValue();
    AMOld.HasBaseReg = true;
    unsigned AS = MRI.getType(Add2).getAddressSpace();
    const auto &TLI = *MF.getSubtarget().getTargetLowering();
    if (TLI.isLegalAddressingMode(MF.getDataLayout(), AMOld, AccessTy, AS) &&
        !TLI.isLegalAddressingMode(MF.getDataLayout(), AMNew, AccessTy, AS))
      return false;
  }

  // Pass the combined immediate to the apply function.
  MatchInfo.Imm = AMNew.BaseOffs;
  MatchInfo.Base = Base;
  MatchInfo.Bank = getRegBank(Imm2);
  return true;
}

void CombinerHelper::applyPtrAddImmedChain(MachineInstr &MI,
                                           PtrAddChain &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_PTR_ADD && "Expected G_PTR_ADD");
  MachineIRBuilder MIB(MI);
  LLT OffsetTy = MRI.getType(MI.getOperand(2).getReg());
  auto NewOffset = MIB.buildConstant(OffsetTy, MatchInfo.Imm);
  setRegBank(NewOffset.getReg(0), MatchInfo.Bank);
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
  auto MaybeImmVal = getIConstantVRegValWithLookThrough(Imm1, MRI);
  if (!MaybeImmVal)
    return false;

  MachineInstr *Shl2Def = MRI.getUniqueVRegDef(Shl2);
  if (Shl2Def->getOpcode() != Opcode)
    return false;

  Register Base = Shl2Def->getOperand(1).getReg();
  Register Imm2 = Shl2Def->getOperand(2).getReg();
  auto MaybeImm2Val = getIConstantVRegValWithLookThrough(Imm2, MRI);
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
  auto MaybeImmVal = getIConstantVRegValWithLookThrough(C1, MRI);
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
        getIConstantVRegValWithLookThrough(MI->getOperand(2).getReg(), MRI);
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
      getIConstantVRegValWithLookThrough(MI.getOperand(2).getReg(), MRI);
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
  auto MaybeShiftAmtVal = getIConstantVRegValWithLookThrough(RHS, MRI);
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
  auto &Unmerge = cast<GUnmerge>(MI);
  Register SrcReg = peekThroughBitcast(Unmerge.getSourceReg(), MRI);

  auto *SrcInstr = getOpcodeDef<GMergeLikeOp>(SrcReg, MRI);
  if (!SrcInstr)
    return false;

  // Check the source type of the merge.
  LLT SrcMergeTy = MRI.getType(SrcInstr->getSourceReg(0));
  LLT Dst0Ty = MRI.getType(Unmerge.getReg(0));
  bool SameSize = Dst0Ty.getSizeInBits() == SrcMergeTy.getSizeInBits();
  if (SrcMergeTy != Dst0Ty && !SameSize)
    return false;
  // They are the same now (modulo a bitcast).
  // We can collect all the src registers.
  for (unsigned Idx = 0; Idx < SrcInstr->getNumSources(); ++Idx)
    Operands.push_back(SrcInstr->getSourceReg(Idx));
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

bool CombinerHelper::matchCombineUnmergeUndef(
    MachineInstr &MI, std::function<void(MachineIRBuilder &)> &MatchInfo) {
  unsigned SrcIdx = MI.getNumOperands() - 1;
  Register SrcReg = MI.getOperand(SrcIdx).getReg();
  MatchInfo = [&MI](MachineIRBuilder &B) {
    unsigned NumElems = MI.getNumOperands() - 1;
    for (unsigned Idx = 0; Idx < NumElems; ++Idx) {
      Register DstReg = MI.getOperand(Idx).getReg();
      B.buildUndef(DstReg);
    }
  };
  return isa<GImplicitDef>(MRI.getVRegDef(SrcReg));
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
      getIConstantVRegValWithLookThrough(MI.getOperand(2).getReg(), MRI);
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
                                                  APInt &NewCst) {
  auto &PtrAdd = cast<GPtrAdd>(MI);
  Register LHS = PtrAdd.getBaseReg();
  Register RHS = PtrAdd.getOffsetReg();
  MachineRegisterInfo &MRI = Builder.getMF().getRegInfo();

  if (auto RHSCst = getIConstantVRegVal(RHS, MRI)) {
    APInt Cst;
    if (mi_match(LHS, MRI, m_GIntToPtr(m_ICst(Cst)))) {
      auto DstTy = MRI.getType(PtrAdd.getReg(0));
      // G_INTTOPTR uses zero-extension
      NewCst = Cst.zextOrTrunc(DstTy.getSizeInBits());
      NewCst += RHSCst->sextOrTrunc(DstTy.getSizeInBits());
      return true;
    }
  }

  return false;
}

void CombinerHelper::applyCombineConstPtrAddToI2P(MachineInstr &MI,
                                                  APInt &NewCst) {
  auto &PtrAdd = cast<GPtrAdd>(MI);
  Register Dst = PtrAdd.getReg(0);

  Builder.setInstrAndDebugLoc(MI);
  Builder.buildConstant(Dst, NewCst);
  PtrAdd.eraseFromParent();
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

bool CombinerHelper::matchCombineFAbsOfFNeg(MachineInstr &MI,
                                            BuildFnTy &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_FABS && "Expected a G_FABS");
  Register Src = MI.getOperand(1).getReg();
  Register NegSrc;

  if (!mi_match(Src, MRI, m_GFNeg(m_Reg(NegSrc))))
    return false;

  MatchInfo = [=, &MI](MachineIRBuilder &B) {
    Observer.changingInstr(MI);
    MI.getOperand(1).setReg(NegSrc);
    Observer.changedInstr(MI);
  };
  return true;
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
    if (Known.countMaxActiveBits() <= Log2_32(Size)) {
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
  GSelect &SelMI = cast<GSelect>(MI);
  auto Cst =
      isConstantOrConstantSplatVector(*MRI.getVRegDef(SelMI.getCondReg()), MRI);
  if (!Cst)
    return false;
  OpIdx = Cst->isZero() ? 3 : 2;
  return true;
}

bool CombinerHelper::eraseInst(MachineInstr &MI) {
  MI.eraseFromParent();
  return true;
}

bool CombinerHelper::matchEqualDefs(const MachineOperand &MOP1,
                                    const MachineOperand &MOP2) {
  if (!MOP1.isReg() || !MOP2.isReg())
    return false;
  auto InstAndDef1 = getDefSrcRegIgnoringCopies(MOP1.getReg(), MRI);
  if (!InstAndDef1)
    return false;
  auto InstAndDef2 = getDefSrcRegIgnoringCopies(MOP2.getReg(), MRI);
  if (!InstAndDef2)
    return false;
  MachineInstr *I1 = InstAndDef1->MI;
  MachineInstr *I2 = InstAndDef2->MI;

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

  // If both instructions are loads or stores, they are equal only if both
  // are dereferenceable invariant loads with the same number of bits.
  if (I1->mayLoadOrStore() && I2->mayLoadOrStore()) {
    GLoadStore *LS1 = dyn_cast<GLoadStore>(I1);
    GLoadStore *LS2 = dyn_cast<GLoadStore>(I2);
    if (!LS1 || !LS2)
      return false;

    if (!I2->isDereferenceableInvariantLoad(nullptr) ||
        (LS1->getMemSizeInBits() != LS2->getMemSizeInBits()))
      return false;
  }

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
  if (Builder.getTII().produceSameValue(*I1, *I2, &MRI)) {
    // Handle instructions with multiple defs that produce same values. Values
    // are same for operands with same index.
    // %0:_(s8), %1:_(s8), %2:_(s8), %3:_(s8) = G_UNMERGE_VALUES %4:_(<4 x s8>)
    // %5:_(s8), %6:_(s8), %7:_(s8), %8:_(s8) = G_UNMERGE_VALUES %4:_(<4 x s8>)
    // I1 and I2 are different instructions but produce same values,
    // %1 and %6 are same, %1 and %7 are not the same value.
    return I1->findRegisterDefOperandIdx(InstAndDef1->Reg) ==
           I2->findRegisterDefOperandIdx(InstAndDef2->Reg);
  }
  return false;
}

bool CombinerHelper::matchConstantOp(const MachineOperand &MOP, int64_t C) {
  if (!MOP.isReg())
    return false;
  auto *MI = MRI.getVRegDef(MOP.getReg());
  auto MaybeCst = isConstantOrConstantSplatVector(*MI, MRI);
  return MaybeCst && MaybeCst->getBitWidth() <= 64 &&
         MaybeCst->getSExtValue() == C;
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
      (LHSBits.Zero | RHSBits.One).isAllOnes()) {
    Replacement = LHS;
    return true;
  }

  // Check if we can replace AndDst with the RHS of the G_AND
  if (canReplaceReg(AndDst, RHS, MRI) &&
      (LHSBits.One | RHSBits.Zero).isAllOnes()) {
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
      (LHSBits.One | RHSBits.Zero).isAllOnes()) {
    Replacement = LHS;
    return true;
  }

  // Check if we can replace OrDst with the RHS of the G_OR
  if (canReplaceReg(OrDst, RHS, MRI) &&
      (LHSBits.Zero | RHSBits.One).isAllOnes()) {
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
    auto MaybeCst = getIConstantSplatSExtVal(*CstDef, MRI);
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
  auto &PtrAdd = cast<GPtrAdd>(MI);
  Register DstReg = PtrAdd.getReg(0);
  LLT Ty = MRI.getType(DstReg);
  const DataLayout &DL = Builder.getMF().getDataLayout();

  if (DL.isNonIntegralAddressSpace(Ty.getScalarType().getAddressSpace()))
    return false;

  if (Ty.isPointer()) {
    auto ConstVal = getIConstantVRegVal(PtrAdd.getBaseReg(), MRI);
    return ConstVal && *ConstVal == 0;
  }

  assert(Ty.isVector() && "Expecting a vector type");
  const MachineInstr *VecMI = MRI.getVRegDef(PtrAdd.getBaseReg());
  return isBuildVectorAllZeros(*VecMI, MRI);
}

void CombinerHelper::applyPtrAddZero(MachineInstr &MI) {
  auto &PtrAdd = cast<GPtrAdd>(MI);
  Builder.setInstrAndDebugLoc(PtrAdd);
  Builder.buildIntToPtr(PtrAdd.getReg(0), PtrAdd.getOffsetReg());
  PtrAdd.eraseFromParent();
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

bool CombinerHelper::matchFoldBinOpIntoSelect(MachineInstr &MI,
                                              unsigned &SelectOpNo) {
  Register LHS = MI.getOperand(1).getReg();
  Register RHS = MI.getOperand(2).getReg();

  Register OtherOperandReg = RHS;
  SelectOpNo = 1;
  MachineInstr *Select = MRI.getVRegDef(LHS);

  // Don't do this unless the old select is going away. We want to eliminate the
  // binary operator, not replace a binop with a select.
  if (Select->getOpcode() != TargetOpcode::G_SELECT ||
      !MRI.hasOneNonDBGUse(LHS)) {
    OtherOperandReg = LHS;
    SelectOpNo = 2;
    Select = MRI.getVRegDef(RHS);
    if (Select->getOpcode() != TargetOpcode::G_SELECT ||
        !MRI.hasOneNonDBGUse(RHS))
      return false;
  }

  MachineInstr *SelectLHS = MRI.getVRegDef(Select->getOperand(2).getReg());
  MachineInstr *SelectRHS = MRI.getVRegDef(Select->getOperand(3).getReg());

  if (!isConstantOrConstantVector(*SelectLHS, MRI,
                                  /*AllowFP*/ true,
                                  /*AllowOpaqueConstants*/ false))
    return false;
  if (!isConstantOrConstantVector(*SelectRHS, MRI,
                                  /*AllowFP*/ true,
                                  /*AllowOpaqueConstants*/ false))
    return false;

  unsigned BinOpcode = MI.getOpcode();

  // We know know one of the operands is a select of constants. Now verify that
  // the other binary operator operand is either a constant, or we can handle a
  // variable.
  bool CanFoldNonConst =
      (BinOpcode == TargetOpcode::G_AND || BinOpcode == TargetOpcode::G_OR) &&
      (isNullOrNullSplat(*SelectLHS, MRI) ||
       isAllOnesOrAllOnesSplat(*SelectLHS, MRI)) &&
      (isNullOrNullSplat(*SelectRHS, MRI) ||
       isAllOnesOrAllOnesSplat(*SelectRHS, MRI));
  if (CanFoldNonConst)
    return true;

  return isConstantOrConstantVector(*MRI.getVRegDef(OtherOperandReg), MRI,
                                    /*AllowFP*/ true,
                                    /*AllowOpaqueConstants*/ false);
}

/// \p SelectOperand is the operand in binary operator \p MI that is the select
/// to fold.
bool CombinerHelper::applyFoldBinOpIntoSelect(MachineInstr &MI,
                                              const unsigned &SelectOperand) {
  Builder.setInstrAndDebugLoc(MI);

  Register Dst = MI.getOperand(0).getReg();
  Register LHS = MI.getOperand(1).getReg();
  Register RHS = MI.getOperand(2).getReg();
  MachineInstr *Select = MRI.getVRegDef(MI.getOperand(SelectOperand).getReg());

  Register SelectCond = Select->getOperand(1).getReg();
  Register SelectTrue = Select->getOperand(2).getReg();
  Register SelectFalse = Select->getOperand(3).getReg();

  LLT Ty = MRI.getType(Dst);
  unsigned BinOpcode = MI.getOpcode();

  Register FoldTrue, FoldFalse;

  // We have a select-of-constants followed by a binary operator with a
  // constant. Eliminate the binop by pulling the constant math into the select.
  // Example: add (select Cond, CT, CF), CBO --> select Cond, CT + CBO, CF + CBO
  if (SelectOperand == 1) {
    // TODO: SelectionDAG verifies this actually constant folds before
    // committing to the combine.

    FoldTrue = Builder.buildInstr(BinOpcode, {Ty}, {SelectTrue, RHS}).getReg(0);
    FoldFalse =
        Builder.buildInstr(BinOpcode, {Ty}, {SelectFalse, RHS}).getReg(0);
  } else {
    FoldTrue = Builder.buildInstr(BinOpcode, {Ty}, {LHS, SelectTrue}).getReg(0);
    FoldFalse =
        Builder.buildInstr(BinOpcode, {Ty}, {LHS, SelectFalse}).getReg(0);
  }

  Builder.buildSelect(Dst, SelectCond, FoldTrue, FoldFalse, MI.getFlags());
  Observer.erasingInstr(*Select);
  Select->eraseFromParent();
  MI.eraseFromParent();

  return true;
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
  if (!IsBigEndian)
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
  LegalityQuery::MemDesc MMDesc(MMO);
  MMDesc.MemoryTy = Ty;
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

/// Check if the store \p Store is a truncstore that can be merged. That is,
/// it's a store of a shifted value of \p SrcVal. If \p SrcVal is an empty
/// Register then it does not need to match and SrcVal is set to the source
/// value found.
/// On match, returns the start byte offset of the \p SrcVal that is being
/// stored.
static Optional<int64_t> getTruncStoreByteOffset(GStore &Store, Register &SrcVal,
                                                 MachineRegisterInfo &MRI) {
  Register TruncVal;
  if (!mi_match(Store.getValueReg(), MRI, m_GTrunc(m_Reg(TruncVal))))
    return None;

  // The shift amount must be a constant multiple of the narrow type.
  // It is translated to the offset address in the wide source value "y".
  //
  // x = G_LSHR y, ShiftAmtC
  // s8 z = G_TRUNC x
  // store z, ...
  Register FoundSrcVal;
  int64_t ShiftAmt;
  if (!mi_match(TruncVal, MRI,
                m_any_of(m_GLShr(m_Reg(FoundSrcVal), m_ICst(ShiftAmt)),
                         m_GAShr(m_Reg(FoundSrcVal), m_ICst(ShiftAmt))))) {
    if (!SrcVal.isValid() || TruncVal == SrcVal) {
      if (!SrcVal.isValid())
        SrcVal = TruncVal;
      return 0; // If it's the lowest index store.
    }
    return None;
  }

  unsigned NarrowBits = Store.getMMO().getMemoryType().getScalarSizeInBits();
  if (ShiftAmt % NarrowBits!= 0)
    return None;
  const unsigned Offset = ShiftAmt / NarrowBits;

  if (SrcVal.isValid() && FoundSrcVal != SrcVal)
    return None;

  if (!SrcVal.isValid())
    SrcVal = FoundSrcVal;
  else if (MRI.getType(SrcVal) != MRI.getType(FoundSrcVal))
    return None;
  return Offset;
}

/// Match a pattern where a wide type scalar value is stored by several narrow
/// stores. Fold it into a single store or a BSWAP and a store if the targets
/// supports it.
///
/// Assuming little endian target:
///  i8 *p = ...
///  i32 val = ...
///  p[0] = (val >> 0) & 0xFF;
///  p[1] = (val >> 8) & 0xFF;
///  p[2] = (val >> 16) & 0xFF;
///  p[3] = (val >> 24) & 0xFF;
/// =>
///  *((i32)p) = val;
///
///  i8 *p = ...
///  i32 val = ...
///  p[0] = (val >> 24) & 0xFF;
///  p[1] = (val >> 16) & 0xFF;
///  p[2] = (val >> 8) & 0xFF;
///  p[3] = (val >> 0) & 0xFF;
/// =>
///  *((i32)p) = BSWAP(val);
bool CombinerHelper::matchTruncStoreMerge(MachineInstr &MI,
                                          MergeTruncStoresInfo &MatchInfo) {
  auto &StoreMI = cast<GStore>(MI);
  LLT MemTy = StoreMI.getMMO().getMemoryType();

  // We only handle merging simple stores of 1-4 bytes.
  if (!MemTy.isScalar())
    return false;
  switch (MemTy.getSizeInBits()) {
  case 8:
  case 16:
  case 32:
    break;
  default:
    return false;
  }
  if (!StoreMI.isSimple())
    return false;

  // We do a simple search for mergeable stores prior to this one.
  // Any potential alias hazard along the way terminates the search.
  SmallVector<GStore *> FoundStores;

  // We're looking for:
  // 1) a (store(trunc(...)))
  // 2) of an LSHR/ASHR of a single wide value, by the appropriate shift to get
  //    the partial value stored.
  // 3) where the offsets form either a little or big-endian sequence.

  auto &LastStore = StoreMI;

  // The single base pointer that all stores must use.
  Register BaseReg;
  int64_t LastOffset;
  if (!mi_match(LastStore.getPointerReg(), MRI,
                m_GPtrAdd(m_Reg(BaseReg), m_ICst(LastOffset)))) {
    BaseReg = LastStore.getPointerReg();
    LastOffset = 0;
  }

  GStore *LowestIdxStore = &LastStore;
  int64_t LowestIdxOffset = LastOffset;

  Register WideSrcVal;
  auto LowestShiftAmt = getTruncStoreByteOffset(LastStore, WideSrcVal, MRI);
  if (!LowestShiftAmt)
    return false; // Didn't match a trunc.
  assert(WideSrcVal.isValid());

  LLT WideStoreTy = MRI.getType(WideSrcVal);
  // The wide type might not be a multiple of the memory type, e.g. s48 and s32.
  if (WideStoreTy.getSizeInBits() % MemTy.getSizeInBits() != 0)
    return false;
  const unsigned NumStoresRequired =
      WideStoreTy.getSizeInBits() / MemTy.getSizeInBits();

  SmallVector<int64_t, 8> OffsetMap(NumStoresRequired, INT64_MAX);
  OffsetMap[*LowestShiftAmt] = LastOffset;
  FoundStores.emplace_back(&LastStore);

  // Search the block up for more stores.
  // We use a search threshold of 10 instructions here because the combiner
  // works top-down within a block, and we don't want to search an unbounded
  // number of predecessor instructions trying to find matching stores.
  // If we moved this optimization into a separate pass then we could probably
  // use a more efficient search without having a hard-coded threshold.
  const int MaxInstsToCheck = 10;
  int NumInstsChecked = 0;
  for (auto II = ++LastStore.getReverseIterator();
       II != LastStore.getParent()->rend() && NumInstsChecked < MaxInstsToCheck;
       ++II) {
    NumInstsChecked++;
    GStore *NewStore;
    if ((NewStore = dyn_cast<GStore>(&*II))) {
      if (NewStore->getMMO().getMemoryType() != MemTy || !NewStore->isSimple())
        break;
    } else if (II->isLoadFoldBarrier() || II->mayLoad()) {
      break;
    } else {
      continue; // This is a safe instruction we can look past.
    }

    Register NewBaseReg;
    int64_t MemOffset;
    // Check we're storing to the same base + some offset.
    if (!mi_match(NewStore->getPointerReg(), MRI,
                  m_GPtrAdd(m_Reg(NewBaseReg), m_ICst(MemOffset)))) {
      NewBaseReg = NewStore->getPointerReg();
      MemOffset = 0;
    }
    if (BaseReg != NewBaseReg)
      break;

    auto ShiftByteOffset = getTruncStoreByteOffset(*NewStore, WideSrcVal, MRI);
    if (!ShiftByteOffset)
      break;
    if (MemOffset < LowestIdxOffset) {
      LowestIdxOffset = MemOffset;
      LowestIdxStore = NewStore;
    }

    // Map the offset in the store and the offset in the combined value, and
    // early return if it has been set before.
    if (*ShiftByteOffset < 0 || *ShiftByteOffset >= NumStoresRequired ||
        OffsetMap[*ShiftByteOffset] != INT64_MAX)
      break;
    OffsetMap[*ShiftByteOffset] = MemOffset;

    FoundStores.emplace_back(NewStore);
    // Reset counter since we've found a matching inst.
    NumInstsChecked = 0;
    if (FoundStores.size() == NumStoresRequired)
      break;
  }

  if (FoundStores.size() != NumStoresRequired) {
    return false;
  }

  const auto &DL = LastStore.getMF()->getDataLayout();
  auto &C = LastStore.getMF()->getFunction().getContext();
  // Check that a store of the wide type is both allowed and fast on the target
  bool Fast = false;
  bool Allowed = getTargetLowering().allowsMemoryAccess(
      C, DL, WideStoreTy, LowestIdxStore->getMMO(), &Fast);
  if (!Allowed || !Fast)
    return false;

  // Check if the pieces of the value are going to the expected places in memory
  // to merge the stores.
  unsigned NarrowBits = MemTy.getScalarSizeInBits();
  auto checkOffsets = [&](bool MatchLittleEndian) {
    if (MatchLittleEndian) {
      for (unsigned i = 0; i != NumStoresRequired; ++i)
        if (OffsetMap[i] != i * (NarrowBits / 8) + LowestIdxOffset)
          return false;
    } else { // MatchBigEndian by reversing loop counter.
      for (unsigned i = 0, j = NumStoresRequired - 1; i != NumStoresRequired;
           ++i, --j)
        if (OffsetMap[j] != i * (NarrowBits / 8) + LowestIdxOffset)
          return false;
    }
    return true;
  };

  // Check if the offsets line up for the native data layout of this target.
  bool NeedBswap = false;
  bool NeedRotate = false;
  if (!checkOffsets(DL.isLittleEndian())) {
    // Special-case: check if byte offsets line up for the opposite endian.
    if (NarrowBits == 8 && checkOffsets(DL.isBigEndian()))
      NeedBswap = true;
    else if (NumStoresRequired == 2 && checkOffsets(DL.isBigEndian()))
      NeedRotate = true;
    else
      return false;
  }

  if (NeedBswap &&
      !isLegalOrBeforeLegalizer({TargetOpcode::G_BSWAP, {WideStoreTy}}))
    return false;
  if (NeedRotate &&
      !isLegalOrBeforeLegalizer({TargetOpcode::G_ROTR, {WideStoreTy}}))
    return false;

  MatchInfo.NeedBSwap = NeedBswap;
  MatchInfo.NeedRotate = NeedRotate;
  MatchInfo.LowestIdxStore = LowestIdxStore;
  MatchInfo.WideSrcVal = WideSrcVal;
  MatchInfo.FoundStores = std::move(FoundStores);
  return true;
}

void CombinerHelper::applyTruncStoreMerge(MachineInstr &MI,
                                          MergeTruncStoresInfo &MatchInfo) {

  Builder.setInstrAndDebugLoc(MI);
  Register WideSrcVal = MatchInfo.WideSrcVal;
  LLT WideStoreTy = MRI.getType(WideSrcVal);

  if (MatchInfo.NeedBSwap) {
    WideSrcVal = Builder.buildBSwap(WideStoreTy, WideSrcVal).getReg(0);
  } else if (MatchInfo.NeedRotate) {
    assert(WideStoreTy.getSizeInBits() % 2 == 0 &&
           "Unexpected type for rotate");
    auto RotAmt =
        Builder.buildConstant(WideStoreTy, WideStoreTy.getSizeInBits() / 2);
    WideSrcVal =
        Builder.buildRotateRight(WideStoreTy, WideSrcVal, RotAmt).getReg(0);
  }

  Builder.buildStore(WideSrcVal, MatchInfo.LowestIdxStore->getPointerReg(),
                     MatchInfo.LowestIdxStore->getMMO().getPointerInfo(),
                     MatchInfo.LowestIdxStore->getMMO().getAlign());

  // Erase the old stores.
  for (auto *ST : MatchInfo.FoundStores)
    ST->eraseFromParent();
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
  for (const MachineOperand &MO : llvm::drop_begin(MI.operands())) {
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

  auto Cst = getIConstantVRegValWithLookThrough(MI.getOperand(2).getReg(), MRI);
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
  for (MachineInstr &II : MRI.use_nodbg_instructions(DstReg)) {
    if (II.getOpcode() != TargetOpcode::G_EXTRACT_VECTOR_ELT)
      return false;
    auto Cst = getIConstantVRegVal(II.getOperand(2).getReg(), MRI);
    if (!Cst)
      return false;
    unsigned Idx = Cst->getZExtValue();
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

bool CombinerHelper::matchOrShiftToFunnelShift(MachineInstr &MI,
                                               BuildFnTy &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_OR);

  Register Dst = MI.getOperand(0).getReg();
  LLT Ty = MRI.getType(Dst);
  unsigned BitWidth = Ty.getScalarSizeInBits();

  Register ShlSrc, ShlAmt, LShrSrc, LShrAmt, Amt;
  unsigned FshOpc = 0;

  // Match (or (shl ...), (lshr ...)).
  if (!mi_match(Dst, MRI,
                // m_GOr() handles the commuted version as well.
                m_GOr(m_GShl(m_Reg(ShlSrc), m_Reg(ShlAmt)),
                      m_GLShr(m_Reg(LShrSrc), m_Reg(LShrAmt)))))
    return false;

  // Given constants C0 and C1 such that C0 + C1 is bit-width:
  // (or (shl x, C0), (lshr y, C1)) -> (fshl x, y, C0) or (fshr x, y, C1)
  int64_t CstShlAmt, CstLShrAmt;
  if (mi_match(ShlAmt, MRI, m_ICstOrSplat(CstShlAmt)) &&
      mi_match(LShrAmt, MRI, m_ICstOrSplat(CstLShrAmt)) &&
      CstShlAmt + CstLShrAmt == BitWidth) {
    FshOpc = TargetOpcode::G_FSHR;
    Amt = LShrAmt;

  } else if (mi_match(LShrAmt, MRI,
                      m_GSub(m_SpecificICstOrSplat(BitWidth), m_Reg(Amt))) &&
             ShlAmt == Amt) {
    // (or (shl x, amt), (lshr y, (sub bw, amt))) -> (fshl x, y, amt)
    FshOpc = TargetOpcode::G_FSHL;

  } else if (mi_match(ShlAmt, MRI,
                      m_GSub(m_SpecificICstOrSplat(BitWidth), m_Reg(Amt))) &&
             LShrAmt == Amt) {
    // (or (shl x, (sub bw, amt)), (lshr y, amt)) -> (fshr x, y, amt)
    FshOpc = TargetOpcode::G_FSHR;

  } else {
    return false;
  }

  LLT AmtTy = MRI.getType(Amt);
  if (!isLegalOrBeforeLegalizer({FshOpc, {Ty, AmtTy}}))
    return false;

  MatchInfo = [=](MachineIRBuilder &B) {
    B.buildInstr(FshOpc, {Dst}, {ShlSrc, LShrSrc, Amt});
  };
  return true;
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
  MI.removeOperand(2);
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

bool CombinerHelper::matchICmpToLHSKnownBits(
    MachineInstr &MI, std::function<void(MachineIRBuilder &)> &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_ICMP);
  // Given:
  //
  // %x = G_WHATEVER (... x is known to be 0 or 1 ...)
  // %cmp = G_ICMP ne %x, 0
  //
  // Or:
  //
  // %x = G_WHATEVER (... x is known to be 0 or 1 ...)
  // %cmp = G_ICMP eq %x, 1
  //
  // We can replace %cmp with %x assuming true is 1 on the target.
  auto Pred = static_cast<CmpInst::Predicate>(MI.getOperand(1).getPredicate());
  if (!CmpInst::isEquality(Pred))
    return false;
  Register Dst = MI.getOperand(0).getReg();
  LLT DstTy = MRI.getType(Dst);
  if (getICmpTrueVal(getTargetLowering(), DstTy.isVector(),
                     /* IsFP = */ false) != 1)
    return false;
  int64_t OneOrZero = Pred == CmpInst::ICMP_EQ;
  if (!mi_match(MI.getOperand(3).getReg(), MRI, m_SpecificICst(OneOrZero)))
    return false;
  Register LHS = MI.getOperand(2).getReg();
  auto KnownLHS = KB->getKnownBits(LHS);
  if (KnownLHS.getMinValue() != 0 || KnownLHS.getMaxValue() != 1)
    return false;
  // Make sure replacing Dst with the LHS is a legal operation.
  LLT LHSTy = MRI.getType(LHS);
  unsigned LHSSize = LHSTy.getSizeInBits();
  unsigned DstSize = DstTy.getSizeInBits();
  unsigned Op = TargetOpcode::COPY;
  if (DstSize != LHSSize)
    Op = DstSize < LHSSize ? TargetOpcode::G_TRUNC : TargetOpcode::G_ZEXT;
  if (!isLegalOrBeforeLegalizer({Op, {DstTy, LHSTy}}))
    return false;
  MatchInfo = [=](MachineIRBuilder &B) { B.buildInstr(Op, {Dst}, {LHS}); };
  return true;
}

// Replace (and (or x, c1), c2) with (and x, c2) iff c1 & c2 == 0
bool CombinerHelper::matchAndOrDisjointMask(
    MachineInstr &MI, std::function<void(MachineIRBuilder &)> &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_AND);

  // Ignore vector types to simplify matching the two constants.
  // TODO: do this for vectors and scalars via a demanded bits analysis.
  LLT Ty = MRI.getType(MI.getOperand(0).getReg());
  if (Ty.isVector())
    return false;

  Register Src;
  Register AndMaskReg;
  int64_t AndMaskBits;
  int64_t OrMaskBits;
  if (!mi_match(MI, MRI,
                m_GAnd(m_GOr(m_Reg(Src), m_ICst(OrMaskBits)),
                       m_all_of(m_ICst(AndMaskBits), m_Reg(AndMaskReg)))))
    return false;

  // Check if OrMask could turn on any bits in Src.
  if (AndMaskBits & OrMaskBits)
    return false;

  MatchInfo = [=, &MI](MachineIRBuilder &B) {
    Observer.changingInstr(MI);
    // Canonicalize the result to have the constant on the RHS.
    if (MI.getOperand(1).getReg() == AndMaskReg)
      MI.getOperand(2).setReg(AndMaskReg);
    MI.getOperand(1).setReg(Src);
    Observer.changedInstr(MI);
  };
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
  LLT ExtractTy = getTargetLowering().getPreferredShiftAmountTy(Ty);
  if (!getTargetLowering().isConstantUnsignedBitfieldExtractLegal(
          TargetOpcode::G_UBFX, Ty, ExtractTy))
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

  uint64_t Width = APInt(Size, AndImm).countTrailingOnes();
  MatchInfo = [=](MachineIRBuilder &B) {
    auto WidthCst = B.buildConstant(ExtractTy, Width);
    auto LSBCst = B.buildConstant(ExtractTy, LSBImm);
    B.buildInstr(TargetOpcode::G_UBFX, {Dst}, {ShiftSrc, LSBCst, WidthCst});
  };
  return true;
}

bool CombinerHelper::matchBitfieldExtractFromShr(
    MachineInstr &MI, std::function<void(MachineIRBuilder &)> &MatchInfo) {
  const unsigned Opcode = MI.getOpcode();
  assert(Opcode == TargetOpcode::G_ASHR || Opcode == TargetOpcode::G_LSHR);

  const Register Dst = MI.getOperand(0).getReg();

  const unsigned ExtrOpcode = Opcode == TargetOpcode::G_ASHR
                                  ? TargetOpcode::G_SBFX
                                  : TargetOpcode::G_UBFX;

  // Check if the type we would use for the extract is legal
  LLT Ty = MRI.getType(Dst);
  LLT ExtractTy = getTargetLowering().getPreferredShiftAmountTy(Ty);
  if (!LI || !LI->isLegalOrCustom({ExtrOpcode, {Ty, ExtractTy}}))
    return false;

  Register ShlSrc;
  int64_t ShrAmt;
  int64_t ShlAmt;
  const unsigned Size = Ty.getScalarSizeInBits();

  // Try to match shr (shl x, c1), c2
  if (!mi_match(Dst, MRI,
                m_BinOp(Opcode,
                        m_OneNonDBGUse(m_GShl(m_Reg(ShlSrc), m_ICst(ShlAmt))),
                        m_ICst(ShrAmt))))
    return false;

  // Make sure that the shift sizes can fit a bitfield extract
  if (ShlAmt < 0 || ShlAmt > ShrAmt || ShrAmt >= Size)
    return false;

  // Skip this combine if the G_SEXT_INREG combine could handle it
  if (Opcode == TargetOpcode::G_ASHR && ShlAmt == ShrAmt)
    return false;

  // Calculate start position and width of the extract
  const int64_t Pos = ShrAmt - ShlAmt;
  const int64_t Width = Size - ShrAmt;

  MatchInfo = [=](MachineIRBuilder &B) {
    auto WidthCst = B.buildConstant(ExtractTy, Width);
    auto PosCst = B.buildConstant(ExtractTy, Pos);
    B.buildInstr(ExtrOpcode, {Dst}, {ShlSrc, PosCst, WidthCst});
  };
  return true;
}

bool CombinerHelper::matchBitfieldExtractFromShrAnd(
    MachineInstr &MI, std::function<void(MachineIRBuilder &)> &MatchInfo) {
  const unsigned Opcode = MI.getOpcode();
  assert(Opcode == TargetOpcode::G_LSHR || Opcode == TargetOpcode::G_ASHR);

  const Register Dst = MI.getOperand(0).getReg();
  LLT Ty = MRI.getType(Dst);
  LLT ExtractTy = getTargetLowering().getPreferredShiftAmountTy(Ty);
  if (!getTargetLowering().isConstantUnsignedBitfieldExtractLegal(
          TargetOpcode::G_UBFX, Ty, ExtractTy))
    return false;

  // Try to match shr (and x, c1), c2
  Register AndSrc;
  int64_t ShrAmt;
  int64_t SMask;
  if (!mi_match(Dst, MRI,
                m_BinOp(Opcode,
                        m_OneNonDBGUse(m_GAnd(m_Reg(AndSrc), m_ICst(SMask))),
                        m_ICst(ShrAmt))))
    return false;

  const unsigned Size = Ty.getScalarSizeInBits();
  if (ShrAmt < 0 || ShrAmt >= Size)
    return false;

  // If the shift subsumes the mask, emit the 0 directly.
  if (0 == (SMask >> ShrAmt)) {
    MatchInfo = [=](MachineIRBuilder &B) {
      B.buildConstant(Dst, 0);
    };
    return true;
  }

  // Check that ubfx can do the extraction, with no holes in the mask.
  uint64_t UMask = SMask;
  UMask |= maskTrailingOnes<uint64_t>(ShrAmt);
  UMask &= maskTrailingOnes<uint64_t>(Size);
  if (!isMask_64(UMask))
    return false;

  // Calculate start position and width of the extract.
  const int64_t Pos = ShrAmt;
  const int64_t Width = countTrailingOnes(UMask) - ShrAmt;

  // It's preferable to keep the shift, rather than form G_SBFX.
  // TODO: remove the G_AND via demanded bits analysis.
  if (Opcode == TargetOpcode::G_ASHR && Width + ShrAmt == Size)
    return false;

  MatchInfo = [=](MachineIRBuilder &B) {
    auto WidthCst = B.buildConstant(ExtractTy, Width);
    auto PosCst = B.buildConstant(ExtractTy, Pos);
    B.buildInstr(TargetOpcode::G_UBFX, {Dst}, {AndSrc, PosCst, WidthCst});
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

  auto C1 = getIConstantVRegVal(Src1Def->getOperand(2).getReg(), MRI);
  if (!C1)
    return false;
  auto C2 = getIConstantVRegVal(Src2Reg, MRI);
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

bool CombinerHelper::matchReassocConstantInnerRHS(GPtrAdd &MI,
                                                  MachineInstr *RHS,
                                                  BuildFnTy &MatchInfo) {
  // G_PTR_ADD(BASE, G_ADD(X, C)) -> G_PTR_ADD(G_PTR_ADD(BASE, X), C)
  Register Src1Reg = MI.getOperand(1).getReg();
  if (RHS->getOpcode() != TargetOpcode::G_ADD)
    return false;
  auto C2 = getIConstantVRegVal(RHS->getOperand(2).getReg(), MRI);
  if (!C2)
    return false;

  MatchInfo = [=, &MI](MachineIRBuilder &B) {
    LLT PtrTy = MRI.getType(MI.getOperand(0).getReg());

    auto NewBase =
        Builder.buildPtrAdd(PtrTy, Src1Reg, RHS->getOperand(1).getReg());
    Observer.changingInstr(MI);
    MI.getOperand(1).setReg(NewBase.getReg(0));
    MI.getOperand(2).setReg(RHS->getOperand(2).getReg());
    Observer.changedInstr(MI);
  };
  return !reassociationCanBreakAddressingModePattern(MI);
}

bool CombinerHelper::matchReassocConstantInnerLHS(GPtrAdd &MI,
                                                  MachineInstr *LHS,
                                                  MachineInstr *RHS,
                                                  BuildFnTy &MatchInfo) {
  // G_PTR_ADD (G_PTR_ADD X, C), Y) -> (G_PTR_ADD (G_PTR_ADD(X, Y), C)
  // if and only if (G_PTR_ADD X, C) has one use.
  Register LHSBase;
  Optional<ValueAndVReg> LHSCstOff;
  if (!mi_match(MI.getBaseReg(), MRI,
                m_OneNonDBGUse(m_GPtrAdd(m_Reg(LHSBase), m_GCst(LHSCstOff)))))
    return false;

  auto *LHSPtrAdd = cast<GPtrAdd>(LHS);
  MatchInfo = [=, &MI](MachineIRBuilder &B) {
    // When we change LHSPtrAdd's offset register we might cause it to use a reg
    // before its def. Sink the instruction so the outer PTR_ADD to ensure this
    // doesn't happen.
    LHSPtrAdd->moveBefore(&MI);
    Register RHSReg = MI.getOffsetReg();
    Observer.changingInstr(MI);
    MI.getOperand(2).setReg(LHSCstOff->VReg);
    Observer.changedInstr(MI);
    Observer.changingInstr(*LHSPtrAdd);
    LHSPtrAdd->getOperand(2).setReg(RHSReg);
    Observer.changedInstr(*LHSPtrAdd);
  };
  return !reassociationCanBreakAddressingModePattern(MI);
}

bool CombinerHelper::matchReassocFoldConstantsInSubTree(GPtrAdd &MI,
                                                        MachineInstr *LHS,
                                                        MachineInstr *RHS,
                                                        BuildFnTy &MatchInfo) {
  // G_PTR_ADD(G_PTR_ADD(BASE, C1), C2) -> G_PTR_ADD(BASE, C1+C2)
  auto *LHSPtrAdd = dyn_cast<GPtrAdd>(LHS);
  if (!LHSPtrAdd)
    return false;

  Register Src2Reg = MI.getOperand(2).getReg();
  Register LHSSrc1 = LHSPtrAdd->getBaseReg();
  Register LHSSrc2 = LHSPtrAdd->getOffsetReg();
  auto C1 = getIConstantVRegVal(LHSSrc2, MRI);
  if (!C1)
    return false;
  auto C2 = getIConstantVRegVal(Src2Reg, MRI);
  if (!C2)
    return false;

  MatchInfo = [=, &MI](MachineIRBuilder &B) {
    auto NewCst = B.buildConstant(MRI.getType(Src2Reg), *C1 + *C2);
    Observer.changingInstr(MI);
    MI.getOperand(1).setReg(LHSSrc1);
    MI.getOperand(2).setReg(NewCst.getReg(0));
    Observer.changedInstr(MI);
  };
  return !reassociationCanBreakAddressingModePattern(MI);
}

bool CombinerHelper::matchReassocPtrAdd(MachineInstr &MI,
                                        BuildFnTy &MatchInfo) {
  auto &PtrAdd = cast<GPtrAdd>(MI);
  // We're trying to match a few pointer computation patterns here for
  // re-association opportunities.
  // 1) Isolating a constant operand to be on the RHS, e.g.:
  // G_PTR_ADD(BASE, G_ADD(X, C)) -> G_PTR_ADD(G_PTR_ADD(BASE, X), C)
  //
  // 2) Folding two constants in each sub-tree as long as such folding
  // doesn't break a legal addressing mode.
  // G_PTR_ADD(G_PTR_ADD(BASE, C1), C2) -> G_PTR_ADD(BASE, C1+C2)
  //
  // 3) Move a constant from the LHS of an inner op to the RHS of the outer.
  // G_PTR_ADD (G_PTR_ADD X, C), Y) -> G_PTR_ADD (G_PTR_ADD(X, Y), C)
  // iif (G_PTR_ADD X, C) has one use.
  MachineInstr *LHS = MRI.getVRegDef(PtrAdd.getBaseReg());
  MachineInstr *RHS = MRI.getVRegDef(PtrAdd.getOffsetReg());

  // Try to match example 2.
  if (matchReassocFoldConstantsInSubTree(PtrAdd, LHS, RHS, MatchInfo))
    return true;

  // Try to match example 3.
  if (matchReassocConstantInnerLHS(PtrAdd, LHS, RHS, MatchInfo))
    return true;

  // Try to match example 1.
  if (matchReassocConstantInnerRHS(PtrAdd, RHS, MatchInfo))
    return true;

  return false;
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

bool CombinerHelper::matchNarrowBinopFeedingAnd(
    MachineInstr &MI, std::function<void(MachineIRBuilder &)> &MatchInfo) {
  // Look for a binop feeding into an AND with a mask:
  //
  // %add = G_ADD %lhs, %rhs
  // %and = G_AND %add, 000...11111111
  //
  // Check if it's possible to perform the binop at a narrower width and zext
  // back to the original width like so:
  //
  // %narrow_lhs = G_TRUNC %lhs
  // %narrow_rhs = G_TRUNC %rhs
  // %narrow_add = G_ADD %narrow_lhs, %narrow_rhs
  // %new_add = G_ZEXT %narrow_add
  // %and = G_AND %new_add, 000...11111111
  //
  // This can allow later combines to eliminate the G_AND if it turns out
  // that the mask is irrelevant.
  assert(MI.getOpcode() == TargetOpcode::G_AND);
  Register Dst = MI.getOperand(0).getReg();
  Register AndLHS = MI.getOperand(1).getReg();
  Register AndRHS = MI.getOperand(2).getReg();
  LLT WideTy = MRI.getType(Dst);

  // If the potential binop has more than one use, then it's possible that one
  // of those uses will need its full width.
  if (!WideTy.isScalar() || !MRI.hasOneNonDBGUse(AndLHS))
    return false;

  // Check if the LHS feeding the AND is impacted by the high bits that we're
  // masking out.
  //
  // e.g. for 64-bit x, y:
  //
  // add_64(x, y) & 65535 == zext(add_16(trunc(x), trunc(y))) & 65535
  MachineInstr *LHSInst = getDefIgnoringCopies(AndLHS, MRI);
  if (!LHSInst)
    return false;
  unsigned LHSOpc = LHSInst->getOpcode();
  switch (LHSOpc) {
  default:
    return false;
  case TargetOpcode::G_ADD:
  case TargetOpcode::G_SUB:
  case TargetOpcode::G_MUL:
  case TargetOpcode::G_AND:
  case TargetOpcode::G_OR:
  case TargetOpcode::G_XOR:
    break;
  }

  // Find the mask on the RHS.
  auto Cst = getIConstantVRegValWithLookThrough(AndRHS, MRI);
  if (!Cst)
    return false;
  auto Mask = Cst->Value;
  if (!Mask.isMask())
    return false;

  // No point in combining if there's nothing to truncate.
  unsigned NarrowWidth = Mask.countTrailingOnes();
  if (NarrowWidth == WideTy.getSizeInBits())
    return false;
  LLT NarrowTy = LLT::scalar(NarrowWidth);

  // Check if adding the zext + truncates could be harmful.
  auto &MF = *MI.getMF();
  const auto &TLI = getTargetLowering();
  LLVMContext &Ctx = MF.getFunction().getContext();
  auto &DL = MF.getDataLayout();
  if (!TLI.isTruncateFree(WideTy, NarrowTy, DL, Ctx) ||
      !TLI.isZExtFree(NarrowTy, WideTy, DL, Ctx))
    return false;
  if (!isLegalOrBeforeLegalizer({TargetOpcode::G_TRUNC, {NarrowTy, WideTy}}) ||
      !isLegalOrBeforeLegalizer({TargetOpcode::G_ZEXT, {WideTy, NarrowTy}}))
    return false;
  Register BinOpLHS = LHSInst->getOperand(1).getReg();
  Register BinOpRHS = LHSInst->getOperand(2).getReg();
  MatchInfo = [=, &MI](MachineIRBuilder &B) {
    auto NarrowLHS = Builder.buildTrunc(NarrowTy, BinOpLHS);
    auto NarrowRHS = Builder.buildTrunc(NarrowTy, BinOpRHS);
    auto NarrowBinOp =
        Builder.buildInstr(LHSOpc, {NarrowTy}, {NarrowLHS, NarrowRHS});
    auto Ext = Builder.buildZExt(WideTy, NarrowBinOp);
    Observer.changingInstr(MI);
    MI.getOperand(1).setReg(Ext.getReg(0));
    Observer.changedInstr(MI);
  };
  return true;
}

bool CombinerHelper::matchMulOBy2(MachineInstr &MI, BuildFnTy &MatchInfo) {
  unsigned Opc = MI.getOpcode();
  assert(Opc == TargetOpcode::G_UMULO || Opc == TargetOpcode::G_SMULO);

  if (!mi_match(MI.getOperand(3).getReg(), MRI, m_SpecificICstOrSplat(2)))
    return false;

  MatchInfo = [=, &MI](MachineIRBuilder &B) {
    Observer.changingInstr(MI);
    unsigned NewOpc = Opc == TargetOpcode::G_UMULO ? TargetOpcode::G_UADDO
                                                   : TargetOpcode::G_SADDO;
    MI.setDesc(Builder.getTII().get(NewOpc));
    MI.getOperand(3).setReg(MI.getOperand(2).getReg());
    Observer.changedInstr(MI);
  };
  return true;
}

bool CombinerHelper::matchMulOBy0(MachineInstr &MI, BuildFnTy &MatchInfo) {
  // (G_*MULO x, 0) -> 0 + no carry out
  assert(MI.getOpcode() == TargetOpcode::G_UMULO ||
         MI.getOpcode() == TargetOpcode::G_SMULO);
  if (!mi_match(MI.getOperand(3).getReg(), MRI, m_SpecificICstOrSplat(0)))
    return false;
  Register Dst = MI.getOperand(0).getReg();
  Register Carry = MI.getOperand(1).getReg();
  if (!isConstantLegalOrBeforeLegalizer(MRI.getType(Dst)) ||
      !isConstantLegalOrBeforeLegalizer(MRI.getType(Carry)))
    return false;
  MatchInfo = [=](MachineIRBuilder &B) {
    B.buildConstant(Dst, 0);
    B.buildConstant(Carry, 0);
  };
  return true;
}

bool CombinerHelper::matchAddOBy0(MachineInstr &MI, BuildFnTy &MatchInfo) {
  // (G_*ADDO x, 0) -> x + no carry out
  assert(MI.getOpcode() == TargetOpcode::G_UADDO ||
         MI.getOpcode() == TargetOpcode::G_SADDO);
  if (!mi_match(MI.getOperand(3).getReg(), MRI, m_SpecificICstOrSplat(0)))
    return false;
  Register Carry = MI.getOperand(1).getReg();
  if (!isConstantLegalOrBeforeLegalizer(MRI.getType(Carry)))
    return false;
  Register Dst = MI.getOperand(0).getReg();
  Register LHS = MI.getOperand(2).getReg();
  MatchInfo = [=](MachineIRBuilder &B) {
    B.buildCopy(Dst, LHS);
    B.buildConstant(Carry, 0);
  };
  return true;
}

MachineInstr *CombinerHelper::buildUDivUsingMul(MachineInstr &MI) {
  assert(MI.getOpcode() == TargetOpcode::G_UDIV);
  auto &UDiv = cast<GenericMachineInstr>(MI);
  Register Dst = UDiv.getReg(0);
  Register LHS = UDiv.getReg(1);
  Register RHS = UDiv.getReg(2);
  LLT Ty = MRI.getType(Dst);
  LLT ScalarTy = Ty.getScalarType();
  const unsigned EltBits = ScalarTy.getScalarSizeInBits();
  LLT ShiftAmtTy = getTargetLowering().getPreferredShiftAmountTy(Ty);
  LLT ScalarShiftAmtTy = ShiftAmtTy.getScalarType();
  auto &MIB = Builder;
  MIB.setInstrAndDebugLoc(MI);

  bool UseNPQ = false;
  SmallVector<Register, 16> PreShifts, PostShifts, MagicFactors, NPQFactors;

  auto BuildUDIVPattern = [&](const Constant *C) {
    auto *CI = cast<ConstantInt>(C);
    const APInt &Divisor = CI->getValue();
    UnsignedDivisonByConstantInfo magics =
        UnsignedDivisonByConstantInfo::get(Divisor);
    unsigned PreShift = 0, PostShift = 0;

    // If the divisor is even, we can avoid using the expensive fixup by
    // shifting the divided value upfront.
    if (magics.IsAdd != 0 && !Divisor[0]) {
      PreShift = Divisor.countTrailingZeros();
      // Get magic number for the shifted divisor.
      magics =
          UnsignedDivisonByConstantInfo::get(Divisor.lshr(PreShift), PreShift);
      assert(magics.IsAdd == 0 && "Should use cheap fixup now");
    }

    APInt Magic = magics.Magic;

    unsigned SelNPQ;
    if (magics.IsAdd == 0 || Divisor.isOneValue()) {
      assert(magics.ShiftAmount < Divisor.getBitWidth() &&
             "We shouldn't generate an undefined shift!");
      PostShift = magics.ShiftAmount;
      SelNPQ = false;
    } else {
      PostShift = magics.ShiftAmount - 1;
      SelNPQ = true;
    }

    PreShifts.push_back(
        MIB.buildConstant(ScalarShiftAmtTy, PreShift).getReg(0));
    MagicFactors.push_back(MIB.buildConstant(ScalarTy, Magic).getReg(0));
    NPQFactors.push_back(
        MIB.buildConstant(ScalarTy,
                          SelNPQ ? APInt::getOneBitSet(EltBits, EltBits - 1)
                                 : APInt::getZero(EltBits))
            .getReg(0));
    PostShifts.push_back(
        MIB.buildConstant(ScalarShiftAmtTy, PostShift).getReg(0));
    UseNPQ |= SelNPQ;
    return true;
  };

  // Collect the shifts/magic values from each element.
  bool Matched = matchUnaryPredicate(MRI, RHS, BuildUDIVPattern);
  (void)Matched;
  assert(Matched && "Expected unary predicate match to succeed");

  Register PreShift, PostShift, MagicFactor, NPQFactor;
  auto *RHSDef = getOpcodeDef<GBuildVector>(RHS, MRI);
  if (RHSDef) {
    PreShift = MIB.buildBuildVector(ShiftAmtTy, PreShifts).getReg(0);
    MagicFactor = MIB.buildBuildVector(Ty, MagicFactors).getReg(0);
    NPQFactor = MIB.buildBuildVector(Ty, NPQFactors).getReg(0);
    PostShift = MIB.buildBuildVector(ShiftAmtTy, PostShifts).getReg(0);
  } else {
    assert(MRI.getType(RHS).isScalar() &&
           "Non-build_vector operation should have been a scalar");
    PreShift = PreShifts[0];
    MagicFactor = MagicFactors[0];
    PostShift = PostShifts[0];
  }

  Register Q = LHS;
  Q = MIB.buildLShr(Ty, Q, PreShift).getReg(0);

  // Multiply the numerator (operand 0) by the magic value.
  Q = MIB.buildUMulH(Ty, Q, MagicFactor).getReg(0);

  if (UseNPQ) {
    Register NPQ = MIB.buildSub(Ty, LHS, Q).getReg(0);

    // For vectors we might have a mix of non-NPQ/NPQ paths, so use
    // G_UMULH to act as a SRL-by-1 for NPQ, else multiply by zero.
    if (Ty.isVector())
      NPQ = MIB.buildUMulH(Ty, NPQ, NPQFactor).getReg(0);
    else
      NPQ = MIB.buildLShr(Ty, NPQ, MIB.buildConstant(ShiftAmtTy, 1)).getReg(0);

    Q = MIB.buildAdd(Ty, NPQ, Q).getReg(0);
  }

  Q = MIB.buildLShr(Ty, Q, PostShift).getReg(0);
  auto One = MIB.buildConstant(Ty, 1);
  auto IsOne = MIB.buildICmp(
      CmpInst::Predicate::ICMP_EQ,
      Ty.isScalar() ? LLT::scalar(1) : Ty.changeElementSize(1), RHS, One);
  return MIB.buildSelect(Ty, IsOne, LHS, Q);
}

bool CombinerHelper::matchUDivByConst(MachineInstr &MI) {
  assert(MI.getOpcode() == TargetOpcode::G_UDIV);
  Register Dst = MI.getOperand(0).getReg();
  Register RHS = MI.getOperand(2).getReg();
  LLT DstTy = MRI.getType(Dst);
  auto *RHSDef = MRI.getVRegDef(RHS);
  if (!isConstantOrConstantVector(*RHSDef, MRI))
    return false;

  auto &MF = *MI.getMF();
  AttributeList Attr = MF.getFunction().getAttributes();
  const auto &TLI = getTargetLowering();
  LLVMContext &Ctx = MF.getFunction().getContext();
  auto &DL = MF.getDataLayout();
  if (TLI.isIntDivCheap(getApproximateEVTForLLT(DstTy, DL, Ctx), Attr))
    return false;

  // Don't do this for minsize because the instruction sequence is usually
  // larger.
  if (MF.getFunction().hasMinSize())
    return false;

  // Don't do this if the types are not going to be legal.
  if (LI) {
    if (!isLegalOrBeforeLegalizer({TargetOpcode::G_MUL, {DstTy, DstTy}}))
      return false;
    if (!isLegalOrBeforeLegalizer({TargetOpcode::G_UMULH, {DstTy}}))
      return false;
    if (!isLegalOrBeforeLegalizer(
            {TargetOpcode::G_ICMP,
             {DstTy.isVector() ? DstTy.changeElementSize(1) : LLT::scalar(1),
              DstTy}}))
      return false;
  }

  auto CheckEltValue = [&](const Constant *C) {
    if (auto *CI = dyn_cast_or_null<ConstantInt>(C))
      return !CI->isZero();
    return false;
  };
  return matchUnaryPredicate(MRI, RHS, CheckEltValue);
}

void CombinerHelper::applyUDivByConst(MachineInstr &MI) {
  auto *NewMI = buildUDivUsingMul(MI);
  replaceSingleDefInstWithReg(MI, NewMI->getOperand(0).getReg());
}

bool CombinerHelper::matchUMulHToLShr(MachineInstr &MI) {
  assert(MI.getOpcode() == TargetOpcode::G_UMULH);
  Register RHS = MI.getOperand(2).getReg();
  Register Dst = MI.getOperand(0).getReg();
  LLT Ty = MRI.getType(Dst);
  LLT ShiftAmtTy = getTargetLowering().getPreferredShiftAmountTy(Ty);
  auto MatchPow2ExceptOne = [&](const Constant *C) {
    if (auto *CI = dyn_cast<ConstantInt>(C))
      return CI->getValue().isPowerOf2() && !CI->getValue().isOne();
    return false;
  };
  if (!matchUnaryPredicate(MRI, RHS, MatchPow2ExceptOne, false))
    return false;
  return isLegalOrBeforeLegalizer({TargetOpcode::G_LSHR, {Ty, ShiftAmtTy}});
}

void CombinerHelper::applyUMulHToLShr(MachineInstr &MI) {
  Register LHS = MI.getOperand(1).getReg();
  Register RHS = MI.getOperand(2).getReg();
  Register Dst = MI.getOperand(0).getReg();
  LLT Ty = MRI.getType(Dst);
  LLT ShiftAmtTy = getTargetLowering().getPreferredShiftAmountTy(Ty);
  unsigned NumEltBits = Ty.getScalarSizeInBits();

  Builder.setInstrAndDebugLoc(MI);
  auto LogBase2 = buildLogBase2(RHS, Builder);
  auto ShiftAmt =
      Builder.buildSub(Ty, Builder.buildConstant(Ty, NumEltBits), LogBase2);
  auto Trunc = Builder.buildZExtOrTrunc(ShiftAmtTy, ShiftAmt);
  Builder.buildLShr(Dst, LHS, Trunc);
  MI.eraseFromParent();
}

bool CombinerHelper::matchRedundantNegOperands(MachineInstr &MI,
                                               BuildFnTy &MatchInfo) {
  unsigned Opc = MI.getOpcode();
  assert(Opc == TargetOpcode::G_FADD || Opc == TargetOpcode::G_FSUB ||
         Opc == TargetOpcode::G_FMUL || Opc == TargetOpcode::G_FDIV ||
         Opc == TargetOpcode::G_FMAD || Opc == TargetOpcode::G_FMA);

  Register Dst = MI.getOperand(0).getReg();
  Register X = MI.getOperand(1).getReg();
  Register Y = MI.getOperand(2).getReg();
  LLT Type = MRI.getType(Dst);

  // fold (fadd x, fneg(y)) -> (fsub x, y)
  // fold (fadd fneg(y), x) -> (fsub x, y)
  // G_ADD is commutative so both cases are checked by m_GFAdd
  if (mi_match(Dst, MRI, m_GFAdd(m_Reg(X), m_GFNeg(m_Reg(Y)))) &&
      isLegalOrBeforeLegalizer({TargetOpcode::G_FSUB, {Type}})) {
    Opc = TargetOpcode::G_FSUB;
  }
  /// fold (fsub x, fneg(y)) -> (fadd x, y)
  else if (mi_match(Dst, MRI, m_GFSub(m_Reg(X), m_GFNeg(m_Reg(Y)))) &&
           isLegalOrBeforeLegalizer({TargetOpcode::G_FADD, {Type}})) {
    Opc = TargetOpcode::G_FADD;
  }
  // fold (fmul fneg(x), fneg(y)) -> (fmul x, y)
  // fold (fdiv fneg(x), fneg(y)) -> (fdiv x, y)
  // fold (fmad fneg(x), fneg(y), z) -> (fmad x, y, z)
  // fold (fma fneg(x), fneg(y), z) -> (fma x, y, z)
  else if ((Opc == TargetOpcode::G_FMUL || Opc == TargetOpcode::G_FDIV ||
            Opc == TargetOpcode::G_FMAD || Opc == TargetOpcode::G_FMA) &&
           mi_match(X, MRI, m_GFNeg(m_Reg(X))) &&
           mi_match(Y, MRI, m_GFNeg(m_Reg(Y)))) {
    // no opcode change
  } else
    return false;

  MatchInfo = [=, &MI](MachineIRBuilder &B) {
    Observer.changingInstr(MI);
    MI.setDesc(B.getTII().get(Opc));
    MI.getOperand(1).setReg(X);
    MI.getOperand(2).setReg(Y);
    Observer.changedInstr(MI);
  };
  return true;
}

/// Checks if \p MI is TargetOpcode::G_FMUL and contractable either
/// due to global flags or MachineInstr flags.
static bool isContractableFMul(MachineInstr &MI, bool AllowFusionGlobally) {
  if (MI.getOpcode() != TargetOpcode::G_FMUL)
    return false;
  return AllowFusionGlobally || MI.getFlag(MachineInstr::MIFlag::FmContract);
}

static bool hasMoreUses(const MachineInstr &MI0, const MachineInstr &MI1,
                        const MachineRegisterInfo &MRI) {
  return std::distance(MRI.use_instr_nodbg_begin(MI0.getOperand(0).getReg()),
                       MRI.use_instr_nodbg_end()) >
         std::distance(MRI.use_instr_nodbg_begin(MI1.getOperand(0).getReg()),
                       MRI.use_instr_nodbg_end());
}

bool CombinerHelper::canCombineFMadOrFMA(MachineInstr &MI,
                                         bool &AllowFusionGlobally,
                                         bool &HasFMAD, bool &Aggressive,
                                         bool CanReassociate) {

  auto *MF = MI.getMF();
  const auto &TLI = *MF->getSubtarget().getTargetLowering();
  const TargetOptions &Options = MF->getTarget().Options;
  LLT DstType = MRI.getType(MI.getOperand(0).getReg());

  if (CanReassociate &&
      !(Options.UnsafeFPMath || MI.getFlag(MachineInstr::MIFlag::FmReassoc)))
    return false;

  // Floating-point multiply-add with intermediate rounding.
  HasFMAD = (LI && TLI.isFMADLegal(MI, DstType));
  // Floating-point multiply-add without intermediate rounding.
  bool HasFMA = TLI.isFMAFasterThanFMulAndFAdd(*MF, DstType) &&
                isLegalOrBeforeLegalizer({TargetOpcode::G_FMA, {DstType}});
  // No valid opcode, do not combine.
  if (!HasFMAD && !HasFMA)
    return false;

  AllowFusionGlobally = Options.AllowFPOpFusion == FPOpFusion::Fast ||
                        Options.UnsafeFPMath || HasFMAD;
  // If the addition is not contractable, do not combine.
  if (!AllowFusionGlobally && !MI.getFlag(MachineInstr::MIFlag::FmContract))
    return false;

  Aggressive = TLI.enableAggressiveFMAFusion(DstType);
  return true;
}

bool CombinerHelper::matchCombineFAddFMulToFMadOrFMA(
    MachineInstr &MI, std::function<void(MachineIRBuilder &)> &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_FADD);

  bool AllowFusionGlobally, HasFMAD, Aggressive;
  if (!canCombineFMadOrFMA(MI, AllowFusionGlobally, HasFMAD, Aggressive))
    return false;

  Register Op1 = MI.getOperand(1).getReg();
  Register Op2 = MI.getOperand(2).getReg();
  DefinitionAndSourceRegister LHS = {MRI.getVRegDef(Op1), Op1};
  DefinitionAndSourceRegister RHS = {MRI.getVRegDef(Op2), Op2};
  unsigned PreferredFusedOpcode =
      HasFMAD ? TargetOpcode::G_FMAD : TargetOpcode::G_FMA;

  // If we have two choices trying to fold (fadd (fmul u, v), (fmul x, y)),
  // prefer to fold the multiply with fewer uses.
  if (Aggressive && isContractableFMul(*LHS.MI, AllowFusionGlobally) &&
      isContractableFMul(*RHS.MI, AllowFusionGlobally)) {
    if (hasMoreUses(*LHS.MI, *RHS.MI, MRI))
      std::swap(LHS, RHS);
  }

  // fold (fadd (fmul x, y), z) -> (fma x, y, z)
  if (isContractableFMul(*LHS.MI, AllowFusionGlobally) &&
      (Aggressive || MRI.hasOneNonDBGUse(LHS.Reg))) {
    MatchInfo = [=, &MI](MachineIRBuilder &B) {
      B.buildInstr(PreferredFusedOpcode, {MI.getOperand(0).getReg()},
                   {LHS.MI->getOperand(1).getReg(),
                    LHS.MI->getOperand(2).getReg(), RHS.Reg});
    };
    return true;
  }

  // fold (fadd x, (fmul y, z)) -> (fma y, z, x)
  if (isContractableFMul(*RHS.MI, AllowFusionGlobally) &&
      (Aggressive || MRI.hasOneNonDBGUse(RHS.Reg))) {
    MatchInfo = [=, &MI](MachineIRBuilder &B) {
      B.buildInstr(PreferredFusedOpcode, {MI.getOperand(0).getReg()},
                   {RHS.MI->getOperand(1).getReg(),
                    RHS.MI->getOperand(2).getReg(), LHS.Reg});
    };
    return true;
  }

  return false;
}

bool CombinerHelper::matchCombineFAddFpExtFMulToFMadOrFMA(
    MachineInstr &MI, std::function<void(MachineIRBuilder &)> &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_FADD);

  bool AllowFusionGlobally, HasFMAD, Aggressive;
  if (!canCombineFMadOrFMA(MI, AllowFusionGlobally, HasFMAD, Aggressive))
    return false;

  const auto &TLI = *MI.getMF()->getSubtarget().getTargetLowering();
  Register Op1 = MI.getOperand(1).getReg();
  Register Op2 = MI.getOperand(2).getReg();
  DefinitionAndSourceRegister LHS = {MRI.getVRegDef(Op1), Op1};
  DefinitionAndSourceRegister RHS = {MRI.getVRegDef(Op2), Op2};
  LLT DstType = MRI.getType(MI.getOperand(0).getReg());

  unsigned PreferredFusedOpcode =
      HasFMAD ? TargetOpcode::G_FMAD : TargetOpcode::G_FMA;

  // If we have two choices trying to fold (fadd (fmul u, v), (fmul x, y)),
  // prefer to fold the multiply with fewer uses.
  if (Aggressive && isContractableFMul(*LHS.MI, AllowFusionGlobally) &&
      isContractableFMul(*RHS.MI, AllowFusionGlobally)) {
    if (hasMoreUses(*LHS.MI, *RHS.MI, MRI))
      std::swap(LHS, RHS);
  }

  // fold (fadd (fpext (fmul x, y)), z) -> (fma (fpext x), (fpext y), z)
  MachineInstr *FpExtSrc;
  if (mi_match(LHS.Reg, MRI, m_GFPExt(m_MInstr(FpExtSrc))) &&
      isContractableFMul(*FpExtSrc, AllowFusionGlobally) &&
      TLI.isFPExtFoldable(MI, PreferredFusedOpcode, DstType,
                          MRI.getType(FpExtSrc->getOperand(1).getReg()))) {
    MatchInfo = [=, &MI](MachineIRBuilder &B) {
      auto FpExtX = B.buildFPExt(DstType, FpExtSrc->getOperand(1).getReg());
      auto FpExtY = B.buildFPExt(DstType, FpExtSrc->getOperand(2).getReg());
      B.buildInstr(PreferredFusedOpcode, {MI.getOperand(0).getReg()},
                   {FpExtX.getReg(0), FpExtY.getReg(0), RHS.Reg});
    };
    return true;
  }

  // fold (fadd z, (fpext (fmul x, y))) -> (fma (fpext x), (fpext y), z)
  // Note: Commutes FADD operands.
  if (mi_match(RHS.Reg, MRI, m_GFPExt(m_MInstr(FpExtSrc))) &&
      isContractableFMul(*FpExtSrc, AllowFusionGlobally) &&
      TLI.isFPExtFoldable(MI, PreferredFusedOpcode, DstType,
                          MRI.getType(FpExtSrc->getOperand(1).getReg()))) {
    MatchInfo = [=, &MI](MachineIRBuilder &B) {
      auto FpExtX = B.buildFPExt(DstType, FpExtSrc->getOperand(1).getReg());
      auto FpExtY = B.buildFPExt(DstType, FpExtSrc->getOperand(2).getReg());
      B.buildInstr(PreferredFusedOpcode, {MI.getOperand(0).getReg()},
                   {FpExtX.getReg(0), FpExtY.getReg(0), LHS.Reg});
    };
    return true;
  }

  return false;
}

bool CombinerHelper::matchCombineFAddFMAFMulToFMadOrFMA(
    MachineInstr &MI, std::function<void(MachineIRBuilder &)> &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_FADD);

  bool AllowFusionGlobally, HasFMAD, Aggressive;
  if (!canCombineFMadOrFMA(MI, AllowFusionGlobally, HasFMAD, Aggressive, true))
    return false;

  Register Op1 = MI.getOperand(1).getReg();
  Register Op2 = MI.getOperand(2).getReg();
  DefinitionAndSourceRegister LHS = {MRI.getVRegDef(Op1), Op1};
  DefinitionAndSourceRegister RHS = {MRI.getVRegDef(Op2), Op2};
  LLT DstTy = MRI.getType(MI.getOperand(0).getReg());

  unsigned PreferredFusedOpcode =
      HasFMAD ? TargetOpcode::G_FMAD : TargetOpcode::G_FMA;

  // If we have two choices trying to fold (fadd (fmul u, v), (fmul x, y)),
  // prefer to fold the multiply with fewer uses.
  if (Aggressive && isContractableFMul(*LHS.MI, AllowFusionGlobally) &&
      isContractableFMul(*RHS.MI, AllowFusionGlobally)) {
    if (hasMoreUses(*LHS.MI, *RHS.MI, MRI))
      std::swap(LHS, RHS);
  }

  MachineInstr *FMA = nullptr;
  Register Z;
  // fold (fadd (fma x, y, (fmul u, v)), z) -> (fma x, y, (fma u, v, z))
  if (LHS.MI->getOpcode() == PreferredFusedOpcode &&
      (MRI.getVRegDef(LHS.MI->getOperand(3).getReg())->getOpcode() ==
       TargetOpcode::G_FMUL) &&
      MRI.hasOneNonDBGUse(LHS.MI->getOperand(0).getReg()) &&
      MRI.hasOneNonDBGUse(LHS.MI->getOperand(3).getReg())) {
    FMA = LHS.MI;
    Z = RHS.Reg;
  }
  // fold (fadd z, (fma x, y, (fmul u, v))) -> (fma x, y, (fma u, v, z))
  else if (RHS.MI->getOpcode() == PreferredFusedOpcode &&
           (MRI.getVRegDef(RHS.MI->getOperand(3).getReg())->getOpcode() ==
            TargetOpcode::G_FMUL) &&
           MRI.hasOneNonDBGUse(RHS.MI->getOperand(0).getReg()) &&
           MRI.hasOneNonDBGUse(RHS.MI->getOperand(3).getReg())) {
    Z = LHS.Reg;
    FMA = RHS.MI;
  }

  if (FMA) {
    MachineInstr *FMulMI = MRI.getVRegDef(FMA->getOperand(3).getReg());
    Register X = FMA->getOperand(1).getReg();
    Register Y = FMA->getOperand(2).getReg();
    Register U = FMulMI->getOperand(1).getReg();
    Register V = FMulMI->getOperand(2).getReg();

    MatchInfo = [=, &MI](MachineIRBuilder &B) {
      Register InnerFMA = MRI.createGenericVirtualRegister(DstTy);
      B.buildInstr(PreferredFusedOpcode, {InnerFMA}, {U, V, Z});
      B.buildInstr(PreferredFusedOpcode, {MI.getOperand(0).getReg()},
                   {X, Y, InnerFMA});
    };
    return true;
  }

  return false;
}

bool CombinerHelper::matchCombineFAddFpExtFMulToFMadOrFMAAggressive(
    MachineInstr &MI, std::function<void(MachineIRBuilder &)> &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_FADD);

  bool AllowFusionGlobally, HasFMAD, Aggressive;
  if (!canCombineFMadOrFMA(MI, AllowFusionGlobally, HasFMAD, Aggressive))
    return false;

  if (!Aggressive)
    return false;

  const auto &TLI = *MI.getMF()->getSubtarget().getTargetLowering();
  LLT DstType = MRI.getType(MI.getOperand(0).getReg());
  Register Op1 = MI.getOperand(1).getReg();
  Register Op2 = MI.getOperand(2).getReg();
  DefinitionAndSourceRegister LHS = {MRI.getVRegDef(Op1), Op1};
  DefinitionAndSourceRegister RHS = {MRI.getVRegDef(Op2), Op2};

  unsigned PreferredFusedOpcode =
      HasFMAD ? TargetOpcode::G_FMAD : TargetOpcode::G_FMA;

  // If we have two choices trying to fold (fadd (fmul u, v), (fmul x, y)),
  // prefer to fold the multiply with fewer uses.
  if (Aggressive && isContractableFMul(*LHS.MI, AllowFusionGlobally) &&
      isContractableFMul(*RHS.MI, AllowFusionGlobally)) {
    if (hasMoreUses(*LHS.MI, *RHS.MI, MRI))
      std::swap(LHS, RHS);
  }

  // Builds: (fma x, y, (fma (fpext u), (fpext v), z))
  auto buildMatchInfo = [=, &MI](Register U, Register V, Register Z, Register X,
                                 Register Y, MachineIRBuilder &B) {
    Register FpExtU = B.buildFPExt(DstType, U).getReg(0);
    Register FpExtV = B.buildFPExt(DstType, V).getReg(0);
    Register InnerFMA =
        B.buildInstr(PreferredFusedOpcode, {DstType}, {FpExtU, FpExtV, Z})
            .getReg(0);
    B.buildInstr(PreferredFusedOpcode, {MI.getOperand(0).getReg()},
                 {X, Y, InnerFMA});
  };

  MachineInstr *FMulMI, *FMAMI;
  // fold (fadd (fma x, y, (fpext (fmul u, v))), z)
  //   -> (fma x, y, (fma (fpext u), (fpext v), z))
  if (LHS.MI->getOpcode() == PreferredFusedOpcode &&
      mi_match(LHS.MI->getOperand(3).getReg(), MRI,
               m_GFPExt(m_MInstr(FMulMI))) &&
      isContractableFMul(*FMulMI, AllowFusionGlobally) &&
      TLI.isFPExtFoldable(MI, PreferredFusedOpcode, DstType,
                          MRI.getType(FMulMI->getOperand(0).getReg()))) {
    MatchInfo = [=](MachineIRBuilder &B) {
      buildMatchInfo(FMulMI->getOperand(1).getReg(),
                     FMulMI->getOperand(2).getReg(), RHS.Reg,
                     LHS.MI->getOperand(1).getReg(),
                     LHS.MI->getOperand(2).getReg(), B);
    };
    return true;
  }

  // fold (fadd (fpext (fma x, y, (fmul u, v))), z)
  //   -> (fma (fpext x), (fpext y), (fma (fpext u), (fpext v), z))
  // FIXME: This turns two single-precision and one double-precision
  // operation into two double-precision operations, which might not be
  // interesting for all targets, especially GPUs.
  if (mi_match(LHS.Reg, MRI, m_GFPExt(m_MInstr(FMAMI))) &&
      FMAMI->getOpcode() == PreferredFusedOpcode) {
    MachineInstr *FMulMI = MRI.getVRegDef(FMAMI->getOperand(3).getReg());
    if (isContractableFMul(*FMulMI, AllowFusionGlobally) &&
        TLI.isFPExtFoldable(MI, PreferredFusedOpcode, DstType,
                            MRI.getType(FMAMI->getOperand(0).getReg()))) {
      MatchInfo = [=](MachineIRBuilder &B) {
        Register X = FMAMI->getOperand(1).getReg();
        Register Y = FMAMI->getOperand(2).getReg();
        X = B.buildFPExt(DstType, X).getReg(0);
        Y = B.buildFPExt(DstType, Y).getReg(0);
        buildMatchInfo(FMulMI->getOperand(1).getReg(),
                       FMulMI->getOperand(2).getReg(), RHS.Reg, X, Y, B);
      };

      return true;
    }
  }

  // fold (fadd z, (fma x, y, (fpext (fmul u, v)))
  //   -> (fma x, y, (fma (fpext u), (fpext v), z))
  if (RHS.MI->getOpcode() == PreferredFusedOpcode &&
      mi_match(RHS.MI->getOperand(3).getReg(), MRI,
               m_GFPExt(m_MInstr(FMulMI))) &&
      isContractableFMul(*FMulMI, AllowFusionGlobally) &&
      TLI.isFPExtFoldable(MI, PreferredFusedOpcode, DstType,
                          MRI.getType(FMulMI->getOperand(0).getReg()))) {
    MatchInfo = [=](MachineIRBuilder &B) {
      buildMatchInfo(FMulMI->getOperand(1).getReg(),
                     FMulMI->getOperand(2).getReg(), LHS.Reg,
                     RHS.MI->getOperand(1).getReg(),
                     RHS.MI->getOperand(2).getReg(), B);
    };
    return true;
  }

  // fold (fadd z, (fpext (fma x, y, (fmul u, v)))
  //   -> (fma (fpext x), (fpext y), (fma (fpext u), (fpext v), z))
  // FIXME: This turns two single-precision and one double-precision
  // operation into two double-precision operations, which might not be
  // interesting for all targets, especially GPUs.
  if (mi_match(RHS.Reg, MRI, m_GFPExt(m_MInstr(FMAMI))) &&
      FMAMI->getOpcode() == PreferredFusedOpcode) {
    MachineInstr *FMulMI = MRI.getVRegDef(FMAMI->getOperand(3).getReg());
    if (isContractableFMul(*FMulMI, AllowFusionGlobally) &&
        TLI.isFPExtFoldable(MI, PreferredFusedOpcode, DstType,
                            MRI.getType(FMAMI->getOperand(0).getReg()))) {
      MatchInfo = [=](MachineIRBuilder &B) {
        Register X = FMAMI->getOperand(1).getReg();
        Register Y = FMAMI->getOperand(2).getReg();
        X = B.buildFPExt(DstType, X).getReg(0);
        Y = B.buildFPExt(DstType, Y).getReg(0);
        buildMatchInfo(FMulMI->getOperand(1).getReg(),
                       FMulMI->getOperand(2).getReg(), LHS.Reg, X, Y, B);
      };
      return true;
    }
  }

  return false;
}

bool CombinerHelper::matchCombineFSubFMulToFMadOrFMA(
    MachineInstr &MI, std::function<void(MachineIRBuilder &)> &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_FSUB);

  bool AllowFusionGlobally, HasFMAD, Aggressive;
  if (!canCombineFMadOrFMA(MI, AllowFusionGlobally, HasFMAD, Aggressive))
    return false;

  Register Op1 = MI.getOperand(1).getReg();
  Register Op2 = MI.getOperand(2).getReg();
  DefinitionAndSourceRegister LHS = {MRI.getVRegDef(Op1), Op1};
  DefinitionAndSourceRegister RHS = {MRI.getVRegDef(Op2), Op2};
  LLT DstTy = MRI.getType(MI.getOperand(0).getReg());

  // If we have two choices trying to fold (fadd (fmul u, v), (fmul x, y)),
  // prefer to fold the multiply with fewer uses.
  int FirstMulHasFewerUses = true;
  if (isContractableFMul(*LHS.MI, AllowFusionGlobally) &&
      isContractableFMul(*RHS.MI, AllowFusionGlobally) &&
      hasMoreUses(*LHS.MI, *RHS.MI, MRI))
    FirstMulHasFewerUses = false;

  unsigned PreferredFusedOpcode =
      HasFMAD ? TargetOpcode::G_FMAD : TargetOpcode::G_FMA;

  // fold (fsub (fmul x, y), z) -> (fma x, y, -z)
  if (FirstMulHasFewerUses &&
      (isContractableFMul(*LHS.MI, AllowFusionGlobally) &&
       (Aggressive || MRI.hasOneNonDBGUse(LHS.Reg)))) {
    MatchInfo = [=, &MI](MachineIRBuilder &B) {
      Register NegZ = B.buildFNeg(DstTy, RHS.Reg).getReg(0);
      B.buildInstr(PreferredFusedOpcode, {MI.getOperand(0).getReg()},
                   {LHS.MI->getOperand(1).getReg(),
                    LHS.MI->getOperand(2).getReg(), NegZ});
    };
    return true;
  }
  // fold (fsub x, (fmul y, z)) -> (fma -y, z, x)
  else if ((isContractableFMul(*RHS.MI, AllowFusionGlobally) &&
            (Aggressive || MRI.hasOneNonDBGUse(RHS.Reg)))) {
    MatchInfo = [=, &MI](MachineIRBuilder &B) {
      Register NegY =
          B.buildFNeg(DstTy, RHS.MI->getOperand(1).getReg()).getReg(0);
      B.buildInstr(PreferredFusedOpcode, {MI.getOperand(0).getReg()},
                   {NegY, RHS.MI->getOperand(2).getReg(), LHS.Reg});
    };
    return true;
  }

  return false;
}

bool CombinerHelper::matchCombineFSubFNegFMulToFMadOrFMA(
    MachineInstr &MI, std::function<void(MachineIRBuilder &)> &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_FSUB);

  bool AllowFusionGlobally, HasFMAD, Aggressive;
  if (!canCombineFMadOrFMA(MI, AllowFusionGlobally, HasFMAD, Aggressive))
    return false;

  Register LHSReg = MI.getOperand(1).getReg();
  Register RHSReg = MI.getOperand(2).getReg();
  LLT DstTy = MRI.getType(MI.getOperand(0).getReg());

  unsigned PreferredFusedOpcode =
      HasFMAD ? TargetOpcode::G_FMAD : TargetOpcode::G_FMA;

  MachineInstr *FMulMI;
  // fold (fsub (fneg (fmul x, y)), z) -> (fma (fneg x), y, (fneg z))
  if (mi_match(LHSReg, MRI, m_GFNeg(m_MInstr(FMulMI))) &&
      (Aggressive || (MRI.hasOneNonDBGUse(LHSReg) &&
                      MRI.hasOneNonDBGUse(FMulMI->getOperand(0).getReg()))) &&
      isContractableFMul(*FMulMI, AllowFusionGlobally)) {
    MatchInfo = [=, &MI](MachineIRBuilder &B) {
      Register NegX =
          B.buildFNeg(DstTy, FMulMI->getOperand(1).getReg()).getReg(0);
      Register NegZ = B.buildFNeg(DstTy, RHSReg).getReg(0);
      B.buildInstr(PreferredFusedOpcode, {MI.getOperand(0).getReg()},
                   {NegX, FMulMI->getOperand(2).getReg(), NegZ});
    };
    return true;
  }

  // fold (fsub x, (fneg (fmul, y, z))) -> (fma y, z, x)
  if (mi_match(RHSReg, MRI, m_GFNeg(m_MInstr(FMulMI))) &&
      (Aggressive || (MRI.hasOneNonDBGUse(RHSReg) &&
                      MRI.hasOneNonDBGUse(FMulMI->getOperand(0).getReg()))) &&
      isContractableFMul(*FMulMI, AllowFusionGlobally)) {
    MatchInfo = [=, &MI](MachineIRBuilder &B) {
      B.buildInstr(PreferredFusedOpcode, {MI.getOperand(0).getReg()},
                   {FMulMI->getOperand(1).getReg(),
                    FMulMI->getOperand(2).getReg(), LHSReg});
    };
    return true;
  }

  return false;
}

bool CombinerHelper::matchCombineFSubFpExtFMulToFMadOrFMA(
    MachineInstr &MI, std::function<void(MachineIRBuilder &)> &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_FSUB);

  bool AllowFusionGlobally, HasFMAD, Aggressive;
  if (!canCombineFMadOrFMA(MI, AllowFusionGlobally, HasFMAD, Aggressive))
    return false;

  Register LHSReg = MI.getOperand(1).getReg();
  Register RHSReg = MI.getOperand(2).getReg();
  LLT DstTy = MRI.getType(MI.getOperand(0).getReg());

  unsigned PreferredFusedOpcode =
      HasFMAD ? TargetOpcode::G_FMAD : TargetOpcode::G_FMA;

  MachineInstr *FMulMI;
  // fold (fsub (fpext (fmul x, y)), z) -> (fma (fpext x), (fpext y), (fneg z))
  if (mi_match(LHSReg, MRI, m_GFPExt(m_MInstr(FMulMI))) &&
      isContractableFMul(*FMulMI, AllowFusionGlobally) &&
      (Aggressive || MRI.hasOneNonDBGUse(LHSReg))) {
    MatchInfo = [=, &MI](MachineIRBuilder &B) {
      Register FpExtX =
          B.buildFPExt(DstTy, FMulMI->getOperand(1).getReg()).getReg(0);
      Register FpExtY =
          B.buildFPExt(DstTy, FMulMI->getOperand(2).getReg()).getReg(0);
      Register NegZ = B.buildFNeg(DstTy, RHSReg).getReg(0);
      B.buildInstr(PreferredFusedOpcode, {MI.getOperand(0).getReg()},
                   {FpExtX, FpExtY, NegZ});
    };
    return true;
  }

  // fold (fsub x, (fpext (fmul y, z))) -> (fma (fneg (fpext y)), (fpext z), x)
  if (mi_match(RHSReg, MRI, m_GFPExt(m_MInstr(FMulMI))) &&
      isContractableFMul(*FMulMI, AllowFusionGlobally) &&
      (Aggressive || MRI.hasOneNonDBGUse(RHSReg))) {
    MatchInfo = [=, &MI](MachineIRBuilder &B) {
      Register FpExtY =
          B.buildFPExt(DstTy, FMulMI->getOperand(1).getReg()).getReg(0);
      Register NegY = B.buildFNeg(DstTy, FpExtY).getReg(0);
      Register FpExtZ =
          B.buildFPExt(DstTy, FMulMI->getOperand(2).getReg()).getReg(0);
      B.buildInstr(PreferredFusedOpcode, {MI.getOperand(0).getReg()},
                   {NegY, FpExtZ, LHSReg});
    };
    return true;
  }

  return false;
}

bool CombinerHelper::matchCombineFSubFpExtFNegFMulToFMadOrFMA(
    MachineInstr &MI, std::function<void(MachineIRBuilder &)> &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_FSUB);

  bool AllowFusionGlobally, HasFMAD, Aggressive;
  if (!canCombineFMadOrFMA(MI, AllowFusionGlobally, HasFMAD, Aggressive))
    return false;

  const auto &TLI = *MI.getMF()->getSubtarget().getTargetLowering();
  LLT DstTy = MRI.getType(MI.getOperand(0).getReg());
  Register LHSReg = MI.getOperand(1).getReg();
  Register RHSReg = MI.getOperand(2).getReg();

  unsigned PreferredFusedOpcode =
      HasFMAD ? TargetOpcode::G_FMAD : TargetOpcode::G_FMA;

  auto buildMatchInfo = [=](Register Dst, Register X, Register Y, Register Z,
                            MachineIRBuilder &B) {
    Register FpExtX = B.buildFPExt(DstTy, X).getReg(0);
    Register FpExtY = B.buildFPExt(DstTy, Y).getReg(0);
    B.buildInstr(PreferredFusedOpcode, {Dst}, {FpExtX, FpExtY, Z});
  };

  MachineInstr *FMulMI;
  // fold (fsub (fpext (fneg (fmul x, y))), z) ->
  //      (fneg (fma (fpext x), (fpext y), z))
  // fold (fsub (fneg (fpext (fmul x, y))), z) ->
  //      (fneg (fma (fpext x), (fpext y), z))
  if ((mi_match(LHSReg, MRI, m_GFPExt(m_GFNeg(m_MInstr(FMulMI)))) ||
       mi_match(LHSReg, MRI, m_GFNeg(m_GFPExt(m_MInstr(FMulMI))))) &&
      isContractableFMul(*FMulMI, AllowFusionGlobally) &&
      TLI.isFPExtFoldable(MI, PreferredFusedOpcode, DstTy,
                          MRI.getType(FMulMI->getOperand(0).getReg()))) {
    MatchInfo = [=, &MI](MachineIRBuilder &B) {
      Register FMAReg = MRI.createGenericVirtualRegister(DstTy);
      buildMatchInfo(FMAReg, FMulMI->getOperand(1).getReg(),
                     FMulMI->getOperand(2).getReg(), RHSReg, B);
      B.buildFNeg(MI.getOperand(0).getReg(), FMAReg);
    };
    return true;
  }

  // fold (fsub x, (fpext (fneg (fmul y, z)))) -> (fma (fpext y), (fpext z), x)
  // fold (fsub x, (fneg (fpext (fmul y, z)))) -> (fma (fpext y), (fpext z), x)
  if ((mi_match(RHSReg, MRI, m_GFPExt(m_GFNeg(m_MInstr(FMulMI)))) ||
       mi_match(RHSReg, MRI, m_GFNeg(m_GFPExt(m_MInstr(FMulMI))))) &&
      isContractableFMul(*FMulMI, AllowFusionGlobally) &&
      TLI.isFPExtFoldable(MI, PreferredFusedOpcode, DstTy,
                          MRI.getType(FMulMI->getOperand(0).getReg()))) {
    MatchInfo = [=, &MI](MachineIRBuilder &B) {
      buildMatchInfo(MI.getOperand(0).getReg(), FMulMI->getOperand(1).getReg(),
                     FMulMI->getOperand(2).getReg(), LHSReg, B);
    };
    return true;
  }

  return false;
}

bool CombinerHelper::matchSelectToLogical(MachineInstr &MI,
                                          BuildFnTy &MatchInfo) {
  GSelect &Sel = cast<GSelect>(MI);
  Register DstReg = Sel.getReg(0);
  Register Cond = Sel.getCondReg();
  Register TrueReg = Sel.getTrueReg();
  Register FalseReg = Sel.getFalseReg();

  auto *TrueDef = getDefIgnoringCopies(TrueReg, MRI);
  auto *FalseDef = getDefIgnoringCopies(FalseReg, MRI);

  const LLT CondTy = MRI.getType(Cond);
  const LLT OpTy = MRI.getType(TrueReg);
  if (CondTy != OpTy || OpTy.getScalarSizeInBits() != 1)
    return false;

  // We have a boolean select.

  // select Cond, Cond, F --> or Cond, F
  // select Cond, 1, F    --> or Cond, F
  auto MaybeCstTrue = isConstantOrConstantSplatVector(*TrueDef, MRI);
  if (Cond == TrueReg || (MaybeCstTrue && MaybeCstTrue->isOne())) {
    MatchInfo = [=](MachineIRBuilder &MIB) {
      MIB.buildOr(DstReg, Cond, FalseReg);
    };
    return true;
  }

  // select Cond, T, Cond --> and Cond, T
  // select Cond, T, 0    --> and Cond, T
  auto MaybeCstFalse = isConstantOrConstantSplatVector(*FalseDef, MRI);
  if (Cond == FalseReg || (MaybeCstFalse && MaybeCstFalse->isZero())) {
    MatchInfo = [=](MachineIRBuilder &MIB) {
      MIB.buildAnd(DstReg, Cond, TrueReg);
    };
    return true;
  }

 // select Cond, T, 1 --> or (not Cond), T
  if (MaybeCstFalse && MaybeCstFalse->isOne()) {
    MatchInfo = [=](MachineIRBuilder &MIB) {
      MIB.buildOr(DstReg, MIB.buildNot(OpTy, Cond), TrueReg);
    };
    return true;
  }

  // select Cond, 0, F --> and (not Cond), F
  if (MaybeCstTrue && MaybeCstTrue->isZero()) {
    MatchInfo = [=](MachineIRBuilder &MIB) {
      MIB.buildAnd(DstReg, MIB.buildNot(OpTy, Cond), FalseReg);
    };
    return true;
  }
  return false;
}

bool CombinerHelper::matchCombineFMinMaxNaN(MachineInstr &MI,
                                            unsigned &IdxToPropagate) {
  bool PropagateNaN;
  switch (MI.getOpcode()) {
  default:
    return false;
  case TargetOpcode::G_FMINNUM:
  case TargetOpcode::G_FMAXNUM:
    PropagateNaN = false;
    break;
  case TargetOpcode::G_FMINIMUM:
  case TargetOpcode::G_FMAXIMUM:
    PropagateNaN = true;
    break;
  }

  auto MatchNaN = [&](unsigned Idx) {
    Register MaybeNaNReg = MI.getOperand(Idx).getReg();
    const ConstantFP *MaybeCst = getConstantFPVRegVal(MaybeNaNReg, MRI);
    if (!MaybeCst || !MaybeCst->getValueAPF().isNaN())
      return false;
    IdxToPropagate = PropagateNaN ? Idx : (Idx == 1 ? 2 : 1);
    return true;
  };

  return MatchNaN(1) || MatchNaN(2);
}

bool CombinerHelper::matchAddSubSameReg(MachineInstr &MI, Register &Src) {
  assert(MI.getOpcode() == TargetOpcode::G_ADD && "Expected a G_ADD");
  Register LHS = MI.getOperand(1).getReg();
  Register RHS = MI.getOperand(2).getReg();

  // Helper lambda to check for opportunities for
  // A + (B - A) -> B
  // (B - A) + A -> B
  auto CheckFold = [&](Register MaybeSub, Register MaybeSameReg) {
    Register Reg;
    return mi_match(MaybeSub, MRI, m_GSub(m_Reg(Src), m_Reg(Reg))) &&
           Reg == MaybeSameReg;
  };
  return CheckFold(LHS, RHS) || CheckFold(RHS, LHS);
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
