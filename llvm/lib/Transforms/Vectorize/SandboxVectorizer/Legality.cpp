//===- Legality.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/Legality.h"
#include "llvm/SandboxIR/Instruction.h"
#include "llvm/SandboxIR/Operator.h"
#include "llvm/SandboxIR/Utils.h"
#include "llvm/SandboxIR/Value.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/InstrMaps.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/VecUtils.h"

namespace llvm::sandboxir {

#ifndef NDEBUG
void ShuffleMask::dump() const {
  print(dbgs());
  dbgs() << "\n";
}

void LegalityResult::dump() const {
  print(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

std::optional<ResultReason>
LegalityAnalysis::notVectorizableBasedOnOpcodesAndTypes(
    ArrayRef<Value *> Bndl) {
  auto *I0 = cast<Instruction>(Bndl[0]);
  auto Opcode = I0->getOpcode();
  // If they have different opcodes, then we cannot form a vector (for now).
  if (any_of(drop_begin(Bndl), [Opcode](Value *V) {
        return cast<Instruction>(V)->getOpcode() != Opcode;
      }))
    return ResultReason::DiffOpcodes;

  // If not the same scalar type, Pack. This will accept scalars and vectors as
  // long as the element type is the same.
  Type *ElmTy0 = VecUtils::getElementType(Utils::getExpectedType(I0));
  if (any_of(drop_begin(Bndl), [ElmTy0](Value *V) {
        return VecUtils::getElementType(Utils::getExpectedType(V)) != ElmTy0;
      }))
    return ResultReason::DiffTypes;

  // TODO: Allow vectorization of instrs with different flags as long as we
  // change them to the least common one.
  // For now pack if differnt FastMathFlags.
  if (isa<FPMathOperator>(I0)) {
    FastMathFlags FMF0 = cast<Instruction>(Bndl[0])->getFastMathFlags();
    if (any_of(drop_begin(Bndl), [FMF0](auto *V) {
          return cast<Instruction>(V)->getFastMathFlags() != FMF0;
        }))
      return ResultReason::DiffMathFlags;
  }

  // TODO: Allow vectorization by using common flags.
  // For now Pack if they don't have the same wrap flags.
  bool CanHaveWrapFlags =
      isa<OverflowingBinaryOperator>(I0) || isa<TruncInst>(I0);
  if (CanHaveWrapFlags) {
    bool NUW0 = I0->hasNoUnsignedWrap();
    bool NSW0 = I0->hasNoSignedWrap();
    if (any_of(drop_begin(Bndl), [NUW0, NSW0](auto *V) {
          return cast<Instruction>(V)->hasNoUnsignedWrap() != NUW0 ||
                 cast<Instruction>(V)->hasNoSignedWrap() != NSW0;
        })) {
      return ResultReason::DiffWrapFlags;
    }
  }

  // Now we need to do further checks for specific opcodes.
  switch (Opcode) {
  case Instruction::Opcode::ZExt:
  case Instruction::Opcode::SExt:
  case Instruction::Opcode::FPToUI:
  case Instruction::Opcode::FPToSI:
  case Instruction::Opcode::FPExt:
  case Instruction::Opcode::PtrToAddr:
  case Instruction::Opcode::PtrToInt:
  case Instruction::Opcode::IntToPtr:
  case Instruction::Opcode::SIToFP:
  case Instruction::Opcode::UIToFP:
  case Instruction::Opcode::Trunc:
  case Instruction::Opcode::FPTrunc:
  case Instruction::Opcode::BitCast: {
    // We have already checked that they are of the same opcode.
    assert(all_of(Bndl,
                  [Opcode](Value *V) {
                    return cast<Instruction>(V)->getOpcode() == Opcode;
                  }) &&
           "Different opcodes, should have early returned!");
    // But for these opcodes we should also check the operand type.
    Type *FromTy0 = Utils::getExpectedType(I0->getOperand(0));
    if (any_of(drop_begin(Bndl), [FromTy0](Value *V) {
          return Utils::getExpectedType(cast<User>(V)->getOperand(0)) !=
                 FromTy0;
        }))
      return ResultReason::DiffTypes;
    return std::nullopt;
  }
  case Instruction::Opcode::FCmp:
  case Instruction::Opcode::ICmp: {
    // We need the same predicate..
    auto Pred0 = cast<CmpInst>(I0)->getPredicate();
    bool Same = all_of(Bndl, [Pred0](Value *V) {
      return cast<CmpInst>(V)->getPredicate() == Pred0;
    });
    if (Same)
      return std::nullopt;
    return ResultReason::DiffOpcodes;
  }
  case Instruction::Opcode::Select: {
    auto *Sel0 = cast<SelectInst>(Bndl[0]);
    auto *Cond0 = Sel0->getCondition();
    if (VecUtils::getNumLanes(Cond0) != VecUtils::getNumLanes(Sel0))
      // TODO: For now we don't vectorize if the lanes in the condition don't
      // match those of the select instruction.
      return ResultReason::Unimplemented;
    return std::nullopt;
  }
  case Instruction::Opcode::FNeg:
  case Instruction::Opcode::Add:
  case Instruction::Opcode::FAdd:
  case Instruction::Opcode::Sub:
  case Instruction::Opcode::FSub:
  case Instruction::Opcode::Mul:
  case Instruction::Opcode::FMul:
  case Instruction::Opcode::FRem:
  case Instruction::Opcode::UDiv:
  case Instruction::Opcode::SDiv:
  case Instruction::Opcode::FDiv:
  case Instruction::Opcode::URem:
  case Instruction::Opcode::SRem:
  case Instruction::Opcode::Shl:
  case Instruction::Opcode::LShr:
  case Instruction::Opcode::AShr:
  case Instruction::Opcode::And:
  case Instruction::Opcode::Or:
  case Instruction::Opcode::Xor:
    return std::nullopt;
  case Instruction::Opcode::Load:
    if (VecUtils::areConsecutive<LoadInst>(Bndl, SE, DL))
      return std::nullopt;
    return ResultReason::NotConsecutive;
  case Instruction::Opcode::Store:
    if (VecUtils::areConsecutive<StoreInst>(Bndl, SE, DL))
      return std::nullopt;
    return ResultReason::NotConsecutive;
  case Instruction::Opcode::PHI:
    return ResultReason::Unimplemented;
  case Instruction::Opcode::Opaque:
    return ResultReason::Unimplemented;
  case Instruction::Opcode::Br:
  case Instruction::Opcode::Ret:
  case Instruction::Opcode::AddrSpaceCast:
  case Instruction::Opcode::InsertElement:
  case Instruction::Opcode::InsertValue:
  case Instruction::Opcode::ExtractElement:
  case Instruction::Opcode::ExtractValue:
  case Instruction::Opcode::ShuffleVector:
  case Instruction::Opcode::Call:
  case Instruction::Opcode::GetElementPtr:
  case Instruction::Opcode::Switch:
    return ResultReason::Unimplemented;
  case Instruction::Opcode::VAArg:
  case Instruction::Opcode::Freeze:
  case Instruction::Opcode::Fence:
  case Instruction::Opcode::Invoke:
  case Instruction::Opcode::CallBr:
  case Instruction::Opcode::LandingPad:
  case Instruction::Opcode::CatchPad:
  case Instruction::Opcode::CleanupPad:
  case Instruction::Opcode::CatchRet:
  case Instruction::Opcode::CleanupRet:
  case Instruction::Opcode::Resume:
  case Instruction::Opcode::CatchSwitch:
  case Instruction::Opcode::AtomicRMW:
  case Instruction::Opcode::AtomicCmpXchg:
  case Instruction::Opcode::Alloca:
  case Instruction::Opcode::Unreachable:
    return ResultReason::Infeasible;
  }

  return std::nullopt;
}

CollectDescr
LegalityAnalysis::getHowToCollectValues(ArrayRef<Value *> Bndl) const {
  SmallVector<CollectDescr::ExtractElementDescr, 4> Vec;
  Vec.reserve(Bndl.size());
  for (auto [Elm, V] : enumerate(Bndl)) {
    if (auto *VecOp = IMaps.getVectorForOrig(V)) {
      // If there is a vector containing `V`, then get the lane it came from.
      std::optional<int> ExtractIdxOpt = IMaps.getOrigLane(VecOp, V);
      // This could be a vector, like <2 x float> in which case the mask needs
      // to enumerate all lanes.
      for (unsigned Ln = 0, Lanes = VecUtils::getNumLanes(V); Ln != Lanes; ++Ln)
        Vec.emplace_back(VecOp, ExtractIdxOpt ? *ExtractIdxOpt + Ln : -1);
    } else {
      Vec.emplace_back(V);
    }
  }
  return CollectDescr(std::move(Vec));
}

const LegalityResult &LegalityAnalysis::canVectorize(ArrayRef<Value *> Bndl,
                                                     bool SkipScheduling) {
  // If Bndl contains values other than instructions, we need to Pack.
  if (any_of(Bndl, [](auto *V) { return !isa<Instruction>(V); }))
    return createLegalityResult<Pack>(ResultReason::NotInstructions);
  // Pack if not in the same BB.
  auto *BB = cast<Instruction>(Bndl[0])->getParent();
  if (any_of(drop_begin(Bndl),
             [BB](auto *V) { return cast<Instruction>(V)->getParent() != BB; }))
    return createLegalityResult<Pack>(ResultReason::DiffBBs);
  // Pack if instructions repeat, i.e., require some sort of broadcast.
  SmallPtrSet<Value *, 8> Unique(llvm::from_range, Bndl);
  if (Unique.size() != Bndl.size())
    return createLegalityResult<Pack>(ResultReason::RepeatedInstrs);

  auto CollectDescrs = getHowToCollectValues(Bndl);
  if (CollectDescrs.hasVectorInputs()) {
    if (auto ValueShuffleOpt = CollectDescrs.getSingleInput()) {
      auto [Vec, Mask] = *ValueShuffleOpt;
      if (Mask.isIdentity())
        return createLegalityResult<DiamondReuse>(Vec);
      return createLegalityResult<DiamondReuseWithShuffle>(Vec, Mask);
    }
    return createLegalityResult<DiamondReuseMultiInput>(
        std::move(CollectDescrs));
  }

  if (auto ReasonOpt = notVectorizableBasedOnOpcodesAndTypes(Bndl))
    return createLegalityResult<Pack>(*ReasonOpt);

  if (!SkipScheduling) {
    // TODO: Try to remove the IBndl vector.
    SmallVector<Instruction *, 8> IBndl;
    IBndl.reserve(Bndl.size());
    for (auto *V : Bndl)
      IBndl.push_back(cast<Instruction>(V));
    if (!Sched.trySchedule(IBndl))
      return createLegalityResult<Pack>(ResultReason::CantSchedule);
  }

  return createLegalityResult<Widen>();
}

void LegalityAnalysis::clear() {
  Sched.clear();
  IMaps.clear();
}
} // namespace llvm::sandboxir
