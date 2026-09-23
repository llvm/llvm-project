//===- ProfDataUtils.cpp - Utility functions for MD_prof Metadata ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utilities for working with Profiling Metadata.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/ProfDataUtils.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/xxhash.h"
#include <optional>

using namespace llvm;

namespace llvm {
extern cl::opt<bool> ProfcheckDisableMetadataFixes;
}

static uint64_t getWaveProfileFunctionId(const Function &F) {
  return xxh3_64bits(F.getName());
}

static const MDNode *getWaveProfileTable(const Function &F) {
  SmallVector<MDNode *> Profiles;
  F.getMetadata(LLVMContext::MD_wave_profile, Profiles);
  if (F.isDeclaration() || Profiles.size() != 1)
    return nullptr;
  const MDNode *MD = Profiles.front();
  if (MD->getNumOperands() < 3)
    return nullptr;
  for (const MDOperand &Op : MD->operands()) {
    const auto *CI = mdconst::dyn_extract_or_null<ConstantInt>(Op);
    if (!CI || !CI->getType()->isIntegerTy(64))
      return nullptr;
  }
  if (mdconst::extract<ConstantInt>(MD->getOperand(0))->getZExtValue() != 2 ||
      mdconst::extract<ConstantInt>(MD->getOperand(1))->getZExtValue() !=
          getWaveProfileFunctionId(F))
    return nullptr;
  return MD;
}

namespace {
struct BlockWaveProfile {
  uint64_t FunctionId;
  uint64_t Id;
  bool HasCount;
  SmallVector<uint64_t, 2> Successors;
};
} // namespace

static std::optional<BlockWaveProfile> getBlockWaveProfile(const MDNode *MD) {
  if (!MD || MD->getNumOperands() < 4)
    return std::nullopt;
  for (const MDOperand &Op : MD->operands()) {
    const auto *CI = mdconst::dyn_extract_or_null<ConstantInt>(Op);
    if (!CI || !CI->getType()->isIntegerTy(64))
      return std::nullopt;
  }
  auto GetInt = [&](unsigned I) {
    return mdconst::extract<ConstantInt>(MD->getOperand(I))->getZExtValue();
  };
  if (GetInt(0) != 2 || GetInt(3) > 1)
    return std::nullopt;
  BlockWaveProfile Profile{GetInt(1), GetInt(2), GetInt(3) != 0, {}};
  for (unsigned I = 4, E = MD->getNumOperands(); I != E; ++I)
    Profile.Successors.push_back(GetInt(I));
  return Profile;
}

void llvm::swapBlockWaveCountSuccessors(CondBrInst &BI) {
  MDNode *MD = BI.getMetadata(LLVMContext::MD_wave_profile_block);
  std::optional<BlockWaveProfile> Profile = getBlockWaveProfile(MD);
  if (!Profile || Profile->Successors.size() != 2)
    return;
  SmallVector<Metadata *> Ops(MD->op_begin(), MD->op_end());
  std::swap(Ops[4], Ops[5]);
  BI.setMetadata(LLVMContext::MD_wave_profile_block,
                 MDNode::get(BI.getContext(), Ops));
}

void llvm::clearBlockWaveCounts(Function &F) {
  F.setMetadata(LLVMContext::MD_wave_profile, nullptr);
  for (BasicBlock &BB : F)
    BB.getTerminator()->setMetadata(LLVMContext::MD_wave_profile_block,
                                    nullptr);
}

static Metadata *getWaveProfileInt(LLVMContext &Context, uint64_t Value) {
  return ConstantAsMetadata::get(
      ConstantInt::get(Type::getInt64Ty(Context), Value));
}

static void writeWaveProfile(Function &F, ArrayRef<Metadata *> Table,
                             const DenseMap<const BasicBlock *, unsigned> &Ids,
                             const BitVector &HasCounts) {
  LLVMContext &Context = F.getContext();
  for (auto [Index, BB] : enumerate(F)) {
    SmallVector<Metadata *> Ops{Table[0], Table[1],
                                getWaveProfileInt(Context, Ids.lookup(&BB)),
                                getWaveProfileInt(Context, HasCounts[Index])};
    for (const BasicBlock *Succ : successors(&BB))
      Ops.push_back(getWaveProfileInt(Context, Ids.lookup(Succ)));
    BB.getTerminator()->setMetadata(LLVMContext::MD_wave_profile_block,
                                    MDNode::get(Context, Ops));
  }
  // Every update invalidates older snapshots, even if only block records
  // changed and their terminators are subsequently removed.
  F.setMetadata(LLVMContext::MD_wave_profile,
                MDNode::getDistinct(Context, Table));
}

void llvm::setBlockWaveCounts(Function &F, ArrayRef<uint64_t> Counts) {
  BitVector HasCounts(Counts.size(), true);
  setBlockWaveCounts(F, Counts, HasCounts);
}

void llvm::setBlockWaveCounts(Function &F, ArrayRef<uint64_t> Counts,
                              const BitVector &HasCounts) {
  assert(Counts.size() == F.size() && "one wave counter per IR block");
  assert(HasCounts.size() == F.size() &&
         "one wave-count validity bit per IR block");
  assert(!Counts.empty() && HasCounts.test(0) &&
         "entry wave count must be measured");
  clearBlockWaveCounts(F);

  LLVMContext &Context = F.getContext();
  SmallVector<Metadata *> Table{
      getWaveProfileInt(Context, 2),
      getWaveProfileInt(Context, getWaveProfileFunctionId(F))};
  for (uint64_t Count : Counts)
    Table.push_back(getWaveProfileInt(Context, Count));

  DenseMap<const BasicBlock *, unsigned> Ids;
  for (const BasicBlock &BB : F)
    Ids.try_emplace(&BB, Ids.size());
  writeWaveProfile(F, Table, Ids, HasCounts);
}

bool llvm::extractBlockWaveCounts(const Function &F,
                                  SmallVectorImpl<uint64_t> &Counts,
                                  BitVector *HasCounts) {
  Counts.clear();
  if (HasCounts)
    HasCounts->clear();
  auto Fail = [&]() {
    Counts.clear();
    if (HasCounts)
      HasCounts->clear();
    return false;
  };
  const MDNode *MD = getWaveProfileTable(F);
  if (!MD || MD->getNumOperands() != F.size() + 2)
    return Fail();
  const uint64_t FunctionId =
      mdconst::extract<ConstantInt>(MD->getOperand(1))->getZExtValue();

  SmallVector<const BasicBlock *> BlocksById(F.size());
  SmallVector<BlockWaveProfile> BlockProfiles;
  for (const BasicBlock &BB : F) {
    auto Block = getBlockWaveProfile(
        BB.getTerminator()->getMetadata(LLVMContext::MD_wave_profile_block));
    if (!Block || Block->FunctionId != FunctionId ||
        Block->Id >= BlocksById.size() || BlocksById[Block->Id])
      return Fail();
    BlocksById[Block->Id] = &BB;
    BlockProfiles.push_back(std::move(*Block));
  }
  if (BlocksById[0] != &F.getEntryBlock())
    return Fail();

  BitVector ExtractedHasCounts;
  for (auto [BB, Block] : zip(F, BlockProfiles)) {
    if (Block.Successors.size() != BB.getTerminator()->getNumSuccessors())
      return Fail();
    for (auto [Succ, ExpectedId] : zip(successors(&BB), Block.Successors))
      if (ExpectedId >= BlocksById.size() || BlocksById[ExpectedId] != Succ)
        return Fail();
    Counts.push_back(mdconst::extract<ConstantInt>(MD->getOperand(Block.Id + 2))
                         ->getZExtValue());
    ExtractedHasCounts.push_back(Block.HasCount);
  }
  if (!HasCounts && ExtractedHasCounts.count() != ExtractedHasCounts.size())
    return Fail();
  if (!ExtractedHasCounts.test(0))
    return Fail();
  if (HasCounts)
    *HasCounts = std::move(ExtractedHasCounts);
  return true;
}

namespace {
struct WaveProfileMapping {
  const MDNode *Table;
  SmallVector<unsigned> BlockIds;
  BitVector HasCounts;
  BitVector DuplicateIds;

  WaveProfileMapping(const MDNode *Table, unsigned NumBlocks)
      : Table(Table), BlockIds(NumBlocks, Table->getNumOperands() - 2),
        HasCounts(NumBlocks), DuplicateIds(Table->getNumOperands() - 2) {}
};
} // namespace

static std::optional<WaveProfileMapping> mapBlockWaveCounts(const Function &F) {
  const MDNode *MD = getWaveProfileTable(F);
  if (!MD)
    return std::nullopt;
  const uint64_t FunctionId =
      mdconst::extract<ConstantInt>(MD->getOperand(1))->getZExtValue();

  const unsigned NumProfileBlocks = MD->getNumOperands() - 2;
  WaveProfileMapping Result(MD, F.size());
  SmallVector<std::optional<BlockWaveProfile>> BlockProfiles(F.size());
  SmallVector<const BasicBlock *> BlocksById(NumProfileBlocks);
  DenseMap<const BasicBlock *, unsigned> CurrentIndices;
  for (auto [Index, BB] : enumerate(F)) {
    CurrentIndices.try_emplace(&BB, Index);
    const MDNode *BlockMD =
        BB.getTerminator()->getMetadata(LLVMContext::MD_wave_profile_block);
    if (!BlockMD)
      continue;
    auto Block = getBlockWaveProfile(BlockMD);
    if (!Block || Block->FunctionId != FunctionId ||
        Block->Id >= NumProfileBlocks)
      return std::nullopt;
    unsigned BlockId = Block->Id;
    BlockProfiles[Index] = std::move(Block);
    Result.BlockIds[Index] = BlockId;
    if (BlocksById[BlockId])
      Result.DuplicateIds.set(BlockId);
    else
      BlocksById[BlockId] = &BB;
  }

  for (auto [Index, BB] : enumerate(F)) {
    const auto &Block = BlockProfiles[Index];
    if (!Block || Result.DuplicateIds.test(Result.BlockIds[Index]))
      continue;
    Result.HasCounts[Index] = Block->HasCount;
  }

  // A changed edge makes the source count and both the old and new target
  // counts ambiguous. Invalidate that local neighborhood while retaining
  // independent blocks whose recorded execution event still matches.
  auto Invalidate = [&](const BasicBlock *BB) {
    auto It = CurrentIndices.find(BB);
    if (It != CurrentIndices.end())
      Result.HasCounts.reset(It->second);
  };
  for (auto [Index, BB] : enumerate(F)) {
    const auto &Block = BlockProfiles[Index];
    if (!Block || Result.DuplicateIds.test(Result.BlockIds[Index])) {
      for (const BasicBlock *Succ : successors(&BB))
        Invalidate(Succ);
      continue;
    }

    if (Block->Successors.size() != BB.getTerminator()->getNumSuccessors()) {
      Result.HasCounts.reset(Index);
      for (const BasicBlock *Succ : successors(&BB))
        Invalidate(Succ);
      for (uint64_t OldSuccId : Block->Successors) {
        if (OldSuccId < BlocksById.size() &&
            !Result.DuplicateIds.test(OldSuccId))
          Invalidate(BlocksById[OldSuccId]);
      }
      continue;
    }

    for (auto [Succ, OldSuccId] : zip(successors(&BB), Block->Successors)) {
      auto Current = CurrentIndices.find(Succ);
      unsigned SuccId = Current == CurrentIndices.end()
                            ? NumProfileBlocks
                            : Result.BlockIds[Current->second];
      if (SuccId < NumProfileBlocks && !Result.DuplicateIds.test(SuccId) &&
          OldSuccId == SuccId)
        continue;

      Result.HasCounts.reset(Index);
      Invalidate(Succ);
      if (OldSuccId < BlocksById.size() && !Result.DuplicateIds.test(OldSuccId))
        Invalidate(BlocksById[OldSuccId]);
    }
  }

  if (Result.DuplicateIds.test(0))
    return std::nullopt;
  if (const BasicBlock *OriginalEntry = BlocksById[0]) {
    unsigned EntryIndex = CurrentIndices.lookup(OriginalEntry);
    if (!BlockProfiles[EntryIndex]->HasCount)
      return std::nullopt;
  }
  return Result;
}

bool llvm::extractMappedBlockWaveCounts(const Function &F,
                                        SmallVectorImpl<uint64_t> &Counts,
                                        BitVector &HasCounts,
                                        uint64_t &EntryCount) {
  Counts.clear();
  HasCounts.clear();
  EntryCount = 0;
  auto Mapping = mapBlockWaveCounts(F);
  if (!Mapping)
    return false;

  unsigned NumProfileBlocks = Mapping->Table->getNumOperands() - 2;
  for (unsigned Id : Mapping->BlockIds)
    Counts.push_back(
        Id < NumProfileBlocks && !Mapping->DuplicateIds.test(Id)
            ? mdconst::extract<ConstantInt>(Mapping->Table->getOperand(Id + 2))
                  ->getZExtValue()
            : 0);
  HasCounts = std::move(Mapping->HasCounts);
  EntryCount = mdconst::extract<ConstantInt>(Mapping->Table->getOperand(2))
                   ->getZExtValue();
  return true;
}

// Wave records contain only constants. Save their operands by value because a
// tracking reference follows in-place edits rather than preserving old values.
static bool hasWaveProfileOperands(const MDNode *MD,
                                   ArrayRef<Metadata *> Operands) {
  return MD && equal(MD->operands(), Operands,
                     [](const MDOperand &Op, Metadata *Saved) {
                       return Op.get() == Saved;
                     });
}

BlockWaveCountPreserver::BlockWaveCountPreserver(Function &F) : F(F) {
  auto Mapping = mapBlockWaveCounts(F);
  if (!Mapping)
    return;
  Profile.reset(F.getMetadata(LLVMContext::MD_wave_profile));
  ProfileOperands.assign(Profile->op_begin(), Profile->op_end());
  for (auto [Index, BB] : enumerate(F)) {
    Instruction *Term = BB.getTerminator();
    MDNode *MD = Term->getMetadata(LLVMContext::MD_wave_profile_block);
    SmallVector<Metadata *, 6> Operands;
    if (MD)
      Operands.assign(MD->op_begin(), MD->op_end());
    Blocks.push_back({&BB, Term, TrackingMDNodeRef(MD), std::move(Operands),
                      Mapping->BlockIds[Index], Mapping->HasCounts[Index]});
  }
}

void BlockWaveCountPreserver::invalidate(const BasicBlock &BB) {
  for (BlockProfile &Block : Blocks)
    if (Block.Block == &BB) {
      Block.HasCount = false;
      Invalidated = true;
    }
}

void BlockWaveCountPreserver::restore() {
  TrackingMDNodeRef SavedProfile(std::move(Profile));
  if (!SavedProfile)
    return;
  auto Abandon = [&]() {
    // Skipping reconstruction must not leave invalidated counts readable.
    if (Invalidated)
      clearBlockWaveCounts(F);
  };
  if (getWaveProfileTable(F) != SavedProfile ||
      !hasWaveProfileOperands(SavedProfile, ProfileOperands))
    return Abandon();

  DenseMap<const BasicBlock *, const BlockProfile *> Saved;
  for (const BlockProfile &Block : Blocks) {
    Value *V = Block.Block;
    auto *BB = dyn_cast_or_null<BasicBlock>(V);
    if (!BB || BB->getParent() != &F)
      continue;
    if (Block.Metadata &&
        !hasWaveProfileOperands(Block.Metadata, Block.Operands))
      return Abandon();
    Instruction *Term = BB->getTerminator();
    MDNode *MD = Term->getMetadata(LLVMContext::MD_wave_profile_block);
    // Missing metadata is expected on a replacement terminator, but clearing
    // metadata on a surviving terminator is an independent invalidation.
    if (MD != Block.Metadata && (MD || Term == Block.Terminator))
      return Abandon();
    Saved.try_emplace(BB, &Block);
  }

  SmallVector<Metadata *> Table(SavedProfile->op_begin(),
                                SavedProfile->op_end());
  unsigned NumOriginalIds = Table.size() - 2;
  BitVector UsedIds(NumOriginalIds);
  DenseMap<const BasicBlock *, unsigned> Ids;
  BitVector HasCounts;
  for (const BasicBlock &BB : F) {
    const BlockProfile *Block = Saved.lookup(&BB);
    unsigned Id = Block ? Block->Id : NumOriginalIds;
    // ID zero owns the normalization count. An invalid current block must not
    // turn that anchor into an unmeasured count.
    if (Id >= NumOriginalIds || UsedIds[Id] || (Id == 0 && !Block->HasCount)) {
      Id = Table.size() - 2;
      Table.push_back(getWaveProfileInt(F.getContext(), 0));
    } else {
      UsedIds.set(Id);
    }
    Ids[&BB] = Id;
    HasCounts.push_back(Block && Block->HasCount);
  }

  writeWaveProfile(F, Table, Ids, HasCounts);
}

// MD_prof nodes have the following layout
//
// In general:
// { String name,         Array of i32   }
//
// In terms of Types:
// { MDString,            [i32, i32, ...]}
//
// Concretely for Branch Weights
// { "branch_weights",    [i32 1, i32 10000]}
//
// We maintain some constants here to ensure that we access the branch weights
// correctly, and can change the behavior in the future if the layout changes

// the minimum number of operands for MD_prof nodes with branch weights
static constexpr unsigned MinBWOps = 3;

// the minimum number of operands for MD_prof nodes with value profiles
static constexpr unsigned MinVPOps = 5;

// We may want to add support for other MD_prof types, so provide an abstraction
// for checking the metadata type.
static bool isTargetMD(const MDNode *ProfData, const char *Name,
                       unsigned MinOps) {
  // TODO: This routine may be simplified if MD_prof used an enum instead of a
  // string to differentiate the types of MD_prof nodes.
  if (!ProfData || !Name || MinOps < 2)
    return false;

  unsigned NOps = ProfData->getNumOperands();
  if (NOps < MinOps)
    return false;

  auto *ProfDataName = dyn_cast<MDString>(ProfData->getOperand(0));
  if (!ProfDataName)
    return false;

  return ProfDataName->getString() == Name;
}

template <typename T,
          typename = typename std::enable_if<std::is_arithmetic_v<T>>>
static void extractFromBranchWeightMD(const MDNode *ProfileData,
                                      SmallVectorImpl<T> &Weights) {
  assert(isBranchWeightMD(ProfileData) && "wrong metadata");

  unsigned NOps = ProfileData->getNumOperands();
  unsigned WeightsIdx = getBranchWeightOffset(ProfileData);
  assert(WeightsIdx < NOps && "Weights Index must be less than NOps.");
  Weights.resize(NOps - WeightsIdx);

  for (unsigned Idx = WeightsIdx, E = NOps; Idx != E; ++Idx) {
    ConstantInt *Weight =
        mdconst::dyn_extract<ConstantInt>(ProfileData->getOperand(Idx));
    assert(Weight && "Malformed branch_weight in MD_prof node");
    assert(Weight->getValue().getActiveBits() <= (sizeof(T) * 8) &&
           "Too many bits for MD_prof branch_weight");
    Weights[Idx - WeightsIdx] = Weight->getZExtValue();
  }
}

/// Push the weights right to fit in uint32_t.
SmallVector<uint32_t> llvm::fitWeights(ArrayRef<uint64_t> Weights) {
  SmallVector<uint32_t> Ret;
  Ret.reserve(Weights.size());
  uint64_t Max = *llvm::max_element(Weights);
  if (Max > UINT_MAX) {
    unsigned Offset = 32 - llvm::countl_zero(Max);
    for (const uint64_t &Value : Weights)
      Ret.push_back(static_cast<uint32_t>(Value >> Offset));
  } else {
    append_range(Ret, Weights);
  }
  return Ret;
}

static cl::opt<bool> ElideAllZeroBranchWeights("elide-all-zero-branch-weights",
#if defined(LLVM_ENABLE_PROFCHECK)
                                               cl::init(false)
#else
                                               cl::init(true)
#endif
);
const char *MDProfLabels::BranchWeights = "branch_weights";
const char *MDProfLabels::ExpectedBranchWeights = "expected";
const char *MDProfLabels::ValueProfile = "VP";
const char *MDProfLabels::FunctionEntryCount = "function_entry_count";
const char *MDProfLabels::SyntheticFunctionEntryCount =
    "synthetic_function_entry_count";
const char *MDProfLabels::UnknownBranchWeightsMarker = "unknown";
const char *llvm::LLVMLoopEstimatedTripCount = "llvm.loop.estimated_trip_count";

bool llvm::hasProfMD(const Instruction &I) {
  return I.hasMetadata(LLVMContext::MD_prof);
}

bool llvm::isBranchWeightMD(const MDNode *ProfileData) {
  return isTargetMD(ProfileData, MDProfLabels::BranchWeights, MinBWOps);
}

bool llvm::isValueProfileMD(const MDNode *ProfileData) {
  return isTargetMD(ProfileData, MDProfLabels::ValueProfile, MinVPOps);
}

bool llvm::hasBranchWeightMD(const Instruction &I) {
  auto *ProfileData = I.getMetadata(LLVMContext::MD_prof);
  return isBranchWeightMD(ProfileData);
}

static bool hasCountTypeMD(const Instruction &I) {
  auto *ProfileData = I.getMetadata(LLVMContext::MD_prof);
  // Value profiles record count-type information.
  if (isValueProfileMD(ProfileData))
    return true;
  // Conservatively assume non CallBase instruction only get taken/not-taken
  // branch probability, so not interpret them as count.
  return isa<CallBase>(I) && !isBranchWeightMD(ProfileData);
}

bool llvm::hasValidBranchWeightMD(const Instruction &I) {
  return getValidBranchWeightMDNode(I);
}

bool llvm::hasBranchWeightOrigin(const Instruction &I) {
  auto *ProfileData = I.getMetadata(LLVMContext::MD_prof);
  return hasBranchWeightOrigin(ProfileData);
}

bool llvm::hasBranchWeightOrigin(const MDNode *ProfileData) {
  if (!isBranchWeightMD(ProfileData))
    return false;
  auto *ProfDataName = dyn_cast<MDString>(ProfileData->getOperand(1));
  // NOTE: if we ever have more types of branch weight provenance,
  // we need to check the string value is "expected". For now, we
  // supply a more generic API, and avoid the spurious comparisons.
  assert(ProfDataName == nullptr ||
         ProfDataName->getString() == MDProfLabels::ExpectedBranchWeights);
  return ProfDataName != nullptr;
}

unsigned llvm::getBranchWeightOffset(const MDNode *ProfileData) {
  return hasBranchWeightOrigin(ProfileData) ? 2 : 1;
}

unsigned llvm::getNumBranchWeights(const MDNode &ProfileData) {
  return ProfileData.getNumOperands() - getBranchWeightOffset(&ProfileData);
}

MDNode *llvm::getBranchWeightMDNode(const Instruction &I) {
  auto *ProfileData = I.getMetadata(LLVMContext::MD_prof);
  if (!isBranchWeightMD(ProfileData))
    return nullptr;
  return ProfileData;
}

MDNode *llvm::getValidBranchWeightMDNode(const Instruction &I) {
  auto *ProfileData = getBranchWeightMDNode(I);
  if (ProfileData && getNumBranchWeights(*ProfileData) == I.getNumSuccessors())
    return ProfileData;
  return nullptr;
}

void llvm::extractFromBranchWeightMD32(const MDNode *ProfileData,
                                       SmallVectorImpl<uint32_t> &Weights) {
  extractFromBranchWeightMD(ProfileData, Weights);
}

void llvm::extractFromBranchWeightMD64(const MDNode *ProfileData,
                                       SmallVectorImpl<uint64_t> &Weights) {
  extractFromBranchWeightMD(ProfileData, Weights);
}

bool llvm::extractBranchWeights(const MDNode *ProfileData,
                                SmallVectorImpl<uint32_t> &Weights) {
  if (!isBranchWeightMD(ProfileData))
    return false;
  extractFromBranchWeightMD(ProfileData, Weights);
  return true;
}

bool llvm::extractBranchWeights(const Instruction &I,
                                SmallVectorImpl<uint32_t> &Weights) {
  auto *ProfileData = I.getMetadata(LLVMContext::MD_prof);
  return extractBranchWeights(ProfileData, Weights);
}

bool llvm::extractBranchWeights(const Instruction &I, uint64_t &TrueVal,
                                uint64_t &FalseVal) {
  assert((isa<CondBrInst, SelectInst>(I)) &&
         "Looking for branch weights on something besides CondBr or Select");

  SmallVector<uint32_t, 2> Weights;
  auto *ProfileData = I.getMetadata(LLVMContext::MD_prof);
  if (!extractBranchWeights(ProfileData, Weights))
    return false;

  if (Weights.size() > 2)
    return false;

  TrueVal = Weights[0];
  FalseVal = Weights[1];
  return true;
}

bool llvm::extractProfTotalWeight(const MDNode *ProfileData,
                                  uint64_t &TotalVal) {
  TotalVal = 0;
  if (!ProfileData)
    return false;

  auto *ProfDataName = dyn_cast<MDString>(ProfileData->getOperand(0));
  if (!ProfDataName)
    return false;

  if (ProfDataName->getString() == MDProfLabels::BranchWeights) {
    unsigned Offset = getBranchWeightOffset(ProfileData);
    for (unsigned Idx = Offset; Idx < ProfileData->getNumOperands(); ++Idx) {
      auto *V = mdconst::extract<ConstantInt>(ProfileData->getOperand(Idx));
      TotalVal += V->getValue().getZExtValue();
    }
    return true;
  }

  if (ProfDataName->getString() == MDProfLabels::ValueProfile &&
      ProfileData->getNumOperands() > 3) {
    TotalVal = mdconst::dyn_extract<ConstantInt>(ProfileData->getOperand(2))
                   ->getValue()
                   .getZExtValue();
    return true;
  }
  return false;
}

bool llvm::extractProfTotalWeight(const Instruction &I, uint64_t &TotalVal) {
  return extractProfTotalWeight(I.getMetadata(LLVMContext::MD_prof), TotalVal);
}

void llvm::setExplicitlyUnknownBranchWeights(Instruction &I,
                                             StringRef PassName) {
  MDBuilder MDB(I.getContext());
  I.setMetadata(
      LLVMContext::MD_prof,
      MDNode::get(I.getContext(),
                  {MDB.createString(MDProfLabels::UnknownBranchWeightsMarker),
                   MDB.createString(PassName)}));
}

void llvm::setExplicitlyUnknownBranchWeightsIfProfiled(Instruction &I,
                                                       StringRef PassName,
                                                       const Function *F) {
  F = F ? F : I.getFunction();
  assert(F && "Either pass a instruction attached to a Function, or explicitly "
              "pass the Function that it will be attached to");
  if (std::optional<uint64_t> EC = F->getEntryCount(); EC && *EC > 0)
    setExplicitlyUnknownBranchWeights(I, PassName);
}

MDNode *llvm::getExplicitlyUnknownBranchWeightsIfProfiled(Function &F,
                                                          StringRef PassName) {
  if (std::optional<uint64_t> EC = F.getEntryCount(); !EC || *EC == 0)
    return nullptr;
  MDBuilder MDB(F.getContext());
  return MDNode::get(
      F.getContext(),
      {MDB.createString(MDProfLabels::UnknownBranchWeightsMarker),
       MDB.createString(PassName)});
}

void llvm::setExplicitlyUnknownFunctionEntryCount(Function &F,
                                                  StringRef PassName) {
  MDBuilder MDB(F.getContext());
  F.setMetadata(
      LLVMContext::MD_prof,
      MDNode::get(F.getContext(),
                  {MDB.createString(MDProfLabels::UnknownBranchWeightsMarker),
                   MDB.createString(PassName)}));
}

bool llvm::isExplicitlyUnknownProfileMetadata(const MDNode &MD) {
  if (MD.getNumOperands() != 2)
    return false;
  return MD.getOperand(0).equalsStr(MDProfLabels::UnknownBranchWeightsMarker);
}

bool llvm::hasExplicitlyUnknownBranchWeights(const Instruction &I) {
  auto *MD = I.getMetadata(LLVMContext::MD_prof);
  if (!MD)
    return false;
  return isExplicitlyUnknownProfileMetadata(*MD);
}

void llvm::setBranchWeights(Instruction &I, ArrayRef<uint32_t> Weights,
                            bool IsExpected, bool ElideAllZero) {
  if ((ElideAllZeroBranchWeights && ElideAllZero) &&
      llvm::all_of(Weights, equal_to(0))) {
    I.setMetadata(LLVMContext::MD_prof, nullptr);
    return;
  }

  MDBuilder MDB(I.getContext());
  MDNode *BranchWeights = MDB.createBranchWeights(Weights, IsExpected);
  I.setMetadata(LLVMContext::MD_prof, BranchWeights);
}

void llvm::setFittedBranchWeights(Instruction &I, ArrayRef<uint64_t> Weights,
                                  bool IsExpected, bool ElideAllZero) {
  setBranchWeights(I, fitWeights(Weights), IsExpected, ElideAllZero);
}

SmallVector<uint32_t>
llvm::downscaleWeights(ArrayRef<uint64_t> Weights,
                       std::optional<uint64_t> KnownMaxCount) {
  uint64_t MaxCount = KnownMaxCount.has_value() ? KnownMaxCount.value()
                                                : *llvm::max_element(Weights);
  assert(MaxCount > 0 && "Bad max count");
  uint64_t Scale = calculateCountScale(MaxCount);
  SmallVector<uint32_t> DownscaledWeights;
  for (const auto &ECI : Weights)
    DownscaledWeights.push_back(scaleBranchCount(ECI, Scale));
  return DownscaledWeights;
}

void llvm::scaleProfData(Instruction &I, uint64_t S, uint64_t T) {
  assert(T != 0 && "Caller should guarantee");
  auto *ProfileData = I.getMetadata(LLVMContext::MD_prof);
  if (ProfileData == nullptr)
    return;

  auto *ProfDataName = dyn_cast<MDString>(ProfileData->getOperand(0));
  if (!ProfDataName ||
      (ProfDataName->getString() != MDProfLabels::BranchWeights &&
       ProfDataName->getString() != MDProfLabels::ValueProfile))
    return;

  if (!hasCountTypeMD(I))
    return;

  LLVMContext &C = I.getContext();

  MDBuilder MDB(C);
  SmallVector<Metadata *, 3> Vals;
  Vals.push_back(ProfileData->getOperand(0));
  APInt APS(128, S), APT(128, T);
  if (ProfDataName->getString() == MDProfLabels::BranchWeights &&
      ProfileData->getNumOperands() > 0) {
    // Using APInt::div may be expensive, but most cases should fit 64 bits.
    APInt Val(128,
              mdconst::dyn_extract<ConstantInt>(
                  ProfileData->getOperand(getBranchWeightOffset(ProfileData)))
                  ->getValue()
                  .getZExtValue());
    Val *= APS;
    Vals.push_back(MDB.createConstant(ConstantInt::get(
        Type::getInt32Ty(C), Val.udiv(APT).getLimitedValue(UINT32_MAX))));
  } else if (ProfDataName->getString() == MDProfLabels::ValueProfile)
    for (unsigned Idx = 1; Idx < ProfileData->getNumOperands(); Idx += 2) {
      // The first value is the key of the value profile, which will not change.
      Vals.push_back(ProfileData->getOperand(Idx));
      uint64_t Count =
          mdconst::dyn_extract<ConstantInt>(ProfileData->getOperand(Idx + 1))
              ->getValue()
              .getZExtValue();
      // Don't scale the magic number.
      if (Count == NOMORE_ICP_MAGICNUM) {
        Vals.push_back(ProfileData->getOperand(Idx + 1));
        continue;
      }
      // Using APInt::div may be expensive, but most cases should fit 64 bits.
      APInt Val(128, Count);
      Val *= APS;
      Vals.push_back(MDB.createConstant(ConstantInt::get(
          Type::getInt64Ty(C), Val.udiv(APT).getLimitedValue())));
    }
  I.setMetadata(LLVMContext::MD_prof, MDNode::get(C, Vals));
}

void llvm::applyProfMetadataIfEnabled(
    Value *V, llvm::function_ref<void(Instruction *)> setMetadataCallback) {
  if (!ProfcheckDisableMetadataFixes) {
    if (Instruction *Inst = dyn_cast<Instruction>(V)) {
      setMetadataCallback(Inst);
    }
  }
}
