//===- AMDGPURankSpecialization.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass clones and specializes regions of a kernel's CFG that are
// predicated on the current wave's rank. By specializing these regions, the
// compiler trades a modest code-size increase for a CFG that is more amenable
// to later optimizations such as LoopOpts, unrolling, and vectorization.
// The pass is a wavegroup mode only feature and is injected just before the
// LoopOpts pipeline.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/Bitset.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/InitializePasses.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/LoopUtils.h"

#define DEBUG_TYPE "amdgpu-rank-specialization"

using namespace llvm;

namespace {

// Evaluate an ICmp predicate between a concrete LHS and RHS.
// Returns true if `LHS pred RHS` holds, unsigned predicates are
// interpreted on the bit-width of `LHS` and `RHS`.
static bool evaluateCmp(CmpInst::Predicate Pred, int64_t LHS, int64_t RHS) {
  switch (Pred) {
  case CmpInst::ICMP_EQ:
    return LHS == RHS;
  case CmpInst::ICMP_NE:
    return LHS != RHS;
  case CmpInst::ICMP_UGT:
    return (uint64_t)LHS > (uint64_t)RHS;
  case CmpInst::ICMP_UGE:
    return (uint64_t)LHS >= (uint64_t)RHS;
  case CmpInst::ICMP_ULT:
    return (uint64_t)LHS < (uint64_t)RHS;
  case CmpInst::ICMP_ULE:
    return (uint64_t)LHS <= (uint64_t)RHS;
  case CmpInst::ICMP_SGT:
    return LHS > RHS;
  case CmpInst::ICMP_SGE:
    return LHS >= RHS;
  case CmpInst::ICMP_SLT:
    return LHS < RHS;
  case CmpInst::ICMP_SLE:
    return LHS <= RHS;
  default:
    llvm_unreachable("Unsupported ICmp predicate.");
  }
}

// Represents a set of wave-IDs for specialization.
using RankMask = Bitset<MAX_WAVES_PER_WAVEGROUP>;

static std::string getRankMaskSuffix(RankMask Mask) {
  std::string S;
  llvm::raw_string_ostream OS(S);
  OS << ".rank";
  for (unsigned I = 0, E = Mask.size(); I != E; ++I) {
    if (Mask[I])
      OS << '_' << I;
  }
  return S;
}

// Return the first (lowest) rank in the mask, if any.
static std::optional<unsigned> getFirstRank(RankMask Mask) {
  for (unsigned I = 0, E = Mask.size(); I != E; ++I) {
    if (Mask[I])
      return I; // Return the single rank.
  }
  return {};
}

// Check if the mask contains a single rank and return it if so.
static std::optional<unsigned> isSingleRank(RankMask Mask) {
  if (Mask.count() == 1)
    return getFirstRank(Mask);
  return {};
}

// Ensure that there is at most one wave-ID query, and return that.
Value *canonicalizeWaveID(Function &F) {
  Value *WaveID = nullptr;
  for (auto &BB : F)
    for (auto &I : BB)
      if (auto *CI = dyn_cast<CallInst>(&I))
        if (CI->getCalledFunction() &&
            CI->getCalledFunction()->getIntrinsicID() ==
                Intrinsic::amdgcn_wave_id_in_wavegroup) {
          if (WaveID == nullptr)
            WaveID = CI;
          else
            return nullptr;
        }
  return WaveID;
}

class DisjointMaskSet {
  // Masks represents the set of disjoint masks.
  SmallVector<RankMask> Masks;

public:
  explicit DisjointMaskSet(unsigned NumRanks) {
    RankMask M;
    for (unsigned i = 0; i < NumRanks; ++i)
      M.set(i);
    Masks.push_back(M);
  }

  // Split the masks of the set such that every mask is either a subset of
  // NewMask or disjoint from it.
  //
  // In other words, there is no mask that "crosses" NewMask.
  void uncross(RankMask NewMask) {
    assert(NewMask.any());

    // Check against all existing masks and split them if they partially
    // intersect with NewMask. We don't need to re-check any of the newly
    // created masks, because they don't cross NewMask by definition.
    for (unsigned I = 0, End = Masks.size(); I != End; ++I) {
      auto Intersection = Masks[I] & NewMask;
      if (!Intersection.any())
        continue;

      if (Intersection != Masks[I]) {
        // Keep the part of Masks[I] which doesn't intersect, split the part
        // that does intersect and add to end of Masks.
        Masks[I] &= ~NewMask;
        Masks.push_back(Intersection);
      }

      // Try remaining bits of NewMask that weren't in Intersection on remaining
      // masks.
      NewMask &= ~Intersection;
      if (!NewMask.any())
        break; // No need to keep going.
    }
  }

  size_t size() const { return Masks.size(); }

  const RankMask &operator[](unsigned I) const { return Masks[I]; }
};

class AMDGPURankSpecializationImpl {
  Value *WaveID = nullptr;

  // The set of I1s/BinaryOps we will replace by constant with when cloning.
  DenseMap<Value *, RankMask> I1Masks;

  // The disjoint set of masks we will create clones from.
  DisjointMaskSet DisjointMasks;

  void analyzeWaveIDUsers();
  void buildSpecializations(Function &Kernel);

public:
  AMDGPURankSpecializationImpl()
      : DisjointMasks(MAX_WAVES_PER_WAVEGROUP) {}
  bool run(Module &M);
};

void AMDGPURankSpecializationImpl::buildSpecializations(Function &Kernel) {
  // Create the clones
  SmallVector<Function *> Specializations;
  ValueToValueMapTy VMap;

  for (unsigned i = 0; i != DisjointMasks.size(); ++i) {
    RankMask Mask = DisjointMasks[i];

    VMap.clear();

    // Create a clone function.
    FunctionType *FTy = Kernel.getFunctionType();
    Function *Specialization = Function::Create(
        FTy, Kernel.getLinkage(), /* AddressSpace= */ 0,
        Kernel.getName() + getRankMaskSuffix(Mask), Kernel.getParent());

    Specializations.push_back(Specialization);

    Specialization->copyAttributesFrom(&Kernel);
    Specialization->copyMetadata(&Kernel, 0);
    Specialization->setVisibility(GlobalValue::DefaultVisibility);
    Specialization->setLinkage(GlobalValue::InternalLinkage);
    Specialization->setDSOLocal(true); // for internal linkage
    Specialization->addFnAttr("amdgpu-wavegroup-enable");
    Specialization->addFnAttr("amdgpu-wavegroup-rank-function");

    // Loop over the arguments, copying the names of the mapped arguments over...
    Function::arg_iterator DestI = Specialization->arg_begin();
    for (const Argument &I : Kernel.args()) {
      DestI->setName(I.getName()); // Copy the name over...
      VMap[&I] = &*DestI++;        // Add mapping to VMap
    }

    const bool IsClone = i + 1 < DisjointMasks.size();
    if (IsClone) {
      SmallVector<ReturnInst *, 8> Returns; // Ignore returns cloned.
      CloneFunctionBodyInto(*Specialization, Kernel, VMap, RF_NoModuleLevelChanges,
                            Returns);
    } else {
      // For the last clone, we can just move the contents of the original function.
      Specialization->splice(Specialization->begin(), &Kernel);

      SmallVector<BasicBlock *> BBs;
      // Remap instructions in Specialization to point at new values.
      for (BasicBlock &BB : *Specialization)
        BBs.push_back(&BB);
      remapInstructionsInBlocks(BBs, VMap);
    }

    // Bake our knowledge of the WaveID into the clone.
    Value *CloneWaveID = IsClone ? &*VMap.lookup(WaveID) : WaveID;
    if (auto UniqueRank = isSingleRank(Mask))
      CloneWaveID->replaceAllUsesWith(ConstantInt::get(WaveID->getType(), *UniqueRank));

    for (const auto &Entry : I1Masks) {
      Value *I1Value = IsClone ? &*VMap.lookup(Entry.first) : Entry.first;
      bool Value = (Entry.second & Mask).any();
      if (Value && (Entry.second & Mask) != Mask)
        continue;
      I1Value->replaceAllUsesWith(ConstantInt::get(I1Value->getType(), Value));
    }

    for (BasicBlock &BB : *Specialization) {
      auto *Switch = dyn_cast<SwitchInst>(BB.getTerminator());
      if (!Switch)
        continue;

      if (Switch->getCondition() == CloneWaveID)
        Switch->setCondition(ConstantInt::get(WaveID->getType(), *getFirstRank(Mask)));
    }
  }

  // Create the jump table in the entry kernel.
  IRBuilder<> Builder(Kernel.getContext());
  BasicBlock *Entry = BasicBlock::Create(Kernel.getContext(), "entry", &Kernel);
  SmallVector<BasicBlock *> Cases;
  SmallVector<Value *> Args;
  for (Value &Arg : Kernel.args())
    Args.push_back(&Arg);

  for (unsigned i = 0; i != DisjointMasks.size(); ++i) {
    RankMask Mask = DisjointMasks[i];
    BasicBlock *BB = BasicBlock::Create(
        Kernel.getContext(), "bb" + getRankMaskSuffix(Mask), &Kernel);
    Builder.SetInsertPoint(BB);

    for (unsigned Rank = 0; Rank < MAX_WAVES_PER_WAVEGROUP; Rank++)
      if (Mask[Rank]) {
        // Intrinsic handles proper set up of calls to rank funcs.
        auto *Callee = Builder.CreateIntrinsic(
            Builder.getVoidTy(), Intrinsic::amdgcn_wavegroup_rank,
            {ConstantInt::get(Builder.getInt32Ty(), Rank), Specializations[i]});

        // Callback metadata is necessary for propagating intrinsic call through
        // call graph.
        auto *WGRFIntrinsic = Callee->getCalledFunction();
        if (!WGRFIntrinsic->hasMetadata(LLVMContext::MD_callback)) {
          LLVMContext &Ctx = WGRFIntrinsic->getContext();
          MDBuilder MDB(Ctx);
          WGRFIntrinsic->addMetadata(
              LLVMContext::MD_callback,
              *MDNode::get(Ctx, {MDB.createCallbackEncoding(
                                    1, {},
                                    /* VarArgsArePassed */ false)}));
        }
      }
    Builder.CreateRetVoid();
    Cases.push_back(BB);
  }

  Builder.SetInsertPoint(Entry);
  WaveID = Builder.CreateIntrinsic(Intrinsic::amdgcn_wave_id_in_wavegroup, {});
  SwitchInst *Switch = Builder.CreateSwitch(WaveID, Cases[0], MAX_WAVES_PER_WAVEGROUP);

  for (unsigned i = 0; i != DisjointMasks.size(); ++i) {
    RankMask Mask = DisjointMasks[i];
    for (int Rank = 0; Rank != MAX_WAVES_PER_WAVEGROUP; ++Rank) {
      if (Mask[Rank])
        Switch->addCase(ConstantInt::get(Builder.getInt32Ty(), Rank), Cases[i]);
    }
  }
}

// Analyze all users of WaveID to:
//  * compute a map from i1 values to the mask of wave IDs for which they are
//    true, for values that are fully determined by the wave ID;
//  * build a set of disjoint masks that will become the specializations
void AMDGPURankSpecializationImpl::analyzeWaveIDUsers() {
  // Search through users of wave-ID, and log all instances where wave-ID
  // directly feeds a switch or directly feeds the condition of a conditional
  // branch.
  SmallVector<Value *> I1Worklist;

  for (User *U : WaveID->users()) {
    if (auto *Cmp = dyn_cast<ICmpInst>(U)) {
      if (auto *CI = dyn_cast<ConstantInt>(Cmp->getOperand(1))) {
        assert(Cmp->getOperand(0) == WaveID);
        RankMask M;
        int64_t C = CI->getSExtValue();
        // Evaluate the predicate for each possible wave-index.
        for (unsigned i = 0; i < MAX_WAVES_PER_WAVEGROUP; ++i) {
          if (evaluateCmp(Cmp->getPredicate(), /*LHS=*/i, /*RHS=*/C))
            M.set(i);
          else
            M.reset(i);
        }
        I1Masks.try_emplace(Cmp, M);

        I1Worklist.push_back(Cmp);
      }
    } else if (auto *Switch = dyn_cast<SwitchInst>(U)) {
      // Build a map from possible destination basic blocks to masks of ranks
      // that branch there.
      MapVector<BasicBlock *, RankMask> DestMasks;
      RankMask DefaultMask = ~RankMask();

      for (auto &Case : Switch->cases()) {
        uint64_t Val = Case.getCaseValue()->getZExtValue();
        if (Val >= MAX_WAVES_PER_WAVEGROUP)
          continue;

        DestMasks[Case.getCaseSuccessor()].set(Val);
        DefaultMask.reset(Val);
      }

      DestMasks[Switch->getDefaultDest()] |= DefaultMask;

      for (const auto &Entry : DestMasks) {
        DisjointMasks.uncross(Entry.second);
        DisjointMasks.uncross(~Entry.second);
      }
    }
  }

  while (!I1Worklist.empty()) {
    Value *I1 = I1Worklist.pop_back_val();

    for (User *U : I1->users()) {
      if (auto *BinOp = dyn_cast<BinaryOperator>(U)) {
        Value *Op0 = BinOp->getOperand(0);
        Value *Op1 = BinOp->getOperand(1);

        // Check if we know the masks for both operands
        auto It0 = I1Masks.find(Op0);
        auto It1 = I1Masks.find(Op1);
        if (It0 == I1Masks.end() || It1 == I1Masks.end())
          continue;

        RankMask Op0Mask = It0->second;
        RankMask Op1Mask = It1->second;
        RankMask ResultMask;

        switch (BinOp->getOpcode()) {
        case Instruction::And:
          ResultMask = Op0Mask & Op1Mask;
          break;
        case Instruction::Or:
          ResultMask = Op0Mask | Op1Mask;
          break;
        case Instruction::Xor:
          ResultMask = Op0Mask ^ Op1Mask;
          break;
        default:
          continue; // Skip other binary operations
        }

        // Only add to worklist if this is a new i1 value we haven't seen
        if (I1Masks.try_emplace(BinOp, ResultMask).second) {
          I1Worklist.push_back(BinOp);
        }
      } else if (auto *Br = dyn_cast<BranchInst>(U)) {
        if (Br->isConditional() && Br->getCondition() == I1) {
          DisjointMasks.uncross(I1Masks[I1]);
          DisjointMasks.uncross(~I1Masks[I1]);
        }
      }
    }
  }
}

bool AMDGPURankSpecializationImpl::run(Module &M) {
  SmallVector<Function *> Kernels;
  for (Function &F : M.functions()) {
    if (AMDGPU::getWavegroupEnable(F) && AMDGPU::getRankSpecializationEnable(F))
      Kernels.push_back(&F);
  }

  bool Changed = false;

  for (Function *F : Kernels) {
    WaveID = canonicalizeWaveID(*F);
    if (!WaveID)
      continue;

    DisjointMasks = DisjointMaskSet(MAX_WAVES_PER_WAVEGROUP);
    I1Masks.clear();
    analyzeWaveIDUsers();

    if (DisjointMasks.size() == 1)
      continue; // No specialization needed

    buildSpecializations(*F);
    Changed = true;
  }

  return Changed;
}

} // namespace

PreservedAnalyses
AMDGPURankSpecializationPass::run(Module &M, ModuleAnalysisManager &MAM) {
  return AMDGPURankSpecializationImpl().run(M)
             ? PreservedAnalyses::none()
             : PreservedAnalyses::all();
}
