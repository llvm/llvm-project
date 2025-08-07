#if LLPC_BUILD_NPI
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

// Evaluate the Instruction using constant folding, with Op0 as LHS and (for two
// operand instructions) Op1 as RHS. This is intended to be used on wave-ID and
// its derived/downstream values.
static Constant*
evaluateWaveIDDerivative(const Instruction *Instruction, Constant *Op0,
                         Constant *Op1) {
  Constant *Folded;
  if (auto *Cast = dyn_cast<CastInst>(Instruction))
    Folded =
        ConstantFoldCastInstruction(Cast->getOpcode(), Op0, Cast->getType());
  else if (auto *Cmp = dyn_cast<ICmpInst>(Instruction)) {
    Folded = ConstantFoldCompareInstruction(Cmp->getPredicate(), Op0, Op1);

    // Result of an ICmp should be a binary valued 0 or 1.
    if (ConstantInt *CI = dyn_cast<ConstantInt>(Folded)) {
      int64_t V = CmpInst::isSigned(Cmp->getPredicate()) ? CI->getSExtValue()
                                                         : CI->getZExtValue();
      if (V == 0 || V == 1)
        return CI;
    }
    return {};
  } else if (auto *BinOp = dyn_cast<BinaryOperator>(Instruction))
    Folded = ConstantFoldBinaryInstruction(BinOp->getOpcode(), Op0, Op1);
  else {
    Folded = nullptr;
    llvm_unreachable("Unexpected instruction type during analyzeWaveIDUsers()");
  }

  // During worklist processing, only store if the result is convertible
  // to a ConstantFP or ConstantInt
  if (!isa<ConstantInt, ConstantFP>(Folded))
    return nullptr;
  return Folded;
}

using RankMask = Bitset<MAX_WAVES_PER_WAVEGROUP>;

// Class to help keep track of the value (or derived value) of a wave-ID at an
// arbitrary point in the program.
class RankTracker {
private:
  std::array<Constant *, MAX_WAVES_PER_WAVEGROUP> Values;

public:
  RankTracker() {};
  explicit RankTracker(LLVMContext &Context) {
    for (unsigned i = 0; i < MAX_WAVES_PER_WAVEGROUP; i++)
      Values[i] = ConstantInt::get(Type::getInt32Ty(Context), APInt(32, i));
  }
  explicit RankTracker(Constant *C) { Values.fill(C); }

  // Convert the vector of values to a RankMask. Expects all elements of Values
  // to be 0 or 1.
  RankMask toRankMask() const {
    RankMask RM;
    for (unsigned i = 0; i < MAX_WAVES_PER_WAVEGROUP; i++) {
      if (auto *CI = dyn_cast<ConstantInt>(Values[i])) {
        uint64_t V = CI->getZExtValue();
        assert((V == 0 || V == 1) && "Expected Values to be binary valued "
                                     "when calling toRankMask()");
        if (V)
          RM.set(i);
        else
          RM.reset(i);
      } else
        llvm_unreachable(
            "Expected Values to be integer when calling toRankMask()");
    }
    return RM;
  }

  Constant *&operator[](unsigned I) { return Values[I]; }
};

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

  // The set of I1s/BinaryOps/CastInsts we will replace by constant with when
  // cloning.
  DenseMap<Value *, RankMask> I1Masks;

  // Keep track of the (possibly derived) wave-ID values after processing by
  // BinaryOps and ICmps.
  DenseMap<Value *, RankTracker> DerivedValues;

  // The disjoint set of masks we will create clones from.
  DisjointMaskSet DisjointMasks;

  void analyzeWaveIDUsers();
  void buildSpecializations(Function &Kernel);

public:
  AMDGPURankSpecializationImpl()
      : DisjointMasks(MAX_WAVES_PER_WAVEGROUP) {}
  bool run(Module &M);
};

// Build clones of Kernel and in each, set wave-ID to the pre-determined value
// represented in the disjoint set. After cloning, create an entry kernel which
// routes waves to their corresponding specialization using the intrinsic.
void AMDGPURankSpecializationImpl::buildSpecializations(Function &Kernel) {

  SmallVector<Function *> Specializations;
  ValueToValueMapTy VMap;
  SmallDenseMap<unsigned, Function *> RankToSpecialization;
  for (unsigned i = 0; i != DisjointMasks.size(); ++i) {
    RankMask Mask = DisjointMasks[i];

    VMap.clear();

    // Create a clone function.
    FunctionType *FTy = Kernel.getFunctionType();
    Function *Specialization = Function::Create(
        FTy, Kernel.getLinkage(), /* AddressSpace= */ 0,
        Kernel.getName() + getRankMaskSuffix(Mask), Kernel.getParent());

    Specializations.push_back(Specialization);

    // Loop over the arguments, copying the names of the mapped arguments over...
    Function::arg_iterator DestI = Specialization->arg_begin();
    for (const Argument &I : Kernel.args()) {
      DestI->setName(I.getName()); // Copy the name over...
      VMap[&I] = &*DestI++;        // Add mapping to VMap
    }

    SmallVector<ReturnInst *, 8> Returns; // Ignore returns cloned.
    CloneFunctionInto(Specialization, &Kernel, VMap,
                      CloneFunctionChangeType::GlobalChanges, Returns);

    // Specialization should have wavegroup attr copied from Kernel during
    // CloneFunctionInto.
    assert(Specialization->hasFnAttribute("amdgpu-wavegroup-enable"));
    if (DISubprogram *SP = Specialization->getSubprogram()) {
      assert(SP != Kernel.getSubprogram() && SP->isDistinct());
      MDString *NewLinkageName =
          MDString::get(Kernel.getContext(), Specialization->getName());
      SP->replaceLinkageName(NewLinkageName);
    }

    Specialization->addFnAttr("amdgpu-wavegroup-rank-function");
    Specialization->setVisibility(GlobalValue::DefaultVisibility);
    Specialization->setLinkage(GlobalValue::InternalLinkage);
    Specialization->setDSOLocal(true);

    // Bake our knowledge of the WaveID into the clone.
    Value *CloneWaveID = &*VMap.lookup(WaveID);
    if (auto UniqueRank = isSingleRank(Mask))
      CloneWaveID->replaceAllUsesWith(ConstantInt::get(WaveID->getType(), *UniqueRank));

    for (const auto &Entry : I1Masks) {
      Value *I1Value = &*VMap.lookup(Entry.first);
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

    // Keep track of which rank gets mapped to which specialization.
    for (unsigned r = 0; r < MAX_WAVES_PER_WAVEGROUP; r++)
      if (Mask.test(r))
        RankToSpecialization[r] = Specialization;
  }

  // Kernel was already cloned for each specialization, so clear its body to use
  // as the entry/jump kernel.
  for (BasicBlock &BB : Kernel)
    BB.dropAllReferences();
  while (Kernel.size() > 0)
    Kernel.begin()->eraseFromParent();

  // Create the jump table in the entry kernel.
  IRBuilder<> Builder(Kernel.getContext());
  BasicBlock *Entry = BasicBlock::Create(Kernel.getContext(), "entry", &Kernel);

  Builder.SetInsertPoint(Entry);

  for (unsigned r = 0; r < MAX_WAVES_PER_WAVEGROUP; r++) {
    // Intrinsic handles proper set up of calls to rank funcs.
    auto *Callee = Builder.CreateIntrinsic(
        Builder.getVoidTy(), Intrinsic::amdgcn_wavegroup_rank,
        {ConstantInt::get(Builder.getInt32Ty(), r), RankToSpecialization[r]});

    // Callback metadata is necessary for propagating intrinsic call through
    // call graph.
    auto *WGRFIntrinsic = Callee->getCalledFunction();
    if (!WGRFIntrinsic->hasMetadata(LLVMContext::MD_callback)) {
      LLVMContext &Ctx = WGRFIntrinsic->getContext();
      MDBuilder MDB(Ctx);
      WGRFIntrinsic->addMetadata(
          LLVMContext::MD_callback,
          *MDNode::get(
              Ctx, {MDB.createCallbackEncoding(1, {},
                                               /* VarArgsArePassed */ false)}));
    }
  }

  Builder.CreateRetVoid();
}

// Analyze all users of WaveID to:
//  * compute a map of instruction Value*'s to the derived values of wave-ID
//  * build a set of disjoint masks that will become the specializations
void AMDGPURankSpecializationImpl::analyzeWaveIDUsers() {

  SmallVector<Value *> Worklist;

  // Define a helper to facilitate the folding. If folding fails, return a
  // nullopt to indicate to worklist processing that we should skip that
  // instruction.
  auto tryFold = [&](Instruction *Inst, RankTracker &LHS,
                     RankTracker &RHS) -> std::optional<RankTracker> {
    LLVMContext &Ctx = Inst->getContext();
    RankTracker RT(Ctx);

    for (unsigned i = 0; i < MAX_WAVES_PER_WAVEGROUP; ++i) {
      Constant* PossibleFoldedConstant = evaluateWaveIDDerivative(Inst, LHS[i], RHS[i]);
      if (!PossibleFoldedConstant) // Bail if the fold fails.
        return std::nullopt;
      RT[i] = PossibleFoldedConstant;
    }

    return RT;
  };

  // Switches on wave-ID are simpler than the other cases, handle separately
  // first.
  for (User *U : WaveID->users()) {
    if (auto *Switch = dyn_cast<SwitchInst>(U)) {

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

  // Handle BinaryOps, ICmps, and CastInsts. Only add to DisjointMasks when the
  // chains feed a branch.

  RankTracker Seed(WaveID->getContext());
  DerivedValues.try_emplace(WaveID, Seed);
  Worklist.push_back(WaveID);

  while (!Worklist.empty()) {
    Value *I = Worklist.pop_back_val();

    for (User *U : I->users()) {
      if (ICmpInst *Cmp = dyn_cast<ICmpInst>(U)) {
        if (Constant *C = dyn_cast<Constant>(Cmp->getOperand(1))) {

          Value *Op0 = Cmp->getOperand(0);
          assert(Op0 == I);

          // Evaluate the current Worklist item on the derived values.
          DenseMapIterator<Value *, RankTracker> It0 = DerivedValues.find(Op0);
          RankTracker RT0 = It0->second;

          RankTracker RT1 = RankTracker(C);
          std::optional<RankTracker> PossibleResultRT = tryFold(Cmp, RT0, RT1);
          if (PossibleResultRT.has_value() &&
              DerivedValues.try_emplace(Cmp, PossibleResultRT.value()).second)
            Worklist.push_back(Cmp);
        }

      } else if (BinaryOperator *BinOp = dyn_cast<BinaryOperator>(U)) {
        Value *Op0 = BinOp->getOperand(0);
        Value *Op1 = BinOp->getOperand(1);

        Constant *Op1Constant = dyn_cast<Constant>(Op1);

        // Check if we know the masks for both operands.
        auto It0 = DerivedValues.find(Op0);
        auto It1 = DerivedValues.find(Op1);

        // If It1 is a Constant, we can still evaluate that on each wave-ID.
        if (It0 == DerivedValues.end() ||
            (It1 == DerivedValues.end() && !Op1Constant))
          continue;

        RankTracker RT0 = It0->second;
        RankTracker RT1 = Op1Constant ? RankTracker(Op1Constant) : It1->second;

        std::optional<RankTracker> PossibleResultRT = tryFold(BinOp, RT0, RT1);
        if (PossibleResultRT.has_value() &&
            DerivedValues.try_emplace(BinOp, PossibleResultRT.value()).second)
          Worklist.push_back(BinOp);

      } else if (CastInst *Cast = dyn_cast<CastInst>(U)) {

        Value *Op0 = Cast->getOperand(0);
        auto It0 = DerivedValues.find(Op0);

        // CastInsts only have 1 src operand so it should already by in
        // DerivedValues.
        if (It0 == DerivedValues.end())
          continue;

        RankTracker RT0 = It0->second;
        RankTracker RT1 = RankTracker(Cast->getContext()); // Used as a dummy.

        std::optional<RankTracker> PossibleResultRT = tryFold(Cast, RT0, RT1);
        if (PossibleResultRT.has_value() &&
            DerivedValues.try_emplace(Cast, PossibleResultRT.value()).second)
          Worklist.push_back(Cast);

      } else if (BranchInst *Br = dyn_cast<BranchInst>(U)) {
        // If I is feeding a branch, it must be an I1 and we can convert it to a
        // RankMask.
        if (Br->isConditional() && Br->getCondition() == I) {
          RankMask RM = DerivedValues[I].toRankMask();
          I1Masks[I] = RM;
          DisjointMasks.uncross(RM);
          DisjointMasks.uncross(~RM);
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
#endif /* LLPC_BUILD_NPI */
