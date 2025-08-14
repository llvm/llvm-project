//===- JumpTableToSwitch.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/JumpTableToSwitch.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/CtxProfAnalysis.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/ProfDataUtils.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include <limits>

using namespace llvm;

static cl::opt<unsigned>
    JumpTableSizeThreshold("jump-table-to-switch-size-threshold", cl::Hidden,
                           cl::desc("Only split jump tables with size less or "
                                    "equal than JumpTableSizeThreshold."),
                           cl::init(10));

// TODO: Consider adding a cost model for profitability analysis of this
// transformation. Currently we replace a jump table with a switch if all the
// functions in the jump table are smaller than the provided threshold.
static cl::opt<unsigned> FunctionSizeThreshold(
    "jump-table-to-switch-function-size-threshold", cl::Hidden,
    cl::desc("Only split jump tables containing functions whose sizes are less "
             "or equal than this threshold."),
    cl::init(50));

extern cl::opt<bool> ProfcheckDisableMetadataFixes;

#define DEBUG_TYPE "jump-table-to-switch"

namespace {
struct JumpTableTy {
  Value *Index;
  SmallVector<Function *, 10> Funcs;
};
} // anonymous namespace

static std::optional<JumpTableTy> parseJumpTable(GetElementPtrInst *GEP,
                                                 PointerType *PtrTy) {
  Constant *Ptr = dyn_cast<Constant>(GEP->getPointerOperand());
  if (!Ptr)
    return std::nullopt;

  GlobalVariable *GV = dyn_cast<GlobalVariable>(Ptr);
  if (!GV || !GV->isConstant() || !GV->hasDefinitiveInitializer())
    return std::nullopt;

  Function &F = *GEP->getParent()->getParent();
  const DataLayout &DL = F.getDataLayout();
  const unsigned BitWidth =
      DL.getIndexSizeInBits(GEP->getPointerAddressSpace());
  SmallMapVector<Value *, APInt, 4> VariableOffsets;
  APInt ConstantOffset(BitWidth, 0);
  if (!GEP->collectOffset(DL, BitWidth, VariableOffsets, ConstantOffset))
    return std::nullopt;
  if (VariableOffsets.size() != 1)
    return std::nullopt;
  // TODO: consider supporting more general patterns
  if (!ConstantOffset.isZero())
    return std::nullopt;
  APInt StrideBytes = VariableOffsets.front().second;
  const uint64_t JumpTableSizeBytes = DL.getTypeAllocSize(GV->getValueType());
  if (JumpTableSizeBytes % StrideBytes.getZExtValue() != 0)
    return std::nullopt;
  const uint64_t N = JumpTableSizeBytes / StrideBytes.getZExtValue();
  if (N > JumpTableSizeThreshold)
    return std::nullopt;

  JumpTableTy JumpTable;
  JumpTable.Index = VariableOffsets.front().first;
  JumpTable.Funcs.reserve(N);
  for (uint64_t Index = 0; Index < N; ++Index) {
    // ConstantOffset is zero.
    APInt Offset = Index * StrideBytes;
    Constant *C =
        ConstantFoldLoadFromConst(GV->getInitializer(), PtrTy, Offset, DL);
    auto *Func = dyn_cast_or_null<Function>(C);
    if (!Func || Func->isDeclaration() ||
        Func->getInstructionCount() > FunctionSizeThreshold)
      return std::nullopt;
    JumpTable.Funcs.push_back(Func);
  }
  return JumpTable;
}

static BasicBlock *
expandToSwitch(CallBase *CB, const JumpTableTy &JT, DomTreeUpdater &DTU,
               OptimizationRemarkEmitter &ORE,
               llvm::function_ref<GlobalValue::GUID(const Function &)>
                   GetGuidForFunction) {
  const bool IsVoid = CB->getType() == Type::getVoidTy(CB->getContext());

  SmallVector<DominatorTree::UpdateType, 8> DTUpdates;
  BasicBlock *BB = CB->getParent();
  BasicBlock *Tail = SplitBlock(BB, CB, &DTU, nullptr, nullptr,
                                BB->getName() + Twine(".tail"));
  DTUpdates.push_back({DominatorTree::Delete, BB, Tail});
  BB->getTerminator()->eraseFromParent();

  Function &F = *BB->getParent();
  BasicBlock *BBUnreachable = BasicBlock::Create(
      F.getContext(), "default.switch.case.unreachable", &F, Tail);
  IRBuilder<> BuilderUnreachable(BBUnreachable);
  BuilderUnreachable.CreateUnreachable();

  IRBuilder<> Builder(BB);
  SwitchInst *Switch = Builder.CreateSwitch(JT.Index, BBUnreachable);
  DTUpdates.push_back({DominatorTree::Insert, BB, BBUnreachable});

  IRBuilder<> BuilderTail(CB);
  PHINode *PHI =
      IsVoid ? nullptr : BuilderTail.CreatePHI(CB->getType(), JT.Funcs.size());
  const auto *ProfMD = CB->getMetadata(LLVMContext::MD_prof);

  SmallVector<uint64_t> BranchWeights;
  DenseMap<GlobalValue::GUID, uint64_t> GuidToCounter;
  const bool HadProfile = isValueProfileMD(ProfMD);
  if (HadProfile) {
    // The assumptions, coming in, are that the functions in JT.Funcs are
    // defined in this module (from parseJumpTable).
    assert(llvm::all_of(
        JT.Funcs, [](const Function *F) { return F && !F->isDeclaration(); }));
    BranchWeights.reserve(JT.Funcs.size() + 1);
    // The first is the default target, which is the unreachable block created
    // above.
    BranchWeights.push_back(0U);
    uint64_t TotalCount = 0;
    auto Targets = getValueProfDataFromInst(
        *CB, InstrProfValueKind::IPVK_IndirectCallTarget,
        std::numeric_limits<uint32_t>::max(), TotalCount);

    for (const auto &[G, C] : Targets) {
      [[maybe_unused]] auto It = GuidToCounter.insert({G, C});
      assert(It.second);
    }
  }
  for (auto [Index, Func] : llvm::enumerate(JT.Funcs)) {
    BasicBlock *B = BasicBlock::Create(Func->getContext(),
                                       "call." + Twine(Index), &F, Tail);
    DTUpdates.push_back({DominatorTree::Insert, BB, B});
    DTUpdates.push_back({DominatorTree::Insert, B, Tail});

    CallBase *Call = cast<CallBase>(CB->clone());
    Call->setCalledFunction(Func);
    Call->insertInto(B, B->end());
    Switch->addCase(
        cast<ConstantInt>(ConstantInt::get(JT.Index->getType(), Index)), B);
    GlobalValue::GUID FctID = GetGuidForFunction(*Func);
    // It'd be OK to _not_ find target functions in GuidToCounter, e.g. suppose
    // just some of the jump targets are taken (for the given profile).
    BranchWeights.push_back(FctID == 0U ? 0U
                                        : GuidToCounter.lookup_or(FctID, 0U));
    BranchInst::Create(Tail, B);
    if (PHI)
      PHI->addIncoming(Call, B);
  }
  DTU.applyUpdates(DTUpdates);
  ORE.emit([&]() {
    return OptimizationRemark(DEBUG_TYPE, "ReplacedJumpTableWithSwitch", CB)
           << "expanded indirect call into switch";
  });
  if (HadProfile && !ProfcheckDisableMetadataFixes) {
    // At least one of the targets must've been taken.
    assert(llvm::any_of(BranchWeights, [](uint64_t V) { return V != 0; }));
    // FIXME: this duplicates logic in instrumentation. Note: since there's at
    // least a nonzero and these are unsigned values, it follows MaxBW != 0.
    uint64_t MaxBW = *llvm::max_element(BranchWeights);
    SmallVector<uint32_t> ScaledBranchWeights(
        llvm::map_range(BranchWeights, [MaxBW](uint64_t V) {
          return static_cast<uint32_t>(V / MaxBW);
        }));
    setBranchWeights(*Switch, ScaledBranchWeights, /*IsExpected=*/false);
  } else
    setExplicitlyUnknownBranchWeights(*Switch);
  if (PHI)
    CB->replaceAllUsesWith(PHI);
  CB->eraseFromParent();
  return Tail;
}

PreservedAnalyses JumpTableToSwitchPass::run(Function &F,
                                             FunctionAnalysisManager &AM) {
  OptimizationRemarkEmitter &ORE =
      AM.getResult<OptimizationRemarkEmitterAnalysis>(F);
  DominatorTree *DT = AM.getCachedResult<DominatorTreeAnalysis>(F);
  PostDominatorTree *PDT = AM.getCachedResult<PostDominatorTreeAnalysis>(F);
  DomTreeUpdater DTU(DT, PDT, DomTreeUpdater::UpdateStrategy::Lazy);
  bool Changed = false;
  InstrProfSymtab Symtab;
  if (auto E = Symtab.create(*F.getParent()))
    F.getContext().emitError(
        "Could not create indirect call table, likely corrupted IR" +
        toString(std::move(E)));
  DenseMap<const Function *, GlobalValue::GUID> FToGuid;
  for (const auto &[G, FPtr] : Symtab.getIDToNameMap())
    FToGuid.insert({FPtr, G});

  for (BasicBlock &BB : make_early_inc_range(F)) {
    BasicBlock *CurrentBB = &BB;
    while (CurrentBB) {
      BasicBlock *SplittedOutTail = nullptr;
      for (Instruction &I : make_early_inc_range(*CurrentBB)) {
        auto *Call = dyn_cast<CallInst>(&I);
        if (!Call || Call->getCalledFunction() || Call->isMustTailCall())
          continue;
        auto *L = dyn_cast<LoadInst>(Call->getCalledOperand());
        // Skip atomic or volatile loads.
        if (!L || !L->isSimple())
          continue;
        auto *GEP = dyn_cast<GetElementPtrInst>(L->getPointerOperand());
        if (!GEP)
          continue;
        auto *PtrTy = dyn_cast<PointerType>(L->getType());
        assert(PtrTy && "call operand must be a pointer");
        std::optional<JumpTableTy> JumpTable = parseJumpTable(GEP, PtrTy);
        if (!JumpTable)
          continue;
        SplittedOutTail = expandToSwitch(
            Call, *JumpTable, DTU, ORE, [&](const Function &Fct) {
              if (Fct.getMetadata(AssignGUIDPass::GUIDMetadataName))
                return AssignGUIDPass::getGUID(Fct);
              return FToGuid.lookup_or(&Fct, 0U);
            });
        Changed = true;
        break;
      }
      CurrentBB = SplittedOutTail ? SplittedOutTail : nullptr;
    }
  }

  if (!Changed)
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  if (DT)
    PA.preserve<DominatorTreeAnalysis>();
  if (PDT)
    PA.preserve<PostDominatorTreeAnalysis>();
  return PA;
}
