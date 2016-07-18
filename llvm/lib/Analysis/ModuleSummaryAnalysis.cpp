//===- ModuleSummaryAnalysis.cpp - Module summary index builder -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass builds a ModuleSummaryIndex object for the module, to be written
// to bitcode or LLVM assembly.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/ModuleSummaryAnalysis.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/BlockFrequencyInfoImpl.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/IndirectCallPromotionAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/ValueSymbolTable.h"
#include "llvm/Pass.h"
using namespace llvm;

#define DEBUG_TYPE "module-summary-analysis"

// Walk through the operands of a given User via worklist iteration and populate
// the set of GlobalValue references encountered. Invoked either on an
// Instruction or a GlobalVariable (which walks its initializer).
static void findRefEdges(const User *CurUser, DenseSet<const Value *> &RefEdges,
                         SmallPtrSet<const User *, 8> &Visited) {
  SmallVector<const User *, 32> Worklist;
  Worklist.push_back(CurUser);

  while (!Worklist.empty()) {
    const User *U = Worklist.pop_back_val();

    if (!Visited.insert(U).second)
      continue;

    ImmutableCallSite CS(U);

    for (const auto &OI : U->operands()) {
      const User *Operand = dyn_cast<User>(OI);
      if (!Operand)
        continue;
      if (isa<BlockAddress>(Operand))
        continue;
      if (isa<GlobalValue>(Operand)) {
        // We have a reference to a global value. This should be added to
        // the reference set unless it is a callee. Callees are handled
        // specially by WriteFunction and are added to a separate list.
        if (!(CS && CS.isCallee(&OI)))
          RefEdges.insert(Operand);
        continue;
      }
      Worklist.push_back(Operand);
    }
  }
}

void ModuleSummaryIndexBuilder::computeFunctionSummary(
    const Function &F, BlockFrequencyInfo *BFI) {
  // Summary not currently supported for anonymous functions, they must
  // be renamed.
  if (!F.hasName())
    return;

  unsigned NumInsts = 0;
  // Map from callee ValueId to profile count. Used to accumulate profile
  // counts for all static calls to a given callee.
  DenseMap<const Value *, CalleeInfo> CallGraphEdges;
  DenseMap<GlobalValue::GUID, CalleeInfo> IndirectCallEdges;
  DenseSet<const Value *> RefEdges;
  ICallPromotionAnalysis ICallAnalysis;

  SmallPtrSet<const User *, 8> Visited;
  for (const BasicBlock &BB : F)
    for (const Instruction &I : BB) {
      if (!isa<DbgInfoIntrinsic>(I))
        ++NumInsts;

      if (auto CS = ImmutableCallSite(&I)) {
        auto *CalledFunction = CS.getCalledFunction();
        // Check if this is a direct call to a known function.
        if (CalledFunction) {
          if (CalledFunction->hasName() && !CalledFunction->isIntrinsic()) {
            auto ScaledCount = BFI ? BFI->getBlockProfileCount(&BB) : None;
            auto *CalleeId =
                M->getValueSymbolTable().lookup(CalledFunction->getName());
            CallGraphEdges[CalleeId] +=
                (ScaledCount ? ScaledCount.getValue() : 0);
          }
        } else {
          // Otherwise, check for an indirect call (call to a non-const value
          // that isn't an inline assembly call).
          const CallInst *CI = dyn_cast<CallInst>(&I);
          if (CS.getCalledValue() && !isa<Constant>(CS.getCalledValue()) &&
              !(CI && CI->isInlineAsm())) {
            uint32_t NumVals, NumCandidates;
            uint64_t TotalCount;
            auto CandidateProfileData =
                ICallAnalysis.getPromotionCandidatesForInstruction(
                    &I, NumVals, TotalCount, NumCandidates);
            for (auto &Candidate : CandidateProfileData)
              IndirectCallEdges[Candidate.Value] += Candidate.Count;
          }
        }
      }
      findRefEdges(&I, RefEdges, Visited);
    }

  GlobalValueSummary::GVFlags Flags(F);
  std::unique_ptr<FunctionSummary> FuncSummary =
      llvm::make_unique<FunctionSummary>(Flags, NumInsts);
  FuncSummary->addCallGraphEdges(CallGraphEdges);
  FuncSummary->addCallGraphEdges(IndirectCallEdges);
  FuncSummary->addRefEdges(RefEdges);
  Index->addGlobalValueSummary(F.getName(), std::move(FuncSummary));
}

void ModuleSummaryIndexBuilder::computeVariableSummary(
    const GlobalVariable &V) {
  DenseSet<const Value *> RefEdges;
  SmallPtrSet<const User *, 8> Visited;
  findRefEdges(&V, RefEdges, Visited);
  GlobalValueSummary::GVFlags Flags(V);
  std::unique_ptr<GlobalVarSummary> GVarSummary =
      llvm::make_unique<GlobalVarSummary>(Flags);
  GVarSummary->addRefEdges(RefEdges);
  Index->addGlobalValueSummary(V.getName(), std::move(GVarSummary));
}

ModuleSummaryIndexBuilder::ModuleSummaryIndexBuilder(
    const Module *M,
    std::function<BlockFrequencyInfo *(const Function &F)> Ftor)
    : Index(llvm::make_unique<ModuleSummaryIndex>()), M(M) {
  // Check if the module can be promoted, otherwise just disable importing from
  // it by not emitting any summary.
  // FIXME: we could still import *into* it most of the time.
  if (!moduleCanBeRenamedForThinLTO(*M))
    return;

  // Compute summaries for all functions defined in module, and save in the
  // index.
  for (auto &F : *M) {
    if (F.isDeclaration())
      continue;

    BlockFrequencyInfo *BFI = nullptr;
    std::unique_ptr<BlockFrequencyInfo> BFIPtr;
    if (Ftor)
      BFI = Ftor(F);
    else if (F.getEntryCount().hasValue()) {
      LoopInfo LI{DominatorTree(const_cast<Function &>(F))};
      BranchProbabilityInfo BPI{F, LI};
      BFIPtr = llvm::make_unique<BlockFrequencyInfo>(F, BPI, LI);
      BFI = BFIPtr.get();
    }

    computeFunctionSummary(F, BFI);
  }

  // Compute summaries for all variables defined in module, and save in the
  // index.
  for (const GlobalVariable &G : M->globals()) {
    if (G.isDeclaration())
      continue;
    computeVariableSummary(G);
  }
}

char ModuleSummaryIndexWrapperPass::ID = 0;
INITIALIZE_PASS_BEGIN(ModuleSummaryIndexWrapperPass, "module-summary-analysis",
                      "Module Summary Analysis", false, true)
INITIALIZE_PASS_DEPENDENCY(BlockFrequencyInfoWrapperPass)
INITIALIZE_PASS_END(ModuleSummaryIndexWrapperPass, "module-summary-analysis",
                    "Module Summary Analysis", false, true)

ModulePass *llvm::createModuleSummaryIndexWrapperPass() {
  return new ModuleSummaryIndexWrapperPass();
}

ModuleSummaryIndexWrapperPass::ModuleSummaryIndexWrapperPass()
    : ModulePass(ID) {
  initializeModuleSummaryIndexWrapperPassPass(*PassRegistry::getPassRegistry());
}

bool ModuleSummaryIndexWrapperPass::runOnModule(Module &M) {
  IndexBuilder = llvm::make_unique<ModuleSummaryIndexBuilder>(
      &M, [this](const Function &F) {
        return &(this->getAnalysis<BlockFrequencyInfoWrapperPass>(
                         *const_cast<Function *>(&F))
                     .getBFI());
      });
  return false;
}

bool ModuleSummaryIndexWrapperPass::doFinalization(Module &M) {
  IndexBuilder.reset();
  return false;
}

void ModuleSummaryIndexWrapperPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<BlockFrequencyInfoWrapperPass>();
}

bool llvm::moduleCanBeRenamedForThinLTO(const Module &M) {
  // We cannot currently promote or rename anything used in inline assembly,
  // which are not visible to the compiler. Detect a possible case by looking
  // for a llvm.used local value, in conjunction with an inline assembly call
  // in the module. Prevent importing of any modules containing these uses by
  // suppressing generation of the index. This also prevents importing
  // into this module, which is also necessary to avoid needing to rename
  // in case of a name clash between a local in this module and an imported
  // global.
  // FIXME: If we find we need a finer-grained approach of preventing promotion
  // and renaming of just the functions using inline assembly we will need to:
  // - Add flag in the function summaries to identify those with inline asm.
  // - Prevent importing of any functions with flag set.
  // - Prevent importing of any global function with the same name as a
  //   function in current module that has the flag set.
  // - For any llvm.used value that is exported and promoted, add a private
  //   alias to the original name in the current module (even if we don't
  //   export the function using those values in inline asm, another function
  //   with a reference could be exported).
  SmallPtrSet<GlobalValue *, 8> Used;
  collectUsedGlobalVariables(M, Used, /*CompilerUsed*/ false);
  bool LocalIsUsed =
      llvm::any_of(Used, [](GlobalValue *V) { return V->hasLocalLinkage(); });
  if (!LocalIsUsed)
    return true;

  // Walk all the instructions in the module and find if one is inline ASM
  auto HasInlineAsm = llvm::any_of(M, [](const Function &F) {
    return llvm::any_of(instructions(F), [](const Instruction &I) {
      const CallInst *CallI = dyn_cast<CallInst>(&I);
      if (!CallI)
        return false;
      return CallI->isInlineAsm();
    });
  });
  return !HasInlineAsm;
}
