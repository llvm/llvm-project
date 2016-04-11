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
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Dominators.h"
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

void ModuleSummaryIndexBuilder::computeFunctionInfo(const Function &F,
                                                    BlockFrequencyInfo *BFI) {
  // Summary not currently supported for anonymous functions, they must
  // be renamed.
  if (!F.hasName())
    return;

  unsigned NumInsts = 0;
  // Map from callee ValueId to profile count. Used to accumulate profile
  // counts for all static calls to a given callee.
  DenseMap<const Value *, CalleeInfo> CallGraphEdges;
  DenseSet<const Value *> RefEdges;

  SmallPtrSet<const User *, 8> Visited;
  for (Function::const_iterator BB = F.begin(), E = F.end(); BB != E; ++BB)
    for (BasicBlock::const_iterator I = BB->begin(), E = BB->end(); I != E;
         ++I) {
      if (!isa<DbgInfoIntrinsic>(I))
        ++NumInsts;

      if (auto CS = ImmutableCallSite(&*I)) {
        auto *CalledFunction = CS.getCalledFunction();
        if (CalledFunction && CalledFunction->hasName() &&
            !CalledFunction->isIntrinsic()) {
          auto ScaledCount = BFI ? BFI->getBlockProfileCount(&*BB) : None;
          auto *CalleeId =
              M->getValueSymbolTable().lookup(CalledFunction->getName());
          CallGraphEdges[CalleeId] +=
              (ScaledCount ? ScaledCount.getValue() : 0);
        }
      }
      findRefEdges(&*I, RefEdges, Visited);
    }

  std::unique_ptr<FunctionSummary> FuncSummary =
      llvm::make_unique<FunctionSummary>(F.getLinkage(), NumInsts);
  FuncSummary->addCallGraphEdges(CallGraphEdges);
  FuncSummary->addRefEdges(RefEdges);
  std::unique_ptr<GlobalValueInfo> GVInfo =
      llvm::make_unique<GlobalValueInfo>(0, std::move(FuncSummary));
  Index->addGlobalValueInfo(F.getName(), std::move(GVInfo));
}

void ModuleSummaryIndexBuilder::computeVariableInfo(const GlobalVariable &V) {
  DenseSet<const Value *> RefEdges;
  SmallPtrSet<const User *, 8> Visited;
  findRefEdges(&V, RefEdges, Visited);
  std::unique_ptr<GlobalVarSummary> GVarSummary =
      llvm::make_unique<GlobalVarSummary>(V.getLinkage());
  GVarSummary->addRefEdges(RefEdges);
  std::unique_ptr<GlobalValueInfo> GVInfo =
      llvm::make_unique<GlobalValueInfo>(0, std::move(GVarSummary));
  Index->addGlobalValueInfo(V.getName(), std::move(GVInfo));
}

ModuleSummaryIndexBuilder::ModuleSummaryIndexBuilder(
    const Module *M,
    std::function<BlockFrequencyInfo *(const Function &F)> Ftor)
    : Index(llvm::make_unique<ModuleSummaryIndex>()), M(M) {
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

    computeFunctionInfo(F, BFI);
  }

  // Compute summaries for all variables defined in module, and save in the
  // index.
  for (const GlobalVariable &G : M->globals()) {
    if (G.isDeclaration())
      continue;
    computeVariableInfo(G);
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
