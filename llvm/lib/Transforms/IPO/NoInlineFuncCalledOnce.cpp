#include "llvm/Transforms/IPO/NoInlineFuncCalledOnce.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

PreservedAnalyses NoInlineFuncCalledOncePass::run(Module &M,
                                                  ModuleAnalysisManager &) {
  DenseMap<Function *, unsigned> DirectCalls;
  DenseSet<Function *> Recursive;

  for (Function &F : M)
    if (!F.isDeclaration() && (F.hasInternalLinkage() || F.hasPrivateLinkage()))
      DirectCalls[&F] = 0;

  for (Function &Caller : M) {
    if (Caller.isDeclaration())
      continue;
    for (Instruction &I : instructions(Caller)) {
      auto *CB = dyn_cast<CallBase>(&I);
      if (!CB)
        continue;
      const Value *Op = CB->getCalledOperand()->stripPointerCasts();
      if (auto *Callee = const_cast<Function *>(dyn_cast<Function>(Op))) {
        if (!DirectCalls.count(Callee))
          continue;
        DirectCalls[Callee] += 1;
        if (&Caller == Callee)
          Recursive.insert(Callee);
      }
    }
  }

  bool Changed = false;
  for (auto &KV : DirectCalls) {
    Function *F = KV.first;
    unsigned N = KV.second;

    if (N != 1)
      continue; // only called-once
    if (Recursive.count(F))
      continue; // skip recursion
    if (F->hasAddressTaken())
      continue; // skip address-taken
    if (F->hasFnAttribute(Attribute::AlwaysInline))
      continue;
    if (F->hasFnAttribute(Attribute::NoInline))
      continue;

    F->addFnAttr(Attribute::NoInline);
    Changed = true;
  }

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
