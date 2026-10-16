//===- ReduceInlineCallSites.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ReduceInlineCallSites.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;

extern cl::OptionCategory LLVMReduceOptions;

static cl::opt<int> CallsiteInlineThreshold(
    "reduce-callsite-inline-threshold",
    cl::desc("Number of instructions in a function to unconditionally inline "
             "(-1 for inline all)"),
    cl::init(5), cl::cat(LLVMReduceOptions));

static bool functionHasMoreThanNonTerminatorInsts(const Function &F,
                                                  uint64_t NumInsts) {
  uint64_t InstCount = 0;
  for (const BasicBlock &BB : F) {
    for (const Instruction &I : make_range(BB.begin(), std::prev(BB.end()))) {
      (void)I;
      if (InstCount++ > NumInsts)
        return true;
    }
  }

  return false;
}

static bool hasOnlyOneCallUse(const Function &F) {
  unsigned UseCount = 0;
  for (const Use &U : F.uses()) {
    const CallBase *CB = dyn_cast<CallBase>(U.getUser());
    if (!CB || !CB->isCallee(&U))
      return false;
    if (UseCount++ > 1)
      return false;
  }

  return UseCount == 1;
}

// TODO: This could use more thought.
static bool inlineWillReduceComplexity(const Function &Caller,
                                       const Function &Callee) {
  // Backdoor to force all possible inlining.
  if (CallsiteInlineThreshold < 0)
    return true;

  if (!hasOnlyOneCallUse(Callee))
    return false;

  // Permit inlining small functions into big functions, or big functions into
  // small functions.
  if (!functionHasMoreThanNonTerminatorInsts(Callee, CallsiteInlineThreshold) &&
      !functionHasMoreThanNonTerminatorInsts(Caller, CallsiteInlineThreshold))
    return true;

  return false;
}

static void reduceCallSites(Oracle &O, Function &F) {
  std::vector<std::pair<CallBase *, InlineFunctionInfo>> CallSitesToInline;

  for (Use &U : F.uses()) {
    if (CallBase *CB = dyn_cast<CallBase>(U.getUser())) {
      // Ignore callsites with wrong call type.
      if (!CB->isCallee(&U))
        continue;

      // We do not consider isInlineViable here. It is overly conservative in
      // cases that the inliner should handle correctly (e.g. disallowing inline
      // of of functions with indirectbr). Some of the other cases are for other
      // correctness issues which we do need to worry about here.

      // TODO: Should we delete the function body?
      InlineFunctionInfo IFI;
      if (CanInlineCallSite(*CB, IFI).isSuccess() &&
          inlineWillReduceComplexity(*CB->getFunction(), F) && !O.shouldKeep())
        CallSitesToInline.emplace_back(CB, std::move(IFI));
    }
  }

  // TODO: InlineFunctionImpl will implicitly perform some simplifications /
  // optimizations which we should be able to opt-out of.
  for (auto [CB, IFI] : CallSitesToInline)
    InlineFunctionImpl(*CB, IFI);
}

void llvm::reduceInlineCallSitesDeltaPass(Oracle &O, ReducerWorkItem &Program) {
  for (Function &F : Program.getModule()) {
    if (!F.isDeclaration())
      reduceCallSites(O, F);
  }
}
