//===- ReduceArguments.cpp - Specialized Delta Pass -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a function which calls the Generic Delta pass in order
// to reduce uninteresting Arguments from defined functions.
//
//===----------------------------------------------------------------------===//

#include "ReduceArguments.h"
#include "Delta.h"
#include "llvm/ADT/SmallVector.h"
#include <set>
#include <vector>

using namespace llvm;

/// Goes over OldF calls and replaces them with a call to NewF
static void replaceFunctionCalls(Function &OldF, Function &NewF,
                                 const std::set<int> &ArgIndexesToKeep) {
  const auto &Users = OldF.users();
  for (auto I = Users.begin(), E = Users.end(); I != E; )
    if (auto *CI = dyn_cast<CallInst>(*I++)) {
      SmallVector<Value *, 8> Args;
      for (auto ArgI = CI->arg_begin(), E = CI->arg_end(); ArgI != E; ++ArgI)
        if (ArgIndexesToKeep.count(ArgI - CI->arg_begin()))
          Args.push_back(*ArgI);

      CallInst *NewCI = CallInst::Create(&NewF, Args);
      NewCI->setCallingConv(NewF.getCallingConv());
      if (!CI->use_empty())
        CI->replaceAllUsesWith(NewCI);
      ReplaceInstWithInst(CI, NewCI);
    }
}

/// Removes out-of-chunk arguments from functions, and modifies their calls
/// accordingly. It also removes allocations of out-of-chunk arguments.
static void extractArgumentsFromModule(std::vector<Chunk> ChunksToKeep,
                                       Module *Program) {
  int I = 0, ArgCount = 0;
  std::set<Argument *> ArgsToKeep;
  std::vector<Function *> Funcs;
  // Get inside-chunk arguments, as well as their parent function
  for (auto &F : *Program)
    if (!F.isDeclaration()) {
      Funcs.push_back(&F);
      for (auto &A : F.args())
        if (I < (int)ChunksToKeep.size()) {
          if (ChunksToKeep[I].contains(++ArgCount))
            ArgsToKeep.insert(&A);
          if (ChunksToKeep[I].end == ArgCount)
            ++I;
        }
    }

  for (auto *F : Funcs) {
    ValueToValueMapTy VMap;
    std::vector<Instruction *> InstToDelete;
    for (auto &A : F->args())
      if (!ArgsToKeep.count(&A)) {
        // By adding undesired arguments to the VMap, CloneFunction will remove
        // them from the resulting Function
        VMap[&A] = UndefValue::get(A.getType());
        for (auto *U : A.users())
          if (auto *I = dyn_cast<Instruction>(*&U))
            InstToDelete.push_back(I);
      }
    // Delete any instruction that uses the argument
    for (auto *I : InstToDelete) {
      I->replaceAllUsesWith(UndefValue::get(I->getType()));
      I->eraseFromParent();
    }

    // No arguments to reduce
    if (VMap.empty())
      continue;

    std::set<int> ArgIndexesToKeep;
    int ArgI = 0;
    for (auto &Arg : F->args())
      if (ArgsToKeep.count(&Arg))
        ArgIndexesToKeep.insert(++ArgI);

    auto *ClonedFunc = CloneFunction(F, VMap);
    // In order to preserve function order, we move Clone after old Function
    ClonedFunc->removeFromParent();
    Program->getFunctionList().insertAfter(F->getIterator(), ClonedFunc);

    replaceFunctionCalls(*F, *ClonedFunc, ArgIndexesToKeep);
    // Rename Cloned Function to Old's name
    std::string FName = F->getName();
    F->eraseFromParent();
    ClonedFunc->setName(FName);
  }
}

/// Counts the amount of arguments in non-declaration functions and prints their
/// respective name, index, and parent function name
static int countArguments(Module *Program) {
  // TODO: Silence index with --quiet flag
  outs() << "----------------------------\n";
  outs() << "Param Index Reference:\n";
  int ArgsCount = 0;
  for (auto &F : *Program)
    if (!F.isDeclaration() && F.arg_size()) {
      outs() << "  " << F.getName() << "\n";
      for (auto &A : F.args())
        outs() << "\t" << ++ArgsCount << ": " << A.getName() << "\n";

      outs() << "----------------------------\n";
    }

  return ArgsCount;
}

void llvm::reduceArgumentsDeltaPass(TestRunner &Test) {
  outs() << "*** Reducing Arguments...\n";
  int ArgCount = countArguments(Test.getProgram());
  runDeltaPass(Test, ArgCount, extractArgumentsFromModule);
}
