//===-- AMDGPUCloneModuleLDSPass.cpp ------------------------------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The purpose of this pass is to ensure that the combined module contains
// as many LDS global variables as there are kernels that (indirectly) access
// them. As LDS variables behave like C++ static variables, it is important that
// each partition contains a unique copy of the variable on a per kernel basis.
// This representation also prepares the combined module to eliminate
// cross-module false dependencies of LDS variables. This pass runs prior to the
// AMDGPULowerModuleLDS pass in the fullLTO pipeline and is used to improve
// the functionality of --lto-partitions.
//
// This pass operates as follows:
// 1. Firstly, traverse the call graph from each kernel to determine the number
//    of kernels calling each device function.
// 2. For each LDS global variable GV, determine the function F that defines it.
//    Collect it's caller functions. Clone F and GV, and finally insert a
//    call/invoke instruction in each caller function.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "Utils/AMDGPUMemoryUtils.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;
using GVToFnMapTy = DenseMap<GlobalVariable *, Function *>;

#define DEBUG_TYPE "amdgpu-clone-module-lds"

static cl::opt<unsigned int> MaxCountForClonedFunctions(
    "clone-lds-functions-max-count", cl::init(16), cl::Hidden,
    cl::desc("Specify a limit to the number of clones of a function"));

/// Return the function that defines \p GV
/// \param GV The global variable in question
/// \return The function defining \p GV
static Function *getFunctionDefiningGV(GlobalVariable &GV) {
  SmallVector<User *> Worklist(GV.users());
  while (!Worklist.empty()) {
    User *U = Worklist.pop_back_val();
    if (auto *Inst = dyn_cast<Instruction>(U))
      return Inst->getFunction();
    if (auto *Op = dyn_cast<Operator>(U))
      append_range(Worklist, Op->users());
  }
  return nullptr;
};

/// Return a map of LDS globals paired with the function defining them
/// \param M Module in question
/// \return Map of LDS global variables and their functions
static GVToFnMapTy collectModuleGlobals(Module &M) {
  GVToFnMapTy GVToFnMap;
  for (auto &GA : M.aliases()) {
    if (auto *GV = dyn_cast<GlobalVariable>(GA.getAliaseeObject())) {
      if (AMDGPU::isLDSVariableToLower(*GV) && !GVToFnMap.contains(GV))
        GVToFnMap.insert({GV, getFunctionDefiningGV(*GV)});
    }
  }

  for (auto &GV : M.globals()) {
    if (AMDGPU::isLDSVariableToLower(GV) && !GVToFnMap.contains(&GV))
      GVToFnMap.insert({&GV, getFunctionDefiningGV(GV)});
  }
  return GVToFnMap;
}

PreservedAnalyses AMDGPUCloneModuleLDSPass::run(Module &M,
                                                ModuleAnalysisManager &AM) {
  if (MaxCountForClonedFunctions.getValue() == 1)
    return PreservedAnalyses::all();

  bool Changed = false;
  auto &CG = AM.getResult<CallGraphAnalysis>(M);

  // For each function in the call graph, determine the number
  // of ancestor-caller kernels.
  DenseMap<Function *, unsigned int> KernelRefsToFuncs;
  for (auto &Fn : M) {
    if (Fn.getCallingConv() != CallingConv::AMDGPU_KERNEL)
      continue;
    for (auto I = df_begin(&CG), E = df_end(&CG); I != E; ++I) {
      if (auto *F = I->getFunction())
        KernelRefsToFuncs[F]++;
    }
  }

  GVToFnMapTy GVToFnMap = collectModuleGlobals(M);
  for (auto [GV, OldF] : GVToFnMap) {
    LLVM_DEBUG(dbgs() << "Found LDS " << GV.getName() << " used in function "
                      << OldF->getName() << '\n');

    // Collect all call instructions to OldF
    SmallVector<Instruction *> InstsCallingOldF;
    for (auto &I : OldF->uses()) {
      if (auto *CI = dyn_cast<CallBase>(I.getUser()))
        InstsCallingOldF.push_back(CI);
    }

    // Create as many clones of the function containing LDS global as
    // there are kernels calling the function (including the function
    // already defining the LDS global). Respectively, clone the
    // LDS global and the call instructions to the function.
    LLVM_DEBUG(dbgs() << "\tFunction is referenced by "
                      << KernelRefsToFuncs[OldF] << " kernels.\n");
    for (unsigned int ID = 0;
         ID + 1 < std::min(KernelRefsToFuncs[OldF],
                           MaxCountForClonedFunctions.getValue());
         ++ID) {
      // Clone LDS global variable
      auto *NewGV = new GlobalVariable(
          M, GV->getValueType(), GV->isConstant(), GlobalValue::InternalLinkage,
          PoisonValue::get(GV->getValueType()),
          GV->getName() + ".clone." + Twine(ID), GV,
          GlobalValue::NotThreadLocal, AMDGPUAS::LOCAL_ADDRESS, false);
      NewGV->copyAttributesFrom(GV);
      NewGV->copyMetadata(GV, 0);
      NewGV->setComdat(GV->getComdat());
      LLVM_DEBUG(dbgs() << "Inserting LDS clone with name " << NewGV->getName()
                        << '\n');

      // Clone function
      ValueToValueMapTy VMap;
      VMap[GV] = NewGV;
      auto *NewF = CloneFunction(OldF, VMap);
      NewF->setName(OldF->getName() + ".clone." + Twine(ID));
      LLVM_DEBUG(dbgs() << "Inserting function clone with name "
                        << NewF->getName() << '\n');

      // Create a new CallInst to call the cloned function
      for (auto *Inst : InstsCallingOldF) {
        Instruction *I = Inst->clone();
        I->setName(Inst->getName() + ".clone." + Twine(ID));
        if (auto *CI = dyn_cast<CallBase>(I))
          CI->setCalledOperand(NewF);
        I->insertAfter(Inst);
        LLVM_DEBUG(dbgs() << "Inserting inst: " << *I << '\n');
      }
      Changed = true;
    }
  }
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
