//===- AlwaysSpecializer.cpp - implementation of always_specialize --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Function specialisation under programmer control.
//
// Specifically, function parameters are marked [[always_specialize]], then call
// sites which pass a constant argument are rewritten to call specialisations.
//
// The difficult parts of function specialisation are the cost model, ensuring
// termination and specialisation to the anticipated extent.
//
// Cost model is under programmer control, exactly like always_inline.
//
// Termination follows from the implementation following a phased structure:
// 1. Functions are identifed in the input IR
// 2. Calls that exist in the input IR are identified
// Those constitute the complete set of specialisations that will be created.
//
// This pass does the _minimum_ specialisation, in the sense that only call
// sites in the input will lead to cloning. A specialised function will call
// another specialised function iff there was a call site with the same
// argument vector in the input.
//
// Running the identifyCalls + createClones sequence N times will behave
// as expected, specialising recursively to that depth. This patch has N=1
// in the first instance, with no commandline argument to override.
// Similarly variadic functions are not yet handled.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/AlwaysSpecializer.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/IPO/FunctionSpecialization.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;

#define DEBUG_TYPE "always-specialize"

namespace {

class AlwaysSpecializer : public ModulePass {
public:
  static char ID;

  AlwaysSpecializer() : ModulePass(ID) {}
  StringRef getPassName() const override { return "Always specializer"; }

  // One constant for each argument, nullptr if that one is non-constant
  using ArgVector = SmallVector<Constant *, 4>;

  // A map from the ArgVector to the matching specialisation
  using FunctionSpecializations = MapVector<ArgVector, Function *>;

  // The four mini-passes populate and then use a map:
  // 1. identifyFunctions writes all keys, with default initialised values.
  // 2. identifyCalls writes all the ArgVector keys in the values of SpecList.
  // 3. createClones writes the Function* values at the leaves.
  // 4. replaceCalls walks the map doing the trivial rewrite.

  // Conceptually a Map<Function*, Specialization> but a vector suffices.
  using SpecListTy =
      SmallVector<std::pair<Function *, FunctionSpecializations>, 4>;

  SpecListTy identifyFunctions(Module &M);
  bool identifyCalls(Module &M, Function *F, FunctionSpecializations &);
  bool createClones(Module &M, Function *F, FunctionSpecializations &);
  bool replaceCalls(Module &M, Function *F, FunctionSpecializations &);

  bool runOnModule(Module &M) override {
    bool Changed = false;

    // Sets all the keys in the structure used in this invocation.
    SpecListTy SpecList = identifyFunctions(M);
    size_t Count = SpecList.size();
    if (Count == 0) {
      return false;
    }

    // Record distinct call sites as vector<Constant*> -> nullptr
    for (auto &[F, spec] : SpecList)
      Changed |= identifyCalls(M, F, spec);

    // Create and record the clones. Note that call sites within the clones
    // cannot trigger creating more clones so no termination risk.
    for (auto &[F, spec] : SpecList)
      Changed |= createClones(M, F, spec);

    // Replacing calls as the final phase means no need to track
    // partially-specialised calls and no creating further clones.
    for (auto &[F, spec] : SpecList)
      Changed |= replaceCalls(M, F, spec);

    return Changed;
  }

  static bool isCandidateFunction(const Function &F);
  static bool callEligible(const Function &F, const CallBase *CB,
                           ArgVector &Out);
  static Function *cloneCandidateFunction(Module &M, Function *F,
                                          const ArgVector &C);

  // Only a member variable to reuse the allocation. Short lived.
  ArgVector ArgVec;
};

AlwaysSpecializer::SpecListTy AlwaysSpecializer::identifyFunctions(Module &M) {
  SpecListTy SpecList;
  for (Function &F : M) {
    if (isCandidateFunction(F)) {
      SpecList.push_back(std::make_pair(&F, FunctionSpecializations()));
    }
  }
  return SpecList;
}

bool AlwaysSpecializer::identifyCalls(Module &M, Function *F,
                                      FunctionSpecializations &Specs) {
  bool Found = false;

  for (User *U : F->users()) {
    CallBase *CB = dyn_cast<CallBase>(U);
    if (!CB || !callEligible(*F, CB, ArgVec)) {
      continue;
    }

    if (!Specs.contains(ArgVec)) {
      Found = true;
      Specs.insert(std::make_pair(ArgVec, nullptr));
    }
  }

  return Found;
}

bool AlwaysSpecializer::createClones(Module &M, Function *F,
                                     FunctionSpecializations &Specs) {
  bool Changed = false;

  for (auto It = Specs.begin(); It != Specs.end(); ++It) {
    if (It->second)
      continue;
    Function *Clone = cloneCandidateFunction(M, F, It->first);
    if (Clone) {
      Changed = true;
      It->second = Clone;
    }
  }

  return Changed;
}

bool AlwaysSpecializer::replaceCalls(Module &M, Function *F,
                                     FunctionSpecializations &Specs) {
  bool Changed = false;

  for (User *u : make_early_inc_range(F->users())) {
    CallBase *CB = dyn_cast<CallBase>(u);
    if (!CB || !callEligible(*F, CB, ArgVec)) {
      continue;
    }

    Function *Clone = Specs[ArgVec];
    if (Clone) {
      Changed = true;
      CB->setCalledFunction(Clone);
    }
  }

  return Changed;
}

bool AlwaysSpecializer::isCandidateFunction(const Function &F) {

  // Test if the function itself can't be specialised
  if (!F.hasExactDefinition() || F.isIntrinsic() ||
      F.hasFnAttribute(Attribute::Naked))
    return false;

  // Variadics are left for a follow up patch
  if (F.isVarArg())
    return false;

  // Need calls to the function for it to be worth considering
  if (F.use_empty())
    return false;

  // Look for the attribute on a non-dead, non-indirect parameter
  for (const Argument &Arg : F.args()) {
    if (Arg.hasPointeeInMemoryValueAttr())
      continue;

    if (F.hasParamAttribute(Arg.getArgNo(), Attribute::AlwaysSpecialize))
      if (!Arg.use_empty())
        return true;
  }

  return false;
}

bool AlwaysSpecializer::callEligible(const Function &F, const CallBase *CB,
                                     ArgVector &Out) {
  const size_t Arity = F.arg_size();
  bool Eligible = false;

  if (CB->getCalledOperand() != &F) {
    return false;
  }

  if (CB->getFunctionType() != F.getFunctionType()) {
    return false;
  }

  if (CB->arg_size() != Arity) {
    return false;
  }

  Out.clear();
  for (size_t I = 0; I < Arity; I++) {
    Constant *Arg = dyn_cast<Constant>(CB->getArgOperand(I));
    if (Arg && F.hasParamAttribute(I, Attribute::AlwaysSpecialize)) {
      Eligible = true;
      Out.push_back(Arg);
    } else {
      Out.push_back(nullptr);
    }
  }

  return Eligible;
}

Function *AlwaysSpecializer::cloneCandidateFunction(Module &M, Function *F,
                                                    const ArgVector &C) {

  Function *Clone =
      Function::Create(F->getFunctionType(), F->getLinkage(),
                       F->getAddressSpace(), F->getName() + ".spec");

  // Roughly CloneFunction but inserting specialisations next to the original.
  ValueToValueMapTy VMap;
  Function::arg_iterator DestI = Clone->arg_begin();
  for (const Argument &I : F->args()) {
    DestI->setName(I.getName());
    VMap[&I] = &*DestI++;
  }
  SmallVector<ReturnInst *, 8> Returns;
  CloneFunctionInto(Clone, F, VMap, CloneFunctionChangeType::LocalChangesOnly,
                    Returns);

  M.getFunctionList().insert(F->getIterator(), Clone);

  // Clones are local things.
  Clone->setDSOLocal(true);
  Clone->setVisibility(GlobalValue::DefaultVisibility);
  Clone->setLinkage(GlobalValue::PrivateLinkage);

  // Replace uses of the argument with the constant.
  for (size_t I = 0; I < C.size(); I++) {
    if (!C[I])
      continue;

    // The argument is going to be dead, drop the specialise attr.
    Clone->removeParamAttr(I, Attribute::AlwaysSpecialize);

    Argument *V = Clone->getArg(I);
    for (User *U : make_early_inc_range(V->users())) {

      if (auto *Inst = dyn_cast<Instruction>(U)) {
        SimplifyQuery SQ = SimplifyQuery(Clone->getDataLayout(), Inst);

        // Do some simplification on the fly so that call sites in the cloned
        // functions can potentially themselves resolve to specialisations
        if (Value *NewInst = simplifyWithOpReplaced(
                Inst, V, C[I], SQ, false /*AllowRefinement*/)) {
          Inst->replaceAllUsesWith(NewInst);
          continue;
        }

        // If we're about to create a load from a constant, try to resolve it
        // immediately so that the uses of the load are now also constant.
        // This covers constant vtable containing pointer to constant vtable.
        if (auto *Load = dyn_cast<LoadInst>(Inst)) {
          if (Load->getOperand(0) == V) {
            if (Value *NewInst = simplifyLoadInst(Load, C[I], SQ)) {
              Load->replaceAllUsesWith(NewInst);
              continue;
            }
          }
        }
      }
    }

    // Replace any remaining uses that the above failed to simplify.
    V->replaceAllUsesWith(C[I]);
  }

  return Clone;
}

} // namespace

char AlwaysSpecializer::ID = 0;

INITIALIZE_PASS(AlwaysSpecializer, DEBUG_TYPE, "TODO", false, false)

ModulePass *createAlwaysSpecializerPass() { return new AlwaysSpecializer(); }

PreservedAnalyses AlwaysSpecializerPass::run(Module &M,
                                             ModuleAnalysisManager &) {
  return AlwaysSpecializer().runOnModule(M) ? PreservedAnalyses::none()
                                            : PreservedAnalyses::all();
}

AlwaysSpecializerPass::AlwaysSpecializerPass() {}
