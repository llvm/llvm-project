//===--- AMDGPUPropagateAttributes.cpp --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief This pass propagates attributes from kernels to the non-entry
/// functions. Most of the library functions were not compiled for specific ABI,
/// yet will be correctly compiled if proper attrbutes are propagated from the
/// caller.
///
/// The pass analyzes call graph and propagates ABI target features through the
/// call graph.
///
/// It can run in two modes: as a function or module pass. A function pass
/// simply propagates attributes. A module pass clones functions if there are
/// callers with different ABI. If a function is clonned all call sites will
/// be updated to use a correct clone.
///
/// A function pass is limited in functionality but can run early in the
/// pipeline. A module pass is more powerful but has to run late, so misses
/// library folding opportunities.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <string>

#define DEBUG_TYPE "amdgpu-propagate-attributes"

using namespace llvm;

namespace llvm {
extern const SubtargetFeatureKV AMDGPUFeatureKV[AMDGPU::NumSubtargetFeatures-1];
}

namespace {

class AMDGPUPropagateAttributes {
  const FeatureBitset TargetFeatures = {
    AMDGPU::FeatureWavefrontSize16,
    AMDGPU::FeatureWavefrontSize32,
    AMDGPU::FeatureWavefrontSize64
  };

  class Clone{
  public:
    Clone(FeatureBitset FeatureMask, Function *OrigF, Function *NewF) :
      FeatureMask(FeatureMask), OrigF(OrigF), NewF(NewF) {}

    FeatureBitset FeatureMask;
    Function *OrigF;
    Function *NewF;
  };

  const TargetMachine *TM;

  // Clone functions as needed or just set attributes.
  bool AllowClone;

  // Option propagation roots.
  SmallSet<Function *, 32> Roots;

  // Clones of functions with their attributes.
  SmallVector<Clone, 32> Clones;

  // Find a clone with required features.
  Function *findFunction(const FeatureBitset &FeaturesNeeded,
                         Function *OrigF);

  // Clone function F and set NewFeatures on the clone.
  // Cole takes the name of original function.
  Function *cloneWithFeatures(Function &F,
                              const FeatureBitset &NewFeatures);

  // Set new function's features in place.
  void setFeatures(Function &F, const FeatureBitset &NewFeatures);

  std::string getFeatureString(const FeatureBitset &Features) const;

  // Propagate attributes from Roots.
  bool process();

public:
  AMDGPUPropagateAttributes(const TargetMachine *TM, bool AllowClone) :
    TM(TM), AllowClone(AllowClone) {}

  // Use F as a root and propagate its attributes.
  bool process(Function &F);

  // Propagate attributes starting from kernel functions.
  bool process(Module &M);
};

// Allows to propagate attributes early, but no clonning is allowed as it must
// be a function pass to run before any optimizations.
// TODO: We shall only need a one instance of module pass, but that needs to be
// in the linker pipeline which is currently not possible.
class AMDGPUPropagateAttributesEarly : public FunctionPass {
  const TargetMachine *TM;

public:
  static char ID; // Pass identification

  AMDGPUPropagateAttributesEarly(const TargetMachine *TM = nullptr) :
    FunctionPass(ID), TM(TM) {
    initializeAMDGPUPropagateAttributesEarlyPass(
      *PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override;
};

// Allows to propagate attributes with clonning but does that late in the
// pipeline.
class AMDGPUPropagateAttributesLate : public ModulePass {
  const TargetMachine *TM;

public:
  static char ID; // Pass identification

  AMDGPUPropagateAttributesLate(const TargetMachine *TM = nullptr) :
    ModulePass(ID), TM(TM) {
    initializeAMDGPUPropagateAttributesLatePass(
      *PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override;
};

}  // end anonymous namespace.

char AMDGPUPropagateAttributesEarly::ID = 0;
char AMDGPUPropagateAttributesLate::ID = 0;

INITIALIZE_PASS(AMDGPUPropagateAttributesEarly,
                "amdgpu-propagate-attributes-early",
                "Early propagate attributes from kernels to functions",
                false, false)
INITIALIZE_PASS(AMDGPUPropagateAttributesLate,
                "amdgpu-propagate-attributes-late",
                "Late propagate attributes from kernels to functions",
                false, false)

Function *
AMDGPUPropagateAttributes::findFunction(const FeatureBitset &FeaturesNeeded,
                                        Function *OrigF) {
  // TODO: search for clone's clones.
  for (Clone &C : Clones)
    if (C.OrigF == OrigF && FeaturesNeeded == C.FeatureMask)
      return C.NewF;

  return nullptr;
}

bool AMDGPUPropagateAttributes::process(Module &M) {
  for (auto &F : M.functions())
    if (AMDGPU::isEntryFunctionCC(F.getCallingConv()))
      Roots.insert(&F);

  return process();
}

bool AMDGPUPropagateAttributes::process(Function &F) {
  Roots.insert(&F);
  return process();
}

bool AMDGPUPropagateAttributes::process() {
  bool Changed = false;
  SmallSet<Function *, 32> NewRoots;
  SmallSet<Function *, 32> Replaced;

  if (Roots.empty())
    return false;
  Module &M = *(*Roots.begin())->getParent();

  do {
    Roots.insert(NewRoots.begin(), NewRoots.end());
    NewRoots.clear();

    for (auto &F : M.functions()) {
      if (F.isDeclaration() || Roots.count(&F) || Roots.count(&F))
        continue;

      const FeatureBitset &CalleeBits =
        TM->getSubtargetImpl(F)->getFeatureBits();
      SmallVector<std::pair<CallBase *, Function *>, 32> ToReplace;

      for (User *U : F.users()) {
        Instruction *I = dyn_cast<Instruction>(U);
        if (!I)
          continue;
        CallBase *CI = dyn_cast<CallBase>(I);
        if (!CI)
          continue;
        Function *Caller = CI->getCaller();
        if (!Caller)
          continue;
        if (!Roots.count(Caller))
          continue;

        const FeatureBitset &CallerBits =
          TM->getSubtargetImpl(*Caller)->getFeatureBits() & TargetFeatures;

        if (CallerBits == (CalleeBits  & TargetFeatures)) {
          NewRoots.insert(&F);
          continue;
        }

        Function *NewF = findFunction(CallerBits, &F);
        if (!NewF) {
          FeatureBitset NewFeatures((CalleeBits & ~TargetFeatures) |
                                    CallerBits);
          if (!AllowClone) {
            // This may set different features on different iteartions if
            // there is a contradiction in callers' attributes. In this case
            // we rely on a second pass running on Module, which is allowed
            // to clone.
            setFeatures(F, NewFeatures);
            NewRoots.insert(&F);
            Changed = true;
            break;
          }

          NewF = cloneWithFeatures(F, NewFeatures);
          Clones.push_back(Clone(CallerBits, &F, NewF));
          NewRoots.insert(NewF);
        }

        ToReplace.push_back(std::make_pair(CI, NewF));
        Replaced.insert(&F);

        Changed = true;
      }

      while (!ToReplace.empty()) {
        auto R = ToReplace.pop_back_val();
        R.first->setCalledFunction(R.second);
      }
    }
  } while (!NewRoots.empty());

  for (Function *F : Replaced) {
    if (F->use_empty())
      F->eraseFromParent();
  }

  return Changed;
}

Function *
AMDGPUPropagateAttributes::cloneWithFeatures(Function &F,
                                             const FeatureBitset &NewFeatures) {
  LLVM_DEBUG(dbgs() << "Cloning " << F.getName() << '\n');

  ValueToValueMapTy dummy;
  Function *NewF = CloneFunction(&F, dummy);
  setFeatures(*NewF, NewFeatures);

  // Swap names. If that is the only clone it will retain the name of now
  // dead value.
  if (F.hasName()) {
    std::string NewName = NewF->getName();
    NewF->takeName(&F);
    F.setName(NewName);

    // Name has changed, it does not need an external symbol.
    F.setVisibility(GlobalValue::DefaultVisibility);
    F.setLinkage(GlobalValue::InternalLinkage);
  }

  return NewF;
}

void AMDGPUPropagateAttributes::setFeatures(Function &F,
                                            const FeatureBitset &NewFeatures) {
  std::string NewFeatureStr = getFeatureString(NewFeatures);

  LLVM_DEBUG(dbgs() << "Set features "
                    << getFeatureString(NewFeatures & TargetFeatures)
                    << " on " << F.getName() << '\n');

  F.removeFnAttr("target-features");
  F.addFnAttr("target-features", NewFeatureStr);
}

std::string
AMDGPUPropagateAttributes::getFeatureString(const FeatureBitset &Features) const
{
  std::string Ret;
  for (const SubtargetFeatureKV &KV : AMDGPUFeatureKV) {
    if (Features[KV.Value])
      Ret += (StringRef("+") + KV.Key + ",").str();
    else if (TargetFeatures[KV.Value])
      Ret += (StringRef("-") + KV.Key + ",").str();
  }
  Ret.pop_back(); // Remove last comma.
  return Ret;
}

bool AMDGPUPropagateAttributesEarly::runOnFunction(Function &F) {
  if (!TM || !AMDGPU::isEntryFunctionCC(F.getCallingConv()))
    return false;

  return AMDGPUPropagateAttributes(TM, false).process(F);
}

bool AMDGPUPropagateAttributesLate::runOnModule(Module &M) {
  if (!TM)
    return false;

  return AMDGPUPropagateAttributes(TM, true).process(M);
}

FunctionPass
*llvm::createAMDGPUPropagateAttributesEarlyPass(const TargetMachine *TM) {
  return new AMDGPUPropagateAttributesEarly(TM);
}

ModulePass
*llvm::createAMDGPUPropagateAttributesLatePass(const TargetMachine *TM) {
  return new AMDGPUPropagateAttributesLate(TM);
}
