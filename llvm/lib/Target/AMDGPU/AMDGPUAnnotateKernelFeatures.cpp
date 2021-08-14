//===- AMDGPUAnnotateKernelFeaturesPass.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This pass propagates the uniform-work-group-size attribute from
/// kernels to leaf functions when possible. It also adds additional attributes
/// to hint ABI lowering optimizations later.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/CallGraphSCCPass.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/IntrinsicsR600.h"
#include "llvm/Target/TargetMachine.h"

#define DEBUG_TYPE "amdgpu-annotate-kernel-features"

using namespace llvm;

namespace {
class AMDGPUAnnotateKernelFeatures : public CallGraphSCCPass {
private:
  const TargetMachine *TM = nullptr;
  SmallVector<CallGraphNode*, 8> NodeList;

  bool processUniformWorkGroupAttribute();
  bool propagateUniformWorkGroupAttribute(Function &Caller, Function &Callee);
  bool addFeatureAttributes(Function &F);

public:
  static char ID;

  AMDGPUAnnotateKernelFeatures() : CallGraphSCCPass(ID) {}

  bool doInitialization(CallGraph &CG) override;
  bool runOnSCC(CallGraphSCC &SCC) override;

  StringRef getPassName() const override {
    return "AMDGPU Annotate Kernel Features";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    CallGraphSCCPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

char AMDGPUAnnotateKernelFeatures::ID = 0;

char &llvm::AMDGPUAnnotateKernelFeaturesID = AMDGPUAnnotateKernelFeatures::ID;

INITIALIZE_PASS(AMDGPUAnnotateKernelFeatures, DEBUG_TYPE,
                "Add AMDGPU function attributes", false, false)

bool AMDGPUAnnotateKernelFeatures::processUniformWorkGroupAttribute() {
  bool Changed = false;

  for (auto *Node : reverse(NodeList)) {
    Function *Caller = Node->getFunction();

    for (auto I : *Node) {
      Function *Callee = std::get<1>(I)->getFunction();
      if (Callee)
        Changed = propagateUniformWorkGroupAttribute(*Caller, *Callee);
    }
  }

  return Changed;
}

bool AMDGPUAnnotateKernelFeatures::propagateUniformWorkGroupAttribute(
       Function &Caller, Function &Callee) {

  // Check for externally defined function
  if (!Callee.hasExactDefinition()) {
    Callee.addFnAttr("uniform-work-group-size", "false");
    if (!Caller.hasFnAttribute("uniform-work-group-size"))
      Caller.addFnAttr("uniform-work-group-size", "false");

    return true;
  }
  // Check if the Caller has the attribute
  if (Caller.hasFnAttribute("uniform-work-group-size")) {
    // Check if the value of the attribute is true
    if (Caller.getFnAttribute("uniform-work-group-size")
        .getValueAsString().equals("true")) {
      // Propagate the attribute to the Callee, if it does not have it
      if (!Callee.hasFnAttribute("uniform-work-group-size")) {
        Callee.addFnAttr("uniform-work-group-size", "true");
        return true;
      }
    } else {
      Callee.addFnAttr("uniform-work-group-size", "false");
      return true;
    }
  } else {
    // If the attribute is absent, set it as false
    Caller.addFnAttr("uniform-work-group-size", "false");
    Callee.addFnAttr("uniform-work-group-size", "false");
    return true;
  }
  return false;
}

bool AMDGPUAnnotateKernelFeatures::addFeatureAttributes(Function &F) {
  bool HaveStackObjects = false;
  bool Changed = false;
  bool HaveCall = false;
  bool IsFunc = !AMDGPU::isEntryFunctionCC(F.getCallingConv());

  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (isa<AllocaInst>(I)) {
        HaveStackObjects = true;
        continue;
      }

      if (auto *CB = dyn_cast<CallBase>(&I)) {
        const Function *Callee =
            dyn_cast<Function>(CB->getCalledOperand()->stripPointerCasts());

        // Note the occurence of indirect call.
        if (!Callee) {
          if (!CB->isInlineAsm())
            HaveCall = true;

          continue;
        }

        Intrinsic::ID IID = Callee->getIntrinsicID();
        if (IID == Intrinsic::not_intrinsic) {
          HaveCall = true;
          Changed = true;
        }
      }
    }
  }

  // TODO: We could refine this to captured pointers that could possibly be
  // accessed by flat instructions. For now this is mostly a poor way of
  // estimating whether there are calls before argument lowering.
  if (!IsFunc && HaveCall) {
    F.addFnAttr("amdgpu-calls");
    Changed = true;
  }

  if (HaveStackObjects) {
    F.addFnAttr("amdgpu-stack-objects");
    Changed = true;
  }

  return Changed;
}

bool AMDGPUAnnotateKernelFeatures::runOnSCC(CallGraphSCC &SCC) {
  bool Changed = false;

  for (CallGraphNode *I : SCC) {
    // Build a list of CallGraphNodes from most number of uses to least
    if (I->getNumReferences())
      NodeList.push_back(I);
    else {
      processUniformWorkGroupAttribute();
      NodeList.clear();
    }

    Function *F = I->getFunction();
    // Ignore functions with graphics calling conventions, these are currently
    // not allowed to have kernel arguments.
    if (!F || F->isDeclaration() || AMDGPU::isGraphics(F->getCallingConv()))
      continue;
    // Add feature attributes
    Changed |= addFeatureAttributes(*F);
  }

  return Changed;
}

bool AMDGPUAnnotateKernelFeatures::doInitialization(CallGraph &CG) {
  auto *TPC = getAnalysisIfAvailable<TargetPassConfig>();
  if (!TPC)
    report_fatal_error("TargetMachine is required");

  TM = &TPC->getTM<TargetMachine>();
  return false;
}

Pass *llvm::createAMDGPUAnnotateKernelFeaturesPass() {
  return new AMDGPUAnnotateKernelFeatures();
}
