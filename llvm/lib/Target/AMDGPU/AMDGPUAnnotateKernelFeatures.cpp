//===- AMDGPUAnnotateKernelFeaturesPass.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This pass adds target attributes to functions which use intrinsics
/// which will impact calling convention lowering.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUSubtarget.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/CallGraphSCCPass.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Use.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Target/TargetMachine.h"

#define DEBUG_TYPE "amdgpu-annotate-kernel-features"

using namespace llvm;

namespace {

class AMDGPUAnnotateKernelFeatures : public CallGraphSCCPass {
private:
  const TargetMachine *TM = nullptr;
  SmallVector<CallGraphNode*, 8> NodeList;

  bool addFeatureAttributes(Function &F);
  bool processUniformWorkGroupAttribute();
  bool propagateUniformWorkGroupAttribute(Function &Caller, Function &Callee);

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

  static bool visitConstantExpr(const ConstantExpr *CE);
  static bool visitConstantExprsRecursively(
    const Constant *EntryC,
    SmallPtrSet<const Constant *, 8> &ConstantExprVisited, bool IsFunc,
    bool HasApertureRegs);
};

} // end anonymous namespace

char AMDGPUAnnotateKernelFeatures::ID = 0;

char &llvm::AMDGPUAnnotateKernelFeaturesID = AMDGPUAnnotateKernelFeatures::ID;

INITIALIZE_PASS(AMDGPUAnnotateKernelFeatures, DEBUG_TYPE,
                "Add AMDGPU function attributes", false, false)


// The queue ptr is only needed when casting to flat, not from it.
static bool castRequiresQueuePtr(unsigned SrcAS) {
  return SrcAS == AMDGPUAS::LOCAL_ADDRESS || SrcAS == AMDGPUAS::PRIVATE_ADDRESS;
}

static bool castRequiresQueuePtr(const AddrSpaceCastInst *ASC) {
  return castRequiresQueuePtr(ASC->getSrcAddressSpace());
}

static bool isDSAddress(const Constant *C) {
  const GlobalValue *GV = dyn_cast<GlobalValue>(C);
  if (!GV)
    return false;
  unsigned AS = GV->getAddressSpace();
  return AS == AMDGPUAS::LOCAL_ADDRESS || AS == AMDGPUAS::REGION_ADDRESS;
}

bool AMDGPUAnnotateKernelFeatures::visitConstantExpr(const ConstantExpr *CE) {
  if (CE->getOpcode() == Instruction::AddrSpaceCast) {
    unsigned SrcAS = CE->getOperand(0)->getType()->getPointerAddressSpace();
    return castRequiresQueuePtr(SrcAS);
  }

  return false;
}

bool AMDGPUAnnotateKernelFeatures::visitConstantExprsRecursively(
  const Constant *EntryC,
  SmallPtrSet<const Constant *, 8> &ConstantExprVisited,
  bool IsFunc, bool HasApertureRegs) {

  if (!ConstantExprVisited.insert(EntryC).second)
    return false;

  SmallVector<const Constant *, 16> Stack;
  Stack.push_back(EntryC);

  while (!Stack.empty()) {
    const Constant *C = Stack.pop_back_val();

    // We need to trap on DS globals in non-entry functions.
    if (IsFunc && isDSAddress(C))
      return true;

    // Check this constant expression.
    if (const auto *CE = dyn_cast<ConstantExpr>(C)) {
      if (!HasApertureRegs && visitConstantExpr(CE))
        return true;
    }

    // Visit all sub-expressions.
    for (const Use &U : C->operands()) {
      const auto *OpC = dyn_cast<Constant>(U);
      if (!OpC)
        continue;

      if (!ConstantExprVisited.insert(OpC).second)
        continue;

      Stack.push_back(OpC);
    }
  }

  return false;
}

// We do not need to note the x workitem or workgroup id because they are always
// initialized.
//
// TODO: We should not add the attributes if the known compile time workgroup
// size is 1 for y/z.
static StringRef intrinsicToAttrName(Intrinsic::ID ID,
                                     bool &NonKernelOnly,
                                     bool &IsQueuePtr) {
  switch (ID) {
  case Intrinsic::amdgcn_workitem_id_x:
    NonKernelOnly = true;
    return "amdgpu-work-item-id-x";
  case Intrinsic::amdgcn_workgroup_id_x:
    NonKernelOnly = true;
    return "amdgpu-work-group-id-x";
  case Intrinsic::amdgcn_workitem_id_y:
  case Intrinsic::r600_read_tidig_y:
    return "amdgpu-work-item-id-y";
  case Intrinsic::amdgcn_workitem_id_z:
  case Intrinsic::r600_read_tidig_z:
    return "amdgpu-work-item-id-z";
  case Intrinsic::amdgcn_workgroup_id_y:
  case Intrinsic::r600_read_tgid_y:
    return "amdgpu-work-group-id-y";
  case Intrinsic::amdgcn_workgroup_id_z:
  case Intrinsic::r600_read_tgid_z:
    return "amdgpu-work-group-id-z";
  case Intrinsic::amdgcn_dispatch_ptr:
    return "amdgpu-dispatch-ptr";
  case Intrinsic::amdgcn_dispatch_id:
    return "amdgpu-dispatch-id";
  case Intrinsic::amdgcn_kernarg_segment_ptr:
    return "amdgpu-kernarg-segment-ptr";
  case Intrinsic::amdgcn_implicitarg_ptr:
    return "amdgpu-implicitarg-ptr";
  case Intrinsic::amdgcn_queue_ptr:
  case Intrinsic::amdgcn_is_shared:
  case Intrinsic::amdgcn_is_private:
    // TODO: Does not require queue ptr on gfx9+
  case Intrinsic::trap:
  case Intrinsic::debugtrap:
    IsQueuePtr = true;
    return "amdgpu-queue-ptr";
  default:
    return "";
  }
}

static bool handleAttr(Function &Parent, const Function &Callee,
                       StringRef Name) {
  if (Callee.hasFnAttribute(Name)) {
    Parent.addFnAttr(Name);
    return true;
  }
  return false;
}

static void copyFeaturesToFunction(Function &Parent, const Function &Callee,
                                   bool &NeedQueuePtr) {
  // X ids unnecessarily propagated to kernels.
  static constexpr StringLiteral AttrNames[] = {
      "amdgpu-work-item-id-x",      "amdgpu-work-item-id-y",
      "amdgpu-work-item-id-z",      "amdgpu-work-group-id-x",
      "amdgpu-work-group-id-y",     "amdgpu-work-group-id-z",
      "amdgpu-dispatch-ptr",        "amdgpu-dispatch-id",
      "amdgpu-kernarg-segment-ptr", "amdgpu-implicitarg-ptr"};

  if (handleAttr(Parent, Callee, "amdgpu-queue-ptr"))
    NeedQueuePtr = true;

  for (StringRef AttrName : AttrNames)
    handleAttr(Parent, Callee, AttrName);
}

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
  const GCNSubtarget &ST = TM->getSubtarget<GCNSubtarget>(F);
  bool HasApertureRegs = ST.hasApertureRegs();
  SmallPtrSet<const Constant *, 8> ConstantExprVisited;

  bool Changed = false;
  bool NeedQueuePtr = false;
  bool HaveCall = false;
  bool IsFunc = !AMDGPU::isEntryFunctionCC(F.getCallingConv());

  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      CallSite CS(&I);
      if (CS) {
        Function *Callee = CS.getCalledFunction();

        // TODO: Do something with indirect calls.
        if (!Callee) {
          if (!CS.isInlineAsm())
            HaveCall = true;
          continue;
        }

        Intrinsic::ID IID = Callee->getIntrinsicID();
        if (IID == Intrinsic::not_intrinsic) {
          HaveCall = true;
          copyFeaturesToFunction(F, *Callee, NeedQueuePtr);
          Changed = true;
        } else {
          bool NonKernelOnly = false;
          StringRef AttrName = intrinsicToAttrName(IID,
                                                   NonKernelOnly, NeedQueuePtr);
          if (!AttrName.empty() && (IsFunc || !NonKernelOnly)) {
            F.addFnAttr(AttrName);
            Changed = true;
          }
        }
      }

      if (NeedQueuePtr || (!IsFunc && HasApertureRegs))
        continue;

      if (const AddrSpaceCastInst *ASC = dyn_cast<AddrSpaceCastInst>(&I)) {
        if (!HasApertureRegs && castRequiresQueuePtr(ASC)) {
          NeedQueuePtr = true;
          continue;
        }
      }

      for (const Use &U : I.operands()) {
        const auto *OpC = dyn_cast<Constant>(U);
        if (!OpC)
          continue;

        if (visitConstantExprsRecursively(OpC, ConstantExprVisited, IsFunc,
                                          HasApertureRegs)) {
          NeedQueuePtr = true;
          break;
        }
      }
    }
  }

  if (NeedQueuePtr) {
    F.addFnAttr("amdgpu-queue-ptr");
    Changed = true;
  }

  // TODO: We could refine this to captured pointers that could possibly be
  // accessed by flat instructions. For now this is mostly a poor way of
  // estimating whether there are calls before argument lowering.
  if (!IsFunc && HaveCall) {
    F.addFnAttr("amdgpu-calls");
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
    // Add feature attributes
    if (!F || F->isDeclaration())
      continue;
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
