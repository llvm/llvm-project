//===-- PPCPrepareIFuncsOnAIX.cpp - Prepare for ifunc lowering in codegen ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass generates...
//
//===----------------------------------------------------------------------===//

#include "PPC.h"
#include "PPCSubtarget.h"
#include "PPCTargetMachine.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include <cassert>

using namespace llvm;

#define DEBUG_TYPE "ppc-prep-ifunc-aix"

STATISTIC(NumIFuncs, "Number of IFuncs prepared");

namespace {
class PPCPrepareIFuncsOnAIX : public ModulePass {
public:
  static char ID;

  PPCPrepareIFuncsOnAIX() : ModulePass(ID) {}

  bool runOnModule(Module &M) override;

  StringRef getPassName() const override {
    return "PPC Prepare for AIX IFunc lowering";
  }
};
} // namespace

char PPCPrepareIFuncsOnAIX::ID = 0;

INITIALIZE_PASS(PPCPrepareIFuncsOnAIX, DEBUG_TYPE,
                "PPC Prepare for AIX IFunc lowering", false, false)

ModulePass *llvm::createPPCPrepareIFuncsOnAIXPass() {
  return new PPCPrepareIFuncsOnAIX();
}

// For each ifunc `foo` with a resolver `foo_resolver`, create a global variable
// `__update_foo` in the `ifunc_sec` section, representing the pair:
//   { ptr @foo, ptr @foo_resolver }
// The compiler arranges for the constructor function `__init_ifuncs` to be
// included on the link step. The constructor walks the `ifunc_sec` section,
// calling the resolver function and storing the result in foo's descriptor.
// On AIX, the address of a function is the address of its descriptor, so the
// constructor accesses foo's descriptor from the first field of the pair.
//
// Since the global `__update_foo` is unreferenced, it's liveness needs to be
// associated to the liveness of ifunc `foo`
//
bool PPCPrepareIFuncsOnAIX::runOnModule(Module &M) {
  if (M.ifuncs().empty())
    return false;

  const DataLayout &DL = M.getDataLayout();
  LLVMContext &Ctx = M.getContext();
  auto *PtrTy = PointerType::getUnqual(Ctx);
  StringRef IFuncUpdatePrefix = "__update_";
  StringRef IFuncUpdateSectionName = "__ifunc_sec";
  StructType *IFuncPairType = StructType::get(PtrTy, PtrTy);

  StringRef IFuncConstructorName = "__init_ifuncs";
  auto *IFuncConstructorFnType =
      FunctionType::get(Type::getVoidTy(Ctx), {}, /*isVarArg=*/false);
  auto *IFuncConstructorDecl = cast<Function>(
      M.getOrInsertFunction(IFuncConstructorName, IFuncConstructorFnType)
          .getCallee());

  for (GlobalIFunc &IFunc : M.ifuncs()) {
    NumIFuncs++;
    LLVM_DEBUG(dbgs() << "expanding ifunc " << IFunc.getName() << "\n");
    // @__update_foo = private global { ptr @foo, ptr @foo_resolver },
    //   section "ifunc_sec"
    std::string Name = (Twine(IFuncUpdatePrefix) + IFunc.getName()).str();
    auto *GV = new GlobalVariable(M, IFuncPairType, /*isConstant*/ false,
                                  GlobalValue::PrivateLinkage, nullptr, Name);
    GV->setAlignment(DL.getPointerPrefAlignment());
    GV->setSection(IFuncUpdateSectionName);

    // Note that on AIX, the address of a function is the address of it's
    // function descriptor, which is what these two values end up being
    // in assembly.
    Constant *InitVals[] = {&IFunc, IFunc.getResolver()};
    GV->setInitializer(ConstantStruct::get(IFuncPairType, InitVals));

    // Liveness of __update_foo is dependent on liveness of ifunc foo.
    IFunc.setMetadata(LLVMContext::MD_implicit_ref,
                      MDNode::get(Ctx, ValueAsMetadata::get(GV)));

    // An implicit.ref creates linkage dependency, so make function foo require
    // the constructor that calls each ifunc's resolver and saves the result in
    // the ifunc's function descriptor.
    IFunc.addMetadata(
        LLVMContext::MD_implicit_ref,
        *MDNode::get(Ctx, ValueAsMetadata::get(IFuncConstructorDecl)));
  }

  return true;
}
