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

// @foo = ifunc i32 (), ptr @foo_resolver, !associated !0
// define ptr @foo_resolver() {
//  ...
//
// %struct.IFUNC_PAIR = type { ptr, ptr }
// @update_foo = internal global %struct.IFUNC_PAIR { ptr @foo, ptr
// @foo_resolver }, section "ifunc_sec", align 8, !associated !1 declare void
// @__init_ifuncs(...)
//
// !0 = !{ptr @update_foo}
// !1 = !{ptr @__init_ifuncs}
bool PPCPrepareIFuncsOnAIX::runOnModule(Module &M) {
  if (M.ifuncs().empty())
    return false;

  const DataLayout &DL = M.getDataLayout();
  LLVMContext &Ctx = M.getContext();
  auto *PtrTy = PointerType::getUnqual(Ctx);
  StringRef IFuncUpdatePrefix = "__update_";
  StringRef IFuncUpdateSectionName = "ifunc_sec";
  StructType *IFuncPairType = StructType::get(PtrTy, PtrTy);

  StringRef IFuncConstructorName = "__init_ifuncs";
  auto *IFuncConstructorFnType =
      FunctionType::get(Type::getVoidTy(Ctx), {}, /*isVarArg=*/false);
  auto *IFuncConstructorDecl =
      Function::Create(IFuncConstructorFnType, GlobalValue::ExternalLinkage,
                       IFuncConstructorName, M);

  for (GlobalIFunc &IFunc : M.ifuncs()) {
    NumIFuncs++;
    LLVM_DEBUG(dbgs() << "doing ifunc " << IFunc.getName() << "\n");
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

    // Associate liveness of function foo with the liveness of update_foo.
    IFunc.setMetadata(LLVMContext::MD_associated,
                      MDNode::get(Ctx, ValueAsMetadata::get(GV)));
    // Make function foo depend on the constructor that calls each ifunc's
    // resolver and updaTes the ifunc's function descriptor with the result.
    // Note: technically, we can associate both the update_foo variable and
    // the constructor function to function foo, but only one MD_associated
    // is allowed on an llvm::Value, so associate the constructor to update_foo
    // here.
    GV->setMetadata(
        LLVMContext::MD_associated,
        MDNode::get(Ctx, ValueAsMetadata::get(IFuncConstructorDecl)));
  }

  return true;
}
