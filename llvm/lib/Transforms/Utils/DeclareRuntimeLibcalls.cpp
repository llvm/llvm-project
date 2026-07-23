//===- DeclareRuntimeLibcalls.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Insert declarations for all runtime library calls known for the target.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/DeclareRuntimeLibcalls.h"
#include "llvm/Analysis/RuntimeLibcallInfo.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/RuntimeLibcalls.h"

using namespace llvm;

static void mergeAttributes(LLVMContext &Ctx, const Module &M,
                            const DataLayout &DL, const Triple &TT,
                            Function *Func, FunctionType *FuncTy,
                            AttributeList FuncAttrs) {
  AttributeList OldAttrs = Func->getAttributes();
  AttributeList NewAttrs = OldAttrs;

  {
    AttrBuilder OldBuilder(Ctx, OldAttrs.getFnAttrs());
    AttrBuilder NewBuilder(Ctx, FuncAttrs.getFnAttrs());
    OldBuilder.merge(NewBuilder);
    NewAttrs = NewAttrs.addFnAttributes(Ctx, OldBuilder);
  }

  {
    AttrBuilder OldBuilder(Ctx, OldAttrs.getRetAttrs());
    AttrBuilder NewBuilder(Ctx, FuncAttrs.getRetAttrs());
    OldBuilder.merge(NewBuilder);
    NewAttrs = NewAttrs.addRetAttributes(Ctx, OldBuilder);
  }

  for (unsigned I = 0, E = FuncTy->getNumParams(); I != E; ++I) {
    AttrBuilder OldBuilder(Ctx, OldAttrs.getParamAttrs(I));
    AttrBuilder NewBuilder(Ctx, FuncAttrs.getParamAttrs(I));
    OldBuilder.merge(NewBuilder);
    NewAttrs = NewAttrs.addParamAttributes(Ctx, I, OldBuilder);
  }

  Func->setAttributes(NewAttrs);
}

PreservedAnalyses DeclareRuntimeLibcallsPass::run(Module &M,
                                                  ModuleAnalysisManager &MAM) {
  const RTLIB::RuntimeLibcallsInfo &RTLCI =
      MAM.getResult<RuntimeLibraryAnalysis>(M);

  LLVMContext &Ctx = M.getContext();
  const DataLayout &DL = M.getDataLayout();
  const Triple &TT = M.getTargetTriple();

  for (RTLIB::LibcallImpl Impl : RTLIB::libcall_impls()) {
    if (!RTLCI.isAvailable(Impl))
      continue;

    auto [FuncTy, FuncAttrs] = RTLCI.getFunctionTy(Ctx, TT, DL, Impl);

    // TODO: Declare with correct type, calling convention, and attributes.
    if (!FuncTy)
      FuncTy = FunctionType::get(Type::getVoidTy(Ctx), {}, /*IsVarArgs=*/true);

    StringRef FuncName = RTLCI.getLibcallImplName(Impl);

    Function *Func =
        cast<Function>(M.getOrInsertFunction(FuncName, FuncTy).getCallee());
    if (Func->getFunctionType() == FuncTy) {
      mergeAttributes(Ctx, M, DL, TT, Func, FuncTy, FuncAttrs);
      Func->setCallingConv(RTLCI.getLibcallImplCallingConv(Impl));
    }
  }

  return PreservedAnalyses::none();
}
