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
#include "llvm/IR/Module.h"
#include "llvm/IR/RuntimeLibcalls.h"

using namespace llvm;

PreservedAnalyses DeclareRuntimeLibcallsPass::run(Module &M,
                                                  ModuleAnalysisManager &MAM) {
  RTLIB::RuntimeLibcallsInfo RTLCI(M.getTargetTriple());
  LLVMContext &Ctx = M.getContext();

  for (RTLIB::LibcallImpl Impl : RTLCI.getLibcallImpls()) {
    if (Impl == RTLIB::Unsupported)
      continue;

    // TODO: Declare with correct type, calling convention, and attributes.

    FunctionType *FuncTy =
        FunctionType::get(Type::getVoidTy(Ctx), {}, /*IsVarArgs=*/true);

    const char *FuncName = RTLCI.getLibcallImplName(Impl);
    M.getOrInsertFunction(FuncName, FuncTy);
  }

  return PreservedAnalyses::none();
}
