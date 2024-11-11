//===- XRayPreparation.cpp - Preparation for XRay instrumentation -------- ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This Pass does some IR-level preparations (e.g. inserting global variable
// that carries default options, if there is any) for XRay instrumentation.
//
//===---------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/XRayPreparation.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

static void createXRayDefaultOptionsVar(Module &M, StringRef DefaultOptions) {
  Constant *DefaultOptionsConst =
      ConstantDataArray::getString(M.getContext(), DefaultOptions, true);
  // This global variable will be passed into XRay's compiler-rt and used as
  // the initial set of XRay options.
  GlobalVariable *DefaultOptsVar = new GlobalVariable(
      M, DefaultOptionsConst->getType(), true, GlobalValue::WeakAnyLinkage,
      DefaultOptionsConst, "__llvm_xray_options");
  DefaultOptsVar->setVisibility(GlobalValue::HiddenVisibility);
  Triple TT(M.getTargetTriple());
  if (TT.supportsCOMDAT()) {
    DefaultOptsVar->setLinkage(GlobalValue::ExternalLinkage);
    DefaultOptsVar->setComdat(M.getOrInsertComdat("__llvm_xray_options"));
  }
}

PreservedAnalyses XRayPreparationPass::run(Module &M,
                                           ModuleAnalysisManager &MAM) {
  // XRay default options.
  if (const auto *DefaultOpts =
          dyn_cast_or_null<MDString>(M.getModuleFlag("xray-default-opts"))) {
    createXRayDefaultOptionsVar(M, DefaultOpts->getString());
    PreservedAnalyses PA;
    PA.preserveSet<CFGAnalyses>();
    return PA;
  }

  return PreservedAnalyses::all();
}
