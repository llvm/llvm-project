//===- AMDGPUExportKernelRuntimeHandles.cpp - Lower enqueued block --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file
//
// Give any globals used for OpenCL block enqueue runtime handles external
// linkage so the runtime may access them. These should behave like internal
// functions for purposes of linking, but need to have an external symbol in the
// final object for the runtime to access them.
//
// TODO: This could be replaced with a new linkage type or global object
// metadata that produces an external symbol in the final object, but allows
// rename on IR linking. Alternatively if we can rely on
// GlobalValue::getGlobalIdentifier we can just make these external symbols to
// begin with.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUExportKernelRuntimeHandles.h"
#include "AMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"

#define DEBUG_TYPE "amdgpu-export-kernel-runtime-handles"

using namespace llvm;

namespace {

/// Lower enqueued blocks.
class AMDGPUExportKernelRuntimeHandlesLegacy : public ModulePass {
public:
  static char ID;

  explicit AMDGPUExportKernelRuntimeHandlesLegacy() : ModulePass(ID) {}

private:
  bool runOnModule(Module &M) override;
};

} // end anonymous namespace

char AMDGPUExportKernelRuntimeHandlesLegacy::ID = 0;

char &llvm::AMDGPUExportKernelRuntimeHandlesLegacyID =
    AMDGPUExportKernelRuntimeHandlesLegacy::ID;

INITIALIZE_PASS(AMDGPUExportKernelRuntimeHandlesLegacy, DEBUG_TYPE,
                "Externalize enqueued block runtime handles", false, false)

ModulePass *llvm::createAMDGPUExportKernelRuntimeHandlesLegacyPass() {
  return new AMDGPUExportKernelRuntimeHandlesLegacy();
}

static bool exportKernelRuntimeHandles(Module &M) {
  bool Changed = false;

  const StringLiteral HandleSectionName(".amdgpu.kernel.runtime.handle");

  for (GlobalVariable &GV : M.globals()) {
    if (GV.getSection() == HandleSectionName) {
      GV.setLinkage(GlobalValue::ExternalLinkage);
      GV.setDSOLocal(false);
      Changed = true;
    }
  }

  if (!Changed)
    return false;

  // FIXME: We shouldn't really need to export the kernel address. We can
  // initialize the runtime handle with the kernel descriptorG
  for (Function &F : M) {
    if (F.getCallingConv() != CallingConv::AMDGPU_KERNEL)
      continue;

    const MDNode *Associated = F.getMetadata(LLVMContext::MD_associated);
    if (!Associated)
      continue;

    auto *VM = cast<ValueAsMetadata>(Associated->getOperand(0));
    auto *Handle = dyn_cast<GlobalObject>(VM->getValue());
    if (Handle && Handle->getSection() == HandleSectionName) {
      F.setLinkage(GlobalValue::ExternalLinkage);
      F.setVisibility(GlobalValue::ProtectedVisibility);
    }
  }

  return Changed;
}

bool AMDGPUExportKernelRuntimeHandlesLegacy::runOnModule(Module &M) {
  return exportKernelRuntimeHandles(M);
}

PreservedAnalyses
AMDGPUExportKernelRuntimeHandlesPass::run(Module &M,
                                          ModuleAnalysisManager &MAM) {
  if (!exportKernelRuntimeHandles(M))
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<AllAnalysesOn<Function>>();
  return PA;
}
