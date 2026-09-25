//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CodeGenTestPass.h"

#include <llvm/CodeGen/Passes.h>
#include <llvm/CodeGen/TargetPassConfig.h>
#include <llvm/Plugins/PassPlugin.h>
#include <llvm/Target/RegisterTargetPassConfigCallback.h>

using namespace llvm;

namespace {
[[maybe_unused]] RegisterTargetPassConfigCallback X{
    [](auto &TM, auto &PM, auto *TPC) {
      TPC->insertPass(&GCLoweringID, &CodeGenTest::ID);
    }};
} // namespace

__attribute__((constructor)) static void initCodeGenPlugin() {
  initializeCodeGenTestPass(*PassRegistry::getPassRegistry());
}

// We use -load-pass-plugin to specify and load the plugin shared-lib, but not
// the actual pass-plugin interface. This isn't available in the codegen
// pipeline, as it doesn't use the New PassManager.
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "CGTestPlugin", LLVM_VERSION_STRING, nullptr,
          nullptr};
}
