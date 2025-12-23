//===---------- Polly.cpp - Initialize the Polly Module -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "polly/RegisterPasses.h"
#include "llvm/PassRegistry.h"
#include "llvm/Passes/PassPlugin.h"

// Pass Plugin Entrypoints

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getPollyPluginInfo();
}
