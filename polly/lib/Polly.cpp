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
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

namespace {

/// Initialize Polly passes when library is loaded.
///
/// We use the constructor of a statically declared object to initialize the
/// different Polly passes right after the Polly library is loaded. This ensures
/// that the Polly passes are available e.g. in the 'opt' tool.
class StaticInitializer {
public:
  StaticInitializer() {
    llvm::PassRegistry &Registry = *llvm::PassRegistry::getPassRegistry();
    polly::initializePollyPasses(Registry);
  }
};
static StaticInitializer InitializeEverything;
} // end of anonymous namespace.
