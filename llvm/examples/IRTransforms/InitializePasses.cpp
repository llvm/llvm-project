//===-- InitializePasses.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements implements the initialization hook for the example
// transforms.
//
//===----------------------------------------------------------------------===//

#include "InitializePasses.h"
#include "llvm/PassRegistry.h"

using namespace llvm;

void initializeExampleIRTransforms(PassRegistry &Registry) {
  initializeSimplifyCFGLegacyPassPass(Registry);
}
