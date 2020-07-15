//===-- ObjCARC.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements common infrastructure for libLLVMObjCARCOpts.a, which
// implements several scalar transformations over the LLVM intermediate
// representation, including the C bindings for that library.
//
//===----------------------------------------------------------------------===//

#include "ObjCARC.h"
#include "llvm-c/Initialization.h"
#include "llvm/InitializePasses.h"

namespace llvm {
  class PassRegistry;
}

using namespace llvm;
using namespace llvm::objcarc;

/// initializeObjCARCOptsPasses - Initialize all passes linked into the
/// ObjCARCOpts library.
void llvm::initializeObjCARCOpts(PassRegistry &Registry) {
  initializeObjCARCAAWrapperPassPass(Registry);
  initializeObjCARCAPElimPass(Registry);
  initializeObjCARCExpandPass(Registry);
  initializeObjCARCContractPass(Registry);
  initializeObjCARCOptLegacyPassPass(Registry);
  initializePAEvalPass(Registry);
}

void LLVMInitializeObjCARCOpts(LLVMPassRegistryRef R) {
  initializeObjCARCOpts(*unwrap(R));
}
