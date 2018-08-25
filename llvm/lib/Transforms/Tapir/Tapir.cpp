//===-- Tapir.cpp ---------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements common infrastructure for libLLVMTapirOpts.a, which
// implements several transformations over the Tapir/LLVM intermediate
// representation, including the C bindings for that library.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Tapir.h"
#include "llvm-c/Initialization.h"
#include "llvm-c/Transforms/Tapir.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/IR/LegacyPassManager.h"

using namespace llvm;

/// initializeTapirOpts - Initialize all passes linked into the
/// TapirOpts library.
void llvm::initializeTapirOpts(PassRegistry &Registry) {
  initializeLoopSpawningPass(Registry);
  initializeLoopSpawningTIPass(Registry);
  initializeLowerTapirToTargetPass(Registry);
}

void LLVMInitializeTapirOpts(LLVMPassRegistryRef R) {
  initializeTapirOpts(*unwrap(R));
}

/*
void LLVMAddLoopSpawningPass(LLVMPassManagerRef PM, TapirTargetID ID) {
  unwrap(PM)->add(createLoopSpawningPass(ID));
}

void LLVMAddLowerTapirToTargetPass(LLVMPassManagerRef PM, TapirTargetID ID) {
  unwrap(PM)->add(createLowerTapirToTargetPass(ID));
}
*/
