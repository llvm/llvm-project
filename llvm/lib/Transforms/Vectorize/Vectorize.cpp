//===-- Vectorize.cpp -----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements common infrastructure for libLLVMVectorizeOpts.a, which
// implements several vectorization transformations over the LLVM intermediate
// representation, including the C bindings for that library.
//
//===----------------------------------------------------------------------===//

#include "llvm-c/Transforms/Vectorize.h"
#include "llvm-c/Initialization.h"
#include "llvm/InitializePasses.h"
#include "llvm/PassManager.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Transforms/Vectorize.h"

using namespace llvm;

/// initializeVectorizationPasses - Initialize all passes linked into the
/// Vectorization library.
void llvm::initializeVectorization(PassRegistry &Registry) {
  initializeBBVectorizePass(Registry);
  initializeLoopVectorizePass(Registry);
}

void LLVMInitializeVectorization(LLVMPassRegistryRef R) {
  initializeVectorization(*unwrap(R));
}

void LLVMAddBBVectorizePass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createBBVectorizePass());
}

void LLVMAddLoopVectorizePass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createLoopVectorizePass());
}
