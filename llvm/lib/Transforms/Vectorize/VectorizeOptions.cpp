//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "VectorizeOptions.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Option/LibraryOptions.h"
#include "llvm/SandboxIR/Context.h"
#include "llvm/Transforms/Vectorize/LoopIdiomVectorize.h"

#define OPTIONS_STRUCT_DEFS
#include "VectorizeOptions.inc"

using namespace llvm;

static llvm::opt::RegisterLibraryOptions<llvm::VectorizeOptions> Registration;

const VectorizeOptions &llvm::getVectorizeOptions(const Loop &L) {
  return getVectorizeOptions(L.getHeader()->getContext());
}

const VectorizeOptions &llvm::getVectorizeOptions(const sandboxir::Context &C) {
  return C.getLibraryOptions().get<VectorizeOptions>();
}
