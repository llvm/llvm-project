//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AnalysisOptions.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Module.h"
#include "llvm/Option/LibraryOptions.h"

#define OPTIONS_STRUCT_DEFS
#include "AnalysisOptions.inc"

using namespace llvm;

static llvm::opt::RegisterLibraryOptions<llvm::AnalysisOptions> Registration;

const AnalysisOptions &llvm::getAnalysisOptions(const Module &M) {
  return getAnalysisOptions(M.getContext());
}

const AnalysisOptions &llvm::getAnalysisOptions(const Loop &L) {
  return getAnalysisOptions(L.getHeader()->getContext());
}
