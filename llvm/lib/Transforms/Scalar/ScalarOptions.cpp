//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ScalarOptions.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Option/LibraryOptions.h"

#define OPTIONS_STRUCT_DEFS
#include "ScalarOptions.inc"

using namespace llvm;

static llvm::opt::RegisterLibraryOptions<llvm::ScalarOptions> Registration;

const ScalarOptions &llvm::getScalarOptions(const Loop &L) {
  return getScalarOptions(L.getHeader()->getContext());
}
