//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InstCombineCLOptions.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Option/LibraryOptions.h"

#define OPTIONS_STRUCT_DEFS
#include "InstCombineCLOptions.inc"

using namespace llvm;

static llvm::opt::RegisterLibraryOptions<llvm::InstCombineCLOptions>
    Registration;

const InstCombineCLOptions &llvm::getInstCombineCLOptions(const Function &F) {
  return F.getContext().getLibraryOptions().get<InstCombineCLOptions>();
}
