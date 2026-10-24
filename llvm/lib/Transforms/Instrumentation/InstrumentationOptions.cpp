//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InstrumentationOptions.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Option/LibraryOptions.h"

#define OPTIONS_STRUCT_DEFS
#include "InstrumentationOptions.inc"

using namespace llvm;

static llvm::opt::RegisterLibraryOptions<llvm::InstrumentationOptions>
    Registration;

const InstrumentationOptions &
llvm::getInstrumentationOptions(const LLVMContext &C) {
  return C.getLibraryOptions().get<InstrumentationOptions>();
}

const InstrumentationOptions &llvm::getInstrumentationOptions(const Module &M) {
  return getInstrumentationOptions(M.getContext());
}

const InstrumentationOptions &
llvm::getInstrumentationOptions(const Function &F) {
  return getInstrumentationOptions(F.getContext());
}
