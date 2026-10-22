//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InstrumentationOptions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/Option/LibraryOptions.h"

#define OPTIONS_STRUCT_DEFS
#include "InstrumentationOptions.inc"

static llvm::opt::RegisterLibraryOptions<llvm::InstrumentationOptions>
    Registration;

namespace llvm {

const InstrumentationOptions &getInstrumentationOptions(const LLVMContext &C) {
  return C.getLibraryOptions().get<InstrumentationOptions>();
}

const InstrumentationOptions &getInstrumentationOptions(const Module &M) {
  return getInstrumentationOptions(M.getContext());
}

const InstrumentationOptions &getInstrumentationOptions(const Value &V) {
  return getInstrumentationOptions(V.getContext());
}

} // namespace llvm
