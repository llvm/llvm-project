//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CodeGenOptions.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Option/LibraryOptions.h"

#define OPTIONS_STRUCT_DEFS
#include "CodeGenOptions.inc"

using namespace llvm;

static llvm::opt::RegisterLibraryOptions<llvm::CodeGenOptions> Registration;

const CodeGenOptions &llvm::getCodeGenOptions(const LLVMContext &C) {
  return C.getLibraryOptions().get<CodeGenOptions>();
}

const CodeGenOptions &llvm::getCodeGenOptions(const Module &M) {
  return getCodeGenOptions(M.getContext());
}

const CodeGenOptions &llvm::getCodeGenOptions(const MachineFunction &MF) {
  return getCodeGenOptions(MF.getFunction().getContext());
}
