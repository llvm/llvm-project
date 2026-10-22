//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_CODEGENOPTIONS_H
#define LLVM_LIB_CODEGEN_CODEGENOPTIONS_H

#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Target/TargetOptions.h"
#include <cstdint>
#include <optional>

#define OPTIONS_STRUCT_DECL
#include "CodeGenOptions.inc"

namespace llvm {

inline const CodeGenOptions &getCodeGenOptions(const LLVMContext &C) {
  return C.getLibraryOptions().get<CodeGenOptions>();
}

inline const CodeGenOptions &getCodeGenOptions(const Function &F) {
  return getCodeGenOptions(F.getContext());
}

inline const CodeGenOptions &getCodeGenOptions(const Module &M) {
  return getCodeGenOptions(M.getContext());
}

inline const CodeGenOptions &getCodeGenOptions(const MachineFunction &MF) {
  return getCodeGenOptions(MF.getFunction());
}

} // namespace llvm

#endif // LLVM_LIB_CODEGEN_CODEGENOPTIONS_H
