//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_CODEGENOPTIONS_H
#define LLVM_LIB_CODEGEN_CODEGENOPTIONS_H

#include "llvm/Target/TargetOptions.h"
#include <cstdint>
#include <optional>

#define OPTIONS_STRUCT_DECL
#include "CodeGenOptions.inc"

namespace llvm {

class LLVMContext;
class MachineFunction;
class Module;

const CodeGenOptions &getCodeGenOptions(const LLVMContext &C);
const CodeGenOptions &getCodeGenOptions(const Module &M);
const CodeGenOptions &getCodeGenOptions(const MachineFunction &MF);

} // namespace llvm

#endif // LLVM_LIB_CODEGEN_CODEGENOPTIONS_H
