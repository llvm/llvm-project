//==------ llvm/CodeGen/LowerEmuTLS.h -------------------------*- C++ -*----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file Add Add __emutls_[vt].* variables.
///
/// This file provide declaration of LowerEmuTLSPass.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_LOWEREMUTLS_H
#define LLVM_CODEGEN_LOWEREMUTLS_H

#include "llvm/IR/PassManager.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class LowerEmuTLSPass : public PassInfoMixin<LowerEmuTLSPass> {
public:
  LLVM_ABI PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};

} // namespace llvm

#endif // LLVM_CODEGEN_LOWEREMUTLS_H
