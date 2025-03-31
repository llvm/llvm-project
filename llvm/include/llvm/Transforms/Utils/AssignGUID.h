//===-- AssignGUID.h - Unique identifier assignment pass --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides a pass which assigns a a GUID (globally unique identifier)
// to every GlobalValue in the module, according to its current name, linkage,
// and originating file.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_ASSIGNGUID_H
#define LLVM_TRANSFORMS_UTILS_ASSIGNGUID_H

#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class AssignGUIDPass : public PassInfoMixin<AssignGUIDPass> {
public:
  AssignGUIDPass() = default;

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

  static bool isRequired() { return true; }
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_ASSIGNGUID_H