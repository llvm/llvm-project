//===- Transforms/Instrumentation/CGProfile.h -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file provides the interface for LLVM's Call Graph Profile pass.
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_CGPROFILE_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_CGPROFILE_H

#include "llvm/ADT/MapVector.h"
#include "llvm/IR/PassManager.h"

namespace llvm {
class CGProfilePass : public PassInfoMixin<CGProfilePass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};
} // end namespace llvm

#endif // LLVM_TRANSFORMS_INSTRUMENTATION_CGPROFILE_H
