//===- MergeFunctionsIgnoringConst.h - Merge Functions ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass transforms simple global variables that never have their address
// taken.  If obviously true, it marks read/write globals as constant, deletes
// variables only stored to, etc.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_MERGEFUNCTIONSIGNORINGCONST_H
#define LLVM_TRANSFORMS_IPO_MERGEFUNCTIONSIGNORINGCONST_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class Module;

/// Merge functions that differ by constants.
class MergeFuncIgnoringConstPass
    : public PassInfoMixin<MergeFuncIgnoringConstPass> {
  std::string MergeFuncSuffix = ".Tm";

public:
  MergeFuncIgnoringConstPass() {}
  MergeFuncIgnoringConstPass(std::string Suffix) : MergeFuncSuffix(Suffix) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_IPO_MERGEFUNCTIONSIGNORINGCONST_H
