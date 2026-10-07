//===- GVN.h - Eliminate redundant values and loads -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file provides the interface for LLVM's Global Value Numbering pass
/// which eliminates fully redundant instructions. It also does somewhat Ad-Hoc
/// PRE and dead load elimination.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_GVN_H
#define LLVM_TRANSFORMS_SCALAR_GVN_H

#include "llvm/IR/PassManager.h"
#include <optional>

namespace llvm {

class FunctionPass;

/// A set of parameters to control various transforms performed by GVN pass.
//  Each of the optional boolean parameters can be set to:
///      true - enabling the transformation.
///      false - disabling the transformation.
///      None - relying on a global default.
/// Intended use is to create a default object, modify parameters with
/// additional setters and then pass it to GVN.
struct GVNOptions {
  std::optional<bool> AllowScalarPRE;
  std::optional<bool> AllowLoadPRE;
  std::optional<bool> AllowLoadInLoopPRE;
  std::optional<bool> AllowLoadPRESplitBackedge;
  std::optional<bool> AllowMemDep;
  std::optional<bool> AllowMemorySSA;

  GVNOptions() = default;

  /// Enables or disables PRE of scalars in GVN.
  GVNOptions &setScalarPRE(bool ScalarPRE) {
    AllowScalarPRE = ScalarPRE;
    return *this;
  }

  /// Enables or disables PRE of loads in GVN.
  GVNOptions &setLoadPRE(bool LoadPRE) {
    AllowLoadPRE = LoadPRE;
    return *this;
  }

  GVNOptions &setLoadInLoopPRE(bool LoadInLoopPRE) {
    AllowLoadInLoopPRE = LoadInLoopPRE;
    return *this;
  }

  /// Enables or disables PRE of loads in GVN.
  GVNOptions &setLoadPRESplitBackedge(bool LoadPRESplitBackedge) {
    AllowLoadPRESplitBackedge = LoadPRESplitBackedge;
    return *this;
  }

  /// Enables or disables use of MemDepAnalysis.
  GVNOptions &setMemDep(bool MemDep) {
    AllowMemDep = MemDep;
    return *this;
  }

  /// Enables or disables use of MemorySSA.
  GVNOptions &setMemorySSA(bool MemSSA) {
    AllowMemorySSA = MemSSA;
    return *this;
  }
};

class GVNPass : public OptionalPassInfoMixin<GVNPass> {
  GVNOptions Options;

public:
  GVNPass(GVNOptions Options = {}) : Options(Options) {}

  /// Run the pass over the function.
  LLVM_ABI PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  LLVM_ABI void
  printPipeline(raw_ostream &OS,
                function_ref<StringRef(StringRef)> MapClassName2PassName);
};

/// Create a legacy GVN pass.
LLVM_ABI FunctionPass *createGVNPass(bool ScalarPRE);
LLVM_ABI FunctionPass *createGVNPass();

} // end namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_GVN_H
