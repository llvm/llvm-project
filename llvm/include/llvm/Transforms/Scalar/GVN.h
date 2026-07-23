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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/PHITransAddr.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Transforms/Scalar/GVNLeaderMap.h"
#include "llvm/Transforms/Scalar/GVNValueTable.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace llvm {

class AAResults;
class AssumeInst;
class AssumptionCache;
class BasicBlock;
class BatchAAResults;
class CallInst;
class CondBrInst;
class EarliestEscapeAnalysis;
class ExtractValueInst;
class Function;
class FunctionPass;
class GVNLegacyPass;
class GetElementPtrInst;
class ImplicitControlFlowTracking;
class LoadInst;
class LoopInfo;
class MemDepResult;
class MemoryAccess;
class MemoryDependenceResults;
class MemoryLocation;
class MemorySSA;
class MemorySSAUpdater;
class NonLocalDepResult;
class OptimizationRemarkEmitter;
class PHINode;
class TargetLibraryInfo;
class Value;
class IntrinsicInst;

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

/// The core GVN pass object.
///
/// FIXME: We should have a good summary of the GVN algorithm implemented by
/// this particular pass here.
class GVNPassImpl;
class GVNPass : public OptionalPassInfoMixin<GVNPass> {
  std::unique_ptr<GVNPassImpl> Impl;

public:
  GVNPass(GVNOptions Options = {});
  ~GVNPass();

  GVNPass(const GVNPass &) = delete;
  GVNPass(GVNPass &&);
  GVNPass &operator=(const GVNPass &) = delete;
  GVNPass &operator=(GVNPass &&);

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
