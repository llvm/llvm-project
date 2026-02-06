//===- OffloadLiveInValueCanonicalization.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass canonicalizes live-in values for regions destined for offloading.
//
// Overview:
// ---------
// When a region is outlined (extracted into a separate function for device
// execution), values defined outside the region but used inside become
// arguments to the outlined function. However, some values cannot be passed
// as arguments because they represent synthetic types (e.g., shape metadata,
// field indices) or are better handled by recreating them inside the region.
//
// This pass identifies such values and either:
// 1. Sinks the defining operation into the region (if all uses are inside)
// 2. Rematerializes (clones) the operation inside the region (if there are
//    uses both inside and outside)
//
// Transforms:
// -----------
// The pass performs two main transformations on live-in values:
//
// 1. Sinking: If a candidate operation's result is only used inside the
//    offload region, the operation is moved into the region.
//
// 2. Rematerialization: If a candidate operation's result is used both
//    inside and outside the region, the operation is cloned inside the
//    region and uses within the region are updated to use the clone.
//
// Candidate operations are:
// - Constants (matching arith.constant, etc.)
// - Operations implementing `acc::OutlineRematerializationOpInterface`
// - Address-of operations (`acc::AddressOfGlobalOpInterface`) referencing
//   symbols that are valid in GPU regions or constant globals
//
// The pass traces through view-like operations (`ViewLikeOpInterface`) and
// partial entity access operations (`acc::PartialEntityAccessOpInterface`)
// to find the original defining operation before making candidate decisions.
//
// Requirements:
// -------------
// To use this pass in a pipeline, the following requirements must be met:
//
// 1. Target Region Identification: Operations representing offload regions
//    must implement `acc::OffloadRegionOpInterface`. This interface marks
//    regions that will be outlined for device execution.
//
// 2. Rematerialization Candidates: Operations producing values that should
//    be rematerialized (rather than passed as arguments) should implement
//    `acc::OutlineRematerializationOpInterface`. Examples include operations
//    producing shape metadata, field indices, or other synthetic types.
//
// 3. Analysis Registration (Optional): If custom behavior is needed for
//    symbol validation (e.g., determining if a global is valid on device),
//    pre-register `acc::OpenACCSupport` analysis on the parent module.
//    If not registered, default behavior will be used.
//
// 4. View-Like Operations: Operations that create views or casts should
//    implement `ViewLikeOpInterface` or `acc::PartialEntityAccessOpInterface`
//    to allow the pass to trace through to the original defining operation.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenACC/Transforms/Passes.h"

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenACC/Analysis/OpenACCSupport.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/RegionUtils.h"

namespace mlir {
namespace acc {
#define GEN_PASS_DEF_OFFLOADLIVEINVALUECANONICALIZATION
#include "mlir/Dialect/OpenACC/Transforms/Passes.h.inc"
} // namespace acc
} // namespace mlir

#define DEBUG_TYPE "offload-livein-value-canonicalization"

using namespace mlir;

namespace {

/// Returns true if all users of the given value are inside the region.
static bool allUsersAreInsideRegion(Value val, Region &region) {
  for (Operation *user : val.getUsers())
    if (!region.isAncestor(user->getParentRegion()))
      return false;
  return true;
}

/// Traces through view-like and partial entity access operations to find the
/// original defining value.
static Value getOriginalValue(Value val) {
  Value prev;
  while (val && val != prev) {
    prev = val;
    if (auto viewLikeOp = val.getDefiningOp<ViewLikeOpInterface>())
      val = viewLikeOp.getViewSource();
    if (auto partialAccess =
            val.getDefiningOp<acc::PartialEntityAccessOpInterface>()) {
      Value base = partialAccess.getBaseEntity();
      if (base)
        val = base;
    }
  }
  return val;
}

/// Returns true if the operation is a candidate for rematerialization.
/// Candidates are operations that:
/// 1. Match the constant pattern (arith.constant, etc.)
/// 2. Implement OutlineRematerializationOpInterface
/// 3. Are address-of operations referencing valid symbols or constant globals
/// The function traces through view-like operations (casts, reinterpret_cast)
/// to find the original defining operation before making the determination.
static bool isRematerializationCandidate(Value val,
                                         acc::OpenACCSupport &accSupport) {
  // Trace through view-like operations to find the original value.
  Value origVal = getOriginalValue(val);
  Operation *definingOp = origVal.getDefiningOp();
  if (!definingOp)
    return false;

  LLVM_DEBUG(llvm::dbgs() << "\tChecking candidate: " << *definingOp << "\n");

  // Constants are trivial and useful to rematerialize.
  if (matchPattern(definingOp, m_Constant())) {
    LLVM_DEBUG(llvm::dbgs() << "\t\t-> constant pattern matched\n");
    return true;
  }

  // Operations implementing OutlineRematerializationOpInterface are candidates.
  if (isa<acc::OutlineRematerializationOpInterface>(definingOp)) {
    LLVM_DEBUG(llvm::dbgs() << "\t\t-> OutlineRematerializationOpInterface\n");
    return true;
  }

  // Address-of operations referencing globals that are valid in GPU regions
  // or referencing constant globals should be rematerialized.
  if (auto addrOfOp = dyn_cast<acc::AddressOfGlobalOpInterface>(definingOp)) {
    SymbolRefAttr symbol = addrOfOp.getSymbol();
    LLVM_DEBUG(llvm::dbgs()
               << "\t\tAddressOfGlobalOpInterface, symbol: " << symbol << "\n");

    // If the symbol is already valid in GPU regions (e.g., has acc.declare),
    // rematerializing ensures the address refers to the device copy.
    Operation *globalOp = nullptr;
    if (accSupport.isValidSymbolUse(definingOp, symbol, &globalOp)) {
      LLVM_DEBUG(llvm::dbgs() << "\t\t-> isValidSymbolUse: true\n");
      return true;
    }
    LLVM_DEBUG(llvm::dbgs() << "\t\t-> isValidSymbolUse: false\n");

    // If the referenced global is constant, prefer rematerialization so the
    // constant can be placed in GPU memory.
    if (globalOp) {
      if (auto globalVarOp =
              dyn_cast<acc::GlobalVariableOpInterface>(globalOp)) {
        if (globalVarOp.isConstant()) {
          LLVM_DEBUG(llvm::dbgs() << "\t\t-> constant global\n");
          return true;
        }
      }
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "\t\t-> not a candidate\n");
  return false;
}

class OffloadLiveInValueCanonicalization
    : public acc::impl::OffloadLiveInValueCanonicalizationBase<
          OffloadLiveInValueCanonicalization> {
public:
  using acc::impl::OffloadLiveInValueCanonicalizationBase<
      OffloadLiveInValueCanonicalization>::
      OffloadLiveInValueCanonicalizationBase;

  /// Canonicalizes live-in values for a region by sinking or rematerializing
  /// operations. Returns true if any changes were made.
  bool canonicalizeLiveInValues(Region &region,
                                acc::OpenACCSupport &accSupport) {
    // 1) Collect live-in values.
    SetVector<Value> liveInValues;
    getUsedValuesDefinedAbove(region, liveInValues);
    LLVM_DEBUG(llvm::dbgs()
               << "\tFound " << liveInValues.size() << " live-in value(s)\n");

    auto isSinkCandidate = [&region, &accSupport](Value val) -> bool {
      return isRematerializationCandidate(val, accSupport) &&
             allUsersAreInsideRegion(val, region);
    };
    auto isCloneCandidate = [&region, &accSupport](Value val) -> bool {
      return isRematerializationCandidate(val, accSupport) &&
             !allUsersAreInsideRegion(val, region);
    };

    // 2) Filter values into two sets - sink and rematerialization candidates.
    SmallVector<Value> sinkCandidates(
        llvm::make_filter_range(liveInValues, isSinkCandidate));
    SmallVector<Value> rematerializationCandidates(
        llvm::make_filter_range(liveInValues, isCloneCandidate));

    LLVM_DEBUG(llvm::dbgs() << "\tSink candidates: " << sinkCandidates.size()
                            << ", clone candidates: "
                            << rematerializationCandidates.size() << "\n");

    if (rematerializationCandidates.empty() && sinkCandidates.empty())
      return false;

    LLVM_DEBUG(llvm::dbgs() << "\tCanonicalizing values into "
                            << *region.getParentOp() << "\n");

    // 3) Handle the sink set by moving the operations into the region.
    for (Value sinkCandidate : sinkCandidates) {
      Operation *sinkOp = sinkCandidate.getDefiningOp();
      assert(sinkOp && "must have op to be considered");
      sinkOp->moveBefore(&region.front().front());
      LLVM_DEBUG(llvm::dbgs() << "\t\tSunk: " << *sinkOp << "\n");
    }

    // 4) Handle the rematerialization set by copying the operations into
    // the region.
    OpBuilder builder(region);
    SmallVector<Operation *> opsToRematerialize;
    for (Value rematerializationCandidate : rematerializationCandidates) {
      Operation *rematerializationOp =
          rematerializationCandidate.getDefiningOp();
      assert(rematerializationOp && "must have op to be considered");
      opsToRematerialize.push_back(rematerializationOp);
    }
    computeTopologicalSorting(opsToRematerialize);
    for (Operation *rematerializationOp : opsToRematerialize) {
      Operation *clonedOp = builder.clone(*rematerializationOp);
      for (auto [oldResult, newResult] : llvm::zip(
               rematerializationOp->getResults(), clonedOp->getResults())) {
        replaceAllUsesInRegionWith(oldResult, newResult, region);
      }
      LLVM_DEBUG(llvm::dbgs() << "\t\tCloned: " << *clonedOp << "\n");
    }

    return true;
  }

  void runOnOperation() override {
    LLVM_DEBUG(llvm::dbgs() << "Enter OffloadLiveInValueCanonicalization\n");

    // Since OpenACCSupport is normally registered on modules, attempt to
    // get it from the parent module first (if available), then fallback
    // to the per-function analysis.
    acc::OpenACCSupport *accSupportPtr = nullptr;
    if (auto parentAnalysis = getCachedParentAnalysis<acc::OpenACCSupport>())
      accSupportPtr = &parentAnalysis->get();
    else
      accSupportPtr = &getAnalysis<acc::OpenACCSupport>();
    acc::OpenACCSupport &accSupport = *accSupportPtr;

    func::FuncOp func = getOperation();
    LLVM_DEBUG(llvm::dbgs()
               << "Processing function: " << func.getName() << "\n");

    func.walk([&](Operation *op) {
      if (isa<acc::OffloadRegionOpInterface>(op)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Found offload region: " << op->getName() << "\n");
        assert(op->getNumRegions() == 1 && "must have 1 region");

        // Canonicalization of values changes live-in set.
        // Rerun the algorithm until convergence.
        bool changes = false;
        [[maybe_unused]] int iteration = 0;
        do {
          LLVM_DEBUG(llvm::dbgs() << "\tIteration " << iteration++ << "\n");
          changes = canonicalizeLiveInValues(op->getRegion(0), accSupport);
        } while (changes);
        LLVM_DEBUG(llvm::dbgs()
                   << "\tConverged after " << iteration << " iteration(s)\n");
      }
    });

    LLVM_DEBUG(llvm::dbgs() << "Exit OffloadLiveInValueCanonicalization\n");
  }
};

} // namespace
