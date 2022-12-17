//===- TestMemRefDependenceCheck.cpp - Test dep analysis ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to run pair-wise memref access dependence checks.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "test-memref-dependence-check"

using namespace mlir;

namespace {

// TODO: Add common surrounding loop depth-wise dependence checks.
/// Checks dependences between all pairs of memref accesses in a Function.
struct TestMemRefDependenceCheck
    : public PassWrapper<TestMemRefDependenceCheck, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestMemRefDependenceCheck)

  StringRef getArgument() const final { return "test-memref-dependence-check"; }
  StringRef getDescription() const final {
    return "Checks dependences between all pairs of memref accesses.";
  }
  SmallVector<Operation *, 4> loadsAndStores;
  void runOnOperation() override;
};

} // namespace

// Returns a result string which represents the direction vector (if there was
// a dependence), returns the string "false" otherwise.
static std::string
getDirectionVectorStr(bool ret, unsigned numCommonLoops, unsigned loopNestDepth,
                      ArrayRef<DependenceComponent> dependenceComponents) {
  if (!ret)
    return "false";
  if (dependenceComponents.empty() || loopNestDepth > numCommonLoops)
    return "true";
  std::string result;
  for (const auto &dependenceComponent : dependenceComponents) {
    std::string lbStr = "-inf";
    if (dependenceComponent.lb.has_value() &&
        dependenceComponent.lb.value() != std::numeric_limits<int64_t>::min())
      lbStr = std::to_string(dependenceComponent.lb.value());

    std::string ubStr = "+inf";
    if (dependenceComponent.ub.has_value() &&
        dependenceComponent.ub.value() != std::numeric_limits<int64_t>::max())
      ubStr = std::to_string(dependenceComponent.ub.value());

    result += "[" + lbStr + ", " + ubStr + "]";
  }
  return result;
}

// For each access in 'loadsAndStores', runs a dependence check between this
// "source" access and all subsequent "destination" accesses in
// 'loadsAndStores'. Emits the result of the dependence check as a note with
// the source access.
static void checkDependences(ArrayRef<Operation *> loadsAndStores) {
  for (unsigned i = 0, e = loadsAndStores.size(); i < e; ++i) {
    auto *srcOpInst = loadsAndStores[i];
    MemRefAccess srcAccess(srcOpInst);
    for (unsigned j = 0; j < e; ++j) {
      auto *dstOpInst = loadsAndStores[j];
      MemRefAccess dstAccess(dstOpInst);

      unsigned numCommonLoops =
          getNumCommonSurroundingLoops(*srcOpInst, *dstOpInst);
      for (unsigned d = 1; d <= numCommonLoops + 1; ++d) {
        FlatAffineValueConstraints dependenceConstraints;
        SmallVector<DependenceComponent, 2> dependenceComponents;
        DependenceResult result = checkMemrefAccessDependence(
            srcAccess, dstAccess, d, &dependenceConstraints,
            &dependenceComponents);
        if (result.value == DependenceResult::Failure) {
          srcOpInst->emitError("dependence check failed");
        } else {
          bool ret = hasDependence(result);
          // TODO: Print dependence type (i.e. RAW, etc) and print
          // distance vectors as: ([2, 3], [0, 10]). Also, shorten distance
          // vectors from ([1, 1], [3, 3]) to (1, 3).
          srcOpInst->emitRemark("dependence from ")
              << i << " to " << j << " at depth " << d << " = "
              << getDirectionVectorStr(ret, numCommonLoops, d,
                                       dependenceComponents);
        }
      }
    }
  }
}

/// Walks the operation adding load and store ops to 'loadsAndStores'. Runs
/// pair-wise dependence checks.
void TestMemRefDependenceCheck::runOnOperation() {
  // Collect the loads and stores within the function.
  loadsAndStores.clear();
  getOperation()->walk([&](Operation *op) {
    if (isa<AffineLoadOp, AffineStoreOp>(op))
      loadsAndStores.push_back(op);
  });

  checkDependences(loadsAndStores);
}

namespace mlir {
namespace test {
void registerTestMemRefDependenceCheck() {
  PassRegistration<TestMemRefDependenceCheck>();
}
} // namespace test
} // namespace mlir
