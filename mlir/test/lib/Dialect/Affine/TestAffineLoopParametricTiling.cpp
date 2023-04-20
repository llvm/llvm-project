//= TestAffineLoopParametricTiling.cpp -- Parametric Affine loop tiling pass =//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a test pass to test parametric tiling of perfectly
// nested affine for loops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;
using namespace mlir::affine;

#define DEBUG_TYPE "test-affine-parametric-tile"

namespace {
struct TestAffineLoopParametricTiling
    : public PassWrapper<TestAffineLoopParametricTiling,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestAffineLoopParametricTiling)

  StringRef getArgument() const final { return "test-affine-parametric-tile"; }
  StringRef getDescription() const final {
    return "Tile affine loops using SSA values as tile sizes";
  }
  void runOnOperation() override;
};
} // namespace

/// Checks if the function enclosing the loop nest has any arguments passed to
/// it, which can be used as tiling parameters. Assumes that atleast 'n'
/// arguments are passed, where 'n' is the number of loops in the loop nest.
static LogicalResult checkIfTilingParametersExist(ArrayRef<AffineForOp> band) {
  assert(!band.empty() && "no loops in input band");
  AffineForOp topLoop = band[0];

  if (func::FuncOp funcOp = dyn_cast<func::FuncOp>(topLoop->getParentOp()))
    if (funcOp.getNumArguments() < band.size())
      return topLoop->emitError(
          "too few tile sizes provided in the argument list of the function "
          "which contains the current band");
  return success();
}

/// Captures tiling parameters, which are expected to be passed as arguments
/// to the function enclosing the loop nest. Also checks if the required
/// parameters are of index type. This approach is temporary for testing
/// purposes.
static LogicalResult
getTilingParameters(ArrayRef<AffineForOp> band,
                    SmallVectorImpl<Value> &tilingParameters) {
  AffineForOp topLoop = band[0];
  Region *funcOpRegion = topLoop->getParentRegion();
  unsigned nestDepth = band.size();

  for (BlockArgument blockArgument :
       funcOpRegion->getArguments().take_front(nestDepth)) {
    if (blockArgument.getArgNumber() < nestDepth) {
      if (!blockArgument.getType().isIndex())
        return topLoop->emitError(
            "expected tiling parameters to be of index type");
      tilingParameters.push_back(blockArgument);
    }
  }
  return success();
}

void TestAffineLoopParametricTiling::runOnOperation() {
  // Bands of loops to tile.
  std::vector<SmallVector<AffineForOp, 6>> bands;
  getTileableBands(getOperation(), &bands);

  // Tile each band.
  for (MutableArrayRef<AffineForOp> band : bands) {
    // Capture the tiling parameters from the arguments to the function
    // enclosing this loop nest.
    SmallVector<AffineForOp, 6> tiledNest;
    SmallVector<Value, 6> tilingParameters;
    // Check if tiling parameters are present.
    if (checkIfTilingParametersExist(band).failed())
      return;

    // Get function arguments as tiling parameters.
    if (getTilingParameters(band, tilingParameters).failed())
      return;

    (void)tilePerfectlyNestedParametric(band, tilingParameters, &tiledNest);
  }
}

namespace mlir {
namespace test {
void registerTestAffineLoopParametricTilingPass() {
  PassRegistration<TestAffineLoopParametricTiling>();
}
} // namespace test
} // namespace mlir
