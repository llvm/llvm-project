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
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Transforms/LoopUtils.h"

using namespace mlir;

#define DEBUG_TYPE "test-affine-parametric-tile"

namespace {
struct TestAffineLoopParametricTiling
    : public PassWrapper<TestAffineLoopParametricTiling, FunctionPass> {
  void runOnFunction() override;
};
} // end anonymous namespace

/// Checks if the function enclosing the loop nest has any arguments passed to
/// it, which can be used as tiling parameters. Assumes that atleast 'n'
/// arguments are passed, where 'n' is the number of loops in the loop nest.
static void checkIfTilingParametersExist(ArrayRef<AffineForOp> band) {
  assert(!band.empty() && "no loops in input band");
  AffineForOp topLoop = band[0];

  if (FuncOp funcOp = dyn_cast<FuncOp>(topLoop->getParentOp()))
    assert(funcOp.getNumArguments() >= band.size() && "Too few tile sizes");
}

/// Captures tiling parameters, which are expected to be passed as arguments
/// to the function enclosing the loop nest. Also checks if the required
/// parameters are of index type. This approach is temporary for testing
/// purposes.
static void getTilingParameters(ArrayRef<AffineForOp> band,
                                SmallVectorImpl<Value> &tilingParameters) {
  AffineForOp topLoop = band[0];
  Region *funcOpRegion = topLoop->getParentRegion();
  unsigned nestDepth = band.size();

  for (BlockArgument blockArgument :
       funcOpRegion->getArguments().take_front(nestDepth)) {
    if (blockArgument.getArgNumber() < nestDepth) {
      assert(blockArgument.getType().isIndex() &&
             "expected tiling parameters to be of index type.");
      tilingParameters.push_back(blockArgument);
    }
  }
}

void TestAffineLoopParametricTiling::runOnFunction() {
  // Bands of loops to tile.
  std::vector<SmallVector<AffineForOp, 6>> bands;
  getTileableBands(getFunction(), &bands);

  // Tile each band.
  for (SmallVectorImpl<AffineForOp> &band : bands) {
    // Capture the tiling parameters from the arguments to the function
    // enclosing this loop nest.
    SmallVector<AffineForOp, 6> tiledNest;
    SmallVector<Value, 6> tilingParameters;
    // Check if tiling parameters are present.
    checkIfTilingParametersExist(band);

    // Get function arguments as tiling parameters.
    getTilingParameters(band, tilingParameters);

    if (failed(
            tilePerfectlyNestedParametric(band, tilingParameters, &tiledNest)))
      return signalPassFailure();
  }
}

namespace mlir {
namespace test {
void registerTestAffineLoopParametricTilingPass() {
  PassRegistration<TestAffineLoopParametricTiling>(
      "test-affine-parametric-tile",
      "Tile affine loops using SSA values as tile sizes");
}
} // namespace test
} // namespace mlir
