//===- Transforms.cpp ---------------------------------------------- C++ --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Mesh/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Mesh/IR/MeshOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <iterator>
#include <numeric>

namespace mlir::mesh {

namespace {

/// Lower `mesh.process_multi_index` into expression using
/// `mesh.process_linear_index` and `mesh.cluster_shape`.
struct ProcessMultiIndexOpLowering : OpRewritePattern<ProcessMultiIndexOp> {
  template <typename... OpRewritePatternArgs>
  ProcessMultiIndexOpLowering(SymbolTableCollection &symbolTableCollection,
                              OpRewritePatternArgs &&...opRewritePatternArgs)
      : OpRewritePattern(
            std::forward<OpRewritePatternArgs...>(opRewritePatternArgs)...),
        symbolTableCollection(symbolTableCollection) {}

  LogicalResult matchAndRewrite(ProcessMultiIndexOp op,
                                PatternRewriter &rewriter) const override {
    ClusterOp mesh =
        symbolTableCollection.lookupNearestSymbolFrom<mesh::ClusterOp>(
            op.getOperation(), op.getMeshAttr());
    if (!mesh) {
      return failure();
    }

    ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
    builder.setInsertionPointAfter(op.getOperation());
    Value linearIndex = builder.create<ProcessLinearIndexOp>(mesh);
    ValueRange meshShape = builder.create<ClusterShapeOp>(mesh).getResults();
    SmallVector<Value> completeMultiIndex =
        builder.create<affine::AffineDelinearizeIndexOp>(linearIndex, meshShape)
            .getMultiIndex();
    SmallVector<Value> multiIndex;
    ArrayRef<MeshAxis> opMeshAxes = op.getAxes();
    SmallVector<MeshAxis> opAxesIota;
    if (opMeshAxes.empty()) {
      opAxesIota.resize(mesh.getRank());
      std::iota(opAxesIota.begin(), opAxesIota.end(), 0);
      opMeshAxes = opAxesIota;
    }
    llvm::transform(opMeshAxes, std::back_inserter(multiIndex),
                    [&completeMultiIndex](MeshAxis meshAxis) {
                      return completeMultiIndex[meshAxis];
                    });
    rewriter.replaceAllUsesWith(op.getResults(), multiIndex);
    return success();
  }

private:
  SymbolTableCollection &symbolTableCollection;
};

} // namespace

void processMultiIndexOpLoweringPopulatePatterns(
    RewritePatternSet &patterns, SymbolTableCollection &symbolTableCollection) {
  patterns.add<ProcessMultiIndexOpLowering>(symbolTableCollection,
                                            patterns.getContext());
}

void processMultiIndexOpLoweringRegisterDialects(DialectRegistry &registry) {
  registry.insert<affine::AffineDialect, mesh::MeshDialect>();
}

} // namespace mlir::mesh
