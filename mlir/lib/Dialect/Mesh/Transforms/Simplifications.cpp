//===- Simplifications.cpp - Mesh Simplifications ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Mesh/Transforms/Simplifications.h"
#include "TransformsDetail.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Mesh/IR/MeshOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <numeric>
#include <utility>

namespace mlir {
namespace mesh {

void populateSimplificationPatterns(
    RewritePatternSet &patterns, SymbolTableCollection &symbolTableCollection) {
  populateAllReduceEndomorphismSimplificationPatterns<arith::AddFOp>(
      patterns, ReductionKind::Sum);
  populateAllReduceEndomorphismSimplificationPatterns<arith::AddIOp>(
      patterns, ReductionKind::Sum);

  populateAllReduceEndomorphismSimplificationPatterns<arith::MinimumFOp>(
      patterns, ReductionKind::Min);
  populateAllReduceEndomorphismSimplificationPatterns<arith::MinSIOp>(
      patterns, ReductionKind::Min);
  populateAllReduceEndomorphismSimplificationPatterns<arith::MinUIOp>(
      patterns, ReductionKind::Min);

  populateAllReduceEndomorphismSimplificationPatterns<arith::MaximumFOp>(
      patterns, ReductionKind::Max);
  populateAllReduceEndomorphismSimplificationPatterns<arith::MaxSIOp>(
      patterns, ReductionKind::Max);
  populateAllReduceEndomorphismSimplificationPatterns<arith::MaxUIOp>(
      patterns, ReductionKind::Max);

  // TODO: add simplifications for all-gather and other collectives.

  populateFoldingPatterns(patterns, symbolTableCollection);
}

namespace {

// This folding can not be done with an operation's fold method or
// DialectFoldInterface, because it needs a SymbolTableCollection to cache the
// symbol tables.
// We can't use DialectFoldInterface since the cache may be invalidated by some
// pass changing the referenced MeshOp ops.
struct MeshShapeFolder
    : OpRewritePatternWithSymbolTableCollection<MeshShapeOp> {
  using OpRewritePatternWithSymbolTableCollection::
      OpRewritePatternWithSymbolTableCollection;
  LogicalResult matchAndRewrite(MeshShapeOp op,
                                PatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
    MeshOp mesh = symbolTableCollection.lookupNearestSymbolFrom<mesh::MeshOp>(
        op.getOperation(), op.getMeshAttr());
    if (!mesh) {
      return failure();
    }
    ArrayRef<MeshAxis> opMeshAxes = op.getAxes();
    SmallVector<MeshAxis> opAxesIota;
    if (opMeshAxes.empty()) {
      opAxesIota.resize(mesh.getRank());
      std::iota(opAxesIota.begin(), opAxesIota.end(), 0);
      opMeshAxes = opAxesIota;
    }
    if (llvm::all_of(opMeshAxes, [&mesh](MeshAxis axis) {
          return ShapedType::isDynamic(mesh.getShape()[axis]);
        })) {
      // All mesh dimensions are dynamic. Nothing to fold.
      return failure();
    }

    SmallVector<Value> newResults(op->getResults().size());
    SmallVector<MeshAxis> newShapeOpMeshAxes;
    SmallVector<size_t> newToOldResultsIndexMap;

    for (size_t i = 0; i < opMeshAxes.size(); ++i) {
      auto meshAxisSize = mesh.getShape()[opMeshAxes[i]];
      if (ShapedType::isDynamic(meshAxisSize)) {
        newToOldResultsIndexMap.push_back(i);
        newShapeOpMeshAxes.push_back(opMeshAxes[i]);
      } else {
        // Fold static mesh axes.
        newResults[i] = builder.create<arith::ConstantOp>(
            builder.getIndexAttr(meshAxisSize));
      }
    }

    // Leave only the dynamic mesh axes to be queried.
    if (!newShapeOpMeshAxes.empty()) {
      MeshShapeOp newShapeOp =
          builder.create<MeshShapeOp>(mesh.getSymName(), newShapeOpMeshAxes);
      for (size_t i = 0; i < newShapeOp->getResults().size(); ++i) {
        newResults[newToOldResultsIndexMap[i]] = newShapeOp->getResults()[i];
      }
    }
    rewriter.replaceOp(op, newResults);

    return success();
  }
};

} // namespace

void populateFoldingPatterns(RewritePatternSet &patterns,
                             SymbolTableCollection &symbolTableCollection) {
  patterns.add<MeshShapeFolder>(symbolTableCollection, patterns.getContext());
}

} // namespace mesh
} // namespace mlir
