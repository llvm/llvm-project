//===- Interchange.cpp - Linalg interchange transformation ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the linalg interchange transformation.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Analysis/DependenceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <type_traits>

#define DEBUG_TYPE "linalg-interchange"

using namespace mlir;
using namespace mlir::linalg;

static LogicalResult
interchangeGenericOpPrecondition(GenericOp genericOp,
                                 ArrayRef<unsigned> interchangeVector) {
  // Interchange vector must be non-empty and match the number of loops.
  if (interchangeVector.empty() ||
      genericOp.getNumLoops() != interchangeVector.size())
    return failure();
  // Permutation map must be invertible.
  if (!inversePermutation(AffineMap::getPermutationMap(interchangeVector,
                                                       genericOp.getContext())))
    return failure();
  return success();
}

FailureOr<GenericOp>
mlir::linalg::interchangeGenericOp(RewriterBase &rewriter, GenericOp genericOp,
                                   ArrayRef<unsigned> interchangeVector) {
  if (failed(interchangeGenericOpPrecondition(genericOp, interchangeVector)))
    return rewriter.notifyMatchFailure(genericOp, "preconditions not met");

  // 1. Compute the inverse permutation map, it must be non-null since the
  // preconditions are satisfied.
  MLIRContext *context = genericOp.getContext();
  AffineMap permutationMap = inversePermutation(
      AffineMap::getPermutationMap(interchangeVector, context));
  assert(permutationMap && "unexpected null map");

  // Start a guarded inplace update.
  rewriter.startRootUpdate(genericOp);
  auto guard =
      llvm::make_scope_exit([&]() { rewriter.finalizeRootUpdate(genericOp); });

  // 2. Compute the interchanged indexing maps.
  SmallVector<AffineMap> newIndexingMaps;
  for (OpOperand *opOperand : genericOp.getInputAndOutputOperands()) {
    AffineMap m = genericOp.getTiedIndexingMap(opOperand);
    if (!permutationMap.isEmpty())
      m = m.compose(permutationMap);
    newIndexingMaps.push_back(m);
  }
  genericOp->setAttr(getIndexingMapsAttrName(),
                     rewriter.getAffineMapArrayAttr(newIndexingMaps));

  // 3. Compute the interchanged iterator types.
  ArrayRef<Attribute> itTypes = genericOp.iterator_types().getValue();
  SmallVector<Attribute> itTypesVector;
  llvm::append_range(itTypesVector, itTypes);
  SmallVector<int64_t> permutation(interchangeVector.begin(),
                                   interchangeVector.end());
  applyPermutationToVector(itTypesVector, permutation);
  genericOp->setAttr(getIteratorTypesAttrName(),
                     ArrayAttr::get(context, itTypesVector));

  // 4. Transform the index operations by applying the permutation map.
  if (genericOp.hasIndexSemantics()) {
    OpBuilder::InsertionGuard guard(rewriter);
    for (IndexOp indexOp :
         llvm::make_early_inc_range(genericOp.getBody()->getOps<IndexOp>())) {
      rewriter.setInsertionPoint(indexOp);
      SmallVector<Value> allIndices;
      allIndices.reserve(genericOp.getNumLoops());
      llvm::transform(llvm::seq<uint64_t>(0, genericOp.getNumLoops()),
                      std::back_inserter(allIndices), [&](uint64_t dim) {
                        return rewriter.create<IndexOp>(indexOp->getLoc(), dim);
                      });
      rewriter.replaceOpWithNewOp<AffineApplyOp>(
          indexOp, permutationMap.getSubMap(indexOp.dim()), allIndices);
    }
  }

  return genericOp;
}
