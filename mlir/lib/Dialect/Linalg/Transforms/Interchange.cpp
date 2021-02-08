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
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/EDSC/Intrinsics.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <type_traits>

#define DEBUG_TYPE "linalg-interchange"

using namespace mlir;
using namespace mlir::linalg;

LogicalResult mlir::linalg::interchangeGenericLinalgOpPrecondition(
    Operation *op, ArrayRef<unsigned> interchangeVector) {
  if (interchangeVector.empty())
    return failure();
  // Transformation applies to generic ops only.
  if (!isa<GenericOp, IndexedGenericOp>(op))
    return failure();
  LinalgOp linOp = cast<LinalgOp>(op);
  // Transformation applies to buffers only.
  if (!linOp.hasBufferSemantics())
    return failure();
  // Permutation must be applicable.
  if (linOp.getIndexingMap(0).getNumInputs() != interchangeVector.size())
    return failure();
  // Permutation map must be invertible.
  if (!inversePermutation(
          AffineMap::getPermutationMap(interchangeVector, op->getContext())))
    return failure();
  return success();
}

LinalgOp mlir::linalg::interchange(LinalgOp op,
                                   ArrayRef<unsigned> interchangeVector) {
  if (interchangeVector.empty())
    return op;

  MLIRContext *context = op.getContext();
  auto permutationMap = inversePermutation(
      AffineMap::getPermutationMap(interchangeVector, context));
  assert(permutationMap && "expected permutation to be invertible");
  SmallVector<Attribute, 4> newIndexingMaps;
  auto indexingMaps = op.indexing_maps().getValue();
  for (unsigned i = 0, e = op.getNumShapedOperands(); i != e; ++i) {
    AffineMap m = indexingMaps[i].cast<AffineMapAttr>().getValue();
    if (!permutationMap.isEmpty())
      m = m.compose(permutationMap);
    newIndexingMaps.push_back(AffineMapAttr::get(m));
  }
  auto itTypes = op.iterator_types().getValue();
  SmallVector<Attribute, 4> itTypesVector;
  for (unsigned i = 0, e = itTypes.size(); i != e; ++i)
    itTypesVector.push_back(itTypes[i]);
  applyPermutationToVector(itTypesVector, interchangeVector);

  op->setAttr(getIndexingMapsAttrName(),
              ArrayAttr::get(newIndexingMaps, context));
  op->setAttr(getIteratorTypesAttrName(),
              ArrayAttr::get(itTypesVector, context));

  return op;
}
