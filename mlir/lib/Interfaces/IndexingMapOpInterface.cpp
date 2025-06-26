//===- IndexingMapOpInterface.cpp -- IndexingMapOpInterface impl ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/IndexingMapOpInterface.h"

using namespace mlir;

namespace mlir {
#include "mlir/Interfaces/IndexingMapOpInterface.cpp.inc"
} // namespace mlir

LogicalResult mlir::IndexingMapOpInterface::verifyImpl() {
  // All input/output operands must be indexed.
  if (static_cast<int64_t>(getIndexingMapsArray().size()) !=
      getOperation()->getNumOperands())
    return this->emitOpError("expected the number of indexing_map (")
           << getIndexingMapsArray().size()
           << ") to be equal to the number of input/output operands ("
           << getOperation()->getNumOperands() << ")";

  AffineMap invertedMap = getShapesToLoopsMap();
  if (!invertedMap) {
    std::string str;
    llvm::raw_string_ostream os(str);
    getLoopsToShapesMap().print(os);
    return this->emitOpError("invalid indexing maps are non-invertible: ")
           << "(" << str << ")";
  }

  SmallVector<int64_t> endLoopRangeValues = getStaticLoopRanges();

  // Set this flag if this op has user defined maps. This is required to guard
  // the below error condition which assume default indexing maps.
  for (OpOperand &opOperand : getOperation()->getOpOperands()) {
    AffineMap indexingMap = getMatchingIndexingMap(&opOperand);

    // Symbols disallowed.
    if (indexingMap.getNumSymbols() != 0)
      return getOperation()->emitOpError("unexpected symbols in indexing_map #")
             << opOperand.getOperandNumber();

    // Domain must be consistent.
    if (indexingMap.getNumDims() != endLoopRangeValues.size())
      return getOperation()->emitOpError("expected indexing_map #")
             << opOperand.getOperandNumber() << " to have "
             << endLoopRangeValues.size()
             << " dim(s) to match the number of loops";

    SmallVector<int64_t> shape = getStaticOperandShape(&opOperand);
    int64_t rank = shape.size();

    if (indexingMap.getNumResults() != rank)
      return getOperation()->emitOpError("expected operand rank (")
             << rank << ") to match the result rank of indexing_map #"
             << opOperand.getOperandNumber() << " ("
             << indexingMap.getNumResults() << ")";
  }

  // Check if given shapes match to inferred shapes.
  SmallVector<int64_t> startLoopRangeValues(endLoopRangeValues.size(), 0);
  // Verify only static cases since we can't get exact dimension sizes and
  // loop ranges for dynamic cases in this stage.
  if (llvm::none_of(endLoopRangeValues, ShapedType::isDynamic)) {
    // Exclusive end range.
    for (int64_t &range : endLoopRangeValues)
      range -= 1;
    for (OpOperand &opOperand : getOperation()->getOpOperands()) {
      AffineMap indexingMap = getMatchingIndexingMap(&opOperand);
      SmallVector<int64_t> startIndices =
          indexingMap.compose(startLoopRangeValues);
      SmallVector<int64_t> endIndices = indexingMap.compose(endLoopRangeValues);
      SmallVector<int64_t> shape = getStaticOperandShape(&opOperand);
      for (auto dim : llvm::seq<int64_t>(0, shape.size())) {
        // Ignore dynamic dimension or the case that the dimension size is 0
        if (ShapedType::isDynamic(shape[dim]) || shape[dim] == 0)
          continue;

        // The first index or last index should be the maximum or the minimum in
        // the inferred index ranges since the range is increasing or
        // decreasing. The size of dimensions of input/output operands and the
        // maximum value + 1 in the inferred range should be the same. But, for
        // now we check if the inferred ranges are in boundary of input/output
        // operands' size or not in case that Affine Expressions are complicated
        // such as d0 * 3
        // + d1 since it is not easy to handle the issues.
        // Found the case that this solution can't check, for example, (d0, d1)
        // -> (d1 - d0)
        int64_t inferredDimSize =
            std::max(startIndices[dim], endIndices[dim]) + 1;
        if (std::min(startIndices[dim], endIndices[dim]) < 0) {
          std::string mapStr;
          {
            llvm::raw_string_ostream os(mapStr);
            os << indexingMap;
          }
          return this->emitOpError(
                     "unexpected result less than 0 at expression #")
                 << dim << " in " << mapStr;
        }
        if (isa<AffineDimExpr>(indexingMap.getResult(dim))) {
          if (inferredDimSize != shape[dim]) {
            return this->emitOpError("inferred input/output operand #")
                   << opOperand.getOperandNumber() << " has shape's dimension #"
                   << dim << " to be " << inferredDimSize << ", but found "
                   << shape[dim];
          }
        } else {
          if (inferredDimSize > shape[dim]) {
            return this->emitOpError("inferred input/output operand #")
                   << opOperand.getOperandNumber() << " has shape's dimension #"
                   << dim << " to be greater than or equal to "
                   << inferredDimSize << ", but found " << shape[dim];
          }
        }
      }
    }
  }

  return success();
}
