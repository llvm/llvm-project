//===- VectorTransforms.cpp - Conversion within the Vector dialect --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements target-independent rewrites as 1->N patterns.
//
//===----------------------------------------------------------------------===//

#include <type_traits>

#include "mlir/Dialect/Affine/EDSC/Builders.h"
#include "mlir/Dialect/Affine/EDSC/Intrinsics.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/EDSC/Intrinsics.h"
#include "mlir/Dialect/SCF/EDSC/Intrinsics.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/EDSC/Intrinsics.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Dialect/Vector/VectorTransforms.h"
#include "mlir/Dialect/Vector/VectorUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/VectorInterfaces.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "vector-to-vector"

using namespace mlir;
using llvm::dbgs;

// Helper to find an index in an affine map.
static Optional<int64_t> getResultIndex(AffineMap map, int64_t index) {
  for (int64_t i = 0, e = map.getNumResults(); i < e; ++i) {
    int64_t idx = map.getDimPosition(i);
    if (idx == index)
      return i;
  }
  return None;
}

// Helper to construct iterator types with one index removed.
static SmallVector<Attribute, 4> adjustIter(ArrayAttr iteratorTypes,
                                            int64_t index) {
  SmallVector<Attribute, 4> results;
  for (auto it : llvm::enumerate(iteratorTypes)) {
    int64_t idx = it.index();
    if (idx == index)
      continue;
    results.push_back(it.value());
  }
  return results;
}

// Helper to construct an affine map with one index removed.
static AffineMap adjustMap(AffineMap map, int64_t index,
                           PatternRewriter &rewriter) {
  auto *ctx = rewriter.getContext();
  SmallVector<AffineExpr, 4> results;
  for (int64_t i = 0, e = map.getNumResults(); i < e; ++i) {
    int64_t idx = map.getDimPosition(i);
    if (idx == index)
      continue;
    // Re-insert remaining indices, but renamed when occurring
    // after the removed index.
    auto targetExpr = getAffineDimExpr(idx < index ? idx : idx - 1, ctx);
    results.push_back(targetExpr);
  }
  return AffineMap::get(map.getNumDims() - 1, 0, results, ctx);
}

// Helper to drop dimension from vector type.
static Type adjustType(VectorType tp, int64_t index) {
  int64_t rank = tp.getRank();
  Type eltType = tp.getElementType();
  if (rank == 1) {
    assert(index == 0 && "index for scalar result out of bounds");
    return eltType;
  }
  SmallVector<int64_t, 4> adjustedShape;
  for (int64_t i = 0; i < rank; ++i) {
    // Omit dimension at the given index.
    if (i == index)
      continue;
    // Otherwise, add dimension back.
    adjustedShape.push_back(tp.getDimSize(i));
  }
  return VectorType::get(adjustedShape, eltType);
}

// Helper method to possibly drop a dimension in a load.
// TODO
static Value reshapeLoad(Location loc, Value val, VectorType type,
                         int64_t index, int64_t pos,
                         PatternRewriter &rewriter) {
  if (index == -1)
    return val;
  Type lowType = adjustType(type, 0);
  // At extraction dimension?
  if (index == 0) {
    auto posAttr = rewriter.getI64ArrayAttr(pos);
    return rewriter.create<vector::ExtractOp>(loc, lowType, val, posAttr);
  }
  // Unroll leading dimensions.
  VectorType vType = lowType.cast<VectorType>();
  VectorType resType = adjustType(type, index).cast<VectorType>();
  Value result =
      rewriter.create<ConstantOp>(loc, resType, rewriter.getZeroAttr(resType));
  for (int64_t d = 0, e = resType.getDimSize(0); d < e; d++) {
    auto posAttr = rewriter.getI64ArrayAttr(d);
    Value ext = rewriter.create<vector::ExtractOp>(loc, vType, val, posAttr);
    Value load = reshapeLoad(loc, ext, vType, index - 1, pos, rewriter);
    result =
        rewriter.create<vector::InsertOp>(loc, resType, load, result, posAttr);
  }
  return result;
}

// Helper method to possibly drop a dimension in a store.
// TODO
static Value reshapeStore(Location loc, Value val, Value result,
                          VectorType type, int64_t index, int64_t pos,
                          PatternRewriter &rewriter) {
  // Unmodified?
  if (index == -1)
    return val;
  // At insertion dimension?
  if (index == 0) {
    auto posAttr = rewriter.getI64ArrayAttr(pos);
    return rewriter.create<vector::InsertOp>(loc, type, val, result, posAttr);
  }
  // Unroll leading dimensions.
  Type lowType = adjustType(type, 0);
  VectorType vType = lowType.cast<VectorType>();
  Type insType = adjustType(vType, 0);
  for (int64_t d = 0, e = type.getDimSize(0); d < e; d++) {
    auto posAttr = rewriter.getI64ArrayAttr(d);
    Value ext = rewriter.create<vector::ExtractOp>(loc, vType, result, posAttr);
    Value ins = rewriter.create<vector::ExtractOp>(loc, insType, val, posAttr);
    Value sto = reshapeStore(loc, ins, ext, vType, index - 1, pos, rewriter);
    result = rewriter.create<vector::InsertOp>(loc, type, sto, result, posAttr);
  }
  return result;
}

// Clones `op` into a new operations that takes `operands` and returns
// `resultTypes`.
static Operation *cloneOpWithOperandsAndTypes(OpBuilder &builder, Location loc,
                                              Operation *op,
                                              ArrayRef<Value> operands,
                                              ArrayRef<Type> resultTypes) {
  OperationState res(loc, op->getName().getStringRef(), operands, resultTypes,
                     op->getAttrs());
  return builder.createOperation(res);
}

// Populates 'resultElements[indexMap[i]]' with elements from 'inputElements[i]'
// for each index 'i' in inputElements with a valid mapping in 'indexMap'.
static void getMappedElements(const DenseMap<int64_t, int64_t> &indexMap,
                              ArrayRef<int64_t> inputElements,
                              SmallVectorImpl<int64_t> &resultElements) {
  assert(indexMap.size() == resultElements.size());
  assert(inputElements.size() >= resultElements.size());
  for (unsigned i = 0, e = inputElements.size(); i < e; ++i) {
    auto it = indexMap.find(i);
    if (it != indexMap.end())
      resultElements[it->second] = inputElements[i];
  }
}

// Returns a tuple type with vector element types for each resulting slice
// of 'vectorType' unrolled by 'sizes' and 'strides'.
// TODO: Move this to a utility function and share it with
// Extract/InsertSlicesOp verification.
static TupleType generateExtractSlicesOpResultType(VectorType vectorType,
                                                   ArrayRef<int64_t> sizes,
                                                   ArrayRef<int64_t> strides,
                                                   OpBuilder &builder) {
  assert(llvm::all_of(strides, [](int64_t s) { return s == 1; }));
  assert(static_cast<int64_t>(sizes.size()) == vectorType.getRank());
  assert(static_cast<int64_t>(strides.size()) == vectorType.getRank());

  // Compute shape ratio of 'shape' and 'sizes'.
  auto shape = vectorType.getShape();
  auto maybeDimSliceCounts = shapeRatio(shape, sizes);
  assert(maybeDimSliceCounts.hasValue());
  auto sliceDimCounts = *maybeDimSliceCounts;

  // Compute strides w.r.t number of slices in each dimension.
  auto sliceStrides = computeStrides(sliceDimCounts);
  int64_t sliceCount = computeMaxLinearIndex(sliceDimCounts);
  SmallVector<Type, 4> vectorTypes(sliceCount);
  for (unsigned i = 0; i < sliceCount; ++i) {
    auto vectorOffsets = delinearize(sliceStrides, i);
    auto elementOffsets =
        computeElementOffsetsFromVectorSliceOffsets(sizes, vectorOffsets);
    auto sliceSizes = computeSliceSizes(shape, sizes, elementOffsets);
    // Create Vector type and add to 'vectorTypes[i]'.
    vectorTypes[i] = VectorType::get(sliceSizes, vectorType.getElementType());
  }
  return TupleType::get(vectorTypes, builder.getContext());
}

// UnrolledVectorState aggregates per-operand/result vector state required for
// unrolling.
struct UnrolledVectorState {
  SmallVector<int64_t, 4> unrolledShape;
  SmallVector<int64_t, 4> unrollFactors;
  SmallVector<int64_t, 8> basis;
  int64_t numInstances;
  Value slicesTuple;
};

// Populates 'state' with unrolled shape, unroll factors, basis and
// num unrolled instances for 'vectorType'.
static void initUnrolledVectorState(VectorType vectorType, Value initValue,
                                    const DenseMap<int64_t, int64_t> &indexMap,
                                    ArrayRef<int64_t> targetShape,
                                    UnrolledVectorState &state,
                                    OpBuilder &builder) {
  // Compute unrolled shape of 'vectorType'.
  state.unrolledShape.resize(vectorType.getRank());
  getMappedElements(indexMap, targetShape, state.unrolledShape);
  // Compute unroll factors for unrolled shape.
  auto maybeUnrollFactors =
      shapeRatio(vectorType.getShape(), state.unrolledShape);
  assert(maybeUnrollFactors.hasValue());
  state.unrollFactors = *maybeUnrollFactors;
  // Compute 'basis' and 'numInstances' based on 'state.unrollFactors'.
  state.basis = computeStrides(state.unrollFactors);
  state.numInstances = computeMaxLinearIndex(state.unrollFactors);
  state.slicesTuple = nullptr;
  if (initValue != nullptr) {
    // Create ExtractSlicesOp.
    SmallVector<int64_t, 4> sizes(state.unrolledShape);
    SmallVector<int64_t, 4> strides(state.unrollFactors.size(), 1);
    auto tupleType =
        generateExtractSlicesOpResultType(vectorType, sizes, strides, builder);
    state.slicesTuple = builder.create<vector::ExtractSlicesOp>(
        initValue.getLoc(), tupleType, initValue, sizes, strides);
  }
}

// Computes and returns the linear index of the unrolled vector at
// 'vectorOffsets' within the vector represented by 'state'.
static int64_t
getUnrolledVectorLinearIndex(UnrolledVectorState &state,
                             ArrayRef<int64_t> vectorOffsets,
                             DenseMap<int64_t, int64_t> &indexMap) {
  // Compute vector offsets.
  SmallVector<int64_t, 4> sliceOffsets(state.unrolledShape.size());
  getMappedElements(indexMap, vectorOffsets, sliceOffsets);
  // Compute and return linear index of 'sliceOffsets' w.r.t 'state.basis'.
  return linearize(sliceOffsets, state.basis);
}

// Returns an unrolled vector at 'vectorOffsets' within the vector
// represented by 'state'. The vector is created from a slice of 'initValue'
// if not present in 'cache'.
static Value getOrCreateUnrolledVectorSlice(
    Location loc, UnrolledVectorState &state, ArrayRef<int64_t> vectorOffsets,
    ArrayRef<int64_t> offsets, DenseMap<int64_t, int64_t> &indexMap,
    Value initValue, SmallVectorImpl<Value> &cache, OpBuilder &builder) {
  // Compute slice offsets.
  SmallVector<int64_t, 4> sliceOffsets(state.unrolledShape.size());
  getMappedElements(indexMap, offsets, sliceOffsets);
  // TODO: Support non-1 strides.
  SmallVector<int64_t, 4> sliceStrides(state.unrolledShape.size(), 1);
  // Compute linear index of 'sliceOffsets' w.r.t 'state.basis'.
  int64_t sliceLinearIndex =
      getUnrolledVectorLinearIndex(state, vectorOffsets, indexMap);
  assert(sliceLinearIndex < static_cast<int64_t>(cache.size()));
  auto valueSlice = cache[sliceLinearIndex];
  if (valueSlice == nullptr) {
    // Return tuple element at 'sliceLinearIndex'.
    auto tupleIndex = builder.getI64IntegerAttr(sliceLinearIndex);
    auto initValueType = initValue.getType().cast<VectorType>();
    auto vectorType =
        VectorType::get(state.unrolledShape, initValueType.getElementType());
    // Initialize 'cache' with slice from 'initValue'.
    valueSlice = builder.create<vector::TupleGetOp>(
        loc, vectorType, state.slicesTuple, tupleIndex);
    // Store value back to 'cache'.
    cache[sliceLinearIndex] = valueSlice;
  }
  return valueSlice;
}

// VectorState aggregates per-operand/result vector state required for
// creating slices of vector operands, and clones of the operation being
// unrolled.
struct VectorState {
  // The type of this vector.
  VectorType type;
  // Map from iteration space index to vector dimension index.
  DenseMap<int64_t, int64_t> indexMap;
  // Index of this value in operation's operand list (-1 if not an operand).
  int64_t operandIndex = -1;
  // Accumulator iterator flag.
  bool isAcc = false;
};

//
// unrollSingleResultStructuredOp
//
// Returns a value representing the result of structured operation 'op'
// with iteration bounds 'iterationBounds' unrolled to 'targetShape'.
// A list of VectorState objects must be specified in 'vectors', where
// each VectorState in the list represents a vector operand or vector result
// (if the operation does not have an accumulator operand).
// The VectorState at index 'resultIndex' in the list must be the state
// associated with the operations single result (i.e. either its accumulator
// operand or vector result value).
//
// Example:
//
//  // Before unrolling
//
//   operand0                operand1                operand2
//       \                      |                      /
//        -------------------- opA --------------------
//
//  // After unrolling by 2
//
//   operand0                operand1                operand2
//   /      \                /      \                /      \
// slice00  slice01       slice10  slice11        slice20  slice21
//   \         |            |          |            /          |
//    -------------------- opA0 --------------------           |
//             |            |          |                       |
//              \           |          |                      /
//               -------------------- opA1 -------------------
//                          |          |
//                           \        /
//                           insertslice
//                                |

// TODO: Add the following canonicalization/simplification patterns:
// *) Add pattern which matches InsertStridedSlice -> StridedSlice and forwards
//    InsertStridedSlice operand to StridedSlice.
// *) Add pattern which matches SourceOp -> StridedSlice -> UserOp which checks
//    if there are duplicate identical StridedSlice ops from SourceOp, and
//    rewrites itself to use the first duplicate. This transformation should
//    cause users of identifical StridedSlice ops to reuse the same StridedSlice
//    operation, and leave the duplicate StridedSlice ops with no users
//    (removable with DCE).

// TODO: Generalize this to support structured ops beyond
// vector ContractionOp, and merge it with 'unrollSingleResultVectorOp'
static Value unrollSingleResultStructuredOp(Operation *op,
                                            ArrayRef<int64_t> iterationBounds,
                                            std::vector<VectorState> &vectors,
                                            unsigned resultIndex,
                                            ArrayRef<int64_t> targetShape,
                                            OpBuilder &builder) {
  auto shapedType = op->getResult(0).getType().dyn_cast_or_null<ShapedType>();
  if (!shapedType || !shapedType.hasStaticShape())
    assert(false && "Expected a statically shaped result type");

  // Compute unroll factors for 'iterationBounds' based on 'targetShape'
  auto maybeUnrollFactors = shapeRatio(iterationBounds, targetShape);
  if (!maybeUnrollFactors.hasValue())
    assert(false && "Failed to compute unroll factors for target shape");
  auto unrollFactors = *maybeUnrollFactors;

  // Compute unrolled vector state for each vector in 'vectors'.
  unsigned numVectors = vectors.size();
  SmallVector<UnrolledVectorState, 3> unrolledVectorState(numVectors);
  for (unsigned i = 0; i < numVectors; ++i) {
    int64_t operandIndex = vectors[i].operandIndex;
    auto operand = operandIndex >= 0 ? op->getOperand(operandIndex) : nullptr;
    initUnrolledVectorState(vectors[i].type, operand, vectors[i].indexMap,
                            targetShape, unrolledVectorState[i], builder);
  }
  // Compute number of total unrolled instances.
  auto numUnrolledInstances = computeMaxLinearIndex(unrollFactors);
  auto sliceStrides = computeStrides(unrollFactors);

  auto &resultValueState = unrolledVectorState[resultIndex];
  auto unrolledResultType = VectorType::get(resultValueState.unrolledShape,
                                            shapedType.getElementType());

  // Initialize caches for intermediate vector results.
  std::vector<SmallVector<Value, 4>> caches(numVectors);
  for (unsigned i = 0; i < numVectors; ++i)
    caches[i].resize(unrolledVectorState[i].numInstances);

  // Unroll 'numUnrolledInstances' of 'op', storing results in 'caches'.
  for (unsigned i = 0; i < numUnrolledInstances; ++i) {
    auto vectorOffsets = delinearize(sliceStrides, i);
    auto elementOffsets =
        computeElementOffsetsFromVectorSliceOffsets(targetShape, vectorOffsets);
    // Get cached slice (or create slice) for each operand at 'offsets'.
    SmallVector<Value, 3> operands;
    operands.resize(op->getNumOperands());
    for (unsigned i = 0; i < numVectors; ++i) {
      int64_t operandIndex = vectors[i].operandIndex;
      if (operandIndex < 0)
        continue; // Output
      auto operand = op->getOperand(operandIndex);
      operands[operandIndex] = getOrCreateUnrolledVectorSlice(
          op->getLoc(), unrolledVectorState[i], vectorOffsets, elementOffsets,
          vectors[i].indexMap, operand, caches[i], builder);
    }
    // Create op on sliced vector arguments.
    auto resultVector =
        cloneOpWithOperandsAndTypes(builder, op->getLoc(), op, operands,
                                    unrolledResultType)
            ->getResult(0);

    // Compute linear result index.
    int64_t linearIndex = getUnrolledVectorLinearIndex(
        resultValueState, vectorOffsets, vectors[resultIndex].indexMap);
    // Update result cache at 'linearIndex'.
    caches[resultIndex][linearIndex] = resultVector;
  }

  // Create TupleOp of unrolled result vectors.
  SmallVector<Type, 4> vectorTupleTypes(resultValueState.numInstances);
  SmallVector<Value, 4> vectorTupleValues(resultValueState.numInstances);
  for (unsigned i = 0; i < resultValueState.numInstances; ++i) {
    vectorTupleTypes[i] = caches[resultIndex][i].getType().cast<VectorType>();
    vectorTupleValues[i] = caches[resultIndex][i];
  }
  TupleType tupleType = builder.getTupleType(vectorTupleTypes);
  Value tupleOp = builder.create<vector::TupleOp>(op->getLoc(), tupleType,
                                                  vectorTupleValues);

  // Create InsertSlicesOp(Tuple(result_vectors)).
  auto resultVectorType = op->getResult(0).getType().cast<VectorType>();
  SmallVector<int64_t, 4> sizes(resultValueState.unrolledShape);
  SmallVector<int64_t, 4> strides(resultValueState.unrollFactors.size(), 1);

  Value insertSlicesOp = builder.create<vector::InsertSlicesOp>(
      op->getLoc(), resultVectorType, tupleOp, builder.getI64ArrayAttr(sizes),
      builder.getI64ArrayAttr(strides));
  return insertSlicesOp;
}

static void getVectorContractionOpUnrollState(
    vector::ContractionOp contractionOp, ArrayRef<int64_t> targetShape,
    std::vector<VectorState> &vectors, unsigned &resultIndex) {
  // Get map from iteration space index to lhs/rhs/result shape index.
  std::vector<DenseMap<int64_t, int64_t>> iterationIndexMapList;
  contractionOp.getIterationIndexMap(iterationIndexMapList);
  unsigned numIterators = iterationIndexMapList.size();
  vectors.resize(numIterators);
  unsigned accOperandIndex = vector::ContractionOp::getAccOperandIndex();
  for (unsigned i = 0; i < numIterators; ++i) {
    vectors[i].type = contractionOp.getOperand(i).getType().cast<VectorType>();
    vectors[i].indexMap = iterationIndexMapList[i];
    vectors[i].operandIndex = i;
    vectors[i].isAcc = i == accOperandIndex ? true : false;
  }

  if (llvm::size(contractionOp.masks()) == 2) {
    // Add vectors for lhs/rhs vector mask arguments. Masks have the
    // same vector shape lhs/rhs args, so copy their index maps.
    vectors.push_back({contractionOp.getLHSVectorMaskType(),
                       vectors[0].indexMap, accOperandIndex + 1, false});
    vectors.push_back({contractionOp.getRHSVectorMaskType(),
                       vectors[1].indexMap, accOperandIndex + 2, false});
  }
  // TODO: Use linalg style 'args_in'/'args_out' to partition
  // 'vectors' instead of 'resultIndex'.
  resultIndex = accOperandIndex;
}

static void getVectorElementwiseOpUnrollState(Operation *op,
                                              ArrayRef<int64_t> targetShape,
                                              std::vector<VectorState> &vectors,
                                              unsigned &resultIndex) {
  // Verify that operation and operands all have the same vector shape.
  auto resultType = op->getResult(0).getType().dyn_cast_or_null<VectorType>();
  assert(resultType && "Expected op with vector result type");
  auto resultShape = resultType.getShape();
  // Verify that all operands have the same vector type as result.
  assert(llvm::all_of(op->getOperandTypes(),
                      [=](Type type) { return type == resultType; }));

  // Create trivial elementwise identity index map based on 'resultShape'.
  DenseMap<int64_t, int64_t> indexMap;
  indexMap.reserve(resultShape.size());
  for (unsigned i = 0; i < resultShape.size(); ++i)
    indexMap[i] = i;

  // Create VectorState each operand and single result.
  unsigned numVectors = op->getNumOperands() + op->getNumResults();
  vectors.resize(numVectors);
  for (unsigned i = 0; i < op->getNumOperands(); ++i)
    vectors[i] = {resultType, indexMap, i, false};
  vectors[numVectors - 1] = {resultType, indexMap, -1, false};
  resultIndex = numVectors - 1;
}

/// Generates slices of 'vectorType' according to 'sizes' and 'strides, and
/// calls 'fn' with linear index and indices for each slice.
static void generateTransferOpSlices(
    Type memrefElementType, VectorType vectorType, TupleType tupleType,
    ArrayRef<int64_t> sizes, ArrayRef<int64_t> strides, ArrayRef<Value> indices,
    OpBuilder &builder, function_ref<void(unsigned, ArrayRef<Value>)> fn) {
  // Compute strides w.r.t. to slice counts in each dimension.
  auto maybeDimSliceCounts = shapeRatio(vectorType.getShape(), sizes);
  assert(maybeDimSliceCounts.hasValue());
  auto sliceDimCounts = *maybeDimSliceCounts;
  auto sliceStrides = computeStrides(sliceDimCounts);

  int64_t numSlices = tupleType.size();
  unsigned numSliceIndices = indices.size();
  // Compute 'indexOffset' at which to update 'indices', which is equal
  // to the memref rank (indices.size) minus the effective 'vectorRank'.
  // The effective 'vectorRank', is equal to the rank of the vector type
  // minus the rank of the memref vector element type (if it has one).
  //
  // For example:
  //
  //   Given memref type 'memref<6x2x1xvector<2x4xf32>>' and vector
  //   transfer_read/write ops which read/write vectors of type
  //   'vector<2x1x2x4xf32>'. The memref rank is 3, and the effective
  //   vector rank is 4 - 2 = 2, and so 'indexOffset' = 3 - 2 = 1.
  //
  unsigned vectorRank = vectorType.getRank();
  if (auto memrefVectorElementType = memrefElementType.dyn_cast<VectorType>()) {
    assert(vectorRank >= memrefVectorElementType.getRank());
    vectorRank -= memrefVectorElementType.getRank();
  }
  unsigned indexOffset = numSliceIndices - vectorRank;

  auto *ctx = builder.getContext();
  for (unsigned i = 0; i < numSlices; ++i) {
    auto vectorOffsets = delinearize(sliceStrides, i);
    auto elementOffsets =
        computeElementOffsetsFromVectorSliceOffsets(sizes, vectorOffsets);
    // Compute 'sliceIndices' by adding 'sliceOffsets[i]' to 'indices[i]'.
    SmallVector<Value, 4> sliceIndices(numSliceIndices);
    for (unsigned j = 0; j < numSliceIndices; ++j) {
      if (j < indexOffset) {
        sliceIndices[j] = indices[j];
      } else {
        auto expr = getAffineDimExpr(0, ctx) +
                    getAffineConstantExpr(elementOffsets[j - indexOffset], ctx);
        auto map = AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0, expr);
        sliceIndices[j] = builder.create<AffineApplyOp>(
            indices[j].getLoc(), map, ArrayRef<Value>(indices[j]));
      }
    }
    // Call 'fn' to generate slice 'i' at 'sliceIndices'.
    fn(i, sliceIndices);
  }
}

/// Returns true if 'map' is a suffix of an identity affine map, false
/// otherwise. Example: affine_map<(d0, d1, d2, d3) -> (d2, d3)>
static bool isIdentitySuffix(AffineMap map) {
  if (map.getNumDims() < map.getNumResults())
    return false;
  ArrayRef<AffineExpr> results = map.getResults();
  Optional<int> lastPos;
  for (unsigned i = 0, e = map.getNumResults(); i < e; ++i) {
    auto expr = results[i].dyn_cast<AffineDimExpr>();
    if (!expr)
      return false;
    int currPos = static_cast<int>(expr.getPosition());
    if (lastPos.hasValue() && currPos != lastPos.getValue() + 1)
      return false;
    lastPos = currPos;
  }
  return true;
}

/// Unroll transfer_read ops to the given shape and create an aggregate with all
/// the chunks.
static Value unrollTransferReadOp(vector::TransferReadOp readOp,
                                  ArrayRef<int64_t> targetShape,
                                  OpBuilder &builder) {
  if (!isIdentitySuffix(readOp.permutation_map()))
    return nullptr;
  auto sourceVectorType = readOp.getVectorType();
  SmallVector<int64_t, 4> strides(targetShape.size(), 1);

  Location loc = readOp.getLoc();
  auto memrefElementType =
      readOp.memref().getType().cast<MemRefType>().getElementType();
  auto tupleType = generateExtractSlicesOpResultType(
      sourceVectorType, targetShape, strides, builder);
  int64_t numSlices = tupleType.size();

  SmallVector<Value, 4> vectorTupleValues(numSlices);
  SmallVector<Value, 4> indices(readOp.indices().begin(),
                                readOp.indices().end());
  auto createSlice = [&](unsigned index, ArrayRef<Value> sliceIndices) {
    // Get VectorType for slice 'i'.
    auto sliceVectorType = tupleType.getType(index);
    // Create split TransferReadOp for 'sliceUser'.
    // `masked` attribute propagates conservatively: if the coarse op didn't
    // need masking, the fine op doesn't either.
    vectorTupleValues[index] = builder.create<vector::TransferReadOp>(
        loc, sliceVectorType, readOp.memref(), sliceIndices,
        readOp.permutation_map(), readOp.padding(),
        readOp.masked() ? *readOp.masked() : ArrayAttr());
  };
  generateTransferOpSlices(memrefElementType, sourceVectorType, tupleType,
                           targetShape, strides, indices, builder, createSlice);

  // Create tuple of splice transfer read operations.
  Value tupleOp =
      builder.create<vector::TupleOp>(loc, tupleType, vectorTupleValues);
  // Replace 'readOp' with result 'insertSlicesResult'.
  Value newVec = builder.create<vector::InsertSlicesOp>(
      loc, sourceVectorType, tupleOp, builder.getI64ArrayAttr(targetShape),
      builder.getI64ArrayAttr(strides));
  return newVec;
}

// Entry point for unrolling declarative pattern rewrite for transfer_write op.
LogicalResult
mlir::vector::unrollTransferWriteOp(OpBuilder &builder, Operation *op,
                                    ArrayRef<int64_t> targetShape) {
  auto writeOp = cast<vector::TransferWriteOp>(op);
  if (!isIdentitySuffix(writeOp.permutation_map()))
    return failure();
  VectorType sourceVectorType = writeOp.getVectorType();
  SmallVector<int64_t, 4> strides(targetShape.size(), 1);
  TupleType tupleType = generateExtractSlicesOpResultType(
      sourceVectorType, targetShape, strides, builder);
  Location loc = writeOp.getLoc();
  Value tuple = builder.create<vector::ExtractSlicesOp>(
      loc, tupleType, writeOp.vector(), targetShape, strides);
  auto memrefElementType =
      writeOp.memref().getType().cast<MemRefType>().getElementType();
  SmallVector<Value, 4> indices(writeOp.indices().begin(),
                                writeOp.indices().end());
  auto createSlice = [&](unsigned index, ArrayRef<Value> sliceIndices) {
    auto element = builder.create<vector::TupleGetOp>(
        loc, tupleType.getType(index), tuple, builder.getI64IntegerAttr(index));
    builder.create<vector::TransferWriteOp>(
        loc, element.getResult(), writeOp.memref(), sliceIndices,
        writeOp.permutation_map(),
        writeOp.masked() ? *writeOp.masked() : ArrayAttr());
  };
  generateTransferOpSlices(memrefElementType, sourceVectorType, tupleType,
                           targetShape, strides, indices, builder, createSlice);
  return success();
}

// Entry point for unrolling declarative pattern rewrites.
SmallVector<Value, 1>
mlir::vector::unrollSingleResultVectorOp(OpBuilder &builder, Operation *op,
                                         ArrayRef<int64_t> targetShape) {
  assert(op->getNumResults() == 1 && "Expected single result operation");

  // Populate 'iterationBounds', 'vectors' and 'resultIndex' to unroll 'op'.
  SmallVector<int64_t, 6> iterationBounds;
  auto unrollableVectorOp = cast<VectorUnrollOpInterface>(op);
  auto maybeUnrollShape = unrollableVectorOp.getShapeForUnroll();
  assert(maybeUnrollShape && "Trying to unroll an incorrect vector op");

  std::vector<VectorState> vectors;
  unsigned resultIndex;

  if (auto readOp = dyn_cast<vector::TransferReadOp>(op))
    return SmallVector<Value, 1>{
        unrollTransferReadOp(readOp, targetShape, builder)};

  if (auto contractionOp = dyn_cast<vector::ContractionOp>(op)) {
    // Populate state for vector ContractionOp.
    getVectorContractionOpUnrollState(contractionOp, targetShape, vectors,
                                      resultIndex);
  } else {
    // Populate state for vector elementwise op.
    getVectorElementwiseOpUnrollState(op, targetShape, vectors, resultIndex);
  }

  // Unroll 'op' with 'iterationBounds' to 'targetShape'.
  return SmallVector<Value, 1>{unrollSingleResultStructuredOp(
      op, *maybeUnrollShape, vectors, resultIndex, targetShape, builder)};
}

namespace {

// Splits vector TransferReadOp into smaller TransferReadOps based on slicing
// scheme of its unique ExtractSlicesOp user.
struct SplitTransferReadOp : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp xferReadOp,
                                PatternRewriter &rewriter) const override {
    // TODO: Support splitting TransferReadOp with non-identity
    // permutation maps. Repurpose code from MaterializeVectors transformation.
    if (!isIdentitySuffix(xferReadOp.permutation_map()))
      return failure();
    // Return unless the unique 'xferReadOp' user is an ExtractSlicesOp.
    Value xferReadResult = xferReadOp.getResult();
    auto extractSlicesOp =
        dyn_cast<vector::ExtractSlicesOp>(*xferReadResult.getUsers().begin());
    if (!xferReadResult.hasOneUse() || !extractSlicesOp)
      return failure();

    // Get 'sizes' and 'strides' parameters from ExtractSlicesOp user.
    SmallVector<int64_t, 4> sizes;
    extractSlicesOp.getSizes(sizes);
    SmallVector<int64_t, 4> strides;
    extractSlicesOp.getStrides(strides);
    assert(llvm::all_of(strides, [](int64_t s) { return s == 1; }));

    Value newVec = unrollTransferReadOp(xferReadOp, sizes, rewriter);
    if (!newVec)
      return failure();
    rewriter.replaceOp(xferReadOp, newVec);
    return success();
  }
};

// Splits vector TransferWriteOp into smaller TransferWriteOps for each source.
struct SplitTransferWriteOp : public OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern<vector::TransferWriteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp xferWriteOp,
                                PatternRewriter &rewriter) const override {
    // TODO: Support splitting TransferWriteOp with non-identity
    // permutation maps. Repurpose code from MaterializeVectors transformation.
    if (!isIdentitySuffix(xferWriteOp.permutation_map()))
      return failure();
    // Return unless the 'xferWriteOp' 'vector' operand is an 'InsertSlicesOp'.
    auto *vectorDefOp = xferWriteOp.vector().getDefiningOp();
    auto insertSlicesOp = dyn_cast_or_null<vector::InsertSlicesOp>(vectorDefOp);
    if (!insertSlicesOp)
      return failure();

    // Get TupleOp operand of 'insertSlicesOp'.
    auto tupleOp = dyn_cast_or_null<vector::TupleOp>(
        insertSlicesOp.vectors().getDefiningOp());
    if (!tupleOp)
      return failure();

    // Get 'sizes' and 'strides' parameters from InsertSlicesOp user.
    auto sourceTupleType = insertSlicesOp.getSourceTupleType();
    auto resultVectorType = insertSlicesOp.getResultVectorType();
    SmallVector<int64_t, 4> sizes;
    insertSlicesOp.getSizes(sizes);
    SmallVector<int64_t, 4> strides;
    insertSlicesOp.getStrides(strides);

    Location loc = xferWriteOp.getLoc();
    auto memrefElementType =
        xferWriteOp.memref().getType().cast<MemRefType>().getElementType();
    SmallVector<Value, 4> indices(xferWriteOp.indices().begin(),
                                  xferWriteOp.indices().end());
    auto createSlice = [&](unsigned index, ArrayRef<Value> sliceIndices) {
      // Create split TransferWriteOp for source vector 'tupleOp.operand[i]'.
      // `masked` attribute propagates conservatively: if the coarse op didn't
      // need masking, the fine op doesn't either.
      rewriter.create<vector::TransferWriteOp>(
          loc, tupleOp.getOperand(index), xferWriteOp.memref(), sliceIndices,
          xferWriteOp.permutation_map(),
          xferWriteOp.masked() ? *xferWriteOp.masked() : ArrayAttr());
    };
    generateTransferOpSlices(memrefElementType, resultVectorType,
                             sourceTupleType, sizes, strides, indices, rewriter,
                             createSlice);

    // Erase old 'xferWriteOp'.
    rewriter.eraseOp(xferWriteOp);
    return success();
  }
};

/// Decomposes ShapeCastOp on tuple-of-vectors to multiple ShapeCastOps, each
/// on vector types.
struct ShapeCastOpDecomposer : public OpRewritePattern<vector::ShapeCastOp> {
  using OpRewritePattern<vector::ShapeCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ShapeCastOp shapeCastOp,
                                PatternRewriter &rewriter) const override {
    // Check if 'shapeCastOp' has tuple source/result type.
    auto sourceTupleType =
        shapeCastOp.source().getType().dyn_cast_or_null<TupleType>();
    auto resultTupleType =
        shapeCastOp.result().getType().dyn_cast_or_null<TupleType>();
    if (!sourceTupleType || !resultTupleType)
      return failure();
    assert(sourceTupleType.size() == resultTupleType.size());

    // Create single-vector ShapeCastOp for each source tuple element.
    Location loc = shapeCastOp.getLoc();
    SmallVector<Value, 8> resultElements;
    resultElements.reserve(resultTupleType.size());
    for (unsigned i = 0, e = sourceTupleType.size(); i < e; ++i) {
      auto sourceElement = rewriter.create<vector::TupleGetOp>(
          loc, sourceTupleType.getType(i), shapeCastOp.source(),
          rewriter.getI64IntegerAttr(i));
      resultElements.push_back(rewriter.create<vector::ShapeCastOp>(
          loc, resultTupleType.getType(i), sourceElement));
    }

    // Replace 'shapeCastOp' with tuple of 'resultElements'.
    rewriter.replaceOpWithNewOp<vector::TupleOp>(shapeCastOp, resultTupleType,
                                                 resultElements);
    return success();
  }
};

/// Returns the producer Value of the same type as 'consumerValue', by tracking
/// the tuple index and offsets of the consumer vector value through the
/// chain of operations (TupleGetOp, InsertSlicesOp, ExtractSlicesOp, TupleOp,
/// and ShapeCastOp) from consumer to producer. Each operation in the chain is
/// structured, and so the tuple index and offsets can be mapped from result to
/// input, while visiting each operation in the chain.
/// Returns nullptr on failure.
static Value getProducerValue(Value consumerValue) {
  auto consumerVectorType = consumerValue.getType().cast<VectorType>();
  // A tupleIndex == -1 indicates that 'offsets' are w.r.t a vector type.
  int64_t tupleIndex = -1;
  SmallVector<int64_t, 4> offsets(consumerVectorType.getRank(), 0);
  auto *op = consumerValue.getDefiningOp();
  while (op != nullptr) {
    if (auto tupleGetOp = dyn_cast<vector::TupleGetOp>(op)) {
      assert(tupleIndex == -1 && "TupleGetOp must have vector result type");

      // Update 'tupleIndex' and next defining 'op' to visit.
      tupleIndex = tupleGetOp.getIndex();
      op = tupleGetOp.vectors().getDefiningOp();
    } else if (auto extractSlicesOp = dyn_cast<vector::ExtractSlicesOp>(op)) {
      assert(tupleIndex >= 0);

      // Compute slice strides for 'extractSlicesOp'.
      SmallVector<int64_t, 4> sizes;
      extractSlicesOp.getSizes(sizes);
      auto sliceStrides = computeStrides(
          extractSlicesOp.getSourceVectorType().getShape(), sizes);

      // Compute 'elementOffsets' into 'extractSlicesOp' input vector type,
      // of 'extractSlicesOp' result vector tuple element at 'tupleIndex'.
      auto vectorOffsets = delinearize(sliceStrides, tupleIndex);
      auto elementOffsets =
          computeElementOffsetsFromVectorSliceOffsets(sizes, vectorOffsets);

      // Add 'elementOffsets' to 'offsets' so that 'offsets' are now relative
      // to the 'extractSlicesOp' input vector type.
      assert(offsets.size() == elementOffsets.size());
      for (unsigned i = 0, e = offsets.size(); i < e; ++i)
        offsets[i] += elementOffsets[i];

      // Clear 'tupleIndex' and update next defining 'op' to visit.
      tupleIndex = -1;
      op = extractSlicesOp.vector().getDefiningOp();
    } else if (auto insertSlicesOp = dyn_cast<vector::InsertSlicesOp>(op)) {
      assert(tupleIndex == -1);

      // Compute slice strides for 'insertSlicesOp'.
      SmallVector<int64_t, 4> sizes;
      insertSlicesOp.getSizes(sizes);
      auto sliceStrides = computeStrides(
          insertSlicesOp.getResultVectorType().getShape(), sizes);

      // Compute 'vectorOffsets' of 'insertSlicesOp' input vector slice,
      // of 'insertSlicesOp' result vector type at 'offsets'.
      SmallVector<int64_t, 4> vectorOffsets(offsets.size());
      assert(offsets.size() == sizes.size());
      for (unsigned i = 0, e = offsets.size(); i < e; ++i)
        vectorOffsets[i] = offsets[i] / sizes[i];

      // Compute the source tuple element index.
      tupleIndex = linearize(vectorOffsets, sliceStrides);

      // Subtract 'elementOffsets' from 'offsets' so that 'offsets' are now
      // relative to input tuple element vector type at 'tupleIndex'.
      auto elementOffsets =
          computeElementOffsetsFromVectorSliceOffsets(sizes, vectorOffsets);
      assert(offsets.size() == elementOffsets.size());
      for (unsigned i = 0, e = offsets.size(); i < e; ++i) {
        offsets[i] -= elementOffsets[i];
        assert(offsets[i] >= 0);
      }

      // Update next defining 'op' to visit.
      op = insertSlicesOp.vectors().getDefiningOp();
    } else if (auto tupleOp = dyn_cast<vector::TupleOp>(op)) {
      assert(tupleIndex >= 0);

      // Return tuple element 'value' at 'tupleIndex' if it matches type.
      auto value = tupleOp.getOperand(tupleIndex);
      if (value.getType() == consumerVectorType)
        return value;

      // Update 'tupleIndex' and next defining 'op' to visit.
      tupleIndex = -1;
      op = value.getDefiningOp();
    } else if (auto shapeCastOp = dyn_cast<vector::ShapeCastOp>(op)) {
      if (shapeCastOp.source().getType().isa<TupleType>())
        return nullptr;
      assert(tupleIndex == -1);
      auto sourceVectorType = shapeCastOp.getSourceVectorType();
      auto sourceVectorShape = sourceVectorType.getShape();
      unsigned sourceVectorRank = sourceVectorType.getRank();
      auto resultVectorType = shapeCastOp.getResultVectorType();
      auto resultVectorShape = resultVectorType.getShape();
      unsigned resultVectorRank = resultVectorType.getRank();

      int i = sourceVectorRank - 1;
      int j = resultVectorRank - 1;

      // Check that source/result vector shape prefixes match while updating
      // 'newOffsets'.
      SmallVector<int64_t, 4> newOffsets(sourceVectorRank, 0);
      for (auto it : llvm::zip(llvm::reverse(sourceVectorShape),
                               llvm::reverse(resultVectorShape))) {
        if (std::get<0>(it) != std::get<1>(it))
          return nullptr;
        newOffsets[i--] = offsets[j--];
      }

      // Check that remaining prefix of source/result vector shapes are all 1s.
      // Currently we only support producer/consumer tracking through trivial
      // shape cast ops. Examples:
      //   %1 = vector.shape_cast %0 : vector<1x1x2x4xf32> to vector<2x4xf32>
      //   %3 = vector.shape_cast %2 : vector<16x8xf32> to vector<1x16x8xf32>
      assert(i == -1 || j == -1);
      if (i >= 0 &&
          !std::all_of(sourceVectorShape.begin(), sourceVectorShape.begin() + i,
                       [](int64_t v) { return v == 1; }))
        return nullptr;
      if (j >= 0 &&
          !std::all_of(resultVectorShape.begin(), resultVectorShape.begin() + j,
                       [](int64_t v) { return v == 1; }))
        return nullptr;

      offsets.swap(newOffsets);
      op = shapeCastOp.source().getDefiningOp();
    } else {
      // Check if 'op' produces a Value with the same type as 'consumerValue'.
      if (op->getNumResults() == 1 &&
          op->getResult(0).getType() == consumerVectorType)
        return op->getResult(0);
      return nullptr;
    }
  }
  return nullptr;
}

/// ShapeCastOpFolder folds cancelling ShapeCastOps away.
//
// Example:
//
//  The following MLIR with cancelling ShapeCastOps:
//
//   %0 = source : vector<5x4x2xf32>
//   %1 = shape_cast %0 : vector<5x4x2xf32> to vector<20x2xf32>
//   %2 = shape_cast %1 : vector<20x2xf32> to vector<5x4x2xf32>
//   %3 = user %2 : vector<5x4x2xf32>
//
//  Should canonicalize to the following:
//
//   %0 = source : vector<5x4x2xf32>
//   %1 = user %0 : vector<5x4x2xf32>
//
struct ShapeCastOpFolder : public OpRewritePattern<vector::ShapeCastOp> {
  using OpRewritePattern<vector::ShapeCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ShapeCastOp shapeCastOp,
                                PatternRewriter &rewriter) const override {
    // Check if we can replace 'shapeCastOp' result with its producer.
    if (auto producer = getProducerValue(shapeCastOp.getResult())) {
      rewriter.replaceOp(shapeCastOp, producer);
      return success();
    }

    // Check if 'shapeCastOp' has vector source/result type.
    auto sourceVectorType =
        shapeCastOp.source().getType().dyn_cast_or_null<VectorType>();
    auto resultVectorType =
        shapeCastOp.result().getType().dyn_cast_or_null<VectorType>();
    if (!sourceVectorType || !resultVectorType)
      return failure();

    // Check if shape cast op source operand is also a shape cast op.
    auto sourceShapeCastOp = dyn_cast_or_null<vector::ShapeCastOp>(
        shapeCastOp.source().getDefiningOp());
    if (!sourceShapeCastOp)
      return failure();
    auto operandSourceVectorType =
        sourceShapeCastOp.source().getType().cast<VectorType>();
    auto operandResultVectorType =
        sourceShapeCastOp.result().getType().cast<VectorType>();

    // Check if shape cast operations invert each other.
    if (operandSourceVectorType != resultVectorType ||
        operandResultVectorType != sourceVectorType)
      return failure();

    rewriter.replaceOp(shapeCastOp, sourceShapeCastOp.source());
    return success();
  }
};

// Patter rewrite which forward tuple elements to their users.
// User(TupleGetOp(ExtractSlicesOp(InsertSlicesOp(TupleOp(Producer)))))
//   -> User(Producer)
struct TupleGetFolderOp : public OpRewritePattern<vector::TupleGetOp> {
  using OpRewritePattern<vector::TupleGetOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TupleGetOp tupleGetOp,
                                PatternRewriter &rewriter) const override {
    if (auto producer = getProducerValue(tupleGetOp.getResult())) {
      rewriter.replaceOp(tupleGetOp, producer);
      return success();
    }
    return failure();
  }
};

/// Progressive lowering of ExtractSlicesOp to tuple of ExtractStridedSliceOp.
/// One:
///   %x = vector.extract_slices %0
/// is replaced by:
///   %a = vector.strided_slice %0
///   %b = vector.strided_slice %0
///   ..
///   %x = vector.tuple %a, %b, ..
class ExtractSlicesOpLowering
    : public OpRewritePattern<vector::ExtractSlicesOp> {
public:
  using OpRewritePattern<vector::ExtractSlicesOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ExtractSlicesOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    VectorType vectorType = op.getSourceVectorType();
    auto shape = vectorType.getShape();

    SmallVector<int64_t, 4> sizes;
    op.getSizes(sizes);
    SmallVector<int64_t, 4> strides;
    op.getStrides(strides); // all-ones at the moment

    // For each element in the tuple, generate the proper strided slice.
    TupleType tupleType = op.getResultTupleType();
    int64_t tupleSize = tupleType.size();
    SmallVector<Value, 4> tupleValues(tupleSize);
    auto sliceStrides = computeStrides(shape, sizes);
    for (int64_t i = 0; i < tupleSize; ++i) {
      auto vectorOffsets = delinearize(sliceStrides, i);
      auto elementOffsets =
          computeElementOffsetsFromVectorSliceOffsets(sizes, vectorOffsets);
      auto sliceSizes = computeSliceSizes(shape, sizes, elementOffsets);
      // Insert in tuple.
      tupleValues[i] = rewriter.create<vector::ExtractStridedSliceOp>(
          loc, op.vector(), elementOffsets, sliceSizes, strides);
    }

    rewriter.replaceOpWithNewOp<vector::TupleOp>(op, tupleType, tupleValues);
    return success();
  }
};

/// Progressive lowering of InsertSlicesOp to series of InsertStridedSliceOp.
/// One:
///   %x = vector.insert_slices %0
/// is replaced by:
///   %r0 = zero-result
///   %t1 = vector.tuple_get %0, 0
///   %r1 = vector.insert_strided_slice %r0, %t1
///   %t2 = vector.tuple_get %0, 1
///   %r2 = vector.insert_strided_slice %r1, %t2
///   ..
///   %x  = ..
class InsertSlicesOpLowering : public OpRewritePattern<vector::InsertSlicesOp> {
public:
  using OpRewritePattern<vector::InsertSlicesOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::InsertSlicesOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    VectorType vectorType = op.getResultVectorType();
    auto shape = vectorType.getShape();

    SmallVector<int64_t, 4> sizes;
    op.getSizes(sizes);
    SmallVector<int64_t, 4> strides;
    op.getStrides(strides); // all-ones at the moment

    // Prepare result.
    Value result = rewriter.create<ConstantOp>(
        loc, vectorType, rewriter.getZeroAttr(vectorType));

    // For each element in the tuple, extract the proper strided slice.
    TupleType tupleType = op.getSourceTupleType();
    int64_t tupleSize = tupleType.size();
    auto sliceStrides = computeStrides(shape, sizes);
    for (int64_t i = 0; i < tupleSize; ++i) {
      auto vectorOffsets = delinearize(sliceStrides, i);
      auto elementOffsets =
          computeElementOffsetsFromVectorSliceOffsets(sizes, vectorOffsets);
      // Extract from tuple into the result.
      auto index = rewriter.getI64IntegerAttr(i);
      auto tupleGet = rewriter.create<vector::TupleGetOp>(
          loc, tupleType.getType(i), op.getOperand(), index);
      result = rewriter.create<vector::InsertStridedSliceOp>(
          loc, tupleGet, result, elementOffsets, strides);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Progressive lowering of BroadcastOp.
class BroadcastOpLowering : public OpRewritePattern<vector::BroadcastOp> {
public:
  using OpRewritePattern<vector::BroadcastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::BroadcastOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    VectorType dstType = op.getVectorType();
    VectorType srcType = op.getSourceType().dyn_cast<VectorType>();
    Type eltType = dstType.getElementType();

    // Determine rank of source and destination.
    int64_t srcRank = srcType ? srcType.getRank() : 0;
    int64_t dstRank = dstType.getRank();

    // Duplicate this rank.
    // For example:
    //   %x = broadcast %y  : k-D to n-D, k < n
    // becomes:
    //   %b = broadcast %y  : k-D to (n-1)-D
    //   %x = [%b,%b,%b,%b] : n-D
    // becomes:
    //   %b = [%y,%y]       : (n-1)-D
    //   %x = [%b,%b,%b,%b] : n-D
    if (srcRank < dstRank) {
      // Scalar to any vector can use splat.
      if (srcRank == 0) {
        rewriter.replaceOpWithNewOp<SplatOp>(op, dstType, op.source());
        return success();
      }
      // Duplication.
      VectorType resType =
          VectorType::get(dstType.getShape().drop_front(), eltType);
      Value bcst =
          rewriter.create<vector::BroadcastOp>(loc, resType, op.source());
      Value result = rewriter.create<ConstantOp>(loc, dstType,
                                                 rewriter.getZeroAttr(dstType));
      for (int64_t d = 0, dim = dstType.getDimSize(0); d < dim; ++d)
        result = rewriter.create<vector::InsertOp>(loc, bcst, result, d);
      rewriter.replaceOp(op, result);
      return success();
    }

    // Find non-matching dimension, if any.
    assert(srcRank == dstRank);
    int64_t m = -1;
    for (int64_t r = 0; r < dstRank; r++)
      if (srcType.getDimSize(r) != dstType.getDimSize(r)) {
        m = r;
        break;
      }

    // All trailing dimensions are the same. Simply pass through.
    if (m == -1) {
      rewriter.replaceOp(op, op.source());
      return success();
    }

    // Stretching scalar inside vector (e.g. vector<1xf32>) can use splat.
    if (srcRank == 1) {
      assert(m == 0);
      Value ext = rewriter.create<vector::ExtractOp>(loc, op.source(), 0);
      rewriter.replaceOpWithNewOp<SplatOp>(op, dstType, ext);
      return success();
    }

    // Any non-matching dimension forces a stretch along this rank.
    // For example:
    //   %x = broadcast %y : vector<4x1x2xf32> to vector<4x2x2xf32>
    // becomes:
    //   %a = broadcast %y[0] : vector<1x2xf32> to vector<2x2xf32>
    //   %b = broadcast %y[1] : vector<1x2xf32> to vector<2x2xf32>
    //   %c = broadcast %y[2] : vector<1x2xf32> to vector<2x2xf32>
    //   %d = broadcast %y[3] : vector<1x2xf32> to vector<2x2xf32>
    //   %x = [%a,%b,%c,%d]
    // becomes:
    //   %u = broadcast %y[0][0] : vector<2xf32> to vector <2x2xf32>
    //   %v = broadcast %y[1][0] : vector<2xf32> to vector <2x2xf32>
    //   %a = [%u, %v]
    //   ..
    //   %x = [%a,%b,%c,%d]
    VectorType resType =
        VectorType::get(dstType.getShape().drop_front(), eltType);
    Value result = rewriter.create<ConstantOp>(loc, dstType,
                                               rewriter.getZeroAttr(dstType));
    if (m == 0) {
      // Stetch at start.
      Value ext = rewriter.create<vector::ExtractOp>(loc, op.source(), 0);
      Value bcst = rewriter.create<vector::BroadcastOp>(loc, resType, ext);
      for (int64_t d = 0, dim = dstType.getDimSize(0); d < dim; ++d)
        result = rewriter.create<vector::InsertOp>(loc, bcst, result, d);
    } else {
      // Stetch not at start.
      for (int64_t d = 0, dim = dstType.getDimSize(0); d < dim; ++d) {
        Value ext = rewriter.create<vector::ExtractOp>(loc, op.source(), d);
        Value bcst = rewriter.create<vector::BroadcastOp>(loc, resType, ext);
        result = rewriter.create<vector::InsertOp>(loc, bcst, result, d);
      }
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Progressive lowering of TransposeOp.
/// One:
///   %x = vector.transpose %y, [1, 0]
/// is replaced by:
///   %z = constant dense<0.000000e+00>
///   %0 = vector.extract %y[0, 0]
///   %1 = vector.insert %0, %z [0, 0]
///   ..
///   %x = vector.insert .., .. [.., ..]
class TransposeOpLowering : public OpRewritePattern<vector::TransposeOp> {
public:
  using OpRewritePattern<vector::TransposeOp>::OpRewritePattern;

  TransposeOpLowering(vector::VectorTransformsOptions vectorTransformsOptions,
                      MLIRContext *context)
      : OpRewritePattern<vector::TransposeOp>(context),
        vectorTransformsOptions(vectorTransformsOptions) {}

  LogicalResult matchAndRewrite(vector::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    VectorType resType = op.getResultType();

    // Set up convenience transposition table.
    SmallVector<int64_t, 4> transp;
    for (auto attr : op.transp())
      transp.push_back(attr.cast<IntegerAttr>().getInt());

    // Handle a true 2-D matrix transpose differently when requested.
    if (vectorTransformsOptions.vectorTransposeLowering ==
            vector::VectorTransposeLowering::Flat &&
        resType.getRank() == 2 && transp[0] == 1 && transp[1] == 0) {
      Type flattenedType =
          VectorType::get(resType.getNumElements(), resType.getElementType());
      auto matrix =
          rewriter.create<vector::ShapeCastOp>(loc, flattenedType, op.vector());
      auto rows = rewriter.getI32IntegerAttr(resType.getShape()[0]);
      auto columns = rewriter.getI32IntegerAttr(resType.getShape()[1]);
      Value trans = rewriter.create<vector::FlatTransposeOp>(
          loc, flattenedType, matrix, rows, columns);
      rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(op, resType, trans);
      return success();
    }

    // Generate fully unrolled extract/insert ops.
    Value result = rewriter.create<ConstantOp>(loc, resType,
                                               rewriter.getZeroAttr(resType));
    SmallVector<int64_t, 4> lhs(transp.size(), 0);
    SmallVector<int64_t, 4> rhs(transp.size(), 0);
    rewriter.replaceOp(op, expandIndices(loc, resType, 0, transp, lhs, rhs,
                                         op.vector(), result, rewriter));
    return success();
  }

private:
  // Builds the indices arrays for the lhs and rhs. Generates the extract/insert
  // operation when al ranks are exhausted.
  Value expandIndices(Location loc, VectorType resType, int64_t pos,
                      SmallVector<int64_t, 4> &transp,
                      SmallVector<int64_t, 4> &lhs,
                      SmallVector<int64_t, 4> &rhs, Value input, Value result,
                      PatternRewriter &rewriter) const {
    if (pos >= resType.getRank()) {
      auto ridx = rewriter.getI64ArrayAttr(rhs);
      auto lidx = rewriter.getI64ArrayAttr(lhs);
      Type eltType = resType.getElementType();
      Value e = rewriter.create<vector::ExtractOp>(loc, eltType, input, ridx);
      return rewriter.create<vector::InsertOp>(loc, resType, e, result, lidx);
    }
    for (int64_t d = 0, e = resType.getDimSize(pos); d < e; ++d) {
      lhs[pos] = d;
      rhs[transp[pos]] = d;
      result = expandIndices(loc, resType, pos + 1, transp, lhs, rhs, input,
                             result, rewriter);
    }
    return result;
  }

  /// Options to control the vector patterns.
  vector::VectorTransformsOptions vectorTransformsOptions;
};

/// Progressive lowering of OuterProductOp.
/// One:
///   %x = vector.outerproduct %lhs, %rhs, %acc
/// is replaced by:
///   %z = zero-result
///   %0 = vector.extract %lhs[0]
///   %1 = vector.broadcast %0
///   %2 = vector.extract %acc[0]
///   %3 = vector.fma %1, %rhs, %2
///   %4 = vector.insert %3, %z[0]
///   ..
///   %x = vector.insert %.., %..[N-1]
///
class OuterProductOpLowering : public OpRewritePattern<vector::OuterProductOp> {
public:
  using OpRewritePattern<vector::OuterProductOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::OuterProductOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    VectorType lhsType = op.getOperandVectorTypeLHS();
    VectorType rhsType = op.getOperandTypeRHS().dyn_cast<VectorType>();
    VectorType resType = op.getVectorType();
    Type eltType = resType.getElementType();
    bool isInt = eltType.isa<IntegerType>();
    Value acc = (op.acc().empty()) ? nullptr : op.acc()[0];

    if (!rhsType) {
      // Special case: AXPY operation.
      Value b = rewriter.create<vector::BroadcastOp>(loc, lhsType, op.rhs());
      rewriter.replaceOp(op, genMult(loc, op.lhs(), b, acc, isInt, rewriter));
      return success();
    }

    Value result = rewriter.create<ConstantOp>(loc, resType,
                                               rewriter.getZeroAttr(resType));
    for (int64_t d = 0, e = resType.getDimSize(0); d < e; ++d) {
      auto pos = rewriter.getI64ArrayAttr(d);
      Value x = rewriter.create<vector::ExtractOp>(loc, eltType, op.lhs(), pos);
      Value a = rewriter.create<vector::BroadcastOp>(loc, rhsType, x);
      Value r = nullptr;
      if (acc)
        r = rewriter.create<vector::ExtractOp>(loc, rhsType, acc, pos);
      Value m = genMult(loc, a, op.rhs(), r, isInt, rewriter);
      result = rewriter.create<vector::InsertOp>(loc, resType, m, result, pos);
    }
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  static Value genMult(Location loc, Value x, Value y, Value acc, bool isInt,
                       PatternRewriter &rewriter) {
    if (acc) {
      if (isInt)
        return rewriter.create<AddIOp>(loc, rewriter.create<MulIOp>(loc, x, y),
                                       acc);
      return rewriter.create<vector::FMAOp>(loc, x, y, acc);
    }
    if (isInt)
      return rewriter.create<MulIOp>(loc, x, y);
    return rewriter.create<MulFOp>(loc, x, y);
  }
};

/// Progressive lowering of ConstantMaskOp.
/// One:
///   %x = vector.constant_mask [a,b]
/// is replaced by:
///   %z = zero-result
///   %l = vector.constant_mask [b]
///   %4 = vector.insert %l, %z[0]
///   ..
///   %x = vector.insert %l, %..[a-1]
/// until a one-dimensional vector is reached. All these operations
/// will be folded at LLVM IR level.
class ConstantMaskOpLowering : public OpRewritePattern<vector::ConstantMaskOp> {
public:
  using OpRewritePattern<vector::ConstantMaskOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ConstantMaskOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto dstType = op.getResult().getType().cast<VectorType>();
    auto eltType = dstType.getElementType();
    auto dimSizes = op.mask_dim_sizes();
    int64_t rank = dimSizes.size();
    int64_t trueDim = std::min(dstType.getDimSize(0),
                               dimSizes[0].cast<IntegerAttr>().getInt());

    if (rank == 1) {
      // Express constant 1-D case in explicit vector form:
      //   [T,..,T,F,..,F].
      SmallVector<bool, 4> values(dstType.getDimSize(0));
      for (int64_t d = 0; d < trueDim; d++)
        values[d] = true;
      rewriter.replaceOpWithNewOp<ConstantOp>(
          op, dstType, rewriter.getBoolVectorAttr(values));
      return success();
    }

    VectorType lowType =
        VectorType::get(dstType.getShape().drop_front(), eltType);
    SmallVector<int64_t, 4> newDimSizes;
    for (int64_t r = 1; r < rank; r++)
      newDimSizes.push_back(dimSizes[r].cast<IntegerAttr>().getInt());
    Value trueVal = rewriter.create<vector::ConstantMaskOp>(
        loc, lowType, rewriter.getI64ArrayAttr(newDimSizes));
    Value result = rewriter.create<ConstantOp>(loc, dstType,
                                               rewriter.getZeroAttr(dstType));
    for (int64_t d = 0; d < trueDim; d++) {
      auto pos = rewriter.getI64ArrayAttr(d);
      result =
          rewriter.create<vector::InsertOp>(loc, dstType, trueVal, result, pos);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Progressive lowering of CreateMaskOp.
/// One:
///   %x = vector.create_mask %a, ... : vector<dx...>
/// is replaced by:
///   %l = vector.create_mask ... : vector<...>  ; one lower rank
///   %0 = cmpi "slt", %ci, %a       |
///   %1 = select %0, %l, %zeroes    |
///   %r = vector.insert %1, %pr [i] | d-times
///   %x = ....
/// until a one-dimensional vector is reached.
class CreateMaskOpLowering : public OpRewritePattern<vector::CreateMaskOp> {
public:
  using OpRewritePattern<vector::CreateMaskOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::CreateMaskOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto dstType = op.getResult().getType().cast<VectorType>();
    auto eltType = dstType.getElementType();
    int64_t dim = dstType.getDimSize(0);
    int64_t rank = dstType.getRank();
    Value idx = op.getOperand(0);

    if (rank == 1)
      return failure(); // leave for lowering

    VectorType lowType =
        VectorType::get(dstType.getShape().drop_front(), eltType);
    Value trueVal = rewriter.create<vector::CreateMaskOp>(
        loc, lowType, op.getOperands().drop_front());
    Value falseVal = rewriter.create<ConstantOp>(loc, lowType,
                                                 rewriter.getZeroAttr(lowType));
    Value result = rewriter.create<ConstantOp>(loc, dstType,
                                               rewriter.getZeroAttr(dstType));
    for (int64_t d = 0; d < dim; d++) {
      Value bnd = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(d));
      Value val = rewriter.create<CmpIOp>(loc, CmpIPredicate::slt, bnd, idx);
      Value sel = rewriter.create<SelectOp>(loc, val, trueVal, falseVal);
      auto pos = rewriter.getI64ArrayAttr(d);
      result =
          rewriter.create<vector::InsertOp>(loc, dstType, sel, result, pos);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// ShapeOp 2D -> 1D downcast serves the purpose of flattening 2-D to 1-D
/// vectors progressively on the way to target llvm.matrix intrinsics.
/// This iterates over the most major dimension of the 2-D vector and performs
/// rewrites into:
///   vector.extract from 2-D + vector.insert_strided_slice offset into 1-D
class ShapeCastOp2DDownCastRewritePattern
    : public OpRewritePattern<vector::ShapeCastOp> {
public:
  using OpRewritePattern<vector::ShapeCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ShapeCastOp op,
                                PatternRewriter &rewriter) const override {
    auto sourceVectorType = op.getSourceVectorType();
    auto resultVectorType = op.getResultVectorType();
    if (sourceVectorType.getRank() != 2 || resultVectorType.getRank() != 1)
      return failure();

    auto loc = op.getLoc();
    Value desc = rewriter.create<ConstantOp>(
        loc, resultVectorType, rewriter.getZeroAttr(resultVectorType));
    unsigned mostMinorVectorSize = sourceVectorType.getShape()[1];
    for (int64_t i = 0, e = sourceVectorType.getShape().front(); i != e; ++i) {
      Value vec = rewriter.create<vector::ExtractOp>(loc, op.source(), i);
      desc = rewriter.create<vector::InsertStridedSliceOp>(
          loc, vec, desc,
          /*offsets=*/i * mostMinorVectorSize, /*strides=*/1);
    }
    rewriter.replaceOp(op, desc);
    return success();
  }
};

/// ShapeOp 1D -> 2D upcast serves the purpose of unflattening 2-D from 1-D
/// vectors progressively on the way from targeting llvm.matrix intrinsics.
/// This iterates over the most major dimension of the 2-D vector and performs
/// rewrites into:
///   vector.strided_slice from 1-D + vector.insert into 2-D
class ShapeCastOp2DUpCastRewritePattern
    : public OpRewritePattern<vector::ShapeCastOp> {
public:
  using OpRewritePattern<vector::ShapeCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ShapeCastOp op,
                                PatternRewriter &rewriter) const override {
    auto sourceVectorType = op.getSourceVectorType();
    auto resultVectorType = op.getResultVectorType();
    if (sourceVectorType.getRank() != 1 || resultVectorType.getRank() != 2)
      return failure();

    auto loc = op.getLoc();
    Value desc = rewriter.create<ConstantOp>(
        loc, resultVectorType, rewriter.getZeroAttr(resultVectorType));
    unsigned mostMinorVectorSize = resultVectorType.getShape()[1];
    for (int64_t i = 0, e = resultVectorType.getShape().front(); i != e; ++i) {
      Value vec = rewriter.create<vector::ExtractStridedSliceOp>(
          loc, op.source(), /*offsets=*/i * mostMinorVectorSize,
          /*sizes=*/mostMinorVectorSize,
          /*strides=*/1);
      desc = rewriter.create<vector::InsertOp>(loc, vec, desc, i);
    }
    rewriter.replaceOp(op, desc);
    return success();
  }
};

// We typically should not lower general shape cast operations into data
// movement instructions, since the assumption is that these casts are
// optimized away during progressive lowering. For completeness, however,
// we fall back to a reference implementation that moves all elements
// into the right place if we get here.
class ShapeCastOpRewritePattern : public OpRewritePattern<vector::ShapeCastOp> {
public:
  using OpRewritePattern<vector::ShapeCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ShapeCastOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto sourceVectorType = op.getSourceVectorType();
    auto resultVectorType = op.getResultVectorType();
    // Intended 2D/1D lowerings with better implementations.
    int64_t srcRank = sourceVectorType.getRank();
    int64_t resRank = resultVectorType.getRank();
    if ((srcRank == 2 && resRank == 1) || (srcRank == 1 && resRank == 2))
      return failure();
    // Compute number of elements involved in the reshape.
    int64_t numElts = 1;
    for (int64_t r = 0; r < srcRank; r++)
      numElts *= sourceVectorType.getDimSize(r);
    // Replace with data movement operations:
    //    x[0,0,0] = y[0,0]
    //    x[0,0,1] = y[0,1]
    //    x[0,1,0] = y[0,2]
    // etc., incrementing the two index vectors "row-major"
    // within the source and result shape.
    SmallVector<int64_t, 4> srcIdx(srcRank);
    SmallVector<int64_t, 4> resIdx(resRank);
    Value result = rewriter.create<ConstantOp>(
        loc, resultVectorType, rewriter.getZeroAttr(resultVectorType));
    for (int64_t i = 0; i < numElts; i++) {
      if (i != 0) {
        incIdx(srcIdx, sourceVectorType, srcRank - 1);
        incIdx(resIdx, resultVectorType, resRank - 1);
      }
      Value e = rewriter.create<vector::ExtractOp>(loc, op.source(), srcIdx);
      result = rewriter.create<vector::InsertOp>(loc, e, result, resIdx);
    }
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  static void incIdx(SmallVector<int64_t, 4> &idx, VectorType tp, int64_t r) {
    assert(0 <= r && r < tp.getRank());
    if (++idx[r] == tp.getDimSize(r)) {
      idx[r] = 0;
      incIdx(idx, tp, r - 1);
    }
  }
};

} // namespace

namespace mlir {

/// Progressively lower a `vector.contract %a, %b, %c` with row-major matmul
/// semantics to:
/// ```
///    %flattened_a = vector.shape_cast %a
///    %flattened_b = vector.shape_cast %b
///    %flattened_d = vector.matmul %flattened_a, %flattened_b
///    %d = vector.shape_cast %%flattened_d
///    %e = add %c, %d
/// ```
/// `vector.matmul` later lowers to `llvm.matrix.multiply`.
//
/// This only kicks in when VectorTransformsOptions is set to OuterProduct and
/// the vector.contract op is a row-major matrix multiply.
LogicalResult ContractionOpToMatmulOpLowering::matchAndRewrite(
    vector::ContractionOp op, PatternRewriter &rewriter) const {
  // TODO: implement masks
  if (llvm::size(op.masks()) != 0)
    return failure();
  if (vectorTransformsOptions.vectorContractLowering !=
      vector::VectorContractLowering::Matmul)
    return failure();
  if (failed(filter(op)))
    return failure();

  auto iteratorTypes = op.iterator_types().getValue();
  if (!isParallelIterator(iteratorTypes[0]) ||
      !isParallelIterator(iteratorTypes[1]) ||
      !isReductionIterator(iteratorTypes[2]))
    return failure();

  if (!isRowMajorMatmul(op.indexing_maps()))
    return failure();

  Type elementType = op.getLhsType().getElementType();
  if (!elementType.isIntOrFloat())
    return failure();

  VectorType lhsType = op.getLhsType();
  VectorType rhsType = op.getRhsType();
  int64_t lhsRows = lhsType.getDimSize(0);
  int64_t lhsColumns = lhsType.getDimSize(1);
  int64_t rhsColumns = rhsType.getDimSize(1);

  Type flattenedLHSType =
      VectorType::get(lhsType.getNumElements(), lhsType.getElementType());
  Type flattenedRHSType =
      VectorType::get(rhsType.getNumElements(), rhsType.getElementType());
  auto lhs = rewriter.create<vector::ShapeCastOp>(op.getLoc(), flattenedLHSType,
                                                  op.lhs());
  auto rhs = rewriter.create<vector::ShapeCastOp>(op.getLoc(), flattenedRHSType,
                                                  op.rhs());

  Value mul = rewriter.create<vector::MatmulOp>(op.getLoc(), lhs, rhs, lhsRows,
                                                lhsColumns, rhsColumns);
  mul = rewriter.create<vector::ShapeCastOp>(op.getLoc(), op.acc().getType(),
                                             mul);
  if (elementType.isa<IntegerType>())
    rewriter.replaceOpWithNewOp<AddIOp>(op, op.acc(), mul);
  else
    rewriter.replaceOpWithNewOp<AddFOp>(op, op.acc(), mul);

  return success();
}

/// Progressively lower a `vector.contract %a, %b, %c` with row-major matmul
/// semantics to a reduction_size-unrolled sequence:
/// ```
///    %at = vector.transpose %a, [1, 0]
///    %bRow0 = vector.extract %b[0]
///    %atRow0 = vector.extract %at[0]
///    %c0 = vector.outerproduct %atRow0, %bRow0, %c
///    ...
///    %bRowK = vector.extract %b[K]
///    %atRowK = vector.extract %at[K]
///    %cK = vector.outerproduct %atRowK, %bRowK, %cK-1
/// ```
///
/// This only kicks in when VectorTransformsOptions is set to OuterProduct but
/// otherwise supports any layout permutation of the matrix-multiply.
LogicalResult ContractionOpToOuterProductOpLowering::matchAndRewrite(
    vector::ContractionOp op, PatternRewriter &rewriter) const {
  // TODO: implement masks
  if (llvm::size(op.masks()) != 0)
    return failure();

  if (vectorTransformsOptions.vectorContractLowering !=
      vector::VectorContractLowering::OuterProduct)
    return failure();

  if (failed(filter(op)))
    return failure();

  Location loc = op.getLoc();
  int64_t reductionSize = 0;
  VectorType lhsType = op.getLhsType();
  Value lhs = op.lhs(), rhs = op.rhs(), res = op.acc();

  // Set up the parallel/reduction structure in right form.
  using MapList = ArrayRef<ArrayRef<AffineExpr>>;
  auto infer = [](MapList m) { return AffineMap::inferFromExprList(m); };
  AffineExpr m, n, k;
  bindDims(rewriter.getContext(), m, n, k);
  static constexpr std::array<int64_t, 2> perm = {1, 0};
  auto iteratorTypes = op.iterator_types().getValue();
  SmallVector<AffineMap, 4> maps = op.getIndexingMaps();
  if (isParallelIterator(iteratorTypes[0]) &&
      isParallelIterator(iteratorTypes[1]) &&
      isReductionIterator(iteratorTypes[2])) {
    //
    // Two outer parallel, one inner reduction (matmat flavor).
    //
    if (maps == infer({{m, k}, {k, n}, {m, n}})) {
      // This is the classical row-major matmul. Just permute the lhs.
      reductionSize = lhsType.getDimSize(1);
      lhs = rewriter.create<vector::TransposeOp>(loc, lhs, perm);
    } else if (maps == infer({{m, k}, {n, k}, {m, n}})) {
      // TODO: may be better to fail and use some vector<k> -> scalar reduction.
      reductionSize = lhsType.getDimSize(1);
      lhs = rewriter.create<vector::TransposeOp>(loc, lhs, perm);
      rhs = rewriter.create<vector::TransposeOp>(loc, rhs, perm);
    } else if (maps == infer({{k, m}, {k, n}, {m, n}})) {
      // No need to permute anything.
      reductionSize = lhsType.getDimSize(0);
    } else if (maps == infer({{k, m}, {n, k}, {m, n}})) {
      // Just permute the rhs.
      reductionSize = lhsType.getDimSize(0);
      rhs = rewriter.create<vector::TransposeOp>(loc, rhs, perm);
    } else if (maps == infer({{m, k}, {k, n}, {n, m}})) {
      // This is the classical row-major matmul. Just permute the lhs.
      reductionSize = lhsType.getDimSize(1);
      Value tmp = rhs;
      rhs = rewriter.create<vector::TransposeOp>(loc, lhs, perm);
      lhs = tmp;
    } else if (maps == infer({{m, k}, {n, k}, {n, m}})) {
      // TODO: may be better to fail and use some vector<k> -> scalar reduction.
      reductionSize = lhsType.getDimSize(1);
      Value tmp = rhs;
      rhs = rewriter.create<vector::TransposeOp>(loc, lhs, perm);
      lhs = rewriter.create<vector::TransposeOp>(loc, tmp, perm);
    } else if (maps == infer({{k, m}, {k, n}, {n, m}})) {
      // No need to permute anything, but still swap lhs and rhs.
      reductionSize = lhsType.getDimSize(0);
      std::swap(lhs, rhs);
    } else if (maps == infer({{k, m}, {n, k}, {n, m}})) {
      // Just permute the rhs.
      reductionSize = lhsType.getDimSize(0);
      Value tmp = lhs;
      lhs = rewriter.create<vector::TransposeOp>(loc, rhs, perm);
      rhs = tmp;
    } else {
      return failure();
    }
  } else if (isParallelIterator(iteratorTypes[0]) &&
             isReductionIterator(iteratorTypes[1])) {
    //
    // One outer parallel, one inner reduction (matvec flavor)
    //
    if (maps == infer({{m, n}, {n}, {m}})) {
      // Case mat-vec: transpose.
      reductionSize = lhsType.getDimSize(1);
      lhs = rewriter.create<vector::TransposeOp>(loc, lhs, perm);
    } else if (maps == infer({{n, m}, {n}, {m}})) {
      // Case mat-trans-vec: ready to go.
      reductionSize = lhsType.getDimSize(0);
    } else if (maps == infer({{n}, {m, n}, {m}})) {
      // Case vec-mat: swap and transpose.
      reductionSize = lhsType.getDimSize(0);
      std::swap(lhs, rhs);
      lhs = rewriter.create<vector::TransposeOp>(loc, lhs, perm);
    } else if (maps == infer({{n}, {n, m}, {m}})) {
      // Case vec-mat-trans: swap and ready to go.
      reductionSize = lhsType.getDimSize(0);
      std::swap(lhs, rhs);
    } else {
      return failure();
    }
  } else {
    return failure();
  }
  assert(reductionSize > 0);

  // Unroll outer-products along reduction.
  for (int64_t k = 0; k < reductionSize; ++k) {
    Value a = rewriter.create<vector::ExtractOp>(op.getLoc(), lhs, k);
    Value b = rewriter.create<vector::ExtractOp>(op.getLoc(), rhs, k);
    res = rewriter.create<vector::OuterProductOp>(op.getLoc(), a, b, res);
  }
  rewriter.replaceOp(op, res);
  return success();
}

LogicalResult
ContractionOpToDotLowering::matchAndRewrite(vector::ContractionOp op,
                                            PatternRewriter &rewriter) const {
  // TODO: implement masks
  if (llvm::size(op.masks()) != 0)
    return failure();

  if (failed(filter(op)))
    return failure();

  if (vectorTransformsOptions.vectorContractLowering !=
      vector::VectorContractLowering::Dot)
    return failure();

  auto iteratorTypes = op.iterator_types().getValue();
  static constexpr std::array<int64_t, 2> perm = {1, 0};
  Location loc = op.getLoc();
  Value lhs = op.lhs(), rhs = op.rhs();

  using MapList = ArrayRef<ArrayRef<AffineExpr>>;
  auto infer = [](MapList m) { return AffineMap::inferFromExprList(m); };
  AffineExpr m, n, k;
  bindDims(rewriter.getContext(), m, n, k);
  SmallVector<AffineMap, 4> maps = op.getIndexingMaps();
  //
  // In the following we wish to make the reduction dimension innermost so we
  // can load vectors and just fmul + reduce into a scalar.
  //
  if (isParallelIterator(iteratorTypes[0]) &&
      isParallelIterator(iteratorTypes[1]) &&
      isReductionIterator(iteratorTypes[2])) {
    //
    // Two outer parallel, one inner reduction (matmat flavor).
    //
    if (maps == infer({{m, k}, {k, n}, {m, n}})) {
      rhs = rewriter.create<vector::TransposeOp>(loc, rhs, perm);
    } else if (maps == infer({{m, k}, {n, k}, {m, n}})) {
      // No need to permute anything.
    } else if (maps == infer({{k, m}, {k, n}, {m, n}})) {
      lhs = rewriter.create<vector::TransposeOp>(loc, lhs, perm);
      rhs = rewriter.create<vector::TransposeOp>(loc, rhs, perm);
    } else if (maps == infer({{k, m}, {n, k}, {m, n}})) {
      lhs = rewriter.create<vector::TransposeOp>(loc, lhs, perm);
    } else if (maps == infer({{m, k}, {k, n}, {n, m}})) {
      // This is the classical row-major matmul. Just permute the lhs.
      Value tmp = lhs;
      lhs = rewriter.create<vector::TransposeOp>(loc, rhs, perm);
      rhs = tmp;
    } else if (maps == infer({{m, k}, {n, k}, {n, m}})) {
      std::swap(lhs, rhs);
    } else if (maps == infer({{k, m}, {k, n}, {n, m}})) {
      Value tmp = lhs;
      lhs = rewriter.create<vector::TransposeOp>(loc, rhs, perm);
      rhs = rewriter.create<vector::TransposeOp>(loc, tmp, perm);
    } else if (maps == infer({{k, m}, {n, k}, {n, m}})) {
      Value tmp = rhs;
      rhs = rewriter.create<vector::TransposeOp>(loc, lhs, perm);
      lhs = tmp;
    } else {
      return failure();
    }
  } else if (isParallelIterator(iteratorTypes[0]) &&
             isReductionIterator(iteratorTypes[1])) {
    //
    // One outer parallel, one inner reduction (matvec flavor)
    //
    if (maps == infer({{m, n}, {n}, {m}})) {
      // No need to permute anything.
    } else if (maps == infer({{n, m}, {n}, {m}})) {
      lhs = rewriter.create<vector::TransposeOp>(loc, lhs, perm);
    } else if (maps == infer({{n}, {m, n}, {m}})) {
      std::swap(lhs, rhs);
    } else if (maps == infer({{n}, {n, m}, {m}})) {
      std::swap(lhs, rhs);
      lhs = rewriter.create<vector::TransposeOp>(loc, lhs, perm);
    } else {
      return failure();
    }
  } else {
    return failure();
  }

  VectorType dstType = op.getResultType().cast<VectorType>();
  assert(dstType.getRank() >= 1 && dstType.getRank() <= 2 &&
         "Expected dst type of rank 1 or 2");

  unsigned rank = dstType.getRank();
  unsigned dstRows = dstType.getShape()[0];
  unsigned dstColumns = rank == 1 ? 1 : dstType.getShape()[1];

  // ExtractOp does not allow dynamic indexing, we must unroll explicitly.
  Value res =
      rewriter.create<ConstantOp>(loc, dstType, rewriter.getZeroAttr(dstType));
  for (unsigned r = 0; r < dstRows; ++r) {
    Value a = rewriter.create<vector::ExtractOp>(op.getLoc(), lhs, r);
    for (unsigned c = 0; c < dstColumns; ++c) {
      Value b = rank == 1
                    ? rhs
                    : rewriter.create<vector::ExtractOp>(op.getLoc(), rhs, c);
      Value m = rewriter.create<MulFOp>(op.getLoc(), a, b);
      Value reduced = rewriter.create<vector::ReductionOp>(
          op.getLoc(), dstType.getElementType(), rewriter.getStringAttr("add"),
          m, ValueRange{});

      SmallVector<int64_t, 2> pos = rank == 1 ? SmallVector<int64_t, 2>{r}
                                              : SmallVector<int64_t, 2>{r, c};
      res = rewriter.create<vector::InsertOp>(op.getLoc(), reduced, res, pos);
    }
  }
  if (auto acc = op.acc())
    res = rewriter.create<AddFOp>(op.getLoc(), res, acc);
  rewriter.replaceOp(op, res);
  return success();
}

/// Progressive lowering of ContractionOp.
/// One:
///   %x = vector.contract with at least one free/batch dimension
/// is replaced by:
///   %a = vector.contract with one less free/batch dimension
///   %b = vector.contract with one less free/batch dimension
///   ..
///   %x = combine %a %b ..
/// until a pure contraction is reached (no free/batch dimensions),
/// which is replaced by a dot-product.
///
/// This only kicks in when either VectorTransformsOptions is set
/// to DOT or when other contraction patterns fail.
//
// TODO: break down into transpose/reshape/cast ops
//               when they become available to avoid code dup
// TODO: investigate lowering order impact on performance
LogicalResult
ContractionOpLowering::matchAndRewrite(vector::ContractionOp op,
                                       PatternRewriter &rewriter) const {
  // TODO: implement masks.
  if (llvm::size(op.masks()) != 0)
    return failure();

  if (failed(filter(op)))
    return failure();

  // TODO: support mixed mode contract lowering.
  if (op.getLhsType().getElementType() !=
          getElementTypeOrSelf(op.getAccType()) ||
      op.getRhsType().getElementType() != getElementTypeOrSelf(op.getAccType()))
    return failure();

  // TODO: implement benefits, cost models.
  MLIRContext *ctx = op.getContext();
  ContractionOpToMatmulOpLowering pat1(vectorTransformsOptions, ctx);
  if (succeeded(pat1.matchAndRewrite(op, rewriter)))
    return success();
  ContractionOpToOuterProductOpLowering pat2(vectorTransformsOptions, ctx);
  if (succeeded(pat2.matchAndRewrite(op, rewriter)))
    return success();
  ContractionOpToDotLowering pat3(vectorTransformsOptions, ctx);
  if (succeeded(pat3.matchAndRewrite(op, rewriter)))
    return success();

  // Find first batch dimension in LHS/RHS, and lower when found.
  std::vector<std::pair<int64_t, int64_t>> batchDimMap = op.getBatchDimMap();
  if (!batchDimMap.empty()) {
    int64_t lhsIndex = batchDimMap[0].first;
    int64_t rhsIndex = batchDimMap[0].second;
    rewriter.replaceOp(op, lowerParallel(op, lhsIndex, rhsIndex, rewriter));
    return success();
  }

  // Collect contracting dimensions.
  std::vector<std::pair<int64_t, int64_t>> contractingDimMap =
      op.getContractingDimMap();
  DenseSet<int64_t> lhsContractingDimSet;
  DenseSet<int64_t> rhsContractingDimSet;
  for (auto &dimPair : contractingDimMap) {
    lhsContractingDimSet.insert(dimPair.first);
    rhsContractingDimSet.insert(dimPair.second);
  }

  // Find first free dimension in LHS, and lower when found.
  VectorType lhsType = op.getLhsType();
  for (int64_t lhsIndex = 0, e = lhsType.getRank(); lhsIndex < e; ++lhsIndex) {
    if (lhsContractingDimSet.count(lhsIndex) == 0) {
      rewriter.replaceOp(
          op, lowerParallel(op, lhsIndex, /*rhsIndex=*/-1, rewriter));
      return success();
    }
  }

  // Find first free dimension in RHS, and lower when found.
  VectorType rhsType = op.getRhsType();
  for (int64_t rhsIndex = 0, e = rhsType.getRank(); rhsIndex < e; ++rhsIndex) {
    if (rhsContractingDimSet.count(rhsIndex) == 0) {
      rewriter.replaceOp(
          op, lowerParallel(op, /*lhsIndex=*/-1, rhsIndex, rewriter));
      return success();
    }
  }

  // Lower the first remaining reduction dimension.
  if (!contractingDimMap.empty()) {
    rewriter.replaceOp(op, lowerReduction(op, rewriter));
    return success();
  }

  return failure();
}

// Lower one parallel dimension.
// TODO: consider reusing existing contract unrolling
Value ContractionOpLowering::lowerParallel(vector::ContractionOp op,
                                           int64_t lhsIndex, int64_t rhsIndex,
                                           PatternRewriter &rewriter) const {
  VectorType lhsType = op.getLhsType();
  VectorType rhsType = op.getRhsType();
  VectorType resType = op.getResultType().cast<VectorType>();
  // Find the iterator type index and result index.
  SmallVector<AffineMap, 4> iMap = op.getIndexingMaps();
  int64_t iterIndex = -1;
  int64_t dimSize = -1;
  if (lhsIndex >= 0) {
    iterIndex = iMap[0].getDimPosition(lhsIndex);
    assert((rhsIndex < 0 || iterIndex == iMap[1].getDimPosition(rhsIndex)) &&
           "parallel index should be free in LHS or batch in LHS/RHS");
    dimSize = lhsType.getDimSize(lhsIndex);
  } else {
    assert(rhsIndex >= 0 && "missing parallel index");
    iterIndex = iMap[1].getDimPosition(rhsIndex);
    dimSize = rhsType.getDimSize(rhsIndex);
  }
  assert(iterIndex >= 0 && "parallel index not listed in operand mapping");
  Optional<int64_t> lookup = getResultIndex(iMap[2], iterIndex);
  assert(lookup.hasValue() && "parallel index not listed in reduction");
  int64_t resIndex = lookup.getValue();
  // Construct new iterator types and affine map array attribute.
  std::array<AffineMap, 3> lowIndexingMaps = {
      adjustMap(iMap[0], iterIndex, rewriter),
      adjustMap(iMap[1], iterIndex, rewriter),
      adjustMap(iMap[2], iterIndex, rewriter)};
  auto lowAffine = rewriter.getAffineMapArrayAttr(lowIndexingMaps);
  auto lowIter =
      rewriter.getArrayAttr(adjustIter(op.iterator_types(), iterIndex));
  // Unroll into a series of lower dimensional vector.contract ops.
  Location loc = op.getLoc();
  Value result =
      rewriter.create<ConstantOp>(loc, resType, rewriter.getZeroAttr(resType));
  for (int64_t d = 0; d < dimSize; ++d) {
    auto lhs = reshapeLoad(loc, op.lhs(), lhsType, lhsIndex, d, rewriter);
    auto rhs = reshapeLoad(loc, op.rhs(), rhsType, rhsIndex, d, rewriter);
    auto acc = reshapeLoad(loc, op.acc(), resType, resIndex, d, rewriter);
    Value lowContract = rewriter.create<vector::ContractionOp>(
        loc, lhs, rhs, acc, lowAffine, lowIter);
    result =
        reshapeStore(loc, lowContract, result, resType, resIndex, d, rewriter);
  }
  return result;
}

// Lower one reduction dimension.
Value ContractionOpLowering::lowerReduction(vector::ContractionOp op,
                                            PatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  VectorType lhsType = op.getLhsType();
  VectorType rhsType = op.getRhsType();
  Type resType = op.getResultType();
  assert(!resType.isa<VectorType>());
  // Use iterator index 0.
  int64_t iterIndex = 0;
  SmallVector<AffineMap, 4> iMap = op.getIndexingMaps();
  Optional<int64_t> lookupLhs = getResultIndex(iMap[0], iterIndex);
  Optional<int64_t> lookupRhs = getResultIndex(iMap[1], iterIndex);
  assert(lookupLhs.hasValue() && "missing LHS parallel index");
  assert(lookupRhs.hasValue() && "missing RHS parallel index");
  int64_t lhsIndex = lookupLhs.getValue();
  int64_t rhsIndex = lookupRhs.getValue();
  int64_t dimSize = lhsType.getDimSize(lhsIndex);
  assert(dimSize == rhsType.getDimSize(rhsIndex) && "corrupt shape");
  // Base case.
  if (lhsType.getRank() == 1) {
    assert(rhsType.getRank() == 1 && "corrupt contraction");
    Value m = rewriter.create<MulFOp>(loc, op.lhs(), op.rhs());
    StringAttr kind = rewriter.getStringAttr("add");
    return rewriter.create<vector::ReductionOp>(loc, resType, kind, m,
                                                op.acc());
  }
  // Construct new iterator types and affine map array attribute.
  std::array<AffineMap, 3> lowIndexingMaps = {
      adjustMap(iMap[0], iterIndex, rewriter),
      adjustMap(iMap[1], iterIndex, rewriter),
      adjustMap(iMap[2], iterIndex, rewriter)};
  auto lowAffine = rewriter.getAffineMapArrayAttr(lowIndexingMaps);
  auto lowIter =
      rewriter.getArrayAttr(adjustIter(op.iterator_types(), iterIndex));
  // Unroll into a series of lower dimensional vector.contract ops.
  // By feeding the initial accumulator into the first contraction,
  // and the result of each contraction into the next, eventually
  // the sum of all reductions is computed.
  Value result = op.acc();
  for (int64_t d = 0; d < dimSize; ++d) {
    auto lhs = reshapeLoad(loc, op.lhs(), lhsType, lhsIndex, d, rewriter);
    auto rhs = reshapeLoad(loc, op.rhs(), rhsType, rhsIndex, d, rewriter);
    result = rewriter.create<vector::ContractionOp>(loc, lhs, rhs, result,
                                                    lowAffine, lowIter);
  }
  return result;
}

} // namespace mlir

static Optional<int64_t> extractConstantIndex(Value v) {
  if (auto cstOp = v.getDefiningOp<ConstantIndexOp>())
    return cstOp.getValue();
  if (auto affineApplyOp = v.getDefiningOp<AffineApplyOp>())
    if (affineApplyOp.getAffineMap().isSingleConstant())
      return affineApplyOp.getAffineMap().getSingleConstantResult();
  return None;
}

// Missing foldings of scf.if make it necessary to perform poor man's folding
// eagerly, especially in the case of unrolling. In the future, this should go
// away once scf.if folds properly.
static Value createScopedFoldedSLE(Value v, Value ub) {
  using namespace edsc::op;
  auto maybeCstV = extractConstantIndex(v);
  auto maybeCstUb = extractConstantIndex(ub);
  if (maybeCstV && maybeCstUb && *maybeCstV < *maybeCstUb)
    return Value();
  return sle(v, ub);
}

// Operates under a scoped context to build the condition to ensure that a
// particular VectorTransferOpInterface is unmasked.
static Value createScopedInBoundsCond(VectorTransferOpInterface xferOp) {
  assert(xferOp.permutation_map().isMinorIdentity() &&
         "Expected minor identity map");
  Value inBoundsCond;
  xferOp.zipResultAndIndexing([&](int64_t resultIdx, int64_t indicesIdx) {
    // Zip over the resulting vector shape and memref indices.
    // If the dimension is known to be unmasked, it does not participate in the
    // construction of `inBoundsCond`.
    if (!xferOp.isMaskedDim(resultIdx))
      return;
    int64_t vectorSize = xferOp.getVectorType().getDimSize(resultIdx);
    using namespace edsc::op;
    using namespace edsc::intrinsics;
    // Fold or create the check that `index + vector_size` <= `memref_size`.
    Value sum = xferOp.indices()[indicesIdx] + std_constant_index(vectorSize);
    Value cond =
        createScopedFoldedSLE(sum, std_dim(xferOp.memref(), indicesIdx));
    if (!cond)
      return;
    // Conjunction over all dims for which we are in-bounds.
    inBoundsCond = inBoundsCond ? inBoundsCond && cond : cond;
  });
  return inBoundsCond;
}

LogicalResult mlir::vector::splitFullAndPartialTransferPrecondition(
    VectorTransferOpInterface xferOp) {
  // TODO: expand support to these 2 cases.
  if (!xferOp.permutation_map().isMinorIdentity())
    return failure();
  // Must have some masked dimension to be a candidate for splitting.
  if (!xferOp.hasMaskedDim())
    return failure();
  // Don't split transfer operations directly under IfOp, this avoids applying
  // the pattern recursively.
  // TODO: improve the filtering condition to make it more applicable.
  if (isa<scf::IfOp>(xferOp.getOperation()->getParentOp()))
    return failure();
  return success();
}

/// Given two MemRefTypes `aT` and `bT`, return a MemRefType to which both can
/// be cast. If the MemRefTypes don't have the same rank or are not strided,
/// return null; otherwise:
///   1. if `aT` and `bT` are cast-compatible, return `aT`.
///   2. else return a new MemRefType obtained by iterating over the shape and
///   strides and:
///     a. keeping the ones that are static and equal across `aT` and `bT`.
///     b. using a dynamic shape and/or stride for the dimensions that don't
///        agree.
static MemRefType getCastCompatibleMemRefType(MemRefType aT, MemRefType bT) {
  if (MemRefCastOp::areCastCompatible(aT, bT))
    return aT;
  if (aT.getRank() != bT.getRank())
    return MemRefType();
  int64_t aOffset, bOffset;
  SmallVector<int64_t, 4> aStrides, bStrides;
  if (failed(getStridesAndOffset(aT, aStrides, aOffset)) ||
      failed(getStridesAndOffset(bT, bStrides, bOffset)) ||
      aStrides.size() != bStrides.size())
    return MemRefType();

  ArrayRef<int64_t> aShape = aT.getShape(), bShape = bT.getShape();
  int64_t resOffset;
  SmallVector<int64_t, 4> resShape(aT.getRank(), 0),
      resStrides(bT.getRank(), 0);
  for (int64_t idx = 0, e = aT.getRank(); idx < e; ++idx) {
    resShape[idx] =
        (aShape[idx] == bShape[idx]) ? aShape[idx] : MemRefType::kDynamicSize;
    resStrides[idx] = (aStrides[idx] == bStrides[idx])
                          ? aStrides[idx]
                          : MemRefType::kDynamicStrideOrOffset;
  }
  resOffset =
      (aOffset == bOffset) ? aOffset : MemRefType::kDynamicStrideOrOffset;
  return MemRefType::get(
      resShape, aT.getElementType(),
      makeStridedLinearLayoutMap(resStrides, resOffset, aT.getContext()));
}

/// Operates under a scoped context to build the intersection between the
/// view `xferOp.memref()` @ `xferOp.indices()` and the view `alloc`.
// TODO: view intersection/union/differences should be a proper std op.
static Value createScopedSubViewIntersection(VectorTransferOpInterface xferOp,
                                             Value alloc) {
  using namespace edsc::intrinsics;
  int64_t memrefRank = xferOp.getMemRefType().getRank();
  // TODO: relax this precondition, will require rank-reducing subviews.
  assert(memrefRank == alloc.getType().cast<MemRefType>().getRank() &&
         "Expected memref rank to match the alloc rank");
  Value one = std_constant_index(1);
  ValueRange leadingIndices =
      xferOp.indices().take_front(xferOp.getLeadingMemRefRank());
  SmallVector<Value, 4> sizes;
  sizes.append(leadingIndices.begin(), leadingIndices.end());
  xferOp.zipResultAndIndexing([&](int64_t resultIdx, int64_t indicesIdx) {
    using MapList = ArrayRef<ArrayRef<AffineExpr>>;
    Value dimMemRef = std_dim(xferOp.memref(), indicesIdx);
    Value dimAlloc = std_dim(alloc, resultIdx);
    Value index = xferOp.indices()[indicesIdx];
    AffineExpr i, j, k;
    bindDims(xferOp.getContext(), i, j, k);
    SmallVector<AffineMap, 4> maps =
        AffineMap::inferFromExprList(MapList{{i - j, k}});
    // affine_min(%dimMemRef - %index, %dimAlloc)
    Value affineMin = affine_min(index.getType(), maps[0],
                                 ValueRange{dimMemRef, index, dimAlloc});
    sizes.push_back(affineMin);
  });
  return std_sub_view(xferOp.memref(), xferOp.indices(), sizes,
                      SmallVector<Value, 4>(memrefRank, one));
}

/// Given an `xferOp` for which:
///   1. `inBoundsCond` and a `compatibleMemRefType` have been computed.
///   2. a memref of single vector `alloc` has been allocated.
/// Produce IR resembling:
/// ```
///    %1:3 = scf.if (%inBounds) {
///      memref_cast %A: memref<A...> to compatibleMemRefType
///      scf.yield %view, ... : compatibleMemRefType, index, index
///    } else {
///      %2 = linalg.fill(%alloc, %pad)
///      %3 = subview %view [...][...][...]
///      linalg.copy(%3, %alloc)
///      memref_cast %alloc: memref<B...> to compatibleMemRefType
///      scf.yield %4, ... : compatibleMemRefType, index, index
///   }
/// ```
/// Return the produced scf::IfOp.
static scf::IfOp createScopedFullPartialLinalgCopy(
    vector::TransferReadOp xferOp, TypeRange returnTypes, Value inBoundsCond,
    MemRefType compatibleMemRefType, Value alloc) {
  using namespace edsc;
  using namespace edsc::intrinsics;
  scf::IfOp fullPartialIfOp;
  Value zero = std_constant_index(0);
  Value memref = xferOp.memref();
  conditionBuilder(
      returnTypes, inBoundsCond,
      [&]() -> scf::ValueVector {
        Value res = memref;
        if (compatibleMemRefType != xferOp.getMemRefType())
          res = std_memref_cast(memref, compatibleMemRefType);
        scf::ValueVector viewAndIndices{res};
        viewAndIndices.insert(viewAndIndices.end(), xferOp.indices().begin(),
                              xferOp.indices().end());
        return viewAndIndices;
      },
      [&]() -> scf::ValueVector {
        linalg_fill(alloc, xferOp.padding());
        // Take partial subview of memref which guarantees no dimension
        // overflows.
        Value memRefSubView = createScopedSubViewIntersection(
            cast<VectorTransferOpInterface>(xferOp.getOperation()), alloc);
        linalg_copy(memRefSubView, alloc);
        Value casted = std_memref_cast(alloc, compatibleMemRefType);
        scf::ValueVector viewAndIndices{casted};
        viewAndIndices.insert(viewAndIndices.end(), xferOp.getTransferRank(),
                              zero);
        return viewAndIndices;
      },
      &fullPartialIfOp);
  return fullPartialIfOp;
}

/// Given an `xferOp` for which:
///   1. `inBoundsCond` and a `compatibleMemRefType` have been computed.
///   2. a memref of single vector `alloc` has been allocated.
/// Produce IR resembling:
/// ```
///    %1:3 = scf.if (%inBounds) {
///      memref_cast %A: memref<A...> to compatibleMemRefType
///      scf.yield %view, ... : compatibleMemRefType, index, index
///    } else {
///      %2 = vector.transfer_read %view[...], %pad : memref<A...>, vector<...>
///      %3 = vector.type_cast %extra_alloc :
///        memref<...> to memref<vector<...>>
///      store %2, %3[] : memref<vector<...>>
///      %4 = memref_cast %alloc: memref<B...> to compatibleMemRefType
///      scf.yield %4, ... : compatibleMemRefType, index, index
///   }
/// ```
/// Return the produced scf::IfOp.
static scf::IfOp createScopedFullPartialVectorTransferRead(
    vector::TransferReadOp xferOp, TypeRange returnTypes, Value inBoundsCond,
    MemRefType compatibleMemRefType, Value alloc) {
  using namespace edsc;
  using namespace edsc::intrinsics;
  scf::IfOp fullPartialIfOp;
  Value zero = std_constant_index(0);
  Value memref = xferOp.memref();
  conditionBuilder(
      returnTypes, inBoundsCond,
      [&]() -> scf::ValueVector {
        Value res = memref;
        if (compatibleMemRefType != xferOp.getMemRefType())
          res = std_memref_cast(memref, compatibleMemRefType);
        scf::ValueVector viewAndIndices{res};
        viewAndIndices.insert(viewAndIndices.end(), xferOp.indices().begin(),
                              xferOp.indices().end());
        return viewAndIndices;
      },
      [&]() -> scf::ValueVector {
        Operation *newXfer =
            ScopedContext::getBuilderRef().clone(*xferOp.getOperation());
        Value vector = cast<VectorTransferOpInterface>(newXfer).vector();
        std_store(vector, vector_type_cast(
                              MemRefType::get({}, vector.getType()), alloc));

        Value casted = std_memref_cast(alloc, compatibleMemRefType);
        scf::ValueVector viewAndIndices{casted};
        viewAndIndices.insert(viewAndIndices.end(), xferOp.getTransferRank(),
                              zero);

        return viewAndIndices;
      },
      &fullPartialIfOp);
  return fullPartialIfOp;
}

/// Split a vector.transfer operation into an unmasked fastpath and a slowpath.
/// If `ifOp` is not null and the result is `success, the `ifOp` points to the
/// newly created conditional upon function return.
/// To accomodate for the fact that the original vector.transfer indexing may be
/// arbitrary and the slow path indexes @[0...0] in the temporary buffer, the
/// scf.if op returns a view and values of type index.
/// At this time, only vector.transfer_read case is implemented.
///
/// Example (a 2-D vector.transfer_read):
/// ```
///    %1 = vector.transfer_read %0[...], %pad : memref<A...>, vector<...>
/// ```
/// is transformed into:
/// ```
///    %1:3 = scf.if (%inBounds) {
///      // fastpath, direct cast
///      memref_cast %A: memref<A...> to compatibleMemRefType
///      scf.yield %view : compatibleMemRefType, index, index
///    } else {
///      // slowpath, masked vector.transfer or linalg.copy.
///      memref_cast %alloc: memref<B...> to compatibleMemRefType
///      scf.yield %4 : compatibleMemRefType, index, index
//     }
///    %0 = vector.transfer_read %1#0[%1#1, %1#2] {masked = [false ... false]}
/// ```
/// where `alloc` is a top of the function alloca'ed buffer of one vector.
///
/// Preconditions:
///  1. `xferOp.permutation_map()` must be a minor identity map
///  2. the rank of the `xferOp.memref()` and the rank of the `xferOp.vector()`
///  must be equal. This will be relaxed in the future but requires
///  rank-reducing subviews.
LogicalResult mlir::vector::splitFullAndPartialTransfer(
    OpBuilder &b, VectorTransferOpInterface xferOp,
    VectorTransformsOptions options, scf::IfOp *ifOp) {
  using namespace edsc;
  using namespace edsc::intrinsics;

  if (options.vectorTransferSplit == VectorTransferSplit::None)
    return failure();

  SmallVector<bool, 4> bools(xferOp.getTransferRank(), false);
  auto unmaskedAttr = b.getBoolArrayAttr(bools);
  if (options.vectorTransferSplit == VectorTransferSplit::ForceUnmasked) {
    xferOp.setAttr(vector::TransferReadOp::getMaskedAttrName(), unmaskedAttr);
    return success();
  }

  assert(succeeded(splitFullAndPartialTransferPrecondition(xferOp)) &&
         "Expected splitFullAndPartialTransferPrecondition to hold");
  auto xferReadOp = dyn_cast<vector::TransferReadOp>(xferOp.getOperation());

  // TODO: add support for write case.
  if (!xferReadOp)
    return failure();

  OpBuilder::InsertionGuard guard(b);
  if (xferOp.memref().getDefiningOp())
    b.setInsertionPointAfter(xferOp.memref().getDefiningOp());
  else
    b.setInsertionPoint(xferOp);
  ScopedContext scope(b, xferOp.getLoc());
  Value inBoundsCond = createScopedInBoundsCond(
      cast<VectorTransferOpInterface>(xferOp.getOperation()));
  if (!inBoundsCond)
    return failure();

  // Top of the function `alloc` for transient storage.
  Value alloc;
  {
    FuncOp funcOp = xferOp.getParentOfType<FuncOp>();
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&funcOp.getRegion().front());
    auto shape = xferOp.getVectorType().getShape();
    Type elementType = xferOp.getVectorType().getElementType();
    alloc = std_alloca(MemRefType::get(shape, elementType), ValueRange{},
                       b.getI64IntegerAttr(32));
  }

  MemRefType compatibleMemRefType = getCastCompatibleMemRefType(
      xferOp.getMemRefType(), alloc.getType().cast<MemRefType>());

  // Read case: full fill + partial copy -> unmasked vector.xfer_read.
  SmallVector<Type, 4> returnTypes(1 + xferOp.getTransferRank(),
                                   b.getIndexType());
  returnTypes[0] = compatibleMemRefType;
  scf::IfOp fullPartialIfOp =
      options.vectorTransferSplit == VectorTransferSplit::VectorTransfer
          ? createScopedFullPartialVectorTransferRead(
                xferReadOp, returnTypes, inBoundsCond, compatibleMemRefType,
                alloc)
          : createScopedFullPartialLinalgCopy(xferReadOp, returnTypes,
                                              inBoundsCond,
                                              compatibleMemRefType, alloc);
  if (ifOp)
    *ifOp = fullPartialIfOp;

  // Unmask the existing read op, it always reads from a full buffer.
  for (unsigned i = 0, e = returnTypes.size(); i != e; ++i)
    xferReadOp.setOperand(i, fullPartialIfOp.getResult(i));
  xferOp.setAttr(vector::TransferReadOp::getMaskedAttrName(), unmaskedAttr);

  return success();
}

LogicalResult mlir::vector::VectorTransferFullPartialRewriter::matchAndRewrite(
    Operation *op, PatternRewriter &rewriter) const {
  auto xferOp = dyn_cast<VectorTransferOpInterface>(op);
  if (!xferOp || failed(splitFullAndPartialTransferPrecondition(xferOp)) ||
      failed(filter(xferOp)))
    return failure();
  rewriter.startRootUpdate(xferOp);
  if (succeeded(splitFullAndPartialTransfer(rewriter, xferOp, options))) {
    rewriter.finalizeRootUpdate(xferOp);
    return success();
  }
  rewriter.cancelRootUpdate(xferOp);
  return failure();
}

LogicalResult mlir::vector::PointwiseExtractPattern::matchAndRewrite(
    ExtractMapOp extract, PatternRewriter &rewriter) const {
  Operation *definedOp = extract.vector().getDefiningOp();
  if (!definedOp || definedOp->getNumResults() != 1)
    return failure();
  // TODO: Create an interfaceOp for elementwise operations.
  if (!isa<AddFOp>(definedOp))
    return failure();
  Location loc = extract.getLoc();
  SmallVector<Value, 4> extractOperands;
  for (OpOperand &operand : definedOp->getOpOperands())
    extractOperands.push_back(rewriter.create<vector::ExtractMapOp>(
        loc, extract.getResultType(), operand.get(), extract.ids()));
  Operation *newOp = cloneOpWithOperandsAndTypes(
      rewriter, loc, definedOp, extractOperands, extract.getResult().getType());
  rewriter.replaceOp(extract, newOp->getResult(0));
  return success();
}

Optional<mlir::vector::DistributeOps> mlir::vector::distributPointwiseVectorOp(
    OpBuilder &builder, Operation *op, ArrayRef<Value> ids,
    ArrayRef<int64_t> multiplicity, const AffineMap &map) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfter(op);
  Location loc = op->getLoc();
  if (op->getNumResults() != 1)
    return {};
  Value result = op->getResult(0);
  VectorType type = op->getResult(0).getType().dyn_cast<VectorType>();
  if (!type || map.getNumResults() != multiplicity.size())
    return {};
  // For each dimension being distributed check that the size is a multiple of
  // the multiplicity. To handle more sizes we would need to support masking.
  unsigned multiplictyCount = 0;
  for (auto exp : map.getResults()) {
    auto affinExp = exp.dyn_cast<AffineDimExpr>();
    if (!affinExp || affinExp.getPosition() >= type.getRank() ||
        type.getDimSize(affinExp.getPosition()) %
                multiplicity[multiplictyCount++] !=
            0)
      return {};
  }
  DistributeOps ops;
  ops.extract =
      builder.create<vector::ExtractMapOp>(loc, result, ids, multiplicity, map);
  ops.insert =
      builder.create<vector::InsertMapOp>(loc, ops.extract, result, ids);
  return ops;
}

struct TransferReadExtractPattern
    : public OpRewritePattern<vector::TransferReadOp> {
  TransferReadExtractPattern(MLIRContext *context)
      : OpRewritePattern<vector::TransferReadOp>(context) {}
  LogicalResult matchAndRewrite(vector::TransferReadOp read,
                                PatternRewriter &rewriter) const override {
    if (!read.getResult().hasOneUse())
      return failure();
    auto extract =
        dyn_cast<vector::ExtractMapOp>(*read.getResult().getUsers().begin());
    if (!extract)
      return failure();
    edsc::ScopedContext scope(rewriter, read.getLoc());
    using mlir::edsc::op::operator+;
    using mlir::edsc::op::operator*;
    using namespace mlir::edsc::intrinsics;
    SmallVector<Value, 4> indices(read.indices().begin(), read.indices().end());
    AffineMap map = extract.map();
    unsigned idCount = 0;
    for (auto expr : map.getResults()) {
      unsigned pos = expr.cast<AffineDimExpr>().getPosition();
      indices[pos] =
          indices[pos] +
          extract.ids()[idCount++] *
              std_constant_index(extract.getResultType().getDimSize(pos));
    }
    Value newRead = vector_transfer_read(extract.getType(), read.memref(),
                                         indices, read.permutation_map(),
                                         read.padding(), read.maskedAttr());
    Value dest = rewriter.create<ConstantOp>(
        read.getLoc(), read.getType(), rewriter.getZeroAttr(read.getType()));
    newRead = rewriter.create<vector::InsertMapOp>(read.getLoc(), newRead, dest,
                                                   extract.ids());
    rewriter.replaceOp(read, newRead);
    return success();
  }
};

struct TransferWriteInsertPattern
    : public OpRewritePattern<vector::TransferWriteOp> {
  TransferWriteInsertPattern(MLIRContext *context)
      : OpRewritePattern<vector::TransferWriteOp>(context) {}
  LogicalResult matchAndRewrite(vector::TransferWriteOp write,
                                PatternRewriter &rewriter) const override {
    auto insert = write.vector().getDefiningOp<vector::InsertMapOp>();
    if (!insert)
      return failure();
    edsc::ScopedContext scope(rewriter, write.getLoc());
    using mlir::edsc::op::operator+;
    using mlir::edsc::op::operator*;
    using namespace mlir::edsc::intrinsics;
    SmallVector<Value, 4> indices(write.indices().begin(),
                                  write.indices().end());
    AffineMap map = insert.map();
    unsigned idCount = 0;
    for (auto expr : map.getResults()) {
      unsigned pos = expr.cast<AffineDimExpr>().getPosition();
      indices[pos] =
          indices[pos] +
          insert.ids()[idCount++] *
              std_constant_index(insert.getSourceVectorType().getDimSize(pos));
    }
    vector_transfer_write(insert.vector(), write.memref(), indices,
                          write.permutation_map(), write.maskedAttr());
    rewriter.eraseOp(write);
    return success();
  }
};

// TODO: Add pattern to rewrite ExtractSlices(ConstantMaskOp).
// TODO: Add this as DRR pattern.
void mlir::vector::populateVectorToVectorTransformationPatterns(
    OwningRewritePatternList &patterns, MLIRContext *context) {
  // clang-format off
  patterns.insert<ShapeCastOpDecomposer,
                  ShapeCastOpFolder,
                  SplitTransferReadOp,
                  SplitTransferWriteOp,
                  TupleGetFolderOp,
                  TransferReadExtractPattern,
                  TransferWriteInsertPattern>(context);
  // clang-format on
}

void mlir::vector::populateVectorSlicesLoweringPatterns(
    OwningRewritePatternList &patterns, MLIRContext *context) {
  patterns.insert<ExtractSlicesOpLowering, InsertSlicesOpLowering>(context);
}

void mlir::vector::populateVectorContractLoweringPatterns(
    OwningRewritePatternList &patterns, MLIRContext *context,
    VectorTransformsOptions parameters) {
  // clang-format off
  patterns.insert<BroadcastOpLowering,
                  CreateMaskOpLowering,
                  ConstantMaskOpLowering,
                  OuterProductOpLowering,
                  ShapeCastOp2DDownCastRewritePattern,
                  ShapeCastOp2DUpCastRewritePattern,
                  ShapeCastOpRewritePattern>(context);
  patterns.insert<TransposeOpLowering,
                  ContractionOpLowering,
                  ContractionOpToMatmulOpLowering,
                  ContractionOpToOuterProductOpLowering>(parameters, context);
  // clang-format on
}
