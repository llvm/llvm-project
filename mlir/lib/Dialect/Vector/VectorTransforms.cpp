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
#include "mlir/Dialect/MemRef/EDSC/Intrinsics.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/VectorInterfaces.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "vector-to-vector"

using namespace mlir;

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
  return builder.getTupleType(vectorTypes);
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
  assert(llvm::all_of(op->getOperandTypes(), [=](Type type) {
    return type.cast<VectorType>().getShape() == resultShape;
  }));

  // Create trivial elementwise identity index map based on 'resultShape'.
  DenseMap<int64_t, int64_t> indexMap;
  indexMap.reserve(resultShape.size());
  for (unsigned i = 0; i < resultShape.size(); ++i)
    indexMap[i] = i;

  // Create VectorState each operand and single result.
  unsigned numVectors = op->getNumOperands() + op->getNumResults();
  vectors.resize(numVectors);
  for (auto it : llvm::enumerate(op->getOperandTypes()))
    vectors[it.index()] = {it.value().cast<VectorType>(), indexMap,
                           static_cast<int64_t>(it.index()), false};
  vectors[numVectors - 1] = {resultType, indexMap, -1, false};
  resultIndex = numVectors - 1;
}

/// Generates slices of 'vectorType' according to 'sizes' and 'strides, and
/// calls 'fn' with linear index and indices for each slice.
static void
generateTransferOpSlices(Type shapedElementType, VectorType vectorType,
                         TupleType tupleType, ArrayRef<int64_t> sizes,
                         ArrayRef<int64_t> strides, ArrayRef<Value> indices,
                         AffineMap permutationMap, OpBuilder &builder,
                         function_ref<void(unsigned, ArrayRef<Value>)> fn) {
  // Compute strides w.r.t. to slice counts in each dimension.
  auto maybeDimSliceCounts = shapeRatio(vectorType.getShape(), sizes);
  assert(maybeDimSliceCounts.hasValue());
  auto sliceDimCounts = *maybeDimSliceCounts;
  auto sliceStrides = computeStrides(sliceDimCounts);

  int64_t numSlices = tupleType.size();
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
  if (auto sourceVectorElementType = shapedElementType.dyn_cast<VectorType>()) {
    assert(vectorRank >= sourceVectorElementType.getRank());
    vectorRank -= sourceVectorElementType.getRank();
  }
  auto isBroadcast = [](AffineExpr expr) {
    if (auto constExpr = expr.dyn_cast<AffineConstantExpr>())
      return constExpr.getValue() == 0;
    return false;
  };
  auto *ctx = builder.getContext();
  for (unsigned i = 0; i < numSlices; ++i) {
    auto vectorOffsets = delinearize(sliceStrides, i);
    auto elementOffsets =
        computeElementOffsetsFromVectorSliceOffsets(sizes, vectorOffsets);
    // Compute 'sliceIndices' by adding 'sliceOffsets[i]' to 'indices[i]'.
    SmallVector<Value, 4> sliceIndices(indices.begin(), indices.end());
    for (auto dim : llvm::enumerate(permutationMap.getResults())) {
      if (isBroadcast(dim.value()))
        continue;
      unsigned pos = dim.value().cast<AffineDimExpr>().getPosition();
      auto expr = getAffineDimExpr(0, ctx) +
                  getAffineConstantExpr(elementOffsets[dim.index()], ctx);
      auto map = AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0, expr);
      sliceIndices[pos] = builder.create<AffineApplyOp>(
          indices[pos].getLoc(), map, ArrayRef<Value>(indices[pos]));
    }
    // Call 'fn' to generate slice 'i' at 'sliceIndices'.
    fn(i, sliceIndices);
  }
}

/// Unroll transfer_read ops to the given shape and create an aggregate with all
/// the chunks.
static Value unrollTransferReadOp(vector::TransferReadOp readOp,
                                  ArrayRef<int64_t> targetShape,
                                  OpBuilder &builder) {
  if (readOp.mask())
    return nullptr;
  auto sourceVectorType = readOp.getVectorType();
  SmallVector<int64_t, 4> strides(targetShape.size(), 1);

  Location loc = readOp.getLoc();
  auto shapedElementType =
      readOp.source().getType().cast<ShapedType>().getElementType();
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
    // `in_bounds` attribute propagates conservatively: if the coarse op didn't
    // need out-of-bounds masking, the fine op doesn't either.
    vectorTupleValues[index] = builder.create<vector::TransferReadOp>(
        loc, sliceVectorType, readOp.source(), sliceIndices,
        readOp.permutation_map(), readOp.padding(),
        readOp.in_bounds() ? *readOp.in_bounds() : ArrayAttr());
  };
  generateTransferOpSlices(shapedElementType, sourceVectorType, tupleType,
                           targetShape, strides, indices,
                           readOp.permutation_map(), builder, createSlice);

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
                                    ArrayRef<int64_t> targetShape,
                                    SmallVector<Value, 1> &result) {
  auto writeOp = cast<vector::TransferWriteOp>(op);
  if (writeOp.mask())
    return failure();
  VectorType sourceVectorType = writeOp.getVectorType();
  SmallVector<int64_t, 4> strides(targetShape.size(), 1);
  TupleType tupleType = generateExtractSlicesOpResultType(
      sourceVectorType, targetShape, strides, builder);
  Location loc = writeOp.getLoc();
  Value tuple = builder.create<vector::ExtractSlicesOp>(
      loc, tupleType, writeOp.vector(), targetShape, strides);
  auto shapedElementType =
      writeOp.source().getType().cast<ShapedType>().getElementType();
  SmallVector<Value, 4> indices(writeOp.indices().begin(),
                                writeOp.indices().end());
  // If the TransferWrite returns a tensor, keep track of the last tensor
  // created.
  Value resultTensor;
  auto createSlice = [&](unsigned index, ArrayRef<Value> sliceIndices) {
    auto element = builder.create<vector::TupleGetOp>(
        loc, tupleType.getType(index), tuple, builder.getI64IntegerAttr(index));
    Operation *write = builder.create<vector::TransferWriteOp>(
        loc, element.getResult(),
        resultTensor ? resultTensor : writeOp.source(), sliceIndices,
        writeOp.permutation_map(),
        writeOp.in_bounds() ? *writeOp.in_bounds() : ArrayAttr());
    if (!write->getResults().empty())
      resultTensor = write->getResult(0);
  };
  generateTransferOpSlices(shapedElementType, sourceVectorType, tupleType,
                           targetShape, strides, indices,
                           writeOp.permutation_map(), builder, createSlice);
  if (resultTensor)
    result.push_back(resultTensor);
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

// Splits a TransferReadOp into smaller TransferReadOps based on slicing
// scheme of its unique ExtractSlicesOp users.
class SplitTransferReadOp : public OpRewritePattern<vector::TransferReadOp> {
public:
  SplitTransferReadOp(MLIRContext *context,
                      std::function<bool(Operation *)> ignoreFilter = nullptr,
                      PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), ignoreFilter(ignoreFilter) {}

  LogicalResult matchAndRewrite(vector::TransferReadOp readOp,
                                PatternRewriter &rewriter) const override {
    if (ignoreFilter && ignoreFilter(readOp))
      return failure();

    if (readOp.mask())
      return failure();

    // Return unless there is only one user, and it is an ExtractSlicesOp.
    Value readResult = readOp.getResult();
    if (!readResult.hasOneUse())
      return failure();

    auto extractSlicesOp =
        dyn_cast<vector::ExtractSlicesOp>(readResult.use_begin()->getOwner());
    if (!extractSlicesOp)
      return failure();

    // Get 'sizes' and 'strides' parameters from ExtractSlicesOp user.
    SmallVector<int64_t, 4> sizes;
    extractSlicesOp.getSizes(sizes);
    SmallVector<int64_t, 4> strides;
    extractSlicesOp.getStrides(strides);
    assert(llvm::all_of(strides, [](int64_t s) { return s == 1; }));

    Value newVec = unrollTransferReadOp(readOp, sizes, rewriter);
    if (!newVec)
      return failure();
    rewriter.replaceOp(readOp, newVec);
    return success();
  }

private:
  std::function<bool(Operation *)> ignoreFilter;
};

// Splits a TransferWriteOp into smaller TransferWriteOps for each source.
class SplitTransferWriteOp : public OpRewritePattern<vector::TransferWriteOp> {
public:
  SplitTransferWriteOp(MLIRContext *context,
                       std::function<bool(Operation *)> ignoreFilter = nullptr,
                       PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), ignoreFilter(ignoreFilter) {}

  LogicalResult matchAndRewrite(vector::TransferWriteOp writeOp,
                                PatternRewriter &rewriter) const override {
    if (ignoreFilter && ignoreFilter(writeOp))
      return failure();

    if (writeOp.mask())
      return failure();

    // Fail to match unless this is writing a vector resulting from an
    // InsertSlicesOp.
    auto insertSlicesOp =
        writeOp.vector().getDefiningOp<vector::InsertSlicesOp>();
    if (!insertSlicesOp)
      return failure();

    // Get the TupleOp operand of the InsertSlicesOp.
    auto tupleOp = insertSlicesOp.vectors().getDefiningOp<vector::TupleOp>();
    if (!tupleOp)
      return failure();

    // Get 'sizes' and 'strides' parameters from the InsertSlicesOp user.
    auto sourceTupleType = insertSlicesOp.getSourceTupleType();
    auto resultVectorType = insertSlicesOp.getResultVectorType();
    SmallVector<int64_t, 4> sizes;
    insertSlicesOp.getSizes(sizes);
    SmallVector<int64_t, 4> strides;
    insertSlicesOp.getStrides(strides);

    Location loc = writeOp.getLoc();
    auto shapedElementType =
        writeOp.source().getType().cast<ShapedType>().getElementType();
    auto indices = llvm::to_vector<4>(writeOp.indices());
    Value resultTensor;
    auto createSlice = [&](unsigned index, ArrayRef<Value> sliceIndices) {
      // Create split TransferWriteOp for source vector 'tupleOp.operand[i]'.
      // 'in_bounds' attribute propagates conservatively: if the coarse op
      // didn't need out-of-bounds masking, the fine op doesn't either.
      Operation *write = rewriter.create<vector::TransferWriteOp>(
          loc, tupleOp.getOperand(index),
          resultTensor ? resultTensor : writeOp.source(), sliceIndices,
          writeOp.permutation_map(),
          writeOp.in_bounds() ? *writeOp.in_bounds() : ArrayAttr());
      if (!write->getResults().empty())
        resultTensor = write->getResult(0);
    };
    generateTransferOpSlices(shapedElementType, resultVectorType,
                             sourceTupleType, sizes, strides, indices,
                             writeOp.permutation_map(), rewriter, createSlice);

    if (resultTensor)
      rewriter.replaceOp(writeOp, ArrayRef<Value>(resultTensor));
    else
      rewriter.eraseOp(writeOp);
    return success();
  }

private:
  std::function<bool(Operation *)> ignoreFilter;
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
    auto operandResultVectorType = sourceShapeCastOp.getType();

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
    bool isInt = eltType.isa<IntegerType, IndexType>();
    Value acc = (op.acc().empty()) ? nullptr : op.acc()[0];
    vector::CombiningKind kind = op.kind();

    if (!rhsType) {
      // Special case: AXPY operation.
      Value b = rewriter.create<vector::BroadcastOp>(loc, lhsType, op.rhs());
      Optional<Value> mult =
          isInt ? genMultI(loc, op.lhs(), b, acc, kind, rewriter)
                : genMultF(loc, op.lhs(), b, acc, kind, rewriter);
      if (!mult.hasValue())
        return failure();
      rewriter.replaceOp(op, mult.getValue());
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
      Optional<Value> m = isInt ? genMultI(loc, a, op.rhs(), r, kind, rewriter)
                                : genMultF(loc, a, op.rhs(), r, kind, rewriter);
      if (!m.hasValue())
        return failure();
      result = rewriter.create<vector::InsertOp>(loc, resType, m.getValue(),
                                                 result, pos);
    }
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  static Optional<Value> genMultI(Location loc, Value x, Value y, Value acc,
                                  vector::CombiningKind kind,
                                  PatternRewriter &rewriter) {
    using vector::CombiningKind;

    MulIOp mul = rewriter.create<MulIOp>(loc, x, y);
    if (!acc)
      return Optional<Value>(mul);

    Value combinedResult;
    switch (kind) {
    case CombiningKind::ADD:
      combinedResult = rewriter.create<AddIOp>(loc, mul, acc);
      break;
    case CombiningKind::MUL:
      combinedResult = rewriter.create<MulIOp>(loc, mul, acc);
      break;
    case CombiningKind::MIN:
      combinedResult = rewriter.create<SelectOp>(
          loc, rewriter.create<CmpIOp>(loc, CmpIPredicate::slt, mul, acc), mul,
          acc);
      break;
    case CombiningKind::MAX:
      combinedResult = rewriter.create<SelectOp>(
          loc, rewriter.create<CmpIOp>(loc, CmpIPredicate::sge, mul, acc), mul,
          acc);
      break;
    case CombiningKind::AND:
      combinedResult = rewriter.create<AndOp>(loc, mul, acc);
      break;
    case CombiningKind::OR:
      combinedResult = rewriter.create<OrOp>(loc, mul, acc);
      break;
    case CombiningKind::XOR:
      combinedResult = rewriter.create<XOrOp>(loc, mul, acc);
      break;
    }
    return Optional<Value>(combinedResult);
  }

  static Optional<Value> genMultF(Location loc, Value x, Value y, Value acc,
                                  vector::CombiningKind kind,
                                  PatternRewriter &rewriter) {
    using vector::CombiningKind;

    // Special case for fused multiply-add.
    if (acc && kind == CombiningKind::ADD) {
      return Optional<Value>(rewriter.create<vector::FMAOp>(loc, x, y, acc));
    }

    MulFOp mul = rewriter.create<MulFOp>(loc, x, y);

    if (!acc)
      return Optional<Value>(mul);

    Value combinedResult;
    switch (kind) {
    case CombiningKind::MUL:
      combinedResult = rewriter.create<MulFOp>(loc, mul, acc);
      break;
    case CombiningKind::MIN:
      combinedResult = rewriter.create<SelectOp>(
          loc, rewriter.create<CmpFOp>(loc, CmpFPredicate::OLE, mul, acc), mul,
          acc);
      break;
    case CombiningKind::MAX:
      combinedResult = rewriter.create<SelectOp>(
          loc, rewriter.create<CmpFOp>(loc, CmpFPredicate::OGT, mul, acc), mul,
          acc);
      break;
    case CombiningKind::ADD: // Already handled this special case above.
    case CombiningKind::AND: // Only valid for integer types.
    case CombiningKind::OR:  // Only valid for integer types.
    case CombiningKind::XOR: // Only valid for integer types.
      return Optional<Value>();
    }
    return Optional<Value>(combinedResult);
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
    auto dstType = op.getType();
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

/// Creates an AddIOp if `isInt` is true otherwise create an AddFOp using
/// operands `x` and `y`.
static Value createAdd(Location loc, Value x, Value y, bool isInt,
                       PatternRewriter &rewriter) {
  if (isInt)
    return rewriter.create<AddIOp>(loc, x, y);
  return rewriter.create<AddFOp>(loc, x, y);
}

/// Creates a MulIOp if `isInt` is true otherwise create an MulFOp using
/// operands `x and `y`.
static Value createMul(Location loc, Value x, Value y, bool isInt,
                       PatternRewriter &rewriter) {
  if (isInt)
    return rewriter.create<MulIOp>(loc, x, y);
  return rewriter.create<MulFOp>(loc, x, y);
}

namespace mlir {

/// Progressively lower a `vector.contract %a, %b, %c` with row-major matmul
/// semantics to:
/// ```
///    %mta = maybe_transpose
///    %mtb = maybe_transpose
///    %flattened_a = vector.shape_cast %mta
///    %flattened_b = vector.shape_cast %mtb
///    %flattened_d = vector.matmul %flattened_a, %flattened_b
///    %mtd = vector.shape_cast %flattened_d
///    %d = maybe_untranspose %mtd
///    %e = add %c, %d
/// ```
/// `vector.matmul` later lowers to `llvm.matrix.multiply`.
//
/// This only kicks in when VectorTransformsOptions is set to `Matmul`.
/// vector.transpose operations are inserted if the vector.contract op is not a
/// row-major matrix multiply.
LogicalResult
ContractionOpToMatmulOpLowering::matchAndRewrite(vector::ContractionOp op,
                                                 PatternRewriter &rew) const {
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

  Type elementType = op.getLhsType().getElementType();
  if (!elementType.isIntOrFloat())
    return failure();

  // Perform lhs + rhs transpositions to conform to matmul row-major semantics.
  // Bail out if the contraction cannot be put in this form.
  MLIRContext *ctx = op.getContext();
  Location loc = op.getLoc();
  AffineExpr m, n, k;
  bindDims(rew.getContext(), m, n, k);
  // LHS must be A(m, k) or A(k, m).
  Value lhs = op.lhs();
  auto lhsMap = op.indexing_maps()[0].cast<AffineMapAttr>().getValue();
  if (lhsMap == AffineMap::get(3, 0, {k, m}, ctx))
    lhs = rew.create<vector::TransposeOp>(loc, lhs, ArrayRef<int64_t>{1, 0});
  else if (lhsMap != AffineMap::get(3, 0, {m, k}, ctx))
    return failure();

  // RHS must be B(k, n) or B(n, k).
  Value rhs = op.rhs();
  auto rhsMap = op.indexing_maps()[1].cast<AffineMapAttr>().getValue();
  if (rhsMap == AffineMap::get(3, 0, {n, k}, ctx))
    rhs = rew.create<vector::TransposeOp>(loc, rhs, ArrayRef<int64_t>{1, 0});
  else if (rhsMap != AffineMap::get(3, 0, {k, n}, ctx))
    return failure();

  // At this point lhs and rhs are in row-major.
  VectorType lhsType = lhs.getType().cast<VectorType>();
  VectorType rhsType = rhs.getType().cast<VectorType>();
  int64_t lhsRows = lhsType.getDimSize(0);
  int64_t lhsColumns = lhsType.getDimSize(1);
  int64_t rhsColumns = rhsType.getDimSize(1);

  Type flattenedLHSType =
      VectorType::get(lhsType.getNumElements(), lhsType.getElementType());
  lhs = rew.create<vector::ShapeCastOp>(loc, flattenedLHSType, lhs);

  Type flattenedRHSType =
      VectorType::get(rhsType.getNumElements(), rhsType.getElementType());
  rhs = rew.create<vector::ShapeCastOp>(loc, flattenedRHSType, rhs);

  Value mul = rew.create<vector::MatmulOp>(loc, lhs, rhs, lhsRows, lhsColumns,
                                           rhsColumns);
  mul = rew.create<vector::ShapeCastOp>(
      loc,
      VectorType::get({lhsRows, rhsColumns},
                      getElementTypeOrSelf(op.acc().getType())),
      mul);

  // ACC must be C(m, n) or C(n, m).
  auto accMap = op.indexing_maps()[2].cast<AffineMapAttr>().getValue();
  if (accMap == AffineMap::get(3, 0, {n, m}, ctx))
    mul = rew.create<vector::TransposeOp>(loc, mul, ArrayRef<int64_t>{1, 0});
  else if (accMap != AffineMap::get(3, 0, {m, n}, ctx))
    llvm_unreachable("invalid contraction semantics");

  Value res = elementType.isa<IntegerType>()
                  ? static_cast<Value>(rew.create<AddIOp>(loc, op.acc(), mul))
                  : static_cast<Value>(rew.create<AddFOp>(loc, op.acc(), mul));

  rew.replaceOp(op, res);
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
    res = rewriter.create<vector::OuterProductOp>(op.getLoc(), res.getType(), a,
                                                  b, res, op.kind());
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
  bool isInt = dstType.getElementType().isa<IntegerType>();
  for (unsigned r = 0; r < dstRows; ++r) {
    Value a = rewriter.create<vector::ExtractOp>(op.getLoc(), lhs, r);
    for (unsigned c = 0; c < dstColumns; ++c) {
      Value b = rank == 1
                    ? rhs
                    : rewriter.create<vector::ExtractOp>(op.getLoc(), rhs, c);
      Value m = createMul(op.getLoc(), a, b, isInt, rewriter);
      Value reduced = rewriter.create<vector::ReductionOp>(
          op.getLoc(), dstType.getElementType(), rewriter.getStringAttr("add"),
          m, ValueRange{});

      SmallVector<int64_t, 2> pos = rank == 1 ? SmallVector<int64_t, 2>{r}
                                              : SmallVector<int64_t, 2>{r, c};
      res = rewriter.create<vector::InsertOp>(op.getLoc(), reduced, res, pos);
    }
  }
  if (auto acc = op.acc())
    res = createAdd(op.getLoc(), res, acc, isInt, rewriter);
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
  bool isInt = resType.isa<IntegerType>();
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
    Value m = createMul(loc, op.lhs(), op.rhs(), isInt, rewriter);
    StringAttr kind = rewriter.getStringAttr("add");
    Value res = rewriter.create<vector::ReductionOp>(loc, resType, kind, m,
                                                     ValueRange{});
    if (auto acc = op.acc())
      res = createAdd(op.getLoc(), res, acc, isInt, rewriter);
    return res;
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
// particular VectorTransferOpInterface is in-bounds.
static Value createScopedInBoundsCond(VectorTransferOpInterface xferOp) {
  assert(xferOp.permutation_map().isMinorIdentity() &&
         "Expected minor identity map");
  Value inBoundsCond;
  xferOp.zipResultAndIndexing([&](int64_t resultIdx, int64_t indicesIdx) {
    // Zip over the resulting vector shape and memref indices.
    // If the dimension is known to be in-bounds, it does not participate in
    // the construction of `inBoundsCond`.
    if (xferOp.isDimInBounds(resultIdx))
      return;
    int64_t vectorSize = xferOp.getVectorType().getDimSize(resultIdx);
    using namespace edsc::op;
    using namespace edsc::intrinsics;
    // Fold or create the check that `index + vector_size` <= `memref_size`.
    Value sum = xferOp.indices()[indicesIdx] + std_constant_index(vectorSize);
    Value cond =
        createScopedFoldedSLE(sum, memref_dim(xferOp.source(), indicesIdx));
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
  // Must have some out-of-bounds dimension to be a candidate for splitting.
  if (!xferOp.hasOutOfBoundsDim())
    return failure();
  // Don't split transfer operations directly under IfOp, this avoids applying
  // the pattern recursively.
  // TODO: improve the filtering condition to make it more applicable.
  if (isa<scf::IfOp>(xferOp->getParentOp()))
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
  if (memref::CastOp::areCastCompatible(aT, bT))
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
/// view `xferOp.source()` @ `xferOp.indices()` and the view `alloc`.
// TODO: view intersection/union/differences should be a proper std op.
static Value createScopedSubViewIntersection(VectorTransferOpInterface xferOp,
                                             Value alloc) {
  using namespace edsc::intrinsics;
  int64_t memrefRank = xferOp.getShapedType().getRank();
  // TODO: relax this precondition, will require rank-reducing subviews.
  assert(memrefRank == alloc.getType().cast<MemRefType>().getRank() &&
         "Expected memref rank to match the alloc rank");
  ValueRange leadingIndices =
      xferOp.indices().take_front(xferOp.getLeadingShapedRank());
  SmallVector<OpFoldResult, 4> sizes;
  sizes.append(leadingIndices.begin(), leadingIndices.end());
  auto isaWrite = isa<vector::TransferWriteOp>(xferOp);
  xferOp.zipResultAndIndexing([&](int64_t resultIdx, int64_t indicesIdx) {
    using MapList = ArrayRef<ArrayRef<AffineExpr>>;
    Value dimMemRef = memref_dim(xferOp.source(), indicesIdx);
    Value dimAlloc = memref_dim(alloc, resultIdx);
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

  SmallVector<OpFoldResult, 4> indices = llvm::to_vector<4>(llvm::map_range(
      xferOp.indices(), [](Value idx) -> OpFoldResult { return idx; }));
  return memref_sub_view(
      isaWrite ? alloc : xferOp.source(), indices, sizes,
      SmallVector<OpFoldResult>(memrefRank, OpBuilder(xferOp).getIndexAttr(1)));
}

/// Given an `xferOp` for which:
///   1. `inBoundsCond` and a `compatibleMemRefType` have been computed.
///   2. a memref of single vector `alloc` has been allocated.
/// Produce IR resembling:
/// ```
///    %1:3 = scf.if (%inBounds) {
///      memref.cast %A: memref<A...> to compatibleMemRefType
///      scf.yield %view, ... : compatibleMemRefType, index, index
///    } else {
///      %2 = linalg.fill(%alloc, %pad)
///      %3 = subview %view [...][...][...]
///      linalg.copy(%3, %alloc)
///      memref.cast %alloc: memref<B...> to compatibleMemRefType
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
  Value memref = xferOp.source();
  conditionBuilder(
      returnTypes, inBoundsCond,
      [&]() -> scf::ValueVector {
        Value res = memref;
        if (compatibleMemRefType != xferOp.getShapedType())
          res = memref_cast(memref, compatibleMemRefType);
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
        Value casted = memref_cast(alloc, compatibleMemRefType);
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
///      memref.cast %A: memref<A...> to compatibleMemRefType
///      scf.yield %view, ... : compatibleMemRefType, index, index
///    } else {
///      %2 = vector.transfer_read %view[...], %pad : memref<A...>, vector<...>
///      %3 = vector.type_cast %extra_alloc :
///        memref<...> to memref<vector<...>>
///      store %2, %3[] : memref<vector<...>>
///      %4 = memref.cast %alloc: memref<B...> to compatibleMemRefType
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
  Value memref = xferOp.source();
  conditionBuilder(
      returnTypes, inBoundsCond,
      [&]() -> scf::ValueVector {
        Value res = memref;
        if (compatibleMemRefType != xferOp.getShapedType())
          res = memref_cast(memref, compatibleMemRefType);
        scf::ValueVector viewAndIndices{res};
        viewAndIndices.insert(viewAndIndices.end(), xferOp.indices().begin(),
                              xferOp.indices().end());
        return viewAndIndices;
      },
      [&]() -> scf::ValueVector {
        Operation *newXfer =
            ScopedContext::getBuilderRef().clone(*xferOp.getOperation());
        Value vector = cast<VectorTransferOpInterface>(newXfer).vector();
        memref_store(vector, vector_type_cast(
                                 MemRefType::get({}, vector.getType()), alloc));

        Value casted = memref_cast(alloc, compatibleMemRefType);
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
///      memref.cast %A: memref<A...> to compatibleMemRefType
///      scf.yield %view, ... : compatibleMemRefType, index, index
///    } else {
///      %3 = vector.type_cast %extra_alloc :
///        memref<...> to memref<vector<...>>
///      %4 = memref.cast %alloc: memref<B...> to compatibleMemRefType
///      scf.yield %4, ... : compatibleMemRefType, index, index
///   }
/// ```
static ValueRange getLocationToWriteFullVec(vector::TransferWriteOp xferOp,
                                            TypeRange returnTypes,
                                            Value inBoundsCond,
                                            MemRefType compatibleMemRefType,
                                            Value alloc) {
  using namespace edsc;
  using namespace edsc::intrinsics;
  Value zero = std_constant_index(0);
  Value memref = xferOp.source();
  return conditionBuilder(
      returnTypes, inBoundsCond,
      [&]() -> scf::ValueVector {
        Value res = memref;
        if (compatibleMemRefType != xferOp.getShapedType())
          res = memref_cast(memref, compatibleMemRefType);
        scf::ValueVector viewAndIndices{res};
        viewAndIndices.insert(viewAndIndices.end(), xferOp.indices().begin(),
                              xferOp.indices().end());
        return viewAndIndices;
      },
      [&]() -> scf::ValueVector {
        Value casted = memref_cast(alloc, compatibleMemRefType);
        scf::ValueVector viewAndIndices{casted};
        viewAndIndices.insert(viewAndIndices.end(), xferOp.getTransferRank(),
                              zero);
        return viewAndIndices;
      });
}

/// Given an `xferOp` for which:
///   1. `inBoundsCond` has been computed.
///   2. a memref of single vector `alloc` has been allocated.
///   3. it originally wrote to %view
/// Produce IR resembling:
/// ```
///    %notInBounds = xor %inBounds, %true
///    scf.if (%notInBounds) {
///      %3 = subview %alloc [...][...][...]
///      linalg.copy(%3, %view)
///   }
/// ```
static void createScopedFullPartialLinalgCopy(vector::TransferWriteOp xferOp,
                                              Value inBoundsCond, Value alloc) {
  using namespace edsc;
  using namespace edsc::intrinsics;
  auto &b = ScopedContext::getBuilderRef();
  auto notInBounds = b.create<XOrOp>(
      xferOp->getLoc(), inBoundsCond,
      b.create<::mlir::ConstantIntOp>(xferOp.getLoc(), true, 1));

  conditionBuilder(notInBounds, [&]() {
    Value memRefSubView = createScopedSubViewIntersection(
        cast<VectorTransferOpInterface>(xferOp.getOperation()), alloc);
    linalg_copy(memRefSubView, xferOp.source());
  });
}

/// Given an `xferOp` for which:
///   1. `inBoundsCond` has been computed.
///   2. a memref of single vector `alloc` has been allocated.
///   3. it originally wrote to %view
/// Produce IR resembling:
/// ```
///    %notInBounds = xor %inBounds, %true
///    scf.if (%notInBounds) {
///      %2 = load %alloc : memref<vector<...>>
///      vector.transfer_write %2, %view[...] : memref<A...>, vector<...>
///   }
/// ```
static void
createScopedFullPartialVectorTransferWrite(vector::TransferWriteOp xferOp,
                                           Value inBoundsCond, Value alloc) {
  using namespace edsc;
  using namespace edsc::intrinsics;
  auto &b = ScopedContext::getBuilderRef();
  auto notInBounds = b.create<XOrOp>(
      xferOp->getLoc(), inBoundsCond,
      b.create<::mlir::ConstantIntOp>(xferOp.getLoc(), true, 1));
  conditionBuilder(notInBounds, [&]() {
    BlockAndValueMapping mapping;

    Value load = memref_load(vector_type_cast(
        MemRefType::get({}, xferOp.vector().getType()), alloc));

    mapping.map(xferOp.vector(), load);
    b.clone(*xferOp.getOperation(), mapping);
  });
}

/// Split a vector.transfer operation into an in-bounds (i.e., no out-of-bounds
/// masking) fastpath and a slowpath.
///
/// For vector.transfer_read:
/// If `ifOp` is not null and the result is `success, the `ifOp` points to the
/// newly created conditional upon function return.
/// To accomodate for the fact that the original vector.transfer indexing may be
/// arbitrary and the slow path indexes @[0...0] in the temporary buffer, the
/// scf.if op returns a view and values of type index.
///
/// Example (a 2-D vector.transfer_read):
/// ```
///    %1 = vector.transfer_read %0[...], %pad : memref<A...>, vector<...>
/// ```
/// is transformed into:
/// ```
///    %1:3 = scf.if (%inBounds) {
///      // fastpath, direct cast
///      memref.cast %A: memref<A...> to compatibleMemRefType
///      scf.yield %view : compatibleMemRefType, index, index
///    } else {
///      // slowpath, not in-bounds vector.transfer or linalg.copy.
///      memref.cast %alloc: memref<B...> to compatibleMemRefType
///      scf.yield %4 : compatibleMemRefType, index, index
//     }
///    %0 = vector.transfer_read %1#0[%1#1, %1#2] {in_bounds = [true ... true]}
/// ```
/// where `alloc` is a top of the function alloca'ed buffer of one vector.
///
/// For vector.transfer_write:
/// There are 2 conditional blocks. First a block to decide which memref and
/// indices to use for an unmasked, inbounds write. Then a conditional block to
/// further copy a partial buffer into the final result in the slow path case.
///
/// Example (a 2-D vector.transfer_write):
/// ```
///    vector.transfer_write %arg, %0[...], %pad : memref<A...>, vector<...>
/// ```
/// is transformed into:
/// ```
///    %1:3 = scf.if (%inBounds) {
///      memref.cast %A: memref<A...> to compatibleMemRefType
///      scf.yield %view : compatibleMemRefType, index, index
///    } else {
///      memref.cast %alloc: memref<B...> to compatibleMemRefType
///      scf.yield %4 : compatibleMemRefType, index, index
///     }
///    %0 = vector.transfer_write %arg, %1#0[%1#1, %1#2] {in_bounds = [true ...
///                                                                    true]}
///    scf.if (%notInBounds) {
///      // slowpath: not in-bounds vector.transfer or linalg.copy.
///    }
/// ```
/// where `alloc` is a top of the function alloca'ed buffer of one vector.
///
/// Preconditions:
///  1. `xferOp.permutation_map()` must be a minor identity map
///  2. the rank of the `xferOp.source()` and the rank of the `xferOp.vector()`
///  must be equal. This will be relaxed in the future but requires
///  rank-reducing subviews.
LogicalResult mlir::vector::splitFullAndPartialTransfer(
    OpBuilder &b, VectorTransferOpInterface xferOp,
    VectorTransformsOptions options, scf::IfOp *ifOp) {
  using namespace edsc;
  using namespace edsc::intrinsics;

  if (options.vectorTransferSplit == VectorTransferSplit::None)
    return failure();

  SmallVector<bool, 4> bools(xferOp.getTransferRank(), true);
  auto inBoundsAttr = b.getBoolArrayAttr(bools);
  if (options.vectorTransferSplit == VectorTransferSplit::ForceInBounds) {
    xferOp->setAttr(xferOp.getInBoundsAttrName(), inBoundsAttr);
    return success();
  }

  // Assert preconditions. Additionally, keep the variables in an inner scope to
  // ensure they aren't used in the wrong scopes further down.
  {
    assert(succeeded(splitFullAndPartialTransferPrecondition(xferOp)) &&
           "Expected splitFullAndPartialTransferPrecondition to hold");

    auto xferReadOp = dyn_cast<vector::TransferReadOp>(xferOp.getOperation());
    auto xferWriteOp = dyn_cast<vector::TransferWriteOp>(xferOp.getOperation());

    if (!(xferReadOp || xferWriteOp))
      return failure();
    if (xferWriteOp && xferWriteOp.mask())
      return failure();
    if (xferReadOp && xferReadOp.mask())
      return failure();
  }

  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(xferOp);
  ScopedContext scope(b, xferOp.getLoc());
  Value inBoundsCond = createScopedInBoundsCond(
      cast<VectorTransferOpInterface>(xferOp.getOperation()));
  if (!inBoundsCond)
    return failure();

  // Top of the function `alloc` for transient storage.
  Value alloc;
  {
    FuncOp funcOp = xferOp->getParentOfType<FuncOp>();
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&funcOp.getRegion().front());
    auto shape = xferOp.getVectorType().getShape();
    Type elementType = xferOp.getVectorType().getElementType();
    alloc = memref_alloca(MemRefType::get(shape, elementType), ValueRange{},
                          b.getI64IntegerAttr(32));
  }

  MemRefType compatibleMemRefType =
      getCastCompatibleMemRefType(xferOp.getShapedType().cast<MemRefType>(),
                                  alloc.getType().cast<MemRefType>());
  SmallVector<Type, 4> returnTypes(1 + xferOp.getTransferRank(),
                                   b.getIndexType());
  returnTypes[0] = compatibleMemRefType;

  if (auto xferReadOp =
          dyn_cast<vector::TransferReadOp>(xferOp.getOperation())) {
    // Read case: full fill + partial copy -> in-bounds vector.xfer_read.
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

    // Set existing read op to in-bounds, it always reads from a full buffer.
    for (unsigned i = 0, e = returnTypes.size(); i != e; ++i)
      xferReadOp.setOperand(i, fullPartialIfOp.getResult(i));

    xferOp->setAttr(xferOp.getInBoundsAttrName(), inBoundsAttr);

    return success();
  }

  auto xferWriteOp = cast<vector::TransferWriteOp>(xferOp.getOperation());

  // Decide which location to write the entire vector to.
  auto memrefAndIndices = getLocationToWriteFullVec(
      xferWriteOp, returnTypes, inBoundsCond, compatibleMemRefType, alloc);

  // Do an in bounds write to either the output or the extra allocated buffer.
  // The operation is cloned to prevent deleting information needed for the
  // later IR creation.
  BlockAndValueMapping mapping;
  mapping.map(xferWriteOp.source(), memrefAndIndices.front());
  mapping.map(xferWriteOp.indices(), memrefAndIndices.drop_front());
  auto *clone = b.clone(*xferWriteOp, mapping);
  clone->setAttr(xferWriteOp.getInBoundsAttrName(), inBoundsAttr);

  // Create a potential copy from the allocated buffer to the final output in
  // the slow path case.
  if (options.vectorTransferSplit == VectorTransferSplit::VectorTransfer)
    createScopedFullPartialVectorTransferWrite(xferWriteOp, inBoundsCond,
                                               alloc);
  else
    createScopedFullPartialLinalgCopy(xferWriteOp, inBoundsCond, alloc);

  xferOp->erase();

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
    if (read.mask())
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
    Value newRead = vector_transfer_read(extract.getType(), read.source(),
                                         indices, read.permutation_map(),
                                         read.padding(), read.in_boundsAttr());
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
    if (write.mask())
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
    vector_transfer_write(insert.vector(), write.source(), indices,
                          write.permutation_map(), write.in_boundsAttr());
    rewriter.eraseOp(write);
    return success();
  }
};

/// Progressive lowering of transfer_read. This pattern supports lowering of
/// `vector.transfer_read` to a combination of `vector.load` and
/// `vector.broadcast` if all of the following hold:
/// - The op reads from a memref with the default layout.
/// - Out-of-bounds masking is not required.
/// - If the memref's element type is a vector type then it coincides with the
///   result type.
/// - The permutation map doesn't perform permutation (broadcasting is allowed).
/// - The op has no mask.
struct TransferReadToVectorLoadLowering
    : public OpRewritePattern<vector::TransferReadOp> {
  TransferReadToVectorLoadLowering(MLIRContext *context)
      : OpRewritePattern<vector::TransferReadOp>(context) {}
  LogicalResult matchAndRewrite(vector::TransferReadOp read,
                                PatternRewriter &rewriter) const override {
    SmallVector<unsigned, 4> broadcastedDims;
    // TODO: Support permutations.
    if (!read.permutation_map().isMinorIdentityWithBroadcasting(
            &broadcastedDims))
      return failure();
    auto memRefType = read.getShapedType().dyn_cast<MemRefType>();
    if (!memRefType)
      return failure();

    // If there is broadcasting involved then we first load the unbroadcasted
    // vector, and then broadcast it with `vector.broadcast`.
    ArrayRef<int64_t> vectorShape = read.getVectorType().getShape();
    SmallVector<int64_t, 4> unbroadcastedVectorShape(vectorShape.begin(),
                                                     vectorShape.end());
    for (unsigned i : broadcastedDims)
      unbroadcastedVectorShape[i] = 1;
    VectorType unbroadcastedVectorType = VectorType::get(
        unbroadcastedVectorShape, read.getVectorType().getElementType());

    // `vector.load` supports vector types as memref's elements only when the
    // resulting vector type is the same as the element type.
    if (memRefType.getElementType().isa<VectorType>() &&
        memRefType.getElementType() != unbroadcastedVectorType)
      return failure();
    // Only the default layout is supported by `vector.load`.
    // TODO: Support non-default layouts.
    if (!memRefType.getAffineMaps().empty())
      return failure();
    // TODO: When out-of-bounds masking is required, we can create a
    //       MaskedLoadOp.
    if (read.hasOutOfBoundsDim())
      return failure();
    if (read.mask())
      return failure();
    Operation *loadOp;
    if (!broadcastedDims.empty() &&
        unbroadcastedVectorType.getNumElements() == 1) {
      // If broadcasting is required and the number of loaded elements is 1 then
      // we can create `memref.load` instead of `vector.load`.
      loadOp = rewriter.create<memref::LoadOp>(read.getLoc(), read.source(),
                                               read.indices());
    } else {
      // Otherwise create `vector.load`.
      loadOp = rewriter.create<vector::LoadOp>(read.getLoc(),
                                               unbroadcastedVectorType,
                                               read.source(), read.indices());
    }

    // Insert a broadcasting op if required.
    if (!broadcastedDims.empty()) {
      rewriter.replaceOpWithNewOp<vector::BroadcastOp>(
          read, read.getVectorType(), loadOp->getResult(0));
    } else {
      rewriter.replaceOp(read, loadOp->getResult(0));
    }

    return success();
  }
};

/// Progressive lowering of transfer_write. This pattern supports lowering of
/// `vector.transfer_write` to `vector.store` if all of the following hold:
/// - The op writes to a memref with the default layout.
/// - Out-of-bounds masking is not required.
/// - If the memref's element type is a vector type then it coincides with the
///   type of the written value.
/// - The permutation map is the minor identity map (neither permutation nor
///   broadcasting is allowed).
/// - The op has no mask.
struct TransferWriteToVectorStoreLowering
    : public OpRewritePattern<vector::TransferWriteOp> {
  TransferWriteToVectorStoreLowering(MLIRContext *context)
      : OpRewritePattern<vector::TransferWriteOp>(context) {}
  LogicalResult matchAndRewrite(vector::TransferWriteOp write,
                                PatternRewriter &rewriter) const override {
    // TODO: Support non-minor-identity maps
    if (!write.permutation_map().isMinorIdentity())
      return failure();
    auto memRefType = write.getShapedType().dyn_cast<MemRefType>();
    if (!memRefType)
      return failure();
    // `vector.store` supports vector types as memref's elements only when the
    // type of the vector value being written is the same as the element type.
    if (memRefType.getElementType().isa<VectorType>() &&
        memRefType.getElementType() != write.getVectorType())
      return failure();
    // Only the default layout is supported by `vector.store`.
    // TODO: Support non-default layouts.
    if (!memRefType.getAffineMaps().empty())
      return failure();
    // TODO: When out-of-bounds masking is required, we can create a
    //       MaskedStoreOp.
    if (write.hasOutOfBoundsDim())
      return failure();
    if (write.mask())
      return failure();
    rewriter.replaceOpWithNewOp<vector::StoreOp>(
        write, write.vector(), write.source(), write.indices());
    return success();
  }
};

/// Lower transfer_read op with permutation into a transfer_read with a
/// permutation map composed of leading zeros followed by a minor identiy +
/// vector.transpose op.
/// Ex:
///     vector.transfer_read ...
///         permutation_map: (d0, d1, d2) -> (0, d1)
/// into:
///     %v = vector.transfer_read ...
///         permutation_map: (d0, d1, d2) -> (d1, 0)
///     vector.transpose %v, [1, 0]
///
///     vector.transfer_read ...
///         permutation_map: (d0, d1, d2, d3) -> (0, 0, 0, d1, d3)
/// into:
///     %v = vector.transfer_read ...
///         permutation_map: (d0, d1, d2, d3) -> (0, 0, d1, 0, d3)
///     vector.transpose %v, [0, 1, 3, 2, 4]
/// Note that an alternative is to transform it to linalg.transpose +
/// vector.transfer_read to do the transpose in memory instead.
struct TransferReadPermutationLowering
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<unsigned> permutation;
    AffineMap map = op.permutation_map();
    if (!map.isPermutationOfMinorIdentityWithBroadcasting(permutation))
      return failure();
    AffineMap permutationMap =
        map.getPermutationMap(permutation, op.getContext());
    if (permutationMap.isIdentity())
      return failure();

    // Caluclate the map of the new read by applying the inverse permutation.
    permutationMap = inversePermutation(permutationMap);
    AffineMap newMap = permutationMap.compose(map);
    // Apply the reverse transpose to deduce the type of the transfer_read.
    ArrayRef<int64_t> originalShape = op.getVectorType().getShape();
    SmallVector<int64_t> newVectorShape(originalShape.size());
    for (auto pos : llvm::enumerate(permutation)) {
      newVectorShape[pos.value()] = originalShape[pos.index()];
    }

    Value newMask;
    if (op.mask()) {
      // Remove unused dims from the permutation map. E.g.:
      // E.g.:  (d0, d1, d2, d3, d4, d5) -> (d5, 0, d3, 0, d2)
      // comp = (d0, d1, d2) -> (d2, 0, d1, 0 d0)
      auto comp = compressUnusedDims(map);
      // Get positions of remaining result dims.
      // E.g.:  (d0, d1, d2) -> (d2, 0, d1, 0 d0)
      // maskTransposeIndices = [ 2,     1,    0]
      SmallVector<int64_t> maskTransposeIndices;
      for (unsigned i = 0; i < comp.getNumResults(); ++i) {
        if (auto expr = comp.getResult(i).dyn_cast<AffineDimExpr>())
          maskTransposeIndices.push_back(expr.getPosition());
      }

      newMask = rewriter.create<vector::TransposeOp>(op.getLoc(), op.mask(),
                                                     maskTransposeIndices);
    }

    VectorType newReadType =
        VectorType::get(newVectorShape, op.getVectorType().getElementType());
    Value newRead = rewriter.create<vector::TransferReadOp>(
        op.getLoc(), newReadType, op.source(), op.indices(), newMap,
        op.padding(), newMask, op.in_bounds() ? *op.in_bounds() : ArrayAttr());

    SmallVector<int64_t> transposePerm(permutation.begin(), permutation.end());
    rewriter.replaceOpWithNewOp<vector::TransposeOp>(op, newRead,
                                                     transposePerm);
    return success();
  }
};

/// Lower transfer_read op with broadcast in the leading dimensions into
/// transfer_read of lower rank + vector.broadcast.
/// Ex: vector.transfer_read ...
///         permutation_map: (d0, d1, d2, d3) -> (0, d1, 0, d3)
/// into:
///     %v = vector.transfer_read ...
///         permutation_map: (d0, d1, d2, d3) -> (d1, 0, d3)
///     vector.broadcast %v
struct TransferOpReduceRank : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp op,
                                PatternRewriter &rewriter) const override {
    AffineMap map = op.permutation_map();
    unsigned numLeadingBroadcast = 0;
    for (auto expr : map.getResults()) {
      auto dimExpr = expr.dyn_cast<AffineConstantExpr>();
      if (!dimExpr || dimExpr.getValue() != 0)
        break;
      numLeadingBroadcast++;
    }
    // If there are no leading zeros in the map there is nothing to do.
    if (numLeadingBroadcast == 0)
      return failure();
    VectorType originalVecType = op.getVectorType();
    unsigned reducedShapeRank = originalVecType.getRank() - numLeadingBroadcast;
    // Calculate new map, vector type and masks without the leading zeros.
    AffineMap newMap = AffineMap::get(
        map.getNumDims(), 0, map.getResults().take_back(reducedShapeRank),
        op.getContext());
    // Only remove the leading zeros if the rest of the map is a minor identity
    // with broadasting. Otherwise we first want to permute the map.
    if (!newMap.isMinorIdentityWithBroadcasting())
      return failure();
    SmallVector<int64_t> newShape = llvm::to_vector<4>(
        originalVecType.getShape().take_back(reducedShapeRank));
    // Vector rank cannot be zero. Handled by TransferReadToVectorLoadLowering.
    if (newShape.empty())
      return failure();
    VectorType newReadType =
        VectorType::get(newShape, originalVecType.getElementType());
    ArrayAttr newInBounds =
        op.in_bounds()
            ? rewriter.getArrayAttr(
                  op.in_boundsAttr().getValue().take_back(reducedShapeRank))
            : ArrayAttr();
    Value newRead = rewriter.create<vector::TransferReadOp>(
        op.getLoc(), newReadType, op.source(), op.indices(), newMap,
        op.padding(), op.mask(), newInBounds);
    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(op, originalVecType,
                                                     newRead);
    return success();
  }
};

// Trims leading one dimensions from `oldType` and returns the result type.
// Returns `vector<1xT>` if `oldType` only has one element.
static VectorType trimLeadingOneDims(VectorType oldType) {
  ArrayRef<int64_t> oldShape = oldType.getShape();
  ArrayRef<int64_t> newShape =
      oldShape.drop_while([](int64_t dim) { return dim == 1; });
  // Make sure we have at least 1 dimension per vector type requirements.
  if (newShape.empty())
    newShape = oldShape.take_back();
  return VectorType::get(newShape, oldType.getElementType());
}

// Casts away leading one dimensions in vector.extract_strided_slice's vector
// input by inserting vector.shape_cast.
struct CastAwayExtractStridedSliceLeadingOneDim
    : public OpRewritePattern<vector::ExtractStridedSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ExtractStridedSliceOp extractOp,
                                PatternRewriter &rewriter) const override {
    // vector.extract_strided_slice requires the input and output vector to have
    // the same rank. Here we drop leading one dimensions from the input vector
    // type to make sure we don't cause mismatch.
    VectorType oldSrcType = extractOp.getVectorType();
    VectorType newSrcType = trimLeadingOneDims(oldSrcType);

    if (newSrcType.getRank() == oldSrcType.getRank())
      return failure();

    int64_t dropCount = oldSrcType.getRank() - newSrcType.getRank();

    VectorType oldDstType = extractOp.getType();
    VectorType newDstType =
        VectorType::get(oldDstType.getShape().drop_front(dropCount),
                        oldDstType.getElementType());

    Location loc = extractOp.getLoc();

    Value newSrcVector = rewriter.create<vector::ShapeCastOp>(
        loc, newSrcType, extractOp.vector());

    // The offsets/sizes/strides attribute can have a less number of elements
    // than the input vector's rank: it is meant for the leading dimensions.
    auto newOffsets = rewriter.getArrayAttr(
        extractOp.offsets().getValue().drop_front(dropCount));
    auto newSizes = rewriter.getArrayAttr(
        extractOp.sizes().getValue().drop_front(dropCount));
    auto newStrides = rewriter.getArrayAttr(
        extractOp.strides().getValue().drop_front(dropCount));

    auto newExtractOp = rewriter.create<vector::ExtractStridedSliceOp>(
        loc, newDstType, newSrcVector, newOffsets, newSizes, newStrides);

    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(extractOp, oldDstType,
                                                     newExtractOp);

    return success();
  }
};

// Casts away leading one dimensions in vector.extract_strided_slice's vector
// inputs by inserting vector.shape_cast.
struct CastAwayInsertStridedSliceLeadingOneDim
    : public OpRewritePattern<vector::InsertStridedSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::InsertStridedSliceOp insertOp,
                                PatternRewriter &rewriter) const override {
    VectorType oldSrcType = insertOp.getSourceVectorType();
    VectorType newSrcType = trimLeadingOneDims(oldSrcType);
    VectorType oldDstType = insertOp.getDestVectorType();
    VectorType newDstType = trimLeadingOneDims(oldDstType);

    if (newSrcType.getRank() == oldSrcType.getRank() &&
        newDstType.getRank() == oldDstType.getRank())
      return failure();

    // Trim leading one dimensions from both operands.
    Location loc = insertOp.getLoc();

    Value newSrcVector = rewriter.create<vector::ShapeCastOp>(
        loc, newSrcType, insertOp.source());
    Value newDstVector =
        rewriter.create<vector::ShapeCastOp>(loc, newDstType, insertOp.dest());

    auto newOffsets = rewriter.getArrayAttr(
        insertOp.offsets().getValue().take_back(newDstType.getRank()));
    auto newStrides = rewriter.getArrayAttr(
        insertOp.strides().getValue().take_back(newSrcType.getRank()));

    auto newInsertOp = rewriter.create<vector::InsertStridedSliceOp>(
        loc, newDstType, newSrcVector, newDstVector, newOffsets, newStrides);

    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(insertOp, oldDstType,
                                                     newInsertOp);

    return success();
  }
};

// Turns vector.transfer_read on vector with leading 1 dimensions into
// vector.shape_cast followed by vector.transfer_read on vector without leading
// 1 dimensions.
struct CastAwayTransferReadLeadingOneDim
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp read,
                                PatternRewriter &rewriter) const override {
    if (read.mask())
      return failure();

    auto shapedType = read.source().getType().cast<ShapedType>();
    if (shapedType.getElementType() != read.getVectorType().getElementType())
      return failure();

    VectorType oldType = read.getVectorType();
    VectorType newType = trimLeadingOneDims(oldType);

    if (newType == oldType)
      return failure();

    AffineMap oldMap = read.permutation_map();
    ArrayRef<AffineExpr> newResults =
        oldMap.getResults().take_back(newType.getRank());
    AffineMap newMap =
        AffineMap::get(oldMap.getNumDims(), oldMap.getNumSymbols(), newResults,
                       rewriter.getContext());

    ArrayAttr inBounds;
    if (read.in_bounds())
      inBounds = rewriter.getArrayAttr(
          read.in_boundsAttr().getValue().take_back(newType.getRank()));

    auto newRead = rewriter.create<vector::TransferReadOp>(
        read.getLoc(), newType, read.source(), read.indices(), newMap,
        read.padding(), inBounds);
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(read, oldType, newRead);

    return success();
  }
};

// Turns vector.transfer_write on vector with leading 1 dimensions into
// vector.shape_cast followed by vector.transfer_write on vector without leading
// 1 dimensions.
struct CastAwayTransferWriteLeadingOneDim
    : public OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp write,
                                PatternRewriter &rewriter) const override {
    if (write.mask())
      return failure();

    auto shapedType = write.source().getType().dyn_cast<ShapedType>();
    if (shapedType.getElementType() != write.getVectorType().getElementType())
      return failure();

    VectorType oldType = write.getVectorType();
    VectorType newType = trimLeadingOneDims(oldType);

    if (newType == oldType)
      return failure();

    AffineMap oldMap = write.permutation_map();
    ArrayRef<AffineExpr> newResults =
        oldMap.getResults().take_back(newType.getRank());
    AffineMap newMap =
        AffineMap::get(oldMap.getNumDims(), oldMap.getNumSymbols(), newResults,
                       rewriter.getContext());

    ArrayAttr inBounds;
    if (write.in_bounds())
      inBounds = rewriter.getArrayAttr(
          write.in_boundsAttr().getValue().take_back(newType.getRank()));

    auto newVector = rewriter.create<vector::ShapeCastOp>(
        write.getLoc(), newType, write.vector());
    rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
        write, newVector, write.source(), write.indices(), newMap, inBounds);

    return success();
  }
};

template <typename BroadCastType>
struct CastAwayBroadcastLeadingOneDim : public OpRewritePattern<BroadCastType> {
  using OpRewritePattern<BroadCastType>::OpRewritePattern;

  LogicalResult matchAndRewrite(BroadCastType broadcastOp,
                                PatternRewriter &rewriter) const override {
    VectorType dstType =
        broadcastOp.getResult().getType().template dyn_cast<VectorType>();
    if (!dstType)
      return failure();
    VectorType newDstType = trimLeadingOneDims(dstType);
    if (newDstType == dstType)
      return failure();
    Location loc = broadcastOp.getLoc();
    Value source = broadcastOp->getOperand(0);
    VectorType srcVecType = source.getType().template dyn_cast<VectorType>();
    if (srcVecType)
      srcVecType = trimLeadingOneDims(srcVecType);
    if (srcVecType && srcVecType != source.getType()) {
      source = rewriter.create<vector::ShapeCastOp>(loc, srcVecType, source);
    }
    Value newBroadcastOp =
        rewriter.create<BroadCastType>(loc, newDstType, source);
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(broadcastOp, dstType,
                                                     newBroadcastOp);
    return success();
  }
};

class CastAwayElementwiseLeadingOneDim : public RewritePattern {
public:
  CastAwayElementwiseLeadingOneDim(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!OpTrait::hasElementwiseMappableTraits(op) || op->getNumResults() != 1)
      return failure();
    auto vecType = op->getResultTypes()[0].dyn_cast<VectorType>();
    if (!vecType)
      return failure();
    VectorType newVecType = trimLeadingOneDims(vecType);
    if (newVecType == vecType)
      return failure();

    SmallVector<Value, 4> newOperands;
    for (Value operand : op->getOperands()) {
      if (auto opVecType = operand.getType().dyn_cast<VectorType>()) {
        auto newType =
            VectorType::get(newVecType.getShape(), opVecType.getElementType());
        newOperands.push_back(rewriter.create<vector::ShapeCastOp>(
            op->getLoc(), newType, operand));
      } else {
        newOperands.push_back(operand);
      }
    }
    OperationState state(op->getLoc(), op->getName());
    state.addAttributes(op->getAttrs());
    state.addOperands(newOperands);
    state.addTypes(newVecType);
    Operation *newOp = rewriter.createOperation(state);
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(op, vecType,
                                                     newOp->getResult(0));
    return success();
  }
};

// Returns the values in `arrayAttr` as an integer vector.
static SmallVector<int64_t, 4> getIntValueVector(ArrayAttr arrayAttr) {
  return llvm::to_vector<4>(
      llvm::map_range(arrayAttr.getAsRange<IntegerAttr>(),
                      [](IntegerAttr attr) { return attr.getInt(); }));
}

// Shuffles vector.bitcast op after vector.extract op.
//
// This transforms IR like:
//   %0 = vector.bitcast %src : vector<4xf32> to vector<8xf16>
//   %1 = vector.extract %0[3] : vector<8xf16>
// Into:
//   %0 = vector.extract %src[1] : vector<4xf32>
//   %1 = vector.bitcast %0: vector<1xf32> to vector<2xf16>
//   %2 = vector.extract %1[1] : vector<2xf16>
struct BubbleDownVectorBitCastForExtract
    : public OpRewritePattern<vector::ExtractOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ExtractOp extractOp,
                                PatternRewriter &rewriter) const override {
    // Only support extracting scalars for now.
    if (extractOp.getVectorType().getRank() != 1)
      return failure();

    auto castOp = extractOp.vector().getDefiningOp<vector::BitCastOp>();
    if (!castOp)
      return failure();

    VectorType castSrcType = castOp.getSourceVectorType();
    VectorType castDstType = castOp.getResultVectorType();
    assert(castSrcType.getRank() == castDstType.getRank());

    // Fail to match if we only have one element in the cast op source.
    // This is to avoid infinite loop given that this pattern can generate
    // such cases.
    if (castSrcType.getNumElements() == 1)
      return failure();

    // Only support casting to a larger number of elements or now.
    // E.g., vector<4xf32> -> vector<8xf16>.
    if (castSrcType.getNumElements() > castDstType.getNumElements())
      return failure();

    unsigned expandRatio =
        castDstType.getNumElements() / castSrcType.getNumElements();

    auto getFirstIntValue = [](ArrayAttr attr) -> uint64_t {
      return (*attr.getAsValueRange<IntegerAttr>().begin()).getZExtValue();
    };

    uint64_t index = getFirstIntValue(extractOp.position());

    // Get the single scalar (as a vector) in the source value that packs the
    // desired scalar. E.g. extract vector<1xf32> from vector<4xf32>
    VectorType oneScalarType =
        VectorType::get({1}, castSrcType.getElementType());
    Value packedValue = rewriter.create<vector::ExtractOp>(
        extractOp.getLoc(), oneScalarType, castOp.source(),
        rewriter.getI64ArrayAttr(index / expandRatio));

    // Cast it to a vector with the desired scalar's type.
    // E.g. f32 -> vector<2xf16>
    VectorType packedType =
        VectorType::get({expandRatio}, castDstType.getElementType());
    Value castedValue = rewriter.create<vector::BitCastOp>(
        extractOp.getLoc(), packedType, packedValue);

    // Finally extract the desired scalar.
    rewriter.replaceOpWithNewOp<vector::ExtractOp>(
        extractOp, extractOp.getType(), castedValue,
        rewriter.getI64ArrayAttr(index % expandRatio));

    return success();
  }
};

// Shuffles vector.bitcast op after vector.extract_strided_slice op.
//
// This transforms IR like:
//    %cast = vector.bitcast %arg0: vector<4xf32> to vector<8xf16>
//     %0 = vector.extract_strided_slice %cast {
//            offsets = [4], sizes = [4], strides = [1]
//          } : vector<8xf16> to vector<4xf16>
// Into:
//   %0 = vector.extract_strided_slice %src {
//          offsets = [2], sizes = [2], strides = [1]
//        } : vector<4xf32> to vector<2xf32>
//   %1 = vector.bitcast %0 : vector<2xf32> to vector<4xf16>
struct BubbleDownBitCastForStridedSliceExtract
    : public OpRewritePattern<vector::ExtractStridedSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ExtractStridedSliceOp extractOp,
                                PatternRewriter &rewriter) const override {
    auto castOp = extractOp.vector().getDefiningOp<vector::BitCastOp>();
    if (!castOp)
      return failure();

    VectorType castSrcType = castOp.getSourceVectorType();
    VectorType castDstType = castOp.getResultVectorType();
    assert(castSrcType.getRank() == castDstType.getRank());

    int64_t castSrcLastDim = castSrcType.getShape().back();
    int64_t castDstLastDim = castDstType.getShape().back();
    // Require casting to more elements for now; other cases to be implemented.
    if (castSrcLastDim > castDstLastDim)
      return failure();

    // Only accept all one strides for now.
    if (llvm::any_of(extractOp.strides().getAsValueRange<IntegerAttr>(),
                     [](const APInt &val) { return !val.isOneValue(); }))
      return failure();

    unsigned rank = extractOp.getVectorType().getRank();
    assert(castDstLastDim % castSrcLastDim == 0);
    int64_t expandRatio = castDstLastDim / castSrcLastDim;

    // If we have a less number of offsets than the rank, then implicitly we
    // are selecting the full range for the last bitcasted dimension; other
    // dimensions aren't affected. Otherwise, we need to scale down the last
    // dimension's offset given we are extracting from less elements now.
    ArrayAttr newOffsets = extractOp.offsets();
    if (newOffsets.size() == rank) {
      SmallVector<int64_t, 4> offsets = getIntValueVector(newOffsets);
      if (offsets.back() % expandRatio != 0)
        return failure();
      offsets.back() = offsets.back() / expandRatio;
      newOffsets = rewriter.getI64ArrayAttr(offsets);
    }

    // Similarly for sizes.
    ArrayAttr newSizes = extractOp.sizes();
    if (newSizes.size() == rank) {
      SmallVector<int64_t, 4> sizes = getIntValueVector(newSizes);
      if (sizes.back() % expandRatio != 0)
        return failure();
      sizes.back() = sizes.back() / expandRatio;
      newSizes = rewriter.getI64ArrayAttr(sizes);
    }

    SmallVector<int64_t, 4> dims =
        llvm::to_vector<4>(extractOp.getType().cast<VectorType>().getShape());
    dims.back() = dims.back() / expandRatio;
    VectorType newExtractType =
        VectorType::get(dims, castSrcType.getElementType());

    auto newExtractOp = rewriter.create<vector::ExtractStridedSliceOp>(
        extractOp.getLoc(), newExtractType, castOp.source(), newOffsets,
        newSizes, extractOp.strides());

    rewriter.replaceOpWithNewOp<vector::BitCastOp>(
        extractOp, extractOp.getType(), newExtractOp);

    return success();
  }
};

// Shuffles vector.bitcast op before vector.insert_strided_slice op.
//
// This transforms IR like:
//   %0 = vector.insert_strided_slice %src, %dst {
//          offsets = [0], strides = [1]} : vector<4xf16> into vector<8xf16>
//   %1 = vector.bitcast %0: vector<8xf16> to vector<4xf32>
// Into:
//   %0 = vector.bitcast %src : vector<4xf16> to vector<2xf32>
//   %1 = vector.bitcast %dst : vector<8xf16> to vector<4xf32>
//   %2 = vector.insert_strided_slice %src, %dst {
//          offsets = [0], strides = [1]} : vector<2xf32> into vector<4xf32>
struct BubbleUpBitCastForStridedSliceInsert
    : public OpRewritePattern<vector::BitCastOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::BitCastOp bitcastOp,
                                PatternRewriter &rewriter) const override {
    VectorType castSrcType = bitcastOp.getSourceVectorType();
    VectorType castDstType = bitcastOp.getResultVectorType();
    assert(castSrcType.getRank() == castDstType.getRank());

    int64_t castSrcLastDim = castSrcType.getShape().back();
    int64_t castDstLastDim = castDstType.getShape().back();
    // Require casting to less elements for now; other cases to be implemented.
    if (castSrcLastDim < castDstLastDim)
      return failure();

    assert(castSrcLastDim % castDstLastDim == 0);
    int64_t shrinkRatio = castSrcLastDim / castDstLastDim;

    auto insertOp =
        bitcastOp.source().getDefiningOp<vector::InsertStridedSliceOp>();
    if (!insertOp)
      return failure();

    // Only accept all one strides for now.
    if (llvm::any_of(insertOp.strides().getAsValueRange<IntegerAttr>(),
                     [](const APInt &val) { return !val.isOneValue(); }))
      return failure();

    unsigned rank = insertOp.getSourceVectorType().getRank();
    // Require insert op to have the same rank for the source and destination
    // vector; other cases to be implemented.
    if (rank != insertOp.getDestVectorType().getRank())
      return failure();

    ArrayAttr newOffsets = insertOp.offsets();
    assert(newOffsets.size() == rank);
    SmallVector<int64_t, 4> offsets = getIntValueVector(newOffsets);
    if (offsets.back() % shrinkRatio != 0)
      return failure();
    offsets.back() = offsets.back() / shrinkRatio;
    newOffsets = rewriter.getI64ArrayAttr(offsets);

    SmallVector<int64_t, 4> srcDims =
        llvm::to_vector<4>(insertOp.getSourceVectorType().getShape());
    srcDims.back() = srcDims.back() / shrinkRatio;
    VectorType newCastSrcType =
        VectorType::get(srcDims, castDstType.getElementType());

    auto newCastSrcOp = rewriter.create<vector::BitCastOp>(
        bitcastOp.getLoc(), newCastSrcType, insertOp.source());

    SmallVector<int64_t, 4> dstDims =
        llvm::to_vector<4>(insertOp.getDestVectorType().getShape());
    dstDims.back() = dstDims.back() / shrinkRatio;
    VectorType newCastDstType =
        VectorType::get(dstDims, castDstType.getElementType());

    auto newCastDstOp = rewriter.create<vector::BitCastOp>(
        bitcastOp.getLoc(), newCastDstType, insertOp.dest());

    rewriter.replaceOpWithNewOp<vector::InsertStridedSliceOp>(
        bitcastOp, bitcastOp.getType(), newCastSrcOp, newCastDstOp, newOffsets,
        insertOp.strides());

    return success();
  }
};

static Value createCastToIndexLike(PatternRewriter &rewriter, Location loc,
                                   Type targetType, Value value) {
  if (targetType == value.getType())
    return value;

  bool targetIsIndex = targetType.isIndex();
  bool valueIsIndex = value.getType().isIndex();
  if (targetIsIndex ^ valueIsIndex)
    return rewriter.create<IndexCastOp>(loc, targetType, value);

  auto targetIntegerType = targetType.dyn_cast<IntegerType>();
  auto valueIntegerType = value.getType().dyn_cast<IntegerType>();
  assert(targetIntegerType && valueIntegerType &&
         "unexpected cast between types other than integers and index");
  assert(targetIntegerType.getSignedness() == valueIntegerType.getSignedness());

  if (targetIntegerType.getWidth() > valueIntegerType.getWidth())
    return rewriter.create<SignExtendIOp>(loc, targetIntegerType, value);
  return rewriter.create<TruncateIOp>(loc, targetIntegerType, value);
}

// Helper that returns a vector comparison that constructs a mask:
//     mask = [0,1,..,n-1] + [o,o,..,o] < [b,b,..,b]
//
// NOTE: The LLVM::GetActiveLaneMaskOp intrinsic would provide an alternative,
//       much more compact, IR for this operation, but LLVM eventually
//       generates more elaborate instructions for this intrinsic since it
//       is very conservative on the boundary conditions.
static Value buildVectorComparison(PatternRewriter &rewriter, Operation *op,
                                   bool enableIndexOptimizations, int64_t dim,
                                   Value b, Value *off = nullptr) {
  auto loc = op->getLoc();
  // If we can assume all indices fit in 32-bit, we perform the vector
  // comparison in 32-bit to get a higher degree of SIMD parallelism.
  // Otherwise we perform the vector comparison using 64-bit indices.
  Value indices;
  Type idxType;
  if (enableIndexOptimizations) {
    indices = rewriter.create<ConstantOp>(
        loc, rewriter.getI32VectorAttr(
                 llvm::to_vector<4>(llvm::seq<int32_t>(0, dim))));
    idxType = rewriter.getI32Type();
  } else {
    indices = rewriter.create<ConstantOp>(
        loc, rewriter.getI64VectorAttr(
                 llvm::to_vector<4>(llvm::seq<int64_t>(0, dim))));
    idxType = rewriter.getI64Type();
  }
  // Add in an offset if requested.
  if (off) {
    Value o = createCastToIndexLike(rewriter, loc, idxType, *off);
    Value ov = rewriter.create<SplatOp>(loc, indices.getType(), o);
    indices = rewriter.create<AddIOp>(loc, ov, indices);
  }
  // Construct the vector comparison.
  Value bound = createCastToIndexLike(rewriter, loc, idxType, b);
  Value bounds = rewriter.create<SplatOp>(loc, indices.getType(), bound);
  return rewriter.create<CmpIOp>(loc, CmpIPredicate::slt, indices, bounds);
}

template <typename ConcreteOp>
struct MaterializeTransferMask : public OpRewritePattern<ConcreteOp> {
public:
  explicit MaterializeTransferMask(MLIRContext *context, bool enableIndexOpt)
      : mlir::OpRewritePattern<ConcreteOp>(context),
        enableIndexOptimizations(enableIndexOpt) {}

  LogicalResult matchAndRewrite(ConcreteOp xferOp,
                                PatternRewriter &rewriter) const override {
    if (!xferOp.hasOutOfBoundsDim())
      return failure();

    if (xferOp.getVectorType().getRank() > 1 ||
        llvm::size(xferOp.indices()) == 0)
      return failure();

    Location loc = xferOp->getLoc();
    VectorType vtp = xferOp.getVectorType();

    // * Create a vector with linear indices [ 0 .. vector_length - 1 ].
    // * Create offsetVector = [ offset + 0 .. offset + vector_length - 1 ].
    // * Let dim the memref dimension, compute the vector comparison mask
    //   (in-bounds mask):
    //   [ offset + 0 .. offset + vector_length - 1 ] < [ dim .. dim ]
    //
    // TODO: when the leaf transfer rank is k > 1, we need the last `k`
    //       dimensions here.
    unsigned vecWidth = vtp.getNumElements();
    unsigned lastIndex = llvm::size(xferOp.indices()) - 1;
    Value off = xferOp.indices()[lastIndex];
    Value dim = rewriter.create<memref::DimOp>(loc, xferOp.source(), lastIndex);
    Value mask = buildVectorComparison(
        rewriter, xferOp, enableIndexOptimizations, vecWidth, dim, &off);

    if (xferOp.mask()) {
      // Intersect the in-bounds with the mask specified as an op parameter.
      mask = rewriter.create<AndOp>(loc, mask, xferOp.mask());
    }

    rewriter.updateRootInPlace(xferOp, [&]() {
      xferOp.maskMutable().assign(mask);
      xferOp.in_boundsAttr(rewriter.getBoolArrayAttr({true}));
    });

    return success();
  }

private:
  const bool enableIndexOptimizations;
};

/// Conversion pattern for a vector.create_mask (1-D only).
class VectorCreateMaskOpConversion
    : public OpRewritePattern<vector::CreateMaskOp> {
public:
  explicit VectorCreateMaskOpConversion(MLIRContext *context,
                                        bool enableIndexOpt)
      : mlir::OpRewritePattern<vector::CreateMaskOp>(context),
        enableIndexOptimizations(enableIndexOpt) {}

  LogicalResult matchAndRewrite(vector::CreateMaskOp op,
                                PatternRewriter &rewriter) const override {
    auto dstType = op.getType();
    int64_t rank = dstType.getRank();
    if (rank == 1) {
      rewriter.replaceOp(
          op, buildVectorComparison(rewriter, op, enableIndexOptimizations,
                                    dstType.getDimSize(0), op.getOperand(0)));
      return success();
    }
    return failure();
  }

private:
  const bool enableIndexOptimizations;
};

// Converts vector.multi_reduction into inner-most reduction form by inserting
// vector.transpose
struct InnerDimReductionConversion
    : public OpRewritePattern<vector::MultiDimReductionOp> {
  using OpRewritePattern<vector::MultiDimReductionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::MultiDimReductionOp multiReductionOp,
                                PatternRewriter &rewriter) const override {
    auto src = multiReductionOp.source();
    auto loc = multiReductionOp.getLoc();
    auto srcRank = multiReductionOp.getSourceVectorType().getRank();

    auto reductionDims = llvm::to_vector<4>(
        llvm::map_range(multiReductionOp.reduction_dims().cast<ArrayAttr>(),
                        [](Attribute attr) -> int64_t {
                          return attr.cast<IntegerAttr>().getInt();
                        }));
    llvm::sort(reductionDims);

    int64_t reductionSize = multiReductionOp.reduction_dims().size();

    // Fails if already inner most reduction.
    bool innerMostReduction = true;
    for (int i = 0; i < reductionSize; ++i) {
      if (reductionDims[reductionSize - i - 1] != srcRank - i - 1) {
        innerMostReduction = false;
      }
    }
    if (innerMostReduction)
      return failure();

    // Permutes the indices so reduction dims are inner most dims.
    SmallVector<int64_t> indices;
    for (int i = 0; i < srcRank; ++i) {
      indices.push_back(i);
    }
    int ir = reductionSize - 1;
    int id = srcRank - 1;
    while (ir >= 0) {
      std::swap(indices[reductionDims[ir--]], indices[id--]);
    }

    // Sets inner most dims as reduction.
    SmallVector<bool> reductionMask(srcRank, false);
    for (int i = 0; i < reductionSize; ++i) {
      reductionMask[srcRank - i - 1] = true;
    }
    auto transposeOp = rewriter.create<vector::TransposeOp>(loc, src, indices);
    rewriter.replaceOpWithNewOp<vector::MultiDimReductionOp>(
        multiReductionOp, transposeOp.result(), reductionMask,
        multiReductionOp.kind());
    return success();
  }
};

// Reduces the rank of vector.mult_reduction nd -> 2d given all reduction
// dimensions are inner most.
struct ReduceMultiDimReductionRank
    : public OpRewritePattern<vector::MultiDimReductionOp> {
  using OpRewritePattern<vector::MultiDimReductionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::MultiDimReductionOp multiReductionOp,
                                PatternRewriter &rewriter) const override {
    auto srcRank = multiReductionOp.getSourceVectorType().getRank();
    auto srcShape = multiReductionOp.getSourceVectorType().getShape();
    if (srcRank == 2)
      return failure();

    auto loc = multiReductionOp.getLoc();
    auto reductionDims = llvm::to_vector<4>(
        llvm::map_range(multiReductionOp.reduction_dims().cast<ArrayAttr>(),
                        [](Attribute attr) -> int64_t {
                          return attr.cast<IntegerAttr>().getInt();
                        }));
    llvm::sort(reductionDims);

    // Fails if not inner most reduction.
    int64_t reductionSize = reductionDims.size();
    bool innerMostReduction = true;
    for (int i = 0; i < reductionSize; ++i) {
      if (reductionDims[reductionSize - i - 1] != srcRank - i - 1) {
        innerMostReduction = false;
      }
    }
    if (!innerMostReduction)
      return failure();

    // Extracts 2d rank reduction shape.
    int innerDims = 1;
    int outterDims = 1;
    SmallVector<int64_t> innerDimsShape;
    for (int i = 0; i < srcRank; ++i) {
      if (i < (srcRank - reductionSize)) {
        innerDims *= srcShape[i];
        innerDimsShape.push_back(srcShape[i]);
      } else {
        outterDims *= srcShape[i];
      }
    }

    // Creates shape cast for the inputs n_d -> 2d
    auto castedType = VectorType::get(
        {innerDims, outterDims},
        multiReductionOp.getSourceVectorType().getElementType());
    auto castedOp = rewriter.create<vector::ShapeCastOp>(
        loc, castedType, multiReductionOp.source());

    // Creates the canonical form of 2d vector.multi_reduction with inner most
    // dim as reduction.
    auto newOp = rewriter.create<vector::MultiDimReductionOp>(
        loc, castedOp.result(), ArrayRef<bool>{false, true},
        multiReductionOp.kind());

    // Creates shape cast for the output 2d -> nd
    auto outputCastedType = VectorType::get(
        innerDimsShape,
        multiReductionOp.getSourceVectorType().getElementType());
    Value castedOutputOp = rewriter.create<vector::ShapeCastOp>(
        loc, outputCastedType, newOp.dest());

    rewriter.replaceOp(multiReductionOp, castedOutputOp);
    return success();
  }
};

// Converts 2d vector.multi_reduction with inner most reduction dimension into a
// sequence of vector.reduction ops.
struct TwoDimMultiReductionToReduction
    : public OpRewritePattern<vector::MultiDimReductionOp> {
  using OpRewritePattern<vector::MultiDimReductionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::MultiDimReductionOp multiReductionOp,
                                PatternRewriter &rewriter) const override {
    auto srcRank = multiReductionOp.getSourceVectorType().getRank();
    if (srcRank != 2)
      return failure();

    if (multiReductionOp.getReductionMask()[0] ||
        !multiReductionOp.getReductionMask()[1])
      return failure();

    auto loc = multiReductionOp.getLoc();

    Value result =
        multiReductionOp.getDestVectorType().getElementType().isIntOrIndex()
            ? rewriter.create<ConstantOp>(
                  loc, multiReductionOp.getDestVectorType(),
                  DenseElementsAttr::get(multiReductionOp.getDestVectorType(),
                                         0))
            : rewriter.create<ConstantOp>(
                  loc, multiReductionOp.getDestVectorType(),
                  DenseElementsAttr::get(multiReductionOp.getDestVectorType(),
                                         0.0f));

    int outerDim = multiReductionOp.getSourceVectorType().getShape()[0];

    // TODO: Add vector::CombiningKind attribute instead of string to
    // vector.reduction.
    auto getKindStr = [](vector::CombiningKind kind) {
      switch (kind) {
      case vector::CombiningKind::ADD:
        return "add";
      case vector::CombiningKind::MUL:
        return "mul";
      case vector::CombiningKind::MIN:
        return "min";
      case vector::CombiningKind::MAX:
        return "max";
      case vector::CombiningKind::AND:
        return "and";
      case vector::CombiningKind::OR:
        return "or";
      case vector::CombiningKind::XOR:
        return "xor";
      }
      llvm_unreachable("unknown combining kind");
    };

    for (int i = 0; i < outerDim; ++i) {
      auto v = rewriter.create<vector::ExtractOp>(
          loc, multiReductionOp.source(), ArrayRef<int64_t>{i});
      auto reducedValue = rewriter.create<vector::ReductionOp>(
          loc, multiReductionOp.getDestVectorType().getElementType(),
          rewriter.getStringAttr(getKindStr(multiReductionOp.kind())), v,
          ValueRange{});
      result = rewriter.create<vector::InsertElementOp>(loc, reducedValue,
                                                        result, i);
    }
    rewriter.replaceOp(multiReductionOp, result);
    return success();
  }
};

void mlir::vector::populateVectorMaskMaterializationPatterns(
    RewritePatternSet &patterns, bool enableIndexOptimizations) {
  patterns.add<VectorCreateMaskOpConversion,
               MaterializeTransferMask<vector::TransferReadOp>,
               MaterializeTransferMask<vector::TransferWriteOp>>(
      patterns.getContext(), enableIndexOptimizations);
}

// TODO: Add pattern to rewrite ExtractSlices(ConstantMaskOp).
// TODO: Add this as DRR pattern.
void mlir::vector::populateVectorToVectorTransformationPatterns(
    RewritePatternSet &patterns) {
  patterns.add<ShapeCastOpDecomposer, ShapeCastOpFolder, TupleGetFolderOp,
               TransferReadExtractPattern, TransferWriteInsertPattern>(
      patterns.getContext());
}

void mlir::vector::populateSplitVectorTransferPatterns(
    RewritePatternSet &patterns,
    std::function<bool(Operation *)> ignoreFilter) {
  patterns.add<SplitTransferReadOp, SplitTransferWriteOp>(patterns.getContext(),
                                                          ignoreFilter);
}

void mlir::vector::populateCastAwayVectorLeadingOneDimPatterns(
    RewritePatternSet &patterns) {
  patterns.add<CastAwayExtractStridedSliceLeadingOneDim,
               CastAwayInsertStridedSliceLeadingOneDim,
               CastAwayTransferReadLeadingOneDim,
               CastAwayTransferWriteLeadingOneDim,
               CastAwayBroadcastLeadingOneDim<vector::BroadcastOp>,
               CastAwayBroadcastLeadingOneDim<SplatOp>,
               CastAwayElementwiseLeadingOneDim, ShapeCastOpFolder>(
      patterns.getContext());
}

void mlir::vector::populateBubbleVectorBitCastOpPatterns(
    RewritePatternSet &patterns) {
  patterns.add<BubbleDownVectorBitCastForExtract,
               BubbleDownBitCastForStridedSliceExtract,
               BubbleUpBitCastForStridedSliceInsert>(patterns.getContext());
}

void mlir::vector::populateVectorSlicesLoweringPatterns(
    RewritePatternSet &patterns) {
  patterns.add<ExtractSlicesOpLowering, InsertSlicesOpLowering>(
      patterns.getContext());
}

void mlir::vector::populateVectorContractLoweringPatterns(
    RewritePatternSet &patterns, VectorTransformsOptions parameters) {
  // clang-format off
  patterns.add<BroadcastOpLowering,
                  CreateMaskOpLowering,
                  ConstantMaskOpLowering,
                  OuterProductOpLowering,
                  ShapeCastOp2DDownCastRewritePattern,
                  ShapeCastOp2DUpCastRewritePattern,
                  ShapeCastOpRewritePattern>(patterns.getContext());
  patterns.add<ContractionOpLowering,
                  ContractionOpToMatmulOpLowering,
                  ContractionOpToOuterProductOpLowering>(parameters, patterns.getContext());
  // clang-format on
}

void mlir::vector::populateVectorTransposeLoweringPatterns(
    RewritePatternSet &patterns,
    VectorTransformsOptions vectorTransformOptions) {
  patterns.add<TransposeOpLowering>(vectorTransformOptions,
                                    patterns.getContext());
}

void mlir::vector::populateVectorTransferLoweringPatterns(
    RewritePatternSet &patterns) {
  patterns
      .add<TransferReadToVectorLoadLowering, TransferWriteToVectorStoreLowering,
           TransferReadPermutationLowering, TransferOpReduceRank>(
          patterns.getContext());
}

void mlir::vector::populateVectorMultiReductionLoweringPatterns(
    RewritePatternSet &patterns) {
  patterns.add<InnerDimReductionConversion, ReduceMultiDimReductionRank,
               TwoDimMultiReductionToReduction>(patterns.getContext());
}
