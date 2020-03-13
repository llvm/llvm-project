//===- VectorToLoops.cpp - Conversion within the Vector dialect -----------===//
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

#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/VectorOps/VectorOps.h"
#include "mlir/Dialect/VectorOps/VectorTransforms.h"
#include "mlir/Dialect/VectorOps/VectorUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/Functional.h"
#include "mlir/Support/STLExtras.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "vector-to-vector"

using namespace mlir;
using llvm::dbgs;
using mlir::functional::zipMap;

static llvm::cl::OptionCategory clOptionsCategory(DEBUG_TYPE " options");

static llvm::cl::opt<bool> lowerToLLVMMatrixIntrinsics(
    "vector-lower-matrix-intrinsics",
    llvm::cl::desc("Lower vector.contract to llvm.intr.matrix.multiply"),
    llvm::cl::init(false), llvm::cl::cat(clOptionsCategory));

/// Given a shape with sizes greater than 0 along all dimensions,
/// returns the distance, in number of elements, between a slice in a dimension
/// and the next slice in the same dimension.
///   e.g. shape[3, 4, 5] -> linearization_basis[20, 5, 1]
static SmallVector<int64_t, 8> computeStrides(ArrayRef<int64_t> shape) {
  if (shape.empty())
    return {};
  SmallVector<int64_t, 8> tmp;
  tmp.reserve(shape.size());
  int64_t running = 1;
  for (auto size : llvm::reverse(shape)) {
    assert(size > 0 && "size must be nonnegative");
    tmp.push_back(running);
    running *= size;
  }
  return SmallVector<int64_t, 8>(tmp.rbegin(), tmp.rend());
}

static int64_t computeMaxLinearIndex(ArrayRef<int64_t> basis) {
  if (basis.empty())
    return 0;
  int64_t res = 1;
  for (auto b : basis)
    res *= b;
  return res;
}

/// Computes and returns the linearized index of 'offsets' w.r.t. 'basis'.
static int64_t linearize(ArrayRef<int64_t> offsets, ArrayRef<int64_t> basis) {
  assert(offsets.size() == basis.size());
  int64_t linearIndex = 0;
  for (unsigned idx = 0, e = basis.size(); idx < e; ++idx)
    linearIndex += offsets[idx] * basis[idx];
  return linearIndex;
}

// Clones `op` into a new operations that takes `operands` and returns
// `resultTypes`.
static Operation *cloneOpWithOperandsAndTypes(PatternRewriter &builder,
                                              Location loc, Operation *op,
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
// TODO(andydavis) Move this to a utility function and share it with
// Extract/InsertSlicesOp verification.
static TupleType generateExtractSlicesOpResultType(VectorType vectorType,
                                                   ArrayRef<int64_t> sizes,
                                                   ArrayRef<int64_t> strides,
                                                   PatternRewriter &builder) {
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
                                    PatternRewriter &builder) {
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
    Value initValue, SmallVectorImpl<Value> &cache, PatternRewriter &builder) {
  // Compute slice offsets.
  SmallVector<int64_t, 4> sliceOffsets(state.unrolledShape.size());
  getMappedElements(indexMap, offsets, sliceOffsets);
  // TODO(b/144845578) Support non-1 strides.
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

// TODO(andydavis) Add the following canonicalization/simplifcation patterns:
// *) Add pattern which matches InsertStridedSlice -> StridedSlice and forwards
//    InsertStridedSlice operand to StridedSlice.
// *) Add pattern which matches SourceOp -> StridedSlice -> UserOp which checks
//    if there are duplicate identical StridedSlice ops from SourceOp, and
//    rewrites itself to use the first duplicate. This transformation should
//    cause users of identifical StridedSlice ops to reuse the same StridedSlice
//    operation, and leave the duplicate StridedSlice ops with no users
//    (removable with DCE).

// TODO(andydavis) Generalize this to support structured ops beyond
// vector ContractionOp, and merge it with 'unrollSingleResultOpMatchingType'
static Value unrollSingleResultStructuredOp(Operation *op,
                                            ArrayRef<int64_t> iterationBounds,
                                            std::vector<VectorState> &vectors,
                                            unsigned resultIndex,
                                            ArrayRef<int64_t> targetShape,
                                            PatternRewriter &builder) {
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
    SmallVectorImpl<int64_t> &iterationBounds,
    std::vector<VectorState> &vectors, unsigned &resultIndex) {
  // Get contraction op iteration bounds.
  contractionOp.getIterationBounds(iterationBounds);
  assert(iterationBounds.size() == targetShape.size());
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
  // Unroll 'op' 'iterationBounds' to 'targetShape'.
  // TODO(andydavis) Use linalg style 'args_in'/'args_out' to partition
  // 'vectors' instead of 'resultIndex'.
  resultIndex = accOperandIndex;
}

static void
getVectorElementwiseOpUnrollState(Operation *op, ArrayRef<int64_t> targetShape,
                                  SmallVectorImpl<int64_t> &iterationBounds,
                                  std::vector<VectorState> &vectors,
                                  unsigned &resultIndex) {
  // Verify that operation and operands all have the same vector shape.
  auto resultType = op->getResult(0).getType().dyn_cast_or_null<VectorType>();
  assert(resultType && "Expected op with vector result type");
  auto resultShape = resultType.getShape();
  // Verify that all operands have the same vector type as result.
  assert(llvm::all_of(op->getOperandTypes(),
                      [=](Type type) { return type == resultType; }));
  // Populate 'iterationBounds' with 'resultShape' for elementwise operations.
  iterationBounds.assign(resultShape.begin(), resultShape.end());

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

// Entry point for unrolling declarative pattern rewrites.
SmallVector<Value, 1> mlir::vector::unrollSingleResultOpMatchingType(
    PatternRewriter &builder, Operation *op, ArrayRef<int64_t> targetShape) {
  assert(op->getNumResults() == 1 && "Expected single result operation");

  // Populate 'iterationBounds', 'vectors' and 'resultIndex' to unroll 'op'.
  SmallVector<int64_t, 6> iterationBounds;
  std::vector<VectorState> vectors;
  unsigned resultIndex;

  if (auto contractionOp = dyn_cast<vector::ContractionOp>(op)) {
    // Populate state for vector ContractionOp.
    getVectorContractionOpUnrollState(contractionOp, targetShape,
                                      iterationBounds, vectors, resultIndex);
  } else {
    // Populate state for vector elementwise op.
    getVectorElementwiseOpUnrollState(op, targetShape, iterationBounds, vectors,
                                      resultIndex);
  }

  // Unroll 'op' with 'iterationBounds' to 'targetShape'.
  return SmallVector<Value, 1>{unrollSingleResultStructuredOp(
      op, iterationBounds, vectors, resultIndex, targetShape, builder)};
}

/// Generates slices of 'vectorType' according to 'sizes' and 'strides, and
/// calls 'fn' with linear index and indices for each slice.
static void
generateTransferOpSlices(Type memrefElementType, VectorType vectorType,
                         TupleType tupleType, ArrayRef<int64_t> sizes,
                         ArrayRef<int64_t> strides, ArrayRef<Value> indices,
                         PatternRewriter &rewriter,
                         function_ref<void(unsigned, ArrayRef<Value>)> fn) {
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

  auto *ctx = rewriter.getContext();
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
        sliceIndices[j] = rewriter.create<AffineApplyOp>(
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

namespace {

// Splits vector TransferReadOp into smaller TransferReadOps based on slicing
// scheme of its unique ExtractSlicesOp user.
struct SplitTransferReadOp : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(vector::TransferReadOp xferReadOp,
                                     PatternRewriter &rewriter) const override {
    // TODO(andydavis, ntv) Support splitting TransferReadOp with non-identity
    // permutation maps. Repurpose code from MaterializeVectors transformation.
    if (!isIdentitySuffix(xferReadOp.permutation_map()))
      return matchFailure();
    // Return unless the unique 'xferReadOp' user is an ExtractSlicesOp.
    Value xferReadResult = xferReadOp.getResult();
    auto extractSlicesOp =
        dyn_cast<vector::ExtractSlicesOp>(*xferReadResult.getUsers().begin());
    if (!xferReadResult.hasOneUse() || !extractSlicesOp)
      return matchFailure();

    // Get 'sizes' and 'strides' parameters from ExtractSlicesOp user.
    auto sourceVectorType = extractSlicesOp.getSourceVectorType();
    auto resultTupleType = extractSlicesOp.getResultTupleType();
    SmallVector<int64_t, 4> sizes;
    extractSlicesOp.getSizes(sizes);
    SmallVector<int64_t, 4> strides;
    extractSlicesOp.getStrides(strides);
    assert(llvm::all_of(strides, [](int64_t s) { return s == 1; }));

    Location loc = xferReadOp.getLoc();
    auto memrefElementType =
        xferReadOp.memref().getType().cast<MemRefType>().getElementType();
    int64_t numSlices = resultTupleType.size();
    SmallVector<Value, 4> vectorTupleValues(numSlices);
    SmallVector<Value, 4> indices(xferReadOp.indices().begin(),
                                  xferReadOp.indices().end());
    auto createSlice = [&](unsigned index, ArrayRef<Value> sliceIndices) {
      // Get VectorType for slice 'i'.
      auto sliceVectorType = resultTupleType.getType(index);
      // Create split TransferReadOp for 'sliceUser'.
      vectorTupleValues[index] = rewriter.create<vector::TransferReadOp>(
          loc, sliceVectorType, xferReadOp.memref(), sliceIndices,
          xferReadOp.permutation_map(), xferReadOp.padding());
    };
    generateTransferOpSlices(memrefElementType, sourceVectorType,
                             resultTupleType, sizes, strides, indices, rewriter,
                             createSlice);

    // Create tuple of splice xfer read operations.
    Value tupleOp = rewriter.create<vector::TupleOp>(loc, resultTupleType,
                                                     vectorTupleValues);
    // Replace 'xferReadOp' with result 'insertSlicesResult'.
    rewriter.replaceOpWithNewOp<vector::InsertSlicesOp>(
        xferReadOp, sourceVectorType, tupleOp, extractSlicesOp.sizes(),
        extractSlicesOp.strides());
    return matchSuccess();
  }
};

// Splits vector TransferWriteOp into smaller TransferWriteOps for each source.
struct SplitTransferWriteOp : public OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern<vector::TransferWriteOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(vector::TransferWriteOp xferWriteOp,
                                     PatternRewriter &rewriter) const override {
    // TODO(andydavis, ntv) Support splitting TransferWriteOp with non-identity
    // permutation maps. Repurpose code from MaterializeVectors transformation.
    if (!isIdentitySuffix(xferWriteOp.permutation_map()))
      return matchFailure();
    // Return unless the 'xferWriteOp' 'vector' operand is an 'InsertSlicesOp'.
    auto *vectorDefOp = xferWriteOp.vector().getDefiningOp();
    auto insertSlicesOp = dyn_cast_or_null<vector::InsertSlicesOp>(vectorDefOp);
    if (!insertSlicesOp)
      return matchFailure();

    // Get TupleOp operand of 'insertSlicesOp'.
    auto tupleOp = dyn_cast_or_null<vector::TupleOp>(
        insertSlicesOp.vectors().getDefiningOp());
    if (!tupleOp)
      return matchFailure();

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
      rewriter.create<vector::TransferWriteOp>(
          loc, tupleOp.getOperand(index), xferWriteOp.memref(), sliceIndices,
          xferWriteOp.permutation_map());
    };
    generateTransferOpSlices(memrefElementType, resultVectorType,
                             sourceTupleType, sizes, strides, indices, rewriter,
                             createSlice);

    // Erase old 'xferWriteOp'.
    rewriter.eraseOp(xferWriteOp);
    return matchSuccess();
  }
};

/// Decomposes ShapeCastOp on tuple-of-vectors to multiple ShapeCastOps, each
/// on vector types.
struct ShapeCastOpDecomposer : public OpRewritePattern<vector::ShapeCastOp> {
  using OpRewritePattern<vector::ShapeCastOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(vector::ShapeCastOp shapeCastOp,
                                     PatternRewriter &rewriter) const override {
    // Check if 'shapeCastOp' has tuple source/result type.
    auto sourceTupleType =
        shapeCastOp.source().getType().dyn_cast_or_null<TupleType>();
    auto resultTupleType =
        shapeCastOp.result().getType().dyn_cast_or_null<TupleType>();
    if (!sourceTupleType || !resultTupleType)
      return matchFailure();
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
    return matchSuccess();
  }
};

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

  PatternMatchResult matchAndRewrite(vector::ShapeCastOp shapeCastOp,
                                     PatternRewriter &rewriter) const override {
    // Check if 'shapeCastOp' has vector source/result type.
    auto sourceVectorType =
        shapeCastOp.source().getType().dyn_cast_or_null<VectorType>();
    auto resultVectorType =
        shapeCastOp.result().getType().dyn_cast_or_null<VectorType>();
    if (!sourceVectorType || !resultVectorType)
      return matchFailure();

    // Check if shape cast op source operand is also a shape cast op.
    auto sourceShapeCastOp = dyn_cast_or_null<vector::ShapeCastOp>(
        shapeCastOp.source().getDefiningOp());
    if (!sourceShapeCastOp)
      return matchFailure();
    auto operandSourceVectorType =
        sourceShapeCastOp.source().getType().cast<VectorType>();
    auto operandResultVectorType =
        sourceShapeCastOp.result().getType().cast<VectorType>();

    // Check if shape cast operations invert each other.
    if (operandSourceVectorType != resultVectorType ||
        operandResultVectorType != sourceVectorType)
      return matchFailure();

    rewriter.replaceOp(shapeCastOp, sourceShapeCastOp.source());
    return matchSuccess();
  }
};

// Patter rewrite which forward tuple elements to their users.
// User(TupleGetOp(ExtractSlicesOp(InsertSlicesOp(TupleOp(Producer)))))
//   -> User(Producer)
struct TupleGetFolderOp : public OpRewritePattern<vector::TupleGetOp> {
  using OpRewritePattern<vector::TupleGetOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(vector::TupleGetOp tupleGetOp,
                                     PatternRewriter &rewriter) const override {
    // Return if 'tupleGetOp.vectors' arg was not defined by ExtractSlicesOp.
    auto extractSlicesOp = dyn_cast_or_null<vector::ExtractSlicesOp>(
        tupleGetOp.vectors().getDefiningOp());
    if (!extractSlicesOp)
      return matchFailure();

    // Return if 'extractSlicesOp.vector' arg was not defined by InsertSlicesOp.
    auto insertSlicesOp = dyn_cast_or_null<vector::InsertSlicesOp>(
        extractSlicesOp.vector().getDefiningOp());
    if (!insertSlicesOp)
      return matchFailure();

    // Return if 'insertSlicesOp.vectors' arg was not defined by TupleOp.
    auto tupleOp = dyn_cast_or_null<vector::TupleOp>(
        insertSlicesOp.vectors().getDefiningOp());
    if (!tupleOp)
      return matchFailure();

    // Forward Value from 'tupleOp' at 'tupleGetOp.index'.
    Value tupleValue = tupleOp.getOperand(tupleGetOp.getIndex());
    rewriter.replaceOp(tupleGetOp, tupleValue);
    return matchSuccess();
  }
};

/// Progressive lowering of ExtractSlicesOp to tuple of StridedSliceOp.
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

  PatternMatchResult matchAndRewrite(vector::ExtractSlicesOp op,
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
      tupleValues[i] = rewriter.create<vector::StridedSliceOp>(
          loc, op.vector(), elementOffsets, sliceSizes, strides);
    }

    rewriter.replaceOpWithNewOp<vector::TupleOp>(op, tupleType, tupleValues);
    return matchSuccess();
  }
};

/// Progressive lowering of InsertSlicesOp to series of InsertStridedSliceOp.
/// One:
///   %x = vector.insert_slices %0
/// is replaced by:
///   %r0 = vector.splat 0
//    %t1 = vector.tuple_get %0, 0
///   %r1 = vector.insert_strided_slice %r0, %t1
//    %t2 = vector.tuple_get %0, 1
///   %r2 = vector.insert_strided_slice %r1, %t2
///   ..
///   %x  = ..
class InsertSlicesOpLowering : public OpRewritePattern<vector::InsertSlicesOp> {
public:
  using OpRewritePattern<vector::InsertSlicesOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(vector::InsertSlicesOp op,
                                     PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    VectorType vectorType = op.getResultVectorType();
    auto shape = vectorType.getShape();

    SmallVector<int64_t, 4> sizes;
    op.getSizes(sizes);
    SmallVector<int64_t, 4> strides;
    op.getStrides(strides); // all-ones at the moment

    // Prepare result.
    auto elemType = vectorType.getElementType();
    Value zero = rewriter.create<ConstantOp>(loc, elemType,
                                             rewriter.getZeroAttr(elemType));
    Value result = rewriter.create<SplatOp>(loc, vectorType, zero);

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
    return matchSuccess();
  }
};

/// Progressive lowering of OuterProductOp.
/// One:
///   %x = vector.outerproduct %lhs, %rhs, %acc
/// is replaced by:
///   %z = zero-result
///   %0 = vector.extract %lhs[0]
///   %1 = vector.broadcast %0
///   %2 = vector.extract %acc[0]
///   %3 = vector.fma %1, %arg1, %2
///   %4 = vector.insert %3, %z[0]
///   ..
///   %x = vector.insert %.., %..[N-1]
///
class OuterProductOpLowering : public OpRewritePattern<vector::OuterProductOp> {
public:
  using OpRewritePattern<vector::OuterProductOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(vector::OuterProductOp op,
                                     PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    VectorType rhsType = op.getOperandVectorTypeRHS();
    VectorType resType = op.getVectorType();
    Type eltType = resType.getElementType();
    Value acc = (op.acc().empty()) ? nullptr : op.acc()[0];

    Value zero = rewriter.create<ConstantOp>(loc, eltType,
                                             rewriter.getZeroAttr(eltType));
    Value result = rewriter.create<SplatOp>(loc, resType, zero);
    for (int64_t d = 0, e = resType.getDimSize(0); d < e; ++d) {
      auto pos = rewriter.getI64ArrayAttr(d);
      Value x = rewriter.create<vector::ExtractOp>(loc, eltType, op.lhs(), pos);
      Value b = rewriter.create<vector::BroadcastOp>(loc, rhsType, x);
      Value m;
      if (acc) {
        Value z = rewriter.create<vector::ExtractOp>(loc, rhsType, acc, pos);
        m = rewriter.create<vector::FMAOp>(loc, b, op.rhs(), z);
      } else {
        m = rewriter.create<MulFOp>(loc, b, op.rhs());
      }
      result = rewriter.create<vector::InsertOp>(loc, resType, m, result, pos);
    }
    rewriter.replaceOp(op, result);
    return matchSuccess();
  }
};

/// Progressive lowering of ContractionOp.
/// One:
///   %x = vector.contract with at least one free/batch dimension
/// is replaced by:
///   %a = vector.contract with one less free/batch dimension
///   %b = vector.contract with one less free/batch dimension
///   ..
///   %x = combine %a %b ..
/// until a pure contraction is reached (no free/batch dimensions),
/// which is replaced by a fma/reduction op.
///
/// TODO(ajcbik): break down into transpose/reshape/cast ops
///               when they become available to avoid code dup
/// TODO(ajcbik): investigate lowering order impact on performance
class ContractionOpLowering : public OpRewritePattern<vector::ContractionOp> {
public:
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(vector::ContractionOp op,
                                     PatternRewriter &rewriter) const override {
    // TODO(ajcbik): implement masks
    if (llvm::size(op.masks()) != 0)
      return matchFailure();

    // TODO(ntv, ajcbik): implement benefits, cost models, separate this out in
    // a new pattern.
    // TODO(ntv, fhahn): once row-major mode is available in LLVM's matrix
    // intrinsics, use that.
    if (lowerToLLVMMatrixIntrinsics &&
        isColumnMajorMatmul(op.indexing_maps())) {
      VectorType lhsType = op.getLhsType();
      VectorType rhsType = op.getRhsType();
      Type flattenedLHSType =
          VectorType::get(lhsType.getNumElements(), lhsType.getElementType());
      Type flattenedRHSType =
          VectorType::get(rhsType.getNumElements(), rhsType.getElementType());
      auto lhs = rewriter.create<vector::ShapeCastOp>(
          op.getLoc(), flattenedLHSType, op.lhs());
      auto rhs = rewriter.create<vector::ShapeCastOp>(
          op.getLoc(), flattenedRHSType, op.rhs());

      unsigned lhsRows = op.getLhsType().getShape()[0];
      unsigned lhsColumns = op.getLhsType().getShape()[1];
      unsigned rhsColumns = op.getRhsType().getShape()[1];
      Value mul = rewriter.create<vector::MatmulOp>(
          op.getLoc(), lhs, rhs, lhsRows, lhsColumns, rhsColumns);
      mul = rewriter.create<vector::ShapeCastOp>(op.getLoc(),
                                                 op.acc().getType(), mul);
      Type elementType = op.getLhsType().getElementType();
      assert(elementType.isIntOrFloat());
      if (elementType.isa<IntegerType>())
        rewriter.replaceOpWithNewOp<AddIOp>(op, op.acc(), mul);
      else
        rewriter.replaceOpWithNewOp<AddFOp>(op, op.acc(), mul);
      return matchSuccess();
    }

    // Find first batch dimension in LHS/RHS, and lower when found.
    std::vector<std::pair<int64_t, int64_t>> batchDimMap = op.getBatchDimMap();
    if (!batchDimMap.empty()) {
      int64_t lhsIndex = batchDimMap[0].first;
      int64_t rhsIndex = batchDimMap[0].second;
      rewriter.replaceOp(op, lowerParallel(op, lhsIndex, rhsIndex, rewriter));
      return matchSuccess();
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
    for (int64_t lhsIndex = 0, e = lhsType.getRank(); lhsIndex < e;
         ++lhsIndex) {
      if (lhsContractingDimSet.count(lhsIndex) == 0) {
        rewriter.replaceOp(
            op, lowerParallel(op, lhsIndex, /*rhsIndex=*/-1, rewriter));
        return matchSuccess();
      }
    }

    // Find first free dimension in RHS, and lower when found.
    VectorType rhsType = op.getRhsType();
    for (int64_t rhsIndex = 0, e = rhsType.getRank(); rhsIndex < e;
         ++rhsIndex) {
      if (rhsContractingDimSet.count(rhsIndex) == 0) {
        rewriter.replaceOp(
            op, lowerParallel(op, /*lhsIndex=*/-1, rhsIndex, rewriter));
        return matchSuccess();
      }
    }

    // Lower the first remaining reduction dimension.
    if (!contractingDimMap.empty()) {
      rewriter.replaceOp(op, lowerReduction(op, rewriter));
      return matchSuccess();
    }

    return matchFailure();
  }

private:
  // Lower one parallel dimension.
  // TODO(ajcbik): consider reusing existing contract unrolling
  Value lowerParallel(vector::ContractionOp op, int64_t lhsIndex,
                      int64_t rhsIndex, PatternRewriter &rewriter) const {
    VectorType lhsType = op.getLhsType();
    VectorType rhsType = op.getRhsType();
    VectorType resType = op.getResultType().cast<VectorType>();
    // Find the iterator type index and result index.
    SmallVector<AffineMap, 4> iMap = op.getIndexingMaps();
    int64_t iterIndex = -1;
    int64_t dimSize = -1;
    if (lhsIndex >= 0) {
      iterIndex =
          iMap[0].getResult(lhsIndex).cast<AffineDimExpr>().getPosition();
      assert((rhsIndex < 0 || iterIndex == iMap[1]
                                               .getResult(rhsIndex)
                                               .cast<AffineDimExpr>()
                                               .getPosition()) &&
             "parallel index should be free in LHS or batch in LHS/RHS");
      dimSize = lhsType.getDimSize(lhsIndex);
    } else {
      assert(rhsIndex >= 0 && "missing parallel index");
      iterIndex =
          iMap[1].getResult(rhsIndex).cast<AffineDimExpr>().getPosition();
      dimSize = rhsType.getDimSize(rhsIndex);
    }
    assert(iterIndex >= 0 && "parallel index not listed in operand mapping");
    Optional<int64_t> lookup = getResultIndex(iMap[2], iterIndex);
    assert(lookup.hasValue() && "parallel index not listed in reduction");
    int64_t resIndex = lookup.getValue();
    // Construct new iterator types and affine map array attribute.
    SmallVector<AffineMap, 4> lowIndexingMaps;
    lowIndexingMaps.push_back(adjustMap(iMap[0], iterIndex, rewriter));
    lowIndexingMaps.push_back(adjustMap(iMap[1], iterIndex, rewriter));
    lowIndexingMaps.push_back(adjustMap(iMap[2], iterIndex, rewriter));
    auto lowAffine = rewriter.getAffineMapArrayAttr(lowIndexingMaps);
    auto lowIter =
        rewriter.getArrayAttr(adjustIter(op.iterator_types(), iterIndex));
    // Unroll into a series of lower dimensional vector.contract ops.
    Location loc = op.getLoc();
    Value result = zeroVector(loc, resType, rewriter);
    for (int64_t d = 0; d < dimSize; ++d) {
      auto lhs = reshapeLoad(loc, op.lhs(), lhsType, lhsIndex, d, rewriter);
      auto rhs = reshapeLoad(loc, op.rhs(), rhsType, rhsIndex, d, rewriter);
      auto acc = reshapeLoad(loc, op.acc(), resType, resIndex, d, rewriter);
      Value lowContract = rewriter.create<vector::ContractionOp>(
          loc, lhs, rhs, acc, lowAffine, lowIter);
      result = reshapeStore(loc, lowContract, result, resType, resIndex, d,
                            rewriter);
    }
    return result;
  }

  // Lower one reduction dimension.
  Value lowerReduction(vector::ContractionOp op,
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
      Value zero = zeroVector(loc, lhsType, rewriter);
      Value fma = rewriter.create<vector::FMAOp>(loc, op.lhs(), op.rhs(), zero);
      StringAttr kind = rewriter.getStringAttr("add");
      return rewriter.create<vector::ReductionOp>(loc, resType, kind, fma,
                                                  op.acc());
    }
    // Construct new iterator types and affine map array attribute.
    SmallVector<AffineMap, 4> lowIndexingMaps;
    lowIndexingMaps.push_back(adjustMap(iMap[0], iterIndex, rewriter));
    lowIndexingMaps.push_back(adjustMap(iMap[1], iterIndex, rewriter));
    lowIndexingMaps.push_back(adjustMap(iMap[2], iterIndex, rewriter));
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

  // Helper method to construct a zero vector.
  static Value zeroVector(Location loc, VectorType vType,
                          PatternRewriter &rewriter) {
    Type eltType = vType.getElementType();
    Value zero = rewriter.create<ConstantOp>(loc, eltType,
                                             rewriter.getZeroAttr(eltType));
    return rewriter.create<SplatOp>(loc, vType, zero);
  }

  // Helper to find an index in an affine map.
  static Optional<int64_t> getResultIndex(AffineMap map, int64_t index) {
    for (int64_t i = 0, e = map.getNumResults(); i < e; ++i) {
      int64_t idx = map.getResult(i).cast<AffineDimExpr>().getPosition();
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
      int64_t idx = map.getResult(i).cast<AffineDimExpr>().getPosition();
      if (idx == index)
        continue;
      // Re-insert remaining indices, but renamed when occurring
      // after the removed index.
      auto targetExpr = getAffineDimExpr(idx < index ? idx : idx - 1, ctx);
      results.push_back(targetExpr);
    }
    // The (...) -> () affine map has its own factory method.
    return results.empty() ? AffineMap::get(map.getNumDims() - 1, 0, ctx)
                           : AffineMap::get(map.getNumDims() - 1, 0, results);
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
  // TODO(ajcbik): use a reshaping vector load (and share lowering code)
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
    Value result = zeroVector(loc, resType, rewriter);
    for (int64_t d = 0, e = resType.getDimSize(0); d < e; d++) {
      auto posAttr = rewriter.getI64ArrayAttr(d);
      Value ext = rewriter.create<vector::ExtractOp>(loc, vType, val, posAttr);
      Value load = reshapeLoad(loc, ext, vType, index - 1, pos, rewriter);
      result = rewriter.create<vector::InsertOp>(loc, resType, load, result,
                                                 posAttr);
    }
    return result;
  }

  // Helper method to possibly drop a dimension in a store.
  // TODO(ajcbik): use a reshaping vector store (and share lowering code)
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
      Value ext =
          rewriter.create<vector::ExtractOp>(loc, vType, result, posAttr);
      Value ins =
          rewriter.create<vector::ExtractOp>(loc, insType, val, posAttr);
      Value sto = reshapeStore(loc, ins, ext, vType, index - 1, pos, rewriter);
      result =
          rewriter.create<vector::InsertOp>(loc, type, sto, result, posAttr);
    }
    return result;
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

  PatternMatchResult matchAndRewrite(vector::ShapeCastOp op,
                                     PatternRewriter &rewriter) const override {
    auto sourceVectorType = op.getSourceVectorType();
    auto resultVectorType = op.getResultVectorType();
    if (sourceVectorType.getRank() != 2 || resultVectorType.getRank() != 1)
      return matchFailure();

    auto loc = op.getLoc();
    auto elemType = sourceVectorType.getElementType();
    Value zero = rewriter.create<ConstantOp>(loc, elemType,
                                             rewriter.getZeroAttr(elemType));
    Value desc = rewriter.create<SplatOp>(loc, resultVectorType, zero);
    unsigned mostMinorVectorSize = sourceVectorType.getShape()[1];
    for (int64_t i = 0, e = sourceVectorType.getShape().front(); i != e; ++i) {
      Value vec = rewriter.create<vector::ExtractOp>(loc, op.source(), i);
      desc = rewriter.create<vector::InsertStridedSliceOp>(
          loc, vec, desc,
          /*offsets=*/i * mostMinorVectorSize, /*strides=*/1);
    }
    rewriter.replaceOp(op, desc);
    return matchSuccess();
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

  PatternMatchResult matchAndRewrite(vector::ShapeCastOp op,
                                     PatternRewriter &rewriter) const override {
    auto sourceVectorType = op.getSourceVectorType();
    auto resultVectorType = op.getResultVectorType();
    if (sourceVectorType.getRank() != 1 || resultVectorType.getRank() != 2)
      return matchFailure();

    auto loc = op.getLoc();
    auto elemType = sourceVectorType.getElementType();
    Value zero = rewriter.create<ConstantOp>(loc, elemType,
                                             rewriter.getZeroAttr(elemType));
    Value desc = rewriter.create<SplatOp>(loc, resultVectorType, zero);
    unsigned mostMinorVectorSize = resultVectorType.getShape()[1];
    for (int64_t i = 0, e = resultVectorType.getShape().front(); i != e; ++i) {
      Value vec = rewriter.create<vector::StridedSliceOp>(
          loc, op.source(), /*offsets=*/i * mostMinorVectorSize,
          /*sizes=*/mostMinorVectorSize,
          /*strides=*/1);
      desc = rewriter.create<vector::InsertOp>(loc, vec, desc, i);
    }
    rewriter.replaceOp(op, desc);
    return matchSuccess();
  }
};

} // namespace

// TODO(andydavis) Add pattern to rewrite ExtractSlices(ConstantMaskOp).
// TODO(andydavis) Add this as DRR pattern.
void mlir::vector::populateVectorToVectorTransformationPatterns(
    OwningRewritePatternList &patterns, MLIRContext *context) {
  patterns.insert<ShapeCastOpDecomposer, ShapeCastOpFolder, SplitTransferReadOp,
                  SplitTransferWriteOp, TupleGetFolderOp>(context);
}

void mlir::vector::populateVectorSlicesLoweringPatterns(
    OwningRewritePatternList &patterns, MLIRContext *context) {
  patterns.insert<ExtractSlicesOpLowering, InsertSlicesOpLowering>(context);
}

void mlir::vector::populateVectorContractLoweringPatterns(
    OwningRewritePatternList &patterns, MLIRContext *context) {
  patterns.insert<ContractionOpLowering, ShapeCastOp2DDownCastRewritePattern,
                  ShapeCastOp2DUpCastRewritePattern, OuterProductOpLowering>(
      context);
}
