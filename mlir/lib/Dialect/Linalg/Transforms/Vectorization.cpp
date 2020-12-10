//===- Vectorization.cpp - Implementation of linalg Vectorization ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the linalg dialect Vectorization transformations.
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

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using namespace mlir::linalg;

using llvm::dbgs;

#define DEBUG_TYPE "linalg-vectorization"

static bool hasMultiplyAddBody(Region &r) {
  if (!llvm::hasSingleElement(r))
    return false;
  if (!llvm::hasNItems(r.front().begin(), r.front().end(), 3))
    return false;

  using mlir::matchers::m_Val;
  auto a = m_Val(r.getArgument(0));
  auto b = m_Val(r.getArgument(1));
  auto c = m_Val(r.getArgument(2));
  // TODO: Update this detection once we have  matcher support for specifying
  // that any permutation of operands matches.
  auto pattern1 = m_Op<linalg::YieldOp>(m_Op<AddFOp>(m_Op<MulFOp>(a, b), c));
  auto pattern2 = m_Op<linalg::YieldOp>(m_Op<AddFOp>(c, m_Op<MulFOp>(a, b)));
  auto pattern3 = m_Op<linalg::YieldOp>(m_Op<AddFOp>(m_Op<MulFOp>(b, a), c));
  auto pattern4 = m_Op<linalg::YieldOp>(m_Op<AddFOp>(c, m_Op<MulFOp>(b, a)));
  auto pattern5 = m_Op<linalg::YieldOp>(m_Op<AddIOp>(m_Op<MulIOp>(a, b), c));
  auto pattern6 = m_Op<linalg::YieldOp>(m_Op<AddIOp>(c, m_Op<MulIOp>(a, b)));
  auto pattern7 = m_Op<linalg::YieldOp>(m_Op<AddIOp>(m_Op<MulIOp>(b, a), c));
  auto pattern8 = m_Op<linalg::YieldOp>(m_Op<AddIOp>(c, m_Op<MulIOp>(b, a)));
  return pattern1.match(&r.front().back()) ||
         pattern2.match(&r.front().back()) ||
         pattern3.match(&r.front().back()) ||
         pattern4.match(&r.front().back()) ||
         pattern5.match(&r.front().back()) ||
         pattern6.match(&r.front().back()) ||
         pattern7.match(&r.front().back()) || pattern8.match(&r.front().back());
}

// TODO: Should be Tablegen'd from a single source that generates the op itself.
static LogicalResult isContraction(Operation *op) {
  // TODO: interface for named ops.
  if (isa<linalg::BatchMatmulOp, linalg::MatmulOp, linalg::MatvecOp,
          linalg::VecmatOp, linalg::DotOp>(op))
    return success();

  auto genericOp = dyn_cast<linalg::GenericOp>(op);
  if (!genericOp)
    return failure();

  auto mapRange = genericOp.indexing_maps().getAsValueRange<AffineMapAttr>();
  return success(
      genericOp.getNumInputs() == 2 && genericOp.getNumOutputs() == 1 &&
      llvm::all_of(mapRange,
                   [](AffineMap m) { return m.isProjectedPermutation(); }) &&
      hasMultiplyAddBody(genericOp.region()));
}

static bool hasOnlyScalarElementwiseOp(Region &r) {
  if (!llvm::hasSingleElement(r))
    return false;
  for (Operation &op : r.front()) {
    if (!(isa<ConstantOp, linalg::YieldOp>(op) ||
          op.hasTrait<OpTrait::ElementwiseMappable>()) ||
        llvm::any_of(op.getResultTypes(),
                     [](Type type) { return !type.isIntOrIndexOrFloat(); }))
      return false;
  }
  return true;
}

// Return true if the op is an element-wise linalg op.
static bool isElementwise(Operation *op) {
  auto genericOp = dyn_cast<linalg::GenericOp>(op);
  if (!genericOp)
    return false;
  if (genericOp.getNumLoops() != genericOp.getNumParallelLoops())
    return false;
  // TODO: relax the restrictions on indexing map.
  for (unsigned i = 0, e = genericOp.getNumOutputs(); i < e; i++) {
    if (!genericOp.getOutputIndexingMap(i).isIdentity())
      return false;
  }
  // Currently limit the input indexing map to minor identity as other
  // permutations might require adding transpose ops to convert the vector read
  // to the right shape.
  for (unsigned i = 0, e = genericOp.getNumInputs(); i < e; i++) {
    if (!genericOp.getInputIndexingMap(i).isMinorIdentity())
      return false;
  }
  return hasOnlyScalarElementwiseOp(genericOp.getRegion());
}

static VectorType extractVectorTypeFromScalarView(Value v) {
  MemRefType mt = v.getType().cast<MemRefType>();
  return mt.getShape().empty()
             ? VectorType()
             : VectorType::get(mt.getShape(), mt.getElementType());
}

static Value transferReadVector(OpBuilder &builder, Value memref) {
  edsc::ScopedContext scope(builder);
  auto memrefType = memref.getType().cast<MemRefType>();
  if (VectorType vectorType = extractVectorTypeFromScalarView(memref)) {
    SmallVector<Value, 4> indices(memrefType.getRank(), std_constant_index(0));
    return vector_transfer_read(vectorType, memref, indices);
  }
  return std_load(memref);
}

static void transferWriteVector(OpBuilder &builder, Value value, Value memref) {
  edsc::ScopedContext scope(builder);
  auto memrefType = memref.getType().cast<MemRefType>();
  if (VectorType vectorType = extractVectorTypeFromScalarView(memref)) {
    SmallVector<Value, 4> indices(memrefType.getRank(), std_constant_index(0));
    if (vectorType != value.getType())
      value = vector_broadcast(vectorType, value);
    vector_transfer_write(value, memref, indices);
  } else {
    std_store(value, memref);
  }
}

namespace {
// Transforms scalar operations into their vectorized counterparts,
// while using the provided generic op to map:
//   * Its arguments to transfer reads from the views of the generic op.
//   * linalg.yield ops to transfer writes to the views of the generic op.
class GenericVectorizer {
public:
  GenericVectorizer(OpBuilder &builder, linalg::GenericOp generic)
      : builder(builder), generic(generic) {}

  // Takes a scalar operation and builds its vectorized counterpart or
  // counterparts using the underlying builder.
  // If operands of the scalar operation are referring to previously vectorized
  // operations, then in their vectorized form these operands will be referring
  // to previous vectorization results.
  void vectorize(Operation &scalarOp) {
    auto yieldOp = dyn_cast<linalg::YieldOp>(scalarOp);
    if (yieldOp) {
      for (auto outputAndMemref :
           llvm::zip(yieldOp.values(), generic.getOutputBuffers())) {
        Value vectorValue = vectorize(std::get<0>(outputAndMemref));
        transferWriteVector(builder, vectorValue, std::get<1>(outputAndMemref));
      }
      return;
    }
    Operation *vectorOp = uncachedVectorize(scalarOp);
    assert(scalarOp.getNumResults() == vectorOp->getNumResults());
    for (auto result :
         llvm::zip(scalarOp.getResults(), vectorOp->getResults())) {
      valueCache[std::get<0>(result)] = std::get<1>(result);
    }
  }

private:
  // Transforms a scalar value into its vectorized counterpart, recursively
  // vectorizing operations as necessary using the underlying builder.
  // Keeps track of previously vectorized values and reuses vectorization
  // results if these values come up again.
  Value vectorize(Value scalarValue) {
    // Don't vectorize values coming from outside the region.
    if (scalarValue.getParentRegion() != &generic.region())
      return scalarValue;
    auto vectorValueIt = valueCache.find(scalarValue);
    if (vectorValueIt != valueCache.end())
      return vectorValueIt->second;

    // If the value is from the region but not in the cache it means it is a
    // block argument.
    auto scalarArg = scalarValue.cast<BlockArgument>();
    assert(scalarArg.getOwner() == &generic.region().front());
    Value vector_arg =
        generic.getInputsAndOutputBuffers()[scalarArg.getArgNumber()];
    Value vectorResult = transferReadVector(builder, vector_arg);
    valueCache[scalarArg] = vectorResult;
    return vectorResult;
  }

  // Return the largest shape of all the given values. Return an empty
  // SmallVector if there are no vector value.
  static SmallVector<int64_t, 4> getLargestShape(ArrayRef<Value> values) {
    SmallVector<int64_t, 4> largestShape;
    int64_t maxSize = 1;
    for (Value value : values) {
      auto vecType = value.getType().dyn_cast<VectorType>();
      if (!vecType)
        continue;
      if (maxSize < vecType.getNumElements()) {
        largestShape.assign(vecType.getShape().begin(),
                            vecType.getShape().end());
      }
    }
    return largestShape;
  }

  // If the value's type doesn't have the given shape broadcast it.
  Value broadcastIfNeeded(Value value, ArrayRef<int64_t> shape) {
    auto vecType = value.getType().dyn_cast<VectorType>();
    if (shape.empty() || (vecType != nullptr && vecType.getShape() == shape))
      return value;
    auto newVecType = VectorType::get(shape, vecType ? vecType.getElementType()
                                                     : value.getType());
    return builder.create<vector::BroadcastOp>(
        builder.getInsertionPoint()->getLoc(), newVecType, value);
  }

  // Takes a scalar operation and builds its vectorized counterpart or
  // counterparts using underlying builder without involving any caches.
  Operation *uncachedVectorize(Operation &base_scalarOp) {
    SmallVector<Value, 4> vectorizedOperands;
    for (Value operand : base_scalarOp.getOperands()) {
      vectorizedOperands.push_back(vectorize(operand));
    }
    SmallVector<int64_t, 4> shape = getLargestShape(vectorizedOperands);
    for (Value &operand : vectorizedOperands)
      operand = broadcastIfNeeded(operand, shape);
    OperationState state(base_scalarOp.getLoc(), base_scalarOp.getName());
    state.addAttributes(base_scalarOp.getAttrs());
    state.addOperands(vectorizedOperands);
    if (shape.empty()) {
      state.addTypes(base_scalarOp.getResultTypes());
    } else {
      SmallVector<VectorType, 4> vectorizedTypes;
      for (auto Type : base_scalarOp.getResultTypes())
        vectorizedTypes.push_back(VectorType::get(shape, Type));
      state.addTypes(vectorizedTypes);
    }
    return builder.createOperation(state);
  }

  OpBuilder &builder;
  linalg::GenericOp generic;
  llvm::DenseMap<Value, Value> valueCache;
};
} // namespace

// Replaces elementwise linalg.generic ops with their bodies with scalar
// operations from these bodies promoted to vector operations.
static void vectorizeElementwise(linalg::GenericOp op, OpBuilder &builder) {
  GenericVectorizer vectorizer(builder, op);
  for (Operation &scalarOp : op.region().front()) {
    vectorizer.vectorize(scalarOp);
  }
}

LogicalResult mlir::linalg::vectorizeLinalgOpPrecondition(Operation *op) {
  auto linalgOp = cast<linalg::LinalgOp>(op);
  // All types must be static shape to go to vector.
  for (Value operand : linalgOp.getInputsAndOutputBuffers())
    if (!operand.getType().cast<ShapedType>().hasStaticShape())
      return failure();
  for (Type outputTensorType : linalgOp.getOutputTensorTypes())
    if (!outputTensorType.cast<ShapedType>().hasStaticShape())
      return failure();

  if (isa<linalg::FillOp, linalg::CopyOp>(op))
    return success();
  if (isElementwise(op))
    return success();
  return isContraction(op);
}

void mlir::linalg::vectorizeLinalgOp(OpBuilder &builder, Operation *op) {
  assert(succeeded(vectorizeLinalgOpPrecondition(op)));

  StringRef dbgPref = "\n[" DEBUG_TYPE "]: ";
  (void)dbgPref;
  edsc::ScopedContext scope(builder, op->getLoc());
  // In the case of 0-D memrefs, return null and special case to scalar load or
  // store later.
  if (auto fillOp = dyn_cast<linalg::FillOp>(op)) {
    // Vectorize fill as a vector.broadcast.
    LLVM_DEBUG(dbgs() << dbgPref
                      << "Rewrite linalg.fill as vector.broadcast: " << *op);
    transferWriteVector(builder, fillOp.value(), fillOp.output());
    return;
  }
  if (auto copyOp = dyn_cast<linalg::CopyOp>(op)) {
    // Vectorize copy as a vector.transfer_read+vector.transfer_write.
    LLVM_DEBUG(dbgs() << dbgPref
                      << "Rewrite linalg.copy as vector.transfer_read + "
                         "vector.transfer_write: "
                      << *op);
    Value vector = transferReadVector(builder, copyOp.input());
    transferWriteVector(builder, vector, copyOp.output());
    return;
  }

  if (isElementwise(op)) {
    LLVM_DEBUG(dbgs() << dbgPref
                      << "Rewrite linalg op as vector.transfer_read + "
                         "vector_op + vector.transfer_write: "
                      << *op);
    return vectorizeElementwise(cast<linalg::GenericOp>(op), builder);
  }

  assert(succeeded(isContraction(op)) && "Expected contraction");

  // Vectorize other ops as vector contraction.
  // TODO: interface.
  LLVM_DEBUG(dbgs() << dbgPref
                    << "Rewrite linalg op as vector.contract: " << *op);
  auto linalgOp = cast<linalg::LinalgOp>(op);
  Value viewA = linalgOp.getInput(0);
  Value viewB = linalgOp.getInput(1);
  Value viewC = linalgOp.getOutputBuffer(0);
  VectorType vtA = extractVectorTypeFromScalarView(viewA);
  VectorType vtB = extractVectorTypeFromScalarView(viewB);
  VectorType vtC = extractVectorTypeFromScalarView(viewC);
  Value zero = std_constant_index(0);
  SmallVector<Value, 4> indicesA, indicesB, indicesC;
  if (vtA)
    indicesA = SmallVector<Value, 4>(vtA.getRank(), zero);
  if (vtB)
    indicesB = SmallVector<Value, 4>(vtB.getRank(), zero);
  if (vtC)
    indicesC = SmallVector<Value, 4>(vtC.getRank(), zero);
  Value a = vtA ? vector_transfer_read(vtA, viewA, indicesA).value
                : std_load(viewA, indicesA).value;
  Value b = vtB ? vector_transfer_read(vtB, viewB, indicesB).value
                : std_load(viewB, indicesB).value;
  Value c = vtC ? vector_transfer_read(vtC, viewC, indicesC).value
                : std_load(viewC, indicesC).value;
  Value res = vector_contract(a, b, c, linalgOp.indexing_maps(),
                              linalgOp.iterator_types());
  if (vtC)
    vector_transfer_write(res, viewC, indicesC);
  else
    std_store(res, viewC, indicesC);
}

/// Check whether there is any interleaved use of any `values` between `firstOp`
/// and `secondOp`. Conservatively return `true` if any op or value is in a
/// different block.
static bool mayExistInterleavedUses(Operation *firstOp, Operation *secondOp,
                                    ValueRange values) {
  StringRef dbgPref = "\n[" DEBUG_TYPE "]: ";
  (void)dbgPref;
  if (firstOp->getBlock() != secondOp->getBlock() ||
      !firstOp->isBeforeInBlock(secondOp)) {
    LLVM_DEBUG(llvm::dbgs()
               << dbgPref << "interleavedUses precondition failed, firstOp: "
               << *firstOp << ", second op: " << *secondOp);
    return true;
  }
  for (auto v : values) {
    for (auto &u : v.getUses()) {
      Operation *owner = u.getOwner();
      if (owner == firstOp || owner == secondOp)
        continue;
      // TODO: this is too conservative, use dominance info in the future.
      if (owner->getBlock() == firstOp->getBlock() &&
          (owner->isBeforeInBlock(firstOp) || secondOp->isBeforeInBlock(owner)))
        continue;
      LLVM_DEBUG(llvm::dbgs()
                 << dbgPref << " found interleaved op " << *owner
                 << ", firstOp: " << *firstOp << ", second op: " << *secondOp);
      return true;
    }
  }
  return false;
}

/// Return the unique subview use of `v` if it is indeed unique, null otherwise.
static SubViewOp getSubViewUseIfUnique(Value v) {
  SubViewOp subViewOp;
  for (auto &u : v.getUses()) {
    if (auto newSubViewOp = dyn_cast<SubViewOp>(u.getOwner())) {
      if (subViewOp)
        return SubViewOp();
      subViewOp = newSubViewOp;
    }
  }
  return subViewOp;
}

/// TODO: use interfaces, side-effects and aliasing analysis as appropriate,
/// when available.
LogicalResult LinalgCopyVTRForwardingPattern::matchAndRewrite(
    vector::TransferReadOp xferOp, PatternRewriter &rewriter) const {

  // Transfer into `view`.
  Value viewOrAlloc = xferOp.memref();
  if (!viewOrAlloc.getDefiningOp<ViewOp>() &&
      !viewOrAlloc.getDefiningOp<AllocOp>())
    return failure();

  StringRef dbgPref = "\n[" DEBUG_TYPE "]: VTRForwarding: ";
  (void)dbgPref;
  LLVM_DEBUG(llvm::dbgs() << dbgPref << viewOrAlloc);

  // Ensure there is exactly one subview of `viewOrAlloc` defining `subView`.
  SubViewOp subViewOp = getSubViewUseIfUnique(viewOrAlloc);
  if (!subViewOp)
    return failure();
  Value subView = subViewOp.getResult();
  LLVM_DEBUG(llvm::dbgs() << dbgPref << "with subView " << subView);

  // Find the copy into `subView` without interleaved uses.
  CopyOp copyOp;
  for (auto &u : subView.getUses()) {
    if (auto newCopyOp = dyn_cast<CopyOp>(u.getOwner())) {
      if (newCopyOp.getOutputBuffer(0) != subView)
        continue;
      LLVM_DEBUG(llvm::dbgs() << dbgPref << "copy candidate " << *newCopyOp);
      if (mayExistInterleavedUses(newCopyOp, xferOp, {viewOrAlloc, subView}))
        continue;
      copyOp = newCopyOp;
      break;
    }
  }
  if (!copyOp)
    return failure();
  LLVM_DEBUG(llvm::dbgs() << dbgPref << "with copy " << *copyOp);

  // Find the fill into `viewOrAlloc` without interleaved uses before the copy.
  FillOp maybeFillOp;
  for (auto &u : viewOrAlloc.getUses()) {
    if (auto newFillOp = dyn_cast<FillOp>(u.getOwner())) {
      if (newFillOp.getOutputBuffer(0) != viewOrAlloc)
        continue;
      LLVM_DEBUG(llvm::dbgs() << dbgPref << "fill candidate " << *newFillOp);
      if (mayExistInterleavedUses(newFillOp, copyOp, {viewOrAlloc, subView}))
        continue;
      maybeFillOp = newFillOp;
      break;
    }
  }
  // Ensure padding matches.
  if (maybeFillOp && xferOp.padding() != maybeFillOp.value())
    return failure();
  if (maybeFillOp)
    LLVM_DEBUG(llvm::dbgs() << dbgPref << "with maybeFillOp " << *maybeFillOp);

  // `in` is the subview that linalg.copy reads. Replace it.
  Value in = copyOp.getInput(0);

  // linalg.copy + linalg.fill can be used to create a padded local buffer.
  // The `masked` attribute is only valid on this padded buffer.
  // When forwarding to vector.transfer_read, the attribute must be reset
  // conservatively.
  Value res = rewriter.create<vector::TransferReadOp>(
      xferOp.getLoc(), xferOp.getVectorType(), in, xferOp.indices(),
      xferOp.permutation_map(), xferOp.padding(), ArrayAttr());

  if (maybeFillOp)
    rewriter.eraseOp(maybeFillOp);
  rewriter.eraseOp(copyOp);
  rewriter.replaceOp(xferOp, res);

  return success();
}

/// TODO: use interfaces, side-effects and aliasing analysis as appropriate,
/// when available.
LogicalResult LinalgCopyVTWForwardingPattern::matchAndRewrite(
    vector::TransferWriteOp xferOp, PatternRewriter &rewriter) const {
  // Transfer into `viewOrAlloc`.
  Value viewOrAlloc = xferOp.memref();
  if (!viewOrAlloc.getDefiningOp<ViewOp>() &&
      !viewOrAlloc.getDefiningOp<AllocOp>())
    return failure();

  // Ensure there is exactly one subview of `viewOrAlloc` defining `subView`.
  SubViewOp subViewOp = getSubViewUseIfUnique(viewOrAlloc);
  if (!subViewOp)
    return failure();
  Value subView = subViewOp.getResult();

  // Find the copy from `subView` without interleaved uses.
  CopyOp copyOp;
  for (auto &u : subViewOp.getResult().getUses()) {
    if (auto newCopyOp = dyn_cast<CopyOp>(u.getOwner())) {
      if (newCopyOp.getInput(0) != subView)
        continue;
      if (mayExistInterleavedUses(xferOp, newCopyOp, {viewOrAlloc, subView}))
        continue;
      copyOp = newCopyOp;
      break;
    }
  }
  if (!copyOp)
    return failure();

  // `out` is the subview copied into that we replace.
  Value out = copyOp.getOutputBuffer(0);

  // Forward vector.transfer into copy.
  // linalg.copy + linalg.fill can be used to create a padded local buffer.
  // The `masked` attribute is only valid on this padded buffer.
  // When forwarding to vector.transfer_write, the attribute must be reset
  // conservatively.
  rewriter.create<vector::TransferWriteOp>(
      xferOp.getLoc(), xferOp.vector(), out, xferOp.indices(),
      xferOp.permutation_map(), ArrayAttr());

  rewriter.eraseOp(copyOp);
  rewriter.eraseOp(xferOp);

  return success();
}

template <class ConvOp, int N>
LogicalResult ConvOpVectorization<ConvOp, N>::matchAndRewrite(
    ConvOp op, PatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  MLIRContext *context = op.getContext();
  edsc::ScopedContext scope(rewriter, loc);

  ShapedType inShapeType = op.getInputShapedType(0);
  ShapedType kShapeType = op.getInputShapedType(1);

  ArrayRef<int64_t> inShape = inShapeType.getShape();
  ArrayRef<int64_t> kShape = kShapeType.getShape();

  if (!inShapeType.hasStaticShape() || !kShapeType.hasStaticShape())
    return failure();

  SmallVector<AffineExpr, 4> mapping;
  SmallVector<int64_t, 4> vectorDims;
  // Fail to apply when the size of not vectorized dimension is not 1.
  for (unsigned i = 0; i < N; i++) {
    if (!mask[i] && (inShape[i] != 1 || kShape[i] != 1))
      return failure();

    if (mask[i] && inShape[i] != kShape[i])
      return failure();

    if (mask[i]) {
      mapping.push_back(getAffineDimExpr(i, context));
      vectorDims.push_back(inShape[i]);
    }
  }

  Value input = op.getInput(0);
  Value kernel = op.getInput(1);
  Value output = op.getOutputBuffer(0);

  unsigned rank = inShapeType.getRank();
  unsigned numDims = mapping.size();
  Type elemType = inShapeType.getElementType();

  auto map = AffineMap::get(rank, 0, mapping, context);
  SmallVector<Value, 4> zeros(rank, std_constant_index(0));
  auto vecType = VectorType::get(vectorDims, elemType);

  auto inputVec = vector_transfer_read(vecType, input, zeros, map);
  auto kernelVec = vector_transfer_read(vecType, kernel, zeros, map);

  auto acc = std_constant(elemType, rewriter.getZeroAttr(elemType));

  std::array<AffineMap, 3> indexingMaps{
      AffineMap::getMultiDimIdentityMap(numDims, context),
      AffineMap::getMultiDimIdentityMap(numDims, context),
      AffineMap::get(numDims, 0, {}, context)};

  std::vector<StringRef> iteratorTypes(numDims, "reduction");

  auto result = rewriter.create<vector::ContractionOp>(
      loc, inputVec, kernelVec, acc,
      rewriter.getAffineMapArrayAttr(indexingMaps),
      rewriter.getStrArrayAttr(iteratorTypes));

  rewriter.create<StoreOp>(loc, result, output, ValueRange(zeros));
  rewriter.eraseOp(op);
  return success();
}

using ConvOpConst = ConvOpVectorization<ConvWOp, 1>;

/// Inserts tiling, promotion and vectorization pattern for ConvOp
/// conversion into corresponding pattern lists.
template <typename ConvOp, unsigned N>
static void
populateVectorizationPatterns(OwningRewritePatternList &tilingPatterns,
                              OwningRewritePatternList &promotionPatterns,
                              OwningRewritePatternList &vectorizationPatterns,
                              ArrayRef<int64_t> tileSizes,
                              MLIRContext *context) {
  if (tileSizes.size() < N)
    return;

  constexpr static StringRef kTiledMarker = "TILED";
  constexpr static StringRef kPromotedMarker = "PROMOTED";
  tilingPatterns.insert<LinalgTilingPattern<ConvOp>>(
      context, LinalgTilingOptions().setTileSizes(tileSizes),
      LinalgMarker({}, Identifier::get(kTiledMarker, context)));

  promotionPatterns.insert<LinalgPromotionPattern<ConvOp>>(
      context, LinalgPromotionOptions().setUseFullTileBuffersByDefault(true),
      LinalgMarker(Identifier::get(kTiledMarker, context),
                   Identifier::get(kPromotedMarker, context)));

  SmallVector<bool, 4> mask(N);
  int offset = tileSizes.size() - N;
  std::transform(tileSizes.begin() + offset, tileSizes.end(), mask.begin(),
                 [](int64_t i) -> bool { return i > 1; });

  vectorizationPatterns.insert<ConvOpVectorization<ConvOp, N>>(context, mask);
}

void mlir::linalg::populateConvVectorizationPatterns(
    MLIRContext *context, SmallVectorImpl<OwningRewritePatternList> &patterns,
    ArrayRef<int64_t> tileSizes) {
  OwningRewritePatternList tiling, promotion, vectorization;
  populateVectorizationPatterns<ConvWOp, 1>(tiling, promotion, vectorization,
                                            tileSizes, context);

  populateVectorizationPatterns<ConvNWCOp, 3>(tiling, promotion, vectorization,
                                              tileSizes, context);

  populateVectorizationPatterns<ConvNCWOp, 3>(tiling, promotion, vectorization,
                                              tileSizes, context);

  populateVectorizationPatterns<ConvHWOp, 2>(tiling, promotion, vectorization,
                                             tileSizes, context);

  populateVectorizationPatterns<ConvNHWCOp, 4>(tiling, promotion, vectorization,
                                               tileSizes, context);

  populateVectorizationPatterns<ConvNCHWOp, 4>(tiling, promotion, vectorization,
                                               tileSizes, context);

  populateVectorizationPatterns<ConvDHWOp, 3>(tiling, promotion, vectorization,
                                              tileSizes, context);

  populateVectorizationPatterns<ConvNDHWCOp, 5>(
      tiling, promotion, vectorization, tileSizes, context);

  populateVectorizationPatterns<ConvNCDHWOp, 5>(
      tiling, promotion, vectorization, tileSizes, context);

  patterns.push_back(std::move(tiling));
  patterns.push_back(std::move(promotion));
  patterns.push_back(std::move(vectorization));
}
