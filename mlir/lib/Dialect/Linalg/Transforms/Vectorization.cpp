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
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <type_traits>

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using namespace mlir::linalg;

using llvm::dbgs;

#define DEBUG_TYPE "linalg-vectorization"

/// Return the unique instance of OpType in `block` if it is indeed unique.
/// Return null if none or more than 1 instances exist.
template <typename OpType>
static OpType getSingleOpOfType(Block &block) {
  OpType res;
  block.walk([&](OpType op) {
    if (res) {
      res = nullptr;
      return WalkResult::interrupt();
    }
    res = op;
    return WalkResult::advance();
  });
  return res;
}

/// Helper data structure to represent the result of vectorization.
/// In certain specific cases, like terminators, we do not want to propagate/
enum VectorizationStatus {
  /// Op failed to vectorize.
  Failure = 0,
  /// Op vectorized and custom function took care of replacement logic
  NoReplace,
  /// Op vectorized into a new Op whose results will replace original Op's
  /// results.
  NewOp
  // TODO: support values if Op vectorized to Many-Ops whose results we need to
  // aggregate for replacement.
};
struct VectorizationResult {
  /// Return status from vectorizing the current op.
  enum VectorizationStatus status = VectorizationStatus::Failure;
  /// New vectorized operation to replace the current op.
  /// Replacement behavior is specified by `status`.
  Operation *newOp;
};

/// Return a vector type of the same shape and element type as the (assumed)
/// ShapedType of `v`.
static VectorType extractVectorTypeFromShapedValue(Value v) {
  auto st = v.getType().cast<ShapedType>();
  if (st.isa<MemRefType>() && st.getShape().empty())
    return VectorType();
  return VectorType::get(st.getShape(), st.getElementType());
}

/// Build a vector.transfer_read from `source` at indices set to all `0`.
/// If source has rank zero, build an std.load.
/// Return the produced value.
static Value buildVectorRead(OpBuilder &builder, Value source) {
  edsc::ScopedContext scope(builder);
  auto shapedType = source.getType().cast<ShapedType>();
  if (VectorType vectorType = extractVectorTypeFromShapedValue(source)) {
    SmallVector<Value> indices(shapedType.getRank(), std_constant_index(0));
    return vector_transfer_read(vectorType, source, indices);
  }
  return std_load(source);
}

/// Build a vector.transfer_write of `value` into `dest` at indices set to all
/// `0`. If `dest` has null rank, build an std.store.
/// Return the produced value or null if no value is produced.
static Value buildVectorWrite(OpBuilder &builder, Value value, Value dest) {
  edsc::ScopedContext scope(builder);
  Operation *write;
  auto shapedType = dest.getType().cast<ShapedType>();
  if (VectorType vectorType = extractVectorTypeFromShapedValue(dest)) {
    SmallVector<Value> indices(shapedType.getRank(), std_constant_index(0));
    if (vectorType != value.getType())
      value = vector_broadcast(vectorType, value);
    write = vector_transfer_write(value, dest, indices);
  } else {
    write = std_store(value, dest);
  }
  LLVM_DEBUG(dbgs() << "\n[" DEBUG_TYPE "]: vectorized op: " << *write);
  if (!write->getResults().empty())
    return write->getResult(0);
  return Value();
}

/// If value of assumed VectorType has a shape different than `shape`, buil and
/// return a new vector.broadcast to `shape`.
/// Otherwise, just return value.
static Value broadcastIfNeeded(OpBuilder &builder, Value value,
                               ArrayRef<int64_t> shape) {
  auto vecType = value.getType().dyn_cast<VectorType>();
  if (shape.empty() || (vecType != nullptr && vecType.getShape() == shape))
    return value;
  auto newVecType = VectorType::get(shape, vecType ? vecType.getElementType()
                                                   : value.getType());
  return builder.create<vector::BroadcastOp>(
      builder.getInsertionPoint()->getLoc(), newVecType, value);
}

// Custom vectorization function type. Produce a vector form of Operation*
// assuming all its vectorized operands are already in the BlockAndValueMapping.
// Return nullptr if the Operation cannot be vectorized.
using CustomVectorizationHook = std::function<VectorizationResult(
    Operation *, const BlockAndValueMapping &)>;

/// Helper function to vectorize the terminator of a `linalgOp`. New result
/// vector values are appended to `newResults`. Return
/// VectorizationStatus::NoReplace to signal the vectorization algorithm that it
/// should not try to map produced operations and instead return the results
/// using the `newResults` vector making them available to the
/// vectorization algorithm for RAUW. This function is meant to be used as a
/// CustomVectorizationHook.
static VectorizationResult
vectorizeLinalgYield(OpBuilder &builder, Operation *op,
                     const BlockAndValueMapping &bvm, LinalgOp linalgOp,
                     SmallVectorImpl<Value> &newResults) {
  auto yieldOp = dyn_cast<linalg::YieldOp>(op);
  if (!yieldOp)
    return VectorizationResult{VectorizationStatus::Failure, nullptr};
  for (auto outputs : llvm::enumerate(yieldOp.values())) {
    // TODO: Scan for an opportunity for reuse.
    // TODO: use a map.
    Value vectorValue = bvm.lookup(outputs.value());
    Value newResult = buildVectorWrite(builder, vectorValue,
                                       linalgOp.getOutput(outputs.index()));
    if (newResult)
      newResults.push_back(newResult);
  }
  return VectorizationResult{VectorizationStatus::NoReplace, nullptr};
}

/// Generic vectorization for a single operation `op`, given already vectorized
/// operands carried by `bvm`. Vectorization occurs as follows:
///   1. Try to apply any of the `customVectorizationHooks` and return its
///   result on success.
///   2. Clone any constant in the current scope without vectorization: each
///   consumer of the constant will later determine the shape to which the
///   constant needs to be broadcast to.
///   3. Fail on any remaining non `ElementwiseMappable` op. It is the purpose
///   of the `customVectorizationHooks` to cover such cases.
///   4. Clone `op` in vector form to a vector of shape prescribed by the first
///   operand of maximal rank. Other operands have smaller rank and are
///   broadcast accordingly. It is assumed this broadcast is always legal,
///   otherwise, it means one of the `customVectorizationHooks` is incorrect.
///
/// This function assumes all operands of `op` have been vectorized and are in
/// the `bvm` mapping. As a consequence, this function is meant to be called on
/// a topologically-sorted list of ops.
/// This function does not update `bvm` but returns a VectorizationStatus that
/// instructs the caller what `bvm` update needs to occur.
static VectorizationResult
vectorizeOneOp(OpBuilder &builder, Operation *op,
               const BlockAndValueMapping &bvm,
               ArrayRef<CustomVectorizationHook> customVectorizationHooks) {
  LLVM_DEBUG(dbgs() << "\n[" DEBUG_TYPE "]: vectorize op " << *op);

  // 1. Try to apply any CustomVectorizationHook.
  if (!customVectorizationHooks.empty()) {
    for (auto &customFunc : customVectorizationHooks) {
      VectorizationResult result = customFunc(op, bvm);
      if (result.status == VectorizationStatus::Failure)
        continue;
      return result;
    }
  }

  // 2. Constant ops don't get vectorized but rather broadcasted at their users.
  // Clone so that the constant is not confined to the linalgOp block .
  if (isa<ConstantOp>(op))
    return VectorizationResult{VectorizationStatus::NewOp, builder.clone(*op)};

  // 3. Only ElementwiseMappable are allowed in the generic vectorization.
  if (!OpTrait::hasElementwiseMappableTraits(op))
    return VectorizationResult{VectorizationStatus::Failure, nullptr};

  // 4. Generic vectorization path for ElementwiseMappable ops.
  //   a. first get the first max ranked shape.
  SmallVector<int64_t, 4> firstMaxRankedShape;
  for (Value operand : op->getOperands()) {
    auto vt = bvm.lookup(operand).getType().dyn_cast<VectorType>();
    if (vt && firstMaxRankedShape.size() < vt.getShape().size())
      firstMaxRankedShape.assign(vt.getShape().begin(), vt.getShape().end());
  }
  //   b. broadcast each op if needed.
  auto vectorizedOperands = llvm::map_range(op->getOperands(), [&](Value v) {
    return firstMaxRankedShape.empty()
               ? bvm.lookup(v)
               : broadcastIfNeeded(builder, bvm.lookup(v), firstMaxRankedShape);
  });
  //   c. for elementwise, the result is the vector with the firstMaxRankedShape
  auto returnTypes = llvm::map_range(op->getResultTypes(), [&](Type t) {
    return firstMaxRankedShape.empty()
               ? t
               : VectorType::get(firstMaxRankedShape, t);
  });

  // Build and return the new op.
  OperationState state(op->getLoc(), op->getName());
  state.addAttributes(op->getAttrs());
  state.addOperands(llvm::to_vector<4>(vectorizedOperands));
  state.addTypes(llvm::to_vector<4>(returnTypes));
  return VectorizationResult{VectorizationStatus::NewOp,
                             builder.createOperation(state)};
}

/// Generic vectorization function that rewrites the body of a `linalgOp` into
/// vector form. Generic vectorization proceeds as follows:
///   1. The region for the linalg op is created if necessary.
///   2. Values defined above the region are mapped to themselves and will be
///   broadcasted on a per-need basis by their consumers.
///   3. Each region argument is vectorized into a vector.transfer_read (or 0-d
///   load).
///   TODO: Reuse opportunities for RAR dependencies.
///   4. Register CustomVectorizationHook for YieldOp to capture the results.
///   5. Iteratively call vectorizeOneOp on the region operations.
LogicalResult vectorizeAsLinalgGeneric(
    OpBuilder &builder, LinalgOp linalgOp, SmallVectorImpl<Value> &newResults,
    ArrayRef<CustomVectorizationHook> customVectorizationHooks = {}) {
  // 1. Certain Linalg ops do not have a region but only a region builder.
  // If so, build the region so we can vectorize.
  std::unique_ptr<Region> owningRegion;
  Region *region;
  if (linalgOp->getNumRegions() > 0) {
    region = &linalgOp->getRegion(0);
  } else {
    // RAII avoid remaining in block.
    OpBuilder::InsertionGuard g(builder);
    owningRegion = std::make_unique<Region>();
    region = owningRegion.get();
    Block *block = builder.createBlock(region);
    auto elementTypes = llvm::to_vector<4>(
        llvm::map_range(linalgOp.getShapedOperandTypes(),
                        [](ShapedType t) { return t.getElementType(); }));
    block->addArguments(elementTypes);
    linalgOp.getRegionBuilder()(*block, /*captures=*/{});
  }
  Block *block = &region->front();

  BlockAndValueMapping bvm;
  // 2. Values defined above the region can only be broadcast for now. Make them
  // map to themselves.
  llvm::SetVector<Value> valuesSet;
  mlir::getUsedValuesDefinedAbove(*region, valuesSet);
  bvm.map(valuesSet.getArrayRef(), valuesSet.getArrayRef());

  // 3. Turn all BBArgs into vector.transfer_read / load.
  SmallVector<AffineMap> indexings;
  for (auto bbarg : block->getArguments()) {
    Value vectorArg = linalgOp.getShapedOperand(bbarg.getArgNumber());
    Value vectorRead = buildVectorRead(builder, vectorArg);
    LLVM_DEBUG(dbgs() << "\n[" DEBUG_TYPE "]: new vectorized bbarg("
                      << bbarg.getArgNumber() << "): " << vectorRead);
    bvm.map(bbarg, vectorRead);
    bvm.map(vectorArg, vectorRead);
  }

  // 4. Register CustomVectorizationHook for yieldOp.
  CustomVectorizationHook vectorizeYield =
      [&](Operation *op,
          const BlockAndValueMapping &bvm) -> VectorizationResult {
    return vectorizeLinalgYield(builder, op, bvm, linalgOp, newResults);
  };
  // Append the vectorizeYield hook.
  auto hooks = llvm::to_vector<4>(customVectorizationHooks);
  hooks.push_back(vectorizeYield);

  // 5. Iteratively call `vectorizeOneOp` to each op in the slice.
  for (Operation &op : block->getOperations()) {
    VectorizationResult result = vectorizeOneOp(builder, &op, bvm, hooks);
    if (result.status == VectorizationStatus::Failure) {
      LLVM_DEBUG(dbgs() << "\n[" DEBUG_TYPE "]: failed to vectorize: " << op);
      return failure();
    }
    if (result.status == VectorizationStatus::NewOp) {
      LLVM_DEBUG(dbgs() << "\n[" DEBUG_TYPE "]: new vector op: "
                        << *result.newOp;);
      bvm.map(op.getResults(), result.newOp->getResults());
    }
  }

  return success();
}

/// Detect whether `r` has only ConstantOp, ElementwiseMappable and YieldOp.
static bool hasOnlyScalarElementwiseOp(Region &r) {
  if (!llvm::hasSingleElement(r))
    return false;
  for (Operation &op : r.front()) {
    if (!(isa<ConstantOp, linalg::YieldOp>(op) ||
          OpTrait::hasElementwiseMappableTraits(&op)) ||
        llvm::any_of(op.getResultTypes(),
                     [](Type type) { return !type.isIntOrIndexOrFloat(); }))
      return false;
  }
  return true;
}

// Return true if the op is an element-wise linalg op.
static bool isElementwise(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp)
    return false;
  if (linalgOp.getNumLoops() != linalgOp.getNumParallelLoops())
    return false;
  // TODO: relax the restrictions on indexing map.
  for (unsigned i = 0, e = linalgOp.getNumOutputs(); i < e; i++) {
    if (!linalgOp.getOutputIndexingMap(i).isIdentity())
      return false;
  }
  // Currently bound the input indexing map to minor identity as other
  // permutations might require adding transpose ops to convert the vector read
  // to the right shape.
  for (unsigned i = 0, e = linalgOp.getNumInputs(); i < e; i++) {
    if (!linalgOp.getInputIndexingMap(i).isMinorIdentity())
      return false;
  }
  if (linalgOp->getNumRegions() != 1)
    return false;
  return hasOnlyScalarElementwiseOp(linalgOp->getRegion(0));
}

static LogicalResult vectorizeContraction(OpBuilder &builder, LinalgOp linalgOp,
                                          SmallVectorImpl<Value> &newResults) {
  assert(isaContractionOpInterface(linalgOp) &&
         "expected vectorizeContraction preconditions to be met");
  Location loc = linalgOp.getLoc();
  // Vectorize other ops as vector contraction.
  // TODO: interface.
  LLVM_DEBUG(dbgs() << "\n[" DEBUG_TYPE "]: "
                    << "Rewrite linalg op as vector.contract: ";
             linalgOp.dump());
  // Special function that describes how to vectorize the multiplication op in a
  // linalg contraction.
  CustomVectorizationHook vectorizeContraction =
      [&](Operation *op,
          const BlockAndValueMapping &bvm) -> VectorizationResult {
    if (!isa<MulIOp, MulFOp>(op))
      return VectorizationResult{VectorizationStatus::Failure, nullptr};
    auto outShape = linalgOp.getOutputShapedType(0).getShape();
    auto vType = outShape.empty()
                     ? op->getResult(0).getType()
                     : VectorType::get(outShape, op->getResult(0).getType());
    auto zero =
        builder.create<ConstantOp>(loc, vType, builder.getZeroAttr(vType));
    Operation *contract = builder.create<vector::ContractionOp>(
        loc, bvm.lookup(op->getOperand(0)), bvm.lookup(op->getOperand(1)), zero,
        linalgOp.indexing_maps(), linalgOp.iterator_types());
    return VectorizationResult{VectorizationStatus::NewOp, contract};
  };
  return vectorizeAsLinalgGeneric(builder, linalgOp, newResults,
                                  {vectorizeContraction});
}

LogicalResult mlir::linalg::vectorizeLinalgOpPrecondition(Operation *op) {
  auto linalgOp = cast<linalg::LinalgOp>(op);
  // All types must be static shape to go to vector.
  for (Value operand : linalgOp.getShapedOperands())
    if (!operand.getType().cast<ShapedType>().hasStaticShape())
      return failure();
  for (Type outputTensorType : linalgOp.getOutputTensorTypes())
    if (!outputTensorType.cast<ShapedType>().hasStaticShape())
      return failure();
  if (isElementwise(op))
    return success();
  return success(isaContractionOpInterface(linalgOp));
}

LogicalResult
mlir::linalg::vectorizeLinalgOp(OpBuilder &builder, Operation *op,
                                SmallVectorImpl<Value> &newResults) {
  if (failed(vectorizeLinalgOpPrecondition(op)))
    return failure();

  edsc::ScopedContext scope(builder, op->getLoc());
  if (isElementwise(op)) {
    LLVM_DEBUG(dbgs() << "\n[" DEBUG_TYPE "]: "
                      << "Vectorize linalg op as a generic: " << *op);
    return vectorizeAsLinalgGeneric(builder, cast<LinalgOp>(op), newResults);
  }

  return vectorizeContraction(builder, cast<LinalgOp>(op), newResults);
}

//----------------------------------------------------------------------------//
// Misc. vectorization patterns.
//----------------------------------------------------------------------------//

/// Rewrite a PadTensorOp into a sequence of InitTensorOp, TransferReadOp and
/// TransferWriteOp. For now, this only applies when all low and high paddings
/// are determined to be zero.
LogicalResult PadTensorOpVectorizationPattern::matchAndRewrite(
    linalg::PadTensorOp padOp, PatternRewriter &rewriter) const {
  // Helper function to determine whether an OpFoldResult is not a zero Index.
  auto isNotZeroIndex = [](OpFoldResult ofr) {
    if (Attribute attr = ofr.dyn_cast<Attribute>())
      return attr.cast<IntegerAttr>().getInt() != 0;
    Value v = ofr.get<Value>();
    if (auto constOp = v.getDefiningOp<ConstantOp>())
      if (auto intAttr = constOp.getValue().dyn_cast<IntegerAttr>())
        return intAttr.getValue().getSExtValue() != 0;
    return true;
  };

  auto resultShapedType = padOp.result().getType().cast<ShapedType>();
  // Bail on non-static shapes.
  if (!resultShapedType.hasStaticShape())
    return failure();

  // If any pad_low is not a static 0, needs a mask. Bail for now.
  if (llvm::any_of(padOp.getMixedLowPad(), isNotZeroIndex))
    return failure();
  VectorType vectorType = extractVectorTypeFromShapedValue(padOp.result());
  if (!vectorType)
    return failure();

  // Only support padding with a constant for now, i.e. either:
  //   1. A BBarg from a different block.
  //   2. A value defined outside of the current block.
  Block &block = padOp.region().front();
  auto yieldOp = cast<YieldOp>(block.getTerminator());
  assert(yieldOp.getNumOperands() == 1 && "expected single operand yield");
  Value padValue = yieldOp.values().front();
  Operation *definingOp = padValue.getDefiningOp();
  if (definingOp && definingOp->getBlock() == &block)
    return failure();
  if (!definingOp && padValue.cast<BlockArgument>().getOwner() == &block)
    return failure();

  // TODO: if any pad_high is not a static 0, needs a mask. For now, just bail.
  if (llvm::any_of(padOp.getMixedHighPad(),
                   [&](OpFoldResult ofr) { return isNotZeroIndex(ofr); }))
    return failure();

  // Now we can rewrite as InitTensorOp + TransferReadOp@[0..0] +
  // TransferWriteOp@[0..0].
  SmallVector<Value> indices(
      resultShapedType.getRank(),
      rewriter.create<ConstantIndexOp>(padOp.getLoc(), 0));
  Value read = rewriter.create<vector::TransferReadOp>(
      padOp.getLoc(), vectorType, padOp.source(), indices, padValue);
  Value init =
      rewriter.create<InitTensorOp>(padOp.getLoc(), resultShapedType.getShape(),
                                    resultShapedType.getElementType());
  rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(padOp, read, init,
                                                       indices);

  return success();
}

// TODO: cleanup all the convolution vectorization patterns.
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
      LinalgTransformationFilter(ArrayRef<Identifier>{},
                                 Identifier::get(kTiledMarker, context)));

  promotionPatterns.insert<LinalgPromotionPattern<ConvOp>>(
      context, LinalgPromotionOptions().setUseFullTileBuffersByDefault(true),
      LinalgTransformationFilter(Identifier::get(kTiledMarker, context),
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
  populateVectorizationPatterns<ConvInputNWCFilterWCFOp, 3>(
      tiling, promotion, vectorization, tileSizes, context);

  populateVectorizationPatterns<ConvNCWOp, 3>(tiling, promotion, vectorization,
                                              tileSizes, context);
  populateVectorizationPatterns<ConvInputNCWFilterWCFOp, 3>(
      tiling, promotion, vectorization, tileSizes, context);

  populateVectorizationPatterns<ConvHWOp, 2>(tiling, promotion, vectorization,
                                             tileSizes, context);

  populateVectorizationPatterns<ConvNHWCOp, 4>(tiling, promotion, vectorization,
                                               tileSizes, context);
  populateVectorizationPatterns<ConvInputNHWCFilterHWCFOp, 4>(
      tiling, promotion, vectorization, tileSizes, context);

  populateVectorizationPatterns<ConvNCHWOp, 4>(tiling, promotion, vectorization,
                                               tileSizes, context);
  populateVectorizationPatterns<ConvInputNCHWFilterHWCFOp, 4>(
      tiling, promotion, vectorization, tileSizes, context);

  populateVectorizationPatterns<ConvDHWOp, 3>(tiling, promotion, vectorization,
                                              tileSizes, context);

  populateVectorizationPatterns<ConvNDHWCOp, 5>(
      tiling, promotion, vectorization, tileSizes, context);
  populateVectorizationPatterns<ConvInputNDHWCFilterDHWCFOp, 5>(
      tiling, promotion, vectorization, tileSizes, context);

  populateVectorizationPatterns<ConvNCDHWOp, 5>(
      tiling, promotion, vectorization, tileSizes, context);
  populateVectorizationPatterns<ConvInputNCDHWFilterDHWCFOp, 5>(
      tiling, promotion, vectorization, tileSizes, context);

  patterns.push_back(std::move(tiling));
  patterns.push_back(std::move(promotion));
  patterns.push_back(std::move(vectorization));
}

//----------------------------------------------------------------------------//
// Forwarding patterns
//----------------------------------------------------------------------------//

/// Check whether there is any interleaved use of any `values` between `firstOp`
/// and `secondOp`. Conservatively return `true` if any op or value is in a
/// different block.
static bool mayExistInterleavedUses(Operation *firstOp, Operation *secondOp,
                                    ValueRange values) {
  if (firstOp->getBlock() != secondOp->getBlock() ||
      !firstOp->isBeforeInBlock(secondOp)) {
    LLVM_DEBUG(llvm::dbgs() << "\n[" DEBUG_TYPE "]: "
                            << "interleavedUses precondition failed, firstOp: "
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
                 << "\n[" DEBUG_TYPE "]: "
                 << " found interleaved op " << *owner
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
  Value viewOrAlloc = xferOp.source();
  if (!viewOrAlloc.getDefiningOp<ViewOp>() &&
      !viewOrAlloc.getDefiningOp<AllocOp>())
    return failure();

  LLVM_DEBUG(llvm::dbgs() << "\n[" DEBUG_TYPE "]: " << viewOrAlloc);

  // Ensure there is exactly one subview of `viewOrAlloc` defining `subView`.
  SubViewOp subViewOp = getSubViewUseIfUnique(viewOrAlloc);
  if (!subViewOp)
    return failure();
  Value subView = subViewOp.getResult();
  LLVM_DEBUG(llvm::dbgs() << "\n[" DEBUG_TYPE "]: "
                          << "with subView " << subView);

  // Find the copy into `subView` without interleaved uses.
  CopyOp copyOp;
  for (auto &u : subView.getUses()) {
    if (auto newCopyOp = dyn_cast<CopyOp>(u.getOwner())) {
      if (newCopyOp.getOutputBuffer(0) != subView)
        continue;
      LLVM_DEBUG(llvm::dbgs() << "\n[" DEBUG_TYPE "]: "
                              << "copy candidate " << *newCopyOp);
      if (mayExistInterleavedUses(newCopyOp, xferOp, {viewOrAlloc, subView}))
        continue;
      copyOp = newCopyOp;
      break;
    }
  }
  if (!copyOp)
    return failure();
  LLVM_DEBUG(llvm::dbgs() << "\n[" DEBUG_TYPE "]: "
                          << "with copy " << *copyOp);

  // Find the fill into `viewOrAlloc` without interleaved uses before the copy.
  FillOp maybeFillOp;
  for (auto &u : viewOrAlloc.getUses()) {
    if (auto newFillOp = dyn_cast<FillOp>(u.getOwner())) {
      if (newFillOp.getOutputBuffer(0) != viewOrAlloc)
        continue;
      LLVM_DEBUG(llvm::dbgs() << "\n[" DEBUG_TYPE "]: "
                              << "fill candidate " << *newFillOp);
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
    LLVM_DEBUG(llvm::dbgs() << "\n[" DEBUG_TYPE "]: "
                            << "with maybeFillOp " << *maybeFillOp);

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
  Value viewOrAlloc = xferOp.source();
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
