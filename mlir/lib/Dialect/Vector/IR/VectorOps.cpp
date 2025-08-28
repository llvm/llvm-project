//===- VectorOps.cpp - MLIR Vector Dialect Operations ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements convenience types for working with super-vectorization
// operations, in particular super-vector loads and stores.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Vector/IR/VectorOps.h"

#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Dialect/Affine/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/SubsetOpInterface.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

#include <cassert>
#include <cstdint>

#include "mlir/Dialect/Vector/IR/VectorDialect.cpp.inc"
// Pull in all enum type and utility function definitions.
#include "mlir/Dialect/Vector/IR/VectorEnums.cpp.inc"

using namespace mlir;
using namespace mlir::vector;

/// Helper enum to classify mask value.
enum class MaskFormat {
  AllTrue = 0,
  AllFalse = 1,
  Unknown = 2,
};

/// Helper method to classify a mask value. Currently, the method
/// looks "under the hood" of a constant value with dense attributes
/// and a constant mask operation (since the client may be called at
/// various stages during progressive lowering).
static MaskFormat getMaskFormat(Value mask) {
  if (auto c = mask.getDefiningOp<arith::ConstantOp>()) {
    // Inspect constant dense values. We count up for bits that
    // are set, count down for bits that are cleared, and bail
    // when a mix is detected.
    if (auto denseElts = llvm::dyn_cast<DenseIntElementsAttr>(c.getValue())) {
      int64_t val = 0;
      for (bool b : denseElts.getValues<bool>())
        if (b && val >= 0)
          val++;
        else if (!b && val <= 0)
          val--;
        else
          return MaskFormat::Unknown;
      if (val > 0)
        return MaskFormat::AllTrue;
      if (val < 0)
        return MaskFormat::AllFalse;
    }
  } else if (auto m = mask.getDefiningOp<ConstantMaskOp>()) {
    // Inspect constant mask index. If the index exceeds the
    // dimension size, all bits are set. If the index is zero
    // or less, no bits are set.
    ArrayRef<int64_t> masks = m.getMaskDimSizes();
    auto shape = m.getType().getShape();
    bool allTrue = true;
    bool allFalse = true;
    for (auto [maskIdx, dimSize] : llvm::zip_equal(masks, shape)) {
      if (maskIdx < dimSize)
        allTrue = false;
      if (maskIdx > 0)
        allFalse = false;
    }
    if (allTrue)
      return MaskFormat::AllTrue;
    if (allFalse)
      return MaskFormat::AllFalse;
  } else if (auto m = mask.getDefiningOp<CreateMaskOp>()) {
    // Finds all-false create_masks. An all-true create_mask requires all
    // dims to be constants, so that'll be folded to a constant_mask, then
    // detected in the constant_mask case.
    auto maskOperands = m.getOperands();
    for (Value operand : maskOperands) {
      if (auto constantOp = operand.getDefiningOp<arith::ConstantOp>()) {
        int64_t dimSize =
            llvm::cast<IntegerAttr>(constantOp.getValue()).getInt();
        if (dimSize <= 0)
          return MaskFormat::AllFalse;
      }
    }
    return MaskFormat::Unknown;
  }
  return MaskFormat::Unknown;
}

/// Default callback to build a region with a 'vector.yield' terminator with no
/// arguments.
void mlir::vector::buildTerminatedBody(OpBuilder &builder, Location loc) {
  vector::YieldOp::create(builder, loc);
}

// Helper for verifying combining kinds in contractions and reductions.
static bool isSupportedCombiningKind(CombiningKind combiningKind,
                                     Type elementType) {
  switch (combiningKind) {
  case CombiningKind::ADD:
  case CombiningKind::MUL:
    return elementType.isIntOrIndexOrFloat();
  case CombiningKind::MINUI:
  case CombiningKind::MINSI:
  case CombiningKind::MAXUI:
  case CombiningKind::MAXSI:
  case CombiningKind::AND:
  case CombiningKind::OR:
  case CombiningKind::XOR:
    return elementType.isIntOrIndex();
  case CombiningKind::MINNUMF:
  case CombiningKind::MAXNUMF:
  case CombiningKind::MINIMUMF:
  case CombiningKind::MAXIMUMF:
    return llvm::isa<FloatType>(elementType);
  }
  return false;
}

///  Returns the effective rank of the vector to read/write for Xfer Ops
///
///  When the element type of the shaped type is _a scalar_, this will simply
///  return the rank of the vector ( the result for xfer_read or the value to
///  store for xfer_write).
///
///  When the element type of the base shaped type is _a vector_, returns the
///  difference between the original vector type and the element type of the
///  shaped type.
///
///  EXAMPLE 1 (element type is _a scalar_):
///   - shapedType = tensor<10x20xf32>, vectorType = vector<2x4xf32>
///     - shapedType.getElementType() = f32 (rank 0)
///     - vectorType.getRank() = 2
///     - Result = 2 - 0 = 2
///
/// EXAMPLE 2 (element type is _a vector_):
///   - shapedType = tensor<10xvector<20xf32>>, vectorType = vector<20xf32>
///     - shapedType.getElementType() = vector<20xf32> (rank 1)
///     - vectorType.getRank() = 1
///     - Result = 1 - 1 = 0
///
/// This is used to determine the number of minor dimensions for identity maps
/// in vector transfer Ops.
static unsigned getEffectiveVectorRankForXferOp(ShapedType shapedType,
                                                VectorType vectorType) {
  unsigned elementVectorRank = 0;
  VectorType elementVectorType =
      llvm::dyn_cast<VectorType>(shapedType.getElementType());
  if (elementVectorType)
    elementVectorRank += elementVectorType.getRank();
  return vectorType.getRank() - elementVectorRank;
}

AffineMap mlir::vector::getTransferMinorIdentityMap(ShapedType shapedType,
                                                    VectorType vectorType) {
  // 0-d transfers are to/from tensor<t>/memref<t> and vector<1xt>.
  // TODO: replace once we have 0-d vectors.
  if (shapedType.getRank() == 0 &&
      vectorType.getShape() == ArrayRef<int64_t>{1})
    return AffineMap::get(
        /*numDims=*/0, /*numSymbols=*/0,
        getAffineConstantExpr(0, shapedType.getContext()));
  return AffineMap::getMinorIdentityMap(
      shapedType.getRank(),
      getEffectiveVectorRankForXferOp(shapedType, vectorType),
      shapedType.getContext());
}

/// Check if `write` is of a constant splat and the masked `read` is padded with
/// the same splat value -- meaning it could be the same value as the initial
/// constant splat.
static bool isSplatWriteConsistentWithMaskedRead(vector::TransferWriteOp write,
                                                 vector::TransferReadOp read) {
  auto readMask = read.getMask();
  auto writeMask = write.getMask();
  // Check if the masks are consistent. The splat value could be the same if the
  // read is masked (and padded with the splat value), and the write is unmasked
  // or has the same mask. Note this does not allow the case where the write is
  // masked and the read is unmasked, as then the read could be of more elements
  // than the write (which may not be the same value).
  bool couldBeSameSplat = readMask && (!writeMask || writeMask == readMask);
  if (!couldBeSameSplat)
    return false;
  // Check for constant splat (as the source of the write).
  DenseElementsAttr splatAttr;
  if (!matchPattern(write.getVector(),
                    m_Constant<DenseElementsAttr>(&splatAttr)) ||
      !splatAttr.isSplat()) {
    return false;
  }
  // The padding of the read and the constant splat value must be the same.
  Attribute padAttr;
  if (!matchPattern(read.getPadding(), m_Constant(&padAttr)))
    return false;
  return padAttr == splatAttr.getSplatValue<Attribute>();
}

bool mlir::vector::checkSameValueRAW(vector::TransferWriteOp defWrite,
                                     vector::TransferReadOp read) {
  return !defWrite.hasOutOfBoundsDim() &&
         defWrite.getIndices() == read.getIndices() &&
         defWrite.getVectorType() == read.getVectorType() &&
         defWrite.getPermutationMap() == read.getPermutationMap() &&
         ((!defWrite.getMask() && !read.getMask()) ||
          isSplatWriteConsistentWithMaskedRead(defWrite, read));
}

bool mlir::vector::checkSameValueWAW(vector::TransferWriteOp write,
                                     vector::TransferWriteOp priorWrite) {
  return priorWrite.getIndices() == write.getIndices() &&
         priorWrite.getMask() == write.getMask() &&
         priorWrite.getVectorType() == write.getVectorType() &&
         priorWrite.getPermutationMap() == write.getPermutationMap();
}

bool mlir::vector::isDisjointTransferIndices(
    VectorTransferOpInterface transferA, VectorTransferOpInterface transferB,
    bool testDynamicValueUsingBounds) {
  // For simplicity only look at transfer of same type.
  if (transferA.getVectorType() != transferB.getVectorType())
    return false;
  unsigned rankOffset = transferA.getLeadingShapedRank();
  for (unsigned i = 0, e = transferA.getIndices().size(); i < e; i++) {
    Value indexA = transferA.getIndices()[i];
    Value indexB = transferB.getIndices()[i];
    std::optional<int64_t> cstIndexA = getConstantIntValue(indexA);
    std::optional<int64_t> cstIndexB = getConstantIntValue(indexB);

    if (i < rankOffset) {
      // For leading dimensions, if we can prove that index are different we
      // know we are accessing disjoint slices.
      if (cstIndexA.has_value() && cstIndexB.has_value()) {
        if (*cstIndexA != *cstIndexB)
          return true;
        continue;
      }
      if (testDynamicValueUsingBounds) {
        // First try to see if we can fully compose and simplify the affine
        // expression as a fast track.
        FailureOr<uint64_t> delta =
            affine::fullyComposeAndComputeConstantDelta(indexA, indexB);
        if (succeeded(delta) && *delta != 0)
          return true;

        FailureOr<bool> testEqual =
            ValueBoundsConstraintSet::areEqual(indexA, indexB);
        if (succeeded(testEqual) && !testEqual.value())
          return true;
      }
    } else {
      // For this dimension, we slice a part of the memref we need to make sure
      // the intervals accessed don't overlap.
      int64_t vectorDim = transferA.getVectorType().getDimSize(i - rankOffset);
      if (cstIndexA.has_value() && cstIndexB.has_value()) {
        int64_t distance = std::abs(*cstIndexA - *cstIndexB);
        if (distance >= vectorDim)
          return true;
        continue;
      }
      if (testDynamicValueUsingBounds) {
        // First try to see if we can fully compose and simplify the affine
        // expression as a fast track.
        FailureOr<int64_t> delta =
            affine::fullyComposeAndComputeConstantDelta(indexA, indexB);
        if (succeeded(delta) && std::abs(*delta) >= vectorDim)
          return true;

        FailureOr<int64_t> computeDelta =
            ValueBoundsConstraintSet::computeConstantDelta(indexA, indexB);
        if (succeeded(computeDelta)) {
          if (std::abs(computeDelta.value()) >= vectorDim)
            return true;
        }
      }
    }
  }
  return false;
}

bool mlir::vector::isDisjointTransferSet(VectorTransferOpInterface transferA,
                                         VectorTransferOpInterface transferB,
                                         bool testDynamicValueUsingBounds) {
  if (transferA.getBase() != transferB.getBase())
    return false;
  return isDisjointTransferIndices(transferA, transferB,
                                   testDynamicValueUsingBounds);
}

// Helper to iterate over n-D vector slice elements. Calculate the next
// `position` in the n-D vector of size `shape`, applying an offset `offsets`.
// Modifies the `position` in place. Returns a failure when `position` becomes
// the end position.
static LogicalResult incSlicePosition(MutableArrayRef<int64_t> position,
                                      ArrayRef<int64_t> shape,
                                      ArrayRef<int64_t> offsets) {
  for (auto [posInDim, dimSize, offsetInDim] :
       llvm::reverse(llvm::zip_equal(position, shape, offsets))) {
    ++posInDim;
    if (posInDim < dimSize + offsetInDim)
      return success();

    // Carry the overflow to the next loop iteration.
    posInDim = offsetInDim;
  }

  return failure();
}

/// Returns the integer numbers in `values`. `values` are expected to be
/// constant operations.
SmallVector<int64_t> vector::getAsIntegers(ArrayRef<Value> values) {
  SmallVector<int64_t> ints;
  llvm::transform(values, std::back_inserter(ints), [](Value value) {
    auto constOp = value.getDefiningOp<arith::ConstantIndexOp>();
    assert(constOp && "Unexpected non-constant index");
    return constOp.value();
  });
  return ints;
}

/// Returns the integer numbers in `foldResults`. `foldResults` are expected to
/// be constant operations.
SmallVector<int64_t> vector::getAsIntegers(ArrayRef<OpFoldResult> foldResults) {
  SmallVector<int64_t> ints;
  llvm::transform(
      foldResults, std::back_inserter(ints), [](OpFoldResult foldResult) {
        assert(isa<Attribute>(foldResult) && "Unexpected non-constant index");
        return cast<IntegerAttr>(cast<Attribute>(foldResult)).getInt();
      });
  return ints;
}

/// Convert `foldResults` into Values. Integer attributes are converted to
/// constant op.
SmallVector<Value> vector::getAsValues(OpBuilder &builder, Location loc,
                                       ArrayRef<OpFoldResult> foldResults) {
  SmallVector<Value> values;
  llvm::transform(foldResults, std::back_inserter(values),
                  [&](OpFoldResult foldResult) {
                    if (auto attr = dyn_cast<Attribute>(foldResult))
                      return arith::ConstantIndexOp::create(
                                 builder, loc, cast<IntegerAttr>(attr).getInt())
                          .getResult();

                    return cast<Value>(foldResult);
                  });
  return values;
}

std::optional<int64_t> vector::getConstantVscaleMultiplier(Value value) {
  if (value.getDefiningOp<vector::VectorScaleOp>())
    return 1;
  auto mul = value.getDefiningOp<arith::MulIOp>();
  if (!mul)
    return {};
  auto lhs = mul.getLhs();
  auto rhs = mul.getRhs();
  if (lhs.getDefiningOp<vector::VectorScaleOp>())
    return getConstantIntValue(rhs);
  if (rhs.getDefiningOp<vector::VectorScaleOp>())
    return getConstantIntValue(lhs);
  return {};
}

/// Converts an IntegerAttr to have the specified type if needed.
/// This handles cases where constant attributes have a different type than the
/// target element type. If the input attribute is not an IntegerAttr or already
/// has the correct type, returns it unchanged.
static Attribute convertIntegerAttr(Attribute attr, Type expectedType) {
  if (auto intAttr = mlir::dyn_cast<IntegerAttr>(attr)) {
    if (intAttr.getType() != expectedType)
      return IntegerAttr::get(expectedType, intAttr.getInt());
  }
  return attr;
}

//===----------------------------------------------------------------------===//
// CombiningKindAttr
//===----------------------------------------------------------------------===//

namespace mlir {
namespace vector {
namespace detail {
struct BitmaskEnumStorage : public AttributeStorage {
  using KeyTy = uint64_t;

  BitmaskEnumStorage(KeyTy val) : value(val) {}

  bool operator==(const KeyTy &key) const { return value == key; }

  static BitmaskEnumStorage *construct(AttributeStorageAllocator &allocator,
                                       const KeyTy &key) {
    return new (allocator.allocate<BitmaskEnumStorage>())
        BitmaskEnumStorage(key);
  }

  KeyTy value = 0;
};
} // namespace detail
} // namespace vector
} // namespace mlir

//===----------------------------------------------------------------------===//
// VectorDialect
//===----------------------------------------------------------------------===//

namespace {
/// This class defines the interface for handling inlining with vector dialect
/// operations.
struct VectorInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  /// All vector dialect ops can be inlined.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }
};
} // namespace

void VectorDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/Vector/IR/VectorAttributes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Vector/IR/VectorOps.cpp.inc"
      >();

  addInterfaces<VectorInlinerInterface>();

  declarePromisedInterfaces<bufferization::BufferizableOpInterface,
                            TransferReadOp, TransferWriteOp, GatherOp, MaskOp,
                            YieldOp>();
  declarePromisedInterfaces<SubsetOpInterface, TransferReadOp,
                            TransferWriteOp>();
  declarePromisedInterface<SubsetExtractionOpInterface, TransferReadOp>();
  declarePromisedInterface<SubsetInsertionOpInterface, TransferWriteOp>();
  declarePromisedInterface<ConvertToLLVMPatternInterface, VectorDialect>();
}

/// Materialize a single constant operation from a given attribute value with
/// the desired resultant type.
Operation *VectorDialect::materializeConstant(OpBuilder &builder,
                                              Attribute value, Type type,
                                              Location loc) {
  if (isa<ub::PoisonAttrInterface>(value))
    return value.getDialect().materializeConstant(builder, value, type, loc);

  return arith::ConstantOp::materialize(builder, value, type, loc);
}

IntegerType vector::getVectorSubscriptType(Builder &builder) {
  return builder.getIntegerType(64);
}

ArrayAttr vector::getVectorSubscriptAttr(Builder &builder,
                                         ArrayRef<int64_t> values) {
  return builder.getI64ArrayAttr(values);
}

//===----------------------------------------------------------------------===//
// MultiDimReductionOp
//===----------------------------------------------------------------------===//

void vector::MultiDimReductionOp::build(OpBuilder &builder,
                                        OperationState &result, Value source,
                                        Value acc, ArrayRef<bool> reductionMask,
                                        CombiningKind kind) {
  SmallVector<int64_t> reductionDims;
  for (const auto &en : llvm::enumerate(reductionMask))
    if (en.value())
      reductionDims.push_back(en.index());
  build(builder, result, kind, source, acc, reductionDims);
}

OpFoldResult MultiDimReductionOp::fold(FoldAdaptor adaptor) {
  // Single parallel dim, this is a noop.
  if (getSourceVectorType().getRank() == 1 && !isReducedDim(0))
    return getSource();
  return {};
}

std::optional<SmallVector<int64_t, 4>>
MultiDimReductionOp::getShapeForUnroll() {
  return llvm::to_vector<4>(getSourceVectorType().getShape());
}

LogicalResult MultiDimReductionOp::verify() {
  SmallVector<int64_t> targetShape;
  SmallVector<bool> scalableDims;
  Type inferredReturnType;
  auto sourceScalableDims = getSourceVectorType().getScalableDims();
  for (auto [dimIdx, dimSize] :
       llvm::enumerate(getSourceVectorType().getShape()))
    if (!llvm::any_of(getReductionDims(),
                      [dimIdx = dimIdx](int64_t reductionDimIdx) {
                        return reductionDimIdx == static_cast<int64_t>(dimIdx);
                      })) {
      targetShape.push_back(dimSize);
      scalableDims.push_back(sourceScalableDims[dimIdx]);
    }
  // TODO: update to also allow 0-d vectors when available.
  if (targetShape.empty())
    inferredReturnType = getSourceVectorType().getElementType();
  else
    inferredReturnType = VectorType::get(
        targetShape, getSourceVectorType().getElementType(), scalableDims);
  if (getType() != inferredReturnType)
    return emitOpError() << "destination type " << getType()
                         << " is incompatible with source type "
                         << getSourceVectorType();

  return success();
}

/// Returns the mask type expected by this operation.
Type MultiDimReductionOp::getExpectedMaskType() {
  auto vecType = getSourceVectorType();
  return VectorType::get(vecType.getShape(),
                         IntegerType::get(vecType.getContext(), /*width=*/1),
                         vecType.getScalableDims());
}

namespace {
// Only unit dimensions that are being reduced are folded. If the dimension is
// unit, but not reduced, it is not folded, thereby keeping the output type the
// same. If not all dimensions which are reduced are of unit dimension, this
// transformation does nothing. This is just a generalization of
// ElideSingleElementReduction for ReduceOp.
struct ElideUnitDimsInMultiDimReduction
    : public OpRewritePattern<MultiDimReductionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MultiDimReductionOp reductionOp,
                                PatternRewriter &rewriter) const override {
    ArrayRef<int64_t> shape = reductionOp.getSourceVectorType().getShape();
    for (const auto &dim : enumerate(shape)) {
      if (reductionOp.isReducedDim(dim.index()) && dim.value() != 1)
        return failure();
    }

    // Vector mask setup.
    OpBuilder::InsertionGuard guard(rewriter);
    Operation *rootOp;
    Value mask;
    if (reductionOp.isMasked()) {
      rewriter.setInsertionPoint(reductionOp.getMaskingOp());
      rootOp = reductionOp.getMaskingOp();
      mask = reductionOp.getMaskingOp().getMask();
    } else {
      rootOp = reductionOp;
    }

    Location loc = reductionOp.getLoc();
    Value acc = reductionOp.getAcc();
    Value cast;
    if (auto dstVecType = dyn_cast<VectorType>(reductionOp.getDestType())) {
      if (mask) {
        VectorType newMaskType =
            VectorType::get(dstVecType.getShape(), rewriter.getI1Type(),
                            dstVecType.getScalableDims());
        mask = vector::ShapeCastOp::create(rewriter, loc, newMaskType, mask);
      }
      cast = vector::ShapeCastOp::create(
          rewriter, loc, reductionOp.getDestType(), reductionOp.getSource());
    } else {
      // This means we are reducing all the dimensions, and all reduction
      // dimensions are of size 1. So a simple extraction would do.
      if (mask)
        mask = vector::ExtractOp::create(rewriter, loc, mask);
      cast = vector::ExtractOp::create(rewriter, loc, reductionOp.getSource());
    }

    Value result =
        vector::makeArithReduction(rewriter, loc, reductionOp.getKind(), acc,
                                   cast, /*fastmath=*/nullptr, mask);
    rewriter.replaceOp(rootOp, result);
    return success();
  }
};
} // namespace

void MultiDimReductionOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<ElideUnitDimsInMultiDimReduction>(context);
}

//===----------------------------------------------------------------------===//
// ReductionOp
//===----------------------------------------------------------------------===//

void vector::ReductionOp::build(OpBuilder &builder, OperationState &result,
                                CombiningKind kind, Value vector,
                                arith::FastMathFlags fastMathFlags) {
  build(builder, result, kind, vector, /*acc=*/Value(), fastMathFlags);
}

void vector::ReductionOp::build(OpBuilder &builder, OperationState &result,
                                CombiningKind kind, Value vector, Value acc,
                                arith::FastMathFlags fastMathFlags) {
  build(builder, result,
        llvm::cast<VectorType>(vector.getType()).getElementType(), kind, vector,
        acc, fastMathFlags);
}

LogicalResult ReductionOp::verify() {
  // Verify for 0-D and 1-D vector.
  int64_t rank = getSourceVectorType().getRank();
  if (rank > 1)
    return emitOpError("unsupported reduction rank: ") << rank;

  // Verify supported reduction kind.
  Type eltType = getDest().getType();
  if (!isSupportedCombiningKind(getKind(), eltType))
    return emitOpError("unsupported reduction type '")
           << eltType << "' for kind '" << stringifyCombiningKind(getKind())
           << "'";

  return success();
}

// MaskableOpInterface methods.

/// Returns the mask type expected by this operation.
Type ReductionOp::getExpectedMaskType() {
  auto vecType = getSourceVectorType();
  return VectorType::get(vecType.getShape(),
                         IntegerType::get(vecType.getContext(), /*width=*/1),
                         vecType.getScalableDims());
}

Value mlir::vector::getVectorReductionOp(arith::AtomicRMWKind op,
                                         OpBuilder &builder, Location loc,
                                         Value vector) {
  switch (op) {
  case arith::AtomicRMWKind::addf:
  case arith::AtomicRMWKind::addi:
    return vector::ReductionOp::create(builder, vector.getLoc(),
                                       CombiningKind::ADD, vector);
  case arith::AtomicRMWKind::mulf:
  case arith::AtomicRMWKind::muli:
    return vector::ReductionOp::create(builder, vector.getLoc(),
                                       CombiningKind::MUL, vector);
  case arith::AtomicRMWKind::minimumf:
    return vector::ReductionOp::create(builder, vector.getLoc(),
                                       CombiningKind::MINIMUMF, vector);
  case arith::AtomicRMWKind::mins:
    return vector::ReductionOp::create(builder, vector.getLoc(),
                                       CombiningKind::MINSI, vector);
  case arith::AtomicRMWKind::minu:
    return vector::ReductionOp::create(builder, vector.getLoc(),
                                       CombiningKind::MINUI, vector);
  case arith::AtomicRMWKind::maximumf:
    return vector::ReductionOp::create(builder, vector.getLoc(),
                                       CombiningKind::MAXIMUMF, vector);
  case arith::AtomicRMWKind::maxs:
    return vector::ReductionOp::create(builder, vector.getLoc(),
                                       CombiningKind::MAXSI, vector);
  case arith::AtomicRMWKind::maxu:
    return vector::ReductionOp::create(builder, vector.getLoc(),
                                       CombiningKind::MAXUI, vector);
  case arith::AtomicRMWKind::andi:
    return vector::ReductionOp::create(builder, vector.getLoc(),
                                       CombiningKind::AND, vector);
  case arith::AtomicRMWKind::ori:
    return vector::ReductionOp::create(builder, vector.getLoc(),
                                       CombiningKind::OR, vector);
  // TODO: Add remaining reduction operations.
  default:
    (void)emitOptionalError(loc, "Reduction operation type not supported");
    break;
  }
  return nullptr;
}

std::optional<SmallVector<int64_t, 4>> ReductionOp::getShapeForUnroll() {
  return llvm::to_vector<4>(getSourceVectorType().getShape());
}

namespace {
struct ElideSingleElementReduction : public OpRewritePattern<ReductionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ReductionOp reductionOp,
                                PatternRewriter &rewriter) const override {
    // Vector mask setup.
    OpBuilder::InsertionGuard guard(rewriter);
    auto maskableOp =
        cast<vector::MaskableOpInterface>(reductionOp.getOperation());
    Operation *rootOp;
    Value mask;
    if (maskableOp.isMasked()) {
      rewriter.setInsertionPoint(maskableOp.getMaskingOp());
      rootOp = maskableOp.getMaskingOp();
      mask = maskableOp.getMaskingOp().getMask();
    } else {
      rootOp = reductionOp;
    }

    auto vectorType = reductionOp.getSourceVectorType();
    if (vectorType.getRank() != 0 && vectorType.getDimSize(0) != 1)
      return failure();

    Location loc = reductionOp.getLoc();
    if (mask)
      mask = ExtractOp::create(rewriter, loc, mask);
    Value result = ExtractOp::create(rewriter, loc, reductionOp.getVector());

    if (Value acc = reductionOp.getAcc())
      result = vector::makeArithReduction(rewriter, loc, reductionOp.getKind(),
                                          result, acc,
                                          reductionOp.getFastmathAttr(), mask);

    rewriter.replaceOp(rootOp, result);
    return success();
  }
};
} // namespace

void ReductionOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.add<ElideSingleElementReduction>(context);
}

//===----------------------------------------------------------------------===//
// ContractionOp
//===----------------------------------------------------------------------===//

void vector::ContractionOp::build(OpBuilder &builder, OperationState &result,
                                  Value lhs, Value rhs, Value acc,
                                  ArrayRef<ArrayRef<AffineExpr>> indexingExprs,
                                  ArrayRef<IteratorType> iteratorTypes) {
  result.addOperands({lhs, rhs, acc});
  result.addTypes(acc.getType());
  result.addAttribute(
      getIndexingMapsAttrName(result.name),
      builder.getAffineMapArrayAttr(
          AffineMap::inferFromExprList(indexingExprs, builder.getContext())));
  result.addAttribute(
      getIteratorTypesAttrName(result.name),
      builder.getArrayAttr(llvm::to_vector(llvm::map_range(
          iteratorTypes, [&](IteratorType t) -> mlir::Attribute {
            return IteratorTypeAttr::get(builder.getContext(), t);
          }))));
}

void vector::ContractionOp::build(OpBuilder &builder, OperationState &result,
                                  Value lhs, Value rhs, Value acc,
                                  ArrayAttr indexingMaps,
                                  ArrayAttr iteratorTypes) {
  build(builder, result, lhs, rhs, acc, indexingMaps, iteratorTypes,
        ContractionOp::getDefaultKind());
}

void vector::ContractionOp::build(OpBuilder &builder, OperationState &result,
                                  Value lhs, Value rhs, Value acc,
                                  ArrayAttr indexingMaps,
                                  ArrayAttr iteratorTypes, CombiningKind kind) {
  result.addOperands({lhs, rhs, acc});
  result.addTypes(acc.getType());
  result.addAttribute(getIndexingMapsAttrName(result.name), indexingMaps);
  result.addAttribute(getIteratorTypesAttrName(result.name), iteratorTypes);
  result.addAttribute(getKindAttrName(result.name),
                      CombiningKindAttr::get(builder.getContext(), kind));
}

ParseResult ContractionOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand lhsInfo;
  OpAsmParser::UnresolvedOperand rhsInfo;
  OpAsmParser::UnresolvedOperand accInfo;
  SmallVector<OpAsmParser::UnresolvedOperand, 2> masksInfo;
  SmallVector<Type, 2> types;
  Type resultType;
  auto loc = parser.getCurrentLocation();
  DictionaryAttr dictAttr;
  // TODO: Unify linalg op attribute parsing.
  if (parser.parseAttribute(dictAttr) || parser.parseOperand(lhsInfo) ||
      parser.parseComma() || parser.parseOperand(rhsInfo) ||
      parser.parseComma() || parser.parseOperand(accInfo) ||
      parser.parseTrailingOperandList(masksInfo) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonTypeList(types) ||
      parser.parseKeywordType("into", resultType) ||
      parser.resolveOperand(lhsInfo, types[0], result.operands) ||
      parser.resolveOperand(rhsInfo, types[1], result.operands) ||
      parser.resolveOperand(accInfo, resultType, result.operands) ||
      parser.addTypeToList(resultType, result.types))
    return failure();
  result.attributes.append(dictAttr.getValue().begin(),
                           dictAttr.getValue().end());

  // Convert array of string into an array of IteratyType enums. This is needed,
  // because tests still use the old format when 'iterator_types' attribute is
  // represented as an array of strings.
  // TODO: Remove this conversion once tests are fixed.
  auto iteratorTypes = dyn_cast_or_null<ArrayAttr>(
      result.attributes.get(getIteratorTypesAttrName(result.name)));
  if (!iteratorTypes) {
    return parser.emitError(loc)
           << "expected " << getIteratorTypesAttrName(result.name)
           << " array attribute";
  }

  SmallVector<Attribute> iteratorTypeAttrs;

  for (StringRef s : iteratorTypes.getAsValueRange<StringAttr>()) {
    auto maybeIteratorType = symbolizeIteratorType(s);
    if (!maybeIteratorType.has_value())
      return parser.emitError(loc) << "unexpected iterator_type (" << s << ")";

    iteratorTypeAttrs.push_back(
        IteratorTypeAttr::get(parser.getContext(), maybeIteratorType.value()));
  }
  result.attributes.set(getIteratorTypesAttrName(result.name),
                        parser.getBuilder().getArrayAttr(iteratorTypeAttrs));

  if (!result.attributes.get(getKindAttrName(result.name))) {
    result.addAttribute(
        getKindAttrName(result.name),
        CombiningKindAttr::get(result.getContext(),
                               ContractionOp::getDefaultKind()));
  }
  if (masksInfo.empty())
    return success();
  if (masksInfo.size() != 2)
    return parser.emitError(parser.getNameLoc(),
                            "expected zero or exactly 2 vector mask operands");
  auto lhsType = llvm::cast<VectorType>(types[0]);
  auto rhsType = llvm::cast<VectorType>(types[1]);
  auto maskElementType = parser.getBuilder().getI1Type();
  std::array<VectorType, 2> maskTypes = {
      VectorType::Builder(lhsType).setElementType(maskElementType),
      VectorType::Builder(rhsType).setElementType(maskElementType)};
  if (parser.resolveOperands(masksInfo, maskTypes, loc, result.operands))
    return failure();
  return success();
}

void ContractionOp::print(OpAsmPrinter &p) {
  // TODO: Unify printing code with linalg ops.
  auto attrNames = getTraitAttrNames();
  llvm::StringSet<> traitAttrsSet;
  traitAttrsSet.insert_range(attrNames);
  SmallVector<NamedAttribute, 8> attrs;
  for (auto attr : (*this)->getAttrs()) {
    if (attr.getName() == getIteratorTypesAttrName()) {
      auto iteratorTypes =
          llvm::cast<ArrayAttr>(attr.getValue())
              .getAsValueRange<IteratorTypeAttr, IteratorType>();
      // Convert IteratorType enums into the string representation. This is
      // needed, because tests still use the old format when 'iterator_types'
      // attribute is represented as an array of strings.
      // TODO: Remove this conversion once tests are fixed.
      SmallVector<Attribute> iteratorTypeNames = llvm::to_vector(
          llvm::map_range(iteratorTypes, [&](IteratorType t) -> Attribute {
            return StringAttr::get(getContext(), stringifyIteratorType(t));
          }));

      attrs.emplace_back(getIteratorTypesAttrName(),
                         ArrayAttr::get(getContext(), iteratorTypeNames));
    } else if (traitAttrsSet.count(attr.getName().strref()) > 0)
      attrs.push_back(attr);
  }

  auto dictAttr = DictionaryAttr::get(getContext(), attrs);
  p << " " << dictAttr << " " << getLhs() << ", ";
  p << getRhs() << ", " << getAcc();

  p.printOptionalAttrDict((*this)->getAttrs(), attrNames);
  p << " : " << getLhs().getType() << ", " << getRhs().getType() << " into "
    << getResultType();
}

static bool verifyDimMap(VectorType lhsType, VectorType rhsType,
                         const std::vector<std::pair<int64_t, int64_t>> &map) {
  for (auto &dimPair : map) {
    if (dimPair.first < 0 || dimPair.first >= lhsType.getRank() ||
        dimPair.second < 0 || dimPair.second >= rhsType.getRank() ||
        lhsType.getDimSize(dimPair.first) != rhsType.getDimSize(dimPair.second))
      return false;
  }
  return true;
}

static LogicalResult verifyOutputShape(
    ContractionOp op, VectorType lhsType, VectorType rhsType, Type accType,
    Type resType,
    const std::vector<std::pair<int64_t, int64_t>> &contractingDimMap,
    const std::vector<std::pair<int64_t, int64_t>> &batchDimMap) {
  DenseSet<int64_t> lhsContractingDimSet;
  DenseSet<int64_t> rhsContractingDimSet;
  for (auto &dimPair : contractingDimMap) {
    lhsContractingDimSet.insert(dimPair.first);
    rhsContractingDimSet.insert(dimPair.second);
  }
  DenseSet<int64_t> rhsBatchDimSet(llvm::from_range,
                                   llvm::make_second_range(batchDimMap));

  // Add free and batch dimensions from 'lhsType' to 'expectedResultDims'.
  SmallVector<int64_t, 4> expectedResultDims;
  for (int64_t i = 0, e = lhsType.getRank(); i < e; ++i) {
    if (lhsContractingDimSet.count(i) > 0)
      continue;
    expectedResultDims.push_back(lhsType.getDimSize(i));
  }

  // Add free dimensions from 'rhsType' to 'expectedResultDims'.
  for (int64_t i = 0, e = rhsType.getRank(); i < e; ++i) {
    if (rhsContractingDimSet.count(i) > 0 || rhsBatchDimSet.count(i) > 0)
      continue;
    expectedResultDims.push_back(rhsType.getDimSize(i));
  }

  // Verify 'expectedResultDims'.
  if (expectedResultDims.empty()) {
    // No batch or free dimension implies a scalar result.
    if (llvm::isa<VectorType>(resType) || llvm::isa<VectorType>(accType))
      return op.emitOpError("invalid accumulator/result vector shape");
  } else {
    // At least one batch or free dimension implies a vector result.
    auto resVectorType = llvm::dyn_cast<VectorType>(resType);
    auto accVectorType = llvm::dyn_cast<VectorType>(accType);
    if (!resVectorType || !accVectorType)
      return op.emitOpError("invalid accumulator/result vector shape");

    // Infer expected result vector type. Lhs + rhs map and lhs + rhs vector
    // types fully define the result vector type. This assumes the affine maps
    // are well-formed, which must have been verified already.
    MLIRContext *ctx = op.getContext();
    AffineMap lhsMap = op.getIndexingMapsArray()[0];
    AffineMap rhsMap = op.getIndexingMapsArray()[1];
    if (getUnusedDimsBitVector({lhsMap, rhsMap}).any())
      return op.emitOpError(
          "expected all dimensions to be either a LHS or a RHS dimension");
    SmallVector<AffineExpr, 4> extents(lhsMap.getNumInputs());
    for (auto pair :
         {std::make_pair(lhsType, lhsMap), std::make_pair(rhsType, rhsMap)}) {
      VectorType v = pair.first;
      auto map = pair.second;
      for (unsigned idx = 0, e = v.getRank(); idx < e; ++idx) {
        unsigned pos = map.getDimPosition(idx);
        if (!extents[pos])
          extents[pos] = getAffineConstantExpr(v.getShape()[idx], ctx);
      }
    }
    if (!llvm::all_of(extents, [](AffineExpr e) { return e; }))
      return op.emitOpError("expected all dimensions to get an extent as "
                            "either a LHS or a RHS dimension");

    AffineMap resMap = op.getIndexingMapsArray()[2];
    auto extentsMap = AffineMap::get(/*dimCount=*/extents.size(),
                                     /*symbolCount=*/0, extents, ctx);
    // Compose the resMap with the extentsMap, which is a constant map.
    AffineMap expectedMap = simplifyAffineMap(resMap.compose(extentsMap));
    assert(llvm::all_of(expectedMap.getResults(),
                        llvm::IsaPred<AffineConstantExpr>) &&
           "expected constant extent along all dimensions.");
    // Extract the expected shape and build the type.
    auto expectedShape = llvm::to_vector<4>(
        llvm::map_range(expectedMap.getResults(), [](AffineExpr e) {
          return cast<AffineConstantExpr>(e).getValue();
        }));
    auto expected =
        VectorType::get(expectedShape, resVectorType.getElementType(),
                        resVectorType.getScalableDims());
    if (resVectorType != expected || accVectorType != expected)
      return op.emitOpError(
                 "invalid accumulator/result vector shape, expected: ")
             << expected;
  }
  return success();
}

LogicalResult ContractionOp::verify() {
  VectorType lhsType = getLhsType();
  VectorType rhsType = getRhsType();
  Type accType = getAccType();
  Type resType = getResultType();

  if (llvm::isa<IntegerType>(lhsType.getElementType())) {
    if (!lhsType.getElementType().isSignlessInteger())
      return emitOpError("only supports signless integer types");
  }

  // Verify that an indexing map was specified for each vector operand.
  if (getIndexingMapsArray().size() != 3)
    return emitOpError("expected an indexing map for each vector operand");

  // Verify that each index map has 'numIterators' inputs, no symbols, and
  // that the number of map outputs equals the rank of its associated
  // vector operand.
  unsigned numIterators = getIteratorTypes().getValue().size();
  for (const auto &it : llvm::enumerate(getIndexingMapsArray())) {
    auto index = it.index();
    auto map = it.value();
    if (map.getNumSymbols() != 0)
      return emitOpError("expected indexing map ")
             << index << " to have no symbols";
    auto vectorType = llvm::dyn_cast<VectorType>(getOperand(index).getType());
    unsigned rank = vectorType ? vectorType.getShape().size() : 0;
    // Verify that the map has the right number of inputs, outputs, and indices.
    // This also correctly accounts for (..) -> () for rank-0 results.
    if (map.getNumDims() != numIterators)
      return emitOpError("expected indexing map ")
             << index << " to have " << numIterators << " number of inputs";
    if (map.getNumResults() != rank)
      return emitOpError("expected indexing map ")
             << index << " to have " << rank << " number of outputs";
    if (!map.isProjectedPermutation())
      return emitOpError("expected indexing map ")
             << index << " to be a projected permutation of its inputs";
  }

  auto contractingDimMap = getContractingDimMap();
  auto batchDimMap = getBatchDimMap();

  // Verify at least one contracting dimension pair was specified.
  if (contractingDimMap.empty())
    return emitOpError("expected at least one contracting dimension pair");

  // Verify contracting dimension map was properly constructed.
  if (!verifyDimMap(lhsType, rhsType, contractingDimMap))
    return emitOpError("invalid contracting dimension map");

  // Verify batch dimension map was properly constructed.
  if (!verifyDimMap(lhsType, rhsType, batchDimMap))
    return emitOpError("invalid batch dimension map");

  // Verify 'accType' and 'resType' shape.
  if (failed(verifyOutputShape(*this, lhsType, rhsType, accType, resType,
                               contractingDimMap, batchDimMap)))
    return failure();

  // Verify supported combining kind.
  auto vectorType = llvm::dyn_cast<VectorType>(resType);
  auto elementType = vectorType ? vectorType.getElementType() : resType;
  if (!isSupportedCombiningKind(getKind(), elementType))
    return emitOpError("unsupported contraction type");

  // Delayed calling of IndexingMapOpInterface::verifyImpl.
  return cast<IndexingMapOpInterface>(this->getOperation()).verifyImpl();
}

// MaskableOpInterface methods.

/// Returns the mask type expected by this operation. Mostly used for
/// verification purposes. It requires the operation to be vectorized."
Type ContractionOp::getExpectedMaskType() {
  auto indexingMaps = this->getIndexingMapsArray();
  AffineMap lhsIdxMap = indexingMaps[0];
  AffineMap rhsIdxMap = indexingMaps[1];
  VectorType lhsType = this->getLhsType();
  VectorType rhsType = this->getRhsType();

  unsigned numVecDims = lhsIdxMap.getNumDims();
  SmallVector<int64_t> maskShape(numVecDims, ShapedType::kDynamic);
  SmallVector<bool> maskShapeScalableDims(numVecDims, false);

  // Using the information in the indexing maps, extract the size of each
  // dimension in the vector.contract operation from the two input operands.
  for (auto [dimIdx, dimSize] : llvm::enumerate(lhsType.getShape())) {
    maskShape[lhsIdxMap.getDimPosition(dimIdx)] = dimSize;
    maskShapeScalableDims[lhsIdxMap.getDimPosition(dimIdx)] =
        lhsType.getScalableDims()[dimIdx];
  }
  for (auto [dimIdx, dimSize] : llvm::enumerate(rhsType.getShape())) {
    maskShape[rhsIdxMap.getDimPosition(dimIdx)] = dimSize;
    maskShapeScalableDims[rhsIdxMap.getDimPosition(dimIdx)] =
        rhsType.getScalableDims()[dimIdx];
  }

  assert(ShapedType::isStaticShape(maskShape) &&
         "Mask shape couldn't be computed");

  return VectorType::get(maskShape,
                         IntegerType::get(lhsType.getContext(), /*width=*/1),
                         maskShapeScalableDims);
}

SmallVector<StringRef> ContractionOp::getTraitAttrNames() {
  return SmallVector<StringRef>{getIndexingMapsAttrName(),
                                getIteratorTypesAttrName(), getKindAttrName()};
}

static int64_t getResultIndex(AffineMap map, AffineExpr targetExpr) {
  for (int64_t i = 0, e = map.getNumResults(); i < e; ++i)
    if (targetExpr == map.getResult(i))
      return i;
  return -1;
}

static std::vector<std::pair<int64_t, int64_t>>
getDimMap(ArrayRef<AffineMap> indexingMaps, ArrayAttr iteratorTypes,
          IteratorType targetIteratorType, MLIRContext *context) {
  std::vector<std::pair<int64_t, int64_t>> dimMap;
  for (const auto &it : llvm::enumerate(iteratorTypes)) {
    auto iteratorType = llvm::cast<IteratorTypeAttr>(it.value()).getValue();
    if (iteratorType != targetIteratorType)
      continue;
    // Search lhs/rhs map results for 'targetExpr'.
    auto targetExpr = getAffineDimExpr(it.index(), context);
    int64_t lhsDim = getResultIndex(indexingMaps[0], targetExpr);
    int64_t rhsDim = getResultIndex(indexingMaps[1], targetExpr);
    if (lhsDim >= 0 && rhsDim >= 0)
      dimMap.emplace_back(lhsDim, rhsDim);
  }
  return dimMap;
}

void ContractionOp::getIterationBounds(
    SmallVectorImpl<int64_t> &iterationBounds) {
  auto lhsShape = getLhsType().getShape();
  auto resVectorType = llvm::dyn_cast<VectorType>(getResultType());
  SmallVector<AffineMap, 4> indexingMaps(getIndexingMapsArray());
  for (const auto &it : llvm::enumerate(getIteratorTypes())) {
    // Search lhs/rhs map results for 'targetExpr'.
    auto targetExpr = getAffineDimExpr(it.index(), getContext());
    auto iteratorType = llvm::cast<IteratorTypeAttr>(it.value()).getValue();
    if (iteratorType == IteratorType::reduction) {
      // Get reduction dim size from lhs shape (same size in rhsShape).
      int64_t lhsDimIndex = getResultIndex(indexingMaps[0], targetExpr);
      assert(lhsDimIndex >= 0);
      iterationBounds.push_back(lhsShape[lhsDimIndex]);
      continue;
    }
    // Get parallel dimension size from result shape.
    int64_t resDimIndex = getResultIndex(indexingMaps[2], targetExpr);
    assert(resDimIndex >= 0);
    assert(resVectorType != nullptr);
    iterationBounds.push_back(resVectorType.getShape()[resDimIndex]);
  }
}

void ContractionOp::getIterationIndexMap(
    std::vector<DenseMap<int64_t, int64_t>> &iterationIndexMap) {
  unsigned numMaps = getIndexingMapsArray().size();
  iterationIndexMap.resize(numMaps);
  for (const auto &it : llvm::enumerate(getIndexingMapsArray())) {
    auto index = it.index();
    auto map = it.value();
    for (unsigned i = 0, e = map.getNumResults(); i < e; ++i) {
      auto dim = cast<AffineDimExpr>(map.getResult(i));
      iterationIndexMap[index][dim.getPosition()] = i;
    }
  }
}

std::vector<std::pair<int64_t, int64_t>> ContractionOp::getContractingDimMap() {
  SmallVector<AffineMap, 4> indexingMaps(getIndexingMapsArray());
  return getDimMap(indexingMaps, getIteratorTypes(), IteratorType::reduction,
                   getContext());
}

std::vector<std::pair<int64_t, int64_t>> ContractionOp::getBatchDimMap() {
  SmallVector<AffineMap, 4> indexingMaps(getIndexingMapsArray());
  return getDimMap(indexingMaps, getIteratorTypes(), IteratorType::parallel,
                   getContext());
}

std::optional<SmallVector<int64_t, 4>> ContractionOp::getShapeForUnroll() {
  SmallVector<int64_t, 4> shape;
  getIterationBounds(shape);
  return shape;
}

/// Return a fused vector::ContractionOp which represents a patterns such as:
///
/// ```mlir
///    %c0 = vector.constant 0: ...
///    %c = vector.contract %a, %b, %c0: ...
///    %e = add %c, %d: ...
/// ```
///
/// by:
///
/// ```mlir
///    %e = vector.contract %a, %b, %d: ...
/// ```
///
/// Return null if the canonicalization does not apply.
// TODO: This should be a folding of Add into Contract in core but while they
// live in different dialects, it is not possible without unnatural
// dependencies.
template <typename AddOpType>
struct CanonicalizeContractAdd : public OpRewritePattern<AddOpType> {
  using OpRewritePattern<AddOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOpType addOp,
                                PatternRewriter &rewriter) const override {
    auto canonicalize = [&](Value maybeContraction,
                            Value otherOperand) -> vector::ContractionOp {
      vector::ContractionOp contractionOp =
          dyn_cast_or_null<vector::ContractionOp>(
              maybeContraction.getDefiningOp());
      if (!contractionOp)
        return vector::ContractionOp();
      if (auto maybeZero = dyn_cast_or_null<arith::ConstantOp>(
              contractionOp.getAcc().getDefiningOp())) {
        if (maybeZero.getValue() ==
            rewriter.getZeroAttr(contractionOp.getAcc().getType())) {
          IRMapping bvm;
          bvm.map(contractionOp.getAcc(), otherOperand);
          auto newContraction =
              cast<vector::ContractionOp>(rewriter.clone(*contractionOp, bvm));
          rewriter.replaceOp(addOp, newContraction.getResult());
          return newContraction;
        }
      }
      return vector::ContractionOp();
    };

    Value a = addOp->getOperand(0), b = addOp->getOperand(1);
    vector::ContractionOp contract = canonicalize(a, b);
    contract = contract ? contract : canonicalize(b, a);
    return contract ? success() : failure();
  }
};

void ContractionOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                MLIRContext *context) {
  results.add<CanonicalizeContractAdd<arith::AddIOp>,
              CanonicalizeContractAdd<arith::AddFOp>>(context);
}

// Returns `true` if `index` is either within [0, maxIndex) or equal to
// `poisonValue`.
static bool isValidPositiveIndexOrPoison(int64_t index, int64_t poisonValue,
                                         int64_t maxIndex) {
  return index == poisonValue || (index >= 0 && index < maxIndex);
}

//===----------------------------------------------------------------------===//
// ExtractOp
//===----------------------------------------------------------------------===//

void ExtractOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                  SetIntRangeFn setResultRanges) {
  setResultRanges(getResult(), argRanges.front());
}

void vector::ExtractOp::build(OpBuilder &builder, OperationState &result,
                              Value source) {
  auto vectorTy = cast<VectorType>(source.getType());
  build(builder, result, source, SmallVector<int64_t>(vectorTy.getRank(), 0));
}

void vector::ExtractOp::build(OpBuilder &builder, OperationState &result,
                              Value source, int64_t position) {
  build(builder, result, source, ArrayRef<int64_t>{position});
}

void vector::ExtractOp::build(OpBuilder &builder, OperationState &result,
                              Value source, OpFoldResult position) {
  build(builder, result, source, ArrayRef<OpFoldResult>{position});
}

void vector::ExtractOp::build(OpBuilder &builder, OperationState &result,
                              Value source, ArrayRef<int64_t> position) {
  build(builder, result, source, /*dynamic_position=*/ArrayRef<Value>(),
        builder.getDenseI64ArrayAttr(position));
}

void vector::ExtractOp::build(OpBuilder &builder, OperationState &result,
                              Value source, ArrayRef<OpFoldResult> position) {
  SmallVector<int64_t> staticPos;
  SmallVector<Value> dynamicPos;
  dispatchIndexOpFoldResults(position, dynamicPos, staticPos);
  build(builder, result, source, dynamicPos,
        builder.getDenseI64ArrayAttr(staticPos));
}

LogicalResult
ExtractOp::inferReturnTypes(MLIRContext *, std::optional<Location>,
                            ExtractOp::Adaptor adaptor,
                            SmallVectorImpl<Type> &inferredReturnTypes) {
  auto vectorType = llvm::cast<VectorType>(adaptor.getVector().getType());
  if (static_cast<int64_t>(adaptor.getStaticPosition().size()) ==
      vectorType.getRank()) {
    inferredReturnTypes.push_back(vectorType.getElementType());
  } else {
    auto n = std::min<size_t>(adaptor.getStaticPosition().size(),
                              vectorType.getRank());
    inferredReturnTypes.push_back(VectorType::get(
        vectorType.getShape().drop_front(n), vectorType.getElementType(),
        vectorType.getScalableDims().drop_front(n)));
  }
  return success();
}

bool ExtractOp::isCompatibleReturnTypes(TypeRange l, TypeRange r) {
  // Allow extracting 1-element vectors instead of scalars.
  auto isCompatible = [](TypeRange l, TypeRange r) {
    auto vectorType = llvm::dyn_cast<VectorType>(l.front());
    return vectorType && vectorType.getShape().equals({1}) &&
           vectorType.getElementType() == r.front();
  };
  if (l.size() == 1 && r.size() == 1 &&
      (isCompatible(l, r) || isCompatible(r, l)))
    return true;
  return l == r;
}

LogicalResult vector::ExtractOp::verify() {
  if (auto resTy = dyn_cast<VectorType>(getResult().getType()))
    if (resTy.getRank() == 0)
      return emitError(
          "expected a scalar instead of a 0-d vector as the result type");

  // Note: This check must come before getMixedPosition() to prevent a crash.
  auto dynamicMarkersCount =
      llvm::count_if(getStaticPosition(), ShapedType::isDynamic);
  if (static_cast<size_t>(dynamicMarkersCount) != getDynamicPosition().size())
    return emitOpError(
        "mismatch between dynamic and static positions (kDynamic marker but no "
        "corresponding dynamic position) -- this can only happen due to an "
        "incorrect fold/rewrite");
  auto position = getMixedPosition();
  if (position.size() > static_cast<unsigned>(getSourceVectorType().getRank()))
    return emitOpError(
        "expected position attribute of rank no greater than vector rank");
  for (auto [idx, pos] : llvm::enumerate(position)) {
    if (auto attr = dyn_cast<Attribute>(pos)) {
      int64_t constIdx = cast<IntegerAttr>(attr).getInt();
      if (!isValidPositiveIndexOrPoison(
              constIdx, kPoisonIndex, getSourceVectorType().getDimSize(idx))) {
        return emitOpError("expected position attribute #")
               << (idx + 1)
               << " to be a non-negative integer smaller than the "
                  "corresponding vector dimension or poison (-1)";
      }
    }
  }
  return success();
}

template <typename IntType>
static SmallVector<IntType> extractVector(ArrayAttr arrayAttr) {
  return llvm::to_vector<4>(llvm::map_range(
      arrayAttr.getAsRange<IntegerAttr>(),
      [](IntegerAttr attr) { return static_cast<IntType>(attr.getInt()); }));
}

/// Fold the result of chains of ExtractOp in place by simply concatenating the
/// positions.
static LogicalResult foldExtractOpFromExtractChain(ExtractOp extractOp) {
  if (!extractOp.getVector().getDefiningOp<ExtractOp>())
    return failure();

  // TODO: Canonicalization for dynamic position not implemented yet.
  if (extractOp.hasDynamicPosition())
    return failure();

  SmallVector<int64_t> globalPosition;
  ExtractOp currentOp = extractOp;
  ArrayRef<int64_t> extrPos = currentOp.getStaticPosition();
  globalPosition.append(extrPos.rbegin(), extrPos.rend());
  while (ExtractOp nextOp = currentOp.getVector().getDefiningOp<ExtractOp>()) {
    currentOp = nextOp;
    // TODO: Canonicalization for dynamic position not implemented yet.
    if (currentOp.hasDynamicPosition())
      return failure();
    ArrayRef<int64_t> extrPos = currentOp.getStaticPosition();
    globalPosition.append(extrPos.rbegin(), extrPos.rend());
  }
  extractOp.setOperand(0, currentOp.getVector());
  // OpBuilder is only used as a helper to build an I64ArrayAttr.
  OpBuilder b(extractOp.getContext());
  std::reverse(globalPosition.begin(), globalPosition.end());
  extractOp.setStaticPosition(globalPosition);
  return success();
}

namespace {
/// Fold an ExtractOp that is fed by a chain of InsertOps and TransposeOps.
/// Walk back a chain of InsertOp/TransposeOp until we hit a match.
/// Compose TransposeOp permutations as we walk back.
/// This helper class keeps an updated extraction position `extractPosition`
/// with extra trailing sentinels.
/// The sentinels encode the internal transposition status of the result vector.
/// As we iterate, extractPosition is permuted and updated.
class ExtractFromInsertTransposeChainState {
public:
  ExtractFromInsertTransposeChainState(ExtractOp e);

  /// Iterate over producing insert and transpose ops until we find a fold.
  Value fold();

private:
  /// Return true if the vector at position `a` is contained within the vector
  /// at position `b`. Under insert/extract semantics, this is the same as `a`
  /// is a prefix of `b`.
  template <typename ContainerA, typename ContainerB>
  bool isContainedWithin(const ContainerA &a, const ContainerB &b) {
    return a.size() <= b.size() &&
           std::equal(a.begin(), a.begin() + a.size(), b.begin());
  }

  /// Return true if the vector at position `a` intersects the vector at
  /// position `b`. Under insert/extract semantics, this is the same as equality
  /// of all entries of `a` that are >=0 with the corresponding entries of b.
  /// Comparison is on the common prefix (i.e. zip).
  template <typename ContainerA, typename ContainerB>
  bool intersectsWhereNonNegative(const ContainerA &a, const ContainerB &b) {
    for (auto [elemA, elemB] : llvm::zip(a, b)) {
      if (elemA < 0 || elemB < 0)
        continue;
      if (elemA != elemB)
        return false;
    }
    return true;
  }

  /// Folding is only possible in the absence of an internal permutation in the
  /// result vector.
  bool canFold() {
    return (sentinels == ArrayRef(extractPosition).drop_front(extractedRank));
  }

  // Helper to get the next defining op of interest.
  void updateStateForNextIteration(Value v) {
    nextInsertOp = v.getDefiningOp<vector::InsertOp>();
    nextTransposeOp = v.getDefiningOp<vector::TransposeOp>();
  };

  // Case 1. If we hit a transpose, just compose the map and iterate.
  // Invariant: insert + transpose do not change rank, we can always compose.
  LogicalResult handleTransposeOp();

  // Case 2: the insert position matches extractPosition exactly, early return.
  LogicalResult handleInsertOpWithMatchingPos(Value &res);

  /// Case 3: if the insert position is a prefix of extractPosition, extract a
  /// portion of the source of the insert.
  /// Example:
  /// ```
  /// %ins = vector.insert %source, %vest[1]: vector<3x4> into vector<2x3x4x5>
  /// // extractPosition == [1, 2, 3]
  /// %ext = vector.extract %ins[1, 0]: vector<5> from vector<3x4x5>
  /// // can fold to vector.extract %source[0, 3]
  /// %ext = vector.extract %source[3]: vector<6> from vector<5x6>
  /// ```
  /// To traverse through %source, we need to set the leading dims to 0 and
  /// drop the extra leading dims.
  /// This method updates the internal state.
  LogicalResult handleInsertOpWithPrefixPos(Value &res);

  /// Try to fold in place to extract(source, extractPosition) and return the
  /// folded result. Return null if folding is not possible (e.g. due to an
  /// internal transposition in the result).
  Value tryToFoldExtractOpInPlace(Value source);

  ExtractOp extractOp;
  int64_t vectorRank;
  int64_t extractedRank;

  InsertOp nextInsertOp;
  TransposeOp nextTransposeOp;

  /// Sentinel values that encode the internal permutation status of the result.
  /// They are set to (-1, ... , -k) at the beginning and appended to
  /// `extractPosition`.
  /// In the end, the tail of `extractPosition` must be exactly `sentinels` to
  /// ensure that there is no internal transposition.
  /// Internal transposition cannot be accounted for with a folding pattern.
  // TODO: We could relax the internal transposition with an extra transposition
  // operation in a future canonicalizer.
  SmallVector<int64_t> sentinels;
  SmallVector<int64_t> extractPosition;
};
} // namespace

ExtractFromInsertTransposeChainState::ExtractFromInsertTransposeChainState(
    ExtractOp e)
    : extractOp(e), vectorRank(extractOp.getSourceVectorType().getRank()),
      extractedRank(extractOp.getNumIndices()) {
  assert(vectorRank >= extractedRank && "Extracted position overflow");
  sentinels.reserve(vectorRank - extractedRank);
  for (int64_t i = 0, e = vectorRank - extractedRank; i < e; ++i)
    sentinels.push_back(-(i + 1));
  extractPosition.assign(extractOp.getStaticPosition().begin(),
                         extractOp.getStaticPosition().end());
  llvm::append_range(extractPosition, sentinels);
}

// Case 1. If we hit a transpose, just compose the map and iterate.
// Invariant: insert + transpose do not change rank, we can always compose.
LogicalResult ExtractFromInsertTransposeChainState::handleTransposeOp() {
  // TODO: Canonicalization for dynamic position not implemented yet.
  if (extractOp.hasDynamicPosition())
    return failure();

  if (!nextTransposeOp)
    return failure();
  AffineMap m = inversePermutation(AffineMap::getPermutationMap(
      nextTransposeOp.getPermutation(), extractOp.getContext()));
  extractPosition = applyPermutationMap(m, ArrayRef(extractPosition));
  return success();
}

// Case 2: the insert position matches extractPosition exactly, early return.
LogicalResult
ExtractFromInsertTransposeChainState::handleInsertOpWithMatchingPos(
    Value &res) {
  // TODO: Canonicalization for dynamic position not implemented yet.
  if (extractOp.hasDynamicPosition() || nextInsertOp.hasDynamicPosition())
    return failure();

  ArrayRef<int64_t> insertedPos = nextInsertOp.getStaticPosition();
  if (insertedPos != llvm::ArrayRef(extractPosition).take_front(extractedRank))
    return failure();
  // Case 2.a. early-exit fold.
  res = nextInsertOp.getValueToStore();
  // Case 2.b. if internal transposition is present, canFold will be false.
  return success(canFold());
}

/// Case 3: if inserted position is a prefix of extractPosition,
/// extract a portion of the source of the insertion.
/// This method updates the internal state.
LogicalResult
ExtractFromInsertTransposeChainState::handleInsertOpWithPrefixPos(Value &res) {
  // TODO: Canonicalization for dynamic position not implemented yet.
  if (extractOp.hasDynamicPosition() || nextInsertOp.hasDynamicPosition())
    return failure();

  ArrayRef<int64_t> insertedPos = nextInsertOp.getStaticPosition();
  if (!isContainedWithin(insertedPos, extractPosition))
    return failure();
  // Set leading dims to zero.
  std::fill_n(extractPosition.begin(), insertedPos.size(), 0);
  // Drop extra leading dims.
  extractPosition.erase(extractPosition.begin(),
                        extractPosition.begin() + insertedPos.size());
  extractedRank = extractPosition.size() - sentinels.size();
  // Case 3.a. early-exit fold (break and delegate to post-while path).
  res = nextInsertOp.getValueToStore();
  // Case 3.b. if internal transposition is present, canFold will be false.
  return success();
}

/// Try to fold in place to extract(source, extractPosition) and return the
/// folded result. Return null if folding is not possible (e.g. due to an
/// internal transposition in the result).
Value ExtractFromInsertTransposeChainState::tryToFoldExtractOpInPlace(
    Value source) {
  // TODO: Canonicalization for dynamic position not implemented yet.
  if (extractOp.hasDynamicPosition())
    return Value();

  // If we can't fold (either internal transposition, or nothing to fold), bail.
  bool nothingToFold = (source == extractOp.getVector());
  if (nothingToFold || !canFold())
    return Value();

  // Otherwise, fold by updating the op inplace and return its result.
  OpBuilder b(extractOp.getContext());
  extractOp.setStaticPosition(
      ArrayRef(extractPosition).take_front(extractedRank));
  extractOp.getVectorMutable().assign(source);
  return extractOp.getResult();
}

/// Iterate over producing insert and transpose ops until we find a fold.
Value ExtractFromInsertTransposeChainState::fold() {
  // TODO: Canonicalization for dynamic position not implemented yet.
  if (extractOp.hasDynamicPosition())
    return Value();

  Value valueToExtractFrom = extractOp.getVector();
  updateStateForNextIteration(valueToExtractFrom);
  while (nextInsertOp || nextTransposeOp) {
    // Case 1. If we hit a transpose, just compose the map and iterate.
    // Invariant: insert + transpose do not change rank, we can always compose.
    if (succeeded(handleTransposeOp())) {
      valueToExtractFrom = nextTransposeOp.getVector();
      updateStateForNextIteration(valueToExtractFrom);
      continue;
    }

    Value result;
    // Case 2: the position match exactly.
    if (succeeded(handleInsertOpWithMatchingPos(result)))
      return result;

    // Case 3: if the inserted position is a prefix of extractPosition, we can
    // just extract a portion of the source of the insert.
    if (succeeded(handleInsertOpWithPrefixPos(result)))
      return tryToFoldExtractOpInPlace(result);

    // Case 4: extractPositionRef intersects insertedPosRef on non-sentinel
    // values. This is a more difficult case and we bail.
    ArrayRef<int64_t> insertedPos = nextInsertOp.getStaticPosition();
    if (isContainedWithin(extractPosition, insertedPos) ||
        intersectsWhereNonNegative(extractPosition, insertedPos))
      return Value();

    // Case 5: No intersection, we forward the extract to insertOp.dest().
    valueToExtractFrom = nextInsertOp.getDest();
    updateStateForNextIteration(valueToExtractFrom);
  }
  // If after all this we can fold, go for it.
  return tryToFoldExtractOpInPlace(valueToExtractFrom);
}

/// Returns true if the operation has a 0-D vector type operand or result.
static bool hasZeroDimVectors(Operation *op) {
  auto hasZeroDimVectorType = [](Type type) -> bool {
    auto vecType = dyn_cast<VectorType>(type);
    return vecType && vecType.getRank() == 0;
  };

  return llvm::any_of(op->getOperandTypes(), hasZeroDimVectorType) ||
         llvm::any_of(op->getResultTypes(), hasZeroDimVectorType);
}

/// All BroadcastOps and SplatOps, as well as ShapeCastOps that only prepend
/// 1s, are considered to be 'broadcastlike'.
static bool isBroadcastLike(Operation *op) {
  if (isa<BroadcastOp, SplatOp>(op))
    return true;

  auto shapeCast = dyn_cast<ShapeCastOp>(op);
  if (!shapeCast)
    return false;

  // Check that shape_cast **only** prepends 1s, like (2,3) -> (1,1,2,3).
  // Checking that the destination shape has a prefix of 1s is not sufficient,
  // for example (2,3) -> (1,3,2) is not broadcastlike. A sufficient condition
  // is that the source shape is a suffix of the destination shape.
  VectorType srcType = shapeCast.getSourceVectorType();
  ArrayRef<int64_t> srcShape = srcType.getShape();
  uint64_t srcRank = srcType.getRank();
  ArrayRef<int64_t> dstShape = shapeCast.getType().getShape();
  return dstShape.size() >= srcRank && dstShape.take_back(srcRank) == srcShape;
}

/// Fold extract(broadcast(X)) to either extract(X) or just X.
///
/// Example:
///
///        broadcast             extract [1][2]
/// (3, 4) --------> (2, 3, 4) ----------------> (4)
///
/// becomes
///                  extract [1]
/// (3,4) -------------------------------------> (4)
///
///
/// The variable names used in this implementation correspond to the above
/// shapes as,
///
/// - (3, 4) is `input` shape.
/// - (2, 3, 4) is `broadcast` shape.
/// - (4) is `extract` shape.
///
/// This folding is possible when the suffix of `input` shape is the same as
/// `extract` shape.
static Value foldExtractFromBroadcast(ExtractOp extractOp) {

  Operation *defOp = extractOp.getVector().getDefiningOp();
  if (!defOp || !isBroadcastLike(defOp))
    return Value();

  Value input = defOp->getOperand(0);

  // Replace extract(broadcast(X)) with X
  if (extractOp.getType() == input.getType())
    return input;

  // Get required types and ranks in the chain
  //    input -> broadcast -> extract
  // (scalars are treated as rank-0).
  auto inputType = llvm::dyn_cast<VectorType>(input.getType());
  auto extractType = llvm::dyn_cast<VectorType>(extractOp.getType());
  unsigned inputRank = inputType ? inputType.getRank() : 0;
  unsigned broadcastRank = extractOp.getSourceVectorType().getRank();
  unsigned extractRank = extractType ? extractType.getRank() : 0;

  // Cannot do without the broadcast if overall the rank increases.
  if (extractRank > inputRank)
    return Value();

  // The above condition guarantees that input is a vector.
  assert(inputType && "input must be a vector type because of previous checks");
  ArrayRef<int64_t> inputShape = inputType.getShape();

  // In the case where there is a broadcast dimension in the suffix, it is not
  // possible to replace extract(broadcast(X)) with extract(X). Example:
  //
  //     broadcast       extract
  // (1) --------> (3,4) ------> (4)
  if (extractType &&
      extractType.getShape() != inputShape.take_back(extractRank))
    return Value();

  // Replace extract(broadcast(X)) with extract(X).
  // First, determine the new extraction position.
  unsigned deltaOverall = inputRank - extractRank;
  unsigned deltaBroadcast = broadcastRank - inputRank;
  SmallVector<OpFoldResult> oldPositions = extractOp.getMixedPosition();
  SmallVector<OpFoldResult> newPositions(deltaOverall);
  IntegerAttr zero = OpBuilder(extractOp.getContext()).getIndexAttr(0);
  for (auto [i, size] : llvm::enumerate(inputShape.take_front(deltaOverall))) {
    newPositions[i] = size == 1 ? zero : oldPositions[i + deltaBroadcast];
  }
  auto [staticPos, dynPos] = decomposeMixedValues(newPositions);
  extractOp->setOperands(
      llvm::to_vector(llvm::concat<Value>(ValueRange(input), dynPos)));
  extractOp.setStaticPosition(staticPos);
  return extractOp.getResult();
}

/// Fold extractOp coming from ShuffleOp.
///
/// Example:
///
///   %shuffle = vector.shuffle %a, %b [0, 8, 7, 15]
///     : vector<8xf32>, vector<8xf32>
///   %extract = vector.extract %shuffle[3] : f32 from vector<4xf32>
/// ->
///   %extract = vector.extract %b[7] : f32 from vector<8xf32>
///
static Value foldExtractFromShuffle(ExtractOp extractOp) {
  // Dynamic positions are not folded as the resulting code would be more
  // complex than the input code.
  if (extractOp.hasDynamicPosition())
    return Value();

  auto shuffleOp = extractOp.getVector().getDefiningOp<ShuffleOp>();
  if (!shuffleOp)
    return Value();

  // TODO: 0-D or multi-dimensional vectors not supported yet.
  if (shuffleOp.getResultVectorType().getRank() != 1)
    return Value();

  int64_t inputVecSize = shuffleOp.getV1().getType().getShape()[0];
  auto shuffleMask = shuffleOp.getMask();
  int64_t extractIdx = extractOp.getStaticPosition()[0];
  int64_t shuffleIdx = shuffleMask[extractIdx];

  // Find the shuffled vector to extract from based on the shuffle index.
  if (shuffleIdx < inputVecSize) {
    extractOp.setOperand(0, shuffleOp.getV1());
    extractOp.setStaticPosition({shuffleIdx});
  } else {
    extractOp.setOperand(0, shuffleOp.getV2());
    extractOp.setStaticPosition({shuffleIdx - inputVecSize});
  }

  return extractOp.getResult();
}

// Fold extractOp with source coming from ShapeCast op.
static Value foldExtractFromShapeCast(ExtractOp extractOp) {
  // TODO: Canonicalization for dynamic position not implemented yet.
  if (extractOp.hasDynamicPosition())
    return Value();

  auto shapeCastOp = extractOp.getVector().getDefiningOp<vector::ShapeCastOp>();
  if (!shapeCastOp)
    return Value();

  // Get the nth dimension size starting from lowest dimension.
  auto getDimReverse = [](VectorType type, int64_t n) {
    return type.getShape().take_back(n + 1).front();
  };
  int64_t destinationRank =
      llvm::isa<VectorType>(extractOp.getType())
          ? llvm::cast<VectorType>(extractOp.getType()).getRank()
          : 0;
  if (destinationRank > shapeCastOp.getSourceVectorType().getRank())
    return Value();
  if (destinationRank > 0) {
    auto destinationType =
        llvm::cast<VectorType>(extractOp.getResult().getType());
    for (int64_t i = 0; i < destinationRank; i++) {
      // The lowest dimension of the destination must match the lowest
      // dimension of the shapecast op source.
      // TODO: This case could be support in a canonicalization pattern.
      if (getDimReverse(shapeCastOp.getSourceVectorType(), i) !=
          getDimReverse(destinationType, i))
        return Value();
    }
  }
  // Extract the strides associated with the extract op vector source. Then use
  // this to calculate a linearized position for the extract.
  SmallVector<int64_t> extractedPos(extractOp.getStaticPosition());
  std::reverse(extractedPos.begin(), extractedPos.end());
  SmallVector<int64_t, 4> strides;
  int64_t stride = 1;
  for (int64_t i = 0, e = extractedPos.size(); i < e; i++) {
    strides.push_back(stride);
    stride *=
        getDimReverse(extractOp.getSourceVectorType(), i + destinationRank);
  }

  int64_t position = linearize(extractedPos, strides);
  // Then extract the strides associated to the shapeCast op vector source and
  // delinearize the position using those strides.
  SmallVector<int64_t, 4> newStrides;
  int64_t numDimension =
      shapeCastOp.getSourceVectorType().getRank() - destinationRank;
  stride = 1;
  for (int64_t i = 0; i < numDimension; i++) {
    newStrides.push_back(stride);
    stride *=
        getDimReverse(shapeCastOp.getSourceVectorType(), i + destinationRank);
  }
  std::reverse(newStrides.begin(), newStrides.end());
  SmallVector<int64_t, 4> newPosition = delinearize(position, newStrides);
  // OpBuilder is only used as a helper to build an I64ArrayAttr.
  OpBuilder b(extractOp.getContext());
  extractOp.setStaticPosition(newPosition);
  extractOp.setOperand(0, shapeCastOp.getSource());
  return extractOp.getResult();
}

/// Fold an ExtractOp from ExtractStridedSliceOp.
static Value foldExtractFromExtractStrided(ExtractOp extractOp) {
  // TODO: Canonicalization for dynamic position not implemented yet.
  if (extractOp.hasDynamicPosition())
    return Value();

  auto extractStridedSliceOp =
      extractOp.getVector().getDefiningOp<vector::ExtractStridedSliceOp>();
  if (!extractStridedSliceOp)
    return Value();

  // 0-D vectors not supported.
  assert(!hasZeroDimVectors(extractOp) && "0-D vectors not supported");
  if (hasZeroDimVectors(extractStridedSliceOp))
    return Value();

  // Return if 'extractStridedSliceOp' has non-unit strides.
  if (extractStridedSliceOp.hasNonUnitStrides())
    return Value();

  // Trim offsets for dimensions fully extracted.
  auto sliceOffsets =
      extractVector<int64_t>(extractStridedSliceOp.getOffsets());
  while (!sliceOffsets.empty()) {
    size_t lastOffset = sliceOffsets.size() - 1;
    if (sliceOffsets.back() != 0 ||
        extractStridedSliceOp.getType().getDimSize(lastOffset) !=
            extractStridedSliceOp.getSourceVectorType().getDimSize(lastOffset))
      break;
    sliceOffsets.pop_back();
  }
  unsigned destinationRank = 0;
  if (auto vecType = llvm::dyn_cast<VectorType>(extractOp.getType()))
    destinationRank = vecType.getRank();
  // The dimensions of the result need to be untouched by the
  // extractStridedSlice op.
  if (destinationRank > extractStridedSliceOp.getSourceVectorType().getRank() -
                            sliceOffsets.size())
    return Value();

  SmallVector<int64_t> extractedPos(extractOp.getStaticPosition());
  assert(extractedPos.size() >= sliceOffsets.size());
  for (size_t i = 0, e = sliceOffsets.size(); i < e; i++)
    extractedPos[i] = extractedPos[i] + sliceOffsets[i];
  extractOp.getVectorMutable().assign(extractStridedSliceOp.getVector());

  // OpBuilder is only used as a helper to build an I64ArrayAttr.
  OpBuilder b(extractOp.getContext());
  extractOp.setStaticPosition(extractedPos);
  return extractOp.getResult();
}

/// Fold extract_op fed from a chain of insertStridedSlice ops.
static Value foldExtractStridedOpFromInsertChain(ExtractOp extractOp) {
  // TODO: Canonicalization for dynamic position not implemented yet.
  if (extractOp.hasDynamicPosition())
    return Value();

  int64_t destinationRank =
      llvm::isa<VectorType>(extractOp.getType())
          ? llvm::cast<VectorType>(extractOp.getType()).getRank()
          : 0;
  auto insertOp = extractOp.getVector().getDefiningOp<InsertStridedSliceOp>();
  if (!insertOp)
    return Value();

  // 0-D vectors not supported.
  assert(!hasZeroDimVectors(extractOp) && "0-D vectors not supported");
  if (hasZeroDimVectors(insertOp))
    return Value();

  while (insertOp) {
    int64_t insertRankDiff = insertOp.getDestVectorType().getRank() -
                             insertOp.getSourceVectorType().getRank();
    if (destinationRank > insertOp.getSourceVectorType().getRank())
      return Value();
    auto insertOffsets = extractVector<int64_t>(insertOp.getOffsets());
    ArrayRef<int64_t> extractOffsets = extractOp.getStaticPosition();

    if (llvm::any_of(insertOp.getStrides(), [](Attribute attr) {
          return llvm::cast<IntegerAttr>(attr).getInt() != 1;
        }))
      return Value();
    bool disjoint = false;
    SmallVector<int64_t, 4> offsetDiffs;
    for (unsigned dim = 0, e = extractOffsets.size(); dim < e; ++dim) {
      int64_t start = insertOffsets[dim];
      int64_t size =
          (dim < insertRankDiff)
              ? 1
              : insertOp.getSourceVectorType().getDimSize(dim - insertRankDiff);
      int64_t end = start + size;
      int64_t offset = extractOffsets[dim];
      // Check if the start of the extract offset is in the interval inserted.
      if (start <= offset && offset < end) {
        if (dim >= insertRankDiff)
          offsetDiffs.push_back(offset - start);
        continue;
      }
      disjoint = true;
      break;
    }
    // The extract element chunk overlap with the vector inserted.
    if (!disjoint) {
      // If any of the inner dimensions are only partially inserted we have a
      // partial overlap.
      int64_t srcRankDiff =
          insertOp.getSourceVectorType().getRank() - destinationRank;
      for (int64_t i = 0; i < destinationRank; i++) {
        if (insertOp.getSourceVectorType().getDimSize(i + srcRankDiff) !=
            insertOp.getDestVectorType().getDimSize(i + srcRankDiff +
                                                    insertRankDiff))
          return Value();
      }
      extractOp.getVectorMutable().assign(insertOp.getValueToStore());
      // OpBuilder is only used as a helper to build an I64ArrayAttr.
      OpBuilder b(extractOp.getContext());
      extractOp.setStaticPosition(offsetDiffs);
      return extractOp.getResult();
    }
    // If the chunk extracted is disjoint from the chunk inserted, keep
    // looking in the insert chain.
    insertOp = insertOp.getDest().getDefiningOp<InsertStridedSliceOp>();
  }
  return Value();
}

/// Try to fold the extraction of a scalar from a vector defined by
/// vector.from_elements. E.g.:
///
/// %0 = vector.from_elements %a, %b : vector<2xf32>
/// %1 = vector.extract %0[0] : f32 from vector<2xf32>
/// ==> fold to %a
static Value foldScalarExtractFromFromElements(ExtractOp extractOp) {
  // Dynamic extractions cannot be folded.
  if (extractOp.hasDynamicPosition())
    return {};

  // Look for extract(from_elements).
  auto fromElementsOp = extractOp.getVector().getDefiningOp<FromElementsOp>();
  if (!fromElementsOp)
    return {};

  // Scalable vectors are not supported.
  auto vecType = llvm::cast<VectorType>(fromElementsOp.getType());
  if (vecType.isScalable())
    return {};

  // Only extractions of scalars are supported.
  int64_t rank = vecType.getRank();
  ArrayRef<int64_t> indices = extractOp.getStaticPosition();
  if (extractOp.getType() != vecType.getElementType())
    return {};
  assert(static_cast<int64_t>(indices.size()) == rank &&
         "unexpected number of indices");

  // Compute flattened/linearized index and fold to operand.
  int flatIndex = 0;
  int stride = 1;
  for (int i = rank - 1; i >= 0; --i) {
    flatIndex += indices[i] * stride;
    stride *= vecType.getDimSize(i);
  }
  return fromElementsOp.getElements()[flatIndex];
}

/// If the dynamic indices of `extractOp` or `insertOp` are in fact constants,
/// then fold it.
template <typename OpType, typename AdaptorType>
static Value extractInsertFoldConstantOp(OpType op, AdaptorType adaptor,
                                         SmallVectorImpl<Value> &operands) {
  std::vector<int64_t> staticPosition = op.getStaticPosition().vec();
  OperandRange dynamicPosition = op.getDynamicPosition();
  ArrayRef<Attribute> dynamicPositionAttr = adaptor.getDynamicPosition();
  ArrayRef<int64_t> vectorShape;
  if constexpr (std::is_same_v<OpType, ExtractOp>)
    vectorShape = op.getSourceVectorType().getShape();
  else
    vectorShape = op.getDestVectorType().getShape();

  // If the dynamic operands is empty, it is returned directly.
  if (!dynamicPosition.size())
    return {};

  // `index` is used to iterate over the `dynamicPosition`.
  unsigned index = 0;

  // `opChange` is a flag. If it is true, it means to update `op` in place.
  bool opChange = false;
  for (unsigned i = 0, e = staticPosition.size(); i < e; ++i) {
    if (ShapedType::isStatic(staticPosition[i]))
      continue;
    Attribute positionAttr = dynamicPositionAttr[index];
    Value position = dynamicPosition[index++];
    if (auto attr = mlir::dyn_cast_if_present<IntegerAttr>(positionAttr)) {
      int64_t value = attr.getInt();
      // Do not fold if the value is out of bounds (-1 signifies a poison
      // value rather than OOB index).
      if (value >= -1 && value < vectorShape[i]) {
        staticPosition[i] = attr.getInt();
        opChange = true;
        continue;
      }
    }
    operands.push_back(position);
  }

  if (opChange) {
    op.setStaticPosition(staticPosition);
    op.getOperation()->setOperands(operands);
    // Return the original result to indicate an in-place folding happened.
    return op.getResult();
  }
  return {};
}

/// Fold an insert or extract operation into an poison value when a poison index
/// is found at any dimension of the static position.
static Attribute foldPoisonIndexInsertExtractOp(MLIRContext *context,
                                                ArrayRef<int64_t> staticPos,
                                                int64_t poisonVal) {
  if (!is_contained(staticPos, poisonVal))
    return {};

  return ub::PoisonAttr::get(context);
}

/// Fold a vector extract from is a poison source.
static Attribute foldPoisonSrcExtractOp(Attribute srcAttr) {
  if (isa_and_nonnull<ub::PoisonAttr>(srcAttr))
    return srcAttr;

  return {};
}

/// Fold a vector extract extracting from a DenseElementsAttr.
static Attribute foldDenseElementsAttrSrcExtractOp(ExtractOp extractOp,
                                                   Attribute srcAttr) {
  auto denseAttr = dyn_cast_if_present<DenseElementsAttr>(srcAttr);
  if (!denseAttr) {
    return {};
  }

  if (denseAttr.isSplat()) {
    Attribute newAttr = denseAttr.getSplatValue<Attribute>();
    if (auto vecDstType = dyn_cast<VectorType>(extractOp.getType()))
      newAttr = DenseElementsAttr::get(vecDstType, newAttr);
    return newAttr;
  }

  auto vecTy = cast<VectorType>(extractOp.getSourceVectorType());
  if (vecTy.isScalable())
    return {};

  if (extractOp.hasDynamicPosition()) {
    return {};
  }

  // Materializing subsets of a large constant array can generally lead to
  // explosion in IR size because of different combination of subsets that
  // can exist. However, vector.extract is a restricted form of subset
  // extract where you can only extract non-overlapping (or the same) subset for
  // a given rank of the subset. Because of this property, the IR size can only
  // increase at most by `rank * size(array)` from a single constant array being
  // extracted by multiple extracts.

  // Calculate the linearized position of the continuous chunk of elements to
  // extract.
  SmallVector<int64_t> completePositions(vecTy.getRank(), 0);
  copy(extractOp.getStaticPosition(), completePositions.begin());
  int64_t startPos =
      linearize(completePositions, computeStrides(vecTy.getShape()));
  auto denseValuesBegin = denseAttr.value_begin<TypedAttr>() + startPos;

  TypedAttr newAttr;
  if (auto resVecTy = dyn_cast<VectorType>(extractOp.getType())) {
    SmallVector<Attribute> elementValues(
        denseValuesBegin, denseValuesBegin + resVecTy.getNumElements());
    newAttr = DenseElementsAttr::get(resVecTy, elementValues);
  } else {
    newAttr = *denseValuesBegin;
  }

  return newAttr;
}

OpFoldResult ExtractOp::fold(FoldAdaptor adaptor) {
  // Fold "vector.extract %v[] : vector<2x2xf32> from vector<2x2xf32>" to %v.
  // Note: Do not fold "vector.extract %v[] : f32 from vector<f32>" (type
  // mismatch).
  if (getNumIndices() == 0 && getVector().getType() == getResult().getType())
    return getVector();
  if (auto res = foldPoisonSrcExtractOp(adaptor.getVector()))
    return res;
  // Fold `arith.constant` indices into the `vector.extract` operation.
  // Do not stop here as this fold may enable subsequent folds that require
  // constant indices.
  SmallVector<Value> operands = {getVector()};
  auto inplaceFolded = extractInsertFoldConstantOp(*this, adaptor, operands);

  if (auto res = foldPoisonIndexInsertExtractOp(
          getContext(), adaptor.getStaticPosition(), kPoisonIndex))
    return res;
  if (auto res = foldDenseElementsAttrSrcExtractOp(*this, adaptor.getVector()))
    return res;
  if (succeeded(foldExtractOpFromExtractChain(*this)))
    return getResult();
  if (auto res = ExtractFromInsertTransposeChainState(*this).fold())
    return res;
  if (auto res = foldExtractFromBroadcast(*this))
    return res;
  if (auto res = foldExtractFromShuffle(*this))
    return res;
  if (auto res = foldExtractFromShapeCast(*this))
    return res;
  if (auto val = foldExtractFromExtractStrided(*this))
    return val;
  if (auto val = foldExtractStridedOpFromInsertChain(*this))
    return val;
  if (auto val = foldScalarExtractFromFromElements(*this))
    return val;

  return inplaceFolded;
}

namespace {

// Pattern to rewrite a ExtractOp(Broadcast) -> Broadcast.
class ExtractOpFromBroadcast final : public OpRewritePattern<ExtractOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ExtractOp extractOp,
                                PatternRewriter &rewriter) const override {

    Operation *defOp = extractOp.getVector().getDefiningOp();
    VectorType outType = dyn_cast<VectorType>(extractOp.getType());
    if (!defOp || !isBroadcastLike(defOp) || !outType)
      return failure();

    Value source = defOp->getOperand(0);
    if (isBroadcastableTo(source.getType(), outType) !=
        BroadcastableToResult::Success)
      return failure();

    rewriter.replaceOpWithNewOp<BroadcastOp>(extractOp, outType, source);
    return success();
  }
};

// Pattern to rewrite a ExtractOp(CreateMask) -> CreateMask.
class ExtractOpFromCreateMask final : public OpRewritePattern<ExtractOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ExtractOp extractOp,
                                PatternRewriter &rewriter) const override {
    auto createMaskOp =
        extractOp.getVector().getDefiningOp<vector::CreateMaskOp>();
    if (!createMaskOp)
      return failure();

    VectorType extractedMaskType =
        llvm::dyn_cast<VectorType>(extractOp.getResult().getType());

    if (!extractedMaskType)
      return failure();

    auto maskOperands = createMaskOp.getOperands();
    ArrayRef<int64_t> extractOpPos = extractOp.getStaticPosition();
    VectorType maskType = createMaskOp.getVectorType();

    bool containsUnknownDims = false;
    bool allFalse = getMaskFormat(createMaskOp) == MaskFormat::AllFalse;

    for (size_t dimIdx = 0; !allFalse && dimIdx < extractOpPos.size();
         dimIdx++) {
      int64_t pos = extractOpPos[dimIdx];
      Value operand = maskOperands[dimIdx];
      auto constantOp = operand.getDefiningOp<arith::ConstantOp>();
      if (!constantOp) {
        // Bounds of this dim unknown.
        containsUnknownDims = true;
        continue;
      }

      int64_t createMaskBound =
          llvm::cast<IntegerAttr>(constantOp.getValue()).getInt();

      if (pos != ShapedType::kDynamic) {
        // If any position is outside the range from the `create_mask`, then the
        // extracted mask will be all-false.
        allFalse |= pos >= createMaskBound;
      } else if (createMaskBound < maskType.getDimSize(dimIdx)) {
        // This dim is not all-true and since this is a dynamic index we don't
        // know if the extraction is within the true or false region.
        // Note: Zero dims have already handled via getMaskFormat().
        containsUnknownDims = true;
      }
    }

    if (allFalse) {
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(
          extractOp, DenseElementsAttr::get(extractedMaskType, false));
    } else if (!containsUnknownDims) {
      rewriter.replaceOpWithNewOp<vector::CreateMaskOp>(
          extractOp, extractedMaskType,
          maskOperands.drop_front(extractOpPos.size()));
    } else {
      return failure();
    }
    return success();
  }
};

// Folds extract(shape_cast(..)) into shape_cast when the total element count
// does not change.
LogicalResult foldExtractFromShapeCastToShapeCast(ExtractOp extractOp,
                                                  PatternRewriter &rewriter) {
  auto castOp = extractOp.getVector().getDefiningOp<ShapeCastOp>();
  if (!castOp)
    return failure();

  VectorType sourceType = castOp.getSourceVectorType();
  auto targetType = dyn_cast<VectorType>(extractOp.getResult().getType());
  if (!targetType)
    return failure();

  if (sourceType.getNumElements() != targetType.getNumElements())
    return failure();

  rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(extractOp, targetType,
                                                   castOp.getSource());
  return success();
}

/// Try to canonicalize the extraction of a subvector from a vector defined by
/// vector.from_elements. E.g.:
///
/// %0 = vector.from_elements %a, %b, %a, %a : vector<2x2xf32>
/// %1 = vector.extract %0[0] : vector<2xf32> from vector<2x2xf32>
/// ==> canonicalize to vector.from_elements %a, %b : vector<2xf32>
LogicalResult foldExtractFromFromElements(ExtractOp extractOp,
                                          PatternRewriter &rewriter) {
  // Dynamic positions are not supported.
  if (extractOp.hasDynamicPosition())
    return failure();

  // Scalar extracts are handled by the folder.
  auto resultType = dyn_cast<VectorType>(extractOp.getType());
  if (!resultType)
    return failure();

  // Look for extracts from a from_elements op.
  auto fromElementsOp = extractOp.getVector().getDefiningOp<FromElementsOp>();
  if (!fromElementsOp)
    return failure();
  VectorType inputType = fromElementsOp.getType();

  // Scalable vectors are not supported.
  if (resultType.isScalable() || inputType.isScalable())
    return failure();

  // Compute the position of first extracted element and flatten/linearize the
  // position.
  SmallVector<int64_t> firstElementPos =
      llvm::to_vector(extractOp.getStaticPosition());
  firstElementPos.append(/*NumInputs=*/resultType.getRank(), /*Elt=*/0);
  int flatIndex = 0;
  int stride = 1;
  for (int64_t i = inputType.getRank() - 1; i >= 0; --i) {
    flatIndex += firstElementPos[i] * stride;
    stride *= inputType.getDimSize(i);
  }

  // Replace the op with a smaller from_elements op.
  rewriter.replaceOpWithNewOp<FromElementsOp>(
      extractOp, resultType,
      fromElementsOp.getElements().slice(flatIndex,
                                         resultType.getNumElements()));
  return success();
}

} // namespace

void ExtractOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<ExtractOpFromBroadcast, ExtractOpFromCreateMask>(context);
  results.add(foldExtractFromShapeCastToShapeCast);
  results.add(foldExtractFromFromElements);
}

static void populateFromInt64AttrArray(ArrayAttr arrayAttr,
                                       SmallVectorImpl<int64_t> &results) {
  for (auto attr : arrayAttr)
    results.push_back(llvm::cast<IntegerAttr>(attr).getInt());
}

//===----------------------------------------------------------------------===//
// FmaOp
//===----------------------------------------------------------------------===//

std::optional<SmallVector<int64_t, 4>> FMAOp::getShapeForUnroll() {
  return llvm::to_vector<4>(getVectorType().getShape());
}

//===----------------------------------------------------------------------===//
// ToElementsOp
//===----------------------------------------------------------------------===//

/// Returns true if all the `operands` are defined by `defOp`.
/// Otherwise, returns false.
static bool haveSameDefiningOp(OperandRange operands, Operation *defOp) {
  if (operands.empty())
    return false;

  return llvm::all_of(operands, [&](Value operand) {
    Operation *currentDef = operand.getDefiningOp();
    return currentDef == defOp;
  });
}

/// Folds vector.to_elements(vector.from_elements(%e0, %e1, ...)) into
/// (%e0, %e1, ...). For example:
///
///   %0 = vector.from_elements %a, %b, %c : vector<3xf32>
///   %1:3 = vector.to_elements %0 : vector<3xf32>
///   user_op %1#0, %1#1, %1#2
///
/// becomes:
///
///   user_op %a, %b, %c
///
static LogicalResult
foldToElementsFromElements(ToElementsOp toElementsOp,
                           SmallVectorImpl<OpFoldResult> &results) {
  auto fromElementsOp =
      toElementsOp.getSource().getDefiningOp<FromElementsOp>();
  if (!fromElementsOp)
    return failure();

  llvm::append_range(results, fromElementsOp.getElements());
  return success();
}

LogicalResult ToElementsOp::fold(FoldAdaptor adaptor,
                                 SmallVectorImpl<OpFoldResult> &results) {
  return foldToElementsFromElements(*this, results);
}

LogicalResult
ToElementsOp::inferReturnTypes(MLIRContext *ctx, std::optional<Location> loc,
                               ToElementsOp::Adaptor adaptor,
                               SmallVectorImpl<Type> &inferredReturnTypes) {
  auto vecType = cast<VectorType>(adaptor.getSource().getType());
  Type elType = vecType.getElementType();
  inferredReturnTypes.append(vecType.getNumElements(), elType);
  return success();
}

//===----------------------------------------------------------------------===//
// FromElementsOp
//===----------------------------------------------------------------------===//

/// Folds vector.from_elements(vector.to_elements(%vector)) into %vector.
///
/// Case #1: Input and output vectors are the same.
///
///   %0:3 = vector.to_elements %a : vector<3xf32>
///   %1 = vector.from_elements %0#0, %0#1, %0#2 : vector<3xf32>
///   user_op %1
///
/// becomes:
///
///   user_op %a
///
static OpFoldResult foldFromElementsToElements(FromElementsOp fromElementsOp) {
  OperandRange fromElemsOperands = fromElementsOp.getElements();
  if (fromElemsOperands.empty())
    return {};

  auto toElementsOp = fromElemsOperands[0].getDefiningOp<ToElementsOp>();
  if (!toElementsOp)
    return {};

  if (!haveSameDefiningOp(fromElemsOperands, toElementsOp))
    return {};

  // Case #1: Input and output vectors are the same. Forward the input vector.
  Value toElementsInput = toElementsOp.getSource();
  if (fromElementsOp.getType() == toElementsInput.getType() &&
      llvm::equal(fromElemsOperands, toElementsOp.getResults())) {
    return toElementsInput;
  }

  // TODO: Support cases with different input and output shapes and different
  // number of elements.

  return {};
}

/// Fold vector.from_elements to a constant when all operands are constants.
/// Example:
///   %c1 = arith.constant 1 : i32
///   %c2 = arith.constant 2 : i32
///   %v = vector.from_elements %c1, %c2 : vector<2xi32>
/// =>
///   %v = arith.constant dense<[1, 2]> : vector<2xi32>
///
static OpFoldResult foldFromElementsToConstant(FromElementsOp fromElementsOp,
                                               ArrayRef<Attribute> elements) {
  if (llvm::any_of(elements, [](Attribute attr) { return !attr; }))
    return {};

  // DenseElementsAttr only supports int/index/float/complex types.
  auto destVecType = fromElementsOp.getDest().getType();
  auto destEltType = destVecType.getElementType();
  if (!destEltType.isIntOrIndexOrFloat() && !isa<ComplexType>(destEltType))
    return {};

  // Constant attributes might have a different type than the return type.
  // Convert them before creating the dense elements attribute.
  auto convertedElements = llvm::map_to_vector(elements, [&](Attribute attr) {
    return convertIntegerAttr(attr, destEltType);
  });

  return DenseElementsAttr::get(destVecType, convertedElements);
}

OpFoldResult FromElementsOp::fold(FoldAdaptor adaptor) {
  if (auto res = foldFromElementsToElements(*this))
    return res;
  if (auto res = foldFromElementsToConstant(*this, adaptor.getElements()))
    return res;

  return {};
}

/// Rewrite vector.from_elements as vector.broadcast if the elements are the
/// same. Example:
///    %0 = vector.from_elements %a, %a, %a : vector<3xf32>
/// =>
///    %0 = vector.broadcast %a : f32 to vector<3xf32>
static LogicalResult
rewriteFromElementsAsBroadcast(FromElementsOp fromElementsOp,
                               PatternRewriter &rewriter) {
  if (!llvm::all_equal(fromElementsOp.getElements()))
    return failure();
  rewriter.replaceOpWithNewOp<BroadcastOp>(
      fromElementsOp, fromElementsOp.getType(),
      fromElementsOp.getElements().front());
  return success();
}

/// Rewrite from_elements on multiple scalar extracts as a shape_cast
/// on a single extract. Example:
///   %0 = vector.extract %source[0, 0] : i8 from vector<2x2xi8>
///   %1 = vector.extract %source[0, 1] : i8 from vector<2x2xi8>
///   %2 = vector.from_elements %0, %1 : vector<2xi8>
///
/// becomes
///   %1 = vector.extract %source[0] : vector<1x2xi8> from vector<2x2xi8>
///   %2 = vector.shape_cast %1 : vector<1x2xi8> to vector<2xi8>
///
/// The requirements for this to be valid are
///
///   i) The elements are extracted from the same vector (%source).
///
///  ii) The elements form a suffix of %source. Specifically, the number
///      of elements is the same as the product of the last N dimension sizes
///      of %source, for some N.
///
/// iii) The elements are extracted contiguously in ascending order.

class FromElementsToShapeCast : public OpRewritePattern<FromElementsOp> {

  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(FromElementsOp fromElements,
                                PatternRewriter &rewriter) const override {

    // Handled by `rewriteFromElementsAsBroadcast`.
    if (fromElements.getType().getNumElements() == 1)
      return failure();

    // The common source that all elements are extracted from, if one exists.
    TypedValue<VectorType> source;
    // The position of the combined extract operation, if one is created.
    ArrayRef<int64_t> combinedPosition;
    // The expected index of extraction of the current element in the loop, if
    // elements are extracted contiguously in ascending order.
    SmallVector<int64_t> expectedPosition;

    for (auto [insertIndex, element] :
         llvm::enumerate(fromElements.getElements())) {

      // Check that the element is from a vector.extract operation.
      auto extractOp = element.getDefiningOp<vector::ExtractOp>();
      if (!extractOp) {
        return rewriter.notifyMatchFailure(fromElements,
                                           "element not from vector.extract");
      }

      // Check condition (i) by checking that all elements have the same source
      // as the first element.
      if (insertIndex == 0) {
        source = extractOp.getVector();
      } else if (extractOp.getVector() != source) {
        return rewriter.notifyMatchFailure(fromElements,
                                           "element from different vector");
      }

      ArrayRef<int64_t> position = extractOp.getStaticPosition();
      int64_t rank = position.size();
      assert(rank == source.getType().getRank() &&
             "scalar extract must have full rank position");

      // Check condition (ii) by checking that the position that the first
      // element is extracted from has sufficient trailing 0s. For example, in
      //
      //   %elm0 = vector.extract %source[1, 0, 0] : i8 from vector<2x3x4xi8>
      //   [...]
      //   %elms = vector.from_elements %elm0, [...] : vector<12xi8>
      //
      // The 2 trailing 0s in the position of extraction of %elm0 cover 3*4 = 12
      // elements, which is the number of elements of %n, so this is valid.
      if (insertIndex == 0) {
        const int64_t numElms = fromElements.getType().getNumElements();
        int64_t numSuffixElms = 1;
        int64_t index = rank;
        while (index > 0 && position[index - 1] == 0 &&
               numSuffixElms < numElms) {
          numSuffixElms *= source.getType().getDimSize(index - 1);
          --index;
        }
        if (numSuffixElms != numElms) {
          return rewriter.notifyMatchFailure(
              fromElements, "elements do not form a suffix of source");
        }
        expectedPosition = llvm::to_vector(position);
        combinedPosition = position.drop_back(rank - index);
      }

      // Check condition (iii).
      else if (expectedPosition != position) {
        return rewriter.notifyMatchFailure(
            fromElements, "elements not in ascending order (static order)");
      }
      increment(expectedPosition, source.getType().getShape());
    }

    auto extracted = rewriter.createOrFold<vector::ExtractOp>(
        fromElements.getLoc(), source, combinedPosition);

    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(
        fromElements, fromElements.getType(), extracted);

    return success();
  }

  /// Increments n-D `indices` by 1 starting from the innermost dimension.
  static void increment(MutableArrayRef<int64_t> indices,
                        ArrayRef<int64_t> shape) {
    for (int dim : llvm::reverse(llvm::seq<int>(0, indices.size()))) {
      indices[dim] += 1;
      if (indices[dim] < shape[dim])
        break;
      indices[dim] = 0;
    }
  }
};

void FromElementsOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
  results.add(rewriteFromElementsAsBroadcast);
  results.add<FromElementsToShapeCast>(context);
}

//===----------------------------------------------------------------------===//
// BroadcastOp
//===----------------------------------------------------------------------===//

void BroadcastOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                    SetIntRangeFn setResultRanges) {
  setResultRanges(getResult(), argRanges.front());
}

std::optional<SmallVector<int64_t, 4>> BroadcastOp::getShapeForUnroll() {
  return llvm::to_vector<4>(getResultVectorType().getShape());
}

/// Return the dimensions of the result vector that were formerly ones in the
/// source tensor and thus correspond to "dim-1" broadcasting.
static llvm::SetVector<int64_t>
computeBroadcastedUnitDims(ArrayRef<int64_t> srcShape,
                           ArrayRef<int64_t> dstShape) {
  int64_t rankDiff = dstShape.size() - srcShape.size();
  int64_t dstDim = rankDiff;
  llvm::SetVector<int64_t> res;
  for (auto [s1, s2] :
       llvm::zip_equal(srcShape, dstShape.drop_front(rankDiff))) {
    if (s1 != s2) {
      assert(s1 == 1 && "expected \"dim-1\" broadcasting");
      res.insert(dstDim);
    }
    ++dstDim;
  }
  return res;
}

llvm::SetVector<int64_t> BroadcastOp::computeBroadcastedUnitDims() {
  // Scalar broadcast is without any unit dim broadcast.
  auto srcVectorType = llvm::dyn_cast<VectorType>(getSourceType());
  if (!srcVectorType)
    return {};
  return ::computeBroadcastedUnitDims(srcVectorType.getShape(),
                                      getResultVectorType().getShape());
}

/// Broadcast `value` to a vector of `dstShape`, knowing that exactly the
/// `broadcastedDims` dimensions in the dstShape are broadcasted.
/// This requires (and asserts) that the broadcast is free of "dim-1"
/// broadcasting.
/// Since vector.broadcast only allows expanding leading dimensions, an extra
/// vector.transpose may be inserted to make the broadcast possible.
/// `value`, `dstShape` and `broadcastedDims` must be properly specified or
/// the helper will assert. This means:
///   1. `dstShape` must not be empty.
///   2. `broadcastedDims` must be confined to [0 .. rank(value.getVectorType)]
///   2. `dstShape` trimmed of the dimensions specified in `broadcastedDims`
//       must match the `value` shape.
Value BroadcastOp::createOrFoldBroadcastOp(
    OpBuilder &b, Value value, ArrayRef<int64_t> dstShape,
    const llvm::SetVector<int64_t> &broadcastedDims) {
  assert(!dstShape.empty() && "unexpected empty dst shape");

  // Well-formedness check.
  SmallVector<int64_t> checkShape;
  for (int i = 0, e = dstShape.size(); i < e; ++i) {
    if (broadcastedDims.contains(i))
      continue;
    checkShape.push_back(dstShape[i]);
  }
  assert(broadcastedDims.size() == dstShape.size() - checkShape.size() &&
         "ill-formed broadcastedDims contains values not confined to "
         "destVectorShape");

  Location loc = value.getLoc();
  Type elementType = getElementTypeOrSelf(value.getType());
  VectorType srcVectorType = llvm::dyn_cast<VectorType>(value.getType());
  VectorType dstVectorType = VectorType::get(dstShape, elementType);

  // Step 2. If scalar -> dstShape broadcast, just do it.
  if (!srcVectorType) {
    assert(checkShape.empty() &&
           "ill-formed createOrFoldBroadcastOp arguments");
    return b.createOrFold<vector::BroadcastOp>(loc, dstVectorType, value);
  }

  assert(srcVectorType.getShape().equals(checkShape) &&
         "ill-formed createOrFoldBroadcastOp arguments");

  // Step 3. Since vector.broadcast only allows creating leading dims,
  //   vector -> dstShape broadcast may require a transpose.
  // Traverse the dims in order and construct:
  //   1. The leading entries of the broadcastShape that is guaranteed to be
  //      achievable by a simple broadcast.
  //   2. The induced permutation for the subsequent vector.transpose that will
  //      bring us from `broadcastShape` back to he desired `dstShape`.
  // If the induced permutation is not the identity, create a vector.transpose.
  SmallVector<int64_t> broadcastShape, permutation(dstShape.size(), -1);
  broadcastShape.reserve(dstShape.size());
  // Consider the example:
  //   srcShape     = 2x4
  //   dstShape     = 1x2x3x4x5
  //   broadcastedDims = [0, 2, 4]
  //
  // We want to build:
  //   broadcastShape  = 1x3x5x2x4
  //   permutation     = [0, 2, 4,                 1, 3]
  //                      ---V---           -----V-----
  //            leading broadcast part      src shape part
  //
  // Note that the trailing dims of broadcastShape are exactly the srcShape
  // by construction.
  // nextSrcShapeDim is used to keep track of where in the permutation the
  // "src shape part" occurs.
  int64_t nextSrcShapeDim = broadcastedDims.size();
  for (int64_t i = 0, e = dstShape.size(); i < e; ++i) {
    if (broadcastedDims.contains(i)) {
      // 3.a. For each dim in the dst shape, if it is a broadcasted dim,
      // bring it to the head of the broadcastShape.
      // It will need to be permuted back from `broadcastShape.size() - 1` into
      // position `i`.
      broadcastShape.push_back(dstShape[i]);
      permutation[i] = broadcastShape.size() - 1;
    } else {
      // 3.b. Otherwise, the dim is not broadcasted, it comes from the src
      // shape and needs to be permuted into position `i`.
      // Don't touch `broadcastShape` here, the whole srcShape will be
      // appended after.
      permutation[i] = nextSrcShapeDim++;
    }
  }
  // 3.c. Append the srcShape.
  llvm::append_range(broadcastShape, srcVectorType.getShape());

  // Ensure there are no "dim-1" broadcasts.
  assert(::computeBroadcastedUnitDims(srcVectorType.getShape(), broadcastShape)
             .empty() &&
         "unexpected \"dim-1\" broadcast");

  VectorType broadcastType = VectorType::get(broadcastShape, elementType);
  assert(vector::isBroadcastableTo(value.getType(), broadcastType) ==
             vector::BroadcastableToResult::Success &&
         "must be broadcastable");
  Value res = b.createOrFold<vector::BroadcastOp>(loc, broadcastType, value);
  // Step 4. If we find any dimension that indeed needs to be permuted,
  // immediately return a new vector.transpose.
  for (int64_t i = 0, e = permutation.size(); i < e; ++i)
    if (permutation[i] != i)
      return b.createOrFold<vector::TransposeOp>(loc, res, permutation);
  // Otherwise return res.
  return res;
}

BroadcastableToResult mlir::vector::isBroadcastableTo(
    Type srcType, VectorType dstVectorType,
    std::pair<VectorDim, VectorDim> *mismatchingDims) {
  // Broadcast scalar to vector of the same element type.
  if (isa<VectorElementTypeInterface>(srcType) && dstVectorType &&
      srcType == getElementTypeOrSelf(dstVectorType))
    return BroadcastableToResult::Success;
  // From now on, only vectors broadcast.
  VectorType srcVectorType = llvm::dyn_cast<VectorType>(srcType);
  if (!srcVectorType)
    return BroadcastableToResult::SourceTypeNotAVector;

  int64_t srcRank = srcVectorType.getRank();
  int64_t dstRank = dstVectorType.getRank();
  if (srcRank > dstRank)
    return BroadcastableToResult::SourceRankHigher;
  // Source has an exact match or singleton value for all trailing dimensions
  // (all leading dimensions are simply duplicated).
  int64_t lead = dstRank - srcRank;
  for (int64_t dimIdx = 0; dimIdx < srcRank; ++dimIdx) {
    // Have mismatching dims (in the sense of vector.broadcast semantics) been
    // encountered?
    bool foundMismatchingDims = false;

    // Check fixed-width dims.
    int64_t srcDim = srcVectorType.getDimSize(dimIdx);
    int64_t dstDim = dstVectorType.getDimSize(lead + dimIdx);
    if (srcDim != 1 && srcDim != dstDim)
      foundMismatchingDims = true;

    // Check scalable flags.
    bool srcDimScalableFlag = srcVectorType.getScalableDims()[dimIdx];
    bool dstDimScalableFlag = dstVectorType.getScalableDims()[lead + dimIdx];
    if ((srcDim == 1 && srcDimScalableFlag && dstDim != 1) ||
        // 1 -> [N] is fine, everything else should be rejected when mixing
        // fixed-width and scalable dims
        (srcDimScalableFlag != dstDimScalableFlag &&
         (srcDim != 1 || srcDimScalableFlag)))
      foundMismatchingDims = true;

    if (foundMismatchingDims) {
      if (mismatchingDims != nullptr) {
        mismatchingDims->first.dim = srcDim;
        mismatchingDims->first.isScalable = srcDimScalableFlag;

        mismatchingDims->second.dim = dstDim;
        mismatchingDims->second.isScalable = dstDimScalableFlag;
      }
      return BroadcastableToResult::DimensionMismatch;
    }
  }

  return BroadcastableToResult::Success;
}

LogicalResult BroadcastOp::verify() {
  std::pair<VectorDim, VectorDim> mismatchingDims;
  BroadcastableToResult res = isBroadcastableTo(
      getSourceType(), getResultVectorType(), &mismatchingDims);
  if (res == BroadcastableToResult::Success)
    return success();
  if (res == BroadcastableToResult::SourceRankHigher)
    return emitOpError("source rank higher than destination rank");
  if (res == BroadcastableToResult::DimensionMismatch) {
    return emitOpError("dimension mismatch (")
           << (mismatchingDims.first.isScalable ? "[" : "")
           << mismatchingDims.first.dim
           << (mismatchingDims.first.isScalable ? "]" : "") << " vs. "
           << (mismatchingDims.second.isScalable ? "[" : "")
           << mismatchingDims.second.dim
           << (mismatchingDims.second.isScalable ? "]" : "") << ")";
  }
  if (res == BroadcastableToResult::SourceTypeNotAVector)
    return emitOpError("source type is not a vector");
  llvm_unreachable("unexpected vector.broadcast op error");
}

// Fold broadcast(shape_cast(x)) into broadcast(x) if x's type is compatible
// with broadcast's result type and shape_cast only adds or removes ones in the
// leading dimensions.
static LogicalResult foldBroadcastOfShapeCast(BroadcastOp broadcastOp) {
  auto srcShapeCast = broadcastOp.getSource().getDefiningOp<ShapeCastOp>();
  if (!srcShapeCast)
    return failure();

  VectorType srcType = srcShapeCast.getSourceVectorType();
  VectorType destType = broadcastOp.getResultVectorType();
  // Check type compatibility.
  if (vector::isBroadcastableTo(srcType, destType) !=
      BroadcastableToResult::Success)
    return failure();

  ArrayRef<int64_t> srcShape = srcType.getShape();
  ArrayRef<int64_t> shapecastShape =
      srcShapeCast.getResultVectorType().getShape();
  // Trailing dimensions should be the same if shape_cast only alters the
  // leading dimensions.
  unsigned numTrailingDims = std::min(srcShape.size(), shapecastShape.size());
  if (!llvm::equal(srcShape.take_back(numTrailingDims),
                   shapecastShape.take_back(numTrailingDims)))
    return failure();

  assert(all_of(srcShape.drop_back(numTrailingDims),
                [](int64_t E) { return E == 1; }) &&
         all_of(shapecastShape.drop_back(numTrailingDims),
                [](int64_t E) { return E == 1; }) &&
         "ill-formed shape_cast");

  broadcastOp.getSourceMutable().assign(srcShapeCast.getSource());
  return success();
}

OpFoldResult BroadcastOp::fold(FoldAdaptor adaptor) {
  if (getSourceType() == getResultVectorType())
    return getSource();
  if (succeeded(foldBroadcastOfShapeCast(*this)))
    return getResult();

  if (!adaptor.getSource())
    return {};
  auto vectorType = getResultVectorType();
  if (auto attr = llvm::dyn_cast<IntegerAttr>(adaptor.getSource())) {
    if (vectorType.getElementType() != attr.getType())
      return {};
    return DenseElementsAttr::get(vectorType, attr);
  }
  if (auto attr = llvm::dyn_cast<FloatAttr>(adaptor.getSource())) {
    if (vectorType.getElementType() != attr.getType())
      return {};
    return DenseElementsAttr::get(vectorType, attr);
  }
  if (auto attr = llvm::dyn_cast<SplatElementsAttr>(adaptor.getSource()))
    return DenseElementsAttr::get(vectorType, attr.getSplatValue<Attribute>());
  if (llvm::dyn_cast<ub::PoisonAttr>(adaptor.getSource()))
    return ub::PoisonAttr::get(getContext());
  return {};
}

namespace {

// Fold broadcast1(broadcast2(x)) into broadcast1(x).
struct BroadcastFolder : public OpRewritePattern<BroadcastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(BroadcastOp broadcastOp,
                                PatternRewriter &rewriter) const override {
    auto srcBroadcast = broadcastOp.getSource().getDefiningOp<BroadcastOp>();
    if (!srcBroadcast)
      return failure();
    rewriter.replaceOpWithNewOp<BroadcastOp>(broadcastOp,
                                             broadcastOp.getResultVectorType(),
                                             srcBroadcast.getSource());
    return success();
  }
};
} // namespace

void BroadcastOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  // BroadcastToShapeCast is not a default canonicalization, it is opt-in by
  // calling `populateCastAwayVectorLeadingOneDimPatterns`
  results.add<BroadcastFolder>(context);
}

//===----------------------------------------------------------------------===//
// ShuffleOp
//===----------------------------------------------------------------------===//

LogicalResult ShuffleOp::verify() {
  VectorType resultType = getResultVectorType();
  VectorType v1Type = getV1VectorType();
  VectorType v2Type = getV2VectorType();
  // Verify ranks.
  int64_t resRank = resultType.getRank();
  int64_t v1Rank = v1Type.getRank();
  int64_t v2Rank = v2Type.getRank();
  bool wellFormed0DCase = v1Rank == 0 && v2Rank == 0 && resRank == 1;
  bool wellFormedNDCase = v1Rank == resRank && v2Rank == resRank;
  if (!wellFormed0DCase && !wellFormedNDCase)
    return emitOpError("rank mismatch");

  // Verify all but leading dimension sizes.
  for (int64_t r = 1; r < v1Rank; ++r) {
    int64_t resDim = resultType.getDimSize(r);
    int64_t v1Dim = v1Type.getDimSize(r);
    int64_t v2Dim = v2Type.getDimSize(r);
    if (resDim != v1Dim || v1Dim != v2Dim)
      return emitOpError("dimension mismatch");
  }
  // Verify mask length.
  ArrayRef<int64_t> mask = getMask();
  int64_t maskLength = mask.size();
  if (maskLength <= 0)
    return emitOpError("invalid mask length");
  if (maskLength != resultType.getDimSize(0))
    return emitOpError("mask length mismatch");
  // Verify all indices.
  int64_t indexSize = (v1Type.getRank() == 0 ? 1 : v1Type.getDimSize(0)) +
                      (v2Type.getRank() == 0 ? 1 : v2Type.getDimSize(0));
  for (auto [idx, maskPos] : llvm::enumerate(mask)) {
    if (!isValidPositiveIndexOrPoison(maskPos, kPoisonIndex, indexSize))
      return emitOpError("mask index #") << (idx + 1) << " out of range";
  }
  return success();
}

LogicalResult
ShuffleOp::inferReturnTypes(MLIRContext *, std::optional<Location>,
                            ShuffleOp::Adaptor adaptor,
                            SmallVectorImpl<Type> &inferredReturnTypes) {
  auto v1Type = llvm::cast<VectorType>(adaptor.getV1().getType());
  auto v1Rank = v1Type.getRank();
  // Construct resulting type: leading dimension matches mask
  // length, all trailing dimensions match the operands.
  SmallVector<int64_t, 4> shape;
  shape.reserve(v1Rank);
  shape.push_back(std::max<size_t>(1, adaptor.getMask().size()));
  // In the 0-D case there is no trailing shape to append.
  if (v1Rank > 0)
    llvm::append_range(shape, v1Type.getShape().drop_front());
  inferredReturnTypes.push_back(
      VectorType::get(shape, v1Type.getElementType()));
  return success();
}

template <typename T>
static bool isStepIndexArray(ArrayRef<T> idxArr, uint64_t begin, size_t width) {
  T expected = begin;
  return idxArr.size() == width && llvm::all_of(idxArr, [&expected](T value) {
           return value == expected++;
         });
}

OpFoldResult vector::ShuffleOp::fold(FoldAdaptor adaptor) {
  auto v1Type = getV1VectorType();
  auto v2Type = getV2VectorType();

  assert(!v1Type.isScalable() && !v2Type.isScalable() &&
         "Vector shuffle does not support scalable vectors");

  // For consistency: 0-D shuffle return type is 1-D, this cannot be a folding
  // but must be a canonicalization into a vector.broadcast.
  if (v1Type.getRank() == 0)
    return {};

  // Fold shuffle V1, V2, [0, 1, 2, 3] : <4xi32>, <2xi32> -> V1.
  auto mask = getMask();
  if (isStepIndexArray(mask, 0, v1Type.getDimSize(0)))
    return getV1();
  // Fold shuffle V1, V2, [4, 5] : <4xi32>, <2xi32> -> V2.
  if (isStepIndexArray(mask, v1Type.getDimSize(0), v2Type.getDimSize(0)))
    return getV2();

  Attribute v1Attr = adaptor.getV1(), v2Attr = adaptor.getV2();
  if (!v1Attr || !v2Attr)
    return {};

  // Fold shuffle poison, poison -> poison.
  bool isV1Poison = isa<ub::PoisonAttr>(v1Attr);
  bool isV2Poison = isa<ub::PoisonAttr>(v2Attr);
  if (isV1Poison && isV2Poison)
    return ub::PoisonAttr::get(getContext());

  // Only support 1-D for now to avoid complicated n-D DenseElementsAttr
  // manipulation.
  if (v1Type.getRank() != 1)
    return {};

  // Poison input attributes need special handling as they are not
  // DenseElementsAttr. If an index is poison, we select the first element of
  // the first non-poison input.
  SmallVector<Attribute> v1Elements, v2Elements;
  Attribute poisonElement;
  if (!isV2Poison) {
    auto v2DenseAttr = dyn_cast<DenseElementsAttr>(v2Attr);
    if (!v2DenseAttr)
      return {};
    v2Elements = to_vector(v2DenseAttr.getValues<Attribute>());
    poisonElement = v2Elements[0];
  }
  if (!isV1Poison) {
    auto v1DenseAttr = dyn_cast<DenseElementsAttr>(v1Attr);
    if (!v1DenseAttr)
      return {};
    v1Elements = to_vector(v1DenseAttr.getValues<Attribute>());
    poisonElement = v1Elements[0];
  }

  SmallVector<Attribute> results;
  int64_t v1Size = v1Type.getDimSize(0);
  for (int64_t maskIdx : mask) {
    Attribute indexedElm;
    // TODO: Return a partial poison vector when supported by the UB dialect.
    if (maskIdx == ShuffleOp::kPoisonIndex) {
      indexedElm = poisonElement;
    } else {
      if (maskIdx < v1Size)
        indexedElm = isV1Poison ? poisonElement : v1Elements[maskIdx];
      else
        indexedElm = isV2Poison ? poisonElement : v2Elements[maskIdx - v1Size];
    }

    results.push_back(indexedElm);
  }

  return DenseElementsAttr::get(getResultVectorType(), results);
}

namespace {

// Pattern to rewrite a 0-D shuffle with [0] or [1] mask returning a 1-D vector
// to a broadcast.
struct Canonicalize0DShuffleOp : public OpRewritePattern<ShuffleOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ShuffleOp shuffleOp,
                                PatternRewriter &rewriter) const override {
    VectorType v1VectorType = shuffleOp.getV1VectorType();
    ArrayRef<int64_t> mask = shuffleOp.getMask();
    if (v1VectorType.getRank() > 0)
      return failure();
    if (mask.size() != 1)
      return failure();
    VectorType resType = VectorType::Builder(v1VectorType).setShape({1});
    if (mask[0] == 0)
      rewriter.replaceOpWithNewOp<vector::BroadcastOp>(shuffleOp, resType,
                                                       shuffleOp.getV1());
    else
      rewriter.replaceOpWithNewOp<vector::BroadcastOp>(shuffleOp, resType,
                                                       shuffleOp.getV2());
    return success();
  }
};

/// Consider the defining operation `defOp` of `value`. If `defOp` is a
/// vector.splat or a vector.broadcast with a scalar operand, return the scalar
/// value that is splatted. Otherwise return null.
///
/// Examples:
///
/// scalar_source --> vector.splat --> value     - return scalar_source
/// scalar_source --> vector.broadcast --> value - return scalar_source
static Value getScalarSplatSource(Value value) {
  // Block argument:
  Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return {};

  // Splat:
  if (auto splat = dyn_cast<vector::SplatOp>(defOp))
    return splat.getInput();

  auto broadcast = dyn_cast<vector::BroadcastOp>(defOp);

  // Not broadcast (and not splat):
  if (!broadcast)
    return {};

  // Broadcast of a vector:
  if (isa<VectorType>(broadcast.getSourceType()))
    return {};

  // Broadcast of a scalar:
  return broadcast.getSource();
}

/// Pattern to rewrite shuffle(splat-like(v), splat-like(v)) as broadcast(v).
class ShuffleSplat final : public OpRewritePattern<ShuffleOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ShuffleOp op,
                                PatternRewriter &rewriter) const override {
    Value splat = getScalarSplatSource(op.getV1());
    if (!splat || getScalarSplatSource(op.getV2()) != splat)
      return failure();

    rewriter.replaceOpWithNewOp<BroadcastOp>(op, op.getType(), splat);
    return success();
  }
};

/// Pattern to rewrite a fixed-size interleave via vector.shuffle to
/// vector.interleave.
class ShuffleInterleave : public OpRewritePattern<ShuffleOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ShuffleOp op,
                                PatternRewriter &rewriter) const override {
    VectorType resultType = op.getResultVectorType();
    if (resultType.isScalable())
      return rewriter.notifyMatchFailure(
          op, "ShuffleOp can't represent a scalable interleave");

    if (resultType.getRank() != 1)
      return rewriter.notifyMatchFailure(
          op, "ShuffleOp can't represent an n-D interleave");

    VectorType sourceType = op.getV1VectorType();
    if (sourceType != op.getV2VectorType() ||
        sourceType.getNumElements() * 2 != resultType.getNumElements()) {
      return rewriter.notifyMatchFailure(
          op, "ShuffleOp types don't match an interleave");
    }

    ArrayRef<int64_t> shuffleMask = op.getMask();
    int64_t resultVectorSize = resultType.getNumElements();
    for (int i = 0, e = resultVectorSize / 2; i < e; ++i) {
      int64_t maskValueA = shuffleMask[i * 2];
      int64_t maskValueB = shuffleMask[(i * 2) + 1];
      if (maskValueA != i || maskValueB != (resultVectorSize / 2) + i)
        return rewriter.notifyMatchFailure(op,
                                           "ShuffleOp mask not interleaving");
    }

    rewriter.replaceOpWithNewOp<InterleaveOp>(op, op.getV1(), op.getV2());
    return success();
  }
};

} // namespace

void ShuffleOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<ShuffleSplat, ShuffleInterleave, Canonicalize0DShuffleOp>(
      context);
}

//===----------------------------------------------------------------------===//
// InsertOp
//===----------------------------------------------------------------------===//

void vector::InsertOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                         SetIntRangeFn setResultRanges) {
  setResultRanges(getResult(), argRanges[0].rangeUnion(argRanges[1]));
}

void vector::InsertOp::build(OpBuilder &builder, OperationState &result,
                             Value source, Value dest) {
  auto vectorTy = cast<VectorType>(dest.getType());
  build(builder, result, source, dest,
        SmallVector<int64_t>(vectorTy.getRank(), 0));
}

void vector::InsertOp::build(OpBuilder &builder, OperationState &result,
                             Value source, Value dest, int64_t position) {
  build(builder, result, source, dest, ArrayRef<int64_t>{position});
}

void vector::InsertOp::build(OpBuilder &builder, OperationState &result,
                             Value source, Value dest, OpFoldResult position) {
  build(builder, result, source, dest, ArrayRef<OpFoldResult>{position});
}

void vector::InsertOp::build(OpBuilder &builder, OperationState &result,
                             Value source, Value dest,
                             ArrayRef<int64_t> position) {
  SmallVector<OpFoldResult> posVals;
  posVals.reserve(position.size());
  llvm::transform(position, std::back_inserter(posVals),
                  [&](int64_t pos) { return builder.getI64IntegerAttr(pos); });
  build(builder, result, source, dest, posVals);
}

void vector::InsertOp::build(OpBuilder &builder, OperationState &result,
                             Value source, Value dest,
                             ArrayRef<OpFoldResult> position) {
  SmallVector<int64_t> staticPos;
  SmallVector<Value> dynamicPos;
  dispatchIndexOpFoldResults(position, dynamicPos, staticPos);
  build(builder, result, source, dest, dynamicPos,
        builder.getDenseI64ArrayAttr(staticPos));
}

LogicalResult InsertOp::verify() {
  if (auto srcTy = dyn_cast<VectorType>(getValueToStoreType()))
    if (srcTy.getRank() == 0)
      return emitError(
          "expected a scalar instead of a 0-d vector as the source operand");

  SmallVector<OpFoldResult> position = getMixedPosition();
  auto destVectorType = getDestVectorType();
  if (position.size() > static_cast<unsigned>(destVectorType.getRank()))
    return emitOpError(
        "expected position attribute of rank no greater than dest vector rank");
  auto srcVectorType = llvm::dyn_cast<VectorType>(getValueToStoreType());
  if (srcVectorType &&
      (static_cast<unsigned>(srcVectorType.getRank()) + position.size() !=
       static_cast<unsigned>(destVectorType.getRank())))
    return emitOpError("expected position attribute rank + source rank to "
                       "match dest vector rank");
  if (!srcVectorType &&
      (position.size() != static_cast<unsigned>(destVectorType.getRank())))
    return emitOpError(
        "expected position attribute rank to match the dest vector rank");
  for (auto [idx, pos] : llvm::enumerate(position)) {
    if (auto attr = dyn_cast<Attribute>(pos)) {
      int64_t constIdx = cast<IntegerAttr>(attr).getInt();
      if (!isValidPositiveIndexOrPoison(constIdx, kPoisonIndex,
                                        destVectorType.getDimSize(idx))) {
        return emitOpError("expected position attribute #")
               << (idx + 1)
               << " to be a non-negative integer smaller than the "
                  "corresponding "
                  "dest vector dimension";
      }
    }
  }
  return success();
}

// Calculate the linearized position of the continuous chunk of elements to
// insert, based on the shape of the value to insert and the positions to insert
// at.
static int64_t calculateInsertPosition(VectorType destTy,
                                       ArrayRef<int64_t> positions) {
  llvm::SmallVector<int64_t> completePositions(destTy.getRank(), 0);
  assert(positions.size() <= completePositions.size() &&
         "positions size must be less than or equal to destTy rank");
  copy(positions, completePositions.begin());
  return linearize(completePositions, computeStrides(destTy.getShape()));
}

namespace {

// If insertOp is only inserting unit dimensions it can be transformed to a
// broadcast.
class InsertToBroadcast final : public OpRewritePattern<InsertOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(InsertOp insertOp,
                                PatternRewriter &rewriter) const override {
    auto srcVecType =
        llvm::dyn_cast<VectorType>(insertOp.getValueToStoreType());
    if (!srcVecType || insertOp.getDestVectorType().getNumElements() !=
                           srcVecType.getNumElements())
      return failure();
    rewriter.replaceOpWithNewOp<BroadcastOp>(
        insertOp, insertOp.getDestVectorType(), insertOp.getValueToStore());
    return success();
  }
};

/// Pattern to rewrite a insert(splat-like(v), splat-like(v)) as broadcast(v).
class InsertSplatToSplat final : public OpRewritePattern<InsertOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(InsertOp op,
                                PatternRewriter &rewriter) const override {

    Value splat = getScalarSplatSource(op.getValueToStore());
    if (!splat || getScalarSplatSource(op.getDest()) != splat)
      return failure();

    rewriter.replaceOpWithNewOp<BroadcastOp>(op, op.getType(), splat);
    return success();
  }
};

/// Pattern to optimize a chain of insertions.
///
/// This pattern identifies chains of vector.insert operations that:
/// 1. Only insert values at static positions.
/// 2. Completely initialize all elements in the resulting vector.
/// 3. All intermediate insert operations have only one use.
///
/// When these conditions are met, the entire chain can be replaced with a
/// single vector.from_elements operation.
///
/// To keep this pattern simple, and avoid spending too much time on matching
/// fragmented insert chains, this pattern only considers the last insert op in
/// the chain.
///
/// Example transformation:
///   %poison = ub.poison : vector<2xi32>
///   %0 = vector.insert %c1, %poison[0] : i32 into vector<2xi32>
///   %1 = vector.insert %c2, %0[1] : i32 into vector<2xi32>
/// ->
///   %result = vector.from_elements %c1, %c2 : vector<2xi32>
class InsertChainFullyInitialized final : public OpRewritePattern<InsertOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(InsertOp op,
                                PatternRewriter &rewriter) const override {

    VectorType destTy = op.getDestVectorType();
    if (destTy.isScalable())
      return failure();
    // Ensure this is the trailing vector.insert op in a chain of inserts.
    for (Operation *user : op.getResult().getUsers())
      if (auto insertOp = dyn_cast<InsertOp>(user))
        if (insertOp.getDest() == op.getResult())
          return failure();

    InsertOp currentOp = op;
    SmallVector<InsertOp> chainInsertOps;
    while (currentOp) {
      // Check cond 1: Dynamic position is not supported.
      if (currentOp.hasDynamicPosition())
        return failure();

      chainInsertOps.push_back(currentOp);
      currentOp = currentOp.getDest().getDefiningOp<InsertOp>();
      // Check cond 3: Intermediate inserts have only one use to avoid an
      // explosion of vectors.
      if (currentOp && !currentOp->hasOneUse())
        return failure();
    }

    int64_t vectorSize = destTy.getNumElements();
    int64_t initializedCount = 0;
    SmallVector<bool> initializedDestIdxs(vectorSize, false);
    SmallVector<int64_t> pendingInsertPos;
    SmallVector<int64_t> pendingInsertSize;
    SmallVector<Value> pendingInsertValues;

    for (auto insertOp : chainInsertOps) {
      // This pattern can do nothing with poison index.
      if (is_contained(insertOp.getStaticPosition(), InsertOp::kPoisonIndex))
        return failure();

      // Calculate the linearized position for inserting elements.
      int64_t insertBeginPosition =
          calculateInsertPosition(destTy, insertOp.getStaticPosition());

      // The valueToStore operand may be a vector or a scalar. Need to handle
      // both cases.
      int64_t insertSize = 1;
      if (auto srcVectorType =
              llvm::dyn_cast<VectorType>(insertOp.getValueToStoreType()))
        insertSize = srcVectorType.getNumElements();

      assert(insertBeginPosition + insertSize <= vectorSize &&
             "insert would overflow the vector");

      for (auto index : llvm::seq<int64_t>(insertBeginPosition,
                                           insertBeginPosition + insertSize)) {
        if (initializedDestIdxs[index])
          continue;
        initializedDestIdxs[index] = true;
        ++initializedCount;
      }

      // Defer the creation of ops before we can make sure the pattern can
      // succeed.
      pendingInsertPos.push_back(insertBeginPosition);
      pendingInsertSize.push_back(insertSize);
      pendingInsertValues.push_back(insertOp.getValueToStore());

      if (initializedCount == vectorSize)
        break;
    }

    // Check cond 2: all positions must be initialized.
    if (initializedCount != vectorSize)
      return failure();

    SmallVector<Value> elements(vectorSize);
    for (auto [insertBeginPosition, insertSize, valueToStore] :
         llvm::reverse(llvm::zip(pendingInsertPos, pendingInsertSize,
                                 pendingInsertValues))) {
      auto srcVectorType = llvm::dyn_cast<VectorType>(valueToStore.getType());

      if (!srcVectorType) {
        elements[insertBeginPosition] = valueToStore;
        continue;
      }

      SmallVector<Type> elementToInsertTypes(insertSize,
                                             srcVectorType.getElementType());
      // Get all elements from the vector in row-major order.
      auto elementsToInsert = rewriter.create<vector::ToElementsOp>(
          op.getLoc(), elementToInsertTypes, valueToStore);
      for (int64_t linearIdx = 0; linearIdx < insertSize; linearIdx++) {
        elements[insertBeginPosition + linearIdx] =
            elementsToInsert.getResult(linearIdx);
      }
    }

    rewriter.replaceOpWithNewOp<vector::FromElementsOp>(op, destTy, elements);
    return success();
  }
};

} // namespace

static Attribute
foldDenseElementsAttrDestInsertOp(InsertOp insertOp, Attribute srcAttr,
                                  Attribute dstAttr,
                                  int64_t maxVectorSizeFoldThreshold) {
  if (insertOp.hasDynamicPosition())
    return {};

  auto denseDst = llvm::dyn_cast_if_present<DenseElementsAttr>(dstAttr);
  if (!denseDst)
    return {};

  if (!srcAttr) {
    return {};
  }

  VectorType destTy = insertOp.getDestVectorType();
  if (destTy.isScalable())
    return {};

  // Make sure we do not create too many large constants.
  if (destTy.getNumElements() > maxVectorSizeFoldThreshold &&
      !insertOp->hasOneUse())
    return {};

  // Calculate the linearized position for inserting elements.
  int64_t insertBeginPosition =
      calculateInsertPosition(destTy, insertOp.getStaticPosition());
  SmallVector<Attribute> insertedValues;
  Type destEltType = destTy.getElementType();

  /// Converts the expected type to an IntegerAttr if there's
  /// a mismatch.
  if (auto denseSource = llvm::dyn_cast<DenseElementsAttr>(srcAttr)) {
    for (auto value : denseSource.getValues<Attribute>())
      insertedValues.push_back(convertIntegerAttr(value, destEltType));
  } else {
    insertedValues.push_back(convertIntegerAttr(srcAttr, destEltType));
  }

  auto allValues = llvm::to_vector(denseDst.getValues<Attribute>());
  copy(insertedValues, allValues.begin() + insertBeginPosition);
  auto newAttr = DenseElementsAttr::get(destTy, allValues);

  return newAttr;
}

/// Folder to replace the `dest` operand of the insert op with the root dest of
/// the insert op use chain.
static Value foldInsertUseChain(InsertOp insertOp) {
  auto destInsert = insertOp.getDest().getDefiningOp<InsertOp>();
  if (!destInsert)
    return {};

  if (insertOp.getMixedPosition() != destInsert.getMixedPosition())
    return {};

  insertOp.setOperand(1, destInsert.getDest());
  return insertOp.getResult();
}

void InsertOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.add<InsertToBroadcast, BroadcastFolder, InsertSplatToSplat,
              InsertChainFullyInitialized>(context);
}

OpFoldResult InsertOp::fold(FoldAdaptor adaptor) {
  // Do not create constants with more than `vectorSizeFoldThreashold` elements,
  // unless the source vector constant has a single use.
  constexpr int64_t vectorSizeFoldThreshold = 256;
  // Fold "vector.insert %v, %dest [] : vector<2x2xf32> from vector<2x2xf32>" to
  // %v. Note: Do not fold "vector.insert %v, %dest [] : f32 into vector<f32>"
  // (type mismatch).
  if (getNumIndices() == 0 && getValueToStoreType() == getType())
    return getValueToStore();
  // Fold `arith.constant` indices into the `vector.insert` operation.
  // Do not stop here as this fold may enable subsequent folds that require
  // constant indices.
  SmallVector<Value> operands = {getValueToStore(), getDest()};
  auto inplaceFolded = extractInsertFoldConstantOp(*this, adaptor, operands);

  if (auto res = foldInsertUseChain(*this))
    return res;
  if (auto res = foldPoisonIndexInsertExtractOp(
          getContext(), adaptor.getStaticPosition(), kPoisonIndex))
    return res;
  if (auto res = foldDenseElementsAttrDestInsertOp(
          *this, adaptor.getValueToStore(), adaptor.getDest(),
          vectorSizeFoldThreshold)) {
    return res;
  }

  return inplaceFolded;
}

//===----------------------------------------------------------------------===//
// InsertStridedSliceOp
//===----------------------------------------------------------------------===//

void InsertStridedSliceOp::build(OpBuilder &builder, OperationState &result,
                                 Value source, Value dest,
                                 ArrayRef<int64_t> offsets,
                                 ArrayRef<int64_t> strides) {
  result.addOperands({source, dest});
  auto offsetsAttr = getVectorSubscriptAttr(builder, offsets);
  auto stridesAttr = getVectorSubscriptAttr(builder, strides);
  result.addTypes(dest.getType());
  result.addAttribute(InsertStridedSliceOp::getOffsetsAttrName(result.name),
                      offsetsAttr);
  result.addAttribute(InsertStridedSliceOp::getStridesAttrName(result.name),
                      stridesAttr);
}

// TODO: Should be moved to Tablegen ConfinedAttr attributes.
template <typename OpType>
static LogicalResult isIntegerArrayAttrSmallerThanShape(OpType op,
                                                        ArrayAttr arrayAttr,
                                                        ArrayRef<int64_t> shape,
                                                        StringRef attrName) {
  if (arrayAttr.size() > shape.size())
    return op.emitOpError("expected ")
           << attrName << " attribute of rank no greater than vector rank";
  return success();
}

// Returns true if all integers in `arrayAttr` are in the half-open [min, max}
// interval. If `halfOpen` is true then the admissible interval is [min, max).
// Otherwise, the admissible interval is [min, max].
template <typename OpType>
static LogicalResult
isIntegerArrayAttrConfinedToRange(OpType op, ArrayAttr arrayAttr, int64_t min,
                                  int64_t max, StringRef attrName,
                                  bool halfOpen = true) {
  for (auto attr : arrayAttr) {
    auto val = llvm::cast<IntegerAttr>(attr).getInt();
    auto upper = max;
    if (!halfOpen)
      upper += 1;
    if (val < min || val >= upper)
      return op.emitOpError("expected ") << attrName << " to be confined to ["
                                         << min << ", " << upper << ")";
  }
  return success();
}

// Returns true if all integers in `arrayAttr` are in the half-open [min, max}
// interval. If `halfOpen` is true then the admissible interval is [min, max).
// Otherwise, the admissible interval is [min, max].
template <typename OpType>
static LogicalResult
isIntegerArrayAttrConfinedToShape(OpType op, ArrayAttr arrayAttr,
                                  ArrayRef<int64_t> shape, StringRef attrName,
                                  bool halfOpen = true, int64_t min = 0) {
  for (auto [index, attrDimPair] :
       llvm::enumerate(llvm::zip_first(arrayAttr, shape))) {
    int64_t val = llvm::cast<IntegerAttr>(std::get<0>(attrDimPair)).getInt();
    int64_t max = std::get<1>(attrDimPair);
    if (!halfOpen)
      max += 1;
    if (val < min || val >= max)
      return op.emitOpError("expected ")
             << attrName << " dimension " << index << " to be confined to ["
             << min << ", " << max << ")";
  }
  return success();
}

// Returns true if, for all indices i = 0..shape.size()-1, val is in the
// [min, max} interval:
//   val = `arrayAttr1[i]` + `arrayAttr2[i]`,
// If `halfOpen` is true then the admissible interval is [min, max). Otherwise,
// the admissible interval is [min, max].
template <typename OpType>
static LogicalResult isSumOfIntegerArrayAttrConfinedToShape(
    OpType op, ArrayAttr arrayAttr1, ArrayAttr arrayAttr2,
    ArrayRef<int64_t> shape, StringRef attrName1, StringRef attrName2,
    bool halfOpen = true, int64_t min = 1) {
  assert(arrayAttr1.size() <= shape.size());
  assert(arrayAttr2.size() <= shape.size());
  for (auto [index, it] :
       llvm::enumerate(llvm::zip(arrayAttr1, arrayAttr2, shape))) {
    auto val1 = llvm::cast<IntegerAttr>(std::get<0>(it)).getInt();
    auto val2 = llvm::cast<IntegerAttr>(std::get<1>(it)).getInt();
    int64_t max = std::get<2>(it);
    if (!halfOpen)
      max += 1;
    if (val1 + val2 < 0 || val1 + val2 >= max)
      return op.emitOpError("expected sum(")
             << attrName1 << ", " << attrName2 << ") dimension " << index
             << " to be confined to [" << min << ", " << max << ")";
  }
  return success();
}

static ArrayAttr makeI64ArrayAttr(ArrayRef<int64_t> values,
                                  MLIRContext *context) {
  auto attrs = llvm::map_range(values, [context](int64_t v) -> Attribute {
    return IntegerAttr::get(IntegerType::get(context, 64), APInt(64, v));
  });
  return ArrayAttr::get(context, llvm::to_vector<8>(attrs));
}

LogicalResult InsertStridedSliceOp::verify() {
  auto sourceVectorType = getSourceVectorType();
  auto destVectorType = getDestVectorType();
  auto offsets = getOffsetsAttr();
  auto strides = getStridesAttr();
  if (offsets.size() != static_cast<unsigned>(destVectorType.getRank()))
    return emitOpError(
        "expected offsets of same size as destination vector rank");
  if (strides.size() != static_cast<unsigned>(sourceVectorType.getRank()))
    return emitOpError("expected strides of same size as source vector rank");
  if (sourceVectorType.getRank() > destVectorType.getRank())
    return emitOpError(
        "expected source rank to be no greater than destination rank");

  auto sourceShape = sourceVectorType.getShape();
  auto destShape = destVectorType.getShape();
  SmallVector<int64_t, 4> sourceShapeAsDestShape(
      destShape.size() - sourceShape.size(), 0);
  sourceShapeAsDestShape.append(sourceShape.begin(), sourceShape.end());
  auto offName = InsertStridedSliceOp::getOffsetsAttrName();
  auto stridesName = InsertStridedSliceOp::getStridesAttrName();
  if (failed(isIntegerArrayAttrConfinedToShape(*this, offsets, destShape,
                                               offName)) ||
      failed(isIntegerArrayAttrConfinedToRange(*this, strides, /*min=*/1,
                                               /*max=*/1, stridesName,
                                               /*halfOpen=*/false)) ||
      failed(isSumOfIntegerArrayAttrConfinedToShape(
          *this, offsets,
          makeI64ArrayAttr(sourceShapeAsDestShape, getContext()), destShape,
          offName, "source vector shape",
          /*halfOpen=*/false, /*min=*/1)))
    return failure();

  unsigned rankDiff = destShape.size() - sourceShape.size();
  for (unsigned idx = 0; idx < sourceShape.size(); ++idx) {
    if (sourceVectorType.getScalableDims()[idx] !=
        destVectorType.getScalableDims()[idx + rankDiff]) {
      return emitOpError("mismatching scalable flags (at source vector idx=")
             << idx << ")";
    }
    if (sourceVectorType.getScalableDims()[idx]) {
      auto sourceSize = sourceShape[idx];
      auto destSize = destShape[idx + rankDiff];
      if (sourceSize != destSize) {
        return emitOpError("expected size at idx=")
               << idx
               << (" to match the corresponding base size from the input "
                   "vector (")
               << sourceSize << (" vs ") << destSize << (")");
      }
    }
  }

  return success();
}

namespace {
/// Rewrite insert_strided_slice(splat-like(v), splat-like(v)) as v.
class FoldInsertStridedSliceSplat final
    : public OpRewritePattern<InsertStridedSliceOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(InsertStridedSliceOp insertStridedSliceOp,
                                PatternRewriter &rewriter) const override {

    auto dst = insertStridedSliceOp.getDest();
    auto splat = getScalarSplatSource(insertStridedSliceOp.getValueToStore());
    if (!splat || getScalarSplatSource(dst) != splat)
      return failure();

    rewriter.replaceOp(insertStridedSliceOp, dst);
    return success();
  }
};

/// Pattern to rewrite an InsertStridedSliceOp(ExtractStridedSliceOp(dst), dst)
/// to dst.
class FoldInsertStridedSliceOfExtract final
    : public OpRewritePattern<InsertStridedSliceOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(InsertStridedSliceOp insertStridedSliceOp,
                                PatternRewriter &rewriter) const override {
    auto extractStridedSliceOp =
        insertStridedSliceOp.getValueToStore()
            .getDefiningOp<vector::ExtractStridedSliceOp>();

    if (!extractStridedSliceOp)
      return failure();

    if (extractStridedSliceOp.getOperand() != insertStridedSliceOp.getDest())
      return failure();

    // Check if have the same strides and offsets.
    if (extractStridedSliceOp.getStrides() !=
            insertStridedSliceOp.getStrides() ||
        extractStridedSliceOp.getOffsets() != insertStridedSliceOp.getOffsets())
      return failure();

    rewriter.replaceOp(insertStridedSliceOp, insertStridedSliceOp.getDest());
    return success();
  }
};

// Pattern to rewrite an InsertStridedSliceOp(ConstantOp into ConstantOp) ->
// ConstantOp.
class InsertStridedSliceConstantFolder final
    : public OpRewritePattern<InsertStridedSliceOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  // Do not create constants with more than `vectorSizeFoldThreashold` elements,
  // unless the source vector constant has a single use.
  static constexpr int64_t vectorSizeFoldThreshold = 256;

  LogicalResult matchAndRewrite(InsertStridedSliceOp op,
                                PatternRewriter &rewriter) const override {
    // Return if 'InsertOp' operand is not defined by a compatible vector
    // ConstantOp.
    TypedValue<VectorType> destVector = op.getDest();
    Attribute vectorDestCst;
    if (!matchPattern(destVector, m_Constant(&vectorDestCst)))
      return failure();

    VectorType destTy = destVector.getType();
    if (destTy.isScalable())
      return failure();

    // Make sure we do not create too many large constants.
    if (destTy.getNumElements() > vectorSizeFoldThreshold &&
        !destVector.hasOneUse())
      return failure();

    TypedValue<VectorType> sourceValue = op.getValueToStore();
    Attribute sourceCst;
    if (!matchPattern(sourceValue, m_Constant(&sourceCst)))
      return failure();

    // TODO: Support poison.
    if (isa<ub::PoisonAttr>(vectorDestCst) || isa<ub::PoisonAttr>(sourceCst))
      return failure();

    // TODO: Handle non-unit strides when they become available.
    if (op.hasNonUnitStrides())
      return failure();

    VectorType sliceVecTy = sourceValue.getType();
    ArrayRef<int64_t> sliceShape = sliceVecTy.getShape();
    int64_t rankDifference = destTy.getRank() - sliceVecTy.getRank();
    SmallVector<int64_t, 4> offsets = getI64SubArray(op.getOffsets());
    SmallVector<int64_t, 4> destStrides = computeStrides(destTy.getShape());

    // Calcualte the destination element indices by enumerating all slice
    // positions within the destination and linearizing them. The enumeration
    // order is lexicographic which yields a sequence of monotonically
    // increasing linearized position indices.
    // Because the destination may have higher dimensionality then the slice,
    // we keep track of two overlapping sets of positions and offsets.
    auto denseDest = llvm::cast<DenseElementsAttr>(vectorDestCst);
    auto denseSlice = llvm::cast<DenseElementsAttr>(sourceCst);
    auto sliceValuesIt = denseSlice.value_begin<Attribute>();
    auto newValues = llvm::to_vector(denseDest.getValues<Attribute>());
    SmallVector<int64_t> currDestPosition(offsets.begin(), offsets.end());
    MutableArrayRef<int64_t> currSlicePosition(
        currDestPosition.begin() + rankDifference, currDestPosition.end());
    ArrayRef<int64_t> sliceOffsets(offsets.begin() + rankDifference,
                                   offsets.end());
    do {
      int64_t linearizedPosition = linearize(currDestPosition, destStrides);
      assert(linearizedPosition < destTy.getNumElements() && "Invalid index");
      assert(sliceValuesIt != denseSlice.value_end<Attribute>() &&
             "Invalid slice element");
      newValues[linearizedPosition] = *sliceValuesIt;
      ++sliceValuesIt;
    } while (succeeded(
        incSlicePosition(currSlicePosition, sliceShape, sliceOffsets)));

    auto newAttr = DenseElementsAttr::get(destTy, newValues);
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, newAttr);
    return success();
  }
};

} // namespace

void vector::InsertStridedSliceOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<FoldInsertStridedSliceSplat, FoldInsertStridedSliceOfExtract,
              InsertStridedSliceConstantFolder>(context);
}

OpFoldResult InsertStridedSliceOp::fold(FoldAdaptor adaptor) {
  if (getSourceVectorType() == getDestVectorType())
    return getValueToStore();
  return {};
}

//===----------------------------------------------------------------------===//
// OuterProductOp
//===----------------------------------------------------------------------===//

/// Build an op without mask, use the type of `acc` as the return type.
void OuterProductOp::build(OpBuilder &builder, OperationState &result,
                           Value lhs, Value rhs, Value acc) {
  result.addOperands({lhs, rhs, acc});
  result.addTypes(acc.getType());
}

void OuterProductOp::print(OpAsmPrinter &p) {
  p << " " << getLhs() << ", " << getRhs();
  if (getAcc()) {
    p << ", " << getAcc();
    p.printOptionalAttrDict((*this)->getAttrs());
  }
  p << " : " << getLhs().getType() << ", " << getRhs().getType();
}

ParseResult OuterProductOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 3> operandsInfo;
  Type tLHS, tRHS;
  if (parser.parseOperandList(operandsInfo) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(tLHS) || parser.parseComma() ||
      parser.parseType(tRHS))
    return failure();
  if (operandsInfo.size() < 2)
    return parser.emitError(parser.getNameLoc(),
                            "expected at least 2 operands");
  VectorType vLHS = llvm::dyn_cast<VectorType>(tLHS);
  VectorType vRHS = llvm::dyn_cast<VectorType>(tRHS);
  if (!vLHS)
    return parser.emitError(parser.getNameLoc(),
                            "expected vector type for operand #1");

  VectorType resType;
  if (vRHS) {
    SmallVector<bool> scalableDimsRes{vLHS.getScalableDims()[0],
                                      vRHS.getScalableDims()[0]};
    resType = VectorType::get({vLHS.getDimSize(0), vRHS.getDimSize(0)},
                              vLHS.getElementType(), scalableDimsRes);
  } else {
    // Scalar RHS operand
    SmallVector<bool> scalableDimsRes{vLHS.getScalableDims()[0]};
    resType = VectorType::get({vLHS.getDimSize(0)}, vLHS.getElementType(),
                              scalableDimsRes);
  }

  if (!result.attributes.get(OuterProductOp::getKindAttrName(result.name))) {
    result.attributes.append(
        OuterProductOp::getKindAttrName(result.name),
        CombiningKindAttr::get(result.getContext(),
                               OuterProductOp::getDefaultKind()));
  }

  return failure(
      parser.resolveOperand(operandsInfo[0], tLHS, result.operands) ||
      parser.resolveOperand(operandsInfo[1], tRHS, result.operands) ||
      (operandsInfo.size() > 2 &&
       parser.resolveOperand(operandsInfo[2], resType, result.operands)) ||
      parser.addTypeToList(resType, result.types));
}

LogicalResult OuterProductOp::verify() {
  Type tRHS = getOperandTypeRHS();
  VectorType vLHS = getOperandVectorTypeLHS(),
             vRHS = llvm::dyn_cast<VectorType>(tRHS),
             vACC = getOperandVectorTypeACC(), vRES = getResultVectorType();

  if (vLHS.getRank() != 1)
    return emitOpError("expected 1-d vector for operand #1");

  if (vRHS) {
    // Proper OUTER operation.
    if (vRHS.getRank() != 1)
      return emitOpError("expected 1-d vector for operand #2");
    if (vRES.getRank() != 2)
      return emitOpError("expected 2-d vector result");
    if (vLHS.getDimSize(0) != vRES.getDimSize(0))
      return emitOpError("expected #1 operand dim to match result dim #1");
    if (vRHS.getDimSize(0) != vRES.getDimSize(1))
      return emitOpError("expected #2 operand dim to match result dim #2");
    if (vLHS.isScalable() && !vRHS.isScalable()) {
      // This restriction reflects what's currently supported in terms of
      // scalable vectors. However, we could relax this if there's a use case.
      return emitOpError(
          "expected either both or only #2 operand dim to be scalable");
    }
  } else {
    // An AXPY operation.
    if (vRES.getRank() != 1)
      return emitOpError("expected 1-d vector result");
    if (vLHS.getDimSize(0) != vRES.getDimSize(0))
      return emitOpError("expected #1 operand dim to match result dim #1");
  }

  if (vACC && vACC != vRES)
    return emitOpError("expected operand #3 of same type as result type");

  // Verify supported combining kind.
  if (!isSupportedCombiningKind(getKind(), vRES.getElementType()))
    return emitOpError("unsupported outerproduct type");

  return success();
}

// MaskableOpInterface methods.

/// Returns the mask type expected by this operation. Mostly used for
/// verification purposes. It requires the operation to be vectorized."
Type OuterProductOp::getExpectedMaskType() {
  auto vecType = this->getResultVectorType();
  return VectorType::get(vecType.getShape(),
                         IntegerType::get(vecType.getContext(), /*width=*/1),
                         vecType.getScalableDims());
}

//===----------------------------------------------------------------------===//
// ExtractStridedSliceOp
//===----------------------------------------------------------------------===//

// Inference works as follows:
//   1. Add 'sizes' from prefix of dims in 'offsets'.
//   2. Add sizes from 'vectorType' for remaining dims.
// Scalable flags are inherited from 'vectorType'.
static Type inferStridedSliceOpResultType(VectorType vectorType,
                                          ArrayAttr offsets, ArrayAttr sizes,
                                          ArrayAttr strides) {
  assert(offsets.size() == sizes.size() && offsets.size() == strides.size());
  SmallVector<int64_t, 4> shape;
  shape.reserve(vectorType.getRank());
  unsigned idx = 0;
  for (unsigned e = offsets.size(); idx < e; ++idx)
    shape.push_back(llvm::cast<IntegerAttr>(sizes[idx]).getInt());
  for (unsigned e = vectorType.getShape().size(); idx < e; ++idx)
    shape.push_back(vectorType.getShape()[idx]);

  return VectorType::get(shape, vectorType.getElementType(),
                         vectorType.getScalableDims());
}

void ExtractStridedSliceOp::build(OpBuilder &builder, OperationState &result,
                                  Value source, ArrayRef<int64_t> offsets,
                                  ArrayRef<int64_t> sizes,
                                  ArrayRef<int64_t> strides) {
  result.addOperands(source);
  auto offsetsAttr = getVectorSubscriptAttr(builder, offsets);
  auto sizesAttr = getVectorSubscriptAttr(builder, sizes);
  auto stridesAttr = getVectorSubscriptAttr(builder, strides);
  result.addTypes(
      inferStridedSliceOpResultType(llvm::cast<VectorType>(source.getType()),
                                    offsetsAttr, sizesAttr, stridesAttr));
  result.addAttribute(ExtractStridedSliceOp::getOffsetsAttrName(result.name),
                      offsetsAttr);
  result.addAttribute(ExtractStridedSliceOp::getSizesAttrName(result.name),
                      sizesAttr);
  result.addAttribute(ExtractStridedSliceOp::getStridesAttrName(result.name),
                      stridesAttr);
}

LogicalResult ExtractStridedSliceOp::verify() {
  auto type = getSourceVectorType();
  auto offsets = getOffsetsAttr();
  auto sizes = getSizesAttr();
  auto strides = getStridesAttr();
  if (offsets.size() != sizes.size() || offsets.size() != strides.size())
    return emitOpError(
        "expected offsets, sizes and strides attributes of same size");

  auto shape = type.getShape();
  auto offName = getOffsetsAttrName();
  auto sizesName = getSizesAttrName();
  auto stridesName = getStridesAttrName();
  if (failed(
          isIntegerArrayAttrSmallerThanShape(*this, offsets, shape, offName)) ||
      failed(
          isIntegerArrayAttrSmallerThanShape(*this, sizes, shape, sizesName)) ||
      failed(isIntegerArrayAttrSmallerThanShape(*this, strides, shape,
                                                stridesName)) ||
      failed(
          isIntegerArrayAttrConfinedToShape(*this, offsets, shape, offName)) ||
      failed(isIntegerArrayAttrConfinedToShape(*this, sizes, shape, sizesName,
                                               /*halfOpen=*/false,
                                               /*min=*/1)) ||
      failed(isIntegerArrayAttrConfinedToRange(*this, strides, /*min=*/1,
                                               /*max=*/1, stridesName,
                                               /*halfOpen=*/false)) ||
      failed(isSumOfIntegerArrayAttrConfinedToShape(*this, offsets, sizes,
                                                    shape, offName, sizesName,
                                                    /*halfOpen=*/false)))
    return failure();

  auto resultType = inferStridedSliceOpResultType(getSourceVectorType(),
                                                  offsets, sizes, strides);
  if (getResult().getType() != resultType)
    return emitOpError("expected result type to be ") << resultType;

  for (unsigned idx = 0; idx < sizes.size(); ++idx) {
    if (type.getScalableDims()[idx]) {
      auto inputDim = type.getShape()[idx];
      auto inputSize = llvm::cast<IntegerAttr>(sizes[idx]).getInt();
      if (inputDim != inputSize)
        return emitOpError("expected size at idx=")
               << idx
               << (" to match the corresponding base size from the input "
                   "vector (")
               << inputSize << (" vs ") << inputDim << (")");
    }
  }

  return success();
}

// When the source of ExtractStrided comes from a chain of InsertStrided ops try
// to use the source of the InsertStrided ops if we can detect that the
// extracted vector is a subset of one of the vector inserted.
static LogicalResult
foldExtractStridedOpFromInsertChain(ExtractStridedSliceOp op) {
  // Helper to extract integer out of ArrayAttr.
  auto getElement = [](ArrayAttr array, int idx) {
    return llvm::cast<IntegerAttr>(array[idx]).getInt();
  };
  ArrayAttr extractOffsets = op.getOffsets();
  ArrayAttr extractStrides = op.getStrides();
  ArrayAttr extractSizes = op.getSizes();
  auto insertOp = op.getVector().getDefiningOp<InsertStridedSliceOp>();
  while (insertOp) {
    if (op.getSourceVectorType().getRank() !=
        insertOp.getSourceVectorType().getRank())
      return failure();
    ArrayAttr insertOffsets = insertOp.getOffsets();
    ArrayAttr insertStrides = insertOp.getStrides();
    // If the rank of extract is greater than the rank of insert, we are likely
    // extracting a partial chunk of the vector inserted.
    if (extractOffsets.size() > insertOffsets.size())
      return failure();
    bool patialoverlap = false;
    bool disjoint = false;
    SmallVector<int64_t, 4> offsetDiffs;
    for (unsigned dim = 0, e = extractOffsets.size(); dim < e; ++dim) {
      if (getElement(extractStrides, dim) != getElement(insertStrides, dim))
        return failure();
      int64_t start = getElement(insertOffsets, dim);
      int64_t end = start + insertOp.getSourceVectorType().getDimSize(dim);
      int64_t offset = getElement(extractOffsets, dim);
      int64_t size = getElement(extractSizes, dim);
      // Check if the start of the extract offset is in the interval inserted.
      if (start <= offset && offset < end) {
        // If the extract interval overlaps but is not fully included we may
        // have a partial overlap that will prevent any folding.
        if (offset + size > end)
          patialoverlap = true;
        offsetDiffs.push_back(offset - start);
        continue;
      }
      disjoint = true;
      break;
    }
    // The extract element chunk is a subset of the insert element.
    if (!disjoint && !patialoverlap) {
      op.setOperand(insertOp.getValueToStore());
      // OpBuilder is only used as a helper to build an I64ArrayAttr.
      OpBuilder b(op.getContext());
      op.setOffsetsAttr(b.getI64ArrayAttr(offsetDiffs));
      return success();
    }
    // If the chunk extracted is disjoint from the chunk inserted, keep looking
    // in the insert chain.
    if (disjoint)
      insertOp = insertOp.getDest().getDefiningOp<InsertStridedSliceOp>();
    else {
      // The extracted vector partially overlap the inserted vector, we cannot
      // fold.
      return failure();
    }
  }
  return failure();
}

// ExtractStridedSliceOp(non-splat ConstantOp) -> ConstantOp.
static OpFoldResult
foldExtractStridedSliceNonSplatConstant(ExtractStridedSliceOp op,
                                        Attribute foldInput) {

  auto dense = llvm::dyn_cast_if_present<DenseElementsAttr>(foldInput);
  if (!dense)
    return {};

  // TODO: Handle non-unit strides when they become available.
  if (op.hasNonUnitStrides())
    return {};

  VectorType sourceVecTy = op.getSourceVectorType();
  ArrayRef<int64_t> sourceShape = sourceVecTy.getShape();
  SmallVector<int64_t, 4> sourceStrides = computeStrides(sourceShape);

  VectorType sliceVecTy = op.getType();
  ArrayRef<int64_t> sliceShape = sliceVecTy.getShape();
  int64_t rank = sliceVecTy.getRank();

  // Expand offsets and sizes to match the vector rank.
  SmallVector<int64_t, 4> offsets(rank, 0);
  copy(getI64SubArray(op.getOffsets()), offsets.begin());

  SmallVector<int64_t, 4> sizes(sourceShape);
  copy(getI64SubArray(op.getSizes()), sizes.begin());

  // Calculate the slice elements by enumerating all slice positions and
  // linearizing them. The enumeration order is lexicographic which yields a
  // sequence of monotonically increasing linearized position indices.
  const auto denseValuesBegin = dense.value_begin<Attribute>();
  SmallVector<Attribute> sliceValues;
  sliceValues.reserve(sliceVecTy.getNumElements());
  SmallVector<int64_t> currSlicePosition(offsets.begin(), offsets.end());
  do {
    int64_t linearizedPosition = linearize(currSlicePosition, sourceStrides);
    assert(linearizedPosition < sourceVecTy.getNumElements() &&
           "Invalid index");
    sliceValues.push_back(*(denseValuesBegin + linearizedPosition));
  } while (succeeded(incSlicePosition(currSlicePosition, sliceShape, offsets)));

  assert(static_cast<int64_t>(sliceValues.size()) ==
             sliceVecTy.getNumElements() &&
         "Invalid number of slice elements");
  return DenseElementsAttr::get(sliceVecTy, sliceValues);
}

OpFoldResult ExtractStridedSliceOp::fold(FoldAdaptor adaptor) {
  if (getSourceVectorType() == getResult().getType())
    return getVector();
  if (succeeded(foldExtractStridedOpFromInsertChain(*this)))
    return getResult();

  // ExtractStridedSliceOp(splat ConstantOp) -> ConstantOp.
  if (auto splat =
          llvm::dyn_cast_if_present<SplatElementsAttr>(adaptor.getVector()))
    DenseElementsAttr::get(getType(), splat.getSplatValue<Attribute>());

  // ExtractStridedSliceOp(non-splat ConstantOp) -> ConstantOp.
  return foldExtractStridedSliceNonSplatConstant(*this, adaptor.getVector());
}

void ExtractStridedSliceOp::getOffsets(SmallVectorImpl<int64_t> &results) {
  populateFromInt64AttrArray(getOffsets(), results);
}

namespace {

// Pattern to rewrite an ExtractStridedSliceOp(CreateMaskOp) to
// CreateMaskOp.
//
// Example:
//
// %mask = vector.create_mask %ub : vector<16xi1>
// %slice = vector.extract_strided_slice [%offset] [8] [1]
//
// to
//
// %new_ub = arith.subi %ub, %offset
// %mask = vector.create_mask %new_ub : vector<8xi1>
class StridedSliceCreateMaskFolder final
    : public OpRewritePattern<ExtractStridedSliceOp> {
  using OpRewritePattern::OpRewritePattern;

public:
  LogicalResult matchAndRewrite(ExtractStridedSliceOp extractStridedSliceOp,
                                PatternRewriter &rewriter) const override {
    Location loc = extractStridedSliceOp.getLoc();
    // Return if 'extractStridedSliceOp' operand is not defined by a
    // CreateMaskOp.
    auto createMaskOp =
        extractStridedSliceOp.getVector().getDefiningOp<CreateMaskOp>();
    if (!createMaskOp)
      return failure();
    // Return if 'extractStridedSliceOp' has non-unit strides.
    if (extractStridedSliceOp.hasNonUnitStrides())
      return failure();
    // Gather constant mask dimension sizes.
    SmallVector<Value> maskDimSizes(createMaskOp.getOperands());
    // Gather strided slice offsets and sizes.
    SmallVector<int64_t> sliceOffsets;
    populateFromInt64AttrArray(extractStridedSliceOp.getOffsets(),
                               sliceOffsets);
    SmallVector<int64_t> sliceSizes;
    populateFromInt64AttrArray(extractStridedSliceOp.getSizes(), sliceSizes);

    // Compute slice of vector mask region.
    SmallVector<Value> sliceMaskDimSizes;
    sliceMaskDimSizes.reserve(maskDimSizes.size());
    // sliceOffsets.size() <= maskDimSizes.size(), so we use llvm::zip and
    // only iterate on the leading dim sizes. The tail accounts for the
    // remaining dim sizes.
    for (auto [maskDimSize, sliceOffset, sliceSize] :
         llvm::zip(maskDimSizes, sliceOffsets, sliceSizes)) {
      // No need to clamp on min/max values, because create_mask has clamping
      // semantics, i.e. the sliceMaskDimSize is allowed to be negative or
      // greater than the vector dim size.
      IntegerAttr offsetAttr =
          rewriter.getIntegerAttr(maskDimSize.getType(), sliceOffset);
      Value offset = arith::ConstantOp::create(rewriter, loc, offsetAttr);
      Value sliceMaskDimSize =
          arith::SubIOp::create(rewriter, loc, maskDimSize, offset);
      sliceMaskDimSizes.push_back(sliceMaskDimSize);
    }
    // Add unchanged dimensions.
    llvm::append_range(
        sliceMaskDimSizes,
        llvm::drop_begin(maskDimSizes, sliceMaskDimSizes.size()));
    // Replace 'extractStridedSliceOp' with CreateMaskOp with sliced mask
    // region.
    rewriter.replaceOpWithNewOp<CreateMaskOp>(
        extractStridedSliceOp, extractStridedSliceOp.getResult().getType(),
        sliceMaskDimSizes);
    return success();
  }
};

// Pattern to rewrite an ExtractStridedSliceOp(ConstantMaskOp) to
// ConstantMaskOp.
class StridedSliceConstantMaskFolder final
    : public OpRewritePattern<ExtractStridedSliceOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ExtractStridedSliceOp extractStridedSliceOp,
                                PatternRewriter &rewriter) const override {
    // Return if 'extractStridedSliceOp' operand is not defined by a
    // ConstantMaskOp.
    auto *defOp = extractStridedSliceOp.getVector().getDefiningOp();
    auto constantMaskOp = dyn_cast_or_null<ConstantMaskOp>(defOp);
    if (!constantMaskOp)
      return failure();
    // Return if 'extractStridedSliceOp' has non-unit strides.
    if (extractStridedSliceOp.hasNonUnitStrides())
      return failure();
    // Gather constant mask dimension sizes.
    ArrayRef<int64_t> maskDimSizes = constantMaskOp.getMaskDimSizes();
    // Gather strided slice offsets and sizes.
    SmallVector<int64_t> sliceOffsets;
    populateFromInt64AttrArray(extractStridedSliceOp.getOffsets(),
                               sliceOffsets);
    SmallVector<int64_t> sliceSizes;
    populateFromInt64AttrArray(extractStridedSliceOp.getSizes(), sliceSizes);

    // Compute slice of vector mask region.
    SmallVector<int64_t> sliceMaskDimSizes;
    sliceMaskDimSizes.reserve(maskDimSizes.size());
    for (auto [maskDimSize, sliceOffset, sliceSize] :
         llvm::zip(maskDimSizes, sliceOffsets, sliceSizes)) {
      int64_t sliceMaskDimSize = std::max(
          static_cast<int64_t>(0),
          std::min(sliceOffset + sliceSize, maskDimSize) - sliceOffset);
      sliceMaskDimSizes.push_back(sliceMaskDimSize);
    }
    // Add unchanged dimensions.
    if (sliceMaskDimSizes.size() < maskDimSizes.size())
      for (size_t i = sliceMaskDimSizes.size(); i < maskDimSizes.size(); ++i)
        sliceMaskDimSizes.push_back(maskDimSizes[i]);
    // If any of 'sliceMaskDimSizes' are zero, then set all to zero (masked
    // region is a conjunction of mask dim intervals).
    if (llvm::is_contained(sliceMaskDimSizes, 0))
      sliceMaskDimSizes.assign(maskDimSizes.size(), 0);

    // Replace 'extractStridedSliceOp' with ConstantMaskOp with sliced mask
    // region.
    rewriter.replaceOpWithNewOp<ConstantMaskOp>(
        extractStridedSliceOp, extractStridedSliceOp.getResult().getType(),
        sliceMaskDimSizes);
    return success();
  }
};

// Pattern to rewrite an ExtractStridedSliceOp(BroadcastOp) to
// BroadcastOp(ExtractStrideSliceOp).
class StridedSliceBroadcast final
    : public OpRewritePattern<ExtractStridedSliceOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ExtractStridedSliceOp op,
                                PatternRewriter &rewriter) const override {
    auto broadcast = op.getVector().getDefiningOp<BroadcastOp>();
    if (!broadcast)
      return failure();
    auto srcVecType =
        llvm::dyn_cast<VectorType>(broadcast.getSource().getType());
    unsigned srcRank = srcVecType ? srcVecType.getRank() : 0;
    auto dstVecType = llvm::cast<VectorType>(op.getType());
    unsigned dstRank = dstVecType.getRank();
    unsigned rankDiff = dstRank - srcRank;
    // Source dimensions can be broadcasted (1 -> n with n > 1) or sliced
    // (n -> m with n > m). If they are originally both broadcasted *and*
    // sliced, this can be simplified to just broadcasting.
    bool needsSlice = false;
    for (unsigned i = 0; i < srcRank; i++) {
      if (srcVecType.getDimSize(i) != 1 &&
          srcVecType.getDimSize(i) != dstVecType.getDimSize(i + rankDiff)) {
        needsSlice = true;
        break;
      }
    }
    Value source = broadcast.getSource();
    if (needsSlice) {
      SmallVector<int64_t> offsets =
          getI64SubArray(op.getOffsets(), /*dropFront=*/rankDiff);
      SmallVector<int64_t> sizes =
          getI64SubArray(op.getSizes(), /*dropFront=*/rankDiff);
      for (unsigned i = 0; i < srcRank; i++) {
        if (srcVecType.getDimSize(i) == 1) {
          // In case this dimension was broadcasted *and* sliced, the offset
          // and size need to be updated now that there is no broadcast before
          // the slice.
          offsets[i] = 0;
          sizes[i] = 1;
        }
      }
      source = ExtractStridedSliceOp::create(
          rewriter, op->getLoc(), source, offsets, sizes,
          getI64SubArray(op.getStrides(), /*dropFront=*/rankDiff));
    }
    rewriter.replaceOpWithNewOp<BroadcastOp>(op, op.getType(), source);
    return success();
  }
};

/// Rewrite extract_strided_slice(splat-like(v)) with broadcast(v).
class StridedSliceSplat final : public OpRewritePattern<ExtractStridedSliceOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ExtractStridedSliceOp op,
                                PatternRewriter &rewriter) const override {

    Value splat = getScalarSplatSource(op.getVector());
    if (!splat)
      return failure();
    rewriter.replaceOpWithNewOp<BroadcastOp>(op, op.getType(), splat);
    return success();
  }
};

/// Pattern to rewrite simple cases of N-D extract_strided_slice, where the
/// slice is contiguous, into extract and shape_cast.
///
/// Example:
///     Before:
///         %1 = vector.extract_strided_slice %arg0 {
///                offsets = [0, 0, 0, 0, 0],
///                sizes = [1, 1, 1, 1, 8],
///                strides = [1, 1, 1, 1, 1]
///              } : vector<8x1x1x2x8xi8> to vector<1x1x1x1x8xi8>
///     After:
///         %0 = vector.extract %arg0[0, 0, 0, 0]
///                : vector<8xi8> from vector<8x1x1x2x8xi8>
///         %1 = vector.shape_cast %0
///                : vector<8xi8> to vector<1x1x1x1x8xi8>
///
class ContiguousExtractStridedSliceToExtract final
    : public OpRewritePattern<ExtractStridedSliceOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ExtractStridedSliceOp op,
                                PatternRewriter &rewriter) const override {
    if (op.hasNonUnitStrides())
      return failure();
    Value source = op.getOperand();
    auto sourceType = cast<VectorType>(source.getType());
    if (sourceType.isScalable() || sourceType.getRank() == 0)
      return failure();

    // Compute the number of offsets to pass to ExtractOp::build. That is the
    // difference between the source rank and the desired slice rank. We walk
    // the dimensions from innermost out, and stop when the next slice dimension
    // is not full-size.
    SmallVector<int64_t> sizes = getI64SubArray(op.getSizes());
    int numOffsets;
    for (numOffsets = sizes.size(); numOffsets > 0; --numOffsets) {
      if (sizes[numOffsets - 1] != sourceType.getDimSize(numOffsets - 1))
        break;
    }

    // If the created extract op would have no offsets, then this whole
    // extract_strided_slice is the identity and should have been handled by
    // other canonicalizations.
    if (numOffsets == 0)
      return failure();

    // If not even the inner-most dimension is full-size, this op can't be
    // rewritten as an ExtractOp.
    if (numOffsets == sourceType.getRank() &&
        static_cast<int>(sizes.size()) == sourceType.getRank())
      return failure();

    // The outer dimensions must have unit size.
    for (int i = 0; i < numOffsets; ++i) {
      if (sizes[i] != 1)
        return failure();
    }

    // Avoid generating slices that have leading unit dimensions. The shape_cast
    // op that we create below would take bad generic fallback patterns
    // (ShapeCastOpRewritePattern).
    while (numOffsets < static_cast<int>(sizes.size()) - 1 &&
           sizes[numOffsets] == 1) {
      ++numOffsets;
    }

    SmallVector<int64_t> offsets = getI64SubArray(op.getOffsets());
    auto extractOffsets = ArrayRef(offsets).take_front(numOffsets);
    Value extract = vector::ExtractOp::create(rewriter, op->getLoc(), source,
                                              extractOffsets);
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(op, op.getType(), extract);
    return success();
  }
};

} // namespace

void ExtractStridedSliceOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  // Pattern to rewrite a ExtractStridedSliceOp(ConstantMaskOp) ->
  // ConstantMaskOp and ExtractStridedSliceOp(ConstantOp) -> ConstantOp.
  results.add<StridedSliceCreateMaskFolder, StridedSliceConstantMaskFolder,
              StridedSliceBroadcast, StridedSliceSplat,
              ContiguousExtractStridedSliceToExtract>(context);
}

//===----------------------------------------------------------------------===//
// TransferReadOp
//===----------------------------------------------------------------------===//

/// 1. Builder that sets padding to zero and an empty mask (variant with attrs).
void TransferReadOp::build(OpBuilder &builder, OperationState &result,
                           VectorType vectorType, Value source,
                           ValueRange indices, std::optional<Value> padding,
                           AffineMapAttr permutationMapAttr,
                           /*optional*/ ArrayAttr inBoundsAttr) {

  Type elemType = llvm::cast<ShapedType>(source.getType()).getElementType();
  if (!padding)
    padding = ub::PoisonOp::create(builder, result.location, elemType);
  build(builder, result, vectorType, source, indices, permutationMapAttr,
        *padding, /*mask=*/Value(), inBoundsAttr);
}

/// 2. Builder that sets padding to zero an empty mask (variant without attrs).
void TransferReadOp::build(OpBuilder &builder, OperationState &result,
                           VectorType vectorType, Value source,
                           ValueRange indices, std::optional<Value> padding,
                           AffineMap permutationMap,
                           std::optional<ArrayRef<bool>> inBounds) {
  auto permutationMapAttr = AffineMapAttr::get(permutationMap);
  auto inBoundsAttr = (inBounds && !inBounds.value().empty())
                          ? builder.getBoolArrayAttr(inBounds.value())
                          : builder.getBoolArrayAttr(
                                SmallVector<bool>(vectorType.getRank(), false));
  Type elemType = llvm::cast<ShapedType>(source.getType()).getElementType();
  if (!padding)
    padding = ub::PoisonOp::create(builder, result.location, elemType);
  build(builder, result, vectorType, source, indices, *padding,
        permutationMapAttr, inBoundsAttr);
}

/// 3. Builder that sets permutation map to 'getMinorIdentityMap'.
void TransferReadOp::build(OpBuilder &builder, OperationState &result,
                           VectorType vectorType, Value source,
                           ValueRange indices, std::optional<Value> padding,
                           std::optional<ArrayRef<bool>> inBounds) {
  AffineMap permutationMap = getTransferMinorIdentityMap(
      llvm::cast<ShapedType>(source.getType()), vectorType);
  auto permutationMapAttr = AffineMapAttr::get(permutationMap);
  auto inBoundsAttr = (inBounds && !inBounds.value().empty())
                          ? builder.getBoolArrayAttr(inBounds.value())
                          : builder.getBoolArrayAttr(
                                SmallVector<bool>(vectorType.getRank(), false));
  Type elemType = llvm::cast<ShapedType>(source.getType()).getElementType();
  if (!padding)
    padding = ub::PoisonOp::create(builder, result.location, elemType);
  build(builder, result, vectorType, source, indices, permutationMapAttr,
        *padding,
        /*mask=*/Value(), inBoundsAttr);
}

template <typename EmitFun>
static LogicalResult verifyPermutationMap(AffineMap permutationMap,
                                          EmitFun emitOpError) {
  SmallVector<bool, 8> seen(permutationMap.getNumInputs(), false);
  for (auto expr : permutationMap.getResults()) {
    auto dim = dyn_cast<AffineDimExpr>(expr);
    auto zero = dyn_cast<AffineConstantExpr>(expr);
    if (zero) {
      if (zero.getValue() != 0) {
        return emitOpError(
            "requires a projected permutation_map (at most one dim or the zero "
            "constant can appear in each result)");
      }
      continue;
    }
    if (!dim) {
      return emitOpError("requires a projected permutation_map (at most one "
                         "dim or the zero constant can appear in each result)");
    }
    if (seen[dim.getPosition()]) {
      return emitOpError(
          "requires a permutation_map that is a permutation (found one dim "
          "used more than once)");
    }
    seen[dim.getPosition()] = true;
  }
  return success();
}

static LogicalResult
verifyTransferOp(VectorTransferOpInterface op, ShapedType shapedType,
                 VectorType vectorType, VectorType maskType,
                 VectorType inferredMaskType, AffineMap permutationMap,
                 ArrayAttr inBounds) {
  if (op->hasAttr("masked")) {
    return op->emitOpError("masked attribute has been removed. "
                           "Use in_bounds instead.");
  }

  if (!llvm::isa<MemRefType, RankedTensorType>(shapedType))
    return op->emitOpError(
        "requires source to be a memref or ranked tensor type");

  auto elementType = shapedType.getElementType();
  DataLayout dataLayout = DataLayout::closest(op);
  if (auto vectorElementType = llvm::dyn_cast<VectorType>(elementType)) {
    // Memref or tensor has vector element type.
    unsigned sourceVecSize =
        dataLayout.getTypeSizeInBits(vectorElementType.getElementType()) *
        vectorElementType.getShape().back();
    unsigned resultVecSize =
        dataLayout.getTypeSizeInBits(vectorType.getElementType()) *
        vectorType.getShape().back();
    if (resultVecSize % sourceVecSize != 0)
      return op->emitOpError(
          "requires the bitwidth of the minor 1-D vector to be an integral "
          "multiple of the bitwidth of the minor 1-D vector of the source");

    unsigned sourceVecEltRank = vectorElementType.getRank();
    unsigned resultVecRank = vectorType.getRank();
    if (sourceVecEltRank > resultVecRank)
      return op->emitOpError(
          "requires source vector element and vector result ranks to match.");
    unsigned rankOffset = resultVecRank - sourceVecEltRank;
    // Check that permutation map results match 'rankOffset' of vector type.
    if (permutationMap.getNumResults() != rankOffset)
      return op->emitOpError("requires a permutation_map with result dims of "
                             "the same rank as the vector type");

    if (maskType)
      return op->emitOpError("does not support masks with vector element type");
  } else {
    // Memref or tensor has scalar element type.
    unsigned minorSize =
        vectorType.getRank() == 0 ? 1 : vectorType.getShape().back();
    unsigned resultVecSize =
        dataLayout.getTypeSizeInBits(vectorType.getElementType()) * minorSize;
    if (resultVecSize % dataLayout.getTypeSizeInBits(elementType) != 0)
      return op->emitOpError(
          "requires the bitwidth of the minor 1-D vector to be an integral "
          "multiple of the bitwidth of the source element type");

    // Check that permutation map results match rank of vector type.
    if (permutationMap.getNumResults() != vectorType.getRank())
      return op->emitOpError("requires a permutation_map with result dims of "
                             "the same rank as the vector type");
  }

  if (permutationMap.getNumSymbols() != 0)
    return op->emitOpError("requires permutation_map without symbols");

  if (permutationMap.getNumInputs() != shapedType.getRank())
    return op->emitOpError("requires a permutation_map with input dims of the "
                           "same rank as the source type");

  if (maskType && maskType != inferredMaskType)
    return op->emitOpError("inferred mask type (")
           << inferredMaskType << ") and mask operand type (" << maskType
           << ") don't match";

  if (permutationMap.getNumResults() != static_cast<int64_t>(inBounds.size()))
    return op->emitOpError("expects the in_bounds attr of same rank "
                           "as permutation_map results: ")
           << AffineMapAttr::get(permutationMap)
           << " vs inBounds of size: " << inBounds.size();

  return success();
}

static void printTransferAttrs(OpAsmPrinter &p, VectorTransferOpInterface op) {
  SmallVector<StringRef, 3> elidedAttrs;
  elidedAttrs.push_back(TransferReadOp::getOperandSegmentSizeAttr());
  if (op.getPermutationMap().isMinorIdentity())
    elidedAttrs.push_back(op.getPermutationMapAttrName());
  // Elide in_bounds attribute if all dims are out-of-bounds.
  if (llvm::none_of(op.getInBoundsValues(), [](bool b) { return b; }))
    elidedAttrs.push_back(op.getInBoundsAttrName());
  p.printOptionalAttrDict(op->getAttrs(), elidedAttrs);
}

void TransferReadOp::print(OpAsmPrinter &p) {
  p << " " << getBase() << "[" << getIndices() << "], " << getPadding();
  if (getMask())
    p << ", " << getMask();
  printTransferAttrs(p, *this);
  p << " : " << getShapedType() << ", " << getVectorType();
}

VectorType mlir::vector::inferTransferOpMaskType(VectorType vecType,
                                                 AffineMap permMap) {
  auto i1Type = IntegerType::get(permMap.getContext(), 1);
  AffineMap invPermMap = inversePermutation(compressUnusedDims(permMap));
  assert(invPermMap && "Inversed permutation map couldn't be computed");
  SmallVector<int64_t, 8> maskShape = invPermMap.compose(vecType.getShape());

  // The MaskOp specification doesn't support 0-D vectors at the moment. Turn a
  // 0-D mask into a single-element 1-D mask.
  if (maskShape.empty())
    maskShape.push_back(1);

  SmallVector<bool> scalableDims =
      applyPermutationMap(invPermMap, vecType.getScalableDims());

  return VectorType::get(maskShape, i1Type, scalableDims);
}

ParseResult TransferReadOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  SMLoc typesLoc;
  OpAsmParser::UnresolvedOperand sourceInfo;
  SmallVector<OpAsmParser::UnresolvedOperand, 8> indexInfo;
  OpAsmParser::UnresolvedOperand paddingInfo;
  SmallVector<Type, 2> types;
  OpAsmParser::UnresolvedOperand maskInfo;
  // Parsing with support for paddingValue.
  if (parser.parseOperand(sourceInfo) ||
      parser.parseOperandList(indexInfo, OpAsmParser::Delimiter::Square) ||
      parser.parseComma() || parser.parseOperand(paddingInfo))
    return failure();
  ParseResult hasMask = parser.parseOptionalComma();
  if (hasMask.succeeded()) {
    if (parser.parseOperand(maskInfo))
      return failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.getCurrentLocation(&typesLoc) || parser.parseColonTypeList(types))
    return failure();
  if (types.size() != 2)
    return parser.emitError(typesLoc, "requires two types");
  auto indexType = builder.getIndexType();
  auto shapedType = llvm::dyn_cast<ShapedType>(types[0]);
  if (!shapedType || !llvm::isa<MemRefType, RankedTensorType>(shapedType))
    return parser.emitError(typesLoc, "requires memref or ranked tensor type");
  VectorType vectorType = llvm::dyn_cast<VectorType>(types[1]);
  if (!vectorType)
    return parser.emitError(typesLoc, "requires vector type");
  auto permMapAttrName = TransferReadOp::getPermutationMapAttrName(result.name);
  Attribute permMapAttr = result.attributes.get(permMapAttrName);
  AffineMap permMap;
  if (!permMapAttr) {
    if (shapedType.getRank() <
        getEffectiveVectorRankForXferOp(shapedType, vectorType))
      return parser.emitError(typesLoc,
                              "expected a custom permutation_map when "
                              "rank(source) != rank(destination)");
    permMap = getTransferMinorIdentityMap(shapedType, vectorType);
    result.attributes.set(permMapAttrName, AffineMapAttr::get(permMap));
  } else {
    permMap = llvm::cast<AffineMapAttr>(permMapAttr).getValue();
  }
  auto inBoundsAttrName = TransferReadOp::getInBoundsAttrName(result.name);
  Attribute inBoundsAttr = result.attributes.get(inBoundsAttrName);
  if (!inBoundsAttr) {
    result.addAttribute(inBoundsAttrName,
                        builder.getBoolArrayAttr(
                            SmallVector<bool>(permMap.getNumResults(), false)));
  }
  if (parser.resolveOperand(sourceInfo, shapedType, result.operands) ||
      parser.resolveOperands(indexInfo, indexType, result.operands) ||
      parser.resolveOperand(paddingInfo, shapedType.getElementType(),
                            result.operands))
    return failure();
  if (hasMask.succeeded()) {
    if (llvm::dyn_cast<VectorType>(shapedType.getElementType()))
      return parser.emitError(
          maskInfo.location, "does not support masks with vector element type");
    if (vectorType.getRank() != permMap.getNumResults()) {
      return parser.emitError(typesLoc,
                              "expected the same rank for the vector and the "
                              "results of the permutation map");
    }
    // Instead of adding the mask type as an op type, compute it based on the
    // vector type and the permutation map (to keep the type signature small).
    auto maskType = inferTransferOpMaskType(vectorType, permMap);
    if (parser.resolveOperand(maskInfo, maskType, result.operands))
      return failure();
  }
  result.addAttribute(TransferReadOp::getOperandSegmentSizeAttr(),
                      builder.getDenseI32ArrayAttr(
                          {1, static_cast<int32_t>(indexInfo.size()), 1,
                           static_cast<int32_t>(hasMask.succeeded())}));
  return parser.addTypeToList(vectorType, result.types);
}

LogicalResult TransferReadOp::verify() {
  // Consistency of elemental types in source and vector.
  ShapedType shapedType = getShapedType();
  VectorType vectorType = getVectorType();
  VectorType maskType = getMaskType();
  auto paddingType = getPadding().getType();
  auto permutationMap = getPermutationMap();
  VectorType inferredMaskType =
      maskType ? inferTransferOpMaskType(vectorType, permutationMap)
               : VectorType();
  auto sourceElementType = shapedType.getElementType();

  if (static_cast<int64_t>(getIndices().size()) != shapedType.getRank())
    return emitOpError("requires ") << shapedType.getRank() << " indices";

  if (failed(verifyTransferOp(cast<VectorTransferOpInterface>(getOperation()),
                              shapedType, vectorType, maskType,
                              inferredMaskType, permutationMap, getInBounds())))
    return failure();

  if (auto sourceVectorElementType =
          llvm::dyn_cast<VectorType>(sourceElementType)) {
    // Source has vector element type.
    // Check that 'sourceVectorElementType' and 'paddingType' types match.
    if (sourceVectorElementType != paddingType)
      return emitOpError(
          "requires source element type and padding type to match.");

  } else {
    // Check that 'paddingType' is valid to store in a vector type.
    if (!VectorType::isValidElementType(paddingType))
      return emitOpError("requires valid padding vector elemental type");

    // Check that padding type and vector element types match.
    if (paddingType != sourceElementType)
      return emitOpError(
          "requires formal padding and source of the same elemental type");
  }

  return verifyPermutationMap(permutationMap,
                              [&](Twine t) { return emitOpError(t); });
}

// MaskableOpInterface methods.

/// Returns the mask type expected by this operation. Mostly used for
/// verification purposes. It requires the operation to be vectorized."
Type TransferReadOp::getExpectedMaskType() {
  return inferTransferOpMaskType(getVectorType(), getPermutationMap());
}

//===----------------------------------------------------------------------===//
// TransferReadOp: VectorTransferOpInterface methods.
//===----------------------------------------------------------------------===//
VectorType TransferReadOp::getVectorType() {
  return cast<VectorType>(getVector().getType());
}

template <typename TransferOp>
static bool isInBounds(TransferOp op, int64_t resultIdx, int64_t indicesIdx) {
  // TODO: support more aggressive createOrFold on:
  // op.getIndices()[indicesIdx] + vectorType < dim(op.getSource(), indicesIdx)
  if (op.getShapedType().isDynamicDim(indicesIdx))
    return false;
  Value index = op.getIndices()[indicesIdx];
  std::optional<int64_t> cstOp = getConstantIntValue(index);
  if (!cstOp.has_value())
    return false;

  int64_t sourceSize = op.getShapedType().getDimSize(indicesIdx);
  int64_t vectorSize = op.getVectorType().getDimSize(resultIdx);

  return cstOp.value() + vectorSize <= sourceSize;
}

template <typename TransferOp>
static LogicalResult foldTransferInBoundsAttribute(TransferOp op) {
  // TODO: support 0-d corner case.
  // TODO: Be less conservative.
  if (op.getTransferRank() == 0)
    return failure();
  AffineMap permutationMap = op.getPermutationMap();
  bool changed = false;
  SmallVector<bool, 4> newInBounds;
  newInBounds.reserve(op.getTransferRank());
  // Idxs of non-bcast dims - used when analysing bcast dims.
  SmallVector<unsigned> nonBcastDims;

  // 1. Process non-broadcast dims
  for (unsigned i = 0; i < op.getTransferRank(); ++i) {
    // 1.1. Already marked as in-bounds, nothing to see here.
    if (op.isDimInBounds(i)) {
      newInBounds.push_back(true);
      continue;
    }
    // 1.2. Currently out-of-bounds, check whether we can statically determine
    // it is inBounds.
    bool inBounds = false;
    auto dimExpr = dyn_cast<AffineDimExpr>(permutationMap.getResult(i));
    if (dimExpr) {
      inBounds = isInBounds(op, /*resultIdx=*/i,
                            /*indicesIdx=*/dimExpr.getPosition());
      nonBcastDims.push_back(i);
    }

    newInBounds.push_back(inBounds);
    // We commit the pattern if it is "more inbounds".
    changed |= inBounds;
  }

  // 2. Handle broadcast dims
  // If all non-broadcast dims are "in bounds", then all bcast dims should be
  // "in bounds" as well.
  bool allNonBcastDimsInBounds = llvm::all_of(
      nonBcastDims, [&newInBounds](unsigned idx) { return newInBounds[idx]; });
  if (allNonBcastDimsInBounds) {
    for (size_t idx : permutationMap.getBroadcastDims()) {
      changed |= !newInBounds[idx];
      newInBounds[idx] = true;
    }
  }

  if (!changed)
    return failure();
  // OpBuilder is only used as a helper to build an I64ArrayAttr.
  OpBuilder b(op.getContext());
  op.setInBoundsAttr(b.getBoolArrayAttr(newInBounds));
  return success();
}

template <typename TransferOp>
static LogicalResult foldTransferFullMask(TransferOp op) {
  auto mask = op.getMask();
  if (!mask)
    return failure();

  if (getMaskFormat(mask) != MaskFormat::AllTrue)
    return failure();

  op.getMaskMutable().clear();
  return success();
}

///  ```
///  %w0 = vector.transfer_write %v0, %arg0[%c1, %c0] {in_bounds = [true, true]}
///    : vector<1x4xf32>, tensor<4x4xf32>
///  %0 = vector.transfer_read %w0[%c1, %c0], %cf0 {in_bounds = [true, true]}
///    : tensor<4x4xf32>, vector<1x4xf32>
///  ```
///  -> Folds into
///  ```
///  %v0
///  ```
static Value foldRAW(TransferReadOp readOp) {
  if (!llvm::isa<RankedTensorType>(readOp.getShapedType()))
    return {};
  auto defWrite = readOp.getBase().getDefiningOp<vector::TransferWriteOp>();
  while (defWrite) {
    if (checkSameValueRAW(defWrite, readOp))
      return defWrite.getVector();
    if (!isDisjointTransferIndices(
            cast<VectorTransferOpInterface>(defWrite.getOperation()),
            cast<VectorTransferOpInterface>(readOp.getOperation())))
      break;
    defWrite = defWrite.getBase().getDefiningOp<vector::TransferWriteOp>();
  }
  return {};
}

OpFoldResult TransferReadOp::fold(FoldAdaptor) {
  if (Value vec = foldRAW(*this))
    return vec;
  /// transfer_read(memrefcast) -> transfer_read
  if (succeeded(foldTransferInBoundsAttribute(*this)))
    return getResult();
  if (succeeded(foldTransferFullMask(*this)))
    return getResult();
  if (succeeded(memref::foldMemRefCast(*this)))
    return getResult();
  if (succeeded(tensor::foldTensorCast(*this)))
    return getResult();
  return OpFoldResult();
}

std::optional<SmallVector<int64_t, 4>> TransferReadOp::getShapeForUnroll() {
  return llvm::to_vector<4>(getVectorType().getShape());
}

void TransferReadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (llvm::isa<MemRefType>(getShapedType()))
    effects.emplace_back(MemoryEffects::Read::get(), &getBaseMutable(),
                         SideEffects::DefaultResource::get());
}

Speculation::Speculatability TransferReadOp::getSpeculatability() {
  if (hasPureTensorSemantics())
    return Speculation::Speculatable;
  return Speculation::NotSpeculatable;
}

namespace {
/// Store to load forwarding for transfer operations with permuation maps.
/// Even if the permutation maps are different we can still propagate the store
/// into the load if the size of the dimensions read and written match. Then we
/// can replace the transfer_read + transfer_write by vector.broadcast and
/// vector.transpose.
/// Example:
/// ```
/// %w0 = vector.transfer_write %v0, %arg0[%c0, %c0, %c0]
///  {in_bounds = [true, true],
///   permutation_map = affine_map<(d0, d1, d2) -> (d2, d1)>} :
///   vector<4x1xf32>, tensor<4x4x4xf32>
///  %r = vector.transfer_read %w0[%c0, %c0, %c0], %cf0
///   {in_bounds = [true, true, true, true],
///   permutation_map = affine_map<(d0, d1, d2) -> (d1, 0, d2, 0)>} :
///   tensor<4x4x4xf32>, vector<1x100x4x5xf32>
/// ```
/// To:
/// ```
/// %0 = vector.broadcast %arg1 : vector<4x1xf32> to vector<100x5x4x1xf32>
/// %r = vector.transpose %0, [3, 0, 2, 1] :
///   vector<100x5x4x1xf32> to vector<1x100x4x5xf32>
/// ```
struct TransferReadAfterWriteToBroadcast
    : public OpRewritePattern<TransferReadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TransferReadOp readOp,
                                PatternRewriter &rewriter) const override {
    auto defWrite = readOp.getBase().getDefiningOp<vector::TransferWriteOp>();
    if (!defWrite)
      return failure();
    // Bail if we need an alias analysis.
    if (!readOp.hasPureTensorSemantics() || !defWrite.hasPureTensorSemantics())
      return failure();
    // Bail if we need a bounds analysis.
    if (readOp.hasOutOfBoundsDim() || defWrite.hasOutOfBoundsDim())
      return failure();
    // TODO: If the written transfer chunk is a superset of the read transfer
    // chunk we could do an extract_strided_slice.
    if (readOp.getTransferChunkAccessed() !=
        defWrite.getTransferChunkAccessed())
      return failure();
    // TODO: Support cases where a dim is explicitly written but implicitly
    // read (i.e., a unit dim that is rank reduced).
    if (getUnusedDimsBitVector({readOp.getPermutationMap()}) !=
        getUnusedDimsBitVector({defWrite.getPermutationMap()}))
      return failure();
    // This pattern should only catch the broadcast case, the non-broadcast case
    // should be done separately to keep application conditions clean and
    // separate.
    AffineMap readMap = compressUnusedDims(readOp.getPermutationMap());
    AffineMap writeMap = compressUnusedDims(defWrite.getPermutationMap());
    bool bcast = !readMap.getBroadcastDims().empty() ||
                 !writeMap.getBroadcastDims().empty();
    if (!bcast)
      return failure();
    // At this point, we know we have a bcast.
    // Bail in the masked case (too complex atm and needed to properly account
    // for padding).
    if (readOp.getMask() || defWrite.getMask())
      return failure();
    // If indices are not the same a shift may be required, bail.
    if (readOp.getIndices() != defWrite.getIndices())
      return failure();

    Value vec = defWrite.getVector();
    // TODO: loop through the chain of transfer_write if we can prove that they
    // don't overlap with the transfer_read. This requires improving
    // `isDisjointTransferIndices` helper.
    AffineMap map = readMap.compose(writeMap);
    if (map.getNumResults() == 0)
      return failure();
    // Calculate the permutation to apply to go from the vector stored to the
    // vector read.
    SmallVector<unsigned> permutation;
    if (!map.isPermutationOfMinorIdentityWithBroadcasting(permutation))
      return failure();

    Location loc = readOp.getLoc();
    // Calculate the broadcast shape by applying the reverse permutation to the
    // final shape we want.
    ArrayRef<int64_t> destShape = readOp.getVectorType().getShape();
    SmallVector<int64_t> broadcastShape(destShape.size());
    SmallVector<bool> broadcastScalableFlags(destShape.size());
    for (const auto &pos : llvm::enumerate(permutation)) {
      broadcastShape[pos.value()] = destShape[pos.index()];
      broadcastScalableFlags[pos.value()] =
          readOp.getVectorType().getScalableDims()[pos.index()];
    }
    VectorType broadcastedType = VectorType::get(
        broadcastShape, defWrite.getVectorType().getElementType(),
        broadcastScalableFlags);
    vec = vector::BroadcastOp::create(rewriter, loc, broadcastedType, vec);
    SmallVector<int64_t> transposePerm(permutation.begin(), permutation.end());
    rewriter.replaceOpWithNewOp<vector::TransposeOp>(readOp, vec,
                                                     transposePerm);
    return success();
  }
};
} // namespace

void TransferReadOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
  results.add<TransferReadAfterWriteToBroadcast>(context);
}

//===----------------------------------------------------------------------===//
// TransferWriteOp
//===----------------------------------------------------------------------===//

/// 1. Builder with type inference.
void TransferWriteOp::build(OpBuilder &builder, OperationState &result,
                            Value vector, Value dest, ValueRange indices,
                            AffineMapAttr permutationMapAttr,
                            /*optional*/ Value mask,
                            /*optional*/ ArrayAttr inBoundsAttr) {
  Type resultType = llvm::dyn_cast<RankedTensorType>(dest.getType());
  build(builder, result, resultType, vector, dest, indices, permutationMapAttr,
        mask, inBoundsAttr);
}

/// 2. Builder with type inference that sets an empty mask (variant with attrs).
void TransferWriteOp::build(OpBuilder &builder, OperationState &result,
                            Value vector, Value dest, ValueRange indices,
                            AffineMapAttr permutationMapAttr,
                            /*optional*/ ArrayAttr inBoundsAttr) {
  build(builder, result, vector, dest, indices, permutationMapAttr,
        /*mask=*/Value(), inBoundsAttr);
}

/// 3. Builder with type inference that sets an empty mask (variant without
/// attrs)
void TransferWriteOp::build(OpBuilder &builder, OperationState &result,
                            Value vector, Value dest, ValueRange indices,
                            AffineMap permutationMap,
                            std::optional<ArrayRef<bool>> inBounds) {
  auto permutationMapAttr = AffineMapAttr::get(permutationMap);
  auto inBoundsAttr =
      (inBounds && !inBounds.value().empty())
          ? builder.getBoolArrayAttr(inBounds.value())
          : builder.getBoolArrayAttr(SmallVector<bool>(
                llvm::cast<VectorType>(vector.getType()).getRank(), false));
  build(builder, result, vector, dest, indices, permutationMapAttr,
        /*mask=*/Value(), inBoundsAttr);
}

/// 4. Builder with type inference that sets an empty mask and sets permutation
///    map to 'getMinorIdentityMap'.
void TransferWriteOp::build(OpBuilder &builder, OperationState &result,
                            Value vector, Value dest, ValueRange indices,
                            std::optional<ArrayRef<bool>> inBounds) {
  auto vectorType = llvm::cast<VectorType>(vector.getType());
  AffineMap permutationMap = getTransferMinorIdentityMap(
      llvm::cast<ShapedType>(dest.getType()), vectorType);
  build(builder, result, vector, dest, indices, permutationMap, inBounds);
}

ParseResult TransferWriteOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
  auto &builder = parser.getBuilder();
  SMLoc typesLoc;
  OpAsmParser::UnresolvedOperand vectorInfo, sourceInfo;
  SmallVector<OpAsmParser::UnresolvedOperand, 8> indexInfo;
  SmallVector<Type, 2> types;
  OpAsmParser::UnresolvedOperand maskInfo;
  if (parser.parseOperand(vectorInfo) || parser.parseComma() ||
      parser.parseOperand(sourceInfo) ||
      parser.parseOperandList(indexInfo, OpAsmParser::Delimiter::Square))
    return failure();
  ParseResult hasMask = parser.parseOptionalComma();
  if (hasMask.succeeded() && parser.parseOperand(maskInfo))
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.getCurrentLocation(&typesLoc) || parser.parseColonTypeList(types))
    return failure();
  if (types.size() != 2)
    return parser.emitError(typesLoc, "requires two types");
  auto indexType = builder.getIndexType();
  VectorType vectorType = llvm::dyn_cast<VectorType>(types[0]);
  if (!vectorType)
    return parser.emitError(typesLoc, "requires vector type");
  ShapedType shapedType = llvm::dyn_cast<ShapedType>(types[1]);
  if (!shapedType || !llvm::isa<MemRefType, RankedTensorType>(shapedType))
    return parser.emitError(typesLoc, "requires memref or ranked tensor type");
  auto permMapAttrName =
      TransferWriteOp::getPermutationMapAttrName(result.name);
  auto permMapAttr = result.attributes.get(permMapAttrName);
  AffineMap permMap;
  if (!permMapAttr) {
    if (shapedType.getRank() <
        getEffectiveVectorRankForXferOp(shapedType, vectorType))
      return parser.emitError(typesLoc,
                              "expected a custom permutation_map when "
                              "rank(source) != rank(destination)");
    permMap = getTransferMinorIdentityMap(shapedType, vectorType);
    result.attributes.set(permMapAttrName, AffineMapAttr::get(permMap));
  } else {
    permMap = llvm::cast<AffineMapAttr>(permMapAttr).getValue();
  }
  auto inBoundsAttrName = TransferWriteOp::getInBoundsAttrName(result.name);
  Attribute inBoundsAttr = result.attributes.get(inBoundsAttrName);
  if (!inBoundsAttr) {
    result.addAttribute(inBoundsAttrName,
                        builder.getBoolArrayAttr(
                            SmallVector<bool>(permMap.getNumResults(), false)));
  }
  if (parser.resolveOperand(vectorInfo, vectorType, result.operands) ||
      parser.resolveOperand(sourceInfo, shapedType, result.operands) ||
      parser.resolveOperands(indexInfo, indexType, result.operands))
    return failure();
  if (hasMask.succeeded()) {
    if (llvm::dyn_cast<VectorType>(shapedType.getElementType()))
      return parser.emitError(
          maskInfo.location, "does not support masks with vector element type");
    if (vectorType.getRank() != permMap.getNumResults()) {
      return parser.emitError(typesLoc,
                              "expected the same rank for the vector and the "
                              "results of the permutation map");
    }
    auto maskType = inferTransferOpMaskType(vectorType, permMap);
    if (parser.resolveOperand(maskInfo, maskType, result.operands))
      return failure();
  }
  result.addAttribute(TransferWriteOp::getOperandSegmentSizeAttr(),
                      builder.getDenseI32ArrayAttr(
                          {1, 1, static_cast<int32_t>(indexInfo.size()),
                           static_cast<int32_t>(hasMask.succeeded())}));
  return failure(llvm::isa<RankedTensorType>(shapedType) &&
                 parser.addTypeToList(shapedType, result.types));
}

void TransferWriteOp::print(OpAsmPrinter &p) {
  p << " " << getVector() << ", " << getBase() << "[" << getIndices() << "]";
  if (getMask())
    p << ", " << getMask();
  printTransferAttrs(p, *this);
  p << " : " << getVectorType() << ", " << getShapedType();
}

LogicalResult TransferWriteOp::verify() {
  // Consistency of elemental types in shape and vector.
  ShapedType shapedType = getShapedType();
  VectorType vectorType = getVectorType();
  VectorType maskType = getMaskType();
  auto permutationMap = getPermutationMap();
  VectorType inferredMaskType =
      maskType ? inferTransferOpMaskType(vectorType, permutationMap)
               : VectorType();

  if (llvm::size(getIndices()) != shapedType.getRank())
    return emitOpError("requires ") << shapedType.getRank() << " indices";

  // We do not allow broadcast dimensions on TransferWriteOps for the moment,
  // as the semantics is unclear. This can be revisited later if necessary.
  if (hasBroadcastDim())
    return emitOpError("should not have broadcast dimensions");

  if (failed(verifyTransferOp(cast<VectorTransferOpInterface>(getOperation()),
                              shapedType, vectorType, maskType,
                              inferredMaskType, permutationMap, getInBounds())))
    return failure();

  return verifyPermutationMap(permutationMap,
                              [&](Twine t) { return emitOpError(t); });
}

//===----------------------------------------------------------------------===//
// TransferWriteOp: MaskableOpInterface methods.
//===----------------------------------------------------------------------===//

/// Returns the mask type expected by this operation. Mostly used for
/// verification purposes.
Type TransferWriteOp::getExpectedMaskType() {
  return inferTransferOpMaskType(getVectorType(), getPermutationMap());
}

//===----------------------------------------------------------------------===//
// TransferWriteOp: VectorTransferOpInterface methods.
//===----------------------------------------------------------------------===//
Value TransferWriteOp::getVector() { return getOperand(0); }
VectorType TransferWriteOp::getVectorType() {
  return cast<VectorType>(getValueToStore().getType());
}

//===----------------------------------------------------------------------===//
// TransferWriteOp: fold methods.
//===----------------------------------------------------------------------===//
/// Fold:
/// ```
///    %t1 = ...
///    %v = vector.transfer_read %t0[%c0...], {in_bounds = [true...]} :
///      tensor<static_sizesxf32>, vector<static_sizesxf32>
///    %t2 = vector.transfer_write %v, %t1[%c0...] {in_bounds = [true...]} :
///      vector<static_sizesxf32>, tensor<static_sizesxf32>
/// ```
///
/// into:
///
/// ```
///    %t0
/// ```
///
/// The producer of t1 may or may not be DCE'd depending on whether it is a
/// block argument or has side effects.
static LogicalResult foldReadInitWrite(TransferWriteOp write,
                                       ArrayRef<Attribute>,
                                       SmallVectorImpl<OpFoldResult> &results) {
  // TODO: support 0-d corner case.
  if (write.getTransferRank() == 0)
    return failure();
  auto rankedTensorType =
      llvm::dyn_cast<RankedTensorType>(write.getBase().getType());
  // If not operating on tensors, bail.
  if (!rankedTensorType)
    return failure();
  // If no read, bail.
  auto read = write.getVector().getDefiningOp<vector::TransferReadOp>();
  if (!read)
    return failure();
  // TODO: support 0-d corner case.
  if (read.getTransferRank() == 0)
    return failure();
  // For now, only accept minor identity. Future: composition is minor identity.
  if (!read.getPermutationMap().isMinorIdentity() ||
      !write.getPermutationMap().isMinorIdentity())
    return failure();
  // Bail on mismatching ranks.
  if (read.getTransferRank() != write.getTransferRank())
    return failure();
  // Bail on potential out-of-bounds accesses.
  if (read.hasOutOfBoundsDim() || write.hasOutOfBoundsDim())
    return failure();
  // Tensor types must be the same.
  if (read.getBase().getType() != rankedTensorType)
    return failure();
  // Vector types must be the same.
  if (read.getVectorType() != write.getVectorType())
    return failure();
  // Vector and Tensor shapes must match.
  if (read.getVectorType().getShape() != rankedTensorType.getShape())
    return failure();
  // If any index is nonzero.
  auto isNotConstantZero = [](Value v) {
    auto cstOp = getConstantIntValue(v);
    return !cstOp.has_value() || cstOp.value() != 0;
  };
  if (llvm::any_of(read.getIndices(), isNotConstantZero) ||
      llvm::any_of(write.getIndices(), isNotConstantZero))
    return failure();
  // Success.
  results.push_back(read.getBase());
  return success();
}

static bool checkSameValueWAR(vector::TransferReadOp read,
                              vector::TransferWriteOp write) {
  return read.getBase() == write.getBase() &&
         read.getIndices() == write.getIndices() &&
         read.getPermutationMap() == write.getPermutationMap() &&
         read.getVectorType() == write.getVectorType() && !read.getMask() &&
         !write.getMask();
}
/// Fold transfer_write write after read:
/// ```
///    %t0 = ...
///    %v = vector.transfer_read %t0[%c0...] :
///      tensor<static_sizesxf32>, vector<static_sizesxf32>
///    %t1 = vector.transfer_write %v, %t0[%c0...] :
///      vector<static_sizesxf32>, tensor<static_sizesxf32>
/// ```
///
/// into:
///
/// ```
///    %t0
/// ```
static LogicalResult foldWAR(TransferWriteOp write,
                             SmallVectorImpl<OpFoldResult> &results) {
  if (!llvm::isa<RankedTensorType>(write.getBase().getType()))
    return failure();
  auto read = write.getVector().getDefiningOp<vector::TransferReadOp>();
  if (!read)
    return failure();

  if (!checkSameValueWAR(read, write))
    return failure();
  results.push_back(read.getBase());
  return success();
}

LogicalResult TransferWriteOp::fold(FoldAdaptor adaptor,
                                    SmallVectorImpl<OpFoldResult> &results) {
  if (succeeded(foldReadInitWrite(*this, adaptor.getOperands(), results)))
    return success();
  if (succeeded(foldWAR(*this, results)))
    return success();
  if (succeeded(foldTransferInBoundsAttribute(*this)))
    return success();
  if (succeeded(foldTransferFullMask(*this)))
    return success();
  return memref::foldMemRefCast(*this);
}

//===----------------------------------------------------------------------===//
// TransferWriteOp: other methods.
//===----------------------------------------------------------------------===//
std::optional<SmallVector<int64_t, 4>> TransferWriteOp::getShapeForUnroll() {
  return llvm::to_vector<4>(getVectorType().getShape());
}

void TransferWriteOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (llvm::isa<MemRefType>(getShapedType()))
    effects.emplace_back(MemoryEffects::Write::get(), &getBaseMutable(),
                         SideEffects::DefaultResource::get());
}

Speculation::Speculatability TransferWriteOp::getSpeculatability() {
  if (hasPureTensorSemantics())
    return Speculation::Speculatable;
  return Speculation::NotSpeculatable;
}

namespace {
/// Remove dead transfer write from the SSA chain so that it an be eliminated by
/// DCE
/// ```
///  %w0 = vector.transfer_write %v0, %arg0[%c1, %c0] {in_bounds = [true, true]}
///    : vector<1x4xf32>, tensor<4x4xf32>
///  %w1 = vector.transfer_write %v0, %w0[%c2, %c0] {in_bounds = [true, true]}
///    : vector<1x4xf32>, tensor<4x4xf32>
///  %w2 = vector.transfer_write %v1, %w1[%c1, %c0] {in_bounds = [true, true]}
///    : vector<1x4xf32>, tensor<4x4xf32>
/// ```
///
/// into:
///
/// ```
///  %w0 = vector.transfer_write %v0, %arg0[%c1, %c0] {in_bounds = [true, true]}
///    : vector<1x4xf32>, tensor<4x4xf32>
///  %w1 = vector.transfer_write %v0, %arg0[%c2, %c0] {in_bounds = [true, true]}
///    : vector<1x4xf32>, tensor<4x4xf32>
///  %w2 = vector.transfer_write %v1, %w1[%c1, %c0] {in_bounds = [true, true]}
///    : vector<1x4xf32>, tensor<4x4xf32>
/// ```
///
/// `%w0 = vector.transfer_write` op will be removed by DCE if it doesn't have
/// any other uses.
class FoldWaw final : public OpRewritePattern<TransferWriteOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TransferWriteOp writeOp,
                                PatternRewriter &rewriter) const override {
    if (!llvm::isa<RankedTensorType>(writeOp.getShapedType()))
      return failure();
    vector::TransferWriteOp writeToModify = writeOp;

    auto defWrite = writeOp.getBase().getDefiningOp<vector::TransferWriteOp>();
    while (defWrite) {
      if (checkSameValueWAW(writeOp, defWrite)) {
        rewriter.modifyOpInPlace(writeToModify, [&]() {
          writeToModify.getBaseMutable().assign(defWrite.getBase());
        });
        return success();
      }
      if (!isDisjointTransferIndices(
              cast<VectorTransferOpInterface>(defWrite.getOperation()),
              cast<VectorTransferOpInterface>(writeOp.getOperation())))
        break;
      // If the previous write op doesn't have any other use we an safely look
      // at the previous store to see if it can be removed.
      if (!defWrite->hasOneUse())
        break;
      writeToModify = defWrite;
      defWrite = defWrite.getBase().getDefiningOp<vector::TransferWriteOp>();
    }
    return failure();
  }
};

/// Rewrite tensor::ExtractSliceOp(vector::TransferWriteOp) to
/// vector::TransferWriteOp(tensor::ExtractSliceOp) if the full slice is
/// overwritten and inserted into another tensor. After this rewrite, the
/// operations bufferize in-place since all of them work on the same slice.
///
/// For example:
/// ```mlir
///   %0 = vector.transfer_write %vec, %init_tensor[%c0, %c0]
///        : vector<8x16xf32>, tensor<8x16xf32>
///   %1 = tensor.extract_slice %0[0, 0] [%sz0, %sz1] [1, 1]
///        : tensor<8x16xf32> to tensor<?x?xf32>
///   %r = tensor.insert_slice %1 into %iter_arg[%iv0, %iv1] [%sz0, %sz1] [1, 1]
///        : tensor<?x?xf32> into tensor<27x37xf32>
/// ```
/// folds to
/// ```mlir
///   %0 = tensor.extract_slice %iter_arg[%iv0, %iv1] [%sz0, %sz1] [1, 1]
///        : tensor<27x37xf32> to tensor<?x?xf32>
///   %1 = vector.transfer_write %vec, %0[%c0, %c0]
///        : vector<8x16xf32>, tensor<?x?xf32>
///   %r = tensor.insert_slice %1 into %iter_arg[%iv0, %iv1] [%sz0, %sz1] [1, 1]
///        : tensor<?x?xf32> into tensor<27x37xf32>
/// ```
struct SwapExtractSliceOfTransferWrite
    : public OpRewritePattern<tensor::InsertSliceOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertSliceOp insertOp,
                                PatternRewriter &rewriter) const override {
    if (!insertOp.hasUnitStride())
      return failure();
    auto extractOp =
        insertOp.getSource().getDefiningOp<tensor::ExtractSliceOp>();
    if (!extractOp || !extractOp.hasUnitStride() || !extractOp->hasOneUse())
      return failure();
    auto transferOp = extractOp.getSource().getDefiningOp<TransferWriteOp>();
    if (!transferOp || !transferOp->hasOneUse())
      return failure();

    // Fail if vector::TransferWriteOp or tensor::ExtractSliceOp is
    // rank-reducing.
    if (insertOp.getSourceType().getRank() != transferOp.getTransferRank()) {
      return rewriter.notifyMatchFailure(insertOp,
                                         "use-def chain is rank-reducing");
    }

    // Fail if tensor::ExtractSliceOp has non-zero offset.
    if (!extractOp.hasZeroOffset()) {
      return rewriter.notifyMatchFailure(insertOp,
                                         "ExtractSliceOp has non-zero offset");
    }

    // Fail if tensor::TransferWriteOp has non-zero offset.
    if (!llvm::all_of(transferOp.getIndices(), [](Value value) {
          return getConstantIntValue(value) == static_cast<int64_t>(0);
        })) {
      return rewriter.notifyMatchFailure(insertOp,
                                         "TranferWriteOp has non-zero offset");
    }

    // Fail if tensor::ExtractSliceOp and tensor::InsertSliceOp sizes differ.
    if (insertOp.getMixedSizes().size() != extractOp.getMixedSizes().size()) {
      return rewriter.notifyMatchFailure(
          insertOp, "InsertSliceOp and ExtractSliceOp ranks differ");
    }

    for (auto [insertSize, extractSize] :
         llvm::zip_equal(insertOp.getMixedSizes(), extractOp.getMixedSizes())) {
      if (!isEqualConstantIntOrValue(insertSize, extractSize)) {
        return rewriter.notifyMatchFailure(
            insertOp, "InsertSliceOp and ExtractSliceOp sizes differ");
      }
    }

    // Fail if the vector::TransferWriteOp may not overwrite the full tensor.
    assert(transferOp.getVectorType().hasStaticShape() &&
           "expected vector to have a static shape");
    ArrayRef<int64_t> vectorShape = transferOp.getVectorType().getShape();
    SmallVector<int64_t> resultShape = applyPermutationMap(
        transferOp.getPermutationMap(), transferOp.getShapedType().getShape());
    if (transferOp.getMask() || !vectorShape.equals(resultShape)) {
      return rewriter.notifyMatchFailure(
          insertOp, "TransferWriteOp may not write the full tensor.");
    }

    // Swap the tensor::ExtractSliceOp in front of the vector::TransferWriteOp.
    // Set all in_bounds to false and let the folder infer them.
    SmallVector<bool> newInBounds(vectorShape.size(), false);
    auto newExtractOp = tensor::ExtractSliceOp::create(
        rewriter, extractOp.getLoc(), insertOp.getSourceType(),
        insertOp.getDest(), insertOp.getMixedOffsets(),
        insertOp.getMixedSizes(), insertOp.getMixedStrides());
    auto newTransferWriteOp = TransferWriteOp::create(
        rewriter, transferOp.getLoc(), transferOp.getVector(),
        newExtractOp.getResult(), transferOp.getIndices(),
        transferOp.getPermutationMapAttr(),
        rewriter.getBoolArrayAttr(newInBounds));
    rewriter.modifyOpInPlace(insertOp, [&]() {
      insertOp.getSourceMutable().assign(newTransferWriteOp.getResult());
    });
    return success();
  }
};

} // namespace

void TransferWriteOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {
  results.add<FoldWaw, SwapExtractSliceOfTransferWrite>(context);
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyLoadStoreMemRefLayout(Operation *op,
                                                 VectorType vecTy,
                                                 MemRefType memRefTy) {
  // If rank==0 or size==1 it's equivalent to scalar load/store, so we don't
  // need any strides limitations.
  if (!vecTy.isScalable() &&
      (vecTy.getRank() == 0 || vecTy.getNumElements() == 1))
    return success();

  if (!memRefTy.isLastDimUnitStride())
    return op->emitOpError("most minor memref dim must have unit stride");
  return success();
}

LogicalResult vector::LoadOp::verify() {
  VectorType resVecTy = getVectorType();
  MemRefType memRefTy = getMemRefType();

  if (failed(verifyLoadStoreMemRefLayout(*this, resVecTy, memRefTy)))
    return failure();

  if (memRefTy.getRank() < resVecTy.getRank())
    return emitOpError(
        "destination memref has lower rank than the result vector");

  // Checks for vector memrefs.
  Type memElemTy = memRefTy.getElementType();
  if (auto memVecTy = llvm::dyn_cast<VectorType>(memElemTy)) {
    if (memVecTy != resVecTy)
      return emitOpError("base memref and result vector types should match");
    memElemTy = memVecTy.getElementType();
  }

  if (resVecTy.getElementType() != memElemTy)
    return emitOpError("base and result element types should match");
  if (llvm::size(getIndices()) != memRefTy.getRank())
    return emitOpError("requires ") << memRefTy.getRank() << " indices";
  return success();
}

OpFoldResult LoadOp::fold(FoldAdaptor) {
  if (succeeded(memref::foldMemRefCast(*this)))
    return getResult();
  return OpFoldResult();
}

std::optional<SmallVector<int64_t, 4>> LoadOp::getShapeForUnroll() {
  return llvm::to_vector<4>(getVectorType().getShape());
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

LogicalResult vector::StoreOp::verify() {
  VectorType valueVecTy = getVectorType();
  MemRefType memRefTy = getMemRefType();

  if (failed(verifyLoadStoreMemRefLayout(*this, valueVecTy, memRefTy)))
    return failure();

  if (memRefTy.getRank() < valueVecTy.getRank())
    return emitOpError("source memref has lower rank than the vector to store");

  // Checks for vector memrefs.
  Type memElemTy = memRefTy.getElementType();
  if (auto memVecTy = llvm::dyn_cast<VectorType>(memElemTy)) {
    if (memVecTy != valueVecTy)
      return emitOpError(
          "base memref and valueToStore vector types should match");
    memElemTy = memVecTy.getElementType();
  }

  if (valueVecTy.getElementType() != memElemTy)
    return emitOpError("base and valueToStore element type should match");
  if (llvm::size(getIndices()) != memRefTy.getRank())
    return emitOpError("requires ") << memRefTy.getRank() << " indices";
  return success();
}

LogicalResult StoreOp::fold(FoldAdaptor adaptor,
                            SmallVectorImpl<OpFoldResult> &results) {
  return memref::foldMemRefCast(*this);
}

std::optional<SmallVector<int64_t, 4>> StoreOp::getShapeForUnroll() {
  return llvm::to_vector<4>(getVectorType().getShape());
}

//===----------------------------------------------------------------------===//
// MaskedLoadOp
//===----------------------------------------------------------------------===//

LogicalResult MaskedLoadOp::verify() {
  VectorType maskVType = getMaskVectorType();
  VectorType passVType = getPassThruVectorType();
  VectorType resVType = getVectorType();
  MemRefType memType = getMemRefType();

  if (resVType.getElementType() != memType.getElementType())
    return emitOpError("base and result element type should match");
  if (llvm::size(getIndices()) != memType.getRank())
    return emitOpError("requires ") << memType.getRank() << " indices";
  if (resVType.getShape() != maskVType.getShape())
    return emitOpError("expected result shape to match mask shape");
  if (resVType != passVType)
    return emitOpError("expected pass_thru of same type as result type");
  return success();
}

namespace {
class MaskedLoadFolder final : public OpRewritePattern<MaskedLoadOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(MaskedLoadOp load,
                                PatternRewriter &rewriter) const override {
    switch (getMaskFormat(load.getMask())) {
    case MaskFormat::AllTrue:
      rewriter.replaceOpWithNewOp<vector::LoadOp>(
          load, load.getType(), load.getBase(), load.getIndices());
      return success();
    case MaskFormat::AllFalse:
      rewriter.replaceOp(load, load.getPassThru());
      return success();
    case MaskFormat::Unknown:
      return failure();
    }
    llvm_unreachable("Unexpected 1DMaskFormat on MaskedLoad");
  }
};
} // namespace

void MaskedLoadOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.add<MaskedLoadFolder>(context);
}

OpFoldResult MaskedLoadOp::fold(FoldAdaptor) {
  if (succeeded(memref::foldMemRefCast(*this)))
    return getResult();
  return OpFoldResult();
}

//===----------------------------------------------------------------------===//
// MaskedStoreOp
//===----------------------------------------------------------------------===//

LogicalResult MaskedStoreOp::verify() {
  VectorType maskVType = getMaskVectorType();
  VectorType valueVType = getVectorType();
  MemRefType memType = getMemRefType();

  if (valueVType.getElementType() != memType.getElementType())
    return emitOpError("base and valueToStore element type should match");
  if (llvm::size(getIndices()) != memType.getRank())
    return emitOpError("requires ") << memType.getRank() << " indices";
  if (valueVType.getShape() != maskVType.getShape())
    return emitOpError("expected valueToStore shape to match mask shape");
  return success();
}

namespace {
class MaskedStoreFolder final : public OpRewritePattern<MaskedStoreOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(MaskedStoreOp store,
                                PatternRewriter &rewriter) const override {
    switch (getMaskFormat(store.getMask())) {
    case MaskFormat::AllTrue:
      rewriter.replaceOpWithNewOp<vector::StoreOp>(
          store, store.getValueToStore(), store.getBase(), store.getIndices());
      return success();
    case MaskFormat::AllFalse:
      rewriter.eraseOp(store);
      return success();
    case MaskFormat::Unknown:
      return failure();
    }
    llvm_unreachable("Unexpected 1DMaskFormat on MaskedStore");
  }
};
} // namespace

void MaskedStoreOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                MLIRContext *context) {
  results.add<MaskedStoreFolder>(context);
}

LogicalResult MaskedStoreOp::fold(FoldAdaptor adaptor,
                                  SmallVectorImpl<OpFoldResult> &results) {
  return memref::foldMemRefCast(*this);
}

//===----------------------------------------------------------------------===//
// GatherOp
//===----------------------------------------------------------------------===//

LogicalResult GatherOp::verify() {
  VectorType indVType = getIndexVectorType();
  VectorType maskVType = getMaskVectorType();
  VectorType resVType = getVectorType();
  ShapedType baseType = getBaseType();

  if (!llvm::isa<MemRefType, RankedTensorType>(baseType))
    return emitOpError("requires base to be a memref or ranked tensor type");

  if (resVType.getElementType() != baseType.getElementType())
    return emitOpError("base and result element type should match");
  if (llvm::size(getOffsets()) != baseType.getRank())
    return emitOpError("requires ") << baseType.getRank() << " indices";
  if (resVType.getShape() != indVType.getShape())
    return emitOpError("expected result dim to match indices dim");
  if (resVType.getShape() != maskVType.getShape())
    return emitOpError("expected result dim to match mask dim");
  if (resVType != getPassThruVectorType())
    return emitOpError("expected pass_thru of same type as result type");
  return success();
}

// MaskableOpInterface methods.

/// Returns the mask type expected by this operation. Mostly used for
/// verification purposes. It requires the operation to be vectorized."
Type GatherOp::getExpectedMaskType() {
  auto vecType = this->getIndexVectorType();
  return VectorType::get(vecType.getShape(),
                         IntegerType::get(vecType.getContext(), /*width=*/1),
                         vecType.getScalableDims());
}

std::optional<SmallVector<int64_t, 4>> GatherOp::getShapeForUnroll() {
  return llvm::to_vector<4>(getVectorType().getShape());
}

/// Cheeck if `indexVec` is constant 1D vec of consecutive values [0, 1, 2, ...]
static LogicalResult isZeroBasedContiguousSeq(Value indexVec) {
  auto vecType = dyn_cast<VectorType>(indexVec.getType());
  if (!vecType || vecType.getRank() != 1 || vecType.isScalable())
    return failure();

  if (indexVec.getDefiningOp<StepOp>())
    return success();

  DenseIntElementsAttr elements;
  if (!matchPattern(indexVec, m_Constant(&elements)))
    return failure();

  return success(
      llvm::equal(elements, llvm::seq<int64_t>(0, vecType.getNumElements())));
}

namespace {
class GatherFolder final : public OpRewritePattern<GatherOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(GatherOp gather,
                                PatternRewriter &rewriter) const override {
    switch (getMaskFormat(gather.getMask())) {
    case MaskFormat::AllTrue:
      return failure(); // no unmasked equivalent
    case MaskFormat::AllFalse:
      rewriter.replaceOp(gather, gather.getPassThru());
      return success();
    case MaskFormat::Unknown:
      return failure();
    }
    llvm_unreachable("Unexpected 1DMaskFormat on GatherFolder");
  }
};

/// Fold gathers with consecutive offsets [0, 1, 2, ...] into contiguous
/// maskedload. Only 1D fixed vectors are supported for now.
class FoldContiguousGather final : public OpRewritePattern<GatherOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(GatherOp op,
                                PatternRewriter &rewriter) const override {
    if (!isa<MemRefType>(op.getBase().getType()))
      return rewriter.notifyMatchFailure(op, "base must be of memref type");

    if (failed(isZeroBasedContiguousSeq(op.getIndices())))
      return failure();

    rewriter.replaceOpWithNewOp<MaskedLoadOp>(op, op.getType(), op.getBase(),
                                              op.getOffsets(), op.getMask(),
                                              op.getPassThru());
    return success();
  }
};
} // namespace

void GatherOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.add<GatherFolder, FoldContiguousGather>(context);
}

//===----------------------------------------------------------------------===//
// ScatterOp
//===----------------------------------------------------------------------===//

LogicalResult ScatterOp::verify() {
  VectorType indVType = getIndexVectorType();
  VectorType maskVType = getMaskVectorType();
  VectorType valueVType = getVectorType();
  MemRefType memType = getMemRefType();

  if (valueVType.getElementType() != memType.getElementType())
    return emitOpError("base and valueToStore element type should match");
  if (llvm::size(getOffsets()) != memType.getRank())
    return emitOpError("requires ") << memType.getRank() << " indices";
  if (valueVType.getShape() != indVType.getShape())
    return emitOpError("expected valueToStore dim to match indices dim");
  if (valueVType.getShape() != maskVType.getShape())
    return emitOpError("expected valueToStore dim to match mask dim");
  return success();
}

namespace {
class ScatterFolder final : public OpRewritePattern<ScatterOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ScatterOp scatter,
                                PatternRewriter &rewriter) const override {
    switch (getMaskFormat(scatter.getMask())) {
    case MaskFormat::AllTrue:
      return failure(); // no unmasked equivalent
    case MaskFormat::AllFalse:
      rewriter.eraseOp(scatter);
      return success();
    case MaskFormat::Unknown:
      return failure();
    }
    llvm_unreachable("Unexpected 1DMaskFormat on ScatterFolder");
  }
};

/// Fold scatters with consecutive offsets [0, 1, 2, ...] into contiguous
/// maskedstore. Only 1D fixed vectors are supported for now.
class FoldContiguousScatter final : public OpRewritePattern<ScatterOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ScatterOp op,
                                PatternRewriter &rewriter) const override {
    if (failed(isZeroBasedContiguousSeq(op.getIndices())))
      return failure();

    rewriter.replaceOpWithNewOp<MaskedStoreOp>(
        op, op.getBase(), op.getOffsets(), op.getMask(), op.getValueToStore());
    return success();
  }
};
} // namespace

void ScatterOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<ScatterFolder, FoldContiguousScatter>(context);
}

//===----------------------------------------------------------------------===//
// ExpandLoadOp
//===----------------------------------------------------------------------===//

LogicalResult ExpandLoadOp::verify() {
  VectorType maskVType = getMaskVectorType();
  VectorType passVType = getPassThruVectorType();
  VectorType resVType = getVectorType();
  MemRefType memType = getMemRefType();

  if (resVType.getElementType() != memType.getElementType())
    return emitOpError("base and result element type should match");
  if (llvm::size(getIndices()) != memType.getRank())
    return emitOpError("requires ") << memType.getRank() << " indices";
  if (resVType.getDimSize(0) != maskVType.getDimSize(0))
    return emitOpError("expected result dim to match mask dim");
  if (resVType != passVType)
    return emitOpError("expected pass_thru of same type as result type");
  return success();
}

namespace {
class ExpandLoadFolder final : public OpRewritePattern<ExpandLoadOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ExpandLoadOp expand,
                                PatternRewriter &rewriter) const override {
    switch (getMaskFormat(expand.getMask())) {
    case MaskFormat::AllTrue:
      rewriter.replaceOpWithNewOp<vector::LoadOp>(
          expand, expand.getType(), expand.getBase(), expand.getIndices());
      return success();
    case MaskFormat::AllFalse:
      rewriter.replaceOp(expand, expand.getPassThru());
      return success();
    case MaskFormat::Unknown:
      return failure();
    }
    llvm_unreachable("Unexpected 1DMaskFormat on ExpandLoadFolder");
  }
};
} // namespace

void ExpandLoadOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.add<ExpandLoadFolder>(context);
}

//===----------------------------------------------------------------------===//
// CompressStoreOp
//===----------------------------------------------------------------------===//

LogicalResult CompressStoreOp::verify() {
  VectorType maskVType = getMaskVectorType();
  VectorType valueVType = getVectorType();
  MemRefType memType = getMemRefType();

  if (valueVType.getElementType() != memType.getElementType())
    return emitOpError("base and valueToStore element type should match");
  if (llvm::size(getIndices()) != memType.getRank())
    return emitOpError("requires ") << memType.getRank() << " indices";
  if (valueVType.getDimSize(0) != maskVType.getDimSize(0))
    return emitOpError("expected valueToStore dim to match mask dim");
  return success();
}

namespace {
class CompressStoreFolder final : public OpRewritePattern<CompressStoreOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CompressStoreOp compress,
                                PatternRewriter &rewriter) const override {
    switch (getMaskFormat(compress.getMask())) {
    case MaskFormat::AllTrue:
      rewriter.replaceOpWithNewOp<vector::StoreOp>(
          compress, compress.getValueToStore(), compress.getBase(),
          compress.getIndices());
      return success();
    case MaskFormat::AllFalse:
      rewriter.eraseOp(compress);
      return success();
    case MaskFormat::Unknown:
      return failure();
    }
    llvm_unreachable("Unexpected 1DMaskFormat on CompressStoreFolder");
  }
};
} // namespace

void CompressStoreOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {
  results.add<CompressStoreFolder>(context);
}

//===----------------------------------------------------------------------===//
// ShapeCastOp
//===----------------------------------------------------------------------===//

void ShapeCastOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                    SetIntRangeFn setResultRanges) {
  setResultRanges(getResult(), argRanges.front());
}

LogicalResult ShapeCastOp::verify() {

  VectorType sourceType = getSourceVectorType();
  VectorType resultType = getResultVectorType();

  // Check that element type is preserved
  if (sourceType.getElementType() != resultType.getElementType())
    return emitOpError("has different source and result element types");

  // Check that number of elements is preserved
  int64_t sourceNElms = sourceType.getNumElements();
  int64_t resultNElms = resultType.getNumElements();
  if (sourceNElms != resultNElms) {
    return emitOpError() << "has different number of elements at source ("
                         << sourceNElms << ") and result (" << resultNElms
                         << ")";
  }

  // Check that (non-)scalability is preserved
  int64_t sourceNScalableDims = sourceType.getNumScalableDims();
  int64_t resultNScalableDims = resultType.getNumScalableDims();
  if (sourceNScalableDims != resultNScalableDims)
    return emitOpError() << "has different number of scalable dims at source ("
                         << sourceNScalableDims << ") and result ("
                         << resultNScalableDims << ")";

  return success();
}

/// Return true if `transpose` does not permute a pair of non-unit dims.
/// By `order preserving` we mean that the flattened versions of the input and
/// output vectors are (numerically) identical. In other words `transpose` is
/// effectively a shape cast.
static bool isOrderPreserving(TransposeOp transpose) {
  ArrayRef<int64_t> permutation = transpose.getPermutation();
  VectorType sourceType = transpose.getSourceVectorType();
  ArrayRef<int64_t> inShape = sourceType.getShape();
  ArrayRef<bool> inDimIsScalable = sourceType.getScalableDims();
  auto isNonScalableUnitDim = [&](int64_t dim) {
    return inShape[dim] == 1 && !inDimIsScalable[dim];
  };
  int64_t current = 0;
  for (auto p : permutation) {
    if (!isNonScalableUnitDim(p)) {
      if (p < current) {
        return false;
      }
      current = p;
    }
  }
  return true;
}

OpFoldResult ShapeCastOp::fold(FoldAdaptor adaptor) {

  VectorType resultType = getType();

  // No-op shape cast.
  if (getSource().getType() == resultType)
    return getSource();

  // shape_cast(shape_cast(x)) -> shape_cast(x)
  if (auto precedingShapeCast = getSource().getDefiningOp<ShapeCastOp>()) {
    setOperand(precedingShapeCast.getSource());
    return getResult();
  }

  // shape_cast(transpose(x)) -> shape_cast(x)
  if (auto transpose = getSource().getDefiningOp<TransposeOp>()) {
    if (isOrderPreserving(transpose)) {
      setOperand(transpose.getVector());
      return getResult();
    }
    return {};
  }

  // Y = shape_cast(broadcast(X))
  //      -> X, if X and Y have same type
  if (auto bcastOp = getSource().getDefiningOp<BroadcastOp>()) {
    if (bcastOp.getSourceType() == resultType)
      return bcastOp.getSource();
  }

  // shape_cast(constant) -> constant
  if (auto denseAttr =
          dyn_cast_if_present<DenseElementsAttr>(adaptor.getSource()))
    return denseAttr.reshape(getType());

  // shape_cast(poison) -> poison
  if (llvm::dyn_cast_if_present<ub::PoisonAttr>(adaptor.getSource()))
    return ub::PoisonAttr::get(getContext());

  return {};
}

namespace {

/// Helper function that computes a new vector type based on the input vector
/// type by removing the trailing one dims:
///
///   vector<4x1x1xi1> --> vector<4x1xi1>
///
static VectorType trimTrailingOneDims(VectorType oldType) {
  ArrayRef<int64_t> oldShape = oldType.getShape();
  ArrayRef<int64_t> newShape = oldShape;

  ArrayRef<bool> oldScalableDims = oldType.getScalableDims();
  ArrayRef<bool> newScalableDims = oldScalableDims;

  while (!newShape.empty() && newShape.back() == 1 && !newScalableDims.back()) {
    newShape = newShape.drop_back(1);
    newScalableDims = newScalableDims.drop_back(1);
  }

  // Make sure we have at least 1 dimension.
  // TODO: Add support for 0-D vectors.
  if (newShape.empty()) {
    newShape = oldShape.take_back();
    newScalableDims = oldScalableDims.take_back();
  }

  return VectorType::get(newShape, oldType.getElementType(), newScalableDims);
}

/// Folds qualifying shape_cast(create_mask) into a new create_mask
///
/// Looks at `vector.shape_cast` Ops that simply "drop" the trailing unit
/// dimension. If the input vector comes from `vector.create_mask` for which
/// the corresponding mask input value is 1 (e.g. `%c1` below), then it is safe
/// to fold shape_cast into create_mask.
///
/// BEFORE:
///    %1 = vector.create_mask %c1, %dim, %c1, %c1 : vector<1x[4]x1x1xi1>
///    %2 = vector.shape_cast %1 : vector<1x[4]x1x1xi1> to vector<1x[4]xi1>
/// AFTER:
///    %0 = vector.create_mask %c1, %dim : vector<1x[4]xi1>
class ShapeCastCreateMaskFolderTrailingOneDim final
    : public OpRewritePattern<ShapeCastOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ShapeCastOp shapeOp,
                                PatternRewriter &rewriter) const override {
    Value shapeOpSrc = shapeOp->getOperand(0);
    auto createMaskOp = shapeOpSrc.getDefiningOp<vector::CreateMaskOp>();
    auto constantMaskOp = shapeOpSrc.getDefiningOp<vector::ConstantMaskOp>();
    if (!createMaskOp && !constantMaskOp)
      return failure();

    VectorType shapeOpResTy = shapeOp.getResultVectorType();
    VectorType shapeOpSrcTy = shapeOp.getSourceVectorType();

    VectorType newVecType = trimTrailingOneDims(shapeOpSrcTy);
    if (newVecType != shapeOpResTy)
      return failure();

    auto numDimsToDrop =
        shapeOpSrcTy.getShape().size() - shapeOpResTy.getShape().size();

    // No unit dims to drop
    if (!numDimsToDrop)
      return failure();

    if (createMaskOp) {
      auto maskOperands = createMaskOp.getOperands();
      auto numMaskOperands = maskOperands.size();

      // Check every mask dim size to see whether it can be dropped
      for (size_t i = numMaskOperands - 1; i >= numMaskOperands - numDimsToDrop;
           --i) {
        auto constant = maskOperands[i].getDefiningOp<arith::ConstantIndexOp>();
        if (!constant || (constant.value() != 1))
          return failure();
      }
      SmallVector<Value> newMaskOperands =
          maskOperands.drop_back(numDimsToDrop);

      rewriter.replaceOpWithNewOp<vector::CreateMaskOp>(shapeOp, shapeOpResTy,
                                                        newMaskOperands);
      return success();
    }

    if (constantMaskOp) {
      auto maskDimSizes = constantMaskOp.getMaskDimSizes();
      auto numMaskOperands = maskDimSizes.size();

      // Check every mask dim size to see whether it can be dropped
      for (size_t i = numMaskOperands - 1; i >= numMaskOperands - numDimsToDrop;
           --i) {
        if (maskDimSizes[i] != 1)
          return failure();
      }

      auto newMaskOperands = maskDimSizes.drop_back(numDimsToDrop);
      rewriter.replaceOpWithNewOp<vector::ConstantMaskOp>(shapeOp, shapeOpResTy,
                                                          newMaskOperands);
      return success();
    }

    return failure();
  }
};

/// Pattern to rewrite Y = ShapeCast(Broadcast(X)) as either
///   i) Y = ShapeCast(X), or
///  ii) Y = Broadcast(X)
/// If both (i) and (ii) are possible, (i) is chosen.
class ShapeCastBroadcastFolder final : public OpRewritePattern<ShapeCastOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ShapeCastOp shapeCastOp,
                                PatternRewriter &rewriter) const override {
    auto broadcastOp =
        shapeCastOp.getSource().getDefiningOp<vector::BroadcastOp>();
    if (!broadcastOp)
      return failure();

    auto srcVectorType = dyn_cast<VectorType>(broadcastOp.getSourceType());
    bool srcIsScalar = !srcVectorType;

    // Replace Y = ShapeCast(Broadcast(X)) with Y = ShapeCast(X).
    // Example:
    // %0 = vector.broadcast %in : vector<3x4xf32> to vector<1x3x4xf32>
    // %1 = vector.shape_cast %0 : vector<1x3x4xf32> to vector<12xf32>
    // to
    // %1 = vector.shape_cast %in : vector<3x4xf32> to vector<12xf32>
    if (srcVectorType) {
      if (srcVectorType.getNumElements() ==
          shapeCastOp.getResultVectorType().getNumElements()) {
        rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(
            shapeCastOp, shapeCastOp.getResultVectorType(),
            broadcastOp.getSource());
        return success();
      }
    }

    // Replace Y = ShapeCast(Broadcast(X)) with Y = Broadcast(X)
    // Example
    // %0 = vector.broadcast %in : vector<3xf32> to vector<2x4x3xf32>
    // %1 = vector.shape_cast %0 : vector<2x4x3xf32> to vector<8x3xf32>
    // to
    // %1 = vector.broadcast %in : vector<3xf32> to vector<8x3xf32>
    VectorType dstVectorType = shapeCastOp.getResultVectorType();
    if (srcIsScalar || isBroadcastableTo(srcVectorType, dstVectorType) ==
                           BroadcastableToResult::Success) {
      rewriter.replaceOpWithNewOp<vector::BroadcastOp>(
          shapeCastOp, dstVectorType, broadcastOp.getSource());
      return success();
    }
    return failure();
  }
};

} // namespace

void ShapeCastOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results
      .add<ShapeCastCreateMaskFolderTrailingOneDim, ShapeCastBroadcastFolder>(
          context);
}

//===----------------------------------------------------------------------===//
// VectorBitCastOp
//===----------------------------------------------------------------------===//

LogicalResult BitCastOp::verify() {
  auto sourceVectorType = getSourceVectorType();
  auto resultVectorType = getResultVectorType();

  for (int64_t i = 0, e = sourceVectorType.getRank() - 1; i < e; i++) {
    if (sourceVectorType.getDimSize(i) != resultVectorType.getDimSize(i))
      return emitOpError("dimension size mismatch at: ") << i;
  }

  DataLayout dataLayout = DataLayout::closest(*this);
  auto sourceElementBits =
      dataLayout.getTypeSizeInBits(sourceVectorType.getElementType());
  auto resultElementBits =
      dataLayout.getTypeSizeInBits(resultVectorType.getElementType());

  if (sourceVectorType.getRank() == 0) {
    if (sourceElementBits != resultElementBits)
      return emitOpError("source/result bitwidth of the 0-D vector element "
                         "types must be equal");
  } else if (sourceElementBits * sourceVectorType.getShape().back() !=
             resultElementBits * resultVectorType.getShape().back()) {
    return emitOpError(
        "source/result bitwidth of the minor 1-D vectors must be equal");
  }

  return success();
}

OpFoldResult BitCastOp::fold(FoldAdaptor adaptor) {
  // Nop cast.
  if (getSource().getType() == getResult().getType())
    return getSource();

  // Canceling bitcasts.
  if (auto otherOp = getSource().getDefiningOp<BitCastOp>()) {
    if (getResult().getType() == otherOp.getSource().getType())
      return otherOp.getSource();

    setOperand(otherOp.getSource());
    return getResult();
  }

  Attribute sourceConstant = adaptor.getSource();
  if (!sourceConstant)
    return {};

  Type srcElemType = getSourceVectorType().getElementType();
  Type dstElemType = getResultVectorType().getElementType();

  if (auto floatPack = llvm::dyn_cast<DenseFPElementsAttr>(sourceConstant)) {
    if (floatPack.isSplat()) {
      auto splat = floatPack.getSplatValue<FloatAttr>();

      // Casting fp16 into fp32.
      if (srcElemType.isF16() && dstElemType.isF32()) {
        uint32_t bits = static_cast<uint32_t>(
            splat.getValue().bitcastToAPInt().getZExtValue());
        // Duplicate the 16-bit pattern.
        bits = (bits << 16) | (bits & 0xffff);
        APInt intBits(32, bits);
        APFloat floatBits(llvm::APFloat::IEEEsingle(), intBits);
        return DenseElementsAttr::get(getResultVectorType(), floatBits);
      }
    }
  }

  if (auto intPack = llvm::dyn_cast<DenseIntElementsAttr>(sourceConstant)) {
    if (intPack.isSplat()) {
      auto splat = intPack.getSplatValue<IntegerAttr>();

      if (llvm::isa<IntegerType>(dstElemType)) {
        uint64_t srcBitWidth = srcElemType.getIntOrFloatBitWidth();
        uint64_t dstBitWidth = dstElemType.getIntOrFloatBitWidth();

        // Casting to a larger integer bit width.
        if (dstBitWidth > srcBitWidth && dstBitWidth % srcBitWidth == 0) {
          APInt intBits = splat.getValue().zext(dstBitWidth);

          // Duplicate the lower width element.
          for (uint64_t i = 0; i < dstBitWidth / srcBitWidth - 1; i++)
            intBits = (intBits << srcBitWidth) | intBits;
          return DenseElementsAttr::get(getResultVectorType(), intBits);
        }
      }
    }
  }

  return {};
}

//===----------------------------------------------------------------------===//
// TypeCastOp
//===----------------------------------------------------------------------===//

static SmallVector<int64_t, 8> extractShape(MemRefType memRefType) {
  auto vectorType = llvm::dyn_cast<VectorType>(memRefType.getElementType());
  SmallVector<int64_t, 8> res(memRefType.getShape());
  if (vectorType)
    res.append(vectorType.getShape().begin(), vectorType.getShape().end());
  return res;
}

/// Build the canonical memRefType with a single vector.
/// E.g. memref<4 x 5 x vector<6 x f32>> -> memref<vector<4 x 5 x 6 x f32>>.
void TypeCastOp::build(OpBuilder &builder, OperationState &result,
                       Value source) {
  result.addOperands(source);
  MemRefType memRefType = llvm::cast<MemRefType>(source.getType());
  VectorType vectorType =
      VectorType::get(extractShape(memRefType),
                      getElementTypeOrSelf(getElementTypeOrSelf(memRefType)));
  result.addTypes(MemRefType::get({}, vectorType, MemRefLayoutAttrInterface(),
                                  memRefType.getMemorySpace()));
}

LogicalResult TypeCastOp::verify() {
  MemRefType canonicalType = getMemRefType().canonicalizeStridedLayout();
  if (!canonicalType.getLayout().isIdentity())
    return emitOpError("expects operand to be a memref with identity layout");
  if (!getResultMemRefType().getLayout().isIdentity())
    return emitOpError("expects result to be a memref with identity layout");
  if (getResultMemRefType().getMemorySpace() !=
      getMemRefType().getMemorySpace())
    return emitOpError("expects result in same memory space");

  auto sourceType = getMemRefType();
  auto resultType = getResultMemRefType();
  if (getElementTypeOrSelf(getElementTypeOrSelf(sourceType)) !=
      getElementTypeOrSelf(getElementTypeOrSelf(resultType)))
    return emitOpError(
               "expects result and operand with same underlying scalar type: ")
           << resultType;
  if (extractShape(sourceType) != extractShape(resultType))
    return emitOpError(
               "expects concatenated result and operand shapes to be equal: ")
           << resultType;
  return success();
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

void vector::TransposeOp::build(OpBuilder &builder, OperationState &result,
                                Value vector, ArrayRef<int64_t> permutation) {
  VectorType vt = llvm::cast<VectorType>(vector.getType());
  SmallVector<int64_t, 4> transposedShape(vt.getRank());
  SmallVector<bool, 4> transposedScalableDims(vt.getRank());
  for (unsigned i = 0; i < permutation.size(); ++i) {
    transposedShape[i] = vt.getShape()[permutation[i]];
    transposedScalableDims[i] = vt.getScalableDims()[permutation[i]];
  }

  result.addOperands(vector);
  result.addTypes(VectorType::get(transposedShape, vt.getElementType(),
                                  transposedScalableDims));
  result.addAttribute(TransposeOp::getPermutationAttrName(result.name),
                      builder.getDenseI64ArrayAttr(permutation));
}

OpFoldResult vector::TransposeOp::fold(FoldAdaptor adaptor) {
  // Eliminate splat constant transpose ops.
  if (auto splat =
          llvm::dyn_cast_if_present<SplatElementsAttr>(adaptor.getVector()))
    return splat.reshape(getResultVectorType());

  // Eliminate poison transpose ops.
  if (llvm::dyn_cast_if_present<ub::PoisonAttr>(adaptor.getVector()))
    return ub::PoisonAttr::get(getContext());

  // Eliminate identity transposes, and more generally any transposes that
  // preserves the shape without permuting elements.
  //
  // Examples of what to fold:
  // %0 = vector.transpose %arg, [0, 1] : vector<1x1xi8> to vector<1x1xi8>
  // %0 = vector.transpose %arg, [0, 1] : vector<2x2xi8> to vector<2x2xi8>
  // %0 = vector.transpose %arg, [1, 0] : vector<1x1xi8> to vector<1x1xi8>
  //
  // Example of what NOT to fold:
  // %0 = vector.transpose %arg, [1, 0] : vector<2x2xi8> to vector<2x2xi8>
  //
  if (getSourceVectorType() == getResultVectorType() &&
      isOrderPreserving(*this))
    return getVector();

  return {};
}

LogicalResult vector::TransposeOp::verify() {
  VectorType vectorType = getSourceVectorType();
  VectorType resultType = getResultVectorType();
  int64_t rank = resultType.getRank();
  if (vectorType.getRank() != rank)
    return emitOpError("vector result rank mismatch: ") << rank;
  // Verify transposition array.
  ArrayRef<int64_t> perm = getPermutation();
  int64_t size = perm.size();
  if (rank != size)
    return emitOpError("transposition length mismatch: ") << size;
  SmallVector<bool, 8> seen(rank, false);
  for (const auto &ta : llvm::enumerate(perm)) {
    if (ta.value() < 0 || ta.value() >= rank)
      return emitOpError("transposition index out of range: ") << ta.value();
    if (seen[ta.value()])
      return emitOpError("duplicate position index: ") << ta.value();
    seen[ta.value()] = true;
    if (resultType.getDimSize(ta.index()) != vectorType.getDimSize(ta.value()))
      return emitOpError("dimension size mismatch at: ") << ta.value();
  }
  return success();
}

std::optional<SmallVector<int64_t, 4>> TransposeOp::getShapeForUnroll() {
  return llvm::to_vector<4>(getResultVectorType().getShape());
}

void TransposeOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                    SetIntRangeFn setResultRanges) {
  setResultRanges(getResult(), argRanges.front());
}

namespace {

// Rewrites two back-to-back TransposeOp operations into a single TransposeOp.
class TransposeFolder final : public OpRewritePattern<vector::TransposeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {
    // Composes two permutations: result[i] = permutation1[permutation2[i]].
    auto composePermutations = [](ArrayRef<int64_t> permutation1,
                                  ArrayRef<int64_t> permutation2) {
      SmallVector<int64_t, 4> result;
      for (auto index : permutation2)
        result.push_back(permutation1[index]);
      return result;
    };

    // Return if the input of 'transposeOp' is not defined by another transpose.
    vector::TransposeOp parentTransposeOp =
        transposeOp.getVector().getDefiningOp<vector::TransposeOp>();
    if (!parentTransposeOp)
      return failure();

    SmallVector<int64_t, 4> permutation = composePermutations(
        parentTransposeOp.getPermutation(), transposeOp.getPermutation());
    // Replace 'transposeOp' with a new transpose operation.
    rewriter.replaceOpWithNewOp<vector::TransposeOp>(
        transposeOp, transposeOp.getResult().getType(),
        parentTransposeOp.getVector(), permutation);
    return success();
  }
};

/// Replace transpose(splat-like(v)) with broadcast(v)
class FoldTransposeSplat final : public OpRewritePattern<TransposeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {
    Value splat = getScalarSplatSource(transposeOp.getVector());
    if (!splat)
      return failure();

    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(
        transposeOp, transposeOp.getResultVectorType(), splat);
    return success();
  }
};

/// Folds transpose(create_mask) into a new transposed create_mask.
class FoldTransposeCreateMask final : public OpRewritePattern<TransposeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TransposeOp transpOp,
                                PatternRewriter &rewriter) const override {
    Value transposeSrc = transpOp.getVector();
    auto createMaskOp = transposeSrc.getDefiningOp<vector::CreateMaskOp>();
    auto constantMaskOp = transposeSrc.getDefiningOp<vector::ConstantMaskOp>();
    if (!createMaskOp && !constantMaskOp)
      return failure();

    // Get the transpose permutation and apply it to the vector.create_mask or
    // vector.constant_mask operands.
    ArrayRef<int64_t> permutation = transpOp.getPermutation();

    if (createMaskOp) {
      auto maskOperands = createMaskOp.getOperands();
      SmallVector<Value> newOperands(maskOperands.begin(), maskOperands.end());
      applyPermutationToVector(newOperands, permutation);

      rewriter.replaceOpWithNewOp<vector::CreateMaskOp>(
          transpOp, transpOp.getResultVectorType(), newOperands);
      return success();
    }

    // ConstantMaskOp case.
    auto maskDimSizes = constantMaskOp.getMaskDimSizes();
    auto newMaskDimSizes = applyPermutation(maskDimSizes, permutation);

    rewriter.replaceOpWithNewOp<vector::ConstantMaskOp>(
        transpOp, transpOp.getResultVectorType(), newMaskDimSizes);
    return success();
  }
};

/// Folds transpose(shape_cast) into a new shape_cast.
class FoldTransposeShapeCast final : public OpRewritePattern<TransposeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {
    auto shapeCastOp =
        transposeOp.getVector().getDefiningOp<vector::ShapeCastOp>();
    if (!shapeCastOp)
      return failure();
    if (!isOrderPreserving(transposeOp))
      return failure();

    VectorType resultType = transposeOp.getType();

    // We don't need to check isValidShapeCast at this point, because it is
    // guaranteed that merging the transpose into the the shape_cast is a valid
    // shape_cast, because the transpose just inserts/removes ones.

    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(transposeOp, resultType,
                                                     shapeCastOp.getSource());
    return success();
  }
};

/// Folds transpose(broadcast(x)) to broadcast(x) if the transpose is
/// 'order preserving', where 'order preserving' means the flattened
/// inputs and outputs of the transpose have identical (numerical) values.
///
/// Example:
/// ```
///  %0 = vector.broadcast %input : vector<1x1xi32> to vector<1x8xi32>
///  %1 = vector.transpose %0, [1, 0] : vector<1x8xi32>
///                                                 to vector<8x1xi32>
/// ```
/// can be rewritten as the equivalent
/// ```
///  %0 = vector.broadcast %input : vector<1x1xi32> to vector<8x1xi32>.
/// ```
/// The algorithm works by partitioning dimensions into groups that can be
/// locally permuted while preserving order, and checks that the transpose
/// only permutes within these groups.
///
/// Groups are either contiguous sequences of 1s, or non-1s (1-element groups).
/// Consider broadcasting 4x1x1x7 to 2x3x4x5x6x7. This is equivalent to
/// broadcasting from 1x1x4x1x1x7.
///                   ^^^ ^ ^^^ ^
///          groups:   0  1  2  3
/// Order preserving permutations for this example are ones that only permute
/// within the groups [0,1] and [3,4], like (1 0 2 4 3 5 6).
class FoldTransposeBroadcast : public OpRewritePattern<vector::TransposeOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  FoldTransposeBroadcast(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern<vector::TransposeOp>(context, benefit) {}

  LogicalResult matchAndRewrite(vector::TransposeOp transpose,
                                PatternRewriter &rewriter) const override {

    vector::BroadcastOp broadcast =
        transpose.getVector().getDefiningOp<vector::BroadcastOp>();
    if (!broadcast) {
      return rewriter.notifyMatchFailure(transpose,
                                         "not preceded by a broadcast");
    }

    auto inputType = dyn_cast<VectorType>(broadcast.getSourceType());
    VectorType outputType = transpose.getResultVectorType();

    // transpose(broadcast(scalar)) -> broadcast(scalar) is always valid
    bool inputIsScalar = !inputType;
    if (inputIsScalar) {
      rewriter.replaceOpWithNewOp<vector::BroadcastOp>(transpose, outputType,
                                                       broadcast.getSource());
      return success();
    }

    ArrayRef<int64_t> permutation = transpose.getPermutation();
    ArrayRef<int64_t> inputShape = inputType.getShape();
    int64_t inputRank = inputType.getRank();
    int64_t outputRank = transpose.getType().getRank();
    int64_t deltaRank = outputRank - inputRank;

    int low = 0;
    for (int inputIndex = 0; inputIndex < inputRank; ++inputIndex) {
      bool notOne = inputShape[inputIndex] != 1;
      bool prevNotOne = (inputIndex != 0 && inputShape[inputIndex - 1] != 1);
      bool groupEndFound = notOne || prevNotOne;
      if (groupEndFound) {
        int high = inputIndex + deltaRank;
        // Return failure if not all permutation destinations for indices in
        // [low, high) are in [low, high), i.e. the permutation is not local to
        // the group.
        for (int i = low; i < high; ++i) {
          if (permutation[i] < low || permutation[i] >= high) {
            return rewriter.notifyMatchFailure(
                transpose, "permutation not local to group");
          }
        }
        low = high;
      }
    }

    // We don't need to check the final group [low, outputRank) because if it is
    // not locally bound, there must be a preceding group that already failed
    // the check (impossible to have just 1 non-locally bound group).

    // The preceding logic also ensures that at this point, the output of the
    // transpose is definitely broadcastable from the input shape, assert so:
    assert(vector::isBroadcastableTo(inputType, outputType) ==
               vector::BroadcastableToResult::Success &&
           "not broadcastable directly to transpose output");

    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(transpose, outputType,
                                                     broadcast.getSource());

    return success();
  }
};

} // namespace

void vector::TransposeOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<FoldTransposeCreateMask, FoldTransposeShapeCast, TransposeFolder,
              FoldTransposeSplat, FoldTransposeBroadcast>(context);
}

//===----------------------------------------------------------------------===//
// ConstantMaskOp
//===----------------------------------------------------------------------===//

void ConstantMaskOp::build(OpBuilder &builder, OperationState &result,
                           VectorType type, ConstantMaskKind kind) {
  assert(kind == ConstantMaskKind::AllTrue ||
         kind == ConstantMaskKind::AllFalse);
  build(builder, result, type,
        kind == ConstantMaskKind::AllTrue
            ? type.getShape()
            : SmallVector<int64_t>(type.getRank(), 0));
}

LogicalResult ConstantMaskOp::verify() {
  auto resultType = llvm::cast<VectorType>(getResult().getType());
  // Check the corner case of 0-D vectors first.
  if (resultType.getRank() == 0) {
    if (getMaskDimSizes().size() != 1)
      return emitError("array attr must have length 1 for 0-D vectors");
    auto dim = getMaskDimSizes()[0];
    if (dim != 0 && dim != 1)
      return emitError("mask dim size must be either 0 or 1 for 0-D vectors");
    return success();
  }

  // Verify that array attr size matches the rank of the vector result.
  if (static_cast<int64_t>(getMaskDimSizes().size()) != resultType.getRank())
    return emitOpError(
        "must specify array attr of size equal vector result rank");
  // Verify that each array attr element is in bounds of corresponding vector
  // result dimension size.
  auto resultShape = resultType.getShape();
  auto resultScalableDims = resultType.getScalableDims();
  ArrayRef<int64_t> maskDimSizes = getMaskDimSizes();
  for (const auto [index, maskDimSize] : llvm::enumerate(maskDimSizes)) {
    if (maskDimSize < 0 || maskDimSize > resultShape[index])
      return emitOpError(
          "array attr of size out of bounds of vector result dimension size");
    if (resultScalableDims[index] && maskDimSize != 0 &&
        maskDimSize != resultShape[index])
      return emitOpError(
          "only supports 'none set' or 'all set' scalable dimensions");
  }
  // Verify that if one mask dim size is zero, they all should be zero (because
  // the mask region is a conjunction of each mask dimension interval).
  bool anyZeros = llvm::is_contained(maskDimSizes, 0);
  bool allZeros = llvm::all_of(maskDimSizes, [](int64_t s) { return s == 0; });
  if (anyZeros && !allZeros)
    return emitOpError("expected all mask dim sizes to be zeros, "
                       "as a result of conjunction with zero mask dim");
  return success();
}

bool ConstantMaskOp::isAllOnesMask() {
  auto resultType = getVectorType();
  // Check the corner case of 0-D vectors first.
  if (resultType.getRank() == 0) {
    assert(getMaskDimSizes().size() == 1 && "invalid sizes for zero rank mask");
    return getMaskDimSizes()[0] == 1;
  }
  for (const auto [resultSize, maskDimSize] :
       llvm::zip_equal(resultType.getShape(), getMaskDimSizes())) {
    if (maskDimSize < resultSize)
      return false;
  }
  return true;
}

OpFoldResult ConstantMaskOp::fold(FoldAdaptor adaptor) {
  ArrayRef<int64_t> bounds = getMaskDimSizes();
  ArrayRef<int64_t> vectorSizes = getVectorType().getShape();

  auto createBoolSplat = [&](bool x) {
    return SplatElementsAttr::get(getVectorType(),
                                  BoolAttr::get(getContext(), x));
  };

  // Check the corner case of 0-D vectors first.
  if (vectorSizes.empty()) {
    assert(bounds.size() == 1 && "invalid sizes for zero rank mask");
    return createBoolSplat(bounds[0] == 1);
  }
  // Fold vector.constant_mask to splat if possible.
  if (bounds == vectorSizes)
    return createBoolSplat(true);
  if (llvm::all_of(bounds, [](int64_t x) { return x == 0; }))
    return createBoolSplat(false);
  return OpFoldResult();
}

//===----------------------------------------------------------------------===//
// CreateMaskOp
//===----------------------------------------------------------------------===//

void CreateMaskOp::build(OpBuilder &builder, OperationState &result,
                         VectorType type,
                         ArrayRef<OpFoldResult> mixedOperands) {
  SmallVector<Value> operands =
      getValueOrCreateConstantIndexOp(builder, result.location, mixedOperands);
  build(builder, result, type, operands);
}

LogicalResult CreateMaskOp::verify() {
  auto vectorType = llvm::cast<VectorType>(getResult().getType());
  // Verify that an operand was specified for each result vector each dimension.
  if (vectorType.getRank() == 0) {
    if (getNumOperands() != 1)
      return emitOpError(
          "must specify exactly one operand for 0-D create_mask");
  } else if (getNumOperands() !=
             llvm::cast<VectorType>(getResult().getType()).getRank()) {
    return emitOpError(
        "must specify an operand for each result vector dimension");
  }
  return success();
}

namespace {

/// Pattern to rewrite a CreateMaskOp with a ConstantMaskOp.
///
/// Ex 1:
///   %c2 = arith.constant 2 : index
///   %c3 = arith.constant 3 : index
///   %0 = vector.create_mask %c3, %c2 : vector<4x3xi1>
/// Becomes:
///    vector.constant_mask [3, 2] : vector<4x3xi1>
///
/// Ex 2:
///   %c_neg_1 = arith.constant -1 : index
///   %0 = vector.create_mask %c_neg_1 : vector<[8]xi1>
/// becomes:
///   vector.constant_mask [0] : vector<[8]xi1>
///
/// Ex 3:
///   %c8 = arith.constant 8 : index
///   %c16 = arith.constant 16 : index
///   %0 = vector.vscale
///   %1 = arith.muli %0, %c16 : index
///   %10 = vector.create_mask %c8, %1 : vector<8x[16]xi1>
/// becomes:
///   %0 = vector.constant_mask [8, 16] : vector<8x[16]xi1>
class CreateMaskFolder final : public OpRewritePattern<CreateMaskOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CreateMaskOp createMaskOp,
                                PatternRewriter &rewriter) const override {
    VectorType maskType = createMaskOp.getVectorType();
    ArrayRef<int64_t> maskTypeDimSizes = maskType.getShape();
    ArrayRef<bool> maskTypeDimScalableFlags = maskType.getScalableDims();

    // Special case: Rank zero shape.
    constexpr std::array<int64_t, 1> rankZeroShape{1};
    constexpr std::array<bool, 1> rankZeroScalableDims{false};
    if (maskType.getRank() == 0) {
      maskTypeDimSizes = rankZeroShape;
      maskTypeDimScalableFlags = rankZeroScalableDims;
    }

    // Determine if this CreateMaskOp can be folded to a ConstantMaskOp and
    // collect the `constantDims` (for the ConstantMaskOp).
    SmallVector<int64_t, 4> constantDims;
    for (auto [i, dimSize] : llvm::enumerate(createMaskOp.getOperands())) {
      if (auto intSize = getConstantIntValue(dimSize)) {
        // Constant value.
        // If the mask dim is non-scalable this can be any value.
        // If the mask dim is scalable only zero (all-false) is supported.
        if (maskTypeDimScalableFlags[i] && intSize >= 0)
          return failure();
        constantDims.push_back(*intSize);
      } else if (auto vscaleMultiplier = getConstantVscaleMultiplier(dimSize)) {
        // Constant vscale multiple (e.g. 4 x vscale).
        // Must be all-true to fold to a ConstantMask.
        if (vscaleMultiplier < maskTypeDimSizes[i])
          return failure();
        constantDims.push_back(*vscaleMultiplier);
      } else {
        return failure();
      }
    }

    // Clamp values to constant_mask bounds.
    for (auto [value, maskDimSize] : llvm::zip(constantDims, maskTypeDimSizes))
      value = std::clamp<int64_t>(value, 0, maskDimSize);

    // If one of dim sizes is zero, set all dims to zero.
    if (llvm::is_contained(constantDims, 0))
      constantDims.assign(constantDims.size(), 0);

    // Replace 'createMaskOp' with ConstantMaskOp.
    rewriter.replaceOpWithNewOp<ConstantMaskOp>(createMaskOp, maskType,
                                                constantDims);
    return success();
  }
};

} // namespace

void CreateMaskOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.add<CreateMaskFolder>(context);
}

//===----------------------------------------------------------------------===//
// MaskOp
//===----------------------------------------------------------------------===//

void MaskOp::build(
    OpBuilder &builder, OperationState &result, Value mask,
    Operation *maskableOp,
    function_ref<void(OpBuilder &, Operation *)> maskRegionBuilder) {
  assert(maskRegionBuilder &&
         "builder callback for 'maskRegion' must be present");

  result.addOperands(mask);
  OpBuilder::InsertionGuard guard(builder);
  Region *maskRegion = result.addRegion();
  builder.createBlock(maskRegion);
  maskRegionBuilder(builder, maskableOp);
}

void MaskOp::build(
    OpBuilder &builder, OperationState &result, TypeRange resultTypes,
    Value mask, Operation *maskableOp,
    function_ref<void(OpBuilder &, Operation *)> maskRegionBuilder) {
  build(builder, result, resultTypes, mask, /*passthru=*/Value(), maskableOp,
        maskRegionBuilder);
}

void MaskOp::build(
    OpBuilder &builder, OperationState &result, TypeRange resultTypes,
    Value mask, Value passthru, Operation *maskableOp,
    function_ref<void(OpBuilder &, Operation *)> maskRegionBuilder) {
  build(builder, result, mask, maskableOp, maskRegionBuilder);
  if (passthru)
    result.addOperands(passthru);
  result.addTypes(resultTypes);
}

ParseResult MaskOp::parse(OpAsmParser &parser, OperationState &result) {
  // Create the op region.
  result.regions.reserve(1);
  Region &maskRegion = *result.addRegion();

  auto &builder = parser.getBuilder();

  // Parse all the operands.
  OpAsmParser::UnresolvedOperand mask;
  if (parser.parseOperand(mask))
    return failure();

  // Optional passthru operand.
  OpAsmParser::UnresolvedOperand passthru;
  ParseResult parsePassthru = parser.parseOptionalComma();
  if (parsePassthru.succeeded() && parser.parseOperand(passthru))
    return failure();

  // Parse op region.
  if (parser.parseRegion(maskRegion, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();

  MaskOp::ensureTerminator(maskRegion, builder, result.location);

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // Parse all the types.
  Type maskType;
  if (parser.parseColonType(maskType))
    return failure();

  SmallVector<Type> resultTypes;
  if (parser.parseOptionalArrowTypeList(resultTypes))
    return failure();
  result.types.append(resultTypes);

  // Resolve operands.
  if (parser.resolveOperand(mask, maskType, result.operands))
    return failure();

  if (parsePassthru.succeeded()) {
    if (resultTypes.empty())
      return parser.emitError(
          parser.getNameLoc(),
          "expects a result if passthru operand is provided");

    if (parser.resolveOperand(passthru, resultTypes[0], result.operands))
      return failure();
  }

  return success();
}

void mlir::vector::MaskOp::print(OpAsmPrinter &p) {
  p << " " << getMask();
  if (getPassthru())
    p << ", " << getPassthru();

  // Print single masked operation and skip terminator.
  p << " { ";
  Block *singleBlock = &getMaskRegion().getBlocks().front();
  if (singleBlock && !singleBlock->getOperations().empty())
    p.printCustomOrGenericOp(&singleBlock->front());
  p << " }";

  p.printOptionalAttrDict(getOperation()->getAttrs());

  p << " : " << getMask().getType();
  if (getNumResults() > 0)
    p << " -> " << getResultTypes();
}

void MaskOp::ensureTerminator(Region &region, Builder &builder, Location loc) {
  // 1. For an empty `vector.mask`, create a default terminator.
  if (region.empty() || region.front().empty()) {
    OpTrait::SingleBlockImplicitTerminator<vector::YieldOp>::Impl<
        MaskOp>::ensureTerminator(region, builder, loc);
    return;
  }

  // 2. For a non-empty `vector.mask` with an explicit terminator, do nothing.
  Block &block = region.front();
  if (isa<vector::YieldOp>(block.back()))
    return;

  // 3. For a non-empty `vector.mask` without an explicit terminator:

  // Create default terminator if the number of masked operations is not
  // one. This case will trigger a verification failure.
  if (block.getOperations().size() != 1) {
    OpTrait::SingleBlockImplicitTerminator<vector::YieldOp>::Impl<
        MaskOp>::ensureTerminator(region, builder, loc);
    return;
  }

  // Create a terminator that yields the results from the masked operation.
  OpBuilder opBuilder(builder.getContext());
  Operation *maskedOp = &block.front();
  opBuilder.setInsertionPointToEnd(&block);
  vector::YieldOp::create(opBuilder, loc, maskedOp->getResults());
}

LogicalResult MaskOp::verify() {
  // Structural checks.
  Block &block = getMaskRegion().getBlocks().front();
  if (block.getOperations().empty())
    return emitOpError("expects a terminator within the mask region");

  unsigned numMaskRegionOps = block.getOperations().size();
  if (numMaskRegionOps > 2)
    return emitOpError("expects only one operation to mask");

  // Terminator checks.
  auto terminator = dyn_cast<vector::YieldOp>(block.back());
  if (!terminator)
    return emitOpError("expects a terminator within the mask region");

  if (terminator->getNumOperands() != getNumResults())
    return emitOpError(
        "expects number of results to match mask region yielded values");

  // Empty vector.mask. Nothing else to check.
  if (numMaskRegionOps == 1)
    return success();

  auto maskableOp = dyn_cast<MaskableOpInterface>(block.front());
  if (!maskableOp)
    return emitOpError("expects a MaskableOpInterface within the mask region");

  // Result checks.
  if (maskableOp->getNumResults() != getNumResults())
    return emitOpError("expects number of results to match maskable operation "
                       "number of results");

  if (!llvm::equal(maskableOp->getResults(), terminator.getOperands()))
    return emitOpError("expects all the results from the MaskableOpInterface "
                       "to match all the values returned by the terminator");

  if (!llvm::equal(maskableOp->getResultTypes(), getResultTypes()))
    return emitOpError(
        "expects result type to match maskable operation result type");

  if (llvm::count_if(maskableOp->getResultTypes(),
                     [](Type t) { return llvm::isa<VectorType>(t); }) > 1)
    return emitOpError("multiple vector results not supported");

  // Mask checks.
  Type expectedMaskType = maskableOp.getExpectedMaskType();
  if (getMask().getType() != expectedMaskType)
    return emitOpError("expects a ")
           << expectedMaskType << " mask for the maskable operation";

  // Passthru checks.
  Value passthru = getPassthru();
  if (passthru) {
    if (!maskableOp.supportsPassthru())
      return emitOpError(
          "doesn't expect a passthru argument for this maskable operation");

    if (maskableOp->getNumResults() != 1)
      return emitOpError("expects result when passthru argument is provided");

    if (passthru.getType() != maskableOp->getResultTypes()[0])
      return emitOpError("expects passthru type to match result type");
  }

  return success();
}

/// Folds empty `vector.mask` with no passthru operand and with or without
/// return values. For example:
///
///   %0 = vector.mask %mask { vector.yield %a : vector<8xf32> } :
///     vector<8xi1> -> vector<8xf32>
///   %1 = user_op %0 : vector<8xf32>
///
/// becomes:
///
///   %0 = user_op %a : vector<8xf32>
///
/// Empty `vector.mask` with passthru operand are handled by the canonicalizer
/// as it requires creating new operations.

static LogicalResult foldEmptyMaskOp(MaskOp maskOp, MaskOp::FoldAdaptor adaptor,
                                     SmallVectorImpl<OpFoldResult> &results) {
  if (!maskOp.isEmpty() || maskOp.hasPassthru())
    return failure();

  Block *block = maskOp.getMaskBlock();
  auto terminator = cast<vector::YieldOp>(block->front());
  if (terminator.getNumOperands() == 0) {
    // `vector.mask` has no results, just remove the `vector.mask`.
    return success();
  }

  // `vector.mask` has results, propagate the results.
  llvm::append_range(results, terminator.getOperands());
  return success();
}

LogicalResult MaskOp::fold(FoldAdaptor adaptor,
                           SmallVectorImpl<OpFoldResult> &results) {
  if (succeeded(foldEmptyMaskOp(*this, adaptor, results)))
    return success();

  MaskFormat maskFormat = getMaskFormat(getMask());
  if (maskFormat != MaskFormat::AllTrue)
    return failure();

  // Move maskable operation outside of the `vector.mask` region.
  Operation *maskableOp = getMaskableOp();
  maskableOp->dropAllUses();
  maskableOp->moveBefore(getOperation());

  llvm::append_range(results, maskableOp->getResults());
  return success();
}

/// Canonialize empty `vector.mask` operations that can't be handled in
/// `VectorMask::fold` as they require creating new operations.
///
/// Example 1: Empty `vector.mask` with passthru operand.
///
///   %0 = vector.mask %mask, %passthru { vector.yield %a : vector<8xf32> } :
///     vector<8xi1> -> vector<8xf32>
///
/// becomes:
///
///   %0 = arith.select %mask, %a, %passthru : vector<8xf32>
///
class CanonializeEmptyMaskOp : public OpRewritePattern<MaskOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MaskOp maskOp,
                                PatternRewriter &rewriter) const override {
    if (!maskOp.isEmpty())
      return failure();

    if (!maskOp.hasPassthru())
      return failure();

    Block *block = maskOp.getMaskBlock();
    auto terminator = cast<vector::YieldOp>(block->front());
    assert(terminator.getNumOperands() == 1 &&
           "expected one result when passthru is provided");

    rewriter.replaceOpWithNewOp<arith::SelectOp>(
        maskOp, maskOp.getResultTypes(), maskOp.getMask(),
        terminator.getOperand(0), maskOp.getPassthru());

    return success();
  }
};

void MaskOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.add<CanonializeEmptyMaskOp>(context);
}

// MaskingOpInterface definitions.

/// Returns the operation masked by this 'vector.mask'.
Operation *MaskOp::getMaskableOp() {
  Block *block = getMaskBlock();
  if (block->getOperations().size() < 2)
    return nullptr;

  return &block->front();
}

/// Returns true if 'vector.mask' has a passthru value.
bool MaskOp::hasPassthru() { return getPassthru() != Value(); }

//===----------------------------------------------------------------------===//
// ScanOp
//===----------------------------------------------------------------------===//

LogicalResult ScanOp::verify() {
  VectorType srcType = getSourceType();
  VectorType initialType = getInitialValueType();
  // Check reduction dimension < rank.
  int64_t srcRank = srcType.getRank();
  int64_t reductionDim = getReductionDim();
  if (reductionDim >= srcRank)
    return emitOpError("reduction dimension ")
           << reductionDim << " has to be less than " << srcRank;

  // Check that rank(initial_value) = rank(src) - 1.
  int64_t initialValueRank = initialType.getRank();
  if (initialValueRank != srcRank - 1)
    return emitOpError("initial value rank ")
           << initialValueRank << " has to be equal to " << srcRank - 1;

  // Check shapes of initial value and src.
  ArrayRef<int64_t> srcShape = srcType.getShape();
  ArrayRef<int64_t> initialValueShapes = initialType.getShape();
  SmallVector<int64_t> expectedShape;
  for (int i = 0; i < srcRank; i++) {
    if (i != reductionDim)
      expectedShape.push_back(srcShape[i]);
  }
  if (!llvm::equal(initialValueShapes, expectedShape)) {
    return emitOpError("incompatible input/initial value shapes");
  }

  // Verify supported reduction kind.
  Type eltType = getDestType().getElementType();
  if (!isSupportedCombiningKind(getKind(), eltType))
    return emitOpError("unsupported reduction type ")
           << eltType << " for kind '" << stringifyCombiningKind(getKind())
           << "'";

  return success();
}

void mlir::vector::populateVectorToVectorCanonicalizationPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns
      .add<CreateMaskFolder, MaskedLoadFolder, MaskedStoreFolder, GatherFolder,
           ScatterFolder, ExpandLoadFolder, CompressStoreFolder,
           StridedSliceConstantMaskFolder, TransposeFolder>(
          patterns.getContext(), benefit);
}

//===----------------------------------------------------------------------===//
// SplatOp
//===----------------------------------------------------------------------===//

OpFoldResult SplatOp::fold(FoldAdaptor adaptor) {
  auto constOperand = adaptor.getInput();
  if (!isa_and_nonnull<IntegerAttr, FloatAttr>(constOperand))
    return {};

  // SplatElementsAttr::get treats single value for second arg as being a splat.
  return SplatElementsAttr::get(getType(), {constOperand});
}

// Canonicalizer for vector.splat. It always gets canonicalized to a
// vector.broadcast.
class SplatToBroadcastPattern final : public OpRewritePattern<SplatOp> {
public:
  using OpRewritePattern<SplatOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(SplatOp splatOp,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(splatOp, splatOp.getType(),
                                                     splatOp.getOperand());
    return success();
  }
};
void SplatOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.add<SplatToBroadcastPattern>(context);
}

void SplatOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                SetIntRangeFn setResultRanges) {
  setResultRanges(getResult(), argRanges.front());
}

Value mlir::vector::makeArithReduction(OpBuilder &b, Location loc,
                                       CombiningKind kind, Value v1, Value acc,
                                       arith::FastMathFlagsAttr fastmath,
                                       Value mask) {
  Type t1 = getElementTypeOrSelf(v1.getType());
  Type tAcc = getElementTypeOrSelf(acc.getType());
  Value result;

  switch (kind) {
  case CombiningKind::ADD:
    if (t1.isIntOrIndex() && tAcc.isIntOrIndex())
      result = b.createOrFold<arith::AddIOp>(loc, v1, acc);
    else if (llvm::isa<FloatType>(t1) && llvm::isa<FloatType>(tAcc))
      result = b.createOrFold<arith::AddFOp>(loc, v1, acc, fastmath);
    else
      llvm_unreachable("invalid value types for ADD reduction");
    break;
  case CombiningKind::AND:
    assert(t1.isIntOrIndex() && tAcc.isIntOrIndex() && "expected int values");
    result = b.createOrFold<arith::AndIOp>(loc, v1, acc);
    break;
  case CombiningKind::MAXNUMF:
    assert(llvm::isa<FloatType>(t1) && llvm::isa<FloatType>(tAcc) &&
           "expected float values");
    result = b.createOrFold<arith::MaxNumFOp>(loc, v1, acc, fastmath);
    break;
  case CombiningKind::MAXIMUMF:
    assert(llvm::isa<FloatType>(t1) && llvm::isa<FloatType>(tAcc) &&
           "expected float values");
    result = b.createOrFold<arith::MaximumFOp>(loc, v1, acc, fastmath);
    break;
  case CombiningKind::MINNUMF:
    assert(llvm::isa<FloatType>(t1) && llvm::isa<FloatType>(tAcc) &&
           "expected float values");
    result = b.createOrFold<arith::MinNumFOp>(loc, v1, acc, fastmath);
    break;
  case CombiningKind::MINIMUMF:
    assert(llvm::isa<FloatType>(t1) && llvm::isa<FloatType>(tAcc) &&
           "expected float values");
    result = b.createOrFold<arith::MinimumFOp>(loc, v1, acc, fastmath);
    break;
  case CombiningKind::MAXSI:
    assert(t1.isIntOrIndex() && tAcc.isIntOrIndex() && "expected int values");
    result = b.createOrFold<arith::MaxSIOp>(loc, v1, acc);
    break;
  case CombiningKind::MINSI:
    assert(t1.isIntOrIndex() && tAcc.isIntOrIndex() && "expected int values");
    result = b.createOrFold<arith::MinSIOp>(loc, v1, acc);
    break;
  case CombiningKind::MAXUI:
    assert(t1.isIntOrIndex() && tAcc.isIntOrIndex() && "expected int values");
    result = b.createOrFold<arith::MaxUIOp>(loc, v1, acc);
    break;
  case CombiningKind::MINUI:
    assert(t1.isIntOrIndex() && tAcc.isIntOrIndex() && "expected int values");
    result = b.createOrFold<arith::MinUIOp>(loc, v1, acc);
    break;
  case CombiningKind::MUL:
    if (t1.isIntOrIndex() && tAcc.isIntOrIndex())
      result = b.createOrFold<arith::MulIOp>(loc, v1, acc);
    else if (llvm::isa<FloatType>(t1) && llvm::isa<FloatType>(tAcc))
      result = b.createOrFold<arith::MulFOp>(loc, v1, acc, fastmath);
    else
      llvm_unreachable("invalid value types for MUL reduction");
    break;
  case CombiningKind::OR:
    assert(t1.isIntOrIndex() && tAcc.isIntOrIndex() && "expected int values");
    result = b.createOrFold<arith::OrIOp>(loc, v1, acc);
    break;
  case CombiningKind::XOR:
    assert(t1.isIntOrIndex() && tAcc.isIntOrIndex() && "expected int values");
    result = b.createOrFold<arith::XOrIOp>(loc, v1, acc);
    break;
  };

  assert(result && "unknown CombiningKind");
  return selectPassthru(b, mask, result, acc);
}

//===----------------------------------------------------------------------===//
// StepOp
//===----------------------------------------------------------------------===//

void StepOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                               SetIntRangeFn setResultRanges) {
  auto resultType = cast<VectorType>(getType());
  if (resultType.isScalable()) {
    return;
  }
  unsigned bitwidth = ConstantIntRanges::getStorageBitwidth(resultType);
  APInt zero(bitwidth, 0);
  APInt high(bitwidth, resultType.getDimSize(0) - 1);
  ConstantIntRanges result = {zero, high, zero, high};
  setResultRanges(getResult(), result);
}

//===----------------------------------------------------------------------===//
// Vector Masking Utilities
//===----------------------------------------------------------------------===//

/// Create the vector.yield-ended region of a vector.mask op with `maskableOp`
/// as masked operation.
void mlir::vector::createMaskOpRegion(OpBuilder &builder,
                                      Operation *maskableOp) {
  assert(maskableOp->getBlock() && "MaskableOp must be inserted into a block");
  Block *insBlock = builder.getInsertionBlock();
  // Create a block and move the op to that block.
  insBlock->getOperations().splice(
      insBlock->begin(), maskableOp->getBlock()->getOperations(), maskableOp);
  YieldOp::create(builder, maskableOp->getLoc(), maskableOp->getResults());
}

/// Creates a vector.mask operation around a maskable operation. Returns the
/// vector.mask operation if the mask provided is valid. Otherwise, returns
/// the maskable operation itself.
Operation *mlir::vector::maskOperation(OpBuilder &builder,
                                       Operation *maskableOp, Value mask,
                                       Value passthru) {
  if (!mask)
    return maskableOp;
  if (passthru)
    return MaskOp::create(builder, maskableOp->getLoc(),
                          maskableOp->getResultTypes(), mask, passthru,
                          maskableOp, createMaskOpRegion);
  return MaskOp::create(builder, maskableOp->getLoc(),
                        maskableOp->getResultTypes(), mask, maskableOp,
                        createMaskOpRegion);
}

/// Creates a vector select operation that picks values from `newValue` or
/// `passthru` for each result vector lane based on `mask`. This utility is used
/// to propagate the pass-thru value of vector.mask or for cases where only the
/// pass-thru value propagation is needed. VP intrinsics do not support
/// pass-thru values and every mask-out lane is set to poison. LLVM backends are
/// usually able to match op + select patterns and fold them into a native
/// target instructions.
Value mlir::vector::selectPassthru(OpBuilder &builder, Value mask,
                                   Value newValue, Value passthru) {
  if (!mask)
    return newValue;

  return arith::SelectOp::create(builder, newValue.getLoc(), newValue.getType(),
                                 mask, newValue, passthru);
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Vector/IR/VectorAttributes.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Vector/IR/VectorOps.cpp.inc"
