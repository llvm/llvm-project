//===- BuiltinTypes.cpp - MLIR Builtin Type Classes -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinTypes.h"
#include "TypeDetail.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TensorEncoding.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::detail;

//===----------------------------------------------------------------------===//
/// Tablegen Type Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/IR/BuiltinTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// BuiltinDialect
//===----------------------------------------------------------------------===//

void BuiltinDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/IR/BuiltinTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
/// ComplexType
//===----------------------------------------------------------------------===//

/// Verify the construction of an integer type.
LogicalResult ComplexType::verify(function_ref<InFlightDiagnostic()> emitError,
                                  Type elementType) {
  if (!elementType.isIntOrFloat())
    return emitError() << "invalid element type for complex";
  return success();
}

//===----------------------------------------------------------------------===//
// Integer Type
//===----------------------------------------------------------------------===//

/// Verify the construction of an integer type.
LogicalResult IntegerType::verify(function_ref<InFlightDiagnostic()> emitError,
                                  unsigned width,
                                  SignednessSemantics signedness) {
  if (width > IntegerType::kMaxWidth) {
    return emitError() << "integer bitwidth is limited to "
                       << IntegerType::kMaxWidth << " bits";
  }
  return success();
}

unsigned IntegerType::getWidth() const { return getImpl()->width; }

IntegerType::SignednessSemantics IntegerType::getSignedness() const {
  return getImpl()->signedness;
}

IntegerType IntegerType::scaleElementBitwidth(unsigned scale) {
  if (!scale)
    return IntegerType();
  return IntegerType::get(getContext(), scale * getWidth(), getSignedness());
}

//===----------------------------------------------------------------------===//
// Float Type
//===----------------------------------------------------------------------===//

unsigned FloatType::getWidth() {
  if (isa<Float8E5M2Type, Float8E4M3FNType>())
    return 8;
  if (isa<Float16Type, BFloat16Type>())
    return 16;
  if (isa<Float32Type>())
    return 32;
  if (isa<Float64Type>())
    return 64;
  if (isa<Float80Type>())
    return 80;
  if (isa<Float128Type>())
    return 128;
  llvm_unreachable("unexpected float type");
}

/// Returns the floating semantics for the given type.
const llvm::fltSemantics &FloatType::getFloatSemantics() {
  if (isa<Float8E5M2Type>())
    return APFloat::Float8E5M2();
  if (isa<Float8E4M3FNType>())
    return APFloat::Float8E4M3FN();
  if (isa<BFloat16Type>())
    return APFloat::BFloat();
  if (isa<Float16Type>())
    return APFloat::IEEEhalf();
  if (isa<Float32Type>())
    return APFloat::IEEEsingle();
  if (isa<Float64Type>())
    return APFloat::IEEEdouble();
  if (isa<Float80Type>())
    return APFloat::x87DoubleExtended();
  if (isa<Float128Type>())
    return APFloat::IEEEquad();
  llvm_unreachable("non-floating point type used");
}

FloatType FloatType::scaleElementBitwidth(unsigned scale) {
  if (!scale)
    return FloatType();
  MLIRContext *ctx = getContext();
  if (isF16() || isBF16()) {
    if (scale == 2)
      return FloatType::getF32(ctx);
    if (scale == 4)
      return FloatType::getF64(ctx);
  }
  if (isF32())
    if (scale == 2)
      return FloatType::getF64(ctx);
  return FloatType();
}

unsigned FloatType::getFPMantissaWidth() {
  return APFloat::semanticsPrecision(getFloatSemantics());
}

//===----------------------------------------------------------------------===//
// FunctionType
//===----------------------------------------------------------------------===//

unsigned FunctionType::getNumInputs() const { return getImpl()->numInputs; }

ArrayRef<Type> FunctionType::getInputs() const {
  return getImpl()->getInputs();
}

unsigned FunctionType::getNumResults() const { return getImpl()->numResults; }

ArrayRef<Type> FunctionType::getResults() const {
  return getImpl()->getResults();
}

FunctionType FunctionType::clone(TypeRange inputs, TypeRange results) const {
  return get(getContext(), inputs, results);
}

/// Returns a new function type with the specified arguments and results
/// inserted.
FunctionType FunctionType::getWithArgsAndResults(
    ArrayRef<unsigned> argIndices, TypeRange argTypes,
    ArrayRef<unsigned> resultIndices, TypeRange resultTypes) {
  SmallVector<Type> argStorage, resultStorage;
  TypeRange newArgTypes = function_interface_impl::insertTypesInto(
      getInputs(), argIndices, argTypes, argStorage);
  TypeRange newResultTypes = function_interface_impl::insertTypesInto(
      getResults(), resultIndices, resultTypes, resultStorage);
  return clone(newArgTypes, newResultTypes);
}

/// Returns a new function type without the specified arguments and results.
FunctionType
FunctionType::getWithoutArgsAndResults(const BitVector &argIndices,
                                       const BitVector &resultIndices) {
  SmallVector<Type> argStorage, resultStorage;
  TypeRange newArgTypes = function_interface_impl::filterTypesOut(
      getInputs(), argIndices, argStorage);
  TypeRange newResultTypes = function_interface_impl::filterTypesOut(
      getResults(), resultIndices, resultStorage);
  return clone(newArgTypes, newResultTypes);
}

//===----------------------------------------------------------------------===//
// OpaqueType
//===----------------------------------------------------------------------===//

/// Verify the construction of an opaque type.
LogicalResult OpaqueType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 StringAttr dialect, StringRef typeData) {
  if (!Dialect::isValidNamespace(dialect.strref()))
    return emitError() << "invalid dialect namespace '" << dialect << "'";

  // Check that the dialect is actually registered.
  MLIRContext *context = dialect.getContext();
  if (!context->allowsUnregisteredDialects() &&
      !context->getLoadedDialect(dialect.strref())) {
    return emitError()
           << "`!" << dialect << "<\"" << typeData << "\">"
           << "` type created with unregistered dialect. If this is "
              "intended, please call allowUnregisteredDialects() on the "
              "MLIRContext, or use -allow-unregistered-dialect with "
              "the MLIR opt tool used";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// VectorType
//===----------------------------------------------------------------------===//

LogicalResult VectorType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 ArrayRef<int64_t> shape, Type elementType,
                                 unsigned numScalableDims) {
  if (!isValidElementType(elementType))
    return emitError()
           << "vector elements must be int/index/float type but got "
           << elementType;

  if (any_of(shape, [](int64_t i) { return i <= 0; }))
    return emitError()
           << "vector types must have positive constant sizes but got "
           << shape;

  return success();
}

VectorType VectorType::scaleElementBitwidth(unsigned scale) {
  if (!scale)
    return VectorType();
  if (auto et = getElementType().dyn_cast<IntegerType>())
    if (auto scaledEt = et.scaleElementBitwidth(scale))
      return VectorType::get(getShape(), scaledEt, getNumScalableDims());
  if (auto et = getElementType().dyn_cast<FloatType>())
    if (auto scaledEt = et.scaleElementBitwidth(scale))
      return VectorType::get(getShape(), scaledEt, getNumScalableDims());
  return VectorType();
}

VectorType VectorType::cloneWith(Optional<ArrayRef<int64_t>> shape,
                                 Type elementType) const {
  return VectorType::get(shape.value_or(getShape()), elementType,
                         getNumScalableDims());
}

//===----------------------------------------------------------------------===//
// TensorType
//===----------------------------------------------------------------------===//

Type TensorType::getElementType() const {
  return llvm::TypeSwitch<TensorType, Type>(*this)
      .Case<RankedTensorType, UnrankedTensorType>(
          [](auto type) { return type.getElementType(); });
}

bool TensorType::hasRank() const { return !isa<UnrankedTensorType>(); }

ArrayRef<int64_t> TensorType::getShape() const {
  return cast<RankedTensorType>().getShape();
}

TensorType TensorType::cloneWith(Optional<ArrayRef<int64_t>> shape,
                                 Type elementType) const {
  if (auto unrankedTy = dyn_cast<UnrankedTensorType>()) {
    if (shape)
      return RankedTensorType::get(*shape, elementType);
    return UnrankedTensorType::get(elementType);
  }

  auto rankedTy = cast<RankedTensorType>();
  if (!shape)
    return RankedTensorType::get(rankedTy.getShape(), elementType,
                                 rankedTy.getEncoding());
  return RankedTensorType::get(shape.value_or(rankedTy.getShape()), elementType,
                               rankedTy.getEncoding());
}

// Check if "elementType" can be an element type of a tensor.
static LogicalResult
checkTensorElementType(function_ref<InFlightDiagnostic()> emitError,
                       Type elementType) {
  if (!TensorType::isValidElementType(elementType))
    return emitError() << "invalid tensor element type: " << elementType;
  return success();
}

/// Return true if the specified element type is ok in a tensor.
bool TensorType::isValidElementType(Type type) {
  // Note: Non standard/builtin types are allowed to exist within tensor
  // types. Dialects are expected to verify that tensor types have a valid
  // element type within that dialect.
  return type.isa<ComplexType, FloatType, IntegerType, OpaqueType, VectorType,
                  IndexType>() ||
         !llvm::isa<BuiltinDialect>(type.getDialect());
}

//===----------------------------------------------------------------------===//
// RankedTensorType
//===----------------------------------------------------------------------===//

LogicalResult
RankedTensorType::verify(function_ref<InFlightDiagnostic()> emitError,
                         ArrayRef<int64_t> shape, Type elementType,
                         Attribute encoding) {
  for (int64_t s : shape)
    if (s < 0 && !ShapedType::isDynamic(s))
      return emitError() << "invalid tensor dimension size";
  if (auto v = encoding.dyn_cast_or_null<VerifiableTensorEncoding>())
    if (failed(v.verifyEncoding(shape, elementType, emitError)))
      return failure();
  return checkTensorElementType(emitError, elementType);
}

//===----------------------------------------------------------------------===//
// UnrankedTensorType
//===----------------------------------------------------------------------===//

LogicalResult
UnrankedTensorType::verify(function_ref<InFlightDiagnostic()> emitError,
                           Type elementType) {
  return checkTensorElementType(emitError, elementType);
}

//===----------------------------------------------------------------------===//
// BaseMemRefType
//===----------------------------------------------------------------------===//

Type BaseMemRefType::getElementType() const {
  return llvm::TypeSwitch<BaseMemRefType, Type>(*this)
      .Case<MemRefType, UnrankedMemRefType>(
          [](auto type) { return type.getElementType(); });
}

bool BaseMemRefType::hasRank() const { return !isa<UnrankedMemRefType>(); }

ArrayRef<int64_t> BaseMemRefType::getShape() const {
  return cast<MemRefType>().getShape();
}

BaseMemRefType BaseMemRefType::cloneWith(Optional<ArrayRef<int64_t>> shape,
                                         Type elementType) const {
  if (auto unrankedTy = dyn_cast<UnrankedMemRefType>()) {
    if (!shape)
      return UnrankedMemRefType::get(elementType, getMemorySpace());
    MemRefType::Builder builder(*shape, elementType);
    builder.setMemorySpace(getMemorySpace());
    return builder;
  }

  MemRefType::Builder builder(cast<MemRefType>());
  if (shape)
    builder.setShape(*shape);
  builder.setElementType(elementType);
  return builder;
}

Attribute BaseMemRefType::getMemorySpace() const {
  if (auto rankedMemRefTy = dyn_cast<MemRefType>())
    return rankedMemRefTy.getMemorySpace();
  return cast<UnrankedMemRefType>().getMemorySpace();
}

unsigned BaseMemRefType::getMemorySpaceAsInt() const {
  if (auto rankedMemRefTy = dyn_cast<MemRefType>())
    return rankedMemRefTy.getMemorySpaceAsInt();
  return cast<UnrankedMemRefType>().getMemorySpaceAsInt();
}

//===----------------------------------------------------------------------===//
// MemRefType
//===----------------------------------------------------------------------===//

/// Given an `originalShape` and a `reducedShape` assumed to be a subset of
/// `originalShape` with some `1` entries erased, return the set of indices
/// that specifies which of the entries of `originalShape` are dropped to obtain
/// `reducedShape`. The returned mask can be applied as a projection to
/// `originalShape` to obtain the `reducedShape`. This mask is useful to track
/// which dimensions must be kept when e.g. compute MemRef strides under
/// rank-reducing operations. Return std::nullopt if reducedShape cannot be
/// obtained by dropping only `1` entries in `originalShape`.
llvm::Optional<llvm::SmallDenseSet<unsigned>>
mlir::computeRankReductionMask(ArrayRef<int64_t> originalShape,
                               ArrayRef<int64_t> reducedShape) {
  size_t originalRank = originalShape.size(), reducedRank = reducedShape.size();
  llvm::SmallDenseSet<unsigned> unusedDims;
  unsigned reducedIdx = 0;
  for (unsigned originalIdx = 0; originalIdx < originalRank; ++originalIdx) {
    // Greedily insert `originalIdx` if match.
    if (reducedIdx < reducedRank &&
        originalShape[originalIdx] == reducedShape[reducedIdx]) {
      reducedIdx++;
      continue;
    }

    unusedDims.insert(originalIdx);
    // If no match on `originalIdx`, the `originalShape` at this dimension
    // must be 1, otherwise we bail.
    if (originalShape[originalIdx] != 1)
      return std::nullopt;
  }
  // The whole reducedShape must be scanned, otherwise we bail.
  if (reducedIdx != reducedRank)
    return std::nullopt;
  return unusedDims;
}

SliceVerificationResult
mlir::isRankReducedType(ShapedType originalType,
                        ShapedType candidateReducedType) {
  if (originalType == candidateReducedType)
    return SliceVerificationResult::Success;

  ShapedType originalShapedType = originalType.cast<ShapedType>();
  ShapedType candidateReducedShapedType =
      candidateReducedType.cast<ShapedType>();

  // Rank and size logic is valid for all ShapedTypes.
  ArrayRef<int64_t> originalShape = originalShapedType.getShape();
  ArrayRef<int64_t> candidateReducedShape =
      candidateReducedShapedType.getShape();
  unsigned originalRank = originalShape.size(),
           candidateReducedRank = candidateReducedShape.size();
  if (candidateReducedRank > originalRank)
    return SliceVerificationResult::RankTooLarge;

  auto optionalUnusedDimsMask =
      computeRankReductionMask(originalShape, candidateReducedShape);

  // Sizes cannot be matched in case empty vector is returned.
  if (!optionalUnusedDimsMask)
    return SliceVerificationResult::SizeMismatch;

  if (originalShapedType.getElementType() !=
      candidateReducedShapedType.getElementType())
    return SliceVerificationResult::ElemTypeMismatch;

  return SliceVerificationResult::Success;
}

bool mlir::detail::isSupportedMemorySpace(Attribute memorySpace) {
  // Empty attribute is allowed as default memory space.
  if (!memorySpace)
    return true;

  // Supported built-in attributes.
  if (memorySpace.isa<IntegerAttr, StringAttr, DictionaryAttr>())
    return true;

  // Allow custom dialect attributes.
  if (!isa<BuiltinDialect>(memorySpace.getDialect()))
    return true;

  return false;
}

Attribute mlir::detail::wrapIntegerMemorySpace(unsigned memorySpace,
                                               MLIRContext *ctx) {
  if (memorySpace == 0)
    return nullptr;

  return IntegerAttr::get(IntegerType::get(ctx, 64), memorySpace);
}

Attribute mlir::detail::skipDefaultMemorySpace(Attribute memorySpace) {
  IntegerAttr intMemorySpace = memorySpace.dyn_cast_or_null<IntegerAttr>();
  if (intMemorySpace && intMemorySpace.getValue() == 0)
    return nullptr;

  return memorySpace;
}

unsigned mlir::detail::getMemorySpaceAsInt(Attribute memorySpace) {
  if (!memorySpace)
    return 0;

  assert(memorySpace.isa<IntegerAttr>() &&
         "Using `getMemorySpaceInteger` with non-Integer attribute");

  return static_cast<unsigned>(memorySpace.cast<IntegerAttr>().getInt());
}

unsigned MemRefType::getMemorySpaceAsInt() const {
  return detail::getMemorySpaceAsInt(getMemorySpace());
}

MemRefType MemRefType::get(ArrayRef<int64_t> shape, Type elementType,
                           MemRefLayoutAttrInterface layout,
                           Attribute memorySpace) {
  // Use default layout for empty attribute.
  if (!layout)
    layout = AffineMapAttr::get(AffineMap::getMultiDimIdentityMap(
        shape.size(), elementType.getContext()));

  // Drop default memory space value and replace it with empty attribute.
  memorySpace = skipDefaultMemorySpace(memorySpace);

  return Base::get(elementType.getContext(), shape, elementType, layout,
                   memorySpace);
}

MemRefType MemRefType::getChecked(
    function_ref<InFlightDiagnostic()> emitErrorFn, ArrayRef<int64_t> shape,
    Type elementType, MemRefLayoutAttrInterface layout, Attribute memorySpace) {

  // Use default layout for empty attribute.
  if (!layout)
    layout = AffineMapAttr::get(AffineMap::getMultiDimIdentityMap(
        shape.size(), elementType.getContext()));

  // Drop default memory space value and replace it with empty attribute.
  memorySpace = skipDefaultMemorySpace(memorySpace);

  return Base::getChecked(emitErrorFn, elementType.getContext(), shape,
                          elementType, layout, memorySpace);
}

MemRefType MemRefType::get(ArrayRef<int64_t> shape, Type elementType,
                           AffineMap map, Attribute memorySpace) {

  // Use default layout for empty map.
  if (!map)
    map = AffineMap::getMultiDimIdentityMap(shape.size(),
                                            elementType.getContext());

  // Wrap AffineMap into Attribute.
  Attribute layout = AffineMapAttr::get(map);

  // Drop default memory space value and replace it with empty attribute.
  memorySpace = skipDefaultMemorySpace(memorySpace);

  return Base::get(elementType.getContext(), shape, elementType, layout,
                   memorySpace);
}

MemRefType
MemRefType::getChecked(function_ref<InFlightDiagnostic()> emitErrorFn,
                       ArrayRef<int64_t> shape, Type elementType, AffineMap map,
                       Attribute memorySpace) {

  // Use default layout for empty map.
  if (!map)
    map = AffineMap::getMultiDimIdentityMap(shape.size(),
                                            elementType.getContext());

  // Wrap AffineMap into Attribute.
  Attribute layout = AffineMapAttr::get(map);

  // Drop default memory space value and replace it with empty attribute.
  memorySpace = skipDefaultMemorySpace(memorySpace);

  return Base::getChecked(emitErrorFn, elementType.getContext(), shape,
                          elementType, layout, memorySpace);
}

MemRefType MemRefType::get(ArrayRef<int64_t> shape, Type elementType,
                           AffineMap map, unsigned memorySpaceInd) {

  // Use default layout for empty map.
  if (!map)
    map = AffineMap::getMultiDimIdentityMap(shape.size(),
                                            elementType.getContext());

  // Wrap AffineMap into Attribute.
  Attribute layout = AffineMapAttr::get(map);

  // Convert deprecated integer-like memory space to Attribute.
  Attribute memorySpace =
      wrapIntegerMemorySpace(memorySpaceInd, elementType.getContext());

  return Base::get(elementType.getContext(), shape, elementType, layout,
                   memorySpace);
}

MemRefType
MemRefType::getChecked(function_ref<InFlightDiagnostic()> emitErrorFn,
                       ArrayRef<int64_t> shape, Type elementType, AffineMap map,
                       unsigned memorySpaceInd) {

  // Use default layout for empty map.
  if (!map)
    map = AffineMap::getMultiDimIdentityMap(shape.size(),
                                            elementType.getContext());

  // Wrap AffineMap into Attribute.
  Attribute layout = AffineMapAttr::get(map);

  // Convert deprecated integer-like memory space to Attribute.
  Attribute memorySpace =
      wrapIntegerMemorySpace(memorySpaceInd, elementType.getContext());

  return Base::getChecked(emitErrorFn, elementType.getContext(), shape,
                          elementType, layout, memorySpace);
}

LogicalResult MemRefType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 ArrayRef<int64_t> shape, Type elementType,
                                 MemRefLayoutAttrInterface layout,
                                 Attribute memorySpace) {
  if (!BaseMemRefType::isValidElementType(elementType))
    return emitError() << "invalid memref element type";

  // Negative sizes are not allowed except for `kDynamic`.
  for (int64_t s : shape)
    if (s < 0 && !ShapedType::isDynamic(s))
      return emitError() << "invalid memref size";

  assert(layout && "missing layout specification");
  if (failed(layout.verifyLayout(shape, emitError)))
    return failure();

  if (!isSupportedMemorySpace(memorySpace))
    return emitError() << "unsupported memory space Attribute";

  return success();
}

//===----------------------------------------------------------------------===//
// UnrankedMemRefType
//===----------------------------------------------------------------------===//

unsigned UnrankedMemRefType::getMemorySpaceAsInt() const {
  return detail::getMemorySpaceAsInt(getMemorySpace());
}

LogicalResult
UnrankedMemRefType::verify(function_ref<InFlightDiagnostic()> emitError,
                           Type elementType, Attribute memorySpace) {
  if (!BaseMemRefType::isValidElementType(elementType))
    return emitError() << "invalid memref element type";

  if (!isSupportedMemorySpace(memorySpace))
    return emitError() << "unsupported memory space Attribute";

  return success();
}

// Fallback cases for terminal dim/sym/cst that are not part of a binary op (
// i.e. single term). Accumulate the AffineExpr into the existing one.
static void extractStridesFromTerm(AffineExpr e,
                                   AffineExpr multiplicativeFactor,
                                   MutableArrayRef<AffineExpr> strides,
                                   AffineExpr &offset) {
  if (auto dim = e.dyn_cast<AffineDimExpr>())
    strides[dim.getPosition()] =
        strides[dim.getPosition()] + multiplicativeFactor;
  else
    offset = offset + e * multiplicativeFactor;
}

/// Takes a single AffineExpr `e` and populates the `strides` array with the
/// strides expressions for each dim position.
/// The convention is that the strides for dimensions d0, .. dn appear in
/// order to make indexing intuitive into the result.
static LogicalResult extractStrides(AffineExpr e,
                                    AffineExpr multiplicativeFactor,
                                    MutableArrayRef<AffineExpr> strides,
                                    AffineExpr &offset) {
  auto bin = e.dyn_cast<AffineBinaryOpExpr>();
  if (!bin) {
    extractStridesFromTerm(e, multiplicativeFactor, strides, offset);
    return success();
  }

  if (bin.getKind() == AffineExprKind::CeilDiv ||
      bin.getKind() == AffineExprKind::FloorDiv ||
      bin.getKind() == AffineExprKind::Mod)
    return failure();

  if (bin.getKind() == AffineExprKind::Mul) {
    auto dim = bin.getLHS().dyn_cast<AffineDimExpr>();
    if (dim) {
      strides[dim.getPosition()] =
          strides[dim.getPosition()] + bin.getRHS() * multiplicativeFactor;
      return success();
    }
    // LHS and RHS may both contain complex expressions of dims. Try one path
    // and if it fails try the other. This is guaranteed to succeed because
    // only one path may have a `dim`, otherwise this is not an AffineExpr in
    // the first place.
    if (bin.getLHS().isSymbolicOrConstant())
      return extractStrides(bin.getRHS(), multiplicativeFactor * bin.getLHS(),
                            strides, offset);
    return extractStrides(bin.getLHS(), multiplicativeFactor * bin.getRHS(),
                          strides, offset);
  }

  if (bin.getKind() == AffineExprKind::Add) {
    auto res1 =
        extractStrides(bin.getLHS(), multiplicativeFactor, strides, offset);
    auto res2 =
        extractStrides(bin.getRHS(), multiplicativeFactor, strides, offset);
    return success(succeeded(res1) && succeeded(res2));
  }

  llvm_unreachable("unexpected binary operation");
}

/// A stride specification is a list of integer values that are either static
/// or dynamic (encoded with ShapedType::kDynamic). Strides encode
/// the distance in the number of elements between successive entries along a
/// particular dimension.
///
/// For example, `memref<42x16xf32, (64 * d0 + d1)>` specifies a view into a
/// non-contiguous memory region of `42` by `16` `f32` elements in which the
/// distance between two consecutive elements along the outer dimension is `1`
/// and the distance between two consecutive elements along the inner dimension
/// is `64`.
///
/// The convention is that the strides for dimensions d0, .. dn appear in
/// order to make indexing intuitive into the result.
static LogicalResult getStridesAndOffset(MemRefType t,
                                         SmallVectorImpl<AffineExpr> &strides,
                                         AffineExpr &offset) {
  AffineMap m = t.getLayout().getAffineMap();

  if (m.getNumResults() != 1 && !m.isIdentity())
    return failure();

  auto zero = getAffineConstantExpr(0, t.getContext());
  auto one = getAffineConstantExpr(1, t.getContext());
  offset = zero;
  strides.assign(t.getRank(), zero);

  // Canonical case for empty map.
  if (m.isIdentity()) {
    // 0-D corner case, offset is already 0.
    if (t.getRank() == 0)
      return success();
    auto stridedExpr =
        makeCanonicalStridedLayoutExpr(t.getShape(), t.getContext());
    if (succeeded(extractStrides(stridedExpr, one, strides, offset)))
      return success();
    assert(false && "unexpected failure: extract strides in canonical layout");
  }

  // Non-canonical case requires more work.
  auto stridedExpr =
      simplifyAffineExpr(m.getResult(0), m.getNumDims(), m.getNumSymbols());
  if (failed(extractStrides(stridedExpr, one, strides, offset))) {
    offset = AffineExpr();
    strides.clear();
    return failure();
  }

  // Simplify results to allow folding to constants and simple checks.
  unsigned numDims = m.getNumDims();
  unsigned numSymbols = m.getNumSymbols();
  offset = simplifyAffineExpr(offset, numDims, numSymbols);
  for (auto &stride : strides)
    stride = simplifyAffineExpr(stride, numDims, numSymbols);

  // In practice, a strided memref must be internally non-aliasing. Test
  // against 0 as a proxy.
  // TODO: static cases can have more advanced checks.
  // TODO: dynamic cases would require a way to compare symbolic
  // expressions and would probably need an affine set context propagated
  // everywhere.
  if (llvm::any_of(strides, [](AffineExpr e) {
        return e == getAffineConstantExpr(0, e.getContext());
      })) {
    offset = AffineExpr();
    strides.clear();
    return failure();
  }

  return success();
}

LogicalResult mlir::getStridesAndOffset(MemRefType t,
                                        SmallVectorImpl<int64_t> &strides,
                                        int64_t &offset) {
  // Happy path: the type uses the strided layout directly.
  if (auto strided = t.getLayout().dyn_cast<StridedLayoutAttr>()) {
    llvm::append_range(strides, strided.getStrides());
    offset = strided.getOffset();
    return success();
  }

  // Otherwise, defer to the affine fallback as layouts are supposed to be
  // convertible to affine maps.
  AffineExpr offsetExpr;
  SmallVector<AffineExpr, 4> strideExprs;
  if (failed(::getStridesAndOffset(t, strideExprs, offsetExpr)))
    return failure();
  if (auto cst = offsetExpr.dyn_cast<AffineConstantExpr>())
    offset = cst.getValue();
  else
    offset = ShapedType::kDynamic;
  for (auto e : strideExprs) {
    if (auto c = e.dyn_cast<AffineConstantExpr>())
      strides.push_back(c.getValue());
    else
      strides.push_back(ShapedType::kDynamic);
  }
  return success();
}

std::pair<SmallVector<int64_t>, int64_t>
mlir::getStridesAndOffset(MemRefType t) {
  SmallVector<int64_t> strides;
  int64_t offset;
  LogicalResult status = getStridesAndOffset(t, strides, offset);
  (void)status;
  assert(succeeded(status) && "Invalid use of check-free getStridesAndOffset");
  return {strides, offset};
}

//===----------------------------------------------------------------------===//
/// TupleType
//===----------------------------------------------------------------------===//

/// Return the elements types for this tuple.
ArrayRef<Type> TupleType::getTypes() const { return getImpl()->getTypes(); }

/// Accumulate the types contained in this tuple and tuples nested within it.
/// Note that this only flattens nested tuples, not any other container type,
/// e.g. a tuple<i32, tensor<i32>, tuple<f32, tuple<i64>>> is flattened to
/// (i32, tensor<i32>, f32, i64)
void TupleType::getFlattenedTypes(SmallVectorImpl<Type> &types) {
  for (Type type : getTypes()) {
    if (auto nestedTuple = type.dyn_cast<TupleType>())
      nestedTuple.getFlattenedTypes(types);
    else
      types.push_back(type);
  }
}

/// Return the number of element types.
size_t TupleType::size() const { return getImpl()->size(); }

//===----------------------------------------------------------------------===//
// Type Utilities
//===----------------------------------------------------------------------===//

/// Return a version of `t` with identity layout if it can be determined
/// statically that the layout is the canonical contiguous strided layout.
/// Otherwise pass `t`'s layout into `simplifyAffineMap` and return a copy of
/// `t` with simplified layout.
/// If `t` has multiple layout maps or a multi-result layout, just return `t`.
MemRefType mlir::canonicalizeStridedLayout(MemRefType t) {
  AffineMap m = t.getLayout().getAffineMap();

  // Already in canonical form.
  if (m.isIdentity())
    return t;

  // Can't reduce to canonical identity form, return in canonical form.
  if (m.getNumResults() > 1)
    return t;

  // Corner-case for 0-D affine maps.
  if (m.getNumDims() == 0 && m.getNumSymbols() == 0) {
    if (auto cst = m.getResult(0).dyn_cast<AffineConstantExpr>())
      if (cst.getValue() == 0)
        return MemRefType::Builder(t).setLayout({});
    return t;
  }

  // 0-D corner case for empty shape that still have an affine map. Example:
  // `memref<f32, affine_map<()[s0] -> (s0)>>`. This is a 1 element memref whose
  // offset needs to remain, just return t.
  if (t.getShape().empty())
    return t;

  // If the canonical strided layout for the sizes of `t` is equal to the
  // simplified layout of `t` we can just return an empty layout. Otherwise,
  // just simplify the existing layout.
  AffineExpr expr =
      makeCanonicalStridedLayoutExpr(t.getShape(), t.getContext());
  auto simplifiedLayoutExpr =
      simplifyAffineExpr(m.getResult(0), m.getNumDims(), m.getNumSymbols());
  if (expr != simplifiedLayoutExpr)
    return MemRefType::Builder(t).setLayout(AffineMapAttr::get(AffineMap::get(
        m.getNumDims(), m.getNumSymbols(), simplifiedLayoutExpr)));
  return MemRefType::Builder(t).setLayout({});
}

AffineExpr mlir::makeCanonicalStridedLayoutExpr(ArrayRef<int64_t> sizes,
                                                ArrayRef<AffineExpr> exprs,
                                                MLIRContext *context) {
  // Size 0 corner case is useful for canonicalizations.
  if (sizes.empty())
    return getAffineConstantExpr(0, context);

  assert(!exprs.empty() && "expected exprs");
  auto maps = AffineMap::inferFromExprList(exprs);
  assert(!maps.empty() && "Expected one non-empty map");
  unsigned numDims = maps[0].getNumDims(), nSymbols = maps[0].getNumSymbols();

  AffineExpr expr;
  bool dynamicPoisonBit = false;
  int64_t runningSize = 1;
  for (auto en : llvm::zip(llvm::reverse(exprs), llvm::reverse(sizes))) {
    int64_t size = std::get<1>(en);
    AffineExpr dimExpr = std::get<0>(en);
    AffineExpr stride = dynamicPoisonBit
                            ? getAffineSymbolExpr(nSymbols++, context)
                            : getAffineConstantExpr(runningSize, context);
    expr = expr ? expr + dimExpr * stride : dimExpr * stride;
    if (size > 0) {
      runningSize *= size;
      assert(runningSize > 0 && "integer overflow in size computation");
    } else {
      dynamicPoisonBit = true;
    }
  }
  return simplifyAffineExpr(expr, numDims, nSymbols);
}

AffineExpr mlir::makeCanonicalStridedLayoutExpr(ArrayRef<int64_t> sizes,
                                                MLIRContext *context) {
  SmallVector<AffineExpr, 4> exprs;
  exprs.reserve(sizes.size());
  for (auto dim : llvm::seq<unsigned>(0, sizes.size()))
    exprs.push_back(getAffineDimExpr(dim, context));
  return makeCanonicalStridedLayoutExpr(sizes, exprs, context);
}

/// Return true if the layout for `t` is compatible with strided semantics.
bool mlir::isStrided(MemRefType t) {
  int64_t offset;
  SmallVector<int64_t, 4> strides;
  auto res = getStridesAndOffset(t, strides, offset);
  return succeeded(res);
}
