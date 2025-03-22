//===- BuiltinTypes.cpp - MLIR Builtin Type Classes -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinTypes.h"
#include "TypeDetail.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/TensorEncoding.h"
#include "mlir/IR/TypeUtilities.h"
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

namespace mlir {
#include "mlir/IR/BuiltinTypeConstraints.cpp.inc"
} // namespace mlir

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
// Float Types
//===----------------------------------------------------------------------===//

// Mapping from MLIR FloatType to APFloat semantics.
#define FLOAT_TYPE_SEMANTICS(TYPE, SEM)                                        \
  const llvm::fltSemantics &TYPE::getFloatSemantics() const {                  \
    return APFloat::SEM();                                                     \
  }
FLOAT_TYPE_SEMANTICS(Float4E2M1FNType, Float4E2M1FN)
FLOAT_TYPE_SEMANTICS(Float6E2M3FNType, Float6E2M3FN)
FLOAT_TYPE_SEMANTICS(Float6E3M2FNType, Float6E3M2FN)
FLOAT_TYPE_SEMANTICS(Float8E5M2Type, Float8E5M2)
FLOAT_TYPE_SEMANTICS(Float8E4M3Type, Float8E4M3)
FLOAT_TYPE_SEMANTICS(Float8E4M3FNType, Float8E4M3FN)
FLOAT_TYPE_SEMANTICS(Float8E5M2FNUZType, Float8E5M2FNUZ)
FLOAT_TYPE_SEMANTICS(Float8E4M3FNUZType, Float8E4M3FNUZ)
FLOAT_TYPE_SEMANTICS(Float8E4M3B11FNUZType, Float8E4M3B11FNUZ)
FLOAT_TYPE_SEMANTICS(Float8E3M4Type, Float8E3M4)
FLOAT_TYPE_SEMANTICS(Float8E8M0FNUType, Float8E8M0FNU)
FLOAT_TYPE_SEMANTICS(BFloat16Type, BFloat)
FLOAT_TYPE_SEMANTICS(Float16Type, IEEEhalf)
FLOAT_TYPE_SEMANTICS(FloatTF32Type, FloatTF32)
FLOAT_TYPE_SEMANTICS(Float32Type, IEEEsingle)
FLOAT_TYPE_SEMANTICS(Float64Type, IEEEdouble)
FLOAT_TYPE_SEMANTICS(Float80Type, x87DoubleExtended)
FLOAT_TYPE_SEMANTICS(Float128Type, IEEEquad)
#undef FLOAT_TYPE_SEMANTICS

FloatType Float16Type::scaleElementBitwidth(unsigned scale) const {
  if (scale == 2)
    return Float32Type::get(getContext());
  if (scale == 4)
    return Float64Type::get(getContext());
  return FloatType();
}

FloatType BFloat16Type::scaleElementBitwidth(unsigned scale) const {
  if (scale == 2)
    return Float32Type::get(getContext());
  if (scale == 4)
    return Float64Type::get(getContext());
  return FloatType();
}

FloatType Float32Type::scaleElementBitwidth(unsigned scale) const {
  if (scale == 2)
    return Float64Type::get(getContext());
  return FloatType();
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
  TypeRange newArgTypes =
      insertTypesInto(getInputs(), argIndices, argTypes, argStorage);
  TypeRange newResultTypes =
      insertTypesInto(getResults(), resultIndices, resultTypes, resultStorage);
  return clone(newArgTypes, newResultTypes);
}

/// Returns a new function type without the specified arguments and results.
FunctionType
FunctionType::getWithoutArgsAndResults(const BitVector &argIndices,
                                       const BitVector &resultIndices) {
  SmallVector<Type> argStorage, resultStorage;
  TypeRange newArgTypes = filterTypesOut(getInputs(), argIndices, argStorage);
  TypeRange newResultTypes =
      filterTypesOut(getResults(), resultIndices, resultStorage);
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

bool VectorType::isValidElementType(Type t) {
  return isValidVectorTypeElementType(t);
}

LogicalResult VectorType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 ArrayRef<int64_t> shape, Type elementType,
                                 ArrayRef<bool> scalableDims) {
  if (!isValidElementType(elementType))
    return emitError()
           << "vector elements must be int/index/float type but got "
           << elementType;

  if (any_of(shape, [](int64_t i) { return i <= 0; }))
    return emitError()
           << "vector types must have positive constant sizes but got "
           << shape;

  if (scalableDims.size() != shape.size())
    return emitError() << "number of dims must match, got "
                       << scalableDims.size() << " and " << shape.size();

  return success();
}

VectorType VectorType::scaleElementBitwidth(unsigned scale) {
  if (!scale)
    return VectorType();
  if (auto et = llvm::dyn_cast<IntegerType>(getElementType()))
    if (auto scaledEt = et.scaleElementBitwidth(scale))
      return VectorType::get(getShape(), scaledEt, getScalableDims());
  if (auto et = llvm::dyn_cast<FloatType>(getElementType()))
    if (auto scaledEt = et.scaleElementBitwidth(scale))
      return VectorType::get(getShape(), scaledEt, getScalableDims());
  return VectorType();
}

VectorType VectorType::cloneWith(std::optional<ArrayRef<int64_t>> shape,
                                 Type elementType) const {
  return VectorType::get(shape.value_or(getShape()), elementType,
                         getScalableDims());
}

//===----------------------------------------------------------------------===//
// TensorType
//===----------------------------------------------------------------------===//

Type TensorType::getElementType() const {
  return llvm::TypeSwitch<TensorType, Type>(*this)
      .Case<RankedTensorType, UnrankedTensorType>(
          [](auto type) { return type.getElementType(); });
}

bool TensorType::hasRank() const {
  return !llvm::isa<UnrankedTensorType>(*this);
}

ArrayRef<int64_t> TensorType::getShape() const {
  return llvm::cast<RankedTensorType>(*this).getShape();
}

TensorType TensorType::cloneWith(std::optional<ArrayRef<int64_t>> shape,
                                 Type elementType) const {
  if (llvm::dyn_cast<UnrankedTensorType>(*this)) {
    if (shape)
      return RankedTensorType::get(*shape, elementType);
    return UnrankedTensorType::get(elementType);
  }

  auto rankedTy = llvm::cast<RankedTensorType>(*this);
  if (!shape)
    return RankedTensorType::get(rankedTy.getShape(), elementType,
                                 rankedTy.getEncoding());
  return RankedTensorType::get(shape.value_or(rankedTy.getShape()), elementType,
                               rankedTy.getEncoding());
}

RankedTensorType TensorType::clone(::llvm::ArrayRef<int64_t> shape,
                                   Type elementType) const {
  return ::llvm::cast<RankedTensorType>(cloneWith(shape, elementType));
}

RankedTensorType TensorType::clone(::llvm::ArrayRef<int64_t> shape) const {
  return ::llvm::cast<RankedTensorType>(cloneWith(shape, getElementType()));
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
  return llvm::isa<ComplexType, FloatType, IntegerType, OpaqueType, VectorType,
                   IndexType>(type) ||
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
  if (auto v = llvm::dyn_cast_or_null<VerifiableTensorEncoding>(encoding))
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

bool BaseMemRefType::hasRank() const {
  return !llvm::isa<UnrankedMemRefType>(*this);
}

ArrayRef<int64_t> BaseMemRefType::getShape() const {
  return llvm::cast<MemRefType>(*this).getShape();
}

BaseMemRefType BaseMemRefType::cloneWith(std::optional<ArrayRef<int64_t>> shape,
                                         Type elementType) const {
  if (llvm::dyn_cast<UnrankedMemRefType>(*this)) {
    if (!shape)
      return UnrankedMemRefType::get(elementType, getMemorySpace());
    MemRefType::Builder builder(*shape, elementType);
    builder.setMemorySpace(getMemorySpace());
    return builder;
  }

  MemRefType::Builder builder(llvm::cast<MemRefType>(*this));
  if (shape)
    builder.setShape(*shape);
  builder.setElementType(elementType);
  return builder;
}

MemRefType BaseMemRefType::clone(::llvm::ArrayRef<int64_t> shape,
                                 Type elementType) const {
  return ::llvm::cast<MemRefType>(cloneWith(shape, elementType));
}

MemRefType BaseMemRefType::clone(::llvm::ArrayRef<int64_t> shape) const {
  return ::llvm::cast<MemRefType>(cloneWith(shape, getElementType()));
}

Attribute BaseMemRefType::getMemorySpace() const {
  if (auto rankedMemRefTy = llvm::dyn_cast<MemRefType>(*this))
    return rankedMemRefTy.getMemorySpace();
  return llvm::cast<UnrankedMemRefType>(*this).getMemorySpace();
}

unsigned BaseMemRefType::getMemorySpaceAsInt() const {
  if (auto rankedMemRefTy = llvm::dyn_cast<MemRefType>(*this))
    return rankedMemRefTy.getMemorySpaceAsInt();
  return llvm::cast<UnrankedMemRefType>(*this).getMemorySpaceAsInt();
}

//===----------------------------------------------------------------------===//
// MemRefType
//===----------------------------------------------------------------------===//

std::optional<llvm::SmallDenseSet<unsigned>>
mlir::computeRankReductionMask(ArrayRef<int64_t> originalShape,
                               ArrayRef<int64_t> reducedShape,
                               bool matchDynamic) {
  size_t originalRank = originalShape.size(), reducedRank = reducedShape.size();
  llvm::SmallDenseSet<unsigned> unusedDims;
  unsigned reducedIdx = 0;
  for (unsigned originalIdx = 0; originalIdx < originalRank; ++originalIdx) {
    // Greedily insert `originalIdx` if match.
    int64_t origSize = originalShape[originalIdx];
    // if `matchDynamic`, count dynamic dims as a match, unless `origSize` is 1.
    if (matchDynamic && reducedIdx < reducedRank && origSize != 1 &&
        (ShapedType::isDynamic(reducedShape[reducedIdx]) ||
         ShapedType::isDynamic(origSize))) {
      reducedIdx++;
      continue;
    }
    if (reducedIdx < reducedRank && origSize == reducedShape[reducedIdx]) {
      reducedIdx++;
      continue;
    }

    unusedDims.insert(originalIdx);
    // If no match on `originalIdx`, the `originalShape` at this dimension
    // must be 1, otherwise we bail.
    if (origSize != 1)
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

  ShapedType originalShapedType = llvm::cast<ShapedType>(originalType);
  ShapedType candidateReducedShapedType =
      llvm::cast<ShapedType>(candidateReducedType);

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
  if (llvm::isa<IntegerAttr, StringAttr, DictionaryAttr>(memorySpace))
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
  IntegerAttr intMemorySpace = llvm::dyn_cast_or_null<IntegerAttr>(memorySpace);
  if (intMemorySpace && intMemorySpace.getValue() == 0)
    return nullptr;

  return memorySpace;
}

unsigned mlir::detail::getMemorySpaceAsInt(Attribute memorySpace) {
  if (!memorySpace)
    return 0;

  assert(llvm::isa<IntegerAttr>(memorySpace) &&
         "Using `getMemorySpaceInteger` with non-Integer attribute");

  return static_cast<unsigned>(llvm::cast<IntegerAttr>(memorySpace).getInt());
}

unsigned MemRefType::getMemorySpaceAsInt() const {
  return detail::getMemorySpaceAsInt(getMemorySpace());
}

MemRefType MemRefType::get(ArrayRef<int64_t> shape, Type elementType,
                           MemRefLayoutAttrInterface layout,
                           Attribute memorySpace) {
  // Use default layout for empty attribute.
  if (!layout)
    layout = ContiguousLayoutAttr::get(elementType.getContext(), /*offset=*/0,
                                       /*rank=*/shape.size());

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
    layout = ContiguousLayoutAttr::get(elementType.getContext(), /*offset=*/0,
                                       /*rank=*/shape.size());

  // Drop default memory space value and replace it with empty attribute.
  memorySpace = skipDefaultMemorySpace(memorySpace);

  return Base::getChecked(emitErrorFn, elementType.getContext(), shape,
                          elementType, layout, memorySpace);
}

MemRefType MemRefType::get(ArrayRef<int64_t> shape, Type elementType,
                           AffineMap map, Attribute memorySpace) {

  MemRefLayoutAttrInterface layout;
  if (map)
    // Wrap AffineMap into Attribute.
    layout = AffineMapAttr::get(map);
  else
    // Represent the default identity map as a contiguous layout.
    layout = ContiguousLayoutAttr::get(elementType.getContext(), /*offset=*/0,
                                       /*rank=*/shape.size());

  // Drop default memory space value and replace it with empty attribute.
  memorySpace = skipDefaultMemorySpace(memorySpace);

  return Base::get(elementType.getContext(), shape, elementType, layout,
                   memorySpace);
}

MemRefType
MemRefType::getChecked(function_ref<InFlightDiagnostic()> emitErrorFn,
                       ArrayRef<int64_t> shape, Type elementType, AffineMap map,
                       Attribute memorySpace) {
  MemRefLayoutAttrInterface layout;
  if (map)
    // Wrap AffineMap into Attribute.
    layout = AffineMapAttr::get(map);
  else
    // Represent the default identity map as a contiguous layout.
    layout = ContiguousLayoutAttr::get(elementType.getContext(), /*offset=*/0,
                                       /*rank=*/shape.size());

  // Drop default memory space value and replace it with empty attribute.
  memorySpace = skipDefaultMemorySpace(memorySpace);

  return Base::getChecked(emitErrorFn, elementType.getContext(), shape,
                          elementType, layout, memorySpace);
}

MemRefType MemRefType::get(ArrayRef<int64_t> shape, Type elementType,
                           AffineMap map, unsigned memorySpaceInd) {

  MemRefLayoutAttrInterface layout;
  if (map)
    // Wrap AffineMap into Attribute.
    layout = AffineMapAttr::get(map);
  else
    // Represent the default identity map as a contiguous layout.
    layout = ContiguousLayoutAttr::get(elementType.getContext(), /*offset=*/0,
                                       /*rank=*/shape.size());

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
  MemRefLayoutAttrInterface layout;
  if (map)
    // Wrap AffineMap into Attribute.
    layout = AffineMapAttr::get(map);
  else
    // Represent the default identity map as a contiguous layout.
    layout = ContiguousLayoutAttr::get(elementType.getContext(), /*offset=*/0,
                                       /*rank=*/shape.size());

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

bool MemRefType::areTrailingDimsContiguous(int64_t n) {
  if (!isLastDimUnitStride())
    return false;

  if (auto contiguousLayout =
          mlir::dyn_cast<ContiguousLayoutAttr>(getLayout())) {
    ArrayRef<int64_t> perm = contiguousLayout.getPermutation();
    int64_t expectedValue = perm.size() - 1;
    for (auto [iter, permVal] : llvm::enumerate(llvm::reverse(perm))) {
      if (static_cast<int64_t>(iter) >= n)
        return true;
      if (permVal != expectedValue)
        return false;
      expectedValue--;
    }
    return true;
  }
  auto memrefShape = getShape().take_back(n);

  if (getLayout().isIdentity())
    return true;

  if (ShapedType::isDynamicShape(memrefShape))
    return false;

  int64_t offset;
  SmallVector<int64_t> stridesFull;
  if (!succeeded(getStridesAndOffset(stridesFull, offset)))
    return false;
  auto strides = ArrayRef<int64_t>(stridesFull).take_back(n);

  if (strides.empty())
    return true;

  // Check whether strides match "flattened" dims.
  SmallVector<int64_t> flattenedDims;
  auto dimProduct = 1;
  for (auto dim : llvm::reverse(memrefShape.drop_front(1))) {
    dimProduct *= dim;
    flattenedDims.push_back(dimProduct);
  }

  strides = strides.drop_back(1);
  return llvm::equal(strides, llvm::reverse(flattenedDims));
}

/// If `layout` is some permutation of the identity layout with an offset
/// applied to the last dimension - that is, if it has the form (d0, d1, ...,
/// dN) -> (dX, dY, ... dZ + E) for some symbol or constant E, succeed and
/// populate `perm` and `offset` with the discovered values.
static LogicalResult asOffsetPermutation(MemRefLayoutAttrInterface layout,
                                         ArrayRef<int64_t> shape,
                                         SmallVectorImpl<int64_t> &perm,
                                         int64_t &offset) {
  if (auto contiguousLayout = mlir::dyn_cast<ContiguousLayoutAttr>(layout)) {
    perm.assign(contiguousLayout.getPermutation().begin(),
                contiguousLayout.getPermutation().end());
    offset = contiguousLayout.getOffset();
    return success();
  }
  if (auto stridedLayout = mlir::dyn_cast<StridedLayoutAttr>(layout)) {
    // We can't reason about dynamic strides
    if (llvm::any_of(stridedLayout.getStrides(), ShapedType::isDynamic))
      return failure();

    int64_t suffixProd = 1;
    bool isRowMajor = true;
    for (auto [stride, dim] : llvm::zip(
             llvm::reverse(stridedLayout.getStrides()), llvm::reverse(shape))) {
      if (stride != suffixProd) {
        isRowMajor = false;
        break;
      }
      suffixProd *= dim;
    }
    if (isRowMajor) {
      llvm::append_range(perm, llvm::iota_range<int64_t>(0, shape.size(),
                                                         /*Inclusive=*/false));
      offset = stridedLayout.getOffset();
      return success();
    }

    SmallVector<std::pair<int64_t, int64_t>> stridesAndLocs;
    for (auto [idx, stride] : llvm::enumerate(stridedLayout.getStrides()))
      stridesAndLocs.emplace_back(stride, static_cast<int64_t>(idx));
    // Sort by increasing stride, ties broken by appearing later in the memref.
    llvm::sort(stridesAndLocs, [](auto a, auto b) {
      if (a.first == b.first)
        return a.second >= b.second;
      return a.first < b.first;
    });
    int64_t expectedStride = 1;
    for (auto [stride, loc] : stridesAndLocs) {
      if (stride != expectedStride)
        return failure();
      expectedStride *= shape[loc];
    }
    perm = llvm::map_to_vector(stridesAndLocs, [](auto x) { return x.second; });
    offset = stridedLayout.getOffset();
    return success();
  }

  auto pullOffset = [&](AffineExpr e) -> bool {
    if (isa<AffineDimExpr>(e))
      return false;
    if (auto constExpr = mlir::dyn_cast<AffineConstantExpr>(e)) {
      offset = constExpr.getValue();
    } else {
      offset = ShapedType::kDynamic;
    }
    return true;
  };

  AffineMap m = layout.getAffineMap();
  if (m.getNumDims() == 0 && m.getNumResults() == 1) {
    if (pullOffset(m.getResult(0)))
      return success();
    return failure();
  }

  int64_t rank = shape.size();
  if (m.getNumResults() != rank || m.getNumDims() != rank)
    return failure();

  llvm::SmallBitVector seen(rank, false);
  for (AffineExpr e : llvm::drop_end(m.getResults())) {
    auto dimE = dyn_cast<AffineDimExpr>(e);
    if (!dimE)
      return failure();
    seen.set(dimE.getPosition());
    perm.push_back(dimE.getPosition());
  }
  AffineDimExpr lastDim = dyn_cast<AffineDimExpr>(m.getResults().back());
  if (!lastDim) {
    auto sum = dyn_cast<AffineBinaryOpExpr>(m.getResults().back());
    if (!sum || sum.getKind() != AffineExprKind::Add)
      return failure();
    if (!(pullOffset(sum.getLHS()) &&
          (lastDim = dyn_cast<AffineDimExpr>(sum.getRHS()))) &&
        !(pullOffset(sum.getRHS()) &&
          (lastDim = dyn_cast<AffineDimExpr>(sum.getLHS()))))
      return failure();
  } else {
    offset = 0;
  }
  seen.set(lastDim.getPosition());
  perm.push_back(lastDim.getPosition());
  if (!seen.all())
    return failure();
  return success();
}

static SmallVector<int64_t>
computeStridesFromPermutedShape(ArrayRef<int64_t> shape,
                                ArrayRef<int64_t> perm) {
  assert(shape.size() == perm.size() &&
         "shape and permutation have same length");
  int64_t rank = shape.size();
  SmallVector<int64_t> strides(rank, ShapedType::kDynamic);
  strides.reserve(rank);

  // invertPermutationVector() might be a circular dependency betwin IR and
  // Utils.
  SmallVector<int64_t> strideOrder(rank, -1);
  for (auto [idx, dim] : llvm::enumerate(perm)) {
    strideOrder[dim] = static_cast<int64_t>(idx);
  }
  SaturatedInteger strideAccum = SaturatedInteger::wrap(1);
  for (int64_t i = rank - 1; i >= 0; --i) {
    strides[strideOrder[i]] = strideAccum.asInteger();
    strideAccum = strideAccum * SaturatedInteger::wrap(shape[strideOrder[i]]);
  }
  return strides;
}

MemRefType MemRefType::canonicalizeStridedLayout() {
  MemRefLayoutAttrInterface layout = getLayout();
  if (mlir::isa<ContiguousLayoutAttr>(layout))
    return *this;

  SmallVector<int64_t> maybePerm;
  int64_t maybeOffset;
  if (succeeded(
          asOffsetPermutation(layout, getShape(), maybePerm, maybeOffset))) {
    return MemRefType::Builder(*this).setLayout(
        ContiguousLayoutAttr::get(getContext(), maybeOffset, maybePerm));
  }

  AffineMap m = getLayout().getAffineMap();

  // Identity maps that aren't contiguous<> aren't canonical
  if (m.isIdentity())
    return MemRefType::Builder(*this).setLayout({});

  // Can't reduce to canonical identity form, return in canonical form.
  if (m.getNumResults() > 1)
    return *this;

  // If the canonical strided layout for the sizes of `t` is equal to the
  // simplified layout of `t` we can just return an empty layout. Otherwise,
  // just simplify the existing layout.
  AffineExpr expr = makeCanonicalStridedLayoutExpr(getShape(), getContext());
  auto simplifiedLayoutExpr =
      simplifyAffineExpr(m.getResult(0), m.getNumDims(), m.getNumSymbols());
  if (expr != simplifiedLayoutExpr)
    return MemRefType::Builder(*this).setLayout(
        AffineMapAttr::get(AffineMap::get(m.getNumDims(), m.getNumSymbols(),
                                          simplifiedLayoutExpr)));
  return MemRefType::Builder(*this).setLayout({});
}

// Fallback cases for terminal dim/sym/cst that are not part of a binary op (
// i.e. single term). Accumulate the AffineExpr into the existing one.
static void extractStridesFromTerm(AffineExpr e,
                                   AffineExpr multiplicativeFactor,
                                   MutableArrayRef<AffineExpr> strides,
                                   AffineExpr &offset) {
  if (auto dim = dyn_cast<AffineDimExpr>(e))
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
  auto bin = dyn_cast<AffineBinaryOpExpr>(e);
  if (!bin) {
    extractStridesFromTerm(e, multiplicativeFactor, strides, offset);
    return success();
  }

  if (bin.getKind() == AffineExprKind::CeilDiv ||
      bin.getKind() == AffineExprKind::FloorDiv ||
      bin.getKind() == AffineExprKind::Mod)
    return failure();

  if (bin.getKind() == AffineExprKind::Mul) {
    auto dim = dyn_cast<AffineDimExpr>(bin.getLHS());
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

  if (m.getNumResults() != 1)
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

  return success();
}

LogicalResult MemRefType::getStridesAndOffset(SmallVectorImpl<int64_t> &strides,
                                              int64_t &offset) {
  // Happy path: the type uses the strided layout directly.
  if (auto strided = llvm::dyn_cast<StridedLayoutAttr>(getLayout())) {
    llvm::append_range(strides, strided.getStrides());
    offset = strided.getOffset();
    return success();
  }

  // Somewhat happy path: the type uses the contiguous layout, we need to
  // compute the strides.
  if (auto contiguous = llvm::dyn_cast<ContiguousLayoutAttr>(getLayout())) {
    strides.append(computeStridesFromPermutedShape(
        getShape(), contiguous.getPermutation()));
    offset = contiguous.getOffset();
    return success();
  }

  SmallVector<int64_t> maybePermutation;
  int64_t maybeOffset;
  if (succeeded(asOffsetPermutation(getLayout(), getShape(), maybePermutation,
                                    maybeOffset))) {
    strides.append(
        computeStridesFromPermutedShape(getShape(), maybePermutation));
    offset = maybeOffset;
    return success();
  }

  // Otherwise, defer to the affine fallback as layouts are supposed to be
  // convertible to affine maps.
  AffineExpr offsetExpr;
  SmallVector<AffineExpr, 4> strideExprs;
  if (failed(::getStridesAndOffset(*this, strideExprs, offsetExpr)))
    return failure();
  if (auto cst = llvm::dyn_cast<AffineConstantExpr>(offsetExpr))
    offset = cst.getValue();
  else
    offset = ShapedType::kDynamic;
  for (auto e : strideExprs) {
    if (auto c = llvm::dyn_cast<AffineConstantExpr>(e))
      strides.push_back(c.getValue());
    else
      strides.push_back(ShapedType::kDynamic);
  }
  return success();
}

std::pair<SmallVector<int64_t>, int64_t> MemRefType::getStridesAndOffset() {
  SmallVector<int64_t> strides;
  int64_t offset;
  LogicalResult status = getStridesAndOffset(strides, offset);
  (void)status;
  assert(succeeded(status) && "Invalid use of check-free getStridesAndOffset");
  return {strides, offset};
}

bool MemRefType::isStrided() {
  int64_t offset;
  SmallVector<int64_t, 4> strides;
  auto res = getStridesAndOffset(strides, offset);
  return succeeded(res);
}

bool MemRefType::isLastDimUnitStride() {
  if (auto contiguousLayout = mlir::dyn_cast<ContiguousLayoutAttr>(getLayout()))
    return getRank() == 0 ||
           contiguousLayout.getPermutation().back() == getRank() - 1;

  int64_t offset;
  SmallVector<int64_t> strides;
  auto successStrides = getStridesAndOffset(strides, offset);
  return succeeded(successStrides) && (strides.empty() || strides.back() == 1);
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
    if (auto nestedTuple = llvm::dyn_cast<TupleType>(type))
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

AffineExpr mlir::makeCanonicalStridedLayoutExpr(ArrayRef<int64_t> sizes,
                                                ArrayRef<AffineExpr> exprs,
                                                MLIRContext *context) {
  // Size 0 corner case is useful for canonicalizations.
  if (sizes.empty())
    return getAffineConstantExpr(0, context);

  assert(!exprs.empty() && "expected exprs");
  auto maps = AffineMap::inferFromExprList(exprs, context);
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
