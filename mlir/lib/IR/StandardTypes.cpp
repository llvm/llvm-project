//===- StandardTypes.cpp - MLIR Standard Type Classes ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/StandardTypes.h"
#include "TypeDetail.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/Twine.h"

using namespace mlir;
using namespace mlir::detail;

//===----------------------------------------------------------------------===//
// Type
//===----------------------------------------------------------------------===//

bool Type::isBF16() { return getKind() == StandardTypes::BF16; }
bool Type::isF16() { return getKind() == StandardTypes::F16; }
bool Type::isF32() { return getKind() == StandardTypes::F32; }
bool Type::isF64() { return getKind() == StandardTypes::F64; }

bool Type::isIndex() { return isa<IndexType>(); }

/// Return true if this is an integer type with the specified width.
bool Type::isInteger(unsigned width) {
  if (auto intTy = dyn_cast<IntegerType>())
    return intTy.getWidth() == width;
  return false;
}

bool Type::isSignlessInteger() {
  if (auto intTy = dyn_cast<IntegerType>())
    return intTy.isSignless();
  return false;
}

bool Type::isSignlessInteger(unsigned width) {
  if (auto intTy = dyn_cast<IntegerType>())
    return intTy.isSignless() && intTy.getWidth() == width;
  return false;
}

bool Type::isSignedInteger() {
  if (auto intTy = dyn_cast<IntegerType>())
    return intTy.isSigned();
  return false;
}

bool Type::isSignedInteger(unsigned width) {
  if (auto intTy = dyn_cast<IntegerType>())
    return intTy.isSigned() && intTy.getWidth() == width;
  return false;
}

bool Type::isUnsignedInteger() {
  if (auto intTy = dyn_cast<IntegerType>())
    return intTy.isUnsigned();
  return false;
}

bool Type::isUnsignedInteger(unsigned width) {
  if (auto intTy = dyn_cast<IntegerType>())
    return intTy.isUnsigned() && intTy.getWidth() == width;
  return false;
}

bool Type::isSignlessIntOrIndex() {
  return isSignlessInteger() || isa<IndexType>();
}

bool Type::isSignlessIntOrIndexOrFloat() {
  return isSignlessInteger() || isa<IndexType, FloatType>();
}

bool Type::isSignlessIntOrFloat() {
  return isSignlessInteger() || isa<FloatType>();
}

bool Type::isIntOrIndex() { return isa<IntegerType>() || isIndex(); }

bool Type::isIntOrFloat() { return isa<IntegerType, FloatType>(); }

bool Type::isIntOrIndexOrFloat() { return isIntOrFloat() || isIndex(); }

//===----------------------------------------------------------------------===//
/// ComplexType
//===----------------------------------------------------------------------===//

ComplexType ComplexType::get(Type elementType) {
  return Base::get(elementType.getContext(), StandardTypes::Complex,
                   elementType);
}

ComplexType ComplexType::getChecked(Type elementType, Location location) {
  return Base::getChecked(location, StandardTypes::Complex, elementType);
}

/// Verify the construction of an integer type.
LogicalResult ComplexType::verifyConstructionInvariants(Location loc,
                                                        Type elementType) {
  if (!elementType.isIntOrFloat())
    return emitError(loc, "invalid element type for complex");
  return success();
}

Type ComplexType::getElementType() { return getImpl()->elementType; }

//===----------------------------------------------------------------------===//
// Integer Type
//===----------------------------------------------------------------------===//

// static constexpr must have a definition (until in C++17 and inline variable).
constexpr unsigned IntegerType::kMaxWidth;

/// Verify the construction of an integer type.
LogicalResult
IntegerType::verifyConstructionInvariants(Location loc, unsigned width,
                                          SignednessSemantics signedness) {
  if (width > IntegerType::kMaxWidth) {
    return emitError(loc) << "integer bitwidth is limited to "
                          << IntegerType::kMaxWidth << " bits";
  }
  return success();
}

unsigned IntegerType::getWidth() const { return getImpl()->getWidth(); }

IntegerType::SignednessSemantics IntegerType::getSignedness() const {
  return getImpl()->getSignedness();
}

//===----------------------------------------------------------------------===//
// Float Type
//===----------------------------------------------------------------------===//

unsigned FloatType::getWidth() {
  switch (getKind()) {
  case StandardTypes::BF16:
  case StandardTypes::F16:
    return 16;
  case StandardTypes::F32:
    return 32;
  case StandardTypes::F64:
    return 64;
  default:
    llvm_unreachable("unexpected type");
  }
}

/// Returns the floating semantics for the given type.
const llvm::fltSemantics &FloatType::getFloatSemantics() {
  if (isBF16())
    return APFloat::BFloat();
  if (isF16())
    return APFloat::IEEEhalf();
  if (isF32())
    return APFloat::IEEEsingle();
  if (isF64())
    return APFloat::IEEEdouble();
  llvm_unreachable("non-floating point type used");
}

unsigned Type::getIntOrFloatBitWidth() {
  assert(isIntOrFloat() && "only integers and floats have a bitwidth");
  if (auto intType = dyn_cast<IntegerType>())
    return intType.getWidth();
  return cast<FloatType>().getWidth();
}

//===----------------------------------------------------------------------===//
// ShapedType
//===----------------------------------------------------------------------===//
constexpr int64_t ShapedType::kDynamicSize;
constexpr int64_t ShapedType::kDynamicStrideOrOffset;

Type ShapedType::getElementType() const {
  return static_cast<ImplType *>(impl)->elementType;
}

unsigned ShapedType::getElementTypeBitWidth() const {
  return getElementType().getIntOrFloatBitWidth();
}

int64_t ShapedType::getNumElements() const {
  assert(hasStaticShape() && "cannot get element count of dynamic shaped type");
  auto shape = getShape();
  int64_t num = 1;
  for (auto dim : shape)
    num *= dim;
  return num;
}

int64_t ShapedType::getRank() const { return getShape().size(); }

bool ShapedType::hasRank() const {
  return !isa<UnrankedMemRefType, UnrankedTensorType>();
}

int64_t ShapedType::getDimSize(unsigned idx) const {
  assert(idx < getRank() && "invalid index for shaped type");
  return getShape()[idx];
}

bool ShapedType::isDynamicDim(unsigned idx) const {
  assert(idx < getRank() && "invalid index for shaped type");
  return isDynamic(getShape()[idx]);
}

unsigned ShapedType::getDynamicDimIndex(unsigned index) const {
  assert(index < getRank() && "invalid index");
  assert(ShapedType::isDynamic(getDimSize(index)) && "invalid index");
  return llvm::count_if(getShape().take_front(index), ShapedType::isDynamic);
}

/// Get the number of bits require to store a value of the given shaped type.
/// Compute the value recursively since tensors are allowed to have vectors as
/// elements.
int64_t ShapedType::getSizeInBits() const {
  assert(hasStaticShape() &&
         "cannot get the bit size of an aggregate with a dynamic shape");

  auto elementType = getElementType();
  if (elementType.isIntOrFloat())
    return elementType.getIntOrFloatBitWidth() * getNumElements();

  // Tensors can have vectors and other tensors as elements, other shaped types
  // cannot.
  assert(isa<TensorType>() && "unsupported element type");
  assert((elementType.isa<VectorType, TensorType>()) &&
         "unsupported tensor element type");
  return getNumElements() * elementType.cast<ShapedType>().getSizeInBits();
}

ArrayRef<int64_t> ShapedType::getShape() const {
  switch (getKind()) {
  case StandardTypes::Vector:
    return cast<VectorType>().getShape();
  case StandardTypes::RankedTensor:
    return cast<RankedTensorType>().getShape();
  case StandardTypes::MemRef:
    return cast<MemRefType>().getShape();
  default:
    llvm_unreachable("not a ShapedType or not ranked");
  }
}

int64_t ShapedType::getNumDynamicDims() const {
  return llvm::count_if(getShape(), isDynamic);
}

bool ShapedType::hasStaticShape() const {
  return hasRank() && llvm::none_of(getShape(), isDynamic);
}

bool ShapedType::hasStaticShape(ArrayRef<int64_t> shape) const {
  return hasStaticShape() && getShape() == shape;
}

//===----------------------------------------------------------------------===//
// VectorType
//===----------------------------------------------------------------------===//

VectorType VectorType::get(ArrayRef<int64_t> shape, Type elementType) {
  return Base::get(elementType.getContext(), StandardTypes::Vector, shape,
                   elementType);
}

VectorType VectorType::getChecked(ArrayRef<int64_t> shape, Type elementType,
                                  Location location) {
  return Base::getChecked(location, StandardTypes::Vector, shape, elementType);
}

LogicalResult VectorType::verifyConstructionInvariants(Location loc,
                                                       ArrayRef<int64_t> shape,
                                                       Type elementType) {
  if (shape.empty())
    return emitError(loc, "vector types must have at least one dimension");

  if (!isValidElementType(elementType))
    return emitError(loc, "vector elements must be int or float type");

  if (any_of(shape, [](int64_t i) { return i <= 0; }))
    return emitError(loc, "vector types must have positive constant sizes");

  return success();
}

ArrayRef<int64_t> VectorType::getShape() const { return getImpl()->getShape(); }

//===----------------------------------------------------------------------===//
// TensorType
//===----------------------------------------------------------------------===//

// Check if "elementType" can be an element type of a tensor. Emit errors if
// location is not nullptr.  Returns failure if check failed.
static inline LogicalResult checkTensorElementType(Location location,
                                                   Type elementType) {
  if (!TensorType::isValidElementType(elementType))
    return emitError(location, "invalid tensor element type");
  return success();
}

//===----------------------------------------------------------------------===//
// RankedTensorType
//===----------------------------------------------------------------------===//

RankedTensorType RankedTensorType::get(ArrayRef<int64_t> shape,
                                       Type elementType) {
  return Base::get(elementType.getContext(), StandardTypes::RankedTensor, shape,
                   elementType);
}

RankedTensorType RankedTensorType::getChecked(ArrayRef<int64_t> shape,
                                              Type elementType,
                                              Location location) {
  return Base::getChecked(location, StandardTypes::RankedTensor, shape,
                          elementType);
}

LogicalResult RankedTensorType::verifyConstructionInvariants(
    Location loc, ArrayRef<int64_t> shape, Type elementType) {
  for (int64_t s : shape) {
    if (s < -1)
      return emitError(loc, "invalid tensor dimension size");
  }
  return checkTensorElementType(loc, elementType);
}

ArrayRef<int64_t> RankedTensorType::getShape() const {
  return getImpl()->getShape();
}

//===----------------------------------------------------------------------===//
// UnrankedTensorType
//===----------------------------------------------------------------------===//

UnrankedTensorType UnrankedTensorType::get(Type elementType) {
  return Base::get(elementType.getContext(), StandardTypes::UnrankedTensor,
                   elementType);
}

UnrankedTensorType UnrankedTensorType::getChecked(Type elementType,
                                                  Location location) {
  return Base::getChecked(location, StandardTypes::UnrankedTensor, elementType);
}

LogicalResult
UnrankedTensorType::verifyConstructionInvariants(Location loc,
                                                 Type elementType) {
  return checkTensorElementType(loc, elementType);
}

//===----------------------------------------------------------------------===//
// MemRefType
//===----------------------------------------------------------------------===//

/// Get or create a new MemRefType based on shape, element type, affine
/// map composition, and memory space.  Assumes the arguments define a
/// well-formed MemRef type.  Use getChecked to gracefully handle MemRefType
/// construction failures.
MemRefType MemRefType::get(ArrayRef<int64_t> shape, Type elementType,
                           ArrayRef<AffineMap> affineMapComposition,
                           unsigned memorySpace) {
  auto result = getImpl(shape, elementType, affineMapComposition, memorySpace,
                        /*location=*/llvm::None);
  assert(result && "Failed to construct instance of MemRefType.");
  return result;
}

/// Get or create a new MemRefType based on shape, element type, affine
/// map composition, and memory space declared at the given location.
/// If the location is unknown, the last argument should be an instance of
/// UnknownLoc.  If the MemRefType defined by the arguments would be
/// ill-formed, emits errors (to the handler registered with the context or to
/// the error stream) and returns nullptr.
MemRefType MemRefType::getChecked(ArrayRef<int64_t> shape, Type elementType,
                                  ArrayRef<AffineMap> affineMapComposition,
                                  unsigned memorySpace, Location location) {
  return getImpl(shape, elementType, affineMapComposition, memorySpace,
                 location);
}

/// Get or create a new MemRefType defined by the arguments.  If the resulting
/// type would be ill-formed, return nullptr.  If the location is provided,
/// emit detailed error messages.  To emit errors when the location is unknown,
/// pass in an instance of UnknownLoc.
MemRefType MemRefType::getImpl(ArrayRef<int64_t> shape, Type elementType,
                               ArrayRef<AffineMap> affineMapComposition,
                               unsigned memorySpace,
                               Optional<Location> location) {
  auto *context = elementType.getContext();

  // Check that memref is formed from allowed types.
  if (!elementType.isIntOrFloat() &&
      !elementType.isa<VectorType, ComplexType>())
    return emitOptionalError(location, "invalid memref element type"),
           MemRefType();

  for (int64_t s : shape) {
    // Negative sizes are not allowed except for `-1` that means dynamic size.
    if (s < -1)
      return emitOptionalError(location, "invalid memref size"), MemRefType();
  }

  // Check that the structure of the composition is valid, i.e. that each
  // subsequent affine map has as many inputs as the previous map has results.
  // Take the dimensionality of the MemRef for the first map.
  auto dim = shape.size();
  unsigned i = 0;
  for (const auto &affineMap : affineMapComposition) {
    if (affineMap.getNumDims() != dim) {
      if (location)
        emitError(*location)
            << "memref affine map dimension mismatch between "
            << (i == 0 ? Twine("memref rank") : "affine map " + Twine(i))
            << " and affine map" << i + 1 << ": " << dim
            << " != " << affineMap.getNumDims();
      return nullptr;
    }

    dim = affineMap.getNumResults();
    ++i;
  }

  // Drop identity maps from the composition.
  // This may lead to the composition becoming empty, which is interpreted as an
  // implicit identity.
  SmallVector<AffineMap, 2> cleanedAffineMapComposition;
  for (const auto &map : affineMapComposition) {
    if (map.isIdentity())
      continue;
    cleanedAffineMapComposition.push_back(map);
  }

  return Base::get(context, StandardTypes::MemRef, shape, elementType,
                   cleanedAffineMapComposition, memorySpace);
}

ArrayRef<int64_t> MemRefType::getShape() const { return getImpl()->getShape(); }

ArrayRef<AffineMap> MemRefType::getAffineMaps() const {
  return getImpl()->getAffineMaps();
}

unsigned MemRefType::getMemorySpace() const { return getImpl()->memorySpace; }

//===----------------------------------------------------------------------===//
// UnrankedMemRefType
//===----------------------------------------------------------------------===//

UnrankedMemRefType UnrankedMemRefType::get(Type elementType,
                                           unsigned memorySpace) {
  return Base::get(elementType.getContext(), StandardTypes::UnrankedMemRef,
                   elementType, memorySpace);
}

UnrankedMemRefType UnrankedMemRefType::getChecked(Type elementType,
                                                  unsigned memorySpace,
                                                  Location location) {
  return Base::getChecked(location, StandardTypes::UnrankedMemRef, elementType,
                          memorySpace);
}

unsigned UnrankedMemRefType::getMemorySpace() const {
  return getImpl()->memorySpace;
}

LogicalResult
UnrankedMemRefType::verifyConstructionInvariants(Location loc, Type elementType,
                                                 unsigned memorySpace) {
  // Check that memref is formed from allowed types.
  if (!elementType.isIntOrFloat() &&
      !elementType.isa<VectorType, ComplexType>())
    return emitError(loc, "invalid memref element type");
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

LogicalResult mlir::getStridesAndOffset(MemRefType t,
                                        SmallVectorImpl<AffineExpr> &strides,
                                        AffineExpr &offset) {
  auto affineMaps = t.getAffineMaps();
  // For now strides are only computed on a single affine map with a single
  // result (i.e. the closed subset of linearization maps that are compatible
  // with striding semantics).
  // TODO: support more forms on a per-need basis.
  if (affineMaps.size() > 1)
    return failure();
  if (affineMaps.size() == 1 && affineMaps[0].getNumResults() != 1)
    return failure();

  auto zero = getAffineConstantExpr(0, t.getContext());
  auto one = getAffineConstantExpr(1, t.getContext());
  offset = zero;
  strides.assign(t.getRank(), zero);

  AffineMap m;
  if (!affineMaps.empty()) {
    m = affineMaps.front();
    assert(!m.isIdentity() && "unexpected identity map");
  }

  // Canonical case for empty map.
  if (!m) {
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

  /// In practice, a strided memref must be internally non-aliasing. Test
  /// against 0 as a proxy.
  /// TODO: static cases can have more advanced checks.
  /// TODO: dynamic cases would require a way to compare symbolic
  /// expressions and would probably need an affine set context propagated
  /// everywhere.
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
  AffineExpr offsetExpr;
  SmallVector<AffineExpr, 4> strideExprs;
  if (failed(::getStridesAndOffset(t, strideExprs, offsetExpr)))
    return failure();
  if (auto cst = offsetExpr.dyn_cast<AffineConstantExpr>())
    offset = cst.getValue();
  else
    offset = ShapedType::kDynamicStrideOrOffset;
  for (auto e : strideExprs) {
    if (auto c = e.dyn_cast<AffineConstantExpr>())
      strides.push_back(c.getValue());
    else
      strides.push_back(ShapedType::kDynamicStrideOrOffset);
  }
  return success();
}

//===----------------------------------------------------------------------===//
/// TupleType
//===----------------------------------------------------------------------===//

/// Get or create a new TupleType with the provided element types. Assumes the
/// arguments define a well-formed type.
TupleType TupleType::get(ArrayRef<Type> elementTypes, MLIRContext *context) {
  return Base::get(context, StandardTypes::Tuple, elementTypes);
}

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

AffineMap mlir::makeStridedLinearLayoutMap(ArrayRef<int64_t> strides,
                                           int64_t offset,
                                           MLIRContext *context) {
  AffineExpr expr;
  unsigned nSymbols = 0;

  // AffineExpr for offset.
  // Static case.
  if (offset != MemRefType::getDynamicStrideOrOffset()) {
    auto cst = getAffineConstantExpr(offset, context);
    expr = cst;
  } else {
    // Dynamic case, new symbol for the offset.
    auto sym = getAffineSymbolExpr(nSymbols++, context);
    expr = sym;
  }

  // AffineExpr for strides.
  for (auto en : llvm::enumerate(strides)) {
    auto dim = en.index();
    auto stride = en.value();
    assert(stride != 0 && "Invalid stride specification");
    auto d = getAffineDimExpr(dim, context);
    AffineExpr mult;
    // Static case.
    if (stride != MemRefType::getDynamicStrideOrOffset())
      mult = getAffineConstantExpr(stride, context);
    else
      // Dynamic case, new symbol for each new stride.
      mult = getAffineSymbolExpr(nSymbols++, context);
    expr = expr + d * mult;
  }

  return AffineMap::get(strides.size(), nSymbols, expr);
}

/// Return a version of `t` with identity layout if it can be determined
/// statically that the layout is the canonical contiguous strided layout.
/// Otherwise pass `t`'s layout into `simplifyAffineMap` and return a copy of
/// `t` with simplified layout.
/// If `t` has multiple layout maps or a multi-result layout, just return `t`.
MemRefType mlir::canonicalizeStridedLayout(MemRefType t) {
  auto affineMaps = t.getAffineMaps();
  // Already in canonical form.
  if (affineMaps.empty())
    return t;

  // Can't reduce to canonical identity form, return in canonical form.
  if (affineMaps.size() > 1 || affineMaps[0].getNumResults() > 1)
    return t;

  // If the canonical strided layout for the sizes of `t` is equal to the
  // simplified layout of `t` we can just return an empty layout. Otherwise,
  // just simplify the existing layout.
  AffineExpr expr =
      makeCanonicalStridedLayoutExpr(t.getShape(), t.getContext());
  auto m = affineMaps[0];
  auto simplifiedLayoutExpr =
      simplifyAffineExpr(m.getResult(0), m.getNumDims(), m.getNumSymbols());
  if (expr != simplifiedLayoutExpr)
    return MemRefType::Builder(t).setAffineMaps({AffineMap::get(
        m.getNumDims(), m.getNumSymbols(), simplifiedLayoutExpr)});
  return MemRefType::Builder(t).setAffineMaps({});
}

AffineExpr mlir::makeCanonicalStridedLayoutExpr(ArrayRef<int64_t> sizes,
                                                ArrayRef<AffineExpr> exprs,
                                                MLIRContext *context) {
  AffineExpr expr;
  bool dynamicPoisonBit = false;
  unsigned numDims = 0;
  unsigned nSymbols = 0;
  // Compute the number of symbols and dimensions of the passed exprs.
  for (AffineExpr expr : exprs) {
    expr.walk([&numDims, &nSymbols](AffineExpr d) {
      if (AffineDimExpr dim = d.dyn_cast<AffineDimExpr>())
        numDims = std::max(numDims, dim.getPosition() + 1);
      else if (AffineSymbolExpr symbol = d.dyn_cast<AffineSymbolExpr>())
        nSymbols = std::max(nSymbols, symbol.getPosition() + 1);
    });
  }
  int64_t runningSize = 1;
  for (auto en : llvm::zip(llvm::reverse(exprs), llvm::reverse(sizes))) {
    int64_t size = std::get<1>(en);
    // Degenerate case, no size =-> no stride
    if (size == 0)
      continue;
    AffineExpr dimExpr = std::get<0>(en);
    AffineExpr stride = dynamicPoisonBit
                            ? getAffineSymbolExpr(nSymbols++, context)
                            : getAffineConstantExpr(runningSize, context);
    expr = expr ? expr + dimExpr * stride : dimExpr * stride;
    if (size > 0)
      runningSize *= size;
    else
      dynamicPoisonBit = true;
  }
  return simplifyAffineExpr(expr, numDims, nSymbols);
}

/// Return a version of `t` with a layout that has all dynamic offset and
/// strides. This is used to erase the static layout.
MemRefType mlir::eraseStridedLayout(MemRefType t) {
  auto val = ShapedType::kDynamicStrideOrOffset;
  return MemRefType::Builder(t).setAffineMaps(makeStridedLinearLayoutMap(
      SmallVector<int64_t, 4>(t.getRank(), val), val, t.getContext()));
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
  SmallVector<int64_t, 4> stridesAndOffset;
  auto res = getStridesAndOffset(t, stridesAndOffset, offset);
  return succeeded(res);
}
