//===- BuiltinAttributeInterfaces.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/Sequence.h"

using namespace mlir;
using namespace mlir::detail;

//===----------------------------------------------------------------------===//
/// Tablegen Interface Definitions
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinAttributeInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// ElementsAttr
//===----------------------------------------------------------------------===//

Type ElementsAttr::getElementType(ElementsAttr elementsAttr) {
  return elementsAttr.getShapedType().getElementType();
}

int64_t ElementsAttr::getNumElements(ElementsAttr elementsAttr) {
  return elementsAttr.getShapedType().getNumElements();
}

bool ElementsAttr::isValidIndex(ShapedType type, ArrayRef<uint64_t> index) {
  // Verify that the rank of the indices matches the held type.
  int64_t rank = type.getRank();
  if (rank == 0 && index.size() == 1 && index[0] == 0)
    return true;
  if (rank != static_cast<int64_t>(index.size()))
    return false;

  // Verify that all of the indices are within the shape dimensions.
  ArrayRef<int64_t> shape = type.getShape();
  return llvm::all_of(llvm::seq<int>(0, rank), [&](int i) {
    int64_t dim = static_cast<int64_t>(index[i]);
    return 0 <= dim && dim < shape[i];
  });
}
bool ElementsAttr::isValidIndex(ElementsAttr elementsAttr,
                                ArrayRef<uint64_t> index) {
  return isValidIndex(elementsAttr.getShapedType(), index);
}

uint64_t ElementsAttr::getFlattenedIndex(Type type, ArrayRef<uint64_t> index) {
  ShapedType shapeType = llvm::cast<ShapedType>(type);
  assert(isValidIndex(shapeType, index) &&
         "expected valid multi-dimensional index");

  // Reduce the provided multidimensional index into a flattended 1D row-major
  // index.
  auto rank = shapeType.getRank();
  ArrayRef<int64_t> shape = shapeType.getShape();
  uint64_t valueIndex = 0;
  uint64_t dimMultiplier = 1;
  for (int i = rank - 1; i >= 0; --i) {
    valueIndex += index[i] * dimMultiplier;
    dimMultiplier *= shape[i];
  }
  return valueIndex;
}

//===----------------------------------------------------------------------===//
// MemRefLayoutAttrInterface
//===----------------------------------------------------------------------===//

LogicalResult mlir::detail::verifyAffineMapAsLayout(
    AffineMap m, ArrayRef<int64_t> shape,
    function_ref<InFlightDiagnostic()> emitError) {
  if (m.getNumDims() != shape.size())
    return emitError() << "memref layout mismatch between rank and affine map: "
                       << shape.size() << " != " << m.getNumDims();

  return success();
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
static LogicalResult getStridesAndOffset(AffineMap m, ArrayRef<int64_t> shape,
                                         SmallVectorImpl<AffineExpr> &strides,
                                         AffineExpr &offset) {
  if (m.getNumResults() != 1 && !m.isIdentity())
    return failure();

  auto zero = getAffineConstantExpr(0, m.getContext());
  auto one = getAffineConstantExpr(1, m.getContext());
  offset = zero;
  strides.assign(shape.size(), zero);

  // Canonical case for empty map.
  if (m.isIdentity()) {
    // 0-D corner case, offset is already 0.
    if (shape.empty())
      return success();
    auto stridedExpr = makeCanonicalStridedLayoutExpr(shape, m.getContext());
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

LogicalResult mlir::detail::getAffineMapStridesAndOffset(
    AffineMap map, ArrayRef<int64_t> shape, SmallVectorImpl<int64_t> &strides,
    int64_t &offset) {
  AffineExpr offsetExpr;
  SmallVector<AffineExpr, 4> strideExprs;
  if (failed(::getStridesAndOffset(map, shape, strideExprs, offsetExpr)))
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
