//===- Traits.cpp - Common op traits shared by dialects -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Traits.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;

bool OpTrait::util::staticallyKnownBroadcastable(ArrayRef<int64_t> shape1,
                                                 ArrayRef<int64_t> shape2) {
  SmallVector<SmallVector<int64_t, 6>, 2> extents;
  extents.emplace_back(shape1.begin(), shape1.end());
  extents.emplace_back(shape2.begin(), shape2.end());
  return staticallyKnownBroadcastable(extents);
}

bool OpTrait::util::staticallyKnownBroadcastable(
    ArrayRef<SmallVector<int64_t, 6>> shapes) {
  assert(!shapes.empty() && "Expected at least one shape");
  size_t maxRank = shapes[0].size();
  for (size_t i = 1; i != shapes.size(); ++i)
    maxRank = std::max(maxRank, shapes[i].size());

  // We look backwards through every column of `shapes`.
  for (size_t i = 0; i != maxRank; ++i) {
    bool seenDynamic = false;
    Optional<int64_t> nonOneDim;
    for (ArrayRef<int64_t> extent : shapes) {
      int64_t dim = i >= extent.size() ? 1 : extent[extent.size() - i - 1];

      if (dim == 1)
        continue;

      // Dimensions are compatible when
      //.  1. One is dynamic, the rest are 1
      if (ShapedType::isDynamic(dim)) {
        if (seenDynamic || nonOneDim)
          return false;
        seenDynamic = true;
      }

      //   2. All are 1 or a specific constant.
      if (nonOneDim && dim != *nonOneDim)
        return false;

      nonOneDim = dim;
    }
  }
  return true;
}

bool OpTrait::util::getBroadcastedShape(ArrayRef<int64_t> shape1,
                                        ArrayRef<int64_t> shape2,
                                        SmallVectorImpl<int64_t> &resultShape) {
  // To compute the result broadcasted shape, we compare operand shapes
  // element-wise: starting with the trailing dimensions, and working the
  // way backward. Two dimensions are compatible when
  //   1. they are equal, or
  //   2. one of them is 1
  // The result shape has the maximum among the two inputs at every
  // dimension index.

  resultShape.clear();
  if (shape1.size() > shape2.size()) {
    std::copy(shape1.begin(), shape1.end(), std::back_inserter(resultShape));
  } else {
    std::copy(shape2.begin(), shape2.end(), std::back_inserter(resultShape));
  }

  auto i1 = shape1.rbegin(), e1 = shape1.rend();
  auto i2 = shape2.rbegin(), e2 = shape2.rend();
  auto iR = resultShape.rbegin();

  // Check each dimension is consistent.
  for (; i1 != e1 && i2 != e2; ++i1, ++i2, ++iR) {
    if (*i1 == -1 || *i2 == -1) {
      // One or both dimensions is unknown. Follow TensorFlow behavior:
      // - If either dimension is greater than 1, we assume that the program is
      //   correct, and the other dimension will be broadcast to match it.
      // - If either dimension is 1, the other dimension is the output.
      if (*i1 > 1) {
        *iR = *i1;
      } else if (*i2 > 1) {
        *iR = *i2;
      } else if (*i1 == 1) {
        *iR = *i2;
      } else if (*i2 == 1) {
        *iR = *i1;
      } else {
        *iR = -1;
      }
    } else {
      if (*i1 == *i2 || *i2 == 1) {
        *iR = *i1;
      } else if (*i1 == 1) {
        *iR = *i2;
      } else {
        // This dimension of the two operand types is incompatible.
        resultShape.clear();
        return false;
      }
    }
  }

  return true;
}

/// Returns the shape of the given type. Scalars will be considered as having a
/// shape with zero dimensions.
static ArrayRef<int64_t> getShape(Type type) {
  if (auto sType = type.dyn_cast<ShapedType>())
    return sType.getShape();
  return {};
}

/// Returns the result broadcast composition type from the two given types by
/// following NumPy broadcast semantics. Returned type may have dynamic shape if
/// either of the input types has dynamic shape. Returns null type if the two
/// given types are not broadcast-compatible.
///
/// elementType, if specified, will be used as the element type of the
/// broadcasted result type. Otherwise it is required that the element type of
/// type1 and type2 is the same and this element type will be used as the
/// resultant element type.
Type OpTrait::util::getBroadcastedType(Type type1, Type type2,
                                       Type elementType) {
  // If the elementType is not specified, then the use the common element type
  // of the inputs or fail if there is no common element type.
  if (!elementType) {
    elementType = getElementTypeOrSelf(type1);
    if (elementType != getElementTypeOrSelf(type2))
      return {};
  }

  // If one of the types is unranked tensor, then the other type shouldn't be
  // vector and the result should have unranked tensor type.
  if (type1.isa<UnrankedTensorType>() || type2.isa<UnrankedTensorType>()) {
    if (type1.isa<VectorType>() || type2.isa<VectorType>())
      return {};
    return UnrankedTensorType::get(elementType);
  }

  // Returns the type kind if the given type is a vector or ranked tensor type.
  // Returns llvm::None otherwise.
  auto getCompositeTypeKind = [](Type type) -> Optional<TypeID> {
    if (type.isa<VectorType, RankedTensorType>())
      return type.getTypeID();
    return llvm::None;
  };

  // Make sure the composite type, if has, is consistent.
  Optional<TypeID> compositeKind1 = getCompositeTypeKind(type1);
  Optional<TypeID> compositeKind2 = getCompositeTypeKind(type2);
  Optional<TypeID> resultCompositeKind;

  if (compositeKind1 && compositeKind2) {
    // Disallow mixing vector and tensor.
    if (compositeKind1 != compositeKind2)
      return {};
    resultCompositeKind = compositeKind1;
  } else if (compositeKind1) {
    resultCompositeKind = compositeKind1;
  } else if (compositeKind2) {
    resultCompositeKind = compositeKind2;
  }

  // Get the shape of each type.
  SmallVector<int64_t, 4> resultShape;
  if (!getBroadcastedShape(getShape(type1), getShape(type2), resultShape))
    return {};

  // Compose the final broadcasted type
  if (resultCompositeKind == VectorType::getTypeID())
    return VectorType::get(resultShape, elementType);
  if (resultCompositeKind == RankedTensorType::getTypeID())
    return RankedTensorType::get(resultShape, elementType);
  return elementType;
}

/// Returns a tuple corresponding to whether range has tensor or vector type.
template <typename iterator_range>
static std::tuple<bool, bool> hasTensorOrVectorType(iterator_range types) {
  return std::make_tuple(
      llvm::any_of(types, [](Type t) { return t.isa<TensorType>(); }),
      llvm::any_of(types, [](Type t) { return t.isa<VectorType>(); }));
}

static bool areCompatibleShapes(ArrayRef<int64_t> shape1,
                                ArrayRef<int64_t> shape2) {
  auto isCompatible = [](int64_t dim1, int64_t dim2) {
    return dim1 == dim2 || dim1 == -1 || dim2 == -1;
  };
  if (shape1.size() != shape2.size())
    return false;
  for (auto p : llvm::zip(shape1, shape2))
    if (!isCompatible(std::get<0>(p), std::get<1>(p)))
      return false;
  return true;
}

static std::string getShapeString(ArrayRef<int64_t> shape) {
  // TODO: should replace with printing shape more uniformly across here and
  // when in type.
  return std::string(
      formatv("'{0:$[x]}'", llvm::make_range(shape.begin(), shape.end())));
}

LogicalResult OpTrait::impl::verifyCompatibleOperandBroadcast(Operation *op) {
  // Ensure broadcasting only tensor or only vector types.
  auto operandsHasTensorVectorType =
      hasTensorOrVectorType(op->getOperandTypes());
  auto resultsHasTensorVectorType = hasTensorOrVectorType(op->getResultTypes());
  if ((std::get<0>(operandsHasTensorVectorType) ||
       std::get<0>(resultsHasTensorVectorType)) &&
      (std::get<1>(operandsHasTensorVectorType) ||
       std::get<1>(resultsHasTensorVectorType)))
    return op->emitError("cannot broadcast vector with tensor");

  auto rankedOperands = make_filter_range(
      op->getOperandTypes(), [](Type t) { return t.isa<RankedTensorType>(); });

  // If all operands are unranked, then all result shapes are possible.
  if (rankedOperands.empty())
    return success();

  // Compute broadcasted shape of operands (which requires that operands are
  // broadcast compatible). The results need to be broadcast compatible with
  // this result shape.
  SmallVector<int64_t, 4> resultShape;
  (void)util::getBroadcastedShape(getShape(*rankedOperands.begin()), {},
                                  resultShape);
  for (auto other : make_early_inc_range(rankedOperands)) {
    SmallVector<int64_t, 4> temp = resultShape;
    if (!util::getBroadcastedShape(temp, getShape(other), resultShape))
      return op->emitOpError("operands don't have broadcast-compatible shapes");
  }

  auto rankedResults = make_filter_range(
      op->getResultTypes(), [](Type t) { return t.isa<RankedTensorType>(); });

  // If all of the results are unranked then no further verification.
  if (rankedResults.empty())
    return success();

  for (auto type : rankedResults) {
    ArrayRef<int64_t> actualSuffix =
        getShape(type).take_back(resultShape.size());
    if (!areCompatibleShapes(actualSuffix, resultShape))
      return op->emitOpError()
             << "result type " << getShapeString(getShape(type))
             << " not broadcast compatible with broadcasted operands's shapes "
             << getShapeString(resultShape);
  }
  return success();
}
