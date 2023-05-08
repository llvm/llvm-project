//===- InferTypeOpInterface.cpp - Infer Type Interfaces ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the definitions of the infer op interfaces defined in
// `InferTypeOpInterface.td`.
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;

namespace mlir {
#include "mlir/Interfaces/InferTypeOpInterface.cpp.inc"
} // namespace mlir

LogicalResult
mlir::reifyResultShapes(OpBuilder &b, Operation *op,
                        ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  auto reifiableOp = dyn_cast<ReifyRankedShapedTypeOpInterface>(op);
  if (!reifiableOp)
    return failure();
  LogicalResult status = reifiableOp.reifyResultShapes(b, reifiedReturnShapes);
#ifndef NDEBUG
  if (failed(status))
    return failure();
  // Assert that ReifyRankedShapedTypeOpInterface::reifyResultShapes produced
  // a correct result.
  int64_t resultIdx = 0;
  for (OpResult result : op->getResults()) {
    auto shapedType = dyn_cast<ShapedType>(result.getType());
    if (!shapedType)
      continue;
    if (!shapedType.hasRank()) {
      // Nothing to check for unranked shaped values.
      ++resultIdx;
      continue;
    }
    // Assert one OpFoldResult per dimension.
    assert(shapedType.getRank() ==
               static_cast<int64_t>(reifiedReturnShapes[resultIdx].size()) &&
           "incorrect implementation of ReifyRankedShapedTypeOpInterface");
    for (int64_t dim = 0; dim < shapedType.getRank(); ++dim) {
      // reifyResultShapes must return:
      // * Attribute for static dimensions
      // * Value for dynamic dimensions
      assert(shapedType.isDynamicDim(dim) ==
                 reifiedReturnShapes[resultIdx][dim].is<Value>() &&
             "incorrect implementation of ReifyRankedShapedTypeOpInterface");
    }
    ++resultIdx;
  }
  // Assert that every shaped value result was reified.
  assert(resultIdx == static_cast<int64_t>(reifiedReturnShapes.size()) &&
         "incorrect implementation of ReifyRankedShapedTypeOpInterface");
#endif // NDEBUG
  return status;
}

bool ShapeAdaptor::hasRank() const {
  if (val.isNull())
    return false;
  if (auto t = val.dyn_cast<Type>())
    return cast<ShapedType>(t).hasRank();
  if (val.is<Attribute>())
    return true;
  return val.get<ShapedTypeComponents *>()->hasRank();
}

Type ShapeAdaptor::getElementType() const {
  if (val.isNull())
    return nullptr;
  if (auto t = val.dyn_cast<Type>())
    return cast<ShapedType>(t).getElementType();
  if (val.is<Attribute>())
    return nullptr;
  return val.get<ShapedTypeComponents *>()->getElementType();
}

void ShapeAdaptor::getDims(SmallVectorImpl<int64_t> &res) const {
  assert(hasRank());
  if (auto t = val.dyn_cast<Type>()) {
    ArrayRef<int64_t> vals = cast<ShapedType>(t).getShape();
    res.assign(vals.begin(), vals.end());
  } else if (auto attr = val.dyn_cast<Attribute>()) {
    auto dattr = cast<DenseIntElementsAttr>(attr);
    res.clear();
    res.reserve(dattr.size());
    for (auto it : dattr.getValues<APInt>())
      res.push_back(it.getSExtValue());
  } else {
    auto vals = val.get<ShapedTypeComponents *>()->getDims();
    res.assign(vals.begin(), vals.end());
  }
}

void ShapeAdaptor::getDims(ShapedTypeComponents &res) const {
  assert(hasRank());
  res.ranked = true;
  getDims(res.dims);
}

int64_t ShapeAdaptor::getDimSize(int index) const {
  assert(hasRank());
  if (auto t = val.dyn_cast<Type>())
    return cast<ShapedType>(t).getDimSize(index);
  if (auto attr = val.dyn_cast<Attribute>())
    return cast<DenseIntElementsAttr>(attr)
        .getValues<APInt>()[index]
        .getSExtValue();
  auto *stc = val.get<ShapedTypeComponents *>();
  return stc->getDims()[index];
}

int64_t ShapeAdaptor::getRank() const {
  assert(hasRank());
  if (auto t = val.dyn_cast<Type>())
    return cast<ShapedType>(t).getRank();
  if (auto attr = val.dyn_cast<Attribute>())
    return cast<DenseIntElementsAttr>(attr).size();
  return val.get<ShapedTypeComponents *>()->getDims().size();
}

bool ShapeAdaptor::hasStaticShape() const {
  if (!hasRank())
    return false;

  if (auto t = val.dyn_cast<Type>())
    return cast<ShapedType>(t).hasStaticShape();
  if (auto attr = val.dyn_cast<Attribute>()) {
    auto dattr = cast<DenseIntElementsAttr>(attr);
    for (auto index : dattr.getValues<APInt>())
      if (ShapedType::isDynamic(index.getSExtValue()))
        return false;
    return true;
  }
  auto *stc = val.get<ShapedTypeComponents *>();
  return llvm::none_of(stc->getDims(), ShapedType::isDynamic);
}

int64_t ShapeAdaptor::getNumElements() const {
  assert(hasStaticShape() && "cannot get element count of dynamic shaped type");

  if (auto t = val.dyn_cast<Type>())
    return cast<ShapedType>(t).getNumElements();

  if (auto attr = val.dyn_cast<Attribute>()) {
    auto dattr = cast<DenseIntElementsAttr>(attr);
    int64_t num = 1;
    for (auto index : dattr.getValues<APInt>()) {
      num *= index.getZExtValue();
      assert(num >= 0 && "integer overflow in element count computation");
    }
    return num;
  }

  auto *stc = val.get<ShapedTypeComponents *>();
  int64_t num = 1;
  for (int64_t dim : stc->getDims()) {
    num *= dim;
    assert(num >= 0 && "integer overflow in element count computation");
  }
  return num;
}

void ShapeAdaptor::dump() const {
  if (!hasRank()) {
    llvm::errs() << "<<unranked>>\n";
    return;
  }

  SmallVector<int64_t> dims;
  getDims(dims);
  auto mapped = llvm::map_range(dims, [](int64_t dim) -> std::string {
    if (ShapedType::isDynamic(dim))
      return "?";
    return llvm::formatv("{0}", dim).str();
  });
  llvm::errs() << "rank = " << getRank() << " dims = [";
  llvm::interleave(mapped, llvm::errs(), "x");
  llvm::errs() << "]\n";
}

ShapeAdaptor ValueShapeRange::getValueAsShape(int index) {
  Value val = operator[](index);
  if (valueToShape)
    if (ShapeAdaptor ret = valueToShape(val))
      return ret;

  DenseIntElementsAttr attr;
  if (!matchPattern(val, m_Constant(&attr)))
    return nullptr;
  if (attr.getType().getRank() != 1)
    return nullptr;
  return attr;
}

ShapeAdaptor ValueShapeRange::getShape(Value val) const {
  if (operandShape)
    if (ShapeAdaptor ret = operandShape(val))
      return ret;
  return val.getType();
}

ShapeAdaptor ValueShapeRange::getShape(int index) const {
  if (index < 0 || static_cast<size_t>(index) >= size())
    return nullptr;
  return getShape(operator[](index));
}

LogicalResult mlir::detail::inferReturnTensorTypes(
    function_ref<
        LogicalResult(MLIRContext *, std::optional<Location> location,
                      ValueShapeRange operands, DictionaryAttr attributes,
                      OpaqueProperties properties, RegionRange regions,
                      SmallVectorImpl<ShapedTypeComponents> &retComponents)>
        componentTypeFn,
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  SmallVector<ShapedTypeComponents, 2> retComponents;
  if (failed(componentTypeFn(context, location, operands, attributes,
                             properties, regions, retComponents)))
    return failure();
  for (const auto &shapeAndType : retComponents) {
    Type elementTy = shapeAndType.getElementType();
    assert(elementTy && "element type required to construct tensor");

    Attribute attr = shapeAndType.getAttribute();
    if (shapeAndType.hasRank()) {
      inferredReturnTypes.push_back(
          RankedTensorType::get(shapeAndType.getDims(), elementTy, attr));
    } else {
      assert(attr == nullptr && "attribute not supported");
      inferredReturnTypes.push_back(UnrankedTensorType::get(elementTy));
    }
  }
  return success();
}

LogicalResult mlir::detail::verifyInferredResultTypes(Operation *op) {
  SmallVector<Type, 4> inferredReturnTypes(op->getResultTypes());
  auto retTypeFn = cast<InferTypeOpInterface>(op);
  auto result = retTypeFn.refineReturnTypes(
      op->getContext(), op->getLoc(), op->getOperands(),
      op->getAttrDictionary(), op->getPropertiesStorage(), op->getRegions(),
      inferredReturnTypes);
  if (failed(result))
    op->emitOpError() << "failed to infer returned types";

  return result;
}
