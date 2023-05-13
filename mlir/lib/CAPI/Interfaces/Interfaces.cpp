

//===- Interfaces.cpp - C Interface for MLIR Interfaces -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Interfaces.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Interfaces.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "llvm/ADT/ScopeExit.h"
#include <optional>

using namespace mlir;

namespace {

std::optional<RegisteredOperationName>
getRegisteredOperationName(MlirContext context, MlirStringRef opName) {
  StringRef name(opName.data, opName.length);
  std::optional<RegisteredOperationName> info =
      RegisteredOperationName::lookup(name, unwrap(context));
  return info;
}

std::optional<Location> maybeGetLocation(MlirLocation location) {
  std::optional<Location> maybeLocation;
  if (!mlirLocationIsNull(location))
    maybeLocation = unwrap(location);
  return maybeLocation;
}

SmallVector<Value> unwrapOperands(intptr_t nOperands, MlirValue *operands) {
  SmallVector<Value> unwrappedOperands;
  (void)unwrapList(nOperands, operands, unwrappedOperands);
  return unwrappedOperands;
}

DictionaryAttr unwrapAttributes(MlirAttribute attributes) {
  DictionaryAttr attributeDict;
  if (!mlirAttributeIsNull(attributes))
    attributeDict = unwrap(attributes).cast<DictionaryAttr>();
  return attributeDict;
}

SmallVector<std::unique_ptr<Region>> unwrapRegions(intptr_t nRegions,
                                                   MlirRegion *regions) {
  // Create a vector of unique pointers to regions and make sure they are not
  // deleted when exiting the scope. This is a hack caused by C++ API expecting
  // an list of unique pointers to regions (without ownership transfer
  // semantics) and C API making ownership transfer explicit.
  SmallVector<std::unique_ptr<Region>> unwrappedRegions;
  unwrappedRegions.reserve(nRegions);
  for (intptr_t i = 0; i < nRegions; ++i)
    unwrappedRegions.emplace_back(unwrap(*(regions + i)));
  auto cleaner = llvm::make_scope_exit([&]() {
    for (auto &region : unwrappedRegions)
      region.release();
  });
  return unwrappedRegions;
}

} // namespace

bool mlirOperationImplementsInterface(MlirOperation operation,
                                      MlirTypeID interfaceTypeID) {
  std::optional<RegisteredOperationName> info =
      unwrap(operation)->getRegisteredInfo();
  return info && info->hasInterface(unwrap(interfaceTypeID));
}

bool mlirOperationImplementsInterfaceStatic(MlirStringRef operationName,
                                            MlirContext context,
                                            MlirTypeID interfaceTypeID) {
  std::optional<RegisteredOperationName> info = RegisteredOperationName::lookup(
      StringRef(operationName.data, operationName.length), unwrap(context));
  return info && info->hasInterface(unwrap(interfaceTypeID));
}

MlirTypeID mlirInferTypeOpInterfaceTypeID() {
  return wrap(InferTypeOpInterface::getInterfaceID());
}

MlirLogicalResult mlirInferTypeOpInterfaceInferReturnTypes(
    MlirStringRef opName, MlirContext context, MlirLocation location,
    intptr_t nOperands, MlirValue *operands, MlirAttribute attributes,
    void *properties, intptr_t nRegions, MlirRegion *regions,
    MlirTypesCallback callback, void *userData) {
  StringRef name(opName.data, opName.length);
  std::optional<RegisteredOperationName> info =
      getRegisteredOperationName(context, opName);
  if (!info)
    return mlirLogicalResultFailure();

  std::optional<Location> maybeLocation = maybeGetLocation(location);
  SmallVector<Value> unwrappedOperands = unwrapOperands(nOperands, operands);
  DictionaryAttr attributeDict = unwrapAttributes(attributes);
  SmallVector<std::unique_ptr<Region>> unwrappedRegions =
      unwrapRegions(nRegions, regions);

  SmallVector<Type> inferredTypes;
  if (failed(info->getInterface<InferTypeOpInterface>()->inferReturnTypes(
          unwrap(context), maybeLocation, unwrappedOperands, attributeDict,
          properties, unwrappedRegions, inferredTypes)))
    return mlirLogicalResultFailure();

  SmallVector<MlirType> wrappedInferredTypes;
  wrappedInferredTypes.reserve(inferredTypes.size());
  for (Type t : inferredTypes)
    wrappedInferredTypes.push_back(wrap(t));
  callback(wrappedInferredTypes.size(), wrappedInferredTypes.data(), userData);
  return mlirLogicalResultSuccess();
}

MlirTypeID mlirInferShapedTypeOpInterfaceTypeID() {
  return wrap(InferShapedTypeOpInterface::getInterfaceID());
}

MlirLogicalResult mlirInferShapedTypeOpInterfaceInferReturnTypes(
    MlirStringRef opName, MlirContext context, MlirLocation location,
    intptr_t nOperands, MlirValue *operands, MlirAttribute attributes,
    void *properties, intptr_t nRegions, MlirRegion *regions,
    MlirShapedTypeComponentsCallback callback, void *userData) {
  std::optional<RegisteredOperationName> info =
      getRegisteredOperationName(context, opName);
  if (!info)
    return mlirLogicalResultFailure();

  std::optional<Location> maybeLocation = maybeGetLocation(location);
  SmallVector<Value> unwrappedOperands = unwrapOperands(nOperands, operands);
  DictionaryAttr attributeDict = unwrapAttributes(attributes);
  SmallVector<std::unique_ptr<Region>> unwrappedRegions =
      unwrapRegions(nRegions, regions);

  SmallVector<ShapedTypeComponents> inferredTypeComponents;
  if (failed(info->getInterface<InferShapedTypeOpInterface>()
                 ->inferReturnTypeComponents(
                     unwrap(context), maybeLocation,
                     mlir::ValueRange(llvm::ArrayRef(unwrappedOperands)),
                     attributeDict, properties, unwrappedRegions,
                     inferredTypeComponents)))
    return mlirLogicalResultFailure();

  bool hasRank;
  intptr_t rank;
  const int64_t *shapeData;
  for (ShapedTypeComponents t : inferredTypeComponents) {
    if (t.hasRank()) {
      hasRank = true;
      rank = t.getDims().size();
      shapeData = t.getDims().data();
    } else {
      hasRank = false;
      rank = 0;
      shapeData = nullptr;
    }
    callback(hasRank, rank, shapeData, wrap(t.getElementType()),
             wrap(t.getAttribute()), userData);
  }
  return mlirLogicalResultSuccess();
}
