//===- FunctionSupport.cpp - Utility types for function-like ops ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/FunctionInterfaces.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Tablegen Interface Definitions
//===----------------------------------------------------------------------===//

#include "mlir/IR/FunctionOpInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// Function Arguments and Results.
//===----------------------------------------------------------------------===//

static bool isEmptyAttrDict(Attribute attr) {
  return attr.cast<DictionaryAttr>().empty();
}

DictionaryAttr mlir::function_interface_impl::getArgAttrDict(Operation *op,
                                                             unsigned index) {
  ArrayAttr attrs = op->getAttrOfType<ArrayAttr>(getArgDictAttrName());
  DictionaryAttr argAttrs =
      attrs ? attrs[index].cast<DictionaryAttr>() : DictionaryAttr();
  return argAttrs;
}

DictionaryAttr
mlir::function_interface_impl::getResultAttrDict(Operation *op,
                                                 unsigned index) {
  ArrayAttr attrs = op->getAttrOfType<ArrayAttr>(getResultDictAttrName());
  DictionaryAttr resAttrs =
      attrs ? attrs[index].cast<DictionaryAttr>() : DictionaryAttr();
  return resAttrs;
}

void mlir::function_interface_impl::detail::setArgResAttrDict(
    Operation *op, StringRef attrName, unsigned numTotalIndices, unsigned index,
    DictionaryAttr attrs) {
  ArrayAttr allAttrs = op->getAttrOfType<ArrayAttr>(attrName);
  if (!allAttrs) {
    if (attrs.empty())
      return;

    // If this attribute is not empty, we need to create a new attribute array.
    SmallVector<Attribute, 8> newAttrs(numTotalIndices,
                                       DictionaryAttr::get(op->getContext()));
    newAttrs[index] = attrs;
    op->setAttr(attrName, ArrayAttr::get(op->getContext(), newAttrs));
    return;
  }
  // Check to see if the attribute is different from what we already have.
  if (allAttrs[index] == attrs)
    return;

  // If it is, check to see if the attribute array would now contain only empty
  // dictionaries.
  ArrayRef<Attribute> rawAttrArray = allAttrs.getValue();
  if (attrs.empty() &&
      llvm::all_of(rawAttrArray.take_front(index), isEmptyAttrDict) &&
      llvm::all_of(rawAttrArray.drop_front(index + 1), isEmptyAttrDict)) {
    op->removeAttr(attrName);
    return;
  }

  // Otherwise, create a new attribute array with the updated dictionary.
  SmallVector<Attribute, 8> newAttrs(rawAttrArray.begin(), rawAttrArray.end());
  newAttrs[index] = attrs;
  op->setAttr(attrName, ArrayAttr::get(op->getContext(), newAttrs));
}

/// Set all of the argument or result attribute dictionaries for a function.
static void setAllArgResAttrDicts(Operation *op, StringRef attrName,
                                  ArrayRef<Attribute> attrs) {
  if (llvm::all_of(attrs, isEmptyAttrDict))
    op->removeAttr(attrName);
  else
    op->setAttr(attrName, ArrayAttr::get(op->getContext(), attrs));
}

void mlir::function_interface_impl::setAllArgAttrDicts(
    Operation *op, ArrayRef<DictionaryAttr> attrs) {
  setAllArgAttrDicts(op, ArrayRef<Attribute>(attrs.data(), attrs.size()));
}
void mlir::function_interface_impl::setAllArgAttrDicts(
    Operation *op, ArrayRef<Attribute> attrs) {
  auto wrappedAttrs = llvm::map_range(attrs, [op](Attribute attr) -> Attribute {
    return !attr ? DictionaryAttr::get(op->getContext()) : attr;
  });
  setAllArgResAttrDicts(op, getArgDictAttrName(),
                        llvm::to_vector<8>(wrappedAttrs));
}

void mlir::function_interface_impl::setAllResultAttrDicts(
    Operation *op, ArrayRef<DictionaryAttr> attrs) {
  setAllResultAttrDicts(op, ArrayRef<Attribute>(attrs.data(), attrs.size()));
}
void mlir::function_interface_impl::setAllResultAttrDicts(
    Operation *op, ArrayRef<Attribute> attrs) {
  auto wrappedAttrs = llvm::map_range(attrs, [op](Attribute attr) -> Attribute {
    return !attr ? DictionaryAttr::get(op->getContext()) : attr;
  });
  setAllArgResAttrDicts(op, getResultDictAttrName(),
                        llvm::to_vector<8>(wrappedAttrs));
}

void mlir::function_interface_impl::insertFunctionArguments(
    Operation *op, ArrayRef<unsigned> argIndices, TypeRange argTypes,
    ArrayRef<DictionaryAttr> argAttrs, ArrayRef<Location> argLocs,
    unsigned originalNumArgs, Type newType) {
  assert(argIndices.size() == argTypes.size());
  assert(argIndices.size() == argAttrs.size() || argAttrs.empty());
  assert(argIndices.size() == argLocs.size());
  if (argIndices.empty())
    return;

  // There are 3 things that need to be updated:
  // - Function type.
  // - Arg attrs.
  // - Block arguments of entry block.
  Block &entry = op->getRegion(0).front();

  // Update the argument attributes of the function.
  auto oldArgAttrs = op->getAttrOfType<ArrayAttr>(getArgDictAttrName());
  if (oldArgAttrs || !argAttrs.empty()) {
    SmallVector<DictionaryAttr, 4> newArgAttrs;
    newArgAttrs.reserve(originalNumArgs + argIndices.size());
    unsigned oldIdx = 0;
    auto migrate = [&](unsigned untilIdx) {
      if (!oldArgAttrs) {
        newArgAttrs.resize(newArgAttrs.size() + untilIdx - oldIdx);
      } else {
        auto oldArgAttrRange = oldArgAttrs.getAsRange<DictionaryAttr>();
        newArgAttrs.append(oldArgAttrRange.begin() + oldIdx,
                           oldArgAttrRange.begin() + untilIdx);
      }
      oldIdx = untilIdx;
    };
    for (unsigned i = 0, e = argIndices.size(); i < e; ++i) {
      migrate(argIndices[i]);
      newArgAttrs.push_back(argAttrs.empty() ? DictionaryAttr{} : argAttrs[i]);
    }
    migrate(originalNumArgs);
    setAllArgAttrDicts(op, newArgAttrs);
  }

  // Update the function type and any entry block arguments.
  op->setAttr(getTypeAttrName(), TypeAttr::get(newType));
  for (unsigned i = 0, e = argIndices.size(); i < e; ++i)
    entry.insertArgument(argIndices[i] + i, argTypes[i], argLocs[i]);
}

void mlir::function_interface_impl::insertFunctionResults(
    Operation *op, ArrayRef<unsigned> resultIndices, TypeRange resultTypes,
    ArrayRef<DictionaryAttr> resultAttrs, unsigned originalNumResults,
    Type newType) {
  assert(resultIndices.size() == resultTypes.size());
  assert(resultIndices.size() == resultAttrs.size() || resultAttrs.empty());
  if (resultIndices.empty())
    return;

  // There are 2 things that need to be updated:
  // - Function type.
  // - Result attrs.

  // Update the result attributes of the function.
  auto oldResultAttrs = op->getAttrOfType<ArrayAttr>(getResultDictAttrName());
  if (oldResultAttrs || !resultAttrs.empty()) {
    SmallVector<DictionaryAttr, 4> newResultAttrs;
    newResultAttrs.reserve(originalNumResults + resultIndices.size());
    unsigned oldIdx = 0;
    auto migrate = [&](unsigned untilIdx) {
      if (!oldResultAttrs) {
        newResultAttrs.resize(newResultAttrs.size() + untilIdx - oldIdx);
      } else {
        auto oldResultAttrsRange = oldResultAttrs.getAsRange<DictionaryAttr>();
        newResultAttrs.append(oldResultAttrsRange.begin() + oldIdx,
                              oldResultAttrsRange.begin() + untilIdx);
      }
      oldIdx = untilIdx;
    };
    for (unsigned i = 0, e = resultIndices.size(); i < e; ++i) {
      migrate(resultIndices[i]);
      newResultAttrs.push_back(resultAttrs.empty() ? DictionaryAttr{}
                                                   : resultAttrs[i]);
    }
    migrate(originalNumResults);
    setAllResultAttrDicts(op, newResultAttrs);
  }

  // Update the function type.
  op->setAttr(getTypeAttrName(), TypeAttr::get(newType));
}

void mlir::function_interface_impl::eraseFunctionArguments(
    Operation *op, const BitVector &argIndices, Type newType) {
  // There are 3 things that need to be updated:
  // - Function type.
  // - Arg attrs.
  // - Block arguments of entry block.
  Block &entry = op->getRegion(0).front();

  // Update the argument attributes of the function.
  if (auto argAttrs = op->getAttrOfType<ArrayAttr>(getArgDictAttrName())) {
    SmallVector<DictionaryAttr, 4> newArgAttrs;
    newArgAttrs.reserve(argAttrs.size());
    for (unsigned i = 0, e = argIndices.size(); i < e; ++i)
      if (!argIndices[i])
        newArgAttrs.emplace_back(argAttrs[i].cast<DictionaryAttr>());
    setAllArgAttrDicts(op, newArgAttrs);
  }

  // Update the function type and any entry block arguments.
  op->setAttr(getTypeAttrName(), TypeAttr::get(newType));
  entry.eraseArguments(argIndices);
}

void mlir::function_interface_impl::eraseFunctionResults(
    Operation *op, const BitVector &resultIndices, Type newType) {
  // There are 2 things that need to be updated:
  // - Function type.
  // - Result attrs.

  // Update the result attributes of the function.
  if (auto resAttrs = op->getAttrOfType<ArrayAttr>(getResultDictAttrName())) {
    SmallVector<DictionaryAttr, 4> newResultAttrs;
    newResultAttrs.reserve(resAttrs.size());
    for (unsigned i = 0, e = resultIndices.size(); i < e; ++i)
      if (!resultIndices[i])
        newResultAttrs.emplace_back(resAttrs[i].cast<DictionaryAttr>());
    setAllResultAttrDicts(op, newResultAttrs);
  }

  // Update the function type.
  op->setAttr(getTypeAttrName(), TypeAttr::get(newType));
}

TypeRange mlir::function_interface_impl::insertTypesInto(
    TypeRange oldTypes, ArrayRef<unsigned> indices, TypeRange newTypes,
    SmallVectorImpl<Type> &storage) {
  assert(indices.size() == newTypes.size() &&
         "mismatch between indice and type count");
  if (indices.empty())
    return oldTypes;

  auto fromIt = oldTypes.begin();
  for (auto it : llvm::zip(indices, newTypes)) {
    const auto toIt = oldTypes.begin() + std::get<0>(it);
    storage.append(fromIt, toIt);
    storage.push_back(std::get<1>(it));
    fromIt = toIt;
  }
  storage.append(fromIt, oldTypes.end());
  return storage;
}

TypeRange
mlir::function_interface_impl::filterTypesOut(TypeRange types,
                                              const BitVector &indices,
                                              SmallVectorImpl<Type> &storage) {
  if (indices.none())
    return types;

  for (unsigned i = 0, e = types.size(); i < e; ++i)
    if (!indices[i])
      storage.emplace_back(types[i]);
  return storage;
}

//===----------------------------------------------------------------------===//
// Function type signature.
//===----------------------------------------------------------------------===//

void mlir::function_interface_impl::setFunctionType(Operation *op,
                                                    Type newType) {
  FunctionOpInterface funcOp = cast<FunctionOpInterface>(op);
  unsigned oldNumArgs = funcOp.getNumArguments();
  unsigned oldNumResults = funcOp.getNumResults();
  op->setAttr(getTypeAttrName(), TypeAttr::get(newType));
  unsigned newNumArgs = funcOp.getNumArguments();
  unsigned newNumResults = funcOp.getNumResults();

  // Functor used to update the argument and result attributes of the function.
  auto updateAttrFn = [&](StringRef attrName, unsigned oldCount,
                          unsigned newCount, auto setAttrFn) {
    if (oldCount == newCount)
      return;
    // The new type has no arguments/results, just drop the attribute.
    if (newCount == 0) {
      op->removeAttr(attrName);
      return;
    }
    ArrayAttr attrs = op->getAttrOfType<ArrayAttr>(attrName);
    if (!attrs)
      return;

    // The new type has less arguments/results, take the first N attributes.
    if (newCount < oldCount)
      return setAttrFn(op, attrs.getValue().take_front(newCount));

    // Otherwise, the new type has more arguments/results. Initialize the new
    // arguments/results with empty attributes.
    SmallVector<Attribute> newAttrs(attrs.begin(), attrs.end());
    newAttrs.resize(newCount);
    setAttrFn(op, newAttrs);
  };

  // Update the argument and result attributes.
  updateAttrFn(
      getArgDictAttrName(), oldNumArgs, newNumArgs,
      [&](Operation *op, auto &&attrs) { setAllArgAttrDicts(op, attrs); });
  updateAttrFn(
      getResultDictAttrName(), oldNumResults, newNumResults,
      [&](Operation *op, auto &&attrs) { setAllResultAttrDicts(op, attrs); });
}
