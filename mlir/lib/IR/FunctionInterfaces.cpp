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
  return llvm::cast<DictionaryAttr>(attr).empty();
}

DictionaryAttr function_interface_impl::getArgAttrDict(FunctionOpInterface op,
                                                       unsigned index) {
  ArrayAttr attrs = op.getArgAttrsAttr();
  DictionaryAttr argAttrs =
      attrs ? llvm::cast<DictionaryAttr>(attrs[index]) : DictionaryAttr();
  return argAttrs;
}

DictionaryAttr
function_interface_impl::getResultAttrDict(FunctionOpInterface op,
                                           unsigned index) {
  ArrayAttr attrs = op.getResAttrsAttr();
  DictionaryAttr resAttrs =
      attrs ? llvm::cast<DictionaryAttr>(attrs[index]) : DictionaryAttr();
  return resAttrs;
}

ArrayRef<NamedAttribute>
function_interface_impl::getArgAttrs(FunctionOpInterface op, unsigned index) {
  auto argDict = getArgAttrDict(op, index);
  return argDict ? argDict.getValue() : std::nullopt;
}

ArrayRef<NamedAttribute>
function_interface_impl::getResultAttrs(FunctionOpInterface op,
                                        unsigned index) {
  auto resultDict = getResultAttrDict(op, index);
  return resultDict ? resultDict.getValue() : std::nullopt;
}

/// Get either the argument or result attributes array.
template <bool isArg>
static ArrayAttr getArgResAttrs(FunctionOpInterface op) {
  if constexpr (isArg)
    return op.getArgAttrsAttr();
  else
    return op.getResAttrsAttr();
}

/// Set either the argument or result attributes array.
template <bool isArg>
static void setArgResAttrs(FunctionOpInterface op, ArrayAttr attrs) {
  if constexpr (isArg)
    op.setArgAttrsAttr(attrs);
  else
    op.setResAttrsAttr(attrs);
}

/// Erase either the argument or result attributes array.
template <bool isArg>
static void removeArgResAttrs(FunctionOpInterface op) {
  if constexpr (isArg)
    op.removeArgAttrsAttr();
  else
    op.removeResAttrsAttr();
}

/// Set all of the argument or result attribute dictionaries for a function.
template <bool isArg>
static void setAllArgResAttrDicts(FunctionOpInterface op,
                                  ArrayRef<Attribute> attrs) {
  if (llvm::all_of(attrs, isEmptyAttrDict))
    removeArgResAttrs<isArg>(op);
  else
    setArgResAttrs<isArg>(op, ArrayAttr::get(op->getContext(), attrs));
}

void function_interface_impl::setAllArgAttrDicts(
    FunctionOpInterface op, ArrayRef<DictionaryAttr> attrs) {
  setAllArgAttrDicts(op, ArrayRef<Attribute>(attrs.data(), attrs.size()));
}

void function_interface_impl::setAllArgAttrDicts(FunctionOpInterface op,
                                                 ArrayRef<Attribute> attrs) {
  auto wrappedAttrs = llvm::map_range(attrs, [op](Attribute attr) -> Attribute {
    return !attr ? DictionaryAttr::get(op->getContext()) : attr;
  });
  setAllArgResAttrDicts</*isArg=*/true>(op, llvm::to_vector<8>(wrappedAttrs));
}

void function_interface_impl::setAllResultAttrDicts(
    FunctionOpInterface op, ArrayRef<DictionaryAttr> attrs) {
  setAllResultAttrDicts(op, ArrayRef<Attribute>(attrs.data(), attrs.size()));
}

void function_interface_impl::setAllResultAttrDicts(FunctionOpInterface op,
                                                    ArrayRef<Attribute> attrs) {
  auto wrappedAttrs = llvm::map_range(attrs, [op](Attribute attr) -> Attribute {
    return !attr ? DictionaryAttr::get(op->getContext()) : attr;
  });
  setAllArgResAttrDicts</*isArg=*/false>(op, llvm::to_vector<8>(wrappedAttrs));
}

/// Update the given index into an argument or result attribute dictionary.
template <bool isArg>
static void setArgResAttrDict(FunctionOpInterface op, unsigned numTotalIndices,
                              unsigned index, DictionaryAttr attrs) {
  ArrayAttr allAttrs = getArgResAttrs<isArg>(op);
  if (!allAttrs) {
    if (attrs.empty())
      return;

    // If this attribute is not empty, we need to create a new attribute array.
    SmallVector<Attribute, 8> newAttrs(numTotalIndices,
                                       DictionaryAttr::get(op->getContext()));
    newAttrs[index] = attrs;
    setArgResAttrs<isArg>(op, ArrayAttr::get(op->getContext(), newAttrs));
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
      llvm::all_of(rawAttrArray.drop_front(index + 1), isEmptyAttrDict))
    return removeArgResAttrs<isArg>(op);

  // Otherwise, create a new attribute array with the updated dictionary.
  SmallVector<Attribute, 8> newAttrs(rawAttrArray.begin(), rawAttrArray.end());
  newAttrs[index] = attrs;
  setArgResAttrs<isArg>(op, ArrayAttr::get(op->getContext(), newAttrs));
}

void function_interface_impl::setArgAttrs(FunctionOpInterface op,
                                          unsigned index,
                                          ArrayRef<NamedAttribute> attributes) {
  assert(index < op.getNumArguments() && "invalid argument number");
  return setArgResAttrDict</*isArg=*/true>(
      op, op.getNumArguments(), index,
      DictionaryAttr::get(op->getContext(), attributes));
}

void function_interface_impl::setArgAttrs(FunctionOpInterface op,
                                          unsigned index,
                                          DictionaryAttr attributes) {
  return setArgResAttrDict</*isArg=*/true>(
      op, op.getNumArguments(), index,
      attributes ? attributes : DictionaryAttr::get(op->getContext()));
}

void function_interface_impl::setResultAttrs(
    FunctionOpInterface op, unsigned index,
    ArrayRef<NamedAttribute> attributes) {
  assert(index < op.getNumResults() && "invalid result number");
  return setArgResAttrDict</*isArg=*/false>(
      op, op.getNumResults(), index,
      DictionaryAttr::get(op->getContext(), attributes));
}

void function_interface_impl::setResultAttrs(FunctionOpInterface op,
                                             unsigned index,
                                             DictionaryAttr attributes) {
  assert(index < op.getNumResults() && "invalid result number");
  return setArgResAttrDict</*isArg=*/false>(
      op, op.getNumResults(), index,
      attributes ? attributes : DictionaryAttr::get(op->getContext()));
}

void function_interface_impl::insertFunctionArguments(
    FunctionOpInterface op, ArrayRef<unsigned> argIndices, TypeRange argTypes,
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
  ArrayAttr oldArgAttrs = op.getArgAttrsAttr();
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
  op.setFunctionTypeAttr(TypeAttr::get(newType));
  for (unsigned i = 0, e = argIndices.size(); i < e; ++i)
    entry.insertArgument(argIndices[i] + i, argTypes[i], argLocs[i]);
}

void function_interface_impl::insertFunctionResults(
    FunctionOpInterface op, ArrayRef<unsigned> resultIndices,
    TypeRange resultTypes, ArrayRef<DictionaryAttr> resultAttrs,
    unsigned originalNumResults, Type newType) {
  assert(resultIndices.size() == resultTypes.size());
  assert(resultIndices.size() == resultAttrs.size() || resultAttrs.empty());
  if (resultIndices.empty())
    return;

  // There are 2 things that need to be updated:
  // - Function type.
  // - Result attrs.

  // Update the result attributes of the function.
  ArrayAttr oldResultAttrs = op.getResAttrsAttr();
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
  op.setFunctionTypeAttr(TypeAttr::get(newType));
}

void function_interface_impl::eraseFunctionArguments(
    FunctionOpInterface op, const BitVector &argIndices, Type newType) {
  // There are 3 things that need to be updated:
  // - Function type.
  // - Arg attrs.
  // - Block arguments of entry block.
  Block &entry = op->getRegion(0).front();

  // Update the argument attributes of the function.
  if (ArrayAttr argAttrs = op.getArgAttrsAttr()) {
    SmallVector<DictionaryAttr, 4> newArgAttrs;
    newArgAttrs.reserve(argAttrs.size());
    for (unsigned i = 0, e = argIndices.size(); i < e; ++i)
      if (!argIndices[i])
        newArgAttrs.emplace_back(llvm::cast<DictionaryAttr>(argAttrs[i]));
    setAllArgAttrDicts(op, newArgAttrs);
  }

  // Update the function type and any entry block arguments.
  op.setFunctionTypeAttr(TypeAttr::get(newType));
  entry.eraseArguments(argIndices);
}

void function_interface_impl::eraseFunctionResults(
    FunctionOpInterface op, const BitVector &resultIndices, Type newType) {
  // There are 2 things that need to be updated:
  // - Function type.
  // - Result attrs.

  // Update the result attributes of the function.
  if (ArrayAttr resAttrs = op.getResAttrsAttr()) {
    SmallVector<DictionaryAttr, 4> newResultAttrs;
    newResultAttrs.reserve(resAttrs.size());
    for (unsigned i = 0, e = resultIndices.size(); i < e; ++i)
      if (!resultIndices[i])
        newResultAttrs.emplace_back(llvm::cast<DictionaryAttr>(resAttrs[i]));
    setAllResultAttrDicts(op, newResultAttrs);
  }

  // Update the function type.
  op.setFunctionTypeAttr(TypeAttr::get(newType));
}

TypeRange function_interface_impl::insertTypesInto(
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

TypeRange function_interface_impl::filterTypesOut(
    TypeRange types, const BitVector &indices, SmallVectorImpl<Type> &storage) {
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

void function_interface_impl::setFunctionType(FunctionOpInterface op,
                                              Type newType) {
  unsigned oldNumArgs = op.getNumArguments();
  unsigned oldNumResults = op.getNumResults();
  op.setFunctionTypeAttr(TypeAttr::get(newType));
  unsigned newNumArgs = op.getNumArguments();
  unsigned newNumResults = op.getNumResults();

  // Functor used to update the argument and result attributes of the function.
  auto emptyDict = DictionaryAttr::get(op.getContext());
  auto updateAttrFn = [&](auto isArg, unsigned oldCount, unsigned newCount) {
    constexpr bool isArgVal = std::is_same_v<decltype(isArg), std::true_type>;

    if (oldCount == newCount)
      return;
    // The new type has no arguments/results, just drop the attribute.
    if (newCount == 0)
      return removeArgResAttrs<isArgVal>(op);
    ArrayAttr attrs = getArgResAttrs<isArgVal>(op);
    if (!attrs)
      return;

    // The new type has less arguments/results, take the first N attributes.
    if (newCount < oldCount)
      return setAllArgResAttrDicts<isArgVal>(
          op, attrs.getValue().take_front(newCount));

    // Otherwise, the new type has more arguments/results. Initialize the new
    // arguments/results with empty dictionary attributes.
    SmallVector<Attribute> newAttrs(attrs.begin(), attrs.end());
    newAttrs.resize(newCount, emptyDict);
    setAllArgResAttrDicts<isArgVal>(op, newAttrs);
  };

  // Update the argument and result attributes.
  updateAttrFn(std::true_type{}, oldNumArgs, newNumArgs);
  updateAttrFn(std::false_type{}, oldNumResults, newNumResults);
}
