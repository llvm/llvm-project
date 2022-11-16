//===- SubElementInterfaces.cpp - Attr and Type SubElement Interfaces -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/SubElementInterfaces.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/DenseSet.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// SubElementInterface
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// WalkSubElements

template <typename InterfaceT>
static void walkSubElementsImpl(InterfaceT interface,
                                function_ref<void(Attribute)> walkAttrsFn,
                                function_ref<void(Type)> walkTypesFn,
                                DenseSet<Attribute> &visitedAttrs,
                                DenseSet<Type> &visitedTypes) {
  interface.walkImmediateSubElements(
      [&](Attribute attr) {
        // Guard against potentially null inputs. This removes the need for the
        // derived attribute/type to do it.
        if (!attr)
          return;

        // Avoid infinite recursion when visiting sub attributes later, if this
        // is a mutable attribute.
        if (LLVM_UNLIKELY(attr.hasTrait<AttributeTrait::IsMutable>())) {
          if (!visitedAttrs.insert(attr).second)
            return;
        }

        // Walk any sub elements first.
        if (auto interface = attr.dyn_cast<SubElementAttrInterface>())
          walkSubElementsImpl(interface, walkAttrsFn, walkTypesFn, visitedAttrs,
                              visitedTypes);

        // Walk this attribute.
        walkAttrsFn(attr);
      },
      [&](Type type) {
        // Guard against potentially null inputs. This removes the need for the
        // derived attribute/type to do it.
        if (!type)
          return;

        // Avoid infinite recursion when visiting sub types later, if this
        // is a mutable type.
        if (LLVM_UNLIKELY(type.hasTrait<TypeTrait::IsMutable>())) {
          if (!visitedTypes.insert(type).second)
            return;
        }

        // Walk any sub elements first.
        if (auto interface = type.dyn_cast<SubElementTypeInterface>())
          walkSubElementsImpl(interface, walkAttrsFn, walkTypesFn, visitedAttrs,
                              visitedTypes);

        // Walk this type.
        walkTypesFn(type);
      });
}

void SubElementAttrInterface::walkSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) {
  assert(walkAttrsFn && walkTypesFn && "expected valid walk functions");
  DenseSet<Attribute> visitedAttrs;
  DenseSet<Type> visitedTypes;
  walkSubElementsImpl(*this, walkAttrsFn, walkTypesFn, visitedAttrs,
                      visitedTypes);
}

void SubElementTypeInterface::walkSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) {
  assert(walkAttrsFn && walkTypesFn && "expected valid walk functions");
  DenseSet<Attribute> visitedAttrs;
  DenseSet<Type> visitedTypes;
  walkSubElementsImpl(*this, walkAttrsFn, walkTypesFn, visitedAttrs,
                      visitedTypes);
}

//===----------------------------------------------------------------------===//
/// AttrTypeReplacer
//===----------------------------------------------------------------------===//

void AttrTypeReplacer::replaceElementsIn(Operation *op, bool replaceAttrs,
                                         bool replaceLocs, bool replaceTypes) {
  // Functor that replaces the given element if the new value is different,
  // otherwise returns nullptr.
  auto replaceIfDifferent = [&](auto element) {
    auto replacement = replace(element);
    return (replacement && replacement != element) ? replacement : nullptr;
  };

  // Update the attribute dictionary.
  if (replaceAttrs) {
    if (auto newAttrs = replaceIfDifferent(op->getAttrDictionary()))
      op->setAttrs(cast<DictionaryAttr>(newAttrs));
  }

  // If we aren't updating locations or types, we're done.
  if (!replaceTypes && !replaceLocs)
    return;

  // Update the location.
  if (replaceLocs) {
    if (Attribute newLoc = replaceIfDifferent(op->getLoc()))
      op->setLoc(cast<LocationAttr>(newLoc));
  }

  // Update the result types.
  if (replaceTypes) {
    for (OpResult result : op->getResults())
      if (Type newType = replaceIfDifferent(result.getType()))
        result.setType(newType);
  }

  // Update any nested block arguments.
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (BlockArgument &arg : block.getArguments()) {
        if (replaceLocs) {
          if (Attribute newLoc = replaceIfDifferent(arg.getLoc()))
            arg.setLoc(cast<LocationAttr>(newLoc));
        }

        if (replaceTypes) {
          if (Type newType = replaceIfDifferent(arg.getType()))
            arg.setType(newType);
        }
      }
    }
  }
}

template <typename T>
static void updateSubElementImpl(T element, AttrTypeReplacer &replacer,
                                 DenseMap<T, T> &elementMap,
                                 SmallVectorImpl<T> &newElements,
                                 FailureOr<bool> &changed) {
  // Bail early if we failed at any point.
  if (failed(changed))
    return;

  // Guard against potentially null inputs. We always map null to null.
  if (!element) {
    newElements.push_back(nullptr);
    return;
  }

  // Replace the element.
  if (T result = replacer.replace(element)) {
    newElements.push_back(result);
    if (result != element)
      changed = true;
  } else {
    changed = failure();
  }
}

template <typename InterfaceT, typename T>
T AttrTypeReplacer::replaceSubElements(InterfaceT interface,
                                       DenseMap<T, T> &interfaceMap) {
  // Walk the current sub-elements, replacing them as necessary.
  SmallVector<Attribute, 16> newAttrs;
  SmallVector<Type, 16> newTypes;
  FailureOr<bool> changed = false;
  interface.walkImmediateSubElements(
      [&](Attribute element) {
        updateSubElementImpl(element, *this, attrMap, newAttrs, changed);
      },
      [&](Type element) {
        updateSubElementImpl(element, *this, typeMap, newTypes, changed);
      });
  if (failed(changed))
    return nullptr;

  // If any sub-elements changed, use the new elements during the replacement.
  T result = interface;
  if (*changed)
    result = interface.replaceImmediateSubElements(newAttrs, newTypes);
  return result;
}

/// Shared implementation of replacing a given attribute or type element.
template <typename InterfaceT, typename ReplaceFns, typename T>
T AttrTypeReplacer::replaceImpl(T element, ReplaceFns &replaceFns,
                                DenseMap<T, T> &map) {
  auto [it, inserted] = map.try_emplace(element, element);
  if (!inserted)
    return it->second;

  T result = element;
  WalkResult walkResult = WalkResult::advance();
  for (auto &replaceFn : llvm::reverse(replaceFns)) {
    if (Optional<std::pair<T, WalkResult>> newRes = replaceFn(element)) {
      std::tie(result, walkResult) = *newRes;
      break;
    }
  }

  // If an error occurred, return nullptr to indicate failure.
  if (walkResult.wasInterrupted() || !result)
    return map[element] = nullptr;

  // Handle replacing sub-elements if this element is also a container.
  if (!walkResult.wasSkipped()) {
    if (auto interface = dyn_cast<InterfaceT>(result)) {
      // Replace the sub elements of this element, bailing if we fail.
      if (!(result = replaceSubElements(interface, map)))
        return map[element] = nullptr;
    }
  }

  return map[element] = result;
}

Attribute AttrTypeReplacer::replace(Attribute attr) {
  return replaceImpl<SubElementAttrInterface>(attr, attrReplacementFns,
                                              attrMap);
}

Type AttrTypeReplacer::replace(Type type) {
  return replaceImpl<SubElementTypeInterface>(type, typeReplacementFns,
                                              typeMap);
}

//===----------------------------------------------------------------------===//
// SubElementInterface Tablegen definitions
//===----------------------------------------------------------------------===//

#include "mlir/IR/SubElementAttrInterfaces.cpp.inc"
#include "mlir/IR/SubElementTypeInterfaces.cpp.inc"
