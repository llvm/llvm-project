//===- SubElementInterfaces.cpp - Attr and Type SubElement Interfaces -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/SubElementInterfaces.h"

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
// ReplaceSubElements

/// Return if the given element is mutable.
static bool isMutable(Attribute attr) {
  return attr.hasTrait<AttributeTrait::IsMutable>();
}
static bool isMutable(Type type) {
  return type.hasTrait<TypeTrait::IsMutable>();
}

template <typename InterfaceT, typename T, typename ReplaceSubElementFnT>
static void updateSubElementImpl(
    T element, function_ref<std::pair<T, WalkResult>(T)> walkFn,
    DenseMap<T, T> &visited, SmallVectorImpl<T> &newElements,
    FailureOr<bool> &changed, ReplaceSubElementFnT &&replaceSubElementFn) {
  // Bail early if we failed at any point.
  if (failed(changed))
    return;
  newElements.push_back(element);

  // Guard against potentially null inputs. We always map null to null.
  if (!element)
    return;

  // Check for an existing mapping for this element, and walk it if we haven't
  // yet.
  T &mappedElement = visited[element];
  if (!mappedElement) {
    WalkResult result = WalkResult::advance();
    std::tie(mappedElement, result) = walkFn(element);

    // Try walking this element.
    if (result.wasInterrupted() || !mappedElement) {
      changed = failure();
      return;
    }

    // Handle replacing sub-elements if this element is also a container.
    if (!result.wasSkipped()) {
      if (auto interface = mappedElement.template dyn_cast<InterfaceT>()) {
        if (!(mappedElement = replaceSubElementFn(interface))) {
          changed = failure();
          return;
        }
      }
    }
  }

  // Update to the mapped element.
  if (mappedElement != element) {
    newElements.back() = mappedElement;
    changed = true;
  }
}

template <typename InterfaceT>
static typename InterfaceT::ValueType
replaceSubElementsImpl(InterfaceT interface,
                       SubElementResultReplFn<Attribute> walkAttrsFn,
                       SubElementResultReplFn<Type> walkTypesFn,
                       DenseMap<Attribute, Attribute> &visitedAttrs,
                       DenseMap<Type, Type> &visitedTypes) {
  // Walk the current sub-elements, replacing them as necessary.
  SmallVector<Attribute, 16> newAttrs;
  SmallVector<Type, 16> newTypes;
  FailureOr<bool> changed = false;
  auto replaceSubElementFn = [&](auto subInterface) {
    return replaceSubElementsImpl(subInterface, walkAttrsFn, walkTypesFn,
                                  visitedAttrs, visitedTypes);
  };
  interface.walkImmediateSubElements(
      [&](Attribute element) {
        updateSubElementImpl<SubElementAttrInterface>(
            element, walkAttrsFn, visitedAttrs, newAttrs, changed,
            replaceSubElementFn);
      },
      [&](Type element) {
        updateSubElementImpl<SubElementTypeInterface>(
            element, walkTypesFn, visitedTypes, newTypes, changed,
            replaceSubElementFn);
      });
  if (failed(changed))
    return {};

  // If the sub-elements didn't change, just return the original value.
  if (!*changed)
    return interface;

  // If this element is mutable, we don't support changing its sub elements, the
  // sub element walk doesn't give us a valid ordering for what we need here. If
  // we want to support mutable elements, we'll need something more.
  if (isMutable(interface))
    return {};

  // Use the new elements during the replacement.
  return interface.replaceImmediateSubElements(newAttrs, newTypes);
}

Attribute SubElementAttrInterface::replaceSubElements(
    SubElementResultReplFn<Attribute> replaceAttrFn,
    SubElementResultReplFn<Type> replaceTypeFn) {
  assert(replaceAttrFn && replaceTypeFn && "expected valid replace functions");
  DenseMap<Attribute, Attribute> visitedAttrs;
  DenseMap<Type, Type> visitedTypes;
  return replaceSubElementsImpl(*this, replaceAttrFn, replaceTypeFn,
                                visitedAttrs, visitedTypes);
}

Type SubElementTypeInterface::replaceSubElements(
    SubElementResultReplFn<Attribute> replaceAttrFn,
    SubElementResultReplFn<Type> replaceTypeFn) {
  assert(replaceAttrFn && replaceTypeFn && "expected valid replace functions");
  DenseMap<Attribute, Attribute> visitedAttrs;
  DenseMap<Type, Type> visitedTypes;
  return replaceSubElementsImpl(*this, replaceAttrFn, replaceTypeFn,
                                visitedAttrs, visitedTypes);
}

//===----------------------------------------------------------------------===//
// SubElementInterface Tablegen definitions
//===----------------------------------------------------------------------===//

#include "mlir/IR/SubElementAttrInterfaces.cpp.inc"
#include "mlir/IR/SubElementTypeInterfaces.cpp.inc"
