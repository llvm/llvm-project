//===- AttrTypeSubElements.cpp - Attr and Type SubElement Interfaces ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Operation.h"
#include <optional>

using namespace mlir;

//===----------------------------------------------------------------------===//
// AttrTypeWalker
//===----------------------------------------------------------------------===//

WalkResult AttrTypeWalker::walkImpl(Attribute attr, WalkOrder order) {
  return walkImpl(attr, attrWalkFns, order);
}
WalkResult AttrTypeWalker::walkImpl(Type type, WalkOrder order) {
  return walkImpl(type, typeWalkFns, order);
}

template <typename T, typename WalkFns>
WalkResult AttrTypeWalker::walkImpl(T element, WalkFns &walkFns,
                                    WalkOrder order) {
  // Check if we've already walk this element before.
  auto key = std::make_pair(element.getAsOpaquePointer(), (int)order);
  auto [it, inserted] =
      visitedAttrTypes.try_emplace(key, WalkResult::advance());
  if (!inserted)
    return it->second;

  // If we are walking in post order, walk the sub elements first.
  if (order == WalkOrder::PostOrder) {
    if (walkSubElements(element, order).wasInterrupted())
      return visitedAttrTypes[key] = WalkResult::interrupt();
  }

  // Walk this element, bailing if skipped or interrupted.
  for (auto &walkFn : llvm::reverse(walkFns)) {
    WalkResult walkResult = walkFn(element);
    if (walkResult.wasInterrupted())
      return visitedAttrTypes[key] = WalkResult::interrupt();
    if (walkResult.wasSkipped())
      return WalkResult::advance();
  }

  // If we are walking in pre-order, walk the sub elements last.
  if (order == WalkOrder::PreOrder) {
    if (walkSubElements(element, order).wasInterrupted())
      return WalkResult::interrupt();
  }
  return WalkResult::advance();
}

template <typename T>
WalkResult AttrTypeWalker::walkSubElements(T interface, WalkOrder order) {
  WalkResult result = WalkResult::advance();
  auto walkFn = [&](auto element) {
    if (element && !result.wasInterrupted())
      result = walkImpl(element, order);
  };
  interface.walkImmediateSubElements(walkFn, walkFn);
  return result.wasInterrupted() ? result : WalkResult::advance();
}

//===----------------------------------------------------------------------===//
/// AttrTypeReplacerBase
//===----------------------------------------------------------------------===//

template <typename Concrete>
void detail::AttrTypeReplacerBase<Concrete>::addReplacement(
    ReplaceFn<Attribute> fn) {
  attrReplacementFns.emplace_back(std::move(fn));
}

template <typename Concrete>
void detail::AttrTypeReplacerBase<Concrete>::addReplacement(
    ReplaceFn<Type> fn) {
  typeReplacementFns.push_back(std::move(fn));
}

template <typename Concrete>
void detail::AttrTypeReplacerBase<Concrete>::replaceElementsIn(
    Operation *op, bool replaceAttrs, bool replaceLocs, bool replaceTypes) {
  // Functor that replaces the given element if the new value is different,
  // otherwise returns nullptr.
  auto replaceIfDifferent = [&](auto element) {
    auto replacement = static_cast<Concrete *>(this)->replace(element);
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

template <typename Concrete>
void detail::AttrTypeReplacerBase<Concrete>::recursivelyReplaceElementsIn(
    Operation *op, bool replaceAttrs, bool replaceLocs, bool replaceTypes) {
  op->walk([&](Operation *nestedOp) {
    replaceElementsIn(nestedOp, replaceAttrs, replaceLocs, replaceTypes);
  });
}

template <typename T, typename Replacer>
static void updateSubElementImpl(T element, Replacer &replacer,
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

template <typename T, typename Replacer>
static T replaceSubElements(T interface, Replacer &replacer) {
  // Walk the current sub-elements, replacing them as necessary.
  SmallVector<Attribute, 16> newAttrs;
  SmallVector<Type, 16> newTypes;
  FailureOr<bool> changed = false;
  interface.walkImmediateSubElements(
      [&](Attribute element) {
        updateSubElementImpl(element, replacer, newAttrs, changed);
      },
      [&](Type element) {
        updateSubElementImpl(element, replacer, newTypes, changed);
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
template <typename T, typename ReplaceFns, typename Replacer>
static T replaceElementImpl(T element, ReplaceFns &replaceFns,
                            Replacer &replacer) {
  T result = element;
  WalkResult walkResult = WalkResult::advance();
  for (auto &replaceFn : llvm::reverse(replaceFns)) {
    if (std::optional<std::pair<T, WalkResult>> newRes = replaceFn(element)) {
      std::tie(result, walkResult) = *newRes;
      break;
    }
  }

  // If an error occurred, return nullptr to indicate failure.
  if (walkResult.wasInterrupted() || !result) {
    return nullptr;
  }

  // Handle replacing sub-elements if this element is also a container.
  if (!walkResult.wasSkipped()) {
    // Replace the sub elements of this element, bailing if we fail.
    if (!(result = replaceSubElements(result, replacer))) {
      return nullptr;
    }
  }

  return result;
}

template <typename Concrete>
Attribute detail::AttrTypeReplacerBase<Concrete>::replaceBase(Attribute attr) {
  return replaceElementImpl(attr, attrReplacementFns,
                            *static_cast<Concrete *>(this));
}

template <typename Concrete>
Type detail::AttrTypeReplacerBase<Concrete>::replaceBase(Type type) {
  return replaceElementImpl(type, typeReplacementFns,
                            *static_cast<Concrete *>(this));
}

//===----------------------------------------------------------------------===//
/// AttrTypeReplacer
//===----------------------------------------------------------------------===//

template class detail::AttrTypeReplacerBase<AttrTypeReplacer>;

template <typename T>
T AttrTypeReplacer::cachedReplaceImpl(T element) {
  const void *opaqueElement = element.getAsOpaquePointer();
  auto [it, inserted] = cache.try_emplace(opaqueElement, opaqueElement);
  if (!inserted)
    return T::getFromOpaquePointer(it->second);

  T result = replaceBase(element);

  cache[opaqueElement] = result.getAsOpaquePointer();
  return result;
}

Attribute AttrTypeReplacer::replace(Attribute attr) {
  return cachedReplaceImpl(attr);
}

Type AttrTypeReplacer::replace(Type type) { return cachedReplaceImpl(type); }

//===----------------------------------------------------------------------===//
/// CyclicAttrTypeReplacer
//===----------------------------------------------------------------------===//

template class detail::AttrTypeReplacerBase<CyclicAttrTypeReplacer>;

CyclicAttrTypeReplacer::CyclicAttrTypeReplacer()
    : cache([&](void *attr) { return breakCycleImpl(attr); }) {}

void CyclicAttrTypeReplacer::addCycleBreaker(CycleBreakerFn<Attribute> fn) {
  attrCycleBreakerFns.emplace_back(std::move(fn));
}

void CyclicAttrTypeReplacer::addCycleBreaker(CycleBreakerFn<Type> fn) {
  typeCycleBreakerFns.emplace_back(std::move(fn));
}

template <typename T>
T CyclicAttrTypeReplacer::cachedReplaceImpl(T element) {
  void *opaqueTaggedElement = AttrOrType(element).getOpaqueValue();
  CyclicReplacerCache<void *, const void *>::CacheEntry cacheEntry =
      cache.lookupOrInit(opaqueTaggedElement);
  if (auto resultOpt = cacheEntry.get())
    return T::getFromOpaquePointer(*resultOpt);

  T result = replaceBase(element);

  cacheEntry.resolve(result.getAsOpaquePointer());
  return result;
}

Attribute CyclicAttrTypeReplacer::replace(Attribute attr) {
  return cachedReplaceImpl(attr);
}

Type CyclicAttrTypeReplacer::replace(Type type) {
  return cachedReplaceImpl(type);
}

std::optional<const void *>
CyclicAttrTypeReplacer::breakCycleImpl(void *element) {
  AttrOrType attrType = AttrOrType::getFromOpaqueValue(element);
  if (auto attr = dyn_cast<Attribute>(attrType)) {
    for (auto &cyclicReplaceFn : llvm::reverse(attrCycleBreakerFns)) {
      if (std::optional<Attribute> newRes = cyclicReplaceFn(attr)) {
        return newRes->getAsOpaquePointer();
      }
    }
  } else {
    auto type = dyn_cast<Type>(attrType);
    for (auto &cyclicReplaceFn : llvm::reverse(typeCycleBreakerFns)) {
      if (std::optional<Type> newRes = cyclicReplaceFn(type)) {
        return newRes->getAsOpaquePointer();
      }
    }
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// AttrTypeImmediateSubElementWalker
//===----------------------------------------------------------------------===//

void AttrTypeImmediateSubElementWalker::walk(Attribute element) {
  if (element)
    walkAttrsFn(element);
}

void AttrTypeImmediateSubElementWalker::walk(Type element) {
  if (element)
    walkTypesFn(element);
}
