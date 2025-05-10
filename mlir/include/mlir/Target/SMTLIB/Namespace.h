//===- Namespace.h - Utilities for generating names -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides utilities for generating new names that do not conflict
// with existing names.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_NAMESPACE_H
#define MLIR_SUPPORT_NAMESPACE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/SMTLIB/SymCache.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"

namespace mlir {

/// A namespace that is used to store existing names and generate new names in
/// some scope within the IR. This exists to work around limitations of
/// SymbolTables. This acts as a base class providing facilities common to all
/// namespaces implementations.
class Namespace {
public:
  Namespace() {
    // This fills an entry for an empty string beforehand so that `newName`
    // doesn't return an empty string.
    nextIndex.insert({"", 0});
  }
  Namespace(const Namespace &other) = default;
  Namespace(Namespace &&other)
      : nextIndex(std::move(other.nextIndex)), locked(other.locked) {}

  Namespace &operator=(const Namespace &other) = default;
  Namespace &operator=(Namespace &&other) {
    nextIndex = std::move(other.nextIndex);
    locked = other.locked;
    return *this;
  }

  void add(mlir::ModuleOp module) {
    assert(module->getNumRegions() == 1);
    for (auto &op : module.getBody(0)->getOperations())
      if (auto symbol = op.getAttrOfType<mlir::StringAttr>(
              mlir::SymbolTable::getSymbolAttrName()))
        nextIndex.insert({symbol.getValue(), 0});
  }

  /// SymbolCache initializer; initialize from every key that is convertible to
  /// a StringAttr in the SymbolCache.
  void add(SymbolCache &symCache) {
    for (auto &&[attr, _] : symCache)
      if (auto strAttr = dyn_cast<mlir::StringAttr>(attr))
        nextIndex.insert({strAttr.getValue(), 0});
  }

  void add(llvm::StringRef name) { nextIndex.insert({name, 0}); }

  /// Removes a symbol from the namespace. Returns true if the symbol was
  /// removed, false if the symbol was not found.
  /// This is only allowed to be called _before_ any call to newName.
  bool erase(llvm::StringRef symbol) {
    assert(!locked && "Cannot erase names from a locked namespace");
    return nextIndex.erase(symbol);
  }

  /// Empty the namespace.
  void clear() {
    nextIndex.clear();
    locked = false;
  }

  /// Return a unique name, derived from the input `name`, and add the new name
  /// to the internal namespace.  There are two possible outcomes for the
  /// returned name:
  ///
  /// 1. The original name is returned.
  /// 2. The name is given a `_<n>` suffix where `<n>` is a number starting from
  ///    `0` and incrementing by one each time (`_0`, ...).
  llvm::StringRef newName(const llvm::Twine &name) {
    locked = true;
    // Special case the situation where there is no name collision to avoid
    // messing with the SmallString allocation below.
    llvm::SmallString<64> tryName;
    auto inserted = nextIndex.insert({name.toStringRef(tryName), 0});
    if (inserted.second)
      return inserted.first->getKey();

    // Try different suffixes until we get a collision-free one.
    if (tryName.empty())
      name.toVector(tryName); // toStringRef may leave tryName unfilled

    // Indexes less than nextIndex[tryName] are lready used, so skip them.
    // Indexes larger than nextIndex[tryName] may be used in another name.
    size_t &i = nextIndex[tryName];
    tryName.push_back('_');
    size_t baseLength = tryName.size();
    do {
      tryName.resize(baseLength);
      llvm::Twine(i++).toVector(tryName); // append integer to tryName
      inserted = nextIndex.insert({tryName, 0});
    } while (!inserted.second);

    return inserted.first->getKey();
  }

  /// Return a unique name, derived from the input `name` and ensure the
  /// returned name has the input `suffix`. Also add the new name to the
  /// internal namespace.
  /// There are two possible outcomes for the returned name:
  /// 1. The original name + `_<suffix>` is returned.
  /// 2. The name is given a suffix `_<n>_<suffix>` where `<n>` is a number
  ///    starting from `0` and incrementing by one each time.
  llvm::StringRef newName(const llvm::Twine &name, const llvm::Twine &suffix) {
    locked = true;
    // Special case the situation where there is no name collision to avoid
    // messing with the SmallString allocation below.
    llvm::SmallString<64> tryName;
    auto inserted = nextIndex.insert(
        {name.concat("_").concat(suffix).toStringRef(tryName), 0});
    if (inserted.second)
      return inserted.first->getKey();

    // Try different suffixes until we get a collision-free one.
    tryName.clear();
    name.toVector(tryName); // toStringRef may leave tryName unfilled
    tryName.push_back('_');
    size_t baseLength = tryName.size();

    // Get the initial number to start from.  Since `:` is not a valid character
    // in a verilog identifier, we use it separate the name and suffix.
    // Next number for name+suffix is stored with key `name_:suffix`.
    tryName.push_back(':');
    suffix.toVector(tryName);

    // Indexes less than nextIndex[tryName] are already used, so skip them.
    // Indexes larger than nextIndex[tryName] may be used in another name.
    size_t &i = nextIndex[tryName];
    do {
      tryName.resize(baseLength);
      llvm::Twine(i++).toVector(tryName); // append integer to tryName
      tryName.push_back('_');
      suffix.toVector(tryName);
      inserted = nextIndex.insert({tryName, 0});
    } while (!inserted.second);

    return inserted.first->getKey();
  }

protected:
  // The "next index" that will be tried when trying to unique a string within a
  // namespace.  It follows that all values less than the "next index" value are
  // already used.
  llvm::StringMap<size_t> nextIndex;

  // When true, no names can be erased from the namespace. This is to prevent
  // erasing names after they have been used, thus leaving users of the
  // namespace in an inconsistent state.
  bool locked = false;
};

} // namespace mlir

#endif // MLIR_SUPPORT_NAMESPACE_H
