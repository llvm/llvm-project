//===- BlockAndValueMapping.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a utility class for maintaining a mapping for multiple
// value types.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_BLOCKANDVALUEMAPPING_H
#define MLIR_IR_BLOCKANDVALUEMAPPING_H

#include "mlir/IR/Block.h"

namespace mlir {
// This is a utility class for mapping one set of values to another. New
// mappings can be inserted via 'map'. Existing mappings can be
// found via the 'lookup*' functions. There are two variants that differ only in
// return value when an existing is not found for the provided key.
// 'lookupOrNull' returns nullptr where as 'lookupOrDefault' will return the
// lookup key.
class BlockAndValueMapping {
public:
  /// Inserts a new mapping for 'from' to 'to'. If there is an existing mapping,
  /// it is overwritten.
  void map(Block *from, Block *to) { blockMap[from] = to; }
  void map(Value from, Value to) { valueMap[from] = to; }

  template <
      typename S, typename T,
      std::enable_if_t<!std::is_assignable<Value, S>::value &&
                       !std::is_assignable<Block *, S>::value> * = nullptr>
  void map(S &&from, T &&to) {
    for (auto pair : llvm::zip(from, to))
      map(std::get<0>(pair), std::get<1>(pair));
  }

  /// Erases a mapping for 'from'.
  void erase(Block *from) { blockMap.erase(from); }
  void erase(Value from) { valueMap.erase(from); }

  /// Checks to see if a mapping for 'from' exists.
  bool contains(Block *from) const { return blockMap.count(from); }
  bool contains(Value from) const { return valueMap.count(from); }

  /// Lookup a mapped value within the map. If a mapping for the provided value
  /// does not exist then return nullptr.
  Block *lookupOrNull(Block *from) const {
    return lookupOrValue(from, (Block *)nullptr);
  }
  Value lookupOrNull(Value from) const { return lookupOrValue(from, Value()); }

  /// Lookup a mapped value within the map. If a mapping for the provided value
  /// does not exist then return the provided value.
  Block *lookupOrDefault(Block *from) const {
    return lookupOrValue(from, from);
  }
  Value lookupOrDefault(Value from) const { return lookupOrValue(from, from); }

  /// Lookup a mapped value within the map. This asserts the provided value
  /// exists within the map.
  template <typename T>
  T lookup(T from) const {
    auto result = lookupOrNull(from);
    assert(result && "expected 'from' to be contained within the map");
    return result;
  }

  /// Clears all mappings held by the mapper.
  void clear() { valueMap.clear(); }

  /// Return the held value mapping.
  const DenseMap<Value, Value> &getValueMap() const { return valueMap; }

  /// Return the held block mapping.
  const DenseMap<Block *, Block *> &getBlockMap() const { return blockMap; }

private:
  /// Utility lookupOrValue that looks up an existing key or returns the
  /// provided value.
  Block *lookupOrValue(Block *from, Block *value) const {
    auto it = blockMap.find(from);
    return it != blockMap.end() ? it->second : value;
  }
  Value lookupOrValue(Value from, Value value) const {
    auto it = valueMap.find(from);
    return it != valueMap.end() ? it->second : value;
  }

  DenseMap<Value, Value> valueMap;
  DenseMap<Block *, Block *> blockMap;
};

} // namespace mlir

#endif // MLIR_IR_BLOCKANDVALUEMAPPING_H
