//===- StringTable.h - Table of strings tracked by offset ----------C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_STRING_TABLE_H
#define LLVM_ADT_STRING_TABLE_H

#include "llvm/ADT/StringRef.h"
#include <limits>

namespace llvm {

/// A table of densely packed, null-terminated strings indexed by offset.
///
/// This table abstracts a densely concatenated list of null-terminated strings,
/// each of which can be referenced using an offset into the table.
///
/// This requires and ensures that the string at offset 0 is also the empty
/// string. This helps allow zero-initialized offsets form empty strings and
/// avoids non-zero initialization when using a string literal pointer would
/// allow a null pointer.
///
/// The primary use case is having a single global string literal for the table
/// contents, and offsets into it in other global data structures to avoid
/// dynamic relocations of individual string literal pointers in those global
/// data structures.
class StringTable {
  StringRef Table;

public:
  // An offset into one of these packed string tables, used to select a string
  // within the table.
  //
  // Typically these are created by TableGen or other code generator from
  // computed offsets, and it just wraps that integer into a type until it is
  // used with the relevant table.
  //
  // We also ensure that the empty string is at offset zero and default
  // constructing this class gives you an offset of zero. This makes default
  // constructing this type work similarly (after indexing the table) to default
  // constructing a `StringRef`.
  class Offset {
    // Note that we ensure the empty string is at offset zero.
    unsigned Value = 0;

  public:
    constexpr Offset() = default;
    constexpr Offset(unsigned Value) : Value(Value) {}

    constexpr unsigned value() const { return Value; }
  };

  // We directly handle string literals with a templated converting constructor
  // because we *don't* want to do `strlen` on them -- we fully expect null
  // bytes in this input. This is somewhat the opposite of how `StringLiteral`
  // works.
  template <size_t N>
  constexpr StringTable(const char (&RawTable)[N]) : Table(RawTable, N) {
    static_assert(N <= std::numeric_limits<unsigned>::max(),
                  "We only support table sizes that can be indexed by an "
                  "`unsigned` offset.");

    // Note that we can only use `empty`, `data`, and `size` in these asserts to
    // support `constexpr`.
    assert(!Table.empty() && "Requires at least a valid empty string.");
    assert(Table.data()[0] == '\0' && "Offset zero must be the empty string.");
    // Ensure that `strlen` from any offset cannot overflow the end of the table
    // by insisting on a null byte at the end.
    assert(Table.data()[Table.size() - 1] == '\0' &&
           "Last byte must be a null byte.");
  }

  // Get a string from the table starting with the provided offset. The returned
  // `StringRef` is in fact null terminated, and so can be converted safely to a
  // C-string if necessary for a system API.
  constexpr StringRef operator[](Offset O) const {
    assert(O.value() < Table.size() && "Out of bounds offset!");
    return Table.data() + O.value();
  }

  /// Returns the byte size of the table.
  constexpr size_t size() const { return Table.size(); }
};

} // namespace llvm

#endif // LLVM_ADT_STRING_TABLE_H
