//===- llvm/Support/Range.h - Range parsing utility -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides utilities for parsing range specifications like
// "1-10,20-30,45" which are commonly used in debugging and bisection tools.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_RANGE_H
#define LLVM_SUPPORT_RANGE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <cassert>
#include <cstdint>

namespace llvm {
class raw_ostream;
} // end namespace llvm

namespace llvm {

/// Represents a range of integers [Begin, End], inclusive on both ends, where
/// Begin <= End.
struct Range {
  int64_t Begin;
  int64_t End;

  /// Create a range [Begin, End].
  Range(int64_t Begin, int64_t End) : Begin(Begin), End(End) {
    assert(Begin <= End && "Range Begin must be <= End");
  }
  /// Create a range [Single, Single].
  Range(int64_t Single) : Begin(Single), End(Single) {}

  /// Check if the given value is within this range (inclusive).
  bool contains(int64_t Value) const { return Value >= Begin && Value <= End; }

  /// Check if this range overlaps with another range.
  bool overlaps(const Range &Other) const {
    return Begin <= Other.End && End >= Other.Begin;
  }

  /// Get the size of this range.
  int64_t size() const { return End - Begin + 1; }

  /// Print the range to the output stream.
  void print(raw_ostream &OS) const {
    if (Begin == End)
      OS << Begin;
    else
      OS << Begin << "-" << End;
  }

  bool operator==(const Range &Other) const {
    return Begin == Other.Begin && End == Other.End;
  }
};

/// Utility class for parsing and managing range specifications.
class RangeUtils {
public:
  using RangeList = SmallVector<Range, 8>;

  /// Parse a range specification string like "1-10,20-30,45" or
  /// "1-10:20-30:45". Ranges must be in increasing order and non-overlapping.
  /// \param RangeStr The string to parse.
  /// \param Separator The separator character to use (',' or ':').
  /// \returns Expected<RangeList> containing the parsed ranges on success,
  ///          or an Error on failure.
  static Expected<RangeList> parseRanges(StringRef RangeStr,
                                         char Separator = ',');

  /// Check if a value is contained in any of the ranges.
  static bool contains(ArrayRef<Range> Ranges, int64_t Value);

  /// Print ranges to output stream.
  /// \param OS The output stream to print to.
  /// \param Ranges The ranges to print.
  /// \param Separator The separator character to use between ranges (i.e. ',' or ':').
  static void printRanges(raw_ostream &OS, ArrayRef<Range> Ranges, char Separator = ',');

  /// Merge adjacent/consecutive ranges into single ranges.
  /// Example: [1-3, 4-6, 8-10] -> [1-6, 8-10].
  static RangeList mergeAdjacentRanges(ArrayRef<Range> Ranges);
};

} // end namespace llvm

#endif // LLVM_SUPPORT_RANGE_H
