//===- InterleavedRange.h - Output stream formatting for ranges -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements format objects for printing ranges to output streams.
// For example:
// ```c++
//    ArrayRef<Type> Types = ...;
//    OS << "Types: " << interleaved(Types); // ==> "Types: i32, f16, i8"
//    ArrayRef<int> Values = ...;
//    OS << "Values: " << interleaved_array(Values); // ==> "Values: [1, 2, 3]"
// ```
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_INTERLEAVED_RANGE_H
#define LLVM_SUPPORT_INTERLEAVED_RANGE_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

/// Format object class for interleaved ranges. Supports specifying the
/// separator and, optionally, the prefix and suffix to be printed surrounding
/// the range.
/// Uses the operator '<<' of the range element type for printing. The range
/// type itself does not have to have an '<<' operator defined.
template <typename Range> class InterleavedRange {
  const Range &TheRange;
  StringRef Separator;
  StringRef Prefix;
  StringRef Suffix;

public:
  InterleavedRange(const Range &R, StringRef Separator, StringRef Prefix,
                   StringRef Suffix)
      : TheRange(R), Separator(Separator), Prefix(Prefix), Suffix(Suffix) {}

  template <typename OStream>
  friend OStream &operator<<(OStream &OS, const InterleavedRange &Interleaved) {
    if (!Interleaved.Prefix.empty())
      OS << Interleaved.Prefix;
    llvm::interleave(Interleaved.TheRange, OS, Interleaved.Separator);
    if (!Interleaved.Suffix.empty())
      OS << Interleaved.Suffix;
    return OS;
  }

  std::string str() const {
    std::string Result;
    raw_string_ostream Stream(Result);
    Stream << *this;
    Stream.flush();
    return Result;
  }

  operator std::string() const { return str(); }
};

/// Output range `R` as a sequence of interleaved elements. Requires the range
/// element type to be printable using `raw_ostream& operator<<`. The
/// `Separator` and `Prefix` / `Suffix` can be customized. Examples:
/// ```c++
///   SmallVector<int> Vals = {1, 2, 3};
///   OS << interleaved(Vals);                 // ==> "1, 2, 3"
///   OS << interleaved(Vals, ";");            // ==> "1;2;3"
///   OS << interleaved(Vals, " ", "{", "}");  // ==> "{1 2 3}"
/// ```
template <typename Range>
InterleavedRange<Range> interleaved(const Range &R, StringRef Separator = ", ",
                                    StringRef Prefix = "",
                                    StringRef Suffix = "") {
  return {R, Separator, Prefix, Suffix};
}

/// Output range `R` as an array of interleaved elements. Requires the range
/// element type to be printable using `raw_ostream& operator<<`. The
/// `Separator` can be customized. Examples:
/// ```c++
///   SmallVector<int> Vals = {1, 2, 3};
///   OS << interleaved_array(Vals);       // ==> "[1, 2, 3]"
///   OS << interleaved_array(Vals, ";");  // ==> "[1;2;3]"
///   OS << interleaved_array(Vals, " ");  // ==> "[1 2 3]"
/// ```
template <typename Range>
InterleavedRange<Range> interleaved_array(const Range &R,
                                          StringRef Separator = ", ") {
  return {R, Separator, "[", "]"};
}

} // end namespace llvm

#endif // LLVM_SUPPORT_INTERLEAVED_RANGE_H
