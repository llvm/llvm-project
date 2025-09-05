//===- llvm/Support/Range.cpp - Range parsing utility ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Range.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"
#include <sstream>

using namespace llvm;

Expected<RangeUtils::RangeList> RangeUtils::parseRanges(StringRef Str,
                                                        char Separator) {
  RangeList Ranges;

  if (Str.empty())
    return std::move(Ranges);

  // Regex to match either single number or range "num1-num2"
  const Regex RangeRegex("^([0-9]+)(-([0-9]+))?$");

  for (StringRef Part : llvm::split(Str, Separator)) {
    Part = Part.trim();
    if (Part.empty())
      continue;

    SmallVector<StringRef, 4> Matches;
    if (!RangeRegex.match(Part, &Matches))
      return createStringError(std::errc::invalid_argument,
                               "Invalid range format: '%s'",
                               Part.str().c_str());

    int64_t Begin, End;
    if (Matches[1].getAsInteger(10, Begin))
      return createStringError(std::errc::invalid_argument,
                               "Failed to parse number: '%s'",
                               Matches[1].str().c_str());

    if (!Matches[3].empty()) {
      // Range format "begin-end"
      if (Matches[3].getAsInteger(10, End))
        return createStringError(std::errc::invalid_argument,
                                 "Failed to parse number: '%s'",
                                 Matches[3].str().c_str());
      if (Begin >= End)
        return createStringError(std::errc::invalid_argument,
                                 "Invalid range: %lld >= %lld", Begin, End);
    } else
      // Single number
      End = Begin;

    // Check ordering constraint (ranges must be in increasing order)
    if (!Ranges.empty() && Begin <= Ranges.back().End)
      return createStringError(
          std::errc::invalid_argument,
          "Expected ranges to be in increasing order: %lld <= %lld", Begin,
          Ranges.back().End);

    Ranges.push_back(Range(Begin, End));
  }

  return Ranges;
}

bool RangeUtils::contains(ArrayRef<Range> Ranges, int64_t Value) {
  for (const Range &R : Ranges) {
    if (R.contains(Value))
      return true;
  }
  return false;
}

void RangeUtils::printRanges(raw_ostream &OS, ArrayRef<Range> Ranges) {
  if (Ranges.empty())
    OS << "empty";
  else {
    bool IsFirst = true;
    for (const Range &R : Ranges) {
      if (!IsFirst)
        OS << ':';
      else
        IsFirst = false;

      if (R.Begin == R.End)
        OS << R.Begin;
      else
        OS << R.Begin << "-" << R.End;
    }
  }
}

RangeUtils::RangeList RangeUtils::mergeAdjacentRanges(ArrayRef<Range> Ranges) {
  if (Ranges.empty())
    return {};

  RangeList Result;
  Result.push_back(Ranges[0]);

  for (size_t I = 1; I < Ranges.size(); ++I) {
    const Range &Current = Ranges[I];
    Range &Last = Result.back();

    // Check if current range is adjacent to the last merged range
    if (Current.Begin == Last.End + 1) {
      // Merge by extending the end of the last range
      Last.End = Current.End;
    } else {
      // Not adjacent, add as separate range
      Result.push_back(Current);
    }
  }

  return Result;
}
