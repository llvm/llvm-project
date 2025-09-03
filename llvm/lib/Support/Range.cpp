//===- llvm/Support/Range.cpp - Range parsing utility ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Range.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"
#include <sstream>

using namespace llvm;

bool RangeUtils::parseRanges(const StringRef Str, RangeList &Ranges, const char Separator) {
  Ranges.clear();
  
  if (Str.empty())
    return true;
  
  // Split by the specified separator
  SmallVector<StringRef, 8> Parts;
  Str.split(Parts, Separator, -1, false);
  
  // Regex to match either single number or range "num1-num2"
  const Regex RangeRegex("^([0-9]+)(-([0-9]+))?$");
  
  for (StringRef Part : Parts) {
    Part = Part.trim();
    if (Part.empty())
      continue;
      
    SmallVector<StringRef, 4> Matches;
    if (!RangeRegex.match(Part, &Matches)) {
      errs() << "Invalid range format: '" << Part << "'\n";
      return false;
    }
    
    int64_t Begin, End;
    if (Matches[1].getAsInteger(10, Begin)) {
      errs() << "Failed to parse number: '" << Matches[1] << "'\n";
      return false;
    }
    
    if (!Matches[3].empty()) {
      // Range format "begin-end"
      if (Matches[3].getAsInteger(10, End)) {
        errs() << "Failed to parse number: '" << Matches[3] << "'\n";
        return false;
      }
      if (Begin >= End) {
        errs() << "Invalid range: " << Begin << " >= " << End << "\n";
        return false;
      }
    } else {
      // Single number
      End = Begin;
    }
    
    // Check ordering constraint (ranges must be in increasing order)
    if (!Ranges.empty() && Begin <= Ranges.back().End) {
      errs() << "Expected ranges to be in increasing order: " << Begin
             << " <= " << Ranges.back().End << "\n";
      return false;
    }
    
    Ranges.push_back(Range(Begin, End));
  }
  
  return true;
}

bool RangeUtils::contains(const ArrayRef<Range> Ranges, const int64_t Value) {
  for (const Range &R : Ranges) {
    if (R.contains(Value))
      return true;
  }
  return false;
}



std::string RangeUtils::rangesToString(const ArrayRef<Range> Ranges, const char Separator) {
  std::ostringstream OS;
  for (size_t I = 0; I < Ranges.size(); ++I) {
    if (I > 0)
      OS << Separator;
    const Range &R = Ranges[I];
    if (R.Begin == R.End) {
      OS << R.Begin;
    } else {
      OS << R.Begin << "-" << R.End;
    }
  }
  return OS.str();
}

void RangeUtils::printRanges(raw_ostream &OS, const ArrayRef<Range> Ranges) {
  if (Ranges.empty()) {
    OS << "empty";
  } else {
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

RangeUtils::RangeList RangeUtils::mergeAdjacentRanges(const ArrayRef<Range> Ranges) {
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
