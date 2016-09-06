//===- CoverageSummaryInfo.h - Coverage summary for function/file ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// These structures are used to represent code coverage metrics
// for functions/files.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_COV_COVERAGESUMMARYINFO_H
#define LLVM_COV_COVERAGESUMMARYINFO_H

#include "llvm/ProfileData/Coverage/CoverageMapping.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

/// \brief Provides information about region coverage for a function/file.
struct RegionCoverageInfo {
  /// \brief The number of regions that were executed at least once.
  size_t Covered;

  /// \brief The number of regions that weren't executed.
  size_t NotCovered;

  /// \brief The total number of regions in a function/file.
  size_t NumRegions;

  RegionCoverageInfo() : Covered(0), NotCovered(0), NumRegions(0) {}

  RegionCoverageInfo(size_t Covered, size_t NumRegions)
      : Covered(Covered), NotCovered(NumRegions - Covered),
        NumRegions(NumRegions) {}

  RegionCoverageInfo &operator+=(const RegionCoverageInfo &RHS) {
    Covered += RHS.Covered;
    NotCovered += RHS.NotCovered;
    NumRegions += RHS.NumRegions;
    return *this;
  }

  bool isFullyCovered() const { return Covered == NumRegions; }

  double getPercentCovered() const {
    if (NumRegions == 0)
      return 0.0;
    return double(Covered) / double(NumRegions) * 100.0;
  }
};

/// \brief Provides information about line coverage for a function/file.
struct LineCoverageInfo {
  /// \brief The number of lines that were executed at least once.
  size_t Covered;

  /// \brief The number of lines that weren't executed.
  size_t NotCovered;

  /// \brief The number of lines that aren't code.
  size_t NonCodeLines;

  /// \brief The total number of lines in a function/file.
  size_t NumLines;

  LineCoverageInfo()
      : Covered(0), NotCovered(0), NonCodeLines(0), NumLines(0) {}

  LineCoverageInfo(size_t Covered, size_t NumNonCodeLines, size_t NumLines)
      : Covered(Covered), NotCovered(NumLines - NumNonCodeLines - Covered),
        NonCodeLines(NumNonCodeLines), NumLines(NumLines) {}

  LineCoverageInfo &operator+=(const LineCoverageInfo &RHS) {
    Covered += RHS.Covered;
    NotCovered += RHS.NotCovered;
    NonCodeLines += RHS.NonCodeLines;
    NumLines += RHS.NumLines;
    return *this;
  }

  bool isFullyCovered() const { return Covered == (NumLines - NonCodeLines); }

  double getPercentCovered() const {
    if (NumLines - NonCodeLines == 0)
      return 0.0;
    return double(Covered) / double(NumLines - NonCodeLines) * 100.0;
  }
};

/// \brief Provides information about function coverage for a file.
struct FunctionCoverageInfo {
  /// \brief The number of functions that were executed.
  size_t Executed;

  /// \brief The total number of functions in this file.
  size_t NumFunctions;

  FunctionCoverageInfo() : Executed(0), NumFunctions(0) {}

  FunctionCoverageInfo(size_t Executed, size_t NumFunctions)
      : Executed(Executed), NumFunctions(NumFunctions) {}

  void addFunction(bool Covered) {
    if (Covered)
      ++Executed;
    ++NumFunctions;
  }

  bool isFullyCovered() const { return Executed == NumFunctions; }

  double getPercentCovered() const {
    if (NumFunctions == 0)
      return 0.0;
    return double(Executed) / double(NumFunctions) * 100.0;
  }
};

/// \brief A summary of function's code coverage.
struct FunctionCoverageSummary {
  StringRef Name;
  uint64_t ExecutionCount;
  RegionCoverageInfo RegionCoverage;
  LineCoverageInfo LineCoverage;

  FunctionCoverageSummary(StringRef Name) : Name(Name), ExecutionCount(0) {}

  FunctionCoverageSummary(StringRef Name, uint64_t ExecutionCount,
                          const RegionCoverageInfo &RegionCoverage,
                          const LineCoverageInfo &LineCoverage)
      : Name(Name), ExecutionCount(ExecutionCount),
        RegionCoverage(RegionCoverage), LineCoverage(LineCoverage) {
  }

  /// \brief Compute the code coverage summary for the given function coverage
  /// mapping record.
  static FunctionCoverageSummary
  get(const coverage::FunctionRecord &Function);
};

/// \brief A summary of file's code coverage.
struct FileCoverageSummary {
  StringRef Name;
  RegionCoverageInfo RegionCoverage;
  LineCoverageInfo LineCoverage;
  FunctionCoverageInfo FunctionCoverage;

  FileCoverageSummary(StringRef Name) : Name(Name) {}

  void addFunction(const FunctionCoverageSummary &Function) {
    RegionCoverage += Function.RegionCoverage;
    LineCoverage += Function.LineCoverage;
    FunctionCoverage.addFunction(/*Covered=*/Function.ExecutionCount > 0);
  }
};

} // namespace llvm

#endif // LLVM_COV_COVERAGESUMMARYINFO_H
