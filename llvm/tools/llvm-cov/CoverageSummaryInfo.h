//===- CoverageSummaryInfo.h - Coverage summary for function/file ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

/// Provides information about region coverage for a function/file.
class RegionCoverageInfo {
  /// The number of regions that were executed at least once.
  size_t Covered;

  /// The total number of regions in a function/file.
  size_t NumRegions;

public:
  RegionCoverageInfo() : Covered(0), NumRegions(0) {}

  RegionCoverageInfo(size_t Covered, size_t NumRegions)
      : Covered(Covered), NumRegions(NumRegions) {
    assert(Covered <= NumRegions && "Covered regions over-counted");
  }

  RegionCoverageInfo &operator+=(const RegionCoverageInfo &RHS) {
    Covered += RHS.Covered;
    NumRegions += RHS.NumRegions;
    return *this;
  }

  size_t getCovered() const { return Covered; }

  size_t getNumRegions() const { return NumRegions; }

  bool isFullyCovered() const { return Covered == NumRegions; }

  double getPercentCovered() const {
    assert(Covered <= NumRegions && "Covered regions over-counted");
    if (NumRegions == 0)
      return 0.0;
    return double(Covered) / double(NumRegions) * 100.0;
  }
};

/// Provides information about line coverage for a function/file.
class LineCoverageInfo {
  /// The number of lines that were executed at least once.
  size_t Covered;

  /// The total number of lines in a function/file.
  size_t NumLines;

public:
  LineCoverageInfo() : Covered(0), NumLines(0) {}

  LineCoverageInfo(size_t Covered, size_t NumLines)
      : Covered(Covered), NumLines(NumLines) {
    assert(Covered <= NumLines && "Covered lines over-counted");
  }

  LineCoverageInfo &operator+=(const LineCoverageInfo &RHS) {
    Covered += RHS.Covered;
    NumLines += RHS.NumLines;
    return *this;
  }

  size_t getCovered() const { return Covered; }

  size_t getNumLines() const { return NumLines; }

  bool isFullyCovered() const { return Covered == NumLines; }

  double getPercentCovered() const {
    assert(Covered <= NumLines && "Covered lines over-counted");
    if (NumLines == 0)
      return 0.0;
    return double(Covered) / double(NumLines) * 100.0;
  }
};

/// Provides information about branches coverage for a function/file.
class BranchCoverageInfo {
  /// The number of branches that were executed at least once.
  size_t Covered;

  /// The total number of branches in a function/file.
  size_t NumBranches;

public:
  BranchCoverageInfo() : Covered(0), NumBranches(0) {}

  BranchCoverageInfo(size_t Covered, size_t NumBranches)
      : Covered(Covered), NumBranches(NumBranches) {
    assert(Covered <= NumBranches && "Covered branches over-counted");
  }

  BranchCoverageInfo &operator+=(const BranchCoverageInfo &RHS) {
    Covered += RHS.Covered;
    NumBranches += RHS.NumBranches;
    return *this;
  }

  size_t getCovered() const { return Covered; }

  size_t getNumBranches() const { return NumBranches; }

  bool isFullyCovered() const { return Covered == NumBranches; }

  double getPercentCovered() const {
    assert(Covered <= NumBranches && "Covered branches over-counted");
    if (NumBranches == 0)
      return 0.0;
    return double(Covered) / double(NumBranches) * 100.0;
  }
};

/// Provides information about MC/DC coverage for a function/file.
class MCDCCoverageInfo {
  /// The number of Independence Pairs that were covered.
  size_t CoveredPairs;

  /// The total number of Independence Pairs in a function/file.
  size_t NumPairs;

public:
  MCDCCoverageInfo() : CoveredPairs(0), NumPairs(0) {}

  MCDCCoverageInfo(size_t CoveredPairs, size_t NumPairs)
      : CoveredPairs(CoveredPairs), NumPairs(NumPairs) {
    assert(CoveredPairs <= NumPairs && "Covered pairs over-counted");
  }

  MCDCCoverageInfo &operator+=(const MCDCCoverageInfo &RHS) {
    CoveredPairs += RHS.CoveredPairs;
    NumPairs += RHS.NumPairs;
    return *this;
  }

  size_t getCoveredPairs() const { return CoveredPairs; }

  size_t getNumPairs() const { return NumPairs; }

  bool isFullyCovered() const { return CoveredPairs == NumPairs; }

  double getPercentCovered() const {
    assert(CoveredPairs <= NumPairs && "Covered pairs over-counted");
    if (NumPairs == 0)
      return 0.0;
    return double(CoveredPairs) / double(NumPairs) * 100.0;
  }
};

/// Provides information about function coverage for a file.
class FunctionCoverageInfo {
  /// The number of functions that were executed.
  size_t Executed;

  /// The total number of functions in this file.
  size_t NumFunctions;

public:
  FunctionCoverageInfo() : Executed(0), NumFunctions(0) {}

  FunctionCoverageInfo(size_t Executed, size_t NumFunctions)
      : Executed(Executed), NumFunctions(NumFunctions) {}

  FunctionCoverageInfo &operator+=(const FunctionCoverageInfo &RHS) {
    Executed += RHS.Executed;
    NumFunctions += RHS.NumFunctions;
    return *this;
  }

  void addFunction(bool Covered) {
    if (Covered)
      ++Executed;
    ++NumFunctions;
  }

  size_t getExecuted() const { return Executed; }

  size_t getNumFunctions() const { return NumFunctions; }

  bool isFullyCovered() const { return Executed == NumFunctions; }

  double getPercentCovered() const {
    assert(Executed <= NumFunctions && "Covered functions over-counted");
    if (NumFunctions == 0)
      return 0.0;
    return double(Executed) / double(NumFunctions) * 100.0;
  }
};

struct CoverageDataSummary {
  RegionCoverageInfo RegionCoverage;
  LineCoverageInfo LineCoverage;
  BranchCoverageInfo BranchCoverage;
  MCDCCoverageInfo MCDCCoverage;

  CoverageDataSummary() = default;
  CoverageDataSummary(const coverage::CoverageData &CD);

  bool empty() const {
    return (RegionCoverage.getNumRegions() == 0 &&
            LineCoverage.getNumLines() == 0 &&
            BranchCoverage.getNumBranches() == 0 &&
            MCDCCoverage.getNumPairs() == 0);
  }

  auto &operator+=(const CoverageDataSummary &RHS) {
    RegionCoverage += RHS.RegionCoverage;
    LineCoverage += RHS.LineCoverage;
    BranchCoverage += RHS.BranchCoverage;
    MCDCCoverage += RHS.MCDCCoverage;
    return *this;
  }
};

/// A summary of function's code coverage.
struct FunctionCoverageSummary : CoverageDataSummary {
  std::string Name;
  uint64_t ExecutionCount;

  FunctionCoverageSummary(const std::string &Name, uint64_t ExecutionCount = 0)
      : Name(Name), ExecutionCount(ExecutionCount) {}

  /// Compute the code coverage summary for the given function coverage
  /// mapping record.
  static FunctionCoverageSummary get(const coverage::CoverageMapping &CM,
                                     const coverage::FunctionRecord &Function);
};

/// A summary of file's code coverage.
struct FileCoverageSummary : CoverageDataSummary {
  StringRef Name;
  FunctionCoverageInfo FunctionCoverage;
  FunctionCoverageInfo InstantiationCoverage;

  FileCoverageSummary() = default;
  FileCoverageSummary(StringRef Name) : Name(Name) {}

  bool empty() const {
    return (CoverageDataSummary::empty() &&
            FunctionCoverage.getNumFunctions() == 0 &&
            InstantiationCoverage.getNumFunctions() == 0);
  }

  FileCoverageSummary &operator+=(const FileCoverageSummary &RHS) {
    *static_cast<CoverageDataSummary *>(this) += RHS;
    FunctionCoverage += RHS.FunctionCoverage;
    InstantiationCoverage += RHS.InstantiationCoverage;
    return *this;
  }
};

/// A cache for demangled symbols.
struct DemangleCache {
  StringMap<std::string> DemangledNames;

  /// Demangle \p Sym if possible. Otherwise, just return \p Sym.
  StringRef demangle(StringRef Sym) const {
    const auto DemangledName = DemangledNames.find(Sym);
    if (DemangledName == DemangledNames.end())
      return Sym;
    return DemangledName->getValue();
  }
};

} // namespace llvm

#endif // LLVM_COV_COVERAGESUMMARYINFO_H
