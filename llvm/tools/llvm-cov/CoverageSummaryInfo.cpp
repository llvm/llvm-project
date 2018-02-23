//===- CoverageSummaryInfo.cpp - Coverage summary for function/file -------===//
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

#include "CoverageSummaryInfo.h"

using namespace llvm;
using namespace coverage;

FunctionCoverageSummary
FunctionCoverageSummary::get(const coverage::FunctionRecord &Function) {
  // Compute the region coverage.
  size_t NumCodeRegions = 0, CoveredRegions = 0;
  for (auto &CR : Function.CountedRegions) {
    if (CR.Kind != CounterMappingRegion::CodeRegion)
      continue;
    ++NumCodeRegions;
    if (CR.ExecutionCount != 0)
      ++CoveredRegions;
  }

  // Compute the line coverage
  size_t NumLines = 0, CoveredLines = 0;
  for (unsigned FileID = 0, E = Function.Filenames.size(); FileID < E;
       ++FileID) {
    // Find the line start and end of the function's source code
    // in that particular file
    unsigned LineStart = std::numeric_limits<unsigned>::max();
    unsigned LineEnd = 0;
    for (auto &CR : Function.CountedRegions) {
      if (CR.FileID != FileID)
        continue;
      LineStart = std::min(LineStart, CR.LineStart);
      LineEnd = std::max(LineEnd, CR.LineEnd);
    }
    unsigned LineCount = LineEnd - LineStart + 1;

    // Get counters
    llvm::SmallVector<uint64_t, 16> ExecutionCounts;
    ExecutionCounts.resize(LineCount, 0);
    for (auto &CR : Function.CountedRegions) {
      if (CR.FileID != FileID)
        continue;
      // Ignore the lines that were skipped by the preprocessor.
      auto ExecutionCount = CR.ExecutionCount;
      if (CR.Kind == CounterMappingRegion::SkippedRegion) {
        LineCount -= CR.LineEnd - CR.LineStart + 1;
        ExecutionCount = 1;
      }
      for (unsigned I = CR.LineStart; I <= CR.LineEnd; ++I)
        ExecutionCounts[I - LineStart] = ExecutionCount;
    }
    CoveredLines += LineCount - std::count(ExecutionCounts.begin(),
                                           ExecutionCounts.end(), 0);
    NumLines += LineCount;
  }
  return FunctionCoverageSummary(
      Function.Name, Function.ExecutionCount,
      RegionCoverageInfo(CoveredRegions, NumCodeRegions),
      LineCoverageInfo(CoveredLines, NumLines));
}

void FunctionCoverageSummary::update(const FunctionCoverageSummary &Summary) {
  ExecutionCount += Summary.ExecutionCount;
  RegionCoverage.Covered =
      std::max(RegionCoverage.Covered, Summary.RegionCoverage.Covered);
  RegionCoverage.NotCovered =
      std::min(RegionCoverage.NotCovered, Summary.RegionCoverage.NotCovered);
  LineCoverage.Covered =
      std::max(LineCoverage.Covered, Summary.LineCoverage.Covered);
  LineCoverage.NotCovered =
      std::min(LineCoverage.NotCovered, Summary.LineCoverage.NotCovered);
}
