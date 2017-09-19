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

LineCoverageStats::LineCoverageStats(
    ArrayRef<const coverage::CoverageSegment *> LineSegments,
    const coverage::CoverageSegment *WrappedSegment) {
  // Find the minimum number of regions which start in this line.
  unsigned MinRegionCount = 0;
  auto isStartOfRegion = [](const coverage::CoverageSegment *S) {
    return !S->IsGapRegion && S->HasCount && S->IsRegionEntry;
  };
  for (unsigned I = 0; I < LineSegments.size() && MinRegionCount < 2; ++I)
    if (isStartOfRegion(LineSegments[I]))
      ++MinRegionCount;

  bool StartOfSkippedRegion = !LineSegments.empty() &&
                              !LineSegments.front()->HasCount &&
                              LineSegments.front()->IsRegionEntry;

  ExecutionCount = 0;
  HasMultipleRegions = MinRegionCount > 1;
  Mapped =
      !StartOfSkippedRegion &&
      ((WrappedSegment && WrappedSegment->HasCount) || (MinRegionCount > 0));

  if (!Mapped)
    return;

  // Pick the max count among regions which start and end on this line, to
  // avoid erroneously using the wrapped count, and to avoid picking region
  // counts which come from deferred regions.
  if (LineSegments.size() > 1) {
    for (unsigned I = 0; I < LineSegments.size() - 1; ++I) {
      if (!LineSegments[I]->IsGapRegion)
        ExecutionCount = std::max(ExecutionCount, LineSegments[I]->Count);
    }
    return;
  }

  // If a non-gap region starts here, use its count. Otherwise use the wrapped
  // count.
  if (MinRegionCount == 1)
    ExecutionCount = LineSegments[0]->Count;
  else
    ExecutionCount = WrappedSegment->Count;
}

FunctionCoverageSummary
FunctionCoverageSummary::get(const CoverageMapping &CM,
                             const coverage::FunctionRecord &Function) {
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
  CoverageData CD = CM.getCoverageForFunction(Function);
  auto NextSegment = CD.begin();
  auto EndSegment = CD.end();
  const coverage::CoverageSegment *WrappedSegment = nullptr;
  SmallVector<const coverage::CoverageSegment *, 4> LineSegments;
  unsigned Line = NextSegment->Line;
  while (NextSegment != EndSegment) {
    // Gather the segments on this line and the wrapped segment.
    if (LineSegments.size())
      WrappedSegment = LineSegments.back();
    LineSegments.clear();
    while (NextSegment != EndSegment && NextSegment->Line == Line)
      LineSegments.push_back(&*NextSegment++);

    LineCoverageStats LCS{LineSegments, WrappedSegment};
    if (LCS.isMapped()) {
      ++NumLines;
      if (LCS.ExecutionCount)
        ++CoveredLines;
    }

    ++Line;
  }

  return FunctionCoverageSummary(
      Function.Name, Function.ExecutionCount,
      RegionCoverageInfo(CoveredRegions, NumCodeRegions),
      LineCoverageInfo(CoveredLines, NumLines));
}

FunctionCoverageSummary
FunctionCoverageSummary::get(const InstantiationGroup &Group,
                             ArrayRef<FunctionCoverageSummary> Summaries) {
  std::string Name;
  if (Group.hasName()) {
    Name = Group.getName();
  } else {
    llvm::raw_string_ostream OS(Name);
    OS << "Definition at line " << Group.getLine() << ", column "
       << Group.getColumn();
  }

  FunctionCoverageSummary Summary(Name);
  Summary.ExecutionCount = Group.getTotalExecutionCount();
  Summary.RegionCoverage = Summaries[0].RegionCoverage;
  Summary.LineCoverage = Summaries[0].LineCoverage;
  for (const auto &FCS : Summaries.drop_front()) {
    Summary.RegionCoverage.merge(FCS.RegionCoverage);
    Summary.LineCoverage.merge(FCS.LineCoverage);
  }
  return Summary;
}
