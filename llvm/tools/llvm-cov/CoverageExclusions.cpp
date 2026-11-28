//===- CoverageExclusions.cpp - Source coverage exclusions ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Reads LCOV exclusion markers and applies the resulting line ranges to
/// llvm-cov coverage data and function records at report time.
///
//===----------------------------------------------------------------------===//

#include "CoverageExclusions.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include <algorithm>
#include <cctype>
#include <limits>
#include <optional>
#include <tuple>

using namespace llvm;
using namespace coverage;

namespace {

constexpr StringLiteral ExcludeLineMarker = "LCOV_EXCL_LINE";
constexpr StringLiteral ExcludeStartMarker = "LCOV_EXCL_START";
constexpr StringLiteral ExcludeStopMarker = "LCOV_EXCL_STOP";

// Marker names must be complete tokens.
bool isMarkerBoundary(char C) {
  return !std::isalnum(static_cast<unsigned char>(C)) && C != '_';
}

// Search the whole line, skipping occurrences embedded in longer names.
bool containsMarker(StringRef Line, StringRef Marker) {
  size_t Pos = 0;
  while ((Pos = Line.find(Marker, Pos)) != StringRef::npos) {
    bool StartBoundary = Pos == 0 || isMarkerBoundary(Line[Pos - 1]);
    size_t End = Pos + Marker.size();
    bool EndBoundary = End == Line.size() || isMarkerBoundary(Line[End]);
    if (StartBoundary && EndBoundary)
      return true;
    Pos = End;
  }
  return false;
}

// Find the one possible containing range in a sorted, disjoint list.
const CoverageExclusions::LineRange *
findContainingRange(ArrayRef<CoverageExclusions::LineRange> Ranges,
                    unsigned Line) {
  auto It =
      llvm::upper_bound(Ranges, Line, [](unsigned Line, const auto &Range) {
        return Line < Range.first;
      });
  if (It == Ranges.begin() || Line > std::prev(It)->second)
    return nullptr;
  return &*std::prev(It);
}

// Coalesce overlaps and adjacent ranges before applying exclusions.
std::vector<CoverageExclusions::LineRange>
mergeLineRanges(std::vector<CoverageExclusions::LineRange> Ranges) {
  llvm::sort(Ranges);
  std::vector<CoverageExclusions::LineRange> Merged;
  for (const auto &[Begin, End] : Ranges) {
    if (Merged.empty() || Begin > Merged.back().second + 1)
      Merged.emplace_back(Begin, End);
    else
      Merged.back().second = std::max(Merged.back().second, End);
  }
  return Merged;
}

// Read single-line markers and non-nested blocks. A block includes its start
// marker line and ends immediately before its stop marker line.
Expected<std::vector<CoverageExclusions::LineRange>>
parseLineRanges(StringRef Source) {
  std::vector<CoverageExclusions::LineRange> Ranges;
  std::optional<unsigned> BlockStart;
  SmallVector<StringRef, 0> Lines;
  Source.split(Lines, '\n', /*MaxSplit=*/-1, /*KeepEmpty=*/true);

  for (unsigned I = 0; I < Lines.size(); ++I) {
    unsigned Line = I + 1;
    StringRef Text = Lines[I].rtrim("\r");
    if (containsMarker(Text, ExcludeStartMarker)) {
      if (BlockStart)
        return createStringError(
            inconvertibleErrorCode(),
            formatv("overlapping LCOV_EXCL_START at line {0}; previous block "
                    "started at line {1}",
                    Line, *BlockStart));
      BlockStart = Line;
      continue;
    }
    if (containsMarker(Text, ExcludeStopMarker)) {
      if (!BlockStart)
        return createStringError(inconvertibleErrorCode(),
                                 formatv("LCOV_EXCL_STOP at line {0} has no "
                                         "matching LCOV_EXCL_START",
                                         Line));
      if (*BlockStart < Line)
        Ranges.emplace_back(*BlockStart, Line - 1);
      BlockStart.reset();
    }
    if (containsMarker(Text, ExcludeLineMarker))
      Ranges.emplace_back(Line, Line);
  }

  if (BlockStart)
    return createStringError(
        inconvertibleErrorCode(),
        formatv("LCOV_EXCL_START at line {0} has no matching LCOV_EXCL_STOP",
                *BlockStart));

  return mergeLineRanges(std::move(Ranges));
}

/// Applies the ranges for one source file to an extracted coverage view.
class CoverageDataExclusionFilter : public CoverageData {
public:
  CoverageDataExclusionFilter(CoverageData &&Data,
                              ArrayRef<CoverageExclusions::LineRange> Ranges)
      : CoverageData(std::move(Data)), Ranges(Ranges) {}

  CoverageData apply() {
    filterSegments();
    filterRecords();
    return std::move(*this);
  }

private:
  enum class SegmentPriority : unsigned {
    Resume,
    Original,
    ExclusionStart,
  };

  struct OrderedSegment {
    CoverageSegment Segment;
    SegmentPriority Priority;
  };

  ArrayRef<CoverageExclusions::LineRange> Ranges;

  bool isExcluded(unsigned Line) const {
    return findContainingRange(Ranges, Line) != nullptr;
  }

  // End active coverage at the first excluded line and restore the preceding
  // count after the range unless a segment there already supplies a count.
  void addRangeBoundaries(const CoverageExclusions::LineRange &Range,
                          std::vector<CoverageSegment> &OriginalSegments,
                          std::vector<OrderedSegment> &FilteredSegments) const {
    const auto [Begin, End] = Range;
    FilteredSegments.push_back({CoverageSegment(Begin, 1,
                                                /*IsRegionEntry=*/true),
                                SegmentPriority::ExclusionStart});

    if (End == std::numeric_limits<unsigned>::max())
      return;

    LineColPair ResumeLoc(End + 1, 1);
    auto Resume =
        llvm::lower_bound(OriginalSegments, ResumeLoc,
                          [](const CoverageSegment &Segment, LineColPair Loc) {
                            return LineColPair(Segment.Line, Segment.Col) < Loc;
                          });
    if (Resume != OriginalSegments.end() &&
        LineColPair(Resume->Line, Resume->Col) == ResumeLoc) {
      auto ResumeEnd = std::upper_bound(
          Resume, OriginalSegments.end(), ResumeLoc,
          [](LineColPair Loc, const CoverageSegment &Segment) {
            return Loc < LineColPair(Segment.Line, Segment.Col);
          });
      auto CountedResume = llvm::find_if(
          make_range(Resume, ResumeEnd),
          [](const CoverageSegment &Segment) { return Segment.HasCount; });
      if (CountedResume != ResumeEnd)
        CountedResume->IsRegionEntry = true;
      return;
    }

    const CoverageSegment *Wrapped =
        Resume == OriginalSegments.begin() ? nullptr : &*std::prev(Resume);
    if (Wrapped && Wrapped->HasCount) {
      FilteredSegments.push_back(
          {CoverageSegment(ResumeLoc.first, ResumeLoc.second, Wrapped->Count,
                           /*IsRegionEntry=*/true, Wrapped->IsGapRegion),
           SegmentPriority::Resume});
      return;
    }

    FilteredSegments.push_back(
        {CoverageSegment(ResumeLoc.first, ResumeLoc.second,
                         /*IsRegionEntry=*/false),
         SegmentPriority::Resume});
  }

  // Drop the segments in excluded lines and install the boundary segments.
  void filterSegments() {
    std::vector<CoverageSegment> OriginalSegments = std::move(Segments);
    std::vector<OrderedSegment> FilteredSegments;
    FilteredSegments.reserve(OriginalSegments.size() + Ranges.size() * 2);

    for (const auto &Range : Ranges)
      addRangeBoundaries(Range, OriginalSegments, FilteredSegments);

    for (const CoverageSegment &Segment : OriginalSegments)
      if (!isExcluded(Segment.Line))
        FilteredSegments.push_back({Segment, SegmentPriority::Original});

    // At a shared location, resume the prior count before original segments
    // and end coverage after them.
    llvm::stable_sort(FilteredSegments, [](const OrderedSegment &LHS,
                                           const OrderedSegment &RHS) {
      return std::tie(LHS.Segment.Line, LHS.Segment.Col, LHS.Priority) <
             std::tie(RHS.Segment.Line, RHS.Segment.Col, RHS.Priority);
    });
    Segments.clear();
    for (const OrderedSegment &Entry : FilteredSegments)
      Segments.push_back(Entry.Segment);
  }

  // Remove expansion sites, branches and decisions anchored on excluded lines.
  void filterRecords() {
    std::vector<ExpansionRecord> FilteredExpansions;
    FilteredExpansions.reserve(Expansions.size());
    for (const ExpansionRecord &Expansion : Expansions)
      if (!isExcluded(Expansion.Region.LineStart))
        FilteredExpansions.push_back(Expansion);
    Expansions.swap(FilteredExpansions);

    llvm::erase_if(BranchRegions, [&](const CountedRegion &Region) {
      return isExcluded(Region.LineStart);
    });
    llvm::erase_if(MCDCRecords, [&](const MCDCRecord &Record) {
      return isExcluded(Record.getDecisionRegion().LineStart);
    });
  }
};

} // namespace

Error CoverageExclusions::parse(StringRef Filename, StringRef Source) {
  auto Ranges = parseLineRanges(Source);
  if (!Ranges)
    return Ranges.takeError();
  ExcludedLineRanges[Filename] = std::move(*Ranges);
  return Error::success();
}

ArrayRef<CoverageExclusions::LineRange>
CoverageExclusions::getRanges(StringRef Filename) const {
  auto It = ExcludedLineRanges.find(Filename);
  if (It == ExcludedLineRanges.end())
    return {};
  return It->second;
}

bool CoverageExclusions::isRegionExcluded(
    const FunctionRecord &Function, const CounterMappingRegion &Region) const {
  if (Region.FileID >= Function.Filenames.size())
    return false;
  const LineRange *Range = findContainingRange(
      getRanges(Function.Filenames[Region.FileID]), Region.LineStart);
  // Region end locations are exclusive, so column 1 of the next line does
  // not include any code on that line.
  return Range &&
         (Region.Kind != CounterMappingRegion::CodeRegion ||
          Region.LineEnd <= Range->second ||
          (Region.LineEnd - Range->second == 1 && Region.ColumnEnd == 1));
}

bool CoverageExclusions::isFunctionExcluded(const FunctionRecord &Function,
                                            StringRef Filename) const {
  if (ExcludedLineRanges.empty())
    return false;

  bool HasCodeRegion = false;
  for (const CountedRegion &Region : Function.CountedRegions) {
    if (Region.Kind != CounterMappingRegion::CodeRegion ||
        Region.FileID >= Function.Filenames.size() ||
        (!Filename.empty() && Function.Filenames[Region.FileID] != Filename))
      continue;
    HasCodeRegion = true;
    if (!isRegionExcluded(Function, Region))
      return false;
  }
  return HasCodeRegion;
}

CoverageData CoverageExclusions::apply(CoverageData Coverage) const {
  ArrayRef<LineRange> Ranges = getRanges(Coverage.getFilename());
  if (Ranges.empty())
    return Coverage;
  return CoverageDataExclusionFilter(std::move(Coverage), Ranges).apply();
}

CoverageData
CoverageExclusions::getCoverageForFile(const CoverageMapping &Mapping,
                                       StringRef Filename) const {
  return apply(Mapping.getCoverageForFile(Filename));
}

CoverageData CoverageExclusions::getCoverageForFunction(
    const CoverageMapping &Mapping, const FunctionRecord &Function) const {
  return apply(Mapping.getCoverageForFunction(Function));
}

CoverageData CoverageExclusions::getCoverageForExpansion(
    const CoverageMapping &Mapping, const ExpansionRecord &Expansion) const {
  return apply(Mapping.getCoverageForExpansion(Expansion));
}
