//===- CoverageExporterJson.cpp - Code coverage export --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements export of code coverage data to JSON.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//
// The json code coverage export follows the following format
// Root: dict => Root Element containing metadata
// -- Data: array => Homogeneous array of one or more export objects
//   -- Export: dict => Json representation of one CoverageMapping
//     -- Files: array => List of objects describing coverage for files
//       -- File: dict => Coverage for a single file
//         -- Branches: array => List of Branches in the file
//           -- Branch: dict => Describes a branch of the file with counters
//         -- MCDC Records: array => List of MCDC records in the file
//           -- MCDC Values: array => List of T/F covered condition values and
//           list of test vectors with execution status
//         -- Segments: array => List of Segments contained in the file
//           -- Segment: dict => Describes a segment of the file with a counter
//         -- Expansions: array => List of expansion records
//           -- Expansion: dict => Object that descibes a single expansion
//             -- CountedRegion: dict => The region to be expanded
//             -- TargetRegions: array => List of Regions in the expansion
//               -- CountedRegion: dict => Single Region in the expansion
//             -- Branches: array => List of Branches in the expansion
//               -- Branch: dict => Describes a branch in expansion and counters
//         -- Summary: dict => Object summarizing the coverage for this file
//           -- LineCoverage: dict => Object summarizing line coverage
//           -- FunctionCoverage: dict => Object summarizing function coverage
//           -- RegionCoverage: dict => Object summarizing region coverage
//           -- BranchCoverage: dict => Object summarizing branch coverage
//           -- MCDCCoverage: dict => Object summarizing MC/DC coverage
//     -- Functions: array => List of objects describing coverage for functions
//       -- Function: dict => Coverage info for a single function
//         -- Filenames: array => List of filenames that the function relates to
//   -- Summary: dict => Object summarizing the coverage for the entire binary
//     -- LineCoverage: dict => Object summarizing line coverage
//     -- FunctionCoverage: dict => Object summarizing function coverage
//     -- InstantiationCoverage: dict => Object summarizing inst. coverage
//     -- RegionCoverage: dict => Object summarizing region coverage
//     -- BranchCoverage: dict => Object summarizing branch coverage
//     -- MCDCCoverage: dict => Object summarizing MC/DC coverage
//
//===----------------------------------------------------------------------===//

#include "CoverageExporterJson.h"
#include "CoverageReport.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/Threading.h"
#include <algorithm>
#include <limits>
#include <numeric>
#include <utility>

/// The semantic version combined as a string.
#define LLVM_COVERAGE_EXPORT_JSON_STR "3.1.0"

/// Unique type identifier for JSON coverage export.
#define LLVM_COVERAGE_EXPORT_JSON_TYPE_STR "llvm.coverage.json.export"

using namespace llvm;

namespace {

// The JSON library accepts int64_t, but profiling counts are stored as uint64_t.
// Therefore we need to explicitly convert from unsigned to signed, since a naive
// cast is implementation-defined behavior when the unsigned value cannot be
// represented as a signed value. We choose to clamp the values to preserve the
// invariant that counts are always >= 0.
int64_t clamp_uint64_to_int64(uint64_t u) {
  return std::min(u, static_cast<uint64_t>(std::numeric_limits<int64_t>::max()));
}

void renderSegment(json::OStream &JOS,
                   const coverage::CoverageSegment &Segment) {
  JOS.array([&] {
    JOS.value(Segment.Line);
    JOS.value(Segment.Col);
    JOS.value(clamp_uint64_to_int64(Segment.Count));
    JOS.value(Segment.HasCount);
    JOS.value(Segment.IsRegionEntry);
    JOS.value(Segment.IsGapRegion);
  });
}

void renderRegion(json::OStream &JOS, const coverage::CountedRegion &Region) {
  JOS.array([&] {
    JOS.value(Region.LineStart);
    JOS.value(Region.ColumnStart);
    JOS.value(Region.LineEnd);
    JOS.value(Region.ColumnEnd);
    JOS.value(clamp_uint64_to_int64(Region.ExecutionCount));
    JOS.value(Region.FileID);
    JOS.value(Region.ExpandedFileID);
    JOS.value(int64_t(Region.Kind));
  });
}

void renderBranch(json::OStream &JOS, const coverage::CountedRegion &Region) {
  JOS.array([&] {
    JOS.value(Region.LineStart);
    JOS.value(Region.ColumnStart);
    JOS.value(Region.LineEnd);
    JOS.value(Region.ColumnEnd);
    JOS.value(clamp_uint64_to_int64(Region.ExecutionCount));
    JOS.value(clamp_uint64_to_int64(Region.FalseExecutionCount));
    JOS.value(Region.FileID);
    JOS.value(Region.ExpandedFileID);
    JOS.value(int64_t(Region.Kind));
  });
}

void gatherConditions(json::OStream &JOS, const coverage::MCDCRecord &Record) {
  JOS.array([&] {
    for (unsigned c = 0; c < Record.getNumConditions(); c++)
      JOS.value(Record.isConditionIndependencePairCovered(c));
  });
}

void renderCondState(json::OStream &JOS,
                     const coverage::MCDCRecord::CondState CondState) {
  switch (CondState) {
  case coverage::MCDCRecord::MCDC_DontCare:
    JOS.value(nullptr);
    return;
  case coverage::MCDCRecord::MCDC_True:
    JOS.value(true);
    return;
  case coverage::MCDCRecord::MCDC_False:
    JOS.value(false);
    return;
  }
  llvm_unreachable("Unknown llvm::coverage::MCDCRecord::CondState enum");
}

void gatherTestVectors(json::OStream &JOS, coverage::MCDCRecord &Record,
                       const CoverageViewOptions &Options) {
  unsigned NumConditions = Record.getNumConditions();
  const bool ShowNonExecutedVectors = Options.ShowMCDCNonExecutedVectors;

  JOS.array([&] {
    for (unsigned tv = 0; tv < Record.getNumTestVectors(); tv++) {
      JOS.object([&] {
        JOS.attributeArray("conditions", [&] {
          for (unsigned c = 0; c < NumConditions; c++)
            renderCondState(JOS, Record.getTVCondition(tv, c));
        });

        JOS.attribute("executed", true);

        JOS.attributeBegin("result");
        renderCondState(JOS, Record.getTVResult(tv));
        JOS.attributeEnd();
      });
    }

    if (ShowNonExecutedVectors) {
      for (unsigned tv = 0; tv < Record.getNumNotExecutedTestVectors(); tv++) {
        JOS.object([&] {
          JOS.attributeArray("conditions", [&] {
            for (unsigned c = 0; c < NumConditions; c++)
              renderCondState(JOS, Record.getNotExecutedTVCondition(tv, c));
          });

          JOS.attribute("executed", false);

          JOS.attributeBegin("result");
          renderCondState(JOS, Record.getNotExecutedTVResult(tv));
          JOS.attributeEnd();
        });
      }
    }
  });
}

void renderMCDCRecord(json::OStream &JOS, const coverage::MCDCRecord &Record,
                      const CoverageViewOptions &Options) {
  const llvm::coverage::CounterMappingRegion &CMR = Record.getDecisionRegion();
  const auto [TrueDecisions, FalseDecisions] = Record.getDecisions();
  JOS.array([&, TrueDecisions = TrueDecisions,
             FalseDecisions = FalseDecisions] {
    JOS.value(CMR.LineStart);
    JOS.value(CMR.ColumnStart);
    JOS.value(CMR.LineEnd);
    JOS.value(CMR.ColumnEnd);
    JOS.value(TrueDecisions);
    JOS.value(FalseDecisions);
    JOS.value(CMR.FileID);
    JOS.value(CMR.ExpandedFileID);
    JOS.value(int64_t(CMR.Kind));
    gatherConditions(JOS, Record);
    gatherTestVectors(JOS, const_cast<coverage::MCDCRecord &>(Record), Options);
  });
}

void renderRegions(json::OStream &JOS,
                   ArrayRef<coverage::CountedRegion> Regions) {
  JOS.array([&] {
    for (const auto &Region : Regions)
      renderRegion(JOS, Region);
  });
}

void renderBranchRegions(json::OStream &JOS,
                         ArrayRef<coverage::CountedRegion> Regions) {
  JOS.array([&] {
    for (const auto &Region : Regions)
      if (!Region.TrueFolded || !Region.FalseFolded)
        renderBranch(JOS, Region);
  });
}

void renderMCDCRecords(json::OStream &JOS,
                       ArrayRef<coverage::MCDCRecord> Records,
                       const CoverageViewOptions &Options) {
  JOS.array([&] {
    for (auto &Record : Records)
      renderMCDCRecord(JOS, Record, Options);
  });
}

std::vector<llvm::coverage::CountedRegion>
collectNestedBranches(const coverage::CoverageMapping &Coverage,
                      ArrayRef<llvm::coverage::ExpansionRecord> Expansions) {
  std::vector<llvm::coverage::CountedRegion> Branches;
  for (const auto &Expansion : Expansions) {
    auto ExpansionCoverage = Coverage.getCoverageForExpansion(Expansion);

    // Recursively collect branches from nested expansions.
    auto NestedExpansions = ExpansionCoverage.getExpansions();
    auto NestedExBranches = collectNestedBranches(Coverage, NestedExpansions);
    append_range(Branches, NestedExBranches);

    // Add branches from this level of expansion.
    auto ExBranches = ExpansionCoverage.getBranches();
    for (auto B : ExBranches)
      if (B.FileID == Expansion.FileID)
        Branches.push_back(B);
  }

  return Branches;
}

void renderExpansion(json::OStream &JOS,
                     const coverage::CoverageMapping &Coverage,
                     const coverage::ExpansionRecord &Expansion) {
  std::vector<llvm::coverage::ExpansionRecord> Expansions = {Expansion};
  JOS.object([&] {
    JOS.attributeArray("filenames", [&] {
      for (const auto &Filename : Expansion.Function.Filenames)
        JOS.value(Filename);
    });
    // Enumerate the branch coverage information for the expansion.
    JOS.attributeBegin("branches");
    renderBranchRegions(JOS, collectNestedBranches(Coverage, Expansions));
    JOS.attributeEnd();
    // Mark the beginning and end of this expansion in the source file.
    JOS.attributeBegin("source_region");
    renderRegion(JOS, Expansion.Region);
    JOS.attributeEnd();
    // Enumerate the coverage information for the expansion.
    JOS.attributeBegin("target_regions");
    renderRegions(JOS, Expansion.Function.CountedRegions);
    JOS.attributeEnd();
  });
}

void renderSummary(json::OStream &JOS, const FileCoverageSummary &Summary) {
  JOS.object([&] {
    JOS.attributeObject("lines", [&] {
      JOS.attribute("count", int64_t(Summary.LineCoverage.getNumLines()));
      JOS.attribute("covered", int64_t(Summary.LineCoverage.getCovered()));
      JOS.attribute("percent", Summary.LineCoverage.getPercentCovered());
    });
    JOS.attributeObject("functions", [&] {
      JOS.attribute("count",
                    int64_t(Summary.FunctionCoverage.getNumFunctions()));
      JOS.attribute("covered", int64_t(Summary.FunctionCoverage.getExecuted()));
      JOS.attribute("percent", Summary.FunctionCoverage.getPercentCovered());
    });
    JOS.attributeObject("instantiations", [&] {
      JOS.attribute("count",
                    int64_t(Summary.InstantiationCoverage.getNumFunctions()));
      JOS.attribute("covered",
                    int64_t(Summary.InstantiationCoverage.getExecuted()));
      JOS.attribute("percent",
                    Summary.InstantiationCoverage.getPercentCovered());
    });
    JOS.attributeObject("regions", [&] {
      JOS.attribute("count", int64_t(Summary.RegionCoverage.getNumRegions()));
      JOS.attribute("covered", int64_t(Summary.RegionCoverage.getCovered()));
      JOS.attribute("notcovered",
                    int64_t(Summary.RegionCoverage.getNumRegions() -
                            Summary.RegionCoverage.getCovered()));
      JOS.attribute("percent", Summary.RegionCoverage.getPercentCovered());
    });
    JOS.attributeObject("branches", [&] {
      JOS.attribute("count", int64_t(Summary.BranchCoverage.getNumBranches()));
      JOS.attribute("covered", int64_t(Summary.BranchCoverage.getCovered()));
      JOS.attribute("notcovered",
                    int64_t(Summary.BranchCoverage.getNumBranches() -
                            Summary.BranchCoverage.getCovered()));
      JOS.attribute("percent", Summary.BranchCoverage.getPercentCovered());
    });
    JOS.attributeObject("mcdc", [&] {
      JOS.attribute("count", int64_t(Summary.MCDCCoverage.getNumPairs()));
      JOS.attribute("covered", int64_t(Summary.MCDCCoverage.getCoveredPairs()));
      JOS.attribute("notcovered",
                    int64_t(Summary.MCDCCoverage.getNumPairs() -
                            Summary.MCDCCoverage.getCoveredPairs()));
      JOS.attribute("percent", Summary.MCDCCoverage.getPercentCovered());
    });
  });
}

void renderFile(json::OStream &JOS, const coverage::CoverageMapping &Coverage,
                const std::string &Filename,
                const FileCoverageSummary &FileReport,
                const CoverageViewOptions &Options) {
  JOS.object([&] {
    JOS.attribute("filename", Filename);
    if (!Options.ExportSummaryOnly) {
      // Calculate and render detailed coverage information for given file.
      auto FileCoverage = Coverage.getCoverageForFile(Filename);
      JOS.attributeArray("branches", [&] {
        for (const auto &Branch : FileCoverage.getBranches())
          renderBranch(JOS, Branch);
      });
      if (!Options.SkipExpansions) {
        JOS.attributeArray("expansions", [&] {
          for (const auto &Expansion : FileCoverage.getExpansions())
            renderExpansion(JOS, Coverage, Expansion);
        });
      }
      JOS.attributeArray("mcdc_records", [&] {
        for (const auto &Record : FileCoverage.getMCDCRecords())
          renderMCDCRecord(JOS, Record, Options);
      });
      JOS.attributeArray("segments", [&] {
        for (const auto &Segment : FileCoverage)
          renderSegment(JOS, Segment);
      });
    }
    JOS.attributeBegin("summary");
    renderSummary(JOS, FileReport);
    JOS.attributeEnd();
  });
}

void renderFiles(json::OStream &JOS, const coverage::CoverageMapping &Coverage,
                 ArrayRef<std::string> SourceFiles,
                 ArrayRef<FileCoverageSummary> FileReports,
                 const CoverageViewOptions &Options) {
  ThreadPoolStrategy S = hardware_concurrency(Options.NumThreads);
  if (Options.NumThreads == 0) {
    // If NumThreads is not specified, create one thread for each input, up to
    // the number of hardware cores.
    S = heavyweight_hardware_concurrency(SourceFiles.size());
    S.Limit = true;
  }

  // Pre-render coverage for each file to separate string.
  std::vector<std::string> RenderedFiles(SourceFiles.size());
  DefaultThreadPool Pool(S);

  for (unsigned I = 0, E = SourceFiles.size(); I < E; ++I) {
    auto &SourceFile = SourceFiles[I];
    auto &FileReport = FileReports[I];
    Pool.async([&, I] {
      std::string Buffer;
      llvm::raw_string_ostream RawSStream(Buffer);
      json::OStream JOS(RawSStream);
      renderFile(JOS, Coverage, SourceFile, FileReport, Options);
      RawSStream.flush();
      RenderedFiles[I] = std::move(Buffer);
    });
  }
  Pool.wait();

  // Dump rendered strings sorted by filename.
  std::vector<unsigned> Indices(SourceFiles.size());
  std::iota(Indices.begin(), Indices.end(), 0u);
  llvm::sort(Indices, [&](unsigned A, unsigned B) {
    return SourceFiles[A] < SourceFiles[B];
  });
  JOS.array([&] {
    for (unsigned I : Indices)
      JOS.rawValue(RenderedFiles[I]);
  });
}

void renderFunctions(
    json::OStream &JOS,
    const iterator_range<coverage::FunctionRecordIterator> &Functions,
    const CoverageViewOptions &Options) {
  JOS.array([&] {
    for (const auto &F : Functions) {
      JOS.object([&] {
        JOS.attributeBegin("branches");
        renderBranchRegions(JOS, F.CountedBranchRegions);
        JOS.attributeEnd();

        JOS.attribute("count", clamp_uint64_to_int64(F.ExecutionCount));

        JOS.attributeArray("filenames", [&] {
          for (const auto &Filename : F.Filenames)
            JOS.value(Filename);
        });

        JOS.attributeBegin("mcdc_records");
        renderMCDCRecords(JOS, F.MCDCRecords, Options);
        JOS.attributeEnd();

        JOS.attribute("name", F.Name);

        JOS.attributeBegin("regions");
        renderRegions(JOS, F.CountedRegions);
        JOS.attributeEnd();
      });
    }
  });
}

} // end anonymous namespace

void CoverageExporterJson::renderRoot(const CoverageFilters &IgnoreFilters) {
  std::vector<std::string> SourceFiles;
  for (StringRef SF : Coverage.getUniqueSourceFiles()) {
    if (!IgnoreFilters.matchesFilename(SF))
      SourceFiles.emplace_back(SF);
  }
  renderRoot(SourceFiles);
}

void CoverageExporterJson::renderRoot(ArrayRef<std::string> SourceFiles) {
  FileCoverageSummary Totals = FileCoverageSummary("Totals");
  auto FileReports = CoverageReport::prepareFileReports(Coverage, Totals,
                                                        SourceFiles, Options);

  json::OStream JOS(OS);
  JOS.object([&] {
    JOS.attributeArray("data", [&] {
      JOS.object([&] {
        JOS.attributeBegin("files");
        renderFiles(JOS, Coverage, SourceFiles, FileReports, Options);
        JOS.attributeEnd();

        // Skip functions-level information if necessary.
        if (!Options.ExportSummaryOnly && !Options.SkipFunctions) {
          JOS.attributeBegin("functions");
          renderFunctions(JOS, Coverage.getCoveredFunctions(), Options);
          JOS.attributeEnd();
        }

        JOS.attributeBegin("totals");
        renderSummary(JOS, Totals);
        JOS.attributeEnd();
      });
    });
    JOS.attribute("type", LLVM_COVERAGE_EXPORT_JSON_TYPE_STR);
    JOS.attribute("version", LLVM_COVERAGE_EXPORT_JSON_STR);
  });
}
