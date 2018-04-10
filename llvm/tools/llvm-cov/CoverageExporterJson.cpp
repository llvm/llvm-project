//===- CoverageExporterJson.cpp - Code coverage export --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
// ---- Export: dict => Json representation of one CoverageMapping
// ------ Files: array => List of objects describing coverage for files
// -------- File: dict => Coverage for a single file
// ---------- Segments: array => List of Segments contained in the file
// ------------ Segment: dict => Describes a segment of the file with a counter
// ---------- Expansions: array => List of expansion records
// ------------ Expansion: dict => Object that descibes a single expansion
// -------------- CountedRegion: dict => The region to be expanded
// -------------- TargetRegions: array => List of Regions in the expansion
// ---------------- CountedRegion: dict => Single Region in the expansion
// ---------- Summary: dict => Object summarizing the coverage for this file
// ------------ LineCoverage: dict => Object summarizing line coverage
// ------------ FunctionCoverage: dict => Object summarizing function coverage
// ------------ RegionCoverage: dict => Object summarizing region coverage
// ------ Functions: array => List of objects describing coverage for functions
// -------- Function: dict => Coverage info for a single function
// ---------- Filenames: array => List of filenames that the function relates to
// ---- Summary: dict => Object summarizing the coverage for the entire binary
// ------ LineCoverage: dict => Object summarizing line coverage
// ------ FunctionCoverage: dict => Object summarizing function coverage
// ------ InstantiationCoverage: dict => Object summarizing inst. coverage
// ------ RegionCoverage: dict => Object summarizing region coverage
//
//===----------------------------------------------------------------------===//

#include "CoverageExporterJson.h"
#include "CoverageReport.h"

/// \brief The semantic version combined as a string.
#define LLVM_COVERAGE_EXPORT_JSON_STR "2.0.0"

/// \brief Unique type identifier for JSON coverage export.
#define LLVM_COVERAGE_EXPORT_JSON_TYPE_STR "llvm.coverage.json.export"

using namespace llvm;

CoverageExporterJson::CoverageExporterJson(
    const coverage::CoverageMapping &CoverageMapping,
    const CoverageViewOptions &Options, raw_ostream &OS)
    : CoverageExporter(CoverageMapping, Options, OS) {
  State.push(JsonState::None);
}

void CoverageExporterJson::emitSerialized(const int64_t Value) { OS << Value; }

void CoverageExporterJson::emitSerialized(const std::string &Value) {
  OS << "\"";
  for (char C : Value) {
    if (C != '\\')
      OS << C;
    else
      OS << "\\\\";
  }
  OS << "\"";
}

void CoverageExporterJson::emitComma() {
  if (State.top() == JsonState::NonEmptyElement) {
    OS << ",";
  } else if (State.top() == JsonState::EmptyElement) {
    State.pop();
    assert((State.size() >= 1) && "Closed too many JSON elements");
    State.push(JsonState::NonEmptyElement);
  }
}

void CoverageExporterJson::emitDictStart() {
  emitComma();
  State.push(JsonState::EmptyElement);
  OS << "{";
}

void CoverageExporterJson::emitDictKey(const std::string &Key) {
  emitComma();
  emitSerialized(Key);
  OS << ":";
  State.pop();
  assert((State.size() >= 1) && "Closed too many JSON elements");

  // We do not want to emit a comma after this key.
  State.push(JsonState::EmptyElement);
}

void CoverageExporterJson::emitDictEnd() {
  State.pop();
  assert((State.size() >= 1) && "Closed too many JSON elements");
  OS << "}";
}

void CoverageExporterJson::emitArrayStart() {
  emitComma();
  State.push(JsonState::EmptyElement);
  OS << "[";
}

void CoverageExporterJson::emitArrayEnd() {
  State.pop();
  assert((State.size() >= 1) && "Closed too many JSON elements");
  OS << "]";
}

void CoverageExporterJson::renderRoot(
    const CoverageFilters &IgnoreFilenameFilters) {
  std::vector<std::string> SourceFiles;
  for (StringRef SF : Coverage.getUniqueSourceFiles()) {
    if (!IgnoreFilenameFilters.matchesFilename(SF))
      SourceFiles.emplace_back(SF);
  }
  renderRoot(SourceFiles);
}

void CoverageExporterJson::renderRoot(
    const std::vector<std::string> &SourceFiles) {
  // Start Root of JSON object.
  emitDictStart();

  emitDictElement("version", LLVM_COVERAGE_EXPORT_JSON_STR);
  emitDictElement("type", LLVM_COVERAGE_EXPORT_JSON_TYPE_STR);
  emitDictKey("data");

  // Start List of Exports.
  emitArrayStart();

  // Start Export.
  emitDictStart();

  emitDictKey("files");

  FileCoverageSummary Totals = FileCoverageSummary("Totals");
  auto FileReports = CoverageReport::prepareFileReports(Coverage, Totals,
                                                        SourceFiles, Options);
  renderFiles(SourceFiles, FileReports);

  // Skip functions-level information for summary-only export mode.
  if (!Options.ExportSummaryOnly) {
    emitDictKey("functions");
    renderFunctions(Coverage.getCoveredFunctions());
  }

  emitDictKey("totals");
  renderSummary(Totals);

  // End Export.
  emitDictEnd();

  // End List of Exports.
  emitArrayEnd();

  // End Root of JSON Object.
  emitDictEnd();

  assert((State.top() == JsonState::None) &&
         "All Elements In JSON were Closed");
}

void CoverageExporterJson::renderFunctions(
    const iterator_range<coverage::FunctionRecordIterator> &Functions) {
  // Start List of Functions.
  emitArrayStart();

  for (const auto &Function : Functions) {
    // Start Function.
    emitDictStart();

    emitDictElement("name", Function.Name);
    emitDictElement("count", Function.ExecutionCount);
    emitDictKey("regions");

    renderRegions(Function.CountedRegions);

    emitDictKey("filenames");

    // Start Filenames for Function.
    emitArrayStart();

    for (const auto &FileName : Function.Filenames)
      emitArrayElement(FileName);

    // End Filenames for Function.
    emitArrayEnd();

    // End Function.
    emitDictEnd();
  }

  // End List of Functions.
  emitArrayEnd();
}

void CoverageExporterJson::renderFiles(
    ArrayRef<std::string> SourceFiles,
    ArrayRef<FileCoverageSummary> FileReports) {
  // Start List of Files.
  emitArrayStart();

  for (unsigned I = 0, E = SourceFiles.size(); I < E; ++I) {
    renderFile(SourceFiles[I], FileReports[I]);
  }

  // End List of Files.
  emitArrayEnd();
}

void CoverageExporterJson::renderFile(const std::string &Filename,
                                      const FileCoverageSummary &FileReport) {
  // Start File.
  emitDictStart();

  emitDictElement("filename", Filename);

  if (!Options.ExportSummaryOnly) {
    // Calculate and render detailed coverage information for given file.
    auto FileCoverage = Coverage.getCoverageForFile(Filename);
    renderFileCoverage(FileCoverage, FileReport);
  }

  emitDictKey("summary");
  renderSummary(FileReport);

  // End File.
  emitDictEnd();
}


void CoverageExporterJson::renderFileCoverage(
    const coverage::CoverageData &FileCoverage,
    const FileCoverageSummary &FileReport) {
  emitDictKey("segments");

  // Start List of Segments.
  emitArrayStart();

  for (const auto &Segment : FileCoverage)
    renderSegment(Segment);

  // End List of Segments.
  emitArrayEnd();

  emitDictKey("expansions");

  // Start List of Expansions.
  emitArrayStart();

  for (const auto &Expansion : FileCoverage.getExpansions())
    renderExpansion(Expansion);

  // End List of Expansions.
  emitArrayEnd();
}

void CoverageExporterJson::renderSegment(
    const coverage::CoverageSegment &Segment) {
  // Start Segment.
  emitArrayStart();

  emitArrayElement(Segment.Line);
  emitArrayElement(Segment.Col);
  emitArrayElement(Segment.Count);
  emitArrayElement(Segment.HasCount);
  emitArrayElement(Segment.IsRegionEntry);

  // End Segment.
  emitArrayEnd();
}

void CoverageExporterJson::renderExpansion(
    const coverage::ExpansionRecord &Expansion) {
  // Start Expansion.
  emitDictStart();

  // Mark the beginning and end of this expansion in the source file.
  emitDictKey("source_region");
  renderRegion(Expansion.Region);

  // Enumerate the coverage information for the expansion.
  emitDictKey("target_regions");
  renderRegions(Expansion.Function.CountedRegions);

  emitDictKey("filenames");
  // Start List of Filenames to map the fileIDs.
  emitArrayStart();
  for (const auto &Filename : Expansion.Function.Filenames)
    emitArrayElement(Filename);
  // End List of Filenames.
  emitArrayEnd();

  // End Expansion.
  emitDictEnd();
}

void CoverageExporterJson::renderRegions(
    ArrayRef<coverage::CountedRegion> Regions) {
  // Start List of Regions.
  emitArrayStart();

  for (const auto &Region : Regions)
    renderRegion(Region);

  // End List of Regions.
  emitArrayEnd();
}

void CoverageExporterJson::renderRegion(const coverage::CountedRegion &Region) {
  // Start CountedRegion.
  emitArrayStart();

  emitArrayElement(Region.LineStart);
  emitArrayElement(Region.ColumnStart);
  emitArrayElement(Region.LineEnd);
  emitArrayElement(Region.ColumnEnd);
  emitArrayElement(Region.ExecutionCount);
  emitArrayElement(Region.FileID);
  emitArrayElement(Region.ExpandedFileID);
  emitArrayElement(Region.Kind);

  // End CountedRegion.
  emitArrayEnd();
}

void CoverageExporterJson::renderSummary(const FileCoverageSummary &Summary) {
  // Start Summary for the file.
  emitDictStart();

  emitDictKey("lines");

  // Start Line Coverage Summary.
  emitDictStart();
  emitDictElement("count", Summary.LineCoverage.getNumLines());
  emitDictElement("covered", Summary.LineCoverage.getCovered());
  emitDictElement("percent", Summary.LineCoverage.getPercentCovered());
  // End Line Coverage Summary.
  emitDictEnd();

  emitDictKey("functions");

  // Start Function Coverage Summary.
  emitDictStart();
  emitDictElement("count", Summary.FunctionCoverage.getNumFunctions());
  emitDictElement("covered", Summary.FunctionCoverage.getExecuted());
  emitDictElement("percent", Summary.FunctionCoverage.getPercentCovered());
  // End Function Coverage Summary.
  emitDictEnd();

  emitDictKey("instantiations");

  // Start Instantiation Coverage Summary.
  emitDictStart();
  emitDictElement("count", Summary.InstantiationCoverage.getNumFunctions());
  emitDictElement("covered", Summary.InstantiationCoverage.getExecuted());
  emitDictElement("percent", Summary.InstantiationCoverage.getPercentCovered());
  // End Function Coverage Summary.
  emitDictEnd();

  emitDictKey("regions");

  // Start Region Coverage Summary.
  emitDictStart();
  emitDictElement("count", Summary.RegionCoverage.getNumRegions());
  emitDictElement("covered", Summary.RegionCoverage.getCovered());
  emitDictElement("notcovered", Summary.RegionCoverage.getNumRegions() -
                                    Summary.RegionCoverage.getCovered());
  emitDictElement("percent", Summary.RegionCoverage.getPercentCovered());
  // End Region Coverage Summary.
  emitDictEnd();

  // End Summary for the file.
  emitDictEnd();
}
