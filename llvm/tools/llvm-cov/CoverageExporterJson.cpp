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

#include "CoverageReport.h"
#include "CoverageSummaryInfo.h"
#include "CoverageViewOptions.h"
#include "llvm/ProfileData/Coverage/CoverageMapping.h"
#include <stack>

/// \brief Major version of the JSON Coverage Export Format.
#define LLVM_COVERAGE_EXPORT_JSON_MAJOR 1

/// \brief Minor version of the JSON Coverage Export Format.
#define LLVM_COVERAGE_EXPORT_JSON_MINOR 0

/// \brief Patch version of the JSON Coverage Export Format.
#define LLVM_COVERAGE_EXPORT_JSON_PATCH 0

/// \brief The semantic version combined as a string.
#define LLVM_COVERAGE_EXPORT_JSON_STR "1.0.0"

/// \brief Unique type identifier for JSON coverage export.
#define LLVM_COVERAGE_EXPORT_JSON_TYPE_STR "llvm.coverage.json.export"

using namespace llvm;
using namespace coverage;

class CoverageExporterJson {
  /// \brief A Name of the object file coverage is for.
  StringRef ObjectFilename;

  /// \brief Output stream to print JSON to.
  raw_ostream &OS;

  /// \brief The full CoverageMapping object to export.
  CoverageMapping Coverage;

  /// \brief States that the JSON rendering machine can be in.
  enum JsonState { None, NonEmptyElement, EmptyElement };

  /// \brief Tracks state of the JSON output.
  std::stack<JsonState> State;

  /// \brief Get the object filename.
  StringRef getObjectFilename() const { return ObjectFilename; }

  /// \brief Emit a serialized scalar.
  void emitSerialized(const int64_t Value) { OS << Value; }

  /// \brief Emit a serialized string.
  void emitSerialized(const std::string &Value) {
    OS << "\"";
    for (char C : Value) {
      if (C != '\\')
        OS << C;
      else
        OS << "\\\\";
    }
    OS << "\"";
  }

  /// \brief Emit a comma if there is a previous element to delimit.
  void emitComma() {
    if (State.top() == JsonState::NonEmptyElement) {
      OS << ",";
    } else if (State.top() == JsonState::EmptyElement) {
      State.pop();
      assert((State.size() >= 1) && "Closed too many JSON elements");
      State.push(JsonState::NonEmptyElement);
    }
  }

  /// \brief Emit a starting dictionary/object character.
  void emitDictStart() {
    emitComma();
    State.push(JsonState::EmptyElement);
    OS << "{";
  }

  /// \brief Emit a dictionary/object key but no value.
  void emitDictKey(const std::string &Key) {
    emitComma();
    emitSerialized(Key);
    OS << ":";
    State.pop();
    assert((State.size() >= 1) && "Closed too many JSON elements");

    // We do not want to emit a comma after this key.
    State.push(JsonState::EmptyElement);
  }

  /// \brief Emit a dictionary/object key/value pair.
  template <typename V>
  void emitDictElement(const std::string &Key, const V &Value) {
    emitComma();
    emitSerialized(Key);
    OS << ":";
    emitSerialized(Value);
  }

  /// \brief Emit a closing dictionary/object character.
  void emitDictEnd() {
    State.pop();
    assert((State.size() >= 1) && "Closed too many JSON elements");
    OS << "}";
  }

  /// \brief Emit a starting array character.
  void emitArrayStart() {
    emitComma();
    State.push(JsonState::EmptyElement);
    OS << "[";
  }

  /// \brief Emit an array element.
  template <typename V> void emitArrayElement(const V &Value) {
    emitComma();
    emitSerialized(Value);
  }

  /// \brief emit a closing array character.
  void emitArrayEnd() {
    State.pop();
    assert((State.size() >= 1) && "Closed too many JSON elements");
    OS << "]";
  }

  /// \brief Render the CoverageMapping object.
  void renderRoot() {
    // Start Root of JSON object.
    emitDictStart();

    emitDictElement("version", LLVM_COVERAGE_EXPORT_JSON_STR);
    emitDictElement("type", LLVM_COVERAGE_EXPORT_JSON_TYPE_STR);
    emitDictKey("data");

    // Start List of Exports.
    emitArrayStart();

    // Start Export.
    emitDictStart();
    emitDictElement("object", getObjectFilename());

    emitDictKey("files");

    FileCoverageSummary Totals = FileCoverageSummary("Totals");
    std::vector<StringRef> SourceFiles = Coverage.getUniqueSourceFiles();
    auto FileReports =
        CoverageReport::prepareFileReports(Coverage, Totals, SourceFiles);
    renderFiles(SourceFiles, FileReports);

    emitDictKey("functions");
    renderFunctions(Coverage.getCoveredFunctions());

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

  /// \brief Render an array of all the given functions.
  void
  renderFunctions(const iterator_range<FunctionRecordIterator> &Functions) {
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

  /// \brief Render an array of all the source files, also pass back a Summary.
  void renderFiles(ArrayRef<StringRef> SourceFiles,
                   ArrayRef<FileCoverageSummary> FileReports) {
    // Start List of Files.
    emitArrayStart();

    for (unsigned I = 0, E = SourceFiles.size(); I < E; ++I) {
      // Render the file.
      auto FileCoverage = Coverage.getCoverageForFile(SourceFiles[I]);
      renderFile(FileCoverage, FileReports[I]);
    }

    // End List of Files.
    emitArrayEnd();
  }

  /// \brief Render a single file.
  void renderFile(const CoverageData &FileCoverage,
                  const FileCoverageSummary &FileReport) {
    // Start File.
    emitDictStart();

    emitDictElement("filename", FileCoverage.getFilename());
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

    emitDictKey("summary");
    renderSummary(FileReport);

    // End File.
    emitDictEnd();
  }

  /// \brief Render a CoverageSegment.
  void renderSegment(const CoverageSegment &Segment) {
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

  /// \brief Render an ExpansionRecord.
  void renderExpansion(const ExpansionRecord &Expansion) {
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

  /// \brief Render a list of CountedRegions.
  void renderRegions(ArrayRef<CountedRegion> Regions) {
    // Start List of Regions.
    emitArrayStart();

    for (const auto &Region : Regions)
      renderRegion(Region);

    // End List of Regions.
    emitArrayEnd();
  }

  /// \brief Render a single CountedRegion.
  void renderRegion(const CountedRegion &Region) {
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

  /// \brief Render a FileCoverageSummary.
  void renderSummary(const FileCoverageSummary &Summary) {
    // Start Summary for the file.
    emitDictStart();

    emitDictKey("lines");

    // Start Line Coverage Summary.
    emitDictStart();
    emitDictElement("count", Summary.LineCoverage.NumLines);
    emitDictElement("covered", Summary.LineCoverage.Covered);
    emitDictElement("percent", Summary.LineCoverage.getPercentCovered());
    emitDictElement("noncode", Summary.LineCoverage.NonCodeLines);
    // End Line Coverage Summary.
    emitDictEnd();

    emitDictKey("functions");

    // Start Function Coverage Summary.
    emitDictStart();
    emitDictElement("count", Summary.FunctionCoverage.NumFunctions);
    emitDictElement("covered", Summary.FunctionCoverage.Executed);
    emitDictElement("percent", Summary.FunctionCoverage.getPercentCovered());
    // End Function Coverage Summary.
    emitDictEnd();

    emitDictKey("instantiations");

    // Start Instantiation Coverage Summary.
    emitDictStart();
    emitDictElement("count", Summary.InstantiationCoverage.NumFunctions);
    emitDictElement("covered", Summary.InstantiationCoverage.Executed);
    emitDictElement("percent",
                    Summary.InstantiationCoverage.getPercentCovered());
    // End Function Coverage Summary.
    emitDictEnd();

    emitDictKey("regions");

    // Start Region Coverage Summary.
    emitDictStart();
    emitDictElement("count", Summary.RegionCoverage.NumRegions);
    emitDictElement("covered", Summary.RegionCoverage.Covered);
    emitDictElement("notcovered", Summary.RegionCoverage.NotCovered);
    emitDictElement("percent", Summary.RegionCoverage.getPercentCovered());
    // End Region Coverage Summary.
    emitDictEnd();

    // End Summary for the file.
    emitDictEnd();
  }

public:
  CoverageExporterJson(StringRef ObjectFilename,
                       const CoverageMapping &CoverageMapping, raw_ostream &OS)
      : ObjectFilename(ObjectFilename), OS(OS), Coverage(CoverageMapping) {
    State.push(JsonState::None);
  }

  /// \brief Print the CoverageMapping.
  void print() { renderRoot(); }
};

/// \brief Export the given CoverageMapping to a JSON Format.
void exportCoverageDataToJson(StringRef ObjectFilename,
                              const CoverageMapping &CoverageMapping,
                              raw_ostream &OS) {
  auto Exporter = CoverageExporterJson(ObjectFilename, CoverageMapping, OS);

  Exporter.print();
}
