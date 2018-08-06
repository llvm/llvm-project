//===- CoverageExporterJson.h - Code coverage JSON exporter ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class implements a code coverage exporter for JSON format.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_COV_COVERAGEEXPORTERJSON_H
#define LLVM_COV_COVERAGEEXPORTERJSON_H

#include "CoverageExporter.h"
#include <stack>

namespace llvm {

class CoverageExporterJson : public CoverageExporter {
  /// States that the JSON rendering machine can be in.
  enum JsonState { None, NonEmptyElement, EmptyElement };

  /// Tracks state of the JSON output.
  std::stack<JsonState> State;

  /// Emit a serialized scalar.
  void emitSerialized(const int64_t Value);

  /// Emit a serialized string.
  void emitSerialized(const std::string &Value);

  /// Emit a comma if there is a previous element to delimit.
  void emitComma();

  /// Emit a starting dictionary/object character.
  void emitDictStart();

  /// Emit a dictionary/object key but no value.
  void emitDictKey(const std::string &Key);

  /// Emit a dictionary/object key/value pair.
  template <typename V>
  void emitDictElement(const std::string &Key, const V &Value) {
    emitComma();
    emitSerialized(Key);
    OS << ":";
    emitSerialized(Value);
  }

  /// Emit a closing dictionary/object character.
  void emitDictEnd();

  /// Emit a starting array character.
  void emitArrayStart();

  /// Emit an array element.
  template <typename V> void emitArrayElement(const V &Value) {
    emitComma();
    emitSerialized(Value);
  }

  /// emit a closing array character.
  void emitArrayEnd();

  /// Render an array of all the given functions.
  void renderFunctions(
      const iterator_range<coverage::FunctionRecordIterator> &Functions);

  /// Render an array of all the source files, also pass back a Summary.
  void renderFiles(ArrayRef<std::string> SourceFiles,
                   ArrayRef<FileCoverageSummary> FileReports);

  /// Render a single file.
  void renderFile(const std::string &Filename,
                  const FileCoverageSummary &FileReport);

  /// Render summary for a single file.
  void renderFileCoverage(const coverage::CoverageData &FileCoverage,
                          const FileCoverageSummary &FileReport);

  /// Render a CoverageSegment.
  void renderSegment(const coverage::CoverageSegment &Segment);

  /// Render an ExpansionRecord.
  void renderExpansion(const coverage::ExpansionRecord &Expansion);

  /// Render a list of CountedRegions.
  void renderRegions(ArrayRef<coverage::CountedRegion> Regions);

  /// Render a single CountedRegion.
  void renderRegion(const coverage::CountedRegion &Region);

  /// Render a FileCoverageSummary.
  void renderSummary(const FileCoverageSummary &Summary);

public:
  CoverageExporterJson(const coverage::CoverageMapping &CoverageMapping,
                       const CoverageViewOptions &Options, raw_ostream &OS);

  /// Render the CoverageMapping object.
  void renderRoot(const CoverageFilters &IgnoreFilenameFilters) override;

  /// Render the CoverageMapping object for specified source files.
  void renderRoot(const std::vector<std::string> &SourceFiles) override;
};

} // end namespace llvm

#endif // LLVM_COV_COVERAGEEXPORTERJSON_H
