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
  /// \brief States that the JSON rendering machine can be in.
  enum JsonState { None, NonEmptyElement, EmptyElement };

  /// \brief Tracks state of the JSON output.
  std::stack<JsonState> State;

  /// \brief Emit a serialized scalar.
  void emitSerialized(const int64_t Value);

  /// \brief Emit a serialized string.
  void emitSerialized(const std::string &Value);

  /// \brief Emit a comma if there is a previous element to delimit.
  void emitComma();

  /// \brief Emit a starting dictionary/object character.
  void emitDictStart();

  /// \brief Emit a dictionary/object key but no value.
  void emitDictKey(const std::string &Key);

  /// \brief Emit a dictionary/object key/value pair.
  template <typename V>
  void emitDictElement(const std::string &Key, const V &Value) {
    emitComma();
    emitSerialized(Key);
    OS << ":";
    emitSerialized(Value);
  }

  /// \brief Emit a closing dictionary/object character.
  void emitDictEnd();

  /// \brief Emit a starting array character.
  void emitArrayStart();

  /// \brief Emit an array element.
  template <typename V> void emitArrayElement(const V &Value) {
    emitComma();
    emitSerialized(Value);
  }

  /// \brief emit a closing array character.
  void emitArrayEnd();

  /// \brief Render an array of all the given functions.
  void renderFunctions(
      const iterator_range<coverage::FunctionRecordIterator> &Functions);

  /// \brief Render an array of all the source files, also pass back a Summary.
  void renderFiles(ArrayRef<std::string> SourceFiles,
                   ArrayRef<FileCoverageSummary> FileReports);

  /// \brief Render a single file.
  void renderFile(const coverage::CoverageData &FileCoverage,
                  const FileCoverageSummary &FileReport);

  /// \brief Render a CoverageSegment.
  void renderSegment(const coverage::CoverageSegment &Segment);

  /// \brief Render an ExpansionRecord.
  void renderExpansion(const coverage::ExpansionRecord &Expansion);

  /// \brief Render a list of CountedRegions.
  void renderRegions(ArrayRef<coverage::CountedRegion> Regions);

  /// \brief Render a single CountedRegion.
  void renderRegion(const coverage::CountedRegion &Region);

  /// \brief Render a FileCoverageSummary.
  void renderSummary(const FileCoverageSummary &Summary);

public:
  CoverageExporterJson(const coverage::CoverageMapping &CoverageMapping,
                       const CoverageViewOptions &Options, raw_ostream &OS);

  /// \brief Render the CoverageMapping object.
  void renderRoot() override;

  /// \brief Render the CoverageMapping object for specified source files.
  void renderRoot(const std::vector<std::string> &SourceFiles) override;
};

} // end namespace llvm

#endif // LLVM_COV_COVERAGEEXPORTERJSON_H
