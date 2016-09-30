//===- CoverageReport.h - Code coverage report ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class implements rendering of a code coverage report.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_COV_COVERAGEREPORT_H
#define LLVM_COV_COVERAGEREPORT_H

#include "CoverageSummaryInfo.h"
#include "CoverageViewOptions.h"

namespace llvm {

/// \brief Displays the code coverage report.
class CoverageReport {
  const CoverageViewOptions &Options;
  const coverage::CoverageMapping &Coverage;

  void render(const FileCoverageSummary &File, raw_ostream &OS) const;
  void render(const FunctionCoverageSummary &Function, raw_ostream &OS) const;

public:
  CoverageReport(const CoverageViewOptions &Options,
                 const coverage::CoverageMapping &Coverage)
      : Options(Options), Coverage(Coverage) {}

  void renderFunctionReports(ArrayRef<std::string> Files, raw_ostream &OS);

  /// Prepare file reports for the files specified in \p Files.
  static std::vector<FileCoverageSummary>
  prepareFileReports(const coverage::CoverageMapping &Coverage,
                     FileCoverageSummary &Totals, ArrayRef<std::string> Files);

  /// Render file reports for every unique file in the coverage mapping.
  void renderFileReports(raw_ostream &OS) const;

  /// Render file reports for the files specified in \p Files.
  void renderFileReports(raw_ostream &OS, ArrayRef<std::string> Files) const;
};

} // end namespace llvm

#endif // LLVM_COV_COVERAGEREPORT_H
