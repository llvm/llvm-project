//===- CoverageReport.h - Code coverage report ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This class implements rendering of a code coverage report.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_COV_COVERAGEREPORT_H
#define LLVM_COV_COVERAGEREPORT_H

#include "CoverageFilters.h"
#include "CoverageSummaryInfo.h"
#include "CoverageViewOptions.h"

namespace llvm {

/// Displays the code coverage report.
class CoverageReport {
  const CoverageViewOptions &Options;
  const coverage::CoverageMapping &Coverage;

  void render(const FileCoverageSummary &File, raw_ostream &OS) const;
  void render(const FunctionCoverageSummary &Function, const DemangleCache &DC,
              raw_ostream &OS) const;

public:
  CoverageReport(const CoverageViewOptions &Options,
                 const coverage::CoverageMapping &Coverage)
      : Options(Options), Coverage(Coverage) {}

  void renderFunctionReports(ArrayRef<std::string> Files,
                             const DemangleCache &DC, raw_ostream &OS);

  /// Prepare file reports for the files specified in \p Files.
  static std::vector<FileCoverageSummary>
  prepareFileReports(const coverage::CoverageMapping &Coverage,
                     FileCoverageSummary &Totals, ArrayRef<std::string> Files,
                     const CoverageViewOptions &Options,
                     const CoverageFilter &Filters = CoverageFiltersMatchAll());

  static void
  prepareSingleFileReport(const StringRef Filename,
                          const coverage::CoverageMapping *Coverage,
                          const CoverageViewOptions &Options,
                          const unsigned LCP,
                          FileCoverageSummary *FileReport,
                          const CoverageFilter *Filters);

  /// Render file reports for every unique file in the coverage mapping.
  void renderFileReports(raw_ostream &OS,
                         const CoverageFilters &IgnoreFilenameFilters) const;

  /// Render file reports for the files specified in \p Files.
  void renderFileReports(raw_ostream &OS, ArrayRef<std::string> Files) const;

  /// Render file reports for the files specified in \p Files and the functions
  /// in \p Filters.
  void renderFileReports(raw_ostream &OS, ArrayRef<std::string> Files,
                         const CoverageFiltersMatchAll &Filters) const;
};

} // end namespace llvm

#endif // LLVM_COV_COVERAGEREPORT_H
