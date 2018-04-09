//===- CoverageExporter.h - Code coverage exporter ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class defines a code coverage exporter interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_COV_COVERAGEEXPORTER_H
#define LLVM_COV_COVERAGEEXPORTER_H

#include "CoverageFilters.h"
#include "CoverageSummaryInfo.h"
#include "CoverageViewOptions.h"
#include "llvm/ProfileData/Coverage/CoverageMapping.h"

namespace llvm {

/// \brief Exports the code coverage information.
class CoverageExporter {
protected:
  /// \brief The full CoverageMapping object to export.
  const coverage::CoverageMapping &Coverage;

  /// \brief The options passed to the tool.
  const CoverageViewOptions &Options;

  /// \brief Output stream to print JSON to.
  raw_ostream &OS;

  CoverageExporter(const coverage::CoverageMapping &CoverageMapping,
                   const CoverageViewOptions &Options, raw_ostream &OS)
      : Coverage(CoverageMapping), Options(Options), OS(OS) {}

public:
  virtual ~CoverageExporter(){};

  /// \brief Render the CoverageMapping object.
  virtual void renderRoot(const CoverageFilters &IgnoreFilenameFilters) = 0;

  /// \brief Render the CoverageMapping object for specified source files.
  virtual void renderRoot(const std::vector<std::string> &SourceFiles) = 0;
};

} // end namespace llvm

#endif // LLVM_COV_COVERAGEEXPORTER_H
