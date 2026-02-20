//===---------------- CoverageProcessor.h - LLVM Advisor -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_ADVISOR_SRC_CORE_COVERAGEPROCESSOR_H
#define LLVM_TOOLS_LLVM_ADVISOR_SRC_CORE_COVERAGEPROCESSOR_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <string>

namespace llvm::advisor {

struct CoverageArtifacts;

class CoverageProcessor {
public:
  /// Merge a raw instrumentation profile into an indexed .profdata file.
  static llvm::Error mergeRawProfile(llvm::StringRef RawProfile,
                                     llvm::StringRef IndexedProfile);

  /// Export coverage information in JSON format using the indexed profile and
  /// the matching instrumented binary.
  static llvm::Error exportCoverageReport(llvm::StringRef InstrumentedBinary,
                                          llvm::StringRef IndexedProfile,
                                          llvm::StringRef ReportPath,
                                          llvm::StringRef CompilationDir = "");

  /// Emit human-readable and JSON summaries for the given indexed profile.
  static llvm::Error summarizeProfile(llvm::StringRef IndexedProfile,
                                      llvm::StringRef TextSummaryPath,
                                      llvm::StringRef JsonSummaryPath);
};

} // namespace llvm::advisor

#endif
