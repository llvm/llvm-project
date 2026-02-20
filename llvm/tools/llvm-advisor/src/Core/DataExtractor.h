//===------------------ DataExtractor.h - LLVM Advisor --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the DataExtractor code generator driver. It provides a convenient
// command-line interface for generating an assembly file or a relocatable file,
// given LLVM bitcode.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_ADVISOR_DATA_EXTRACTOR_H
#define LLVM_ADVISOR_DATA_EXTRACTOR_H

#include "../Config/AdvisorConfig.h"
#include "CompilationUnit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <string>

namespace llvm {
namespace advisor {

class DataExtractor {
public:
  DataExtractor(const AdvisorConfig &Config);

  Error extractAllData(CompilationUnit &Unit, llvm::StringRef TempDir);

private:
  llvm::SmallVector<std::string, 8>
  getBaseCompilerArgs(const CompilationUnitInfo &UnitInfo) const;

  Error extractIR(CompilationUnit &Unit, llvm::StringRef TempDir);
  Error extractAssembly(CompilationUnit &Unit, llvm::StringRef TempDir);
  Error extractAST(CompilationUnit &Unit, llvm::StringRef TempDir);
  Error extractPreprocessed(CompilationUnit &Unit, llvm::StringRef TempDir);
  Error extractIncludeTree(CompilationUnit &Unit, llvm::StringRef TempDir);
  Error extractDependencies(CompilationUnit &Unit, llvm::StringRef TempDir);
  Error extractDebugInfo(CompilationUnit &Unit, llvm::StringRef TempDir);
  Error extractStaticAnalysis(CompilationUnit &Unit, llvm::StringRef TempDir);
  Error extractMacroExpansion(CompilationUnit &Unit, llvm::StringRef TempDir);
  Error extractCompilationPhases(CompilationUnit &Unit,
                                 llvm::StringRef TempDir);
  Error extractFTimeReport(CompilationUnit &Unit, llvm::StringRef TempDir);
  Error extractVersionInfo(CompilationUnit &Unit, llvm::StringRef TempDir);
  Error extractSources(CompilationUnit &Unit, llvm::StringRef TempDir);
  Error extractASTJSON(CompilationUnit &Unit, llvm::StringRef TempDir);
  Error extractDiagnostics(CompilationUnit &Unit, llvm::StringRef TempDir);
  Error extractCoverage(CompilationUnit &Unit, llvm::StringRef TempDir);
  Error extractTimeTrace(CompilationUnit &Unit, llvm::StringRef TempDir);
  Error extractRuntimeTrace(CompilationUnit &Unit, llvm::StringRef TempDir);
  Error extractSARIF(CompilationUnit &Unit, llvm::StringRef TempDir);
  Error extractBinarySize(CompilationUnit &Unit, llvm::StringRef TempDir);
  Error extractPGO(CompilationUnit &Unit, llvm::StringRef TempDir);
  Error extractSymbols(CompilationUnit &Unit, llvm::StringRef TempDir);
  Error extractObjdump(CompilationUnit &Unit, llvm::StringRef TempDir);
  Error extractXRay(CompilationUnit &Unit, llvm::StringRef TempDir);
  Error extractOptDot(CompilationUnit &Unit, llvm::StringRef TempDir);

  Error runCompilerWithFlags(const llvm::SmallVector<std::string, 8> &Args);

  using ExtractorMethod = Error (DataExtractor::*)(CompilationUnit &Unit,
                                                   llvm::StringRef TempDir);
  struct ExtractorInfo {
    ExtractorMethod method;
    const char *name;
  };

  static const ExtractorInfo extractors[];
  static const size_t numExtractors;

  const AdvisorConfig &Config;
};

} // namespace advisor
} // namespace llvm

#endif
