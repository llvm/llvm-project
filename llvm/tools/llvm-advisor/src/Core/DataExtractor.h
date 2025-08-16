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
  DataExtractor(const AdvisorConfig &config);

  Error extractAllData(CompilationUnit &unit, llvm::StringRef tempDir);

private:
  llvm::SmallVector<std::string, 8>
  getBaseCompilerArgs(const CompilationUnitInfo &unitInfo) const;

  Error extractIR(CompilationUnit &unit, llvm::StringRef tempDir);
  Error extractAssembly(CompilationUnit &unit, llvm::StringRef tempDir);
  Error extractAST(CompilationUnit &unit, llvm::StringRef tempDir);
  Error extractPreprocessed(CompilationUnit &unit, llvm::StringRef tempDir);
  Error extractIncludeTree(CompilationUnit &unit, llvm::StringRef tempDir);
  Error extractDebugInfo(CompilationUnit &unit, llvm::StringRef tempDir);
  Error extractStaticAnalysis(CompilationUnit &unit, llvm::StringRef tempDir);
  Error extractMacroExpansion(CompilationUnit &unit, llvm::StringRef tempDir);
  Error extractCompilationPhases(CompilationUnit &unit,
                                 llvm::StringRef tempDir);

  Error runCompilerWithFlags(const llvm::SmallVector<std::string, 8> &args);

  const AdvisorConfig &config_;
};

} // namespace advisor
} // namespace llvm

#endif