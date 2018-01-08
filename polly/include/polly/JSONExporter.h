//===- polly/JSONExporter.h - Import/Export to/from jscop files.-*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_JSONEXPORTER_H
#define POLLY_JSONEXPORTER_H

#include "polly/ScopPass.h"
#include "llvm/IR/PassManager.h"

namespace polly {
/// This pass exports a scop to a jscop file. The filename is generated from the
/// concatenation of the function and scop name.
struct JSONExportPass : public llvm::PassInfoMixin<JSONExportPass> {
  llvm::PreservedAnalyses run(Scop &, ScopAnalysisManager &,
                              ScopStandardAnalysisResults &, SPMUpdater &);
};

/// This pass imports a scop from a jscop file. The filename is deduced from the
/// concatenation of the function and scop name.
struct JSONImportPass : public llvm::PassInfoMixin<JSONExportPass> {
  llvm::PreservedAnalyses run(Scop &, ScopAnalysisManager &,
                              ScopStandardAnalysisResults &, SPMUpdater &);
};
} // namespace polly

#endif /* POLLY_JSONEXPORTER_H */
