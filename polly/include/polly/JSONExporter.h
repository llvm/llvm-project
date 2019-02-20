//===- polly/JSONExporter.h - Import/Export to/from jscop files.-*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
