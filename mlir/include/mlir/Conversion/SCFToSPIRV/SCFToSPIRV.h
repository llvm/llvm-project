//===- SCFToSPIRV.h - SCF to SPIR-V Patterns --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides patterns to convert SCF dialect to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_SCFTOSPIRV_SCFTOSPIRV_H_
#define MLIR_CONVERSION_SCFTOSPIRV_SCFTOSPIRV_H_

#include <memory>

namespace mlir {
class Pass;

// Owning list of rewriting patterns.
class SPIRVTypeConverter;
struct ScfToSPIRVContextImpl;
class RewritePatternSet;
using OwningRewritePatternList = RewritePatternSet;

struct ScfToSPIRVContext {
  ScfToSPIRVContext();
  ~ScfToSPIRVContext();

  ScfToSPIRVContextImpl *getImpl() { return impl.get(); }

private:
  std::unique_ptr<ScfToSPIRVContextImpl> impl;
};

/// Collects a set of patterns to lower from scf.for, scf.if, and
/// loop.terminator to CFG operations within the SPIR-V dialect.
void populateSCFToSPIRVPatterns(SPIRVTypeConverter &typeConverter,
                                ScfToSPIRVContext &scfToSPIRVContext,
                                OwningRewritePatternList &patterns);
} // namespace mlir

#endif // MLIR_CONVERSION_SCFTOSPIRV_SCFTOSPIRV_H_
