//===-- TosaToLinalg.h - TOSA optimization pass declarations ----------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the passes for the TOSA Linalg Dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_TOSATOLINALG_TOSATOLINALG_H
#define MLIR_CONVERSION_TOSATOLINALG_TOSATOLINALG_H

#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

#define GEN_PASS_DECL_TOSATOLINALG
#define GEN_PASS_DECL_TOSATOLINALGNAMED
#include "mlir/Conversion/Passes.h.inc"

namespace tosa {

std::unique_ptr<Pass> createTosaToLinalg();
std::unique_ptr<Pass> createTosaToLinalgNamed(
    const TosaToLinalgNamedOptions &options = TosaToLinalgNamedOptions());

/// Populates passes to convert from TOSA to Linalg on buffers. At the end of
/// the pass, the function will only contain linalg ops or standard ops if the
/// pipeline succeeds.  The option to disable decompositions is available for
/// benchmarking performance improvements from the canonicalizations.
void addTosaToLinalgPasses(
    OpPassManager &pm, const TosaToLinalgOptions &options,
    const TosaToLinalgNamedOptions &tosaToLinalgNamedOptions =
        TosaToLinalgNamedOptions(),
    // Note: Default to 'none' level unless otherwise specified.
    std::optional<tosa::TosaValidationOptions> validationOptions =
        tosa::TosaValidationOptions{
            {"none"}, false, tosa::TosaLevelEnum::None});

/// Populates TOSA to linalg pipelines
/// Currently, this includes only the "tosa-to-linalg-pipeline".
void registerTosaToLinalgPipelines();

/// Populates conversion passes from TOSA dialect to Linalg dialect.
void populateTosaToLinalgConversionPatterns(const TypeConverter &converter,
                                            RewritePatternSet *patterns);

/// Populates conversion passes from TOSA dialect to Linalg named operations.
void populateTosaToLinalgNamedConversionPatterns(
    RewritePatternSet *patterns, const TosaToLinalgNamedOptions &options);

} // namespace tosa
} // namespace mlir

#endif // MLIR_CONVERSION_TOSATOLINALG_TOSATOLINALG_H
