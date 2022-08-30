//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors in the loop
// transformation library.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_PASSES_H
#define MLIR_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LocationSnapshot.h"
#include "mlir/Transforms/ViewOpGraph.h"
#include "llvm/Support/Debug.h"
#include <limits>

namespace mlir {

class GreedyRewriteConfig;

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL_CANONICALIZERPASS
#define GEN_PASS_DECL_CONTROLFLOWSINKPASS
#define GEN_PASS_DECL_CSEPASS
#define GEN_PASS_DECL_LOOPINVARIANTCODEMOTIONPASS
#define GEN_PASS_DECL_STRIPDEBUGINFOPASS
#define GEN_PASS_DECL_PRINTOPSTATSPASS
#define GEN_PASS_DECL_INLINERPASS
#define GEN_PASS_DECL_SCCPPASS
#define GEN_PASS_DECL_SYMBOLDCEPASS
#define GEN_PASS_DECL_SYMBOLPRIVATIZEPASS
#define GEN_PASS_DECL_TOPOLOGICALSORTPASS
#include "mlir/Transforms/Passes.h.inc"

/// Creates an instance of the Canonicalizer pass, configured with default
/// settings (which can be overridden by pass options on the command line).
std::unique_ptr<Pass> createCanonicalizerPass();

/// Creates an instance of the Canonicalizer pass with the specified config.
/// `disabledPatterns` is a set of labels used to filter out input patterns with
/// a debug label or debug name in this set. `enabledPatterns` is a set of
/// labels used to filter out input patterns that do not have one of the labels
/// in this set. Debug labels must be set explicitly on patterns or when adding
/// them with `RewritePatternSet::addWithLabel`. Debug names may be empty, but
/// patterns created with `RewritePattern::create` have their default debug name
/// set to their type name.
std::unique_ptr<Pass>
createCanonicalizerPass(const GreedyRewriteConfig &config,
                        ArrayRef<std::string> disabledPatterns = llvm::None,
                        ArrayRef<std::string> enabledPatterns = llvm::None);

/// Creates a pass which prints the list of ops and the number of occurrences in
/// the module.
std::unique_ptr<Pass> createPrintOpStatsPass(raw_ostream &os = llvm::errs());

/// Creates a pass which prints the list of ops and the number of occurrences in
/// the module with the output format option.
std::unique_ptr<Pass> createPrintOpStatsPass(raw_ostream &os, bool printAsJSON);

/// Creates a pass which inlines calls and callable operations as defined by
/// the CallGraph.
std::unique_ptr<Pass> createInlinerPass();
/// Creates an instance of the inliner pass, and use the provided pass managers
/// when optimizing callable operations with names matching the key type.
/// Callable operations with a name not within the provided map will use the
/// default inliner pipeline during optimization.
std::unique_ptr<Pass>
createInlinerPass(llvm::StringMap<OpPassManager> opPipelines);
/// Creates an instance of the inliner pass, and use the provided pass managers
/// when optimizing callable operations with names matching the key type.
/// Callable operations with a name not within the provided map will use the
/// provided default pipeline builder.
std::unique_ptr<Pass>
createInlinerPass(llvm::StringMap<OpPassManager> opPipelines,
                  std::function<void(OpPassManager &)> defaultPipelineBuilder);

/// Creates a pass which marks top-level symbol operations as `private` unless
/// listed in `excludeSymbols`.
std::unique_ptr<Pass>
createSymbolPrivatizePass(ArrayRef<std::string> excludeSymbols = {});

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Transforms/Passes.h.inc"

} // namespace mlir

#endif // MLIR_TRANSFORMS_PASSES_H
