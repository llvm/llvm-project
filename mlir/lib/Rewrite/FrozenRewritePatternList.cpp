//===- FrozenRewritePatternList.cpp - Frozen Pattern List -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Rewrite/FrozenRewritePatternList.h"
#include "ByteCode.h"
#include "mlir/Conversion/PDLToPDLInterp/PDLToPDLInterp.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;

static LogicalResult convertPDLToPDLInterp(ModuleOp pdlModule) {
  // Skip the conversion if the module doesn't contain pdl.
  if (llvm::empty(pdlModule.getOps<pdl::PatternOp>()))
    return success();

  // Simplify the provided PDL module. Note that we can't use the canonicalizer
  // here because it would create a cyclic dependency.
  auto simplifyFn = [](Operation *op) {
    // TODO: Add folding here if ever necessary.
    if (isOpTriviallyDead(op))
      op->erase();
  };
  pdlModule.getBody()->walk(simplifyFn);

  /// Lower the PDL pattern module to the interpreter dialect.
  PassManager pdlPipeline(pdlModule.getContext());
#ifdef NDEBUG
  // We don't want to incur the hit of running the verifier when in release
  // mode.
  pdlPipeline.enableVerifier(false);
#endif
  pdlPipeline.addPass(createPDLToPDLInterpPass());
  if (failed(pdlPipeline.run(pdlModule)))
    return failure();

  // Simplify again after running the lowering pipeline.
  pdlModule.getBody()->walk(simplifyFn);
  return success();
}

//===----------------------------------------------------------------------===//
// FrozenRewritePatternList
//===----------------------------------------------------------------------===//

FrozenRewritePatternList::FrozenRewritePatternList(
    OwningRewritePatternList &&patterns)
    : nativePatterns(std::move(patterns.getNativePatterns())) {
  PDLPatternModule &pdlPatterns = patterns.getPDLPatterns();

  // Generate the bytecode for the PDL patterns if any were provided.
  ModuleOp pdlModule = pdlPatterns.getModule();
  if (!pdlModule)
    return;
  if (failed(convertPDLToPDLInterp(pdlModule)))
    llvm::report_fatal_error(
        "failed to lower PDL pattern module to the PDL Interpreter");

  // Generate the pdl bytecode.
  pdlByteCode = std::make_unique<detail::PDLByteCode>(
      pdlModule, pdlPatterns.takeConstraintFunctions(),
      pdlPatterns.takeCreateFunctions(), pdlPatterns.takeRewriteFunctions());
}

FrozenRewritePatternList::FrozenRewritePatternList(
    FrozenRewritePatternList &&patterns)
    : nativePatterns(std::move(patterns.nativePatterns)),
      pdlByteCode(std::move(patterns.pdlByteCode)) {}

FrozenRewritePatternList::~FrozenRewritePatternList() {}
