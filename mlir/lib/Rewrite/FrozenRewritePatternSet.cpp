//===- FrozenRewritePatternSet.cpp - Frozen Pattern List -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "ByteCode.h"
#include "mlir/Conversion/PDLToPDLInterp/PDLToPDLInterp.h"
#include "mlir/Dialect/PDL/IR/PDLOps.h"
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
// FrozenRewritePatternSet
//===----------------------------------------------------------------------===//

FrozenRewritePatternSet::FrozenRewritePatternSet()
    : impl(std::make_shared<Impl>()) {}

FrozenRewritePatternSet::FrozenRewritePatternSet(
    RewritePatternSet &&patterns, ArrayRef<std::string> disabledPatternLabels,
    ArrayRef<std::string> enabledPatternLabels)
    : impl(std::make_shared<Impl>()) {
  DenseSet<StringRef> disabledPatterns, enabledPatterns;
  disabledPatterns.insert(disabledPatternLabels.begin(),
                          disabledPatternLabels.end());
  enabledPatterns.insert(enabledPatternLabels.begin(),
                         enabledPatternLabels.end());

  // Functor used to walk all of the operations registered in the context. This
  // is useful for patterns that get applied to multiple operations, such as
  // interface and trait based patterns.
  std::vector<AbstractOperation *> abstractOps;
  auto addToOpsWhen = [&](std::unique_ptr<RewritePattern> &pattern,
                          function_ref<bool(AbstractOperation *)> callbackFn) {
    if (abstractOps.empty())
      abstractOps = pattern->getContext()->getRegisteredOperations();
    for (AbstractOperation *absOp : abstractOps) {
      if (callbackFn(absOp)) {
        OperationName opName(absOp);
        impl->nativeOpSpecificPatternMap[opName].push_back(pattern.get());
      }
    }
    impl->nativeOpSpecificPatternList.push_back(std::move(pattern));
  };

  for (std::unique_ptr<RewritePattern> &pat : patterns.getNativePatterns()) {
    // Don't add patterns that haven't been enabled by the user.
    if (!enabledPatterns.empty()) {
      auto isEnabledFn = [&](StringRef label) {
        return enabledPatterns.count(label);
      };
      if (!isEnabledFn(pat->getDebugName()) &&
          llvm::none_of(pat->getDebugLabels(), isEnabledFn))
        continue;
    }
    // Don't add patterns that have been disabled by the user.
    if (!disabledPatterns.empty()) {
      auto isDisabledFn = [&](StringRef label) {
        return disabledPatterns.count(label);
      };
      if (isDisabledFn(pat->getDebugName()) ||
          llvm::any_of(pat->getDebugLabels(), isDisabledFn))
        continue;
    }

    if (Optional<OperationName> rootName = pat->getRootKind()) {
      impl->nativeOpSpecificPatternMap[*rootName].push_back(pat.get());
      impl->nativeOpSpecificPatternList.push_back(std::move(pat));
      continue;
    }
    if (Optional<TypeID> interfaceID = pat->getRootInterfaceID()) {
      addToOpsWhen(pat, [&](AbstractOperation *absOp) {
        return absOp->hasInterface(*interfaceID);
      });
      continue;
    }
    if (Optional<TypeID> traitID = pat->getRootTraitID()) {
      addToOpsWhen(pat, [&](AbstractOperation *absOp) {
        return absOp->hasTrait(*traitID);
      });
      continue;
    }
    impl->nativeAnyOpPatterns.push_back(std::move(pat));
  }

  // Generate the bytecode for the PDL patterns if any were provided.
  PDLPatternModule &pdlPatterns = patterns.getPDLPatterns();
  ModuleOp pdlModule = pdlPatterns.getModule();
  if (!pdlModule)
    return;
  if (failed(convertPDLToPDLInterp(pdlModule)))
    llvm::report_fatal_error(
        "failed to lower PDL pattern module to the PDL Interpreter");

  // Generate the pdl bytecode.
  impl->pdlByteCode = std::make_unique<detail::PDLByteCode>(
      pdlModule, pdlPatterns.takeConstraintFunctions(),
      pdlPatterns.takeRewriteFunctions());
}

FrozenRewritePatternSet::~FrozenRewritePatternSet() {}
