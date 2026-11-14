//===- FrozenRewritePatternSet.cpp - Frozen Pattern List -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "ByteCode.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include <optional>

using namespace mlir;

// Include the PDL rewrite support.
#if MLIR_ENABLE_PDL_IN_PATTERNMATCH
#include "mlir/Conversion/PDLToPDLInterp/PDLToPDLInterp.h"
#include "mlir/Dialect/PDL/IR/PDLOps.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"

static LogicalResult
convertPDLToPDLInterp(ModuleOp pdlModule,
                      DenseMap<Operation *, PDLPatternConfigSet *> &configMap) {
  // Skip the conversion if the module doesn't contain pdl.
  if (pdlModule.getOps<pdl::PatternOp>().empty())
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
  PassManager pdlPipeline(pdlModule->getName());
#ifdef NDEBUG
  // We don't want to incur the hit of running the verifier when in release
  // mode.
  pdlPipeline.enableVerifier(false);
#endif
  pdlPipeline.addPass(createConvertPDLToPDLInterpPass(configMap));
  if (failed(pdlPipeline.run(pdlModule)))
    return failure();

  // Simplify again after running the lowering pipeline.
  pdlModule.getBody()->walk(simplifyFn);
  return success();
}
#endif // MLIR_ENABLE_PDL_IN_PATTERNMATCH

//===----------------------------------------------------------------------===//
// FrozenRewritePatternSet
//===----------------------------------------------------------------------===//

FrozenRewritePatternSet::FrozenRewritePatternSet()
    : impl(std::make_shared<Impl>()) {}

FrozenRewritePatternSet::FrozenRewritePatternSet(
    RewritePatternSet &&patterns, ArrayRef<std::string> disabledPatternLabels,
    ArrayRef<std::string> enabledPatternLabels)
    : impl(std::make_shared<Impl>()) {
  // Functor used to walk all of the operations registered in the context. This
  // is useful for patterns that get applied to multiple operations, such as
  // interface and trait based patterns.
  std::vector<RegisteredOperationName> opInfos;
  auto addToOpsWhen =
      [&](std::unique_ptr<RewritePattern> &pattern,
          function_ref<bool(RegisteredOperationName)> callbackFn) {
        if (opInfos.empty())
          opInfos = pattern->getContext()->getRegisteredOperations();
        for (RegisteredOperationName info : opInfos)
          if (callbackFn(info))
            impl->nativeOpSpecificPatternMap[info].push_back(pattern.get());
        impl->nativeOpSpecificPatternList.push_back(std::move(pattern));
      };

  // Returns true if `label` (a pattern's debug name or one of its debug
  // labels) matches any entry in `userLabels`. A user label matches on exact
  // string equality; additionally, a user label that does not contain "::"
  // matches against the suffix of `label` after its last "::", so users can
  // write e.g. `disable-patterns=FooBar` instead of
  // `disable-patterns=(anonymous namespace)::FooBar`. Note: an unqualified
  // user label matches *any* pattern whose unqualified name is the same,
  // regardless of namespace.
  auto matchesAnyUserLabel = [](StringRef label,
                                ArrayRef<std::string> userLabels) {
    size_t pos = label.rfind("::");
    StringRef unqualified =
        (pos == StringRef::npos) ? label : label.substr(pos + 2);
    for (StringRef ul : userLabels) {
      if (label == ul)
        return true;
      if (!ul.contains("::") && unqualified == ul)
        return true;
    }
    return false;
  };

  for (std::unique_ptr<RewritePattern> &pat : patterns.getNativePatterns()) {
    // Don't add patterns that haven't been enabled by the user.
    if (!enabledPatternLabels.empty()) {
      auto isEnabledFn = [&](StringRef label) {
        return matchesAnyUserLabel(label, enabledPatternLabels);
      };
      if (!isEnabledFn(pat->getDebugName()) &&
          llvm::none_of(pat->getDebugLabels(), isEnabledFn))
        continue;
    }
    // Don't add patterns that have been disabled by the user.
    if (!disabledPatternLabels.empty()) {
      auto isDisabledFn = [&](StringRef label) {
        return matchesAnyUserLabel(label, disabledPatternLabels);
      };
      if (isDisabledFn(pat->getDebugName()) ||
          llvm::any_of(pat->getDebugLabels(), isDisabledFn))
        continue;
    }

    if (std::optional<OperationName> rootName = pat->getRootKind()) {
      impl->nativeOpSpecificPatternMap[*rootName].push_back(pat.get());
      impl->nativeOpSpecificPatternList.push_back(std::move(pat));
      continue;
    }
    if (std::optional<TypeID> interfaceID = pat->getRootInterfaceID()) {
      addToOpsWhen(pat, [&](RegisteredOperationName info) {
        return info.hasInterface(*interfaceID);
      });
      continue;
    }
    if (std::optional<TypeID> traitID = pat->getRootTraitID()) {
      addToOpsWhen(pat, [&](RegisteredOperationName info) {
        return info.hasTrait(*traitID);
      });
      continue;
    }
    impl->nativeAnyOpPatterns.push_back(std::move(pat));
  }

#if MLIR_ENABLE_PDL_IN_PATTERNMATCH
  // Generate the bytecode for the PDL patterns if any were provided.
  PDLPatternModule &pdlPatterns = patterns.getPDLPatterns();
  ModuleOp pdlModule = pdlPatterns.getModule();
  if (!pdlModule)
    return;
  DenseMap<Operation *, PDLPatternConfigSet *> configMap =
      pdlPatterns.takeConfigMap();
  if (failed(convertPDLToPDLInterp(pdlModule, configMap)))
    llvm::report_fatal_error(
        "failed to lower PDL pattern module to the PDL Interpreter");

  // Verify that the PDL module was actually lowered to the interpreter
  // dialect. If the lowering pass was skipped (e.g., by a debug counter
  // via --mlir-debug-counter), the matcher function will not be present and
  // we skip bytecode construction. PDL patterns will not be applied in this
  // case.
  if (!pdlModule.lookupSymbol(
          pdl_interp::PDLInterpDialect::getMatcherFunctionName()))
    return;

  // Generate the pdl bytecode.
  impl->pdlByteCode = std::make_unique<detail::PDLByteCode>(
      pdlModule, pdlPatterns.takeConfigs(), configMap,
      pdlPatterns.takeConstraintFunctions(),
      pdlPatterns.takeRewriteFunctions());
#endif // MLIR_ENABLE_PDL_IN_PATTERNMATCH
}

FrozenRewritePatternSet::~FrozenRewritePatternSet() = default;
