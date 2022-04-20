//===- TransformDialect.cpp - Transform dialect operations ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/PDL/IR/PDLOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "llvm/ADT/ScopeExit.h"

using namespace mlir;

#define GET_OP_CLASSES
#include "mlir/Dialect/Transform/IR/TransformOps.cpp.inc"

//===----------------------------------------------------------------------===//
// PatternApplicatorExtension
//===----------------------------------------------------------------------===//

namespace {
/// A simple pattern rewriter that can be constructed from a context. This is
/// necessary to apply patterns to a specific op locally.
class TrivialPatternRewriter : public PatternRewriter {
public:
  explicit TrivialPatternRewriter(MLIRContext *context)
      : PatternRewriter(context) {}
};

/// A TransformState extension that keeps track of compiled PDL pattern sets.
/// This is intended to be used along the WithPDLPatterns op. The extension
/// can be constructed given an operation that has a SymbolTable trait and
/// contains pdl::PatternOp instances. The patterns are compiled lazily and one
/// by one when requested; this behavior is subject to change.
class PatternApplicatorExtension : public transform::TransformState::Extension {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PatternApplicatorExtension)

  /// Creates the extension for patterns contained in `patternContainer`.
  explicit PatternApplicatorExtension(transform::TransformState &state,
                                      Operation *patternContainer)
      : Extension(state), patterns(patternContainer) {}

  /// Appends to `results` the operations contained in `root` that matched the
  /// PDL pattern with the given name. Note that `root` may or may not be the
  /// operation that contains PDL patterns. Reports an error if the pattern
  /// cannot be found. Note that when no operations are matched, this still
  /// succeeds as long as the pattern exists.
  LogicalResult findAllMatches(StringRef patternName, Operation *root,
                               SmallVectorImpl<Operation *> &results);

private:
  /// Map from the pattern name to a singleton set of rewrite patterns that only
  /// contains the pattern with this name. Populated when the pattern is first
  /// requested.
  // TODO: reconsider the efficiency of this storage when more usage data is
  // available. Storing individual patterns in a set and triggering compilation
  // for each of them has overhead. So does compiling a large set of patterns
  // only to apply a handlful of them.
  llvm::StringMap<FrozenRewritePatternSet> compiledPatterns;

  /// A symbol table operation containing the relevant PDL patterns.
  SymbolTable patterns;
};

LogicalResult PatternApplicatorExtension::findAllMatches(
    StringRef patternName, Operation *root,
    SmallVectorImpl<Operation *> &results) {
  auto it = compiledPatterns.find(patternName);
  if (it == compiledPatterns.end()) {
    auto patternOp = patterns.lookup<pdl::PatternOp>(patternName);
    if (!patternOp)
      return failure();

    OwningOpRef<ModuleOp> pdlModuleOp = ModuleOp::create(patternOp.getLoc());
    patternOp->moveBefore(pdlModuleOp->getBody(),
                          pdlModuleOp->getBody()->end());
    PDLPatternModule patternModule(std::move(pdlModuleOp));

    // Merge in the hooks owned by the dialect. Make a copy as they may be
    // also used by the following operations.
    auto *dialect =
        root->getContext()->getLoadedDialect<transform::TransformDialect>();
    for (const auto &pair : dialect->getPDLConstraintHooks())
      patternModule.registerConstraintFunction(pair.first(), pair.second);

    // Register a noop rewriter because PDL requires patterns to end with some
    // rewrite call.
    patternModule.registerRewriteFunction(
        "transform.dialect", [](PatternRewriter &, Operation *) {});

    it = compiledPatterns
             .try_emplace(patternOp.getName(), std::move(patternModule))
             .first;
  }

  PatternApplicator applicator(it->second);
  TrivialPatternRewriter rewriter(root->getContext());
  applicator.applyDefaultCostModel();
  root->walk([&](Operation *op) {
    if (succeeded(applicator.matchAndRewrite(op, rewriter)))
      results.push_back(op);
  });

  return success();
}
} // namespace

//===----------------------------------------------------------------------===//
// PDLMatchOp
//===----------------------------------------------------------------------===//

LogicalResult transform::PDLMatchOp::apply(transform::TransformResults &results,
                                           transform::TransformState &state) {
  auto *extension = state.getExtension<PatternApplicatorExtension>();
  assert(extension &&
         "expected PatternApplicatorExtension to be attached by the parent op");
  SmallVector<Operation *> targets;
  for (Operation *root : state.getPayloadOps(getRoot())) {
    if (failed(extension->findAllMatches(
            getPatternName().getLeafReference().getValue(), root, targets))) {
      return emitOpError() << "could not find pattern '" << getPatternName()
                           << "'";
    }
  }
  results.set(getResult().cast<OpResult>(), targets);
  return success();
}

//===----------------------------------------------------------------------===//
// SequenceOp
//===----------------------------------------------------------------------===//

LogicalResult transform::SequenceOp::apply(transform::TransformResults &results,
                                           transform::TransformState &state) {
  // Map the entry block argument to the list of operations.
  auto scope = state.make_region_scope(*getBodyBlock()->getParent());
  if (failed(mapBlockArguments(state)))
    return failure();

  // Apply the sequenced ops one by one.
  for (Operation &transform : getBodyBlock()->without_terminator())
    if (failed(state.applyTransform(cast<TransformOpInterface>(transform))))
      return failure();

  // Forward the operation mapping for values yielded from the sequence to the
  // values produced by the sequence op.
  for (const auto &pair :
       llvm::zip(getBodyBlock()->getTerminator()->getOperands(),
                 getOperation()->getOpResults())) {
    Value terminatorOperand = std::get<0>(pair);
    OpResult result = std::get<1>(pair);
    results.set(result, state.getPayloadOps(terminatorOperand));
  }

  return success();
}

LogicalResult transform::SequenceOp::verify() {
  for (Operation &child : *getBodyBlock()) {
    if (!isa<TransformOpInterface>(child) &&
        &child != &getBodyBlock()->back()) {
      InFlightDiagnostic diag =
          emitOpError()
          << "expected children ops to implement TransformOpInterface";
      diag.attachNote(child.getLoc()) << "op without interface";
      return diag;
    }

    for (OpResult result : child.getResults()) {
      if (llvm::hasNItemsOrLess(result.getUses(), 1))
        continue;
      InFlightDiagnostic diag = child.emitError()
                                << "result #" << result.getResultNumber()
                                << " has more than one use";
      for (OpOperand &use : result.getUses()) {
        diag.attachNote(use.getOwner()->getLoc())
            << "used here as operand #" << use.getOperandNumber();
      }
      return diag;
    }
  }

  if (getBodyBlock()->getTerminator()->getOperandTypes() !=
      getOperation()->getResultTypes()) {
    InFlightDiagnostic diag = emitOpError()
                              << "expects the types of the terminator operands "
                                 "to match the types of the result";
    diag.attachNote(getBodyBlock()->getTerminator()->getLoc()) << "terminator";
    return diag;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// WithPDLPatternsOp
//===----------------------------------------------------------------------===//

LogicalResult
transform::WithPDLPatternsOp::apply(transform::TransformResults &results,
                                    transform::TransformState &state) {
  OwningOpRef<ModuleOp> pdlModuleOp =
      ModuleOp::create(getOperation()->getLoc());
  TransformOpInterface transformOp = nullptr;
  for (Operation &nested : getBody().front()) {
    if (!isa<pdl::PatternOp>(nested)) {
      transformOp = cast<TransformOpInterface>(nested);
      break;
    }
  }

  state.addExtension<PatternApplicatorExtension>(getOperation());
  auto guard = llvm::make_scope_exit(
      [&]() { state.removeExtension<PatternApplicatorExtension>(); });

  auto scope = state.make_region_scope(getBody());
  if (failed(mapBlockArguments(state)))
    return failure();
  return state.applyTransform(transformOp);
}

LogicalResult transform::WithPDLPatternsOp::verify() {
  Block *body = getBodyBlock();
  Operation *topLevelOp = nullptr;
  for (Operation &op : body->getOperations()) {
    if (isa<pdl::PatternOp>(op))
      continue;

    if (op.hasTrait<::mlir::transform::PossibleTopLevelTransformOpTrait>()) {
      if (topLevelOp) {
        InFlightDiagnostic diag =
            emitOpError() << "expects only one non-pattern op in its body";
        diag.attachNote(topLevelOp->getLoc()) << "first non-pattern op";
        diag.attachNote(op.getLoc()) << "second non-pattern op";
        return diag;
      }
      topLevelOp = &op;
      continue;
    }

    InFlightDiagnostic diag =
        emitOpError()
        << "expects only pattern and top-level transform ops in its body";
    diag.attachNote(op.getLoc()) << "offending op";
    return diag;
  }

  if (auto parent = getOperation()->getParentOfType<WithPDLPatternsOp>()) {
    InFlightDiagnostic diag = emitOpError() << "cannot be nested";
    diag.attachNote(parent.getLoc()) << "parent operation";
    return diag;
  }

  return success();
}
