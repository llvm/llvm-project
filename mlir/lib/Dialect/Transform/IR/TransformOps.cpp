//===- TransformDialect.cpp - Transform dialect operations ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/PDL/IR/PDLOps.h"
#include "mlir/Dialect/Transform/IR/MatchInterfaces.h"
#include "mlir/Dialect/Transform/IR/TransformAttrs.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include <optional>

#define DEBUG_TYPE "transform-dialect"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "] ")

#define DEBUG_TYPE_MATCHER "transform-matcher"
#define DBGS_MATCHER() (llvm::dbgs() << "[" DEBUG_TYPE_MATCHER "] ")
#define DEBUG_MATCHER(x) DEBUG_WITH_TYPE(DEBUG_TYPE_MATCHER, x)

using namespace mlir;

static ParseResult parseSequenceOpOperands(
    OpAsmParser &parser, std::optional<OpAsmParser::UnresolvedOperand> &root,
    Type &rootType,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &extraBindings,
    SmallVectorImpl<Type> &extraBindingTypes);
static void printSequenceOpOperands(OpAsmPrinter &printer, Operation *op,
                                    Value root, Type rootType,
                                    ValueRange extraBindings,
                                    TypeRange extraBindingTypes);
static void printForeachMatchSymbols(OpAsmPrinter &printer, Operation *op,
                                     ArrayAttr matchers, ArrayAttr actions);
static ParseResult parseForeachMatchSymbols(OpAsmParser &parser,
                                            ArrayAttr &matchers,
                                            ArrayAttr &actions);

#define GET_OP_CLASSES
#include "mlir/Dialect/Transform/IR/TransformOps.cpp.inc"

//===----------------------------------------------------------------------===//
// PatternApplicatorExtension
//===----------------------------------------------------------------------===//

namespace {
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

    // Copy the pattern operation into a new module that is compiled and
    // consumed by the PDL interpreter.
    OwningOpRef<ModuleOp> pdlModuleOp = ModuleOp::create(patternOp.getLoc());
    auto builder = OpBuilder::atBlockEnd(pdlModuleOp->getBody());
    builder.clone(*patternOp);
    PDLPatternModule patternModule(std::move(pdlModuleOp));

    // Merge in the hooks owned by the dialect. Make a copy as they may be
    // also used by the following operations.
    auto *dialect =
        root->getContext()->getLoadedDialect<transform::TransformDialect>();
    for (const auto &[name, constraintFn] : dialect->getPDLConstraintHooks())
      patternModule.registerConstraintFunction(name, constraintFn);

    // Register a noop rewriter because PDL requires patterns to end with some
    // rewrite call.
    patternModule.registerRewriteFunction(
        "transform.dialect", [](PatternRewriter &, Operation *) {});

    it = compiledPatterns
             .try_emplace(patternOp.getName(), std::move(patternModule))
             .first;
  }

  PatternApplicator applicator(it->second);
  // We want to discourage direct use of PatternRewriter in APIs but In this
  // very specific case, an IRRewriter is not enough.
  struct TrivialPatternRewriter : public PatternRewriter {
  public:
    explicit TrivialPatternRewriter(MLIRContext *context)
        : PatternRewriter(context) {}
  };
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
// TrackingListener
//===----------------------------------------------------------------------===//

Operation *transform::TrackingListener::getCommonDefiningOp(ValueRange values) {
  Operation *defOp = nullptr;
  for (Value v : values) {
    // Skip empty values.
    if (!v)
      continue;
    if (!defOp) {
      defOp = v.getDefiningOp();
      continue;
    }
    if (defOp != v.getDefiningOp())
      return nullptr;
  }
  return defOp;
}

Operation *
transform::TrackingListener::findReplacementOp(Operation *op,
                                               ValueRange newValues) const {
  assert(op->getNumResults() == newValues.size() &&
         "invalid number of replacement values");

  // If the replacement values belong to different ops, drop the mapping.
  Operation *defOp = getCommonDefiningOp(newValues);
  if (!defOp)
    return nullptr;

  // If the replacement op has a different type, drop the mapping.
  if (op->getName() != defOp->getName())
    return nullptr;

  // If the replacement op is not a new op, drop the mapping.
  if (!isNewOp(defOp))
    return nullptr;

  return defOp;
}

bool transform::TrackingListener::isNewOp(Operation *op) const {
  auto it = newOps.find(op->getName());
  if (it == newOps.end())
    return false;
  return it->second.contains(op);
}

LogicalResult transform::TrackingListener::notifyMatchFailure(
    Location loc, function_ref<void(Diagnostic &)> reasonCallback) {
  LLVM_DEBUG({
    Diagnostic diag(loc, DiagnosticSeverity::Remark);
    reasonCallback(diag);
    DBGS() << "Match Failure : " << diag.str() << "\n";
  });
  return failure();
}

void transform::TrackingListener::notifyOperationInserted(Operation *op) {
  newOps[op->getName()].insert(op);
}

void transform::TrackingListener::notifyOperationRemoved(Operation *op) {
  // TODO: Walk can be removed when D144193 has landed.
  op->walk([&](Operation *op) {
    // Keep set of new ops up-to-date.
    auto it = newOps.find(op->getName());
    if (it != newOps.end())
      it->second.erase(op);
    // Remove mappings for result values.
    for (OpResult value : op->getResults())
      (void)replacePayloadValue(value, nullptr);
    // Remove mapping for op.
    (void)replacePayloadOp(op, nullptr);
  });
}

/// Return true if `a` happens before `b`, i.e., `a` or one of its ancestors
/// properly dominates `b` and `b` is not inside `a`.
static bool happensBefore(Operation *a, Operation *b) {
  do {
    if (a->isProperAncestor(b))
      return false;
    if (Operation *bAncestor = a->getBlock()->findAncestorOpInBlock(*b)) {
      return a->isBeforeInBlock(bAncestor);
    }
  } while ((a = a->getParentOp()));
  return false;
}

void transform::TrackingListener::notifyOperationReplaced(
    Operation *op, ValueRange newValues) {
  assert(op->getNumResults() == newValues.size() &&
         "invalid number of replacement values");

  // Replace value handles.
  for (auto [oldValue, newValue] : llvm::zip(op->getResults(), newValues))
    (void)replacePayloadValue(oldValue, newValue);

  // Replace op handle.
  SmallVector<Value> opHandles;
  if (failed(getTransformState().getHandlesForPayloadOp(op, opHandles))) {
    // Op is not tracked.
    return;
  }
  auto hasAliveUser = [&]() {
    for (Value v : opHandles)
      for (Operation *user : v.getUsers())
        if (!happensBefore(user, transformOp))
          return true;
    return false;
  };
  if (!hasAliveUser()) {
    // The op is tracked but the corresponding handles are dead.
    (void)replacePayloadOp(op, nullptr);
    return;
  }

  Operation *replacement = findReplacementOp(op, newValues);
  // If the op is tracked but no replacement op was found, send a
  // notification.
  if (!replacement)
    notifyPayloadReplacementNotFound(op, newValues);
  (void)replacePayloadOp(op, replacement);
}

//===----------------------------------------------------------------------===//
// AlternativesOp
//===----------------------------------------------------------------------===//

OperandRange transform::AlternativesOp::getSuccessorEntryOperands(
    std::optional<unsigned> index) {
  if (index && getOperation()->getNumOperands() == 1)
    return getOperation()->getOperands();
  return OperandRange(getOperation()->operand_end(),
                      getOperation()->operand_end());
}

void transform::AlternativesOp::getSuccessorRegions(
    std::optional<unsigned> index, ArrayRef<Attribute> operands,
    SmallVectorImpl<RegionSuccessor> &regions) {
  for (Region &alternative : llvm::drop_begin(
           getAlternatives(), index.has_value() ? *index + 1 : 0)) {
    regions.emplace_back(&alternative, !getOperands().empty()
                                           ? alternative.getArguments()
                                           : Block::BlockArgListType());
  }
  if (index.has_value())
    regions.emplace_back(getOperation()->getResults());
}

void transform::AlternativesOp::getRegionInvocationBounds(
    ArrayRef<Attribute> operands, SmallVectorImpl<InvocationBounds> &bounds) {
  (void)operands;
  // The region corresponding to the first alternative is always executed, the
  // remaining may or may not be executed.
  bounds.reserve(getNumRegions());
  bounds.emplace_back(1, 1);
  bounds.resize(getNumRegions(), InvocationBounds(0, 1));
}

static void forwardEmptyOperands(Block *block, transform::TransformState &state,
                                 transform::TransformResults &results) {
  for (const auto &res : block->getParentOp()->getOpResults())
    results.set(res, {});
}

DiagnosedSilenceableFailure
transform::AlternativesOp::apply(transform::TransformResults &results,
                                 transform::TransformState &state) {
  SmallVector<Operation *> originals;
  if (Value scopeHandle = getScope())
    llvm::append_range(originals, state.getPayloadOps(scopeHandle));
  else
    originals.push_back(state.getTopLevel());

  for (Operation *original : originals) {
    if (original->isAncestor(getOperation())) {
      auto diag = emitDefiniteFailure()
                  << "scope must not contain the transforms being applied";
      diag.attachNote(original->getLoc()) << "scope";
      return diag;
    }
    if (!original->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
      auto diag = emitDefiniteFailure()
                  << "only isolated-from-above ops can be alternative scopes";
      diag.attachNote(original->getLoc()) << "scope";
      return diag;
    }
  }

  for (Region &reg : getAlternatives()) {
    // Clone the scope operations and make the transforms in this alternative
    // region apply to them by virtue of mapping the block argument (the only
    // visible handle) to the cloned scope operations. This effectively prevents
    // the transformation from accessing any IR outside the scope.
    auto scope = state.make_region_scope(reg);
    auto clones = llvm::to_vector(
        llvm::map_range(originals, [](Operation *op) { return op->clone(); }));
    auto deleteClones = llvm::make_scope_exit([&] {
      for (Operation *clone : clones)
        clone->erase();
    });
    if (failed(state.mapBlockArguments(reg.front().getArgument(0), clones)))
      return DiagnosedSilenceableFailure::definiteFailure();

    bool failed = false;
    for (Operation &transform : reg.front().without_terminator()) {
      DiagnosedSilenceableFailure result =
          state.applyTransform(cast<TransformOpInterface>(transform));
      if (result.isSilenceableFailure()) {
        LLVM_DEBUG(DBGS() << "alternative failed: " << result.getMessage()
                          << "\n");
        failed = true;
        break;
      }

      if (::mlir::failed(result.silence()))
        return DiagnosedSilenceableFailure::definiteFailure();
    }

    // If all operations in the given alternative succeeded, no need to consider
    // the rest. Replace the original scoping operation with the clone on which
    // the transformations were performed.
    if (!failed) {
      // We will be using the clones, so cancel their scheduled deletion.
      deleteClones.release();
      TrackingListener listener(state, *this);
      IRRewriter rewriter(getContext(), &listener);
      for (const auto &kvp : llvm::zip(originals, clones)) {
        Operation *original = std::get<0>(kvp);
        Operation *clone = std::get<1>(kvp);
        original->getBlock()->getOperations().insert(original->getIterator(),
                                                     clone);
        rewriter.replaceOp(original, clone->getResults());
      }
      detail::forwardTerminatorOperands(&reg.front(), state, results);
      return DiagnosedSilenceableFailure::success();
    }
  }
  return emitSilenceableError() << "all alternatives failed";
}

void transform::AlternativesOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getOperands(), effects);
  producesHandle(getResults(), effects);
  for (Region *region : getRegions()) {
    if (!region->empty())
      producesHandle(region->front().getArguments(), effects);
  }
  modifiesPayload(effects);
}

LogicalResult transform::AlternativesOp::verify() {
  for (Region &alternative : getAlternatives()) {
    Block &block = alternative.front();
    Operation *terminator = block.getTerminator();
    if (terminator->getOperands().getTypes() != getResults().getTypes()) {
      InFlightDiagnostic diag = emitOpError()
                                << "expects terminator operands to have the "
                                   "same type as results of the operation";
      diag.attachNote(terminator->getLoc()) << "terminator";
      return diag;
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::CastOp::applyToOne(Operation *target, ApplyToEachResultList &results,
                              transform::TransformState &state) {
  results.push_back(target);
  return DiagnosedSilenceableFailure::success();
}

void transform::CastOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsPayload(effects);
  onlyReadsHandle(getInput(), effects);
  producesHandle(getOutput(), effects);
}

bool transform::CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  assert(inputs.size() == 1 && "expected one input");
  assert(outputs.size() == 1 && "expected one output");
  return llvm::all_of(
      std::initializer_list<Type>{inputs.front(), outputs.front()},
      [](Type ty) {
        return ty
            .isa<pdl::OperationType, transform::TransformHandleTypeInterface>();
      });
}

//===----------------------------------------------------------------------===//
// ForeachMatchOp
//===----------------------------------------------------------------------===//

/// Applies matcher operations from the given `block` assigning `op` as the
/// payload of the block's first argument. Updates `state` accordingly. If any
/// of the matcher produces a silenceable failure, discards it (printing the
/// content to the debug output stream) and returns failure. If any of the
/// matchers produces a definite failure, reports it and returns failure. If all
/// matchers in the block succeed, populates `mappings` with the payload
/// entities associated with the block terminator operands.
static DiagnosedSilenceableFailure
matchBlock(Block &block, Operation *op, transform::TransformState &state,
           SmallVectorImpl<SmallVector<transform::MappedValue>> &mappings) {
  assert(block.getParent() && "cannot match using a detached block");
  auto matchScope = state.make_isolated_region_scope(*block.getParent());
  if (failed(state.mapBlockArgument(block.getArgument(0), {op})))
    return DiagnosedSilenceableFailure::definiteFailure();

  for (Operation &match : block.without_terminator()) {
    if (!isa<transform::MatchOpInterface>(match)) {
      return emitDefiniteFailure(match.getLoc())
             << "expected operations in the match part to "
                "implement MatchOpInterface";
    }
    DiagnosedSilenceableFailure diag =
        state.applyTransform(cast<transform::TransformOpInterface>(match));
    if (diag.succeeded())
      continue;

    return diag;
  }

  // Remember the values mapped to the terminator operands so we can
  // forward them to the action.
  ValueRange yieldedValues = block.getTerminator()->getOperands();
  transform::detail::prepareValueMappings(mappings, yieldedValues, state);
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure
transform::ForeachMatchOp::apply(transform::TransformResults &results,
                                 transform::TransformState &state) {
  SmallVector<std::pair<FunctionOpInterface, FunctionOpInterface>>
      matchActionPairs;
  matchActionPairs.reserve(getMatchers().size());
  SymbolTableCollection symbolTable;
  for (auto &&[matcher, action] :
       llvm::zip_equal(getMatchers(), getActions())) {
    auto matcherSymbol =
        symbolTable.lookupNearestSymbolFrom<FunctionOpInterface>(
            getOperation(), cast<SymbolRefAttr>(matcher));
    auto actionSymbol =
        symbolTable.lookupNearestSymbolFrom<FunctionOpInterface>(
            getOperation(), cast<SymbolRefAttr>(action));
    assert(matcherSymbol && actionSymbol &&
           "unresolved symbols not caught by the verifier");

    if (matcherSymbol.isExternal())
      return emitDefiniteFailure() << "unresolved external symbol " << matcher;
    if (actionSymbol.isExternal())
      return emitDefiniteFailure() << "unresolved external symbol " << action;

    matchActionPairs.emplace_back(matcherSymbol, actionSymbol);
  }

  for (Operation *root : state.getPayloadOps(getRoot())) {
    WalkResult walkResult = root->walk([&](Operation *op) {
      // Skip over the root op itself so we don't invalidate it.
      if (op == root)
        return WalkResult::advance();

      DEBUG_MATCHER({
        DBGS_MATCHER() << "matching ";
        op->print(llvm::dbgs(),
                  OpPrintingFlags().assumeVerified().skipRegions());
        llvm::dbgs() << " @" << op << "\n";
      });

      // Try all the match/action pairs until the first successful match.
      for (auto [matcher, action] : matchActionPairs) {
        SmallVector<SmallVector<MappedValue>> mappings;
        DiagnosedSilenceableFailure diag =
            matchBlock(matcher.getFunctionBody().front(), op, state, mappings);
        if (diag.isDefiniteFailure())
          return WalkResult::interrupt();
        if (diag.isSilenceableFailure()) {
          DEBUG_MATCHER(DBGS_MATCHER() << "matcher " << matcher.getName()
                                       << " failed: " << diag.getMessage());
          continue;
        }

        auto scope = state.make_isolated_region_scope(action.getFunctionBody());
        for (auto &&[arg, map] : llvm::zip_equal(
                 action.getFunctionBody().front().getArguments(), mappings)) {
          if (failed(state.mapBlockArgument(arg, map)))
            return WalkResult::interrupt();
        }

        for (Operation &transform :
             action.getFunctionBody().front().without_terminator()) {
          DiagnosedSilenceableFailure result =
              state.applyTransform(cast<TransformOpInterface>(transform));
          if (failed(result.checkAndReport()))
            return WalkResult::interrupt();
        }
        break;
      }
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted())
      return DiagnosedSilenceableFailure::definiteFailure();
  }

  // The root operation should not have been affected, so we can just reassign
  // the payload to the result. Note that we need to consume the root handle to
  // make sure any handles to operations inside, that could have been affected
  // by actions, are invalidated.
  results.set(getUpdated().cast<OpResult>(), state.getPayloadOps(getRoot()));
  return DiagnosedSilenceableFailure::success();
}

void transform::ForeachMatchOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  // Bail if invalid.
  if (getOperation()->getNumOperands() < 1 ||
      getOperation()->getNumResults() < 1) {
    return modifiesPayload(effects);
  }

  consumesHandle(getRoot(), effects);
  producesHandle(getUpdated(), effects);
  modifiesPayload(effects);
}

/// Parses the comma-separated list of symbol reference pairs of the format
/// `@matcher -> @action`.
static ParseResult parseForeachMatchSymbols(OpAsmParser &parser,
                                            ArrayAttr &matchers,
                                            ArrayAttr &actions) {
  StringAttr matcher;
  StringAttr action;
  SmallVector<Attribute> matcherList;
  SmallVector<Attribute> actionList;
  do {
    if (parser.parseSymbolName(matcher) || parser.parseArrow() ||
        parser.parseSymbolName(action)) {
      return failure();
    }
    matcherList.push_back(SymbolRefAttr::get(matcher));
    actionList.push_back(SymbolRefAttr::get(action));
  } while (parser.parseOptionalComma().succeeded());

  matchers = parser.getBuilder().getArrayAttr(matcherList);
  actions = parser.getBuilder().getArrayAttr(actionList);
  return success();
}

/// Prints the comma-separated list of symbol reference pairs of the format
/// `@matcher -> @action`.
static void printForeachMatchSymbols(OpAsmPrinter &printer, Operation *op,
                                     ArrayAttr matchers, ArrayAttr actions) {
  printer.increaseIndent();
  printer.increaseIndent();
  for (auto &&[matcher, action, idx] : llvm::zip_equal(
           matchers, actions, llvm::seq<unsigned>(0, matchers.size()))) {
    printer.printNewline();
    printer << cast<SymbolRefAttr>(matcher) << " -> "
            << cast<SymbolRefAttr>(action);
    if (idx != matchers.size() - 1)
      printer << ", ";
  }
  printer.decreaseIndent();
  printer.decreaseIndent();
}

LogicalResult transform::ForeachMatchOp::verify() {
  if (getMatchers().size() != getActions().size())
    return emitOpError() << "expected the same number of matchers and actions";
  if (getMatchers().empty())
    return emitOpError() << "expected at least one match/action pair";

  llvm::SmallPtrSet<Attribute, 8> matcherNames;
  for (Attribute name : getMatchers()) {
    if (matcherNames.insert(name).second)
      continue;
    emitWarning() << "matcher " << name
                  << " is used more than once, only the first match will apply";
  }

  return success();
}

/// Returns `true` if both types implement one of the interfaces provided as
/// template parameters.
template <typename... Tys>
static bool implementSameInterface(Type t1, Type t2) {
  return ((isa<Tys>(t1) && isa<Tys>(t2)) || ... || false);
}

/// Returns `true` if both types implement one of the transform dialect
/// interfaces.
static bool implementSameTransformInterface(Type t1, Type t2) {
  return implementSameInterface<transform::TransformHandleTypeInterface,
                                transform::TransformParamTypeInterface,
                                transform::TransformValueHandleTypeInterface>(
      t1, t2);
}

/// Checks that the attributes of the function-like operation have correct
/// consumption effect annotations. If `alsoVerifyInternal`, checks for
/// annotations being present even if they can be inferred from the body.
static DiagnosedSilenceableFailure
verifyFunctionLikeConsumeAnnotations(FunctionOpInterface op,
                                     bool alsoVerifyInternal = false) {
  auto transformOp = cast<transform::TransformOpInterface>(op.getOperation());
  llvm::SmallDenseSet<unsigned> consumedArguments;
  if (!op.isExternal()) {
    transform::getConsumedBlockArguments(op.getFunctionBody().front(),
                                         consumedArguments);
  }
  for (unsigned i = 0, e = op.getNumArguments(); i < e; ++i) {
    bool isConsumed =
        op.getArgAttr(i, transform::TransformDialect::kArgConsumedAttrName) !=
        nullptr;
    bool isReadOnly =
        op.getArgAttr(i, transform::TransformDialect::kArgReadOnlyAttrName) !=
        nullptr;
    if (isConsumed && isReadOnly) {
      return transformOp.emitSilenceableError()
             << "argument #" << i << " cannot be both readonly and consumed";
    }
    if ((op.isExternal() || alsoVerifyInternal) && !isConsumed && !isReadOnly) {
      return transformOp.emitSilenceableError()
             << "must provide consumed/readonly status for arguments of "
                "external or called ops";
    }
    if (op.isExternal())
      continue;

    if (consumedArguments.contains(i) && !isConsumed && isReadOnly) {
      return transformOp.emitSilenceableError()
             << "argument #" << i
             << " is consumed in the body but is not marked as such";
    }
    if (!consumedArguments.contains(i) && isConsumed) {
      Diagnostic warning(op->getLoc(), DiagnosticSeverity::Warning);
      warning << "argument #" << i
              << " is not consumed in the body but is marked as consumed";
      return DiagnosedSilenceableFailure::silenceableFailure(
          std::move(warning));
    }
  }
  return DiagnosedSilenceableFailure::success();
}

LogicalResult transform::ForeachMatchOp::verifySymbolUses(
    SymbolTableCollection &symbolTable) {
  assert(getMatchers().size() == getActions().size());
  auto consumedAttr =
      StringAttr::get(getContext(), TransformDialect::kArgConsumedAttrName);
  for (auto &&[matcher, action] :
       llvm::zip_equal(getMatchers(), getActions())) {
    auto matcherSymbol = dyn_cast_or_null<FunctionOpInterface>(
        symbolTable.lookupNearestSymbolFrom(getOperation(),
                                            cast<SymbolRefAttr>(matcher)));
    auto actionSymbol = dyn_cast_or_null<FunctionOpInterface>(
        symbolTable.lookupNearestSymbolFrom(getOperation(),
                                            cast<SymbolRefAttr>(action)));
    if (!matcherSymbol ||
        !isa<TransformOpInterface>(matcherSymbol.getOperation()))
      return emitError() << "unresolved matcher symbol " << matcher;
    if (!actionSymbol ||
        !isa<TransformOpInterface>(actionSymbol.getOperation()))
      return emitError() << "unresolved action symbol " << action;

    if (failed(verifyFunctionLikeConsumeAnnotations(matcherSymbol,
                                                    /*alsoVerifyInternal=*/true)
                   .checkAndReport())) {
      return failure();
    }
    if (failed(verifyFunctionLikeConsumeAnnotations(actionSymbol,
                                                    /*alsoVerifyInternal=*/true)
                   .checkAndReport())) {
      return failure();
    }

    ArrayRef<Type> matcherResults = matcherSymbol.getResultTypes();
    ArrayRef<Type> actionArguments = actionSymbol.getArgumentTypes();
    if (matcherResults.size() != actionArguments.size()) {
      return emitError() << "mismatching number of matcher results and "
                            "action arguments between "
                         << matcher << " (" << matcherResults.size() << ") and "
                         << action << " (" << actionArguments.size() << ")";
    }
    for (auto &&[i, matcherType, actionType] :
         llvm::enumerate(matcherResults, actionArguments)) {
      if (implementSameTransformInterface(matcherType, actionType))
        continue;

      return emitError() << "mismatching type interfaces for matcher result "
                            "and action argument #"
                         << i;
    }

    if (!actionSymbol.getResultTypes().empty()) {
      InFlightDiagnostic diag =
          emitError() << "action symbol is not expected to have results";
      diag.attachNote(actionSymbol->getLoc()) << "symbol declaration";
      return diag;
    }

    if (matcherSymbol.getArgumentTypes().size() != 1 ||
        !implementSameTransformInterface(matcherSymbol.getArgumentTypes()[0],
                                         getRoot().getType())) {
      InFlightDiagnostic diag =
          emitOpError() << "expects matcher symbol to have one argument with "
                           "the same transform interface as the first operand";
      diag.attachNote(matcherSymbol->getLoc()) << "symbol declaration";
      return diag;
    }

    if (matcherSymbol.getArgAttr(0, consumedAttr)) {
      InFlightDiagnostic diag =
          emitOpError()
          << "does not expect matcher symbol to consume its operand";
      diag.attachNote(matcherSymbol->getLoc()) << "symbol declaration";
      return diag;
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ForeachOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::ForeachOp::apply(transform::TransformResults &results,
                            transform::TransformState &state) {
  ArrayRef<Operation *> payloadOps = state.getPayloadOps(getTarget());
  SmallVector<SmallVector<Operation *>> resultOps(getNumResults(), {});

  for (Operation *op : payloadOps) {
    auto scope = state.make_region_scope(getBody());
    if (failed(state.mapBlockArguments(getIterationVariable(), {op})))
      return DiagnosedSilenceableFailure::definiteFailure();

    // Execute loop body.
    for (Operation &transform : getBody().front().without_terminator()) {
      DiagnosedSilenceableFailure result = state.applyTransform(
          cast<transform::TransformOpInterface>(transform));
      if (!result.succeeded())
        return result;
    }

    // Append yielded payload ops to result list (if any).
    for (unsigned i = 0; i < getNumResults(); ++i) {
      ArrayRef<Operation *> yieldedOps =
          state.getPayloadOps(getYieldOp().getOperand(i));
      resultOps[i].append(yieldedOps.begin(), yieldedOps.end());
    }
  }

  for (unsigned i = 0; i < getNumResults(); ++i)
    results.set(getResult(i).cast<OpResult>(), resultOps[i]);

  return DiagnosedSilenceableFailure::success();
}

void transform::ForeachOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  BlockArgument iterVar = getIterationVariable();
  if (any_of(getBody().front().without_terminator(), [&](Operation &op) {
        return isHandleConsumed(iterVar, cast<TransformOpInterface>(&op));
      })) {
    consumesHandle(getTarget(), effects);
  } else {
    onlyReadsHandle(getTarget(), effects);
  }

  for (Value result : getResults())
    producesHandle(result, effects);
}

void transform::ForeachOp::getSuccessorRegions(
    std::optional<unsigned> index, ArrayRef<Attribute> operands,
    SmallVectorImpl<RegionSuccessor> &regions) {
  Region *bodyRegion = &getBody();
  if (!index) {
    regions.emplace_back(bodyRegion, bodyRegion->getArguments());
    return;
  }

  // Branch back to the region or the parent.
  assert(*index == 0 && "unexpected region index");
  regions.emplace_back(bodyRegion, bodyRegion->getArguments());
  regions.emplace_back();
}

OperandRange
transform::ForeachOp::getSuccessorEntryOperands(std::optional<unsigned> index) {
  // The iteration variable op handle is mapped to a subset (one op to be
  // precise) of the payload ops of the ForeachOp operand.
  assert(index && *index == 0 && "unexpected region index");
  return getOperation()->getOperands();
}

transform::YieldOp transform::ForeachOp::getYieldOp() {
  return cast<transform::YieldOp>(getBody().front().getTerminator());
}

LogicalResult transform::ForeachOp::verify() {
  auto yieldOp = getYieldOp();
  if (getNumResults() != yieldOp.getNumOperands())
    return emitOpError() << "expects the same number of results as the "
                            "terminator has operands";
  for (Value v : yieldOp.getOperands())
    if (!v.getType().isa<TransformHandleTypeInterface>())
      return yieldOp->emitOpError("expects operands to have types implementing "
                                  "TransformHandleTypeInterface");
  return success();
}

//===----------------------------------------------------------------------===//
// GetClosestIsolatedParentOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::GetClosestIsolatedParentOp::apply(
    transform::TransformResults &results, transform::TransformState &state) {
  SetVector<Operation *> parents;
  for (Operation *target : state.getPayloadOps(getTarget())) {
    Operation *parent =
        target->getParentWithTrait<OpTrait::IsIsolatedFromAbove>();
    if (!parent) {
      DiagnosedSilenceableFailure diag =
          emitSilenceableError()
          << "could not find an isolated-from-above parent op";
      diag.attachNote(target->getLoc()) << "target op";
      return diag;
    }
    parents.insert(parent);
  }
  results.set(getResult().cast<OpResult>(), parents.getArrayRef());
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// GetConsumersOfResult
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::GetConsumersOfResult::apply(transform::TransformResults &results,
                                       transform::TransformState &state) {
  int64_t resultNumber = getResultNumber();
  ArrayRef<Operation *> payloadOps = state.getPayloadOps(getTarget());
  if (payloadOps.empty()) {
    results.set(getResult().cast<OpResult>(), {});
    return DiagnosedSilenceableFailure::success();
  }
  if (payloadOps.size() != 1)
    return emitDefiniteFailure()
           << "handle must be mapped to exactly one payload op";

  Operation *target = payloadOps.front();
  if (target->getNumResults() <= resultNumber)
    return emitDefiniteFailure() << "result number overflow";
  results.set(getResult().cast<OpResult>(),
              llvm::to_vector(target->getResult(resultNumber).getUsers()));
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// GetDefiningOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::GetDefiningOp::apply(transform::TransformResults &results,
                                transform::TransformState &state) {
  SmallVector<Operation *> definingOps;
  for (Value v : state.getPayloadValues(getTarget())) {
    if (v.isa<BlockArgument>()) {
      DiagnosedSilenceableFailure diag =
          emitSilenceableError() << "cannot get defining op of block argument";
      diag.attachNote(v.getLoc()) << "target value";
      return diag;
    }
    definingOps.push_back(v.getDefiningOp());
  }
  results.set(getResult().cast<OpResult>(), definingOps);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// GetProducerOfOperand
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::GetProducerOfOperand::apply(transform::TransformResults &results,
                                       transform::TransformState &state) {
  int64_t operandNumber = getOperandNumber();
  SmallVector<Operation *> producers;
  for (Operation *target : state.getPayloadOps(getTarget())) {
    Operation *producer =
        target->getNumOperands() <= operandNumber
            ? nullptr
            : target->getOperand(operandNumber).getDefiningOp();
    if (!producer) {
      DiagnosedSilenceableFailure diag =
          emitSilenceableError()
          << "could not find a producer for operand number: " << operandNumber
          << " of " << *target;
      diag.attachNote(target->getLoc()) << "target op";
      return diag;
    }
    producers.push_back(producer);
  }
  results.set(getResult().cast<OpResult>(), producers);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// GetResultOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::GetResultOp::apply(transform::TransformResults &results,
                              transform::TransformState &state) {
  int64_t resultNumber = getResultNumber();
  SmallVector<Value> opResults;
  for (Operation *target : state.getPayloadOps(getTarget())) {
    if (resultNumber >= target->getNumResults()) {
      DiagnosedSilenceableFailure diag =
          emitSilenceableError() << "targeted op does not have enough results";
      diag.attachNote(target->getLoc()) << "target op";
      return diag;
    }
    opResults.push_back(target->getOpResult(resultNumber));
  }
  results.setValues(getResult().cast<OpResult>(), opResults);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// IncludeOp
//===----------------------------------------------------------------------===//

/// Applies the transform ops contained in `block`. Maps `results` to the same
/// values as the operands of the block terminator.
static DiagnosedSilenceableFailure
applySequenceBlock(Block &block, transform::FailurePropagationMode mode,
                   transform::TransformState &state,
                   transform::TransformResults &results) {
  // Apply the sequenced ops one by one.
  for (Operation &transform : block.without_terminator()) {
    DiagnosedSilenceableFailure result =
        state.applyTransform(cast<transform::TransformOpInterface>(transform));
    if (result.isDefiniteFailure())
      return result;

    if (result.isSilenceableFailure()) {
      if (mode == transform::FailurePropagationMode::Propagate) {
        // Propagate empty results in case of early exit.
        forwardEmptyOperands(&block, state, results);
        return result;
      }
      (void)result.silence();
    }
  }

  // Forward the operation mapping for values yielded from the sequence to the
  // values produced by the sequence op.
  transform::detail::forwardTerminatorOperands(&block, state, results);
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure
transform::IncludeOp::apply(transform::TransformResults &results,
                            transform::TransformState &state) {
  auto callee = SymbolTable::lookupNearestSymbolFrom<NamedSequenceOp>(
      getOperation(), getTarget());
  assert(callee && "unverified reference to unknown symbol");

  if (callee.isExternal())
    return emitDefiniteFailure() << "unresolved external named sequence";

  // Map operands to block arguments.
  SmallVector<SmallVector<MappedValue>> mappings;
  detail::prepareValueMappings(mappings, getOperands(), state);
  auto scope = state.make_isolated_region_scope(callee.getBody());
  for (auto &&[arg, map] :
       llvm::zip_equal(callee.getBody().front().getArguments(), mappings)) {
    if (failed(state.mapBlockArgument(arg, map)))
      return DiagnosedSilenceableFailure::definiteFailure();
  }

  DiagnosedSilenceableFailure result = applySequenceBlock(
      callee.getBody().front(), getFailurePropagationMode(), state, results);
  mappings.clear();
  detail::prepareValueMappings(
      mappings, callee.getBody().front().getTerminator()->getOperands(), state);
  for (auto &&[result, mapping] : llvm::zip_equal(getResults(), mappings))
    results.setMappedValues(result, mapping);
  return result;
}

/// Appends to `effects` the memory effect instances on `target` with the same
/// resource and effect as the ones the operation `iface` having on `source`.
static void
remapEffects(MemoryEffectOpInterface iface, BlockArgument source, Value target,
             SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  SmallVector<MemoryEffects::EffectInstance> nestedEffects;
  iface.getEffectsOnValue(source, nestedEffects);
  for (const auto &effect : nestedEffects)
    effects.emplace_back(effect.getEffect(), target, effect.getResource());
}

/// Appends to `effects` the same effects as the operations of `block` have on
/// block arguments but associated with `operands.`
static void
remapArgumentEffects(Block &block, ValueRange operands,
                     SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  for (Operation &op : block) {
    auto iface = dyn_cast<MemoryEffectOpInterface>(&op);
    if (!iface)
      continue;

    for (auto &&[source, target] : llvm::zip(block.getArguments(), operands)) {
      remapEffects(iface, source, target, effects);
    }

    SmallVector<MemoryEffects::EffectInstance> nestedEffects;
    iface.getEffectsOnResource(transform::PayloadIRResource::get(),
                               nestedEffects);
    llvm::append_range(effects, nestedEffects);
  }
}

static DiagnosedSilenceableFailure
verifyNamedSequenceOp(transform::NamedSequenceOp op);

void transform::IncludeOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  // Always mark as modifying the payload.
  // TODO: a mechanism to annotate effects on payload. Even when all handles are
  // only read, the payload may still be modified, so we currently stay on the
  // conservative side and always indicate modification. This may prevent some
  // code reordering.
  modifiesPayload(effects);

  // Results are always produced.
  producesHandle(getResults(), effects);

  // Adds default effects to operands and results. This will be added if
  // preconditions fail so the trait verifier doesn't complain about missing
  // effects and the real precondition failure is reported later on.
  auto defaultEffects = [&] { onlyReadsHandle(getOperands(), effects); };

  // Bail if the callee is unknown. This may run as part of the verification
  // process before we verified the validity of the callee or of this op.
  auto target =
      getOperation()->getAttrOfType<SymbolRefAttr>(getTargetAttrName());
  if (!target)
    return defaultEffects();
  auto callee = SymbolTable::lookupNearestSymbolFrom<NamedSequenceOp>(
      getOperation(), getTarget());
  if (!callee)
    return defaultEffects();
  DiagnosedSilenceableFailure earlyVerifierResult =
      verifyNamedSequenceOp(callee);
  if (!earlyVerifierResult.succeeded()) {
    (void)earlyVerifierResult.silence();
    return defaultEffects();
  }

  for (unsigned i = 0, e = getNumOperands(); i < e; ++i) {
    if (callee.getArgAttr(i, TransformDialect::kArgConsumedAttrName))
      consumesHandle(getOperand(i), effects);
    else
      onlyReadsHandle(getOperand(i), effects);
  }
}

LogicalResult
transform::IncludeOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Access through indirection and do additional checking because this may be
  // running before the main op verifier.
  auto targetAttr = getOperation()->getAttrOfType<SymbolRefAttr>("target");
  if (!targetAttr)
    return emitOpError() << "expects a 'target' symbol reference attribute";

  auto target = symbolTable.lookupNearestSymbolFrom<transform::NamedSequenceOp>(
      *this, targetAttr);
  if (!target)
    return emitOpError() << "does not reference a named transform sequence";

  FunctionType fnType = target.getFunctionType();
  if (fnType.getNumInputs() != getNumOperands())
    return emitError("incorrect number of operands for callee");

  for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i) {
    if (getOperand(i).getType() != fnType.getInput(i)) {
      return emitOpError("operand type mismatch: expected operand type ")
             << fnType.getInput(i) << ", but provided "
             << getOperand(i).getType() << " for operand number " << i;
    }
  }

  if (fnType.getNumResults() != getNumResults())
    return emitError("incorrect number of results for callee");

  for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i) {
    Type resultType = getResult(i).getType();
    Type funcType = fnType.getResult(i);
    if (!implementSameTransformInterface(resultType, funcType)) {
      return emitOpError() << "type of result #" << i
                           << " must implement the same transform dialect "
                              "interface as the corresponding callee result";
    }
  }

  return verifyFunctionLikeConsumeAnnotations(
             cast<FunctionOpInterface>(*target),
             /*alsoVerifyInternal=*/true)
      .checkAndReport();
}

//===----------------------------------------------------------------------===//
// MatchOperationNameOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::MatchOperationNameOp::matchOperation(
    Operation *current, transform::TransformResults &results,
    transform::TransformState &state) {
  StringRef currentOpName = current->getName().getStringRef();
  for (auto acceptedAttr : getOpNames().getAsRange<StringAttr>()) {
    if (acceptedAttr.getValue() == currentOpName)
      return DiagnosedSilenceableFailure::success();
  }
  return emitSilenceableError() << "wrong operation name";
}

//===----------------------------------------------------------------------===//
// MatchParamCmpIOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::MatchParamCmpIOp::apply(transform::TransformResults &results,
                                   transform::TransformState &state) {
  auto signedAPIntAsString = [&](APInt value) {
    std::string str;
    llvm::raw_string_ostream os(str);
    value.print(os, /*isSigned=*/true);
    return os.str();
  };

  ArrayRef<Attribute> params = state.getParams(getParam());
  ArrayRef<Attribute> references = state.getParams(getReference());

  if (params.size() != references.size()) {
    return emitSilenceableError()
           << "parameters have different payload lengths (" << params.size()
           << " vs " << references.size() << ")";
  }

  for (auto &&[i, param, reference] : llvm::enumerate(params, references)) {
    auto intAttr = param.dyn_cast<IntegerAttr>();
    auto refAttr = reference.dyn_cast<IntegerAttr>();
    if (!intAttr || !refAttr) {
      return emitDefiniteFailure()
             << "non-integer parameter value not expected";
    }
    if (intAttr.getType() != refAttr.getType()) {
      return emitDefiniteFailure()
             << "mismatching integer attribute types in parameter #" << i;
    }
    APInt value = intAttr.getValue();
    APInt refValue = refAttr.getValue();

    // TODO: this copy will not be necessary in C++20.
    int64_t position = i;
    auto reportError = [&](StringRef direction) {
      DiagnosedSilenceableFailure diag =
          emitSilenceableError() << "expected parameter to be " << direction
                                 << " " << signedAPIntAsString(refValue)
                                 << ", got " << signedAPIntAsString(value);
      diag.attachNote(getParam().getLoc())
          << "value # " << position
          << " associated with the parameter defined here";
      return diag;
    };

    switch (getPredicate()) {
    case MatchCmpIPredicate::eq:
      if (value.eq(refValue))
        break;
      return reportError("equal to");
    case MatchCmpIPredicate::ne:
      if (value.ne(refValue))
        break;
      return reportError("not equal to");
    case MatchCmpIPredicate::lt:
      if (value.slt(refValue))
        break;
      return reportError("less than");
    case MatchCmpIPredicate::le:
      if (value.sle(refValue))
        break;
      return reportError("less than or equal to");
    case MatchCmpIPredicate::gt:
      if (value.sgt(refValue))
        break;
      return reportError("greater than");
    case MatchCmpIPredicate::ge:
      if (value.sge(refValue))
        break;
      return reportError("greater than or equal to");
    }
  }
  return DiagnosedSilenceableFailure::success();
}

void transform::MatchParamCmpIOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getParam(), effects);
  onlyReadsHandle(getReference(), effects);
}

//===----------------------------------------------------------------------===//
// ParamConstantOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::ParamConstantOp::apply(transform::TransformResults &results,
                                  transform::TransformState &state) {
  results.setParams(cast<OpResult>(getParam()), {getValue()});
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// MergeHandlesOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::MergeHandlesOp::apply(transform::TransformResults &results,
                                 transform::TransformState &state) {
  SmallVector<Operation *> operations;
  for (Value operand : getHandles())
    llvm::append_range(operations, state.getPayloadOps(operand));
  if (!getDeduplicate()) {
    results.set(getResult().cast<OpResult>(), operations);
    return DiagnosedSilenceableFailure::success();
  }

  SetVector<Operation *> uniqued(operations.begin(), operations.end());
  results.set(getResult().cast<OpResult>(), uniqued.getArrayRef());
  return DiagnosedSilenceableFailure::success();
}

bool transform::MergeHandlesOp::allowsRepeatedHandleOperands() {
  // Handles may be the same if deduplicating is enabled.
  return getDeduplicate();
}

void transform::MergeHandlesOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getHandles(), effects);
  producesHandle(getResult(), effects);

  // There are no effects on the Payload IR as this is only a handle
  // manipulation.
}

OpFoldResult transform::MergeHandlesOp::fold(FoldAdaptor adaptor) {
  if (getDeduplicate() || getHandles().size() != 1)
    return {};

  // If deduplication is not required and there is only one operand, it can be
  // used directly instead of merging.
  return getHandles().front();
}

//===----------------------------------------------------------------------===//
// NamedSequenceOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::NamedSequenceOp::apply(transform::TransformResults &results,
                                  transform::TransformState &state) {
  // Nothing to do here.
  return DiagnosedSilenceableFailure::success();
}

void transform::NamedSequenceOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {}

ParseResult transform::NamedSequenceOp::parse(OpAsmParser &parser,
                                              OperationState &result) {
  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name),
      [](Builder &builder, ArrayRef<Type> inputs, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(inputs, results); },
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void transform::NamedSequenceOp::print(OpAsmPrinter &printer) {
  function_interface_impl::printFunctionOp(
      printer, cast<FunctionOpInterface>(getOperation()), /*isVariadic=*/false,
      getFunctionTypeAttrName().getValue(), getArgAttrsAttrName(),
      getResAttrsAttrName());
}

/// Verifies that a symbol function-like transform dialect operation has the
/// signature and the terminator that have conforming types, i.e., types
/// implementing the same transform dialect type interface. If `allowExternal`
/// is set, allow external symbols (declarations) and don't check the terminator
/// as it may not exist.
static DiagnosedSilenceableFailure
verifyYieldingSingleBlockOp(FunctionOpInterface op, bool allowExternal) {
  if (auto parent = op->getParentOfType<transform::TransformOpInterface>()) {
    DiagnosedSilenceableFailure diag =
        emitSilenceableFailure(op)
        << "cannot be defined inside another transform op";
    diag.attachNote(parent.getLoc()) << "ancestor transform op";
    return diag;
  }

  if (op.isExternal() || op.getFunctionBody().empty()) {
    if (allowExternal)
      return DiagnosedSilenceableFailure::success();

    return emitSilenceableFailure(op) << "cannot be external";
  }

  if (op.getFunctionBody().front().empty())
    return emitSilenceableFailure(op) << "expected a non-empty body block";

  Operation *terminator = &op.getFunctionBody().front().back();
  if (!isa<transform::YieldOp>(terminator)) {
    DiagnosedSilenceableFailure diag = emitSilenceableFailure(op)
                                       << "expected '"
                                       << transform::YieldOp::getOperationName()
                                       << "' as terminator";
    diag.attachNote(terminator->getLoc()) << "terminator";
    return diag;
  }

  if (terminator->getNumOperands() != op.getResultTypes().size()) {
    return emitSilenceableFailure(terminator)
           << "expected terminator to have as many operands as the parent op "
              "has results";
  }
  for (auto [i, operandType, resultType] : llvm::zip_equal(
           llvm::seq<unsigned>(0, terminator->getNumOperands()),
           terminator->getOperands().getType(), op.getResultTypes())) {
    if (operandType == resultType)
      continue;
    return emitSilenceableFailure(terminator)
           << "the type of the terminator operand #" << i
           << " must match the type of the corresponding parent op result ("
           << operandType << " vs " << resultType << ")";
  }

  return DiagnosedSilenceableFailure::success();
}

/// Verification of a NamedSequenceOp. This does not report the error
/// immediately, so it can be used to check for op's well-formedness before the
/// verifier runs, e.g., during trait verification.
static DiagnosedSilenceableFailure
verifyNamedSequenceOp(transform::NamedSequenceOp op) {
  if (Operation *parent = op->getParentWithTrait<OpTrait::SymbolTable>()) {
    if (!parent->getAttr(
            transform::TransformDialect::kWithNamedSequenceAttrName)) {
      DiagnosedSilenceableFailure diag =
          emitSilenceableFailure(op)
          << "expects the parent symbol table to have the '"
          << transform::TransformDialect::kWithNamedSequenceAttrName
          << "' attribute";
      diag.attachNote(parent->getLoc()) << "symbol table operation";
      return diag;
    }
  }

  if (auto parent = op->getParentOfType<transform::TransformOpInterface>()) {
    DiagnosedSilenceableFailure diag =
        emitSilenceableFailure(op)
        << "cannot be defined inside another transform op";
    diag.attachNote(parent.getLoc()) << "ancestor transform op";
    return diag;
  }

  if (op.isExternal() || op.getBody().empty())
    return verifyFunctionLikeConsumeAnnotations(cast<FunctionOpInterface>(*op));

  if (op.getBody().front().empty())
    return emitSilenceableFailure(op) << "expected a non-empty body block";

  Operation *terminator = &op.getBody().front().back();
  if (!isa<transform::YieldOp>(terminator)) {
    DiagnosedSilenceableFailure diag = emitSilenceableFailure(op)
                                       << "expected '"
                                       << transform::YieldOp::getOperationName()
                                       << "' as terminator";
    diag.attachNote(terminator->getLoc()) << "terminator";
    return diag;
  }

  if (terminator->getNumOperands() != op.getFunctionType().getNumResults()) {
    return emitSilenceableFailure(terminator)
           << "expected terminator to have as many operands as the parent op "
              "has results";
  }
  for (auto [i, operandType, resultType] :
       llvm::zip_equal(llvm::seq<unsigned>(0, terminator->getNumOperands()),
                       terminator->getOperands().getType(),
                       op.getFunctionType().getResults())) {
    if (operandType == resultType)
      continue;
    return emitSilenceableFailure(terminator)
           << "the type of the terminator operand #" << i
           << " must match the type of the corresponding parent op result ("
           << operandType << " vs " << resultType << ")";
  }

  auto funcOp = cast<FunctionOpInterface>(*op);
  DiagnosedSilenceableFailure diag =
      verifyFunctionLikeConsumeAnnotations(funcOp);
  if (!diag.succeeded())
    return diag;

  return verifyYieldingSingleBlockOp(funcOp,
                                     /*allowExternal=*/true);
}

LogicalResult transform::NamedSequenceOp::verify() {
  // Actual verification happens in a separate function for reusability.
  return verifyNamedSequenceOp(*this).checkAndReport();
}

//===----------------------------------------------------------------------===//
// SplitHandleOp
//===----------------------------------------------------------------------===//

void transform::SplitHandleOp::build(OpBuilder &builder, OperationState &result,
                                     Value target, int64_t numResultHandles) {
  result.addOperands(target);
  auto pdlOpType = pdl::OperationType::get(builder.getContext());
  result.addTypes(SmallVector<pdl::OperationType>(numResultHandles, pdlOpType));
}

DiagnosedSilenceableFailure
transform::SplitHandleOp::apply(transform::TransformResults &results,
                                transform::TransformState &state) {
  int64_t numPayloadOps = state.getPayloadOps(getHandle()).size();

  // Empty handle corner case: all result handles are empty.
  if (numPayloadOps == 0) {
    for (OpResult result : getResults())
      results.set(result, {});
    return DiagnosedSilenceableFailure::success();
  }

  // If the input handle was not empty and the number of payload ops does not
  // match, this is a legit silenceable error.
  if (numPayloadOps != getNumResults())
    return emitSilenceableError()
           << getHandle() << " expected to contain " << getNumResults()
           << " payload ops but it contains " << numPayloadOps
           << " payload ops";

  for (const auto &en : llvm::enumerate(state.getPayloadOps(getHandle())))
    results.set(getResults()[en.index()].cast<OpResult>(), en.value());

  return DiagnosedSilenceableFailure::success();
}

void transform::SplitHandleOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getHandle(), effects);
  producesHandle(getResults(), effects);
  // There are no effects on the Payload IR as this is only a handle
  // manipulation.
}

//===----------------------------------------------------------------------===//
// PDLMatchOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::PDLMatchOp::apply(transform::TransformResults &results,
                             transform::TransformState &state) {
  auto *extension = state.getExtension<PatternApplicatorExtension>();
  assert(extension &&
         "expected PatternApplicatorExtension to be attached by the parent op");
  SmallVector<Operation *> targets;
  for (Operation *root : state.getPayloadOps(getRoot())) {
    if (failed(extension->findAllMatches(
            getPatternName().getLeafReference().getValue(), root, targets))) {
      emitDefiniteFailure()
          << "could not find pattern '" << getPatternName() << "'";
    }
  }
  results.set(getResult().cast<OpResult>(), targets);
  return DiagnosedSilenceableFailure::success();
}

void transform::PDLMatchOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getRoot(), effects);
  producesHandle(getMatched(), effects);
  onlyReadsPayload(effects);
}

//===----------------------------------------------------------------------===//
// ReplicateOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::ReplicateOp::apply(transform::TransformResults &results,
                              transform::TransformState &state) {
  unsigned numRepetitions = state.getPayloadOps(getPattern()).size();
  for (const auto &en : llvm::enumerate(getHandles())) {
    Value handle = en.value();
    if (handle.getType().isa<TransformHandleTypeInterface>()) {
      ArrayRef<Operation *> current = state.getPayloadOps(handle);
      SmallVector<Operation *> payload;
      payload.reserve(numRepetitions * current.size());
      for (unsigned i = 0; i < numRepetitions; ++i)
        llvm::append_range(payload, current);
      results.set(getReplicated()[en.index()].cast<OpResult>(), payload);
    } else {
      assert(handle.getType().isa<TransformParamTypeInterface>() &&
             "expected param type");
      ArrayRef<Attribute> current = state.getParams(handle);
      SmallVector<Attribute> params;
      params.reserve(numRepetitions * current.size());
      for (unsigned i = 0; i < numRepetitions; ++i)
        llvm::append_range(params, current);
      results.setParams(getReplicated()[en.index()].cast<OpResult>(), params);
    }
  }
  return DiagnosedSilenceableFailure::success();
}

void transform::ReplicateOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getPattern(), effects);
  onlyReadsHandle(getHandles(), effects);
  producesHandle(getReplicated(), effects);
}

//===----------------------------------------------------------------------===//
// SequenceOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::SequenceOp::apply(transform::TransformResults &results,
                             transform::TransformState &state) {
  // Map the entry block argument to the list of operations.
  auto scope = state.make_region_scope(*getBodyBlock()->getParent());
  if (failed(mapBlockArguments(state)))
    return DiagnosedSilenceableFailure::definiteFailure();

  return applySequenceBlock(*getBodyBlock(), getFailurePropagationMode(), state,
                            results);
}

static ParseResult parseSequenceOpOperands(
    OpAsmParser &parser, std::optional<OpAsmParser::UnresolvedOperand> &root,
    Type &rootType,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &extraBindings,
    SmallVectorImpl<Type> &extraBindingTypes) {
  OpAsmParser::UnresolvedOperand rootOperand;
  OptionalParseResult hasRoot = parser.parseOptionalOperand(rootOperand);
  if (!hasRoot.has_value()) {
    root = std::nullopt;
    return success();
  }
  if (failed(hasRoot.value()))
    return failure();
  root = rootOperand;

  if (succeeded(parser.parseOptionalComma())) {
    if (failed(parser.parseOperandList(extraBindings)))
      return failure();
  }
  if (failed(parser.parseColon()))
    return failure();

  // The paren is truly optional.
  (void)parser.parseOptionalLParen();

  if (failed(parser.parseType(rootType))) {
    return failure();
  }

  if (!extraBindings.empty()) {
    if (parser.parseComma() || parser.parseTypeList(extraBindingTypes))
      return failure();
  }

  if (extraBindingTypes.size() != extraBindings.size()) {
    return parser.emitError(parser.getNameLoc(),
                            "expected types to be provided for all operands");
  }

  // The paren is truly optional.
  (void)parser.parseOptionalRParen();
  return success();
}

static void printSequenceOpOperands(OpAsmPrinter &printer, Operation *op,
                                    Value root, Type rootType,
                                    ValueRange extraBindings,
                                    TypeRange extraBindingTypes) {
  if (!root)
    return;

  printer << root;
  bool hasExtras = !extraBindings.empty();
  if (hasExtras) {
    printer << ", ";
    printer.printOperands(extraBindings);
  }

  printer << " : ";
  if (hasExtras)
    printer << "(";

  printer << rootType;
  if (hasExtras) {
    printer << ", ";
    llvm::interleaveComma(extraBindingTypes, printer.getStream());
    printer << ")";
  }
}

/// Returns `true` if the given op operand may be consuming the handle value in
/// the Transform IR. That is, if it may have a Free effect on it.
static bool isValueUsePotentialConsumer(OpOperand &use) {
  // Conservatively assume the effect being present in absence of the interface.
  auto iface = dyn_cast<transform::TransformOpInterface>(use.getOwner());
  if (!iface)
    return true;

  return isHandleConsumed(use.get(), iface);
}

LogicalResult
checkDoubleConsume(Value value,
                   function_ref<InFlightDiagnostic()> reportError) {
  OpOperand *potentialConsumer = nullptr;
  for (OpOperand &use : value.getUses()) {
    if (!isValueUsePotentialConsumer(use))
      continue;

    if (!potentialConsumer) {
      potentialConsumer = &use;
      continue;
    }

    InFlightDiagnostic diag = reportError()
                              << " has more than one potential consumer";
    diag.attachNote(potentialConsumer->getOwner()->getLoc())
        << "used here as operand #" << potentialConsumer->getOperandNumber();
    diag.attachNote(use.getOwner()->getLoc())
        << "used here as operand #" << use.getOperandNumber();
    return diag;
  }

  return success();
}

LogicalResult transform::SequenceOp::verify() {
  assert(getBodyBlock()->getNumArguments() >= 1 &&
         "the number of arguments must have been verified to be more than 1 by "
         "PossibleTopLevelTransformOpTrait");

  if (!getRoot() && !getExtraBindings().empty()) {
    return emitOpError()
           << "does not expect extra operands when used as top-level";
  }

  // Check if a block argument has more than one consuming use.
  for (BlockArgument arg : getBodyBlock()->getArguments()) {
    if (failed(checkDoubleConsume(arg, [this, arg]() {
          return (emitOpError() << "block argument #" << arg.getArgNumber());
        }))) {
      return failure();
    }
  }

  // Check properties of the nested operations they cannot check themselves.
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
      auto report = [&]() {
        return (child.emitError() << "result #" << result.getResultNumber());
      };
      if (failed(checkDoubleConsume(result, report)))
        return failure();
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

/// Populate `effects` with transform dialect memory effects for the potential
/// top-level operation. Such operations have recursive effects from nested
/// operations. When they have an operand, we can additionally remap effects on
/// the block argument to be effects on the operand.
template <typename OpTy>
static void getPotentialTopLevelEffects(
    OpTy operation, SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(operation->getOperands(), effects);
  transform::producesHandle(operation->getResults(), effects);

  if (!operation.getRoot()) {
    for (Operation &op : *operation.getBodyBlock()) {
      auto iface = dyn_cast<MemoryEffectOpInterface>(&op);
      if (!iface)
        continue;

      SmallVector<MemoryEffects::EffectInstance, 2> nestedEffects;
      iface.getEffects(effects);
    }
    return;
  }

  // Carry over all effects on arguments of the entry block as those on the
  // operands, this is the same value just remapped.
  remapArgumentEffects(*operation.getBodyBlock(), operation->getOperands(),
                       effects);
}

void transform::SequenceOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  getPotentialTopLevelEffects(*this, effects);
}

OperandRange transform::SequenceOp::getSuccessorEntryOperands(
    std::optional<unsigned> index) {
  assert(index && *index == 0 && "unexpected region index");
  if (getOperation()->getNumOperands() > 0)
    return getOperation()->getOperands();
  return OperandRange(getOperation()->operand_end(),
                      getOperation()->operand_end());
}

void transform::SequenceOp::getSuccessorRegions(
    std::optional<unsigned> index, ArrayRef<Attribute> operands,
    SmallVectorImpl<RegionSuccessor> &regions) {
  if (!index) {
    Region *bodyRegion = &getBody();
    regions.emplace_back(bodyRegion, !operands.empty()
                                         ? bodyRegion->getArguments()
                                         : Block::BlockArgListType());
    return;
  }

  assert(*index == 0 && "unexpected region index");
  regions.emplace_back(getOperation()->getResults());
}

void transform::SequenceOp::getRegionInvocationBounds(
    ArrayRef<Attribute> operands, SmallVectorImpl<InvocationBounds> &bounds) {
  (void)operands;
  bounds.emplace_back(1, 1);
}

template <typename FnTy>
static void buildSequenceBody(OpBuilder &builder, OperationState &state,
                              Type bbArgType, TypeRange extraBindingTypes,
                              FnTy bodyBuilder) {
  SmallVector<Type> types;
  types.reserve(1 + extraBindingTypes.size());
  types.push_back(bbArgType);
  llvm::append_range(types, extraBindingTypes);

  OpBuilder::InsertionGuard guard(builder);
  Region *region = state.regions.back().get();
  Block *bodyBlock =
      builder.createBlock(region, region->begin(), types,
                          SmallVector<Location>(types.size(), state.location));

  // Populate body.
  builder.setInsertionPointToStart(bodyBlock);
  if constexpr (llvm::function_traits<FnTy>::num_args == 3) {
    bodyBuilder(builder, state.location, bodyBlock->getArgument(0));
  } else {
    bodyBuilder(builder, state.location, bodyBlock->getArgument(0),
                bodyBlock->getArguments().drop_front());
  }
}

void transform::SequenceOp::build(OpBuilder &builder, OperationState &state,
                                  TypeRange resultTypes,
                                  FailurePropagationMode failurePropagationMode,
                                  Value root,
                                  SequenceBodyBuilderFn bodyBuilder) {
  build(builder, state, resultTypes, failurePropagationMode, root,
        /*extra_bindings=*/ValueRange());
  Type bbArgType = root.getType();
  buildSequenceBody(builder, state, bbArgType,
                    /*extraBindingTypes=*/TypeRange(), bodyBuilder);
}

void transform::SequenceOp::build(OpBuilder &builder, OperationState &state,
                                  TypeRange resultTypes,
                                  FailurePropagationMode failurePropagationMode,
                                  Value root, ValueRange extraBindings,
                                  SequenceBodyBuilderArgsFn bodyBuilder) {
  build(builder, state, resultTypes, failurePropagationMode, root,
        extraBindings);
  buildSequenceBody(builder, state, root.getType(), extraBindings.getTypes(),
                    bodyBuilder);
}

void transform::SequenceOp::build(OpBuilder &builder, OperationState &state,
                                  TypeRange resultTypes,
                                  FailurePropagationMode failurePropagationMode,
                                  Type bbArgType,
                                  SequenceBodyBuilderFn bodyBuilder) {
  build(builder, state, resultTypes, failurePropagationMode, /*root=*/Value(),
        /*extra_bindings=*/ValueRange());
  buildSequenceBody(builder, state, bbArgType,
                    /*extraBindingTypes=*/TypeRange(), bodyBuilder);
}

void transform::SequenceOp::build(OpBuilder &builder, OperationState &state,
                                  TypeRange resultTypes,
                                  FailurePropagationMode failurePropagationMode,
                                  Type bbArgType, TypeRange extraBindingTypes,
                                  SequenceBodyBuilderArgsFn bodyBuilder) {
  build(builder, state, resultTypes, failurePropagationMode, /*root=*/Value(),
        /*extra_bindings=*/ValueRange());
  buildSequenceBody(builder, state, bbArgType, extraBindingTypes, bodyBuilder);
}

//===----------------------------------------------------------------------===//
// WithPDLPatternsOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::WithPDLPatternsOp::apply(transform::TransformResults &results,
                                    transform::TransformState &state) {
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
    return DiagnosedSilenceableFailure::definiteFailure();
  return state.applyTransform(transformOp);
}

void transform::WithPDLPatternsOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  getPotentialTopLevelEffects(*this, effects);
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

  if (!topLevelOp) {
    InFlightDiagnostic diag = emitOpError()
                              << "expects at least one non-pattern op";
    return diag;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// PrintOp
//===----------------------------------------------------------------------===//

void transform::PrintOp::build(OpBuilder &builder, OperationState &result,
                               StringRef name) {
  if (!name.empty()) {
    result.addAttribute(PrintOp::getNameAttrName(result.name),
                        builder.getStrArrayAttr(name));
  }
}

void transform::PrintOp::build(OpBuilder &builder, OperationState &result,
                               Value target, StringRef name) {
  result.addOperands({target});
  build(builder, result, name);
}

DiagnosedSilenceableFailure
transform::PrintOp::apply(transform::TransformResults &results,
                          transform::TransformState &state) {
  llvm::outs() << "[[[ IR printer: ";
  if (getName().has_value())
    llvm::outs() << *getName() << " ";

  if (!getTarget()) {
    llvm::outs() << "top-level ]]]\n" << *state.getTopLevel() << "\n";
    return DiagnosedSilenceableFailure::success();
  }

  llvm::outs() << "]]]\n";
  ArrayRef<Operation *> targets = state.getPayloadOps(getTarget());
  for (Operation *target : targets)
    llvm::outs() << *target << "\n";

  return DiagnosedSilenceableFailure::success();
}

void transform::PrintOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTarget(), effects);
  onlyReadsPayload(effects);

  // There is no resource for stderr file descriptor, so just declare print
  // writes into the default resource.
  effects.emplace_back(MemoryEffects::Write::get());
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

void transform::YieldOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getOperands(), effects);
}
