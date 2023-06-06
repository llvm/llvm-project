//===- TransformDialect.cpp - Transform dialect operations ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/MatchInterfaces.h"
#include "mlir/Dialect/Transform/IR/TransformAttrs.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
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

MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::transform::PatternRegistry)

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
  SmallVector<Value> values(newValues.begin(), newValues.end());

  do {
    // If the replacement values belong to different ops, drop the mapping.
    Operation *defOp = getCommonDefiningOp(values);
    if (!defOp)
      return nullptr;

    // If the defining op has the same type, we take it as a replacement.
    if (op->getName() == defOp->getName())
      return defOp;

    // Replacing an op with a constant-like equivalent is a common
    // canonicalization.
    if (defOp->hasTrait<OpTrait::ConstantLike>())
      return defOp;

    values.clear();

    // Skip through ops that implement FindPayloadReplacementOpInterface.
    if (auto findReplacementOpInterface =
            dyn_cast<FindPayloadReplacementOpInterface>(defOp)) {
      values.assign(findReplacementOpInterface.getNextOperands());
      continue;
    }

    // Skip through ops that implement CastOpInterface.
    if (isa<CastOpInterface>(defOp)) {
      values.assign(defOp->getOperands().begin(), defOp->getOperands().end());
      continue;
    }
  } while (!values.empty());

  return nullptr;
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

void transform::TrackingListener::notifyOperationRemoved(Operation *op) {
  // TODO: Walk can be removed when D144193 has landed.
  op->walk([&](Operation *op) {
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

transform::ErrorCheckingTrackingListener::~ErrorCheckingTrackingListener() {
  // The state of the ErrorCheckingTrackingListener must be checked and reset
  // if there was an error. This is to prevent errors from accidentally being
  // missed.
  assert(status.succeeded() && "listener state was not checked");
}

DiagnosedSilenceableFailure
transform::ErrorCheckingTrackingListener::checkAndResetError() {
  DiagnosedSilenceableFailure s = std::move(status);
  status = DiagnosedSilenceableFailure::success();
  errorCounter = 0;
  return s;
}

bool transform::ErrorCheckingTrackingListener::failed() const {
  return !status.succeeded();
}

void transform::ErrorCheckingTrackingListener::notifyPayloadReplacementNotFound(
    Operation *op, ValueRange values) {
  if (status.succeeded()) {
    status = emitSilenceableFailure(
        getTransformOp(), "tracking listener failed to find replacement op");
  }

  status.attachNote(op->getLoc()) << "[" << errorCounter << "] replaced op";
  for (auto &&[index, value] : llvm::enumerate(values))
    status.attachNote(value.getLoc())
        << "[" << errorCounter << "] replacement value " << index;

  ++errorCounter;
}

//===----------------------------------------------------------------------===//
// PatternRegistry
//===----------------------------------------------------------------------===//

void transform::PatternRegistry::registerPatterns(StringRef identifier,
                                                  PopulatePatternsFn &&fn) {
  StringAttr attr = builder.getStringAttr(identifier);
  assert(!patterns.contains(attr) && "patterns identifier is already in use");
  patterns.try_emplace(attr, std::move(fn));
}

void transform::PatternRegistry::registerPatterns(
    StringRef identifier, PopulatePatternsWithBenefitFn &&fn) {
  StringAttr attr = builder.getStringAttr(identifier);
  assert(!patterns.contains(attr) && "patterns identifier is already in use");
  patterns.try_emplace(attr, [f = std::move(fn)](RewritePatternSet &patternSet) {
    f(patternSet, /*benefit=*/1);
  });
}

void transform::PatternRegistry::populatePatterns(
    StringAttr identifier, RewritePatternSet &patternSet) const {
  auto it = patterns.find(identifier);
  assert(it != patterns.end() && "patterns not registered in registry");
  it->second(patternSet);
}

bool transform::PatternRegistry::hasPatterns(StringAttr identifier) const {
  return patterns.contains(identifier);
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
// AnnotateOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::AnnotateOp::apply(transform::TransformResults &results,
                             transform::TransformState &state) {
  SmallVector<Operation *> targets =
      llvm::to_vector(state.getPayloadOps(getTarget()));

  Attribute attr = UnitAttr::get(getContext());
  if (auto paramH = getParam()) {
    ArrayRef<Attribute> params = state.getParams(paramH);
    if (params.size() != 1) {
      if (targets.size() != params.size()) {
        return emitSilenceableError()
               << "parameter and target have different payload lengths ("
               << params.size() << " vs " << targets.size() << ")";
      }
      for (auto &&[target, attr] : llvm::zip_equal(targets, params))
        target->setAttr(getName(), attr);
      return DiagnosedSilenceableFailure::success();
    }
    attr = params[0];
  }
  for (auto target : targets)
    target->setAttr(getName(), attr);
  return DiagnosedSilenceableFailure::success();
}

void transform::AnnotateOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTarget(), effects);
  onlyReadsHandle(getParam(), effects);
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// ApplyPatternsOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::ApplyPatternsOp::applyToOne(Operation *target,
                                       ApplyToEachResultList &results,
                                       transform::TransformState &state) {
  // Gather all specified patterns.
  MLIRContext *ctx = target->getContext();
  RewritePatternSet patterns(ctx);
  const auto &registry = getContext()
                             ->getLoadedDialect<transform::TransformDialect>()
                             ->getExtraData<transform::PatternRegistry>();
  for (Attribute attr : getPatterns())
    registry.populatePatterns(attr.cast<StringAttr>(), patterns);
  if (!getRegion().empty()) {
    for (Operation &op : getRegion().front()) {
      cast<transform::PatternDescriptorOpInterface>(&op).populatePatterns(
          patterns);
    }
  }

  // Configure the GreedyPatternRewriteDriver.
  ErrorCheckingTrackingListener listener(state, *this);
  GreedyRewriteConfig config;
  config.listener = &listener;

  // Manually gather list of ops because the other GreedyPatternRewriteDriver
  // overloads only accepts ops that are isolated from above. This way, patterns
  // can be applied to ops that are not isolated from above.
  SmallVector<Operation *> ops;
  target->walk([&](Operation *nestedOp) {
    if (target != nestedOp)
      ops.push_back(nestedOp);
  });
  LogicalResult result =
      applyOpPatternsAndFold(ops, std::move(patterns), config);
  // A failure typically indicates that the pattern application did not
  // converge.
  if (failed(result)) {
    return emitSilenceableFailure(target)
           << "greedy pattern application failed";
  }

  // Check listener state for tracking errors.
  if (listener.failed()) {
    DiagnosedSilenceableFailure status = listener.checkAndResetError();
    if (getFailOnPayloadReplacementNotFound())
      return status;
    (void)status.silence();
  }

  return DiagnosedSilenceableFailure::success();
}

LogicalResult transform::ApplyPatternsOp::verify() {
  const auto &registry = getContext()
                             ->getLoadedDialect<transform::TransformDialect>()
                             ->getExtraData<transform::PatternRegistry>();
  for (Attribute attr : getPatterns()) {
    auto strAttr = attr.dyn_cast<StringAttr>();
    if (!strAttr)
      return emitOpError() << "expected " << getPatternsAttrName()
                           << " to be an array of strings";
    if (!registry.hasPatterns(strAttr))
      return emitOpError() << "patterns not registered: " << strAttr.strref();
  }
  if (!getRegion().empty()) {
    for (Operation &op : getRegion().front()) {
      if (!isa<transform::PatternDescriptorOpInterface>(&op)) {
        InFlightDiagnostic diag = emitOpError()
                                  << "expected children ops to implement "
                                     "PatternDescriptorOpInterface";
        diag.attachNote(op.getLoc()) << "op without interface";
        return diag;
      }
    }
  }
  return success();
}

void transform::ApplyPatternsOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTarget(), effects);
  transform::modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// ApplyCanonicalizationPatternsOp
//===----------------------------------------------------------------------===//

void transform::ApplyCanonicalizationPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  for (Dialect *dialect : ctx->getLoadedDialects())
    dialect->getCanonicalizationPatterns(patterns);
  for (RegisteredOperationName op : ctx->getRegisteredOperations())
    op.getCanonicalizationPatterns(patterns, ctx);
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
      [](Type ty) { return isa<transform::TransformHandleTypeInterface>(ty); });
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
  results.set(llvm::cast<OpResult>(getUpdated()),
              state.getPayloadOps(getRoot()));
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
  SmallVector<SmallVector<Operation *>> resultOps(getNumResults(), {});

  for (Operation *op : state.getPayloadOps(getTarget())) {
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
      auto yieldedOps = state.getPayloadOps(getYieldOp().getOperand(i));
      resultOps[i].append(yieldedOps.begin(), yieldedOps.end());
    }
  }

  for (unsigned i = 0; i < getNumResults(); ++i)
    results.set(llvm::cast<OpResult>(getResult(i)), resultOps[i]);

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
    if (!llvm::isa<TransformHandleTypeInterface>(v.getType()))
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
  results.set(llvm::cast<OpResult>(getResult()), parents.getArrayRef());
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// GetConsumersOfResult
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::GetConsumersOfResult::apply(transform::TransformResults &results,
                                       transform::TransformState &state) {
  int64_t resultNumber = getResultNumber();
  auto payloadOps = state.getPayloadOps(getTarget());
  if (std::empty(payloadOps)) {
    results.set(cast<OpResult>(getResult()), {});
    return DiagnosedSilenceableFailure::success();
  }
  if (!llvm::hasSingleElement(payloadOps))
    return emitDefiniteFailure()
           << "handle must be mapped to exactly one payload op";

  Operation *target = *payloadOps.begin();
  if (target->getNumResults() <= resultNumber)
    return emitDefiniteFailure() << "result number overflow";
  results.set(llvm::cast<OpResult>(getResult()),
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
    if (llvm::isa<BlockArgument>(v)) {
      DiagnosedSilenceableFailure diag =
          emitSilenceableError() << "cannot get defining op of block argument";
      diag.attachNote(v.getLoc()) << "target value";
      return diag;
    }
    definingOps.push_back(v.getDefiningOp());
  }
  results.set(llvm::cast<OpResult>(getResult()), definingOps);
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
  results.set(llvm::cast<OpResult>(getResult()), producers);
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
  results.setValues(llvm::cast<OpResult>(getResult()), opResults);
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
    auto intAttr = llvm::dyn_cast<IntegerAttr>(param);
    auto refAttr = llvm::dyn_cast<IntegerAttr>(reference);
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
    results.set(llvm::cast<OpResult>(getResult()), operations);
    return DiagnosedSilenceableFailure::success();
  }

  SetVector<Operation *> uniqued(operations.begin(), operations.end());
  results.set(llvm::cast<OpResult>(getResult()), uniqued.getArrayRef());
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
  result.addTypes(SmallVector<Type>(numResultHandles, target.getType()));
}

DiagnosedSilenceableFailure
transform::SplitHandleOp::apply(transform::TransformResults &results,
                                transform::TransformState &state) {
  int64_t numPayloadOps = llvm::range_size(state.getPayloadOps(getHandle()));
  auto produceNumOpsError = [&]() {
    return emitSilenceableError()
           << getHandle() << " expected to contain " << this->getNumResults()
           << " payload ops but it contains " << numPayloadOps
           << " payload ops";
  };

  // Fail if there are more payload ops than results and no overflow result was
  // specified.
  if (numPayloadOps > getNumResults() && !getOverflowResult().has_value())
    return produceNumOpsError();

  // Fail if there are more results than payload ops. Unless:
  // - "fail_on_payload_too_small" is set to "false", or
  // - "pass_through_empty_handle" is set to "true" and there are 0 payload ops.
  if (numPayloadOps < getNumResults() && getFailOnPayloadTooSmall() &&
      !(numPayloadOps == 0 && getPassThroughEmptyHandle()))
    return produceNumOpsError();

  // Distribute payload ops.
  SmallVector<SmallVector<Operation *, 1>> resultHandles(getNumResults(), {});
  if (getOverflowResult())
    resultHandles[*getOverflowResult()].reserve(numPayloadOps -
                                                getNumResults());
  for (auto &&en : llvm::enumerate(state.getPayloadOps(getHandle()))) {
    int64_t resultNum = en.index();
    if (resultNum >= getNumResults())
      resultNum = *getOverflowResult();
    resultHandles[resultNum].push_back(en.value());
  }

  // Set transform op results.
  for (auto &&it : llvm::enumerate(resultHandles))
    results.set(llvm::cast<OpResult>(getResult(it.index())), it.value());

  return DiagnosedSilenceableFailure::success();
}

void transform::SplitHandleOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getHandle(), effects);
  producesHandle(getResults(), effects);
  // There are no effects on the Payload IR as this is only a handle
  // manipulation.
}

LogicalResult transform::SplitHandleOp::verify() {
  if (getOverflowResult().has_value() &&
      !(*getOverflowResult() >= 0 && *getOverflowResult() < getNumResults()))
    return emitOpError("overflow_result is not a valid result index");
  return success();
}

//===----------------------------------------------------------------------===//
// ReplicateOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::ReplicateOp::apply(transform::TransformResults &results,
                              transform::TransformState &state) {
  unsigned numRepetitions = llvm::range_size(state.getPayloadOps(getPattern()));
  for (const auto &en : llvm::enumerate(getHandles())) {
    Value handle = en.value();
    if (isa<TransformHandleTypeInterface>(handle.getType())) {
      SmallVector<Operation *> current =
          llvm::to_vector(state.getPayloadOps(handle));
      SmallVector<Operation *> payload;
      payload.reserve(numRepetitions * current.size());
      for (unsigned i = 0; i < numRepetitions; ++i)
        llvm::append_range(payload, current);
      results.set(llvm::cast<OpResult>(getReplicated()[en.index()]), payload);
    } else {
      assert(llvm::isa<TransformParamTypeInterface>(handle.getType()) &&
             "expected param type");
      ArrayRef<Attribute> current = state.getParams(handle);
      SmallVector<Attribute> params;
      params.reserve(numRepetitions * current.size());
      for (unsigned i = 0; i < numRepetitions; ++i)
        llvm::append_range(params, current);
      results.setParams(llvm::cast<OpResult>(getReplicated()[en.index()]),
                        params);
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

void transform::SequenceOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  getPotentialTopLevelEffects(effects);
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
  for (Operation *target : state.getPayloadOps(getTarget()))
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
