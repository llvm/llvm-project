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
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "transform-dialect"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "] ")

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
// AlternativesOp
//===----------------------------------------------------------------------===//

OperandRange
transform::AlternativesOp::getSuccessorEntryOperands(Optional<unsigned> index) {
  if (index && getOperation()->getNumOperands() == 1)
    return getOperation()->getOperands();
  return OperandRange(getOperation()->operand_end(),
                      getOperation()->operand_end());
}

void transform::AlternativesOp::getSuccessorRegions(
    Optional<unsigned> index, ArrayRef<Attribute> operands,
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

static void forwardTerminatorOperands(Block *block,
                                      transform::TransformState &state,
                                      transform::TransformResults &results) {
  for (const auto &pair : llvm::zip(block->getTerminator()->getOperands(),
                                    block->getParentOp()->getOpResults())) {
    Value terminatorOperand = std::get<0>(pair);
    OpResult result = std::get<1>(pair);
    results.set(result, state.getPayloadOps(terminatorOperand));
  }
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
      InFlightDiagnostic diag =
          emitError() << "scope must not contain the transforms being applied";
      diag.attachNote(original->getLoc()) << "scope";
      return DiagnosedSilenceableFailure::definiteFailure();
    }
    if (!original->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
      InFlightDiagnostic diag =
          emitError()
          << "only isolated-from-above ops can be alternative scopes";
      diag.attachNote(original->getLoc()) << "scope";
      return DiagnosedSilenceableFailure(std::move(diag));
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
      IRRewriter rewriter(getContext());
      for (const auto &kvp : llvm::zip(originals, clones)) {
        Operation *original = std::get<0>(kvp);
        Operation *clone = std::get<1>(kvp);
        original->getBlock()->getOperations().insert(original->getIterator(),
                                                     clone);
        rewriter.replaceOp(original, clone->getResults());
      }
      forwardTerminatorOperands(&reg.front(), state, results);
      return DiagnosedSilenceableFailure::success();
    }
  }
  return emitSilenceableError() << "all alternatives failed";
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
// ForeachOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::CastOp::applyToOne(Operation *target,
                              SmallVectorImpl<Operation *> &results,
                              transform::TransformState &state) {
  results.push_back(target);
  return DiagnosedSilenceableFailure::success();
}

void transform::CastOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsPayload(effects);
  consumesHandle(getInput(), effects);
  producesHandle(getOutput(), effects);
}

bool transform::CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  assert(inputs.size() == 1 && "expected one input");
  assert(outputs.size() == 1 && "expected one output");
  return llvm::all_of(
      std::initializer_list<Type>{inputs.front(), outputs.front()},
      [](Type ty) {
        return ty.isa<pdl::OperationType, transform::TransformTypeInterface>();
      });
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
    Optional<unsigned> index, ArrayRef<Attribute> operands,
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
transform::ForeachOp::getSuccessorEntryOperands(Optional<unsigned> index) {
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
    if (!v.getType().isa<TransformTypeInterface>())
      return yieldOp->emitOpError(
          "expects operands to have types implementing TransformTypeInterface");
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
      results.set(getResult().cast<OpResult>(),
                  SmallVector<mlir::Operation *>{});
      return diag;
    }
    producers.push_back(producer);
  }
  results.set(getResult().cast<OpResult>(), producers);
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

void transform::MergeHandlesOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getHandles(), effects);
  producesHandle(getResult(), effects);

  // There are no effects on the Payload IR as this is only a handle
  // manipulation.
}

OpFoldResult transform::MergeHandlesOp::fold(ArrayRef<Attribute> operands) {
  if (getDeduplicate() || getHandles().size() != 1)
    return {};

  // If deduplication is not required and there is only one operand, it can be
  // used directly instead of merging.
  return getHandles().front();
}

//===----------------------------------------------------------------------===//
// SplitHandlesOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::SplitHandlesOp::apply(transform::TransformResults &results,
                                 transform::TransformState &state) {
  int64_t numResultHandles =
      getHandle() ? state.getPayloadOps(getHandle()).size() : 0;
  int64_t expectedNumResultHandles = getNumResultHandles();
  if (numResultHandles != expectedNumResultHandles) {
    // Failing case needs to propagate gracefully for both suppress and
    // propagate modes.
    for (int64_t idx = 0; idx < expectedNumResultHandles; ++idx)
      results.set(getResults()[idx].cast<OpResult>(), {});
    // Empty input handle corner case: always propagates empty handles in both
    // suppress and propagate modes.
    if (numResultHandles == 0)
      return DiagnosedSilenceableFailure::success();
    // If the input handle was not empty and the number of result handles does
    // not match, this is a legit silenceable error.
    return emitSilenceableError()
           << getHandle() << " expected to contain " << expectedNumResultHandles
           << " operation handles but it only contains " << numResultHandles
           << " handles";
  }
  // Normal successful case.
  for (auto en : llvm::enumerate(state.getPayloadOps(getHandle())))
    results.set(getResults()[en.index()].cast<OpResult>(), en.value());
  return DiagnosedSilenceableFailure::success();
}

void transform::SplitHandlesOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getHandle(), effects);
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
      emitOpError() << "could not find pattern '" << getPatternName() << "'";
      return DiagnosedSilenceableFailure::definiteFailure();
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
    ArrayRef<Operation *> current = state.getPayloadOps(handle);
    SmallVector<Operation *> payload;
    payload.reserve(numRepetitions * current.size());
    for (unsigned i = 0; i < numRepetitions; ++i)
      llvm::append_range(payload, current);
    results.set(getReplicated()[en.index()].cast<OpResult>(), payload);
  }
  return DiagnosedSilenceableFailure::success();
}

void transform::ReplicateOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getPattern(), effects);
  consumesHandle(getHandles(), effects);
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

  // Apply the sequenced ops one by one.
  for (Operation &transform : getBodyBlock()->without_terminator()) {
    DiagnosedSilenceableFailure result =
        state.applyTransform(cast<TransformOpInterface>(transform));
    if (result.isDefiniteFailure())
      return result;

    if (result.isSilenceableFailure()) {
      if (getFailurePropagationMode() == FailurePropagationMode::Propagate)
        return result;
      (void)result.silence();
    }
  }

  // Forward the operation mapping for values yielded from the sequence to the
  // values produced by the sequence op.
  forwardTerminatorOperands(getBodyBlock(), state, results);
  return DiagnosedSilenceableFailure::success();
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
  // Check if the block argument has more than one consuming use.
  for (BlockArgument argument : getBodyBlock()->getArguments()) {
    auto report = [&]() {
      return (emitOpError() << "block argument #" << argument.getArgNumber());
    };
    if (failed(checkDoubleConsume(argument, report)))
      return failure();
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
  auto *mappingResource = TransformMappingResource::get();
  effects.emplace_back(MemoryEffects::Read::get(), getRoot(), mappingResource);

  for (Value result : getResults()) {
    effects.emplace_back(MemoryEffects::Allocate::get(), result,
                         mappingResource);
    effects.emplace_back(MemoryEffects::Write::get(), result, mappingResource);
  }

  if (!getRoot()) {
    for (Operation &op : *getBodyBlock()) {
      auto iface = dyn_cast<MemoryEffectOpInterface>(&op);
      if (!iface) {
        // TODO: fill all possible effects; or require ops to actually implement
        // the memory effect interface always
        assert(false);
      }

      SmallVector<MemoryEffects::EffectInstance, 2> nestedEffects;
      iface.getEffects(effects);
    }
    return;
  }

  // Carry over all effects on the argument of the entry block as those on the
  // operand, this is the same value just remapped.
  for (Operation &op : *getBodyBlock()) {
    auto iface = dyn_cast<MemoryEffectOpInterface>(&op);
    if (!iface) {
      // TODO: fill all possible effects; or require ops to actually implement
      // the memory effect interface always
      assert(false);
    }

    SmallVector<MemoryEffects::EffectInstance, 2> nestedEffects;
    iface.getEffectsOnValue(getBodyBlock()->getArgument(0), nestedEffects);
    for (const auto &effect : nestedEffects)
      effects.emplace_back(effect.getEffect(), getRoot(), effect.getResource());
  }
}

OperandRange
transform::SequenceOp::getSuccessorEntryOperands(Optional<unsigned> index) {
  assert(index && *index == 0 && "unexpected region index");
  if (getOperation()->getNumOperands() == 1)
    return getOperation()->getOperands();
  return OperandRange(getOperation()->operand_end(),
                      getOperation()->operand_end());
}

void transform::SequenceOp::getSuccessorRegions(
    Optional<unsigned> index, ArrayRef<Attribute> operands,
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

//===----------------------------------------------------------------------===//
// WithPDLPatternsOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
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
    return DiagnosedSilenceableFailure::definiteFailure();
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
