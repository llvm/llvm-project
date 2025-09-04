//===- TransformOps.cpp - Transform dialect operations --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Transform/IR/TransformOps.h"

#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Transform/IR/TransformAttrs.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/Interfaces/MatchInterfaces.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InterleavedRange.h"
#include <optional>

#define DEBUG_TYPE "transform-dialect"
#define DEBUG_TYPE_MATCHER "transform-matcher"

using namespace mlir;

static ParseResult parseApplyRegisteredPassOptions(
    OpAsmParser &parser, DictionaryAttr &options,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &dynamicOptions);
static void printApplyRegisteredPassOptions(OpAsmPrinter &printer,
                                            Operation *op,
                                            DictionaryAttr options,
                                            ValueRange dynamicOptions);
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

/// Helper function to check if the given transform op is contained in (or
/// equal to) the given payload target op. In that case, an error is returned.
/// Transforming transform IR that is currently executing is generally unsafe.
static DiagnosedSilenceableFailure
ensurePayloadIsSeparateFromTransform(transform::TransformOpInterface transform,
                                     Operation *payload) {
  Operation *transformAncestor = transform.getOperation();
  while (transformAncestor) {
    if (transformAncestor == payload) {
      DiagnosedDefiniteFailure diag =
          transform.emitDefiniteFailure()
          << "cannot apply transform to itself (or one of its ancestors)";
      diag.attachNote(payload->getLoc()) << "target payload op";
      return diag;
    }
    transformAncestor = transformAncestor->getParentOp();
  }
  return DiagnosedSilenceableFailure::success();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/Transform/IR/TransformOps.cpp.inc"

//===----------------------------------------------------------------------===//
// AlternativesOp
//===----------------------------------------------------------------------===//

OperandRange
transform::AlternativesOp::getEntrySuccessorOperands(RegionBranchPoint point) {
  if (!point.isParent() && getOperation()->getNumOperands() == 1)
    return getOperation()->getOperands();
  return OperandRange(getOperation()->operand_end(),
                      getOperation()->operand_end());
}

void transform::AlternativesOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  for (Region &alternative : llvm::drop_begin(
           getAlternatives(),
           point.isParent() ? 0
                            : point.getRegionOrNull()->getRegionNumber() + 1)) {
    regions.emplace_back(&alternative, !getOperands().empty()
                                           ? alternative.getArguments()
                                           : Block::BlockArgListType());
  }
  if (!point.isParent())
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
transform::AlternativesOp::apply(transform::TransformRewriter &rewriter,
                                 transform::TransformResults &results,
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
        LDBG() << "alternative failed: " << result.getMessage();
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
  consumesHandle(getOperation()->getOpOperands(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
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
transform::AnnotateOp::apply(transform::TransformRewriter &rewriter,
                             transform::TransformResults &results,
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
  for (auto *target : targets)
    target->setAttr(getName(), attr);
  return DiagnosedSilenceableFailure::success();
}

void transform::AnnotateOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);
  onlyReadsHandle(getParamMutable(), effects);
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// ApplyCommonSubexpressionEliminationOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::ApplyCommonSubexpressionEliminationOp::applyToOne(
    transform::TransformRewriter &rewriter, Operation *target,
    ApplyToEachResultList &results, transform::TransformState &state) {
  // Make sure that this transform is not applied to itself. Modifying the
  // transform IR while it is being interpreted is generally dangerous.
  DiagnosedSilenceableFailure payloadCheck =
      ensurePayloadIsSeparateFromTransform(*this, target);
  if (!payloadCheck.succeeded())
    return payloadCheck;

  DominanceInfo domInfo;
  mlir::eliminateCommonSubExpressions(rewriter, domInfo, target);
  return DiagnosedSilenceableFailure::success();
}

void transform::ApplyCommonSubexpressionEliminationOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTargetMutable(), effects);
  transform::modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// ApplyDeadCodeEliminationOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::ApplyDeadCodeEliminationOp::applyToOne(
    transform::TransformRewriter &rewriter, Operation *target,
    ApplyToEachResultList &results, transform::TransformState &state) {
  // Make sure that this transform is not applied to itself. Modifying the
  // transform IR while it is being interpreted is generally dangerous.
  DiagnosedSilenceableFailure payloadCheck =
      ensurePayloadIsSeparateFromTransform(*this, target);
  if (!payloadCheck.succeeded())
    return payloadCheck;

  // Maintain a worklist of potentially dead ops.
  SetVector<Operation *> worklist;

  // Helper function that adds all defining ops of used values (operands and
  // operands of nested ops).
  auto addDefiningOpsToWorklist = [&](Operation *op) {
    op->walk([&](Operation *op) {
      for (Value v : op->getOperands())
        if (Operation *defOp = v.getDefiningOp())
          if (target->isProperAncestor(defOp))
            worklist.insert(defOp);
    });
  };

  // Helper function that erases an op.
  auto eraseOp = [&](Operation *op) {
    // Remove op and nested ops from the worklist.
    op->walk([&](Operation *op) {
      const auto *it = llvm::find(worklist, op);
      if (it != worklist.end())
        worklist.erase(it);
    });
    rewriter.eraseOp(op);
  };

  // Initial walk over the IR.
  target->walk<WalkOrder::PostOrder>([&](Operation *op) {
    if (op != target && isOpTriviallyDead(op)) {
      addDefiningOpsToWorklist(op);
      eraseOp(op);
    }
  });

  // Erase all ops that have become dead.
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (!isOpTriviallyDead(op))
      continue;
    addDefiningOpsToWorklist(op);
    eraseOp(op);
  }

  return DiagnosedSilenceableFailure::success();
}

void transform::ApplyDeadCodeEliminationOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTargetMutable(), effects);
  transform::modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// ApplyPatternsOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::ApplyPatternsOp::applyToOne(
    transform::TransformRewriter &rewriter, Operation *target,
    ApplyToEachResultList &results, transform::TransformState &state) {
  // Make sure that this transform is not applied to itself. Modifying the
  // transform IR while it is being interpreted is generally dangerous. Even
  // more so for the ApplyPatternsOp because the GreedyPatternRewriteDriver
  // performs many additional simplifications such as dead code elimination.
  DiagnosedSilenceableFailure payloadCheck =
      ensurePayloadIsSeparateFromTransform(*this, target);
  if (!payloadCheck.succeeded())
    return payloadCheck;

  // Gather all specified patterns.
  MLIRContext *ctx = target->getContext();
  RewritePatternSet patterns(ctx);
  if (!getRegion().empty()) {
    for (Operation &op : getRegion().front()) {
      cast<transform::PatternDescriptorOpInterface>(&op)
          .populatePatternsWithState(patterns, state);
    }
  }

  // Configure the GreedyPatternRewriteDriver.
  GreedyRewriteConfig config;
  config.setListener(
      static_cast<RewriterBase::Listener *>(rewriter.getListener()));
  FrozenRewritePatternSet frozenPatterns(std::move(patterns));

  config.setMaxIterations(getMaxIterations() == static_cast<uint64_t>(-1)
                              ? GreedyRewriteConfig::kNoLimit
                              : getMaxIterations());
  config.setMaxNumRewrites(getMaxNumRewrites() == static_cast<uint64_t>(-1)
                               ? GreedyRewriteConfig::kNoLimit
                               : getMaxNumRewrites());

  // Apply patterns and CSE repetitively until a fixpoint is reached. If no CSE
  // was requested, apply the greedy pattern rewrite only once. (The greedy
  // pattern rewrite driver already iterates to a fixpoint internally.)
  bool cseChanged = false;
  // One or two iterations should be sufficient. Stop iterating after a certain
  // threshold to make debugging easier.
  static const int64_t kNumMaxIterations = 50;
  int64_t iteration = 0;
  do {
    LogicalResult result = failure();
    if (target->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
      // Op is isolated from above. Apply patterns and also perform region
      // simplification.
      result = applyPatternsGreedily(target, frozenPatterns, config);
    } else {
      // Manually gather list of ops because the other
      // GreedyPatternRewriteDriver overloads only accepts ops that are isolated
      // from above. This way, patterns can be applied to ops that are not
      // isolated from above. Regions are not being simplified. Furthermore,
      // only a single greedy rewrite iteration is performed.
      SmallVector<Operation *> ops;
      target->walk([&](Operation *nestedOp) {
        if (target != nestedOp)
          ops.push_back(nestedOp);
      });
      result = applyOpPatternsGreedily(ops, frozenPatterns, config);
    }

    // A failure typically indicates that the pattern application did not
    // converge.
    if (failed(result)) {
      return emitSilenceableFailure(target)
             << "greedy pattern application failed";
    }

    if (getApplyCse()) {
      DominanceInfo domInfo;
      mlir::eliminateCommonSubExpressions(rewriter, domInfo, target,
                                          &cseChanged);
    }
  } while (cseChanged && ++iteration < kNumMaxIterations);

  if (iteration == kNumMaxIterations)
    return emitDefiniteFailure() << "fixpoint iteration did not converge";

  return DiagnosedSilenceableFailure::success();
}

LogicalResult transform::ApplyPatternsOp::verify() {
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
  transform::onlyReadsHandle(getTargetMutable(), effects);
  transform::modifiesPayload(effects);
}

void transform::ApplyPatternsOp::build(
    OpBuilder &builder, OperationState &result, Value target,
    function_ref<void(OpBuilder &, Location)> bodyBuilder) {
  result.addOperands(target);

  OpBuilder::InsertionGuard g(builder);
  Region *region = result.addRegion();
  builder.createBlock(region);
  if (bodyBuilder)
    bodyBuilder(builder, result.location);
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
// ApplyConversionPatternsOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::ApplyConversionPatternsOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state) {
  MLIRContext *ctx = getContext();

  // Instantiate the default type converter if a type converter builder is
  // specified.
  std::unique_ptr<TypeConverter> defaultTypeConverter;
  transform::TypeConverterBuilderOpInterface typeConverterBuilder =
      getDefaultTypeConverter();
  if (typeConverterBuilder)
    defaultTypeConverter = typeConverterBuilder.getTypeConverter();

  // Configure conversion target.
  ConversionTarget conversionTarget(*getContext());
  if (getLegalOps())
    for (Attribute attr : cast<ArrayAttr>(*getLegalOps()))
      conversionTarget.addLegalOp(
          OperationName(cast<StringAttr>(attr).getValue(), ctx));
  if (getIllegalOps())
    for (Attribute attr : cast<ArrayAttr>(*getIllegalOps()))
      conversionTarget.addIllegalOp(
          OperationName(cast<StringAttr>(attr).getValue(), ctx));
  if (getLegalDialects())
    for (Attribute attr : cast<ArrayAttr>(*getLegalDialects()))
      conversionTarget.addLegalDialect(cast<StringAttr>(attr).getValue());
  if (getIllegalDialects())
    for (Attribute attr : cast<ArrayAttr>(*getIllegalDialects()))
      conversionTarget.addIllegalDialect(cast<StringAttr>(attr).getValue());

  // Gather all specified patterns.
  RewritePatternSet patterns(ctx);
  // Need to keep the converters alive until after pattern application because
  // the patterns take a reference to an object that would otherwise get out of
  // scope.
  SmallVector<std::unique_ptr<TypeConverter>> keepAliveConverters;
  if (!getPatterns().empty()) {
    for (Operation &op : getPatterns().front()) {
      auto descriptor =
          cast<transform::ConversionPatternDescriptorOpInterface>(&op);

      // Check if this pattern set specifies a type converter.
      std::unique_ptr<TypeConverter> typeConverter =
          descriptor.getTypeConverter();
      TypeConverter *converter = nullptr;
      if (typeConverter) {
        keepAliveConverters.emplace_back(std::move(typeConverter));
        converter = keepAliveConverters.back().get();
      } else {
        // No type converter specified: Use the default type converter.
        if (!defaultTypeConverter) {
          auto diag = emitDefiniteFailure()
                      << "pattern descriptor does not specify type "
                         "converter and apply_conversion_patterns op has "
                         "no default type converter";
          diag.attachNote(op.getLoc()) << "pattern descriptor op";
          return diag;
        }
        converter = defaultTypeConverter.get();
      }

      // Add descriptor-specific updates to the conversion target, which may
      // depend on the final type converter. In structural converters, the
      // legality of types dictates the dynamic legality of an operation.
      descriptor.populateConversionTargetRules(*converter, conversionTarget);

      descriptor.populatePatterns(*converter, patterns);
    }
  }

  // Attach a tracking listener if handles should be preserved. We configure the
  // listener to allow op replacements with different names, as conversion
  // patterns typically replace ops with replacement ops that have a different
  // name.
  TrackingListenerConfig trackingConfig;
  trackingConfig.requireMatchingReplacementOpName = false;
  ErrorCheckingTrackingListener trackingListener(state, *this, trackingConfig);
  ConversionConfig conversionConfig;
  if (getPreserveHandles())
    conversionConfig.listener = &trackingListener;

  FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  for (Operation *target : state.getPayloadOps(getTarget())) {
    // Make sure that this transform is not applied to itself. Modifying the
    // transform IR while it is being interpreted is generally dangerous.
    DiagnosedSilenceableFailure payloadCheck =
        ensurePayloadIsSeparateFromTransform(*this, target);
    if (!payloadCheck.succeeded())
      return payloadCheck;

    LogicalResult status = failure();
    if (getPartialConversion()) {
      status = applyPartialConversion(target, conversionTarget, frozenPatterns,
                                      conversionConfig);
    } else {
      status = applyFullConversion(target, conversionTarget, frozenPatterns,
                                   conversionConfig);
    }

    // Check dialect conversion state.
    DiagnosedSilenceableFailure diag = DiagnosedSilenceableFailure::success();
    if (failed(status)) {
      diag = emitSilenceableError() << "dialect conversion failed";
      diag.attachNote(target->getLoc()) << "target op";
    }

    // Check tracking listener error state.
    DiagnosedSilenceableFailure trackingFailure =
        trackingListener.checkAndResetError();
    if (!trackingFailure.succeeded()) {
      if (diag.succeeded()) {
        // Tracking failure is the only failure.
        return trackingFailure;
      } else {
        diag.attachNote() << "tracking listener also failed: "
                          << trackingFailure.getMessage();
        (void)trackingFailure.silence();
      }
    }

    if (!diag.succeeded())
      return diag;
  }

  return DiagnosedSilenceableFailure::success();
}

LogicalResult transform::ApplyConversionPatternsOp::verify() {
  if (getNumRegions() != 1 && getNumRegions() != 2)
    return emitOpError() << "expected 1 or 2 regions";
  if (!getPatterns().empty()) {
    for (Operation &op : getPatterns().front()) {
      if (!isa<transform::ConversionPatternDescriptorOpInterface>(&op)) {
        InFlightDiagnostic diag =
            emitOpError() << "expected pattern children ops to implement "
                             "ConversionPatternDescriptorOpInterface";
        diag.attachNote(op.getLoc()) << "op without interface";
        return diag;
      }
    }
  }
  if (getNumRegions() == 2) {
    Region &typeConverterRegion = getRegion(1);
    if (!llvm::hasSingleElement(typeConverterRegion.front()))
      return emitOpError()
             << "expected exactly one op in default type converter region";
    Operation *maybeTypeConverter = &typeConverterRegion.front().front();
    auto typeConverterOp = dyn_cast<transform::TypeConverterBuilderOpInterface>(
        maybeTypeConverter);
    if (!typeConverterOp) {
      InFlightDiagnostic diag = emitOpError()
                                << "expected default converter child op to "
                                   "implement TypeConverterBuilderOpInterface";
      diag.attachNote(maybeTypeConverter->getLoc()) << "op without interface";
      return diag;
    }
    // Check default type converter type.
    if (!getPatterns().empty()) {
      for (Operation &op : getPatterns().front()) {
        auto descriptor =
            cast<transform::ConversionPatternDescriptorOpInterface>(&op);
        if (failed(descriptor.verifyTypeConverter(typeConverterOp)))
          return failure();
      }
    }
  }
  return success();
}

void transform::ApplyConversionPatternsOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  if (!getPreserveHandles()) {
    transform::consumesHandle(getTargetMutable(), effects);
  } else {
    transform::onlyReadsHandle(getTargetMutable(), effects);
  }
  transform::modifiesPayload(effects);
}

void transform::ApplyConversionPatternsOp::build(
    OpBuilder &builder, OperationState &result, Value target,
    function_ref<void(OpBuilder &, Location)> patternsBodyBuilder,
    function_ref<void(OpBuilder &, Location)> typeConverterBodyBuilder) {
  result.addOperands(target);

  {
    OpBuilder::InsertionGuard g(builder);
    Region *region1 = result.addRegion();
    builder.createBlock(region1);
    if (patternsBodyBuilder)
      patternsBodyBuilder(builder, result.location);
  }
  {
    OpBuilder::InsertionGuard g(builder);
    Region *region2 = result.addRegion();
    builder.createBlock(region2);
    if (typeConverterBodyBuilder)
      typeConverterBodyBuilder(builder, result.location);
  }
}

//===----------------------------------------------------------------------===//
// ApplyToLLVMConversionPatternsOp
//===----------------------------------------------------------------------===//

void transform::ApplyToLLVMConversionPatternsOp::populatePatterns(
    TypeConverter &typeConverter, RewritePatternSet &patterns) {
  Dialect *dialect = getContext()->getLoadedDialect(getDialectName());
  assert(dialect && "expected that dialect is loaded");
  auto *iface = cast<ConvertToLLVMPatternInterface>(dialect);
  // ConversionTarget is currently ignored because the enclosing
  // apply_conversion_patterns op sets up its own ConversionTarget.
  ConversionTarget target(*getContext());
  iface->populateConvertToLLVMConversionPatterns(
      target, static_cast<LLVMTypeConverter &>(typeConverter), patterns);
}

LogicalResult transform::ApplyToLLVMConversionPatternsOp::verifyTypeConverter(
    transform::TypeConverterBuilderOpInterface builder) {
  if (builder.getTypeConverterType() != "LLVMTypeConverter")
    return emitOpError("expected LLVMTypeConverter");
  return success();
}

LogicalResult transform::ApplyToLLVMConversionPatternsOp::verify() {
  Dialect *dialect = getContext()->getLoadedDialect(getDialectName());
  if (!dialect)
    return emitOpError("unknown dialect or dialect not loaded: ")
           << getDialectName();
  auto *iface = dyn_cast<ConvertToLLVMPatternInterface>(dialect);
  if (!iface)
    return emitOpError(
               "dialect does not implement ConvertToLLVMPatternInterface or "
               "extension was not loaded: ")
           << getDialectName();
  return success();
}

//===----------------------------------------------------------------------===//
// ApplyLoopInvariantCodeMotionOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::ApplyLoopInvariantCodeMotionOp::applyToOne(
    transform::TransformRewriter &rewriter, LoopLikeOpInterface target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  // Currently, LICM does not remove operations, so we don't need tracking.
  // If this ever changes, add a LICM entry point that takes a rewriter.
  moveLoopInvariantCode(target);
  return DiagnosedSilenceableFailure::success();
}

void transform::ApplyLoopInvariantCodeMotionOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTargetMutable(), effects);
  transform::modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// ApplyRegisteredPassOp
//===----------------------------------------------------------------------===//

void transform::ApplyRegisteredPassOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTargetMutable(), effects);
  onlyReadsHandle(getDynamicOptionsMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure
transform::ApplyRegisteredPassOp::apply(transform::TransformRewriter &rewriter,
                                        transform::TransformResults &results,
                                        transform::TransformState &state) {
  // Obtain a single options-string to pass to the pass(-pipeline) from options
  // passed in as a dictionary of keys mapping to values which are either
  // attributes or param-operands pointing to attributes.
  OperandRange dynamicOptions = getDynamicOptions();

  std::string options;
  llvm::raw_string_ostream optionsStream(options); // For "printing" attrs.

  // A helper to convert an option's attribute value into a corresponding
  // string representation, with the ability to obtain the attr(s) from a param.
  std::function<void(Attribute)> appendValueAttr = [&](Attribute valueAttr) {
    if (auto paramOperand = dyn_cast<transform::ParamOperandAttr>(valueAttr)) {
      // The corresponding value attribute(s) is/are passed in via a param.
      // Obtain the param-operand via its specified index.
      int64_t dynamicOptionIdx = paramOperand.getIndex().getInt();
      assert(dynamicOptionIdx < static_cast<int64_t>(dynamicOptions.size()) &&
             "the number of ParamOperandAttrs in the options DictionaryAttr"
             "should be the same as the number of options passed as params");
      ArrayRef<Attribute> attrsAssociatedToParam =
          state.getParams(dynamicOptions[dynamicOptionIdx]);
      // Recursive so as to append all attrs associated to the param.
      llvm::interleave(attrsAssociatedToParam, optionsStream, appendValueAttr,
                       ",");
    } else if (auto arrayAttr = dyn_cast<ArrayAttr>(valueAttr)) {
      // Recursive so as to append all nested attrs of the array.
      llvm::interleave(arrayAttr, optionsStream, appendValueAttr, ",");
    } else if (auto strAttr = dyn_cast<StringAttr>(valueAttr)) {
      // Convert to unquoted string.
      optionsStream << strAttr.getValue().str();
    } else {
      // For all other attributes, ask the attr to print itself (without type).
      valueAttr.print(optionsStream, /*elideType=*/true);
    }
  };

  // Convert the options DictionaryAttr into a single string.
  llvm::interleave(
      getOptions(), optionsStream,
      [&](auto namedAttribute) {
        optionsStream << namedAttribute.getName().str(); // Append the key.
        optionsStream << "="; // And the key-value separator.
        appendValueAttr(namedAttribute.getValue()); // And the attr's str repr.
      },
      " ");
  optionsStream.flush();

  // Get pass or pass pipeline from registry.
  const PassRegistryEntry *info = PassPipelineInfo::lookup(getPassName());
  if (!info)
    info = PassInfo::lookup(getPassName());
  if (!info)
    return emitDefiniteFailure()
           << "unknown pass or pass pipeline: " << getPassName();

  // Create pass manager and add the pass or pass pipeline.
  PassManager pm(getContext());
  if (failed(info->addToPipeline(pm, options, [&](const Twine &msg) {
        emitError(msg);
        return failure();
      }))) {
    return emitDefiniteFailure()
           << "failed to add pass or pass pipeline to pipeline: "
           << getPassName();
  }

  auto targets = SmallVector<Operation *>(state.getPayloadOps(getTarget()));
  for (Operation *target : targets) {
    // Make sure that this transform is not applied to itself. Modifying the
    // transform IR while it is being interpreted is generally dangerous. Even
    // more so when applying passes because they may perform a wide range of IR
    // modifications.
    DiagnosedSilenceableFailure payloadCheck =
        ensurePayloadIsSeparateFromTransform(*this, target);
    if (!payloadCheck.succeeded())
      return payloadCheck;

    // Run the pass or pass pipeline on the current target operation.
    if (failed(pm.run(target))) {
      auto diag = emitSilenceableError() << "pass pipeline failed";
      diag.attachNote(target->getLoc()) << "target op";
      return diag;
    }
  }

  // The applied pass will have directly modified the payload IR(s).
  results.set(llvm::cast<OpResult>(getResult()), targets);
  return DiagnosedSilenceableFailure::success();
}

static ParseResult parseApplyRegisteredPassOptions(
    OpAsmParser &parser, DictionaryAttr &options,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &dynamicOptions) {
  // Construct the options DictionaryAttr per a `{ key = value, ... }` syntax.
  SmallVector<NamedAttribute> keyValuePairs;
  size_t dynamicOptionsIdx = 0;

  // Helper for allowing parsing of option values which can be of the form:
  // - a normal attribute
  // - an operand (which would be converted to an attr referring to the operand)
  // - ArrayAttrs containing the foregoing (in correspondence with ListOptions)
  std::function<ParseResult(Attribute &)> parseValue =
      [&](Attribute &valueAttr) -> ParseResult {
    // Allow for array syntax, e.g. `[0 : i64, %param, true, %other_param]`:
    if (succeeded(parser.parseOptionalLSquare())) {
      SmallVector<Attribute> attrs;

      // Recursively parse the array's elements, which might be operands.
      if (parser.parseCommaSeparatedList(
              AsmParser::Delimiter::None,
              [&]() -> ParseResult { return parseValue(attrs.emplace_back()); },
              " in options dictionary") ||
          parser.parseRSquare())
        return failure(); // NB: Attempted parse should've output error message.

      valueAttr = ArrayAttr::get(parser.getContext(), attrs);

      return success();
    }

    // Parse the value, which can be either an attribute or an operand.
    OptionalParseResult parsedValueAttr =
        parser.parseOptionalAttribute(valueAttr);
    if (!parsedValueAttr.has_value()) {
      OpAsmParser::UnresolvedOperand operand;
      ParseResult parsedOperand = parser.parseOperand(operand);
      if (failed(parsedOperand))
        return failure(); // NB: Attempted parse should've output error message.
      // To make use of the operand, we need to store it in the options dict.
      // As SSA-values cannot occur in attributes, what we do instead is store
      // an attribute in its place that contains the index of the param-operand,
      // so that an attr-value associated to the param can be resolved later on.
      dynamicOptions.push_back(operand);
      auto wrappedIndex = IntegerAttr::get(
          IntegerType::get(parser.getContext(), 64), dynamicOptionsIdx++);
      valueAttr =
          transform::ParamOperandAttr::get(parser.getContext(), wrappedIndex);
    } else if (failed(parsedValueAttr.value())) {
      return failure(); // NB: Attempted parse should have output error message.
    } else if (isa<transform::ParamOperandAttr>(valueAttr)) {
      return parser.emitError(parser.getCurrentLocation())
             << "the param_operand attribute is a marker reserved for "
             << "indicating a value will be passed via params and is only used "
             << "in the generic print format";
    }

    return success();
  };

  // Helper for `key = value`-pair parsing where `key` is a bare identifier or a
  // string and `value` looks like either an attribute or an operand-in-an-attr.
  std::function<ParseResult()> parseKeyValuePair = [&]() -> ParseResult {
    std::string key;
    Attribute valueAttr;

    if (failed(parser.parseOptionalKeywordOrString(&key)) || key.empty())
      return parser.emitError(parser.getCurrentLocation())
             << "expected key to either be an identifier or a string";

    if (failed(parser.parseEqual()))
      return parser.emitError(parser.getCurrentLocation())
             << "expected '=' after key in key-value pair";

    if (failed(parseValue(valueAttr)))
      return parser.emitError(parser.getCurrentLocation())
             << "expected a valid attribute or operand as value associated "
             << "to key '" << key << "'";

    keyValuePairs.push_back(NamedAttribute(key, valueAttr));

    return success();
  };

  if (parser.parseCommaSeparatedList(AsmParser::Delimiter::Braces,
                                     parseKeyValuePair,
                                     " in options dictionary"))
    return failure(); // NB: Attempted parse should have output error message.

  if (DictionaryAttr::findDuplicate(
          keyValuePairs, /*isSorted=*/false) // Also sorts the keyValuePairs.
          .has_value())
    return parser.emitError(parser.getCurrentLocation())
           << "duplicate keys found in options dictionary";

  options = DictionaryAttr::getWithSorted(parser.getContext(), keyValuePairs);

  return success();
}

static void printApplyRegisteredPassOptions(OpAsmPrinter &printer,
                                            Operation *op,
                                            DictionaryAttr options,
                                            ValueRange dynamicOptions) {
  if (options.empty())
    return;

  std::function<void(Attribute)> printOptionValue = [&](Attribute valueAttr) {
    if (auto paramOperandAttr =
            dyn_cast<transform::ParamOperandAttr>(valueAttr)) {
      // Resolve index of param-operand to its actual SSA-value and print that.
      printer.printOperand(
          dynamicOptions[paramOperandAttr.getIndex().getInt()]);
    } else if (auto arrayAttr = dyn_cast<ArrayAttr>(valueAttr)) {
      // This case is so that ArrayAttr-contained operands are pretty-printed.
      printer << "[";
      llvm::interleaveComma(arrayAttr, printer, printOptionValue);
      printer << "]";
    } else {
      printer.printAttribute(valueAttr);
    }
  };

  printer << "{";
  llvm::interleaveComma(options, printer, [&](NamedAttribute namedAttribute) {
    printer << namedAttribute.getName();
    printer << " = ";
    printOptionValue(namedAttribute.getValue());
  });
  printer << "}";
}

LogicalResult transform::ApplyRegisteredPassOp::verify() {
  // Check that there is a one-to-one correspondence between param operands
  // and references to dynamic options in the options dictionary.

  auto dynamicOptions = SmallVector<Value>(getDynamicOptions());

  // Helper for option values to mark seen operands as having been seen (once).
  std::function<LogicalResult(Attribute)> checkOptionValue =
      [&](Attribute valueAttr) -> LogicalResult {
    if (auto paramOperand = dyn_cast<transform::ParamOperandAttr>(valueAttr)) {
      int64_t dynamicOptionIdx = paramOperand.getIndex().getInt();
      if (dynamicOptionIdx < 0 ||
          dynamicOptionIdx >= static_cast<int64_t>(dynamicOptions.size()))
        return emitOpError()
               << "dynamic option index " << dynamicOptionIdx
               << " is out of bounds for the number of dynamic options: "
               << dynamicOptions.size();
      if (dynamicOptions[dynamicOptionIdx] == nullptr)
        return emitOpError() << "dynamic option index " << dynamicOptionIdx
                             << " is already used in options";
      dynamicOptions[dynamicOptionIdx] = nullptr; // Mark this option as used.
    } else if (auto arrayAttr = dyn_cast<ArrayAttr>(valueAttr)) {
      // Recurse into ArrayAttrs as they may contain references to operands.
      for (auto eltAttr : arrayAttr)
        if (failed(checkOptionValue(eltAttr)))
          return failure();
    }
    return success();
  };

  for (NamedAttribute namedAttr : getOptions())
    if (failed(checkOptionValue(namedAttr.getValue())))
      return failure();

  // All dynamicOptions-params seen in the dict will have been set to null.
  for (Value dynamicOption : dynamicOptions)
    if (dynamicOption)
      return emitOpError() << "a param operand does not have a corresponding "
                           << "param_operand attr in the options dict";

  return success();
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::CastOp::applyToOne(transform::TransformRewriter &rewriter,
                              Operation *target, ApplyToEachResultList &results,
                              transform::TransformState &state) {
  results.push_back(target);
  return DiagnosedSilenceableFailure::success();
}

void transform::CastOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsPayload(effects);
  onlyReadsHandle(getInputMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
}

bool transform::CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  assert(inputs.size() == 1 && "expected one input");
  assert(outputs.size() == 1 && "expected one output");
  return llvm::all_of(
      std::initializer_list<Type>{inputs.front(), outputs.front()},
      llvm::IsaPred<transform::TransformHandleTypeInterface>);
}

//===----------------------------------------------------------------------===//
// CollectMatchingOp
//===----------------------------------------------------------------------===//

/// Applies matcher operations from the given `block` using
/// `blockArgumentMapping` to initialize block arguments. Updates `state`
/// accordingly. If any of the matcher produces a silenceable failure, discards
/// it (printing the content to the debug output stream) and returns failure. If
/// any of the matchers produces a definite failure, reports it and returns
/// failure. If all matchers in the block succeed, populates `mappings` with the
/// payload entities associated with the block terminator operands. Note that
/// `mappings` will be cleared before that.
static DiagnosedSilenceableFailure
matchBlock(Block &block,
           ArrayRef<SmallVector<transform::MappedValue>> blockArgumentMapping,
           transform::TransformState &state,
           SmallVectorImpl<SmallVector<transform::MappedValue>> &mappings) {
  assert(block.getParent() && "cannot match using a detached block");
  auto matchScope = state.make_region_scope(*block.getParent());
  if (failed(
          state.mapBlockArguments(block.getArguments(), blockArgumentMapping)))
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
  // Our contract with the caller is that the mappings will contain only the
  // newly mapped values, clear the rest.
  mappings.clear();
  transform::detail::prepareValueMappings(mappings, yieldedValues, state);
  return DiagnosedSilenceableFailure::success();
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

//===----------------------------------------------------------------------===//
// CollectMatchingOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::CollectMatchingOp::apply(transform::TransformRewriter &rewriter,
                                    transform::TransformResults &results,
                                    transform::TransformState &state) {
  auto matcher = SymbolTable::lookupNearestSymbolFrom<FunctionOpInterface>(
      getOperation(), getMatcher());
  if (matcher.isExternal()) {
    return emitDefiniteFailure()
           << "unresolved external symbol " << getMatcher();
  }

  SmallVector<SmallVector<MappedValue>, 2> rawResults;
  rawResults.resize(getOperation()->getNumResults());
  std::optional<DiagnosedSilenceableFailure> maybeFailure;
  for (Operation *root : state.getPayloadOps(getRoot())) {
    WalkResult walkResult = root->walk([&](Operation *op) {
      LDBG(1, DEBUG_TYPE_MATCHER)
          << "matching "
          << OpWithFlags(op, OpPrintingFlags().assumeVerified().skipRegions())
          << " @" << op;

      // Try matching.
      SmallVector<SmallVector<MappedValue>> mappings;
      SmallVector<transform::MappedValue> inputMapping({op});
      DiagnosedSilenceableFailure diag = matchBlock(
          matcher.getFunctionBody().front(),
          ArrayRef<SmallVector<transform::MappedValue>>(inputMapping), state,
          mappings);
      if (diag.isDefiniteFailure())
        return WalkResult::interrupt();
      if (diag.isSilenceableFailure()) {
        LDBG(1, DEBUG_TYPE_MATCHER) << "matcher " << matcher.getName()
                                    << " failed: " << diag.getMessage();
        return WalkResult::advance();
      }

      // If succeeded, collect results.
      for (auto &&[i, mapping] : llvm::enumerate(mappings)) {
        if (mapping.size() != 1) {
          maybeFailure.emplace(emitSilenceableError()
                               << "result #" << i << ", associated with "
                               << mapping.size()
                               << " payload objects, expected 1");
          return WalkResult::interrupt();
        }
        rawResults[i].push_back(mapping[0]);
      }
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted())
      return std::move(*maybeFailure);
    assert(!maybeFailure && "failure set but the walk was not interrupted");

    for (auto &&[opResult, rawResult] :
         llvm::zip_equal(getOperation()->getResults(), rawResults)) {
      results.setMappedValues(opResult, rawResult);
    }
  }
  return DiagnosedSilenceableFailure::success();
}

void transform::CollectMatchingOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getRootMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  onlyReadsPayload(effects);
}

LogicalResult transform::CollectMatchingOp::verifySymbolUses(
    SymbolTableCollection &symbolTable) {
  auto matcherSymbol = dyn_cast_or_null<FunctionOpInterface>(
      symbolTable.lookupNearestSymbolFrom(getOperation(), getMatcher()));
  if (!matcherSymbol ||
      !isa<TransformOpInterface>(matcherSymbol.getOperation()))
    return emitError() << "unresolved matcher symbol " << getMatcher();

  ArrayRef<Type> argumentTypes = matcherSymbol.getArgumentTypes();
  if (argumentTypes.size() != 1 ||
      !isa<TransformHandleTypeInterface>(argumentTypes[0])) {
    return emitError()
           << "expected the matcher to take one operation handle argument";
  }
  if (!matcherSymbol.getArgAttr(
          0, transform::TransformDialect::kArgReadOnlyAttrName)) {
    return emitError() << "expected the matcher argument to be marked readonly";
  }

  ArrayRef<Type> resultTypes = matcherSymbol.getResultTypes();
  if (resultTypes.size() != getOperation()->getNumResults()) {
    return emitError()
           << "expected the matcher to yield as many values as op has results ("
           << getOperation()->getNumResults() << "), got "
           << resultTypes.size();
  }

  for (auto &&[i, matcherType, resultType] :
       llvm::enumerate(resultTypes, getOperation()->getResultTypes())) {
    if (implementSameTransformInterface(matcherType, resultType))
      continue;

    return emitError()
           << "mismatching type interfaces for matcher result and op result #"
           << i;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ForeachMatchOp
//===----------------------------------------------------------------------===//

// This is fine because nothing is actually consumed by this op.
bool transform::ForeachMatchOp::allowsRepeatedHandleOperands() { return true; }

DiagnosedSilenceableFailure
transform::ForeachMatchOp::apply(transform::TransformRewriter &rewriter,
                                 transform::TransformResults &results,
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

  DiagnosedSilenceableFailure overallDiag =
      DiagnosedSilenceableFailure::success();

  SmallVector<SmallVector<MappedValue>> matchInputMapping;
  SmallVector<SmallVector<MappedValue>> matchOutputMapping;
  SmallVector<SmallVector<MappedValue>> actionResultMapping;
  // Explicitly add the mapping for the first block argument (the op being
  // matched).
  matchInputMapping.emplace_back();
  transform::detail::prepareValueMappings(matchInputMapping,
                                          getForwardedInputs(), state);
  SmallVector<MappedValue> &firstMatchArgument = matchInputMapping.front();
  actionResultMapping.resize(getForwardedOutputs().size());

  for (Operation *root : state.getPayloadOps(getRoot())) {
    WalkResult walkResult = root->walk([&](Operation *op) {
      // If getRestrictRoot is not present, skip over the root op itself so we
      // don't invalidate it.
      if (!getRestrictRoot() && op == root)
        return WalkResult::advance();

      LDBG(1, DEBUG_TYPE_MATCHER)
          << "matching "
          << OpWithFlags(op, OpPrintingFlags().assumeVerified().skipRegions())
          << " @" << op;

      firstMatchArgument.clear();
      firstMatchArgument.push_back(op);

      // Try all the match/action pairs until the first successful match.
      for (auto [matcher, action] : matchActionPairs) {
        DiagnosedSilenceableFailure diag =
            matchBlock(matcher.getFunctionBody().front(), matchInputMapping,
                       state, matchOutputMapping);
        if (diag.isDefiniteFailure())
          return WalkResult::interrupt();
        if (diag.isSilenceableFailure()) {
          LDBG(1, DEBUG_TYPE_MATCHER) << "matcher " << matcher.getName()
                                      << " failed: " << diag.getMessage();
          continue;
        }

        auto scope = state.make_region_scope(action.getFunctionBody());
        if (failed(state.mapBlockArguments(
                action.getFunctionBody().front().getArguments(),
                matchOutputMapping))) {
          return WalkResult::interrupt();
        }

        for (Operation &transform :
             action.getFunctionBody().front().without_terminator()) {
          DiagnosedSilenceableFailure result =
              state.applyTransform(cast<TransformOpInterface>(transform));
          if (result.isDefiniteFailure())
            return WalkResult::interrupt();
          if (result.isSilenceableFailure()) {
            if (overallDiag.succeeded()) {
              overallDiag = emitSilenceableError() << "actions failed";
            }
            overallDiag.attachNote(action->getLoc())
                << "failed action: " << result.getMessage();
            overallDiag.attachNote(op->getLoc())
                << "when applied to this matching payload";
            (void)result.silence();
            continue;
          }
        }
        if (failed(detail::appendValueMappings(
                MutableArrayRef<SmallVector<MappedValue>>(actionResultMapping),
                action.getFunctionBody().front().getTerminator()->getOperands(),
                state, getFlattenResults()))) {
          emitDefiniteFailure()
              << "action @" << action.getName()
              << " has results associated with multiple payload entities, "
                 "but flattening was not requested";
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
  for (auto &&[result, mapping] :
       llvm::zip_equal(getForwardedOutputs(), actionResultMapping)) {
    results.setMappedValues(result, mapping);
  }
  return overallDiag;
}

void transform::ForeachMatchOp::getAsmResultNames(
    OpAsmSetValueNameFn setNameFn) {
  setNameFn(getUpdated(), "updated_root");
  for (Value v : getForwardedOutputs()) {
    setNameFn(v, "yielded");
  }
}

void transform::ForeachMatchOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  // Bail if invalid.
  if (getOperation()->getNumOperands() < 1 ||
      getOperation()->getNumResults() < 1) {
    return modifiesPayload(effects);
  }

  consumesHandle(getRootMutable(), effects);
  onlyReadsHandle(getForwardedInputsMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
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

/// Checks that the attributes of the function-like operation have correct
/// consumption effect annotations. If `alsoVerifyInternal`, checks for
/// annotations being present even if they can be inferred from the body.
static DiagnosedSilenceableFailure
verifyFunctionLikeConsumeAnnotations(FunctionOpInterface op, bool emitWarnings,
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
    if (emitWarnings && !consumedArguments.contains(i) && isConsumed) {
      // Cannot use op.emitWarning() here as it would attempt to verify the op
      // before printing, resulting in infinite recursion.
      emitWarning(op->getLoc())
          << "op argument #" << i
          << " is not consumed in the body but is marked as consumed";
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
    // Presence and typing.
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
                                                    /*emitWarnings=*/false,
                                                    /*alsoVerifyInternal=*/true)
                   .checkAndReport())) {
      return failure();
    }
    if (failed(verifyFunctionLikeConsumeAnnotations(actionSymbol,
                                                    /*emitWarnings=*/false,
                                                    /*alsoVerifyInternal=*/true)
                   .checkAndReport())) {
      return failure();
    }

    // Input -> matcher forwarding.
    TypeRange operandTypes = getOperandTypes();
    TypeRange matcherArguments = matcherSymbol.getArgumentTypes();
    if (operandTypes.size() != matcherArguments.size()) {
      InFlightDiagnostic diag =
          emitError() << "the number of operands (" << operandTypes.size()
                      << ") doesn't match the number of matcher arguments ("
                      << matcherArguments.size() << ") for " << matcher;
      diag.attachNote(matcherSymbol->getLoc()) << "symbol declaration";
      return diag;
    }
    for (auto &&[i, operand, argument] :
         llvm::enumerate(operandTypes, matcherArguments)) {
      if (matcherSymbol.getArgAttr(i, consumedAttr)) {
        InFlightDiagnostic diag =
            emitOpError()
            << "does not expect matcher symbol to consume its operand #" << i;
        diag.attachNote(matcherSymbol->getLoc()) << "symbol declaration";
        return diag;
      }

      if (implementSameTransformInterface(operand, argument))
        continue;

      InFlightDiagnostic diag =
          emitError()
          << "mismatching type interfaces for operand and matcher argument #"
          << i << " of matcher " << matcher;
      diag.attachNote(matcherSymbol->getLoc()) << "symbol declaration";
      return diag;
    }

    // Matcher -> action forwarding.
    TypeRange matcherResults = matcherSymbol.getResultTypes();
    TypeRange actionArguments = actionSymbol.getArgumentTypes();
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
                         << i << "of matcher " << matcher << " and action "
                         << action;
    }

    // Action -> result forwarding.
    TypeRange actionResults = actionSymbol.getResultTypes();
    auto resultTypes = TypeRange(getResultTypes()).drop_front();
    if (actionResults.size() != resultTypes.size()) {
      InFlightDiagnostic diag =
          emitError() << "the number of action results ("
                      << actionResults.size() << ") for " << action
                      << " doesn't match the number of extra op results ("
                      << resultTypes.size() << ")";
      diag.attachNote(actionSymbol->getLoc()) << "symbol declaration";
      return diag;
    }
    for (auto &&[i, resultType, actionType] :
         llvm::enumerate(resultTypes, actionResults)) {
      if (implementSameTransformInterface(resultType, actionType))
        continue;

      InFlightDiagnostic diag =
          emitError() << "mismatching type interfaces for action result #" << i
                      << " of action " << action << " and op result";
      diag.attachNote(actionSymbol->getLoc()) << "symbol declaration";
      return diag;
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ForeachOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::ForeachOp::apply(transform::TransformRewriter &rewriter,
                            transform::TransformResults &results,
                            transform::TransformState &state) {
  // We store the payloads before executing the body as ops may be removed from
  // the mapping by the TrackingRewriter while iteration is in progress.
  SmallVector<SmallVector<MappedValue>> payloads;
  detail::prepareValueMappings(payloads, getTargets(), state);
  size_t numIterations = payloads.empty() ? 0 : payloads.front().size();
  bool withZipShortest = getWithZipShortest();

  // In case of `zip_shortest`, set the number of iterations to the
  // smallest payload in the targets.
  if (withZipShortest) {
    numIterations =
        llvm::min_element(payloads, [&](const SmallVector<MappedValue> &a,
                                        const SmallVector<MappedValue> &b) {
          return a.size() < b.size();
        })->size();

    for (size_t argIdx = 0; argIdx < payloads.size(); argIdx++)
      payloads[argIdx].resize(numIterations);
  }

  // As we will be "zipping" over them, check all payloads have the same size.
  // `zip_shortest` adjusts all payloads to the same size, so skip this check
  // when true.
  for (size_t argIdx = 1; !withZipShortest && argIdx < payloads.size();
       argIdx++) {
    if (payloads[argIdx].size() != numIterations) {
      return emitSilenceableError()
             << "prior targets' payload size (" << numIterations
             << ") differs from payload size (" << payloads[argIdx].size()
             << ") of target " << getTargets()[argIdx];
    }
  }

  // Start iterating, indexing into payloads to obtain the right arguments to
  // call the body with - each slice of payloads at the same argument index
  // corresponding to a tuple to use as the body's block arguments.
  ArrayRef<BlockArgument> blockArguments = getBody().front().getArguments();
  SmallVector<SmallVector<MappedValue>> zippedResults(getNumResults(), {});
  for (size_t iterIdx = 0; iterIdx < numIterations; iterIdx++) {
    auto scope = state.make_region_scope(getBody());
    // Set up arguments to the region's block.
    for (auto &&[argIdx, blockArg] : llvm::enumerate(blockArguments)) {
      MappedValue argument = payloads[argIdx][iterIdx];
      // Note that each blockArg's handle gets associated with just a single
      // element from the corresponding target's payload.
      if (failed(state.mapBlockArgument(blockArg, {argument})))
        return DiagnosedSilenceableFailure::definiteFailure();
    }

    // Execute loop body.
    for (Operation &transform : getBody().front().without_terminator()) {
      DiagnosedSilenceableFailure result = state.applyTransform(
          llvm::cast<transform::TransformOpInterface>(transform));
      if (!result.succeeded())
        return result;
    }

    // Append yielded payloads to corresponding results from prior iterations.
    OperandRange yieldOperands = getYieldOp().getOperands();
    for (auto &&[result, yieldOperand, resTuple] :
         llvm::zip_equal(getResults(), yieldOperands, zippedResults))
      // NB: each iteration we add any number of ops/vals/params to a result.
      if (isa<TransformHandleTypeInterface>(result.getType()))
        llvm::append_range(resTuple, state.getPayloadOps(yieldOperand));
      else if (isa<TransformValueHandleTypeInterface>(result.getType()))
        llvm::append_range(resTuple, state.getPayloadValues(yieldOperand));
      else if (isa<TransformParamTypeInterface>(result.getType()))
        llvm::append_range(resTuple, state.getParams(yieldOperand));
      else
        assert(false && "unhandled handle type");
  }

  // Associate the accumulated result payloads to the op's actual results.
  for (auto &&[result, resPayload] : zip_equal(getResults(), zippedResults))
    results.setMappedValues(llvm::cast<OpResult>(result), resPayload);

  return DiagnosedSilenceableFailure::success();
}

void transform::ForeachOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  // NB: this `zip` should be `zip_equal` - while this op's verifier catches
  // arity errors, this method might get called before/in absence of `verify()`.
  for (auto &&[target, blockArg] :
       llvm::zip(getTargetsMutable(), getBody().front().getArguments())) {
    BlockArgument blockArgument = blockArg;
    if (any_of(getBody().front().without_terminator(), [&](Operation &op) {
          return isHandleConsumed(blockArgument,
                                  cast<TransformOpInterface>(&op));
        })) {
      consumesHandle(target, effects);
    } else {
      onlyReadsHandle(target, effects);
    }
  }

  if (any_of(getBody().front().without_terminator(), [&](Operation &op) {
        return doesModifyPayload(cast<TransformOpInterface>(&op));
      })) {
    modifiesPayload(effects);
  } else if (any_of(getBody().front().without_terminator(), [&](Operation &op) {
               return doesReadPayload(cast<TransformOpInterface>(&op));
             })) {
    onlyReadsPayload(effects);
  }

  producesHandle(getOperation()->getOpResults(), effects);
}

void transform::ForeachOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  Region *bodyRegion = &getBody();
  if (point.isParent()) {
    regions.emplace_back(bodyRegion, bodyRegion->getArguments());
    return;
  }

  // Branch back to the region or the parent.
  assert(point == getBody() && "unexpected region index");
  regions.emplace_back(bodyRegion, bodyRegion->getArguments());
  regions.emplace_back();
}

OperandRange
transform::ForeachOp::getEntrySuccessorOperands(RegionBranchPoint point) {
  // Each block argument handle is mapped to a subset (one op to be precise)
  // of the payload of the corresponding `targets` operand of ForeachOp.
  assert(point == getBody() && "unexpected region index");
  return getOperation()->getOperands();
}

transform::YieldOp transform::ForeachOp::getYieldOp() {
  return cast<transform::YieldOp>(getBody().front().getTerminator());
}

LogicalResult transform::ForeachOp::verify() {
  for (auto [targetOpt, bodyArgOpt] :
       llvm::zip_longest(getTargets(), getBody().front().getArguments())) {
    if (!targetOpt || !bodyArgOpt)
      return emitOpError() << "expects the same number of targets as the body "
                              "has block arguments";
    if (targetOpt.value().getType() != bodyArgOpt.value().getType())
      return emitOpError(
          "expects co-indexed targets and the body's "
          "block arguments to have the same op/value/param type");
  }

  for (auto [resultOpt, yieldOperandOpt] :
       llvm::zip_longest(getResults(), getYieldOp().getOperands())) {
    if (!resultOpt || !yieldOperandOpt)
      return emitOpError() << "expects the same number of results as the "
                              "yield terminator has operands";
    if (resultOpt.value().getType() != yieldOperandOpt.value().getType())
      return emitOpError("expects co-indexed results and yield "
                         "operands to have the same op/value/param type");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// GetParentOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::GetParentOp::apply(transform::TransformRewriter &rewriter,
                              transform::TransformResults &results,
                              transform::TransformState &state) {
  SmallVector<Operation *> parents;
  DenseSet<Operation *> resultSet;
  for (Operation *target : state.getPayloadOps(getTarget())) {
    Operation *parent = target;
    for (int64_t i = 0, e = getNthParent(); i < e; ++i) {
      parent = parent->getParentOp();
      while (parent) {
        bool checkIsolatedFromAbove =
            !getIsolatedFromAbove() ||
            parent->hasTrait<OpTrait::IsIsolatedFromAbove>();
        bool checkOpName = !getOpName().has_value() ||
                           parent->getName().getStringRef() == *getOpName();
        if (checkIsolatedFromAbove && checkOpName)
          break;
        parent = parent->getParentOp();
      }
      if (!parent) {
        if (getAllowEmptyResults()) {
          results.set(llvm::cast<OpResult>(getResult()), parents);
          return DiagnosedSilenceableFailure::success();
        }
        DiagnosedSilenceableFailure diag =
            emitSilenceableError()
            << "could not find a parent op that matches all requirements";
        diag.attachNote(target->getLoc()) << "target op";
        return diag;
      }
    }
    if (getDeduplicate()) {
      if (resultSet.insert(parent).second)
        parents.push_back(parent);
    } else {
      parents.push_back(parent);
    }
  }
  results.set(llvm::cast<OpResult>(getResult()), parents);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// GetConsumersOfResult
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::GetConsumersOfResult::apply(transform::TransformRewriter &rewriter,
                                       transform::TransformResults &results,
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
transform::GetDefiningOp::apply(transform::TransformRewriter &rewriter,
                                transform::TransformResults &results,
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
transform::GetProducerOfOperand::apply(transform::TransformRewriter &rewriter,
                                       transform::TransformResults &results,
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
// GetOperandOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::GetOperandOp::apply(transform::TransformRewriter &rewriter,
                               transform::TransformResults &results,
                               transform::TransformState &state) {
  SmallVector<Value> operands;
  for (Operation *target : state.getPayloadOps(getTarget())) {
    SmallVector<int64_t> operandPositions;
    DiagnosedSilenceableFailure diag = expandTargetSpecification(
        getLoc(), getIsAll(), getIsInverted(), getRawPositionList(),
        target->getNumOperands(), operandPositions);
    if (diag.isSilenceableFailure()) {
      diag.attachNote(target->getLoc())
          << "while considering positions of this payload operation";
      return diag;
    }
    llvm::append_range(operands,
                       llvm::map_range(operandPositions, [&](int64_t pos) {
                         return target->getOperand(pos);
                       }));
  }
  results.setValues(cast<OpResult>(getResult()), operands);
  return DiagnosedSilenceableFailure::success();
}

LogicalResult transform::GetOperandOp::verify() {
  return verifyTransformMatchDimsOp(getOperation(), getRawPositionList(),
                                    getIsInverted(), getIsAll());
}

//===----------------------------------------------------------------------===//
// GetResultOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::GetResultOp::apply(transform::TransformRewriter &rewriter,
                              transform::TransformResults &results,
                              transform::TransformState &state) {
  SmallVector<Value> opResults;
  for (Operation *target : state.getPayloadOps(getTarget())) {
    SmallVector<int64_t> resultPositions;
    DiagnosedSilenceableFailure diag = expandTargetSpecification(
        getLoc(), getIsAll(), getIsInverted(), getRawPositionList(),
        target->getNumResults(), resultPositions);
    if (diag.isSilenceableFailure()) {
      diag.attachNote(target->getLoc())
          << "while considering positions of this payload operation";
      return diag;
    }
    llvm::append_range(opResults,
                       llvm::map_range(resultPositions, [&](int64_t pos) {
                         return target->getResult(pos);
                       }));
  }
  results.setValues(cast<OpResult>(getResult()), opResults);
  return DiagnosedSilenceableFailure::success();
}

LogicalResult transform::GetResultOp::verify() {
  return verifyTransformMatchDimsOp(getOperation(), getRawPositionList(),
                                    getIsInverted(), getIsAll());
}

//===----------------------------------------------------------------------===//
// GetTypeOp
//===----------------------------------------------------------------------===//

void transform::GetTypeOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getValueMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  onlyReadsPayload(effects);
}

DiagnosedSilenceableFailure
transform::GetTypeOp::apply(transform::TransformRewriter &rewriter,
                            transform::TransformResults &results,
                            transform::TransformState &state) {
  SmallVector<Attribute> params;
  for (Value value : state.getPayloadValues(getValue())) {
    Type type = value.getType();
    if (getElemental()) {
      if (auto shaped = dyn_cast<ShapedType>(type)) {
        type = shaped.getElementType();
      }
    }
    params.push_back(TypeAttr::get(type));
  }
  results.setParams(cast<OpResult>(getResult()), params);
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
transform::IncludeOp::apply(transform::TransformRewriter &rewriter,
                            transform::TransformResults &results,
                            transform::TransformState &state) {
  auto callee = SymbolTable::lookupNearestSymbolFrom<NamedSequenceOp>(
      getOperation(), getTarget());
  assert(callee && "unverified reference to unknown symbol");

  if (callee.isExternal())
    return emitDefiniteFailure() << "unresolved external named sequence";

  // Map operands to block arguments.
  SmallVector<SmallVector<MappedValue>> mappings;
  detail::prepareValueMappings(mappings, getOperands(), state);
  auto scope = state.make_region_scope(callee.getBody());
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
verifyNamedSequenceOp(transform::NamedSequenceOp op, bool emitWarnings);

void transform::IncludeOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  // Always mark as modifying the payload.
  // TODO: a mechanism to annotate effects on payload. Even when all handles are
  // only read, the payload may still be modified, so we currently stay on the
  // conservative side and always indicate modification. This may prevent some
  // code reordering.
  modifiesPayload(effects);

  // Results are always produced.
  producesHandle(getOperation()->getOpResults(), effects);

  // Adds default effects to operands and results. This will be added if
  // preconditions fail so the trait verifier doesn't complain about missing
  // effects and the real precondition failure is reported later on.
  auto defaultEffects = [&] {
    onlyReadsHandle(getOperation()->getOpOperands(), effects);
  };

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
      verifyNamedSequenceOp(callee, /*emitWarnings=*/false);
  if (!earlyVerifierResult.succeeded()) {
    (void)earlyVerifierResult.silence();
    return defaultEffects();
  }

  for (unsigned i = 0, e = getNumOperands(); i < e; ++i) {
    if (callee.getArgAttr(i, TransformDialect::kArgConsumedAttrName))
      consumesHandle(getOperation()->getOpOperand(i), effects);
    else
      onlyReadsHandle(getOperation()->getOpOperand(i), effects);
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
             cast<FunctionOpInterface>(*target), /*emitWarnings=*/false,
             /*alsoVerifyInternal=*/true)
      .checkAndReport();
}

//===----------------------------------------------------------------------===//
// MatchOperationEmptyOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::MatchOperationEmptyOp::matchOperation(
    ::std::optional<::mlir::Operation *> maybeCurrent,
    transform::TransformResults &results, transform::TransformState &state) {
  if (!maybeCurrent.has_value()) {
    LDBG(1, DEBUG_TYPE_MATCHER) << "MatchOperationEmptyOp success";
    return DiagnosedSilenceableFailure::success();
  }
  LDBG(1, DEBUG_TYPE_MATCHER) << "MatchOperationEmptyOp failure";
  return emitSilenceableError() << "operation is not empty";
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
transform::MatchParamCmpIOp::apply(transform::TransformRewriter &rewriter,
                                   transform::TransformResults &results,
                                   transform::TransformState &state) {
  auto signedAPIntAsString = [&](const APInt &value) {
    std::string str;
    llvm::raw_string_ostream os(str);
    value.print(os, /*isSigned=*/true);
    return str;
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
  onlyReadsHandle(getParamMutable(), effects);
  onlyReadsHandle(getReferenceMutable(), effects);
}

//===----------------------------------------------------------------------===//
// ParamConstantOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::ParamConstantOp::apply(transform::TransformRewriter &rewriter,
                                  transform::TransformResults &results,
                                  transform::TransformState &state) {
  results.setParams(cast<OpResult>(getParam()), {getValue()});
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// MergeHandlesOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::MergeHandlesOp::apply(transform::TransformRewriter &rewriter,
                                 transform::TransformResults &results,
                                 transform::TransformState &state) {
  ValueRange handles = getHandles();
  if (isa<TransformHandleTypeInterface>(handles.front().getType())) {
    SmallVector<Operation *> operations;
    for (Value operand : handles)
      llvm::append_range(operations, state.getPayloadOps(operand));
    if (!getDeduplicate()) {
      results.set(llvm::cast<OpResult>(getResult()), operations);
      return DiagnosedSilenceableFailure::success();
    }

    SetVector<Operation *> uniqued(llvm::from_range, operations);
    results.set(llvm::cast<OpResult>(getResult()), uniqued.getArrayRef());
    return DiagnosedSilenceableFailure::success();
  }

  if (llvm::isa<TransformParamTypeInterface>(handles.front().getType())) {
    SmallVector<Attribute> attrs;
    for (Value attribute : handles)
      llvm::append_range(attrs, state.getParams(attribute));
    if (!getDeduplicate()) {
      results.setParams(cast<OpResult>(getResult()), attrs);
      return DiagnosedSilenceableFailure::success();
    }

    SetVector<Attribute> uniqued(llvm::from_range, attrs);
    results.setParams(cast<OpResult>(getResult()), uniqued.getArrayRef());
    return DiagnosedSilenceableFailure::success();
  }

  assert(
      llvm::isa<TransformValueHandleTypeInterface>(handles.front().getType()) &&
      "expected value handle type");
  SmallVector<Value> payloadValues;
  for (Value value : handles)
    llvm::append_range(payloadValues, state.getPayloadValues(value));
  if (!getDeduplicate()) {
    results.setValues(cast<OpResult>(getResult()), payloadValues);
    return DiagnosedSilenceableFailure::success();
  }

  SetVector<Value> uniqued(llvm::from_range, payloadValues);
  results.setValues(cast<OpResult>(getResult()), uniqued.getArrayRef());
  return DiagnosedSilenceableFailure::success();
}

bool transform::MergeHandlesOp::allowsRepeatedHandleOperands() {
  // Handles may be the same if deduplicating is enabled.
  return getDeduplicate();
}

void transform::MergeHandlesOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getHandlesMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);

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
transform::NamedSequenceOp::apply(transform::TransformRewriter &rewriter,
                                  transform::TransformResults &results,
                                  transform::TransformState &state) {
  if (isExternal())
    return emitDefiniteFailure() << "unresolved external named sequence";

  // Map the entry block argument to the list of operations.
  // Note: this is the same implementation as PossibleTopLevelTransformOp but
  // without attaching the interface / trait since that is tailored to a
  // dangling top-level op that does not get "called".
  auto scope = state.make_region_scope(getBody());
  if (failed(detail::mapPossibleTopLevelTransformOpBlockArguments(
          state, this->getOperation(), getBody())))
    return DiagnosedSilenceableFailure::definiteFailure();

  return applySequenceBlock(getBody().front(),
                            FailurePropagationMode::Propagate, state, results);
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
verifyNamedSequenceOp(transform::NamedSequenceOp op, bool emitWarnings) {
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
    return verifyFunctionLikeConsumeAnnotations(cast<FunctionOpInterface>(*op),
                                                emitWarnings);

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
      verifyFunctionLikeConsumeAnnotations(funcOp, emitWarnings);
  if (!diag.succeeded())
    return diag;

  return verifyYieldingSingleBlockOp(funcOp,
                                     /*allowExternal=*/true);
}

LogicalResult transform::NamedSequenceOp::verify() {
  // Actual verification happens in a separate function for reusability.
  return verifyNamedSequenceOp(*this, /*emitWarnings=*/true).checkAndReport();
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

void transform::NamedSequenceOp::build(OpBuilder &builder,
                                       OperationState &state, StringRef symName,
                                       Type rootType, TypeRange resultTypes,
                                       SequenceBodyBuilderFn bodyBuilder,
                                       ArrayRef<NamedAttribute> attrs,
                                       ArrayRef<DictionaryAttr> argAttrs) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(symName));
  state.addAttribute(getFunctionTypeAttrName(state.name),
                     TypeAttr::get(FunctionType::get(builder.getContext(),
                                                     rootType, resultTypes)));
  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();

  buildSequenceBody(builder, state, rootType,
                    /*extraBindingTypes=*/TypeRange(), bodyBuilder);
}

//===----------------------------------------------------------------------===//
// NumAssociationsOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::NumAssociationsOp::apply(transform::TransformRewriter &rewriter,
                                    transform::TransformResults &results,
                                    transform::TransformState &state) {
  size_t numAssociations =
      llvm::TypeSwitch<Type, size_t>(getHandle().getType())
          .Case([&](TransformHandleTypeInterface opHandle) {
            return llvm::range_size(state.getPayloadOps(getHandle()));
          })
          .Case([&](TransformValueHandleTypeInterface valueHandle) {
            return llvm::range_size(state.getPayloadValues(getHandle()));
          })
          .Case([&](TransformParamTypeInterface param) {
            return llvm::range_size(state.getParams(getHandle()));
          })
          .Default([](Type) {
            llvm_unreachable("unknown kind of transform dialect type");
            return 0;
          });
  results.setParams(cast<OpResult>(getNum()),
                    rewriter.getI64IntegerAttr(numAssociations));
  return DiagnosedSilenceableFailure::success();
}

LogicalResult transform::NumAssociationsOp::verify() {
  // Verify that the result type accepts an i64 attribute as payload.
  auto resultType = cast<TransformParamTypeInterface>(getNum().getType());
  return resultType
      .checkPayload(getLoc(), {Builder(getContext()).getI64IntegerAttr(0)})
      .checkAndReport();
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::SelectOp::apply(transform::TransformRewriter &rewriter,
                           transform::TransformResults &results,
                           transform::TransformState &state) {
  SmallVector<Operation *> result;
  auto payloadOps = state.getPayloadOps(getTarget());
  for (Operation *op : payloadOps) {
    if (op->getName().getStringRef() == getOpName())
      result.push_back(op);
  }
  results.set(cast<OpResult>(getResult()), result);
  return DiagnosedSilenceableFailure::success();
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
transform::SplitHandleOp::apply(transform::TransformRewriter &rewriter,
                                transform::TransformResults &results,
                                transform::TransformState &state) {
  int64_t numPayloads =
      llvm::TypeSwitch<Type, int64_t>(getHandle().getType())
          .Case<TransformHandleTypeInterface>([&](auto x) {
            return llvm::range_size(state.getPayloadOps(getHandle()));
          })
          .Case<TransformValueHandleTypeInterface>([&](auto x) {
            return llvm::range_size(state.getPayloadValues(getHandle()));
          })
          .Case<TransformParamTypeInterface>([&](auto x) {
            return llvm::range_size(state.getParams(getHandle()));
          })
          .Default([](auto x) {
            llvm_unreachable("unknown transform dialect type interface");
            return -1;
          });

  auto produceNumOpsError = [&]() {
    return emitSilenceableError()
           << getHandle() << " expected to contain " << this->getNumResults()
           << " payloads but it contains " << numPayloads << " payloads";
  };

  // Fail if there are more payload ops than results and no overflow result was
  // specified.
  if (numPayloads > getNumResults() && !getOverflowResult().has_value())
    return produceNumOpsError();

  // Fail if there are more results than payload ops. Unless:
  // - "fail_on_payload_too_small" is set to "false", or
  // - "pass_through_empty_handle" is set to "true" and there are 0 payload ops.
  if (numPayloads < getNumResults() && getFailOnPayloadTooSmall() &&
      (numPayloads != 0 || !getPassThroughEmptyHandle()))
    return produceNumOpsError();

  // Distribute payloads.
  SmallVector<SmallVector<MappedValue, 1>> resultHandles(getNumResults(), {});
  if (getOverflowResult())
    resultHandles[*getOverflowResult()].reserve(numPayloads - getNumResults());

  auto container = [&]() {
    if (isa<TransformHandleTypeInterface>(getHandle().getType())) {
      return llvm::map_to_vector(
          state.getPayloadOps(getHandle()),
          [](Operation *op) -> MappedValue { return op; });
    }
    if (isa<TransformValueHandleTypeInterface>(getHandle().getType())) {
      return llvm::map_to_vector(state.getPayloadValues(getHandle()),
                                 [](Value v) -> MappedValue { return v; });
    }
    assert(isa<TransformParamTypeInterface>(getHandle().getType()) &&
           "unsupported kind of transform dialect type");
    return llvm::map_to_vector(state.getParams(getHandle()),
                               [](Attribute a) -> MappedValue { return a; });
  }();

  for (auto &&en : llvm::enumerate(container)) {
    int64_t resultNum = en.index();
    if (resultNum >= getNumResults())
      resultNum = *getOverflowResult();
    resultHandles[resultNum].push_back(en.value());
  }

  // Set transform op results.
  for (auto &&it : llvm::enumerate(resultHandles))
    results.setMappedValues(llvm::cast<OpResult>(getResult(it.index())),
                            it.value());

  return DiagnosedSilenceableFailure::success();
}

void transform::SplitHandleOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getHandleMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  // There are no effects on the Payload IR as this is only a handle
  // manipulation.
}

LogicalResult transform::SplitHandleOp::verify() {
  if (getOverflowResult().has_value() &&
      !(*getOverflowResult() < getNumResults()))
    return emitOpError("overflow_result is not a valid result index");

  for (Type resultType : getResultTypes()) {
    if (implementSameTransformInterface(getHandle().getType(), resultType))
      continue;

    return emitOpError("expects result types to implement the same transform "
                       "interface as the operand type");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ReplicateOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::ReplicateOp::apply(transform::TransformRewriter &rewriter,
                              transform::TransformResults &results,
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
  onlyReadsHandle(getPatternMutable(), effects);
  onlyReadsHandle(getHandlesMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
}

//===----------------------------------------------------------------------===//
// SequenceOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::SequenceOp::apply(transform::TransformRewriter &rewriter,
                             transform::TransformResults &results,
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
  if (hasExtras)
    printer << ", " << llvm::interleaved(extraBindingTypes) << ')';
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

  if (!getBodyBlock()->mightHaveTerminator())
    return emitOpError() << "expects to have a terminator in the body";

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

OperandRange
transform::SequenceOp::getEntrySuccessorOperands(RegionBranchPoint point) {
  assert(point == getBody() && "unexpected region index");
  if (getOperation()->getNumOperands() > 0)
    return getOperation()->getOperands();
  return OperandRange(getOperation()->operand_end(),
                      getOperation()->operand_end());
}

void transform::SequenceOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  if (point.isParent()) {
    Region *bodyRegion = &getBody();
    regions.emplace_back(bodyRegion, getNumOperands() != 0
                                         ? bodyRegion->getArguments()
                                         : Block::BlockArgListType());
    return;
  }

  assert(point == getBody() && "unexpected region index");
  regions.emplace_back(getOperation()->getResults());
}

void transform::SequenceOp::getRegionInvocationBounds(
    ArrayRef<Attribute> operands, SmallVectorImpl<InvocationBounds> &bounds) {
  (void)operands;
  bounds.emplace_back(1, 1);
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
  if (!name.empty())
    result.getOrAddProperties<Properties>().name = builder.getStringAttr(name);
}

void transform::PrintOp::build(OpBuilder &builder, OperationState &result,
                               Value target, StringRef name) {
  result.addOperands({target});
  build(builder, result, name);
}

DiagnosedSilenceableFailure
transform::PrintOp::apply(transform::TransformRewriter &rewriter,
                          transform::TransformResults &results,
                          transform::TransformState &state) {
  llvm::outs() << "[[[ IR printer: ";
  if (getName().has_value())
    llvm::outs() << *getName() << " ";

  OpPrintingFlags printFlags;
  if (getAssumeVerified().value_or(false))
    printFlags.assumeVerified();
  if (getUseLocalScope().value_or(false))
    printFlags.useLocalScope();
  if (getSkipRegions().value_or(false))
    printFlags.skipRegions();

  if (!getTarget()) {
    llvm::outs() << "top-level ]]]\n";
    state.getTopLevel()->print(llvm::outs(), printFlags);
    llvm::outs() << "\n";
    llvm::outs().flush();
    return DiagnosedSilenceableFailure::success();
  }

  llvm::outs() << "]]]\n";
  for (Operation *target : state.getPayloadOps(getTarget())) {
    target->print(llvm::outs(), printFlags);
    llvm::outs() << "\n";
  }

  llvm::outs().flush();
  return DiagnosedSilenceableFailure::success();
}

void transform::PrintOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  // We don't really care about mutability here, but `getTarget` now
  // unconditionally casts to a specific type before verification could run
  // here.
  if (!getTargetMutable().empty())
    onlyReadsHandle(getTargetMutable()[0], effects);
  onlyReadsPayload(effects);

  // There is no resource for stderr file descriptor, so just declare print
  // writes into the default resource.
  effects.emplace_back(MemoryEffects::Write::get());
}

//===----------------------------------------------------------------------===//
// VerifyOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::VerifyOp::applyToOne(transform::TransformRewriter &rewriter,
                                Operation *target,
                                transform::ApplyToEachResultList &results,
                                transform::TransformState &state) {
  if (failed(::mlir::verify(target))) {
    DiagnosedDefiniteFailure diag = emitDefiniteFailure()
                                    << "failed to verify payload op";
    diag.attachNote(target->getLoc()) << "payload op";
    return diag;
  }
  return DiagnosedSilenceableFailure::success();
}

void transform::VerifyOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTargetMutable(), effects);
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

void transform::YieldOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getOperandsMutable(), effects);
}
