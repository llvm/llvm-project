//===- ModuleBufferization.cpp - Bufferization across Func. Boundaries ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Module Bufferization is an extension of Comprehensive Bufferize that
// bufferizes function boundaries. It provides `BufferizableOpInterface`
// implementations for FuncOp, CallOp and ReturnOp.
//
// Module Bufferization is run via `runModuleBufferize(ModuleOp, ...)`. This
// function analyzes the given module and determines the order of analysis and
// bufferization: Functions that are called are processed before their
// respective callers.
//
// After analyzing a FuncOp, additional information about its bbArgs is
// gathered through PostAnalysisStepFns and stored in
// `ModuleAnalysisState`.
//
// * `equivalentFuncOpBBArgsAnalysis` determines the equivalent bbArg for each
//   tensor return value (if any).
// * `funcOpBbArgReadWriteAnalysis` determines whether or not a tensor bbArg is
//   read/written.
//
// Only tensors that are equivalent to some FuncOp bbArg may be returned.
// Bufferization currently fails if other tensors (in particular tensors that
// bufferize out-of-place and result in a new buffer allocation) are returned.
// In the future, such allocations could be hoisted to the caller.
//
// Example: `foo` fails bufferization because %0 is not equivalent to any bbArg.
// ```
// func @foo() -> tensor<?xf32> {
//   %0 = linalg.init_tensor [...] : tensor<?xf32>
//   return %0 : tensor<?xf32>
// }
// ```
//
// Module Bufferization implements the following calling convention.
//
// * In the absence of conflicts within a FuncOp, the FuncOp's bbArgs may always
//   be written to in-place.
// * If a tensor operand of a CallOp is read after the CallOp, the operand of
//   the CallOp must bufferize out-of-place.
//
// Example: The tensor.insert op bufferizes in-place because it is allowed to
// modify the buffer of `%t1` directly. The CallOp in `caller` must bufferize
// out-of-place because `%t0` is modified by the callee but read by the
// tensor.extract op. The analysis of CallOps decides whether an OpOperand must
// bufferize out-of-place based on results of `funcOpBbArgReadWriteAnalysis`.
// ```
// func @callee(%t1 : tensor<?xf32>) -> tensor<?xf32> {
//   %f = ... : f32
//   %0 = tensor.insert %f into %t1[...] : tensor<?xf32>
//   return %0 : tensor<?xf32>
// }
//
// func @caller() -> () {
//   %t0 = ... : tensor<?xf32>
//   %1 = call @callee(%t0) : (tensor<?xf32>) -> (tensor<?xf32>)
//   %2 = tensor.extract %1[...]  : tensor<?xf32>
// }
// ```
//
// Note: If a function is external, `funcOpBbArgReadWriteAnalysis` cannot
// analyze the function body. In such a case, the CallOp analysis conservatively
// assumes that each tensor OpOperand is both read and written.
//
// TODO: Add FuncOp attributes so that bbArgs of external FuncOps can be marked
// as "not reading" and/or "not writing".

#include "mlir/Dialect/Linalg/ComprehensiveBufferize/ModuleBufferization.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace linalg;
using namespace tensor;
using namespace comprehensive_bufferize;
using namespace mlir::bufferization;

namespace {
/// The state of analysis of a FuncOp.
enum class FuncOpAnalysisState { NotAnalyzed, InProgress, Analyzed };

/// Extra analysis state that is required for bufferization of function
/// boundaries.
struct ModuleAnalysisState : public DialectAnalysisState {
  /// A set of block argument indices.
  using BbArgIndexSet = DenseSet<int64_t>;

  /// A mapping of indices to indices.
  using IndexMapping = DenseMap<int64_t, int64_t>;

  /// A mapping of ReturnOp OpOperand indices to equivalent FuncOp BBArg
  /// indices.
  DenseMap<FuncOp, IndexMapping> equivalentFuncArgs;

  /// A set of all read BlockArguments of FuncOps.
  DenseMap<FuncOp, BbArgIndexSet> readBbArgs;

  /// A set of all written-to BlockArguments of FuncOps.
  DenseMap<FuncOp, BbArgIndexSet> writtenBbArgs;

  /// Keep track of which FuncOps are fully analyzed or currently being
  /// analyzed.
  DenseMap<FuncOp, FuncOpAnalysisState> analyzedFuncOps;

  /// A list of functions in the order in which they are analyzed + bufferized.
  SmallVector<FuncOp> orderedFuncOps;

  /// A mapping of FuncOps to their callers.
  DenseMap<FuncOp, DenseSet<Operation *>> callerMap;

  /// This function is called right before analyzing the given FuncOp. It
  /// initializes the data structures for the FuncOp in this state object.
  void startFunctionAnalysis(FuncOp funcOp) {
    analyzedFuncOps[funcOp] = FuncOpAnalysisState::InProgress;
    auto createdEquiv = equivalentFuncArgs.try_emplace(funcOp, IndexMapping());
    auto createdRead = readBbArgs.try_emplace(funcOp, BbArgIndexSet());
    auto createdWritten = writtenBbArgs.try_emplace(funcOp, BbArgIndexSet());
    (void)createdEquiv;
    (void)createdRead;
    (void)createdWritten;
#ifndef NDEBUG
    assert(createdEquiv.second && "equivalence info exists already");
    assert(createdRead.second && "bbarg access info exists already");
    assert(createdWritten.second && "bbarg access info exists already");
#endif // NDEBUG
  }
};
} // namespace

/// Get ModuleAnalysisState.
static const ModuleAnalysisState &
getModuleAnalysisState(const AnalysisState &state) {
  Optional<const ModuleAnalysisState *> maybeState =
      state.getDialectState<ModuleAnalysisState>(
          func::FuncDialect::getDialectNamespace());
  assert(maybeState.hasValue() && "ModuleAnalysisState does not exist");
  return **maybeState;
}

/// Get or create ModuleAnalysisState.
static ModuleAnalysisState &getModuleAnalysisState(AnalysisState &state) {
  return state.getOrCreateDialectState<ModuleAnalysisState>(
      func::FuncDialect::getDialectNamespace());
}

/// Return the state (phase) of analysis of the FuncOp.
static FuncOpAnalysisState getFuncOpAnalysisState(const AnalysisState &state,
                                                  FuncOp funcOp) {
  const ModuleAnalysisState &moduleState = getModuleAnalysisState(state);
  auto it = moduleState.analyzedFuncOps.find(funcOp);
  if (it == moduleState.analyzedFuncOps.end())
    return FuncOpAnalysisState::NotAnalyzed;
  return it->second;
}

/// Return the unique ReturnOp that terminates `funcOp`.
/// Return nullptr if there is no such unique ReturnOp.
static func::ReturnOp getAssumedUniqueReturnOp(FuncOp funcOp) {
  func::ReturnOp returnOp;
  for (Block &b : funcOp.getBody()) {
    if (auto candidateOp = dyn_cast<func::ReturnOp>(b.getTerminator())) {
      if (returnOp)
        return nullptr;
      returnOp = candidateOp;
    }
  }
  return returnOp;
}

namespace {

/// Annotate IR with the results of the analysis. For testing purposes only.
static void annotateEquivalentReturnBbArg(OpOperand &returnVal,
                                          BlockArgument bbArg) {
  const char *kEquivalentArgsAttr = "__equivalent_func_args__";
  Operation *op = returnVal.getOwner();

  SmallVector<int64_t> equivBbArgs;
  if (op->hasAttr(kEquivalentArgsAttr)) {
    auto attr = op->getAttr(kEquivalentArgsAttr).cast<ArrayAttr>();
    equivBbArgs = llvm::to_vector<4>(llvm::map_range(attr, [](Attribute a) {
      return a.cast<IntegerAttr>().getValue().getSExtValue();
    }));
  } else {
    equivBbArgs.append(op->getNumOperands(), -1);
  }
  equivBbArgs[returnVal.getOperandNumber()] = bbArg.getArgNumber();

  OpBuilder b(op->getContext());
  op->setAttr(kEquivalentArgsAttr, b.getI64ArrayAttr(equivBbArgs));
}

/// Store function BlockArguments that are equivalent to a returned value in
/// ModuleAnalysisState.
static LogicalResult
equivalentFuncOpBBArgsAnalysis(Operation *op, AnalysisState &state,
                               BufferizationAliasInfo &aliasInfo,
                               SmallVector<Operation *> &newOps) {
  ModuleAnalysisState &moduleState = getModuleAnalysisState(state);

  // Support only single return-terminated block in the function.
  auto funcOp = cast<FuncOp>(op);
  func::ReturnOp returnOp = getAssumedUniqueReturnOp(funcOp);
  assert(returnOp && "expected func with single return op");

  for (OpOperand &returnVal : returnOp->getOpOperands())
    if (returnVal.get().getType().isa<RankedTensorType>())
      for (BlockArgument bbArg : funcOp.getArguments())
        if (bbArg.getType().isa<RankedTensorType>())
          if (aliasInfo.areEquivalentBufferizedValues(returnVal.get(), bbArg)) {
            moduleState
                .equivalentFuncArgs[funcOp][returnVal.getOperandNumber()] =
                bbArg.getArgNumber();
            if (state.getOptions().testAnalysisOnly)
              annotateEquivalentReturnBbArg(returnVal, bbArg);
          }

  return success();
}

/// Return true if the buffer of the given tensor value is written to. Must not
/// be called for values inside not yet analyzed functions. (Post-analysis
/// steps do not have to be run yet, i.e., "in progress" is also OK.)
static bool isValueWritten(Value value, const AnalysisState &state,
                           const BufferizationAliasInfo &aliasInfo) {
#ifndef NDEBUG
  assert(value.getType().isa<TensorType>() && "expected TensorType");
  FuncOp funcOp;
  if (auto bbArg = value.dyn_cast<BlockArgument>()) {
    Operation *owner = bbArg.getOwner()->getParentOp();
    funcOp = isa<FuncOp>(owner) ? cast<FuncOp>(owner)
                                : owner->getParentOfType<FuncOp>();
  } else {
    funcOp = value.getDefiningOp()->getParentOfType<FuncOp>();
  }
  assert(getFuncOpAnalysisState(state, funcOp) !=
             FuncOpAnalysisState::NotAnalyzed &&
         "FuncOp must be fully analyzed or analysis in progress");
#endif // NDEBUG

  bool isWritten = false;
  aliasInfo.applyOnAliases(value, [&](Value val) {
    for (OpOperand &use : val.getUses())
      if (state.isInPlace(use) && state.bufferizesToMemoryWrite(use))
        isWritten = true;
  });
  return isWritten;
}

static void annotateFuncArgAccess(FuncOp funcOp, BlockArgument bbArg,
                                  bool isRead, bool isWritten) {
  OpBuilder b(funcOp.getContext());
  Attribute accessType;
  if (isRead && isWritten) {
    accessType = b.getStringAttr("read-write");
  } else if (isRead) {
    accessType = b.getStringAttr("read");
  } else if (isWritten) {
    accessType = b.getStringAttr("write");
  } else {
    accessType = b.getStringAttr("none");
  }
  funcOp.setArgAttr(bbArg.getArgNumber(), "bufferization.access", accessType);
}

/// Determine which FuncOp bbArgs are read and which are written. If this
/// PostAnalysisStepFn is run on a function with unknown ops, it will
/// conservatively assume that such ops bufferize to a read + write.
static LogicalResult
funcOpBbArgReadWriteAnalysis(Operation *op, AnalysisState &state,
                             BufferizationAliasInfo &aliasInfo,
                             SmallVector<Operation *> &newOps) {
  ModuleAnalysisState &moduleState = getModuleAnalysisState(state);
  auto funcOp = cast<FuncOp>(op);

  // If the function has no body, conservatively assume that all args are
  // read + written.
  if (funcOp.getBody().empty()) {
    for (BlockArgument bbArg : funcOp.getArguments()) {
      moduleState.readBbArgs[funcOp].insert(bbArg.getArgNumber());
      moduleState.writtenBbArgs[funcOp].insert(bbArg.getArgNumber());
    }

    return success();
  }

  for (BlockArgument bbArg : funcOp.getArguments()) {
    if (!bbArg.getType().isa<TensorType>())
      continue;
    bool isRead = state.isValueRead(bbArg);
    bool isWritten = isValueWritten(bbArg, state, aliasInfo);
    if (state.getOptions().testAnalysisOnly)
      annotateFuncArgAccess(funcOp, bbArg, isRead, isWritten);
    if (isRead)
      moduleState.readBbArgs[funcOp].insert(bbArg.getArgNumber());
    if (isWritten)
      moduleState.writtenBbArgs[funcOp].insert(bbArg.getArgNumber());
  }

  return success();
}
} // namespace

static bool isaTensor(Type t) { return t.isa<TensorType>(); }

/// If `value` is a memref::CastOp, return its source. Otherwise, return
/// `value` directly.
static Value getNonCastedValue(Value value) {
  while (auto castOp = value.getDefiningOp<memref::CastOp>())
    value = castOp.source();
  return value;
}

/// Remove the attribute that triggers inplace bufferization on a FuncOp
/// argument `bbArg`.
static void removeBufferizationFuncArguments(BlockArgument bbArg) {
  auto funcOp = cast<FuncOp>(bbArg.getOwner()->getParentOp());
  funcOp.removeArgAttr(bbArg.getArgNumber(),
                       BufferizableOpInterface::kBufferLayoutAttrName);
  funcOp.removeArgAttr(bbArg.getArgNumber(),
                       BufferizableOpInterface::kInplaceableAttrName);
}

/// Return the FuncOp called by `callOp`.
static FuncOp getCalledFunction(CallOpInterface callOp) {
  SymbolRefAttr sym = callOp.getCallableForCallee().dyn_cast<SymbolRefAttr>();
  if (!sym)
    return nullptr;
  return dyn_cast_or_null<FuncOp>(
      SymbolTable::lookupNearestSymbolFrom(callOp, sym));
}

/// Return the FunctionType with `argumentTypes` and `resultTypes` where each
/// tensor is replaced by the corresponding buffer type.
/// In order for all the callers to agree, this *must* bufferize to the most
/// dynamic buffer type supported.
/// A later pass across all CallOps in the module can decide whether to simplify
/// the types of to version according to some cost model.
static FunctionType
getBufferizedFunctionType(MLIRContext *ctx, TypeRange argumentTypes,
                          TypeRange resultTypes,
                          const BufferizationOptions &options) {
  auto rewrite = [&](Type t) -> Type {
    // TODO: non-zero address space.
    // TODO: layout information if relevant.
    if (auto tensorType = t.dyn_cast<TensorType>())
      return getMemRefType(tensorType, options);
    return t;
  };
  auto argTypes = llvm::to_vector<4>(llvm::map_range(argumentTypes, rewrite));
  auto retTypes = llvm::to_vector<4>(llvm::map_range(resultTypes, rewrite));
  return FunctionType::get(ctx, argTypes, retTypes);
}

/// Gather equivalence info of CallOps.
/// Note: This only adds new equivalence info if `funcOp` was already analyzed.
// TODO: This does not handle cyclic function call graphs etc.
static void equivalenceAnalysis(FuncOp funcOp,
                                BufferizationAliasInfo &aliasInfo,
                                ModuleAnalysisState &moduleState) {
  funcOp->walk([&](func::CallOp callOp) {
    FuncOp calledFunction = getCalledFunction(callOp);
    assert(calledFunction && "could not retrieved called FuncOp");

    // No equivalence info available for the called function.
    if (!moduleState.equivalentFuncArgs.count(calledFunction))
      return WalkResult::skip();

    for (auto it : moduleState.equivalentFuncArgs[calledFunction]) {
      int64_t returnIdx = it.first;
      int64_t bbargIdx = it.second;
      Value returnVal = callOp.getResult(returnIdx);
      Value argVal = callOp->getOperand(bbargIdx);
      aliasInfo.unionEquivalenceClasses(returnVal, argVal);
    }

    return WalkResult::advance();
  });
}

/// Rewrite the `funcOp` arguments analysis return values and terminator into
/// buffer form (using the canonical memref layout for now), according to the
/// inPlace-bufferizable information of the function arguments.
///
/// This relies on a buffer equivalence analysis of each return operand. When a
/// result buffer is equivalent to a BlockArgument of `funcOp`, it can be
/// dropped from the return values and becomes inplaceable at all callers. This
/// assumes all CallOp perform the necessary work to clone operands so as to
/// make them inplaceable. Reliance on this logic will need to be relaxed in the
/// future.
///
/// Note: Returning a memref currently fails bufferization. If such memrefs
/// originate from an op with an Alloc effect, they could be hoisted in the
/// future.
static LogicalResult bufferizeFuncOpBoundary(FuncOp funcOp,
                                             RewriterBase &rewriter,
                                             BufferizationState &state) {
  const ModuleAnalysisState &moduleState =
      getModuleAnalysisState(state.getAnalysisState());

  // If nothing to do then we are done.
  if (!llvm::any_of(funcOp.getFunctionType().getInputs(), isaTensor) &&
      !llvm::any_of(funcOp.getFunctionType().getResults(), isaTensor))
    return success();

  // Get the bufferized FunctionType for funcOp or construct it if not yet
  // available.
  // TODO: Atm we have 3 cases:
  // 1. if a function is called from within the Module, it must have bufferized
  //    to inplaceable tensor results.
  // 2. if it is bodiless, it must have bufferized and is not allowed to have
  //    result tensors.
  // 3. if it is not called internally, it still must bufferize to inplaceable
  //    tensor results and we construct it now (e.g. top-level function called
  //    externally).
  // -> Figure out a better layering.
  TypeRange resultTypes;

  // Corner case: Bodiless FuncOp
  // ============================
  // The body of such functions is assumed opaque and we can't know the
  // bufferization contract they want to enforce atm.
  // As a consequence, only support functions that don't return any tensor atm.
  if (funcOp.getBody().empty()) {
    if (llvm::any_of(funcOp.getFunctionType().getResults(), isaTensor))
      return funcOp->emitError() << "cannot bufferize bodiless function that "
                                 << "returns a tensor";
    FunctionType bufferizedFuncType = getBufferizedFunctionType(
        funcOp.getContext(), funcOp.getFunctionType().getInputs(),
        funcOp.getFunctionType().getResults(), state.getOptions());
    funcOp.setType(bufferizedFuncType);
    return success();
  }

  // Support only single return-terminated block in the function.
  func::ReturnOp returnOp = getAssumedUniqueReturnOp(funcOp);
  assert(returnOp && "expected func with single return op");

  // 1. For each FuncOp result, keep track of which inplace argument it reuses.
  SmallVector<Value> returnValues;
  for (OpOperand &returnOperand : returnOp->getOpOperands()) {
    Value returnVal = returnOperand.get();

    // If not a renturn tensor type just forward it.
    if (!returnVal.getType().isa<RankedTensorType>()) {
      returnValues.push_back(returnVal);
      continue;
    }

    // If return operand is equivalent to some bbArg, no need to return it.
    auto funcOpIt = moduleState.equivalentFuncArgs.find(funcOp);
    if (funcOpIt != moduleState.equivalentFuncArgs.end() &&
        funcOpIt->second.count(returnOperand.getOperandNumber()))
      continue;

    // Cast values at the call site if necessary.
    returnValues.push_back(
        getNonCastedValue(*state.getBuffer(rewriter, returnOperand)));
  }

  // 2. Rewrite the terminator without the inPlace bufferizable values.
  ValueRange retValues{returnValues};
  FunctionType bufferizedFuncType = getBufferizedFunctionType(
      funcOp.getContext(), funcOp.getFunctionType().getInputs(),
      retValues.getTypes(), state.getOptions());
  OpBuilder b(returnOp);
  b.create<func::ReturnOp>(returnOp.getLoc(), returnValues);
  returnOp->erase();

  // 3. Rewrite the bbArgs.
  // Iterate on the original `numArgs` and replace them in order.
  // This guarantees the argument order still matches after the rewrite.
  Block &frontBlock = funcOp.getBody().front();
  unsigned numArgs = frontBlock.getNumArguments();
  for (unsigned idx = 0; idx < numArgs; ++idx) {
    auto bbArg = frontBlock.getArgument(0);
    auto tensorType = bbArg.getType().dyn_cast<TensorType>();
    // Non-tensor types are just forwarded.
    if (!tensorType) {
      frontBlock.addArgument(bbArg.getType(), bbArg.getLoc());
      bbArg.replaceAllUsesWith(frontBlock.getArguments().back());
      frontBlock.eraseArgument(0);
      continue;
    }

    // Get the buffer type from the bufferized function type.
    Type memrefType = bufferizedFuncType.getInput(idx);
    Value memref = frontBlock.addArgument(memrefType, bbArg.getLoc());
    OpBuilder b(funcOp->getContext());
    b.setInsertionPointToStart(&frontBlock);
    // Replace all uses of bbArg through a ToMemRefOp.
    for (auto &use : llvm::make_early_inc_range(bbArg.getUses())) {
      if (auto toMemrefOp =
              dyn_cast<bufferization::ToMemrefOp>(use.getOwner())) {
        if (memref.getType() != toMemrefOp.memref().getType()) {
          // Type has changed, insert a cast.
          assert(memref::CastOp::areCastCompatible(
                     memref.getType(), toMemrefOp.memref().getType()) &&
                 "bufferizeFuncOpBoundary: cast incompatible");
          auto castOp = b.create<memref::CastOp>(
              funcOp.getLoc(), toMemrefOp.memref().getType(), memref);
          toMemrefOp.memref().replaceAllUsesWith(castOp);
        } else {
          // Type did not change, replace directly.
          toMemrefOp.memref().replaceAllUsesWith(memref);
        }
      }
    }
    // Replace all remaining uses by a to_tensor.
    if (!bbArg.use_empty()) {
      auto toTensorOp =
          b.create<bufferization::ToTensorOp>(funcOp.getLoc(), memref);
      bbArg.replaceAllUsesWith(toTensorOp);
    }
    frontBlock.eraseArgument(0);
    // TODO: add support to erase aliasInfo entries if deemed necessary.
  }

  // 4. Rewrite the FuncOp type to buffer form.
  funcOp.setType(bufferizedFuncType);

  return success();
}

/// Store all functions of the `moduleOp` in `orderedFuncOps`, sorted by
/// callee-caller order (i.e. callees without callers first).
/// Store the map of FuncOp to all its callers in `callerMap`.
/// Return `failure()` if a cycle of calls is detected or if we are unable to
/// retrieve the called FuncOp from any CallOpInterface.
static LogicalResult
getFuncOpsOrderedByCalls(ModuleOp moduleOp,
                         SmallVectorImpl<FuncOp> &orderedFuncOps,
                         DenseMap<FuncOp, DenseSet<Operation *>> &callerMap) {
  // For each FuncOp, the set of functions called by it (i.e. the union of
  // symbols of all nested CallOpInterfaceOp).
  DenseMap<FuncOp, DenseSet<FuncOp>> calledBy;
  // For each FuncOp, the number of CallOpInterface it contains.
  DenseMap<FuncOp, unsigned> numberCallOpsContainedInFuncOp;
  WalkResult res = moduleOp.walk([&](FuncOp funcOp) -> WalkResult {
    if (!funcOp.getBody().empty()) {
      func::ReturnOp returnOp = getAssumedUniqueReturnOp(funcOp);
      if (!returnOp)
        return funcOp->emitError()
               << "cannot bufferize a FuncOp with tensors and "
                  "without a unique ReturnOp";
    }

    numberCallOpsContainedInFuncOp[funcOp] = 0;
    return funcOp.walk([&](CallOpInterface callOp) -> WalkResult {
      // Only support CallOp for now.
      if (!isa<func::CallOp>(callOp.getOperation()))
        return callOp->emitError() << "expected a CallOp";
      FuncOp calledFunction = getCalledFunction(callOp);
      assert(calledFunction && "could not retrieved called FuncOp");
      auto it = callerMap.try_emplace(calledFunction, DenseSet<Operation *>{});
      it.first->getSecond().insert(callOp);
      if (calledBy[calledFunction].count(funcOp) == 0) {
        calledBy[calledFunction].insert(funcOp);
        numberCallOpsContainedInFuncOp[funcOp]++;
      }
      return WalkResult::advance();
    });
  });
  if (res.wasInterrupted())
    return failure();
  // Iteratively remove function operation that do not call any of the
  // functions remaining in the callCounter map and add them to the worklist.
  while (!numberCallOpsContainedInFuncOp.empty()) {
    auto it = llvm::find_if(numberCallOpsContainedInFuncOp,
                            [](auto entry) { return entry.getSecond() == 0; });
    if (it == numberCallOpsContainedInFuncOp.end())
      return moduleOp.emitOpError(
          "expected callgraph to be free of circular dependencies.");
    orderedFuncOps.push_back(it->getFirst());
    for (auto callee : calledBy[it->getFirst()])
      numberCallOpsContainedInFuncOp[callee]--;
    numberCallOpsContainedInFuncOp.erase(it);
  }
  return success();
}

static void
foreachCaller(const DenseMap<FuncOp, DenseSet<Operation *>> &callerMap,
              FuncOp callee, llvm::function_ref<void(Operation *)> doit) {
  auto itCallers = callerMap.find(callee);
  if (itCallers == callerMap.end())
    return;
  for (Operation *caller : itCallers->second)
    doit(caller);
}

/// Postprocess the linalg.buffer_layout annotation across function boundaries.
/// This is a purely mechanical process that may later become part of a
/// separate pass with its own layout assignment heuristic.
static void layoutPostProcessing(ModuleOp moduleOp) {
  SmallVector<FuncOp> orderedFuncOps;
  DenseMap<FuncOp, DenseSet<Operation *>> callerMap;
  auto res = getFuncOpsOrderedByCalls(moduleOp, orderedFuncOps, callerMap);
  (void)res;
  assert(succeeded(res) && "unexpected getFuncOpsOrderedByCalls failure");

  for (FuncOp funcOp : orderedFuncOps) {
    DenseMap<Operation *, SmallVector<Value>> operandsPerCaller;
    foreachCaller(callerMap, funcOp, [&](Operation *caller) {
      operandsPerCaller.try_emplace(caller, SmallVector<Value>());
    });

    SmallVector<Type> argumentTypes;
    // Iterate on each function argument and check it it was marked with a
    // desired layout.
    for (const auto &it :
         llvm::enumerate(funcOp.getFunctionType().getInputs())) {
      int argNumber = it.index();
      Type inputType = it.value();
      auto memrefType = inputType.dyn_cast<MemRefType>();
      auto layoutAttr = funcOp.getArgAttrOfType<AffineMapAttr>(
          argNumber, BufferizableOpInterface::kBufferLayoutAttrName);
      AffineMap desiredLayoutMap =
          layoutAttr ? layoutAttr.getValue() : AffineMap();
      AffineMap currentLayoutMap =
          memrefType ? getStridedLinearLayoutMap(memrefType) : AffineMap();
      if (!memrefType || !layoutAttr || desiredLayoutMap == currentLayoutMap) {
        argumentTypes.push_back(inputType);
        foreachCaller(callerMap, funcOp, [&](Operation *caller) {
          operandsPerCaller.find(caller)->getSecond().push_back(
              caller->getOperand(argNumber));
        });
        continue;
      }

      // Compute the buffer type with desired layout and add to input argument
      // types.
      MemRefType desiredMemrefType = MemRefType::get(
          memrefType.getShape(), memrefType.getElementType(), desiredLayoutMap);
      argumentTypes.push_back(desiredMemrefType);

      // If funcOp's body is not empty, change the bbArg type and propagate.
      if (!funcOp.getBody().empty()) {
        BlockArgument bbArg = funcOp.getArgument(argNumber);
        bbArg.setType(desiredMemrefType);
        OpBuilder b(bbArg.getContext());
        b.setInsertionPointToStart(bbArg.getOwner());
        assert(memref::CastOp::areCastCompatible(bbArg.getType(), memrefType) &&
               "layoutPostProcessing: cast incompatible");
        // Cast back to the original memrefType and let it canonicalize.
        Value cast =
            b.create<memref::CastOp>(funcOp.getLoc(), memrefType, bbArg);
        bbArg.replaceAllUsesExcept(cast, cast.getDefiningOp());
      }

      // Cast to desired buffer type on all callers to `funcOp`.
      // TODO: on the callee side, this may even have to trigger a copy to
      // change the layout. For now let the memref::CastOp fail to verify in
      // such cases.
      auto castArg = [&](Operation *caller) {
        OpBuilder b(caller);
        assert(
            memref::CastOp::areCastCompatible(
                caller->getOperand(argNumber).getType(), desiredMemrefType) &&
            "layoutPostProcessing.2: cast incompatible");
        Value newOperand = b.create<memref::CastOp>(
            funcOp.getLoc(), desiredMemrefType, caller->getOperand(argNumber));
        operandsPerCaller.find(caller)->getSecond().push_back(newOperand);
      };
      foreachCaller(callerMap, funcOp, castArg);
    }

    // Set operands with cast buffer on all callers to `funcOp`.
    foreachCaller(callerMap, funcOp, [&](Operation *caller) {
      caller->setOperands(operandsPerCaller.lookup(caller));
    });

    // Finally set the funcOp type to update the arguments.
    auto newFuncType = FunctionType::get(moduleOp.getContext(), argumentTypes,
                                         funcOp.getFunctionType().getResults());
    funcOp.setType(newFuncType);
  }
}

namespace mlir {
namespace linalg {
namespace comprehensive_bufferize {
namespace std_ext {

/// Return the index of the bbArg in the given FuncOp that is equivalent to the
/// specified return value (if any).
static Optional<int64_t>
getEquivalentFuncArgIdx(FuncOp funcOp, const ModuleAnalysisState &state,
                        int64_t returnValIdx) {
  auto funcOpIt = state.equivalentFuncArgs.find(funcOp);
  if (funcOpIt == state.equivalentFuncArgs.end())
    // No equivalence info stores for funcOp.
    return None;

  auto retValIt = funcOpIt->getSecond().find(returnValIdx);
  if (retValIt == funcOpIt->getSecond().end())
    // Return value has no equivalent bbArg.
    return None;

  return retValIt->getSecond();
}

struct CallOpInterface
    : public BufferizableOpInterface::ExternalModel<CallOpInterface,
                                                    func::CallOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    func::CallOp callOp = cast<func::CallOp>(op);
    FuncOp funcOp = getCalledFunction(callOp);
    assert(funcOp && "expected CallOp to a FuncOp");

    const ModuleAnalysisState &moduleState = getModuleAnalysisState(state);
    if (getFuncOpAnalysisState(state, funcOp) != FuncOpAnalysisState::Analyzed)
      // FuncOp not analyzed yet. Assume that OpOperand is read.
      return true;

    return moduleState.readBbArgs.lookup(funcOp).contains(
        opOperand.getOperandNumber());
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    func::CallOp callOp = cast<func::CallOp>(op);
    FuncOp funcOp = getCalledFunction(callOp);
    assert(funcOp && "expected CallOp to a FuncOp");

    const ModuleAnalysisState &moduleState = getModuleAnalysisState(state);
    if (getFuncOpAnalysisState(state, funcOp) != FuncOpAnalysisState::Analyzed)
      // FuncOp not analyzed yet. Assume that OpOperand is written.
      return true;

    return moduleState.writtenBbArgs.lookup(funcOp).contains(
        opOperand.getOperandNumber());
  }

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    func::CallOp callOp = cast<func::CallOp>(op);
    FuncOp funcOp = getCalledFunction(callOp);
    assert(funcOp && "expected CallOp to a FuncOp");
    const ModuleAnalysisState &moduleState = getModuleAnalysisState(state);

    SmallVector<OpResult> result;
    for (int64_t resultIdx = 0; resultIdx < callOp->getNumResults();
         ++resultIdx)
      if (Optional<int64_t> maybeArgNumber =
              getEquivalentFuncArgIdx(funcOp, moduleState, resultIdx))
        if (*maybeArgNumber == opOperand.getOperandNumber())
          result.push_back(callOp->getOpResult(resultIdx));

    return result;
  }

  SmallVector<OpOperand *>
  getAliasingOpOperand(Operation *op, OpResult opResult,
                       const AnalysisState &state) const {
    func::CallOp callOp = cast<func::CallOp>(op);
    FuncOp funcOp = getCalledFunction(callOp);
    assert(funcOp && "expected CallOp to a FuncOp");
    const ModuleAnalysisState &moduleState = getModuleAnalysisState(state);

    // TODO: We should be looking for aliasing block arguments here. The current
    // condition is actually stronger than neccesary. Once we check for aliasing
    // block arguments, we may be multiple.
    if (Optional<int64_t> maybeArgNumber = getEquivalentFuncArgIdx(
            funcOp, moduleState, opResult.getResultNumber()))
      return {&op->getOpOperand(*maybeArgNumber)};

    // Note: Returning a non-equivalent tensor from a FuncOp is currently not
    // supported an will fail bufferization.
    return {};
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const AnalysisState &state) const {
    return BufferRelation::Equivalent;
  }

  /// In a first approximation, all the function arguments of a FuncOp are
  /// marked inplaceable. For now, it is the responsibility of the `callOp`
  /// bufferization to allow FuncOp that are inplaceable to write inPlace.
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          BufferizationState &state) const {
    func::CallOp callOp = cast<func::CallOp>(op);
    unsigned numResults = callOp.getNumResults();
    unsigned numOperands = callOp->getNumOperands();
    FuncOp funcOp = getCalledFunction(callOp);
    assert(funcOp && "expected CallOp to a FuncOp");
    const ModuleAnalysisState &moduleState =
        getModuleAnalysisState(state.getAnalysisState());

    // Result types of the bufferized CallOp.
    SmallVector<Type> resultTypes;
    // Replacement values for the existing CallOp. These are usually the results
    // of the bufferized CallOp, unless a tensor result folds onto an operand.
    SmallVector<Value> replacementValues(numResults, Value());
    // For non-tensor results: A mapping from return val indices of the old
    // CallOp to return val indices of the bufferized CallOp.
    SmallVector<Optional<unsigned>> retValMapping(numResults, None);
    // Operands of the bufferized CallOp.
    SmallVector<Value> newOperands(numOperands, Value());

    // Based on previously gathered equivalence information, we know if a
    // tensor result folds onto an operand. These are the only tensor value
    // results that are supported at the moment.
    //
    // For tensors return values that do not fold onto an operand, additional
    // work is needed (TODO) to either:
    // * hoist a result into an inplaceable operand or
    // * devise a better representation to truly return a buffer.
    //
    // Note: If a function has no body, no equivalence information is
    // available. Consequently, a tensor return value cannot be proven to fold
    // onto a FuncOp bbArg, so calls to such functions are not bufferizable at
    // the moment.

    // 1. Compute the result types of the new CallOp. Tensor results that are
    // equivalent to a FuncOp bbArg are no longer returned.
    for (const auto &it : llvm::enumerate(callOp.getResultTypes())) {
      unsigned returnValIdx = it.index();
      Type returnType = it.value();
      if (!isaTensor(returnType)) {
        // Non-tensor values are returned.
        retValMapping[returnValIdx] = resultTypes.size();
        resultTypes.push_back(returnType);
        continue;
      }

      if (Optional<int64_t> bbArgIdx =
              getEquivalentFuncArgIdx(funcOp, moduleState, returnValIdx)) {
        // Return operands that are equivalent to some bbArg, are not
        // returned.
        FailureOr<Value> bufferOrFailure =
            state.getBuffer(rewriter, callOp->getOpOperand(*bbArgIdx));
        if (failed(bufferOrFailure))
          return failure();
        replacementValues[returnValIdx] = *bufferOrFailure;
        newOperands[*bbArgIdx] = *bufferOrFailure;
        continue;
      }

      return callOp->emitError(
          "call to FuncOp that returns non-equivalent tensors not supported");
    }

    // 2. Compute bufferized FunctionType.
    SmallVector<Type> argumentTypes{callOp->getOperandTypes()};
    // Get the bufferized FunctionType for funcOp or construct it if not yet
    // available.
    FunctionType bufferizedFuncType = getBufferizedFunctionType(
        funcOp.getContext(), argumentTypes, resultTypes, state.getOptions());

    // 3. Rewrite tensor operands as memrefs based on `bufferizedFuncType`.
    for (OpOperand &opOperand : callOp->getOpOperands()) {
      unsigned idx = opOperand.getOperandNumber();
      Value tensorOperand = opOperand.get();

      // Non-tensor operands are just copied.
      if (!tensorOperand.getType().isa<TensorType>()) {
        newOperands[idx] = tensorOperand;
        continue;
      }

      // Retrieve buffers for tensor operands. Tensor operand buffers, who's
      // corresponding FuncOp bbArgs are equivalent to a returned tensor, were
      // already stored in `newOperands` during Step 1.
      Value buffer = newOperands[idx];
      if (!buffer) {
        FailureOr<Value> bufferOrFailure = state.getBuffer(rewriter, opOperand);
        if (failed(bufferOrFailure))
          return failure();
        buffer = *bufferOrFailure;
      }

      // Caller / callee type mismatch is handled with a CastOp.
      auto memRefType = bufferizedFuncType.getInput(idx);
      // Since we don't yet have a clear layout story, to_memref may
      // conservatively turn tensors into more dynamic memref than necessary.
      // If the memref type of the callee fails, introduce an extra memref.cast
      // that will either canonicalize away or fail compilation until we can do
      // something better.
      if (buffer.getType() != memRefType) {
        assert(
            memref::CastOp::areCastCompatible(buffer.getType(), memRefType) &&
            "CallOp::bufferize: cast incompatible");
        Value castBuffer = rewriter.create<memref::CastOp>(callOp.getLoc(),
                                                           memRefType, buffer);
        buffer = castBuffer;
      }
      newOperands[idx] = buffer;
    }

    // 4. Create the new CallOp.
    Operation *newCallOp = rewriter.create<func::CallOp>(
        callOp.getLoc(), funcOp.getSymName(), resultTypes, newOperands);
    newCallOp->setAttrs(callOp->getAttrs());
    // Get replacement values for non-tensor / non-equivalent results.
    for (unsigned i = 0; i < replacementValues.size(); ++i) {
      if (replacementValues[i])
        continue;
      replacementValues[i] = newCallOp->getResult(*retValMapping[i]);
    }

    // 5. Replace the old op with the new op.
    replaceOpWithBufferizedValues(rewriter, callOp, replacementValues);

    return success();
  }
};

struct ReturnOpInterface
    : public BufferizableOpInterface::ExternalModel<ReturnOpInterface,
                                                    func::ReturnOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          BufferizationState &state) const {
#ifndef NDEBUG
    auto returnOp = cast<func::ReturnOp>(op);
    assert(isa<FuncOp>(returnOp->getParentOp()) &&
           "only support FuncOp parent for ReturnOp");
#endif // NDEBUG
    return failure();
  }
};

struct FuncOpInterface
    : public BufferizableOpInterface::ExternalModel<FuncOpInterface, FuncOp> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          BufferizationState &state) const {
    return failure();
  }

  /// Return `true` if the given function argument is writable.
  bool isWritable(Operation *op, Value value,
                  const AnalysisState &state) const {
    auto funcOp = cast<FuncOp>(op);
    BlockArgument bbArg = value.dyn_cast<BlockArgument>();
    assert(bbArg && "expected BlockArgument");

    // "linalg.inplaceable" overrides other writability decisions. This is
    // currently used for testing only.
    if (BoolAttr inplaceAttr = funcOp.getArgAttrOfType<BoolAttr>(
            bbArg.getArgNumber(),
            BufferizableOpInterface::kInplaceableAttrName))
      return inplaceAttr.getValue();

    // All function arguments are writable by default.
    return true;
  }

  bool isAllocationHoistingBarrier(Operation *op) const { return true; }
};

} // namespace std_ext
} // namespace comprehensive_bufferize
} // namespace linalg
} // namespace mlir

void mlir::linalg::comprehensive_bufferize::std_ext::
    registerModuleBufferizationExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, func::FuncDialect *dialect) {
    func::CallOp::attachInterface<std_ext::CallOpInterface>(*ctx);
    func::ReturnOp::attachInterface<std_ext::ReturnOpInterface>(*ctx);
    func::FuncOp::attachInterface<std_ext::FuncOpInterface>(*ctx);
  });
}

/// Set the attribute that triggers inplace bufferization on a FuncOp argument
/// `bbArg`.
static void setInPlaceFuncArgument(BlockArgument bbArg, bool inPlace) {
  auto funcOp = cast<FuncOp>(bbArg.getOwner()->getParentOp());
  funcOp.setArgAttr(bbArg.getArgNumber(),
                    BufferizableOpInterface::kInplaceableAttrName,
                    BoolAttr::get(bbArg.getContext(), inPlace));
}

/// Annotate the IR with the result of the analysis. For testing/debugging only.
static void annotateOpsWithBufferizationMarkers(FuncOp funcOp,
                                                const AnalysisState &state) {
  auto bufferizableOp = cast<BufferizableOpInterface>(funcOp.getOperation());
  for (BlockArgument bbArg : funcOp.getArguments())
    if (bbArg.getType().isa<TensorType>())
      setInPlaceFuncArgument(bbArg, bufferizableOp.isWritable(bbArg, state));
}

LogicalResult mlir::linalg::comprehensive_bufferize::runModuleBufferize(
    ModuleOp moduleOp, OneShotBufferizationOptions options) {
  IRRewriter rewriter(moduleOp.getContext());
  OneShotAnalysisState analysisState(moduleOp, options);
  BufferizationState bufferizationState(analysisState);
  ModuleAnalysisState &moduleState = getModuleAnalysisState(analysisState);
  BufferizationAliasInfo &aliasInfo = analysisState.getAliasInfo();

  if (failed(getFuncOpsOrderedByCalls(moduleOp, moduleState.orderedFuncOps,
                                      moduleState.callerMap)))
    return failure();

  // Collect bbArg/return value information after the analysis.
  options.addPostAnalysisStep(equivalentFuncOpBBArgsAnalysis);
  options.addPostAnalysisStep(funcOpBbArgReadWriteAnalysis);

  // Analyze ops.
  for (FuncOp funcOp : moduleState.orderedFuncOps) {
    // No body => no analysis.
    if (funcOp.getBody().empty())
      continue;

    // Now analyzing function.
    moduleState.startFunctionAnalysis(funcOp);

    // Gather equivalence info for CallOps.
    equivalenceAnalysis(funcOp, aliasInfo, moduleState);

    // Analyze funcOp.
    if (failed(analyzeOp(funcOp, analysisState)))
      return failure();

    // Mark op as fully analyzed.
    moduleState.analyzedFuncOps[funcOp] = FuncOpAnalysisState::Analyzed;

    // Add annotations to function arguments.
    if (options.testAnalysisOnly)
      annotateOpsWithBufferizationMarkers(funcOp, analysisState);
  }

  if (options.testAnalysisOnly)
    return success();

  // Bufferize functions.
  for (FuncOp funcOp : moduleState.orderedFuncOps) {
    // No body => no analysis.
    if (!funcOp.getBody().empty())
      if (failed(bufferizeOp(funcOp, bufferizationState)))
        return failure();

    // Note: It would be good to apply cleanups here but we cannot as aliasInfo
    // would be invalidated.
    if (failed(bufferizeFuncOpBoundary(funcOp, rewriter, bufferizationState)))
      return failure();
  }

  // Check result.
  for (FuncOp funcOp : moduleState.orderedFuncOps) {
    if (!options.allowReturnAllocs &&
        llvm::any_of(funcOp.getFunctionType().getResults(), [](Type t) {
          return t.isa<MemRefType, UnrankedMemRefType>();
        })) {
      funcOp->emitError("memref return type is unsupported");
      return failure();
    }
  }

  // Finalize all buffers.
  if (failed(finalizeBuffers(moduleOp, options)))
    return failure();

  // Perform a post-processing pass of layout modification at function boundary
  // according to the kBufferLayoutAttrName.
  layoutPostProcessing(moduleOp);

  // Post-pass cleanup of inplaceable and buffer_layout attributes.
  moduleOp.walk([&](FuncOp op) {
    for (BlockArgument bbArg : op.getArguments())
      removeBufferizationFuncArguments(bbArg);
  });

  return success();
}
