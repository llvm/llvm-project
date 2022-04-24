//===- ModuleBufferization.cpp - Bufferization across Func. Boundaries ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Module Bufferization is an extension of One-Shot Bufferize that
// bufferizes function boundaries. It provides `BufferizableOpInterface`
// implementations for FuncOp, CallOp and ReturnOp.
//
// Module Bufferization is run via `runModuleBufferize(ModuleOp, ...)`. This
// function analyzes the given module and determines the order of analysis and
// bufferization: Functions that are called are processed before their
// respective callers.
//
// After analyzing a FuncOp, additional information about its bbArgs is
// gathered through PostAnalysisStepFns and stored in `FuncAnalysisState`.
//
// * `aliasingFuncOpBBArgsAnalysis` determines the equivalent/aliasing bbArgs
// for
//   each tensor return value (if any).
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

/// A mapping of FuncOps to their callers.
using FuncCallerMap = DenseMap<func::FuncOp, DenseSet<Operation *>>;

namespace {
/// The state of analysis of a FuncOp.
enum class FuncOpAnalysisState { NotAnalyzed, InProgress, Analyzed };

/// Extra analysis state that is required for bufferization of function
/// boundaries.
struct FuncAnalysisState : public DialectAnalysisState {
  // Note: Function arguments and/or function return values may disappear during
  // bufferization. Functions and their CallOps are analyzed and bufferized
  // separately. To ensure that a CallOp analysis/bufferization can access an
  // already bufferized function's analysis results, we store bbArg/return value
  // indices instead of BlockArguments/OpOperand pointers.

  /// A set of block argument indices.
  using BbArgIndexSet = DenseSet<int64_t>;

  /// A mapping of indices to indices.
  using IndexMapping = DenseMap<int64_t, int64_t>;

  /// A mapping of indices to a list of indices.
  using IndexToIndexListMapping = DenseMap<int64_t, SmallVector<int64_t>>;

  /// A mapping of ReturnOp OpOperand indices to equivalent FuncOp BBArg
  /// indices.
  DenseMap<func::FuncOp, IndexMapping> equivalentFuncArgs;

  /// A mapping of ReturnOp OpOperand indices to aliasing FuncOp BBArg indices.
  DenseMap<func::FuncOp, IndexToIndexListMapping> aliasingFuncArgs;

  /// A mapping of FuncOp BBArg indices to aliasing ReturnOp OpOperand indices.
  DenseMap<func::FuncOp, IndexToIndexListMapping> aliasingReturnVals;

  /// A set of all read BlockArguments of FuncOps.
  DenseMap<func::FuncOp, BbArgIndexSet> readBbArgs;

  /// A set of all written-to BlockArguments of FuncOps.
  DenseMap<func::FuncOp, BbArgIndexSet> writtenBbArgs;

  /// Keep track of which FuncOps are fully analyzed or currently being
  /// analyzed.
  DenseMap<func::FuncOp, FuncOpAnalysisState> analyzedFuncOps;

  /// This function is called right before analyzing the given FuncOp. It
  /// initializes the data structures for the FuncOp in this state object.
  void startFunctionAnalysis(func::FuncOp funcOp) {
    analyzedFuncOps[funcOp] = FuncOpAnalysisState::InProgress;
    auto createdEquiv = equivalentFuncArgs.try_emplace(funcOp, IndexMapping());
    auto createdAliasingOperands =
        aliasingFuncArgs.try_emplace(funcOp, IndexToIndexListMapping());
    auto createdAliasingResults =
        aliasingReturnVals.try_emplace(funcOp, IndexToIndexListMapping());
    auto createdRead = readBbArgs.try_emplace(funcOp, BbArgIndexSet());
    auto createdWritten = writtenBbArgs.try_emplace(funcOp, BbArgIndexSet());
    (void)createdEquiv;
    (void)createdAliasingOperands;
    (void)createdAliasingResults;
    (void)createdRead;
    (void)createdWritten;
#ifndef NDEBUG
    assert(createdEquiv.second && "equivalence info exists already");
    assert(createdAliasingOperands.second && "aliasing info exists already");
    assert(createdAliasingResults.second && "aliasing info exists already");
    assert(createdRead.second && "bbarg access info exists already");
    assert(createdWritten.second && "bbarg access info exists already");
#endif // NDEBUG
  }
};
} // namespace

/// Get FuncAnalysisState.
static const FuncAnalysisState &
getFuncAnalysisState(const AnalysisState &state) {
  Optional<const FuncAnalysisState *> maybeState =
      state.getDialectState<FuncAnalysisState>(
          func::FuncDialect::getDialectNamespace());
  assert(maybeState.hasValue() && "FuncAnalysisState does not exist");
  return **maybeState;
}

/// Get or create FuncAnalysisState.
static FuncAnalysisState &getFuncAnalysisState(AnalysisState &state) {
  return state.getOrCreateDialectState<FuncAnalysisState>(
      func::FuncDialect::getDialectNamespace());
}

/// Return the state (phase) of analysis of the FuncOp.
static FuncOpAnalysisState getFuncOpAnalysisState(const AnalysisState &state,
                                                  func::FuncOp funcOp) {
  const FuncAnalysisState &moduleState = getFuncAnalysisState(state);
  auto it = moduleState.analyzedFuncOps.find(funcOp);
  if (it == moduleState.analyzedFuncOps.end())
    return FuncOpAnalysisState::NotAnalyzed;
  return it->second;
}

/// Return the unique ReturnOp that terminates `funcOp`.
/// Return nullptr if there is no such unique ReturnOp.
static func::ReturnOp getAssumedUniqueReturnOp(func::FuncOp funcOp) {
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

/// Store function BlockArguments that are equivalent to/aliasing a returned
/// value in FuncAnalysisState.
static LogicalResult
aliasingFuncOpBBArgsAnalysis(Operation *op, AnalysisState &state,
                             BufferizationAliasInfo &aliasInfo,
                             SmallVector<Operation *> &newOps) {
  FuncAnalysisState &funcState = getFuncAnalysisState(state);

  // Support only single return-terminated block in the function.
  auto funcOp = cast<func::FuncOp>(op);
  func::ReturnOp returnOp = getAssumedUniqueReturnOp(funcOp);
  assert(returnOp && "expected func with single return op");

  for (OpOperand &returnVal : returnOp->getOpOperands())
    if (returnVal.get().getType().isa<RankedTensorType>())
      for (BlockArgument bbArg : funcOp.getArguments())
        if (bbArg.getType().isa<RankedTensorType>()) {
          int64_t returnIdx = returnVal.getOperandNumber();
          int64_t bbArgIdx = bbArg.getArgNumber();
          if (aliasInfo.areEquivalentBufferizedValues(returnVal.get(), bbArg)) {
            funcState.equivalentFuncArgs[funcOp][returnIdx] = bbArgIdx;
            if (state.getOptions().testAnalysisOnly)
              annotateEquivalentReturnBbArg(returnVal, bbArg);
          }
          if (aliasInfo.areAliasingBufferizedValues(returnVal.get(), bbArg)) {
            funcState.aliasingFuncArgs[funcOp][returnIdx].push_back(bbArgIdx);
            funcState.aliasingReturnVals[funcOp][bbArgIdx].push_back(returnIdx);
          }
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
  func::FuncOp funcOp;
  if (auto bbArg = value.dyn_cast<BlockArgument>()) {
    Operation *owner = bbArg.getOwner()->getParentOp();
    funcOp = isa<func::FuncOp>(owner) ? cast<func::FuncOp>(owner)
                                      : owner->getParentOfType<func::FuncOp>();
  } else {
    funcOp = value.getDefiningOp()->getParentOfType<func::FuncOp>();
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

static void annotateFuncArgAccess(func::FuncOp funcOp, BlockArgument bbArg,
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
  FuncAnalysisState &funcState = getFuncAnalysisState(state);
  auto funcOp = cast<func::FuncOp>(op);

  // If the function has no body, conservatively assume that all args are
  // read + written.
  if (funcOp.getBody().empty()) {
    for (BlockArgument bbArg : funcOp.getArguments()) {
      funcState.readBbArgs[funcOp].insert(bbArg.getArgNumber());
      funcState.writtenBbArgs[funcOp].insert(bbArg.getArgNumber());
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
      funcState.readBbArgs[funcOp].insert(bbArg.getArgNumber());
    if (isWritten)
      funcState.writtenBbArgs[funcOp].insert(bbArg.getArgNumber());
  }

  return success();
}
} // namespace

/// Remove the attribute that triggers inplace bufferization on a func::FuncOp
/// argument `bbArg`.
static void removeBufferizationFuncArguments(BlockArgument bbArg) {
  auto funcOp = cast<func::FuncOp>(bbArg.getOwner()->getParentOp());
  funcOp.removeArgAttr(bbArg.getArgNumber(),
                       BufferizableOpInterface::kBufferLayoutAttrName);
  funcOp.removeArgAttr(bbArg.getArgNumber(),
                       BufferizableOpInterface::kInplaceableAttrName);
}

/// Return the func::FuncOp called by `callOp`.
static func::FuncOp getCalledFunction(CallOpInterface callOp) {
  SymbolRefAttr sym = callOp.getCallableForCallee().dyn_cast<SymbolRefAttr>();
  if (!sym)
    return nullptr;
  return dyn_cast_or_null<func::FuncOp>(
      SymbolTable::lookupNearestSymbolFrom(callOp, sym));
}

/// Return the index-th bufferized function argument type. This assumes that the
/// specified argument is a tensor. If the tensor is ranked, a layout map may be
/// specified by the user. If no layout map is specified, a fully dynamic map is
/// used.
static BaseMemRefType
getBufferizedFunctionArgType(func::FuncOp funcOp, int64_t index,
                             const BufferizationOptions &options) {
  auto tensorType =
      funcOp.getFunctionType().getInput(index).dyn_cast<TensorType>();
  assert(tensorType && "expected TensorType");
  BaseMemRefType memrefType = getMemRefType(tensorType, options);

  auto layoutAttr = funcOp.getArgAttrOfType<AffineMapAttr>(
      index, BufferizableOpInterface::kBufferLayoutAttrName);
  if (!layoutAttr)
    return memrefType;

  auto rankedMemrefType = memrefType.dyn_cast<MemRefType>();
  assert(rankedMemrefType && "buffer layout not supported on unranked tensors");
  return MemRefType::get(
      rankedMemrefType.getShape(), rankedMemrefType.getElementType(),
      layoutAttr.getValue(), rankedMemrefType.getMemorySpaceAsInt());
}

/// Gather equivalence info of CallOps.
/// Note: This only adds new equivalence info if the called function was already
/// analyzed.
// TODO: This does not handle cyclic function call graphs etc.
static void equivalenceAnalysis(func::FuncOp funcOp,
                                BufferizationAliasInfo &aliasInfo,
                                FuncAnalysisState &funcState) {
  funcOp->walk([&](func::CallOp callOp) {
    func::FuncOp calledFunction = getCalledFunction(callOp);
    assert(calledFunction && "could not retrieved called func::FuncOp");

    // No equivalence info available for the called function.
    if (!funcState.equivalentFuncArgs.count(calledFunction))
      return WalkResult::skip();

    for (auto it : funcState.equivalentFuncArgs[calledFunction]) {
      int64_t returnIdx = it.first;
      int64_t bbargIdx = it.second;
      Value returnVal = callOp.getResult(returnIdx);
      Value argVal = callOp->getOperand(bbargIdx);
      aliasInfo.unionEquivalenceClasses(returnVal, argVal);
    }

    return WalkResult::advance();
  });
}

/// Store all functions of the `moduleOp` in `orderedFuncOps`, sorted by
/// callee-caller order (i.e. callees without callers first).
/// Store the map of FuncOp to all its callers in `callerMap`.
/// Return `failure()` if a cycle of calls is detected or if we are unable to
/// retrieve the called FuncOp from any CallOpInterface.
static LogicalResult
getFuncOpsOrderedByCalls(ModuleOp moduleOp,
                         SmallVectorImpl<func::FuncOp> &orderedFuncOps,
                         FuncCallerMap &callerMap) {
  // For each FuncOp, the set of functions called by it (i.e. the union of
  // symbols of all nested CallOpInterfaceOp).
  DenseMap<func::FuncOp, DenseSet<func::FuncOp>> calledBy;
  // For each FuncOp, the number of CallOpInterface it contains.
  DenseMap<func::FuncOp, unsigned> numberCallOpsContainedInFuncOp;
  WalkResult res = moduleOp.walk([&](func::FuncOp funcOp) -> WalkResult {
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
      func::FuncOp calledFunction = getCalledFunction(callOp);
      assert(calledFunction && "could not retrieved called func::FuncOp");
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

namespace mlir {
namespace linalg {
namespace comprehensive_bufferize {
namespace std_ext {

/// Return the index of the bbArg in the given func::FuncOp that is equivalent
/// to the specified return value (if any).
static Optional<int64_t> getEquivalentFuncArgIdx(func::FuncOp funcOp,
                                                 const FuncAnalysisState &state,
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
    func::FuncOp funcOp = getCalledFunction(callOp);
    assert(funcOp && "expected CallOp to a func::FuncOp");

    const FuncAnalysisState &funcState = getFuncAnalysisState(state);
    if (getFuncOpAnalysisState(state, funcOp) != FuncOpAnalysisState::Analyzed)
      // FuncOp not analyzed yet. Assume that OpOperand is read.
      return true;

    return funcState.readBbArgs.lookup(funcOp).contains(
        opOperand.getOperandNumber());
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    func::CallOp callOp = cast<func::CallOp>(op);
    func::FuncOp funcOp = getCalledFunction(callOp);
    assert(funcOp && "expected CallOp to a func::FuncOp");

    const FuncAnalysisState &funcState = getFuncAnalysisState(state);
    if (getFuncOpAnalysisState(state, funcOp) != FuncOpAnalysisState::Analyzed)
      // FuncOp not analyzed yet. Assume that OpOperand is written.
      return true;

    return funcState.writtenBbArgs.lookup(funcOp).contains(
        opOperand.getOperandNumber());
  }

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    func::CallOp callOp = cast<func::CallOp>(op);
    func::FuncOp funcOp = getCalledFunction(callOp);
    assert(funcOp && "expected CallOp to a func::FuncOp");
    const FuncAnalysisState &funcState = getFuncAnalysisState(state);
    if (getFuncOpAnalysisState(state, funcOp) !=
        FuncOpAnalysisState::Analyzed) {
      // FuncOp not analyzed yet. Any OpResult may be aliasing.
      SmallVector<OpResult> result;
      for (OpResult opResult : op->getOpResults())
        if (opResult.getType().isa<TensorType>())
          result.push_back(opResult);
      return result;
    }

    // Get aliasing results from state.
    auto aliasingReturnVals =
        funcState.aliasingReturnVals.lookup(funcOp).lookup(
            opOperand.getOperandNumber());
    SmallVector<OpResult> result;
    for (int64_t resultIdx : aliasingReturnVals)
      result.push_back(callOp->getOpResult(resultIdx));
    return result;
  }

  SmallVector<OpOperand *>
  getAliasingOpOperand(Operation *op, OpResult opResult,
                       const AnalysisState &state) const {
    func::CallOp callOp = cast<func::CallOp>(op);
    func::FuncOp funcOp = getCalledFunction(callOp);
    assert(funcOp && "expected CallOp to a func::FuncOp");
    const FuncAnalysisState &funcState = getFuncAnalysisState(state);
    if (getFuncOpAnalysisState(state, funcOp) !=
        FuncOpAnalysisState::Analyzed) {
      // FuncOp not analyzed yet. Any OpOperand may be aliasing.
      SmallVector<OpOperand *> result;
      for (OpOperand &opOperand : op->getOpOperands())
        if (opOperand.get().getType().isa<TensorType>())
          result.push_back(&opOperand);
      return result;
    }

    // Get aliasing bbArgs from state.
    auto aliasingFuncArgs = funcState.aliasingFuncArgs.lookup(funcOp).lookup(
        opResult.getResultNumber());
    SmallVector<OpOperand *> result;
    for (int64_t bbArgIdx : aliasingFuncArgs)
      result.push_back(&callOp->getOpOperand(bbArgIdx));
    return result;
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const AnalysisState &state) const {
    return BufferRelation::Equivalent;
  }

  /// All function arguments are writable. It is the responsibility of the
  /// CallOp to insert buffer copies where necessary.
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          BufferizationState &state) const {
    func::CallOp callOp = cast<func::CallOp>(op);
    unsigned numResults = callOp.getNumResults();
    unsigned numOperands = callOp->getNumOperands();
    func::FuncOp funcOp = getCalledFunction(callOp);
    assert(funcOp && "expected CallOp to a func::FuncOp");
    const FuncAnalysisState &funcState =
        getFuncAnalysisState(state.getAnalysisState());
    const OneShotBufferizationOptions &options =
        static_cast<const OneShotBufferizationOptions &>(state.getOptions());

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
    // onto a func::FuncOp bbArg, so calls to such functions are not
    // bufferizable at the moment.

    // 1. Compute the result types of the new CallOp. Tensor results that are
    // equivalent to a func::FuncOp bbArg are no longer returned.
    for (const auto &it : llvm::enumerate(callOp.getResultTypes())) {
      unsigned returnValIdx = it.index();
      Type returnType = it.value();
      if (!returnType.isa<TensorType>()) {
        // Non-tensor values are returned.
        retValMapping[returnValIdx] = resultTypes.size();
        resultTypes.push_back(returnType);
        continue;
      }

      if (Optional<int64_t> bbArgIdx =
              getEquivalentFuncArgIdx(funcOp, funcState, returnValIdx)) {
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

      if (!options.allowReturnAllocs)
        return callOp->emitError(
            "call to FuncOp that returns non-equivalent tensors not supported");

      // Returning a memref. This memref is not equivalent to any bbArg. It is
      // likely a newly allocated buffer. We may want to hoist such allocations
      // to the call site in the future.
      retValMapping[returnValIdx] = resultTypes.size();
      resultTypes.push_back(
          funcOp.getFunctionType().getResult(resultTypes.size()));
    }

    // 2. Get the bufferized FunctionType of the called function. Recursive or
    // circular call graphs are not currently supported, so we can be sure that
    // the called function was already bufferized.
    FunctionType bufferizedFuncType = funcOp.getFunctionType();

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
      // corresponding func::FuncOp bbArgs are equivalent to a returned tensor,
      // were already stored in `newOperands` during Step 1.
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
    assert(isa<func::FuncOp>(returnOp->getParentOp()) &&
           "only support FuncOp parent for ReturnOp");
#endif // NDEBUG

    // ReturnOps are bufferized as part of FuncOps.
    return failure();
  }
};

struct FuncOpInterface
    : public BufferizableOpInterface::ExternalModel<FuncOpInterface,
                                                    func::FuncOp> {
  /// Rewrite function bbArgs and return values into buffer form (using the
  /// canonical memref layout for now). This function bufferizes the function
  /// signature and the ReturnOp. When the entire function body has been
  /// bufferized, function return types can be switched to more concise memref
  /// types as part of `foldMemRefCasts`.
  ///
  /// When a tensor function argument is known to be equivalent to a tensor
  /// result, it is dropped from the return values.
  ///
  /// All function bbArgs are writable unless they are explicitly marked as
  /// read-only. Callers must insert copies when needed.
  ///
  /// Note: Returning a memref is possible, but corresponding CallOp
  /// bufferizations fail unless `allowReturnAllocs`.
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          BufferizationState &state) const {
    auto funcOp = cast<func::FuncOp>(op);
    FunctionType funcType = funcOp.getFunctionType();
    const FuncAnalysisState &moduleState =
        getFuncAnalysisState(state.getAnalysisState());
    const BufferizationOptions &options = state.getOptions();

    // Construct the bufferized function type.
    SmallVector<Type> argTypes;
    for (const auto &it : llvm::enumerate(funcType.getInputs())) {
      Type argType = it.value();
      if (auto tensorType = argType.dyn_cast<TensorType>()) {
        argTypes.push_back(
            getBufferizedFunctionArgType(funcOp, it.index(), options));
        continue;
      }
      argTypes.push_back(argType);
    }

    // Bodiless functions are assumed opaque and we cannot know the
    // bufferization contract they want to enforce. As a consequence, only
    // support functions that don't return any tensors atm.
    if (funcOp.getBody().empty()) {
      SmallVector<Type> retTypes;
      for (Type resultType : funcType.getResults()) {
        if (resultType.isa<TensorType>())
          return funcOp->emitError() << "cannot bufferize bodiless function "
                                     << "that returns a tensor";
        retTypes.push_back(resultType);
      }
      funcOp.setType(FunctionType::get(op->getContext(), argTypes, retTypes));
      return success();
    }

    // TODO: Support functions with multiple returns.
    func::ReturnOp returnOp = getAssumedUniqueReturnOp(funcOp);
    assert(returnOp && "expected func with single return op");

    // 1. Rewrite the bbArgs. Turn every tensor bbArg into a memref bbArg.
    Block &frontBlock = funcOp.getBody().front();
    for (BlockArgument &bbArg : frontBlock.getArguments()) {
      auto tensorType = bbArg.getType().dyn_cast<TensorType>();
      // Non-tensor types stay the same.
      if (!tensorType)
        continue;

      // Collect all uses of the bbArg.
      SmallVector<OpOperand *> bbArgUses;
      for (OpOperand &use : bbArg.getUses())
        bbArgUses.push_back(&use);

      // Change the bbArg type to memref.
      Type memrefType =
          getBufferizedFunctionArgType(funcOp, bbArg.getArgNumber(), options);
      bbArg.setType(memrefType);

      // Replace all uses of the original tensor bbArg.
      rewriter.setInsertionPointToStart(&frontBlock);
      if (!bbArgUses.empty()) {
        // Insert to_tensor because the remaining function body has not been
        // bufferized yet.
        Value toTensorOp =
            rewriter.create<bufferization::ToTensorOp>(funcOp.getLoc(), bbArg);
        for (OpOperand *use : bbArgUses)
          use->set(toTensorOp);
      }
    }

    // 2. For each result, keep track of which inplace argument it reuses.
    SmallVector<Value> returnValues;
    for (OpOperand &returnOperand : returnOp->getOpOperands()) {
      Value returnVal = returnOperand.get();

      // If not a tensor type just forward it.
      if (!returnVal.getType().isa<RankedTensorType>()) {
        returnValues.push_back(returnVal);
        continue;
      }

      // If return operand is equivalent to some bbArg, no need to return it.
      if (Optional<int64_t> equivBbArgIdx = getEquivalentFuncArgIdx(
              funcOp, moduleState, returnOperand.getOperandNumber())) {
        rewriter.setInsertionPoint(returnOp);
        Location loc = returnOp.getLoc();
        Value toMemrefOp = rewriter.create<bufferization::ToMemrefOp>(
            loc, getMemRefType(returnVal.getType().cast<TensorType>(), options),
            returnVal);
        BlockArgument equivBbArg = funcOp.getArgument(*equivBbArgIdx);
        // Note: This copy will fold away. It must be inserted here to ensure
        // that `returnVal` still has at least one use and does not fold away.
        if (failed(
                createMemCpy(rewriter, loc, toMemrefOp, equivBbArg, options)))
          return funcOp->emitError("could not generate copy for bbArg");
        continue;
      }

      returnValues.push_back(*state.getBuffer(rewriter, returnOperand));
    }

    // 3. Rewrite the terminator without the in-place bufferizable values.
    returnOp.operandsMutable().assign(returnValues);

    // 4. Rewrite the FuncOp type to buffer form.
    funcOp.setType(FunctionType::get(op->getContext(), argTypes,
                                     ValueRange(returnValues).getTypes()));

    return success();
  }

  /// Return `true` if the given function argument is writable.
  bool isWritable(Operation *op, Value value,
                  const AnalysisState &state) const {
    auto funcOp = cast<func::FuncOp>(op);
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

/// Set the attribute that triggers inplace bufferization on a func::FuncOp
/// argument `bbArg`.
static void setInPlaceFuncArgument(BlockArgument bbArg, bool inPlace) {
  auto funcOp = cast<func::FuncOp>(bbArg.getOwner()->getParentOp());
  funcOp.setArgAttr(bbArg.getArgNumber(),
                    BufferizableOpInterface::kInplaceableAttrName,
                    BoolAttr::get(bbArg.getContext(), inPlace));
}

/// Annotate the IR with the result of the analysis. For testing/debugging only.
static void annotateOpsWithBufferizationMarkers(func::FuncOp funcOp,
                                                const AnalysisState &state) {
  auto bufferizableOp = cast<BufferizableOpInterface>(funcOp.getOperation());
  for (BlockArgument bbArg : funcOp.getArguments())
    if (bbArg.getType().isa<TensorType>())
      setInPlaceFuncArgument(bbArg, bufferizableOp.isWritable(bbArg, state));
}

/// Fold return values that are memref casts and update function return types.
///
/// During FuncOp bufferization, the exact type of the returned memrefs (if any)
/// is not known yet. Therefore, the bufferization uses memref types with the
/// most generic layout map as function return types. After bufferizing the
/// entire function body, a more concise memref type can potentially be used for
/// the return type of the function.
static void foldMemRefCasts(func::FuncOp funcOp) {
  if (funcOp.getBody().empty())
    return;

  func::ReturnOp returnOp = getAssumedUniqueReturnOp(funcOp);
  SmallVector<Type> resultTypes;

  for (OpOperand &operand : returnOp->getOpOperands()) {
    if (auto castOp = operand.get().getDefiningOp<memref::CastOp>()) {
      operand.set(castOp.source());
      resultTypes.push_back(castOp.source().getType());
    } else {
      resultTypes.push_back(operand.get().getType());
    }
  }

  auto newFuncType = FunctionType::get(
      funcOp.getContext(), funcOp.getFunctionType().getInputs(), resultTypes);
  funcOp.setType(newFuncType);
}

LogicalResult mlir::linalg::comprehensive_bufferize::runModuleBufferize(
    ModuleOp moduleOp, OneShotBufferizationOptions options) {
  IRRewriter rewriter(moduleOp.getContext());
  OneShotAnalysisState analysisState(moduleOp, options);
  BufferizationState bufferizationState(analysisState);
  FuncAnalysisState &funcState = getFuncAnalysisState(analysisState);
  BufferizationAliasInfo &aliasInfo = analysisState.getAliasInfo();

  // A list of functions in the order in which they are analyzed + bufferized.
  SmallVector<func::FuncOp> orderedFuncOps;

  // A mapping of FuncOps to their callers.
  FuncCallerMap callerMap;

  if (failed(getFuncOpsOrderedByCalls(moduleOp, orderedFuncOps, callerMap)))
    return failure();

  // Collect bbArg/return value information after the analysis.
  options.addPostAnalysisStep(aliasingFuncOpBBArgsAnalysis);
  options.addPostAnalysisStep(funcOpBbArgReadWriteAnalysis);

  // Analyze ops.
  for (func::FuncOp funcOp : orderedFuncOps) {
    // No body => no analysis.
    if (funcOp.getBody().empty())
      continue;

    // Now analyzing function.
    funcState.startFunctionAnalysis(funcOp);

    // Gather equivalence info for CallOps.
    equivalenceAnalysis(funcOp, aliasInfo, funcState);

    // Analyze funcOp.
    if (failed(analyzeOp(funcOp, analysisState)))
      return failure();

    // Mark op as fully analyzed.
    funcState.analyzedFuncOps[funcOp] = FuncOpAnalysisState::Analyzed;

    // Add annotations to function arguments.
    if (options.testAnalysisOnly)
      annotateOpsWithBufferizationMarkers(funcOp, analysisState);
  }

  if (options.testAnalysisOnly)
    return success();

  // Bufferize functions.
  for (func::FuncOp funcOp : orderedFuncOps) {
    // Note: It would be good to apply cleanups here but we cannot as aliasInfo
    // would be invalidated.
    if (failed(bufferizeOp(funcOp, bufferizationState)))
      return failure();
    foldMemRefCasts(funcOp);
  }

  // Check result.
  for (func::FuncOp funcOp : orderedFuncOps) {
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

  // Post-pass cleanup of inplaceable and buffer_layout attributes.
  moduleOp.walk([&](func::FuncOp op) {
    for (BlockArgument bbArg : op.getArguments())
      removeBufferizationFuncArguments(bbArg);
  });

  return success();
}
