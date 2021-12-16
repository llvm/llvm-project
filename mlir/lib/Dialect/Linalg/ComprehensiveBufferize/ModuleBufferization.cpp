//===- ModuleBufferization.cpp - Bufferization across Func. Boundaries ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/ComprehensiveBufferize/ModuleBufferization.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/BufferizableOpInterface.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/ComprehensiveBufferize.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace linalg;
using namespace tensor;
using namespace comprehensive_bufferize;

namespace {
/// Extra bufferization state that is required for bufferization of function
/// boundaries.
struct ModuleBufferizationState : public DialectBufferizationState {
  /// A map for looking up bufferized function types.
  DenseMap<FuncOp, FunctionType> bufferizedFunctionTypes;

  /// A mapping of ReturnOp OpOperand indices to equivalent FuncOp BBArg
  /// indices.
  DenseMap<FuncOp, DenseMap<int64_t, int64_t>> equivalentFuncArgs;

  SmallVector<FuncOp> orderedFuncOps;

  DenseMap<FuncOp, DenseSet<Operation *>> callerMap;
};
} // namespace

static ModuleBufferizationState &
getModuleBufferizationState(BufferizationState &state) {
  return state.getDialectState<ModuleBufferizationState>(
      StandardOpsDialect::getDialectNamespace());
}

/// Return the unique ReturnOp that terminates `funcOp`.
/// Return nullptr if there is no such unique ReturnOp.
static ReturnOp getAssumedUniqueReturnOp(FuncOp funcOp) {
  ReturnOp returnOp;
  for (Block &b : funcOp.body()) {
    if (auto candidateOp = dyn_cast<ReturnOp>(b.getTerminator())) {
      if (returnOp)
        return nullptr;
      returnOp = candidateOp;
    }
  }
  return returnOp;
}

namespace {
/// Store function BlockArguments that are equivalent to a returned value in
/// ModuleBufferizationState.
struct EquivalentFuncOpBBArgsAnalysis : public PostAnalysisStep {
  /// Annotate IR with the results of the analysis. For testing purposes only.
  static void annotateReturnOp(OpOperand &returnVal, BlockArgument bbArg) {
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

  LogicalResult run(Operation *op, BufferizationState &state,
                    BufferizationAliasInfo &aliasInfo,
                    SmallVector<Operation *> &newOps) override {
    ModuleBufferizationState &moduleState = getModuleBufferizationState(state);

    // Support only single return-terminated block in the function.
    auto funcOp = cast<FuncOp>(op);
    ReturnOp returnOp = getAssumedUniqueReturnOp(funcOp);
    assert(returnOp && "expected func with single return op");

    for (OpOperand &returnVal : returnOp->getOpOperands())
      if (returnVal.get().getType().isa<RankedTensorType>())
        for (BlockArgument bbArg : funcOp.getArguments())
          if (bbArg.getType().isa<RankedTensorType>())
            if (aliasInfo.areEquivalentBufferizedValues(returnVal.get(),
                                                        bbArg)) {
              moduleState
                  .equivalentFuncArgs[funcOp][returnVal.getOperandNumber()] =
                  bbArg.getArgNumber();
              if (state.getOptions().testAnalysisOnly)
                annotateReturnOp(returnVal, bbArg);
            }

    return success();
  }
};
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
static FunctionType getBufferizedFunctionType(MLIRContext *ctx,
                                              TypeRange argumentTypes,
                                              TypeRange resultTypes) {
  auto rewrite = [](Type t) -> Type {
    // TODO: non-zero address space.
    // TODO: layout information if relevant.
    if (auto rankedTensorType = t.dyn_cast<RankedTensorType>())
      return getDynamicMemRefType(rankedTensorType);
    if (auto tensorType = t.dyn_cast<TensorType>())
      return getContiguousOrUnrankedMemRefType(tensorType);
    return t;
  };
  auto argTypes = llvm::to_vector<4>(llvm::map_range(argumentTypes, rewrite));
  auto retTypes = llvm::to_vector<4>(llvm::map_range(resultTypes, rewrite));
  return FunctionType::get(ctx, argTypes, retTypes);
}

/// If an entry for `funcOp` is available in `bufferizedFunctionTypes`, return
/// it. Otherwise, construct a new entry based on `argumentTypes` and
/// `resultTypes`.
// TODO: improve the layering.
static FunctionType getOrCreateBufferizedFunctionType(
    FuncOp funcOp, TypeRange argumentTypes, TypeRange resultTypes,
    DenseMap<FuncOp, FunctionType> &bufferizedFunctionTypes) {
  auto it = bufferizedFunctionTypes.find(funcOp);
  if (it != bufferizedFunctionTypes.end())
    return it->second;

  auto it2 = bufferizedFunctionTypes.try_emplace(
      funcOp, getBufferizedFunctionType(funcOp.getContext(), argumentTypes,
                                        resultTypes));
  return it2.first->second;
}

/// Gather equivalence info of CallOps.
/// Note: This only adds new equivalence info if `funcOp` was already analyzed.
// TODO: This does not handle cyclic function call graphs etc.
static void equivalenceAnalysis(FuncOp funcOp,
                                BufferizationAliasInfo &aliasInfo,
                                ModuleBufferizationState &moduleState) {
  funcOp->walk([&](CallOp callOp) {
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
                                             BufferizationState &state) {
  ModuleBufferizationState &moduleState = getModuleBufferizationState(state);

  // If nothing to do then we are done.
  if (!llvm::any_of(funcOp.getType().getInputs(), isaTensor) &&
      !llvm::any_of(funcOp.getType().getResults(), isaTensor))
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
    if (llvm::any_of(funcOp.getType().getResults(), isaTensor))
      return funcOp->emitError() << "cannot bufferize bodiless function that "
                                 << "returns a tensor";
    FunctionType bufferizedFuncType = getOrCreateBufferizedFunctionType(
        funcOp, funcOp.getType().getInputs(), TypeRange{},
        moduleState.bufferizedFunctionTypes);
    funcOp.setType(bufferizedFuncType);
    return success();
  }

  // Support only single return-terminated block in the function.
  ReturnOp returnOp = getAssumedUniqueReturnOp(funcOp);
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
    if (moduleState.equivalentFuncArgs[funcOp].count(
            returnOperand.getOperandNumber()))
      continue;

    // Cast values at the call site if necessary.
    returnValues.push_back(getNonCastedValue(state.lookupBuffer(returnVal)));
  }

  // 2. Rewrite the terminator without the inPlace bufferizable values.
  ValueRange retValues{returnValues};
  FunctionType bufferizedFuncType = getOrCreateBufferizedFunctionType(
      funcOp, funcOp.getType().getInputs(), retValues.getTypes(),
      moduleState.bufferizedFunctionTypes);
  OpBuilder b(returnOp);
  b.create<ReturnOp>(returnOp.getLoc(), returnValues);
  returnOp->erase();

  // 3. Rewrite the bbArgs.
  // Iterate on the original `numArgs` and replace them in order.
  // This guarantees the argument order still matches after the rewrite.
  Block &frontBlock = funcOp.body().front();
  unsigned numArgs = frontBlock.getNumArguments();
  for (unsigned idx = 0; idx < numArgs; ++idx) {
    auto bbArg = frontBlock.getArgument(0);
    auto tensorType = bbArg.getType().dyn_cast<TensorType>();
    // Non-tensor types are just forwarded.
    if (!tensorType) {
      frontBlock.addArgument(bbArg.getType());
      bbArg.replaceAllUsesWith(frontBlock.getArguments().back());
      frontBlock.eraseArgument(0);
      continue;
    }

    // Get the buffer type from the bufferized function type.
    Type memrefType = bufferizedFuncType.getInput(idx);
    Value memref = frontBlock.addArgument(memrefType);
    OpBuilder b(funcOp->getContext());
    b.setInsertionPointToStart(&frontBlock);
    // Replace all uses of bbArg through a ToMemRefOp by a memref::CastOp.
    for (auto &use : llvm::make_early_inc_range(bbArg.getUses())) {
      if (auto toMemrefOp =
          dyn_cast<bufferization::ToMemrefOp>(use.getOwner())) {
        auto castOp = b.create<memref::CastOp>(
            funcOp.getLoc(), toMemrefOp.memref().getType(), memref);
        toMemrefOp.memref().replaceAllUsesWith(castOp);
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
    if (!funcOp.body().empty()) {
      ReturnOp returnOp = getAssumedUniqueReturnOp(funcOp);
      if (!returnOp)
        return funcOp->emitError()
               << "cannot bufferize a FuncOp with tensors and "
                  "without a unique ReturnOp";
    }

    numberCallOpsContainedInFuncOp[funcOp] = 0;
    return funcOp.walk([&](CallOpInterface callOp) -> WalkResult {
      // Only support CallOp for now.
      if (!isa<CallOp>(callOp.getOperation()))
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
    for (const auto &it : llvm::enumerate(funcOp.getType().getInputs())) {
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
      if (!funcOp.body().empty()) {
        BlockArgument bbArg = funcOp.getArgument(argNumber);
        bbArg.setType(desiredMemrefType);
        OpBuilder b(bbArg.getContext());
        b.setInsertionPointToStart(bbArg.getOwner());
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
                                         funcOp.getType().getResults());
    funcOp.setType(newFuncType);
  }
}

namespace mlir {
namespace linalg {
namespace comprehensive_bufferize {
namespace std_ext {

struct CallOpInterface
    : public BufferizableOpInterface::ExternalModel<CallOpInterface, CallOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              BufferizationState &state) const {
    // CallOpInterface alone doesn't bufferize to a memory read, one of the uses
    // of the matching bbArg may. It is the responsibility of the caller to
    // inspect bbArgs. In the absence of a BufferizationAliasInfo, we need to be
    // conservative.
    return true;
  }

  OpResult getAliasingOpResult(Operation *op, OpOperand &opOperand,
                               BufferizationState &state) const {
    // CallOpInterface is special, it needs to wait for the callee to be
    // bufferized and needs to inspect the BufferAliasInfo object. It can't
    // make a proper determination by itself and needs to be conservative.
    return OpResult();
  }

  /// In a first approximation, all the function arguments of a FuncOp are
  /// marked inplaceable. For now, it is the responsibility of the `callOp`
  /// bufferization to allow FuncOp that are inplaceable to write inPlace.
  LogicalResult bufferize(Operation *op, OpBuilder &b,
                          BufferizationState &state) const {
    CallOp callOp = cast<CallOp>(op);
    FuncOp funcOp = getCalledFunction(callOp);
    assert(isa<CallOp>(callOp.getOperation()) && funcOp &&
           "expected Callop to a FuncOp");
    ModuleBufferizationState &moduleState = getModuleBufferizationState(state);

    // 1. Filter return types:
    //    - if the callee is bodiless / external, we cannot inspect it and we
    //      cannot assume anything. We can just assert that it does not return a
    //      tensor as this would have to bufferize to "return a memref", whose
    //      semantics is ill-defined.
    //    - if the callee has a body, we perform inter-procedural equivalence
    //      analysis. When successful, a result folds onto an operand. When
    //      unsuccessful, additional work is needed (TODO) to either:
    //        * hoist a result into an inplaceable operand or
    //        * devise a better representation to truly return a buffer.
    SmallVector<Type> resultTypes;
    if (funcOp.body().empty()) {
      if (llvm::any_of(funcOp.getType().getResults(), isaTensor))
        return callOp->emitError()
               << "cannot bufferize bodiless function that returns a tensor";
    } else {
      ReturnOp returnOp = getAssumedUniqueReturnOp(funcOp);
      assert(returnOp && "expected func with single return op");

      // For each FuncOp result, keep track of which inplace argument it reuses.
      for (OpOperand &returnOperand : returnOp->getOpOperands()) {
        Type returnType = returnOperand.get().getType();
        if (!isaTensor(returnType)) {
          resultTypes.push_back(returnType);
          continue;
        }

        // If return operand is equivalent to some bbArg, no need to return it.
        if (moduleState.equivalentFuncArgs[funcOp].count(
                returnOperand.getOperandNumber())) {
          int64_t idx =
              moduleState
                  .equivalentFuncArgs[funcOp][returnOperand.getOperandNumber()];
          Value oldRes = callOp->getResult(returnOperand.getOperandNumber());
          Value buffer = state.lookupBuffer(callOp->getOperand(idx));
          // Add a ToTensorOp to kill all uses of the CallOp return.
          // Replace all uses of the CallOp results so we can erase the CallOp.
          // This ToTensorOp must fold/DCE away or bufferization should be
          // considered failed.
          Value toTensorOp =
              b.create<bufferization::ToTensorOp>(callOp.getLoc(), buffer);
          oldRes.replaceAllUsesWith(toTensorOp);
          continue;
        }

        resultTypes.push_back(returnType);
      }
    }

    // 2. Compute bufferized FunctionType.
    SmallVector<Type> argumentTypes{callOp->getOperandTypes()};
    // Get the bufferized FunctionType for funcOp or construct it if not yet
    // available.
    FunctionType bufferizedFuncType =
        getOrCreateBufferizedFunctionType(funcOp, argumentTypes, resultTypes,
                                          moduleState.bufferizedFunctionTypes);

    // 3. Rewrite tensor operands as memrefs based on `bufferizedFuncType`.
    SmallVector<Value> newOperands;
    newOperands.reserve(callOp->getNumOperands());
    for (OpOperand &opOperand : callOp->getOpOperands()) {
      Value tensorOperand = opOperand.get();
      // Non-tensor operands are just copied.
      if (!tensorOperand.getType().isa<TensorType>()) {
        newOperands.push_back(tensorOperand);
        continue;
      }

      // Tensor operands are guaranteed to have been buferized.
      int64_t idx = opOperand.getOperandNumber();
      Value buffer = state.lookupBuffer(tensorOperand);

      // Caller / callee type mistmatch is handled with a CastOp.
      auto memRefType = bufferizedFuncType.getInput(idx);
      // Since we don't yet have a clear layout story, buffer_cast may
      // conservatively turn tensors into more dynamic memref than necessary.
      // If the memref type of the callee fails, introduce an extra memref.cast
      // that will either canonicalize away or fail compilation until we can do
      // something better.
      if (buffer.getType() != memRefType) {
        Value castBuffer =
            b.create<memref::CastOp>(callOp.getLoc(), memRefType, buffer);
        buffer = castBuffer;
      }
      newOperands.push_back(buffer);
    }

    // 4. Create the new CallOp.
    Operation *newCallOp = b.create<CallOp>(callOp.getLoc(), funcOp.sym_name(),
                                            resultTypes, newOperands);
    newCallOp->setAttrs(callOp->getAttrs());

    // 5. Delete the op at the end of bufferization.
    callOp->erase();

    return success();
  }
};

struct ReturnOpInterface
    : public BufferizableOpInterface::ExternalModel<ReturnOpInterface,
                                                    ReturnOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              BufferizationState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               BufferizationState &state) const {
    return false;
  }

  OpResult getAliasingOpResult(Operation *op, OpOperand &opOperand,
                               BufferizationState &state) const {
    return OpResult();
  }

  LogicalResult bufferize(Operation *op, OpBuilder &b,
                          BufferizationState &state) const {
    auto returnOp = cast<ReturnOp>(op);
    assert(isa<FuncOp>(returnOp->getParentOp()) &&
           "only support FuncOp parent for ReturnOp");

    for (OpOperand &operand : returnOp->getOpOperands()) {
      auto tensorType = operand.get().getType().dyn_cast<TensorType>();
      if (!tensorType)
        continue;
      Value v = state.lookupBuffer(operand.get());
      Value returnTensor = b.create<bufferization::ToTensorOp>(
          returnOp.getLoc(), v);
      operand.set(returnTensor);
    }
    return success();
  }
};

struct FuncOpInterface
    : public BufferizableOpInterface::ExternalModel<FuncOpInterface, FuncOp> {
  LogicalResult bufferize(Operation *op, OpBuilder &b,
                          BufferizationState &state) const {
    auto funcOp = cast<FuncOp>(op);

    // Bufferize function body.
    return comprehensive_bufferize::bufferize(&funcOp.body(), state);
  }

  /// Return `true` if the given function argument is writable.
  bool isWritable(Operation *op, Value value, BufferizationState &state) const {
    auto funcOp = cast<FuncOp>(op);
    BlockArgument bbArg = value.dyn_cast<BlockArgument>();
    assert(bbArg && "expected BlockArgument");
    ModuleBufferizationState &moduleState = getModuleBufferizationState(state);

    // In a first approximation:
    // =========================
    // If the function is called, we can allocate on the caller side which lets
    // us force inplace arguments at function boundaries.
    // TODO: do not rely on this behavior.
    if (moduleState.callerMap.find(funcOp) != moduleState.callerMap.end())
      return true;

    // Set the function arguments marked with inplaceable to be known as
    // bufferizing to a writeable memory.
    BoolAttr inplaceAttr = funcOp.getArgAttrOfType<BoolAttr>(
        bbArg.getArgNumber(), BufferizableOpInterface::kInplaceableAttrName);
    if (inplaceAttr && inplaceAttr.getValue())
      return true;

    // All other function arguments are not writable.
    return false;
  }

  bool isAllocationHoistingBarrier(Operation *op) const { return true; }
};

} // namespace std_ext
} // namespace comprehensive_bufferize
} // namespace linalg
} // namespace mlir

void mlir::linalg::comprehensive_bufferize::std_ext::
    registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addOpInterface<CallOp, std_ext::CallOpInterface>();
  registry.addOpInterface<ReturnOp, std_ext::ReturnOpInterface>();
  registry.addOpInterface<FuncOp, std_ext::FuncOpInterface>();
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
                                                BufferizationState &state) {
  auto bufferizableOp = cast<BufferizableOpInterface>(funcOp.getOperation());
  for (BlockArgument bbArg : funcOp.getArguments())
    if (bbArg.getType().isa<TensorType>())
      setInPlaceFuncArgument(bbArg, bufferizableOp.isWritable(bbArg, state));
}

LogicalResult mlir::linalg::comprehensive_bufferize::runComprehensiveBufferize(
    ModuleOp moduleOp, const BufferizationOptions &options) {
  BufferizationState state(moduleOp, options);
  ModuleBufferizationState &moduleState = getModuleBufferizationState(state);
  BufferizationAliasInfo &aliasInfo = state.aliasInfo;

  if (failed(getFuncOpsOrderedByCalls(moduleOp, moduleState.orderedFuncOps,
                                      moduleState.callerMap)))
    return failure();

  // Interestingly, all function args that are not visible outside of a module
  // can be fully bufferized inplace by guaranteeing the CallOp is bufferized
  // inplace. Therefore, we just bufferize funcOp as if none of its results were
  // inplaceable, detect which operands are cloned internally and decide what to
  // do at call sites.
  for (FuncOp funcOp : moduleState.orderedFuncOps) {
    // No body => no analysis.
    if (funcOp.body().empty())
      continue;

    // Register extra post analysis steps. These cannot be stored in `options`
    // because `options` is immutable.
    PostAnalysisStepList extraSteps;
    extraSteps.emplace_back(std::make_unique<EquivalentFuncOpBBArgsAnalysis>());

    // Gather equivalence info for CallOps.
    equivalenceAnalysis(funcOp, aliasInfo, moduleState);

    // Analyze and bufferize funcOp.
    if (failed(runComprehensiveBufferize(funcOp, options, state, extraSteps)))
      return failure();

    // Add annotations to function arguments.
    if (options.testAnalysisOnly)
      annotateOpsWithBufferizationMarkers(funcOp, state);
  }

  if (options.testAnalysisOnly)
    return success();

  for (FuncOp funcOp : moduleState.orderedFuncOps) {
    // Note: It would be good to apply cleanups here but we cannot as aliasInfo
    // would be invalidated.
    if (failed(bufferizeFuncOpBoundary(funcOp, state)))
      return failure();

    if (!options.allowReturnMemref &&
        llvm::any_of(funcOp.getType().getResults(), [](Type t) {
          return t.isa<MemRefType, UnrankedMemRefType>();
        })) {
      funcOp->emitError("memref return type is unsupported");
      return failure();
    }
  }

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
