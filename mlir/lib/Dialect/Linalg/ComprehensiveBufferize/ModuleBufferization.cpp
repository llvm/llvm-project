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
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

#define DEBUG_TYPE "comprehensive-module-bufferize"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X)

using namespace mlir;
using namespace linalg;
using namespace tensor;
using namespace comprehensive_bufferize;

namespace {
/// A specialization of BufferizationState that keeps track of additional
/// state required for bufferization of function boundaries.
struct ModuleBufferizationState : public BufferizationState {
  using BufferizationState::BufferizationState;

  /// A map for looking up bufferized function types.
  DenseMap<FuncOp, FunctionType> bufferizedFunctionTypes;
};
} // namespace

static bool isaTensor(Type t) { return t.isa<TensorType>(); }

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
  LDBG("FT: " << funcOp.getType() << " -> " << it2.first->second << "\n");
  return it2.first->second;
}

/// Return the op with Allocate MemoryEffect if `v` is equivalent to such an
/// an op. Return null otherwise.
static Operation *getEquivalentAlloc(Value value,
                                     const BufferizationAliasInfo &aliasInfo) {
  Operation *res = nullptr;
  aliasInfo.applyOnEquivalenceClass(value, [&](Value v) {
    if (!res)
      if (auto interface =
              dyn_cast_or_null<MemoryEffectOpInterface>(v.getDefiningOp()))
        if (auto effect =
                interface.getEffectOnValue<MemoryEffects::Allocate>(v))
          res = v.getDefiningOp();
  });
  return res;
}

/// Return the first argument of the enclosing FuncOp that is equivalent to `v`.
/// Return null if no such bbArg can be found.
static BlockArgument
getEquivalentEnclosingFuncBBArg(Value v,
                                const BufferizationAliasInfo &aliasInfo) {
  if (!v.getType().isa<RankedTensorType>())
    return nullptr;
  Operation *op = v.getParentBlock()->getParentOp();
  FuncOp funcOp = dyn_cast<FuncOp>(op);
  if (!funcOp)
    funcOp = op->getParentOfType<FuncOp>();
  assert(funcOp && "expected non-null FuncOp");
  for (BlockArgument bbArg : funcOp.getArguments()) {
    if (!bbArg.getType().isa<RankedTensorType>())
      continue;
    if (aliasInfo.areEquivalentBufferizedValues(v, bbArg))
      return bbArg;
  }
  return nullptr;
}

/// Rewrite the `funcOp` arguments analysis return values and terminator into
/// buffer form (using the canonical memref layout for now), according to the
/// inPlace-bufferizable information of the function arguments.
/// This relies on a buffer equivalence analysis of each return operand. When a
/// result buffer is equivalent to:
///   1. a BlockArgument of `funcOp`, it can be dropped from the return values
///      and becomes inplaceable at all callers. This assumes all CallOp perform
///      the necessary work to clone operands so as to make them inplaceable.
//       Reliance on this logic will need to be relaxed in thefuture.
///   2. an op with an Alloc effect, this currently fails bufferization but is a
///      candidate for hoisting and creating a new inplace operand at all caller
///      sites.
///   3. if such a hoisting for 2. is not possible (e.g. data-dependent that
///      prevents hoisting), this is currently unsupported and will require a
///      refcounted buffer type.
static LogicalResult bufferizeFuncOpBoundary(
    FuncOp funcOp, BufferizationAliasInfo &aliasInfo,
    DenseMap<FuncOp, FunctionType> &bufferizedFunctionTypes) {
  LLVM_DEBUG(DBGS() << "Begin bufferizeFuncOpBoundary:\n" << funcOp << "\n");

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
    FunctionType bufferizedFuncType =
        getOrCreateBufferizedFunctionType(funcOp, funcOp.getType().getInputs(),
                                          TypeRange{}, bufferizedFunctionTypes);
    funcOp.setType(bufferizedFuncType);
    LLVM_DEBUG(DBGS() << "End bufferizeFuncOpBoundary no fun body: " << funcOp);
    return success();
  }

  // Support only single return-terminated block in the function.
  ReturnOp returnOp = getAssumedUniqueReturnOp(funcOp);
  assert(returnOp && "expected func with single return op");

  // 1. For each FuncOp result, keep track of which inplace argument it reuses.
  SmallVector<Value> returnValues;
  for (OpOperand &returnOperand : returnOp->getOpOperands()) {
    // If not a renturn tensor type just forward it.
    if (!returnOperand.get().getType().isa<RankedTensorType>()) {
      returnValues.push_back(returnOperand.get());
      continue;
    }

    // If return operand is equivalent to some bbArg, no need to return it.
    Value returnVal = returnOperand.get();
    if (getEquivalentEnclosingFuncBBArg(returnVal, aliasInfo))
      continue;

    // TODO: Need to hoist above function boundary.
    if (Operation *allocOp = getEquivalentAlloc(returnVal, aliasInfo)) {
      returnValues.push_back(allocOp->getResult(0));
      continue;
    }

    // Other cases legitimately need to return a tensor, this is currently not
    // supported. For instance, if hoisting across function boundary has
    // failed, it may be due to e.g. data-dependent sizes. In such a case, we
    // would need a better type than memref.
    int64_t returnIdx = returnOperand.getOperandNumber();
    return returnOp->emitError()
           << "buffer result #" << returnIdx << " not produced by an alloc\n";
  }

  // 2. Rewrite the terminator without the inPlace bufferizable values.
  ValueRange retValues{returnValues};
  FunctionType bufferizedFuncType = getOrCreateBufferizedFunctionType(
      funcOp, funcOp.getType().getInputs(), retValues.getTypes(),
      bufferizedFunctionTypes);
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
        aliasInfo.insertNewBufferEquivalence(castOp.dest(),
                                             toMemrefOp.memref());
      }
    }
    // Replace all remaining uses by a to_tensor.
    if (!bbArg.use_empty()) {
      auto toTensorOp =
          b.create<bufferization::ToTensorOp>(funcOp.getLoc(), memref);
      aliasInfo.insertNewBufferEquivalence(toTensorOp, bbArg);
      bbArg.replaceAllUsesWith(toTensorOp);
    }
    frontBlock.eraseArgument(0);
    // TODO: add support to erase aliasInfo entries if deemed necessary.
  }

  // 4. Rewrite the FuncOp type to buffer form.
  funcOp.setType(bufferizedFuncType);

  LLVM_DEBUG(DBGS() << "End bufferizeFuncOpBoundary:\n" << funcOp);

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
    for (auto it : llvm::enumerate(funcOp.getType().getInputs())) {
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
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand) const {
    // CallOpInterface alone doesn't bufferize to a memory read, one of the uses
    // of the matching bbArg may. It is the responsibility of the caller to
    // inspect bbArgs. In the absence of a BufferizationAliasInfo, we need to be
    // conservative.
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand) const {
    // CallOpInterface alone doesn't bufferize to a memory write, one of the
    // uses of the matching bbArg may. It is the responsibility of the caller to
    // inspect bbArgs. In the absence of a BufferizationAliasInfo, we need to be
    // conservative.
    return true;
  }

  SmallVector<OpOperand *> getAliasingOpOperand(Operation *op,
                                                OpResult opResult) const {
    // TODO: Can we do better?
    return {};
  }

  OpResult getAliasingOpResult(Operation *op, OpOperand &opOperand) const {
    // CallOpInterface is special, it needs to wait for the callee to be
    // bufferized and needs to inspect the BufferAliasInfo object. It can't
    // make a proper determination by itself and needs to be conservative.
    return OpResult();
  }

  BufferRelation bufferRelation(Operation *op, OpOperand &opOperand) const {
    return BufferRelation::Equivalent;
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

    // Take a guard before anything else.
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(callOp);

    // 1. Filter return types:
    //    - if the callee is bodiless / external, we cannot inspect it and we
    //      cannot assume anything. We can just assert that it does not return a
    //      tensor as this would have to bufferize to "return a memref", whose
    //      semantics is ill-defined.
    //    - if the callee has a body, we perform inter-procedural equivalence
    //      analysis. When successful, a result folds onto an operand. When
    //      unsuccessful, additional work is needed to either:
    //        * hoist a result into an inplaceable operand or
    //        * devise a better representation to truly return a buffer.
    SmallVector<Type> resultTypes;
    SmallVector<Value> hoistedArguments;
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
        Value returnVal = returnOperand.get();
        if (BlockArgument bbArg =
                getEquivalentEnclosingFuncBBArg(returnVal, state.aliasInfo)) {
          Value oldRes = callOp->getResult(returnOperand.getOperandNumber());
          int64_t idx = bbArg.getArgNumber();
          Value buffer = state.lookupBuffer(callOp->getOperand(idx));
          // Add CallOp operand/result equivalence: this is interprocedural
          // info.
          state.aliasInfo.insertNewBufferEquivalence(oldRes, buffer);
          state.mapBuffer(oldRes, buffer);
          // Add a ToTensorOp to kill all uses of the CallOp return.
          // Replace all uses of the CallOp results so we can erase the CallOp.
          // This ToTensorOp must fold/DCE away or bufferization should be
          // considered failed.
          Value toTensorOp =
              b.create<bufferization::ToTensorOp>(callOp.getLoc(), buffer);
          oldRes.replaceAllUsesWith(toTensorOp);
          // Add new op equivalence info.
          state.aliasInfo.insertNewBufferEquivalence(toTensorOp, buffer);
          state.mapBuffer(toTensorOp, buffer);
          continue;
        }

        // TODO: Need to hoist above function boundary.
        if (Operation *allocOp =
                getEquivalentAlloc(returnVal, state.aliasInfo)) {
          hoistedArguments.push_back(allocOp->getResult(0));
          continue;
        }

        // Other cases legitimately need to return a tensor, this is currently
        // not supported. For instance, if hoisting across function boundary has
        // failed, it may be due to e.g. data-dependent sizes. In such a case,
        // we would we need a better type than memref.
        resultTypes.push_back(returnType);

        int64_t returnIdx = returnOperand.getOperandNumber();
        return returnOp->emitError() << "buffer result #" << returnIdx
                                     << " not produced by an alloc\n";
      }
    }

    // 2. Compute bufferized FunctionType.
    SmallVector<Type> argumentTypes{callOp->getOperandTypes()};
    ValueRange hoistedArgs{hoistedArguments};
    llvm::append_range(argumentTypes, hoistedArgs.getTypes());
    // Get the bufferized FunctionType for funcOp or construct it if not yet
    // available.
    // TODO: Assert that `state` is a ModuleBufferizationState.
    FunctionType bufferizedFuncType = getOrCreateBufferizedFunctionType(
        funcOp, argumentTypes, resultTypes,
        static_cast<ModuleBufferizationState &>(state).bufferizedFunctionTypes);

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
        // Add new op equivalence info.
        state.aliasInfo.insertNewBufferEquivalence(castBuffer, buffer);
        state.mapBuffer(tensorOperand, castBuffer);
        buffer = castBuffer;
      }
      newOperands.push_back(buffer);
    }

    // 4. Create the new CallOp.
    Operation *newCallOp = b.create<CallOp>(callOp.getLoc(), funcOp.sym_name(),
                                            resultTypes, newOperands);
    newCallOp->setAttrs(callOp->getAttrs());

    // 5. Delete the op at the end of bufferization.
    state.markOpObsolete(callOp);

    return success();
  }
};

struct ReturnOpInterface
    : public BufferizableOpInterface::ExternalModel<ReturnOpInterface,
                                                    ReturnOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand) const {
    return false;
  }

  OpResult getAliasingOpResult(Operation *op, OpOperand &opOperand) const {
    return OpResult();
  }

  LogicalResult bufferize(Operation *op, OpBuilder &b,
                          BufferizationState &state) const {
    auto returnOp = cast<ReturnOp>(op);

    // Take a guard before anything else.
    OpBuilder::InsertionGuard g(b);
    // Cannot insert after returnOp.
    b.setInsertionPoint(returnOp);

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
      state.aliasInfo.insertNewBufferEquivalence(returnTensor, v);
      state.mapBuffer(returnTensor, v);
    }
    return success();
  }
};

} // namespace std_ext
} // namespace comprehensive_bufferize
} // namespace linalg
} // namespace mlir

void mlir::linalg::comprehensive_bufferize::std_ext::
    registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addOpInterface<CallOp, std_ext::CallOpInterface>();
  registry.addOpInterface<ReturnOp, std_ext::ReturnOpInterface>();
  registry.addOpInterface<FuncOp, AllocationHoistingBarrierOnly<FuncOp>>();
}

LogicalResult mlir::linalg::comprehensive_bufferize::runComprehensiveBufferize(
    ModuleOp moduleOp, const BufferizationOptions &options) {
  SmallVector<FuncOp> orderedFuncOps;
  DenseMap<FuncOp, DenseSet<Operation *>> callerMap;
  if (failed(getFuncOpsOrderedByCalls(moduleOp, orderedFuncOps, callerMap)))
    return failure();

  ModuleBufferizationState state(moduleOp, *options.allocationFns);
  BufferizationAliasInfo &aliasInfo = state.aliasInfo;

  // Interestingly, all function args that are not visible outside of a module
  // can be fully bufferized inplace by guaranteeing the CallOp is bufferized
  // inplace. Therefore, we just bufferize funcOp as if none of its results were
  // inplaceable, detect which operands are cloned internally and decide what to
  // do at call sites.
  for (FuncOp funcOp : orderedFuncOps) {
    // No body => no analysis.
    if (funcOp.body().empty())
      continue;

    // In a first approximation:
    // =========================
    // If the function is called, we can allocate on the caller side which lets
    // us force inplace arguments at function boundaries.
    // TODO: do not rely on this behavior.
    if (callerMap.find(funcOp) != callerMap.end())
      for (BlockArgument bbArg : funcOp.getArguments())
        if (bbArg.getType().isa<TensorType>())
          aliasInfo.setBufferizesToWritableMemory(bbArg);

    // Analyze and bufferize funcOp.
    if (failed(runComprehensiveBufferize(funcOp, options, state)))
      return failure();
  }

  if (options.testAnalysisOnly)
    return success();

  for (FuncOp funcOp : orderedFuncOps) {
    // Note: It would be good to apply cleanups here but we cannot as aliasInfo
    // would be invalidated.
    if (failed(bufferizeFuncOpBoundary(funcOp, aliasInfo,
                                       state.bufferizedFunctionTypes)))
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
