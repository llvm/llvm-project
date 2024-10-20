//===- BufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/IR/UnstructuredControlFlow.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include <optional>

namespace mlir {
namespace bufferization {
namespace func_ext {

void FuncAnalysisState::startFunctionAnalysis(FunctionOpInterface funcOp) {
  analyzedFuncOps[funcOp] = FuncOpAnalysisState::InProgress;
  auto createdEquiv = equivalentFuncArgs.try_emplace(funcOp, IndexMapping());
  auto createdAliasingResults =
      aliasingReturnVals.try_emplace(funcOp, IndexToIndexListMapping());
  auto createdRead = readBbArgs.try_emplace(funcOp, BbArgIndexSet());
  auto createdWritten = writtenBbArgs.try_emplace(funcOp, BbArgIndexSet());
  (void)createdEquiv;
  (void)createdAliasingResults;
  (void)createdRead;
  (void)createdWritten;
#ifndef NDEBUG
  assert(createdEquiv.second && "equivalence info exists already");
  assert(createdAliasingResults.second && "aliasing info exists already");
  assert(createdRead.second && "bbarg access info exists already");
  assert(createdWritten.second && "bbarg access info exists already");
#endif // NDEBUG
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

/// Return the index-th bufferized function argument type. This assumes that the
/// specified argument is a tensor. If the tensor is ranked, a layout map may be
/// specified by the user (as per `options.functionArgTypeConverterFn`).
static BaseMemRefType
getBufferizedFunctionArgType(FuncOp funcOp, int64_t index,
                             const BufferizationOptions &options) {
  auto tensorType =
      dyn_cast<TensorType>(funcOp.getFunctionType().getInput(index));
  assert(tensorType && "expected TensorType");

  BaseMemRefType memrefType = options.functionArgTypeConverterFn(
      tensorType, *options.defaultMemorySpaceFn(tensorType), funcOp, options);

  auto layoutAttr = funcOp.getArgAttrOfType<AffineMapAttr>(
      index, BufferizationDialect::kBufferLayoutAttrName);
  if (!layoutAttr)
    return memrefType;

  auto rankedMemrefType = dyn_cast<MemRefType>(memrefType);
  assert(rankedMemrefType && "buffer layout not supported on unranked tensors");
  return MemRefType::get(
      rankedMemrefType.getShape(), rankedMemrefType.getElementType(),
      layoutAttr.getValue(), rankedMemrefType.getMemorySpace());
}

/// Return the FuncOp called by `callOp`.
static FuncOp getCalledFunction(CallOpInterface callOp) {
  SymbolRefAttr sym = llvm::dyn_cast_if_present<SymbolRefAttr>(callOp.getCallableForCallee());
  if (!sym)
    return nullptr;
  return dyn_cast_or_null<FuncOp>(
      SymbolTable::lookupNearestSymbolFrom(callOp, sym));
}

/// Get FuncAnalysisState.
static const FuncAnalysisState &
getFuncAnalysisState(const AnalysisState &state) {
  assert(isa<OneShotAnalysisState>(state) && "expected OneShotAnalysisState");
  auto *result = static_cast<const OneShotAnalysisState &>(state)
                     .getExtension<FuncAnalysisState>();
  assert(result && "FuncAnalysisState does not exist");
  return *result;
}

/// Return the state (phase) of analysis of the FuncOp.
static FuncOpAnalysisState getFuncOpAnalysisState(const AnalysisState &state,
                                                  FuncOp funcOp) {
  if (!isa<OneShotAnalysisState>(state))
    return FuncOpAnalysisState::NotAnalyzed;
  auto *funcState = static_cast<const OneShotAnalysisState &>(state)
                        .getExtension<FuncAnalysisState>();
  if (!funcState)
    return FuncOpAnalysisState::NotAnalyzed;
  const auto &analyzedFuncOps = funcState->analyzedFuncOps;
  auto it = analyzedFuncOps.find(funcOp);
  if (it == analyzedFuncOps.end())
    return FuncOpAnalysisState::NotAnalyzed;
  return it->second;
}

/// Return the index of the bbArg in the given FuncOp that is equivalent to the
/// specified return value (if any).
static std::optional<int64_t>
getEquivalentFuncArgIdx(FuncOp funcOp, const FuncAnalysisState &state,
                        int64_t returnValIdx) {
  auto funcOpIt = state.equivalentFuncArgs.find(funcOp);
  if (funcOpIt == state.equivalentFuncArgs.end())
    // No equivalence info stores for funcOp.
    return std::nullopt;

  auto retValIt = funcOpIt->getSecond().find(returnValIdx);
  if (retValIt == funcOpIt->getSecond().end())
    // Return value has no equivalent bbArg.
    return std::nullopt;

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

    if (getFuncOpAnalysisState(state, funcOp) != FuncOpAnalysisState::Analyzed)
      // FuncOp not analyzed yet. Assume that OpOperand is read.
      return true;

    const FuncAnalysisState &funcState = getFuncAnalysisState(state);
    return funcState.readBbArgs.lookup(funcOp).contains(
        opOperand.getOperandNumber());
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    func::CallOp callOp = cast<func::CallOp>(op);
    FuncOp funcOp = getCalledFunction(callOp);
    assert(funcOp && "expected CallOp to a FuncOp");

    if (getFuncOpAnalysisState(state, funcOp) != FuncOpAnalysisState::Analyzed)
      // FuncOp not analyzed yet. Assume that OpOperand is written.
      return true;

    const FuncAnalysisState &funcState = getFuncAnalysisState(state);
    return funcState.writtenBbArgs.lookup(funcOp).contains(
        opOperand.getOperandNumber());
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    func::CallOp callOp = cast<func::CallOp>(op);
    FuncOp funcOp = getCalledFunction(callOp);
    assert(funcOp && "expected CallOp to a FuncOp");
    if (getFuncOpAnalysisState(state, funcOp) != FuncOpAnalysisState::Analyzed)
      // FuncOp not analyzed yet. Any OpResult may be aliasing.
      return detail::unknownGetAliasingValues(opOperand);

    // Get aliasing results from state.
    const FuncAnalysisState &funcState = getFuncAnalysisState(state);
    auto aliasingReturnVals =
        funcState.aliasingReturnVals.lookup(funcOp).lookup(
            opOperand.getOperandNumber());

    // Check if the aliasing OpResult is equivalent to the OpOperand.
    std::optional<int64_t> equivalent = {};
    if (aliasingReturnVals.size() == 1) {
      equivalent = getEquivalentFuncArgIdx(funcOp, funcState,
                                           aliasingReturnVals.front());
      assert((!equivalent.has_value() ||
              *equivalent == opOperand.getOperandNumber()) &&
             "inconsistent analysis state");
    }
    AliasingValueList result;
    for (int64_t resultIdx : aliasingReturnVals)
      result.addAlias({callOp->getOpResult(resultIdx),
                       equivalent.has_value() ? BufferRelation::Equivalent
                                              : BufferRelation::Unknown,
                       /*isDefinite=*/equivalent.has_value()});
    return result;
  }

  FailureOr<BaseMemRefType>
  getBufferType(Operation *op, Value value, const BufferizationOptions &options,
                SmallVector<Value> &invocationStack) const {
    auto callOp = cast<func::CallOp>(op);
    FuncOp funcOp = getCalledFunction(callOp);
    assert(funcOp && "expected CallOp to a FuncOp");

    // The callee was already bufferized, so we can directly take the type from
    // its signature.
    FunctionType funcType = funcOp.getFunctionType();
    return cast<BaseMemRefType>(
        funcType.getResult(cast<OpResult>(value).getResultNumber()));
  }

  /// All function arguments are writable. It is the responsibility of the
  /// CallOp to insert buffer copies where necessary.
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    func::CallOp callOp = cast<func::CallOp>(op);

    // 1. Compute the result types of the new CallOp.
    SmallVector<Type> resultTypes;
    for (Value result : callOp.getResults()) {
      Type returnType = result.getType();
      if (!isa<TensorType>(returnType)) {
        // Non-tensor values are returned.
        resultTypes.push_back(returnType);
        continue;
      }

      // Returning a memref.
      FailureOr<BaseMemRefType> resultType =
          bufferization::getBufferType(result, options);
      if (failed(resultType))
        return failure();
      resultTypes.push_back(*resultType);
    }

    // 2. Rewrite tensor operands as memrefs based on type of the already
    //    bufferized callee.
    SmallVector<Value> newOperands;
    FuncOp funcOp = getCalledFunction(callOp);
    assert(funcOp && "expected CallOp to a FuncOp");
    FunctionType funcType = funcOp.getFunctionType();

    for (OpOperand &opOperand : callOp->getOpOperands()) {
      // Non-tensor operands are just copied.
      if (!isa<TensorType>(opOperand.get().getType())) {
        newOperands.push_back(opOperand.get());
        continue;
      }

      // Retrieve buffers for tensor operands.
      FailureOr<Value> maybeBuffer =
          getBuffer(rewriter, opOperand.get(), options);
      if (failed(maybeBuffer))
        return failure();
      Value buffer = *maybeBuffer;

      // Caller / callee type mismatch is handled with castOrReallocMemRefValue.
      auto memRefType = funcType.getInput(opOperand.getOperandNumber());
      // Since we don't yet have a clear layout story, to_memref may
      // conservatively turn tensors into more dynamic memref than necessary.
      // If the memref type of the callee fails, introduce an extra memref.cast
      // that will either canonicalize away or fail compilation until we can do
      // something better. Insert a reallocation + copy if it cannot be
      // statically guaranteed that a direct cast would be valid.
      if (buffer.getType() != memRefType) {
        auto memrefDstType = dyn_cast<MemRefType>(memRefType);
        assert(memrefDstType &&
               "buffer layout not supported on unranked tensors");
        FailureOr<Value> replacement = bufferization::castOrReallocMemRefValue(
            rewriter, buffer, memrefDstType, options);
        if (failed(replacement))
          return failure();
        buffer = *replacement;
      }
      newOperands.push_back(buffer);
    }

    // 3. Create the new CallOp.
    Operation *newCallOp = rewriter.create<func::CallOp>(
        callOp.getLoc(), funcOp.getSymName(), resultTypes, newOperands);
    newCallOp->setAttrs(callOp->getAttrs());

    // 4. Replace the old op with the new op.
    replaceOpWithBufferizedValues(rewriter, callOp, newCallOp->getResults());

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

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
#ifndef NDEBUG
    auto returnOp = cast<func::ReturnOp>(op);
    assert(isa<FuncOp>(returnOp->getParentOp()) &&
           "only support FuncOp parent for ReturnOp");
#endif // NDEBUG

    // ReturnOps are bufferized as part of FuncOps.
    return success();
  }
};

struct FuncOpInterface
    : public OpWithUnstructuredControlFlowBufferizableOpInterfaceExternalModel<
          FuncOpInterface, FuncOp> {

  static bool supportsUnstructuredControlFlow() { return true; }

  bool hasTensorSemantics(Operation *op) const {
    auto isaTensor = llvm::IsaPred<TensorType>;

    // A function has tensor semantics if it has tensor arguments/results.
    auto funcOp = cast<FuncOp>(op);
    bool hasTensorArg = any_of(funcOp.getArgumentTypes(), isaTensor);
    bool hasTensorResult = any_of(funcOp.getResultTypes(), isaTensor);
    if (hasTensorArg || hasTensorResult)
      return true;

    // It also has tensor semantics if it has tensor block arguments.
    // TODO: Decouple bufferization of unstructured control flow from
    // BufferizableOpInterface implementations. We should only care about
    // region entry block arguments here (which are already covered by the
    // argument types of the function).
    for (Block &block : funcOp.getBody())
      if (any_of(block.getArgumentTypes(), isaTensor))
        return true;

    return false;
  }

  AliasingOpOperandList
  getAliasingOpOperands(Operation *op, Value value,
                        const AnalysisState &state) const {
    return getAliasingBranchOpOperands(op, cast<BlockArgument>(value), state);
  }

  FailureOr<BaseMemRefType>
  getBufferType(Operation *op, Value value, const BufferizationOptions &options,
                SmallVector<Value> &invocationStack) const {
    auto funcOp = cast<FuncOp>(op);
    auto bbArg = cast<BlockArgument>(value);

    // Function arguments are special.
    if (bbArg.getOwner() == &funcOp.getBody().front())
      return getBufferizedFunctionArgType(funcOp, bbArg.getArgNumber(),
                                          options);

    return OpWithUnstructuredControlFlowBufferizableOpInterfaceExternalModel::
        getBufferType(op, value, options, invocationStack);
  }

  LogicalResult verifyAnalysis(Operation *op,
                               const AnalysisState &state) const {
    auto funcOp = cast<func::FuncOp>(op);
    // TODO: func.func with multiple returns are not supported.
    if (!getAssumedUniqueReturnOp(funcOp) && !funcOp.isExternal())
      return op->emitOpError("op without unique func.return is not supported");
    return success();
  }

  /// Rewrite function bbArgs and return values into buffer form. This function
  /// bufferizes the function signature and the ReturnOp. When the entire
  /// function body has been bufferized, function return types can be switched
  /// to more concise memref types as part of `foldMemRefCasts`.
  ///
  /// All function bbArgs are writable unless they are explicitly marked as
  /// read-only. Callers must insert copies when needed.
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto funcOp = cast<FuncOp>(op);
    FunctionType funcType = funcOp.getFunctionType();

    // Construct the bufferized function type.
    SmallVector<Type> argTypes;
    for (const auto &it : llvm::enumerate(funcType.getInputs())) {
      Type argType = it.value();
      if (dyn_cast<TensorType>(argType)) {
        argTypes.push_back(
            getBufferizedFunctionArgType(funcOp, it.index(), options));
        continue;
      }
      argTypes.push_back(argType);
    }

    // Bodiless functions are assumed opaque and we cannot know the
    // bufferization contract they want to enforce. As a consequence, only
    // support functions that don't return any tensors atm.
    if (funcOp.isExternal()) {
      SmallVector<Type> retTypes;
      for (Type resultType : funcType.getResults()) {
        if (isa<TensorType>(resultType))
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
    Location loc = returnOp.getLoc();

    // 1. Bufferize every block.
    for (Block &block : funcOp.getBody())
      if (failed(bufferization::bufferizeBlockSignature(&block, rewriter,
                                                        options)))
        return failure();

    // 2. For each result, keep track of which inplace argument it reuses.
    SmallVector<Value> returnValues;
    for (OpOperand &returnOperand : returnOp->getOpOperands()) {
      Value returnVal = returnOperand.get();
      auto tensorType = dyn_cast<TensorType>(returnVal.getType());
      rewriter.setInsertionPoint(returnOp);

      // If not a tensor type just forward it.
      if (!tensorType) {
        returnValues.push_back(returnVal);
        continue;
      }

      // Note: If `inferFunctionResultLayout = true`, cast are later folded
      // away.
      BaseMemRefType resultType = options.functionArgTypeConverterFn(
          tensorType, *options.defaultMemorySpaceFn(tensorType), funcOp,
          options);
      Value toMemrefOp = rewriter.create<bufferization::ToMemrefOp>(
          loc, resultType, returnVal);
      returnValues.push_back(toMemrefOp);
    }

    // 3. Rewrite the terminator without the in-place bufferizable values.
    returnOp.getOperandsMutable().assign(returnValues);

    // 4. Rewrite the FuncOp type to buffer form.
    funcOp.setType(FunctionType::get(op->getContext(), argTypes,
                                     ValueRange(returnValues).getTypes()));

    return success();
  }

  /// Return `true` if the given function argument is writable.
  bool isWritable(Operation *op, Value value,
                  const AnalysisState &state) const {
    auto funcOp = cast<FuncOp>(op);
    BlockArgument bbArg = dyn_cast<BlockArgument>(value);
    assert(bbArg && "expected BlockArgument");

    // Non-entry block arguments are always writable. (They may alias with
    // values that are not writable, which will turn them into read-only.)
    if (bbArg.getOwner() != &funcOp.getBody().front())
      return true;

    // "bufferization.writable" overrides other writability decisions. This is
    // currently used for testing only.
    if (BoolAttr writable = funcOp.getArgAttrOfType<BoolAttr>(
            bbArg.getArgNumber(), BufferizationDialect::kWritableAttrName))
      return writable.getValue();

    // All function arguments are writable by default.
    return true;
  }
};

} // namespace func_ext
} // namespace bufferization
} // namespace mlir

void mlir::bufferization::func_ext::
    registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, func::FuncDialect *dialect) {
    func::CallOp::attachInterface<func_ext::CallOpInterface>(*ctx);
    func::FuncOp::attachInterface<func_ext::FuncOpInterface>(*ctx);
    func::ReturnOp::attachInterface<func_ext::ReturnOpInterface>(*ctx);
  });
}
