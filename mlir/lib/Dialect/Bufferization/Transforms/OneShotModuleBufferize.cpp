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
// Module Bufferization is run via `runOneShotModuleBufferize(ModuleOp, ...)`.
// This function analyzes the given module and determines the order of analysis
// and bufferization: Functions that are called are processed before their
// respective callers.
//
// After analyzing a FuncOp, additional information about its bbArgs is
// gathered and stored in `FuncAnalysisState`.
//
// * `aliasingFuncOpBBArgsAnalysis` determines the equivalent/aliasing bbArgs
// for
//   each tensor return value (if any).
// * `funcOpBbArgReadWriteAnalysis` determines whether or not a tensor bbArg is
//   read/written.
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

#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::bufferization::func_ext;

/// A mapping of FuncOps to their callers.
using FuncCallerMap = DenseMap<func::FuncOp, DenseSet<Operation *>>;

/// Get or create FuncAnalysisState.
static FuncAnalysisState &
getOrCreateFuncAnalysisState(OneShotAnalysisState &state) {
  auto *result = state.getExtension<FuncAnalysisState>();
  if (result)
    return *result;
  return state.addExtension<FuncAnalysisState>();
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
aliasingFuncOpBBArgsAnalysis(FuncOp funcOp, OneShotAnalysisState &state,
                             FuncAnalysisState &funcState) {
  if (funcOp.getBody().empty()) {
    // No function body available. Conservatively assume that every tensor
    // return value may alias with any tensor bbArg.
    FunctionType type = funcOp.getFunctionType();
    for (const auto &inputIt : llvm::enumerate(type.getInputs())) {
      if (!inputIt.value().isa<TensorType>())
        continue;
      for (const auto &resultIt : llvm::enumerate(type.getResults())) {
        if (!resultIt.value().isa<TensorType>())
          continue;
        int64_t returnIdx = resultIt.index();
        int64_t bbArgIdx = inputIt.index();
        funcState.aliasingReturnVals[funcOp][bbArgIdx].push_back(returnIdx);
      }
    }
    return success();
  }

  // Support only single return-terminated block in the function.
  func::ReturnOp returnOp = getAssumedUniqueReturnOp(funcOp);
  assert(returnOp && "expected func with single return op");

  for (OpOperand &returnVal : returnOp->getOpOperands())
    if (returnVal.get().getType().isa<RankedTensorType>())
      for (BlockArgument bbArg : funcOp.getArguments())
        if (bbArg.getType().isa<RankedTensorType>()) {
          int64_t returnIdx = returnVal.getOperandNumber();
          int64_t bbArgIdx = bbArg.getArgNumber();
          if (state.areEquivalentBufferizedValues(returnVal.get(), bbArg)) {
            funcState.equivalentFuncArgs[funcOp][returnIdx] = bbArgIdx;
            if (state.getOptions().testAnalysisOnly)
              annotateEquivalentReturnBbArg(returnVal, bbArg);
          }
          if (state.areAliasingBufferizedValues(returnVal.get(), bbArg))
            funcState.aliasingReturnVals[funcOp][bbArgIdx].push_back(returnIdx);
        }

  return success();
}

static void annotateFuncArgAccess(func::FuncOp funcOp, int64_t idx, bool isRead,
                                  bool isWritten) {
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
  funcOp.setArgAttr(idx, BufferizationDialect::kBufferAccessAttrName,
                    accessType);
}

/// Determine which FuncOp bbArgs are read and which are written. When run on a
/// function with unknown ops, we conservatively assume that such ops bufferize
/// to a read + write.
static LogicalResult
funcOpBbArgReadWriteAnalysis(FuncOp funcOp, OneShotAnalysisState &state,
                             FuncAnalysisState &funcState) {
  for (int64_t idx = 0, e = funcOp.getFunctionType().getNumInputs(); idx < e;
       ++idx) {
    // Skip non-tensor arguments.
    if (!funcOp.getFunctionType().getInput(idx).isa<TensorType>())
      continue;
    bool isRead;
    bool isWritten;
    if (auto accessAttr = funcOp.getArgAttrOfType<StringAttr>(
            idx, BufferizationDialect::kBufferAccessAttrName)) {
      // Buffer access behavior is specified on the function. Skip the analysis.
      StringRef str = accessAttr.getValue();
      isRead = str == "read" || str == "read-write";
      isWritten = str == "write" || str == "read-write";
    } else if (funcOp.getBody().empty()) {
      // If the function has no body, conservatively assume that all args are
      // read + written.
      isRead = true;
      isWritten = true;
    } else {
      // Analyze the body of the function.
      BlockArgument bbArg = funcOp.getArgument(idx);
      isRead = state.isValueRead(bbArg);
      isWritten = state.isValueWritten(bbArg);
    }

    if (state.getOptions().testAnalysisOnly)
      annotateFuncArgAccess(funcOp, idx, isRead, isWritten);
    if (isRead)
      funcState.readBbArgs[funcOp].insert(idx);
    if (isWritten)
      funcState.writtenBbArgs[funcOp].insert(idx);
  }

  return success();
}
} // namespace

/// Remove bufferization attributes on FuncOp arguments.
static void removeBufferizationAttributes(BlockArgument bbArg) {
  auto funcOp = cast<func::FuncOp>(bbArg.getOwner()->getParentOp());
  funcOp.removeArgAttr(bbArg.getArgNumber(),
                       BufferizationDialect::kBufferLayoutAttrName);
  funcOp.removeArgAttr(bbArg.getArgNumber(),
                       BufferizationDialect::kWritableAttrName);
}

/// Return the func::FuncOp called by `callOp`.
static func::FuncOp getCalledFunction(CallOpInterface callOp) {
  SymbolRefAttr sym = callOp.getCallableForCallee().dyn_cast<SymbolRefAttr>();
  if (!sym)
    return nullptr;
  return dyn_cast_or_null<func::FuncOp>(
      SymbolTable::lookupNearestSymbolFrom(callOp, sym));
}

/// Gather equivalence info of CallOps.
/// Note: This only adds new equivalence info if the called function was already
/// analyzed.
// TODO: This does not handle cyclic function call graphs etc.
static void equivalenceAnalysis(func::FuncOp funcOp,
                                BufferizationAliasInfo &aliasInfo,
                                OneShotAnalysisState &state,
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
      if (!state.isInPlace(callOp->getOpOperand(bbargIdx)))
        continue;
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
      callerMap[calledFunction].insert(callOp);
      if (calledBy[calledFunction].insert(funcOp).second) {
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
      operand.set(castOp.getSource());
      resultTypes.push_back(castOp.getSource().getType());
    } else {
      resultTypes.push_back(operand.get().getType());
    }
  }

  auto newFuncType = FunctionType::get(
      funcOp.getContext(), funcOp.getFunctionType().getInputs(), resultTypes);
  funcOp.setType(newFuncType);
}

LogicalResult
mlir::bufferization::analyzeModuleOp(ModuleOp moduleOp,
                                     OneShotAnalysisState &state,
                                     BufferizationStatistics *statistics) {
  assert(state.getOptions().bufferizeFunctionBoundaries &&
         "expected that function boundary bufferization is activated");
  FuncAnalysisState &funcState = getOrCreateFuncAnalysisState(state);
  BufferizationAliasInfo &aliasInfo = state.getAliasInfo();

  // A list of functions in the order in which they are analyzed + bufferized.
  SmallVector<func::FuncOp> orderedFuncOps;

  // A mapping of FuncOps to their callers.
  FuncCallerMap callerMap;

  if (failed(getFuncOpsOrderedByCalls(moduleOp, orderedFuncOps, callerMap)))
    return failure();

  // Analyze ops.
  for (func::FuncOp funcOp : orderedFuncOps) {
    // Now analyzing function.
    funcState.startFunctionAnalysis(funcOp);

    // Gather equivalence info for CallOps.
    equivalenceAnalysis(funcOp, aliasInfo, state, funcState);

    // Analyze funcOp.
    if (failed(analyzeOp(funcOp, state, statistics)))
      return failure();

    // Run some extra function analyses.
    if (failed(aliasingFuncOpBBArgsAnalysis(funcOp, state, funcState)) ||
        failed(funcOpBbArgReadWriteAnalysis(funcOp, state, funcState)))
      return failure();

    // Mark op as fully analyzed.
    funcState.analyzedFuncOps[funcOp] = FuncOpAnalysisState::Analyzed;
  }

  return success();
}

void mlir::bufferization::removeBufferizationAttributesInModule(
    ModuleOp moduleOp) {
  moduleOp.walk([&](func::FuncOp op) {
    for (BlockArgument bbArg : op.getArguments())
      removeBufferizationAttributes(bbArg);
  });
}

LogicalResult mlir::bufferization::bufferizeModuleOp(
    ModuleOp moduleOp, const OneShotBufferizationOptions &options,
    BufferizationStatistics *statistics) {
  assert(options.bufferizeFunctionBoundaries &&
         "expected that function boundary bufferization is activated");
  IRRewriter rewriter(moduleOp.getContext());

  // A list of functions in the order in which they are analyzed + bufferized.
  SmallVector<func::FuncOp> orderedFuncOps;

  // A mapping of FuncOps to their callers.
  FuncCallerMap callerMap;

  if (failed(getFuncOpsOrderedByCalls(moduleOp, orderedFuncOps, callerMap)))
    return failure();

  // Bufferize functions.
  for (func::FuncOp funcOp : orderedFuncOps) {
    // Note: It would be good to apply cleanups here but we cannot as aliasInfo
    // would be invalidated.
    if (failed(bufferizeOp(funcOp, options, options.copyBeforeWrite,
                           /*opFilter=*/nullptr, statistics)))
      return failure();
    // Change buffer return types to more precise layout maps.
    if (options.functionBoundaryTypeConversion ==
        LayoutMapOption::InferLayoutMap)
      foldMemRefCasts(funcOp);
  }

  // Post-pass cleanup of function argument attributes.
  removeBufferizationAttributesInModule(moduleOp);

  return success();
}

LogicalResult mlir::bufferization::runOneShotModuleBufferize(
    ModuleOp moduleOp, const OneShotBufferizationOptions &options,
    BufferizationStatistics *statistics) {
  assert(options.bufferizeFunctionBoundaries &&
         "expected that function boundary bufferization is activated");
  assert(!(options.copyBeforeWrite && options.testAnalysisOnly) &&
         "invalid combination of bufferization flags");
  if (!options.copyBeforeWrite) {
    if (failed(insertTensorCopies(moduleOp, options, statistics)))
      return failure();
  }
  if (options.testAnalysisOnly)
    return success();
  if (failed(bufferizeModuleOp(moduleOp, options, statistics)))
    return failure();
  return success();
}
