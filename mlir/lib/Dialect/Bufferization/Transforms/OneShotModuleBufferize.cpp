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
#include "mlir/IR/BuiltinTypes.h"
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

namespace {

/// Annotate IR with the results of the analysis. For testing purposes only.
static void annotateEquivalentReturnBbArg(OpOperand &returnVal,
                                          BlockArgument bbArg) {
  const char *kEquivalentArgsAttr = "__equivalent_func_args__";
  Operation *op = returnVal.getOwner();

  SmallVector<int64_t> equivBbArgs;
  if (op->hasAttr(kEquivalentArgsAttr)) {
    auto attr = cast<ArrayAttr>(op->getAttr(kEquivalentArgsAttr));
    equivBbArgs = llvm::to_vector<4>(llvm::map_range(attr, [](Attribute a) {
      return cast<IntegerAttr>(a).getValue().getSExtValue();
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
      if (!isa<TensorType>(inputIt.value()))
        continue;
      for (const auto &resultIt : llvm::enumerate(type.getResults())) {
        if (!isa<TensorType>(resultIt.value()))
          continue;
        int64_t returnIdx = resultIt.index();
        int64_t bbArgIdx = inputIt.index();
        funcState.aliasingReturnVals[funcOp][bbArgIdx].push_back(returnIdx);
      }
    }
    return success();
  }

  // Find all func.return ops.
  SmallVector<func::ReturnOp> returnOps = getReturnOps(funcOp);
  assert(!returnOps.empty() && "expected at least one ReturnOp");

  // Build alias sets. Merge all aliases from all func.return ops.
  for (BlockArgument bbArg : funcOp.getArguments()) {
    if (isa<RankedTensorType>(bbArg.getType())) {
      int64_t bbArgIdx = bbArg.getArgNumber();
      // Store aliases in a set, so that we don't add the same alias twice.
      SetVector<int64_t> aliases;
      for (func::ReturnOp returnOp : returnOps) {
        for (OpOperand &returnVal : returnOp->getOpOperands()) {
          if (isa<RankedTensorType>(returnVal.get().getType())) {
            int64_t returnIdx = returnVal.getOperandNumber();
            if (state.areAliasingBufferizedValues(returnVal.get(), bbArg))
              aliases.insert(returnIdx);
          }
        }
      }
      for (int64_t alias : aliases)
        funcState.aliasingReturnVals[funcOp][bbArgIdx].push_back(alias);
    }
  }

  // Build equivalence sets.
  // Helper function that finds an equivalent block argument index for the
  // given OpOperand. Return std::nullopt if no equivalent block argument could
  // be found.
  auto findEquivalentBlockArgIdx =
      [&](OpOperand &opOperand) -> std::optional<int64_t> {
    Value v = opOperand.get();
    if (!isa<TensorType>(v.getType()))
      return std::nullopt;
    for (BlockArgument bbArg : funcOp.getArguments()) {
      if (isa<RankedTensorType>(bbArg.getType())) {
        if (state.areEquivalentBufferizedValues(v, bbArg)) {
          if (state.getOptions().testAnalysisOnly)
            annotateEquivalentReturnBbArg(opOperand, bbArg);
          return bbArg.getArgNumber();
        }
      }
    }
    return std::nullopt;
  };

  int64_t numResults = returnOps.front()->getNumOperands();
  for (int64_t i = 0; i < numResults; ++i) {
    // Find the equivalent block argument index for the i-th operand of the
    // first func.return op.
    std::optional<int64_t> maybeEquiv =
        findEquivalentBlockArgIdx(returnOps.front()->getOpOperand(i));
    if (!maybeEquiv.has_value())
      continue;
    int64_t bbArgIdx = *maybeEquiv;
    bool allEquiv = true;

    // Check if all other func.return ops have the same equivalent block
    // argument for the i-th operand. In contrast to aliasing information,
    // which is just "merged", equivalence information must match across all
    // func.return ops.
    for (func::ReturnOp returnOp : ArrayRef(returnOps).drop_front()) {
      std::optional<int64_t> maybeEquiv =
          findEquivalentBlockArgIdx(returnOp->getOpOperand(i));
      if (maybeEquiv != bbArgIdx) {
        allEquiv = false;
        break;
      }
    }

    // All func.return ops have the same equivalent block argument for the i-th
    // operand.
    if (allEquiv)
      funcState.equivalentFuncArgs[funcOp][i] = bbArgIdx;
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
    if (!isa<TensorType>(funcOp.getFunctionType().getInput(idx)))
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
static func::FuncOp getCalledFunction(func::CallOp callOp) {
  SymbolRefAttr sym =
      llvm::dyn_cast_if_present<SymbolRefAttr>(callOp.getCallableForCallee());
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
      state.unionEquivalenceClasses(returnVal, argVal);
    }

    return WalkResult::advance();
  });
}

/// Return "true" if the given function signature has tensor semantics.
static bool hasTensorSignature(func::FuncOp funcOp) {
  return llvm::any_of(funcOp.getFunctionType().getInputs(),
                      llvm::IsaPred<TensorType>) ||
         llvm::any_of(funcOp.getFunctionType().getResults(),
                      llvm::IsaPred<TensorType>);
}

/// Store all functions of the `moduleOp` in `orderedFuncOps`, sorted by
/// callee-caller order (i.e., callees without callers first). Store all
/// remaining functions (i.e., the ones that call each other recursively) in
/// `remainingFuncOps`.
///
/// Store the map of FuncOp to all its callers in `callerMap`.
///
/// Return `failure()` if we are unable to retrieve the called FuncOp from
/// any func::CallOp.
static LogicalResult getFuncOpsOrderedByCalls(
    ModuleOp moduleOp, SmallVectorImpl<func::FuncOp> &orderedFuncOps,
    SmallVectorImpl<func::FuncOp> &remainingFuncOps, FuncCallerMap &callerMap) {
  // For each FuncOp, the set of functions called by it (i.e. the union of
  // symbols of all nested func::CallOp).
  DenseMap<func::FuncOp, DenseSet<func::FuncOp>> calledBy;
  // For each FuncOp, the number of func::CallOp it contains.
  DenseMap<func::FuncOp, unsigned> numberCallOpsContainedInFuncOp;
  WalkResult res = moduleOp.walk([&](func::FuncOp funcOp) -> WalkResult {
    // Collect function calls and populate the caller map.
    numberCallOpsContainedInFuncOp[funcOp] = 0;
    return funcOp.walk([&](func::CallOp callOp) -> WalkResult {
      func::FuncOp calledFunction = getCalledFunction(callOp);
      assert(calledFunction && "could not retrieved called func::FuncOp");
      // If the called function does not have any tensors in its signature, then
      // it is not necessary to bufferize the callee before the caller.
      if (!hasTensorSignature(calledFunction))
        return WalkResult::skip();

      callerMap[calledFunction].insert(callOp);
      if (calledBy[calledFunction].insert(funcOp).second) {
        numberCallOpsContainedInFuncOp[funcOp]++;
      }
      return WalkResult::advance();
    });
  });
  if (res.wasInterrupted())
    return failure();

  // Iteratively remove function operations that do not call any of the
  // functions remaining in the callCounter map and add them to ordered list.
  while (!numberCallOpsContainedInFuncOp.empty()) {
    auto it = llvm::find_if(numberCallOpsContainedInFuncOp,
                            [](auto entry) { return entry.getSecond() == 0; });
    if (it == numberCallOpsContainedInFuncOp.end())
      break;
    orderedFuncOps.push_back(it->getFirst());
    for (auto callee : calledBy[it->getFirst()])
      numberCallOpsContainedInFuncOp[callee]--;
    numberCallOpsContainedInFuncOp.erase(it);
  }

  // Put all other functions in the list of remaining functions. These are
  // functions that call each other circularly.
  for (auto it : numberCallOpsContainedInFuncOp)
    remainingFuncOps.push_back(it.first);

  return success();
}

/// Helper function that extracts the source from a memref.cast. If the given
/// value is not a memref.cast result, simply returns the given value.
static Value unpackCast(Value v) {
  auto castOp = v.getDefiningOp<memref::CastOp>();
  if (!castOp)
    return v;
  return castOp.getSource();
}

/// Helper function that returns the return types (skipping casts) of the given
/// func.return ops. This function returns as many types as the return ops have
/// operands. If the i-th operand is not the same for all func.return ops, then
/// the i-th returned type is an "empty" type.
static SmallVector<Type> getReturnTypes(SmallVector<func::ReturnOp> returnOps) {
  assert(!returnOps.empty() && "expected at least one ReturnOp");
  int numOperands = returnOps.front()->getNumOperands();

  // Helper function that unpacks memref.cast ops and returns the type.
  auto getSourceType = [&](Value v) { return unpackCast(v).getType(); };

  SmallVector<Type> result;
  for (int i = 0; i < numOperands; ++i) {
    // Get the type of the i-th operand of the first func.return ops.
    Type t = getSourceType(returnOps.front()->getOperand(i));

    // Check if all other func.return ops have a matching operand type.
    for (int j = 1; j < static_cast<int>(returnOps.size()); ++j)
      if (getSourceType(returnOps[j]->getOperand(i)) != t)
        t = Type();

    result.push_back(t);
  }

  return result;
}

/// Fold return values that are memref casts and update function return types.
///
/// During FuncOp bufferization, the exact type of the returned memrefs (if any)
/// is not known yet. Therefore, the bufferization uses memref types with the
/// most generic layout map as function return types. After bufferizing the
/// entire function body, a more concise memref type can potentially be used for
/// the return type of the function.
static void foldMemRefCasts(func::FuncOp funcOp) {
  // There is nothing to do for bodiless ops.
  if (funcOp.getBody().empty())
    return;

  // Compute the common result types of all return ops.
  SmallVector<func::ReturnOp> returnOps = getReturnOps(funcOp);
  SmallVector<Type> resultTypes = getReturnTypes(returnOps);

  // Remove direct casts.
  for (func::ReturnOp returnOp : returnOps) {
    for (OpOperand &operand : returnOp->getOpOperands()) {
      // Bail if no common result type was found.
      if (resultTypes[operand.getOperandNumber()]) {
        operand.set(unpackCast(operand.get()));
      }
    }
  }

  // Fill in the missing result types that were not the same among all
  // func.return ops.
  for (int i = 0; i < static_cast<int>(resultTypes.size()); ++i) {
    if (resultTypes[i])
      continue;
    resultTypes[i] = funcOp.getFunctionType().getResult(i);
  }

  // Update the function type.
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

  // A list of non-circular functions in the order in which they are analyzed
  // and bufferized.
  SmallVector<func::FuncOp> orderedFuncOps;
  // A list of all other functions. I.e., functions that call each other
  // recursively. For these, we analyze the function body but not the function
  // boundary.
  SmallVector<func::FuncOp> remainingFuncOps;

  // A mapping of FuncOps to their callers.
  FuncCallerMap callerMap;

  if (failed(getFuncOpsOrderedByCalls(moduleOp, orderedFuncOps,
                                      remainingFuncOps, callerMap)))
    return failure();

  // Analyze functions in order. Starting with functions that are not calling
  // any other functions.
  for (func::FuncOp funcOp : orderedFuncOps) {
    if (!state.getOptions().isOpAllowed(funcOp))
      continue;

    // Now analyzing function.
    funcState.startFunctionAnalysis(funcOp);

    // Gather equivalence info for CallOps.
    equivalenceAnalysis(funcOp, state, funcState);

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

  // Analyze all other functions. All function boundary analyses are skipped.
  for (func::FuncOp funcOp : remainingFuncOps) {
    if (!state.getOptions().isOpAllowed(funcOp))
      continue;

    // Gather equivalence info for CallOps.
    equivalenceAnalysis(funcOp, state, funcState);

    // Analyze funcOp.
    if (failed(analyzeOp(funcOp, state, statistics)))
      return failure();

    // TODO: We currently skip all function argument analyses for functions
    // that call each other circularly. These analyses do not support recursive
    // calls yet. The `BufferizableOpInterface` implementations of `func`
    // dialect ops return conservative results in the absence of analysis
    // information.
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

  // A list of non-circular functions in the order in which they are analyzed
  // and bufferized.
  SmallVector<func::FuncOp> orderedFuncOps;
  // A list of all other functions. I.e., functions that call each other
  // recursively. For these, we analyze the function body but not the function
  // boundary.
  SmallVector<func::FuncOp> remainingFuncOps;

  // A mapping of FuncOps to their callers.
  FuncCallerMap callerMap;

  // Try to bufferize functions in calling order. I.e., first bufferize
  // functions that do not call other functions. This allows us to infer
  // accurate buffer types for function return values. Functions that call
  // each other recursively are bufferized in an unspecified order at the end.
  // We may use unnecessarily "complex" (in terms of layout map) buffer types.
  if (failed(getFuncOpsOrderedByCalls(moduleOp, orderedFuncOps,
                                      remainingFuncOps, callerMap)))
    return failure();
  llvm::append_range(orderedFuncOps, remainingFuncOps);

  // Bufferize functions.
  for (func::FuncOp funcOp : orderedFuncOps) {
    // Note: It would be good to apply cleanups here but we cannot as aliasInfo
    // would be invalidated.

    if (llvm::is_contained(options.noAnalysisFuncFilter, funcOp.getSymName())) {
      // This function was not analyzed and RaW conflicts were not resolved.
      // Buffer copies must be inserted before every write.
      OneShotBufferizationOptions updatedOptions = options;
      updatedOptions.copyBeforeWrite = true;
      if (failed(bufferizeOp(funcOp, updatedOptions, statistics)))
        return failure();
    } else {
      if (failed(bufferizeOp(funcOp, options, statistics)))
        return failure();
    }

    // Change buffer return types to more precise layout maps.
    if (options.inferFunctionResultLayout)
      foldMemRefCasts(funcOp);
  }

  // Bufferize all other ops.
  for (Operation &op : llvm::make_early_inc_range(moduleOp.getOps())) {
    // Functions were already bufferized.
    if (isa<func::FuncOp>(&op))
      continue;
    if (failed(bufferizeOp(&op, options, statistics)))
      return failure();
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
    if (options.noAnalysisFuncFilter.empty()) {
      if (failed(insertTensorCopies(moduleOp, options, statistics)))
        return failure();
    } else {
      // FuncOps whose names are specified in options.noAnalysisFuncFilter will
      // not be analyzed. Ops in these FuncOps will not be analyzed as well.
      OpFilter::Entry::FilterFn analysisFilterFn = [=](Operation *op) {
        auto func = dyn_cast<func::FuncOp>(op);
        if (!func)
          func = op->getParentOfType<func::FuncOp>();
        if (func)
          return llvm::is_contained(options.noAnalysisFuncFilter,
                                    func.getSymName());
        return false;
      };
      OneShotBufferizationOptions updatedOptions(options);
      updatedOptions.opFilter.denyOperation(analysisFilterFn);
      if (failed(insertTensorCopies(moduleOp, updatedOptions, statistics)))
        return failure();
    }
  }
  if (options.testAnalysisOnly)
    return success();
  if (failed(bufferizeModuleOp(moduleOp, options, statistics)))
    return failure();
  return success();
}
