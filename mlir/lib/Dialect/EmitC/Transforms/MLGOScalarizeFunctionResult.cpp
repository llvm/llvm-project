//===- MLGOScalarizeFunctionResult.cpp - Scalarize tensor returns ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/EmitC/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir {
namespace emitc {
#define GEN_PASS_DEF_MLGOSCALARIZESINGLEELEMENTTENSORRETURNPASS
#include "mlir/Dialect/EmitC/Transforms/Passes.h.inc"
} // namespace emitc
} // namespace mlir

using namespace mlir;

namespace {

enum class ScalarizationState {
  // No DFS classification has been computed for this function yet.
  unknown,
  // The function is currently on the recursive DFS stack. Re-entering this
  // state means a cycle was found, which this pass conservatively blocks.
  visiting,
  // The function cannot be scalarized safely because either it is not locally
  // eligible or one of its transitive users blocks the rewrite.
  blocked,
  // The function is locally eligible and all transitive users considered by
  // this pass can be updated consistently.
  rewritable
};

struct ScalarizableFunctionInfo {
  RankedTensorType tensorType;
  SmallVector<func::ReturnOp> returnOps;
};

// Returns per-function scalarization info when this function is locally
// eligible, i.e. it is a private definition with one statically-shaped ranked
// tensor result containing exactly one element.
static FailureOr<ScalarizableFunctionInfo>
getScalarizableFunctionInfoIfEligible(func::FuncOp func) {
  if (func.isDeclaration() || !func.isPrivate())
    return failure();

  FunctionType functionType = func.getFunctionType();
  if (functionType.getNumResults() != 1)
    return failure();

  auto tensorType = dyn_cast<RankedTensorType>(functionType.getResult(0));
  if (!tensorType || !tensorType.hasStaticShape() ||
      tensorType.getNumElements() != 1)
    return failure();

  ScalarizableFunctionInfo info{tensorType, {}};
  for (Block &block : func.getBody()) {
    auto returnOp = dyn_cast<func::ReturnOp>(block.getTerminator());
    if (returnOp)
      info.returnOps.push_back(returnOp);
  }

  if (info.returnOps.empty())
    return failure();
  return info;
}

struct ScalarizationAnalysis {
  explicit ScalarizationAnalysis(SymbolUserMap &userMap) : userMap(userMap) {}

  SymbolUserMap &userMap;
  DenseSet<func::FuncOp> moduleFunctions;
  DenseMap<func::FuncOp, ScalarizableFunctionInfo> candidateInfos;
  DenseMap<func::FuncOp, SmallVector<func::CallOp>> callUsers;
  DenseMap<func::FuncOp, ScalarizationState> states;
  // DFS completion order for rewritable functions. The rewrite phase walks
  // this list in reverse so callees are rewritten before their call sites.
  SmallVector<func::FuncOp> rewriteOrder;
};

// Returns whether `func` is one of the direct function definitions in the
// current module-level analysis scope.
static bool isDirectFunctionInModule(func::FuncOp func,
                                     const ScalarizationAnalysis &analysis) {
  return analysis.moduleFunctions.contains(func);
}

// Runs the memoized DFS that classifies one candidate function by walking its
// transitive private call users.
static ScalarizationState
computeScalarizationState(func::FuncOp func, ScalarizationAnalysis &analysis) {
  auto [it, inserted] =
      analysis.states.try_emplace(func, ScalarizationState::unknown);
  if (!inserted) {
    // Conservatively reject recursive cycles instead of reasoning about SCCs.
    if (it->second == ScalarizationState::visiting)
      return ScalarizationState::blocked;
    return it->second;
  }

  // Starting from one locally-eligible private function, walk its symbol users
  // upward through private callers. Memoization avoids re-traversing the same
  // subgraph from the outer linear scan, and early blocking truncates the DFS
  // as soon as a public/unsupported user is found.
  auto setBlocked = [&] {
    analysis.states[func] = ScalarizationState::blocked;
    return ScalarizationState::blocked;
  };
  auto setRewritable = [&] {
    analysis.states[func] = ScalarizationState::rewritable;
    analysis.rewriteOrder.push_back(func);
    return ScalarizationState::rewritable;
  };

  if (!analysis.candidateInfos.contains(func))
    return setBlocked();
  analysis.states[func] = ScalarizationState::visiting;

  SmallVector<func::CallOp> callUsers;
  for (Operation *user : analysis.userMap.getUsers(func.getOperation())) {
    auto callOp = dyn_cast<func::CallOp>(user);
    // Non-call symbol uses, such as func.constant, prevent updating all users
    // consistently, so the current function stays blocked.
    if (!callOp)
      return setBlocked();
    callUsers.push_back(callOp);

    func::FuncOp caller = callOp->getParentOfType<func::FuncOp>();
    // Only direct callers in the same module participate in this analysis.
    if (!caller)
      return setBlocked();
    // `computeScalarizationState` assumes `getScalarizableFunctionInfo` has
    // already categorized every direct func.func in the current module, so any
    // direct caller in this module must already appear in the analysis tables.
    if (!isDirectFunctionInModule(caller, analysis))
      return setBlocked();
    // Public and external callers keep the current function blocked because the
    // pass cannot rewrite every visible call boundary.
    if (caller.isPublic())
      return setBlocked();
    if (caller.isExternal())
      return setBlocked();

    // A private non-candidate caller can absorb the scalarized call by
    // reboxing the scalar result back into a single-element tensor.
    if (!analysis.candidateInfos.contains(caller))
      continue;

    if (computeScalarizationState(caller, analysis) !=
        ScalarizationState::rewritable)
      return setBlocked();
  }

  analysis.callUsers.try_emplace(func, std::move(callUsers));
  return setRewritable();
}

// Builds the module-level analysis state used by the rewrite phase.
static void computeScalarizationAnalysis(ModuleOp module,
                                         ScalarizationAnalysis &analysis) {
  // First collect all direct functions and the subset that is locally
  // eligible.
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    analysis.moduleFunctions.insert(func);
    FailureOr<ScalarizableFunctionInfo> info =
        getScalarizableFunctionInfoIfEligible(func);
    if (succeeded(info))
      analysis.candidateInfos.try_emplace(func, std::move(*info));
  }
  // Then run the memoized DFS for candidate roots.
  for (func::FuncOp func : module.getOps<func::FuncOp>())
    if (analysis.candidateInfos.contains(func))
      (void)computeScalarizationState(func, analysis);
}

// Rewrites one function that has already been proven rewritable and updates
// the direct call users cached before any IR mutation started.
static void rewriteScalarizableFunction(func::FuncOp func,
                                        const ScalarizableFunctionInfo &info,
                                        ArrayRef<func::CallOp> callOps,
                                        RewriterBase &rewriter) {
  // Extract the unique element before each return, update the function type,
  // and fix direct call users that were recorded during analysis.
  RankedTensorType tensorType = info.tensorType;
  SmallVector<Value> zeroIndices;
  if (tensorType.getRank() != 0) {
    rewriter.setInsertionPointToStart(&func.getBody().front());
    Value zero = arith::ConstantIndexOp::create(rewriter, func.getLoc(), 0);
    zeroIndices.assign(tensorType.getRank(), zero);
  }

  Type scalarType = tensorType.getElementType();
  for (func::ReturnOp returnOp : info.returnOps) {
    assert(returnOp.getNumOperands() == 1 &&
           "func.return must have exactly one operand");
    assert(returnOp.getOperand(0).getType() == tensorType &&
           "func.return operand type must match the function result type");
    rewriter.setInsertionPoint(returnOp);
    Value scalar = rewriter.createOrFold<tensor::ExtractOp>(
        returnOp.getLoc(), returnOp.getOperand(0), zeroIndices);
    rewriter.replaceOpWithNewOp<func::ReturnOp>(returnOp, scalar);
  }

  FunctionType functionType = func.getFunctionType();
  // This is a 1-result to 1-result type replacement, so the existing result
  // attribute dictionary remains attached to result #0 without reordering,
  // hence the rewrite is done directly without function_interface methods.
  func.setType(FunctionType::get(func.getContext(), functionType.getInputs(),
                                 TypeRange{scalarType}));

  for (func::CallOp callOp : callOps) {
    rewriter.setInsertionPoint(callOp);
    func::CallOp newCallOp = func::CallOp::create(rewriter, callOp.getLoc(),
                                                  func, callOp.getOperands());
    newCallOp->setAttrs(callOp->getAttrs());

    if (!callOp.getResult(0).use_empty()) {
      Value wrappedResult =
          tensor::FromElementsOp::create(rewriter, callOp.getLoc(), tensorType,
                                         ValueRange{newCallOp.getResult(0)});
      rewriter.replaceOp(callOp, wrappedResult);
    } else {
      rewriter.eraseOp(callOp);
    }
  }
}

/// Drives the complete module-level transform: analyze the original module,
/// determine which private functions can be scalarized safely, then
/// rewrite only that precomputed set.
///
/// BEFORE (both callee and private caller rewritten)
///   private callee(x : tensor<1xT>) -> tensor<1xT> { return x }
///   private caller(x : tensor<1xT>) -> tensor<1xT> {
///     y = call callee(x)
///     return y
///   }
///
/// AFTER
///   private callee(x : tensor<1xT>) -> T {
///     return tensor.extract x[0]
///   }
///   private caller(x : tensor<1xT>) -> T {
///     y = call callee(x)
///     return y
///   }
///
/// BEFORE (callee rewritten, unchanged private caller reboxes)
///   private callee(x : tensor<1xT>) -> tensor<1xT> { return x }
///   private caller(x : tensor<1xT>, z : tensor<1xT>) -> tensor<1xT> {
///     y = call callee(x)
///     r = tensor_op(y, z)
///     return r
///   }
///
/// AFTER
///   private callee(x : tensor<1xT>) -> T {
///     return tensor.extract x[0]
///   }
///   private caller(x : tensor<1xT>, z : tensor<1xT>) -> tensor<1xT> {
///     y = call callee(x)
///     y_boxed = tensor.from_elements y
///     r = tensor_op(y_boxed, z)
///     return r
///   }
static LogicalResult
MLGOScalarizeSingleElementTensorReturns(ModuleOp module,
                                        RewriterBase &rewriter) {
  // The transform depends on module-scoped `SymbolUserMap` state and on a
  // precomputed DFS result, so it runs as a direct module analysis + rewrite
  // instead of exposing a pattern-testing / transform-dialect pattern path.
  SymbolTableCollection symbolTable;
  // Take a snapshot of symbol users for the original module. This is safe
  // because it is consulted only during `computeScalarizationAnalysis`, before
  // rewriting starts, and this pass invocation has exclusive access to the
  // current module.
  SymbolUserMap userMap(symbolTable, module);
  ScalarizationAnalysis analysis(userMap);
  computeScalarizationAnalysis(module, analysis);

  for (func::FuncOp func : llvm::reverse(analysis.rewriteOrder)) {
    const ScalarizableFunctionInfo &info =
        analysis.candidateInfos.find(func)->second;
    ArrayRef<func::CallOp> callOps = analysis.callUsers.find(func)->second;
    rewriteScalarizableFunction(func, info, callOps, rewriter);
  }

  return success();
}

struct MLGOScalarizeSingleElementTensorReturnPass
    : public emitc::impl::MLGOScalarizeSingleElementTensorReturnPassBase<
          MLGOScalarizeSingleElementTensorReturnPass> {
  using Base::Base;

  void runOnOperation() override {
    IRRewriter rewriter(&getContext());
    if (failed(
            MLGOScalarizeSingleElementTensorReturns(getOperation(), rewriter)))
      signalPassFailure();
  }
};

} // namespace
