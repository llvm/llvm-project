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

enum class ScalarizationState { unknown, visiting, blocked, rewritable };

struct ScalarizableFunctionInfo {
  RankedTensorType tensorType;
  SmallVector<func::ReturnOp> returnOps;
};

static FailureOr<ScalarizableFunctionInfo>
getScalarizableFunctionInfo(func::FuncOp funcOp) {
  // Only private function definitions with one single-element ranked tensor
  // result are locally eligible for scalarization.
  if (funcOp.isDeclaration() || !funcOp.isPrivate())
    return failure();

  FunctionType functionType = funcOp.getFunctionType();
  if (functionType.getNumResults() != 1)
    return failure();

  auto tensorType = dyn_cast<RankedTensorType>(functionType.getResult(0));
  if (!tensorType || !tensorType.hasStaticShape() ||
      tensorType.getNumElements() != 1)
    return failure();

  ScalarizableFunctionInfo info{tensorType, {}};
  for (Block &block : funcOp.getBody()) {
    auto returnOp = dyn_cast<func::ReturnOp>(block.getTerminator());
    if (!returnOp)
      return failure();
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
  // Functions are appended here when proven rewritable by the DFS. The final
  // rewrite walks this list in reverse so callees are rewritten before private
  // callers that may need their updated call signatures.
  SmallVector<func::FuncOp> rewriteOrder;
};

static bool isDirectFunctionInModule(func::FuncOp funcOp,
                                     const ScalarizationAnalysis &analysis) {
  return analysis.moduleFunctions.contains(funcOp);
}

static ScalarizationState
computeScalarizationState(func::FuncOp funcOp,
                          ScalarizationAnalysis &analysis) {
  auto [it, inserted] =
      analysis.states.try_emplace(funcOp, ScalarizationState::unknown);
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
    analysis.states[funcOp] = ScalarizationState::blocked;
    return ScalarizationState::blocked;
  };
  auto setRewritable = [&] {
    analysis.states[funcOp] = ScalarizationState::rewritable;
    analysis.rewriteOrder.push_back(funcOp);
    return ScalarizationState::rewritable;
  };

  if (!analysis.candidateInfos.contains(funcOp))
    return setBlocked();
  analysis.states[funcOp] = ScalarizationState::visiting;

  SmallVector<func::CallOp> callUsers;
  for (Operation *user : analysis.userMap.getUsers(funcOp.getOperation())) {
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
    // computeScalarizationState assumes getScalarizableFunctionInfo has already
    // categorized every direct func.func in the current module, so any direct
    // caller in this module must already appear in the analysis tables.
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

  analysis.callUsers.try_emplace(funcOp, std::move(callUsers));
  return setRewritable();
}

static void analyzeModule(ModuleOp module, ScalarizationAnalysis &analysis) {
  // First collect every direct function in the module and record the subset
  // that is locally eligible. The second pass runs the memoized DFS only for
  // candidates to determine the transitive blocked/rewritable state and build
  // the rewrite order.
  for (func::FuncOp funcOp : module.getOps<func::FuncOp>()) {
    analysis.moduleFunctions.insert(funcOp);
    FailureOr<ScalarizableFunctionInfo> info =
        getScalarizableFunctionInfo(funcOp);
    if (succeeded(info))
      analysis.candidateInfos.try_emplace(funcOp, std::move(*info));
  }

  for (func::FuncOp funcOp : module.getOps<func::FuncOp>())
    if (analysis.candidateInfos.contains(funcOp))
      (void)computeScalarizationState(funcOp, analysis);
}

static void rewriteScalarizableFunction(func::FuncOp funcOp,
                                        const ScalarizableFunctionInfo &info,
                                        ArrayRef<func::CallOp> callOps,
                                        RewriterBase &rewriter) {
  // Scalarize eligible functions, as decided by analyzeModule: extract the
  // unique element before each return, update the function type, and fix direct
  // call users that were recorded during analysis.
  RankedTensorType tensorType = info.tensorType;
  SmallVector<Value> zeroIndices;
  if (tensorType.getRank() != 0) {
    rewriter.setInsertionPointToStart(&funcOp.getBody().front());
    Value zero = arith::ConstantIndexOp::create(rewriter, funcOp.getLoc(), 0);
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

  FunctionType functionType = funcOp.getFunctionType();
  // This is a 1-result to 1-result type replacement, so the existing result
  // attribute dictionary remains attached to result #0 without reordering,
  // hence the rewrite is done directly without function_interface methods.
  funcOp.setType(FunctionType::get(
      funcOp.getContext(), functionType.getInputs(), TypeRange{scalarType}));

  for (func::CallOp callOp : callOps) {
    rewriter.setInsertionPoint(callOp);
    func::CallOp newCallOp = func::CallOp::create(rewriter, callOp.getLoc(),
                                                  funcOp, callOp.getOperands());
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

/// Scalarizes private functions that return a statically-shaped ranked
/// tensor with exactly one element.
///
/// The transform first analyzes module-wide symbol users and only rewrites
/// functions whose transitive private call users can be updated safely.
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
///
/// Public callers and callees, non-call symbol users, and recursive cycles
/// conservatively block scalarization.
static LogicalResult
MLGOScalarizeSingleElementTensorReturns(ModuleOp module,
                                        RewriterBase &rewriter) {
  // This pass is intentionally run as a direct module analysis + rewrite,
  // rather than through matchAndRewrite on a ModuleOp pattern. The transform
  // depends on module-scoped SymbolUserMap state and on a precomputed DFS
  // result, so it does not expose a pattern-testing / transform-dialect pattern
  // path.
  SymbolTableCollection symbolTable;
  // Take a snapshot of symbol users for the original module. This is
  // safe because the snapshot is consulted only during analyzeModule, before
  // any rewriting starts, and the rewrite phase relies exclusively on the
  // cached analysis result instead of querying SymbolUserMap again. It is also
  // safe with pass-manager multithreading: this pass invocation has exclusive
  // access to the current ModuleOp, so no other pass mutates the same module
  // concurrently and invalidates the snapshot underneath this analysis.
  SymbolUserMap userMap(symbolTable, module);
  ScalarizationAnalysis analysis(userMap);
  analyzeModule(module, analysis);

  for (func::FuncOp funcOp : llvm::reverse(analysis.rewriteOrder)) {
    const ScalarizableFunctionInfo &info =
        analysis.candidateInfos.find(funcOp)->second;
    ArrayRef<func::CallOp> callOps = analysis.callUsers.find(funcOp)->second;
    rewriteScalarizableFunction(funcOp, info, callOps, rewriter);
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
