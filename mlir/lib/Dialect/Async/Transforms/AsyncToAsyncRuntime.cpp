//===- AsyncToAsyncRuntime.cpp - Lower from Async to Async Runtime --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering from high level async operations to async.coro
// and async.runtime operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Async/Passes.h"

#include "PassDetail.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

namespace mlir {
#define GEN_PASS_DEF_ASYNCTOASYNCRUNTIME
#include "mlir/Dialect/Async/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::async;

#define DEBUG_TYPE "async-to-async-runtime"
// Prefix for functions outlined from `async.execute` op regions.
static constexpr const char kAsyncFnPrefix[] = "async_execute_fn";

namespace {

class AsyncToAsyncRuntimePass
    : public impl::AsyncToAsyncRuntimeBase<AsyncToAsyncRuntimePass> {
public:
  AsyncToAsyncRuntimePass() = default;
  void runOnOperation() override;
};

} // namespace

//===----------------------------------------------------------------------===//
// async.execute op outlining to the coroutine functions.
//===----------------------------------------------------------------------===//

/// Function targeted for coroutine transformation has two additional blocks at
/// the end: coroutine cleanup and coroutine suspension.
///
/// async.await op lowering additionaly creates a resume block for each
/// operation to enable non-blocking waiting via coroutine suspension.
namespace {
struct CoroMachinery {
  func::FuncOp func;

  // Async execute region returns a completion token, and an async value for
  // each yielded value.
  //
  //   %token, %result = async.execute -> !async.value<T> {
  //     %0 = arith.constant ... : T
  //     async.yield %0 : T
  //   }
  Value asyncToken; // token representing completion of the async region
  llvm::SmallVector<Value, 4> returnValues; // returned async values

  Value coroHandle; // coroutine handle (!async.coro.handle value)
  Block *entry;     // coroutine entry block
  Block *setError;  // switch completion token and all values to error state
  Block *cleanup;   // coroutine cleanup block
  Block *suspend;   // coroutine suspension block
};
} // namespace

/// Utility to partially update the regular function CFG to the coroutine CFG
/// compatible with LLVM coroutines switched-resume lowering using
/// `async.runtime.*` and `async.coro.*` operations. Adds a new entry block
/// that branches into preexisting entry block. Also inserts trailing blocks.
///
/// The result types of the passed `func` must start with an `async.token`
/// and be continued with some number of `async.value`s.
///
/// The func given to this function needs to have been preprocessed to have
/// either branch or yield ops as terminators. Branches to the cleanup block are
/// inserted after each yield.
///
/// See LLVM coroutines documentation: https://llvm.org/docs/Coroutines.html
///
///  - `entry` block sets up the coroutine.
///  - `set_error` block sets completion token and async values state to error.
///  - `cleanup` block cleans up the coroutine state.
///  - `suspend block after the @llvm.coro.end() defines what value will be
///    returned to the initial caller of a coroutine. Everything before the
///    @llvm.coro.end() will be executed at every suspension point.
///
/// Coroutine structure (only the important bits):
///
///   func @some_fn(<function-arguments>) -> (!async.token, !async.value<T>)
///   {
///     ^entry(<function-arguments>):
///       %token = <async token> : !async.token    // create async runtime token
///       %value = <async value> : !async.value<T> // create async value
///       %id = async.coro.id                      // create a coroutine id
///       %hdl = async.coro.begin %id              // create a coroutine handle
///       cf.br ^preexisting_entry_block
///
///     /*  preexisting blocks modified to branch to the cleanup block */
///
///     ^set_error: // this block created lazily only if needed (see code below)
///       async.runtime.set_error %token : !async.token
///       async.runtime.set_error %value : !async.value<T>
///       cf.br ^cleanup
///
///     ^cleanup:
///       async.coro.free %hdl // delete the coroutine state
///       cf.br ^suspend
///
///     ^suspend:
///       async.coro.end %hdl // marks the end of a coroutine
///       return %token, %value : !async.token, !async.value<T>
///   }
///
static CoroMachinery setupCoroMachinery(func::FuncOp func) {
  assert(!func.getBlocks().empty() && "Function must have an entry block");

  MLIRContext *ctx = func.getContext();
  Block *entryBlock = &func.getBlocks().front();
  Block *originalEntryBlock =
      entryBlock->splitBlock(entryBlock->getOperations().begin());
  auto builder = ImplicitLocOpBuilder::atBlockBegin(func->getLoc(), entryBlock);

  // ------------------------------------------------------------------------ //
  // Allocate async token/values that we will return from a ramp function.
  // ------------------------------------------------------------------------ //
  auto retToken = builder.create<RuntimeCreateOp>(TokenType::get(ctx)).result();

  llvm::SmallVector<Value, 4> retValues;
  for (auto resType : func.getCallableResults().drop_front())
    retValues.emplace_back(builder.create<RuntimeCreateOp>(resType).result());

  // ------------------------------------------------------------------------ //
  // Initialize coroutine: get coroutine id and coroutine handle.
  // ------------------------------------------------------------------------ //
  auto coroIdOp = builder.create<CoroIdOp>(CoroIdType::get(ctx));
  auto coroHdlOp =
      builder.create<CoroBeginOp>(CoroHandleType::get(ctx), coroIdOp.id());
  builder.create<cf::BranchOp>(originalEntryBlock);

  Block *cleanupBlock = func.addBlock();
  Block *suspendBlock = func.addBlock();

  // ------------------------------------------------------------------------ //
  // Coroutine cleanup block: deallocate coroutine frame, free the memory.
  // ------------------------------------------------------------------------ //
  builder.setInsertionPointToStart(cleanupBlock);
  builder.create<CoroFreeOp>(coroIdOp.id(), coroHdlOp.handle());

  // Branch into the suspend block.
  builder.create<cf::BranchOp>(suspendBlock);

  // ------------------------------------------------------------------------ //
  // Coroutine suspend block: mark the end of a coroutine and return allocated
  // async token.
  // ------------------------------------------------------------------------ //
  builder.setInsertionPointToStart(suspendBlock);

  // Mark the end of a coroutine: async.coro.end
  builder.create<CoroEndOp>(coroHdlOp.handle());

  // Return created `async.token` and `async.values` from the suspend block.
  // This will be the return value of a coroutine ramp function.
  SmallVector<Value, 4> ret{retToken};
  ret.insert(ret.end(), retValues.begin(), retValues.end());
  builder.create<func::ReturnOp>(ret);

  // `async.await` op lowering will create resume blocks for async
  // continuations, and will conditionally branch to cleanup or suspend blocks.

  for (Block &block : func.getBody().getBlocks()) {
    if (&block == entryBlock || &block == cleanupBlock ||
        &block == suspendBlock)
      continue;
    Operation *terminator = block.getTerminator();
    if (auto yield = dyn_cast<YieldOp>(terminator)) {
      builder.setInsertionPointToEnd(&block);
      builder.create<cf::BranchOp>(cleanupBlock);
    }
  }

  // The switch-resumed API based coroutine should be marked with
  // coroutine.presplit attribute to mark the function as a coroutine.
  func->setAttr("passthrough", builder.getArrayAttr(
                                   StringAttr::get(ctx, "presplitcoroutine")));

  CoroMachinery machinery;
  machinery.func = func;
  machinery.asyncToken = retToken;
  machinery.returnValues = retValues;
  machinery.coroHandle = coroHdlOp.handle();
  machinery.entry = entryBlock;
  machinery.setError = nullptr; // created lazily only if needed
  machinery.cleanup = cleanupBlock;
  machinery.suspend = suspendBlock;
  return machinery;
}

// Lazily creates `set_error` block only if it is required for lowering to the
// runtime operations (see for example lowering of assert operation).
static Block *setupSetErrorBlock(CoroMachinery &coro) {
  if (coro.setError)
    return coro.setError;

  coro.setError = coro.func.addBlock();
  coro.setError->moveBefore(coro.cleanup);

  auto builder =
      ImplicitLocOpBuilder::atBlockBegin(coro.func->getLoc(), coro.setError);

  // Coroutine set_error block: set error on token and all returned values.
  builder.create<RuntimeSetErrorOp>(coro.asyncToken);
  for (Value retValue : coro.returnValues)
    builder.create<RuntimeSetErrorOp>(retValue);

  // Branch into the cleanup block.
  builder.create<cf::BranchOp>(coro.cleanup);

  return coro.setError;
}

/// Outline the body region attached to the `async.execute` op into a standalone
/// function.
///
/// Note that this is not reversible transformation.
static std::pair<func::FuncOp, CoroMachinery>
outlineExecuteOp(SymbolTable &symbolTable, ExecuteOp execute) {
  ModuleOp module = execute->getParentOfType<ModuleOp>();

  MLIRContext *ctx = module.getContext();
  Location loc = execute.getLoc();

  // Make sure that all constants will be inside the outlined async function to
  // reduce the number of function arguments.
  cloneConstantsIntoTheRegion(execute.body());

  // Collect all outlined function inputs.
  SetVector<mlir::Value> functionInputs(execute.dependencies().begin(),
                                        execute.dependencies().end());
  functionInputs.insert(execute.operands().begin(), execute.operands().end());
  getUsedValuesDefinedAbove(execute.body(), functionInputs);

  // Collect types for the outlined function inputs and outputs.
  auto typesRange = llvm::map_range(
      functionInputs, [](Value value) { return value.getType(); });
  SmallVector<Type, 4> inputTypes(typesRange.begin(), typesRange.end());
  auto outputTypes = execute.getResultTypes();

  auto funcType = FunctionType::get(ctx, inputTypes, outputTypes);
  auto funcAttrs = ArrayRef<NamedAttribute>();

  // TODO: Derive outlined function name from the parent FuncOp (support
  // multiple nested async.execute operations).
  func::FuncOp func =
      func::FuncOp::create(loc, kAsyncFnPrefix, funcType, funcAttrs);
  symbolTable.insert(func);

  SymbolTable::setSymbolVisibility(func, SymbolTable::Visibility::Private);
  auto builder = ImplicitLocOpBuilder::atBlockBegin(loc, func.addEntryBlock());

  // Prepare for coroutine conversion by creating the body of the function.
  {
    size_t numDependencies = execute.dependencies().size();
    size_t numOperands = execute.operands().size();

    // Await on all dependencies before starting to execute the body region.
    for (size_t i = 0; i < numDependencies; ++i)
      builder.create<AwaitOp>(func.getArgument(i));

    // Await on all async value operands and unwrap the payload.
    SmallVector<Value, 4> unwrappedOperands(numOperands);
    for (size_t i = 0; i < numOperands; ++i) {
      Value operand = func.getArgument(numDependencies + i);
      unwrappedOperands[i] = builder.create<AwaitOp>(loc, operand).result();
    }

    // Map from function inputs defined above the execute op to the function
    // arguments.
    BlockAndValueMapping valueMapping;
    valueMapping.map(functionInputs, func.getArguments());
    valueMapping.map(execute.body().getArguments(), unwrappedOperands);

    // Clone all operations from the execute operation body into the outlined
    // function body.
    for (Operation &op : execute.body().getOps())
      builder.clone(op, valueMapping);
  }

  // Adding entry/cleanup/suspend blocks.
  CoroMachinery coro = setupCoroMachinery(func);

  // Suspend async function at the end of an entry block, and resume it using
  // Async resume operation (execution will be resumed in a thread managed by
  // the async runtime).
  {
    cf::BranchOp branch = cast<cf::BranchOp>(coro.entry->getTerminator());
    builder.setInsertionPointToEnd(coro.entry);

    // Save the coroutine state: async.coro.save
    auto coroSaveOp =
        builder.create<CoroSaveOp>(CoroStateType::get(ctx), coro.coroHandle);

    // Pass coroutine to the runtime to be resumed on a runtime managed
    // thread.
    builder.create<RuntimeResumeOp>(coro.coroHandle);

    // Add async.coro.suspend as a suspended block terminator.
    builder.create<CoroSuspendOp>(coroSaveOp.state(), coro.suspend,
                                  branch.getDest(), coro.cleanup);

    branch.erase();
  }

  // Replace the original `async.execute` with a call to outlined function.
  {
    ImplicitLocOpBuilder callBuilder(loc, execute);
    auto callOutlinedFunc = callBuilder.create<func::CallOp>(
        func.getName(), execute.getResultTypes(), functionInputs.getArrayRef());
    execute.replaceAllUsesWith(callOutlinedFunc.getResults());
    execute.erase();
  }

  return {func, coro};
}

//===----------------------------------------------------------------------===//
// Convert async.create_group operation to async.runtime.create_group
//===----------------------------------------------------------------------===//

namespace {
class CreateGroupOpLowering : public OpConversionPattern<CreateGroupOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CreateGroupOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<RuntimeCreateGroupOp>(
        op, GroupType::get(op->getContext()), adaptor.getOperands());
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Convert async.add_to_group operation to async.runtime.add_to_group.
//===----------------------------------------------------------------------===//

namespace {
class AddToGroupOpLowering : public OpConversionPattern<AddToGroupOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AddToGroupOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<RuntimeAddToGroupOp>(
        op, rewriter.getIndexType(), adaptor.getOperands());
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Convert async.await and async.await_all operations to the async.runtime.await
// or async.runtime.await_and_resume operations.
//===----------------------------------------------------------------------===//

namespace {
template <typename AwaitType, typename AwaitableType>
class AwaitOpLoweringBase : public OpConversionPattern<AwaitType> {
  using AwaitAdaptor = typename AwaitType::Adaptor;

public:
  AwaitOpLoweringBase(
      MLIRContext *ctx,
      llvm::DenseMap<func::FuncOp, CoroMachinery> &outlinedFunctions)
      : OpConversionPattern<AwaitType>(ctx),
        outlinedFunctions(outlinedFunctions) {}

  LogicalResult
  matchAndRewrite(AwaitType op, typename AwaitType::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // We can only await on one the `AwaitableType` (for `await` it can be
    // a `token` or a `value`, for `await_all` it must be a `group`).
    if (!op.operand().getType().template isa<AwaitableType>())
      return rewriter.notifyMatchFailure(op, "unsupported awaitable type");

    // Check if await operation is inside the outlined coroutine function.
    auto func = op->template getParentOfType<func::FuncOp>();
    auto outlined = outlinedFunctions.find(func);
    const bool isInCoroutine = outlined != outlinedFunctions.end();

    Location loc = op->getLoc();
    Value operand = adaptor.operand();

    Type i1 = rewriter.getI1Type();

    // Inside regular functions we use the blocking wait operation to wait for
    // the async object (token, value or group) to become available.
    if (!isInCoroutine) {
      ImplicitLocOpBuilder builder(loc, op, rewriter.getListener());
      builder.create<RuntimeAwaitOp>(loc, operand);

      // Assert that the awaited operands is not in the error state.
      Value isError = builder.create<RuntimeIsErrorOp>(i1, operand);
      Value notError = builder.create<arith::XOrIOp>(
          isError, builder.create<arith::ConstantOp>(
                       loc, i1, builder.getIntegerAttr(i1, 1)));

      builder.create<cf::AssertOp>(notError,
                                   "Awaited async operand is in error state");
    }

    // Inside the coroutine we convert await operation into coroutine suspension
    // point, and resume execution asynchronously.
    if (isInCoroutine) {
      CoroMachinery &coro = outlined->getSecond();
      Block *suspended = op->getBlock();

      ImplicitLocOpBuilder builder(loc, op, rewriter.getListener());
      MLIRContext *ctx = op->getContext();

      // Save the coroutine state and resume on a runtime managed thread when
      // the operand becomes available.
      auto coroSaveOp =
          builder.create<CoroSaveOp>(CoroStateType::get(ctx), coro.coroHandle);
      builder.create<RuntimeAwaitAndResumeOp>(operand, coro.coroHandle);

      // Split the entry block before the await operation.
      Block *resume = rewriter.splitBlock(suspended, Block::iterator(op));

      // Add async.coro.suspend as a suspended block terminator.
      builder.setInsertionPointToEnd(suspended);
      builder.create<CoroSuspendOp>(coroSaveOp.state(), coro.suspend, resume,
                                    coro.cleanup);

      // Split the resume block into error checking and continuation.
      Block *continuation = rewriter.splitBlock(resume, Block::iterator(op));

      // Check if the awaited value is in the error state.
      builder.setInsertionPointToStart(resume);
      auto isError = builder.create<RuntimeIsErrorOp>(loc, i1, operand);
      builder.create<cf::CondBranchOp>(isError,
                                       /*trueDest=*/setupSetErrorBlock(coro),
                                       /*trueArgs=*/ArrayRef<Value>(),
                                       /*falseDest=*/continuation,
                                       /*falseArgs=*/ArrayRef<Value>());

      // Make sure that replacement value will be constructed in the
      // continuation block.
      rewriter.setInsertionPointToStart(continuation);
    }

    // Erase or replace the await operation with the new value.
    if (Value replaceWith = getReplacementValue(op, operand, rewriter))
      rewriter.replaceOp(op, replaceWith);
    else
      rewriter.eraseOp(op);

    return success();
  }

  virtual Value getReplacementValue(AwaitType op, Value operand,
                                    ConversionPatternRewriter &rewriter) const {
    return Value();
  }

private:
  llvm::DenseMap<func::FuncOp, CoroMachinery> &outlinedFunctions;
};

/// Lowering for `async.await` with a token operand.
class AwaitTokenOpLowering : public AwaitOpLoweringBase<AwaitOp, TokenType> {
  using Base = AwaitOpLoweringBase<AwaitOp, TokenType>;

public:
  using Base::Base;
};

/// Lowering for `async.await` with a value operand.
class AwaitValueOpLowering : public AwaitOpLoweringBase<AwaitOp, ValueType> {
  using Base = AwaitOpLoweringBase<AwaitOp, ValueType>;

public:
  using Base::Base;

  Value
  getReplacementValue(AwaitOp op, Value operand,
                      ConversionPatternRewriter &rewriter) const override {
    // Load from the async value storage.
    auto valueType = operand.getType().cast<ValueType>().getValueType();
    return rewriter.create<RuntimeLoadOp>(op->getLoc(), valueType, operand);
  }
};

/// Lowering for `async.await_all` operation.
class AwaitAllOpLowering : public AwaitOpLoweringBase<AwaitAllOp, GroupType> {
  using Base = AwaitOpLoweringBase<AwaitAllOp, GroupType>;

public:
  using Base::Base;
};

} // namespace

//===----------------------------------------------------------------------===//
// Convert async.yield operation to async.runtime operations.
//===----------------------------------------------------------------------===//

class YieldOpLowering : public OpConversionPattern<async::YieldOp> {
public:
  YieldOpLowering(
      MLIRContext *ctx,
      const llvm::DenseMap<func::FuncOp, CoroMachinery> &outlinedFunctions)
      : OpConversionPattern<async::YieldOp>(ctx),
        outlinedFunctions(outlinedFunctions) {}

  LogicalResult
  matchAndRewrite(async::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Check if yield operation is inside the async coroutine function.
    auto func = op->template getParentOfType<func::FuncOp>();
    auto outlined = outlinedFunctions.find(func);
    if (outlined == outlinedFunctions.end())
      return rewriter.notifyMatchFailure(
          op, "operation is not inside the async coroutine function");

    Location loc = op->getLoc();
    const CoroMachinery &coro = outlined->getSecond();

    // Store yielded values into the async values storage and switch async
    // values state to available.
    for (auto tuple : llvm::zip(adaptor.getOperands(), coro.returnValues)) {
      Value yieldValue = std::get<0>(tuple);
      Value asyncValue = std::get<1>(tuple);
      rewriter.create<RuntimeStoreOp>(loc, yieldValue, asyncValue);
      rewriter.create<RuntimeSetAvailableOp>(loc, asyncValue);
    }

    // Switch the coroutine completion token to available state.
    rewriter.replaceOpWithNewOp<RuntimeSetAvailableOp>(op, coro.asyncToken);

    return success();
  }

private:
  const llvm::DenseMap<func::FuncOp, CoroMachinery> &outlinedFunctions;
};

//===----------------------------------------------------------------------===//
// Convert cf.assert operation to cf.cond_br into `set_error` block.
//===----------------------------------------------------------------------===//

class AssertOpLowering : public OpConversionPattern<cf::AssertOp> {
public:
  AssertOpLowering(
      MLIRContext *ctx,
      llvm::DenseMap<func::FuncOp, CoroMachinery> &outlinedFunctions)
      : OpConversionPattern<cf::AssertOp>(ctx),
        outlinedFunctions(outlinedFunctions) {}

  LogicalResult
  matchAndRewrite(cf::AssertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Check if assert operation is inside the async coroutine function.
    auto func = op->template getParentOfType<func::FuncOp>();
    auto outlined = outlinedFunctions.find(func);
    if (outlined == outlinedFunctions.end())
      return rewriter.notifyMatchFailure(
          op, "operation is not inside the async coroutine function");

    Location loc = op->getLoc();
    CoroMachinery &coro = outlined->getSecond();

    Block *cont = rewriter.splitBlock(op->getBlock(), Block::iterator(op));
    rewriter.setInsertionPointToEnd(cont->getPrevNode());
    rewriter.create<cf::CondBranchOp>(loc, adaptor.getArg(),
                                      /*trueDest=*/cont,
                                      /*trueArgs=*/ArrayRef<Value>(),
                                      /*falseDest=*/setupSetErrorBlock(coro),
                                      /*falseArgs=*/ArrayRef<Value>());
    rewriter.eraseOp(op);

    return success();
  }

private:
  llvm::DenseMap<func::FuncOp, CoroMachinery> &outlinedFunctions;
};

//===----------------------------------------------------------------------===//

/// Rewrite a func as a coroutine by:
/// 1) Wrapping the results into `async.value`.
/// 2) Prepending the results with `async.token`.
/// 3) Setting up coroutine blocks.
/// 4) Rewriting return ops as yield op and branch op into the suspend block.
static CoroMachinery rewriteFuncAsCoroutine(func::FuncOp func) {
  auto *ctx = func->getContext();
  auto loc = func.getLoc();
  SmallVector<Type> resultTypes;
  resultTypes.reserve(func.getCallableResults().size());
  llvm::transform(func.getCallableResults(), std::back_inserter(resultTypes),
                  [](Type type) { return ValueType::get(type); });
  func.setType(
      FunctionType::get(ctx, func.getFunctionType().getInputs(), resultTypes));
  func.insertResult(0, TokenType::get(ctx), {});
  for (Block &block : func.getBlocks()) {
    Operation *terminator = block.getTerminator();
    if (auto returnOp = dyn_cast<func::ReturnOp>(*terminator)) {
      ImplicitLocOpBuilder builder(loc, returnOp);
      builder.create<YieldOp>(returnOp.getOperands());
      returnOp.erase();
    }
  }
  return setupCoroMachinery(func);
}

/// Rewrites a call into a function that has been rewritten as a coroutine.
///
/// The invocation of this function is safe only when call ops are traversed in
/// reverse order of how they appear in a single block. See `funcsToCoroutines`.
static void rewriteCallsiteForCoroutine(func::CallOp oldCall,
                                        func::FuncOp func) {
  auto loc = func.getLoc();
  ImplicitLocOpBuilder callBuilder(loc, oldCall);
  auto newCall = callBuilder.create<func::CallOp>(
      func.getName(), func.getCallableResults(), oldCall.getArgOperands());

  // Await on the async token and all the value results and unwrap the latter.
  callBuilder.create<AwaitOp>(loc, newCall.getResults().front());
  SmallVector<Value> unwrappedResults;
  unwrappedResults.reserve(newCall->getResults().size() - 1);
  for (Value result : newCall.getResults().drop_front())
    unwrappedResults.push_back(
        callBuilder.create<AwaitOp>(loc, result).result());
  // Careful, when result of a call is piped into another call this could lead
  // to a dangling pointer.
  oldCall.replaceAllUsesWith(unwrappedResults);
  oldCall.erase();
}

static bool isAllowedToBlock(func::FuncOp func) {
  return !!func->getAttrOfType<UnitAttr>(AsyncDialect::kAllowedToBlockAttrName);
}

static LogicalResult funcsToCoroutines(
    ModuleOp module,
    llvm::DenseMap<func::FuncOp, CoroMachinery> &outlinedFunctions) {
  // The following code supports the general case when 2 functions mutually
  // recurse into each other. Because of this and that we are relying on
  // SymbolUserMap to find pointers to calling FuncOps, we cannot simply erase
  // a FuncOp while inserting an equivalent coroutine, because that could lead
  // to dangling pointers.

  SmallVector<func::FuncOp> funcWorklist;

  // Careful, it's okay to add a func to the worklist multiple times if and only
  // if the loop processing the worklist will skip the functions that have
  // already been converted to coroutines.
  auto addToWorklist = [&](func::FuncOp func) {
    if (isAllowedToBlock(func))
      return;
    // N.B. To refactor this code into a separate pass the lookup in
    // outlinedFunctions is the most obvious obstacle. Looking at an arbitrary
    // func and recognizing if it has a coroutine structure is messy. Passing
    // this dict between the passes is ugly.
    if (isAllowedToBlock(func) ||
        outlinedFunctions.find(func) == outlinedFunctions.end()) {
      for (Operation &op : func.getBody().getOps()) {
        if (dyn_cast<AwaitOp>(op) || dyn_cast<AwaitAllOp>(op)) {
          funcWorklist.push_back(func);
          break;
        }
      }
    }
  };

  // Traverse in post-order collecting for each func op the await ops it has.
  for (func::FuncOp func : module.getOps<func::FuncOp>())
    addToWorklist(func);

  SymbolTableCollection symbolTable;
  SymbolUserMap symbolUserMap(symbolTable, module);

  // Rewrite funcs, while updating call sites and adding them to the worklist.
  while (!funcWorklist.empty()) {
    auto func = funcWorklist.pop_back_val();
    auto insertion = outlinedFunctions.insert({func, CoroMachinery{}});
    if (!insertion.second)
      // This function has already been processed because this is either
      // the corecursive case, or a caller with multiple calls to a newly
      // created corouting. Either way, skip updating the call sites.
      continue;
    insertion.first->second = rewriteFuncAsCoroutine(func);
    SmallVector<Operation *> users(symbolUserMap.getUsers(func).begin(),
                                   symbolUserMap.getUsers(func).end());
    // If there are multiple calls from the same block they need to be traversed
    // in reverse order so that symbolUserMap references are not invalidated
    // when updating the users of the call op which is earlier in the block.
    llvm::sort(users, [](Operation *a, Operation *b) {
      Block *blockA = a->getBlock();
      Block *blockB = b->getBlock();
      // Impose arbitrary order on blocks so that there is a well-defined order.
      return blockA > blockB || (blockA == blockB && !a->isBeforeInBlock(b));
    });
    // Rewrite the callsites to await on results of the newly created coroutine.
    for (Operation *op : users) {
      if (func::CallOp call = dyn_cast<func::CallOp>(*op)) {
        func::FuncOp caller = call->getParentOfType<func::FuncOp>();
        rewriteCallsiteForCoroutine(call, func); // Careful, erases the call op.
        addToWorklist(caller);
      } else {
        op->emitError("Unexpected reference to func referenced by symbol");
        return failure();
      }
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
void AsyncToAsyncRuntimePass::runOnOperation() {
  ModuleOp module = getOperation();
  SymbolTable symbolTable(module);

  // Outline all `async.execute` body regions into async functions (coroutines).
  llvm::DenseMap<func::FuncOp, CoroMachinery> outlinedFunctions;

  module.walk([&](ExecuteOp execute) {
    outlinedFunctions.insert(outlineExecuteOp(symbolTable, execute));
  });

  LLVM_DEBUG({
    llvm::dbgs() << "Outlined " << outlinedFunctions.size()
                 << " functions built from async.execute operations\n";
  });

  // Returns true if operation is inside the coroutine.
  auto isInCoroutine = [&](Operation *op) -> bool {
    auto parentFunc = op->getParentOfType<func::FuncOp>();
    return outlinedFunctions.find(parentFunc) != outlinedFunctions.end();
  };

  if (eliminateBlockingAwaitOps &&
      failed(funcsToCoroutines(module, outlinedFunctions))) {
    signalPassFailure();
    return;
  }

  // Lower async operations to async.runtime operations.
  MLIRContext *ctx = module->getContext();
  RewritePatternSet asyncPatterns(ctx);

  // Conversion to async runtime augments original CFG with the coroutine CFG,
  // and we have to make sure that structured control flow operations with async
  // operations in nested regions will be converted to branch-based control flow
  // before we add the coroutine basic blocks.
  populateSCFToControlFlowConversionPatterns(asyncPatterns);

  // Async lowering does not use type converter because it must preserve all
  // types for async.runtime operations.
  asyncPatterns.add<CreateGroupOpLowering, AddToGroupOpLowering>(ctx);
  asyncPatterns.add<AwaitTokenOpLowering, AwaitValueOpLowering,
                    AwaitAllOpLowering, YieldOpLowering>(ctx,
                                                         outlinedFunctions);

  // Lower assertions to conditional branches into error blocks.
  asyncPatterns.add<AssertOpLowering>(ctx, outlinedFunctions);

  // All high level async operations must be lowered to the runtime operations.
  ConversionTarget runtimeTarget(*ctx);
  runtimeTarget.addLegalDialect<AsyncDialect>();
  runtimeTarget.addIllegalOp<CreateGroupOp, AddToGroupOp>();
  runtimeTarget.addIllegalOp<ExecuteOp, AwaitOp, AwaitAllOp, async::YieldOp>();

  // Decide if structured control flow has to be lowered to branch-based CFG.
  runtimeTarget.addDynamicallyLegalDialect<scf::SCFDialect>([&](Operation *op) {
    auto walkResult = op->walk([&](Operation *nested) {
      bool isAsync = isa<async::AsyncDialect>(nested->getDialect());
      return isAsync && isInCoroutine(nested) ? WalkResult::interrupt()
                                              : WalkResult::advance();
    });
    return !walkResult.wasInterrupted();
  });
  runtimeTarget.addLegalOp<cf::AssertOp, arith::XOrIOp, arith::ConstantOp,
                           func::ConstantOp, cf::BranchOp, cf::CondBranchOp>();

  // Assertions must be converted to runtime errors inside async functions.
  runtimeTarget.addDynamicallyLegalOp<cf::AssertOp>(
      [&](cf::AssertOp op) -> bool {
        auto func = op->getParentOfType<func::FuncOp>();
        return outlinedFunctions.find(func) == outlinedFunctions.end();
      });

  if (eliminateBlockingAwaitOps)
    runtimeTarget.addDynamicallyLegalOp<RuntimeAwaitOp>(
        [&](RuntimeAwaitOp op) -> bool {
          return isAllowedToBlock(op->getParentOfType<func::FuncOp>());
        });

  if (failed(applyPartialConversion(module, runtimeTarget,
                                    std::move(asyncPatterns)))) {
    signalPassFailure();
    return;
  }
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::createAsyncToAsyncRuntimePass() {
  return std::make_unique<AsyncToAsyncRuntimePass>();
}
