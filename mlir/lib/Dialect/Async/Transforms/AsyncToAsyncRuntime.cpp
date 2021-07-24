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

#include "PassDetail.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Async/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::async;

#define DEBUG_TYPE "async-to-async-runtime"
// Prefix for functions outlined from `async.execute` op regions.
static constexpr const char kAsyncFnPrefix[] = "async_execute_fn";

namespace {

class AsyncToAsyncRuntimePass
    : public AsyncToAsyncRuntimeBase<AsyncToAsyncRuntimePass> {
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
  FuncOp func;

  // Async execute region returns a completion token, and an async value for
  // each yielded value.
  //
  //   %token, %result = async.execute -> !async.value<T> {
  //     %0 = constant ... : T
  //     async.yield %0 : T
  //   }
  Value asyncToken; // token representing completion of the async region
  llvm::SmallVector<Value, 4> returnValues; // returned async values

  Value coroHandle; // coroutine handle (!async.coro.handle value)
  Block *setError;  // switch completion token and all values to error state
  Block *cleanup;   // coroutine cleanup block
  Block *suspend;   // coroutine suspension block
};
} // namespace

/// Builds an coroutine template compatible with LLVM coroutines switched-resume
/// lowering using `async.runtime.*` and `async.coro.*` operations.
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
///   func @async_execute_fn(<function-arguments>)
///        -> (!async.token, !async.value<T>)
///   {
///     ^entry(<function-arguments>):
///       %token = <async token> : !async.token    // create async runtime token
///       %value = <async value> : !async.value<T> // create async value
///       %id = async.coro.id                      // create a coroutine id
///       %hdl = async.coro.begin %id              // create a coroutine handle
///       br ^cleanup
///
///     ^set_error: // this block created lazily only if needed (see code below)
///       async.runtime.set_error %token : !async.token
///       async.runtime.set_error %value : !async.value<T>
///       br ^cleanup
///
///     ^cleanup:
///       async.coro.free %hdl // delete the coroutine state
///       br ^suspend
///
///     ^suspend:
///       async.coro.end %hdl // marks the end of a coroutine
///       return %token, %value : !async.token, !async.value<T>
///   }
///
/// The actual code for the async.execute operation body region will be inserted
/// before the entry block terminator.
///
///
static CoroMachinery setupCoroMachinery(FuncOp func) {
  assert(func.getBody().empty() && "Function must have empty body");

  MLIRContext *ctx = func.getContext();
  Block *entryBlock = func.addEntryBlock();

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

  Block *cleanupBlock = func.addBlock();
  Block *suspendBlock = func.addBlock();

  // ------------------------------------------------------------------------ //
  // Coroutine cleanup block: deallocate coroutine frame, free the memory.
  // ------------------------------------------------------------------------ //
  builder.setInsertionPointToStart(cleanupBlock);
  builder.create<CoroFreeOp>(coroIdOp.id(), coroHdlOp.handle());

  // Branch into the suspend block.
  builder.create<BranchOp>(suspendBlock);

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
  builder.create<ReturnOp>(ret);

  // Branch from the entry block to the cleanup block to create a valid CFG.
  builder.setInsertionPointToEnd(entryBlock);
  builder.create<BranchOp>(cleanupBlock);

  // `async.await` op lowering will create resume blocks for async
  // continuations, and will conditionally branch to cleanup or suspend blocks.

  CoroMachinery machinery;
  machinery.func = func;
  machinery.asyncToken = retToken;
  machinery.returnValues = retValues;
  machinery.coroHandle = coroHdlOp.handle();
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
  builder.create<BranchOp>(coro.cleanup);

  return coro.setError;
}

/// Outline the body region attached to the `async.execute` op into a standalone
/// function.
///
/// Note that this is not reversible transformation.
static std::pair<FuncOp, CoroMachinery>
outlineExecuteOp(SymbolTable &symbolTable, ExecuteOp execute) {
  ModuleOp module = execute->getParentOfType<ModuleOp>();

  MLIRContext *ctx = module.getContext();
  Location loc = execute.getLoc();

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
  FuncOp func = FuncOp::create(loc, kAsyncFnPrefix, funcType, funcAttrs);
  symbolTable.insert(func);

  SymbolTable::setSymbolVisibility(func, SymbolTable::Visibility::Private);

  // Prepare a function for coroutine lowering by adding entry/cleanup/suspend
  // blocks, adding async.coro operations and setting up control flow.
  CoroMachinery coro = setupCoroMachinery(func);

  // Suspend async function at the end of an entry block, and resume it using
  // Async resume operation (execution will be resumed in a thread managed by
  // the async runtime).
  Block *entryBlock = &func.getBlocks().front();
  auto builder = ImplicitLocOpBuilder::atBlockTerminator(loc, entryBlock);

  // Save the coroutine state: async.coro.save
  auto coroSaveOp =
      builder.create<CoroSaveOp>(CoroStateType::get(ctx), coro.coroHandle);

  // Pass coroutine to the runtime to be resumed on a runtime managed thread.
  builder.create<RuntimeResumeOp>(coro.coroHandle);

  // Split the entry block before the terminator (branch to suspend block).
  auto *terminatorOp = entryBlock->getTerminator();
  Block *suspended = terminatorOp->getBlock();
  Block *resume = suspended->splitBlock(terminatorOp);

  // Add async.coro.suspend as a suspended block terminator.
  builder.setInsertionPointToEnd(suspended);
  builder.create<CoroSuspendOp>(coroSaveOp.state(), coro.suspend, resume,
                                coro.cleanup);

  size_t numDependencies = execute.dependencies().size();
  size_t numOperands = execute.operands().size();

  // Await on all dependencies before starting to execute the body region.
  builder.setInsertionPointToStart(resume);
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

  // Replace the original `async.execute` with a call to outlined function.
  ImplicitLocOpBuilder callBuilder(loc, execute);
  auto callOutlinedFunc = callBuilder.create<CallOp>(
      func.getName(), execute.getResultTypes(), functionInputs.getArrayRef());
  execute.replaceAllUsesWith(callOutlinedFunc.getResults());
  execute.erase();

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
  matchAndRewrite(CreateGroupOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<RuntimeCreateGroupOp>(
        op, GroupType::get(op->getContext()), operands);
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
  matchAndRewrite(AddToGroupOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<RuntimeAddToGroupOp>(
        op, rewriter.getIndexType(), operands);
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
  AwaitOpLoweringBase(MLIRContext *ctx,
                      llvm::DenseMap<FuncOp, CoroMachinery> &outlinedFunctions)
      : OpConversionPattern<AwaitType>(ctx),
        outlinedFunctions(outlinedFunctions) {}

  LogicalResult
  matchAndRewrite(AwaitType op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // We can only await on one the `AwaitableType` (for `await` it can be
    // a `token` or a `value`, for `await_all` it must be a `group`).
    if (!op.operand().getType().template isa<AwaitableType>())
      return rewriter.notifyMatchFailure(op, "unsupported awaitable type");

    // Check if await operation is inside the outlined coroutine function.
    auto func = op->template getParentOfType<FuncOp>();
    auto outlined = outlinedFunctions.find(func);
    const bool isInCoroutine = outlined != outlinedFunctions.end();

    Location loc = op->getLoc();
    Value operand = AwaitAdaptor(operands).operand();

    // Inside regular functions we use the blocking wait operation to wait for
    // the async object (token, value or group) to become available.
    if (!isInCoroutine)
      rewriter.create<RuntimeAwaitOp>(loc, operand);

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
      auto isError =
          builder.create<RuntimeIsErrorOp>(loc, rewriter.getI1Type(), operand);
      builder.create<CondBranchOp>(isError,
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
  llvm::DenseMap<FuncOp, CoroMachinery> &outlinedFunctions;
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
      const llvm::DenseMap<FuncOp, CoroMachinery> &outlinedFunctions)
      : OpConversionPattern<async::YieldOp>(ctx),
        outlinedFunctions(outlinedFunctions) {}

  LogicalResult
  matchAndRewrite(async::YieldOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Check if yield operation is inside the async coroutine function.
    auto func = op->template getParentOfType<FuncOp>();
    auto outlined = outlinedFunctions.find(func);
    if (outlined == outlinedFunctions.end())
      return rewriter.notifyMatchFailure(
          op, "operation is not inside the async coroutine function");

    Location loc = op->getLoc();
    const CoroMachinery &coro = outlined->getSecond();

    // Store yielded values into the async values storage and switch async
    // values state to available.
    for (auto tuple : llvm::zip(operands, coro.returnValues)) {
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
  const llvm::DenseMap<FuncOp, CoroMachinery> &outlinedFunctions;
};

//===----------------------------------------------------------------------===//
// Convert std.assert operation to cond_br into `set_error` block.
//===----------------------------------------------------------------------===//

class AssertOpLowering : public OpConversionPattern<AssertOp> {
public:
  AssertOpLowering(MLIRContext *ctx,
                   llvm::DenseMap<FuncOp, CoroMachinery> &outlinedFunctions)
      : OpConversionPattern<AssertOp>(ctx),
        outlinedFunctions(outlinedFunctions) {}

  LogicalResult
  matchAndRewrite(AssertOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Check if assert operation is inside the async coroutine function.
    auto func = op->template getParentOfType<FuncOp>();
    auto outlined = outlinedFunctions.find(func);
    if (outlined == outlinedFunctions.end())
      return rewriter.notifyMatchFailure(
          op, "operation is not inside the async coroutine function");

    Location loc = op->getLoc();
    CoroMachinery &coro = outlined->getSecond();

    Block *cont = rewriter.splitBlock(op->getBlock(), Block::iterator(op));
    rewriter.setInsertionPointToEnd(cont->getPrevNode());
    rewriter.create<CondBranchOp>(loc, AssertOpAdaptor(operands).arg(),
                                  /*trueDest=*/cont,
                                  /*trueArgs=*/ArrayRef<Value>(),
                                  /*falseDest=*/setupSetErrorBlock(coro),
                                  /*falseArgs=*/ArrayRef<Value>());
    rewriter.eraseOp(op);

    return success();
  }

private:
  llvm::DenseMap<FuncOp, CoroMachinery> &outlinedFunctions;
};

//===----------------------------------------------------------------------===//

void AsyncToAsyncRuntimePass::runOnOperation() {
  ModuleOp module = getOperation();
  SymbolTable symbolTable(module);

  // Outline all `async.execute` body regions into async functions (coroutines).
  llvm::DenseMap<FuncOp, CoroMachinery> outlinedFunctions;

  module.walk([&](ExecuteOp execute) {
    outlinedFunctions.insert(outlineExecuteOp(symbolTable, execute));
  });

  LLVM_DEBUG({
    llvm::dbgs() << "Outlined " << outlinedFunctions.size()
                 << " functions built from async.execute operations\n";
  });

  // Returns true if operation is inside the coroutine.
  auto isInCoroutine = [&](Operation *op) -> bool {
    auto parentFunc = op->getParentOfType<FuncOp>();
    return outlinedFunctions.find(parentFunc) != outlinedFunctions.end();
  };

  // Lower async operations to async.runtime operations.
  MLIRContext *ctx = module->getContext();
  RewritePatternSet asyncPatterns(ctx);

  // Conversion to async runtime augments original CFG with the coroutine CFG,
  // and we have to make sure that structured control flow operations with async
  // operations in nested regions will be converted to branch-based control flow
  // before we add the coroutine basic blocks.
  populateLoopToStdConversionPatterns(asyncPatterns);

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
  runtimeTarget.addLegalOp<BranchOp, CondBranchOp>();

  // Assertions must be converted to runtime errors inside async functions.
  runtimeTarget.addDynamicallyLegalOp<AssertOp>([&](AssertOp op) -> bool {
    auto func = op->getParentOfType<FuncOp>();
    return outlinedFunctions.find(func) == outlinedFunctions.end();
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
