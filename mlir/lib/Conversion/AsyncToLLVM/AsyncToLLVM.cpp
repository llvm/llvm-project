//===- AsyncToLLVM.cpp - Convert Async to LLVM dialect --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h"

#include "../PassDetail.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/FormatVariadic.h"

#define DEBUG_TYPE "convert-async-to-llvm"

using namespace mlir;
using namespace mlir::async;

// Prefix for functions outlined from `async.execute` op regions.
static constexpr const char kAsyncFnPrefix[] = "async_execute_fn";

//===----------------------------------------------------------------------===//
// Async Runtime C API declaration.
//===----------------------------------------------------------------------===//

static constexpr const char *kCreateToken = "mlirAsyncRuntimeCreateToken";
static constexpr const char *kEmplaceToken = "mlirAsyncRuntimeEmplaceToken";
static constexpr const char *kAwaitToken = "mlirAsyncRuntimeAwaitToken";
static constexpr const char *kExecute = "mlirAsyncRuntimeExecute";
static constexpr const char *kAwaitAndExecute =
    "mlirAsyncRuntimeAwaitTokenAndExecute";

namespace {
// Async Runtime API function types.
struct AsyncAPI {
  static FunctionType createTokenFunctionType(MLIRContext *ctx) {
    return FunctionType::get({}, {TokenType::get(ctx)}, ctx);
  }

  static FunctionType emplaceTokenFunctionType(MLIRContext *ctx) {
    return FunctionType::get({TokenType::get(ctx)}, {}, ctx);
  }

  static FunctionType awaitTokenFunctionType(MLIRContext *ctx) {
    return FunctionType::get({TokenType::get(ctx)}, {}, ctx);
  }

  static FunctionType executeFunctionType(MLIRContext *ctx) {
    auto hdl = LLVM::LLVMType::getInt8PtrTy(ctx);
    auto resume = resumeFunctionType(ctx).getPointerTo();
    return FunctionType::get({hdl, resume}, {}, ctx);
  }

  static FunctionType awaitAndExecuteFunctionType(MLIRContext *ctx) {
    auto hdl = LLVM::LLVMType::getInt8PtrTy(ctx);
    auto resume = resumeFunctionType(ctx).getPointerTo();
    return FunctionType::get({TokenType::get(ctx), hdl, resume}, {}, ctx);
  }

  // Auxiliary coroutine resume intrinsic wrapper.
  static LLVM::LLVMType resumeFunctionType(MLIRContext *ctx) {
    auto voidTy = LLVM::LLVMType::getVoidTy(ctx);
    auto i8Ptr = LLVM::LLVMType::getInt8PtrTy(ctx);
    return LLVM::LLVMType::getFunctionTy(voidTy, {i8Ptr}, false);
  }
};
} // namespace

// Adds Async Runtime C API declarations to the module.
static void addAsyncRuntimeApiDeclarations(ModuleOp module) {
  auto builder = OpBuilder::atBlockTerminator(module.getBody());

  MLIRContext *ctx = module.getContext();
  Location loc = module.getLoc();

  if (!module.lookupSymbol(kCreateToken))
    builder.create<FuncOp>(loc, kCreateToken,
                           AsyncAPI::createTokenFunctionType(ctx));

  if (!module.lookupSymbol(kEmplaceToken))
    builder.create<FuncOp>(loc, kEmplaceToken,
                           AsyncAPI::emplaceTokenFunctionType(ctx));

  if (!module.lookupSymbol(kAwaitToken))
    builder.create<FuncOp>(loc, kAwaitToken,
                           AsyncAPI::awaitTokenFunctionType(ctx));

  if (!module.lookupSymbol(kExecute))
    builder.create<FuncOp>(loc, kExecute, AsyncAPI::executeFunctionType(ctx));

  if (!module.lookupSymbol(kAwaitAndExecute))
    builder.create<FuncOp>(loc, kAwaitAndExecute,
                           AsyncAPI::awaitAndExecuteFunctionType(ctx));
}

//===----------------------------------------------------------------------===//
// LLVM coroutines intrinsics declarations.
//===----------------------------------------------------------------------===//

static constexpr const char *kCoroId = "llvm.coro.id";
static constexpr const char *kCoroSizeI64 = "llvm.coro.size.i64";
static constexpr const char *kCoroBegin = "llvm.coro.begin";
static constexpr const char *kCoroSave = "llvm.coro.save";
static constexpr const char *kCoroSuspend = "llvm.coro.suspend";
static constexpr const char *kCoroEnd = "llvm.coro.end";
static constexpr const char *kCoroFree = "llvm.coro.free";
static constexpr const char *kCoroResume = "llvm.coro.resume";

/// Adds coroutine intrinsics declarations to the module.
static void addCoroutineIntrinsicsDeclarations(ModuleOp module) {
  using namespace mlir::LLVM;

  MLIRContext *ctx = module.getContext();
  Location loc = module.getLoc();

  OpBuilder builder(module.getBody()->getTerminator());

  auto token = LLVMTokenType::get(ctx);
  auto voidTy = LLVMType::getVoidTy(ctx);

  auto i8 = LLVMType::getInt8Ty(ctx);
  auto i1 = LLVMType::getInt1Ty(ctx);
  auto i32 = LLVMType::getInt32Ty(ctx);
  auto i64 = LLVMType::getInt64Ty(ctx);
  auto i8Ptr = LLVMType::getInt8PtrTy(ctx);

  if (!module.lookupSymbol(kCoroId))
    builder.create<LLVMFuncOp>(
        loc, kCoroId,
        LLVMType::getFunctionTy(token, {i32, i8Ptr, i8Ptr, i8Ptr}, false));

  if (!module.lookupSymbol(kCoroSizeI64))
    builder.create<LLVMFuncOp>(loc, kCoroSizeI64,
                               LLVMType::getFunctionTy(i64, false));

  if (!module.lookupSymbol(kCoroBegin))
    builder.create<LLVMFuncOp>(
        loc, kCoroBegin, LLVMType::getFunctionTy(i8Ptr, {token, i8Ptr}, false));

  if (!module.lookupSymbol(kCoroSave))
    builder.create<LLVMFuncOp>(loc, kCoroSave,
                               LLVMType::getFunctionTy(token, i8Ptr, false));

  if (!module.lookupSymbol(kCoroSuspend))
    builder.create<LLVMFuncOp>(loc, kCoroSuspend,
                               LLVMType::getFunctionTy(i8, {token, i1}, false));

  if (!module.lookupSymbol(kCoroEnd))
    builder.create<LLVMFuncOp>(loc, kCoroEnd,
                               LLVMType::getFunctionTy(i1, {i8Ptr, i1}, false));

  if (!module.lookupSymbol(kCoroFree))
    builder.create<LLVMFuncOp>(
        loc, kCoroFree, LLVMType::getFunctionTy(i8Ptr, {token, i8Ptr}, false));

  if (!module.lookupSymbol(kCoroResume))
    builder.create<LLVMFuncOp>(loc, kCoroResume,
                               LLVMType::getFunctionTy(voidTy, i8Ptr, false));
}

//===----------------------------------------------------------------------===//
// Add malloc/free declarations to the module.
//===----------------------------------------------------------------------===//

static constexpr const char *kMalloc = "malloc";
static constexpr const char *kFree = "free";

/// Adds malloc/free declarations to the module.
static void addCRuntimeDeclarations(ModuleOp module) {
  using namespace mlir::LLVM;

  MLIRContext *ctx = module.getContext();
  Location loc = module.getLoc();

  OpBuilder builder(module.getBody()->getTerminator());

  auto voidTy = LLVMType::getVoidTy(ctx);
  auto i64 = LLVMType::getInt64Ty(ctx);
  auto i8Ptr = LLVMType::getInt8PtrTy(ctx);

  if (!module.lookupSymbol(kMalloc))
    builder.create<LLVM::LLVMFuncOp>(
        loc, kMalloc, LLVMType::getFunctionTy(i8Ptr, {i64}, false));

  if (!module.lookupSymbol(kFree))
    builder.create<LLVM::LLVMFuncOp>(
        loc, kFree, LLVMType::getFunctionTy(voidTy, i8Ptr, false));
}

//===----------------------------------------------------------------------===//
// Coroutine resume function wrapper.
//===----------------------------------------------------------------------===//

static constexpr const char *kResume = "__resume";

// A function that takes a coroutine handle and calls a `llvm.coro.resume`
// intrinsics. We need this function to be able to pass it to the async
// runtime execute API.
static void addResumeFunction(ModuleOp module) {
  MLIRContext *ctx = module.getContext();

  OpBuilder moduleBuilder(module.getBody()->getTerminator());
  Location loc = module.getLoc();

  if (module.lookupSymbol(kResume))
    return;

  auto voidTy = LLVM::LLVMType::getVoidTy(ctx);
  auto i8Ptr = LLVM::LLVMType::getInt8PtrTy(ctx);

  auto resumeOp = moduleBuilder.create<LLVM::LLVMFuncOp>(
      loc, kResume, LLVM::LLVMFunctionType::get(voidTy, {i8Ptr}));
  SymbolTable::setSymbolVisibility(resumeOp, SymbolTable::Visibility::Private);

  auto *block = resumeOp.addEntryBlock();
  OpBuilder blockBuilder = OpBuilder::atBlockEnd(block);

  blockBuilder.create<LLVM::CallOp>(loc, Type(),
                                    blockBuilder.getSymbolRefAttr(kCoroResume),
                                    resumeOp.getArgument(0));

  blockBuilder.create<LLVM::ReturnOp>(loc, ValueRange());
}

//===----------------------------------------------------------------------===//
// async.execute op outlining to the coroutine functions.
//===----------------------------------------------------------------------===//

// Function targeted for coroutine transformation has two additional blocks at
// the end: coroutine cleanup and coroutine suspension.
//
// async.await op lowering additionaly creates a resume block for each
// operation to enable non-blocking waiting via coroutine suspension.
namespace {
struct CoroMachinery {
  Value asyncToken;
  Value coroHandle;
  Block *cleanup;
  Block *suspend;
};
} // namespace

// Builds an coroutine template compatible with LLVM coroutines lowering.
//
//  - `entry` block sets up the coroutine.
//  - `cleanup` block cleans up the coroutine state.
//  - `suspend block after the @llvm.coro.end() defines what value will be
//    returned to the initial caller of a coroutine. Everything before the
//    @llvm.coro.end() will be executed at every suspension point.
//
// Coroutine structure (only the important bits):
//
//   func @async_execute_fn(<function-arguments>) -> !async.token {
//     ^entryBlock(<function-arguments>):
//       %token = <async token> : !async.token // create async runtime token
//       %hdl = llvm.call @llvm.coro.id(...)   // create a coroutine handle
//       br ^cleanup
//
//     ^cleanup:
//       llvm.call @llvm.coro.free(...)        // delete coroutine state
//       br ^suspend
//
//     ^suspend:
//       llvm.call @llvm.coro.end(...)         // marks the end of a coroutine
//       return %token : !async.token
//   }
//
// The actual code for the async.execute operation body region will be inserted
// before the entry block terminator.
//
//
static CoroMachinery setupCoroMachinery(FuncOp func) {
  assert(func.getBody().empty() && "Function must have empty body");

  MLIRContext *ctx = func.getContext();

  auto token = LLVM::LLVMTokenType::get(ctx);
  auto i1 = LLVM::LLVMType::getInt1Ty(ctx);
  auto i32 = LLVM::LLVMType::getInt32Ty(ctx);
  auto i64 = LLVM::LLVMType::getInt64Ty(ctx);
  auto i8Ptr = LLVM::LLVMType::getInt8PtrTy(ctx);

  Block *entryBlock = func.addEntryBlock();
  Location loc = func.getBody().getLoc();

  OpBuilder builder = OpBuilder::atBlockBegin(entryBlock);

  // ------------------------------------------------------------------------ //
  // Allocate async tokens/values that we will return from a ramp function.
  // ------------------------------------------------------------------------ //
  auto createToken =
      builder.create<CallOp>(loc, kCreateToken, TokenType::get(ctx));

  // ------------------------------------------------------------------------ //
  // Initialize coroutine: allocate frame, get coroutine handle.
  // ------------------------------------------------------------------------ //

  // Constants for initializing coroutine frame.
  auto constZero =
      builder.create<LLVM::ConstantOp>(loc, i32, builder.getI32IntegerAttr(0));
  auto constFalse =
      builder.create<LLVM::ConstantOp>(loc, i1, builder.getBoolAttr(false));
  auto nullPtr = builder.create<LLVM::NullOp>(loc, i8Ptr);

  // Get coroutine id: @llvm.coro.id
  auto coroId = builder.create<LLVM::CallOp>(
      loc, token, builder.getSymbolRefAttr(kCoroId),
      ValueRange({constZero, nullPtr, nullPtr, nullPtr}));

  // Get coroutine frame size: @llvm.coro.size.i64
  auto coroSize = builder.create<LLVM::CallOp>(
      loc, i64, builder.getSymbolRefAttr(kCoroSizeI64), ValueRange());

  // Allocate memory for coroutine frame.
  auto coroAlloc = builder.create<LLVM::CallOp>(
      loc, i8Ptr, builder.getSymbolRefAttr(kMalloc),
      ValueRange(coroSize.getResult(0)));

  // Begin a coroutine: @llvm.coro.begin
  auto coroHdl = builder.create<LLVM::CallOp>(
      loc, i8Ptr, builder.getSymbolRefAttr(kCoroBegin),
      ValueRange({coroId.getResult(0), coroAlloc.getResult(0)}));

  Block *cleanupBlock = func.addBlock();
  Block *suspendBlock = func.addBlock();

  // ------------------------------------------------------------------------ //
  // Coroutine cleanup block: deallocate coroutine frame, free the memory.
  // ------------------------------------------------------------------------ //
  builder.setInsertionPointToStart(cleanupBlock);

  // Get a pointer to the coroutine frame memory: @llvm.coro.free.
  auto coroMem = builder.create<LLVM::CallOp>(
      loc, i8Ptr, builder.getSymbolRefAttr(kCoroFree),
      ValueRange({coroId.getResult(0), coroHdl.getResult(0)}));

  // Free the memory.
  builder.create<LLVM::CallOp>(loc, Type(), builder.getSymbolRefAttr(kFree),
                               ValueRange(coroMem.getResult(0)));
  // Branch into the suspend block.
  builder.create<BranchOp>(loc, suspendBlock);

  // ------------------------------------------------------------------------ //
  // Coroutine suspend block: mark the end of a coroutine and return allocated
  // async token.
  // ------------------------------------------------------------------------ //
  builder.setInsertionPointToStart(suspendBlock);

  // Mark the end of a coroutine: @llvm.coro.end.
  builder.create<LLVM::CallOp>(loc, i1, builder.getSymbolRefAttr(kCoroEnd),
                               ValueRange({coroHdl.getResult(0), constFalse}));

  // Return created `async.token` from the suspend block. This will be the
  // return value of a coroutine ramp function.
  builder.create<ReturnOp>(loc, createToken.getResult(0));

  // Branch from the entry block to the cleanup block to create a valid CFG.
  builder.setInsertionPointToEnd(entryBlock);

  builder.create<BranchOp>(loc, cleanupBlock);

  // `async.await` op lowering will create resume blocks for async
  // continuations, and will conditionally branch to cleanup or suspend blocks.

  return {createToken.getResult(0), coroHdl.getResult(0), cleanupBlock,
          suspendBlock};
}

// Adds a suspension point before the `op`, and moves `op` and all operations
// after it into the resume block. Returns a pointer to the resume block.
//
// `coroState` must be a value returned from the call to @llvm.coro.save(...)
// intrinsic (saved coroutine state).
//
// Before:
//
//   ^bb0:
//     "opBefore"(...)
//     "op"(...)
//   ^cleanup: ...
//   ^suspend: ...
//
// After:
//
//   ^bb0:
//     "opBefore"(...)
//     %suspend = llmv.call @llvm.coro.suspend(...)
//     switch %suspend [-1: ^suspend, 0: ^resume, 1: ^cleanup]
//   ^resume:
//     "op"(...)
//   ^cleanup: ...
//   ^suspend: ...
//
static Block *addSuspensionPoint(CoroMachinery coro, Value coroState,
                                 Operation *op) {
  MLIRContext *ctx = op->getContext();
  auto i1 = LLVM::LLVMType::getInt1Ty(ctx);
  auto i8 = LLVM::LLVMType::getInt8Ty(ctx);

  Location loc = op->getLoc();
  Block *splitBlock = op->getBlock();

  // Split the block before `op`, newly added block is the resume block.
  Block *resume = splitBlock->splitBlock(op);

  // Add a coroutine suspension in place of original `op` in the split block.
  OpBuilder builder = OpBuilder::atBlockEnd(splitBlock);

  auto constFalse =
      builder.create<LLVM::ConstantOp>(loc, i1, builder.getBoolAttr(false));

  // Suspend a coroutine: @llvm.coro.suspend
  auto coroSuspend = builder.create<LLVM::CallOp>(
      loc, i8, builder.getSymbolRefAttr(kCoroSuspend),
      ValueRange({coroState, constFalse}));

  // After a suspension point decide if we should branch into resume, cleanup
  // or suspend block of the coroutine (see @llvm.coro.suspend return code
  // documentation).
  auto constZero =
      builder.create<LLVM::ConstantOp>(loc, i8, builder.getI8IntegerAttr(0));
  auto constNegOne =
      builder.create<LLVM::ConstantOp>(loc, i8, builder.getI8IntegerAttr(-1));

  Block *resumeOrCleanup = builder.createBlock(resume);

  // Suspend the coroutine ...?
  builder.setInsertionPointToEnd(splitBlock);
  auto isNegOne = builder.create<LLVM::ICmpOp>(
      loc, LLVM::ICmpPredicate::eq, coroSuspend.getResult(0), constNegOne);
  builder.create<LLVM::CondBrOp>(loc, isNegOne, /*trueDest=*/coro.suspend,
                                 /*falseDest=*/resumeOrCleanup);

  // ... or resume or cleanup the coroutine?
  builder.setInsertionPointToStart(resumeOrCleanup);
  auto isZero = builder.create<LLVM::ICmpOp>(
      loc, LLVM::ICmpPredicate::eq, coroSuspend.getResult(0), constZero);
  builder.create<LLVM::CondBrOp>(loc, isZero, /*trueDest=*/resume,
                                 /*falseDest=*/coro.cleanup);

  return resume;
}

// Outline the body region attached to the `async.execute` op into a standalone
// function.
static std::pair<FuncOp, CoroMachinery>
outlineExecuteOp(SymbolTable &symbolTable, ExecuteOp execute) {
  ModuleOp module = execute.getParentOfType<ModuleOp>();

  MLIRContext *ctx = module.getContext();
  Location loc = execute.getLoc();

  OpBuilder moduleBuilder(module.getBody()->getTerminator());

  // Get values captured by the async region
  llvm::SetVector<mlir::Value> usedAbove;
  getUsedValuesDefinedAbove(execute.body(), usedAbove);

  // Collect types of the captured values.
  auto usedAboveTypes =
      llvm::map_range(usedAbove, [](Value value) { return value.getType(); });
  SmallVector<Type, 4> inputTypes(usedAboveTypes.begin(), usedAboveTypes.end());
  auto outputTypes = execute.getResultTypes();

  auto funcType = moduleBuilder.getFunctionType(inputTypes, outputTypes);
  auto funcAttrs = ArrayRef<NamedAttribute>();

  // TODO: Derive outlined function name from the parent FuncOp (support
  // multiple nested async.execute operations).
  FuncOp func = FuncOp::create(loc, kAsyncFnPrefix, funcType, funcAttrs);
  symbolTable.insert(func, moduleBuilder.getInsertionPoint());

  SymbolTable::setSymbolVisibility(func, SymbolTable::Visibility::Private);

  // Prepare a function for coroutine lowering by adding entry/cleanup/suspend
  // blocks, adding llvm.coro instrinsics and setting up control flow.
  CoroMachinery coro = setupCoroMachinery(func);

  // Suspend async function at the end of an entry block, and resume it using
  // Async execute API (execution will be resumed in a thread managed by the
  // async runtime).
  Block *entryBlock = &func.getBlocks().front();
  OpBuilder builder = OpBuilder::atBlockTerminator(entryBlock);

  // A pointer to coroutine resume intrinsic wrapper.
  auto resumeFnTy = AsyncAPI::resumeFunctionType(ctx);
  auto resumePtr = builder.create<LLVM::AddressOfOp>(
      loc, resumeFnTy.getPointerTo(), kResume);

  // Save the coroutine state: @llvm.coro.save
  auto coroSave = builder.create<LLVM::CallOp>(
      loc, LLVM::LLVMTokenType::get(ctx), builder.getSymbolRefAttr(kCoroSave),
      ValueRange({coro.coroHandle}));

  // Call async runtime API to execute a coroutine in the managed thread.
  SmallVector<Value, 2> executeArgs = {coro.coroHandle, resumePtr.res()};
  builder.create<CallOp>(loc, Type(), kExecute, executeArgs);

  // Split the entry block before the terminator.
  Block *resume = addSuspensionPoint(coro, coroSave.getResult(0),
                                     entryBlock->getTerminator());

  // Map from values defined above the execute op to the function arguments.
  BlockAndValueMapping valueMapping;
  valueMapping.map(usedAbove, func.getArguments());

  // Clone all operations from the execute operation body into the outlined
  // function body, and replace all `async.yield` operations with a call
  // to async runtime to emplace the result token.
  builder.setInsertionPointToStart(resume);
  for (Operation &op : execute.body().getOps()) {
    if (isa<async::YieldOp>(op)) {
      builder.create<CallOp>(loc, kEmplaceToken, Type(), coro.asyncToken);
      continue;
    }
    builder.clone(op, valueMapping);
  }

  // Replace the original `async.execute` with a call to outlined function.
  OpBuilder callBuilder(execute);
  SmallVector<Value, 4> usedAboveArgs(usedAbove.begin(), usedAbove.end());
  auto callOutlinedFunc = callBuilder.create<CallOp>(
      loc, func.getName(), execute.getResultTypes(), usedAboveArgs);
  execute.replaceAllUsesWith(callOutlinedFunc.getResults());
  execute.erase();

  return {func, coro};
}

//===----------------------------------------------------------------------===//
// Convert Async dialect types to LLVM types.
//===----------------------------------------------------------------------===//

namespace {
class AsyncRuntimeTypeConverter : public TypeConverter {
public:
  AsyncRuntimeTypeConverter() { addConversion(convertType); }

  static Type convertType(Type type) {
    MLIRContext *ctx = type.getContext();
    // Convert async tokens to opaque pointers.
    if (type.isa<TokenType>())
      return LLVM::LLVMType::getInt8PtrTy(ctx);
    return type;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Convert types for all call operations to lowered async types.
//===----------------------------------------------------------------------===//

namespace {
class CallOpOpConversion : public ConversionPattern {
public:
  explicit CallOpOpConversion(MLIRContext *ctx)
      : ConversionPattern(CallOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    AsyncRuntimeTypeConverter converter;

    SmallVector<Type, 5> resultTypes;
    converter.convertTypes(op->getResultTypes(), resultTypes);

    CallOp call = cast<CallOp>(op);
    rewriter.replaceOpWithNewOp<CallOp>(op, resultTypes, call.callee(),
                                        call.getOperands());

    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// async.await op lowering to mlirAsyncRuntimeAwaitToken function call.
//===----------------------------------------------------------------------===//

namespace {
class AwaitOpLowering : public ConversionPattern {
public:
  explicit AwaitOpLowering(
      MLIRContext *ctx,
      const llvm::DenseMap<FuncOp, CoroMachinery> &outlinedFunctions)
      : ConversionPattern(AwaitOp::getOperationName(), 1, ctx),
        outlinedFunctions(outlinedFunctions) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // We can only await on the token operand. Async valus are not supported.
    auto await = cast<AwaitOp>(op);
    if (!await.operand().getType().isa<TokenType>())
      return failure();

    // Check if `async.await` is inside the outlined coroutine function.
    auto func = await.getParentOfType<FuncOp>();
    auto outlined = outlinedFunctions.find(func);
    const bool isInCoroutine = outlined != outlinedFunctions.end();

    Location loc = op->getLoc();

    // Inside regular function we convert await operation to the blocking
    // async API await function call.
    if (!isInCoroutine)
      rewriter.create<CallOp>(loc, Type(), kAwaitToken,
                              ValueRange(op->getOperand(0)));

    // Inside the coroutine we convert await operation into coroutine suspension
    // point, and resume execution asynchronously.
    if (isInCoroutine) {
      const CoroMachinery &coro = outlined->getSecond();

      OpBuilder builder(op);
      MLIRContext *ctx = op->getContext();

      // A pointer to coroutine resume intrinsic wrapper.
      auto resumeFnTy = AsyncAPI::resumeFunctionType(ctx);
      auto resumePtr = builder.create<LLVM::AddressOfOp>(
          loc, resumeFnTy.getPointerTo(), kResume);

      // Save the coroutine state: @llvm.coro.save
      auto coroSave = builder.create<LLVM::CallOp>(
          loc, LLVM::LLVMTokenType::get(ctx),
          builder.getSymbolRefAttr(kCoroSave), ValueRange(coro.coroHandle));

      // Call async runtime API to resume a coroutine in the managed thread when
      // the async await argument becomes ready.
      SmallVector<Value, 3> awaitAndExecuteArgs = {
          await.getOperand(), coro.coroHandle, resumePtr.res()};
      builder.create<CallOp>(loc, Type(), kAwaitAndExecute,
                             awaitAndExecuteArgs);

      // Split the entry block before the await operation.
      addSuspensionPoint(coro, coroSave.getResult(0), op);
    }

    // Original operation was replaced by function call or suspension point.
    rewriter.eraseOp(op);

    return success();
  }

private:
  const llvm::DenseMap<FuncOp, CoroMachinery> &outlinedFunctions;
};
} // namespace

//===----------------------------------------------------------------------===//

namespace {
struct ConvertAsyncToLLVMPass
    : public ConvertAsyncToLLVMBase<ConvertAsyncToLLVMPass> {
  void runOnOperation() override;
};

void ConvertAsyncToLLVMPass::runOnOperation() {
  ModuleOp module = getOperation();
  SymbolTable symbolTable(module);

  // Outline all `async.execute` body regions into async functions (coroutines).
  llvm::DenseMap<FuncOp, CoroMachinery> outlinedFunctions;

  WalkResult outlineResult = module.walk([&](ExecuteOp execute) {
    // We currently do not support execute operations that take async
    // token dependencies, async value arguments or produce async results.
    if (!execute.dependencies().empty() || !execute.operands().empty() ||
        !execute.results().empty()) {
      execute.emitOpError(
          "Can't outline async.execute op with async dependencies, arguments "
          "or returned async results");
      return WalkResult::interrupt();
    }

    outlinedFunctions.insert(outlineExecuteOp(symbolTable, execute));

    return WalkResult::advance();
  });

  // Failed to outline all async execute operations.
  if (outlineResult.wasInterrupted()) {
    signalPassFailure();
    return;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Outlined " << outlinedFunctions.size()
                 << " async functions\n";
  });

  // Add declarations for all functions required by the coroutines lowering.
  addResumeFunction(module);
  addAsyncRuntimeApiDeclarations(module);
  addCoroutineIntrinsicsDeclarations(module);
  addCRuntimeDeclarations(module);

  MLIRContext *ctx = &getContext();

  // Convert async dialect types and operations to LLVM dialect.
  AsyncRuntimeTypeConverter converter;
  OwningRewritePatternList patterns;

  populateFuncOpTypeConversionPattern(patterns, ctx, converter);
  patterns.insert<CallOpOpConversion>(ctx);
  patterns.insert<AwaitOpLowering>(ctx, outlinedFunctions);

  ConversionTarget target(*ctx);
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addIllegalDialect<AsyncDialect>();
  target.addDynamicallyLegalOp<FuncOp>(
      [&](FuncOp op) { return converter.isSignatureLegal(op.getType()); });
  target.addDynamicallyLegalOp<CallOp>(
      [&](CallOp op) { return converter.isLegal(op.getResultTypes()); });

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::createConvertAsyncToLLVMPass() {
  return std::make_unique<ConvertAsyncToLLVMPass>();
}
