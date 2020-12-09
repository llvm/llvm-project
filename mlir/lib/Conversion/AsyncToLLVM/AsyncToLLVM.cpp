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

static constexpr const char *kAddRef = "mlirAsyncRuntimeAddRef";
static constexpr const char *kDropRef = "mlirAsyncRuntimeDropRef";
static constexpr const char *kCreateToken = "mlirAsyncRuntimeCreateToken";
static constexpr const char *kCreateGroup = "mlirAsyncRuntimeCreateGroup";
static constexpr const char *kEmplaceToken = "mlirAsyncRuntimeEmplaceToken";
static constexpr const char *kAwaitToken = "mlirAsyncRuntimeAwaitToken";
static constexpr const char *kAwaitGroup = "mlirAsyncRuntimeAwaitAllInGroup";
static constexpr const char *kExecute = "mlirAsyncRuntimeExecute";
static constexpr const char *kAddTokenToGroup =
    "mlirAsyncRuntimeAddTokenToGroup";
static constexpr const char *kAwaitAndExecute =
    "mlirAsyncRuntimeAwaitTokenAndExecute";
static constexpr const char *kAwaitAllAndExecute =
    "mlirAsyncRuntimeAwaitAllInGroupAndExecute";

namespace {
// Async Runtime API function types.
struct AsyncAPI {
  static FunctionType addOrDropRefFunctionType(MLIRContext *ctx) {
    auto ref = LLVM::LLVMType::getInt8PtrTy(ctx);
    auto count = IntegerType::get(32, ctx);
    return FunctionType::get({ref, count}, {}, ctx);
  }

  static FunctionType createTokenFunctionType(MLIRContext *ctx) {
    return FunctionType::get({}, {TokenType::get(ctx)}, ctx);
  }

  static FunctionType createGroupFunctionType(MLIRContext *ctx) {
    return FunctionType::get({}, {GroupType::get(ctx)}, ctx);
  }

  static FunctionType emplaceTokenFunctionType(MLIRContext *ctx) {
    return FunctionType::get({TokenType::get(ctx)}, {}, ctx);
  }

  static FunctionType awaitTokenFunctionType(MLIRContext *ctx) {
    return FunctionType::get({TokenType::get(ctx)}, {}, ctx);
  }

  static FunctionType awaitGroupFunctionType(MLIRContext *ctx) {
    return FunctionType::get({GroupType::get(ctx)}, {}, ctx);
  }

  static FunctionType executeFunctionType(MLIRContext *ctx) {
    auto hdl = LLVM::LLVMType::getInt8PtrTy(ctx);
    auto resume = resumeFunctionType(ctx).getPointerTo();
    return FunctionType::get({hdl, resume}, {}, ctx);
  }

  static FunctionType addTokenToGroupFunctionType(MLIRContext *ctx) {
    auto i64 = IntegerType::get(64, ctx);
    return FunctionType::get({TokenType::get(ctx), GroupType::get(ctx)}, {i64},
                             ctx);
  }

  static FunctionType awaitAndExecuteFunctionType(MLIRContext *ctx) {
    auto hdl = LLVM::LLVMType::getInt8PtrTy(ctx);
    auto resume = resumeFunctionType(ctx).getPointerTo();
    return FunctionType::get({TokenType::get(ctx), hdl, resume}, {}, ctx);
  }

  static FunctionType awaitAllAndExecuteFunctionType(MLIRContext *ctx) {
    auto hdl = LLVM::LLVMType::getInt8PtrTy(ctx);
    auto resume = resumeFunctionType(ctx).getPointerTo();
    return FunctionType::get({GroupType::get(ctx), hdl, resume}, {}, ctx);
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

  auto addFuncDecl = [&](StringRef name, FunctionType type) {
    if (module.lookupSymbol(name))
      return;
    builder.create<FuncOp>(module.getLoc(), name, type).setPrivate();
  };

  MLIRContext *ctx = module.getContext();
  addFuncDecl(kAddRef, AsyncAPI::addOrDropRefFunctionType(ctx));
  addFuncDecl(kDropRef, AsyncAPI::addOrDropRefFunctionType(ctx));
  addFuncDecl(kCreateToken, AsyncAPI::createTokenFunctionType(ctx));
  addFuncDecl(kCreateGroup, AsyncAPI::createGroupFunctionType(ctx));
  addFuncDecl(kEmplaceToken, AsyncAPI::emplaceTokenFunctionType(ctx));
  addFuncDecl(kAwaitToken, AsyncAPI::awaitTokenFunctionType(ctx));
  addFuncDecl(kAwaitGroup, AsyncAPI::awaitGroupFunctionType(ctx));
  addFuncDecl(kExecute, AsyncAPI::executeFunctionType(ctx));
  addFuncDecl(kAddTokenToGroup, AsyncAPI::addTokenToGroupFunctionType(ctx));
  addFuncDecl(kAwaitAndExecute, AsyncAPI::awaitAndExecuteFunctionType(ctx));
  addFuncDecl(kAwaitAllAndExecute,
              AsyncAPI::awaitAllAndExecuteFunctionType(ctx));
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

/// Adds an LLVM function declaration to a module.
static void addLLVMFuncDecl(ModuleOp module, OpBuilder &builder, StringRef name,
                            LLVM::LLVMType ret,
                            ArrayRef<LLVM::LLVMType> params) {
  if (module.lookupSymbol(name))
    return;
  LLVM::LLVMType type = LLVM::LLVMType::getFunctionTy(ret, params, false);
  builder.create<LLVM::LLVMFuncOp>(module.getLoc(), name, type);
}

/// Adds coroutine intrinsics declarations to the module.
static void addCoroutineIntrinsicsDeclarations(ModuleOp module) {
  using namespace mlir::LLVM;

  MLIRContext *ctx = module.getContext();
  OpBuilder builder(module.getBody()->getTerminator());

  auto token = LLVMTokenType::get(ctx);
  auto voidTy = LLVMType::getVoidTy(ctx);

  auto i8 = LLVMType::getInt8Ty(ctx);
  auto i1 = LLVMType::getInt1Ty(ctx);
  auto i32 = LLVMType::getInt32Ty(ctx);
  auto i64 = LLVMType::getInt64Ty(ctx);
  auto i8Ptr = LLVMType::getInt8PtrTy(ctx);

  addLLVMFuncDecl(module, builder, kCoroId, token, {i32, i8Ptr, i8Ptr, i8Ptr});
  addLLVMFuncDecl(module, builder, kCoroSizeI64, i64, {});
  addLLVMFuncDecl(module, builder, kCoroBegin, i8Ptr, {token, i8Ptr});
  addLLVMFuncDecl(module, builder, kCoroSave, token, {i8Ptr});
  addLLVMFuncDecl(module, builder, kCoroSuspend, i8, {token, i1});
  addLLVMFuncDecl(module, builder, kCoroEnd, i1, {i8Ptr, i1});
  addLLVMFuncDecl(module, builder, kCoroFree, i8Ptr, {token, i8Ptr});
  addLLVMFuncDecl(module, builder, kCoroResume, voidTy, {i8Ptr});
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
  OpBuilder builder(module.getBody()->getTerminator());

  auto voidTy = LLVMType::getVoidTy(ctx);
  auto i64 = LLVMType::getInt64Ty(ctx);
  auto i8Ptr = LLVMType::getInt8PtrTy(ctx);

  addLLVMFuncDecl(module, builder, kMalloc, i8Ptr, {i64});
  addLLVMFuncDecl(module, builder, kFree, voidTy, {i8Ptr});
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
      loc, kResume, LLVM::LLVMType::getFunctionTy(voidTy, {i8Ptr}, false));
  resumeOp.setPrivate();

  auto *block = resumeOp.addEntryBlock();
  OpBuilder blockBuilder = OpBuilder::atBlockEnd(block);

  blockBuilder.create<LLVM::CallOp>(loc, TypeRange(),
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
  builder.create<LLVM::CallOp>(loc, TypeRange(),
                               builder.getSymbolRefAttr(kFree),
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

// Add a LLVM coroutine suspension point to the end of suspended block, to
// resume execution in resume block. The caller is responsible for creating the
// two suspended/resume blocks with the desired ops contained in each block.
// This function merely provides the required control flow logic.
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
//   ^resume:
//     "op"(...)
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
static void addSuspensionPoint(CoroMachinery coro, Value coroState,
                               Operation *op, Block *suspended, Block *resume,
                               OpBuilder &builder) {
  Location loc = op->getLoc();
  MLIRContext *ctx = op->getContext();
  auto i1 = LLVM::LLVMType::getInt1Ty(ctx);
  auto i8 = LLVM::LLVMType::getInt8Ty(ctx);

  // Add a coroutine suspension in place of original `op` in the split block.
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(suspended);

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
  builder.setInsertionPointToEnd(suspended);
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
}

// Outline the body region attached to the `async.execute` op into a standalone
// function.
//
// Note that this is not reversible transformation.
static std::pair<FuncOp, CoroMachinery>
outlineExecuteOp(SymbolTable &symbolTable, ExecuteOp execute) {
  ModuleOp module = execute.getParentOfType<ModuleOp>();

  MLIRContext *ctx = module.getContext();
  Location loc = execute.getLoc();

  OpBuilder moduleBuilder(module.getBody()->getTerminator());

  // Collect all outlined function inputs.
  llvm::SetVector<mlir::Value> functionInputs(execute.dependencies().begin(),
                                              execute.dependencies().end());
  getUsedValuesDefinedAbove(execute.body(), functionInputs);

  // Collect types for the outlined function inputs and outputs.
  auto typesRange = llvm::map_range(
      functionInputs, [](Value value) { return value.getType(); });
  SmallVector<Type, 4> inputTypes(typesRange.begin(), typesRange.end());
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
  builder.create<CallOp>(loc, TypeRange(), kExecute, executeArgs);

  // Split the entry block before the terminator.
  auto *terminatorOp = entryBlock->getTerminator();
  Block *suspended = terminatorOp->getBlock();
  Block *resume = suspended->splitBlock(terminatorOp);
  addSuspensionPoint(coro, coroSave.getResult(0), terminatorOp, suspended,
                     resume, builder);

  // Await on all dependencies before starting to execute the body region.
  builder.setInsertionPointToStart(resume);
  for (size_t i = 0; i < execute.dependencies().size(); ++i)
    builder.create<AwaitOp>(loc, func.getArgument(i));

  // Map from function inputs defined above the execute op to the function
  // arguments.
  BlockAndValueMapping valueMapping;
  valueMapping.map(functionInputs, func.getArguments());

  // Clone all operations from the execute operation body into the outlined
  // function body, and replace all `async.yield` operations with a call
  // to async runtime to emplace the result token.
  for (Operation &op : execute.body().getOps()) {
    if (isa<async::YieldOp>(op)) {
      builder.create<CallOp>(loc, kEmplaceToken, TypeRange(), coro.asyncToken);
      continue;
    }
    builder.clone(op, valueMapping);
  }

  // Replace the original `async.execute` with a call to outlined function.
  OpBuilder callBuilder(execute);
  auto callOutlinedFunc =
      callBuilder.create<CallOp>(loc, func.getName(), execute.getResultTypes(),
                                 functionInputs.getArrayRef());
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
    // Convert async tokens and groups to opaque pointers.
    if (type.isa<TokenType, GroupType>())
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
                                        operands);

    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Async reference counting ops lowering (`async.add_ref` and `async.drop_ref`
// to the corresponding API calls).
//===----------------------------------------------------------------------===//

namespace {

template <typename RefCountingOp>
class RefCountingOpLowering : public ConversionPattern {
public:
  explicit RefCountingOpLowering(MLIRContext *ctx, StringRef apiFunctionName)
      : ConversionPattern(RefCountingOp::getOperationName(), 1, ctx),
        apiFunctionName(apiFunctionName) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    RefCountingOp refCountingOp = cast<RefCountingOp>(op);

    auto count = rewriter.create<ConstantOp>(
        op->getLoc(), rewriter.getI32Type(),
        rewriter.getI32IntegerAttr(refCountingOp.count()));

    rewriter.replaceOpWithNewOp<CallOp>(op, TypeRange(), apiFunctionName,
                                        ValueRange({operands[0], count}));

    return success();
  }

private:
  StringRef apiFunctionName;
};

// async.drop_ref op lowering to mlirAsyncRuntimeDropRef function call.
class AddRefOpLowering : public RefCountingOpLowering<AddRefOp> {
public:
  explicit AddRefOpLowering(MLIRContext *ctx)
      : RefCountingOpLowering(ctx, kAddRef) {}
};

// async.create_group op lowering to mlirAsyncRuntimeCreateGroup function call.
class DropRefOpLowering : public RefCountingOpLowering<DropRefOp> {
public:
  explicit DropRefOpLowering(MLIRContext *ctx)
      : RefCountingOpLowering(ctx, kDropRef) {}
};

} // namespace

//===----------------------------------------------------------------------===//
// async.create_group op lowering to mlirAsyncRuntimeCreateGroup function call.
//===----------------------------------------------------------------------===//

namespace {
class CreateGroupOpLowering : public ConversionPattern {
public:
  explicit CreateGroupOpLowering(MLIRContext *ctx)
      : ConversionPattern(CreateGroupOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto retTy = GroupType::get(op->getContext());
    rewriter.replaceOpWithNewOp<CallOp>(op, kCreateGroup, retTy);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// async.add_to_group op lowering to runtime function call.
//===----------------------------------------------------------------------===//

namespace {
class AddToGroupOpLowering : public ConversionPattern {
public:
  explicit AddToGroupOpLowering(MLIRContext *ctx)
      : ConversionPattern(AddToGroupOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Currently we can only add tokens to the group.
    auto addToGroup = cast<AddToGroupOp>(op);
    if (!addToGroup.operand().getType().isa<TokenType>())
      return failure();

    auto i64 = IntegerType::get(64, op->getContext());
    rewriter.replaceOpWithNewOp<CallOp>(op, kAddTokenToGroup, i64, operands);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// async.await and async.await_all op lowerings to the corresponding async
// runtime function calls.
//===----------------------------------------------------------------------===//

namespace {

template <typename AwaitType, typename AwaitableType>
class AwaitOpLoweringBase : public ConversionPattern {
protected:
  explicit AwaitOpLoweringBase(
      MLIRContext *ctx,
      const llvm::DenseMap<FuncOp, CoroMachinery> &outlinedFunctions,
      StringRef blockingAwaitFuncName, StringRef coroAwaitFuncName)
      : ConversionPattern(AwaitType::getOperationName(), 1, ctx),
        outlinedFunctions(outlinedFunctions),
        blockingAwaitFuncName(blockingAwaitFuncName),
        coroAwaitFuncName(coroAwaitFuncName) {}

public:
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // We can only await on one the `AwaitableType` (for `await` it can be
    // only a `token`, for `await_all` it is a `group`).
    auto await = cast<AwaitType>(op);
    if (!await.operand().getType().template isa<AwaitableType>())
      return failure();

    // Check if await operation is inside the outlined coroutine function.
    auto func = await.template getParentOfType<FuncOp>();
    auto outlined = outlinedFunctions.find(func);
    const bool isInCoroutine = outlined != outlinedFunctions.end();

    Location loc = op->getLoc();

    // Inside regular function we convert await operation to the blocking
    // async API await function call.
    if (!isInCoroutine)
      rewriter.create<CallOp>(loc, TypeRange(), blockingAwaitFuncName,
                              ValueRange(operands[0]));

    // Inside the coroutine we convert await operation into coroutine suspension
    // point, and resume execution asynchronously.
    if (isInCoroutine) {
      const CoroMachinery &coro = outlined->getSecond();

      OpBuilder builder(op, rewriter.getListener());
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
      SmallVector<Value, 3> awaitAndExecuteArgs = {operands[0], coro.coroHandle,
                                                   resumePtr.res()};
      builder.create<CallOp>(loc, TypeRange(), coroAwaitFuncName,
                             awaitAndExecuteArgs);

      Block *suspended = op->getBlock();

      // Split the entry block before the await operation.
      Block *resume = rewriter.splitBlock(suspended, Block::iterator(op));
      addSuspensionPoint(coro, coroSave.getResult(0), op, suspended, resume,
                         builder);
    }

    // Original operation was replaced by function call or suspension point.
    rewriter.eraseOp(op);

    return success();
  }

private:
  const llvm::DenseMap<FuncOp, CoroMachinery> &outlinedFunctions;
  StringRef blockingAwaitFuncName;
  StringRef coroAwaitFuncName;
};

// Lowering for `async.await` operation (only token operands are supported).
class AwaitOpLowering : public AwaitOpLoweringBase<AwaitOp, TokenType> {
  using Base = AwaitOpLoweringBase<AwaitOp, TokenType>;

public:
  explicit AwaitOpLowering(
      MLIRContext *ctx,
      const llvm::DenseMap<FuncOp, CoroMachinery> &outlinedFunctions)
      : Base(ctx, outlinedFunctions, kAwaitToken, kAwaitAndExecute) {}
};

// Lowering for `async.await_all` operation.
class AwaitAllOpLowering : public AwaitOpLoweringBase<AwaitAllOp, GroupType> {
  using Base = AwaitOpLoweringBase<AwaitAllOp, GroupType>;

public:
  explicit AwaitAllOpLowering(
      MLIRContext *ctx,
      const llvm::DenseMap<FuncOp, CoroMachinery> &outlinedFunctions)
      : Base(ctx, outlinedFunctions, kAwaitGroup, kAwaitAllAndExecute) {}
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
    // We currently do not support execute operations that have async value
    // operands or produce async results.
    if (!execute.operands().empty() || !execute.results().empty()) {
      execute.emitOpError("can't outline async.execute op with async value "
                          "operands or returned async results");
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
  patterns.insert<AddRefOpLowering, DropRefOpLowering>(ctx);
  patterns.insert<CreateGroupOpLowering, AddToGroupOpLowering>(ctx);
  patterns.insert<AwaitOpLowering, AwaitAllOpLowering>(ctx, outlinedFunctions);

  ConversionTarget target(*ctx);
  target.addLegalOp<ConstantOp>();
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
