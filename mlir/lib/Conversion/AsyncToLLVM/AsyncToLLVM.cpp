//===- AsyncToLLVM.cpp - Convert Async to LLVM dialect --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h"

#include "../PassDetail.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/FuncConversions.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
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
static constexpr const char *kCreateValue = "mlirAsyncRuntimeCreateValue";
static constexpr const char *kCreateGroup = "mlirAsyncRuntimeCreateGroup";
static constexpr const char *kEmplaceToken = "mlirAsyncRuntimeEmplaceToken";
static constexpr const char *kEmplaceValue = "mlirAsyncRuntimeEmplaceValue";
static constexpr const char *kAwaitToken = "mlirAsyncRuntimeAwaitToken";
static constexpr const char *kAwaitValue = "mlirAsyncRuntimeAwaitValue";
static constexpr const char *kAwaitGroup = "mlirAsyncRuntimeAwaitAllInGroup";
static constexpr const char *kExecute = "mlirAsyncRuntimeExecute";
static constexpr const char *kGetValueStorage =
    "mlirAsyncRuntimeGetValueStorage";
static constexpr const char *kAddTokenToGroup =
    "mlirAsyncRuntimeAddTokenToGroup";
static constexpr const char *kAwaitTokenAndExecute =
    "mlirAsyncRuntimeAwaitTokenAndExecute";
static constexpr const char *kAwaitValueAndExecute =
    "mlirAsyncRuntimeAwaitValueAndExecute";
static constexpr const char *kAwaitAllAndExecute =
    "mlirAsyncRuntimeAwaitAllInGroupAndExecute";

namespace {
/// Async Runtime API function types.
///
/// Because we can't create API function signature for type parametrized
/// async.value type, we use opaque pointers (!llvm.ptr<i8>) instead. After
/// lowering all async data types become opaque pointers at runtime.
struct AsyncAPI {
  // All async types are lowered to opaque i8* LLVM pointers at runtime.
  static LLVM::LLVMPointerType opaquePointerType(MLIRContext *ctx) {
    return LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8));
  }

  static LLVM::LLVMTokenType tokenType(MLIRContext *ctx) {
    return LLVM::LLVMTokenType::get(ctx);
  }

  static FunctionType addOrDropRefFunctionType(MLIRContext *ctx) {
    auto ref = opaquePointerType(ctx);
    auto count = IntegerType::get(ctx, 32);
    return FunctionType::get(ctx, {ref, count}, {});
  }

  static FunctionType createTokenFunctionType(MLIRContext *ctx) {
    return FunctionType::get(ctx, {}, {TokenType::get(ctx)});
  }

  static FunctionType createValueFunctionType(MLIRContext *ctx) {
    auto i32 = IntegerType::get(ctx, 32);
    auto value = opaquePointerType(ctx);
    return FunctionType::get(ctx, {i32}, {value});
  }

  static FunctionType createGroupFunctionType(MLIRContext *ctx) {
    return FunctionType::get(ctx, {}, {GroupType::get(ctx)});
  }

  static FunctionType getValueStorageFunctionType(MLIRContext *ctx) {
    auto value = opaquePointerType(ctx);
    auto storage = opaquePointerType(ctx);
    return FunctionType::get(ctx, {value}, {storage});
  }

  static FunctionType emplaceTokenFunctionType(MLIRContext *ctx) {
    return FunctionType::get(ctx, {TokenType::get(ctx)}, {});
  }

  static FunctionType emplaceValueFunctionType(MLIRContext *ctx) {
    auto value = opaquePointerType(ctx);
    return FunctionType::get(ctx, {value}, {});
  }

  static FunctionType awaitTokenFunctionType(MLIRContext *ctx) {
    return FunctionType::get(ctx, {TokenType::get(ctx)}, {});
  }

  static FunctionType awaitValueFunctionType(MLIRContext *ctx) {
    auto value = opaquePointerType(ctx);
    return FunctionType::get(ctx, {value}, {});
  }

  static FunctionType awaitGroupFunctionType(MLIRContext *ctx) {
    return FunctionType::get(ctx, {GroupType::get(ctx)}, {});
  }

  static FunctionType executeFunctionType(MLIRContext *ctx) {
    auto hdl = opaquePointerType(ctx);
    auto resume = LLVM::LLVMPointerType::get(resumeFunctionType(ctx));
    return FunctionType::get(ctx, {hdl, resume}, {});
  }

  static FunctionType addTokenToGroupFunctionType(MLIRContext *ctx) {
    auto i64 = IntegerType::get(ctx, 64);
    return FunctionType::get(ctx, {TokenType::get(ctx), GroupType::get(ctx)},
                             {i64});
  }

  static FunctionType awaitTokenAndExecuteFunctionType(MLIRContext *ctx) {
    auto hdl = opaquePointerType(ctx);
    auto resume = LLVM::LLVMPointerType::get(resumeFunctionType(ctx));
    return FunctionType::get(ctx, {TokenType::get(ctx), hdl, resume}, {});
  }

  static FunctionType awaitValueAndExecuteFunctionType(MLIRContext *ctx) {
    auto value = opaquePointerType(ctx);
    auto hdl = opaquePointerType(ctx);
    auto resume = LLVM::LLVMPointerType::get(resumeFunctionType(ctx));
    return FunctionType::get(ctx, {value, hdl, resume}, {});
  }

  static FunctionType awaitAllAndExecuteFunctionType(MLIRContext *ctx) {
    auto hdl = opaquePointerType(ctx);
    auto resume = LLVM::LLVMPointerType::get(resumeFunctionType(ctx));
    return FunctionType::get(ctx, {GroupType::get(ctx), hdl, resume}, {});
  }

  // Auxiliary coroutine resume intrinsic wrapper.
  static Type resumeFunctionType(MLIRContext *ctx) {
    auto voidTy = LLVM::LLVMVoidType::get(ctx);
    auto i8Ptr = opaquePointerType(ctx);
    return LLVM::LLVMFunctionType::get(voidTy, {i8Ptr}, false);
  }
};
} // namespace

/// Adds Async Runtime C API declarations to the module.
static void addAsyncRuntimeApiDeclarations(ModuleOp module) {
  auto builder = ImplicitLocOpBuilder::atBlockTerminator(module.getLoc(),
                                                         module.getBody());

  auto addFuncDecl = [&](StringRef name, FunctionType type) {
    if (module.lookupSymbol(name))
      return;
    builder.create<FuncOp>(name, type).setPrivate();
  };

  MLIRContext *ctx = module.getContext();
  addFuncDecl(kAddRef, AsyncAPI::addOrDropRefFunctionType(ctx));
  addFuncDecl(kDropRef, AsyncAPI::addOrDropRefFunctionType(ctx));
  addFuncDecl(kCreateToken, AsyncAPI::createTokenFunctionType(ctx));
  addFuncDecl(kCreateValue, AsyncAPI::createValueFunctionType(ctx));
  addFuncDecl(kCreateGroup, AsyncAPI::createGroupFunctionType(ctx));
  addFuncDecl(kEmplaceToken, AsyncAPI::emplaceTokenFunctionType(ctx));
  addFuncDecl(kEmplaceValue, AsyncAPI::emplaceValueFunctionType(ctx));
  addFuncDecl(kAwaitToken, AsyncAPI::awaitTokenFunctionType(ctx));
  addFuncDecl(kAwaitValue, AsyncAPI::awaitValueFunctionType(ctx));
  addFuncDecl(kAwaitGroup, AsyncAPI::awaitGroupFunctionType(ctx));
  addFuncDecl(kExecute, AsyncAPI::executeFunctionType(ctx));
  addFuncDecl(kGetValueStorage, AsyncAPI::getValueStorageFunctionType(ctx));
  addFuncDecl(kAddTokenToGroup, AsyncAPI::addTokenToGroupFunctionType(ctx));
  addFuncDecl(kAwaitTokenAndExecute,
              AsyncAPI::awaitTokenAndExecuteFunctionType(ctx));
  addFuncDecl(kAwaitValueAndExecute,
              AsyncAPI::awaitValueAndExecuteFunctionType(ctx));
  addFuncDecl(kAwaitAllAndExecute,
              AsyncAPI::awaitAllAndExecuteFunctionType(ctx));
}

//===----------------------------------------------------------------------===//
// Add malloc/free declarations to the module.
//===----------------------------------------------------------------------===//

static constexpr const char *kMalloc = "malloc";
static constexpr const char *kFree = "free";

static void addLLVMFuncDecl(ModuleOp module, ImplicitLocOpBuilder &builder,
                            StringRef name, Type ret, ArrayRef<Type> params) {
  if (module.lookupSymbol(name))
    return;
  Type type = LLVM::LLVMFunctionType::get(ret, params);
  builder.create<LLVM::LLVMFuncOp>(name, type);
}

/// Adds malloc/free declarations to the module.
static void addCRuntimeDeclarations(ModuleOp module) {
  using namespace mlir::LLVM;

  MLIRContext *ctx = module.getContext();
  ImplicitLocOpBuilder builder(module.getLoc(),
                               module.getBody()->getTerminator());

  auto voidTy = LLVMVoidType::get(ctx);
  auto i64 = IntegerType::get(ctx, 64);
  auto i8Ptr = LLVMPointerType::get(IntegerType::get(ctx, 8));

  addLLVMFuncDecl(module, builder, kMalloc, i8Ptr, {i64});
  addLLVMFuncDecl(module, builder, kFree, voidTy, {i8Ptr});
}

//===----------------------------------------------------------------------===//
// Coroutine resume function wrapper.
//===----------------------------------------------------------------------===//

static constexpr const char *kResume = "__resume";

/// A function that takes a coroutine handle and calls a `llvm.coro.resume`
/// intrinsics. We need this function to be able to pass it to the async
/// runtime execute API.
static void addResumeFunction(ModuleOp module) {
  MLIRContext *ctx = module.getContext();

  OpBuilder moduleBuilder(module.getBody()->getTerminator());
  Location loc = module.getLoc();

  if (module.lookupSymbol(kResume))
    return;

  auto voidTy = LLVM::LLVMVoidType::get(ctx);
  auto i8Ptr = LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8));

  auto resumeOp = moduleBuilder.create<LLVM::LLVMFuncOp>(
      loc, kResume, LLVM::LLVMFunctionType::get(voidTy, {i8Ptr}));
  resumeOp.setPrivate();

  auto *block = resumeOp.addEntryBlock();
  auto blockBuilder = ImplicitLocOpBuilder::atBlockEnd(loc, block);

  blockBuilder.create<LLVM::CoroResumeOp>(resumeOp.getArgument(0));
  blockBuilder.create<LLVM::ReturnOp>(ValueRange());
}

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
  machinery.asyncToken = retToken;
  machinery.returnValues = retValues;
  machinery.coroHandle = coroHdlOp.handle();
  machinery.cleanup = cleanupBlock;
  machinery.suspend = suspendBlock;
  return machinery;
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
  llvm::SetVector<mlir::Value> functionInputs(execute.dependencies().begin(),
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
  symbolTable.insert(func, Block::iterator(module.getBody()->getTerminator()));

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
// Convert Async dialect types to LLVM types.
//===----------------------------------------------------------------------===//

namespace {
/// AsyncRuntimeTypeConverter only converts types from the Async dialect to
/// their runtime type (opaque pointers) and does not convert any other types.
class AsyncRuntimeTypeConverter : public TypeConverter {
public:
  AsyncRuntimeTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion(convertAsyncTypes);
  }

  static Optional<Type> convertAsyncTypes(Type type) {
    if (type.isa<TokenType, GroupType, ValueType>())
      return AsyncAPI::opaquePointerType(type.getContext());

    if (type.isa<CoroIdType, CoroStateType>())
      return AsyncAPI::tokenType(type.getContext());
    if (type.isa<CoroHandleType>())
      return AsyncAPI::opaquePointerType(type.getContext());

    return llvm::None;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Convert async.coro.id to @llvm.coro.id intrinsic.
//===----------------------------------------------------------------------===//

namespace {
class CoroIdOpConversion : public OpConversionPattern<CoroIdOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CoroIdOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto token = AsyncAPI::tokenType(op->getContext());
    auto i8Ptr = AsyncAPI::opaquePointerType(op->getContext());
    auto loc = op->getLoc();

    // Constants for initializing coroutine frame.
    auto constZero = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
    auto nullPtr = rewriter.create<LLVM::NullOp>(loc, i8Ptr);

    // Get coroutine id: @llvm.coro.id.
    rewriter.replaceOpWithNewOp<LLVM::CoroIdOp>(
        op, token, ValueRange({constZero, nullPtr, nullPtr, nullPtr}));

    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Convert async.coro.begin to @llvm.coro.begin intrinsic.
//===----------------------------------------------------------------------===//

namespace {
class CoroBeginOpConversion : public OpConversionPattern<CoroBeginOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CoroBeginOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto i8Ptr = AsyncAPI::opaquePointerType(op->getContext());
    auto loc = op->getLoc();

    // Get coroutine frame size: @llvm.coro.size.i64.
    auto coroSize =
        rewriter.create<LLVM::CoroSizeOp>(loc, rewriter.getI64Type());

    // Allocate memory for the coroutine frame.
    auto coroAlloc = rewriter.create<LLVM::CallOp>(
        loc, i8Ptr, rewriter.getSymbolRefAttr(kMalloc),
        ValueRange(coroSize.getResult()));

    // Begin a coroutine: @llvm.coro.begin.
    auto coroId = CoroBeginOpAdaptor(operands).id();
    rewriter.replaceOpWithNewOp<LLVM::CoroBeginOp>(
        op, i8Ptr, ValueRange({coroId, coroAlloc.getResult(0)}));

    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Convert async.coro.free to @llvm.coro.free intrinsic.
//===----------------------------------------------------------------------===//

namespace {
class CoroFreeOpConversion : public OpConversionPattern<CoroFreeOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CoroFreeOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto i8Ptr = AsyncAPI::opaquePointerType(op->getContext());
    auto loc = op->getLoc();

    // Get a pointer to the coroutine frame memory: @llvm.coro.free.
    auto coroMem = rewriter.create<LLVM::CoroFreeOp>(loc, i8Ptr, operands);

    // Free the memory.
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange(),
                                              rewriter.getSymbolRefAttr(kFree),
                                              ValueRange(coroMem.getResult()));

    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Convert async.coro.end to @llvm.coro.end intrinsic.
//===----------------------------------------------------------------------===//

namespace {
class CoroEndOpConversion : public OpConversionPattern<CoroEndOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CoroEndOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // We are not in the block that is part of the unwind sequence.
    auto constFalse = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), rewriter.getI1Type(), rewriter.getBoolAttr(false));

    // Mark the end of a coroutine: @llvm.coro.end.
    auto coroHdl = CoroEndOpAdaptor(operands).handle();
    rewriter.create<LLVM::CoroEndOp>(op->getLoc(), rewriter.getI1Type(),
                                     ValueRange({coroHdl, constFalse}));
    rewriter.eraseOp(op);

    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Convert async.coro.save to @llvm.coro.save intrinsic.
//===----------------------------------------------------------------------===//

namespace {
class CoroSaveOpConversion : public OpConversionPattern<CoroSaveOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CoroSaveOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Save the coroutine state: @llvm.coro.save
    rewriter.replaceOpWithNewOp<LLVM::CoroSaveOp>(
        op, AsyncAPI::tokenType(op->getContext()), operands);

    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Convert async.coro.suspend to @llvm.coro.suspend intrinsic.
//===----------------------------------------------------------------------===//

namespace {

/// Convert async.coro.suspend to the @llvm.coro.suspend intrinsic call, and
/// branch to the appropriate block based on the return code.
///
/// Before:
///
///   ^suspended:
///     "opBefore"(...)
///     async.coro.suspend %state, ^suspend, ^resume, ^cleanup
///   ^resume:
///     "op"(...)
///   ^cleanup: ...
///   ^suspend: ...
///
/// After:
///
///   ^suspended:
///     "opBefore"(...)
///     %suspend = llmv.intr.coro.suspend ...
///     switch %suspend [-1: ^suspend, 0: ^resume, 1: ^cleanup]
///   ^resume:
///     "op"(...)
///   ^cleanup: ...
///   ^suspend: ...
///
class CoroSuspendOpConversion : public OpConversionPattern<CoroSuspendOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CoroSuspendOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto i8 = rewriter.getIntegerType(8);
    auto i32 = rewriter.getI32Type();
    auto loc = op->getLoc();

    // This is not a final suspension point.
    auto constFalse = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI1Type(), rewriter.getBoolAttr(false));

    // Suspend a coroutine: @llvm.coro.suspend
    auto coroState = CoroSuspendOpAdaptor(operands).state();
    auto coroSuspend = rewriter.create<LLVM::CoroSuspendOp>(
        loc, i8, ValueRange({coroState, constFalse}));

    // Cast return code to i32.

    // After a suspension point decide if we should branch into resume, cleanup
    // or suspend block of the coroutine (see @llvm.coro.suspend return code
    // documentation).
    llvm::SmallVector<int32_t, 2> caseValues = {0, 1};
    llvm::SmallVector<Block *, 2> caseDest = {op.resumeDest(),
                                              op.cleanupDest()};
    rewriter.replaceOpWithNewOp<LLVM::SwitchOp>(
        op, rewriter.create<LLVM::SExtOp>(loc, i32, coroSuspend.getResult()),
        /*defaultDestination=*/op.suspendDest(),
        /*defaultOperands=*/ValueRange(),
        /*caseValues=*/caseValues,
        /*caseDestinations=*/caseDest,
        /*caseOperands=*/ArrayRef<ValueRange>(),
        /*branchWeights=*/ArrayRef<int32_t>());

    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Convert async.runtime.create to the corresponding runtime API call.
//
// To allocate storage for the async values we use getelementptr trick:
// http://nondot.org/sabre/LLVMNotes/SizeOf-OffsetOf-VariableSizedStructs.txt
//===----------------------------------------------------------------------===//

namespace {
class RuntimeCreateOpLowering : public OpConversionPattern<RuntimeCreateOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RuntimeCreateOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    TypeConverter *converter = getTypeConverter();
    Type resultType = op->getResultTypes()[0];

    // Tokens and Groups lowered to function calls without arguments.
    if (resultType.isa<TokenType>() || resultType.isa<GroupType>()) {
      rewriter.replaceOpWithNewOp<CallOp>(
          op, resultType.isa<TokenType>() ? kCreateToken : kCreateGroup,
          converter->convertType(resultType));
      return success();
    }

    // To create a value we need to compute the storage requirement.
    if (auto value = resultType.dyn_cast<ValueType>()) {
      // Returns the size requirements for the async value storage.
      auto sizeOf = [&](ValueType valueType) -> Value {
        auto loc = op->getLoc();
        auto i32 = rewriter.getI32Type();

        auto storedType = converter->convertType(valueType.getValueType());
        auto storagePtrType = LLVM::LLVMPointerType::get(storedType);

        // %Size = getelementptr %T* null, int 1
        // %SizeI = ptrtoint %T* %Size to i32
        auto nullPtr = rewriter.create<LLVM::NullOp>(loc, storagePtrType);
        auto one = rewriter.create<LLVM::ConstantOp>(
            loc, i32, rewriter.getI32IntegerAttr(1));
        auto gep = rewriter.create<LLVM::GEPOp>(loc, storagePtrType, nullPtr,
                                                one.getResult());
        return rewriter.create<LLVM::PtrToIntOp>(loc, i32, gep);
      };

      rewriter.replaceOpWithNewOp<CallOp>(op, kCreateValue, resultType,
                                          sizeOf(value));

      return success();
    }

    return rewriter.notifyMatchFailure(op, "unsupported async type");
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Convert async.runtime.set_available to the corresponding runtime API call.
//===----------------------------------------------------------------------===//

namespace {
class RuntimeSetAvailableOpLowering
    : public OpConversionPattern<RuntimeSetAvailableOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RuntimeSetAvailableOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Type operandType = op.operand().getType();

    if (operandType.isa<TokenType>() || operandType.isa<ValueType>()) {
      rewriter.create<CallOp>(op->getLoc(),
                              operandType.isa<TokenType>() ? kEmplaceToken
                                                           : kEmplaceValue,
                              TypeRange(), operands);
      rewriter.eraseOp(op);
      return success();
    }

    return rewriter.notifyMatchFailure(op, "unsupported async type");
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Convert async.runtime.await to the corresponding runtime API call.
//===----------------------------------------------------------------------===//

namespace {
class RuntimeAwaitOpLowering : public OpConversionPattern<RuntimeAwaitOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RuntimeAwaitOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Type operandType = op.operand().getType();

    StringRef apiFuncName;
    if (operandType.isa<TokenType>())
      apiFuncName = kAwaitToken;
    else if (operandType.isa<ValueType>())
      apiFuncName = kAwaitValue;
    else if (operandType.isa<GroupType>())
      apiFuncName = kAwaitGroup;
    else
      return rewriter.notifyMatchFailure(op, "unsupported async type");

    rewriter.create<CallOp>(op->getLoc(), apiFuncName, TypeRange(), operands);
    rewriter.eraseOp(op);

    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Convert async.runtime.await_and_resume to the corresponding runtime API call.
//===----------------------------------------------------------------------===//

namespace {
class RuntimeAwaitAndResumeOpLowering
    : public OpConversionPattern<RuntimeAwaitAndResumeOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RuntimeAwaitAndResumeOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Type operandType = op.operand().getType();

    StringRef apiFuncName;
    if (operandType.isa<TokenType>())
      apiFuncName = kAwaitTokenAndExecute;
    else if (operandType.isa<ValueType>())
      apiFuncName = kAwaitValueAndExecute;
    else if (operandType.isa<GroupType>())
      apiFuncName = kAwaitAllAndExecute;
    else
      return rewriter.notifyMatchFailure(op, "unsupported async type");

    Value operand = RuntimeAwaitAndResumeOpAdaptor(operands).operand();
    Value handle = RuntimeAwaitAndResumeOpAdaptor(operands).handle();

    // A pointer to coroutine resume intrinsic wrapper.
    auto resumeFnTy = AsyncAPI::resumeFunctionType(op->getContext());
    auto resumePtr = rewriter.create<LLVM::AddressOfOp>(
        op->getLoc(), LLVM::LLVMPointerType::get(resumeFnTy), kResume);

    rewriter.create<CallOp>(op->getLoc(), apiFuncName, TypeRange(),
                            ValueRange({operand, handle, resumePtr.res()}));
    rewriter.eraseOp(op);

    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Convert async.runtime.resume to the corresponding runtime API call.
//===----------------------------------------------------------------------===//

namespace {
class RuntimeResumeOpLowering : public OpConversionPattern<RuntimeResumeOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RuntimeResumeOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // A pointer to coroutine resume intrinsic wrapper.
    auto resumeFnTy = AsyncAPI::resumeFunctionType(op->getContext());
    auto resumePtr = rewriter.create<LLVM::AddressOfOp>(
        op->getLoc(), LLVM::LLVMPointerType::get(resumeFnTy), kResume);

    // Call async runtime API to execute a coroutine in the managed thread.
    auto coroHdl = RuntimeResumeOpAdaptor(operands).handle();
    rewriter.replaceOpWithNewOp<CallOp>(op, TypeRange(), kExecute,
                                        ValueRange({coroHdl, resumePtr.res()}));

    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Convert async.runtime.store to the corresponding runtime API call.
//===----------------------------------------------------------------------===//

namespace {
class RuntimeStoreOpLowering : public OpConversionPattern<RuntimeStoreOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RuntimeStoreOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    // Get a pointer to the async value storage from the runtime.
    auto i8Ptr = AsyncAPI::opaquePointerType(rewriter.getContext());
    auto storage = RuntimeStoreOpAdaptor(operands).storage();
    auto storagePtr = rewriter.create<CallOp>(loc, kGetValueStorage,
                                              TypeRange(i8Ptr), storage);

    // Cast from i8* to the LLVM pointer type.
    auto valueType = op.value().getType();
    auto llvmValueType = getTypeConverter()->convertType(valueType);
    auto castedStoragePtr = rewriter.create<LLVM::BitcastOp>(
        loc, LLVM::LLVMPointerType::get(llvmValueType),
        storagePtr.getResult(0));

    // Store the yielded value into the async value storage.
    auto value = RuntimeStoreOpAdaptor(operands).value();
    rewriter.create<LLVM::StoreOp>(loc, value, castedStoragePtr.getResult());

    // Erase the original runtime store operation.
    rewriter.eraseOp(op);

    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Convert async.runtime.load to the corresponding runtime API call.
//===----------------------------------------------------------------------===//

namespace {
class RuntimeLoadOpLowering : public OpConversionPattern<RuntimeLoadOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RuntimeLoadOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    // Get a pointer to the async value storage from the runtime.
    auto i8Ptr = AsyncAPI::opaquePointerType(rewriter.getContext());
    auto storage = RuntimeLoadOpAdaptor(operands).storage();
    auto storagePtr = rewriter.create<CallOp>(loc, kGetValueStorage,
                                              TypeRange(i8Ptr), storage);

    // Cast from i8* to the LLVM pointer type.
    auto valueType = op.result().getType();
    auto llvmValueType = getTypeConverter()->convertType(valueType);
    auto castedStoragePtr = rewriter.create<LLVM::BitcastOp>(
        loc, LLVM::LLVMPointerType::get(llvmValueType),
        storagePtr.getResult(0));

    // Load from the casted pointer.
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, castedStoragePtr.getResult());

    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Convert async.runtime.add_to_group to the corresponding runtime API call.
//===----------------------------------------------------------------------===//

namespace {
class RuntimeAddToGroupOpLowering
    : public OpConversionPattern<RuntimeAddToGroupOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RuntimeAddToGroupOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Currently we can only add tokens to the group.
    if (!op.operand().getType().isa<TokenType>())
      return rewriter.notifyMatchFailure(op, "only token type is supported");

    // Replace with a runtime API function call.
    rewriter.replaceOpWithNewOp<CallOp>(op, kAddTokenToGroup,
                                        rewriter.getI64Type(), operands);

    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Async reference counting ops lowering (`async.runtime.add_ref` and
// `async.runtime.drop_ref` to the corresponding API calls).
//===----------------------------------------------------------------------===//

namespace {
template <typename RefCountingOp>
class RefCountingOpLowering : public OpConversionPattern<RefCountingOp> {
public:
  explicit RefCountingOpLowering(TypeConverter &converter, MLIRContext *ctx,
                                 StringRef apiFunctionName)
      : OpConversionPattern<RefCountingOp>(converter, ctx),
        apiFunctionName(apiFunctionName) {}

  LogicalResult
  matchAndRewrite(RefCountingOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto count =
        rewriter.create<ConstantOp>(op->getLoc(), rewriter.getI32Type(),
                                    rewriter.getI32IntegerAttr(op.count()));

    auto operand = typename RefCountingOp::Adaptor(operands).operand();
    rewriter.replaceOpWithNewOp<CallOp>(op, TypeRange(), apiFunctionName,
                                        ValueRange({operand, count}));

    return success();
  }

private:
  StringRef apiFunctionName;
};

class RuntimeAddRefOpLowering : public RefCountingOpLowering<RuntimeAddRefOp> {
public:
  explicit RuntimeAddRefOpLowering(TypeConverter &converter, MLIRContext *ctx)
      : RefCountingOpLowering(converter, ctx, kAddRef) {}
};

class RuntimeDropRefOpLowering
    : public RefCountingOpLowering<RuntimeDropRefOp> {
public:
  explicit RuntimeDropRefOpLowering(TypeConverter &converter, MLIRContext *ctx)
      : RefCountingOpLowering(converter, ctx, kDropRef) {}
};
} // namespace

//===----------------------------------------------------------------------===//
// Convert return operations that return async values from async regions.
//===----------------------------------------------------------------------===//

namespace {
class ReturnOpOpConversion : public OpConversionPattern<ReturnOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReturnOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ReturnOp>(op, operands);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Convert async.create_group operation to async.runtime.create
//===----------------------------------------------------------------------===//

namespace {
class CreateGroupOpLowering : public OpConversionPattern<CreateGroupOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CreateGroupOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<RuntimeCreateOp>(
        op, GroupType::get(op->getContext()));
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
  AwaitOpLoweringBase(
      MLIRContext *ctx,
      const llvm::DenseMap<FuncOp, CoroMachinery> &outlinedFunctions)
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
      const CoroMachinery &coro = outlined->getSecond();
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

      // Make sure that replacement value will be constructed in resume block.
      rewriter.setInsertionPointToStart(resume);
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
  const llvm::DenseMap<FuncOp, CoroMachinery> &outlinedFunctions;
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
    // Check if yield operation is inside the outlined coroutine function.
    auto func = op->template getParentOfType<FuncOp>();
    auto outlined = outlinedFunctions.find(func);
    if (outlined == outlinedFunctions.end())
      return rewriter.notifyMatchFailure(
          op, "operation is not inside the outlined async.execute function");

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

namespace {
struct ConvertAsyncToLLVMPass
    : public ConvertAsyncToLLVMBase<ConvertAsyncToLLVMPass> {
  void runOnOperation() override;
};
} // namespace

void ConvertAsyncToLLVMPass::runOnOperation() {
  ModuleOp module = getOperation();
  SymbolTable symbolTable(module);

  MLIRContext *ctx = &getContext();

  // Outline all `async.execute` body regions into async functions (coroutines).
  llvm::DenseMap<FuncOp, CoroMachinery> outlinedFunctions;

  // We use conversion to LLVM type to ensure that all `async.value` operands
  // and results can be lowered to LLVM load and store operations.
  LLVMTypeConverter llvmConverter(ctx);
  llvmConverter.addConversion(AsyncRuntimeTypeConverter::convertAsyncTypes);

  // Returns true if the `async.value` payload is convertible to LLVM.
  auto isConvertibleToLlvm = [&](Type type) -> bool {
    auto valueType = type.cast<ValueType>().getValueType();
    return static_cast<bool>(llvmConverter.convertType(valueType));
  };

  WalkResult outlineResult = module.walk([&](ExecuteOp execute) {
    // All operands and results must be convertible to LLVM.
    if (!llvm::all_of(execute.operands().getTypes(), isConvertibleToLlvm)) {
      execute.emitOpError("operands payload must be convertible to LLVM type");
      return WalkResult::interrupt();
    }
    if (!llvm::all_of(execute.results().getTypes(), isConvertibleToLlvm)) {
      execute.emitOpError("results payload must be convertible to LLVM type");
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
  addCRuntimeDeclarations(module);

  // ------------------------------------------------------------------------ //
  // Lower async operations to async.runtime operations.
  // ------------------------------------------------------------------------ //
  OwningRewritePatternList asyncPatterns;

  // Async lowering does not use type converter because it must preserve all
  // types for async.runtime operations.
  asyncPatterns.insert<CreateGroupOpLowering, AddToGroupOpLowering>(ctx);
  asyncPatterns.insert<AwaitTokenOpLowering, AwaitValueOpLowering,
                       AwaitAllOpLowering, YieldOpLowering>(ctx,
                                                            outlinedFunctions);

  // All high level async operations must be lowered to the runtime operations.
  ConversionTarget runtimeTarget(*ctx);
  runtimeTarget.addLegalDialect<AsyncDialect>();
  runtimeTarget.addIllegalOp<CreateGroupOp, AddToGroupOp>();
  runtimeTarget.addIllegalOp<ExecuteOp, AwaitOp, AwaitAllOp, async::YieldOp>();

  if (failed(applyPartialConversion(module, runtimeTarget,
                                    std::move(asyncPatterns)))) {
    signalPassFailure();
    return;
  }

  // ------------------------------------------------------------------------ //
  // Lower async.runtime and async.coro operations to Async Runtime API and
  // LLVM coroutine intrinsics.
  // ------------------------------------------------------------------------ //

  // Convert async dialect types and operations to LLVM dialect.
  AsyncRuntimeTypeConverter converter;
  OwningRewritePatternList patterns;

  // Convert async types in function signatures and function calls.
  populateFuncOpTypeConversionPattern(patterns, ctx, converter);
  populateCallOpTypeConversionPattern(patterns, ctx, converter);

  // Convert return operations inside async.execute regions.
  patterns.insert<ReturnOpOpConversion>(converter, ctx);

  // Lower async.runtime operations to the async runtime API calls.
  patterns.insert<RuntimeSetAvailableOpLowering, RuntimeAwaitOpLowering,
                  RuntimeAwaitAndResumeOpLowering, RuntimeResumeOpLowering,
                  RuntimeAddToGroupOpLowering, RuntimeAddRefOpLowering,
                  RuntimeDropRefOpLowering>(converter, ctx);

  // Lower async.runtime operations that rely on LLVM type converter to convert
  // from async value payload type to the LLVM type.
  patterns.insert<RuntimeCreateOpLowering, RuntimeStoreOpLowering,
                  RuntimeLoadOpLowering>(llvmConverter, ctx);

  // Lower async coroutine operations to LLVM coroutine intrinsics.
  patterns.insert<CoroIdOpConversion, CoroBeginOpConversion,
                  CoroFreeOpConversion, CoroEndOpConversion,
                  CoroSaveOpConversion, CoroSuspendOpConversion>(converter,
                                                                 ctx);

  ConversionTarget target(*ctx);
  target.addLegalOp<ConstantOp>();
  target.addLegalDialect<LLVM::LLVMDialect>();

  // All operations from Async dialect must be lowered to the runtime API and
  // LLVM intrinsics calls.
  target.addIllegalDialect<AsyncDialect>();

  // Add dynamic legality constraints to apply conversions defined above.
  target.addDynamicallyLegalOp<FuncOp>(
      [&](FuncOp op) { return converter.isSignatureLegal(op.getType()); });
  target.addDynamicallyLegalOp<ReturnOp>(
      [&](ReturnOp op) { return converter.isLegal(op.getOperandTypes()); });
  target.addDynamicallyLegalOp<CallOp>([&](CallOp op) {
    return converter.isSignatureLegal(op.getCalleeType());
  });

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

//===----------------------------------------------------------------------===//
// Patterns for structural type conversions for the Async dialect operations.
//===----------------------------------------------------------------------===//

namespace {
class ConvertExecuteOpTypes : public OpConversionPattern<ExecuteOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ExecuteOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    ExecuteOp newOp =
        cast<ExecuteOp>(rewriter.cloneWithoutRegions(*op.getOperation()));
    rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(),
                                newOp.getRegion().end());

    // Set operands and update block argument and result types.
    newOp->setOperands(operands);
    if (failed(rewriter.convertRegionTypes(&newOp.getRegion(), *typeConverter)))
      return failure();
    for (auto result : newOp.getResults())
      result.setType(typeConverter->convertType(result.getType()));

    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

// Dummy pattern to trigger the appropriate type conversion / materialization.
class ConvertAwaitOpTypes : public OpConversionPattern<AwaitOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AwaitOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<AwaitOp>(op, operands.front());
    return success();
  }
};

// Dummy pattern to trigger the appropriate type conversion / materialization.
class ConvertYieldOpTypes : public OpConversionPattern<async::YieldOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(async::YieldOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<async::YieldOp>(op, operands);
    return success();
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::createConvertAsyncToLLVMPass() {
  return std::make_unique<ConvertAsyncToLLVMPass>();
}

void mlir::populateAsyncStructuralTypeConversionsAndLegality(
    MLIRContext *context, TypeConverter &typeConverter,
    OwningRewritePatternList &patterns, ConversionTarget &target) {
  typeConverter.addConversion([&](TokenType type) { return type; });
  typeConverter.addConversion([&](ValueType type) {
    return ValueType::get(typeConverter.convertType(type.getValueType()));
  });

  patterns
      .insert<ConvertExecuteOpTypes, ConvertAwaitOpTypes, ConvertYieldOpTypes>(
          typeConverter, context);

  target.addDynamicallyLegalOp<AwaitOp, ExecuteOp, async::YieldOp>(
      [&](Operation *op) { return typeConverter.isLegal(op); });
}
