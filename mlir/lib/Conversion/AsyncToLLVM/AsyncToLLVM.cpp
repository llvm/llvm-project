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

static void addLLVMFuncDecl(ModuleOp module, ImplicitLocOpBuilder &builder,
                            StringRef name, Type ret, ArrayRef<Type> params) {
  if (module.lookupSymbol(name))
    return;
  Type type = LLVM::LLVMFunctionType::get(ret, params);
  builder.create<LLVM::LLVMFuncOp>(name, type);
}

/// Adds coroutine intrinsics declarations to the module.
static void addCoroutineIntrinsicsDeclarations(ModuleOp module) {
  using namespace mlir::LLVM;

  MLIRContext *ctx = module.getContext();
  ImplicitLocOpBuilder builder(module.getLoc(),
                               module.getBody()->getTerminator());

  auto token = LLVMTokenType::get(ctx);
  auto voidTy = LLVMVoidType::get(ctx);

  auto i8 = IntegerType::get(ctx, 8);
  auto i1 = IntegerType::get(ctx, 1);
  auto i32 = IntegerType::get(ctx, 32);
  auto i64 = IntegerType::get(ctx, 64);
  auto i8Ptr = LLVMPointerType::get(i8);

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

  blockBuilder.create<LLVM::CallOp>(TypeRange(),
                                    blockBuilder.getSymbolRefAttr(kCoroResume),
                                    resumeOp.getArgument(0));

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

  Value coroHandle;
  Block *cleanup;
  Block *suspend;
};
} // namespace

/// Builds an coroutine template compatible with LLVM coroutines lowering.
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
///     ^entryBlock(<function-arguments>):
///       %token = <async token> : !async.token    // create async runtime token
///       %value = <async value> : !async.value<T> // create async value
///       %hdl = llvm.call @llvm.coro.id(...)      // create a coroutine handle
///       br ^cleanup
///
///     ^cleanup:
///       llvm.call @llvm.coro.free(...)  // delete coroutine state
///       br ^suspend
///
///     ^suspend:
///       llvm.call @llvm.coro.end(...)  // marks the end of a coroutine
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

  auto token = LLVM::LLVMTokenType::get(ctx);
  auto i1 = IntegerType::get(ctx, 1);
  auto i32 = IntegerType::get(ctx, 32);
  auto i64 = IntegerType::get(ctx, 64);
  auto i8Ptr = LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8));

  Block *entryBlock = func.addEntryBlock();
  Location loc = func.getBody().getLoc();

  auto builder = ImplicitLocOpBuilder::atBlockBegin(loc, entryBlock);

  // ------------------------------------------------------------------------ //
  // Allocate async tokens/values that we will return from a ramp function.
  // ------------------------------------------------------------------------ //
  auto createToken = builder.create<CallOp>(kCreateToken, TokenType::get(ctx));

  // Async value operands and results must be convertible to LLVM types. This is
  // verified before the function outlining.
  LLVMTypeConverter converter(ctx);

  // Returns the size requirements for the async value storage.
  // http://nondot.org/sabre/LLVMNotes/SizeOf-OffsetOf-VariableSizedStructs.txt
  auto sizeOf = [&](ValueType valueType) -> Value {
    auto storedType = converter.convertType(valueType.getValueType());
    auto storagePtrType = LLVM::LLVMPointerType::get(storedType);

    // %Size = getelementptr %T* null, int 1
    // %SizeI = ptrtoint %T* %Size to i32
    auto nullPtr = builder.create<LLVM::NullOp>(loc, storagePtrType);
    auto one = builder.create<LLVM::ConstantOp>(loc, i32,
                                                builder.getI32IntegerAttr(1));
    auto gep = builder.create<LLVM::GEPOp>(loc, storagePtrType, nullPtr,
                                           one.getResult());
    return builder.create<LLVM::PtrToIntOp>(loc, i32, gep);
  };

  // We use the `async.value` type as a return type although it does not match
  // the `kCreateValue` function signature, because it will be later lowered to
  // the runtime type (opaque i8* pointer).
  llvm::SmallVector<CallOp, 4> createValues;
  for (auto resultType : func.getCallableResults().drop_front(1))
    createValues.emplace_back(builder.create<CallOp>(
        loc, kCreateValue, resultType, sizeOf(resultType.cast<ValueType>())));

  auto createdValues = llvm::map_range(
      createValues, [](CallOp call) { return call.getResult(0); });
  llvm::SmallVector<Value, 4> returnValues(createdValues.begin(),
                                           createdValues.end());

  // ------------------------------------------------------------------------ //
  // Initialize coroutine: allocate frame, get coroutine handle.
  // ------------------------------------------------------------------------ //

  // Constants for initializing coroutine frame.
  auto constZero =
      builder.create<LLVM::ConstantOp>(i32, builder.getI32IntegerAttr(0));
  auto constFalse =
      builder.create<LLVM::ConstantOp>(i1, builder.getBoolAttr(false));
  auto nullPtr = builder.create<LLVM::NullOp>(i8Ptr);

  // Get coroutine id: @llvm.coro.id
  auto coroId = builder.create<LLVM::CallOp>(
      token, builder.getSymbolRefAttr(kCoroId),
      ValueRange({constZero, nullPtr, nullPtr, nullPtr}));

  // Get coroutine frame size: @llvm.coro.size.i64
  auto coroSize = builder.create<LLVM::CallOp>(
      i64, builder.getSymbolRefAttr(kCoroSizeI64), ValueRange());

  // Allocate memory for coroutine frame.
  auto coroAlloc =
      builder.create<LLVM::CallOp>(i8Ptr, builder.getSymbolRefAttr(kMalloc),
                                   ValueRange(coroSize.getResult(0)));

  // Begin a coroutine: @llvm.coro.begin
  auto coroHdl = builder.create<LLVM::CallOp>(
      i8Ptr, builder.getSymbolRefAttr(kCoroBegin),
      ValueRange({coroId.getResult(0), coroAlloc.getResult(0)}));

  Block *cleanupBlock = func.addBlock();
  Block *suspendBlock = func.addBlock();

  // ------------------------------------------------------------------------ //
  // Coroutine cleanup block: deallocate coroutine frame, free the memory.
  // ------------------------------------------------------------------------ //
  builder.setInsertionPointToStart(cleanupBlock);

  // Get a pointer to the coroutine frame memory: @llvm.coro.free.
  auto coroMem = builder.create<LLVM::CallOp>(
      i8Ptr, builder.getSymbolRefAttr(kCoroFree),
      ValueRange({coroId.getResult(0), coroHdl.getResult(0)}));

  // Free the memory.
  builder.create<LLVM::CallOp>(TypeRange(), builder.getSymbolRefAttr(kFree),
                               ValueRange(coroMem.getResult(0)));
  // Branch into the suspend block.
  builder.create<BranchOp>(suspendBlock);

  // ------------------------------------------------------------------------ //
  // Coroutine suspend block: mark the end of a coroutine and return allocated
  // async token.
  // ------------------------------------------------------------------------ //
  builder.setInsertionPointToStart(suspendBlock);

  // Mark the end of a coroutine: @llvm.coro.end.
  builder.create<LLVM::CallOp>(i1, builder.getSymbolRefAttr(kCoroEnd),
                               ValueRange({coroHdl.getResult(0), constFalse}));

  // Return created `async.token` and `async.values` from the suspend block.
  // This will be the return value of a coroutine ramp function.
  SmallVector<Value, 4> ret{createToken.getResult(0)};
  ret.insert(ret.end(), returnValues.begin(), returnValues.end());
  builder.create<ReturnOp>(loc, ret);

  // Branch from the entry block to the cleanup block to create a valid CFG.
  builder.setInsertionPointToEnd(entryBlock);

  builder.create<BranchOp>(cleanupBlock);

  // `async.await` op lowering will create resume blocks for async
  // continuations, and will conditionally branch to cleanup or suspend blocks.

  CoroMachinery machinery;
  machinery.asyncToken = createToken.getResult(0);
  machinery.returnValues = returnValues;
  machinery.coroHandle = coroHdl.getResult(0);
  machinery.cleanup = cleanupBlock;
  machinery.suspend = suspendBlock;
  return machinery;
}

/// Add a LLVM coroutine suspension point to the end of suspended block, to
/// resume execution in resume block. The caller is responsible for creating the
/// two suspended/resume blocks with the desired ops contained in each block.
/// This function merely provides the required control flow logic.
///
/// `coroState` must be a value returned from the call to @llvm.coro.save(...)
/// intrinsic (saved coroutine state).
///
/// Before:
///
///   ^bb0:
///     "opBefore"(...)
///     "op"(...)
///   ^cleanup: ...
///   ^suspend: ...
///   ^resume:
///     "op"(...)
///
/// After:
///
///   ^bb0:
///     "opBefore"(...)
///     %suspend = llmv.call @llvm.coro.suspend(...)
///     switch %suspend [-1: ^suspend, 0: ^resume, 1: ^cleanup]
///   ^resume:
///     "op"(...)
///   ^cleanup: ...
///   ^suspend: ...
///
static void addSuspensionPoint(CoroMachinery coro, Value coroState,
                               Operation *op, Block *suspended, Block *resume,
                               OpBuilder &builder) {
  Location loc = op->getLoc();
  MLIRContext *ctx = op->getContext();
  auto i1 = IntegerType::get(ctx, 1);
  auto i8 = IntegerType::get(ctx, 8);

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
  // blocks, adding llvm.coro instrinsics and setting up control flow.
  CoroMachinery coro = setupCoroMachinery(func);

  // Suspend async function at the end of an entry block, and resume it using
  // Async execute API (execution will be resumed in a thread managed by the
  // async runtime).
  Block *entryBlock = &func.getBlocks().front();
  auto builder = ImplicitLocOpBuilder::atBlockTerminator(loc, entryBlock);

  // A pointer to coroutine resume intrinsic wrapper.
  auto resumeFnTy = AsyncAPI::resumeFunctionType(ctx);
  auto resumePtr = builder.create<LLVM::AddressOfOp>(
      LLVM::LLVMPointerType::get(resumeFnTy), kResume);

  // Save the coroutine state: @llvm.coro.save
  auto coroSave = builder.create<LLVM::CallOp>(
      LLVM::LLVMTokenType::get(ctx), builder.getSymbolRefAttr(kCoroSave),
      ValueRange({coro.coroHandle}));

  // Call async runtime API to execute a coroutine in the managed thread.
  SmallVector<Value, 2> executeArgs = {coro.coroHandle, resumePtr.res()};
  builder.create<CallOp>(TypeRange(), kExecute, executeArgs);

  // Split the entry block before the terminator.
  auto *terminatorOp = entryBlock->getTerminator();
  Block *suspended = terminatorOp->getBlock();
  Block *resume = suspended->splitBlock(terminatorOp);
  addSuspensionPoint(coro, coroSave.getResult(0), terminatorOp, suspended,
                     resume, builder);

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
    return llvm::None;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Convert return operations that return async values from async regions.
//===----------------------------------------------------------------------===//

namespace {
class ReturnOpOpConversion : public ConversionPattern {
public:
  explicit ReturnOpOpConversion(TypeConverter &converter, MLIRContext *ctx)
      : ConversionPattern(ReturnOp::getOperationName(), 1, converter, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ReturnOp>(op, operands);
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
  explicit RefCountingOpLowering(TypeConverter &converter, MLIRContext *ctx,
                                 StringRef apiFunctionName)
      : ConversionPattern(RefCountingOp::getOperationName(), 1, converter, ctx),
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

/// async.drop_ref op lowering to mlirAsyncRuntimeDropRef function call.
class AddRefOpLowering : public RefCountingOpLowering<AddRefOp> {
public:
  explicit AddRefOpLowering(TypeConverter &converter, MLIRContext *ctx)
      : RefCountingOpLowering(converter, ctx, kAddRef) {}
};

/// async.create_group op lowering to mlirAsyncRuntimeCreateGroup function call.
class DropRefOpLowering : public RefCountingOpLowering<DropRefOp> {
public:
  explicit DropRefOpLowering(TypeConverter &converter, MLIRContext *ctx)
      : RefCountingOpLowering(converter, ctx, kDropRef) {}
};

} // namespace

//===----------------------------------------------------------------------===//
// async.create_group op lowering to mlirAsyncRuntimeCreateGroup function call.
//===----------------------------------------------------------------------===//

namespace {
class CreateGroupOpLowering : public ConversionPattern {
public:
  explicit CreateGroupOpLowering(TypeConverter &converter, MLIRContext *ctx)
      : ConversionPattern(CreateGroupOp::getOperationName(), 1, converter,
                          ctx) {}

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
  explicit AddToGroupOpLowering(TypeConverter &converter, MLIRContext *ctx)
      : ConversionPattern(AddToGroupOp::getOperationName(), 1, converter, ctx) {
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Currently we can only add tokens to the group.
    auto addToGroup = cast<AddToGroupOp>(op);
    if (!addToGroup.operand().getType().isa<TokenType>())
      return failure();

    auto i64 = IntegerType::get(op->getContext(), 64);
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
      TypeConverter &converter, MLIRContext *ctx,
      const llvm::DenseMap<FuncOp, CoroMachinery> &outlinedFunctions,
      StringRef blockingAwaitFuncName, StringRef coroAwaitFuncName)
      : ConversionPattern(AwaitType::getOperationName(), 1, converter, ctx),
        outlinedFunctions(outlinedFunctions),
        blockingAwaitFuncName(blockingAwaitFuncName),
        coroAwaitFuncName(coroAwaitFuncName) {}

public:
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // We can only await on one the `AwaitableType` (for `await` it can be
    // a `token` or a `value`, for `await_all` it must be a `group`).
    auto await = cast<AwaitType>(op);
    if (!await.operand().getType().template isa<AwaitableType>())
      return failure();

    // Check if await operation is inside the outlined coroutine function.
    auto func = await->template getParentOfType<FuncOp>();
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

      ImplicitLocOpBuilder builder(loc, op, rewriter.getListener());
      MLIRContext *ctx = op->getContext();

      // A pointer to coroutine resume intrinsic wrapper.
      auto resumeFnTy = AsyncAPI::resumeFunctionType(ctx);
      auto resumePtr = builder.create<LLVM::AddressOfOp>(
          LLVM::LLVMPointerType::get(resumeFnTy), kResume);

      // Save the coroutine state: @llvm.coro.save
      auto coroSave = builder.create<LLVM::CallOp>(
          LLVM::LLVMTokenType::get(ctx), builder.getSymbolRefAttr(kCoroSave),
          ValueRange(coro.coroHandle));

      // Call async runtime API to resume a coroutine in the managed thread when
      // the async await argument becomes ready.
      SmallVector<Value, 3> awaitAndExecuteArgs = {operands[0], coro.coroHandle,
                                                   resumePtr.res()};
      builder.create<CallOp>(TypeRange(), coroAwaitFuncName,
                             awaitAndExecuteArgs);

      Block *suspended = op->getBlock();

      // Split the entry block before the await operation.
      Block *resume = rewriter.splitBlock(suspended, Block::iterator(op));
      addSuspensionPoint(coro, coroSave.getResult(0), op, suspended, resume,
                         builder);

      // Make sure that replacement value will be constructed in resume block.
      rewriter.setInsertionPointToStart(resume);
    }

    // Replace or erase the await operation with the new value.
    if (Value replaceWith = getReplacementValue(op, operands[0], rewriter))
      rewriter.replaceOp(op, replaceWith);
    else
      rewriter.eraseOp(op);

    return success();
  }

  virtual Value getReplacementValue(Operation *op, Value operand,
                                    ConversionPatternRewriter &rewriter) const {
    return Value();
  }

private:
  const llvm::DenseMap<FuncOp, CoroMachinery> &outlinedFunctions;
  StringRef blockingAwaitFuncName;
  StringRef coroAwaitFuncName;
};

/// Lowering for `async.await` with a token operand.
class AwaitTokenOpLowering : public AwaitOpLoweringBase<AwaitOp, TokenType> {
  using Base = AwaitOpLoweringBase<AwaitOp, TokenType>;

public:
  explicit AwaitTokenOpLowering(
      TypeConverter &converter, MLIRContext *ctx,
      const llvm::DenseMap<FuncOp, CoroMachinery> &outlinedFunctions)
      : Base(converter, ctx, outlinedFunctions, kAwaitToken,
             kAwaitTokenAndExecute) {}
};

/// Lowering for `async.await` with a value operand.
class AwaitValueOpLowering : public AwaitOpLoweringBase<AwaitOp, ValueType> {
  using Base = AwaitOpLoweringBase<AwaitOp, ValueType>;

public:
  explicit AwaitValueOpLowering(
      TypeConverter &converter, MLIRContext *ctx,
      const llvm::DenseMap<FuncOp, CoroMachinery> &outlinedFunctions)
      : Base(converter, ctx, outlinedFunctions, kAwaitValue,
             kAwaitValueAndExecute) {}

  Value
  getReplacementValue(Operation *op, Value operand,
                      ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto i8Ptr = AsyncAPI::opaquePointerType(rewriter.getContext());

    // Get the underlying value type from the `async.value`.
    auto await = cast<AwaitOp>(op);
    auto valueType = await.operand().getType().cast<ValueType>().getValueType();

    // Get a pointer to an async value storage from the runtime.
    auto storage = rewriter.create<CallOp>(loc, kGetValueStorage,
                                           TypeRange(i8Ptr), operand);

    // Cast from i8* to the pointer pointer to LLVM type.
    auto llvmValueType = getTypeConverter()->convertType(valueType);
    auto castedStorage = rewriter.create<LLVM::BitcastOp>(
        loc, LLVM::LLVMPointerType::get(llvmValueType), storage.getResult(0));

    // Load from the async value storage.
    return rewriter.create<LLVM::LoadOp>(loc, castedStorage.getResult());
  }
};

/// Lowering for `async.await_all` operation.
class AwaitAllOpLowering : public AwaitOpLoweringBase<AwaitAllOp, GroupType> {
  using Base = AwaitOpLoweringBase<AwaitAllOp, GroupType>;

public:
  explicit AwaitAllOpLowering(
      TypeConverter &converter, MLIRContext *ctx,
      const llvm::DenseMap<FuncOp, CoroMachinery> &outlinedFunctions)
      : Base(converter, ctx, outlinedFunctions, kAwaitGroup,
             kAwaitAllAndExecute) {}
};

} // namespace

//===----------------------------------------------------------------------===//
// async.yield op lowerings to the corresponding async runtime function calls.
//===----------------------------------------------------------------------===//

class YieldOpLowering : public ConversionPattern {
public:
  explicit YieldOpLowering(
      TypeConverter &converter, MLIRContext *ctx,
      const llvm::DenseMap<FuncOp, CoroMachinery> &outlinedFunctions)
      : ConversionPattern(async::YieldOp::getOperationName(), 1, converter,
                          ctx),
        outlinedFunctions(outlinedFunctions) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Check if yield operation is inside the outlined coroutine function.
    auto func = op->template getParentOfType<FuncOp>();
    auto outlined = outlinedFunctions.find(func);
    if (outlined == outlinedFunctions.end())
      return op->emitOpError(
          "async.yield is not inside the outlined coroutine function");

    Location loc = op->getLoc();
    const CoroMachinery &coro = outlined->getSecond();

    // Store yielded values into the async values storage and emplace them.
    auto i8Ptr = AsyncAPI::opaquePointerType(rewriter.getContext());

    for (auto tuple : llvm::zip(operands, coro.returnValues)) {
      // Store `yieldValue` into the `asyncValue` storage.
      Value yieldValue = std::get<0>(tuple);
      Value asyncValue = std::get<1>(tuple);

      // Get an opaque i8* pointer to an async value storage from the runtime.
      auto storage = rewriter.create<CallOp>(loc, kGetValueStorage,
                                             TypeRange(i8Ptr), asyncValue);

      // Cast storage pointer to the yielded value type.
      auto castedStorage = rewriter.create<LLVM::BitcastOp>(
          loc, LLVM::LLVMPointerType::get(yieldValue.getType()),
          storage.getResult(0));

      // Store the yielded value into the async value storage.
      rewriter.create<LLVM::StoreOp>(loc, yieldValue,
                                     castedStorage.getResult());

      // Emplace the `async.value` to mark it ready.
      rewriter.create<CallOp>(loc, kEmplaceValue, TypeRange(), asyncValue);
    }

    // Emplace the completion token to mark it ready.
    rewriter.create<CallOp>(loc, kEmplaceToken, TypeRange(), coro.asyncToken);

    // Original operation was replaced by the function call(s).
    rewriter.eraseOp(op);

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
  addCoroutineIntrinsicsDeclarations(module);
  addCRuntimeDeclarations(module);

  // Convert async dialect types and operations to LLVM dialect.
  AsyncRuntimeTypeConverter converter;
  OwningRewritePatternList patterns;

  // Convert async types in function signatures and function calls.
  populateFuncOpTypeConversionPattern(patterns, ctx, converter);
  populateCallOpTypeConversionPattern(patterns, ctx, converter);

  // Convert return operations inside async.execute regions.
  patterns.insert<ReturnOpOpConversion>(converter, ctx);

  // Lower async operations to async runtime API calls.
  patterns.insert<AddRefOpLowering, DropRefOpLowering>(converter, ctx);
  patterns.insert<CreateGroupOpLowering, AddToGroupOpLowering>(converter, ctx);

  // Use LLVM type converter to automatically convert between the async value
  // payload type and LLVM type when loading/storing from/to the async
  // value storage which is an opaque i8* pointer using LLVM load/store ops.
  patterns
      .insert<AwaitTokenOpLowering, AwaitValueOpLowering, AwaitAllOpLowering>(
          llvmConverter, ctx, outlinedFunctions);
  patterns.insert<YieldOpLowering>(llvmConverter, ctx, outlinedFunctions);

  ConversionTarget target(*ctx);
  target.addLegalOp<ConstantOp>();
  target.addLegalDialect<LLVM::LLVMDialect>();

  // All operations from Async dialect must be lowered to the runtime API calls.
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
} // namespace

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
