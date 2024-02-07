//===- AsyncToLLVM.cpp - Convert Async to LLVM dialect --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h"

#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTASYNCTOLLVMPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "convert-async-to-llvm"

using namespace mlir;
using namespace mlir::async;

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
static constexpr const char *kSetTokenError = "mlirAsyncRuntimeSetTokenError";
static constexpr const char *kSetValueError = "mlirAsyncRuntimeSetValueError";
static constexpr const char *kIsTokenError = "mlirAsyncRuntimeIsTokenError";
static constexpr const char *kIsValueError = "mlirAsyncRuntimeIsValueError";
static constexpr const char *kIsGroupError = "mlirAsyncRuntimeIsGroupError";
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
static constexpr const char *kGetNumWorkerThreads =
    "mlirAsyncRuntimGetNumWorkerThreads";

namespace {
/// Async Runtime API function types.
///
/// Because we can't create API function signature for type parametrized
/// async.getValue type, we use opaque pointers (!llvm.ptr) instead. After
/// lowering all async data types become opaque pointers at runtime.
struct AsyncAPI {
  // All async types are lowered to opaque LLVM pointers at runtime.
  static LLVM::LLVMPointerType opaquePointerType(MLIRContext *ctx,
                                                 bool useLLVMOpaquePointers) {
    if (useLLVMOpaquePointers)
      return LLVM::LLVMPointerType::get(ctx);
    return LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8));
  }

  static LLVM::LLVMTokenType tokenType(MLIRContext *ctx) {
    return LLVM::LLVMTokenType::get(ctx);
  }

  static FunctionType addOrDropRefFunctionType(MLIRContext *ctx,
                                               bool useLLVMOpaquePointers) {
    auto ref = opaquePointerType(ctx, useLLVMOpaquePointers);
    auto count = IntegerType::get(ctx, 64);
    return FunctionType::get(ctx, {ref, count}, {});
  }

  static FunctionType createTokenFunctionType(MLIRContext *ctx) {
    return FunctionType::get(ctx, {}, {TokenType::get(ctx)});
  }

  static FunctionType createValueFunctionType(MLIRContext *ctx,
                                              bool useLLVMOpaquePointers) {
    auto i64 = IntegerType::get(ctx, 64);
    auto value = opaquePointerType(ctx, useLLVMOpaquePointers);
    return FunctionType::get(ctx, {i64}, {value});
  }

  static FunctionType createGroupFunctionType(MLIRContext *ctx) {
    auto i64 = IntegerType::get(ctx, 64);
    return FunctionType::get(ctx, {i64}, {GroupType::get(ctx)});
  }

  static FunctionType getValueStorageFunctionType(MLIRContext *ctx,
                                                  bool useLLVMOpaquePointers) {
    auto value = opaquePointerType(ctx, useLLVMOpaquePointers);
    auto storage = opaquePointerType(ctx, useLLVMOpaquePointers);
    return FunctionType::get(ctx, {value}, {storage});
  }

  static FunctionType emplaceTokenFunctionType(MLIRContext *ctx) {
    return FunctionType::get(ctx, {TokenType::get(ctx)}, {});
  }

  static FunctionType emplaceValueFunctionType(MLIRContext *ctx,
                                               bool useLLVMOpaquePointers) {
    auto value = opaquePointerType(ctx, useLLVMOpaquePointers);
    return FunctionType::get(ctx, {value}, {});
  }

  static FunctionType setTokenErrorFunctionType(MLIRContext *ctx) {
    return FunctionType::get(ctx, {TokenType::get(ctx)}, {});
  }

  static FunctionType setValueErrorFunctionType(MLIRContext *ctx,
                                                bool useLLVMOpaquePointers) {
    auto value = opaquePointerType(ctx, useLLVMOpaquePointers);
    return FunctionType::get(ctx, {value}, {});
  }

  static FunctionType isTokenErrorFunctionType(MLIRContext *ctx) {
    auto i1 = IntegerType::get(ctx, 1);
    return FunctionType::get(ctx, {TokenType::get(ctx)}, {i1});
  }

  static FunctionType isValueErrorFunctionType(MLIRContext *ctx,
                                               bool useLLVMOpaquePointers) {
    auto value = opaquePointerType(ctx, useLLVMOpaquePointers);
    auto i1 = IntegerType::get(ctx, 1);
    return FunctionType::get(ctx, {value}, {i1});
  }

  static FunctionType isGroupErrorFunctionType(MLIRContext *ctx) {
    auto i1 = IntegerType::get(ctx, 1);
    return FunctionType::get(ctx, {GroupType::get(ctx)}, {i1});
  }

  static FunctionType awaitTokenFunctionType(MLIRContext *ctx) {
    return FunctionType::get(ctx, {TokenType::get(ctx)}, {});
  }

  static FunctionType awaitValueFunctionType(MLIRContext *ctx,
                                             bool useLLVMOpaquePointers) {
    auto value = opaquePointerType(ctx, useLLVMOpaquePointers);
    return FunctionType::get(ctx, {value}, {});
  }

  static FunctionType awaitGroupFunctionType(MLIRContext *ctx) {
    return FunctionType::get(ctx, {GroupType::get(ctx)}, {});
  }

  static FunctionType executeFunctionType(MLIRContext *ctx,
                                          bool useLLVMOpaquePointers) {
    auto hdl = opaquePointerType(ctx, useLLVMOpaquePointers);
    Type resume;
    if (useLLVMOpaquePointers)
      resume = LLVM::LLVMPointerType::get(ctx);
    else
      resume = LLVM::LLVMPointerType::get(
          resumeFunctionType(ctx, useLLVMOpaquePointers));
    return FunctionType::get(ctx, {hdl, resume}, {});
  }

  static FunctionType addTokenToGroupFunctionType(MLIRContext *ctx) {
    auto i64 = IntegerType::get(ctx, 64);
    return FunctionType::get(ctx, {TokenType::get(ctx), GroupType::get(ctx)},
                             {i64});
  }

  static FunctionType
  awaitTokenAndExecuteFunctionType(MLIRContext *ctx,
                                   bool useLLVMOpaquePointers) {
    auto hdl = opaquePointerType(ctx, useLLVMOpaquePointers);
    Type resume;
    if (useLLVMOpaquePointers)
      resume = LLVM::LLVMPointerType::get(ctx);
    else
      resume = LLVM::LLVMPointerType::get(
          resumeFunctionType(ctx, useLLVMOpaquePointers));
    return FunctionType::get(ctx, {TokenType::get(ctx), hdl, resume}, {});
  }

  static FunctionType
  awaitValueAndExecuteFunctionType(MLIRContext *ctx,
                                   bool useLLVMOpaquePointers) {
    auto value = opaquePointerType(ctx, useLLVMOpaquePointers);
    auto hdl = opaquePointerType(ctx, useLLVMOpaquePointers);
    Type resume;
    if (useLLVMOpaquePointers)
      resume = LLVM::LLVMPointerType::get(ctx);
    else
      resume = LLVM::LLVMPointerType::get(
          resumeFunctionType(ctx, useLLVMOpaquePointers));
    return FunctionType::get(ctx, {value, hdl, resume}, {});
  }

  static FunctionType
  awaitAllAndExecuteFunctionType(MLIRContext *ctx, bool useLLVMOpaquePointers) {
    auto hdl = opaquePointerType(ctx, useLLVMOpaquePointers);
    Type resume;
    if (useLLVMOpaquePointers)
      resume = LLVM::LLVMPointerType::get(ctx);
    else
      resume = LLVM::LLVMPointerType::get(
          resumeFunctionType(ctx, useLLVMOpaquePointers));
    return FunctionType::get(ctx, {GroupType::get(ctx), hdl, resume}, {});
  }

  static FunctionType getNumWorkerThreads(MLIRContext *ctx) {
    return FunctionType::get(ctx, {}, {IndexType::get(ctx)});
  }

  // Auxiliary coroutine resume intrinsic wrapper.
  static Type resumeFunctionType(MLIRContext *ctx, bool useLLVMOpaquePointers) {
    auto voidTy = LLVM::LLVMVoidType::get(ctx);
    auto ptrType = opaquePointerType(ctx, useLLVMOpaquePointers);
    return LLVM::LLVMFunctionType::get(voidTy, {ptrType}, false);
  }
};
} // namespace

/// Adds Async Runtime C API declarations to the module.
static void addAsyncRuntimeApiDeclarations(ModuleOp module,
                                           bool useLLVMOpaquePointers) {
  auto builder =
      ImplicitLocOpBuilder::atBlockEnd(module.getLoc(), module.getBody());

  auto addFuncDecl = [&](StringRef name, FunctionType type) {
    if (module.lookupSymbol(name))
      return;
    builder.create<func::FuncOp>(name, type).setPrivate();
  };

  MLIRContext *ctx = module.getContext();
  addFuncDecl(kAddRef,
              AsyncAPI::addOrDropRefFunctionType(ctx, useLLVMOpaquePointers));
  addFuncDecl(kDropRef,
              AsyncAPI::addOrDropRefFunctionType(ctx, useLLVMOpaquePointers));
  addFuncDecl(kCreateToken, AsyncAPI::createTokenFunctionType(ctx));
  addFuncDecl(kCreateValue,
              AsyncAPI::createValueFunctionType(ctx, useLLVMOpaquePointers));
  addFuncDecl(kCreateGroup, AsyncAPI::createGroupFunctionType(ctx));
  addFuncDecl(kEmplaceToken, AsyncAPI::emplaceTokenFunctionType(ctx));
  addFuncDecl(kEmplaceValue,
              AsyncAPI::emplaceValueFunctionType(ctx, useLLVMOpaquePointers));
  addFuncDecl(kSetTokenError, AsyncAPI::setTokenErrorFunctionType(ctx));
  addFuncDecl(kSetValueError,
              AsyncAPI::setValueErrorFunctionType(ctx, useLLVMOpaquePointers));
  addFuncDecl(kIsTokenError, AsyncAPI::isTokenErrorFunctionType(ctx));
  addFuncDecl(kIsValueError,
              AsyncAPI::isValueErrorFunctionType(ctx, useLLVMOpaquePointers));
  addFuncDecl(kIsGroupError, AsyncAPI::isGroupErrorFunctionType(ctx));
  addFuncDecl(kAwaitToken, AsyncAPI::awaitTokenFunctionType(ctx));
  addFuncDecl(kAwaitValue,
              AsyncAPI::awaitValueFunctionType(ctx, useLLVMOpaquePointers));
  addFuncDecl(kAwaitGroup, AsyncAPI::awaitGroupFunctionType(ctx));
  addFuncDecl(kExecute,
              AsyncAPI::executeFunctionType(ctx, useLLVMOpaquePointers));
  addFuncDecl(kGetValueStorage, AsyncAPI::getValueStorageFunctionType(
                                    ctx, useLLVMOpaquePointers));
  addFuncDecl(kAddTokenToGroup, AsyncAPI::addTokenToGroupFunctionType(ctx));
  addFuncDecl(kAwaitTokenAndExecute, AsyncAPI::awaitTokenAndExecuteFunctionType(
                                         ctx, useLLVMOpaquePointers));
  addFuncDecl(kAwaitValueAndExecute, AsyncAPI::awaitValueAndExecuteFunctionType(
                                         ctx, useLLVMOpaquePointers));
  addFuncDecl(kAwaitAllAndExecute, AsyncAPI::awaitAllAndExecuteFunctionType(
                                       ctx, useLLVMOpaquePointers));
  addFuncDecl(kGetNumWorkerThreads, AsyncAPI::getNumWorkerThreads(ctx));
}

//===----------------------------------------------------------------------===//
// Coroutine resume function wrapper.
//===----------------------------------------------------------------------===//

static constexpr const char *kResume = "__resume";

/// A function that takes a coroutine handle and calls a `llvm.coro.resume`
/// intrinsics. We need this function to be able to pass it to the async
/// runtime execute API.
static void addResumeFunction(ModuleOp module, bool useOpaquePointers) {
  if (module.lookupSymbol(kResume))
    return;

  MLIRContext *ctx = module.getContext();
  auto loc = module.getLoc();
  auto moduleBuilder = ImplicitLocOpBuilder::atBlockEnd(loc, module.getBody());

  auto voidTy = LLVM::LLVMVoidType::get(ctx);
  Type ptrType;
  if (useOpaquePointers)
    ptrType = LLVM::LLVMPointerType::get(ctx);
  else
    ptrType = LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8));

  auto resumeOp = moduleBuilder.create<LLVM::LLVMFuncOp>(
      kResume, LLVM::LLVMFunctionType::get(voidTy, {ptrType}));
  resumeOp.setPrivate();

  auto *block = resumeOp.addEntryBlock();
  auto blockBuilder = ImplicitLocOpBuilder::atBlockEnd(loc, block);

  blockBuilder.create<LLVM::CoroResumeOp>(resumeOp.getArgument(0));
  blockBuilder.create<LLVM::ReturnOp>(ValueRange());
}

//===----------------------------------------------------------------------===//
// Convert Async dialect types to LLVM types.
//===----------------------------------------------------------------------===//

namespace {
/// AsyncRuntimeTypeConverter only converts types from the Async dialect to
/// their runtime type (opaque pointers) and does not convert any other types.
class AsyncRuntimeTypeConverter : public TypeConverter {
  bool llvmOpaquePointers = false;

public:
  AsyncRuntimeTypeConverter(const LowerToLLVMOptions &options)
      : llvmOpaquePointers(options.useOpaquePointers) {
    addConversion([](Type type) { return type; });
    addConversion([this](Type type) {
      return convertAsyncTypes(type, llvmOpaquePointers);
    });

    // Use UnrealizedConversionCast as the bridge so that we don't need to pull
    // in patterns for other dialects.
    auto addUnrealizedCast = [](OpBuilder &builder, Type type,
                                ValueRange inputs, Location loc) {
      auto cast = builder.create<UnrealizedConversionCastOp>(loc, type, inputs);
      return std::optional<Value>(cast.getResult(0));
    };

    addSourceMaterialization(addUnrealizedCast);
    addTargetMaterialization(addUnrealizedCast);
  }

  /// Returns whether LLVM opaque pointers should be used instead of typed
  /// pointers.
  bool useOpaquePointers() const { return llvmOpaquePointers; }

  /// Creates an LLVM pointer type which may either be a typed pointer or an
  /// opaque pointer, depending on what options the converter was constructed
  /// with.
  LLVM::LLVMPointerType getPointerType(Type elementType) {
    if (llvmOpaquePointers)
      return LLVM::LLVMPointerType::get(elementType.getContext());
    return LLVM::LLVMPointerType::get(elementType);
  }

  static std::optional<Type> convertAsyncTypes(Type type,
                                               bool useOpaquePointers) {
    if (isa<TokenType, GroupType, ValueType>(type))
      return AsyncAPI::opaquePointerType(type.getContext(), useOpaquePointers);

    if (isa<CoroIdType, CoroStateType>(type))
      return AsyncAPI::tokenType(type.getContext());
    if (isa<CoroHandleType>(type))
      return AsyncAPI::opaquePointerType(type.getContext(), useOpaquePointers);

    return std::nullopt;
  }
};

/// Base class for conversion patterns requiring AsyncRuntimeTypeConverter
/// as type converter. Allows access to it via the 'getTypeConverter'
/// convenience method.
template <typename SourceOp>
class AsyncOpConversionPattern : public OpConversionPattern<SourceOp> {

  using Base = OpConversionPattern<SourceOp>;

public:
  AsyncOpConversionPattern(AsyncRuntimeTypeConverter &typeConverter,
                           MLIRContext *context)
      : Base(typeConverter, context) {}

  /// Returns the 'AsyncRuntimeTypeConverter' of the pattern.
  AsyncRuntimeTypeConverter *getTypeConverter() const {
    return static_cast<AsyncRuntimeTypeConverter *>(Base::getTypeConverter());
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Convert async.coro.id to @llvm.coro.id intrinsic.
//===----------------------------------------------------------------------===//

namespace {
class CoroIdOpConversion : public AsyncOpConversionPattern<CoroIdOp> {
public:
  using AsyncOpConversionPattern::AsyncOpConversionPattern;

  LogicalResult
  matchAndRewrite(CoroIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto token = AsyncAPI::tokenType(op->getContext());
    auto ptrType = AsyncAPI::opaquePointerType(
        op->getContext(), getTypeConverter()->useOpaquePointers());
    auto loc = op->getLoc();

    // Constants for initializing coroutine frame.
    auto constZero =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), 0);
    auto nullPtr = rewriter.create<LLVM::NullOp>(loc, ptrType);

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
class CoroBeginOpConversion : public AsyncOpConversionPattern<CoroBeginOp> {
public:
  using AsyncOpConversionPattern::AsyncOpConversionPattern;

  LogicalResult
  matchAndRewrite(CoroBeginOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ptrType = AsyncAPI::opaquePointerType(
        op->getContext(), getTypeConverter()->useOpaquePointers());
    auto loc = op->getLoc();

    // Get coroutine frame size: @llvm.coro.size.i64.
    Value coroSize =
        rewriter.create<LLVM::CoroSizeOp>(loc, rewriter.getI64Type());
    // Get coroutine frame alignment: @llvm.coro.align.i64.
    Value coroAlign =
        rewriter.create<LLVM::CoroAlignOp>(loc, rewriter.getI64Type());

    // Round up the size to be multiple of the alignment. Since aligned_alloc
    // requires the size parameter be an integral multiple of the alignment
    // parameter.
    auto makeConstant = [&](uint64_t c) {
      return rewriter.create<LLVM::ConstantOp>(op->getLoc(),
                                               rewriter.getI64Type(), c);
    };
    coroSize = rewriter.create<LLVM::AddOp>(op->getLoc(), coroSize, coroAlign);
    coroSize =
        rewriter.create<LLVM::SubOp>(op->getLoc(), coroSize, makeConstant(1));
    Value negCoroAlign =
        rewriter.create<LLVM::SubOp>(op->getLoc(), makeConstant(0), coroAlign);
    coroSize =
        rewriter.create<LLVM::AndOp>(op->getLoc(), coroSize, negCoroAlign);

    // Allocate memory for the coroutine frame.
    auto allocFuncOp = LLVM::lookupOrCreateAlignedAllocFn(
        op->getParentOfType<ModuleOp>(), rewriter.getI64Type(),
        getTypeConverter()->useOpaquePointers());
    auto coroAlloc = rewriter.create<LLVM::CallOp>(
        loc, allocFuncOp, ValueRange{coroAlign, coroSize});

    // Begin a coroutine: @llvm.coro.begin.
    auto coroId = CoroBeginOpAdaptor(adaptor.getOperands()).getId();
    rewriter.replaceOpWithNewOp<LLVM::CoroBeginOp>(
        op, ptrType, ValueRange({coroId, coroAlloc.getResult()}));

    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Convert async.coro.free to @llvm.coro.free intrinsic.
//===----------------------------------------------------------------------===//

namespace {
class CoroFreeOpConversion : public AsyncOpConversionPattern<CoroFreeOp> {
public:
  using AsyncOpConversionPattern::AsyncOpConversionPattern;

  LogicalResult
  matchAndRewrite(CoroFreeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ptrType = AsyncAPI::opaquePointerType(
        op->getContext(), getTypeConverter()->useOpaquePointers());
    auto loc = op->getLoc();

    // Get a pointer to the coroutine frame memory: @llvm.coro.free.
    auto coroMem =
        rewriter.create<LLVM::CoroFreeOp>(loc, ptrType, adaptor.getOperands());

    // Free the memory.
    auto freeFuncOp =
        LLVM::lookupOrCreateFreeFn(op->getParentOfType<ModuleOp>(),
                                   getTypeConverter()->useOpaquePointers());
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, freeFuncOp,
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
  matchAndRewrite(CoroEndOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // We are not in the block that is part of the unwind sequence.
    auto constFalse = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), rewriter.getI1Type(), rewriter.getBoolAttr(false));
    auto noneToken = rewriter.create<LLVM::NoneTokenOp>(op->getLoc());

    // Mark the end of a coroutine: @llvm.coro.end.
    auto coroHdl = adaptor.getHandle();
    rewriter.create<LLVM::CoroEndOp>(op->getLoc(), rewriter.getI1Type(),
                                     ValueRange({coroHdl, constFalse, noneToken}));
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
  matchAndRewrite(CoroSaveOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Save the coroutine state: @llvm.coro.save
    rewriter.replaceOpWithNewOp<LLVM::CoroSaveOp>(
        op, AsyncAPI::tokenType(op->getContext()), adaptor.getOperands());

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
  matchAndRewrite(CoroSuspendOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto i8 = rewriter.getIntegerType(8);
    auto i32 = rewriter.getI32Type();
    auto loc = op->getLoc();

    // This is not a final suspension point.
    auto constFalse = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI1Type(), rewriter.getBoolAttr(false));

    // Suspend a coroutine: @llvm.coro.suspend
    auto coroState = adaptor.getState();
    auto coroSuspend = rewriter.create<LLVM::CoroSuspendOp>(
        loc, i8, ValueRange({coroState, constFalse}));

    // Cast return code to i32.

    // After a suspension point decide if we should branch into resume, cleanup
    // or suspend block of the coroutine (see @llvm.coro.suspend return code
    // documentation).
    llvm::SmallVector<int32_t, 2> caseValues = {0, 1};
    llvm::SmallVector<Block *, 2> caseDest = {op.getResumeDest(),
                                              op.getCleanupDest()};
    rewriter.replaceOpWithNewOp<LLVM::SwitchOp>(
        op, rewriter.create<LLVM::SExtOp>(loc, i32, coroSuspend.getResult()),
        /*defaultDestination=*/op.getSuspendDest(),
        /*defaultOperands=*/ValueRange(),
        /*caseValues=*/caseValues,
        /*caseDestinations=*/caseDest,
        /*caseOperands=*/ArrayRef<ValueRange>({ValueRange(), ValueRange()}),
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
class RuntimeCreateOpLowering : public ConvertOpToLLVMPattern<RuntimeCreateOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(RuntimeCreateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    TypeConverter *converter = getTypeConverter();
    Type resultType = op->getResultTypes()[0];

    // Tokens creation maps to a simple function call.
    if (isa<TokenType>(resultType)) {
      rewriter.replaceOpWithNewOp<func::CallOp>(
          op, kCreateToken, converter->convertType(resultType));
      return success();
    }

    // To create a value we need to compute the storage requirement.
    if (auto value = dyn_cast<ValueType>(resultType)) {
      // Returns the size requirements for the async value storage.
      auto sizeOf = [&](ValueType valueType) -> Value {
        auto loc = op->getLoc();
        auto i64 = rewriter.getI64Type();

        auto storedType = converter->convertType(valueType.getValueType());
        auto storagePtrType = getTypeConverter()->getPointerType(storedType);

        // %Size = getelementptr %T* null, int 1
        // %SizeI = ptrtoint %T* %Size to i64
        auto nullPtr = rewriter.create<LLVM::NullOp>(loc, storagePtrType);
        auto gep =
            rewriter.create<LLVM::GEPOp>(loc, storagePtrType, storedType,
                                         nullPtr, ArrayRef<LLVM::GEPArg>{1});
        return rewriter.create<LLVM::PtrToIntOp>(loc, i64, gep);
      };

      rewriter.replaceOpWithNewOp<func::CallOp>(op, kCreateValue, resultType,
                                                sizeOf(value));

      return success();
    }

    return rewriter.notifyMatchFailure(op, "unsupported async type");
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Convert async.runtime.create_group to the corresponding runtime API call.
//===----------------------------------------------------------------------===//

namespace {
class RuntimeCreateGroupOpLowering
    : public ConvertOpToLLVMPattern<RuntimeCreateGroupOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(RuntimeCreateGroupOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    TypeConverter *converter = getTypeConverter();
    Type resultType = op.getResult().getType();

    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, kCreateGroup, converter->convertType(resultType),
        adaptor.getOperands());
    return success();
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
  matchAndRewrite(RuntimeSetAvailableOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef apiFuncName =
        TypeSwitch<Type, StringRef>(op.getOperand().getType())
            .Case<TokenType>([](Type) { return kEmplaceToken; })
            .Case<ValueType>([](Type) { return kEmplaceValue; });

    rewriter.replaceOpWithNewOp<func::CallOp>(op, apiFuncName, TypeRange(),
                                              adaptor.getOperands());

    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Convert async.runtime.set_error to the corresponding runtime API call.
//===----------------------------------------------------------------------===//

namespace {
class RuntimeSetErrorOpLowering
    : public OpConversionPattern<RuntimeSetErrorOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RuntimeSetErrorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef apiFuncName =
        TypeSwitch<Type, StringRef>(op.getOperand().getType())
            .Case<TokenType>([](Type) { return kSetTokenError; })
            .Case<ValueType>([](Type) { return kSetValueError; });

    rewriter.replaceOpWithNewOp<func::CallOp>(op, apiFuncName, TypeRange(),
                                              adaptor.getOperands());

    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Convert async.runtime.is_error to the corresponding runtime API call.
//===----------------------------------------------------------------------===//

namespace {
class RuntimeIsErrorOpLowering : public OpConversionPattern<RuntimeIsErrorOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RuntimeIsErrorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef apiFuncName =
        TypeSwitch<Type, StringRef>(op.getOperand().getType())
            .Case<TokenType>([](Type) { return kIsTokenError; })
            .Case<GroupType>([](Type) { return kIsGroupError; })
            .Case<ValueType>([](Type) { return kIsValueError; });

    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, apiFuncName, rewriter.getI1Type(), adaptor.getOperands());
    return success();
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
  matchAndRewrite(RuntimeAwaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef apiFuncName =
        TypeSwitch<Type, StringRef>(op.getOperand().getType())
            .Case<TokenType>([](Type) { return kAwaitToken; })
            .Case<ValueType>([](Type) { return kAwaitValue; })
            .Case<GroupType>([](Type) { return kAwaitGroup; });

    rewriter.create<func::CallOp>(op->getLoc(), apiFuncName, TypeRange(),
                                  adaptor.getOperands());
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
    : public AsyncOpConversionPattern<RuntimeAwaitAndResumeOp> {
public:
  using AsyncOpConversionPattern::AsyncOpConversionPattern;

  LogicalResult
  matchAndRewrite(RuntimeAwaitAndResumeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef apiFuncName =
        TypeSwitch<Type, StringRef>(op.getOperand().getType())
            .Case<TokenType>([](Type) { return kAwaitTokenAndExecute; })
            .Case<ValueType>([](Type) { return kAwaitValueAndExecute; })
            .Case<GroupType>([](Type) { return kAwaitAllAndExecute; });

    Value operand = adaptor.getOperand();
    Value handle = adaptor.getHandle();

    // A pointer to coroutine resume intrinsic wrapper.
    addResumeFunction(op->getParentOfType<ModuleOp>(),
                      getTypeConverter()->useOpaquePointers());
    auto resumeFnTy = AsyncAPI::resumeFunctionType(
        op->getContext(), getTypeConverter()->useOpaquePointers());
    auto resumePtr = rewriter.create<LLVM::AddressOfOp>(
        op->getLoc(), getTypeConverter()->getPointerType(resumeFnTy), kResume);

    rewriter.create<func::CallOp>(
        op->getLoc(), apiFuncName, TypeRange(),
        ValueRange({operand, handle, resumePtr.getRes()}));
    rewriter.eraseOp(op);

    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Convert async.runtime.resume to the corresponding runtime API call.
//===----------------------------------------------------------------------===//

namespace {
class RuntimeResumeOpLowering
    : public AsyncOpConversionPattern<RuntimeResumeOp> {
public:
  using AsyncOpConversionPattern::AsyncOpConversionPattern;

  LogicalResult
  matchAndRewrite(RuntimeResumeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // A pointer to coroutine resume intrinsic wrapper.
    addResumeFunction(op->getParentOfType<ModuleOp>(),
                      getTypeConverter()->useOpaquePointers());
    auto resumeFnTy = AsyncAPI::resumeFunctionType(
        op->getContext(), getTypeConverter()->useOpaquePointers());
    auto resumePtr = rewriter.create<LLVM::AddressOfOp>(
        op->getLoc(), getTypeConverter()->getPointerType(resumeFnTy), kResume);

    // Call async runtime API to execute a coroutine in the managed thread.
    auto coroHdl = adaptor.getHandle();
    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, TypeRange(), kExecute, ValueRange({coroHdl, resumePtr.getRes()}));

    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Convert async.runtime.store to the corresponding runtime API call.
//===----------------------------------------------------------------------===//

namespace {
class RuntimeStoreOpLowering : public ConvertOpToLLVMPattern<RuntimeStoreOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(RuntimeStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    // Get a pointer to the async value storage from the runtime.
    auto ptrType = AsyncAPI::opaquePointerType(
        rewriter.getContext(), getTypeConverter()->useOpaquePointers());
    auto storage = adaptor.getStorage();
    auto storagePtr = rewriter.create<func::CallOp>(
        loc, kGetValueStorage, TypeRange(ptrType), storage);

    // Cast from i8* to the LLVM pointer type.
    auto valueType = op.getValue().getType();
    auto llvmValueType = getTypeConverter()->convertType(valueType);
    if (!llvmValueType)
      return rewriter.notifyMatchFailure(
          op, "failed to convert stored value type to LLVM type");

    Value castedStoragePtr = storagePtr.getResult(0);
    if (!getTypeConverter()->useOpaquePointers())
      castedStoragePtr = rewriter.create<LLVM::BitcastOp>(
          loc, getTypeConverter()->getPointerType(llvmValueType),
          castedStoragePtr);

    // Store the yielded value into the async value storage.
    auto value = adaptor.getValue();
    rewriter.create<LLVM::StoreOp>(loc, value, castedStoragePtr);

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
class RuntimeLoadOpLowering : public ConvertOpToLLVMPattern<RuntimeLoadOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(RuntimeLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    // Get a pointer to the async value storage from the runtime.
    auto ptrType = AsyncAPI::opaquePointerType(
        rewriter.getContext(), getTypeConverter()->useOpaquePointers());
    auto storage = adaptor.getStorage();
    auto storagePtr = rewriter.create<func::CallOp>(
        loc, kGetValueStorage, TypeRange(ptrType), storage);

    // Cast from i8* to the LLVM pointer type.
    auto valueType = op.getResult().getType();
    auto llvmValueType = getTypeConverter()->convertType(valueType);
    if (!llvmValueType)
      return rewriter.notifyMatchFailure(
          op, "failed to convert loaded value type to LLVM type");

    Value castedStoragePtr = storagePtr.getResult(0);
    if (!getTypeConverter()->useOpaquePointers())
      castedStoragePtr = rewriter.create<LLVM::BitcastOp>(
          loc, getTypeConverter()->getPointerType(llvmValueType),
          castedStoragePtr);

    // Load from the casted pointer.
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, llvmValueType,
                                              castedStoragePtr);

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
  matchAndRewrite(RuntimeAddToGroupOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Currently we can only add tokens to the group.
    if (!isa<TokenType>(op.getOperand().getType()))
      return rewriter.notifyMatchFailure(op, "only token type is supported");

    // Replace with a runtime API function call.
    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, kAddTokenToGroup, rewriter.getI64Type(), adaptor.getOperands());

    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Convert async.runtime.num_worker_threads to the corresponding runtime API
// call.
//===----------------------------------------------------------------------===//

namespace {
class RuntimeNumWorkerThreadsOpLowering
    : public OpConversionPattern<RuntimeNumWorkerThreadsOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RuntimeNumWorkerThreadsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Replace with a runtime API function call.
    rewriter.replaceOpWithNewOp<func::CallOp>(op, kGetNumWorkerThreads,
                                              rewriter.getIndexType());

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
  matchAndRewrite(RefCountingOp op, typename RefCountingOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto count = rewriter.create<arith::ConstantOp>(
        op->getLoc(), rewriter.getI64Type(),
        rewriter.getI64IntegerAttr(op.getCount()));

    auto operand = adaptor.getOperand();
    rewriter.replaceOpWithNewOp<func::CallOp>(op, TypeRange(), apiFunctionName,
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
class ReturnOpOpConversion : public OpConversionPattern<func::ReturnOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//

namespace {
struct ConvertAsyncToLLVMPass
    : public impl::ConvertAsyncToLLVMPassBase<ConvertAsyncToLLVMPass> {
  using Base::Base;

  void runOnOperation() override;
};
} // namespace

void ConvertAsyncToLLVMPass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *ctx = module->getContext();

  LowerToLLVMOptions options(ctx);
  options.useOpaquePointers = useOpaquePointers;

  // Add declarations for most functions required by the coroutines lowering.
  // We delay adding the resume function until it's needed because it currently
  // fails to compile unless '-O0' is specified.
  addAsyncRuntimeApiDeclarations(module, useOpaquePointers);

  // Lower async.runtime and async.coro operations to Async Runtime API and
  // LLVM coroutine intrinsics.

  // Convert async dialect types and operations to LLVM dialect.
  AsyncRuntimeTypeConverter converter(options);
  RewritePatternSet patterns(ctx);

  // We use conversion to LLVM type to lower async.runtime load and store
  // operations.
  LLVMTypeConverter llvmConverter(ctx, options);
  llvmConverter.addConversion([&](Type type) {
    return AsyncRuntimeTypeConverter::convertAsyncTypes(
        type, llvmConverter.useOpaquePointers());
  });

  // Convert async types in function signatures and function calls.
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                 converter);
  populateCallOpTypeConversionPattern(patterns, converter);

  // Convert return operations inside async.execute regions.
  patterns.add<ReturnOpOpConversion>(converter, ctx);

  // Lower async.runtime operations to the async runtime API calls.
  patterns.add<RuntimeSetAvailableOpLowering, RuntimeSetErrorOpLowering,
               RuntimeIsErrorOpLowering, RuntimeAwaitOpLowering,
               RuntimeAwaitAndResumeOpLowering, RuntimeResumeOpLowering,
               RuntimeAddToGroupOpLowering, RuntimeNumWorkerThreadsOpLowering,
               RuntimeAddRefOpLowering, RuntimeDropRefOpLowering>(converter,
                                                                  ctx);

  // Lower async.runtime operations that rely on LLVM type converter to convert
  // from async value payload type to the LLVM type.
  patterns.add<RuntimeCreateOpLowering, RuntimeCreateGroupOpLowering,
               RuntimeStoreOpLowering, RuntimeLoadOpLowering>(llvmConverter);

  // Lower async coroutine operations to LLVM coroutine intrinsics.
  patterns
      .add<CoroIdOpConversion, CoroBeginOpConversion, CoroFreeOpConversion,
           CoroEndOpConversion, CoroSaveOpConversion, CoroSuspendOpConversion>(
          converter, ctx);

  ConversionTarget target(*ctx);
  target.addLegalOp<arith::ConstantOp, func::ConstantOp,
                    UnrealizedConversionCastOp>();
  target.addLegalDialect<LLVM::LLVMDialect>();

  // All operations from Async dialect must be lowered to the runtime API and
  // LLVM intrinsics calls.
  target.addIllegalDialect<AsyncDialect>();

  // Add dynamic legality constraints to apply conversions defined above.
  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    return converter.isSignatureLegal(op.getFunctionType());
  });
  target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
    return converter.isLegal(op.getOperandTypes());
  });
  target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
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
  matchAndRewrite(ExecuteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ExecuteOp newOp =
        cast<ExecuteOp>(rewriter.cloneWithoutRegions(*op.getOperation()));
    rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(),
                                newOp.getRegion().end());

    // Set operands and update block argument and result types.
    newOp->setOperands(adaptor.getOperands());
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
  matchAndRewrite(AwaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<AwaitOp>(op, adaptor.getOperands().front());
    return success();
  }
};

// Dummy pattern to trigger the appropriate type conversion / materialization.
class ConvertYieldOpTypes : public OpConversionPattern<async::YieldOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(async::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<async::YieldOp>(op, adaptor.getOperands());
    return success();
  }
};
} // namespace

void mlir::populateAsyncStructuralTypeConversionsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target) {
  typeConverter.addConversion([&](TokenType type) { return type; });
  typeConverter.addConversion([&](ValueType type) {
    Type converted = typeConverter.convertType(type.getValueType());
    return converted ? ValueType::get(converted) : converted;
  });

  patterns.add<ConvertExecuteOpTypes, ConvertAwaitOpTypes, ConvertYieldOpTypes>(
      typeConverter, patterns.getContext());

  target.addDynamicallyLegalOp<AwaitOp, ExecuteOp, async::YieldOp>(
      [&](Operation *op) { return typeConverter.isLegal(op); });
}
