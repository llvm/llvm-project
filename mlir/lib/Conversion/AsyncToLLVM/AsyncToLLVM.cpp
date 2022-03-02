//===- AsyncToLLVM.cpp - Convert Async to LLVM dialect --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h"

#include "../PassDetail.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
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
    auto count = IntegerType::get(ctx, 64);
    return FunctionType::get(ctx, {ref, count}, {});
  }

  static FunctionType createTokenFunctionType(MLIRContext *ctx) {
    return FunctionType::get(ctx, {}, {TokenType::get(ctx)});
  }

  static FunctionType createValueFunctionType(MLIRContext *ctx) {
    auto i64 = IntegerType::get(ctx, 64);
    auto value = opaquePointerType(ctx);
    return FunctionType::get(ctx, {i64}, {value});
  }

  static FunctionType createGroupFunctionType(MLIRContext *ctx) {
    auto i64 = IntegerType::get(ctx, 64);
    return FunctionType::get(ctx, {i64}, {GroupType::get(ctx)});
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

  static FunctionType setTokenErrorFunctionType(MLIRContext *ctx) {
    return FunctionType::get(ctx, {TokenType::get(ctx)}, {});
  }

  static FunctionType setValueErrorFunctionType(MLIRContext *ctx) {
    auto value = opaquePointerType(ctx);
    return FunctionType::get(ctx, {value}, {});
  }

  static FunctionType isTokenErrorFunctionType(MLIRContext *ctx) {
    auto i1 = IntegerType::get(ctx, 1);
    return FunctionType::get(ctx, {TokenType::get(ctx)}, {i1});
  }

  static FunctionType isValueErrorFunctionType(MLIRContext *ctx) {
    auto value = opaquePointerType(ctx);
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

  static FunctionType getNumWorkerThreads(MLIRContext *ctx) {
    return FunctionType::get(ctx, {}, {IndexType::get(ctx)});
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
  auto builder =
      ImplicitLocOpBuilder::atBlockEnd(module.getLoc(), module.getBody());

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
  addFuncDecl(kSetTokenError, AsyncAPI::setTokenErrorFunctionType(ctx));
  addFuncDecl(kSetValueError, AsyncAPI::setValueErrorFunctionType(ctx));
  addFuncDecl(kIsTokenError, AsyncAPI::isTokenErrorFunctionType(ctx));
  addFuncDecl(kIsValueError, AsyncAPI::isValueErrorFunctionType(ctx));
  addFuncDecl(kIsGroupError, AsyncAPI::isGroupErrorFunctionType(ctx));
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
  addFuncDecl(kGetNumWorkerThreads, AsyncAPI::getNumWorkerThreads(ctx));
}

//===----------------------------------------------------------------------===//
// Coroutine resume function wrapper.
//===----------------------------------------------------------------------===//

static constexpr const char *kResume = "__resume";

/// A function that takes a coroutine handle and calls a `llvm.coro.resume`
/// intrinsics. We need this function to be able to pass it to the async
/// runtime execute API.
static void addResumeFunction(ModuleOp module) {
  if (module.lookupSymbol(kResume))
    return;

  MLIRContext *ctx = module.getContext();
  auto loc = module.getLoc();
  auto moduleBuilder = ImplicitLocOpBuilder::atBlockEnd(loc, module.getBody());

  auto voidTy = LLVM::LLVMVoidType::get(ctx);
  auto i8Ptr = LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8));

  auto resumeOp = moduleBuilder.create<LLVM::LLVMFuncOp>(
      kResume, LLVM::LLVMFunctionType::get(voidTy, {i8Ptr}));
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
  matchAndRewrite(CoroIdOp op, OpAdaptor adaptor,
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
  matchAndRewrite(CoroBeginOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto i8Ptr = AsyncAPI::opaquePointerType(op->getContext());
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
      return rewriter.create<LLVM::ConstantOp>(
          op->getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(c));
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
        op->getParentOfType<ModuleOp>(), rewriter.getI64Type());
    auto coroAlloc = rewriter.create<LLVM::CallOp>(
        loc, i8Ptr, SymbolRefAttr::get(allocFuncOp),
        ValueRange{coroAlign, coroSize});

    // Begin a coroutine: @llvm.coro.begin.
    auto coroId = CoroBeginOpAdaptor(adaptor.getOperands()).id();
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
  matchAndRewrite(CoroFreeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto i8Ptr = AsyncAPI::opaquePointerType(op->getContext());
    auto loc = op->getLoc();

    // Get a pointer to the coroutine frame memory: @llvm.coro.free.
    auto coroMem =
        rewriter.create<LLVM::CoroFreeOp>(loc, i8Ptr, adaptor.getOperands());

    // Free the memory.
    auto freeFuncOp =
        LLVM::lookupOrCreateFreeFn(op->getParentOfType<ModuleOp>());
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange(),
                                              SymbolRefAttr::get(freeFuncOp),
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

    // Mark the end of a coroutine: @llvm.coro.end.
    auto coroHdl = adaptor.handle();
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
    auto coroState = adaptor.state();
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
class RuntimeCreateOpLowering : public OpConversionPattern<RuntimeCreateOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RuntimeCreateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    TypeConverter *converter = getTypeConverter();
    Type resultType = op->getResultTypes()[0];

    // Tokens creation maps to a simple function call.
    if (resultType.isa<TokenType>()) {
      rewriter.replaceOpWithNewOp<func::CallOp>(
          op, kCreateToken, converter->convertType(resultType));
      return success();
    }

    // To create a value we need to compute the storage requirement.
    if (auto value = resultType.dyn_cast<ValueType>()) {
      // Returns the size requirements for the async value storage.
      auto sizeOf = [&](ValueType valueType) -> Value {
        auto loc = op->getLoc();
        auto i64 = rewriter.getI64Type();

        auto storedType = converter->convertType(valueType.getValueType());
        auto storagePtrType = LLVM::LLVMPointerType::get(storedType);

        // %Size = getelementptr %T* null, int 1
        // %SizeI = ptrtoint %T* %Size to i64
        auto nullPtr = rewriter.create<LLVM::NullOp>(loc, storagePtrType);
        auto one = rewriter.create<LLVM::ConstantOp>(
            loc, i64, rewriter.getI64IntegerAttr(1));
        auto gep = rewriter.create<LLVM::GEPOp>(loc, storagePtrType, nullPtr,
                                                one.getResult());
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
    : public OpConversionPattern<RuntimeCreateGroupOp> {
public:
  using OpConversionPattern::OpConversionPattern;

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
        TypeSwitch<Type, StringRef>(op.operand().getType())
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
        TypeSwitch<Type, StringRef>(op.operand().getType())
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
        TypeSwitch<Type, StringRef>(op.operand().getType())
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
        TypeSwitch<Type, StringRef>(op.operand().getType())
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
    : public OpConversionPattern<RuntimeAwaitAndResumeOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RuntimeAwaitAndResumeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef apiFuncName =
        TypeSwitch<Type, StringRef>(op.operand().getType())
            .Case<TokenType>([](Type) { return kAwaitTokenAndExecute; })
            .Case<ValueType>([](Type) { return kAwaitValueAndExecute; })
            .Case<GroupType>([](Type) { return kAwaitAllAndExecute; });

    Value operand = adaptor.operand();
    Value handle = adaptor.handle();

    // A pointer to coroutine resume intrinsic wrapper.
    addResumeFunction(op->getParentOfType<ModuleOp>());
    auto resumeFnTy = AsyncAPI::resumeFunctionType(op->getContext());
    auto resumePtr = rewriter.create<LLVM::AddressOfOp>(
        op->getLoc(), LLVM::LLVMPointerType::get(resumeFnTy), kResume);

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
class RuntimeResumeOpLowering : public OpConversionPattern<RuntimeResumeOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RuntimeResumeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // A pointer to coroutine resume intrinsic wrapper.
    addResumeFunction(op->getParentOfType<ModuleOp>());
    auto resumeFnTy = AsyncAPI::resumeFunctionType(op->getContext());
    auto resumePtr = rewriter.create<LLVM::AddressOfOp>(
        op->getLoc(), LLVM::LLVMPointerType::get(resumeFnTy), kResume);

    // Call async runtime API to execute a coroutine in the managed thread.
    auto coroHdl = adaptor.handle();
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
class RuntimeStoreOpLowering : public OpConversionPattern<RuntimeStoreOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RuntimeStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    // Get a pointer to the async value storage from the runtime.
    auto i8Ptr = AsyncAPI::opaquePointerType(rewriter.getContext());
    auto storage = adaptor.storage();
    auto storagePtr = rewriter.create<func::CallOp>(loc, kGetValueStorage,
                                                    TypeRange(i8Ptr), storage);

    // Cast from i8* to the LLVM pointer type.
    auto valueType = op.value().getType();
    auto llvmValueType = getTypeConverter()->convertType(valueType);
    if (!llvmValueType)
      return rewriter.notifyMatchFailure(
          op, "failed to convert stored value type to LLVM type");

    auto castedStoragePtr = rewriter.create<LLVM::BitcastOp>(
        loc, LLVM::LLVMPointerType::get(llvmValueType),
        storagePtr.getResult(0));

    // Store the yielded value into the async value storage.
    auto value = adaptor.value();
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
  matchAndRewrite(RuntimeLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    // Get a pointer to the async value storage from the runtime.
    auto i8Ptr = AsyncAPI::opaquePointerType(rewriter.getContext());
    auto storage = adaptor.storage();
    auto storagePtr = rewriter.create<func::CallOp>(loc, kGetValueStorage,
                                                    TypeRange(i8Ptr), storage);

    // Cast from i8* to the LLVM pointer type.
    auto valueType = op.result().getType();
    auto llvmValueType = getTypeConverter()->convertType(valueType);
    if (!llvmValueType)
      return rewriter.notifyMatchFailure(
          op, "failed to convert loaded value type to LLVM type");

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
  matchAndRewrite(RuntimeAddToGroupOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Currently we can only add tokens to the group.
    if (!op.operand().getType().isa<TokenType>())
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
        rewriter.getI64IntegerAttr(op.count()));

    auto operand = adaptor.operand();
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
    : public ConvertAsyncToLLVMBase<ConvertAsyncToLLVMPass> {
  void runOnOperation() override;
};
} // namespace

void ConvertAsyncToLLVMPass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *ctx = module->getContext();

  // Add declarations for most functions required by the coroutines lowering.
  // We delay adding the resume function until it's needed because it currently
  // fails to compile unless '-O0' is specified.
  addAsyncRuntimeApiDeclarations(module);

  // Lower async.runtime and async.coro operations to Async Runtime API and
  // LLVM coroutine intrinsics.

  // Convert async dialect types and operations to LLVM dialect.
  AsyncRuntimeTypeConverter converter;
  RewritePatternSet patterns(ctx);

  // We use conversion to LLVM type to lower async.runtime load and store
  // operations.
  LLVMTypeConverter llvmConverter(ctx);
  llvmConverter.addConversion(AsyncRuntimeTypeConverter::convertAsyncTypes);

  // Convert async types in function signatures and function calls.
  populateFunctionOpInterfaceTypeConversionPattern<FuncOp>(patterns, converter);
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
               RuntimeStoreOpLowering, RuntimeLoadOpLowering>(llvmConverter,
                                                              ctx);

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
  target.addDynamicallyLegalOp<FuncOp>(
      [&](FuncOp op) { return converter.isSignatureLegal(op.getType()); });
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

std::unique_ptr<OperationPass<ModuleOp>> mlir::createConvertAsyncToLLVMPass() {
  return std::make_unique<ConvertAsyncToLLVMPass>();
}

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
