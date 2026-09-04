//===- ACCExecutableDirectivePatterns.cpp - ACC exec patterns ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lowers OpenACC executable directives (init, shutdown, wait, set) to calls to
// an OpenACC offloading runtime compiler interface.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/OpenACCToLLVM/ACCToLLVM.h"
#include "mlir/Conversion/OpenACCToLLVM/ACCToLLVMUtils.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"

#include <cstdint>
#include <iterator>

using namespace mlir;
using namespace mlir::acc;

namespace {
static Value castToI64(Location loc, Value value,
                       ConversionPatternRewriter &rewriter) {
  Type i64Ty = IntegerType::get(rewriter.getContext(), 64);
  unsigned bitwidth = value.getType().getIntOrFloatBitWidth();
  if (bitwidth > 64)
    return arith::TruncIOp::create(rewriter, loc, i64Ty, value);
  if (bitwidth < 64)
    return arith::ExtSIOp::create(rewriter, loc, i64Ty, value);
  return value;
}

static Value getAsyncQueue(WaitOp op, ConversionPatternRewriter &rewriter,
                           const ACCRuntimeCallConfig &config) {
  Location loc = op->getLoc();
  Type i64Ty = IntegerType::get(rewriter.getContext(), 64);
  if (op.getAsync())
    return LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                    config.getAsyncNoValueRuntimeValue());
  if (Value asyncValue = op.getAsyncOperand()) {
    asyncValue = rewriter.getRemappedValue(asyncValue);
    return castToI64(loc, asyncValue, rewriter);
  }
  return LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                  config.getAsyncSyncRuntimeValue());
}

static LogicalResult createIfThen(Location loc, Value ifCond,
                                  ConversionPatternRewriter &rewriter,
                                  function_ref<LogicalResult()> thenFn) {
  Block *parentBlock = rewriter.getInsertionBlock();
  Block *continueBlock =
      rewriter.splitBlock(parentBlock, rewriter.getInsertionPoint());
  Block *thenBlock = rewriter.createBlock(
      parentBlock->getParent(), std::next(Region::iterator(parentBlock)));

  rewriter.setInsertionPointToEnd(parentBlock);
  LLVM::CondBrOp::create(rewriter, loc, ifCond, thenBlock, ValueRange{},
                         continueBlock, ValueRange{});

  rewriter.setInsertionPointToStart(thenBlock);
  LogicalResult result = thenFn();
  rewriter.setInsertionPointToEnd(thenBlock);
  LLVM::BrOp::create(rewriter, loc, ValueRange{}, continueBlock);
  rewriter.setInsertionPointToStart(continueBlock);
  return result;
}

/// Run \p emitFn, guarded by a branch on \p ifCond when it is present.
static LogicalResult emitGuardedByIfCond(Location loc, Value ifCond,
                                         ConversionPatternRewriter &rewriter,
                                         function_ref<LogicalResult()> emitFn) {
  if (ifCond)
    return createIfThen(loc, ifCond, rewriter, emitFn);
  return emitFn();
}

template <typename OpTy>
struct ACCExecutableDirectivePattern : public ConvertOpToLLVMPattern<OpTy> {
  ACCExecutableDirectivePattern(const LLVMTypeConverter &converter,
                                const ACCRuntimeCallConfig &config,
                                PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<OpTy>(converter, benefit), config(config) {}

  ACCRuntimeCallConfig config;
};

struct WaitOpLowering : public ACCExecutableDirectivePattern<WaitOp> {
  using ACCExecutableDirectivePattern<WaitOp>::ACCExecutableDirectivePattern;

  LogicalResult
  matchAndRewrite(WaitOp op, WaitOp::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type i32Ty = rewriter.getI32Type();
    Type i64Ty = rewriter.getI64Type();
    Type ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());

    auto emitWait = [&]() -> LogicalResult {
      Value asyncQueue = getAsyncQueue(op, rewriter, config);
      SmallVector<Value> waitValues;
      for (Value operand : op.getWaitOperands())
        waitValues.push_back(
            castToI64(loc, rewriter.getRemappedValue(operand), rewriter));

      unsigned size = waitValues.size();
      Value waitNum = LLVM::ConstantOp::create(rewriter, loc, i32Ty, size);
      Value waitList;
      if (size == 0) {
        waitList = LLVM::ZeroOp::create(rewriter, loc, ptrTy);
      } else {
        waitList = LLVM::AllocaOp::create(rewriter, loc, ptrTy, i64Ty, waitNum);
        for (auto [index, waitValue] : llvm::enumerate(waitValues)) {
          Value idx = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                               static_cast<int64_t>(index));
          Value elementPtr = LLVM::GEPOp::create(
              rewriter, loc, ptrTy, i64Ty, waitList, ArrayRef<Value>{idx});
          LLVM::StoreOp::create(rewriter, loc, waitValue, elementPtr);
        }
      }

      StringRef functionName = getParentFunctionName(waitValues);
      if (functionName.empty())
        functionName = getParentFunctionName(op);
      Value ident = createIdent(loc, functionName, rewriter, module, config);
      Value flags = LLVM::ConstantOp::create(rewriter, loc, i64Ty, 0);
      Value deviceType = LLVM::ConstantOp::create(
          rewriter, loc, i64Ty,
          config.getDeviceTypeRuntimeValue(DeviceType::None));
      Value deviceNum = LLVM::ConstantOp::create(rewriter, loc, i32Ty, 0);

      return createRuntimeCall(
          loc, rewriter, module, RuntimeFunction::ACCRTL_tgt_acc_wait, config,
          {ident, flags, deviceType, deviceNum, waitNum, waitList, asyncQueue});
    };

    if (failed(emitGuardedByIfCond(loc, op.getIfCond(), rewriter, emitWait)))
      return failure();

    rewriter.eraseOp(op);
    return success();
  }
};

/// Emit a call to a runtime entry point taking
/// `(ident, flags, deviceType, deviceNum)`. A null `deviceNum` selects the
/// current device.
static LogicalResult
emitDeviceOperationCall(Location loc, RuntimeFunction fn, DeviceType deviceType,
                        Value deviceNum, StringRef functionName,
                        ModuleOp module, ConversionPatternRewriter &rewriter,
                        const ACCRuntimeCallConfig &config) {
  Type i64Ty = rewriter.getI64Type();
  Value deviceTypeValue = LLVM::ConstantOp::create(
      rewriter, loc, i64Ty, config.getDeviceTypeRuntimeValue(deviceType));
  Value ident = createIdent(loc, functionName, rewriter, module, config);
  Value flags = LLVM::ConstantOp::create(rewriter, loc, i64Ty, 0);
  Value deviceNumValue =
      deviceNum ? castToI64(loc, deviceNum, rewriter)
                : LLVM::ConstantOp::create(rewriter, loc, i64Ty, -1);
  return createRuntimeCall(loc, rewriter, module, fn, config,
                           {ident, flags, deviceTypeValue, deviceNumValue});
}

static LogicalResult rewriteInitOrShutdown(Operation *op, Value deviceNum,
                                           ArrayAttr deviceTypesAttr,
                                           Value ifCond, bool isInit,
                                           ConversionPatternRewriter &rewriter,
                                           const ACCRuntimeCallConfig &config) {
  ModuleOp module = op->getParentOfType<ModuleOp>();
  Location loc = op->getLoc();

  auto emitCalls = [&]() -> LogicalResult {
    StringRef functionName = deviceNum ? getParentFunctionName(deviceNum)
                                       : getParentFunctionName(op);
    RuntimeFunction fn = isInit ? RuntimeFunction::ACCRTL_tgt_acc_init
                                : RuntimeFunction::ACCRTL_tgt_acc_shutdown;

    auto emitOne = [&](DeviceType deviceType) {
      return emitDeviceOperationCall(loc, fn, deviceType, deviceNum,
                                     functionName, module, rewriter, config);
    };

    if (!deviceTypesAttr)
      return emitOne(DeviceType::None);

    for (Attribute attr : deviceTypesAttr) {
      if (auto typeAttr = dyn_cast<DeviceTypeAttr>(attr))
        if (failed(emitOne(typeAttr.getValue())))
          return failure();
    }
    return success();
  };

  if (failed(emitGuardedByIfCond(loc, ifCond, rewriter, emitCalls)))
    return failure();

  rewriter.eraseOp(op);
  return success();
}

struct InitOpLowering : public ACCExecutableDirectivePattern<InitOp> {
  using ACCExecutableDirectivePattern<InitOp>::ACCExecutableDirectivePattern;

  LogicalResult
  matchAndRewrite(InitOp op, InitOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteInitOrShutdown(op, adaptor.getDeviceNum(),
                                 op.getDeviceTypesAttr(), op.getIfCond(),
                                 /*isInit=*/true, rewriter, config);
  }
};

struct ShutdownOpLowering : public ACCExecutableDirectivePattern<ShutdownOp> {
  using ACCExecutableDirectivePattern<
      ShutdownOp>::ACCExecutableDirectivePattern;

  LogicalResult
  matchAndRewrite(ShutdownOp op, ShutdownOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteInitOrShutdown(op, adaptor.getDeviceNum(),
                                 op.getDeviceTypesAttr(), op.getIfCond(),
                                 /*isInit=*/false, rewriter, config);
  }
};

struct SetOpLowering : public ACCExecutableDirectivePattern<SetOp> {
  using ACCExecutableDirectivePattern<SetOp>::ACCExecutableDirectivePattern;

  LogicalResult
  matchAndRewrite(SetOp op, SetOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = op.getLoc();
    Type i64Ty = rewriter.getI64Type();

    auto emitSet = [&]() -> LogicalResult {
      if (Value asyncValue = adaptor.getDefaultAsync()) {
        asyncValue = castToI64(loc, asyncValue, rewriter);
        Value ident = createIdent(loc, getParentFunctionName(asyncValue),
                                  rewriter, module, config);
        if (failed(createRuntimeCall(
                loc, rewriter, module,
                RuntimeFunction::ACCRTL_tgt_acc_set_default_async, config,
                {ident, asyncValue})))
          return failure();
      }

      if (op.getDeviceNum()) {
        Value deviceNum = adaptor.getDeviceNum();
        DeviceType deviceType = DeviceType::None;
        if (auto deviceTypeAttr = op.getDeviceTypeAttr())
          deviceType = deviceTypeAttr.getValue();
        return emitDeviceOperationCall(
            loc, RuntimeFunction::ACCRTL_tgt_acc_set_device_num, deviceType,
            deviceNum, getParentFunctionName(deviceNum), module, rewriter,
            config);
      }
      if (auto deviceTypeAttr = op.getDeviceTypeAttr()) {
        Value deviceTypeValue = LLVM::ConstantOp::create(
            rewriter, loc, i64Ty,
            config.getDeviceTypeRuntimeValue(deviceTypeAttr.getValue()));
        Value ident = createIdent(loc, StringRef(), rewriter, module, config);
        Value flags = LLVM::ConstantOp::create(rewriter, loc, i64Ty, 0);
        return createRuntimeCall(
            loc, rewriter, module,
            RuntimeFunction::ACCRTL_tgt_acc_set_device_type, config,
            {ident, flags, deviceTypeValue});
      }
      return success();
    };

    if (failed(emitGuardedByIfCond(loc, op.getIfCond(), rewriter, emitSet)))
      return failure();

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::configureACCExecutableDirectiveConversionLegality(
    ConversionTarget &target) {
  target.addIllegalOp<acc::InitOp, acc::ShutdownOp, acc::WaitOp, acc::SetOp>();
}

void mlir::populateACCExecutableDirectivePatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns,
    const acc::ACCRuntimeCallConfig &config) {
  patterns
      .add<WaitOpLowering, InitOpLowering, ShutdownOpLowering, SetOpLowering>(
          converter, config);
}
