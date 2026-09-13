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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"

#include <cstdint>

using namespace mlir;
using namespace mlir::acc;

namespace {
template <typename OpTy>
struct ACCExecutableDirectivePattern : public ConvertOpToLLVMPattern<OpTy> {
  ACCExecutableDirectivePattern(const LLVMTypeConverter &converter,
                                Region &globalSymbolRegion,
                                SymbolTable &symbolTable,
                                const ACCRuntimeCallConfig &config,
                                PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<OpTy>(converter, benefit),
        globalSymbolRegion(globalSymbolRegion), symbolTable(symbolTable),
        config(config) {}

  Region &globalSymbolRegion;
  SymbolTable &symbolTable;
  ACCRuntimeCallConfig config;
};

struct WaitOpLowering : public ACCExecutableDirectivePattern<WaitOp> {
  using ACCExecutableDirectivePattern<WaitOp>::ACCExecutableDirectivePattern;

  LogicalResult
  matchAndRewrite(WaitOp op, WaitOp::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    auto emitWait = [&]() -> LogicalResult {
      Value asyncOperand = op.getAsyncOperand()
                               ? rewriter.getRemappedValue(op.getAsyncOperand())
                               : Value();
      Value asyncQueue =
          getAsyncQueue(loc, asyncOperand, op.getAsync(), rewriter, config);
      SmallVector<Value> waitValues;
      for (Value operand : op.getWaitOperands())
        waitValues.push_back(rewriter.getRemappedValue(operand));
      return emitWaitCall(loc, waitValues, asyncQueue, rewriter,
                          globalSymbolRegion, symbolTable, config);
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
                        Region &globalSymbolRegion, SymbolTable &symbolTable,
                        ConversionPatternRewriter &rewriter,
                        const ACCRuntimeCallConfig &config) {
  Type i64Ty = rewriter.getI64Type();
  Value deviceTypeValue = LLVM::ConstantOp::create(
      rewriter, loc, i64Ty, config.getDeviceTypeRuntimeValue(deviceType));
  Value ident = createIdent(loc, functionName, rewriter, globalSymbolRegion,
                            config, &symbolTable);
  Value flags = LLVM::ConstantOp::create(rewriter, loc, i64Ty, 0);
  Value deviceNumValue =
      deviceNum ? castToI64(loc, deviceNum, rewriter)
                : LLVM::ConstantOp::create(rewriter, loc, i64Ty, -1);
  return createRuntimeCall(loc, rewriter, globalSymbolRegion, symbolTable, fn,
                           config,
                           {ident, flags, deviceTypeValue, deviceNumValue});
}

static LogicalResult rewriteInitOrShutdown(Operation *op, Value deviceNum,
                                           ArrayAttr deviceTypesAttr,
                                           Value ifCond, bool isInit,
                                           ConversionPatternRewriter &rewriter,
                                           Region &globalSymbolRegion,
                                           SymbolTable &symbolTable,
                                           const ACCRuntimeCallConfig &config) {
  Location loc = op->getLoc();

  auto emitCalls = [&]() -> LogicalResult {
    StringRef functionName = deviceNum ? getParentFunctionName(deviceNum)
                                       : getParentFunctionName(op);
    RuntimeFunction fn = isInit ? RuntimeFunction::ACCRTL_tgt_acc_init
                                : RuntimeFunction::ACCRTL_tgt_acc_shutdown;

    auto emitOne = [&](DeviceType deviceType) {
      return emitDeviceOperationCall(loc, fn, deviceType, deviceNum,
                                     functionName, globalSymbolRegion,
                                     symbolTable, rewriter, config);
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
    return rewriteInitOrShutdown(
        op, adaptor.getDeviceNum(), op.getDeviceTypesAttr(), op.getIfCond(),
        /*isInit=*/true, rewriter, globalSymbolRegion, symbolTable, config);
  }
};

struct ShutdownOpLowering : public ACCExecutableDirectivePattern<ShutdownOp> {
  using ACCExecutableDirectivePattern<
      ShutdownOp>::ACCExecutableDirectivePattern;

  LogicalResult
  matchAndRewrite(ShutdownOp op, ShutdownOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteInitOrShutdown(
        op, adaptor.getDeviceNum(), op.getDeviceTypesAttr(), op.getIfCond(),
        /*isInit=*/false, rewriter, globalSymbolRegion, symbolTable, config);
  }
};

struct SetOpLowering : public ACCExecutableDirectivePattern<SetOp> {
  using ACCExecutableDirectivePattern<SetOp>::ACCExecutableDirectivePattern;

  LogicalResult
  matchAndRewrite(SetOp op, SetOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type i64Ty = rewriter.getI64Type();

    auto emitSet = [&]() -> LogicalResult {
      if (Value asyncValue = adaptor.getDefaultAsync()) {
        asyncValue = castToI64(loc, asyncValue, rewriter);
        Value ident =
            createIdent(loc, getParentFunctionName(asyncValue), rewriter,
                        globalSymbolRegion, config, &symbolTable);
        if (failed(createRuntimeCall(
                loc, rewriter, globalSymbolRegion, symbolTable,
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
            deviceNum, getParentFunctionName(deviceNum), globalSymbolRegion,
            symbolTable, rewriter, config);
      }
      if (auto deviceTypeAttr = op.getDeviceTypeAttr()) {
        Value deviceTypeValue = LLVM::ConstantOp::create(
            rewriter, loc, i64Ty,
            config.getDeviceTypeRuntimeValue(deviceTypeAttr.getValue()));
        Value ident = createIdent(loc, StringRef(), rewriter,
                                  globalSymbolRegion, config, &symbolTable);
        Value flags = LLVM::ConstantOp::create(rewriter, loc, i64Ty, 0);
        return createRuntimeCall(
            loc, rewriter, globalSymbolRegion, symbolTable,
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
    Region &globalSymbolRegion, SymbolTable &symbolTable,
    const acc::ACCRuntimeCallConfig &config) {
  patterns
      .add<WaitOpLowering, InitOpLowering, ShutdownOpLowering, SetOpLowering>(
          converter, globalSymbolRegion, symbolTable, config);
}
