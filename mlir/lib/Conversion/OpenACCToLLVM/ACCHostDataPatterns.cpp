//===- ACCHostDataPatterns.cpp - ACC host_data to LLVM ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lowering of the OpenACC host_data construct to the OpenACC runtime. Each
// use_device clause of the construct becomes the call that asks the runtime
// for the device address its object is mapped to, so that the body of the
// construct works on device addresses. An if clause on the construct decides
// between that address and the host one, and the body is left in place of the
// construct.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/OpenACCToLLVM/ACCToLLVM.h"
#include "mlir/Conversion/OpenACCToLLVM/ACCToLLVMUtils.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::acc;

/// Returns the host_data construct \p op gives a device address to, or null
/// when the clause belongs to no such construct.
static HostDataOp getHostDataConstruct(UseDeviceOp op) {
  for (Operation *user : op->getUsers())
    if (auto hostData = dyn_cast<HostDataOp>(user))
      return hostData;
  return nullptr;
}

namespace {
/// A use_device clause stands for the device address of the object it names,
/// so it becomes the call that asks the runtime for that address. The call is
/// emitted before the construct, so that every use of the clause in its body
/// is reached by it.
struct UseDeviceOpLowering : public ConvertOpToLLVMPattern<UseDeviceOp> {
  UseDeviceOpLowering(const LLVMTypeConverter &converter,
                      Region &globalSymbolRegion, SymbolTable &symbolTable,
                      const ACCRuntimeCallConfig &config,
                      PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<UseDeviceOp>(converter, benefit),
        globalSymbolRegion(globalSymbolRegion), symbolTable(symbolTable),
        config(config) {}

  LogicalResult
  matchAndRewrite(UseDeviceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    HostDataOp hostData = getHostDataConstruct(op);
    if (!hostData)
      return rewriter.notifyMatchFailure(
          op, "use_device clause of no host_data construct");

    Location loc = op.getLoc();
    Value hostPtr = adaptor.getVar();
    auto emitCall = [&]() -> FailureOr<Value> {
      return acc::emitGetDevicePtrCall(op, hostPtr, hostData.getIfPresent(),
                                       rewriter, globalSymbolRegion,
                                       symbolTable, config);
    };
    // An if clause that does not hold leaves the body on host addresses.
    auto keepHostPtr = [&]() -> FailureOr<Value> { return hostPtr; };

    Value ifCond = hostData.getIfCond()
                       ? rewriter.getRemappedValue(hostData.getIfCond())
                       : Value();
    if (ifCond)
      rewriter.setInsertionPoint(hostData);
    FailureOr<Value> devicePtr = acc::emitValueSelectedByIfCond(
        loc, ifCond, rewriter, emitCall, keepHostPtr);
    if (failed(devicePtr))
      return failure();

    rewriter.replaceOp(op, *devicePtr);
    return success();
  }

  Region &globalSymbolRegion;
  SymbolTable &symbolTable;
  ACCRuntimeCallConfig config;
};

/// The construct itself has nothing left to do once its clauses hold device
/// addresses, so its body takes its place.
struct HostDataOpLowering : public ConvertOpToLLVMPattern<HostDataOp> {
  using ConvertOpToLLVMPattern<HostDataOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(HostDataOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    acc::spliceConstructRegion(op, op.getRegion(), rewriter);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::configureACCHostDataConversionLegality(ConversionTarget &target) {
  target.addIllegalOp<acc::HostDataOp, acc::UseDeviceOp>();
}

void mlir::populateACCHostDataPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns,
    Region &globalSymbolRegion, SymbolTable &symbolTable,
    const acc::ACCRuntimeCallConfig &config) {
  patterns.add<UseDeviceOpLowering>(converter, globalSymbolRegion, symbolTable,
                                    config);
  patterns.add<HostDataOpLowering>(converter);
}
