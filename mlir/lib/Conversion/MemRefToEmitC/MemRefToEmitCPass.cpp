//===- MemRefToEmitC.cpp - MemRef to EmitC conversion ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert memref ops into emitc ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/MemRefToEmitC/MemRefToEmitCPass.h"

#include "mlir/Conversion/EmitCCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToEmitC/MemRefToEmitC.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/EmitC/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRefDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTMEMREFTOEMITC
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct ConvertMemRefToEmitCPass
    : public impl::ConvertMemRefToEmitCBase<ConvertMemRefToEmitCPass> {
  using Base::Base;
  void runOnOperation() override {
    EmitCTypeConverter converter(&getContext());
    ConvertMemRefToEmitCOptions options;
    options.lowerToCpp = this->lowerToCpp;

    RewritePatternSet patterns(&getContext());
    populateMemRefToEmitCConversionPatterns(patterns, converter);

    ConversionTarget target(getContext());
    target.addIllegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<emitc::EmitCDialect>();

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();

    mlir::ModuleOp module = getOperation();
    llvm::SmallSetVector<StringRef, 2> requiredHeaders;
    module.walk([&](mlir::emitc::CallOpaqueOp callOp) {
      StringRef expectedHeader;
      if (callOp.getCallee() == alignedAllocFunctionName ||
          callOp.getCallee() == freeFunctionName ||
          callOp.getCallee() == mallocFunctionName)
        expectedHeader = options.lowerToCpp ? cppStandardLibraryHeader
                                            : cStandardLibraryHeader;
      else if (callOp.getCallee() == memcpyFunctionName)
        expectedHeader =
            options.lowerToCpp ? cppStringLibraryHeader : cStringLibraryHeader;
      else
        return;
      requiredHeaders.insert(expectedHeader);
    });
    emitc::addStandardIncludes(getOperation(), requiredHeaders.getArrayRef());
  }
};
} // namespace
