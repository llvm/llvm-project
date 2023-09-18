//===- FuncTransformOps.cpp - Implementation of CF transform ops ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/TransformOps/FuncTransformOps.h"

#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Apply...ConversionPatternsOp
//===----------------------------------------------------------------------===//

void transform::ApplyFuncToLLVMConversionPatternsOp::populatePatterns(
    TypeConverter &typeConverter, RewritePatternSet &patterns) {
  populateFuncToLLVMConversionPatterns(
      static_cast<LLVMTypeConverter &>(typeConverter), patterns);
}

LogicalResult
transform::ApplyFuncToLLVMConversionPatternsOp::verifyTypeConverter(
    transform::TypeConverterBuilderOpInterface builder) {
  if (builder.getTypeConverterType() != "LLVMTypeConverter")
    return emitOpError("expected LLVMTypeConverter");
  return success();
}

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {
class FuncTransformDialectExtension
    : public transform::TransformDialectExtension<
          FuncTransformDialectExtension> {
public:
  using Base::Base;

  void init() {
    declareGeneratedDialect<LLVM::LLVMDialect>();

    registerTransformOps<
#define GET_OP_LIST
#include "mlir/Dialect/Func/TransformOps/FuncTransformOps.cpp.inc"
        >();
  }
};
} // namespace

#define GET_OP_CLASSES
#include "mlir/Dialect/Func/TransformOps/FuncTransformOps.cpp.inc"

void mlir::func::registerTransformDialectExtension(DialectRegistry &registry) {
  registry.addExtensions<FuncTransformDialectExtension>();
}
