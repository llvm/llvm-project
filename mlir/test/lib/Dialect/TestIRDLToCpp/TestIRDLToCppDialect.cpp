//===- TestDialect.cpp - MLIR Test Dialect Types ------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file includes TestIRDLToCpp dialect.
//
//===----------------------------------------------------------------------===//

// #include "mlir/IR/Dialect.h"
#include "mlir/IR/Region.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/TypeSwitch.h"

#include "TestIRDLToCppDialect.h"

#define GEN_DIALECT_DEF
#define GET_TYPEDEF_CLASSES
#define GET_OP_CLASSES
#include "test_irdl_to_cpp.irdl.mlir.cpp.inc"

namespace test {
using namespace mlir;
struct TestOpConversion : public OpConversionPattern<test_irdl_to_cpp::BeefOp> {
  using OpConversionPattern<test_irdl_to_cpp::BeefOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::test_irdl_to_cpp::BeefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return success();
  }
};

void registerIrdlTestDialect(mlir::DialectRegistry &registry) {
  registry.insert<mlir::test_irdl_to_cpp::Test_irdl_to_cppDialect>();
}
} // namespace test
