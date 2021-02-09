//===- Invoke.cpp ------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/RunnerUtils.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include "gmock/gmock.h"

using namespace mlir;

static struct LLVMInitializer {
  LLVMInitializer() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
  }
} initializer;

/// Simple conversion pipeline for the purpose of testing sources written in
/// dialects lowering to LLVM Dialect.
static LogicalResult lowerToLLVMDialect(ModuleOp module) {
  PassManager pm(module.getContext());
  pm.addPass(mlir::createLowerToLLVMPass());
  return pm.run(module);
}

// The JIT isn't supported on Windows at that time
#ifndef _WIN32

TEST(MLIRExecutionEngine, AddInteger) {
  std::string moduleStr = R"mlir(
  func @foo(%arg0 : i32) -> i32 attributes { llvm.emit_c_interface } {
    %res = std.addi %arg0, %arg0 : i32
    return %res : i32
  }
  )mlir";
  MLIRContext context;
  registerAllDialects(context.getDialectRegistry());
  OwningModuleRef module = parseSourceString(moduleStr, &context);
  ASSERT_TRUE(!!module);
  ASSERT_TRUE(succeeded(lowerToLLVMDialect(*module)));
  auto jitOrError = ExecutionEngine::create(*module);
  ASSERT_TRUE(!!jitOrError);
  std::unique_ptr<ExecutionEngine> jit = std::move(jitOrError.get());
  // The result of the function must be passed as output argument.
  int result = 0;
  llvm::Error error =
      jit->invoke("foo", 42, ExecutionEngine::Result<int>(result));
  ASSERT_TRUE(!error);
  ASSERT_EQ(result, 42 + 42);
}

TEST(MLIRExecutionEngine, SubtractFloat) {
  std::string moduleStr = R"mlir(
  func @foo(%arg0 : f32, %arg1 : f32) -> f32 attributes { llvm.emit_c_interface } {
    %res = std.subf %arg0, %arg1 : f32
    return %res : f32
  }
  )mlir";
  MLIRContext context;
  registerAllDialects(context.getDialectRegistry());
  OwningModuleRef module = parseSourceString(moduleStr, &context);
  ASSERT_TRUE(!!module);
  ASSERT_TRUE(succeeded(lowerToLLVMDialect(*module)));
  auto jitOrError = ExecutionEngine::create(*module);
  ASSERT_TRUE(!!jitOrError);
  std::unique_ptr<ExecutionEngine> jit = std::move(jitOrError.get());
  // The result of the function must be passed as output argument.
  float result = -1;
  llvm::Error error =
      jit->invoke("foo", 43.0f, 1.0f, ExecutionEngine::result(result));
  ASSERT_TRUE(!error);
  ASSERT_EQ(result, 42.f);
}

#endif // _WIN32
