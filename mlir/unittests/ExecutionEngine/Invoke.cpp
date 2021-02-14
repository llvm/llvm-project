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
#include "mlir/ExecutionEngine/MemRefUtils.h"
#include "mlir/ExecutionEngine/RunnerUtils.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
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
  DialectRegistry registry;
  registerAllDialects(registry);
  registerLLVMDialectTranslation(registry);
  MLIRContext context(registry);
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
  DialectRegistry registry;
  registerAllDialects(registry);
  registerLLVMDialectTranslation(registry);
  MLIRContext context(registry);
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

TEST(NativeMemRefJit, ZeroRankMemref) {
  OwningMemRef<float, 0> A({});
  A[{}] = 42.;
  ASSERT_EQ(*A->data, 42);
  A[{}] = 0;
  std::string moduleStr = R"mlir(
  func @zero_ranked(%arg0 : memref<f32>) attributes { llvm.emit_c_interface } {
    %cst42 = constant 42.0 : f32
    store %cst42, %arg0[] : memref<f32>
    return
  }
  )mlir";
  DialectRegistry registry;
  registerAllDialects(registry);
  registerLLVMDialectTranslation(registry);
  MLIRContext context(registry);
  auto module = parseSourceString(moduleStr, &context);
  ASSERT_TRUE(!!module);
  ASSERT_TRUE(succeeded(lowerToLLVMDialect(*module)));
  auto jitOrError = ExecutionEngine::create(*module);
  ASSERT_TRUE(!!jitOrError);
  auto jit = std::move(jitOrError.get());

  llvm::Error error = jit->invoke("zero_ranked", &*A);
  ASSERT_TRUE(!error);
  EXPECT_EQ((A[{}]), 42.);
  for (float &elt : *A)
    EXPECT_EQ(&elt, &(A[{}]));
}

TEST(NativeMemRefJit, RankOneMemref) {
  int64_t shape[] = {9};
  OwningMemRef<float, 1> A(shape);
  int count = 1;
  for (float &elt : *A) {
    EXPECT_EQ(&elt, &(A[{count - 1}]));
    elt = count++;
  }

  std::string moduleStr = R"mlir(
  func @one_ranked(%arg0 : memref<?xf32>) attributes { llvm.emit_c_interface } {
    %cst42 = constant 42.0 : f32
    %cst5 = constant 5 : index
    store %cst42, %arg0[%cst5] : memref<?xf32>
    return
  }
  )mlir";
  DialectRegistry registry;
  registerAllDialects(registry);
  registerLLVMDialectTranslation(registry);
  MLIRContext context(registry);
  auto module = parseSourceString(moduleStr, &context);
  ASSERT_TRUE(!!module);
  ASSERT_TRUE(succeeded(lowerToLLVMDialect(*module)));
  auto jitOrError = ExecutionEngine::create(*module);
  ASSERT_TRUE(!!jitOrError);
  auto jit = std::move(jitOrError.get());

  llvm::Error error = jit->invoke("one_ranked", &*A);
  ASSERT_TRUE(!error);
  count = 1;
  for (float &elt : *A) {
    if (count == 6)
      EXPECT_EQ(elt, 42.);
    else
      EXPECT_EQ(elt, count);
    count++;
  }
}

TEST(NativeMemRefJit, BasicMemref) {
  constexpr int K = 3;
  constexpr int M = 7;
  // Prepare arguments beforehand.
  auto init = [=](float &elt, ArrayRef<int64_t> indices) {
    assert(indices.size() == 2);
    elt = M * indices[0] + indices[1];
  };
  int64_t shape[] = {K, M};
  int64_t shapeAlloc[] = {K + 1, M + 1};
  OwningMemRef<float, 2> A(shape, shapeAlloc, init);
  ASSERT_EQ(A->sizes[0], K);
  ASSERT_EQ(A->sizes[1], M);
  ASSERT_EQ(A->strides[0], M + 1);
  ASSERT_EQ(A->strides[1], 1);
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < M; ++j) {
      EXPECT_EQ((A[{i, j}]), i * M + j);
      EXPECT_EQ(&(A[{i, j}]), &((*A)[i][j]));
    }
  }
  std::string moduleStr = R"mlir(
  func @rank2_memref(%arg0 : memref<?x?xf32>, %arg1 : memref<?x?xf32>) attributes { llvm.emit_c_interface } {
    %x = constant 2 : index
    %y = constant 1 : index
    %cst42 = constant 42.0 : f32
    store %cst42, %arg0[%y, %x] : memref<?x?xf32>
    store %cst42, %arg1[%x, %y] : memref<?x?xf32>
    return
  }
  )mlir";
  DialectRegistry registry;
  registerAllDialects(registry);
  registerLLVMDialectTranslation(registry);
  MLIRContext context(registry);
  OwningModuleRef module = parseSourceString(moduleStr, &context);
  ASSERT_TRUE(!!module);
  ASSERT_TRUE(succeeded(lowerToLLVMDialect(*module)));
  auto jitOrError = ExecutionEngine::create(*module);
  ASSERT_TRUE(!!jitOrError);
  std::unique_ptr<ExecutionEngine> jit = std::move(jitOrError.get());

  llvm::Error error = jit->invoke("rank2_memref", &*A, &*A);
  ASSERT_TRUE(!error);
  EXPECT_EQ(((*A)[1][2]), 42.);
  EXPECT_EQ((A[{2, 1}]), 42.);
}

// A helper function that will be called from the JIT
static void memref_multiply(::StridedMemRefType<float, 2> *memref,
                            int32_t coefficient) {
  for (float &elt : *memref)
    elt *= coefficient;
}

TEST(NativeMemRefJit, JITCallback) {
  constexpr int K = 2;
  constexpr int M = 2;
  int64_t shape[] = {K, M};
  int64_t shapeAlloc[] = {K + 1, M + 1};
  OwningMemRef<float, 2> A(shape, shapeAlloc);
  int count = 1;
  for (float &elt : *A)
    elt = count++;

  std::string moduleStr = R"mlir(
  func private @callback(%arg0: memref<?x?xf32>, %coefficient: i32)  attributes { llvm.emit_c_interface } 
  func @caller_for_callback(%arg0: memref<?x?xf32>, %coefficient: i32) attributes { llvm.emit_c_interface } {
    %unranked = memref_cast %arg0: memref<?x?xf32> to memref<*xf32>
    call @callback(%arg0, %coefficient) : (memref<?x?xf32>, i32) -> ()
    return
  }
  )mlir";
  DialectRegistry registry;
  registerAllDialects(registry);
  registerLLVMDialectTranslation(registry);
  MLIRContext context(registry);
  auto module = parseSourceString(moduleStr, &context);
  ASSERT_TRUE(!!module);
  ASSERT_TRUE(succeeded(lowerToLLVMDialect(*module)));
  auto jitOrError = ExecutionEngine::create(*module);
  ASSERT_TRUE(!!jitOrError);
  auto jit = std::move(jitOrError.get());
  // Define any extra symbols so they're available at runtime.
  jit->registerSymbols([&](llvm::orc::MangleAndInterner interner) {
    llvm::orc::SymbolMap symbolMap;
    symbolMap[interner("_mlir_ciface_callback")] =
        llvm::JITEvaluatedSymbol::fromPointer(memref_multiply);
    return symbolMap;
  });

  int32_t coefficient = 3.;
  llvm::Error error = jit->invoke("caller_for_callback", &*A, coefficient);
  ASSERT_TRUE(!error);
  count = 1;
  for (float elt : *A)
    ASSERT_EQ(elt, coefficient * count++);
}

#endif // _WIN32
