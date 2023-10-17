//===- Invoke.cpp ------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/MemRefUtils.h"
#include "mlir/ExecutionEngine/RunnerUtils.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include "gmock/gmock.h"

// SPARC currently lacks JIT support.
#ifdef __sparc__
#define SKIP_WITHOUT_JIT(x) DISABLED_##x
#else
#define SKIP_WITHOUT_JIT(x) x
#endif

using namespace mlir;

// The JIT isn't supported on Windows at that time
#ifndef _WIN32

static struct LLVMInitializer {
  LLVMInitializer() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
  }
} initializer;

/// Simple conversion pipeline for the purpose of testing sources written in
/// dialects lowering to LLVM Dialect.
static LogicalResult lowerToLLVMDialect(ModuleOp module) {
  PassManager pm(module->getName());
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  pm.addNestedPass<func::FuncOp>(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  return pm.run(module);
}

TEST(MLIRExecutionEngine, SKIP_WITHOUT_JIT(AddInteger)) {
  std::string moduleStr = R"mlir(
  func.func @foo(%arg0 : i32) -> i32 attributes { llvm.emit_c_interface } {
    %res = arith.addi %arg0, %arg0 : i32
    return %res : i32
  }
  )mlir";
  DialectRegistry registry;
  registerAllDialects(registry);
  registerBuiltinDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);
  MLIRContext context(registry);
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(moduleStr, &context);
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

TEST(MLIRExecutionEngine, SKIP_WITHOUT_JIT(SubtractFloat)) {
  std::string moduleStr = R"mlir(
  func.func @foo(%arg0 : f32, %arg1 : f32) -> f32 attributes { llvm.emit_c_interface } {
    %res = arith.subf %arg0, %arg1 : f32
    return %res : f32
  }
  )mlir";
  DialectRegistry registry;
  registerAllDialects(registry);
  registerBuiltinDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);
  MLIRContext context(registry);
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(moduleStr, &context);
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

TEST(NativeMemRefJit, SKIP_WITHOUT_JIT(ZeroRankMemref)) {
  OwningMemRef<float, 0> a({});
  a[{}] = 42.;
  ASSERT_EQ(*a->data, 42);
  a[{}] = 0;
  std::string moduleStr = R"mlir(
  func.func @zero_ranked(%arg0 : memref<f32>) attributes { llvm.emit_c_interface } {
    %cst42 = arith.constant 42.0 : f32
    memref.store %cst42, %arg0[] : memref<f32>
    return
  }
  )mlir";
  DialectRegistry registry;
  registerAllDialects(registry);
  registerBuiltinDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);
  MLIRContext context(registry);
  auto module = parseSourceString<ModuleOp>(moduleStr, &context);
  ASSERT_TRUE(!!module);
  ASSERT_TRUE(succeeded(lowerToLLVMDialect(*module)));
  auto jitOrError = ExecutionEngine::create(*module);
  ASSERT_TRUE(!!jitOrError);
  auto jit = std::move(jitOrError.get());

  llvm::Error error = jit->invoke("zero_ranked", &*a);
  ASSERT_TRUE(!error);
  EXPECT_EQ((a[{}]), 42.);
  for (float &elt : *a)
    EXPECT_EQ(&elt, &(a[{}]));
}

TEST(NativeMemRefJit, SKIP_WITHOUT_JIT(RankOneMemref)) {
  int64_t shape[] = {9};
  OwningMemRef<float, 1> a(shape);
  int count = 1;
  for (float &elt : *a) {
    EXPECT_EQ(&elt, &(a[{count - 1}]));
    elt = count++;
  }

  std::string moduleStr = R"mlir(
  func.func @one_ranked(%arg0 : memref<?xf32>) attributes { llvm.emit_c_interface } {
    %cst42 = arith.constant 42.0 : f32
    %cst5 = arith.constant 5 : index
    memref.store %cst42, %arg0[%cst5] : memref<?xf32>
    return
  }
  )mlir";
  DialectRegistry registry;
  registerAllDialects(registry);
  registerBuiltinDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);
  MLIRContext context(registry);
  auto module = parseSourceString<ModuleOp>(moduleStr, &context);
  ASSERT_TRUE(!!module);
  ASSERT_TRUE(succeeded(lowerToLLVMDialect(*module)));
  auto jitOrError = ExecutionEngine::create(*module);
  ASSERT_TRUE(!!jitOrError);
  auto jit = std::move(jitOrError.get());

  llvm::Error error = jit->invoke("one_ranked", &*a);
  ASSERT_TRUE(!error);
  count = 1;
  for (float &elt : *a) {
    if (count == 6)
      EXPECT_EQ(elt, 42.);
    else
      EXPECT_EQ(elt, count);
    count++;
  }
}

TEST(NativeMemRefJit, SKIP_WITHOUT_JIT(BasicMemref)) {
  constexpr int k = 3;
  constexpr int m = 7;
  // Prepare arguments beforehand.
  auto init = [=](float &elt, ArrayRef<int64_t> indices) {
    assert(indices.size() == 2);
    elt = m * indices[0] + indices[1];
  };
  int64_t shape[] = {k, m};
  int64_t shapeAlloc[] = {k + 1, m + 1};
  OwningMemRef<float, 2> a(shape, shapeAlloc, init);
  ASSERT_EQ(a->sizes[0], k);
  ASSERT_EQ(a->sizes[1], m);
  ASSERT_EQ(a->strides[0], m + 1);
  ASSERT_EQ(a->strides[1], 1);
  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < m; ++j) {
      EXPECT_EQ((a[{i, j}]), i * m + j);
      EXPECT_EQ(&(a[{i, j}]), &((*a)[i][j]));
    }
  }
  std::string moduleStr = R"mlir(
  func.func @rank2_memref(%arg0 : memref<?x?xf32>, %arg1 : memref<?x?xf32>) attributes { llvm.emit_c_interface } {
    %x = arith.constant 2 : index
    %y = arith.constant 1 : index
    %cst42 = arith.constant 42.0 : f32
    memref.store %cst42, %arg0[%y, %x] : memref<?x?xf32>
    memref.store %cst42, %arg1[%x, %y] : memref<?x?xf32>
    return
  }
  )mlir";
  DialectRegistry registry;
  registerAllDialects(registry);
  registerBuiltinDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);
  MLIRContext context(registry);
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(moduleStr, &context);
  ASSERT_TRUE(!!module);
  ASSERT_TRUE(succeeded(lowerToLLVMDialect(*module)));
  auto jitOrError = ExecutionEngine::create(*module);
  ASSERT_TRUE(!!jitOrError);
  std::unique_ptr<ExecutionEngine> jit = std::move(jitOrError.get());

  llvm::Error error = jit->invoke("rank2_memref", &*a, &*a);
  ASSERT_TRUE(!error);
  EXPECT_EQ(((*a)[1][2]), 42.);
  EXPECT_EQ((a[{2, 1}]), 42.);
}

// A helper function that will be called from the JIT
static void memrefMultiply(::StridedMemRefType<float, 2> *memref,
                           int32_t coefficient) {
  for (float &elt : *memref)
    elt *= coefficient;
}

// MSAN does not work with JIT.
#if __has_feature(memory_sanitizer)
#define MAYBE_JITCallback DISABLED_JITCallback
#else
#define MAYBE_JITCallback SKIP_WITHOUT_JIT(JITCallback)
#endif
TEST(NativeMemRefJit, MAYBE_JITCallback) {
  constexpr int k = 2;
  constexpr int m = 2;
  int64_t shape[] = {k, m};
  int64_t shapeAlloc[] = {k + 1, m + 1};
  OwningMemRef<float, 2> a(shape, shapeAlloc);
  int count = 1;
  for (float &elt : *a)
    elt = count++;

  std::string moduleStr = R"mlir(
  func.func private @callback(%arg0: memref<?x?xf32>, %coefficient: i32)  attributes { llvm.emit_c_interface }
  func.func @caller_for_callback(%arg0: memref<?x?xf32>, %coefficient: i32) attributes { llvm.emit_c_interface } {
    %unranked = memref.cast %arg0: memref<?x?xf32> to memref<*xf32>
    call @callback(%arg0, %coefficient) : (memref<?x?xf32>, i32) -> ()
    return
  }
  )mlir";
  DialectRegistry registry;
  registerAllDialects(registry);
  registerBuiltinDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);
  MLIRContext context(registry);
  auto module = parseSourceString<ModuleOp>(moduleStr, &context);
  ASSERT_TRUE(!!module);
  ASSERT_TRUE(succeeded(lowerToLLVMDialect(*module)));
  auto jitOrError = ExecutionEngine::create(*module);
  ASSERT_TRUE(!!jitOrError);
  auto jit = std::move(jitOrError.get());
  // Define any extra symbols so they're available at runtime.
  jit->registerSymbols([&](llvm::orc::MangleAndInterner interner) {
    llvm::orc::SymbolMap symbolMap;
    symbolMap[interner("_mlir_ciface_callback")] = {
        llvm::orc::ExecutorAddr::fromPtr(memrefMultiply),
        llvm::JITSymbolFlags::Exported};
    return symbolMap;
  });

  int32_t coefficient = 3.;
  llvm::Error error = jit->invoke("caller_for_callback", &*a, coefficient);
  ASSERT_TRUE(!error);
  count = 1;
  for (float elt : *a)
    ASSERT_EQ(elt, coefficient * count++);
}

#endif // _WIN32
