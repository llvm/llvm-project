//===- SerializeNVVMTarget.cpp ----------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Config/mlir-config.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Target/LLVM/NVVM/Target.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"

#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"

#include "gmock/gmock.h"

using namespace mlir;

// Skip the test if the NVPTX target was not built.
#if LLVM_HAS_NVPTX_TARGET
#define SKIP_WITHOUT_NVPTX(x) x
#else
#define SKIP_WITHOUT_NVPTX(x) DISABLED_##x
#endif

class MLIRTargetLLVMNVVM : public ::testing::Test {
protected:
  void SetUp() override {
    registerBuiltinDialectTranslation(registry);
    registerLLVMDialectTranslation(registry);
    registerGPUDialectTranslation(registry);
    registerNVVMDialectTranslation(registry);
    NVVM::registerNVVMTargetInterfaceExternalModels(registry);
  }

  // Checks if PTXAS is in PATH.
  bool hasPtxas() {
    // Find the `ptxas` compiler.
    std::optional<std::string> ptxasCompiler =
        llvm::sys::Process::FindInEnvPath("PATH", "ptxas");
    return ptxasCompiler.has_value();
  }

  // Dialect registry.
  DialectRegistry registry;

  // MLIR module used for the tests.
  const std::string moduleStr = R"mlir(
      gpu.module @nvvm_test {
        llvm.func @nvvm_kernel(%arg0: f32) attributes {gpu.kernel, nvvm.kernel} {
        llvm.return
      }
    })mlir";
};

// Test NVVM serialization to LLVM.
TEST_F(MLIRTargetLLVMNVVM, SKIP_WITHOUT_NVPTX(SerializeNVVMMToLLVM)) {
  MLIRContext context(registry);

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(moduleStr, &context);
  ASSERT_TRUE(!!module);

  // Create an NVVM target.
  NVVM::NVVMTargetAttr target = NVVM::NVVMTargetAttr::get(&context);

  // Serialize the module.
  auto serializer = dyn_cast<gpu::TargetAttrInterface>(target);
  ASSERT_TRUE(!!serializer);
  gpu::TargetOptions options("", {}, "", gpu::CompilationTarget::Offload);
  for (auto gpuModule : (*module).getBody()->getOps<gpu::GPUModuleOp>()) {
    std::optional<SmallVector<char, 0>> object =
        serializer.serializeToObject(gpuModule, options);
    // Check that the serializer was successful.
    ASSERT_TRUE(object != std::nullopt);
    ASSERT_TRUE(!object->empty());

    // Read the serialized module.
    llvm::MemoryBufferRef buffer(StringRef(object->data(), object->size()),
                                 "module");
    llvm::LLVMContext llvmContext;
    llvm::Expected<std::unique_ptr<llvm::Module>> llvmModule =
        llvm::getLazyBitcodeModule(buffer, llvmContext);
    ASSERT_TRUE(!!llvmModule);
    ASSERT_TRUE(!!*llvmModule);

    // Check that it has a function named `foo`.
    ASSERT_TRUE((*llvmModule)->getFunction("nvvm_kernel") != nullptr);
  }
}

// Test NVVM serialization to PTX.
TEST_F(MLIRTargetLLVMNVVM, SKIP_WITHOUT_NVPTX(SerializeNVVMToPTX)) {
  MLIRContext context(registry);

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(moduleStr, &context);
  ASSERT_TRUE(!!module);

  // Create an NVVM target.
  NVVM::NVVMTargetAttr target = NVVM::NVVMTargetAttr::get(&context);

  // Serialize the module.
  auto serializer = dyn_cast<gpu::TargetAttrInterface>(target);
  ASSERT_TRUE(!!serializer);
  gpu::TargetOptions options("", {}, "", gpu::CompilationTarget::Assembly);
  for (auto gpuModule : (*module).getBody()->getOps<gpu::GPUModuleOp>()) {
    std::optional<SmallVector<char, 0>> object =
        serializer.serializeToObject(gpuModule, options);
    // Check that the serializer was successful.
    ASSERT_TRUE(object != std::nullopt);
    ASSERT_TRUE(!object->empty());

    ASSERT_TRUE(
        StringRef(object->data(), object->size()).contains("nvvm_kernel"));
  }
}

// Test NVVM serialization to Binary.
TEST_F(MLIRTargetLLVMNVVM, SKIP_WITHOUT_NVPTX(SerializeNVVMToBinary)) {
  if (!hasPtxas())
    GTEST_SKIP() << "PTXAS compiler not found, skipping test.";

  MLIRContext context(registry);

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(moduleStr, &context);
  ASSERT_TRUE(!!module);

  // Create an NVVM target.
  NVVM::NVVMTargetAttr target = NVVM::NVVMTargetAttr::get(&context);

  // Serialize the module.
  auto serializer = dyn_cast<gpu::TargetAttrInterface>(target);
  ASSERT_TRUE(!!serializer);
  gpu::TargetOptions options("", {}, "", gpu::CompilationTarget::Binary);
  for (auto gpuModule : (*module).getBody()->getOps<gpu::GPUModuleOp>()) {
    std::optional<SmallVector<char, 0>> object =
        serializer.serializeToObject(gpuModule, options);
    // Check that the serializer was successful.
    ASSERT_TRUE(object != std::nullopt);
    ASSERT_TRUE(!object->empty());
  }
}
