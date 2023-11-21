//===- SerializeROCDLTarget.cpp ---------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Target/LLVM/ROCDL/Target.h"
#include "mlir/Target/LLVM/ROCDL/Utils.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"

#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"

#include "gmock/gmock.h"

using namespace mlir;

// Skip the test if the AMDGPU target was not built.
#if MLIR_ROCM_CONVERSIONS_ENABLED == 0
#define SKIP_WITHOUT_AMDGPU(x) DISABLED_##x
#else
#define SKIP_WITHOUT_AMDGPU(x) x
#endif

class MLIRTargetLLVMROCDL : public ::testing::Test {
protected:
  virtual void SetUp() {
    registerBuiltinDialectTranslation(registry);
    registerLLVMDialectTranslation(registry);
    registerGPUDialectTranslation(registry);
    registerROCDLDialectTranslation(registry);
    ROCDL::registerROCDLTargetInterfaceExternalModels(registry);
  }

  // Checks if a ROCm installation is available.
  bool hasROCMTools() {
    StringRef rocmPath = ROCDL::getROCMPath();
    if (rocmPath.empty())
      return false;
    llvm::SmallString<128> lldPath(rocmPath);
    llvm::sys::path::append(lldPath, "llvm", "bin", "ld.lld");
    return llvm::sys::fs::can_execute(lldPath);
  }

  // Dialect registry.
  DialectRegistry registry;

  // MLIR module used for the tests.
  const std::string moduleStr = R"mlir(
      gpu.module @rocdl_test {
        llvm.func @rocdl_kernel(%arg0: f32) attributes {gpu.kernel, rocdl.kernel} {
        llvm.return
      }
    })mlir";
};

// Test ROCDL serialization to LLVM.
TEST_F(MLIRTargetLLVMROCDL, SKIP_WITHOUT_AMDGPU(SerializeROCDLMToLLVM)) {
  MLIRContext context(registry);

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(moduleStr, &context);
  ASSERT_TRUE(!!module);

  // Create a ROCDL target.
  ROCDL::ROCDLTargetAttr target = ROCDL::ROCDLTargetAttr::get(&context);

  // Serialize the module.
  auto serializer = dyn_cast<gpu::TargetAttrInterface>(target);
  ASSERT_TRUE(!!serializer);
  gpu::TargetOptions options("", {}, "", gpu::CompilationTarget::Offload);
  for (auto gpuModule : (*module).getBody()->getOps<gpu::GPUModuleOp>()) {
    std::optional<SmallVector<char, 0>> object =
        serializer.serializeToObject(gpuModule, options);
    // Check that the serializer was successful.
    ASSERT_TRUE(object != std::nullopt);
    ASSERT_TRUE(object->size() > 0);

    // Read the serialized module.
    llvm::MemoryBufferRef buffer(StringRef(object->data(), object->size()),
                                 "module");
    llvm::LLVMContext llvmContext;
    llvm::Expected<std::unique_ptr<llvm::Module>> llvmModule =
        llvm::getLazyBitcodeModule(buffer, llvmContext);
    ASSERT_TRUE(!!llvmModule);
    ASSERT_TRUE(!!*llvmModule);

    // Check that it has a function named `foo`.
    ASSERT_TRUE((*llvmModule)->getFunction("rocdl_kernel") != nullptr);
  }
}

// Test ROCDL serialization to PTX.
TEST_F(MLIRTargetLLVMROCDL, SKIP_WITHOUT_AMDGPU(SerializeROCDLToPTX)) {
  MLIRContext context(registry);

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(moduleStr, &context);
  ASSERT_TRUE(!!module);

  // Create a ROCDL target.
  ROCDL::ROCDLTargetAttr target = ROCDL::ROCDLTargetAttr::get(&context);

  // Serialize the module.
  auto serializer = dyn_cast<gpu::TargetAttrInterface>(target);
  ASSERT_TRUE(!!serializer);
  gpu::TargetOptions options("", {}, "", gpu::CompilationTarget::Assembly);
  for (auto gpuModule : (*module).getBody()->getOps<gpu::GPUModuleOp>()) {
    std::optional<SmallVector<char, 0>> object =
        serializer.serializeToObject(gpuModule, options);
    // Check that the serializer was successful.
    ASSERT_TRUE(object != std::nullopt);
    ASSERT_TRUE(object->size() > 0);

    ASSERT_TRUE(
        StringRef(object->data(), object->size()).contains("rocdl_kernel"));
  }
}

// Test ROCDL serialization to Binary.
TEST_F(MLIRTargetLLVMROCDL, SKIP_WITHOUT_AMDGPU(SerializeROCDLToBinary)) {
  if (!hasROCMTools())
    GTEST_SKIP() << "ROCm installation not found, skipping test.";

  MLIRContext context(registry);

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(moduleStr, &context);
  ASSERT_TRUE(!!module);

  // Create a ROCDL target.
  ROCDL::ROCDLTargetAttr target = ROCDL::ROCDLTargetAttr::get(&context);

  // Serialize the module.
  auto serializer = dyn_cast<gpu::TargetAttrInterface>(target);
  ASSERT_TRUE(!!serializer);
  gpu::TargetOptions options("", {}, "", gpu::CompilationTarget::Binary);
  for (auto gpuModule : (*module).getBody()->getOps<gpu::GPUModuleOp>()) {
    std::optional<SmallVector<char, 0>> object =
        serializer.serializeToObject(gpuModule, options);
    // Check that the serializer was successful.
    ASSERT_TRUE(object != std::nullopt);
    ASSERT_FALSE(object->empty());
  }
}
