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

#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Config/Targets.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
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
#if LLVM_HAS_AMDGPU_TARGET
#define SKIP_WITHOUT_AMDGPU(x) x
#else
#define SKIP_WITHOUT_AMDGPU(x) DISABLED_##x
#endif

class MLIRTargetLLVMROCDL : public ::testing::Test {
protected:
  void SetUp() override {
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

  // Checks if the ROCm device libraries are available.
  bool hasROCMDeviceLibs() {
    StringRef rocmPath = ROCDL::getROCMPath();
    if (rocmPath.empty())
      return false;
    llvm::SmallString<128> ocmlPath(rocmPath);
    llvm::sys::path::append(ocmlPath, "amdgcn", "bitcode", "ocml.bc");
    return llvm::sys::fs::is_regular_file(ocmlPath);
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
TEST_F(MLIRTargetLLVMROCDL, SKIP_WITHOUT_AMDGPU(SerializeROCDLToLLVM)) {
  MLIRContext context(registry);

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(moduleStr, &context);
  ASSERT_TRUE(!!module);

  // Create a ROCDL target.
  ROCDL::ROCDLTargetAttr target = ROCDL::ROCDLTargetAttr::get(&context);

  // Serialize the module.
  auto serializer = dyn_cast<gpu::TargetAttrInterface>(target);
  ASSERT_TRUE(!!serializer);
  gpu::TargetOptions options("", {}, "", "", gpu::CompilationTarget::Offload);
  for (auto gpuModule : (*module).getBody()->getOps<gpu::GPUModuleOp>()) {
    std::optional<mlir::gpu::SerializedObject> object =
        serializer.serializeToObject(gpuModule, options);
    // Check that the serializer was successful.
    ASSERT_TRUE(object != std::nullopt);
    ASSERT_TRUE(!object->getObject().empty());

    // Read the serialized module.
    llvm::MemoryBufferRef buffer(
        StringRef(object->getObject().data(), object->getObject().size()),
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

TEST_F(MLIRTargetLLVMROCDL, SKIP_WITHOUT_AMDGPU(ISAVersionControlVariable)) {
  if (!(hasROCMDeviceLibs() && hasROCMTools()))
    GTEST_SKIP() << "ROCm device libraries or tools not found, skipping test.";

  // The call to an `__ocml_` function makes the serializer add the control
  // variables.
  const std::string moduleStr = R"mlir(
      gpu.module @rocdl_test {
        llvm.func @__ocml_exp_f32(f32) -> f32
        llvm.func @rocdl_kernel(%arg0: f32) attributes {gpu.kernel, rocdl.kernel} {
          %0 = llvm.call @__ocml_exp_f32(%arg0) : (f32) -> f32
          llvm.return
        }
      })mlir";

  struct TestCase {
    StringRef chip;
    uint64_t isaVersion;
  };
  const TestCase testCases[] = {
      {"gfx908", 9008},   {"gfx90a", 9010},   {"gfx942", 9402},
      {"gfx950", 9500},   {"gfx1030", 10300}, {"gfx1100", 11000},
      {"gfx1201", 12001}, {"gfx1250", 12500},
  };

  for (const TestCase &testCase : testCases) {
    SCOPED_TRACE(testCase.chip);
    MLIRContext context(registry);
    OwningOpRef<ModuleOp> module =
        parseSourceString<ModuleOp>(moduleStr, &context);
    ASSERT_TRUE(!!module);

    // Use `-O0` so the optimizer doesn't remove the unused control variable.
    ROCDL::ROCDLTargetAttr target = ROCDL::ROCDLTargetAttr::get(
        &context, 0, "amdgcn-amd-amdhsa", testCase.chip, "", "600");
    auto serializer = dyn_cast<gpu::TargetAttrInterface>(target);
    ASSERT_TRUE(!!serializer);

    gpu::TargetOptions options("", {}, "", "", gpu::CompilationTarget::Offload);
    std::optional<uint64_t> isaVersion;
    for (auto gpuModule : (*module).getBody()->getOps<gpu::GPUModuleOp>()) {
      std::optional<mlir::gpu::SerializedObject> object =
          serializer.serializeToObject(gpuModule, options);
      ASSERT_TRUE(object != std::nullopt);

      // Read the serialized module and get the control variable.
      llvm::MemoryBufferRef buffer(
          StringRef(object->getObject().data(), object->getObject().size()),
          "module");
      llvm::LLVMContext llvmContext;
      llvm::Expected<std::unique_ptr<llvm::Module>> llvmModule =
          llvm::parseBitcodeFile(buffer, llvmContext);
      ASSERT_TRUE(!!llvmModule);
      const llvm::GlobalVariable *global =
          (*llvmModule)->getNamedGlobal("__oclc_ISA_version");
      ASSERT_TRUE(global != nullptr);
      const auto *value = dyn_cast<llvm::ConstantInt>(global->getInitializer());
      ASSERT_TRUE(value != nullptr);
      isaVersion = value->getZExtValue();
    }
    ASSERT_TRUE(isaVersion.has_value());
    EXPECT_EQ(*isaVersion, testCase.isaVersion);
  }
}

// Test ROCDL serialization to ISA with default code object version.
TEST_F(MLIRTargetLLVMROCDL,
       SKIP_WITHOUT_AMDGPU(SerializeROCDLToISAWithDefaultCOV)) {
  MLIRContext context(registry);

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(moduleStr, &context);
  ASSERT_TRUE(!!module);

  // Create a ROCDL target.
  ROCDL::ROCDLTargetAttr target = ROCDL::ROCDLTargetAttr::get(&context);

  // Serialize the module.
  auto serializer = dyn_cast<gpu::TargetAttrInterface>(target);
  ASSERT_TRUE(!!serializer);
  gpu::TargetOptions options("", {}, "", "", gpu::CompilationTarget::Assembly);
  for (auto gpuModule : (*module).getBody()->getOps<gpu::GPUModuleOp>()) {
    std::optional<mlir::gpu::SerializedObject> object =
        serializer.serializeToObject(gpuModule, options);
    // Check that the serializer was successful.
    EXPECT_TRUE(
        StringRef(object->getObject().data(), object->getObject().size())
            .contains(".amdhsa_code_object_version 6"));
  }
}

// Test ROCDL serialization to ISA with non-default code object version.
TEST_F(MLIRTargetLLVMROCDL,
       SKIP_WITHOUT_AMDGPU(SerializeROCDLToISAWithNonDefaultCOV)) {
  MLIRContext context(registry);

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(moduleStr, &context);
  ASSERT_TRUE(!!module);

  // Create a ROCDL target.
  ROCDL::ROCDLTargetAttr target = ROCDL::ROCDLTargetAttr::get(
      &context, 2, "amdgcn-amd-amdhsa", "gfx900", "", "400");

  // Serialize the module.
  auto serializer = dyn_cast<gpu::TargetAttrInterface>(target);
  ASSERT_TRUE(!!serializer);
  gpu::TargetOptions options("", {}, "", "", gpu::CompilationTarget::Assembly);
  for (auto gpuModule : (*module).getBody()->getOps<gpu::GPUModuleOp>()) {
    std::optional<mlir::gpu::SerializedObject> object =
        serializer.serializeToObject(gpuModule, options);
    // Check that the serializer was successful.
    EXPECT_TRUE(
        StringRef(object->getObject().data(), object->getObject().size())
            .contains(".amdhsa_code_object_version 4"));
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
  gpu::TargetOptions options("", {}, "", "", gpu::CompilationTarget::Assembly);
  for (auto gpuModule : (*module).getBody()->getOps<gpu::GPUModuleOp>()) {
    std::optional<mlir::gpu::SerializedObject> object =
        serializer.serializeToObject(gpuModule, options);
    // Check that the serializer was successful.
    ASSERT_TRUE(object != std::nullopt);
    ASSERT_TRUE(!object->getObject().empty());

    ASSERT_TRUE(
        StringRef(object->getObject().data(), object->getObject().size())
            .contains("rocdl_kernel"));
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
  gpu::TargetOptions options("", {}, "", "", gpu::CompilationTarget::Binary);
  for (auto gpuModule : (*module).getBody()->getOps<gpu::GPUModuleOp>()) {
    std::optional<mlir::gpu::SerializedObject> object =
        serializer.serializeToObject(gpuModule, options);
    // Check that the serializer was successful.
    ASSERT_TRUE(object != std::nullopt);
    ASSERT_FALSE(object->getObject().empty());
  }
}

// Test ROCDL metadata.
TEST_F(MLIRTargetLLVMROCDL, SKIP_WITHOUT_AMDGPU(GetELFMetadata)) {
  if (!hasROCMTools())
    GTEST_SKIP() << "ROCm installation not found, skipping test.";

  MLIRContext context(registry);

  // MLIR module used for the tests.
  const std::string moduleStr = R"mlir(
    gpu.module @rocdl_test {
    llvm.func @rocdl_kernel_1(%arg0: f32) attributes {gpu.kernel, rocdl.kernel} {
      llvm.return
    }
    llvm.func @rocdl_kernel_0(%arg0: f32) attributes {gpu.kernel, rocdl.kernel} {
      llvm.return
    }
    llvm.func @rocdl_kernel_2(%arg0: f32) attributes {gpu.kernel, rocdl.kernel} {
      llvm.return
    }
    llvm.func @a_kernel(%arg0: f32) attributes {gpu.kernel, rocdl.kernel} {
      llvm.return
    }
  })mlir";

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(moduleStr, &context);
  ASSERT_TRUE(!!module);

  // Create a ROCDL target.
  ROCDL::ROCDLTargetAttr target = ROCDL::ROCDLTargetAttr::get(&context);

  // Serialize the module.
  auto serializer = dyn_cast<gpu::TargetAttrInterface>(target);
  ASSERT_TRUE(!!serializer);
  gpu::TargetOptions options("", {}, "", "", gpu::CompilationTarget::Binary);
  for (auto gpuModule : (*module).getBody()->getOps<gpu::GPUModuleOp>()) {
    std::optional<mlir::gpu::SerializedObject> object =
        serializer.serializeToObject(gpuModule, options);
    // Check that the serializer was successful.
    ASSERT_TRUE(object != std::nullopt);
    ASSERT_FALSE(object->getObject().empty());
    if (!object)
      continue;
    // Get the metadata.
    gpu::KernelTableAttr metadata =
        ROCDL::getKernelMetadata(gpuModule, object->getObject());
    ASSERT_TRUE(metadata != nullptr);
    // There should be 4 kernels.
    ASSERT_TRUE(metadata.size() == 4);
    // Check that the lookup method returns finds the kernel.
    ASSERT_TRUE(metadata.lookup("a_kernel") != nullptr);
    ASSERT_TRUE(metadata.lookup("rocdl_kernel_0") != nullptr);
    // Check that the kernel doesn't exist.
    ASSERT_TRUE(metadata.lookup("not_existent_kernel") == nullptr);
    // Test the `KernelMetadataAttr` iterators.
    for (gpu::KernelMetadataAttr kernel : metadata) {
      // Check that the ELF metadata is present.
      ASSERT_TRUE(kernel.getMetadata() != nullptr);
      // Verify that `sgpr_count` is present and it is an integer attribute.
      ASSERT_TRUE(kernel.getAttr<IntegerAttr>("sgpr_count") != nullptr);
      // Verify that `vgpr_count` is present and it is an integer attribute.
      ASSERT_TRUE(kernel.getAttr<IntegerAttr>("vgpr_count") != nullptr);
    }
  }
}
