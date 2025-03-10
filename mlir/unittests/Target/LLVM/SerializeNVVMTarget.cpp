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

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Config/llvm-config.h" // for LLVM_HAS_NVPTX_TARGET
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"

#include "gmock/gmock.h"
#include <cstdint>

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
  gpu::TargetOptions options("", {}, "", "", gpu::CompilationTarget::Offload);
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
  gpu::TargetOptions options("", {}, "", "", gpu::CompilationTarget::Assembly);
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
  gpu::TargetOptions options("", {}, "", "", gpu::CompilationTarget::Binary);
  for (auto gpuModule : (*module).getBody()->getOps<gpu::GPUModuleOp>()) {
    std::optional<SmallVector<char, 0>> object =
        serializer.serializeToObject(gpuModule, options);
    // Check that the serializer was successful.
    ASSERT_TRUE(object != std::nullopt);
    ASSERT_TRUE(!object->empty());
  }
}

// Test callback functions invoked with LLVM IR and ISA.
TEST_F(MLIRTargetLLVMNVVM,
       SKIP_WITHOUT_NVPTX(CallbackInvokedWithLLVMIRAndISA)) {
  MLIRContext context(registry);

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(moduleStr, &context);
  ASSERT_TRUE(!!module);

  NVVM::NVVMTargetAttr target = NVVM::NVVMTargetAttr::get(&context);

  auto serializer = dyn_cast<gpu::TargetAttrInterface>(target);
  ASSERT_TRUE(!!serializer);

  std::string initialLLVMIR;
  auto initialCallback = [&initialLLVMIR](llvm::Module &module) {
    llvm::raw_string_ostream ros(initialLLVMIR);
    module.print(ros, nullptr);
  };

  std::string linkedLLVMIR;
  auto linkedCallback = [&linkedLLVMIR](llvm::Module &module) {
    llvm::raw_string_ostream ros(linkedLLVMIR);
    module.print(ros, nullptr);
  };

  std::string optimizedLLVMIR;
  auto optimizedCallback = [&optimizedLLVMIR](llvm::Module &module) {
    llvm::raw_string_ostream ros(optimizedLLVMIR);
    module.print(ros, nullptr);
  };

  std::string isaResult;
  auto isaCallback = [&isaResult](llvm::StringRef isa) {
    isaResult = isa.str();
  };

  gpu::TargetOptions options({}, {}, {}, {}, gpu::CompilationTarget::Assembly,
                             {}, initialCallback, linkedCallback,
                             optimizedCallback, isaCallback);

  for (auto gpuModule : (*module).getBody()->getOps<gpu::GPUModuleOp>()) {
    std::optional<SmallVector<char, 0>> object =
        serializer.serializeToObject(gpuModule, options);

    ASSERT_TRUE(object != std::nullopt);
    ASSERT_TRUE(!object->empty());
    ASSERT_TRUE(!initialLLVMIR.empty());
    ASSERT_TRUE(!linkedLLVMIR.empty());
    ASSERT_TRUE(!optimizedLLVMIR.empty());
    ASSERT_TRUE(!isaResult.empty());

    initialLLVMIR.clear();
    linkedLLVMIR.clear();
    optimizedLLVMIR.clear();
    isaResult.clear();
  }
}

// Test linking LLVM IR from a resource attribute.
TEST_F(MLIRTargetLLVMNVVM, SKIP_WITHOUT_NVPTX(LinkedLLVMIRResource)) {
  MLIRContext context(registry);
  std::string moduleStr = R"mlir(
    gpu.module @nvvm_test {
      llvm.func @bar()
      llvm.func @nvvm_kernel(%arg0: f32) attributes {gpu.kernel, nvvm.kernel} {
        llvm.call @bar() : () -> ()
        llvm.return
      }
    }
  )mlir";
  // Provide the library to link as a serialized bitcode blob.
  SmallVector<char> bitcodeToLink;
  {
    std::string linkedLib = R"llvm(
  define void @bar() {
    ret void
  }
  )llvm";
    llvm::SMDiagnostic err;
    llvm::MemoryBufferRef buffer(linkedLib, "linkedLib");
    llvm::LLVMContext llvmCtx;
    std::unique_ptr<llvm::Module> module = llvm::parseIR(buffer, err, llvmCtx);
    ASSERT_TRUE(module) << " Can't parse IR: " << err.getMessage();
    {
      llvm::raw_svector_ostream os(bitcodeToLink);
      WriteBitcodeToFile(*module, os);
    }
  }

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(moduleStr, &context);
  ASSERT_TRUE(!!module);
  Builder builder(&context);

  NVVM::NVVMTargetAttr target = NVVM::NVVMTargetAttr::get(&context);
  auto serializer = dyn_cast<gpu::TargetAttrInterface>(target);

  // Hook to intercept the LLVM IR after linking external libs.
  std::string linkedLLVMIR;
  auto linkedCallback = [&linkedLLVMIR](llvm::Module &module) {
    llvm::raw_string_ostream ros(linkedLLVMIR);
    module.print(ros, nullptr);
  };

  // Store the bitcode as a DenseI8ArrayAttr.
  SmallVector<Attribute> librariesToLink;
  librariesToLink.push_back(DenseI8ArrayAttr::get(
      &context,
      ArrayRef<int8_t>((int8_t *)bitcodeToLink.data(), bitcodeToLink.size())));
  gpu::TargetOptions options({}, librariesToLink, {}, {},
                             gpu::CompilationTarget::Assembly, {}, {},
                             linkedCallback);
  for (auto gpuModule : (*module).getBody()->getOps<gpu::GPUModuleOp>()) {
    std::optional<SmallVector<char, 0>> object =
        serializer.serializeToObject(gpuModule, options);

    // Verify that we correctly linked in the library: the external call is
    // replaced by the definition.
    ASSERT_TRUE(!linkedLLVMIR.empty());
    {
      llvm::SMDiagnostic err;
      llvm::MemoryBufferRef buffer(linkedLLVMIR, "linkedLLVMIR");
      llvm::LLVMContext llvmCtx;
      std::unique_ptr<llvm::Module> module =
          llvm::parseIR(buffer, err, llvmCtx);
      ASSERT_TRUE(module) << " Can't parse linkedLLVMIR: " << err.getMessage()
                          << " IR: \n\b" << linkedLLVMIR;
      llvm::Function *bar = module->getFunction("bar");
      ASSERT_TRUE(bar);
      ASSERT_FALSE(bar->empty());
    }
    ASSERT_TRUE(object != std::nullopt);
    ASSERT_TRUE(!object->empty());
  }
}
