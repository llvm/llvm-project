//===- SerializeToLLVMBitcode.cpp -------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Target/LLVM/ModuleToObject.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"

#include "gmock/gmock.h"

using namespace mlir;

// Skip the test if the native target was not built.
#if LLVM_NATIVE_TARGET_TEST_ENABLED == 0
#define SKIP_WITHOUT_NATIVE(x) DISABLED_##x
#else
#define SKIP_WITHOUT_NATIVE(x) x
#endif

namespace {
// Dummy interface for testing.
class TargetAttrImpl
    : public gpu::TargetAttrInterface::FallbackModel<TargetAttrImpl> {
public:
  std::optional<SmallVector<char, 0>>
  serializeToObject(Attribute attribute, Operation *module,
                    const gpu::TargetOptions &options) const;

  Attribute createObject(Attribute attribute, Operation *module,
                         const SmallVector<char, 0> &object,
                         const gpu::TargetOptions &options) const;
};
} // namespace

class MLIRTargetLLVM : public ::testing::Test {
protected:
  void SetUp() override {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    registry.addExtension(+[](MLIRContext *ctx, BuiltinDialect *dialect) {
      IntegerAttr::attachInterface<TargetAttrImpl>(*ctx);
    });
    registerBuiltinDialectTranslation(registry);
    registerLLVMDialectTranslation(registry);
    registry.insert<gpu::GPUDialect>();
  }

  // Dialect registry.
  DialectRegistry registry;

  // MLIR module used for the tests.
  std::string moduleStr = R"mlir(
  llvm.func @foo(%arg0 : i32) {
    llvm.return
  }
  )mlir";
};

TEST_F(MLIRTargetLLVM, SKIP_WITHOUT_NATIVE(SerializeToLLVMBitcode)) {
  MLIRContext context(registry);

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(moduleStr, &context);
  ASSERT_TRUE(!!module);

  // Serialize the module.
  std::string targetTriple = llvm::sys::getProcessTriple();
  LLVM::ModuleToObject serializer(*(module->getOperation()), targetTriple, "",
                                  "");
  std::optional<SmallVector<char, 0>> serializedModule = serializer.run();
  ASSERT_TRUE(!!serializedModule);
  ASSERT_TRUE(!serializedModule->empty());

  // Read the serialized module.
  llvm::MemoryBufferRef buffer(
      StringRef(serializedModule->data(), serializedModule->size()), "module");
  llvm::LLVMContext llvmContext;
  llvm::Expected<std::unique_ptr<llvm::Module>> llvmModule =
      llvm::getLazyBitcodeModule(buffer, llvmContext);
  ASSERT_TRUE(!!llvmModule);
  ASSERT_TRUE(!!*llvmModule);

  // Check that it has a function named `foo`.
  ASSERT_TRUE((*llvmModule)->getFunction("foo") != nullptr);
}

std::optional<SmallVector<char, 0>>
TargetAttrImpl::serializeToObject(Attribute attribute, Operation *module,
                                  const gpu::TargetOptions &options) const {
  // Set a dummy attr to be retrieved by `createObject`.
  module->setAttr("serialize_attr", UnitAttr::get(module->getContext()));
  std::string targetTriple = llvm::sys::getProcessTriple();
  LLVM::ModuleToObject serializer(
      *module, targetTriple, "", "", 3, options.getInitialLlvmIRCallback(),
      options.getLinkedLlvmIRCallback(), options.getOptimizedLlvmIRCallback());
  return serializer.run();
}

Attribute
TargetAttrImpl::createObject(Attribute attribute, Operation *module,
                             const SmallVector<char, 0> &object,
                             const gpu::TargetOptions &options) const {
  // Create a GPU object with the GPU module dictionary as the object
  // properties.
  return gpu::ObjectAttr::get(
      module->getContext(), attribute, gpu::CompilationTarget::Offload,
      StringAttr::get(module->getContext(),
                      StringRef(object.data(), object.size())),
      module->getAttrDictionary(), /*kernels=*/nullptr);
}

// This test checks the correct functioning of `TargetAttrInterface` as an API.
// In particular, it shows how `TargetAttrInterface::createObject` can leverage
// the `module` operation argument to retrieve information from the module.
TEST_F(MLIRTargetLLVM, SKIP_WITHOUT_NATIVE(TargetAttrAPI)) {
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(moduleStr, &context);
  ASSERT_TRUE(!!module);
  Builder builder(&context);
  IntegerAttr target = builder.getI32IntegerAttr(0);
  auto targetAttr = dyn_cast<gpu::TargetAttrInterface>(target);
  // Check the attribute holds the interface.
  ASSERT_TRUE(!!targetAttr);
  gpu::TargetOptions opts;
  std::optional<SmallVector<char, 0>> serializedBinary =
      targetAttr.serializeToObject(*module, opts);
  // Check the serialized string.
  ASSERT_TRUE(!!serializedBinary);
  ASSERT_TRUE(!serializedBinary->empty());
  // Create the object attribute.
  auto object = cast<gpu::ObjectAttr>(
      targetAttr.createObject(*module, *serializedBinary, opts));
  // Check the object has properties.
  DictionaryAttr properties = object.getProperties();
  ASSERT_TRUE(!!properties);
  // Check that it contains the attribute added to the module in
  // `serializeToObject`.
  ASSERT_TRUE(properties.contains("serialize_attr"));
}

// Test callback function invoked with initial LLVM IR
TEST_F(MLIRTargetLLVM, SKIP_WITHOUT_NATIVE(CallbackInvokedWithInitialLLVMIR)) {
  MLIRContext context(registry);

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(moduleStr, &context);
  ASSERT_TRUE(!!module);
  Builder builder(&context);
  IntegerAttr target = builder.getI32IntegerAttr(0);
  auto targetAttr = dyn_cast<gpu::TargetAttrInterface>(target);

  std::string initialLLVMIR;
  auto initialCallback = [&initialLLVMIR](llvm::Module &module) {
    llvm::raw_string_ostream ros(initialLLVMIR);
    module.print(ros, nullptr);
  };

  gpu::TargetOptions opts(
      {}, {}, {}, {}, mlir::gpu::TargetOptions::getDefaultCompilationTarget(),
      {}, initialCallback);
  std::optional<SmallVector<char, 0>> serializedBinary =
      targetAttr.serializeToObject(*module, opts);

  ASSERT_TRUE(serializedBinary != std::nullopt);
  ASSERT_TRUE(!serializedBinary->empty());
  ASSERT_TRUE(!initialLLVMIR.empty());
}

// Test callback function invoked with linked LLVM IR
TEST_F(MLIRTargetLLVM, SKIP_WITHOUT_NATIVE(CallbackInvokedWithLinkedLLVMIR)) {
  MLIRContext context(registry);

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(moduleStr, &context);
  ASSERT_TRUE(!!module);
  Builder builder(&context);
  IntegerAttr target = builder.getI32IntegerAttr(0);
  auto targetAttr = dyn_cast<gpu::TargetAttrInterface>(target);

  std::string linkedLLVMIR;
  auto linkedCallback = [&linkedLLVMIR](llvm::Module &module) {
    llvm::raw_string_ostream ros(linkedLLVMIR);
    module.print(ros, nullptr);
  };

  gpu::TargetOptions opts(
      {}, {}, {}, {}, mlir::gpu::TargetOptions::getDefaultCompilationTarget(),
      {}, {}, linkedCallback);
  std::optional<SmallVector<char, 0>> serializedBinary =
      targetAttr.serializeToObject(*module, opts);

  ASSERT_TRUE(serializedBinary != std::nullopt);
  ASSERT_TRUE(!serializedBinary->empty());
  ASSERT_TRUE(!linkedLLVMIR.empty());
}

// Test callback function invoked with optimized LLVM IR
TEST_F(MLIRTargetLLVM,
       SKIP_WITHOUT_NATIVE(CallbackInvokedWithOptimizedLLVMIR)) {
  MLIRContext context(registry);

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(moduleStr, &context);
  ASSERT_TRUE(!!module);
  Builder builder(&context);
  IntegerAttr target = builder.getI32IntegerAttr(0);
  auto targetAttr = dyn_cast<gpu::TargetAttrInterface>(target);

  std::string optimizedLLVMIR;
  auto optimizedCallback = [&optimizedLLVMIR](llvm::Module &module) {
    llvm::raw_string_ostream ros(optimizedLLVMIR);
    module.print(ros, nullptr);
  };

  gpu::TargetOptions opts(
      {}, {}, {}, {}, mlir::gpu::TargetOptions::getDefaultCompilationTarget(),
      {}, {}, {}, optimizedCallback);
  std::optional<SmallVector<char, 0>> serializedBinary =
      targetAttr.serializeToObject(*module, opts);

  ASSERT_TRUE(serializedBinary != std::nullopt);
  ASSERT_TRUE(!serializedBinary->empty());
  ASSERT_TRUE(!optimizedLLVMIR.empty());
}