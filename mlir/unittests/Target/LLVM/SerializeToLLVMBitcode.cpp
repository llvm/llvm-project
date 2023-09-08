//===- SerializeToLLVMBitcode.cpp -------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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

class MLIRTargetLLVM : public ::testing::Test {
protected:
  virtual void SetUp() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
  }
};

TEST_F(MLIRTargetLLVM, SKIP_WITHOUT_NATIVE(SerializeToLLVMBitcode)) {
  std::string moduleStr = R"mlir(
  llvm.func @foo(%arg0 : i32) {
    llvm.return
  }
  )mlir";

  DialectRegistry registry;
  registerBuiltinDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);
  MLIRContext context(registry);

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(moduleStr, &context);
  ASSERT_TRUE(!!module);

  // Serialize the module.
  std::string targetTriple = llvm::sys::getDefaultTargetTriple();
  LLVM::ModuleToObject serializer(*(module->getOperation()), targetTriple, "",
                                  "");
  std::optional<SmallVector<char, 0>> serializedModule = serializer.run();
  ASSERT_TRUE(!!serializedModule);
  ASSERT_TRUE(serializedModule->size() > 0);

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
