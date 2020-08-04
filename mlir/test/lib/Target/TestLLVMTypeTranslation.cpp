//===- TestLLVMTypeTranslation.cpp - Test MLIR/LLVM IR type translation ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Target/LLVMIR/TypeTranslation.h"
#include "mlir/Translation.h"

using namespace mlir;

namespace {
class TestLLVMTypeTranslation : public LLVM::ModuleTranslation {
  // Allow access to the constructors under MSVC.
  friend LLVM::ModuleTranslation;

public:
  using LLVM::ModuleTranslation::ModuleTranslation;

protected:
  /// Simple test facility for translating types from MLIR LLVM dialect to LLVM
  /// IR. This converts the "llvm.test_introduce_func" operation into an LLVM IR
  /// function with the name extracted from the `name` attribute that returns
  /// the type contained in the `type` attribute if it is a non-function type or
  /// that has the signature obtained by converting `type` if it is a function
  /// type. This is a temporary check before type translation is substituted
  /// into the main translation flow and exercised here.
  LogicalResult convertOperation(Operation &op,
                                 llvm::IRBuilder<> &builder) override {
    if (op.getName().getStringRef() == "llvm.test_introduce_func") {
      auto attr = op.getAttrOfType<TypeAttr>("type");
      assert(attr && "expected 'type' attribute");
      auto type = attr.getValue().cast<LLVM::LLVMType>();

      auto nameAttr = op.getAttrOfType<StringAttr>("name");
      assert(nameAttr && "expected 'name' attributes");

      llvm::Type *translated =
          LLVM::translateTypeToLLVMIR(type, builder.getContext());

      llvm::Module *module = builder.GetInsertBlock()->getModule();
      if (auto *funcType = dyn_cast<llvm::FunctionType>(translated))
        module->getOrInsertFunction(nameAttr.getValue(), funcType);
      else
        module->getOrInsertFunction(nameAttr.getValue(), translated);

      std::string roundtripName = (Twine(nameAttr.getValue()) + "_round").str();
      LLVM::LLVMType translatedBack =
          LLVM::translateTypeFromLLVMIR(translated, *op.getContext());
      llvm::Type *translatedBackAndForth =
          LLVM::translateTypeToLLVMIR(translatedBack, builder.getContext());
      if (auto *funcType = dyn_cast<llvm::FunctionType>(translatedBackAndForth))
        module->getOrInsertFunction(roundtripName, funcType);
      else
        module->getOrInsertFunction(roundtripName, translatedBackAndForth);
      return success();
    }

    return LLVM::ModuleTranslation::convertOperation(op, builder);
  }
};
} // namespace

namespace mlir {
void registerTestLLVMTypeTranslation() {
  TranslateFromMLIRRegistration reg(
      "test-mlir-to-llvmir", [](ModuleOp module, raw_ostream &output) {
        std::unique_ptr<llvm::Module> llvmModule =
            LLVM::ModuleTranslation::translateModule<TestLLVMTypeTranslation>(
                module.getOperation());
        llvmModule->print(output, nullptr);
        return success();
      });
}
} // namespace mlir
