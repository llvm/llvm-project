//===- TestToLLVMIRTranslation.cpp - Translate Test dialect to LLVM IR ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR Test dialect and LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "TestDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

namespace {

class TestDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  LogicalResult
  amendOperation(Operation *op, ArrayRef<llvm::Instruction *> instructions,
                 NamedAttribute attribute,
                 LLVM::ModuleTranslation &moduleTranslation) const final;

  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const final;
};

} // namespace

LogicalResult TestDialectLLVMIRTranslationInterface::amendOperation(
    Operation *op, ArrayRef<llvm::Instruction *> instructions,
    NamedAttribute attribute,
    LLVM::ModuleTranslation &moduleTranslation) const {
  return llvm::StringSwitch<llvm::function_ref<LogicalResult(Attribute)>>(
             attribute.getName())
      // The `test.discardable_mod_attr` attribute, if present and set to
      // `true`, results in the addition of a `test.symbol` in the module it is
      // attached to with name "sym_from_attr".
      .Case("test.discardable_mod_attr",
            [&](Attribute attr) {
              if (!isa<ModuleOp>(op)) {
                op->emitOpError("attribute 'test.discardable_mod_attr' only "
                                "supported in modules");
                return failure();
              }

              bool createSymbol = false;
              if (auto boolAttr = attr.dyn_cast<BoolAttr>())
                createSymbol = boolAttr.getValue();

              if (createSymbol) {
                OpBuilder builder(op->getRegion(0));
                builder.create<test::SymbolOp>(
                    op->getLoc(),
                    StringAttr::get(op->getContext(), "sym_from_attr"),
                    /*sym_visibility=*/nullptr);
              }

              return success();
            })
      .Case("test.add_annotation",
            [&](Attribute attr) {
              for (llvm::Instruction *instruction : instructions) {
                if (auto strAttr = dyn_cast<StringAttr>(attr)) {
                  instruction->addAnnotationMetadata("annotation_from_test: " +
                                                     strAttr.getValue().str());
                } else {
                  instruction->addAnnotationMetadata("annotation_from_test");
                }
              }
              return success();
            })
      .Default([](Attribute) {
        // Skip other discardable dialect attributes.
        return success();
      })(attribute.getValue());
}

LogicalResult TestDialectLLVMIRTranslationInterface::convertOperation(
    Operation *op, llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation) const {
  return llvm::TypeSwitch<Operation *, LogicalResult>(op)
      // `test.symbol`s are translated into global integers in LLVM IR, with a
      // name equal to the symbol they are translated from.
      .Case([&](test::SymbolOp symOp) {
        llvm::Module *mod = moduleTranslation.getLLVMModule();
        llvm::IntegerType *i32Type =
            llvm::IntegerType::get(moduleTranslation.getLLVMContext(), 32);
        mod->getOrInsertGlobal(symOp.getSymName(), i32Type);
        return success();
      })
      .Default([&](Operation *) {
        return op->emitOpError("unsupported translation of test operation");
      });
}

namespace mlir {

void registerTestToLLVMIR() {
  TranslateFromMLIRRegistration registration(
      "test-to-llvmir", "test dialect to LLVM IR",
      [](Operation *op, raw_ostream &output) {
        llvm::LLVMContext llvmContext;
        auto llvmModule = translateModuleToLLVMIR(op, llvmContext);
        if (!llvmModule)
          return failure();

        llvmModule->print(output, nullptr);
        return success();
      },
      [](DialectRegistry &registry) {
        registry.insert<test::TestDialect>();
        registerBuiltinDialectTranslation(registry);
        registerLLVMDialectTranslation(registry);
        registry.addExtension(
            +[](MLIRContext *ctx, test::TestDialect *dialect) {
              dialect->addInterfaces<TestDialectLLVMIRTranslationInterface>();
            });
      });
}

} // namespace mlir
