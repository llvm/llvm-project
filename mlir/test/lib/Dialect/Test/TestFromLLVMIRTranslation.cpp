//===- TestFromLLVMIRTranslation.cpp - Import Test dialect from LLVM IR ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between LLVM IR and the MLIR Test dialect.
//
//===----------------------------------------------------------------------===//

#include "TestDialect.h"
#include "TestOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMIRToLLVMTranslation.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/Target/LLVMIR/ModuleImport.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace test;

static ArrayRef<unsigned> getSupportedInstructionsImpl() {
  static unsigned instructions[] = {llvm::Instruction::Load};
  return instructions;
}

static LogicalResult convertLoad(OpBuilder &builder, llvm::Instruction *inst,
                                 ArrayRef<llvm::Value *> llvmOperands,
                                 LLVM::ModuleImport &moduleImport) {
  FailureOr<Value> addr = moduleImport.convertValue(llvmOperands[0]);
  if (failed(addr))
    return failure();
  // Create the LoadOp
  Value loadOp = builder.create<LLVM::LoadOp>(
      moduleImport.translateLoc(inst->getDebugLoc()),
      moduleImport.convertType(inst->getType()), *addr);
  moduleImport.mapValue(inst) = builder.create<SameOperandElementTypeOp>(
      loadOp.getLoc(), loadOp.getType(), loadOp, loadOp);
  return success();
}

namespace {
class TestDialectLLVMImportDialectInterface
    : public LLVMImportDialectInterface {
public:
  using LLVMImportDialectInterface::LLVMImportDialectInterface;

  LogicalResult
  convertInstruction(OpBuilder &builder, llvm::Instruction *inst,
                     ArrayRef<llvm::Value *> llvmOperands,
                     LLVM::ModuleImport &moduleImport) const override {
    switch (inst->getOpcode()) {
    case llvm::Instruction::Load:
      return convertLoad(builder, inst, llvmOperands, moduleImport);
    default:
      break;
    }
    return failure();
  }

  ArrayRef<unsigned> getSupportedInstructions() const override {
    return getSupportedInstructionsImpl();
  }
};
} // namespace

namespace mlir {
void registerTestFromLLVMIR() {
  TranslateToMLIRRegistration registration(
      "test-import-llvmir", "test dialect from LLVM IR",
      [](llvm::SourceMgr &sourceMgr,
         MLIRContext *context) -> OwningOpRef<Operation *> {
        llvm::SMDiagnostic err;
        llvm::LLVMContext llvmContext;
        std::unique_ptr<llvm::Module> llvmModule =
            llvm::parseIR(*sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID()),
                          err, llvmContext);
        if (!llvmModule) {
          std::string errStr;
          llvm::raw_string_ostream errStream(errStr);
          err.print(/*ProgName=*/"", errStream);
          emitError(UnknownLoc::get(context)) << errStream.str();
          return {};
        }
        if (llvm::verifyModule(*llvmModule, &llvm::errs()))
          return nullptr;

        return translateLLVMIRToModule(std::move(llvmModule), context, false);
      },
      [](DialectRegistry &registry) {
        registry.insert<DLTIDialect>();
        registry.insert<test::TestDialect>();
        registerLLVMDialectImport(registry);
        registry.addExtension(
            +[](MLIRContext *ctx, test::TestDialect *dialect) {
              dialect->addInterfaces<TestDialectLLVMImportDialectInterface>();
            });
      });
}
} // namespace mlir
