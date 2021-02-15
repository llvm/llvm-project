//===- ConvertToLLVMIR.cpp - MLIR to LLVM IR conversion -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR LLVM dialect and LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR.h"

#include "mlir/Dialect/LLVMIR/LLVMAVX512Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMArmNeonDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMArmSVEDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMAVX512/LLVMAVX512ToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMArmNeon/LLVMArmNeonToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMArmSVE/LLVMArmSVEToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Translation.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;

std::unique_ptr<llvm::Module>
mlir::translateModuleToLLVMIR(Operation *op, llvm::LLVMContext &llvmContext,
                              StringRef name) {
  auto llvmModule =
      LLVM::ModuleTranslation::translateModule<>(op, llvmContext, name);
  if (!llvmModule)
    emitError(op->getLoc(), "Fail to convert MLIR to LLVM IR");
  else if (verifyModule(*llvmModule))
    emitError(op->getLoc(), "LLVM IR fails to verify");
  return llvmModule;
}

void mlir::registerLLVMDialectTranslation(DialectRegistry &registry) {
  registry.insert<LLVM::LLVMDialect>();
  registry.addDialectInterface<LLVM::LLVMDialect,
                               LLVMDialectLLVMIRTranslationInterface>();
}

void mlir::registerLLVMDialectTranslation(MLIRContext &context) {
  auto *dialect = context.getLoadedDialect<LLVM::LLVMDialect>();
  if (!dialect || dialect->getRegisteredInterface<
                      LLVMDialectLLVMIRTranslationInterface>() == nullptr) {
    DialectRegistry registry;
    registry.insert<LLVM::LLVMDialect>();
    registry.addDialectInterface<LLVM::LLVMDialect,
                                 LLVMDialectLLVMIRTranslationInterface>();
    context.appendDialectRegistry(registry);
  }
}

namespace mlir {
void registerToLLVMIRTranslation() {
  TranslateFromMLIRRegistration registration(
      "mlir-to-llvmir",
      [](ModuleOp module, raw_ostream &output) {
        llvm::LLVMContext llvmContext;
        auto llvmModule = LLVM::ModuleTranslation::translateModule<>(
            module, llvmContext, "LLVMDialectModule");
        if (!llvmModule)
          return failure();

        llvmModule->print(output, nullptr);
        return success();
      },
      [](DialectRegistry &registry) {
        registry.insert<omp::OpenMPDialect, LLVM::LLVMAVX512Dialect,
                        LLVM::LLVMArmSVEDialect, LLVM::LLVMArmNeonDialect,
                        NVVM::NVVMDialect, ROCDL::ROCDLDialect>();
        registry.addDialectInterface<omp::OpenMPDialect,
                                     OpenMPDialectLLVMIRTranslationInterface>();
        registry
            .addDialectInterface<LLVM::LLVMAVX512Dialect,
                                 LLVMAVX512DialectLLVMIRTranslationInterface>();
        registry.addDialectInterface<
            LLVM::LLVMArmNeonDialect,
            LLVMArmNeonDialectLLVMIRTranslationInterface>();
        registry
            .addDialectInterface<LLVM::LLVMArmSVEDialect,
                                 LLVMArmSVEDialectLLVMIRTranslationInterface>();
        registry.addDialectInterface<NVVM::NVVMDialect,
                                     NVVMDialectLLVMIRTranslationInterface>();
        registry.addDialectInterface<ROCDL::ROCDLDialect,
                                     ROCDLDialectLLVMIRTranslationInterface>();
        registerLLVMDialectTranslation(registry);
      });
}
} // namespace mlir
