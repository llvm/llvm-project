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

#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Translation.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;

std::unique_ptr<llvm::Module>
mlir::translateModuleToLLVMIR(ModuleOp m, llvm::LLVMContext &llvmContext,
                              StringRef name) {
  auto llvmModule =
      LLVM::ModuleTranslation::translateModule<>(m, llvmContext, name);
  if (!llvmModule)
    emitError(m.getLoc(), "Fail to convert MLIR to LLVM IR");
  else if (verifyModule(*llvmModule))
    emitError(m.getLoc(), "LLVM IR fails to verify");
  return llvmModule;
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
        registry.insert<LLVM::LLVMDialect, omp::OpenMPDialect>();
      });
}
} // namespace mlir
