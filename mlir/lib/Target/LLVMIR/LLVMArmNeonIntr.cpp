//===- ArmNeonIntr.cpp - Convert MLIR LLVM dialect to LLVM intrinsics -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR LLVM and ArmNeon dialects
// and LLVM IR with ArmNeon intrinsics.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMArmNeonDialect.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Translation.h"
#include "llvm/IR/IntrinsicsAArch64.h"

using namespace mlir;

namespace {
class LLVMArmNeonModuleTranslation : public LLVM::ModuleTranslation {
  friend LLVM::ModuleTranslation;

public:
  using LLVM::ModuleTranslation::ModuleTranslation;

protected:
  LogicalResult convertOperation(Operation &opInst,
                                 llvm::IRBuilder<> &builder) override {
#include "mlir/Dialect/LLVMIR/LLVMArmNeonConversions.inc"

    return LLVM::ModuleTranslation::convertOperation(opInst, builder);
  }
};

std::unique_ptr<llvm::Module>
translateLLVMArmNeonModuleToLLVMIR(Operation *m, llvm::LLVMContext &llvmContext,
                                   StringRef name) {
  return LLVM::ModuleTranslation::translateModule<LLVMArmNeonModuleTranslation>(
      m, llvmContext, name);
}
} // end namespace

namespace mlir {
void registerArmNeonToLLVMIRTranslation() {
  TranslateFromMLIRRegistration reg(
      "arm-neon-mlir-to-llvmir",
      [](ModuleOp module, raw_ostream &output) {
        llvm::LLVMContext llvmContext;
        auto llvmModule = translateLLVMArmNeonModuleToLLVMIR(
            module, llvmContext, "LLVMDialectModule");
        if (!llvmModule)
          return failure();

        llvmModule->print(output, nullptr);
        return success();
      },
      [](DialectRegistry &registry) {
        registry.insert<LLVM::LLVMArmNeonDialect, LLVM::LLVMDialect>();
      });
}
} // namespace mlir
