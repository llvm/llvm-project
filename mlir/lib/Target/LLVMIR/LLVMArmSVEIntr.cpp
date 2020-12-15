//===- LLVMArmSVEIntr.cpp - Convert MLIR LLVM dialect to LLVM intrinsics --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR LLVM and ArmSVE dialects
// and LLVM IR with Arm SVE intrinsics.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMArmSVEDialect.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Translation.h"
#include "llvm/IR/IntrinsicsAArch64.h"

using namespace mlir;

namespace {
class LLVMArmSVEModuleTranslation : public LLVM::ModuleTranslation {
  friend LLVM::ModuleTranslation;

public:
  using LLVM::ModuleTranslation::ModuleTranslation;

protected:
  LogicalResult convertOperation(Operation &opInst,
                                 llvm::IRBuilder<> &builder) override {
#include "mlir/Dialect/LLVMIR/LLVMArmSVEConversions.inc"

    return LLVM::ModuleTranslation::convertOperation(opInst, builder);
  }
};
} // end namespace

static std::unique_ptr<llvm::Module>
translateLLVMArmSVEModuleToLLVMIR(Operation *m, llvm::LLVMContext &llvmContext,
                                  StringRef name) {
  return LLVM::ModuleTranslation::translateModule<LLVMArmSVEModuleTranslation>(
      m, llvmContext, name);
}

namespace mlir {
void registerArmSVEToLLVMIRTranslation() {
  TranslateFromMLIRRegistration reg(
      "arm-sve-mlir-to-llvmir",
      [](ModuleOp module, raw_ostream &output) {
        llvm::LLVMContext llvmContext;
        auto llvmModule = translateLLVMArmSVEModuleToLLVMIR(
            module, llvmContext, "LLVMDialectModule");
        if (!llvmModule)
          return failure();

        llvmModule->print(output, nullptr);
        return success();
      },
      [](DialectRegistry &registry) {
        registry.insert<LLVM::LLVMArmSVEDialect, LLVM::LLVMDialect>();
      });
}
} // namespace mlir
