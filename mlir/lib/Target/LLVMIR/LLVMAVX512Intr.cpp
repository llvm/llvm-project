//===- AVX512Intr.cpp - Convert MLIR LLVM dialect to LLVM intrinsics ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR LLVM and AVX512 dialects
// and LLVM IR with AVX intrinsics.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMAVX512Dialect.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Translation.h"
#include "llvm/IR/IntrinsicsX86.h"

using namespace mlir;

namespace {
class LLVMAVX512ModuleTranslation : public LLVM::ModuleTranslation {
  friend LLVM::ModuleTranslation;

public:
  using LLVM::ModuleTranslation::ModuleTranslation;

protected:
  LogicalResult convertOperation(Operation &opInst,
                                 llvm::IRBuilder<> &builder) override {
#include "mlir/Dialect/LLVMIR/LLVMAVX512Conversions.inc"

    return LLVM::ModuleTranslation::convertOperation(opInst, builder);
  }
};

std::unique_ptr<llvm::Module>
translateLLVMAVX512ModuleToLLVMIR(Operation *m, llvm::LLVMContext &llvmContext,
                                  StringRef name) {
  return LLVM::ModuleTranslation::translateModule<LLVMAVX512ModuleTranslation>(
      m, llvmContext, name);
}
} // end namespace

namespace mlir {
void registerAVX512ToLLVMIRTranslation() {
  TranslateFromMLIRRegistration reg(
      "avx512-mlir-to-llvmir", [](ModuleOp module, raw_ostream &output) {
        llvm::LLVMContext llvmContext;
        auto llvmModule = translateLLVMAVX512ModuleToLLVMIR(
            module, llvmContext, "LLVMDialectModule");
        if (!llvmModule)
          return failure();

        llvmModule->print(output, nullptr);
        return success();
      });
}
} // namespace mlir
