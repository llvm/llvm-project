//===-- LLVMIR.h - C Interface for MLIR LLVMIR Target ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Target/LLVMIR.h"
#include "llvm-c/Support.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include <memory>

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

using namespace mlir;

LLVMModuleRef mlirTranslateModuleToLLVMIR(MlirOperation module,
                                          LLVMContextRef context) {
  Operation *moduleOp = unwrap(module);

  llvm::LLVMContext *ctx = llvm::unwrap(context);

  std::unique_ptr<llvm::Module> llvmModule =
      mlir::translateModuleToLLVMIR(moduleOp, *ctx);

  LLVMModuleRef moduleRef = llvm::wrap(llvmModule.release());

  return moduleRef;
}
