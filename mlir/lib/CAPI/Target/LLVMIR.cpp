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
#include <memory>

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

using namespace mlir;

LLVMModuleRef mlirTranslateModuleToLLVMIR(MlirOperation module,
                                          LLVMContextRef context,
                                          MlirStringRef llvmModuleName) {
  Operation *moduleOp = unwrap(module);

  llvm::LLVMContext *ctx = reinterpret_cast<llvm::LLVMContext *>(context);

  std::unique_ptr<llvm::Module> llvmModule = mlir::translateModuleToLLVMIR(
      moduleOp, *ctx,
      llvm::StringRef(llvmModuleName.data, llvmModuleName.length));

  LLVMModuleRef moduleRef = reinterpret_cast<LLVMModuleRef>(
      const_cast<llvm::Module *>(llvmModule.release()));

  return moduleRef;
}
