//===-- LLVMIR.h - C Interface for MLIR LLVMIR Target ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Target/LLVMIR.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Target/LLVMIR/TypeFromLLVM.h"

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

DEFINE_C_API_PTR_METHODS(MlirTypeFromLLVMIRTranslator,
                         mlir::LLVM::TypeFromLLVMIRTranslator)

MlirTypeFromLLVMIRTranslator
mlirTypeFromLLVMIRTranslatorCreate(MlirContext ctx) {
  MLIRContext *context = unwrap(ctx);
  auto *translator = new LLVM::TypeFromLLVMIRTranslator(*context);
  return wrap(translator);
}

void mlirTypeFromLLVMIRTranslatorDestroy(
    MlirTypeFromLLVMIRTranslator translator) {
  delete static_cast<LLVM::TypeFromLLVMIRTranslator *>(unwrap(translator));
}

MlirType mlirTypeFromLLVMIRTranslatorTranslateType(
    MlirTypeFromLLVMIRTranslator translator, LLVMTypeRef llvmType) {
  LLVM::TypeFromLLVMIRTranslator *translator_ = unwrap(translator);
  mlir::Type type = translator_->translateType(llvm::unwrap(llvmType));
  return wrap(type);
}

DEFINE_C_API_PTR_METHODS(MlirTypeToLLVMIRTranslator,
                         mlir::LLVM::TypeToLLVMIRTranslator)

MlirTypeToLLVMIRTranslator
mlirTypeToLLVMIRTranslatorCreate(LLVMContextRef ctx) {
  llvm::LLVMContext *context = llvm::unwrap(ctx);
  auto *translator = new LLVM::TypeToLLVMIRTranslator(*context);
  return wrap(translator);
}

void mlirTypeToLLVMIRTranslatorDestroy(MlirTypeToLLVMIRTranslator translator) {
  delete static_cast<LLVM::TypeToLLVMIRTranslator *>(unwrap(translator));
}

LLVMTypeRef
mlirTypeToLLVMIRTranslatorTranslateType(MlirTypeToLLVMIRTranslator translator,
                                        MlirType mlirType) {
  LLVM::TypeToLLVMIRTranslator *translator_ = unwrap(translator);
  llvm::Type *type = translator_->translateType(unwrap(mlirType));
  return llvm::wrap(type);
}
