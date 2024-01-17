//===- LLVMExternalModels.cpp - Implementation of LLVM external models ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the OpenMP external models for the LLVM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/ExternalModels.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"

using namespace mlir;

namespace {
struct LLVMPointerPointerLikeModel
    : public omp::PointerLikeType::ExternalModel<LLVMPointerPointerLikeModel,
                                                 LLVM::LLVMPointerType> {
  Type getElementType(Type pointer) const { return Type(); }
};
} // namespace

void omp::registerLLVMExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, LLVM::LLVMDialect *dialect) {
    LLVM::LLVMPointerType::attachInterface<LLVMPointerPointerLikeModel>(*ctx);
    // Attach default declare target interfaces to operations which can be
    // marked as declare target (Global Operations and Functions/Subroutines in
    // dialects that Fortran (or other languages that lower to MLIR) translates
    // too
    mlir::LLVM::GlobalOp::attachInterface<
        mlir::omp::DeclareTargetDefaultModel<mlir::LLVM::GlobalOp>>(*ctx);
    mlir::LLVM::LLVMFuncOp::attachInterface<
        mlir::omp::DeclareTargetDefaultModel<mlir::LLVM::LLVMFuncOp>>(*ctx);
    // Attach default early outlining interface to func ops.
    mlir::LLVM::LLVMFuncOp::attachInterface<
        mlir::omp::EarlyOutliningDefaultModel<mlir::LLVM::LLVMFuncOp>>(*ctx);
  });
}
