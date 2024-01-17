//===- BuiltinExternalModels.cpp - Impl of Builtin external models --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the OpenMP external models for the Builtin dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenMP/ExternalModels.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/BuiltinDialect.h"

using namespace mlir;

namespace {
struct MemRefPointerLikeModel
    : public omp::PointerLikeType::ExternalModel<MemRefPointerLikeModel,
                                                 MemRefType> {
  Type getElementType(Type pointer) const {
    return llvm::cast<MemRefType>(pointer).getElementType();
  }
};
} // namespace

void omp::registerBuiltinExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, BuiltinDialect *dialect) {
    MemRefType::attachInterface<MemRefPointerLikeModel>(*ctx);
    mlir::ModuleOp::attachInterface<
        mlir::omp::OffloadModuleDefaultModel<mlir::ModuleOp>>(*ctx);
  });
}
