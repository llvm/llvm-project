//===- FuncExternalModels.cpp - Implementation of Func external models ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the OpenMP external models for the Func dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenMP/ExternalModels.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"

using namespace mlir;

void omp::registerFuncExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, func::FuncDialect *dialect) {
    // Attach default declare target interfaces to operations which can be
    // marked as declare target (Global Operations and Functions/Subroutines in
    // dialects that Fortran (or other languages that lower to MLIR) translates
    // too
    mlir::func::FuncOp::attachInterface<
        mlir::omp::DeclareTargetDefaultModel<mlir::func::FuncOp>>(*ctx);
    // Attach default early outlining interface to func ops.
    mlir::func::FuncOp::attachInterface<
        mlir::omp::EarlyOutliningDefaultModel<mlir::func::FuncOp>>(*ctx);
  });
}
