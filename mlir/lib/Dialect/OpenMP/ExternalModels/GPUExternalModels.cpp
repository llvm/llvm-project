//===- GPUExternalModels.cpp - Implementation of GPU external models ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the OpenMP external models for the GPU dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/OpenMP/ExternalModels.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"

using namespace mlir;

void omp::registerGPUExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, gpu::GPUDialect *dialect) {
    gpu::GPUModuleOp::attachInterface<
        omp::OffloadModuleDefaultModel<gpu::GPUModuleOp>>(*ctx);
  });
}
