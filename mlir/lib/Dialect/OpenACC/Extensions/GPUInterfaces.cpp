//===- GPUInterfaces.cpp - OpenACC external models ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenACC/Extensions/GPUInterfaces.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/DialectRegistry.h"

using namespace mlir;
using namespace mlir::acc;

namespace {
struct GPULaunchOffloadRegionModel
    : public acc::OffloadRegionOpInterface::ExternalModel<
          GPULaunchOffloadRegionModel, gpu::LaunchOp> {
  mlir::Region &getOffloadRegion(mlir::Operation *op) const {
    return cast<gpu::LaunchOp>(op).getBody();
  }
};
} // namespace

void mlir::acc::registerGPUInterfaces(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, OpenACCDialect *, gpu::GPUDialect *) {
        gpu::LaunchOp::attachInterface<GPULaunchOffloadRegionModel>(*ctx);
      });
}
