//===- ShardingExtensions.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/Extensions/ShardingExtensions.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Shard/Interfaces/ShardingInterfaceImpl.h"
#include "mlir/IR/DialectRegistry.h"

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::shard;

void mlir::bufferization::shard_ext::registerShardingInterfaceExternalModels(
    DialectRegistry &registry) {

  registry.addExtension(+[](MLIRContext *ctx, BufferizationDialect *dialect) {
    AllocTensorOp::attachInterface<
        IndependentParallelIteratorDomainShardingInterface<AllocTensorOp>>(
        *ctx);
    DeallocTensorOp::attachInterface<
        IndependentParallelIteratorDomainShardingInterface<DeallocTensorOp>>(
        *ctx);
    MaterializeInDestinationOp::attachInterface<
        ElementwiseShardingInterface<MaterializeInDestinationOp>>(*ctx);
  });
}
