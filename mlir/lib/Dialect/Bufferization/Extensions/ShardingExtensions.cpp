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

/// Variadic helper function.
template <typename... OpTypes>
static void registerAll(MLIRContext *ctx) {
  (OpTypes::template attachInterface<
       shard::IndependentParallelIteratorDomainShardingInterface<OpTypes>>(
       *ctx),
   ...);
}

void mlir::bufferization::shard_ext::registerShardingInterfaceExternalModels(
    DialectRegistry &registry) {

  registry.addExtension(+[](MLIRContext *ctx,
                            bufferization::BufferizationDialect *dialect) {
    registerAll<bufferization::AllocTensorOp, bufferization::DeallocTensorOp,
                bufferization::MaterializeInDestinationOp>(ctx);
  });
}
