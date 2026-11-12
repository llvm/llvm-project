//===- PassDetail.cpp - Async Pass class details ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/RegionUtils.h"

using namespace mlir;

void mlir::async::cloneConstantsIntoTheRegion(Region &region) {
  OpBuilder builder(&region);
  cloneConstantsIntoTheRegion(region, builder);
}

void mlir::async::cloneConstantsIntoTheRegion(Region &region,
                                              OpBuilder &builder) {
  // Values implicitly captured by the region.
  llvm::SetVector<Value> captures;
  getUsedValuesDefinedAbove(region, region, captures);

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&region.front());

  // Clone ConstantLike operations into the region.
  for (Value capture : captures) {
    Operation *op = capture.getDefiningOp();
    if (!op || !op->hasTrait<OpTrait::ConstantLike>())
      continue;

    Operation *cloned = builder.clone(*op);

    for (auto tuple : llvm::zip(op->getResults(), cloned->getResults())) {
      Value orig = std::get<0>(tuple);
      Value replacement = std::get<1>(tuple);
      replaceAllUsesInRegionWith(orig, replacement, region);
    }
  }
}
