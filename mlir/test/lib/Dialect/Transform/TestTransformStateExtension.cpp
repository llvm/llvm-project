//===- TestTransformStateExtension.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestTransformStateExtension.h"

using namespace mlir;

LogicalResult
test::TestTransformStateExtension::updateMapping(Operation *previous,
                                                 Operation *updated) {
  // Update value handles. The new ops should have at least as many results as
  // the replacement op. Fewer results are acceptable, if those results are not
  // mapped to any handle.
  for (auto r = updated->getNumResults(); r < previous->getNumResults(); ++r) {
    SmallVector<Value> handles;
    (void)getTransformState().getHandlesForPayloadValue(previous->getResult(r),
                                                        handles);
    if (!handles.empty())
      return emitError(previous->getLoc())
             << "cannot replace an op with another op producing fewer results "
                "while tracking handles";
  }

  for (auto [oldValue, newValue] :
       llvm::zip(previous->getResults(), updated->getResults()))
    if (failed(replacePayloadValue(oldValue, newValue)))
      return failure();

  // Update op handle.
  return replacePayloadOp(previous, updated);
}
