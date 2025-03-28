//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Linker/IRMover.h"
#include "mlir/Linker/LinkerInterface.h"

#define DEBUG_TYPE "mlir-linker-irmover"

using namespace mlir;
using namespace mlir::link;

LogicalResult IRMover::move(const Summary &summary) {
  worklist.reserve(summary.size());
  for (const auto &[_, pair] : summary) {
    worklist.push_back(pair);
  }

  while (!worklist.empty()) {
    ConflictPair pair = worklist.back();
    worklist.pop_back();

    if (mapping.contains(pair.src))
      continue;

    if (!remap(pair))
      return failure();
  }

  return success();
}

Operation *IRMover::remap(ConflictPair pair) {
  Operation *remaped = materialize(pair);
  if (!remaped)
    return nullptr;

  // TODO this should be part of liker interface
  if (!composite->isProperAncestor(remaped)) {
    // TODO investigate why we modify different module than the composite
    remaped->remove();
    composite.push_back(remaped);
  }

  mapping.map(pair.src, remaped);
  return remaped;
}

Operation *IRMover::materialize(ConflictPair pair) const {
  return cast<SymbolLinkerInterface>(pair.src->getDialect())
      ->materialize(pair, composite);
}
