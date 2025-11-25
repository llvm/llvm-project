//===- Unit.cpp - Support for manipulating IR Unit ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Unit.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Region.h"
#include "llvm/Support/raw_ostream.h"
#include <iterator>

using namespace mlir;

static void printOp(llvm::raw_ostream &os, Operation *op,
                    OpPrintingFlags &flags) {
  if (!op) {
    os << "<Operation:nullptr>";
    return;
  }
  op->print(os, flags);
}

static void printRegion(llvm::raw_ostream &os, Region *region,
                        OpPrintingFlags &flags) {
  if (!region) {
    os << "<Region:nullptr>";
    return;
  }
  os << "Region #" << region->getRegionNumber() << " for op ";
  printOp(os, region->getParentOp(), flags);
}

static void printBlock(llvm::raw_ostream &os, Block *block,
                       OpPrintingFlags &flags) {
  Region *region = block->getParent();
  Block *entry = &region->front();
  int blockId = std::distance(entry->getIterator(), block->getIterator());
  os << "Block #" << blockId << " for ";
  bool shouldSkipRegions = flags.shouldSkipRegions();
  printRegion(os, region, flags.skipRegions());
  if (!shouldSkipRegions)
    block->print(os);
}

void mlir::IRUnit::print(llvm::raw_ostream &os, OpPrintingFlags flags) const {
  if (auto *op = llvm::dyn_cast_if_present<Operation *>(*this))
    return printOp(os, op, flags);
  if (auto *region = llvm::dyn_cast_if_present<Region *>(*this))
    return printRegion(os, region, flags);
  if (auto *block = llvm::dyn_cast_if_present<Block *>(*this))
    return printBlock(os, block, flags);
  llvm_unreachable("unknown IRUnit");
}

llvm::raw_ostream &mlir::operator<<(llvm::raw_ostream &os, const IRUnit &unit) {
  unit.print(os);
  return os;
}
