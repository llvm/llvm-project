//===- DumpNotifications.h - Test notifications -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TEST_LIB_IR_DUMPNOTIFICATIONS_H
#define MLIR_TEST_LIB_IR_DUMPNOTIFICATIONS_H

#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
struct DumpNotifications : public RewriterBase::Listener {
  void notifyBlockInserted(Block *block, Region *previous,
                           Region::iterator previousIt) override {
    llvm::outs() << "notifyBlockInserted";
    if (block->getParentOp()) {
      llvm::outs() << " into " << block->getParentOp()->getName() << ": ";
    } else {
      llvm::outs() << " into unknown op: ";
    }
    if (previous == nullptr) {
      llvm::outs() << "was unlinked\n";
    } else {
      llvm::outs() << "was linked\n";
    }
  }
  void notifyOperationInserted(Operation *op,
                               OpBuilder::InsertPoint previous) override {
    llvm::outs() << "notifyOperationInserted: " << op->getName();
    if (!previous.isSet()) {
      llvm::outs() << ", was unlinked\n";
    } else {
      if (!previous.getPoint().getNodePtr()) {
        llvm::outs() << ", was linked, exact position unknown\n";
      } else if (previous.getPoint() == previous.getBlock()->end()) {
        llvm::outs() << ", was last in block\n";
      } else {
        llvm::outs() << ", previous = " << previous.getPoint()->getName()
                     << "\n";
      }
    }
  }
  void notifyBlockErased(Block *block) override {
    llvm::outs() << "notifyBlockErased\n";
  }
  void notifyOperationErased(Operation *op) override {
    llvm::outs() << "notifyOperationErased: " << op->getName() << "\n";
  }
  void notifyOperationModified(Operation *op) override {
    llvm::outs() << "notifyOperationModified: " << op->getName() << "\n";
  }
  void notifyOperationReplaced(Operation *op, ValueRange values) override {
    llvm::outs() << "notifyOperationReplaced: " << op->getName() << "\n";
  }
};
} // namespace mlir

#endif // MLIR_TEST_LIB_IR_DUMPNOTIFICATIONS_H
