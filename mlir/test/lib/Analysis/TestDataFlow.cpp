//===- TestDataFlow.cpp - Test data flow analysis system -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains test passes for defining and running a dataflow analysis.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlowAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace {
struct WasAnalyzed {
  WasAnalyzed(bool wasAnalyzed) : wasAnalyzed(wasAnalyzed) {}

  static WasAnalyzed join(const WasAnalyzed &a, const WasAnalyzed &b) {
    return a.wasAnalyzed && b.wasAnalyzed;
  }

  static WasAnalyzed getPessimisticValueState(MLIRContext *context) {
    return false;
  }

  static WasAnalyzed getPessimisticValueState(Value v) {
    return getPessimisticValueState(v.getContext());
  }

  bool operator==(const WasAnalyzed &other) const {
    return wasAnalyzed == other.wasAnalyzed;
  }

  bool wasAnalyzed;
};

struct TestAnalysis : public ForwardDataFlowAnalysis<WasAnalyzed> {
  using ForwardDataFlowAnalysis<WasAnalyzed>::ForwardDataFlowAnalysis;

  ChangeResult
  visitOperation(Operation *op,
                 ArrayRef<LatticeElement<WasAnalyzed> *> operands) final {
    ChangeResult ret = ChangeResult::NoChange;
    llvm::errs() << "Visiting : ";
    op->print(llvm::errs());
    llvm::errs() << "\n";

    WasAnalyzed result(true);
    for (auto &pair : llvm::enumerate(operands)) {
      LatticeElement<WasAnalyzed> *elem = pair.value();
      llvm::errs() << "Arg " << pair.index();
      if (!elem->isUninitialized()) {
        llvm::errs() << " : " << elem->getValue().wasAnalyzed << "\n";
        result = WasAnalyzed::join(result, elem->getValue());
      } else {
        llvm::errs() << " uninitialized\n";
      }
    }
    for (const auto &pair : llvm::enumerate(op->getResults())) {
      LatticeElement<WasAnalyzed> &lattice = getLatticeElement(pair.value());
      llvm::errs() << "Result " << pair.index() << " moved from ";
      if (lattice.isUninitialized())
        llvm::errs() << "uninitialized";
      else
        llvm::errs() << lattice.getValue().wasAnalyzed;
      ret |= lattice.join({result});
      llvm::errs() << " to " << lattice.getValue().wasAnalyzed << "\n";
    }
    return ret;
  }

  ChangeResult visitNonControlFlowArguments(
      Operation *op, const RegionSuccessor &successor,
      ArrayRef<LatticeElement<WasAnalyzed> *> operands) final {
    ChangeResult ret = ChangeResult::NoChange;
    llvm::errs() << "Visiting region branch op : ";
    op->print(llvm::errs());
    llvm::errs() << "\n";

    Region *region = successor.getSuccessor();
    Block *block = &region->front();
    Block::BlockArgListType arguments = block->getArguments();
    // Mark all arguments to blocks as analyzed unless they already have
    // an unanalyzed state.
    for (const auto &pair : llvm::enumerate(arguments)) {
      LatticeElement<WasAnalyzed> &lattice = getLatticeElement(pair.value());
      llvm::errs() << "Block argument " << pair.index() << " moved from ";
      if (lattice.isUninitialized())
        llvm::errs() << "uninitialized";
      else
        llvm::errs() << lattice.getValue().wasAnalyzed;
      ret |= lattice.join({true});
      llvm::errs() << " to " << lattice.getValue().wasAnalyzed << "\n";
    }
    return ret;
  }
};

struct TestDataFlowPass
    : public PassWrapper<TestDataFlowPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestDataFlowPass)

  StringRef getArgument() const final { return "test-data-flow"; }
  StringRef getDescription() const final {
    return "Print the actions taken during a dataflow analysis.";
  }
  void runOnOperation() override {
    llvm::errs() << "Testing : " << getOperation()->getAttr("test.name")
                 << "\n";
    TestAnalysis analysis(getOperation().getContext());
    analysis.run(getOperation());
  }
};
} // namespace

namespace mlir {
namespace test {
void registerTestDataFlowPass() { PassRegistration<TestDataFlowPass>(); }
} // namespace test
} // namespace mlir
