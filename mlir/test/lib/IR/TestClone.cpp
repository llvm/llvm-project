//===- TestClone.cpp - Pass to test operation cloning  --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

/// This is a test pass which clones the body of a function. Specifically
/// this pass replaces f(x) to instead return f(f(x)) in which the cloned body
/// takes the result of the first operation return as an input.
struct ClonePass
    : public PassWrapper<ClonePass, InterfacePass<FunctionOpInterface>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ClonePass)

  StringRef getArgument() const final { return "test-clone"; }
  StringRef getDescription() const final { return "Test clone of op"; }
  void runOnOperation() override {
    FunctionOpInterface op = getOperation();

    // Limit testing to ops with only one region.
    if (op->getNumRegions() != 1)
      return;

    Region &region = op->getRegion(0);
    if (!region.hasOneBlock())
      return;

    Block &regionEntry = region.front();
    Operation *terminator = regionEntry.getTerminator();

    // Only handle functions whose returns match the inputs.
    if (terminator->getNumOperands() != regionEntry.getNumArguments())
      return;

    IRMapping map;
    for (auto tup :
         llvm::zip(terminator->getOperands(), regionEntry.getArguments())) {
      if (std::get<0>(tup).getType() != std::get<1>(tup).getType())
        return;
      map.map(std::get<1>(tup), std::get<0>(tup));
    }

    OpBuilder builder(op->getContext());
    builder.setInsertionPointToEnd(&regionEntry);
    SmallVector<Operation *> toClone;
    for (Operation &inst : regionEntry)
      toClone.push_back(&inst);
    for (Operation *inst : toClone)
      builder.clone(*inst, map);
    terminator->erase();
  }
};
} // namespace

namespace mlir {
void registerCloneTestPasses() { PassRegistration<ClonePass>(); }
} // namespace mlir
