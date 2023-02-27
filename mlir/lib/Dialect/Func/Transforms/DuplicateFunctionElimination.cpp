//===- DuplicateFunctionElimination.cpp - Duplicate function elimination --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"

namespace mlir {
namespace {

#define GEN_PASS_DEF_DUPLICATEFUNCTIONELIMINATIONPASS
#include "mlir/Dialect/Func/Transforms/Passes.h.inc"

// Define a notion of function equivalence that allows for reuse. Ignore the
// symbol name for this purpose.
struct DuplicateFuncOpEquivalenceInfo
    : public llvm::DenseMapInfo<func::FuncOp> {

  static unsigned getHashValue(const func::FuncOp cFunc) {
    if (!cFunc) {
      return DenseMapInfo<func::FuncOp>::getHashValue(cFunc);
    }

    // Aggregate attributes, ignoring the symbol name.
    llvm::hash_code hash = {};
    func::FuncOp func = const_cast<func::FuncOp &>(cFunc);
    StringAttr symNameAttrName = func.getSymNameAttrName();
    for (NamedAttribute namedAttr : cFunc->getAttrs()) {
      StringAttr attrName = namedAttr.getName();
      if (attrName == symNameAttrName)
        continue;
      hash = llvm::hash_combine(hash, namedAttr);
    }

    // Also hash the func body.
    func.getBody().walk([&](Operation *op) {
      hash = llvm::hash_combine(
          hash, OperationEquivalence::computeHash(
                    op, /*hashOperands=*/OperationEquivalence::ignoreHashValue,
                    /*hashResults=*/OperationEquivalence::ignoreHashValue,
                    OperationEquivalence::IgnoreLocations));
    });

    return hash;
  }

  static bool isEqual(const func::FuncOp cLhs, const func::FuncOp cRhs) {
    if (cLhs == cRhs) {
      return true;
    }
    if (cLhs == getTombstoneKey() || cLhs == getEmptyKey() ||
        cRhs == getTombstoneKey() || cRhs == getEmptyKey()) {
      return false;
    }

    // Check attributes equivalence, ignoring the symbol name.
    if (cLhs->getAttrDictionary().size() != cRhs->getAttrDictionary().size()) {
      return false;
    }
    func::FuncOp lhs = const_cast<func::FuncOp &>(cLhs);
    StringAttr symNameAttrName = lhs.getSymNameAttrName();
    for (NamedAttribute namedAttr : cLhs->getAttrs()) {
      StringAttr attrName = namedAttr.getName();
      if (attrName == symNameAttrName) {
        continue;
      }
      if (namedAttr.getValue() != cRhs->getAttr(attrName)) {
        return false;
      }
    }

    // Compare inner workings.
    func::FuncOp rhs = const_cast<func::FuncOp &>(cRhs);
    return OperationEquivalence::isRegionEquivalentTo(
        &lhs.getBody(), &rhs.getBody(), OperationEquivalence::IgnoreLocations);
  }
};

struct DuplicateFunctionEliminationPass
    : public impl::DuplicateFunctionEliminationPassBase<
          DuplicateFunctionEliminationPass> {

  using DuplicateFunctionEliminationPassBase<
      DuplicateFunctionEliminationPass>::DuplicateFunctionEliminationPassBase;

  void runOnOperation() override {
    auto module = getOperation();

    // Find unique representant per equivalent func ops.
    DenseSet<func::FuncOp, DuplicateFuncOpEquivalenceInfo> uniqueFuncOps;
    DenseMap<StringAttr, func::FuncOp> getRepresentant;
    DenseSet<func::FuncOp> toBeErased;
    module.walk([&](func::FuncOp f) {
      auto [repr, inserted] = uniqueFuncOps.insert(f);
      getRepresentant[f.getSymNameAttr()] = *repr;
      if (!inserted) {
        toBeErased.insert(f);
      }
    });

    // Update call ops to call unique func op representants.
    module.walk([&](func::CallOp callOp) {
      func::FuncOp callee = getRepresentant[callOp.getCalleeAttr().getAttr()];
      callOp.setCallee(callee.getSymName());
    });

    // Erase redundant func ops.
    for (auto it : toBeErased) {
      it.erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::func::createDuplicateFunctionEliminationPass() {
  return std::make_unique<DuplicateFunctionEliminationPass>();
}

} // namespace mlir
