//===- ConstantPropagationAnalysis.cpp - Constant propagation analysis ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "constant-propagation"

using namespace mlir;
using namespace mlir::dataflow;

//===----------------------------------------------------------------------===//
// ConstantValue
//===----------------------------------------------------------------------===//

void ConstantValue::print(raw_ostream &os) const {
  if (constant)
    return constant.print(os);
  os << "<NO VALUE>";
}

//===----------------------------------------------------------------------===//
// SparseConstantPropagation
//===----------------------------------------------------------------------===//

void SparseConstantPropagation::visitOperation(
    Operation *op, ArrayRef<const Lattice<ConstantValue> *> operands,
    ArrayRef<Lattice<ConstantValue> *> results) {
  LLVM_DEBUG(llvm::dbgs() << "SCP: Visiting operation: " << *op << "\n");

  // Don't try to simulate the results of a region operation as we can't
  // guarantee that folding will be out-of-place. We don't allow in-place
  // folds as the desire here is for simulated execution, and not general
  // folding.
  if (op->getNumRegions())
    return;

  SmallVector<Attribute, 8> constantOperands;
  constantOperands.reserve(op->getNumOperands());
  for (auto *operandLattice : operands)
    constantOperands.push_back(operandLattice->getValue().getConstantValue());

  // Save the original operands and attributes just in case the operation
  // folds in-place. The constant passed in may not correspond to the real
  // runtime value, so in-place updates are not allowed.
  SmallVector<Value, 8> originalOperands(op->getOperands());
  DictionaryAttr originalAttrs = op->getAttrDictionary();

  // Simulate the result of folding this operation to a constant. If folding
  // fails or was an in-place fold, mark the results as overdefined.
  SmallVector<OpFoldResult, 8> foldResults;
  foldResults.reserve(op->getNumResults());
  if (failed(op->fold(constantOperands, foldResults))) {
    markAllPessimisticFixpoint(results);
    return;
  }

  // If the folding was in-place, mark the results as overdefined and reset
  // the operation. We don't allow in-place folds as the desire here is for
  // simulated execution, and not general folding.
  if (foldResults.empty()) {
    op->setOperands(originalOperands);
    op->setAttrs(originalAttrs);
    markAllPessimisticFixpoint(results);
    return;
  }

  // Merge the fold results into the lattice for this operation.
  assert(foldResults.size() == op->getNumResults() && "invalid result size");
  for (const auto it : llvm::zip(results, foldResults)) {
    Lattice<ConstantValue> *lattice = std::get<0>(it);

    // Merge in the result of the fold, either a constant or a value.
    OpFoldResult foldResult = std::get<1>(it);
    if (Attribute attr = foldResult.dyn_cast<Attribute>()) {
      LLVM_DEBUG(llvm::dbgs() << "Folded to constant: " << attr << "\n");
      propagateIfChanged(lattice,
                         lattice->join(ConstantValue(attr, op->getDialect())));
    } else {
      LLVM_DEBUG(llvm::dbgs()
                 << "Folded to value: " << foldResult.get<Value>() << "\n");
      AbstractSparseDataFlowAnalysis::join(
          lattice, *getLatticeElement(foldResult.get<Value>()));
    }
  }
}
