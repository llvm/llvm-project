//===-- MetadataAnalysis.cpp - dataflow tutorial ----------------*- c++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is contains the implementations of the methods in the
// metadata-related classes in the dataflow tutorial.
//
//===----------------------------------------------------------------------===//

#include "MetadataAnalysis.h"

using namespace mlir;
using namespace dataflow;

namespace mlir {

/// This method conservatively joins the information held by `lhs` and `rhs`
/// into a new value. This method is required to be monotonic. `monotonicity`
/// is implied by the satisfaction of the following axioms:
///   * idempotence:   join(x,x) == x
///   * commutativity: join(x,y) == join(y,x)
///   * associativity: join(x,join(y,z)) == join(join(x,y),z)
///
/// When the above axioms are satisfied, we achieve `monotonicity`:
///   * monotonicity: join(x, join(x,y)) == join(x,y)
MetadataLatticeValue
MetadataLatticeValue::join(const MetadataLatticeValue &lhs,
                           const MetadataLatticeValue &rhs) {
  // To join `lhs` and `rhs` we will define a simple policy, which is that we
  // directly insert the metadata of rhs into the metadata of lhs.If lhs and rhs
  // have overlapping attributes, keep the attribute value in lhs unchanged.
  MetadataLatticeValue result;
  for (auto &&lhsIt : lhs.metadata) {
    result.metadata.insert(
        std::pair<StringRef, Attribute>(lhsIt.getKey(), lhsIt.getValue()));
  }

  for (auto &&rhsIt : rhs.metadata) {
    result.metadata.insert(
        std::pair<StringRef, Attribute>(rhsIt.getKey(), rhsIt.getValue()));
  }
  return result;
}

/// A simple comparator that checks to see if this value is equal to the one
/// provided.
bool MetadataLatticeValue::operator==(const MetadataLatticeValue &rhs) const {
  if (metadata.size() != rhs.metadata.size())
    return false;

  // Check that `rhs` contains the same metadata.
  for (auto &&it : metadata) {
    auto rhsIt = rhs.metadata.find(it.getKey());
    if (rhsIt == rhs.metadata.end() || it.second != rhsIt->second)
      return false;
  }
  return true;
}

void MetadataLatticeValue::print(llvm::raw_ostream &os) const {
  os << "{";
  for (auto &&iter : metadata) {
    os << iter.getKey() << ": " << iter.getValue() << ", ";
  }
  os << "\b\b}\n";
}

namespace dataflow {
LogicalResult MetadataAnalysis::visitOperation(
    Operation *op, ArrayRef<const MetadataLatticeValueLattice *> operands,
    ArrayRef<MetadataLatticeValueLattice *> results) {
  DictionaryAttr metadata = op->getAttrOfType<DictionaryAttr>("metadata");
  // If we have no metadata for this operation and the operands is empty, we
  // will conservatively mark all of the results as having reached a pessimistic
  // fixpoint.
  if (!metadata && operands.empty()) {
    setAllToEntryStates(results);
    return success();
  }

  MetadataLatticeValue latticeValue;
  if (metadata)
    latticeValue = MetadataLatticeValue(metadata);

  // Otherwise, we will compute a lattice value for the metadata and join it
  // into the current lattice element for all of our results.`results` stores
  // the lattices corresponding to the results of op, We use a loop to traverse
  // them.
  for (int i = 0, e = results.size(); i < e; ++i) {

    // `isChanged` records whether the result has been changed.
    ChangeResult isChanged = ChangeResult::NoChange;

    // Op's metadata is joined result's lattice.
    isChanged |= results[i]->join(latticeValue);

    // All lattice of operands of op are joined to the lattice of result.
    for (int j = 0, m = operands.size(); j < m; ++j) {
      isChanged |= results[i]->join(*operands[j]);
    }
    propagateIfChanged(results[i], isChanged);
  }
  return success();
}

/// At an entry point, We leave its function body empty because no metadata can
/// be joined to Lattice.
void MetadataAnalysis::setToEntryState(MetadataLatticeValueLattice *lattice) {}
} // namespace dataflow
} // namespace mlir
