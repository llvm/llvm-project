//===-- MetadataAnalysis.h - dataflow tutorial ------------------*- c++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is contains the dataflow tutorial's classes related to metadata.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TUTORIAL_METADATA_ANALYSIS_H_
#define MLIR_TUTORIAL_METADATA_ANALYSIS_H_

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace llvm;

namespace mlir {
/// The value of our lattice represents the inner structure of a DictionaryAttr,
/// for the `metadata`.
struct MetadataLatticeValue {
  MetadataLatticeValue() = default;
  /// Compute a lattice value from the provided dictionary.
  MetadataLatticeValue(DictionaryAttr attr) {
    for (NamedAttribute pair : attr) {
      metadata.insert(
          std::pair<StringAttr, Attribute>(pair.getName(), pair.getValue()));
    }
  }

  /// This method conservatively joins the information held by `lhs` and `rhs`
  /// into a new value. This method is required to be monotonic. `monotonicity`
  /// is implied by the satisfaction of the following axioms:
  ///   * idempotence:   join(x,x) == x
  ///   * commutativity: join(x,y) == join(y,x)
  ///   * associativity: join(x,join(y,z)) == join(join(x,y),z)
  ///
  /// When the above axioms are satisfied, we achieve `monotonicity`:
  ///   * monotonicity: join(x, join(x,y)) == join(x,y)
  static MetadataLatticeValue join(const MetadataLatticeValue &lhs,
                                   const MetadataLatticeValue &rhs);

  /// A simple comparator that checks to see if this value is equal to the one
  /// provided.
  bool operator==(const MetadataLatticeValue &rhs) const;

  /// Print data in metadata.
  void print(llvm::raw_ostream &os) const;

  /// Our value represents the combined metadata, which is originally a
  /// DictionaryAttr, so we use a map.
  DenseMap<StringAttr, Attribute> metadata;
};

namespace dataflow {
class MetadataLatticeValueLattice : public Lattice<MetadataLatticeValue> {
public:
  using Lattice::Lattice;
};

class MetadataAnalysis
    : public SparseForwardDataFlowAnalysis<MetadataLatticeValueLattice> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;
  LogicalResult
  visitOperation(Operation *op,
                 ArrayRef<const MetadataLatticeValueLattice *> operands,
                 ArrayRef<MetadataLatticeValueLattice *> results) override;
  void setToEntryState(MetadataLatticeValueLattice *lattice) override;
};

} // namespace dataflow
} // namespace mlir

#endif // MLIR_TUTORIAL_METADATA_ANALYSIS_H_
