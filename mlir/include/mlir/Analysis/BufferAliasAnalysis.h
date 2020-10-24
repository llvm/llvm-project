//===- BufferAliasAnalysis.h - Buffer alias analysis for MLIR ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_BUFFERALIASANALYSIS_H
#define MLIR_ANALYSIS_BUFFERALIASANALYSIS_H

#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir {

/// A straight-forward alias analysis which ensures that all aliases of all
/// values will be determined. This is a requirement for the BufferPlacement
/// class since you need to determine safe positions to place alloc and
/// deallocs.
class BufferAliasAnalysis {
public:
  using ValueSetT = SmallPtrSet<Value, 16>;
  using ValueMapT = llvm::DenseMap<Value, ValueSetT>;

public:
  /// Constructs a new alias analysis using the op provided.
  BufferAliasAnalysis(Operation *op);

  /// Find all immediate aliases this value could potentially have.
  ValueMapT::const_iterator find(Value value) const {
    return aliases.find(value);
  }

  /// Returns the begin iterator to iterate over all aliases.
  ValueMapT::const_iterator begin() const { return aliases.begin(); }

  /// Returns the end iterator that can be used in combination with find.
  ValueMapT::const_iterator end() const { return aliases.end(); }

  /// Find all immediate and indirect aliases this value could potentially
  /// have. Note that the resulting set will also contain the value provided as
  /// it is an alias of itself.
  ValueSetT resolve(Value value) const;

  /// Removes the given values from all alias sets.
  void remove(const SmallPtrSetImpl<Value> &aliasValues);

private:
  /// This function constructs a mapping from values to its immediate aliases.
  void build(Operation *op);

  /// Maps values to all immediate aliases this value can have.
  ValueMapT aliases;
};

} // namespace mlir

#endif // MLIR_ANALYSIS_BUFFERALIASANALYSIS_H
