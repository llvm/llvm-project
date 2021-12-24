//===- NumberOfExecutions.h - Number of executions analysis -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains an analysis for computing how many times a block within a
// region is executed *each time* that region is entered. The analysis
// iterates over all associated regions that are attached to the given top-level
// operation.
//
// It is possible to query number of executions information on block level.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_NUMBER_OF_EXECUTIONS_H
#define MLIR_ANALYSIS_NUMBER_OF_EXECUTIONS_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"

namespace mlir {

class Block;
class BlockNumberOfExecutionsInfo;
class Operation;
class Region;

/// Represents an analysis for computing how many times a block or an operation
/// within a region is executed *each time* that region is entered. The analysis
/// iterates over all associated regions that are attached to the given
/// top-level operation.
///
/// This analysis assumes that all operations complete in a finite amount of
/// time (do not abort and do not go into the infinite loop).
class NumberOfExecutions {
public:
  /// Creates a new NumberOfExecutions analysis that computes how many times a
  /// block within a region is executed for all associated regions.
  explicit NumberOfExecutions(Operation *op);

  /// Returns the number of times operations `op` is executed *each time* the
  /// control flow enters the region `perEntryOfThisRegion`. Returns empty
  /// optional if this is not known statically.
  Optional<int64_t> getNumberOfExecutions(Operation *op,
                                          Region *perEntryOfThisRegion) const;

  /// Returns the number of times block `block` is executed *each time* the
  /// control flow enters the region `perEntryOfThisRegion`. Returns empty
  /// optional if this is not known statically.
  Optional<int64_t> getNumberOfExecutions(Block *block,
                                          Region *perEntryOfThisRegion) const;

  /// Dumps the number of block executions *each time* the control flow enters
  /// the region `perEntryOfThisRegion` to the given stream.
  void printBlockExecutions(raw_ostream &os,
                            Region *perEntryOfThisRegion) const;

  /// Dumps the number of operation executions *each time* the control flow
  /// enters the region `perEntryOfThisRegion` to the given stream.
  void printOperationExecutions(raw_ostream &os,
                                Region *perEntryOfThisRegion) const;

private:
  /// The operation this analysis was constructed from.
  Operation *operation;

  /// A mapping from blocks to number of executions information.
  DenseMap<Block *, BlockNumberOfExecutionsInfo> blockNumbersOfExecution;
};

/// Represents number of block executions information.
class BlockNumberOfExecutionsInfo {
public:
  BlockNumberOfExecutionsInfo(Block *block,
                              Optional<int64_t> numberOfRegionInvocations,
                              Optional<int64_t> numberOfBlockExecutions);

  /// Returns the number of times this block will be executed *each time* the
  /// parent operation is executed.
  Optional<int64_t> getNumberOfExecutions() const;

  /// Returns the number of times this block will be executed if the parent
  /// region is invoked `numberOfRegionInvocations` times. This can be different
  /// from the number of region invocations by the parent operation.
  Optional<int64_t>
  getNumberOfExecutions(int64_t numberOfRegionInvocations) const;

  Block *getBlock() const { return block; }

private:
  Block *block;

  /// Number of `block` parent region invocations *each time* parent operation
  /// is executed.
  Optional<int64_t> numberOfRegionInvocations;

  /// Number of `block` executions *each time* parent region is invoked.
  Optional<int64_t> numberOfBlockExecutions;
};

} // namespace mlir

#endif // MLIR_ANALYSIS_NUMBER_OF_EXECUTIONS_H
