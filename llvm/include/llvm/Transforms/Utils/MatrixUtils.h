//===- MatrixUtils.h - Utilities to lower matrix intrinsics -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for generating tiled loops for matrix operations.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_MATRIXUTILS_H
#define LLVM_TRANSFORMS_UTILS_MATRIXUTILS_H

#include "llvm/ADT/StringRef.h"

namespace llvm {
class DomTreeUpdater;
class BasicBlock;
class Value;
class Loop;
class LoopInfo;
class IRBuilderBase;

/// A helper struct to create IR loop nests for tiling in IR of the following
/// form:
///   for CurrentColumn = 0..NumColumns
///     for CurrentRow = 0..NumRows
///       for CurrentInner = 0..NumInner
struct TileInfo {
  /// Number of rows of the matrix.
  unsigned NumRows;

  /// Number of columns of the matrix.
  unsigned NumColumns;

  /// Number of columns of the first matrix of a multiply /
  /// number of rows of the second matrix of a multiply.
  unsigned NumInner;

  /// Number of rows/columns in a tile.
  unsigned TileSize = -1;

  /// Start row of the current tile to compute.
  Value *CurrentRow;

  /// Start column of the current tile to compute.
  Value *CurrentCol;

  /// Current tile offset during the tile computation.
  Value *CurrentK;

  /// Header of the outermost loop iterating from 0..NumColumns.
  BasicBlock *ColumnLoopHeader = nullptr;

  /// Header of the second loop iterating from 0..NumRows.
  BasicBlock *RowLoopHeader = nullptr;
  /// Latch of the second loop iterating from 0..NumRows.
  BasicBlock *RowLoopLatch = nullptr;
  /// Header of the innermost loop iterating from 0..NumInner.
  BasicBlock *InnerLoopHeader = nullptr;
  /// Latch of the innermost loop iterating from 0..NumInner.
  BasicBlock *InnerLoopLatch = nullptr;

  TileInfo(unsigned NumRows, unsigned NumColumns, unsigned NumInner,
           unsigned TileSize)
      : NumRows(NumRows), NumColumns(NumColumns), NumInner(NumInner),
        TileSize(TileSize) {}

  /// Creates an IR loop nests for tiling of the form below. Returns the block
  /// for the inner loop body and sets {Column,Row,Inner}LoopHeader/Latch
  /// fields.
  ///
  /// for CurrentColumn = 0..NumColumns
  ///   for CurrentRow = 0..NumRows
  ///     for CurrentInner = 0..NumInner
  BasicBlock *CreateTiledLoops(BasicBlock *Start, BasicBlock *End,
                               IRBuilderBase &B, DomTreeUpdater &DTU,
                               LoopInfo &LI);

private:
  /// Creates a new loop with header, body and latch blocks that iterates from
  /// [0, Bound). Updates \p Preheader to branch to the new header and uses \p
  /// Exit as exit block.  Adds the new loop blocks to \L and applies dominator
  /// tree updates to \p DTU.
  static BasicBlock *CreateLoop(BasicBlock *Preheader, BasicBlock *Exit,
                                Value *Bound, Value *Step, StringRef Name,
                                IRBuilderBase &B, DomTreeUpdater &DTU, Loop *L,
                                LoopInfo &LI);
};
} // namespace llvm

#endif
