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
class Type;
class LLVMContext;
class DataLayout;
class Constant;

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

  /// Number of rows in a tile.
  unsigned TileNumRows = -1;
  /// Inner dimension of the tile.
  unsigned TileNumInner = -1;
  /// Number of columns in a tile.
  unsigned TileNumColumns = -1;

  Type *EltType;

  /// Properties of a single loop used when generating the tiled loop nest.
  struct MatrixLoop {
    /// The index updated on every iteration.
    Value *Index = nullptr;
    /// The header and latch of the loop.
    BasicBlock *Header = nullptr;
    BasicBlock *Latch = nullptr;
  };

  /// See the comment before LowerMatrixIntrinsics::createTiledLoops for a view
  /// of all the loops and their operations.
  ///
  /// Main tiling loop nest.
  /// The loop iterating on the rows.
  MatrixLoop RowLoop;
  /// The loop iterating on the columns.
  MatrixLoop ColumnLoop;
  /// The loop iterating on k (inner dimension).
  MatrixLoop KLoop;

  /// Remainder columns loop nest: computes the remainder columns of the
  /// current row.
  /// This loop nest is optional, and only created if there are remainder
  /// columns that don't fit in the main loop nest.
  ///
  /// The loop iterating on k (inner dimension) for the remainder columns.
  MatrixLoop RemainderColumnsKLoop;

  /// Remainder rows loop nest: computes the remainder rows of the matrix.
  /// This loop nest is optional, and only created if there are remainder
  /// rows that don't fit in the main loop nest.
  ///
  /// The loop iterating on the columns for the remainder rows.
  MatrixLoop RemainderRowsColumnLoop;
  /// The loop iterating on k (inner dimension) for the remainder rows.
  MatrixLoop RemainderRowsKLoop;

  /// Remainder columns of the remainder rows loop nest: computes the remainder
  /// columns of the remainder rows.
  /// This loop nest is optional, and only created if there are remainder
  /// columns and remainder rows that don't fit in the main loop nest.
  ///
  /// The loop iterating on k (inner dimension) for the remainder columns of the
  /// remainder rows.
  MatrixLoop RemainderRowsRemainderColumnsKLoop;

  TileInfo(unsigned NumRows, unsigned NumColumns, unsigned NumInner,
           unsigned TileSize, Type *EltTy)
      : NumRows(NumRows), NumColumns(NumColumns), NumInner(NumInner),
        TileNumRows(TileSize), TileNumInner(TileSize), TileNumColumns(TileSize),
        EltType(EltTy) {}

  TileInfo(unsigned NumRows, unsigned NumColumns, unsigned NumInner,
           unsigned TileNumRows, unsigned TileNumInner, unsigned TileNumColumns,
           Type *EltTy)
      : NumRows(NumRows), NumColumns(NumColumns), NumInner(NumInner),
        TileNumRows(std::min(TileNumRows, NumRows)),
        TileNumInner(std::min(TileNumInner, NumInner)),
        TileNumColumns(std::min(TileNumColumns, NumColumns)), EltType(EltTy) {}

  /// Creates an IR loop nests for tiling of the form below. Returns the block
  /// for the inner loop body and sets {Column,Row,K}LoopHeader/Latch fields.
  ///
  /// for Column = 0..NumColumns; += TileNumColumns
  ///   for Row = 0..NumRows; += TileNumRows
  ///     for K = 0..NumInner; += TileNumInner
  LLVM_ABI BasicBlock *CreateTiledLoops(BasicBlock *Start, BasicBlock *End,
                                        IRBuilderBase &B, DomTreeUpdater &DTU,
                                        LoopInfo &LI);

  /// Creates an IR loop nest for tiling with remainder handling.
  /// Sets {Remainder}{Column,Row,K}LoopHeader/Latch fields with all the
  /// resulting basic blocks.
  ///
  /// See comment in LowerMatrixIntrinsics::createTiledLoops for the structure
  /// of the loop nest.
  void CreateTiledLoopsWithRemainder(BasicBlock *Start, BasicBlock *End,
                                     IRBuilderBase &B, DomTreeUpdater &DTU,
                                     LoopInfo &LI);

private:
  /// Creates a new loop with header, body and latch blocks that iterates from
  /// [Start, Bound). Updates \p Preheader to branch to the new header and uses
  /// \p Exit as exit block.  Adds the new loop blocks to \L and applies
  /// dominator tree updates to \p DTU.
  static BasicBlock *CreateLoop(BasicBlock *Preheader, BasicBlock *Exit,
                                Value *Start, Value *Bound, Value *Step,
                                StringRef Name, IRBuilderBase &B,
                                DomTreeUpdater &DTU, Loop *L, LoopInfo &LI,
                                unsigned Successor = 0);

  /// Same as above, with Start == 0.
  static BasicBlock *CreateLoop(BasicBlock *Preheader, BasicBlock *Exit,
                                Value *Bound, Value *Step, StringRef Name,
                                IRBuilderBase &B, DomTreeUpdater &DTU, Loop *L,
                                LoopInfo &LI, unsigned Successor = 0);
};

Type *indexType(LLVMContext &Ctx, const DataLayout &DL);

Constant *asIndex(IRBuilderBase &Builder, unsigned Val);

} // namespace llvm

#endif
