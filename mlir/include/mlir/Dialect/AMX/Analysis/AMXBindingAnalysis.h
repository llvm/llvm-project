//===- AMXBindingAnalysis.h - Analyse AMX Binding Info --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains implementations of two analysis:
// 1. TileBindingAnalysis: verify and propagate pre-set tile register binding
// info for `vector`s used by tile operations.
// 2. TileScopeAnalysis: verify and find out proper tile configuration
// domination scopes for tile operations w.r.t correctness and performance.
// These analysis would be invoked by pass `--enable-amx-tile-binding` and used
// as indicator for tile binding info validation in pass
// `--convert-vector-to-llvm`
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AMX_ANALYSIS_AMXBINDINGANALYSIS_H_
#define MLIR_DIALECT_AMX_ANALYSIS_AMXBINDINGANALYSIS_H_

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace amx {

/// A class for analyzing (propagating) tile register binding for each tile
/// vector.
class TileBindingAnalysis {
private:
  bool isValidAnalysis;
  DenseMap<Value, int> bindings;

  // Ensure that tile operations are not wrapped by out-of-scope operations.
  bool isViableTileOps(Operation *root);

public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TileBindingAnalysis)
  explicit TileBindingAnalysis(Operation *);
  bool isValid() const { return isValidAnalysis; }
  int getBinding(Value val) const {
    auto iter = bindings.find(val);
    if (iter == bindings.end())
      return -1;
    return iter->second;
  }
  void setBinding(Value val, int index) { bindings[val] = index; }
};

// A class for analyzing tile configuration domination (a.k.a. tile scope).
class TileScopeAnalysis {
public:
  // A list of 2-dim {rows x colBytes} shapes representing tmm register shape,
  // the length should always be 8.
  struct PaletteInfo {
    bool overflow;
    SmallVector<std::pair<int, int>, 8> palette;
    PaletteInfo() {
      palette.resize(8, {0, 0});
      clear();
    }
    void clear();
    bool isEmpty(int idx) const {
      return palette[idx].first == 0 && palette[idx].second == 0;
    }
    void set(int idx, std::pair<int, int> shape) { palette[idx] = shape; }
    void merge(const PaletteInfo &rhs);
    bool isConflict(const PaletteInfo &rhs) const;
  };

private:
  typedef llvm::iterator_range<Block::iterator> BlockSeg;
  struct TileScope {
    // The BlockSeg here is inclusive (containing `end` Op).
    BlockSeg seg;
    PaletteInfo pi;
    TileScope() : seg(Block::iterator(), Block::iterator()) { clear(); }
    void clear() { pi.clear(); }
  };

  bool isValidAnalysis;
  // Storing parallel Ops that would break tile context & scope.
  DenseSet<Operation *> parallelOps;
  // Storing needed palette info for each concerned Op.
  DenseMap<Operation *, PaletteInfo> neededPalette;
  // Storing the usage scope for each concerned tile Op.
  // The BlockSeg here is inclusive (containing `end` Op).
  DenseMap<Operation *, BlockSeg> tileUsage;
  // Storing final tile scope results for injecting tilecfg/tilerelease.
  SmallVector<TileScope, 10> tileScopes;

  void addParallelOp(Operation *op) { parallelOps.insert(op); }
  bool isParallelOp(Operation *op) {
    return parallelOps.find(op) != parallelOps.end();
  }

  void setTileUsage(Operation *op, BlockSeg seg) {
    tileUsage.insert({op, std::move(seg)});
  }

  PaletteInfo collectBlockPalette(Block &block);
  PaletteInfo collectPalette(Operation *op);
  // Below two functions are the leaf functions of recursive collection, will
  // actually insert PaletteInfo into map storage.
  PaletteInfo collectPaletteForScf(Operation *op);
  PaletteInfo collectPaletteForTile(Operation *op);
  std::optional<PaletteInfo> getPalette(Operation *op);
  std::optional<PaletteInfo> getPalette(BlockSeg seg);

  void doTileScope(Block &block);
  // The input BlockSeg here is exclusive (not containing `end` Op).
  void doTileScope(BlockSeg seg);
  void doTileScope(Operation *op);

public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TileScopeAnalysis)
  explicit TileScopeAnalysis(Operation *);
  const SmallVector<TileScope, 10> &getTileScopes() const {
    return tileScopes;
  };
  bool isValid() const { return isValidAnalysis; }
};

} // namespace amx
} // namespace mlir

#endif // MLIR_DIALECT_AMX_ANALYSIS_AMXBINDINGANALYSIS_H_
