//===- TileReducerAnalyses.h - non-mutating analyses ------------*- C++ -*-===//
//
// Milestone 5: bounds, layout, and reduction classification. Analyses do
// not mutate IR. The planning pass reads them and writes attributes.
//
//===----------------------------------------------------------------------===//

#ifndef TILE_REDUCER_TILEREDUCERANALYSES_H
#define TILE_REDUCER_TILEREDUCERANALYSES_H

#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace tr {

enum class ReductionKind { Row, Column, Full, Unknown };

struct ReductionInfo {
  ReductionKind kind = ReductionKind::Unknown;
  int64_t axis = -1;
  bool loadReduceCandidate = false;
};

class BoundsAnalysis {
public:
  BoundsAnalysis(Operation *op);
  /// Tile extent along `dim` of `v`'s type, or -1 if unknown.
  int64_t getTileExtent(Value v, int64_t dim) const;
  bool isKnownMultipleOf(int64_t value, int64_t factor) const {
    return value > 0 && factor > 0 && (value % factor) == 0;
  }
};

class LayoutAnalysis {
public:
  LayoutAnalysis(Operation *op);
  /// Assumed source layout. TileReducer buffers are row-major until a
  /// later layout attribute says otherwise.
  StringRef getBufferLayout(Value buffer) const { return "row_major"; }
};

class ReductionAnalysis {
public:
  ReductionAnalysis(Operation *op);
  const ReductionInfo *get(Operation *op) const;
  const llvm::DenseMap<Operation *, ReductionInfo> &all() const {
    return infos;
  }

private:
  llvm::DenseMap<Operation *, ReductionInfo> infos;
};

} // namespace tr
} // namespace mlir

#endif // TILE_REDUCER_TILEREDUCERANALYSES_H
