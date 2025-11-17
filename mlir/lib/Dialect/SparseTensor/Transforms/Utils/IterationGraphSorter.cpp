//===- IterationGraphSorter.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IterationGraphSorter.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::sparse_tensor;

namespace {

/// A helper class that visits an affine expression and tries to find
/// an AffineDimExpr to which the corresponding iterator from a GenericOp
/// matches the desired iterator type. If there is no matched iterator
/// type, the method returns the first DimExpr in the expression.
class AffineDimFinder : public AffineExprVisitor<AffineDimFinder> {
public:
  explicit AffineDimFinder(ArrayRef<utils::IteratorType> itTypes)
      : iterTypes(itTypes) {}

  /// Overrides the visit method from AffineExprVisitor.
  void visitDimExpr(AffineDimExpr expr) {
    if (pickedDim == nullptr || pickIterType == iterTypes[expr.getPosition()])
      pickedDim = expr;
  }

  /// Sets the desired iterator type that we want to pick.
  void setPickedIterType(utils::IteratorType iterType) {
    pickIterType = iterType;
  }

  /// Gets the desired AffineDimExpr.
  AffineDimExpr getDimExpr() const {
    return llvm::cast<AffineDimExpr>(pickedDim);
  }

  /// Walks the graph in post order to find dim expr.
  void walkPostOrder(AffineExpr expr) {
    pickedDim = nullptr;
    AffineExprVisitor<AffineDimFinder>::walkPostOrder(expr);
  }

private:
  /// The picked AffineDimExpr after visit.
  AffineExpr pickedDim;
  /// The iterator type that we want.
  utils::IteratorType pickIterType;
  /// The mapping between levels and iterator types.
  ArrayRef<utils::IteratorType> iterTypes;
};

/// Flattens an affine expression into a list of AffineDimExprs.
struct AffineDimCollector : public AffineExprVisitor<AffineDimCollector> {
  // Overrides method from AffineExprVisitor.
  void visitDimExpr(AffineDimExpr expr) { dims.push_back(expr); }
  SmallVector<AffineDimExpr> dims;
};

} // namespace

inline static bool includesAny(SortMask mask1, SortMask mask2) {
  return static_cast<unsigned>(mask1) & static_cast<unsigned>(mask2);
}

inline static bool includesDenseInput(SortMask mask) {
  return includesAny(mask, SortMask::kIncludeDenseInput);
}

inline static bool includesDenseOutput(SortMask mask) {
  return includesAny(mask, SortMask::kIncludeDenseOutput);
}

/// Returns a sparsity rank for loop ordering: lower values indicate
/// dimensions that should be placed in outer loops.
/// 0 = Dense, 1 = Compressed, 2 = Singleton, 3 = Other/Unknown.
static unsigned getLoopSparsityRank(unsigned loop, ArrayRef<Value> allTensors,
                                    ArrayRef<AffineMap> allMaps) {
  // Start with highest rank.
  unsigned minRank = 3;

  for (auto [tensor, map] : llvm::zip(allTensors, allMaps)) {
    // Check if this loop accesses this tensor.
    bool loopAccessesTensor = false;
    unsigned tensorDim = 0;
    for (AffineExpr expr : map.getResults()) {
      if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
        if (dimExpr.getPosition() == loop) {
          loopAccessesTensor = true;
          break;
        }
      }
      tensorDim++;
    }

    if (loopAccessesTensor) {
      const auto enc = getSparseTensorEncoding(tensor.getType());
      if (!enc) {
        // Dense tensor - lowest rank.
        return 0;
      } else {
        // Sparse tensor - check the level type for this dimension.
        auto lvlTypes = enc.getLvlTypes();
        if (tensorDim < lvlTypes.size()) {
          auto lvlType = lvlTypes[tensorDim];
          if (isDenseLT(lvlType)) {
            return 0; // Dense level.
          } else if (isCompressedLT(lvlType)) {
            minRank = std::min(minRank, 1u); // Compressed level.
          } else if (isSingletonLT(lvlType)) {
            minRank = std::min(minRank, 2u); // Singleton level.
          }
        }
      }
    }
  }

  return minRank;
}

AffineMap IterationGraphSorter::topoSort() {
  // The sorted result will put the first Reduction iterator to the
  // latest possible position.
  std::vector<unsigned> redIt; // reduce iterator with 0 degree
  std::vector<unsigned> parIt; // parallel iterator with 0 degree
  const unsigned numLoops = getNumLoops();
  for (unsigned i = 0; i < numLoops; i++) {
    if (inDegree[i] == 0) {
      if (iterTypes[i] == utils::IteratorType::reduction)
        redIt.push_back(i);
      else
        parIt.push_back(i);
    }
  }

  SmallVector<unsigned> loopOrder;
  while (!redIt.empty() || !parIt.empty()) {
    // We always prefer a parallel loop over a reduction loop because putting
    // a reduction loop early might make the loop sequence inadmissible.
    auto &it = !parIt.empty() ? parIt : redIt;

    // Select loop based on strategy.
    unsigned src;
    switch (strategy) {
    case sparse_tensor::LoopOrderingStrategy::kDefault:
      src = it.back();
      break;
    case sparse_tensor::LoopOrderingStrategy::kDenseOuter: {
      // Prefer dense, then compressed, then singleton dimensions outermost.
      // Create combined tensor and map lists for analysis.
      SmallVector<Value> allTensors = ins;
      allTensors.push_back(out);
      SmallVector<AffineMap> allMaps = loop2InsLvl;
      allMaps.push_back(loop2OutLvl);

      // Find loop with minimum (lowest) sparsity rank.
      unsigned minLoop = it[0];
      unsigned minRank = getLoopSparsityRank(minLoop, allTensors, allMaps);

      for (auto candidateLoop : it) {
        unsigned rank = getLoopSparsityRank(candidateLoop, allTensors, allMaps);
        if (rank < minRank || (rank == minRank && candidateLoop < minLoop)) {
          minLoop = candidateLoop;
          minRank = rank;
        }
      }
      src = minLoop;
      break;
    }
    }

    loopOrder.push_back(src);
    // Remove the selected loop from the worklist.
    it.erase(std::find(it.begin(), it.end(), src));
    // Update in-degree, and push 0-degree node into worklist.
    for (unsigned dst = 0; dst < numLoops; dst++) {
      if (itGraph[src][dst] && --inDegree[dst] == 0) {
        if (iterTypes[dst] == utils::IteratorType::reduction)
          redIt.push_back(dst);
        else
          parIt.push_back(dst);
      }
    }
  }

  // Return the topological sort on success.
  if (loopOrder.size() == numLoops)
    return AffineMap::getPermutationMap(loopOrder, out.getContext());

  // Cycle detected.
  return AffineMap();
}

IterationGraphSorter IterationGraphSorter::fromGenericOp(
    linalg::GenericOp genericOp, sparse_tensor::LoopOrderingStrategy strategy) {
  // Must be a demapped sparse kernel.
  assert(!hasAnyNonIdentityOperandsOrResults(genericOp) &&
         hasAnySparseOperandOrResult(genericOp) &&
         genericOp.getNumDpsInits() == 1);

  SmallVector<AffineMap> loopMap = genericOp.getIndexingMapsArray();
  SmallVector<Value> ins = genericOp.getDpsInputs();

  AffineMap outMap = loopMap.back();
  loopMap.pop_back();

  Value out = genericOp.getDpsInitOperand(0)->get();
  SmallVector<utils::IteratorType> iterTypes =
      genericOp.getIteratorTypesArray();

  return IterationGraphSorter(std::move(ins), std::move(loopMap), out, outMap,
                              std::move(iterTypes), strategy);
}

IterationGraphSorter::IterationGraphSorter(
    SmallVector<Value> &&insArg, SmallVector<AffineMap> &&loop2InsLvlArg,
    Value out, AffineMap loop2OutLvl,
    SmallVector<utils::IteratorType> &&iterTypesArg,
    sparse_tensor::LoopOrderingStrategy strategy)
    : ins(std::move(insArg)), loop2InsLvl(std::move(loop2InsLvlArg)), out(out),
      loop2OutLvl(loop2OutLvl), iterTypes(std::move(iterTypesArg)),
      strategy(strategy) {
  // One map per tensor.
  assert(loop2InsLvl.size() == ins.size());
  // All the affine maps have the same number of dimensions (loops).
  assert(llvm::all_equal(llvm::map_range(
      loop2InsLvl, [](AffineMap m) { return m.getNumDims(); })));
  // The number of results of the map should match the rank of the tensor.
  assert(llvm::all_of(llvm::zip(loop2InsLvl, ins), [](auto mvPair) {
    auto [m, v] = mvPair;

    // For ranked types the rank must match.
    // Simply return true for UnrankedTensorType
    if (auto shapedType = llvm::dyn_cast<ShapedType>(v.getType())) {
      return !shapedType.hasRank() ||
             (m.getNumResults() == shapedType.getRank());
    }
    // Non-shaped (scalar) types behave like rank-0.
    return m.getNumResults() == 0;
  }));

  itGraph.resize(getNumLoops(), std::vector<bool>(getNumLoops(), false));
  inDegree.resize(getNumLoops());
}

AffineMap IterationGraphSorter::sort(SortMask mask, Value ignored) {
  // Reset the adjacency matrix that represents the iteration graph.
  for (auto &row : itGraph)
    llvm::fill(row, false);

  // Reset in-degree.
  llvm::fill(inDegree, 0);

  // Add the constraints for the loop to level map.
  for (auto [in, map] : llvm::zip(ins, loop2InsLvl)) {
    // Get map and encoding.
    const auto enc = getSparseTensorEncoding(in.getType());
    // Skip dense inputs when not requested.
    if ((!enc && !includesDenseInput(mask)) || in == ignored)
      continue;
    addConstraints(in, map);
  }

  // Add the constraints for the output map.
  const auto enc = getSparseTensorEncoding(out.getType());
  if ((enc || includesDenseOutput(mask)) && out != ignored)
    addConstraints(out, loop2OutLvl);

  // Return the topological sort (empty for cyclic).
  return topoSort();
}

void IterationGraphSorter::addConstraints(Value t, AffineMap loop2LvlMap) {
  auto addIterOrdering = [this](unsigned f, unsigned t) {
    if (!itGraph[f][t] && f != t) {
      itGraph[f][t] = true;
      inDegree[t]++;
    }
  };

  // Set up a reduction finder.
  AffineDimFinder finder(iterTypes);
  finder.setPickedIterType(utils::IteratorType::reduction);

  // To compute iteration graph for tensor[d0 + d1 + d3, d4 + d5 + d6],
  // we require there exist d_x \in {d0, d1, d3} and d_y \in {d4, d5, d6},
  // and d_x > d_y && {d0, d1, d3} - d_x > {d4, d5, d6} - d_y
  const Level lvlRank = loop2LvlMap.getNumResults();
  for (Level lvl = 1; lvl < lvlRank; lvl++) {
    const AffineExpr fa = loop2LvlMap.getResult(lvl - 1);
    const AffineExpr ta = loop2LvlMap.getResult(lvl);

    if (llvm::isa<AffineDimExpr>(fa) || llvm::isa<AffineDimExpr>(ta)) {
      // Special case when at least one loop2LvlExp is a simple AffineDimExpr
      // (say, d0) and we require d0 > {d1, d2, ...} or {d1, d2, ...} > d0
      AffineDimCollector fCollector;
      fCollector.walkPostOrder(fa);
      AffineDimCollector tCollector;
      tCollector.walkPostOrder(ta);

      for (auto fd : fCollector.dims) {
        for (auto td : tCollector.dims) {
          const unsigned f = fd.getPosition();
          const unsigned t = td.getPosition();
          addIterOrdering(f, t);
        }
      }
      continue;
    }

    // When both loop2LvlExpr is compound, we pick an abitrary reduction loop
    // from lhs and rhs and use them as d_x and d_y.
    finder.walkPostOrder(fa);
    const AffineDimExpr fexp = finder.getDimExpr();
    const unsigned fldx = fexp.getPosition();

    finder.walkPostOrder(ta);
    const AffineDimExpr texp = finder.getDimExpr();
    const unsigned tldx = texp.getPosition();

    // d_x > d_y
    addIterOrdering(fldx, tldx);

    AffineDimCollector fCollector;
    fCollector.walkPostOrder(fa);
    AffineDimCollector tCollector;
    tCollector.walkPostOrder(ta);

    // Make sure dx and dy is the last.
    for (auto fd : fCollector.dims) {
      const unsigned f = fd.getPosition();
      addIterOrdering(f, fldx);
    }
    for (auto td : tCollector.dims) {
      const unsigned t = td.getPosition();
      addIterOrdering(t, tldx);
    }
    // {d0, d1, d3} - d_x > {d4, d5, d6} - d_y
    // This is to ensure that the affine expressions are reduced in sparse
    // tensor level ordering.
    for (auto fd : fCollector.dims) {
      const unsigned f = fd.getPosition();
      if (f == fldx) // skip d_x
        continue;
      for (auto td : tCollector.dims) {
        const unsigned t = td.getPosition();
        if (t == tldx) // skip d_y
          continue;
        addIterOrdering(f, t);
      }
    }
  }
}
