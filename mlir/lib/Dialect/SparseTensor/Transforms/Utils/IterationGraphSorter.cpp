//===- IterationGraphSorter.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>

#include "IterationGraphSorter.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SparseTensor/IR/Enums.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensorType.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/Support/CommandLine.h"

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

AffineMap IterationGraphSorter::topoSort() {    
  // Run memory analysis for strategies that can benefit from it
  switch (getLoopOrderingStrategy()) {
    case LoopOrderingStrategy::kMemoryAware:
    case LoopOrderingStrategy::kSequentialFirst:
    case LoopOrderingStrategy::kAdaptive:
      analyzeMemoryPatterns();
      break;
    case LoopOrderingStrategy::kDefault:
    case LoopOrderingStrategy::kDenseOuter:
    case LoopOrderingStrategy::kSparseOuter:
    case LoopOrderingStrategy::kParallelFirst:
      break;
  }

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
    unsigned src;

    switch (getLoopOrderingStrategy()) {
      case LoopOrderingStrategy::kMemoryAware:
        src = selectBestCandidateByMemory(it);
        it.erase(std::find(it.begin(), it.end(), src));
        break;
      case LoopOrderingStrategy::kDenseOuter:
        src = selectBestCandidateByDensity(it, true); // dense first
        it.erase(std::find(it.begin(), it.end(), src));
        break;
      case LoopOrderingStrategy::kSparseOuter:
        src = selectBestCandidateByDensity(it, false); // sparse first
        it.erase(std::find(it.begin(), it.end(), src));
        break;
      case LoopOrderingStrategy::kSequentialFirst:
        src = selectBestCandidateBySequentiality(it);
        it.erase(std::find(it.begin(), it.end(), src));
        break;
      case LoopOrderingStrategy::kParallelFirst:
        src = selectBestCandidateByParallelism(it);
        it.erase(std::find(it.begin(), it.end(), src));
        break;
      case LoopOrderingStrategy::kAdaptive:
        src = selectBestCandidateByAdaptive(it);
        it.erase(std::find(it.begin(), it.end(), src));
        break;
      case LoopOrderingStrategy::kDefault:
        // Default strategy: pick the last loop (original behavior)
        src = it.back();
        it.pop_back();
        break;
    }

    loopOrder.push_back(src);

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

IterationGraphSorter
IterationGraphSorter::fromGenericOp(linalg::GenericOp genericOp) {
  // Original behavior - no strategy parameter, uses default behavior
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

  // Use original constructor with explicit default strategy parameter
  return IterationGraphSorter(std::move(ins), std::move(loopMap), out, outMap,
                              std::move(iterTypes), LoopOrderingStrategy::kDefault);
}

IterationGraphSorter
IterationGraphSorter::fromGenericOp(linalg::GenericOp genericOp,
                                     LoopOrderingStrategy strategy) {
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
    SmallVector<Value> &&ins, SmallVector<AffineMap> &&loop2InsLvl, Value out,
    AffineMap loop2OutLvl, SmallVector<utils::IteratorType> &&iterTypes,
    LoopOrderingStrategy strategy)
    : loopOrderingStrategy(strategy), ins(std::move(ins)),
      loop2InsLvl(std::move(loop2InsLvl)), out(out), loop2OutLvl(loop2OutLvl),
      iterTypes(std::move(iterTypes)) {
  // One map per tensor.
  assert(loop2InsLvl.size() == ins.size());
  // All the affine maps have the same number of dimensions (loops).
  assert(llvm::all_equal(llvm::map_range(
      loop2InsLvl, [](AffineMap m) { return m.getNumDims(); })));
  // The number of results of the map should match the rank of the tensor.
  assert(llvm::all_of(llvm::zip(loop2InsLvl, ins), [](auto mvPair) {
    auto [m, v] = mvPair;
    return m.getNumResults() == cast<ShapedType>(v.getType()).getRank();
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

    // When both loop2LvlExpr is compound, we pick an arbitrary reduction loop
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

// get encoding info (storage format, level types, etc)
SparseTensorEncodingAttr getEncodingInfo(Value tensor) {
  auto tensorType = dyn_cast<RankedTensorType>(tensor.getType());
  if (!tensorType)
    return nullptr; // Not a ranked tensor type
  return getSparseTensorEncoding(tensorType);
}

void IterationGraphSorter::analyzeMemoryPatterns() {
  const unsigned numLoops = getNumLoops();
  loopMemoryAnalysis.resize(numLoops);

  // Initialize memory analysis for each loop
  for (unsigned loop = 0; loop < numLoops; ++loop) {
    auto &memInfo = loopMemoryAnalysis[loop];
    memInfo.totalTensorAccesses = 0;
    memInfo.sparseAccessCost = 0;
    memInfo.compressedSequentialAccesses.clear();
    memInfo.randomSparseAccesses.clear();
    memInfo.unitStrideAccesses.clear();
    memInfo.avgStrideComplexity = 0.0;
    memInfo.spatialLocalityScore = 0.0;
    memInfo.temporalReuseScore = 0.0;
    memInfo.accessPatternRand = 0.0;
  }

  // Analyze input tensors
  for (auto [tensorIdx, tensor] : llvm::enumerate(ins)) {
    const AffineMap &map = loop2InsLvl[tensorIdx];
    analyzeMapForMemoryPatterns(map, tensorIdx, tensor, false);
  }

  // Analyze output tensor
  analyzeMapForMemoryPatterns(loop2OutLvl, ins.size(), out, true);

  // Compute final scores without architecture assumptions
  for (unsigned loop = 0; loop < numLoops; ++loop) {
    computeArchitectureScore(loop);
  }
}

IterationGraphSorter::SparseAccessPattern
IterationGraphSorter::analyzeSparseAccessPattern(
    AffineMap map, unsigned dim, unsigned loopIdx,
    SparseTensorEncodingAttr encoding, unsigned tensorIdx) {

  SparseAccessPattern pattern;

  // Get the level types for this encoding
  auto lvlTypes = encoding.getLvlTypes();
  if (dim >= lvlTypes.size()) {
    pattern.type = IterationGraphSorter::SparseAccessType::kRandomSparse;
    pattern.expectedSparsity = 0.01;
    pattern.memoryIndirections = 3;
    pattern.hasGoodLocality = false;
    return pattern;
  }

  LevelType levelType = lvlTypes[dim];
  AffineExpr dimExpr = map.getResult(dim);

  // Analyze the affine expression for this dimension
  if (auto dimExprCast = dyn_cast<AffineDimExpr>(dimExpr)) {
    // Simple case: dimension expression is just a loop variable
    if (dimExprCast.getPosition() == loopIdx) {

      if (isCompressedLT(levelType)) {
        // Sequential access through compressed dimension
        pattern.type = SparseAccessType::kCompressedSequential;
        pattern.expectedSparsity = 1.0;
        pattern.memoryIndirections = 1;
        pattern.hasGoodLocality = true;
      } else if (isSingletonLT(levelType)) {
        // Sequential scan through singleton dimension
        pattern.type = SparseAccessType::kSingletonScan;
        pattern.expectedSparsity = 0.1;
        pattern.memoryIndirections = 2;
        pattern.hasGoodLocality = false;
      } else {
        // Dense level
        pattern.type = SparseAccessType::kDenseSubtensor;
        pattern.expectedSparsity = 1.0;
        pattern.memoryIndirections = 1;
        pattern.hasGoodLocality = true;
      }
    } else {
      // Loop variable doesn't match this dimension
      pattern.type = IterationGraphSorter::SparseAccessType::kRandomSparse;
      pattern.expectedSparsity = 0.01;
      pattern.memoryIndirections = 3;
      pattern.hasGoodLocality = false;
    }
  } else {
    // Complex affine expression - generally bad for sparse access
    pattern.type = IterationGraphSorter::SparseAccessType::kRandomSparse;
    pattern.expectedSparsity = 0.01;
    pattern.memoryIndirections = 3;
    pattern.hasGoodLocality = false;
  }

  return pattern;
}

void IterationGraphSorter::analyzeMapForMemoryPatterns(AffineMap map,
                                                       unsigned tensorIdx,
                                                       Value tensor,
                                                       bool isOutput) {

  auto encoding = getEncodingInfo(tensor);
  bool isSparse = static_cast<bool>(encoding);

  const unsigned tensorRank = map.getNumResults();

  for (unsigned dim = 0; dim < tensorRank; ++dim) {
    AffineExpr dimExpr = map.getResult(dim);

    AffineDimCollector collector;
    collector.walkPostOrder(dimExpr);

    for (auto dimExprNode : collector.dims) {
      unsigned loopIdx = dimExprNode.getPosition();
      auto &loopInfo = loopMemoryAnalysis[loopIdx];
      loopInfo.totalTensorAccesses++;

      if (isSparse) {
        // Sparse tensor analysis
        SparseAccessPattern pattern =
            analyzeSparseAccessPattern(map, dim, loopIdx, encoding, tensorIdx);

        switch (pattern.type) {
        case SparseAccessType::kCompressedSequential:
          loopInfo.compressedSequentialAccesses.push_back(tensorIdx);
          break;
        case SparseAccessType::kSingletonScan:
          loopInfo.singletonScanAccesses.push_back(tensorIdx);
          break;
        case SparseAccessType::kRandomSparse:
          loopInfo.randomSparseAccesses.push_back(tensorIdx);
          break;
        case SparseAccessType::kDenseSubtensor:
          loopInfo.unitStrideAccesses.push_back(tensorIdx);
          break;
        }
      } else {
        // Dense tensor analysis (your original code)
        unsigned strideComplexity =
            computeStrideComplexity(map.getResult(dim), loopIdx);
        if (strideComplexity == 1) {
          loopInfo.unitStrideAccesses.push_back(tensorIdx);
        } else if (strideComplexity == 2) {
          loopInfo.linearStrideAccesses.push_back(tensorIdx);
        } else {
          loopInfo.complexAccesses.push_back(tensorIdx);
        }
      }
    }
  }
}

unsigned IterationGraphSorter::computeStrideComplexity(AffineExpr expr,
                                                       unsigned targetLoop) {
  if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
    return dimExpr.getPosition() == targetLoop ? 1 : 3;
  }

  AffineDimCollector collector;
  collector.walkPostOrder(expr);

  unsigned targetLoopCount = 0;
  unsigned otherLoopCount = 0;

  for (auto dim : collector.dims) {
    if (dim.getPosition() == targetLoop) {
      targetLoopCount++;
    } else {
      otherLoopCount++;
    }
  }

  if (targetLoopCount == 1 && otherLoopCount == 0) {
    return 1; // Unit stride
  } else if (targetLoopCount == 1 && otherLoopCount <= 1) {
    return 2; // Linear stride
  } else {
    return 3; // Complex
  }
}

void IterationGraphSorter::computeArchitectureScore(unsigned loopIdx) {
  auto &memInfo = loopMemoryAnalysis[loopIdx];

  if (memInfo.totalTensorAccesses == 0) {
    memInfo.avgStrideComplexity = 0.0;
    return;
  }

  // Compute sparse access cost
  double sparseAccessScore = 0.0;
  unsigned totalSparseAccesses = memInfo.compressedSequentialAccesses.size() +
                                 memInfo.singletonScanAccesses.size() +
                                 memInfo.randomSparseAccesses.size();

  if (totalSparseAccesses > 0) {
    // Weighted scoring based on access pattern efficiency
    double compressedRatio =
        (double)memInfo.compressedSequentialAccesses.size() /
        totalSparseAccesses;
    double singletonRatio =
        (double)memInfo.singletonScanAccesses.size() / totalSparseAccesses;
    double randomRatio =
        (double)memInfo.randomSparseAccesses.size() / totalSparseAccesses;

    double unitStrideRatio =
        memInfo.totalTensorAccesses > 0
            ? (double)(memInfo.unitStrideAccesses.size() +
                       memInfo.compressedSequentialAccesses.size()) /
                  memInfo.totalTensorAccesses
            : 0.0;
    memInfo.spatialLocalityScore = unitStrideRatio;

    // Temporal reuse: reward loops that access multiple tensors (more reuse
    // potential)
    memInfo.temporalReuseScore =
        std::min(1.0, memInfo.totalTensorAccesses / 3.0);

    // Apply locality bonuses to final score
    memInfo.avgStrideComplexity *= (1.0 + memInfo.spatialLocalityScore * 0.1);
    memInfo.avgStrideComplexity *= (1.0 + memInfo.temporalReuseScore * 0.05);

    // Scoring: compressed access = 1.0, singleton = 0.4, random = 0.1
    sparseAccessScore =
        compressedRatio * 1.0 + singletonRatio * 0.4 + randomRatio * 0.1;
  }

  // Compute dense access score
  double denseAccessScore = 0.0;
  unsigned totalDenseAccesses = memInfo.unitStrideAccesses.size() +
                                memInfo.linearStrideAccesses.size() +
                                memInfo.complexAccesses.size();

  if (totalDenseAccesses > 0) {
    double unitStrideRatio =
        (double)memInfo.unitStrideAccesses.size() / totalDenseAccesses;
    double linearStrideRatio =
        (double)memInfo.linearStrideAccesses.size() / totalDenseAccesses;
    double complexAccessRatio =
        (double)memInfo.complexAccesses.size() / totalDenseAccesses;

    denseAccessScore = unitStrideRatio * 1.0 + linearStrideRatio * 0.7 +
                       complexAccessRatio * 0.2;
  }

  // Combine sparse and dense scores
  double totalAccesses = totalSparseAccesses + totalDenseAccesses;
  if (totalAccesses > 0) {
    double sparseWeight = (double)totalSparseAccesses / totalAccesses;
    double denseWeight = (double)totalDenseAccesses / totalAccesses;

    memInfo.avgStrideComplexity =
        sparseWeight * sparseAccessScore + denseWeight * denseAccessScore;
  } else {
    memInfo.avgStrideComplexity = 0.0;
  }

  // Apply existing bonuses (reduction preference, fan-out penalty)
  if (iterTypes[loopIdx] == utils::IteratorType::reduction) {
    memInfo.avgStrideComplexity *= 1.15;
  }

  // Fan-out penalty
  unsigned fanOut = 0;
  for (unsigned j = 0; j < getNumLoops(); ++j) {
    if (itGraph[loopIdx][j])
      fanOut++;
  }

  double fanOutRatio = (double)fanOut / getNumLoops();
  if (fanOutRatio > 0.5) {
    memInfo.avgStrideComplexity *= (1.0 - fanOutRatio * 0.2);
  }
}

double IterationGraphSorter::computePortableScore(unsigned loopIdx) {
  const auto &memInfo = loopMemoryAnalysis[loopIdx];

  double memoryScore = memInfo.avgStrideComplexity;

  // Bonus for loops that enable sparse optimizations
  if (memInfo.compressedSequentialAccesses.size() > 0) {
    memoryScore *=
        1.2; // Prefer loops that access compressed dimensions sequentially
  }

  // Penalty for loops that cause random sparse access
  if (memInfo.randomSparseAccesses.size() >
      memInfo.compressedSequentialAccesses.size()) {
    memoryScore *= 0.8; // Penalize loops that cause poor sparse access patterns
  }

  // Existing logic
  double parallelScore =
      (iterTypes[loopIdx] == utils::IteratorType::parallel) ? 1.1 : 1.0;

  unsigned outDegree = 0;
  unsigned inDegree = 0;
  for (unsigned j = 0; j < getNumLoops(); ++j) {
    if (itGraph[loopIdx][j])
      outDegree++;
    if (itGraph[j][loopIdx])
      inDegree++;
  }

  double graphScore = 1.0 / (1.0 + outDegree * 0.1) + inDegree * 0.05;

  return memoryScore * parallelScore * graphScore;
}

unsigned IterationGraphSorter::selectBestCandidateByMemory(
    const std::vector<unsigned> &candidates) {
  
  if (candidates.empty()) return 0;

  if (candidates.size() == 1)
    return candidates[0];

  unsigned bestCandidate = candidates[0];
  double bestScore = computePortableScore(bestCandidate);

  for (unsigned i = 1; i < candidates.size(); ++i) {
    unsigned candidate = candidates[i];
    double score = computePortableScore(candidate);

    if (score > bestScore) {
      bestScore = score;
      
    bestCandidate = candidate;
    }
  }

  return bestCandidate;
}

// Dense-outer heuristic: prefer dense dimensions first
unsigned IterationGraphSorter::selectBestCandidateByDensity(
    const std::vector<unsigned> &candidates, bool denseFirst) {
  unsigned bestCandidate = candidates[0];
  int bestScore = denseFirst ? -1000 : 1000; // Start with worst possible score
  
  for (unsigned candidate : candidates) {
    int score = 0;
    
    // Count dense vs sparse accesses for this loop
    for (unsigned tensorIdx = 0; tensorIdx < ins.size(); tensorIdx++) {
      Value tensor = ins[tensorIdx];
      if (getSparseTensorEncoding(tensor.getType())) {
        AffineMap dimToLvlMap = loop2InsLvl[tensorIdx];
        if (candidate < dimToLvlMap.getNumResults()) {
          auto lvlExpr = dimToLvlMap.getResult(candidate);
          if (auto dimExpr = dyn_cast<AffineDimExpr>(lvlExpr)) {
            unsigned lvl = dimExpr.getPosition();
            auto enc = getSparseTensorEncoding(tensor.getType());
            if (enc && lvl < enc.getLvlTypes().size()) {
              auto lvlType = enc.getLvlTypes()[lvl];
              if (isDenseLT(lvlType)) {
                score += 10; // Dense is good
              } else {
                score -= 5;  // Sparse is bad
              }
            }
          }
        }
      } else {
        score += 5; // Dense tensor access is always good
      }
    }
    
    
    bool isBetter = denseFirst ? (score > bestScore) : (score < bestScore);
    if (isBetter) {
      bestScore = score;
      
    bestCandidate = candidate;
    }
  }
  
  return bestCandidate;
}

// Sequential-first heuristic: prefer unit stride accesses
unsigned IterationGraphSorter::selectBestCandidateBySequentiality(
    const std::vector<unsigned> &candidates) {
  unsigned bestCandidate = candidates[0];
  int bestScore = -1000;
  
  for (unsigned candidate : candidates) {
    int score = 0;
    
    // Simple heuristic: prefer lower-numbered loops (often more sequential)
    // In practice, this would need more sophisticated stride analysis
    for (unsigned tensorIdx = 0; tensorIdx < ins.size(); tensorIdx++) {
      AffineMap map = loop2InsLvl[tensorIdx];
      if (candidate < map.getNumResults()) {
        auto expr = map.getResult(candidate);
        // Simple approximation: direct dimension access is better
        if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
          if (dimExpr.getPosition() == candidate) {
            score += 10; // Direct access is good
          }
        } else {
          score -= 5; // Complex expression is worse
        }
      }
    }
        
    if (score > bestScore) {
      bestScore = score;
      
    bestCandidate = candidate;
    }
  }
  
  return bestCandidate;
}

// Parallel-first heuristic: parallel loops first, then by density
unsigned IterationGraphSorter::selectBestCandidateByParallelism(
    const std::vector<unsigned> &candidates) {
  
  unsigned bestCandidate = candidates[0];
  int bestScore = -1000;
  
  for (unsigned candidate : candidates) {
    int score = 0;
    
    // Strongly prefer parallel loops
    if (candidate < iterTypes.size() && iterTypes[candidate] == utils::IteratorType::parallel) {
      score += 100; // Big bonus for parallel
    } else {
      score -= 50;  // Penalty for reduction
    }
    
    // Secondary criteria: prefer dense accesses
    for (unsigned tensorIdx = 0; tensorIdx < ins.size(); tensorIdx++) {
      Value tensor = ins[tensorIdx];
      if (getSparseTensorEncoding(tensor.getType())) {
        AffineMap dimToLvlMap = loop2InsLvl[tensorIdx];
        if (candidate < dimToLvlMap.getNumResults()) {
          auto lvlExpr = dimToLvlMap.getResult(candidate);
          if (auto dimExpr = dyn_cast<AffineDimExpr>(lvlExpr)) {
            unsigned lvl = dimExpr.getPosition();
            auto enc = getSparseTensorEncoding(tensor.getType());
            if (enc && lvl < enc.getLvlTypes().size()) {
              auto lvlType = enc.getLvlTypes()[lvl];
              if (isDenseLT(lvlType)) {
                score += 5;
              }
            }
          }
        }
      }
    }
        
    if (score > bestScore) {
      bestScore = score;

    bestCandidate = candidate;
    }
  }
  
  return bestCandidate;
}

// Adaptive heuristic: intelligently choose the best strategy based on kernel characteristics
unsigned IterationGraphSorter::selectBestCandidateByAdaptive(
    const std::vector<unsigned> &candidates) {
  
  LoopOrderingStrategy adaptiveStrategy = selectAdaptiveStrategy();
  
  // Delegate to the selected strategy
  switch (adaptiveStrategy) {
    case LoopOrderingStrategy::kParallelFirst:
      return selectBestCandidateByParallelism(candidates);
    case LoopOrderingStrategy::kMemoryAware:
      return selectBestCandidateByMemory(candidates);
    case LoopOrderingStrategy::kSequentialFirst:
      return selectBestCandidateBySequentiality(candidates);
    case LoopOrderingStrategy::kDenseOuter:
      return selectBestCandidateByDensity(candidates, true);
    case LoopOrderingStrategy::kSparseOuter:
      return selectBestCandidateByDensity(candidates, false);
    case LoopOrderingStrategy::kDefault:
      // For default, use the first candidate (matches default behavior)
      return candidates[0];
    default:
      // Fallback to memory_aware
      return selectBestCandidateByMemory(candidates);
  }
}

// Determine the best strategy based on kernel characteristics
LoopOrderingStrategy IterationGraphSorter::selectAdaptiveStrategy() const {  
  
  // Get kernel characteristics
  bool hasHighParallelism = hasHighParallelismPotential();
  unsigned numLoops = getNumLoops();
  uint64_t totalElements = getTotalElementsHeuristic();
  bool hasGoodLocality = hasGoodMemoryLocalityPotential();
  
  // Calculate derived metrics for principled decisions
  unsigned parallelLoops = 0;
  unsigned reductionLoops = 0;
  for (auto iterType : iterTypes) {
    if (iterType == utils::IteratorType::parallel) parallelLoops++;
    if (iterType == utils::IteratorType::reduction) reductionLoops++;
  }
  
  double parallelRatio = numLoops > 0 ? (double)parallelLoops / numLoops : 0.0;
  double reductionRatio = numLoops > 0 ? (double)reductionLoops / numLoops : 0.0;
  bool isSimplePattern = (parallelLoops + reductionLoops == numLoops) && numLoops <= 4;
    
  // Ultra-deep loops with high parallelism --> parallel-first
  if (numLoops >= 10 && hasHighParallelism) {
    return LoopOrderingStrategy::kParallelFirst;
  }
  
  // Reduction-heavy workloads --> sequential-first
  if (reductionRatio >= 0.5 && numLoops >= 4) {
    return LoopOrderingStrategy::kSequentialFirst;
  }
  
  // High parallelism with large scale --> parallel-first
  if (parallelRatio >= 0.6 && totalElements >= 100000) {
    return LoopOrderingStrategy::kParallelFirst;
  }

  // Simple patterns with good locality --> memory-aware or dense-outer
  if (isSimplePattern && hasGoodLocality) {
    if (totalElements <= 50000) {
      return LoopOrderingStrategy::kMemoryAware;
    } else {
      return LoopOrderingStrategy::kDenseOuter;
    }
  }

  // Medium complexity with good locality --> memory-aware
  if (hasGoodLocality && numLoops >= 3 && numLoops <= 8) {
    return LoopOrderingStrategy::kMemoryAware;
  }
  
  // Fall back based on dominant pattern type
  if (parallelRatio > reductionRatio && parallelRatio >= 0.3) {
    return LoopOrderingStrategy::kParallelFirst;
  }
  
  // Default: Safe fallback to memory-aware
  return LoopOrderingStrategy::kMemoryAware;
}

// Essential helper functions for principle-based adaptive strategy
bool IterationGraphSorter::hasGoodMemoryLocalityPotential() const {
  // Principle: Operations with regular access patterns benefit from memory-aware analysis
  // This includes: sparse matvec (CSR), dense operations, unit-stride accesses
  
  // Check for sparse tensors with compressed formats (good locality)
  for (const auto& in : ins) {
    if (auto tensorType = dyn_cast<RankedTensorType>(in.getType())) {
      if (auto encoding = dyn_cast_or_null<SparseTensorEncodingAttr>(tensorType.getEncoding())) {
        auto dimLevelTypes = encoding.getLvlTypes();
        for (auto dimType : dimLevelTypes) {
          if (dimType.isa<LevelFormat::Compressed>()) {
            return true; // Compressed sparse has good locality
          }
        }
      }
    }
  }
  
  // Check for simple affine maps (good for cache analysis)  
  auto hasSimpleMap = [](const AffineMap &map) -> bool {
    for (unsigned i = 0; i < map.getNumResults(); ++i) {
      AffineExpr expr = map.getResult(i);
      if (!llvm::isa<AffineDimExpr>(expr)) {
        return false; // Complex expression
      }
    }
    return true; // All simple dimension accesses
  };
  
  // If most maps are simple, memory analysis will be effective
  int simpleMapCount = 0;
  int totalMaps = loop2InsLvl.size() + 1; // inputs + output
  
  for (const AffineMap &map : loop2InsLvl) {
    if (hasSimpleMap(map)) simpleMapCount++;
  }
  if (hasSimpleMap(loop2OutLvl)) simpleMapCount++;
  
  return (double)simpleMapCount / totalMaps >= 0.5; // Majority are simple
}

bool IterationGraphSorter::hasStrongSequentialDependencies() const {
  // Principle: Operations with many inter-loop dependencies benefit from sequential ordering
  
  // Count dependencies in the iteration graph
  unsigned totalDependencies = 0;
  unsigned numLoops = getNumLoops();
  
  for (unsigned i = 0; i < numLoops; ++i) {
    for (unsigned j = 0; j < numLoops; ++j) {
      if (i != j && itGraph[i][j]) {
        totalDependencies++;
      }
    }
  }
  
  unsigned maxPossibleDeps = numLoops * (numLoops - 1);
  return maxPossibleDeps > 0 && (double)totalDependencies / maxPossibleDeps > 0.5;
}

bool IterationGraphSorter::hasHighParallelismPotential() const {
  unsigned parallelLoops = 0;
  for (auto iterType : iterTypes) {
    if (iterType == utils::IteratorType::parallel) {
      parallelLoops++;
    }
  }
  
  unsigned totalLoops = iterTypes.size();
  double parallelRatio = totalLoops > 0 ? (double)parallelLoops / totalLoops : 0.0;
  
  return parallelRatio > 0.6;
}

double IterationGraphSorter::computeAverageSparsity() const {
  unsigned sparseTensorCount = 0;
  for (auto [tensorIdx, tensor] : llvm::enumerate(ins)) {
    if (auto tensorType = llvm::dyn_cast<RankedTensorType>(tensor.getType())) {
      if (auto encoding = llvm::dyn_cast_or_null<SparseTensorEncodingAttr>(tensorType.getEncoding())) {
        sparseTensorCount++;
      }
    }
  }
  
  if (sparseTensorCount == 0) return 1.0; // Dense
  return 0.1; // 10% sparsity estimate for sparse tensors
}

bool IterationGraphSorter::hasComplexMemoryPattern() const {
  // Check for non-trivial affine expressions in access patterns
  auto checkComplexMap = [](const AffineMap &map) -> bool {
    for (unsigned i = 0; i < map.getNumResults(); ++i) {
      AffineExpr expr = map.getResult(i);
      // Complex if not just a simple dimension expression
      if (!llvm::isa<AffineDimExpr>(expr)) {
        return true;
      }
    }
    return false;
  };
  
  // Check input maps
  for (const AffineMap &map : loop2InsLvl) {
    if (checkComplexMap(map)) return true;
  }
  
  // Check output map
  return checkComplexMap(loop2OutLvl);
}

bool IterationGraphSorter::hasMemoryIntensiveScanPattern() const {
  // Heuristic: operations with mostly reduction dimensions suggest scans
  unsigned reductionCount = 0;
  for (auto iterType : iterTypes) {
    if (iterType == utils::IteratorType::reduction) {
      reductionCount++;
    }
  }
  
  // Memory scans typically have many reduction dimensions
  return reductionCount >= 2 && reductionCount == iterTypes.size();
}

bool IterationGraphSorter::hasTensorContractionPattern() const {
  // 3D or higher dimensional operations with mixed parallel/reduction
  if (iterTypes.size() < 3) return false;
  
  bool hasParallel = false, hasReduction = false;
  for (auto iterType : iterTypes) {
    if (iterType == utils::IteratorType::parallel) hasParallel = true;
    if (iterType == utils::IteratorType::reduction) hasReduction = true;
  }
  
  // Tensor contractions have both parallel and reduction dimensions
  return hasParallel && hasReduction && iterTypes.size() >= 3;
}

bool IterationGraphSorter::hasMatrixVectorPattern() const {  
  unsigned totalLoops = iterTypes.size();
  if (totalLoops != 2) return false;
  
  unsigned reductionLoops = 0;
  unsigned parallelLoops = 0;
  
  for (auto iterType : iterTypes) {
    if (iterType == utils::IteratorType::reduction) reductionLoops++;
    else if (iterType == utils::IteratorType::parallel) parallelLoops++;
  }
  
  if (reductionLoops == 1 && parallelLoops == 1) {
    // Check tensor dimensionalities
    bool hasMatrixInput = false;
    bool hasVectorInput = false;
    
    for (unsigned i = 0; i < ins.size(); i++) {
      auto tensorType = dyn_cast<RankedTensorType>(ins[i].getType());
      if (tensorType) {
        int rank = tensorType.getRank();
        if (rank == 2) hasMatrixInput = true;
        else if (rank == 1) hasVectorInput = true;
      }
    }
    
    auto outType = dyn_cast<RankedTensorType>(out.getType());
    bool hasVectorOutput = outType && outType.getRank() == 1;
    
    return hasMatrixInput && (hasVectorInput || hasVectorOutput);
  }
  
  return false;
}

bool IterationGraphSorter::hasMatrixMatrixPattern() const {
  // - 3 loops (2 parallel for output dims, 1 reduction for inner product)
  // - Two matrix inputs, one matrix output
  // - Specific loop structure: (i,j,k) where k is reduction
  
  unsigned totalLoops = iterTypes.size();
  if (totalLoops != 3) return false;
  
  unsigned reductionLoops = 0;
  unsigned parallelLoops = 0;
  
  for (auto iterType : iterTypes) {
    if (iterType == utils::IteratorType::reduction) reductionLoops++;
    else if (iterType == utils::IteratorType::parallel) parallelLoops++;
  }
  
  // Classic matmul: 2 parallel, 1 reduction
  if (reductionLoops != 1 || parallelLoops != 2) return false;
  
  // Check tensor dimensionalities - should have matrix inputs and output
  bool hasMatrixInputs = true;
  for (unsigned i = 0; i < ins.size(); i++) {
    auto tensorType = dyn_cast<RankedTensorType>(ins[i].getType());
    if (!tensorType || tensorType.getRank() != 2) {
      hasMatrixInputs = false;
      break;
    }
  }
  
  auto outType = dyn_cast<RankedTensorType>(out.getType());
  bool hasMatrixOutput = outType && outType.getRank() == 2;
  
  return hasMatrixInputs && hasMatrixOutput && ins.size() >= 2;
}

int64_t IterationGraphSorter::getTotalElementsHeuristic() const {
  int64_t maxElements = 1;
  
  // Check output tensor dimensions
  if (auto outType = dyn_cast<RankedTensorType>(out.getType())) {
    auto shape = outType.getShape();
    int64_t elements = 1;
    for (auto dim : shape) {
      if (dim != ShapedType::kDynamic) {
        elements *= dim;
      } else {
        elements *= 1000; // Assume 1000 for dynamic dimensions
      }
    }
    maxElements = std::max(maxElements, elements);
  }
  
  // Check input tensor dimensions
  for (const auto& in : ins) {
    if (auto tensorType = dyn_cast<RankedTensorType>(in.getType())) {
      auto shape = tensorType.getShape();
      int64_t elements = 1;
      for (auto dim : shape) {
        if (dim != ShapedType::kDynamic) {
          elements *= dim;
        } else {
          elements *= 1000; // Assume 1000 for dynamic dimensions
        }
      }
      maxElements = std::max(maxElements, elements);
    }
  }
  
  return maxElements;
}

bool IterationGraphSorter::hasBlockSparsePattern() const {
  // Block sparse operations typically have:
  // - Multiple reduction dimensions
  // - Structured sparsity patterns
  // - Regular block access patterns
  
  // Look for sparse encodings with multiple compressed dimensions
  for (const auto& in : ins) {
    if (auto tensorType = dyn_cast<RankedTensorType>(in.getType())) {
      if (auto encoding = dyn_cast_or_null<SparseTensorEncodingAttr>(tensorType.getEncoding())) {
        auto dimLevelTypes = encoding.getLvlTypes();
        int compressedDims = 0;
        for (auto dimType : dimLevelTypes) {
          if (dimType.isa<LevelFormat::Compressed>()) compressedDims++;
        }
        if (compressedDims >= 2) return true; // Likely block pattern
      }
    }
  }
  
  // Alternative heuristic: multiple reduction loops
  unsigned reductionLoops = 0;
  for (auto iterType : iterTypes) {
    if (iterType == utils::IteratorType::reduction) reductionLoops++;
  }
  
  return reductionLoops >= 2;
}

bool IterationGraphSorter::hasComplexReductionPattern() const {
  // Complex reductions have:
  // - Multiple reduction dimensions
  // - Nested loop structures
  // - Complex mathematical operations
  
  unsigned reductionLoops = 0;
  unsigned totalLoops = iterTypes.size();
  
  for (auto iterType : iterTypes) {
    if (iterType == utils::IteratorType::reduction) reductionLoops++;
  }
  
  // Complex if multiple reductions and deep nesting
  return reductionLoops >= 2 && totalLoops >= 4;
}

bool IterationGraphSorter::hasTriangularSolvePattern() const {
  // Triangular solve patterns:
  // - Lower/upper triangular matrix access
  // - Dependencies between iterations
  // - Solver-like computation pattern
  
  // Look for triangular structure in sparse encodings
  for (const auto& in : ins) {
    if (auto tensorType = dyn_cast<RankedTensorType>(in.getType())) {
      if (auto encoding = dyn_cast_or_null<SparseTensorEncodingAttr>(tensorType.getEncoding())) {
        auto dimLevelTypes = encoding.getLvlTypes();
        for (auto dimType : dimLevelTypes) {
          // Look for compressed formats which might indicate structure
          if (dimType.isa<LevelFormat::Compressed>() || 
              dimType.isa<LevelFormat::LooseCompressed>()) {
            return true; // Compressed sparse often indicates triangular structure
          }
        }
      }
    }
  }
  
  // Fallback: check for triangular-like patterns
  unsigned reductionLoops = 0;
  for (auto iterType : iterTypes) {
    if (iterType == utils::IteratorType::reduction) reductionLoops++;
  }
  
  return reductionLoops >= 1 && iterTypes.size() >= 2;
}

bool IterationGraphSorter::hasStreamingReductionPattern() const {
  // Streaming reductions have:
  // 1. At least one reduction dimension
  // 2. Large data size (streaming)
  // 3. Sequential access patterns
  
  unsigned reductionCount = 0;
  for (auto iterType : iterTypes) {
    if (iterType == utils::IteratorType::reduction) reductionCount++;
  }
  
  // Must have reductions and be reasonably large
  if (reductionCount == 0 || getTotalElementsHeuristic() < 16777216) { // < 4K*4K
    return false;
  }
  
  // Streaming pattern: more parallel than reduction dimensions
  unsigned parallelCount = 0;
  for (auto iterType : iterTypes) {
    if (iterType == utils::IteratorType::parallel) parallelCount++;
  }
  
  return parallelCount > reductionCount;
}
