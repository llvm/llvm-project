//===- IterationGraphSorter.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines the iteration graph sorter (top-sort scheduling).
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_UTILS_ITERATIONGRAPHSORTER_H_
#define MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_UTILS_ITERATIONGRAPHSORTER_H_

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {

// Forward declarations.
class Value;
namespace utils {
enum class IteratorType : uint32_t;
} // namespace utils
namespace linalg {
class GenericOp;
} // namespace linalg

namespace sparse_tensor {

// Forward declaration for sparse tensor encoding
class SparseTensorEncodingAttr;

/// Iteration graph sorting mask,
enum class SortMask : unsigned {
  kIncludeDenseOutput = 0x1, // b001
  kIncludeDenseInput = 0x2,  // b010
  kIncludeAll = 0x7,   // b111
  kIncludeDense = 0x3, // b011
  kSparseOnly = 0x0,   // b000
};

class IterationGraphSorter {
public:
  /// Factory method that constructs an iteration graph sorter
  /// for the given linalg.generic operation (original behavior).
  static IterationGraphSorter fromGenericOp(linalg::GenericOp genericOp);
  
  /// Factory method that constructs an iteration graph sorter
  /// for the given linalg.generic operation with the specified loop ordering strategy.
  static IterationGraphSorter fromGenericOp(linalg::GenericOp genericOp, 
                                          LoopOrderingStrategy strategy);

  /// Returns a permutation that represents the scheduled loop order.
  /// Note that the returned AffineMap could be null if the kernel
  /// cannot be scheduled due to cyclic iteration graph.
  [[nodiscard]] AffineMap sort(SortMask mask, Value ignored = nullptr);

  /// Returns the number of loops in the iteration graph.
  unsigned getNumLoops() const { return loop2OutLvl.getNumDims(); }

private:
  // Private constructor.
  IterationGraphSorter(SmallVector<Value> &&ins,
                       SmallVector<AffineMap> &&loop2InsLvl, Value out,
                       AffineMap loop2OutLvl,
                       SmallVector<utils::IteratorType> &&iterTypes,
                       LoopOrderingStrategy strategy = LoopOrderingStrategy::kDefault);

  // Adds all the constraints in the given loop to level map.
  void addConstraints(Value t, AffineMap loop2LvlMap);

  /// A helper to compute a topological sort. The method has an
  /// O(n^2) time complexity since we use an adjacency matrix
  /// representation for the iteration graph.
  AffineMap topoSort();

  // The loop ordering strategy to use
  LoopOrderingStrategy loopOrderingStrategy;

  // Input tensors and associated loop to level maps.
  SmallVector<Value> ins;
  SmallVector<AffineMap> loop2InsLvl;

  // Output tensor and associated loop to level map.
  Value out;
  AffineMap loop2OutLvl;

  // Loop iteration types;
  SmallVector<utils::IteratorType> iterTypes;

  // Adjacency matrix that represents the iteration graph.
  std::vector<std::vector<bool>> itGraph;

  // InDegree used for topo sort.
  std::vector<unsigned> inDegree;

public:
  enum class SparseAccessType {
    kCompressedSequential,
    kSingletonScan,
    kRandomSparse,
    kDenseSubtensor
  };

  struct SparseAccessPattern {
    SparseAccessType type;
    double expectedSparsity;
    unsigned memoryIndirections;
    bool hasGoodLocality;
  };

private:

  // Add these fields to your LoopMemoryInfo struct:
  struct LoopMemoryInfo {
    unsigned totalTensorAccesses;
    double avgStrideComplexity;
    double spatialLocalityScore;
    double temporalReuseScore;
    double accessPatternRand;

    // Dense tensor access patterns
    SmallVector<unsigned> unitStrideAccesses;
    SmallVector<unsigned> linearStrideAccesses;
    SmallVector<unsigned> complexAccesses;

    // Sparse tensor access patterns
    SmallVector<unsigned> compressedSequentialAccesses;
    SmallVector<unsigned> singletonScanAccesses;
    SmallVector<unsigned> randomSparseAccesses;
    double sparseAccessCost;
    double expectedWorkingSet;
  };

  // Loop memory access information.
  SmallVector<LoopMemoryInfo, 0> loopMemoryAnalysis;

  // Analyze memory access patterns across all tensors.
  void analyzeMemoryPatterns();

  // Analyze memory patterns for a specific tensor mapping.
  void analyzeMapForMemoryPatterns(AffineMap map, unsigned tensorIdx,
                                   Value tensor, bool isOutput);

  // Compute stride complexity for a given affine expression.
  unsigned computeStrideComplexity(AffineExpr expr, unsigned targetLoop);

  // Select best loop candidate based on memory access patterns.
  unsigned selectBestCandidateByMemory(const std::vector<unsigned> &candidates);
  
  // Select best loop candidate based on density (dense first or sparse first).
  unsigned selectBestCandidateByDensity(const std::vector<unsigned> &candidates, bool denseFirst);
  
  // Select best loop candidate based on sequentiality (unit stride first).
  unsigned selectBestCandidateBySequentiality(const std::vector<unsigned> &candidates);
  
  // Select best loop candidate based on parallelism (parallel loops first).
  unsigned selectBestCandidateByParallelism(const std::vector<unsigned> &candidates);
  
  // Adaptive selection: automatically choose the best strategy based on kernel characteristics.
  unsigned selectBestCandidateByAdaptive(const std::vector<unsigned> &candidates);
  
  // Essential pattern detection functions for adaptive strategy
  bool hasMatrixVectorPattern() const;
  bool hasMatrixMatrixPattern() const;
  bool hasBlockSparsePattern() const;
  bool hasComplexReductionPattern() const;
  bool hasTriangularSolvePattern() const;
  bool hasMemoryIntensiveScanPattern() const;
  bool hasStreamingReductionPattern() const;
  bool hasTensorContractionPattern() const;
  
  // Essential helper functions
  bool hasHighParallelismPotential() const;
  bool hasSignificantReductions() const;
  bool hasComplexMemoryPattern() const;
  double computeAverageSparsity() const;
  int64_t getTotalElementsHeuristic() const;
  
  // Principle-based helper functions for adaptive strategy
  bool hasGoodMemoryLocalityPotential() const;
  bool hasStrongSequentialDependencies() const;
  
  LoopOrderingStrategy selectAdaptiveStrategy() const;
  
  // Get the current loop ordering strategy
  LoopOrderingStrategy getLoopOrderingStrategy() const { return loopOrderingStrategy; }

  // Compute architecture memory score for a loop.
  void computeArchitectureScore(unsigned loopIdx);

  // Compute combined portability score for loop ordering.
  double computePortableScore(unsigned loopIdx);

  // Analyze data access pattern characteristics.
  void analyzeDataAccessPatterns();

  // Analyze access patterns - fixed return type
  SparseAccessPattern
  analyzeSparseAccessPattern(AffineMap map, unsigned dim, unsigned loopIdx,
                             SparseTensorEncodingAttr encoding,
                             unsigned tensorIdx);

  // Analyze sparse format for a tensor
  void analyzeSparseFormat(Value tensor, unsigned tensorIdx);
};

} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_UTILS_ITERATIONGRAPHSORTER_H_
