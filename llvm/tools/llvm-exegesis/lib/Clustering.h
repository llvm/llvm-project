//===-- Clustering.h --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Utilities to compute benchmark result clusters.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_EXEGESIS_CLUSTERING_H
#define LLVM_TOOLS_LLVM_EXEGESIS_CLUSTERING_H

#include "BenchmarkResult.h"
#include "llvm/Support/Error.h"
#include <vector>

namespace exegesis {

class InstructionBenchmarkClustering {
public:
  // Clusters `Points` using DBSCAN with the given parameters. See the cc file
  // for more explanations on the algorithm.
  static llvm::Expected<InstructionBenchmarkClustering>
  create(const std::vector<InstructionBenchmark> &Points, size_t MinPts,
         double Epsilon);

  class ClusterId {
  public:
    static ClusterId noise() { return ClusterId(kNoise); }
    static ClusterId error() { return ClusterId(kError); }
    static ClusterId makeValid(size_t Id) { return ClusterId(Id); }
    ClusterId() : Id_(kUndef) {}
    bool operator==(const ClusterId &O) const { return Id_ == O.Id_; }
    bool operator<(const ClusterId &O) const { return Id_ < O.Id_; }

    bool isValid() const { return Id_ <= kMaxValid; }
    bool isUndef() const { return Id_ == kUndef; }
    bool isNoise() const { return Id_ == kNoise; }
    bool isError() const { return Id_ == kError; }

    // Precondition: isValid().
    size_t getId() const {
      assert(isValid());
      return Id_;
    }

  private:
    explicit ClusterId(size_t Id) : Id_(Id) {}
    static constexpr const size_t kMaxValid =
        std::numeric_limits<size_t>::max() - 4;
    static constexpr const size_t kNoise = kMaxValid + 1;
    static constexpr const size_t kError = kMaxValid + 2;
    static constexpr const size_t kUndef = kMaxValid + 3;
    size_t Id_;
  };

  struct Cluster {
    Cluster() = delete;
    explicit Cluster(const ClusterId &Id) : Id(Id) {}

    const ClusterId Id;
    // Indices of benchmarks within the cluster.
    std::vector<int> PointIndices;
  };

  ClusterId getClusterIdForPoint(size_t P) const {
    return ClusterIdForPoint_[P];
  }

  const std::vector<InstructionBenchmark> &getPoints() const { return Points_; }

  const Cluster &getCluster(ClusterId Id) const {
    assert(!Id.isUndef() && "unlabeled cluster");
    if (Id.isNoise()) {
      return NoiseCluster_;
    }
    if (Id.isError()) {
      return ErrorCluster_;
    }
    return Clusters_[Id.getId()];
  }

  const std::vector<Cluster> &getValidClusters() const { return Clusters_; }

  // Returns true if the given point is within a distance Epsilon of each other.
  bool isNeighbour(const std::vector<BenchmarkMeasure> &P,
                   const std::vector<BenchmarkMeasure> &Q) const;

private:
  InstructionBenchmarkClustering(
      const std::vector<InstructionBenchmark> &Points, double EpsilonSquared);
  llvm::Error validateAndSetup();
  void dbScan(size_t MinPts);
  std::vector<size_t> rangeQuery(size_t Q) const;

  const std::vector<InstructionBenchmark> &Points_;
  const double EpsilonSquared_;
  int NumDimensions_ = 0;
  // ClusterForPoint_[P] is the cluster id for Points[P].
  std::vector<ClusterId> ClusterIdForPoint_;
  std::vector<Cluster> Clusters_;
  Cluster NoiseCluster_;
  Cluster ErrorCluster_;
};

} // namespace exegesis

#endif // LLVM_TOOLS_LLVM_EXEGESIS_CLUSTERING_H
