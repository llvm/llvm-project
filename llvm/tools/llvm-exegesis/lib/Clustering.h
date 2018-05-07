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
    static ClusterId makeValid(int Id) {
      assert(Id >= 0);
      return ClusterId(Id);
    }
    ClusterId() : Id_(kUndef) {}
    bool operator==(const ClusterId &O) const { return Id_ == O.Id_; }

    bool isValid() const { return Id_ >= 0; }
    bool isUndef() const { return Id_ == kUndef; }
    bool isNoise() const { return Id_ == kNoise; }
    bool isError() const { return Id_ == kError; }

    // Precondition: isValid().
    size_t getId() const {
      assert(isValid());
      return static_cast<size_t>(Id_);
    }

  private:
    explicit ClusterId(int Id) : Id_(Id) {}
    static constexpr const int kUndef = -1;
    static constexpr const int kNoise = -2;
    static constexpr const int kError = -3;
    int Id_;
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

private:
  InstructionBenchmarkClustering();
  llvm::Error validateAndSetup(const std::vector<InstructionBenchmark> &Points);
  void dbScan(const std::vector<InstructionBenchmark> &Points, size_t MinPts,
              double EpsilonSquared);
  int NumDimensions_ = 0;
  // ClusterForPoint_[P] is the cluster id for Points[P].
  std::vector<ClusterId> ClusterIdForPoint_;
  std::vector<Cluster> Clusters_;
  Cluster NoiseCluster_;
  Cluster ErrorCluster_;
};

} // namespace exegesis

#endif // LLVM_TOOLS_LLVM_EXEGESIS_CLUSTERING_H
