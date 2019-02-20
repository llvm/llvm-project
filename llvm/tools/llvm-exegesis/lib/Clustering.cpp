//===-- Clustering.cpp ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Clustering.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>
#include <string>
#include <vector>

namespace llvm {
namespace exegesis {

// The clustering problem has the following characteristics:
//  (A) - Low dimension (dimensions are typically proc resource units,
//    typically < 10).
//  (B) - Number of points : ~thousands (points are measurements of an MCInst)
//  (C) - Number of clusters: ~tens.
//  (D) - The number of clusters is not known /a priory/.
//  (E) - The amount of noise is relatively small.
// The problem is rather small. In terms of algorithms, (D) disqualifies
// k-means and makes algorithms such as DBSCAN[1] or OPTICS[2] more applicable.
//
// We've used DBSCAN here because it's simple to implement. This is a pretty
// straightforward and inefficient implementation of the pseudocode in [2].
//
// [1] https://en.wikipedia.org/wiki/DBSCAN
// [2] https://en.wikipedia.org/wiki/OPTICS_algorithm

// Finds the points at distance less than sqrt(EpsilonSquared) of Q (not
// including Q).
void InstructionBenchmarkClustering::rangeQuery(
    const size_t Q, std::vector<size_t> &Neighbors) const {
  Neighbors.clear();
  Neighbors.reserve(Points_.size() - 1); // The Q itself isn't a neighbor.
  const auto &QMeasurements = Points_[Q].Measurements;
  for (size_t P = 0, NumPoints = Points_.size(); P < NumPoints; ++P) {
    if (P == Q)
      continue;
    const auto &PMeasurements = Points_[P].Measurements;
    if (PMeasurements.empty()) // Error point.
      continue;
    if (isNeighbour(PMeasurements, QMeasurements)) {
      Neighbors.push_back(P);
    }
  }
}

InstructionBenchmarkClustering::InstructionBenchmarkClustering(
    const std::vector<InstructionBenchmark> &Points,
    const double EpsilonSquared)
    : Points_(Points), EpsilonSquared_(EpsilonSquared),
      NoiseCluster_(ClusterId::noise()), ErrorCluster_(ClusterId::error()) {}

llvm::Error InstructionBenchmarkClustering::validateAndSetup() {
  ClusterIdForPoint_.resize(Points_.size());
  // Mark erroneous measurements out.
  // All points must have the same number of dimensions, in the same order.
  const std::vector<BenchmarkMeasure> *LastMeasurement = nullptr;
  for (size_t P = 0, NumPoints = Points_.size(); P < NumPoints; ++P) {
    const auto &Point = Points_[P];
    if (!Point.Error.empty()) {
      ClusterIdForPoint_[P] = ClusterId::error();
      ErrorCluster_.PointIndices.push_back(P);
      continue;
    }
    const auto *CurMeasurement = &Point.Measurements;
    if (LastMeasurement) {
      if (LastMeasurement->size() != CurMeasurement->size()) {
        return llvm::make_error<llvm::StringError>(
            "inconsistent measurement dimensions",
            llvm::inconvertibleErrorCode());
      }
      for (size_t I = 0, E = LastMeasurement->size(); I < E; ++I) {
        if (LastMeasurement->at(I).Key != CurMeasurement->at(I).Key) {
          return llvm::make_error<llvm::StringError>(
              "inconsistent measurement dimensions keys",
              llvm::inconvertibleErrorCode());
        }
      }
    }
    LastMeasurement = CurMeasurement;
  }
  if (LastMeasurement) {
    NumDimensions_ = LastMeasurement->size();
  }
  return llvm::Error::success();
}

void InstructionBenchmarkClustering::dbScan(const size_t MinPts) {
  std::vector<size_t> Neighbors; // Persistent buffer to avoid allocs.
  for (size_t P = 0, NumPoints = Points_.size(); P < NumPoints; ++P) {
    if (!ClusterIdForPoint_[P].isUndef())
      continue; // Previously processed in inner loop.
    rangeQuery(P, Neighbors);
    if (Neighbors.size() + 1 < MinPts) { // Density check.
      // The region around P is not dense enough to create a new cluster, mark
      // as noise for now.
      ClusterIdForPoint_[P] = ClusterId::noise();
      continue;
    }

    // Create a new cluster, add P.
    Clusters_.emplace_back(ClusterId::makeValid(Clusters_.size()));
    Cluster &CurrentCluster = Clusters_.back();
    ClusterIdForPoint_[P] = CurrentCluster.Id; /* Label initial point */
    CurrentCluster.PointIndices.push_back(P);

    // Process P's neighbors.
    llvm::SetVector<size_t, std::deque<size_t>> ToProcess;
    ToProcess.insert(Neighbors.begin(), Neighbors.end());
    while (!ToProcess.empty()) {
      // Retrieve a point from the set.
      const size_t Q = *ToProcess.begin();
      ToProcess.erase(ToProcess.begin());

      if (ClusterIdForPoint_[Q].isNoise()) {
        // Change noise point to border point.
        ClusterIdForPoint_[Q] = CurrentCluster.Id;
        CurrentCluster.PointIndices.push_back(Q);
        continue;
      }
      if (!ClusterIdForPoint_[Q].isUndef()) {
        continue; // Previously processed.
      }
      // Add Q to the current custer.
      ClusterIdForPoint_[Q] = CurrentCluster.Id;
      CurrentCluster.PointIndices.push_back(Q);
      // And extend to the neighbors of Q if the region is dense enough.
      rangeQuery(Q, Neighbors);
      if (Neighbors.size() + 1 >= MinPts) {
        ToProcess.insert(Neighbors.begin(), Neighbors.end());
      }
    }
  }
  // assert(Neighbors.capacity() == (Points_.size() - 1));
  // ^ True, but it is not quaranteed to be true in all the cases.

  // Add noisy points to noise cluster.
  for (size_t P = 0, NumPoints = Points_.size(); P < NumPoints; ++P) {
    if (ClusterIdForPoint_[P].isNoise()) {
      NoiseCluster_.PointIndices.push_back(P);
    }
  }
}

// Given an instruction Opcode, we can make benchmarks (measurements) of the
// instruction characteristics/performance. Then, to facilitate further analysis
// we group the benchmarks with *similar* characteristics into clusters.
// Now, this is all not entirely deterministic. Some instructions have variable
// characteristics, depending on their arguments. And thus, if we do several
// benchmarks of the same instruction Opcode, we may end up with *different*
// performance characteristics measurements. And when we then do clustering,
// these several benchmarks of the same instruction Opcode may end up being
// clustered into *different* clusters. This is not great for further analysis.
// We shall find every opcode with benchmarks not in just one cluster, and move
// *all* the benchmarks of said Opcode into one new unstable cluster per Opcode.
void InstructionBenchmarkClustering::stabilize(unsigned NumOpcodes) {
  // Given an instruction Opcode, in which clusters do benchmarks of this
  // instruction lie? Normally, they all should be in the same cluster.
  std::vector<llvm::SmallSet<ClusterId, 1>> OpcodeToClusterIDs;
  OpcodeToClusterIDs.resize(NumOpcodes);
  // The list of opcodes that have more than one cluster.
  llvm::SetVector<size_t> UnstableOpcodes;
  // Populate OpcodeToClusterIDs and UnstableOpcodes data structures.
  assert(ClusterIdForPoint_.size() == Points_.size() && "size mismatch");
  for (const auto &Point : zip(Points_, ClusterIdForPoint_)) {
    const ClusterId &ClusterIdOfPoint = std::get<1>(Point);
    if (!ClusterIdOfPoint.isValid())
      continue; // Only process fully valid clusters.
    const unsigned Opcode = std::get<0>(Point).keyInstruction().getOpcode();
    assert(Opcode < NumOpcodes && "NumOpcodes is incorrect (too small)");
    llvm::SmallSet<ClusterId, 1> &ClusterIDsOfOpcode =
        OpcodeToClusterIDs[Opcode];
    ClusterIDsOfOpcode.insert(ClusterIdOfPoint);
    // Is there more than one ClusterID for this opcode?.
    if (ClusterIDsOfOpcode.size() < 2)
      continue; // If not, then at this moment this Opcode is stable.
    // Else let's record this unstable opcode for future use.
    UnstableOpcodes.insert(Opcode);
  }
  assert(OpcodeToClusterIDs.size() == NumOpcodes && "sanity check");

  // We know with how many [new] clusters we will end up with.
  const auto NewTotalClusterCount = Clusters_.size() + UnstableOpcodes.size();
  Clusters_.reserve(NewTotalClusterCount);
  for (const size_t UnstableOpcode : UnstableOpcodes.getArrayRef()) {
    const llvm::SmallSet<ClusterId, 1> &ClusterIDs =
        OpcodeToClusterIDs[UnstableOpcode];
    assert(ClusterIDs.size() > 1 &&
           "Should only have Opcodes with more than one cluster.");

    // Create a new unstable cluster, one per Opcode.
    Clusters_.emplace_back(ClusterId::makeValidUnstable(Clusters_.size()));
    Cluster &UnstableCluster = Clusters_.back();
    // We will find *at least* one point in each of these clusters.
    UnstableCluster.PointIndices.reserve(ClusterIDs.size());

    // Go through every cluster which we recorded as containing benchmarks
    // of this UnstableOpcode. NOTE: we only recorded valid clusters.
    for (const ClusterId &CID : ClusterIDs) {
      assert(CID.isValid() &&
             "We only recorded valid clusters, not noise/error clusters.");
      Cluster &OldCluster = Clusters_[CID.getId()]; // Valid clusters storage.
      // Within each cluster, go through each point, and either move it to the
      // new unstable cluster, or 'keep' it.
      // In this case, we'll reshuffle OldCluster.PointIndices vector
      // so that all the points that are *not* for UnstableOpcode are first,
      // and the rest of the points is for the UnstableOpcode.
      const auto it = std::stable_partition(
          OldCluster.PointIndices.begin(), OldCluster.PointIndices.end(),
          [this, UnstableOpcode](size_t P) {
            return Points_[P].keyInstruction().getOpcode() != UnstableOpcode;
          });
      assert(std::distance(it, OldCluster.PointIndices.end()) > 0 &&
             "Should have found at least one bad point");
      // Mark to-be-moved points as belonging to the new cluster.
      std::for_each(it, OldCluster.PointIndices.end(),
                    [this, &UnstableCluster](size_t P) {
                      ClusterIdForPoint_[P] = UnstableCluster.Id;
                    });
      // Actually append to-be-moved points to the new cluster.
      UnstableCluster.PointIndices.insert(UnstableCluster.PointIndices.cend(),
                                          it, OldCluster.PointIndices.end());
      // And finally, remove "to-be-moved" points form the old cluster.
      OldCluster.PointIndices.erase(it, OldCluster.PointIndices.cend());
      // Now, the old cluster may end up being empty, but let's just keep it
      // in whatever state it ended up. Purging empty clusters isn't worth it.
    };
    assert(UnstableCluster.PointIndices.size() > 1 &&
           "New unstable cluster should end up with more than one point.");
    assert(UnstableCluster.PointIndices.size() >= ClusterIDs.size() &&
           "New unstable cluster should end up with no less points than there "
           "was clusters");
  }
  assert(Clusters_.size() == NewTotalClusterCount && "sanity check");
}

llvm::Expected<InstructionBenchmarkClustering>
InstructionBenchmarkClustering::create(
    const std::vector<InstructionBenchmark> &Points, const size_t MinPts,
    const double Epsilon, llvm::Optional<unsigned> NumOpcodes) {
  InstructionBenchmarkClustering Clustering(Points, Epsilon * Epsilon);
  if (auto Error = Clustering.validateAndSetup()) {
    return std::move(Error);
  }
  if (Clustering.ErrorCluster_.PointIndices.size() == Points.size()) {
    return Clustering; // Nothing to cluster.
  }

  Clustering.dbScan(MinPts);

  if (NumOpcodes.hasValue())
    Clustering.stabilize(NumOpcodes.getValue());

  return Clustering;
}

} // namespace exegesis
} // namespace llvm
