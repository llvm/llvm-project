//===-- ClusteringTest.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Clustering.h"
#include "BenchmarkResult.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace exegesis {

namespace {

using testing::Field;
using testing::UnorderedElementsAre;
using testing::UnorderedElementsAreArray;

TEST(ClusteringTest, Clusters3D) {
  std::vector<InstructionBenchmark> Points(6);

  // Cluster around (x=0, y=1, z=2): points {0, 3}.
  Points[0].Measurements = {{"x", 0.01, ""}, {"y", 1.02, ""}, {"z", 1.98, "A"}};
  Points[3].Measurements = {{"x", -0.01, ""}, {"y", 1.02, ""}, {"z", 1.98, ""}};
  // Cluster around (x=1, y=1, z=2): points {1, 4}.
  Points[1].Measurements = {{"x", 1.01, ""}, {"y", 1.02, ""}, {"z", 1.98, ""}};
  Points[4].Measurements = {{"x", 0.99, ""}, {"y", 1.02, ""}, {"z", 1.98, ""}};
  // Cluster around (x=0, y=0, z=0): points {5}, marked as noise.
  Points[5].Measurements = {{"x", 0.0, ""}, {"y", 0.01, ""}, {"z", -0.02, ""}};
  // Error cluster: points {2}
  Points[2].Error = "oops";

  auto HasPoints = [](const std::vector<int> &Indices) {
    return Field(&InstructionBenchmarkClustering::Cluster::PointIndices,
                 UnorderedElementsAreArray(Indices));
  };

  auto Clustering = InstructionBenchmarkClustering::create(Points, 2, 0.25);
  ASSERT_TRUE((bool)Clustering);
  EXPECT_THAT(Clustering.get().getValidClusters(),
              UnorderedElementsAre(HasPoints({0, 3}), HasPoints({1, 4})));
  EXPECT_THAT(Clustering.get().getCluster(
                  InstructionBenchmarkClustering::ClusterId::noise()),
              HasPoints({5}));
  EXPECT_THAT(Clustering.get().getCluster(
                  InstructionBenchmarkClustering::ClusterId::error()),
              HasPoints({2}));

  EXPECT_EQ(Clustering.get().getClusterIdForPoint(2),
            InstructionBenchmarkClustering::ClusterId::error());
  EXPECT_EQ(Clustering.get().getClusterIdForPoint(5),
            InstructionBenchmarkClustering::ClusterId::noise());
  EXPECT_EQ(Clustering.get().getClusterIdForPoint(0),
            Clustering.get().getClusterIdForPoint(3));
  EXPECT_EQ(Clustering.get().getClusterIdForPoint(1),
            Clustering.get().getClusterIdForPoint(4));
}

TEST(ClusteringTest, Clusters3D_InvalidSize) {
  std::vector<InstructionBenchmark> Points(6);
  Points[0].Measurements = {{"x", 0.01, ""}, {"y", 1.02, ""}, {"z", 1.98, ""}};
  Points[1].Measurements = {{"y", 1.02, ""}, {"z", 1.98, ""}};
  auto Error =
      InstructionBenchmarkClustering::create(Points, 2, 0.25).takeError();
  ASSERT_TRUE((bool)Error);
  consumeError(std::move(Error));
}

TEST(ClusteringTest, Clusters3D_InvalidOrder) {
  std::vector<InstructionBenchmark> Points(6);
  Points[0].Measurements = {{"x", 0.01, ""}, {"y", 1.02, ""}};
  Points[1].Measurements = {{"y", 1.02, ""}, {"x", 1.98, ""}};
  auto Error =
      InstructionBenchmarkClustering::create(Points, 2, 0.25).takeError();
  ASSERT_TRUE((bool)Error);
  consumeError(std::move(Error));
}

} // namespace
} // namespace exegesis
