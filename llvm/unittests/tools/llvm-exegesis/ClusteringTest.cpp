//===-- ClusteringTest.cpp --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Clustering.h"
#include "BenchmarkResult.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace llvm {
namespace exegesis {

namespace {

using testing::Field;
using testing::UnorderedElementsAre;
using testing::UnorderedElementsAreArray;

static const auto HasPoints = [](const std::vector<int> &Indices) {
  return Field(&BenchmarkClustering::Cluster::PointIndices,
                 UnorderedElementsAreArray(Indices));
};

TEST(ClusteringTest, Clusters3D) {
  std::vector<Benchmark> Points(6);

  // Cluster around (x=0, y=1, z=2): points {0, 3}.
  Points[0].Measurements = {
      {"x", 0.01, 0.0, {}}, {"y", 1.02, 0.0, {}}, {"z", 1.98, 0.0, {}}};
  Points[3].Measurements = {
      {"x", -0.01, 0.0, {}}, {"y", 1.02, 0.0, {}}, {"z", 1.98, 0.0, {}}};
  // Cluster around (x=1, y=1, z=2): points {1, 4}.
  Points[1].Measurements = {
      {"x", 1.01, 0.0, {}}, {"y", 1.02, 0.0, {}}, {"z", 1.98, 0.0, {}}};
  Points[4].Measurements = {
      {"x", 0.99, 0.0, {}}, {"y", 1.02, 0.0, {}}, {"z", 1.98, 0.0, {}}};
  // Cluster around (x=0, y=0, z=0): points {5}, marked as noise.
  Points[5].Measurements = {
      {"x", 0.0, 0.0, {}}, {"y", 0.01, 0.0, {}}, {"z", -0.02, 0.0, {}}};
  // Error cluster: points {2}
  Points[2].Error = "oops";

  auto Clustering = BenchmarkClustering::create(
      Points, BenchmarkClustering::ModeE::Dbscan, 2, 0.25);
  ASSERT_TRUE((bool)Clustering);
  EXPECT_THAT(Clustering.get().getValidClusters(),
              UnorderedElementsAre(HasPoints({0, 3}), HasPoints({1, 4})));
  EXPECT_THAT(Clustering.get().getCluster(
                  BenchmarkClustering::ClusterId::noise()),
              HasPoints({5}));
  EXPECT_THAT(Clustering.get().getCluster(
                  BenchmarkClustering::ClusterId::error()),
              HasPoints({2}));

  EXPECT_EQ(Clustering.get().getClusterIdForPoint(2),
            BenchmarkClustering::ClusterId::error());
  EXPECT_EQ(Clustering.get().getClusterIdForPoint(5),
            BenchmarkClustering::ClusterId::noise());
  EXPECT_EQ(Clustering.get().getClusterIdForPoint(0),
            Clustering.get().getClusterIdForPoint(3));
  EXPECT_EQ(Clustering.get().getClusterIdForPoint(1),
            Clustering.get().getClusterIdForPoint(4));
}

TEST(ClusteringTest, Clusters3D_InvalidSize) {
  std::vector<Benchmark> Points(6);
  Points[0].Measurements = {
      {"x", 0.01, 0.0, {}}, {"y", 1.02, 0.0, {}}, {"z", 1.98, 0.0, {}}};
  Points[1].Measurements = {{"y", 1.02, 0.0, {}}, {"z", 1.98, 0.0, {}}};
  auto Error =
      BenchmarkClustering::create(
          Points, BenchmarkClustering::ModeE::Dbscan, 2, 0.25)
          .takeError();
  ASSERT_TRUE((bool)Error);
  consumeError(std::move(Error));
}

TEST(ClusteringTest, Clusters3D_InvalidOrder) {
  std::vector<Benchmark> Points(6);
  Points[0].Measurements = {{"x", 0.01, 0.0, {}}, {"y", 1.02, 0.0, {}}};
  Points[1].Measurements = {{"y", 1.02, 0.0, {}}, {"x", 1.98, 0.0, {}}};
  auto Error =
      BenchmarkClustering::create(
          Points, BenchmarkClustering::ModeE::Dbscan, 2, 0.25)
          .takeError();
  ASSERT_TRUE((bool)Error);
  consumeError(std::move(Error));
}

TEST(ClusteringTest, Ordering) {
  ASSERT_LT(BenchmarkClustering::ClusterId::makeValid(1),
            BenchmarkClustering::ClusterId::makeValid(2));

  ASSERT_LT(BenchmarkClustering::ClusterId::makeValid(2),
            BenchmarkClustering::ClusterId::noise());

  ASSERT_LT(BenchmarkClustering::ClusterId::makeValid(2),
            BenchmarkClustering::ClusterId::error());

  ASSERT_LT(BenchmarkClustering::ClusterId::noise(),
            BenchmarkClustering::ClusterId::error());
}

TEST(ClusteringTest, Ordering1) {
  std::vector<Benchmark> Points(3);

  Points[0].Measurements = {{"x", 0.0, 0.0, {}}};
  Points[1].Measurements = {{"x", 1.0, 0.0, {}}};
  Points[2].Measurements = {{"x", 2.0, 0.0, {}}};

  auto Clustering = BenchmarkClustering::create(
      Points, BenchmarkClustering::ModeE::Dbscan, 2, 1.1);
  ASSERT_TRUE((bool)Clustering);
  EXPECT_THAT(Clustering.get().getValidClusters(),
              UnorderedElementsAre(HasPoints({0, 1, 2})));
}

TEST(ClusteringTest, Ordering2) {
  std::vector<Benchmark> Points(3);

  Points[0].Measurements = {{"x", 0.0, 0.0, {}}};
  Points[1].Measurements = {{"x", 2.0, 0.0, {}}};
  Points[2].Measurements = {{"x", 1.0, 0.0, {}}};

  auto Clustering = BenchmarkClustering::create(
      Points, BenchmarkClustering::ModeE::Dbscan, 2, 1.1);
  ASSERT_TRUE((bool)Clustering);
  EXPECT_THAT(Clustering.get().getValidClusters(),
              UnorderedElementsAre(HasPoints({0, 1, 2})));
}

} // namespace
} // namespace exegesis
} // namespace llvm
