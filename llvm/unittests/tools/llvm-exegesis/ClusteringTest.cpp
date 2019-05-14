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
  return Field(&InstructionBenchmarkClustering::Cluster::PointIndices,
                 UnorderedElementsAreArray(Indices));
};

TEST(ClusteringTest, Clusters3D) {
  std::vector<InstructionBenchmark> Points(6);

  // Cluster around (x=0, y=1, z=2): points {0, 3}.
  Points[0].Measurements = {
      {"x", 0.01, 0.0}, {"y", 1.02, 0.0}, {"z", 1.98, 0.0}};
  Points[3].Measurements = {
      {"x", -0.01, 0.0}, {"y", 1.02, 0.0}, {"z", 1.98, 0.0}};
  // Cluster around (x=1, y=1, z=2): points {1, 4}.
  Points[1].Measurements = {
      {"x", 1.01, 0.0}, {"y", 1.02, 0.0}, {"z", 1.98, 0.0}};
  Points[4].Measurements = {
      {"x", 0.99, 0.0}, {"y", 1.02, 0.0}, {"z", 1.98, 0.0}};
  // Cluster around (x=0, y=0, z=0): points {5}, marked as noise.
  Points[5].Measurements = {
      {"x", 0.0, 0.0}, {"y", 0.01, 0.0}, {"z", -0.02, 0.0}};
  // Error cluster: points {2}
  Points[2].Error = "oops";

  auto Clustering = InstructionBenchmarkClustering::create(
      Points, InstructionBenchmarkClustering::ModeE::Dbscan, 2, 0.25);
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
  Points[0].Measurements = {
      {"x", 0.01, 0.0}, {"y", 1.02, 0.0}, {"z", 1.98, 0.0}};
  Points[1].Measurements = {{"y", 1.02, 0.0}, {"z", 1.98, 0.0}};
  auto Error =
      InstructionBenchmarkClustering::create(
          Points, InstructionBenchmarkClustering::ModeE::Dbscan, 2, 0.25)
          .takeError();
  ASSERT_TRUE((bool)Error);
  consumeError(std::move(Error));
}

TEST(ClusteringTest, Clusters3D_InvalidOrder) {
  std::vector<InstructionBenchmark> Points(6);
  Points[0].Measurements = {{"x", 0.01, 0.0}, {"y", 1.02, 0.0}};
  Points[1].Measurements = {{"y", 1.02, 0.0}, {"x", 1.98, 0.0}};
  auto Error =
      InstructionBenchmarkClustering::create(
          Points, InstructionBenchmarkClustering::ModeE::Dbscan, 2, 0.25)
          .takeError();
  ASSERT_TRUE((bool)Error);
  consumeError(std::move(Error));
}

TEST(ClusteringTest, Ordering) {
  ASSERT_LT(InstructionBenchmarkClustering::ClusterId::makeValid(1),
            InstructionBenchmarkClustering::ClusterId::makeValid(2));

  ASSERT_LT(InstructionBenchmarkClustering::ClusterId::makeValid(2),
            InstructionBenchmarkClustering::ClusterId::noise());

  ASSERT_LT(InstructionBenchmarkClustering::ClusterId::makeValid(2),
            InstructionBenchmarkClustering::ClusterId::error());

  ASSERT_LT(InstructionBenchmarkClustering::ClusterId::noise(),
            InstructionBenchmarkClustering::ClusterId::error());
}

TEST(ClusteringTest, Ordering1) {
  std::vector<InstructionBenchmark> Points(3);

  Points[0].Measurements = {
      {"x", 0.0, 0.0}};
  Points[1].Measurements = {
      {"x", 1.0, 0.0}};
  Points[2].Measurements = {
      {"x", 2.0, 0.0}};

  auto Clustering = InstructionBenchmarkClustering::create(
      Points, InstructionBenchmarkClustering::ModeE::Dbscan, 2, 1.1);
  ASSERT_TRUE((bool)Clustering);
  EXPECT_THAT(Clustering.get().getValidClusters(),
              UnorderedElementsAre(HasPoints({0, 1, 2})));
}

TEST(ClusteringTest, Ordering2) {
  std::vector<InstructionBenchmark> Points(3);

  Points[0].Measurements = {
      {"x", 0.0, 0.0}};
  Points[1].Measurements = {
      {"x", 2.0, 0.0}};
  Points[2].Measurements = {
      {"x", 1.0, 0.0}};

  auto Clustering = InstructionBenchmarkClustering::create(
      Points, InstructionBenchmarkClustering::ModeE::Dbscan, 2, 1.1);
  ASSERT_TRUE((bool)Clustering);
  EXPECT_THAT(Clustering.get().getValidClusters(),
              UnorderedElementsAre(HasPoints({0, 1, 2})));
}

} // namespace
} // namespace exegesis
} // namespace llvm
