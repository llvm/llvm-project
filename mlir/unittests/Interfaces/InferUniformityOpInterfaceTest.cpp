//===- InferUniformityOpInterfaceTest.cpp - Uniformity lattice tests ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/InferUniformityOpInterface.h"

#include <gtest/gtest.h>

using namespace mlir;

TEST(Uniformity, ScopesNest) {
  EXPECT_LT(UniformityScope::Divergent, UniformityScope::Subgroup);
  EXPECT_LT(UniformityScope::Subgroup, UniformityScope::Workgroup);
  EXPECT_LT(UniformityScope::Workgroup, UniformityScope::Cluster);
  EXPECT_LT(UniformityScope::Cluster, UniformityScope::Uniform);
  EXPECT_EQ(meet(UniformityScope::Workgroup, UniformityScope::Subgroup),
            UniformityScope::Subgroup);
  EXPECT_EQ(meet(UniformityScope::Subgroup, UniformityScope::Workgroup),
            UniformityScope::Subgroup);
  EXPECT_EQ(meet(UniformityScope::Uniform, UniformityScope::Uniform),
            UniformityScope::Uniform);
}

TEST(Uniformity, JoinIsTheNarrowerScope) {
  Uniformity workgroup(UniformityScope::Workgroup);
  Uniformity cluster(UniformityScope::Cluster);
  EXPECT_EQ(Uniformity::join(workgroup, cluster), workgroup);
  EXPECT_EQ(Uniformity::join(cluster, workgroup), workgroup);
  EXPECT_EQ(
      Uniformity::join(Uniformity::getUniform(), Uniformity::getDivergent()),
      Uniformity::getDivergent());
  EXPECT_EQ(Uniformity::join(workgroup, workgroup), workgroup);
}

TEST(Uniformity, UninitializedIsNeutral) {
  Uniformity none;
  Uniformity subgroup(UniformityScope::Subgroup);
  EXPECT_TRUE(none.isUninitialized());
  EXPECT_FALSE(subgroup.isUninitialized());
  EXPECT_EQ(Uniformity::join(none, subgroup), subgroup);
  EXPECT_EQ(Uniformity::join(subgroup, none), subgroup);
  EXPECT_TRUE(Uniformity::join(none, none).isUninitialized());
}

TEST(Uniformity, JoinOfList) {
  EXPECT_TRUE(Uniformity::join(ArrayRef<Uniformity>()).isUninitialized());
  Uniformity allNone[] = {Uniformity(), Uniformity()};
  EXPECT_TRUE(Uniformity::join(allNone).isUninitialized());
  Uniformity values[] = {Uniformity(), Uniformity::getUniform(),
                         Uniformity(UniformityScope::Cluster), Uniformity(),
                         Uniformity(UniformityScope::Workgroup)};
  EXPECT_EQ(Uniformity::join(values).getScope(), UniformityScope::Workgroup);
}

TEST(Uniformity, Names) {
  EXPECT_EQ(stringifyUniformityScope(UniformityScope::Divergent), "divergent");
  EXPECT_EQ(stringifyUniformityScope(UniformityScope::Uniform), "uniform");
  for (UniformityScope scope :
       {UniformityScope::Divergent, UniformityScope::Subgroup,
        UniformityScope::Workgroup, UniformityScope::Cluster,
        UniformityScope::Uniform}) {
    std::optional<UniformityScope> back =
        symbolizeUniformityScope(stringifyUniformityScope(scope));
    ASSERT_TRUE(back.has_value());
    EXPECT_EQ(*back, scope);
  }
  EXPECT_FALSE(symbolizeUniformityScope("warp").has_value());
  EXPECT_FALSE(symbolizeUniformityScope("Uniform").has_value());
}
