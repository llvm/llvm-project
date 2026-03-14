//===- llvm/unittest/Support/EditDistanceTest.cpp - Edit distance tests ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/edit_distance.h"
#include "gtest/gtest.h"
#include <cstdlib>

using namespace llvm;

namespace {

struct Result {
  unsigned NumMaps;
  unsigned EditDist;
};
} // namespace

static Result editDistanceAndMaps(StringRef A, StringRef B,
                                  unsigned MaxEditDistance = 0) {
  unsigned NumMaps = 0;
  auto TrackMaps = [&](const char X) {
    ++NumMaps;
    return X;
  };
  unsigned EditDist = llvm::ComputeMappedEditDistance(
      ArrayRef(A.data(), A.size()), ArrayRef(B.data(), B.size()), TrackMaps,
      true, MaxEditDistance);
  return {NumMaps, EditDist};
}

TEST(EditDistance, VerifyShortCircuit) {
  StringRef Hello = "Hello";
  StringRef HelloWorld = "HelloWorld";
  Result R = editDistanceAndMaps(Hello, HelloWorld, 5);
  EXPECT_EQ(R.EditDist, 5U);
  EXPECT_GT(R.NumMaps, 0U);

  R = editDistanceAndMaps(Hello, HelloWorld);
  EXPECT_EQ(R.EditDist, 5U);
  EXPECT_GT(R.NumMaps, 0U);

  R = editDistanceAndMaps(Hello, HelloWorld, 4);
  EXPECT_EQ(R.EditDist, 5U);
  EXPECT_EQ(R.NumMaps, 0U);

  R = editDistanceAndMaps(HelloWorld, Hello, 4);
  EXPECT_EQ(R.EditDist, 5U);
  EXPECT_EQ(R.NumMaps, 0U);

  R = editDistanceAndMaps(Hello, HelloWorld, 1);
  EXPECT_EQ(R.EditDist, 2U);
  EXPECT_EQ(R.NumMaps, 0U);

  R = editDistanceAndMaps(HelloWorld, Hello, 1);
  EXPECT_EQ(R.EditDist, 2U);
  EXPECT_EQ(R.NumMaps, 0U);
}
