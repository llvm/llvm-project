//===-- BenchmarkRunnerTest.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BenchmarkRunner.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace llvm {
namespace exegesis {

namespace {

TEST(ScratchSpaceTest, Works) {
  BenchmarkRunner::ScratchSpace Space;
  EXPECT_EQ(reinterpret_cast<uintptr_t>(Space.ptr()) %
                BenchmarkRunner::ScratchSpace::kAlignment,
            0u);
  Space.ptr()[0] = 42;
  Space.ptr()[BenchmarkRunner::ScratchSpace::kSize - 1] = 43;
  Space.clear();
  EXPECT_EQ(Space.ptr()[0], 0);
  EXPECT_EQ(Space.ptr()[BenchmarkRunner::ScratchSpace::kSize - 1], 0);
}

} // namespace
} // namespace exegesis
} // namespace llvm
