//===-- BenchmarkRunnerTest.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
  EXPECT_EQ(reinterpret_cast<intptr_t>(Space.ptr()) %
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
