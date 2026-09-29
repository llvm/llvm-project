//===-- ResultAggregatorTest.cpp --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ResultAggregator.h"
#include "gtest/gtest.h"

namespace llvm {
namespace exegesis {

namespace {

TEST(ResultAggregatorTest, DefaultAggregator) {
  std::vector<Benchmark> Results(1);

  Results[0].Measurements = {BenchmarkMeasure::Create("x", 1, {})};

  Benchmark &Result = Results[0];

  std::unique_ptr<ResultAggregator> AggregatorToTest =
      ResultAggregator::CreateAggregator(Benchmark::RepetitionModeE::Duplicate);
  AggregatorToTest->AggregateResults(Result, ArrayRef(Results).drop_front());

  EXPECT_EQ(Result.Measurements[0].PerInstructionValue, 1);
  EXPECT_EQ(Result.Measurements[0].PerSnippetValue, 1);
  EXPECT_EQ(Result.Measurements[0].RawValue, 1);
}

TEST(ResultAggregatorTest, MinimumResultAggregator) {
  std::vector<Benchmark> Results(2);

  Results[0].Measurements = {BenchmarkMeasure::Create("x", 2, {})};
  Results[1].Measurements = {BenchmarkMeasure::Create("x", 1, {})};

  Benchmark &Result = Results[0];

  std::unique_ptr<ResultAggregator> AggregatorToTest =
      ResultAggregator::CreateAggregator(
          Benchmark::RepetitionModeE::AggregateMin);
  AggregatorToTest->AggregateResults(Result, ArrayRef(Results).drop_front());

  EXPECT_EQ(Result.Measurements[0].PerInstructionValue, 1);
  EXPECT_EQ(Result.Measurements[0].PerSnippetValue, 1);
  EXPECT_EQ(Result.Measurements[0].RawValue, 1);
}

TEST(ResultAggregatorTest, MiddleHalfAggregator) {
  std::vector<Benchmark> Results(2);

  Results[0].Measurements = {BenchmarkMeasure::Create("x", 2, {})};
  Results[1].Measurements = {BenchmarkMeasure::Create("x", 6, {})};

  Results[0].Key.Instructions.push_back(MCInst());
  Results[1].Key.Instructions.push_back(MCInst());

  Results[0].MinInstructions = 1;
  Results[1].MinInstructions = 3;

  Benchmark &Result = Results[0];

  std::unique_ptr<ResultAggregator> AggregatorToTest =
      ResultAggregator::CreateAggregator(
          Benchmark::RepetitionModeE::MiddleHalfLoop);
  AggregatorToTest->AggregateResults(Result, ArrayRef(Results).drop_front());

  EXPECT_EQ(Result.Measurements[0].PerInstructionValue, 4);
  EXPECT_EQ(Result.Measurements[0].PerSnippetValue, 4);
  EXPECT_EQ(Result.Measurements[0].RawValue, 4);
}

} // namespace

} // namespace exegesis
} // namespace llvm
