//===-- BenchmarkResultTest.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "BenchmarkResult.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace exegesis {

bool operator==(const BenchmarkMeasure &A, const BenchmarkMeasure &B) {
  return std::tie(A.Key, A.Value) == std::tie(B.Key, B.Value);
}

namespace {

TEST(BenchmarkResultTest, WriteToAndReadFromDisk) {
  InstructionBenchmark ToDisk;

  ToDisk.AsmTmpl.Name = "name";
  ToDisk.CpuName = "cpu_name";
  ToDisk.LLVMTriple = "llvm_triple";
  ToDisk.NumRepetitions = 1;
  ToDisk.Measurements.push_back(BenchmarkMeasure{"a", 1, "debug a"});
  ToDisk.Measurements.push_back(BenchmarkMeasure{"b", 2, ""});
  ToDisk.Error = "error";

  const llvm::StringRef Filename("data.yaml");

  ToDisk.writeYamlOrDie(Filename);

  {
    const auto FromDisk = InstructionBenchmark::readYamlOrDie(Filename);

    EXPECT_EQ(FromDisk.AsmTmpl.Name, ToDisk.AsmTmpl.Name);
    EXPECT_EQ(FromDisk.CpuName, ToDisk.CpuName);
    EXPECT_EQ(FromDisk.LLVMTriple, ToDisk.LLVMTriple);
    EXPECT_EQ(FromDisk.NumRepetitions, ToDisk.NumRepetitions);
    EXPECT_THAT(FromDisk.Measurements, ToDisk.Measurements);
    EXPECT_THAT(FromDisk.Error, ToDisk.Error);
  }
}

} // namespace
} // namespace exegesis
