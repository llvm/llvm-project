//===-- BenchmarkResult.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines classes to represent measurements and serialize/deserialize them to
//  Yaml.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_EXEGESIS_BENCHMARKRESULT_H
#define LLVM_TOOLS_LLVM_EXEGESIS_BENCHMARKRESULT_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/YAMLTraits.h"
#include <string>
#include <vector>

namespace exegesis {

struct InstructionBenchmarkKey {
  // The LLVM opcode name.
  std::string OpcodeName;
  // The benchmark mode.
  std::string Mode;
  // An opaque configuration, that can be used to separate several benchmarks of
  // the same instruction under different configurations.
  std::string Config;
};

struct BenchmarkMeasure {
  std::string Key;
  double Value;
  std::string DebugString;
};

// The result of an instruction benchmark.
struct InstructionBenchmark {
  InstructionBenchmarkKey Key;
  std::string CpuName;
  std::string LLVMTriple;
  int NumRepetitions = 0;
  std::vector<BenchmarkMeasure> Measurements;
  std::string Error;
  std::string Info;

  static InstructionBenchmark readYamlOrDie(llvm::StringRef Filename);
  static std::vector<InstructionBenchmark> readYamlsOrDie(llvm::StringRef Filename);

  // Unfortunately this function is non const because of YAML traits.
  void writeYamlOrDie(const llvm::StringRef Filename);
};

} // namespace exegesis

#endif // LLVM_TOOLS_LLVM_EXEGESIS_BENCHMARKRESULT_H
