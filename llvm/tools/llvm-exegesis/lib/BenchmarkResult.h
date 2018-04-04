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

struct AsmTemplate {
  std::string Name;
};

struct BenchmarkMeasure {
  std::string Key;
  double Value;
  std::string DebugString;
};

// The result of an instruction benchmark.
struct InstructionBenchmark {
  AsmTemplate AsmTmpl;
  std::string CpuName;
  std::string LLVMTriple;
  int NumRepetitions = 0;
  std::vector<BenchmarkMeasure> Measurements;
  std::string Error;

  static InstructionBenchmark readYamlOrDie(llvm::StringRef Filename);

  // Unfortunately this function is non const because of YAML traits.
  void writeYamlOrDie(const llvm::StringRef Filename);
};

} // namespace exegesis

#endif // LLVM_TOOLS_LLVM_EXEGESIS_BENCHMARKRESULT_H
