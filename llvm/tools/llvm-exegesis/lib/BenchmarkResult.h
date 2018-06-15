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

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/Support/YAMLTraits.h"
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace exegesis {

struct BenchmarkResultContext; // Forward declaration.

struct InstructionBenchmarkKey {
  // The LLVM opcode name.
  std::vector<llvm::MCInst> Instructions;
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
  enum ModeE { Unknown, Latency, Uops };
  ModeE Mode;
  std::string CpuName;
  std::string LLVMTriple;
  // The number of instructions inside the repeated snippet. For example, if a
  // snippet of 3 instructions is repeated 4 times, this is 12.
  int NumRepetitions = 0;
  // Note that measurements are per instruction.
  std::vector<BenchmarkMeasure> Measurements;
  std::string Error;
  std::string Info;
  std::vector<uint8_t> AssembledSnippet;

  // Read functions.
  static llvm::Expected<InstructionBenchmark>
  readYaml(const BenchmarkResultContext &Context, llvm::StringRef Filename);

  static llvm::Expected<std::vector<InstructionBenchmark>>
  readYamls(const BenchmarkResultContext &Context, llvm::StringRef Filename);

  void readYamlFrom(const BenchmarkResultContext &Context,
                    llvm::StringRef InputContent);

  // Write functions, non-const because of YAML traits.
  void writeYamlTo(const BenchmarkResultContext &Context, llvm::raw_ostream &S);

  llvm::Error writeYaml(const BenchmarkResultContext &Context,
                        const llvm::StringRef Filename);
};

//------------------------------------------------------------------------------
// Utilities to work with Benchmark measures.

// A class that measures stats over benchmark measures.
class BenchmarkMeasureStats {
public:
  void push(const BenchmarkMeasure &BM);

  double avg() const {
    assert(NumValues);
    return SumValues / NumValues;
  }
  double min() const { return MinValue; }
  double max() const { return MaxValue; }

  const std::string &key() const { return Key; }

private:
  std::string Key;
  double SumValues = 0.0;
  int NumValues = 0;
  double MaxValue = std::numeric_limits<double>::min();
  double MinValue = std::numeric_limits<double>::max();
};

// This context is used when de/serializing InstructionBenchmark to guarantee
// that Registers and Instructions are human readable and preserved accross
// different versions of LLVM.
struct BenchmarkResultContext {
  BenchmarkResultContext() = default;
  BenchmarkResultContext(BenchmarkResultContext &&) = default;
  BenchmarkResultContext &operator=(BenchmarkResultContext &&) = default;
  BenchmarkResultContext(const BenchmarkResultContext &) = delete;
  BenchmarkResultContext &operator=(const BenchmarkResultContext &) = delete;

  // Populate Registers and Instruction mapping.
  void addRegEntry(unsigned RegNo, llvm::StringRef Name);
  void addInstrEntry(unsigned Opcode, llvm::StringRef Name);

  // Register accessors.
  llvm::StringRef getRegName(unsigned RegNo) const;
  unsigned getRegNo(llvm::StringRef Name) const; // 0 is not found.

  // Instruction accessors.
  llvm::StringRef getInstrName(unsigned Opcode) const;
  unsigned getInstrOpcode(llvm::StringRef Name) const; // 0 is not found.

private:
  // Ideally we would like to use MCRegisterInfo and MCInstrInfo but doing so
  // would make testing harder, instead we create a mapping that we can easily
  // populate.
  std::unordered_map<unsigned, llvm::StringRef> InstrOpcodeToName;
  std::unordered_map<unsigned, llvm::StringRef> RegNoToName;
  llvm::StringMap<unsigned> InstrNameToOpcode;
  llvm::StringMap<unsigned> RegNameToNo;
};

} // namespace exegesis

#endif // LLVM_TOOLS_LLVM_EXEGESIS_BENCHMARKRESULT_H
