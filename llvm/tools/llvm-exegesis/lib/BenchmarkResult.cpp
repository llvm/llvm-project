//===-- BenchmarkResult.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "BenchmarkResult.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

// Defining YAML traits for IO.
namespace llvm {
namespace yaml {

// std::vector<exegesis::Measure> will be rendered as a list.
template <> struct SequenceElementTraits<exegesis::BenchmarkMeasure> {
  static const bool flow = false;
};

// exegesis::Measure is rendererd as a flow instead of a list.
// e.g. { "key": "the key", "value": 0123 }
template <> struct MappingTraits<exegesis::BenchmarkMeasure> {
  static void mapping(IO &Io, exegesis::BenchmarkMeasure &Obj) {
    Io.mapRequired("key", Obj.Key);
    Io.mapRequired("value", Obj.Value);
    Io.mapOptional("debug_string", Obj.DebugString);
  }
  static const bool flow = true;
};

template <> struct MappingTraits<exegesis::InstructionBenchmarkKey> {
  static void mapping(IO &Io, exegesis::InstructionBenchmarkKey &Obj) {
    Io.mapRequired("opcode_name", Obj.OpcodeName);
    Io.mapRequired("mode", Obj.Mode);
    Io.mapOptional("config", Obj.Config);
  }
};

template <> struct MappingTraits<exegesis::InstructionBenchmark> {
  static void mapping(IO &Io, exegesis::InstructionBenchmark &Obj) {
    Io.mapRequired("key", Obj.Key);
    Io.mapRequired("cpu_name", Obj.CpuName);
    Io.mapRequired("llvm_triple", Obj.LLVMTriple);
    Io.mapRequired("num_repetitions", Obj.NumRepetitions);
    Io.mapRequired("measurements", Obj.Measurements);
    Io.mapRequired("error", Obj.Error);
    Io.mapOptional("info", Obj.Info);
  }
};

} // namespace yaml
} // namespace llvm

LLVM_YAML_IS_DOCUMENT_LIST_VECTOR(exegesis::InstructionBenchmark)

namespace exegesis {

namespace {

template <typename ObjectOrList>
ObjectOrList readYamlOrDieCommon(llvm::StringRef Filename) {
  std::unique_ptr<llvm::MemoryBuffer> MemBuffer = llvm::cantFail(
      llvm::errorOrToExpected(llvm::MemoryBuffer::getFile(Filename)));
  llvm::yaml::Input Yin(*MemBuffer);
  ObjectOrList Benchmark;
  Yin >> Benchmark;
  return Benchmark;
}

} // namespace

InstructionBenchmark
InstructionBenchmark::readYamlOrDie(llvm::StringRef Filename) {
  return readYamlOrDieCommon<InstructionBenchmark>(Filename);
}

std::vector<InstructionBenchmark>
InstructionBenchmark::readYamlsOrDie(llvm::StringRef Filename) {
  return readYamlOrDieCommon<std::vector<InstructionBenchmark>>(Filename);
}

void InstructionBenchmark::writeYamlOrDie(const llvm::StringRef Filename) {
  if (Filename == "-") {
    llvm::yaml::Output Yout(llvm::outs());
    Yout << *this;
  } else {
    llvm::SmallString<1024> Buffer;
    llvm::raw_svector_ostream Ostr(Buffer);
    llvm::yaml::Output Yout(Ostr);
    Yout << *this;
    std::unique_ptr<llvm::FileOutputBuffer> File =
        llvm::cantFail(llvm::FileOutputBuffer::create(Filename, Buffer.size()));
    memcpy(File->getBufferStart(), Buffer.data(), Buffer.size());
    llvm::cantFail(File->commit());
  }
}

} // namespace exegesis
