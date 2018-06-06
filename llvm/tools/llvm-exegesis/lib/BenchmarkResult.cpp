//===-- BenchmarkResult.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "BenchmarkResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

// Defining YAML traits for IO.
namespace llvm {
namespace yaml {

// std::vector<llvm::MCInst> will be rendered as a list.
template <> struct SequenceElementTraits<llvm::MCInst> {
  static const bool flow = false;
};

template <> struct ScalarTraits<llvm::MCInst> {

  static void output(const llvm::MCInst &Value, void *Ctx,
                     llvm::raw_ostream &Out) {
    assert(Ctx);
    auto *Context = static_cast<const exegesis::BenchmarkResultContext *>(Ctx);
    const StringRef Name = Context->getInstrName(Value.getOpcode());
    assert(!Name.empty());
    Out << Name;
  }

  static StringRef input(StringRef Scalar, void *Ctx, llvm::MCInst &Value) {
    assert(Ctx);
    auto *Context = static_cast<const exegesis::BenchmarkResultContext *>(Ctx);
    const unsigned Opcode = Context->getInstrOpcode(Scalar);
    if (Opcode == 0) {
      return "Unable to parse instruction";
    }
    Value.setOpcode(Opcode);
    return StringRef();
  }

  static QuotingType mustQuote(StringRef) { return QuotingType::Single; }

  static const bool flow = true;
};

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

template <>
struct ScalarEnumerationTraits<exegesis::InstructionBenchmark::ModeE> {
  static void enumeration(IO &Io,
                          exegesis::InstructionBenchmark::ModeE &Value) {
    Io.enumCase(Value, "", exegesis::InstructionBenchmark::Unknown);
    Io.enumCase(Value, "latency", exegesis::InstructionBenchmark::Latency);
    Io.enumCase(Value, "uops", exegesis::InstructionBenchmark::Uops);
  }
};

template <> struct MappingTraits<exegesis::InstructionBenchmarkKey> {
  static void mapping(IO &Io, exegesis::InstructionBenchmarkKey &Obj) {
    Io.mapRequired("opcode_name", Obj.OpcodeName);
    Io.mapOptional("instructions", Obj.Instructions);
    Io.mapOptional("config", Obj.Config);
  }
};

template <> struct MappingTraits<exegesis::InstructionBenchmark> {
  static void mapping(IO &Io, exegesis::InstructionBenchmark &Obj) {
    Io.mapRequired("mode", Obj.Mode);
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

void BenchmarkResultContext::addRegEntry(unsigned RegNo, llvm::StringRef Name) {
  assert(RegNoToName.find(RegNo) == RegNoToName.end());
  assert(RegNameToNo.find(Name) == RegNameToNo.end());
  RegNoToName[RegNo] = Name;
  RegNameToNo[Name] = RegNo;
}

llvm::StringRef BenchmarkResultContext::getRegName(unsigned RegNo) const {
  const auto Itr = RegNoToName.find(RegNo);
  if (Itr != RegNoToName.end())
    return Itr->second;
  return {};
}

unsigned BenchmarkResultContext::getRegNo(llvm::StringRef Name) const {
  const auto Itr = RegNameToNo.find(Name);
  if (Itr != RegNameToNo.end())
    return Itr->second;
  return 0;
}

void BenchmarkResultContext::addInstrEntry(unsigned Opcode,
                                           llvm::StringRef Name) {
  assert(InstrOpcodeToName.find(Opcode) == InstrOpcodeToName.end());
  assert(InstrNameToOpcode.find(Name) == InstrNameToOpcode.end());
  InstrOpcodeToName[Opcode] = Name;
  InstrNameToOpcode[Name] = Opcode;
}

llvm::StringRef BenchmarkResultContext::getInstrName(unsigned Opcode) const {
  const auto Itr = InstrOpcodeToName.find(Opcode);
  if (Itr != InstrOpcodeToName.end())
    return Itr->second;
  return {};
}

unsigned BenchmarkResultContext::getInstrOpcode(llvm::StringRef Name) const {
  const auto Itr = InstrNameToOpcode.find(Name);
  if (Itr != InstrNameToOpcode.end())
    return Itr->second;
  return 0;
}

template <typename ObjectOrList>
static ObjectOrList readYamlOrDieCommon(const BenchmarkResultContext &Context,
                                        llvm::StringRef Filename) {
  std::unique_ptr<llvm::MemoryBuffer> MemBuffer = llvm::cantFail(
      llvm::errorOrToExpected(llvm::MemoryBuffer::getFile(Filename)));
  // YAML IO requires a mutable pointer to Context but we guarantee to not
  // modify it.
  llvm::yaml::Input Yin(*MemBuffer,
                        const_cast<BenchmarkResultContext *>(&Context));
  ObjectOrList Benchmark;
  Yin >> Benchmark;
  return Benchmark;
}

InstructionBenchmark
InstructionBenchmark::readYamlOrDie(const BenchmarkResultContext &Context,
                                    llvm::StringRef Filename) {
  return readYamlOrDieCommon<InstructionBenchmark>(Context, Filename);
}

std::vector<InstructionBenchmark>
InstructionBenchmark::readYamlsOrDie(const BenchmarkResultContext &Context,
                                     llvm::StringRef Filename) {
  return readYamlOrDieCommon<std::vector<InstructionBenchmark>>(Context,
                                                                Filename);
}

void InstructionBenchmark::writeYamlTo(const BenchmarkResultContext &Context,
                                       llvm::raw_ostream &S) {
  // YAML IO requires a mutable pointer to Context but we guarantee to not
  // modify it.
  llvm::yaml::Output Yout(S, const_cast<BenchmarkResultContext *>(&Context));
  Yout << *this;
}

void InstructionBenchmark::readYamlFrom(const BenchmarkResultContext &Context,
                                        llvm::StringRef InputContent) {
  // YAML IO requires a mutable pointer to Context but we guarantee to not
  // modify it.
  llvm::yaml::Input Yin(InputContent,
                        const_cast<BenchmarkResultContext *>(&Context));
  Yin >> *this;
}

// FIXME: Change the API to let the caller handle errors.
void InstructionBenchmark::writeYamlOrDie(const BenchmarkResultContext &Context,
                                          const llvm::StringRef Filename) {
  if (Filename == "-") {
    writeYamlTo(Context, llvm::outs());
  } else {
    int ResultFD = 0;
    llvm::cantFail(llvm::errorCodeToError(
        openFileForWrite(Filename, ResultFD, llvm::sys::fs::F_Text)));
    llvm::raw_fd_ostream Ostr(ResultFD, true /*shouldClose*/);
    writeYamlTo(Context, Ostr);
  }
}

void BenchmarkMeasureStats::push(const BenchmarkMeasure &BM) {
  if (Key.empty())
    Key = BM.Key;
  assert(Key == BM.Key);
  ++NumValues;
  SumValues += BM.Value;
  MaxValue = std::max(MaxValue, BM.Value);
  MinValue = std::min(MinValue, BM.Value);
}

} // namespace exegesis
