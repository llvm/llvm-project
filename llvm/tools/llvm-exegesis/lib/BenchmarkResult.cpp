//===-- BenchmarkResult.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BenchmarkResult.h"
#include "BenchmarkRunner.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/bit.h"
#include "llvm/ObjectYAML/YAML.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

static constexpr const char kIntegerPrefix[] = "i_0x";
static constexpr const char kDoublePrefix[] = "f_";
static constexpr const char kInvalidOperand[] = "INVALID";
static constexpr llvm::StringLiteral kNoRegister("%noreg");

namespace llvm {

namespace {

// A mutable struct holding an LLVMState that can be passed through the
// serialization process to encode/decode registers and instructions.
struct YamlContext {
  YamlContext(const exegesis::LLVMState &State)
      : State(&State), ErrorStream(LastError),
        OpcodeNameToOpcodeIdx(
            generateOpcodeNameToOpcodeIdxMapping(State.getInstrInfo())),
        RegNameToRegNo(generateRegNameToRegNoMapping(State.getRegInfo())) {}

  static llvm::StringMap<unsigned>
  generateOpcodeNameToOpcodeIdxMapping(const llvm::MCInstrInfo &InstrInfo) {
    llvm::StringMap<unsigned> Map(InstrInfo.getNumOpcodes());
    for (unsigned I = 0, E = InstrInfo.getNumOpcodes(); I < E; ++I)
      Map[InstrInfo.getName(I)] = I;
    assert(Map.size() == InstrInfo.getNumOpcodes() && "Size prediction failed");
    return Map;
  };

  llvm::StringMap<unsigned>
  generateRegNameToRegNoMapping(const llvm::MCRegisterInfo &RegInfo) {
    llvm::StringMap<unsigned> Map(RegInfo.getNumRegs());
    // Special-case RegNo 0, which would otherwise be spelled as ''.
    Map[kNoRegister] = 0;
    for (unsigned I = 1, E = RegInfo.getNumRegs(); I < E; ++I)
      Map[RegInfo.getName(I)] = I;
    assert(Map.size() == RegInfo.getNumRegs() && "Size prediction failed");
    return Map;
  };

  void serializeMCInst(const llvm::MCInst &MCInst, llvm::raw_ostream &OS) {
    OS << getInstrName(MCInst.getOpcode());
    for (const auto &Op : MCInst) {
      OS << ' ';
      serializeMCOperand(Op, OS);
    }
  }

  void deserializeMCInst(llvm::StringRef String, llvm::MCInst &Value) {
    llvm::SmallVector<llvm::StringRef, 16> Pieces;
    String.split(Pieces, " ", /* MaxSplit */ -1, /* KeepEmpty */ false);
    if (Pieces.empty()) {
      ErrorStream << "Unknown Instruction: '" << String << "'\n";
      return;
    }
    bool ProcessOpcode = true;
    for (llvm::StringRef Piece : Pieces) {
      if (ProcessOpcode)
        Value.setOpcode(getInstrOpcode(Piece));
      else
        Value.addOperand(deserializeMCOperand(Piece));
      ProcessOpcode = false;
    }
  }

  std::string &getLastError() { return ErrorStream.str(); }

  llvm::raw_string_ostream &getErrorStream() { return ErrorStream; }

  llvm::StringRef getRegName(unsigned RegNo) {
    // Special case: RegNo 0 is NoRegister. We have to deal with it explicitly.
    if (RegNo == 0)
      return kNoRegister;
    const llvm::StringRef RegName = State->getRegInfo().getName(RegNo);
    if (RegName.empty())
      ErrorStream << "No register with enum value '" << RegNo << "'\n";
    return RegName;
  }

  llvm::Optional<unsigned> getRegNo(llvm::StringRef RegName) {
    auto Iter = RegNameToRegNo.find(RegName);
    if (Iter != RegNameToRegNo.end())
      return Iter->second;
    ErrorStream << "No register with name '" << RegName << "'\n";
    return llvm::None;
  }

private:
  void serializeIntegerOperand(llvm::raw_ostream &OS, int64_t Value) {
    OS << kIntegerPrefix;
    OS.write_hex(llvm::bit_cast<uint64_t>(Value));
  }

  bool tryDeserializeIntegerOperand(llvm::StringRef String, int64_t &Value) {
    if (!String.consume_front(kIntegerPrefix))
      return false;
    return !String.consumeInteger(16, Value);
  }

  void serializeFPOperand(llvm::raw_ostream &OS, double Value) {
    OS << kDoublePrefix << llvm::format("%la", Value);
  }

  bool tryDeserializeFPOperand(llvm::StringRef String, double &Value) {
    if (!String.consume_front(kDoublePrefix))
      return false;
    char *EndPointer = nullptr;
    Value = strtod(String.begin(), &EndPointer);
    return EndPointer == String.end();
  }

  void serializeMCOperand(const llvm::MCOperand &MCOperand,
                          llvm::raw_ostream &OS) {
    if (MCOperand.isReg()) {
      OS << getRegName(MCOperand.getReg());
    } else if (MCOperand.isImm()) {
      serializeIntegerOperand(OS, MCOperand.getImm());
    } else if (MCOperand.isFPImm()) {
      serializeFPOperand(OS, MCOperand.getFPImm());
    } else {
      OS << kInvalidOperand;
    }
  }

  llvm::MCOperand deserializeMCOperand(llvm::StringRef String) {
    assert(!String.empty());
    int64_t IntValue = 0;
    double DoubleValue = 0;
    if (tryDeserializeIntegerOperand(String, IntValue))
      return llvm::MCOperand::createImm(IntValue);
    if (tryDeserializeFPOperand(String, DoubleValue))
      return llvm::MCOperand::createFPImm(DoubleValue);
    if (auto RegNo = getRegNo(String))
      return llvm::MCOperand::createReg(*RegNo);
    if (String != kInvalidOperand)
      ErrorStream << "Unknown Operand: '" << String << "'\n";
    return {};
  }

  llvm::StringRef getInstrName(unsigned InstrNo) {
    const llvm::StringRef InstrName = State->getInstrInfo().getName(InstrNo);
    if (InstrName.empty())
      ErrorStream << "No opcode with enum value '" << InstrNo << "'\n";
    return InstrName;
  }

  unsigned getInstrOpcode(llvm::StringRef InstrName) {
    auto Iter = OpcodeNameToOpcodeIdx.find(InstrName);
    if (Iter != OpcodeNameToOpcodeIdx.end())
      return Iter->second;
    ErrorStream << "No opcode with name '" << InstrName << "'\n";
    return 0;
  }

  const llvm::exegesis::LLVMState *State;
  std::string LastError;
  llvm::raw_string_ostream ErrorStream;
  const llvm::StringMap<unsigned> OpcodeNameToOpcodeIdx;
  const llvm::StringMap<unsigned> RegNameToRegNo;
};
} // namespace

// Defining YAML traits for IO.
namespace yaml {

static YamlContext &getTypedContext(void *Ctx) {
  return *reinterpret_cast<YamlContext *>(Ctx);
}

// std::vector<llvm::MCInst> will be rendered as a list.
template <> struct SequenceElementTraits<llvm::MCInst> {
  static const bool flow = false;
};

template <> struct ScalarTraits<llvm::MCInst> {

  static void output(const llvm::MCInst &Value, void *Ctx,
                     llvm::raw_ostream &Out) {
    getTypedContext(Ctx).serializeMCInst(Value, Out);
  }

  static StringRef input(StringRef Scalar, void *Ctx, llvm::MCInst &Value) {
    YamlContext &Context = getTypedContext(Ctx);
    Context.deserializeMCInst(Scalar, Value);
    return Context.getLastError();
  }

  // By default strings are quoted only when necessary.
  // We force the use of single quotes for uniformity.
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
    if (!Io.outputting()) {
      // For backward compatibility, interpret debug_string as a key.
      Io.mapOptional("debug_string", Obj.Key);
    }
    Io.mapRequired("value", Obj.PerInstructionValue);
    Io.mapOptional("per_snippet_value", Obj.PerSnippetValue);
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
    Io.enumCase(Value, "inverse_throughput",
                exegesis::InstructionBenchmark::InverseThroughput);
  }
};

// std::vector<exegesis::RegisterValue> will be rendered as a list.
template <> struct SequenceElementTraits<exegesis::RegisterValue> {
  static const bool flow = false;
};

template <> struct ScalarTraits<exegesis::RegisterValue> {
  static constexpr const unsigned kRadix = 16;
  static constexpr const bool kSigned = false;

  static void output(const exegesis::RegisterValue &RV, void *Ctx,
                     llvm::raw_ostream &Out) {
    YamlContext &Context = getTypedContext(Ctx);
    Out << Context.getRegName(RV.Register) << "=0x"
        << RV.Value.toString(kRadix, kSigned);
  }

  static StringRef input(StringRef String, void *Ctx,
                         exegesis::RegisterValue &RV) {
    llvm::SmallVector<llvm::StringRef, 2> Pieces;
    String.split(Pieces, "=0x", /* MaxSplit */ -1,
                 /* KeepEmpty */ false);
    YamlContext &Context = getTypedContext(Ctx);
    llvm::Optional<unsigned> RegNo;
    if (Pieces.size() == 2 && (RegNo = Context.getRegNo(Pieces[0]))) {
      RV.Register = *RegNo;
      const unsigned BitsNeeded = llvm::APInt::getBitsNeeded(Pieces[1], kRadix);
      RV.Value = llvm::APInt(BitsNeeded, Pieces[1], kRadix);
    } else {
      Context.getErrorStream()
          << "Unknown initial register value: '" << String << "'";
    }
    return Context.getLastError();
  }

  static QuotingType mustQuote(StringRef) { return QuotingType::Single; }

  static const bool flow = true;
};

template <>
struct MappingContextTraits<exegesis::InstructionBenchmarkKey, YamlContext> {
  static void mapping(IO &Io, exegesis::InstructionBenchmarkKey &Obj,
                      YamlContext &Context) {
    Io.setContext(&Context);
    Io.mapRequired("instructions", Obj.Instructions);
    Io.mapOptional("config", Obj.Config);
    Io.mapRequired("register_initial_values", Obj.RegisterInitialValues);
  }
};

template <>
struct MappingContextTraits<exegesis::InstructionBenchmark, YamlContext> {
  struct NormalizedBinary {
    NormalizedBinary(IO &io) {}
    NormalizedBinary(IO &, std::vector<uint8_t> &Data) : Binary(Data) {}
    std::vector<uint8_t> denormalize(IO &) {
      std::vector<uint8_t> Data;
      std::string Str;
      raw_string_ostream OSS(Str);
      Binary.writeAsBinary(OSS);
      OSS.flush();
      Data.assign(Str.begin(), Str.end());
      return Data;
    }

    BinaryRef Binary;
  };

  static void mapping(IO &Io, exegesis::InstructionBenchmark &Obj,
                      YamlContext &Context) {
    Io.mapRequired("mode", Obj.Mode);
    Io.mapRequired("key", Obj.Key, Context);
    Io.mapRequired("cpu_name", Obj.CpuName);
    Io.mapRequired("llvm_triple", Obj.LLVMTriple);
    Io.mapRequired("num_repetitions", Obj.NumRepetitions);
    Io.mapRequired("measurements", Obj.Measurements);
    Io.mapRequired("error", Obj.Error);
    Io.mapOptional("info", Obj.Info);
    // AssembledSnippet
    MappingNormalization<NormalizedBinary, std::vector<uint8_t>> BinaryString(
        Io, Obj.AssembledSnippet);
    Io.mapOptional("assembled_snippet", BinaryString->Binary);
  }
};

} // namespace yaml

namespace exegesis {

llvm::Expected<InstructionBenchmark>
InstructionBenchmark::readYaml(const LLVMState &State,
                               llvm::StringRef Filename) {
  if (auto ExpectedMemoryBuffer =
          llvm::errorOrToExpected(llvm::MemoryBuffer::getFile(Filename))) {
    llvm::yaml::Input Yin(*ExpectedMemoryBuffer.get());
    YamlContext Context(State);
    InstructionBenchmark Benchmark;
    if (Yin.setCurrentDocument())
      llvm::yaml::yamlize(Yin, Benchmark, /*unused*/ true, Context);
    if (!Context.getLastError().empty())
      return llvm::make_error<BenchmarkFailure>(Context.getLastError());
    return Benchmark;
  } else {
    return ExpectedMemoryBuffer.takeError();
  }
}

llvm::Expected<std::vector<InstructionBenchmark>>
InstructionBenchmark::readYamls(const LLVMState &State,
                                llvm::StringRef Filename) {
  if (auto ExpectedMemoryBuffer =
          llvm::errorOrToExpected(llvm::MemoryBuffer::getFile(Filename))) {
    llvm::yaml::Input Yin(*ExpectedMemoryBuffer.get());
    YamlContext Context(State);
    std::vector<InstructionBenchmark> Benchmarks;
    while (Yin.setCurrentDocument()) {
      Benchmarks.emplace_back();
      yamlize(Yin, Benchmarks.back(), /*unused*/ true, Context);
      if (Yin.error())
        return llvm::errorCodeToError(Yin.error());
      if (!Context.getLastError().empty())
        return llvm::make_error<BenchmarkFailure>(Context.getLastError());
      Yin.nextDocument();
    }
    return Benchmarks;
  } else {
    return ExpectedMemoryBuffer.takeError();
  }
}

llvm::Error InstructionBenchmark::writeYamlTo(const LLVMState &State,
                                              llvm::raw_ostream &OS) {
  auto Cleanup = make_scope_exit([&] { OS.flush(); });
  llvm::yaml::Output Yout(OS, nullptr /*Ctx*/, 200 /*WrapColumn*/);
  YamlContext Context(State);
  Yout.beginDocuments();
  llvm::yaml::yamlize(Yout, *this, /*unused*/ true, Context);
  if (!Context.getLastError().empty())
    return llvm::make_error<BenchmarkFailure>(Context.getLastError());
  Yout.endDocuments();
  return Error::success();
}

llvm::Error InstructionBenchmark::readYamlFrom(const LLVMState &State,
                                               llvm::StringRef InputContent) {
  llvm::yaml::Input Yin(InputContent);
  YamlContext Context(State);
  if (Yin.setCurrentDocument())
    llvm::yaml::yamlize(Yin, *this, /*unused*/ true, Context);
  if (!Context.getLastError().empty())
    return llvm::make_error<BenchmarkFailure>(Context.getLastError());
  return Error::success();
}

llvm::Error InstructionBenchmark::writeYaml(const LLVMState &State,
                                            const llvm::StringRef Filename) {
  if (Filename == "-") {
    if (auto Err = writeYamlTo(State, llvm::outs()))
      return Err;
  } else {
    int ResultFD = 0;
    if (auto E = llvm::errorCodeToError(
            openFileForWrite(Filename, ResultFD, llvm::sys::fs::CD_CreateAlways,
                             llvm::sys::fs::F_Text))) {
      return E;
    }
    llvm::raw_fd_ostream Ostr(ResultFD, true /*shouldClose*/);
    if (auto Err = writeYamlTo(State, Ostr))
      return Err;
  }
  return llvm::Error::success();
}

void PerInstructionStats::push(const BenchmarkMeasure &BM) {
  if (Key.empty())
    Key = BM.Key;
  assert(Key == BM.Key);
  ++NumValues;
  SumValues += BM.PerInstructionValue;
  MaxValue = std::max(MaxValue, BM.PerInstructionValue);
  MinValue = std::min(MinValue, BM.PerInstructionValue);
}

} // namespace exegesis
} // namespace llvm
