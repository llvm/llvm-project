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

static constexpr const char kIntegerFormat[] = "i_0x%" PRId64 "x";
static constexpr const char kDoubleFormat[] = "f_%la";

static void serialize(const exegesis::BenchmarkResultContext &Context,
                      const llvm::MCOperand &MCOperand, llvm::raw_ostream &OS) {
  if (MCOperand.isReg()) {
    OS << Context.getRegName(MCOperand.getReg());
  } else if (MCOperand.isImm()) {
    OS << llvm::format(kIntegerFormat, MCOperand.getImm());
  } else if (MCOperand.isFPImm()) {
    OS << llvm::format(kDoubleFormat, MCOperand.getFPImm());
  } else {
    OS << "INVALID";
  }
}

static void serialize(const exegesis::BenchmarkResultContext &Context,
                      const llvm::MCInst &MCInst, llvm::raw_ostream &OS) {
  OS << Context.getInstrName(MCInst.getOpcode());
  for (const auto &Op : MCInst) {
    OS << ' ';
    serialize(Context, Op, OS);
  }
}

static llvm::MCOperand
deserialize(const exegesis::BenchmarkResultContext &Context,
            llvm::StringRef String) {
  assert(!String.empty());
  int64_t IntValue = 0;
  double DoubleValue = 0;
  if (sscanf(String.data(), kIntegerFormat, &IntValue) == 1)
    return llvm::MCOperand::createImm(IntValue);
  if (sscanf(String.data(), kDoubleFormat, &DoubleValue) == 1)
    return llvm::MCOperand::createFPImm(DoubleValue);
  if (unsigned RegNo = Context.getRegNo(String)) // Returns 0 if invalid.
    return llvm::MCOperand::createReg(RegNo);
  return {};
}

static llvm::StringRef
deserialize(const exegesis::BenchmarkResultContext &Context,
            llvm::StringRef String, llvm::MCInst &Value) {
  llvm::SmallVector<llvm::StringRef, 8> Pieces;
  String.split(Pieces, " ");
  if (Pieces.empty())
    return "Invalid Instruction";
  bool ProcessOpcode = true;
  for (llvm::StringRef Piece : Pieces) {
    if (ProcessOpcode) {
      ProcessOpcode = false;
      Value.setOpcode(Context.getInstrOpcode(Piece));
      if (Value.getOpcode() == 0)
        return "Unknown Opcode Name";
    } else {
      Value.addOperand(deserialize(Context, Piece));
    }
  }
  return {};
}

// YAML IO requires a mutable pointer to Context but we guarantee to not
// modify it.
static void *getUntypedContext(const exegesis::BenchmarkResultContext &Ctx) {
  return const_cast<exegesis::BenchmarkResultContext *>(&Ctx);
}

static const exegesis::BenchmarkResultContext &getTypedContext(void *Ctx) {
  assert(Ctx);
  return *static_cast<const exegesis::BenchmarkResultContext *>(Ctx);
}

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
    serialize(getTypedContext(Ctx), Value, Out);
  }

  static StringRef input(StringRef Scalar, void *Ctx, llvm::MCInst &Value) {
    return deserialize(getTypedContext(Ctx), Scalar, Value);
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
static llvm::Expected<ObjectOrList>
readYamlCommon(const BenchmarkResultContext &Context,
               llvm::StringRef Filename) {
  if (auto ExpectedMemoryBuffer =
          llvm::errorOrToExpected(llvm::MemoryBuffer::getFile(Filename))) {
    std::unique_ptr<llvm::MemoryBuffer> MemoryBuffer =
        std::move(ExpectedMemoryBuffer.get());
    llvm::yaml::Input Yin(*MemoryBuffer, getUntypedContext(Context));
    ObjectOrList Benchmark;
    Yin >> Benchmark;
    return Benchmark;
  } else {
    return ExpectedMemoryBuffer.takeError();
  }
}

llvm::Expected<InstructionBenchmark>
InstructionBenchmark::readYaml(const BenchmarkResultContext &Context,
                               llvm::StringRef Filename) {
  return readYamlCommon<InstructionBenchmark>(Context, Filename);
}

llvm::Expected<std::vector<InstructionBenchmark>>
InstructionBenchmark::readYamls(const BenchmarkResultContext &Context,
                                llvm::StringRef Filename) {
  return readYamlCommon<std::vector<InstructionBenchmark>>(Context, Filename);
}

void InstructionBenchmark::writeYamlTo(const BenchmarkResultContext &Context,
                                       llvm::raw_ostream &OS) {
  llvm::yaml::Output Yout(OS, getUntypedContext(Context));
  Yout << *this;
}

void InstructionBenchmark::readYamlFrom(const BenchmarkResultContext &Context,
                                        llvm::StringRef InputContent) {
  llvm::yaml::Input Yin(InputContent, getUntypedContext(Context));
  Yin >> *this;
}

llvm::Error
InstructionBenchmark::writeYaml(const BenchmarkResultContext &Context,
                                const llvm::StringRef Filename) {
  if (Filename == "-") {
    writeYamlTo(Context, llvm::outs());
  } else {
    int ResultFD = 0;
    if (auto E = llvm::errorCodeToError(
            openFileForWrite(Filename, ResultFD, llvm::sys::fs::CD_CreateAlways,
                             llvm::sys::fs::F_Text))) {
      return E;
    }
    llvm::raw_fd_ostream Ostr(ResultFD, true /*shouldClose*/);
    writeYamlTo(Context, Ostr);
  }
  return llvm::Error::success();
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
