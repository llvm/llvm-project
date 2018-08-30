//===- FDRTraceWriter.cpp - XRay FDR Trace Writer ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Test a utility that can write out XRay FDR Mode formatted trace files.
//
//===----------------------------------------------------------------------===//
#include "llvm/XRay/FDRTraceWriter.h"
#include <tuple>

namespace llvm {
namespace xray {

namespace {

struct alignas(32) FileHeader {
  uint16_t Version;
  uint16_t Type;
  bool ConstantTSC : 1;
  bool NonstopTSC : 1;
  alignas(8) uint64_t CycleFrequency;
  char FreeForm[16];
};

struct MetadataBlob {
  uint8_t Type : 1;
  uint8_t RecordKind : 7;
  char Data[15];
};

struct FunctionDeltaBlob {
  uint8_t Type : 1;
  uint8_t RecordKind : 3;
  int FuncId : 28;
  uint32_t TSCDelta;
};

template <size_t Index> struct IndexedMemcpy {
  template <
      class Tuple,
      typename std::enable_if<
          (Index <
           std::tuple_size<typename std::remove_reference<Tuple>::type>::value),
          int>::type = 0>
  static void Copy(char *Dest, Tuple &&T) {
    auto Next = static_cast<char *>(std::memcpy(
                    Dest, reinterpret_cast<const char *>(&std::get<Index>(T)),
                    sizeof(std::get<Index>(T)))) +
                sizeof(std::get<Index>(T));
    IndexedMemcpy<Index + 1>::Copy(Next, T);
  }

  template <
      class Tuple,
      typename std::enable_if<
          (Index >=
           std::tuple_size<typename std::remove_reference<Tuple>::type>::value),
          int>::type = 0>
  static void Copy(char *, Tuple &&) {}
};

template <uint8_t Kind, class... Values>
Error writeMetadata(raw_ostream &OS, Values&&... Ds) {
  MetadataBlob B;
  B.Type = 1;
  B.RecordKind = Kind;
  std::memset(B.Data, 0, 15);
  auto T = std::make_tuple(std::forward<Values>(std::move(Ds))...);
  IndexedMemcpy<0>::Copy(B.Data, T);
  OS.write(reinterpret_cast<const char *>(&B), sizeof(MetadataBlob));
  return Error::success();
}

} // namespace

FDRTraceWriter::FDRTraceWriter(raw_ostream &O, const XRayFileHeader &H)
    : OS(O) {
  // We need to re-construct a header, by writing the fields we care about for
  // traces, in the format that the runtime would have written.
  FileHeader Raw;
  Raw.Version = H.Version;
  Raw.Type = H.Type;
  Raw.ConstantTSC = H.ConstantTSC;
  Raw.NonstopTSC = H.NonstopTSC;
  Raw.CycleFrequency = H.CycleFrequency;
  memcpy(&Raw.FreeForm, H.FreeFormData, 16);
  OS.write(reinterpret_cast<const char *>(&Raw), sizeof(XRayFileHeader));
}

FDRTraceWriter::~FDRTraceWriter() {}

Error FDRTraceWriter::visit(BufferExtents &R) {
  return writeMetadata<7u>(OS, R.size());
}

Error FDRTraceWriter::visit(WallclockRecord &R) {
  return writeMetadata<4u>(OS, R.seconds(), R.nanos());
}

Error FDRTraceWriter::visit(NewCPUIDRecord &R) {
  return writeMetadata<2u>(OS, R.cpuid());
}

Error FDRTraceWriter::visit(TSCWrapRecord &R) {
  return writeMetadata<3u>(OS, R.tsc());
}

Error FDRTraceWriter::visit(CustomEventRecord &R) {
  if (auto E = writeMetadata<5u>(OS, R.size(), R.tsc()))
    return E;
  OS.write(R.data().data(), R.data().size());
  return Error::success();
}

Error FDRTraceWriter::visit(CallArgRecord &R) {
  return writeMetadata<6u>(OS, R.arg());
}

Error FDRTraceWriter::visit(PIDRecord &R) {
  return writeMetadata<9u>(OS, R.pid());
}

Error FDRTraceWriter::visit(NewBufferRecord &R) {
  return writeMetadata<0u>(OS, R.tid());
}

Error FDRTraceWriter::visit(EndBufferRecord &R) {
  return writeMetadata<1u>(OS, 0);
}

Error FDRTraceWriter::visit(FunctionRecord &R) {
  FunctionDeltaBlob B;
  B.Type = 0;
  B.RecordKind = static_cast<uint8_t>(R.recordType());
  B.FuncId = R.functionId();
  B.TSCDelta = R.delta();
  OS.write(reinterpret_cast<const char *>(&B), sizeof(FunctionDeltaBlob));
  return Error::success();
}

} // namespace xray
} // namespace llvm
