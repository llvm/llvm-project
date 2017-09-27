//===- YAMLXRayRecord.h - XRay Record YAML Support Definitions ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Types and traits specialisations for YAML I/O of XRay log entries.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_XRAY_YAML_XRAY_RECORD_H
#define LLVM_XRAY_YAML_XRAY_RECORD_H

#include <type_traits>

#include "llvm/Support/YAMLTraits.h"
#include "llvm/XRay/XRayRecord.h"

namespace llvm {
namespace xray {

struct YAMLXRayFileHeader {
  uint16_t Version;
  uint16_t Type;
  bool ConstantTSC;
  bool NonstopTSC;
  uint64_t CycleFrequency;
};

struct YAMLXRayRecord {
  uint16_t RecordType;
  uint16_t CPU;
  RecordTypes Type;
  int32_t FuncId;
  std::string Function;
  uint64_t TSC;
  uint32_t TId;
  std::vector<uint64_t> CallArgs;
};

struct YAMLXRayTrace {
  YAMLXRayFileHeader Header;
  std::vector<YAMLXRayRecord> Records;
};

} // namespace xray

namespace yaml {

// YAML Traits
// -----------
template <> struct ScalarEnumerationTraits<xray::RecordTypes> {
  static void enumeration(IO &IO, xray::RecordTypes &Type) {
    IO.enumCase(Type, "function-enter", xray::RecordTypes::ENTER);
    IO.enumCase(Type, "function-exit", xray::RecordTypes::EXIT);
    IO.enumCase(Type, "function-tail-exit", xray::RecordTypes::TAIL_EXIT);
    IO.enumCase(Type, "function-enter-arg", xray::RecordTypes::ENTER_ARG);
  }
};

template <> struct MappingTraits<xray::YAMLXRayFileHeader> {
  static void mapping(IO &IO, xray::YAMLXRayFileHeader &Header) {
    IO.mapRequired("version", Header.Version);
    IO.mapRequired("type", Header.Type);
    IO.mapRequired("constant-tsc", Header.ConstantTSC);
    IO.mapRequired("nonstop-tsc", Header.NonstopTSC);
    IO.mapRequired("cycle-frequency", Header.CycleFrequency);
  }
};

template <> struct MappingTraits<xray::YAMLXRayRecord> {
  static void mapping(IO &IO, xray::YAMLXRayRecord &Record) {
    // FIXME: Make this type actually be descriptive
    IO.mapRequired("type", Record.RecordType);
    IO.mapRequired("func-id", Record.FuncId);
    IO.mapOptional("function", Record.Function);
    IO.mapOptional("args", Record.CallArgs);
    IO.mapRequired("cpu", Record.CPU);
    IO.mapRequired("thread", Record.TId);
    IO.mapRequired("kind", Record.Type);
    IO.mapRequired("tsc", Record.TSC);
  }

  static constexpr bool flow = true;
};

template <> struct SequenceTraits<std::vector<uint64_t>> {
  static constexpr bool flow = true;
  static size_t size(IO &IO, std::vector<uint64_t> &V) { return V.size(); }
  static uint64_t &element(IO &IO, std::vector<uint64_t> &V, size_t Index) {
    if (Index >= V.size())
      V.resize(Index + 1);
    return V[Index];
  }
};

template <> struct MappingTraits<xray::YAMLXRayTrace> {
  static void mapping(IO &IO, xray::YAMLXRayTrace &Trace) {
    // A trace file contains two parts, the header and the list of all the
    // trace records.
    IO.mapRequired("header", Trace.Header);
    IO.mapRequired("records", Trace.Records);
  }
};

} // namespace yaml
} // namespace llvm

LLVM_YAML_IS_SEQUENCE_VECTOR(xray::YAMLXRayRecord)

#endif // LLVM_XRAY_YAML_XRAY_RECORD_H
