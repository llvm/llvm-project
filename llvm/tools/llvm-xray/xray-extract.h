//===- xray-extract.h - XRay Instrumentation Map Extraction ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Defines the interface for extracting the instrumentation map from an
// XRay-instrumented binary.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_XRAY_EXTRACT_H
#define LLVM_TOOLS_XRAY_EXTRACT_H

#include <deque>
#include <map>
#include <string>
#include <unordered_map>

#include "xray-sleds.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
namespace xray {

class InstrumentationMapExtractor {
public:
  typedef std::unordered_map<int32_t, uint64_t> FunctionAddressMap;
  typedef std::unordered_map<uint64_t, int32_t> FunctionAddressReverseMap;

  enum class InputFormats { ELF, YAML };

private:
  std::deque<SledEntry> Sleds;
  FunctionAddressMap FunctionAddresses;
  FunctionAddressReverseMap FunctionIds;

public:
  /// Loads the instrumentation map from |Filename|. Updates |EC| in case there
  /// were errors encountered opening the file. |Format| defines what the input
  /// instrumentation map is in.
  InstrumentationMapExtractor(std::string Filename, InputFormats Format,
                              Error &EC);

  const FunctionAddressMap &getFunctionAddresses() { return FunctionAddresses; }

  /// Exports the loaded function address map as YAML through |OS|.
  void exportAsYAML(raw_ostream &OS);
};

} // namespace xray
} // namespace llvm

#endif // LLVM_TOOLS_XRAY_EXTRACT_H
