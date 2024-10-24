//===- CallSiteInfo.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_GSYM_CALLSITEINFO_H
#define LLVM_DEBUGINFO_GSYM_CALLSITEINFO_H

#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Error.h"
#include <vector>

namespace llvm {
class DataExtractor;
class raw_ostream;

namespace yaml {
struct FunctionsYAML;
} // namespace yaml

namespace gsym {
class FileWriter;
class GsymCreator;
struct FunctionInfo;
struct CallSiteInfo {
  enum Flags : uint8_t {
    None = 0,
    // This flag specifies that the call site can only call a function within
    // the same link unit as the call site.
    InternalCall = 1 << 0,
    // This flag specifies that the call site can only call a function outside
    // the link unit that the call site is in.
    ExternalCall = 1 << 1,

    LLVM_MARK_AS_BITMASK_ENUM(/*LargestValue*/ ExternalCall),
  };

  /// The return address of the call site.
  uint64_t ReturnAddress = 0;

  /// Offsets into the string table for function names regex patterns.
  std::vector<uint32_t> MatchRegex;

  /// Bitwise OR of CallSiteInfo::Flags values
  uint8_t Flags = CallSiteInfo::Flags::None;

  /// Decode a CallSiteInfo object from a binary data stream.
  ///
  /// \param Data The binary stream to read the data from.
  /// \param Offset The current offset within the data stream.
  /// \returns A CallSiteInfo or an error describing the issue.
  static llvm::Expected<CallSiteInfo> decode(DataExtractor &Data,
                                             uint64_t &Offset);

  /// Encode this CallSiteInfo object into a FileWriter stream.
  ///
  /// \param O The binary stream to write the data to.
  /// \returns An error object that indicates success or failure.
  llvm::Error encode(FileWriter &O) const;
};

struct CallSiteInfoCollection {
  std::vector<CallSiteInfo> CallSites;

  /// Decode a CallSiteInfoCollection object from a binary data stream.
  ///
  /// \param Data The binary stream to read the data from.
  /// \returns A CallSiteInfoCollection or an error describing the issue.
  static llvm::Expected<CallSiteInfoCollection> decode(DataExtractor &Data);

  /// Encode this CallSiteInfoCollection object into a FileWriter stream.
  ///
  /// \param O The binary stream to write the data to.
  /// \returns An error object that indicates success or failure.
  llvm::Error encode(FileWriter &O) const;
};

class CallSiteInfoLoader {
public:
  /// Constructor that initializes the CallSiteInfoLoader with necessary data
  /// structures.
  ///
  /// \param GCreator A reference to the GsymCreator.
  CallSiteInfoLoader(GsymCreator &GCreator) : GCreator(GCreator) {}

  /// This method reads the specified YAML file, parses its content, and updates
  /// the `Funcs` vector with call site information based on the YAML data.
  ///
  /// \param Funcs A reference to a vector of FunctionInfo objects to be
  /// populated.
  /// \param YAMLFile A StringRef representing the path to the YAML
  /// file to be loaded.
  /// \returns An `llvm::Error` indicating success or describing any issues
  /// encountered during the loading process.
  llvm::Error loadYAML(std::vector<FunctionInfo> &Funcs, StringRef YAMLFile);

private:
  /// Builds a map from function names to FunctionInfo pointers based on the
  /// provided `Funcs` vector.
  ///
  /// \param Funcs A reference to a vector of FunctionInfo objects.
  /// \returns A StringMap mapping function names (StringRef) to their
  /// corresponding FunctionInfo pointers.
  StringMap<FunctionInfo *> buildFunctionMap(std::vector<FunctionInfo> &Funcs);

  /// Processes the parsed YAML functions and updates the `FuncMap` accordingly.
  ///
  /// \param functionsYAML A constant reference to an llvm::yaml::FunctionsYAML
  /// object containing parsed YAML data.
  /// \param FuncMap A reference to a StringMap mapping function names to
  /// FunctionInfo pointers.
  /// \returns An `llvm::Error` indicating success or describing any issues
  /// encountered during processing.
  llvm::Error
  processYAMLFunctions(const llvm::yaml::FunctionsYAML &functionsYAML,
                       StringMap<FunctionInfo *> &FuncMap);

  /// Reference to the parent Gsym Creator object.
  GsymCreator &GCreator;
};

raw_ostream &operator<<(raw_ostream &OS, const CallSiteInfo &CSI);
raw_ostream &operator<<(raw_ostream &OS, const CallSiteInfoCollection &CSIC);

} // namespace gsym
} // namespace llvm

#endif // LLVM_DEBUGINFO_GSYM_CALLSITEINFO_H
