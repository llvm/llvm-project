//===- CallSiteInfo.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_GSYM_CALLSITEINFO_H
#define LLVM_DEBUGINFO_GSYM_CALLSITEINFO_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/DebugInfo/GSYM/ExtractRanges.h"
#include "llvm/Support/YAMLParser.h"
#include <string>
#include <unordered_map>
#include <vector>

namespace llvm {
class DataExtractor;
class raw_ostream;
class StringTableBuilder;
class CachedHashStringRef;

namespace yaml {
struct CallSiteYAML;
struct FunctionYAML;
struct FunctionsYAML;
} // namespace yaml

namespace gsym {
class FileWriter;
struct FunctionInfo;
struct CallSiteInfo {
public:
  enum Flags : uint8_t {
    None = 0,
    // This flag specifies that the call site can only call a function within
    // the same link unit as the call site.
    InternalCall = 1 << 0,
    // This flag specifies that the call site can only call a function outside
    // the link unit that the call site is in.
    ExternalCall = 1 << 1,
  };

  /// The return address of the call site.
  uint64_t ReturnAddress;

  /// Offsets into the string table for function names regex patterns.
  std::vector<uint32_t> MatchRegex;

  /// Bitwise OR of CallSiteInfo::Flags values
  uint8_t Flags;

  /// Decode a CallSiteInfo object from a binary data stream.
  ///
  /// \param Data The binary stream to read the data from.
  /// \param Offset The current offset within the data stream.
  /// \param BaseAddr The base address for decoding (unused here but included
  /// for consistency).
  ///
  /// \returns A CallSiteInfo or an error describing the issue.
  static llvm::Expected<CallSiteInfo>
  decode(DataExtractor &Data, uint64_t &Offset, uint64_t BaseAddr);

  /// Encode this CallSiteInfo object into a FileWriter stream.
  ///
  /// \param O The binary stream to write the data to.
  /// \returns An error object that indicates success or failure.
  llvm::Error encode(FileWriter &O) const;
};

struct CallSiteInfoCollection {
public:
  std::vector<CallSiteInfo> CallSites;

  void clear() { CallSites.clear(); }

  /// Query if a CallSiteInfoCollection object is valid.
  ///
  /// \returns True if the collection is not empty.
  bool isValid() const { return !CallSites.empty(); }

  /// Decode a CallSiteInfoCollection object from a binary data stream.
  ///
  /// \param Data The binary stream to read the data from.
  /// \param BaseAddr The base address for decoding (unused here but included
  /// for consistency).
  ///
  /// \returns A CallSiteInfoCollection or an error describing the issue.
  static llvm::Expected<CallSiteInfoCollection> decode(DataExtractor &Data,
                                                       uint64_t BaseAddr);

  /// Encode this CallSiteInfoCollection object into a FileWriter stream.
  ///
  /// \param O The binary stream to write the data to.
  /// \returns An error object that indicates success or failure.
  llvm::Error encode(FileWriter &O) const;
};

bool operator==(const CallSiteInfoCollection &LHS,
                const CallSiteInfoCollection &RHS);

bool operator==(const CallSiteInfo &LHS, const CallSiteInfo &RHS);

class CallSiteInfoLoader {
public:
  /// Constructor that initializes the CallSiteInfoLoader with necessary data
  /// structures.
  ///
  /// \param StringOffsetMap A reference to a DenseMap that maps existing string
  /// offsets to CachedHashStringRef. \param StrTab A reference to a
  /// StringTableBuilder used for managing looking up and creating new strings.
  /// \param StringStorage A reference to a StringSet for storing the data for
  /// generated strings.
  CallSiteInfoLoader(DenseMap<uint64_t, CachedHashStringRef> &StringOffsetMap,
                     StringTableBuilder &StrTab, StringSet<> &StringStorage)
      : StringOffsetMap(StringOffsetMap), StrTab(StrTab),
        StringStorage(StringStorage) {}

  /// Loads call site information from a YAML file and populates the provided
  /// FunctionInfo vector.
  ///
  /// This method reads the specified YAML file, parses its content, and updates
  /// the `Funcs` vector with call site information based on the YAML data.
  ///
  /// \param Funcs A reference to a vector of FunctionInfo objects to be
  /// populated.
  /// \param YAMLFile A StringRef representing the path to the YAML
  /// file to be loaded.
  ///
  /// \returns An `llvm::Error` indicating success or describing any issues
  /// encountered during the loading process.
  llvm::Error loadYAML(std::vector<FunctionInfo> &Funcs, StringRef YAMLFile);

private:
  /// Retrieves an existing string from the StringOffsetMap using the provided
  /// offset.
  ///
  /// \param offset A 32-bit unsigned integer representing the offset of the
  /// string.
  ///
  /// \returns A StringRef corresponding to the string for the given offset.
  ///
  /// \note This method asserts that the offset exists in the StringOffsetMap.
  StringRef stringFromOffset(uint32_t offset) const;

  /// Obtains the offset corresponding to a given string in the StrTab. If the
  /// string does not already exist, it is created.
  ///
  /// \param str A StringRef representing the string for which the offset is
  /// requested.
  ///
  /// \returns A 32-bit unsigned integer representing the offset of the string.
  uint32_t offsetFromString(StringRef str);

  /// Reads the content of the YAML file specified by `YAMLFile` into
  /// `yamlContent`.
  ///
  /// \param YAMLFile A StringRef representing the path to the YAML file.
  /// \param Buffer The memory buffer containing the YAML content.
  ///
  /// \returns An `llvm::Error` indicating success or describing any issues
  /// encountered while reading the file.
  llvm::Error readYAMLFile(StringRef YAMLFile,
                           std::unique_ptr<llvm::MemoryBuffer> &Buffer);

  /// Parses the YAML content and populates `functionsYAML` with the parsed
  /// data.
  ///
  /// \param Buffer The memory buffer containing the YAML content.
  /// \param functionsYAML A reference to an llvm::yaml::FunctionsYAML object to
  /// be populated.
  ///
  /// \returns An `llvm::Error` indicating success or describing any issues
  /// encountered during parsing.
  llvm::Error parseYAML(llvm::MemoryBuffer &Buffer,
                        llvm::yaml::FunctionsYAML &functionsYAML);

  /// Builds a map from function names to FunctionInfo pointers based on the
  /// provided `Funcs` vector.
  ///
  /// \param Funcs A reference to a vector of FunctionInfo objects.
  ///
  /// \returns An unordered_map mapping function names (std::string) to their
  /// corresponding FunctionInfo pointers.
  std::unordered_map<std::string, FunctionInfo *>
  buildFunctionMap(std::vector<FunctionInfo> &Funcs);

  /// Processes the parsed YAML functions and updates the `FuncMap` accordingly.
  ///
  /// \param functionsYAML A constant reference to an llvm::yaml::FunctionsYAML
  /// object containing parsed YAML data.
  /// \param FuncMap A reference to an unordered_map mapping function names to
  /// FunctionInfo pointers.
  /// \param YAMLFile A StringRef representing the name of the YAML file (used
  /// for error messages).
  ///
  /// \returns An `llvm::Error` indicating success or describing any issues
  /// encountered during processing.
  llvm::Error
  processYAMLFunctions(const llvm::yaml::FunctionsYAML &functionsYAML,
                       std::unordered_map<std::string, FunctionInfo *> &FuncMap,
                       StringRef YAMLFile);

  /// Map of existing string offsets to CachedHashStringRef.
  DenseMap<uint64_t, CachedHashStringRef> &StringOffsetMap;

  /// The gSYM string table builder.
  StringTableBuilder &StrTab;

  /// The gSYM string storage - we store generated strings here.
  StringSet<> &StringStorage;
};

raw_ostream &operator<<(raw_ostream &OS, const CallSiteInfo &CSI);
raw_ostream &operator<<(raw_ostream &OS, const CallSiteInfoCollection &CSIC);

} // namespace gsym
} // namespace llvm

#endif // LLVM_DEBUGINFO_GSYM_CALLSITEINFO_H
