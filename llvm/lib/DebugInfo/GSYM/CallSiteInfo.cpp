//===- CallSiteInfo.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/GSYM/CallSiteInfo.h"
#include "llvm/ADT/CachedHashString.h"
#include "llvm/DebugInfo/GSYM/FileWriter.h"
#include "llvm/DebugInfo/GSYM/FunctionInfo.h"
#include "llvm/MC/StringTableBuilder.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace llvm;
using namespace gsym;

llvm::Error CallSiteInfo::encode(FileWriter &O) const {
  O.writeU64(ReturnAddress);
  O.writeU8(Flags);
  O.writeU32(MatchRegex.size());
  for (uint32_t Entry : MatchRegex)
    O.writeU32(Entry);
  return llvm::Error::success();
}

llvm::Expected<CallSiteInfo>
CallSiteInfo::decode(DataExtractor &Data, uint64_t &Offset, uint64_t BaseAddr) {
  CallSiteInfo CSI;

  // Read ReturnAddress
  if (!Data.isValidOffsetForDataOfSize(Offset, sizeof(uint64_t)))
    return createStringError(std::errc::io_error,
                             "0x%8.8" PRIx64 ": missing ReturnAddress", Offset);
  CSI.ReturnAddress = Data.getU64(&Offset);

  // Read Flags
  if (!Data.isValidOffsetForDataOfSize(Offset, sizeof(uint8_t)))
    return createStringError(std::errc::io_error,
                             "0x%8.8" PRIx64 ": missing Flags", Offset);
  CSI.Flags = Data.getU8(&Offset);

  // Read number of MatchRegex entries
  if (!Data.isValidOffsetForDataOfSize(Offset, sizeof(uint32_t)))
    return createStringError(std::errc::io_error,
                             "0x%8.8" PRIx64 ": missing MatchRegex count",
                             Offset);
  uint32_t NumEntries = Data.getU32(&Offset);

  CSI.MatchRegex.reserve(NumEntries);
  for (uint32_t i = 0; i < NumEntries; ++i) {
    if (!Data.isValidOffsetForDataOfSize(Offset, sizeof(uint32_t)))
      return createStringError(std::errc::io_error,
                               "0x%8.8" PRIx64 ": missing MatchRegex entry",
                               Offset);
    uint32_t Entry = Data.getU32(&Offset);
    CSI.MatchRegex.push_back(Entry);
  }

  return CSI;
}

llvm::Error CallSiteInfoCollection::encode(FileWriter &O) const {
  O.writeU32(CallSites.size());
  for (const CallSiteInfo &CSI : CallSites) {
    if (llvm::Error Err = CSI.encode(O))
      return Err;
  }
  return llvm::Error::success();
}

llvm::Expected<CallSiteInfoCollection>
CallSiteInfoCollection::decode(DataExtractor &Data, uint64_t BaseAddr) {
  CallSiteInfoCollection CSC;
  uint64_t Offset = 0;

  // Read number of CallSiteInfo entries
  if (!Data.isValidOffsetForDataOfSize(Offset, sizeof(uint32_t)))
    return createStringError(std::errc::io_error,
                             "0x%8.8" PRIx64 ": missing CallSiteInfo count",
                             Offset);
  uint32_t NumCallSites = Data.getU32(&Offset);

  CSC.CallSites.reserve(NumCallSites);
  for (uint32_t i = 0; i < NumCallSites; ++i) {
    llvm::Expected<CallSiteInfo> ECSI =
        CallSiteInfo::decode(Data, Offset, BaseAddr);
    if (!ECSI)
      return ECSI.takeError();
    CSC.CallSites.emplace_back(*ECSI);
  }

  return CSC;
}

/// Structures necessary for reading CallSiteInfo from YAML.
namespace llvm {
namespace yaml {

struct CallSiteYAML {
  // The offset of the return address of the call site - relative to the start
  // of the function.
  llvm::yaml::Hex64 return_offset;
  std::vector<std::string> match_regex;
  std::vector<std::string> flags;
};

struct FunctionYAML {
  std::string name;
  std::vector<CallSiteYAML> callsites;
};

struct FunctionsYAML {
  std::vector<FunctionYAML> functions;
};

template <> struct MappingTraits<CallSiteYAML> {
  static void mapping(IO &io, CallSiteYAML &callsite) {
    io.mapRequired("return_offset", callsite.return_offset);
    io.mapRequired("match_regex", callsite.match_regex);
    io.mapOptional("flags", callsite.flags);
  }
};

template <> struct MappingTraits<FunctionYAML> {
  static void mapping(IO &io, FunctionYAML &func) {
    io.mapRequired("name", func.name);
    io.mapOptional("callsites", func.callsites);
  }
};

template <> struct MappingTraits<FunctionsYAML> {
  static void mapping(IO &io, FunctionsYAML &functionsYAML) {
    io.mapRequired("functions", functionsYAML.functions);
  }
};

} // namespace yaml
} // namespace llvm

LLVM_YAML_IS_SEQUENCE_VECTOR(CallSiteYAML)
LLVM_YAML_IS_SEQUENCE_VECTOR(FunctionYAML)

// Implementation of CallSiteInfoLoader
StringRef CallSiteInfoLoader::stringFromOffset(uint32_t offset) const {
  assert(StringOffsetMap.count(offset) &&
         "expected function name offset to already be in StringOffsetMap");
  return StringOffsetMap.find(offset)->second.val();
}

uint32_t CallSiteInfoLoader::offsetFromString(StringRef str) {
  return StrTab.add(StringStorage.insert(str).first->getKey());
}

llvm::Error CallSiteInfoLoader::loadYAML(std::vector<FunctionInfo> &Funcs,
                                         StringRef YAMLFile) {
  std::unique_ptr<llvm::MemoryBuffer> Buffer;
  // Step 1: Read YAML file
  if (auto Err = readYAMLFile(YAMLFile, Buffer))
    return Err;

  // Step 2: Parse YAML content
  llvm::yaml::FunctionsYAML functionsYAML;
  if (auto Err = parseYAML(*Buffer, functionsYAML))
    return Err;

  // Step 3: Build function map from Funcs
  auto FuncMap = buildFunctionMap(Funcs);

  // Step 4: Process parsed YAML functions and update FuncMap
  return processYAMLFunctions(functionsYAML, FuncMap, YAMLFile);
}

llvm::Error
CallSiteInfoLoader::readYAMLFile(StringRef YAMLFile,
                                 std::unique_ptr<llvm::MemoryBuffer> &Buffer) {
  auto BufferOrError = llvm::MemoryBuffer::getFile(YAMLFile);
  if (!BufferOrError)
    return errorCodeToError(BufferOrError.getError());
  Buffer = std::move(*BufferOrError);
  return llvm::Error::success();
}

llvm::Error
CallSiteInfoLoader::parseYAML(llvm::MemoryBuffer &Buffer,
                              llvm::yaml::FunctionsYAML &functionsYAML) {
  // Use the MemoryBufferRef constructor
  llvm::yaml::Input yin(Buffer.getMemBufferRef());
  yin >> functionsYAML;
  if (yin.error()) {
    return llvm::createStringError(yin.error(), "Error parsing YAML file: %s\n",
                                   Buffer.getBufferIdentifier().str().c_str());
  }
  return llvm::Error::success();
}

std::unordered_map<std::string, FunctionInfo *>
CallSiteInfoLoader::buildFunctionMap(std::vector<FunctionInfo> &Funcs) {
  std::unordered_map<std::string, FunctionInfo *> FuncMap;
  auto insertFunc = [&](auto &Function) {
    std::string FuncName = stringFromOffset(Function.Name).str();
    // If the function name is already in the map, don't update it. This way we
    // preferentially use the first encountered function. Since symbols are
    // loaded from dSYM first, we end up preferring keeping track of symbols
    // from dSYM rather than from the symbol table - which is what we want to
    // do.
    if (FuncMap.count(FuncName))
      return;
    FuncMap[FuncName] = &Function;
  };
  for (auto &Func : Funcs) {
    insertFunc(Func);
    if (Func.MergedFunctions.has_value())
      for (auto &MFunc : Func.MergedFunctions->MergedFunctions)
        insertFunc(MFunc);
  }
  return FuncMap;
}

llvm::Error CallSiteInfoLoader::processYAMLFunctions(
    const llvm::yaml::FunctionsYAML &functionsYAML,
    std::unordered_map<std::string, FunctionInfo *> &FuncMap,
    StringRef YAMLFile) {
  // For each function in the YAML file
  for (const auto &FuncYAML : functionsYAML.functions) {
    auto it = FuncMap.find(FuncYAML.name);
    if (it == FuncMap.end()) {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "Can't find function '%s' specified in callsite YAML\n",
          FuncYAML.name.c_str());
    }
    FunctionInfo *FuncInfo = it->second;
    // Create a CallSiteInfoCollection if not already present
    if (!FuncInfo->CallSites)
      FuncInfo->CallSites = CallSiteInfoCollection();
    for (const auto &CallSiteYAML : FuncYAML.callsites) {
      CallSiteInfo CSI;
      // Since YAML has specifies relative return offsets, add the function
      // start address to make the offset absolute.
      CSI.ReturnAddress = FuncInfo->Range.start() + CallSiteYAML.return_offset;
      for (const auto &regex : CallSiteYAML.match_regex) {
        CSI.MatchRegex.push_back(offsetFromString(regex));
      }
      // Initialize flags to None
      CSI.Flags = CallSiteInfo::None;
      // Parse flags and combine them
      for (const auto &FlagStr : CallSiteYAML.flags) {
        if (FlagStr == "InternalCall") {
          CSI.Flags |= static_cast<uint8_t>(CallSiteInfo::InternalCall);
        } else if (FlagStr == "ExternalCall") {
          CSI.Flags |= static_cast<uint8_t>(CallSiteInfo::ExternalCall);
        } else {
          return llvm::createStringError(std::errc::invalid_argument,
                                         "Unknown flag in callsite YAML: %s\n",
                                         FlagStr.c_str());
        }
      }
      FuncInfo->CallSites->CallSites.push_back(CSI);
    }
  }
  return llvm::Error::success();
}

raw_ostream &llvm::gsym::operator<<(raw_ostream &OS, const CallSiteInfo &CSI) {
  OS << "  Return=" << HEX64(CSI.ReturnAddress);
  OS << "  Flags=" << HEX8(CSI.Flags);

  OS << "  RegEx=";
  for (uint32_t i = 0; i < CSI.MatchRegex.size(); ++i) {
    if (i > 0)
      OS << ",";
    OS << CSI.MatchRegex[i];
  }
  return OS;
}

raw_ostream &llvm::gsym::operator<<(raw_ostream &OS,
                                    const CallSiteInfoCollection &CSIC) {
  for (const auto &CS : CSIC.CallSites) {
    OS << CS;
    OS << "\n";
  }
  return OS;
}
