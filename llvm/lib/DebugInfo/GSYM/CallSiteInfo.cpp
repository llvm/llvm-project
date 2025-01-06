//===- CallSiteInfo.cpp -----------------------------------------*- C++ -*-===//
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
#include "llvm/DebugInfo/GSYM/GsymCreator.h"
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

Error CallSiteInfo::encode(FileWriter &O) const {
  O.writeU64(ReturnOffset);
  O.writeU8(Flags);
  O.writeU32(MatchRegex.size());
  for (uint32_t Entry : MatchRegex)
    O.writeU32(Entry);
  return Error::success();
}

Expected<CallSiteInfo> CallSiteInfo::decode(DataExtractor &Data,
                                            uint64_t &Offset) {
  CallSiteInfo CSI;

  // Read ReturnOffset
  if (!Data.isValidOffsetForDataOfSize(Offset, sizeof(uint64_t)))
    return createStringError(std::errc::io_error,
                             "0x%8.8" PRIx64 ": missing ReturnOffset", Offset);
  CSI.ReturnOffset = Data.getU64(&Offset);

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

Error CallSiteInfoCollection::encode(FileWriter &O) const {
  O.writeU32(CallSites.size());
  for (const CallSiteInfo &CSI : CallSites)
    if (Error Err = CSI.encode(O))
      return Err;

  return Error::success();
}

Expected<CallSiteInfoCollection>
CallSiteInfoCollection::decode(DataExtractor &Data) {
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
    Expected<CallSiteInfo> ECSI = CallSiteInfo::decode(Data, Offset);
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
  Hex64 return_offset;
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
  static void mapping(IO &io, FunctionsYAML &FuncYAMLs) {
    io.mapRequired("functions", FuncYAMLs.functions);
  }
};

} // namespace yaml
} // namespace llvm

LLVM_YAML_IS_SEQUENCE_VECTOR(CallSiteYAML)
LLVM_YAML_IS_SEQUENCE_VECTOR(FunctionYAML)

Error CallSiteInfoLoader::loadYAML(StringRef YAMLFile) {
  // Step 1: Read YAML file
  auto BufferOrError = MemoryBuffer::getFile(YAMLFile, /*IsText=*/true);
  if (!BufferOrError)
    return errorCodeToError(BufferOrError.getError());

  std::unique_ptr<MemoryBuffer> Buffer = std::move(*BufferOrError);

  // Step 2: Parse YAML content
  yaml::FunctionsYAML FuncsYAML;
  yaml::Input Yin(Buffer->getMemBufferRef());
  Yin >> FuncsYAML;
  if (Yin.error())
    return createStringError(Yin.error(), "Error parsing YAML file: %s\n",
                             Buffer->getBufferIdentifier().str().c_str());

  // Step 3: Build function map from Funcs
  auto FuncMap = buildFunctionMap();

  // Step 4: Process parsed YAML functions and update FuncMap
  return processYAMLFunctions(FuncsYAML, FuncMap);
}

StringMap<FunctionInfo *> CallSiteInfoLoader::buildFunctionMap() {
  // If the function name is already in the map, don't update it. This way we
  // preferentially use the first encountered function. Since symbols are
  // loaded from dSYM first, we end up preferring keeping track of symbols
  // from dSYM rather than from the symbol table - which is what we want to
  // do.
  StringMap<FunctionInfo *> FuncMap;
  for (auto &Func : Funcs) {
    FuncMap.try_emplace(GCreator.getString(Func.Name), &Func);
    if (auto &MFuncs = Func.MergedFunctions)
      for (auto &MFunc : MFuncs->MergedFunctions)
        FuncMap.try_emplace(GCreator.getString(MFunc.Name), &MFunc);
  }
  return FuncMap;
}

Error CallSiteInfoLoader::processYAMLFunctions(
    const yaml::FunctionsYAML &FuncYAMLs, StringMap<FunctionInfo *> &FuncMap) {
  // For each function in the YAML file
  for (const auto &FuncYAML : FuncYAMLs.functions) {
    auto It = FuncMap.find(FuncYAML.name);
    if (It == FuncMap.end())
      return createStringError(
          std::errc::invalid_argument,
          "Can't find function '%s' specified in callsite YAML\n",
          FuncYAML.name.c_str());

    FunctionInfo *FuncInfo = It->second;
    // Create a CallSiteInfoCollection if not already present
    if (!FuncInfo->CallSites)
      FuncInfo->CallSites = CallSiteInfoCollection();
    for (const auto &CallSiteYAML : FuncYAML.callsites) {
      CallSiteInfo CSI;
      // Since YAML has specifies relative return offsets, add the function
      // start address to make the offset absolute.
      CSI.ReturnOffset = CallSiteYAML.return_offset;
      for (const auto &Regex : CallSiteYAML.match_regex) {
        uint32_t StrOffset = GCreator.insertString(Regex);
        CSI.MatchRegex.push_back(StrOffset);
      }

      // Parse flags and combine them
      for (const auto &FlagStr : CallSiteYAML.flags) {
        if (FlagStr == "InternalCall") {
          CSI.Flags |= static_cast<uint8_t>(CallSiteInfo::InternalCall);
        } else if (FlagStr == "ExternalCall") {
          CSI.Flags |= static_cast<uint8_t>(CallSiteInfo::ExternalCall);
        } else {
          return createStringError(std::errc::invalid_argument,
                                   "Unknown flag in callsite YAML: %s\n",
                                   FlagStr.c_str());
        }
      }
      FuncInfo->CallSites->CallSites.push_back(CSI);
    }
  }
  return Error::success();
}

raw_ostream &gsym::operator<<(raw_ostream &OS, const CallSiteInfo &CSI) {
  OS << "  Return=" << HEX64(CSI.ReturnOffset);
  OS << "  Flags=" << HEX8(CSI.Flags);

  OS << "  RegEx=";
  for (uint32_t i = 0; i < CSI.MatchRegex.size(); ++i) {
    if (i > 0)
      OS << ",";
    OS << CSI.MatchRegex[i];
  }
  return OS;
}

raw_ostream &gsym::operator<<(raw_ostream &OS,
                              const CallSiteInfoCollection &CSIC) {
  for (const auto &CS : CSIC.CallSites) {
    OS << CS;
    OS << "\n";
  }
  return OS;
}
