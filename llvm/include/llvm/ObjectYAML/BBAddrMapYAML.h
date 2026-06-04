//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares the YAML representation of BB address maps
/// (SHT_LLVM_BB_ADDR_MAP / .llvm_bb_addr_map). The types here are
/// format-agnostic so they can be reused by ELFYAML and COFFYAML.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECTYAML_BBADDRMAPYAML_H
#define LLVM_OBJECTYAML_BBADDRMAPYAML_H

#include "llvm/Support/YAMLTraits.h"
#include <cstdint>
#include <optional>
#include <vector>

namespace llvm {
namespace BBAddrMapYAML {

struct BBAddrMapEntry {
  struct BBEntry {
    uint32_t ID;
    llvm::yaml::Hex64 AddressOffset;
    llvm::yaml::Hex64 Size;
    llvm::yaml::Hex64 Metadata;
    std::optional<std::vector<llvm::yaml::Hex64>> CallsiteEndOffsets;
    std::optional<llvm::yaml::Hex64> Hash;
  };
  uint8_t Version;
  llvm::yaml::Hex16 Feature;

  struct BBRangeEntry {
    llvm::yaml::Hex64 BaseAddress;
    std::optional<uint64_t> NumBlocks;
    std::optional<std::vector<BBEntry>> BBEntries;
  };

  std::optional<uint64_t> NumBBRanges;
  std::optional<std::vector<BBRangeEntry>> BBRanges;

  llvm::yaml::Hex64 getFunctionAddress() const {
    if (!BBRanges || BBRanges->empty())
      return 0;
    return BBRanges->front().BaseAddress;
  }

  // Returns if any BB entries have non-empty callsite offsets.
  bool hasAnyCallsiteEndOffsets() const {
    if (!BBRanges)
      return false;
    for (const BBRangeEntry &BBR : *BBRanges) {
      if (!BBR.BBEntries)
        continue;
      for (const BBEntry &BBE : *BBR.BBEntries)
        if (BBE.CallsiteEndOffsets && !BBE.CallsiteEndOffsets->empty())
          return true;
    }
    return false;
  }
};

struct PGOAnalysisMapEntry {
  struct PGOBBEntry {
    struct SuccessorEntry {
      uint32_t ID;
      llvm::yaml::Hex32 BrProb;
      std::optional<uint32_t> PostLinkBrFreq;
    };
    std::optional<uint64_t> BBFreq;
    std::optional<uint32_t> PostLinkBBFreq;
    std::optional<std::vector<SuccessorEntry>> Successors;
  };
  std::optional<uint64_t> FuncEntryCount;
  std::optional<std::vector<PGOBBEntry>> PGOBBEntries;
};

} // end namespace BBAddrMapYAML
} // end namespace llvm

LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::BBAddrMapYAML::BBAddrMapEntry)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::BBAddrMapYAML::BBAddrMapEntry::BBEntry)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::BBAddrMapYAML::BBAddrMapEntry::BBRangeEntry)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::BBAddrMapYAML::PGOAnalysisMapEntry)
LLVM_YAML_IS_SEQUENCE_VECTOR(
    llvm::BBAddrMapYAML::PGOAnalysisMapEntry::PGOBBEntry)
LLVM_YAML_IS_SEQUENCE_VECTOR(
    llvm::BBAddrMapYAML::PGOAnalysisMapEntry::PGOBBEntry::SuccessorEntry)

namespace llvm {
namespace yaml {

template <> struct MappingTraits<BBAddrMapYAML::BBAddrMapEntry> {
  static void mapping(IO &IO, BBAddrMapYAML::BBAddrMapEntry &E);
};

template <> struct MappingTraits<BBAddrMapYAML::BBAddrMapEntry::BBRangeEntry> {
  static void mapping(IO &IO, BBAddrMapYAML::BBAddrMapEntry::BBRangeEntry &E);
};

template <> struct MappingTraits<BBAddrMapYAML::BBAddrMapEntry::BBEntry> {
  static void mapping(IO &IO, BBAddrMapYAML::BBAddrMapEntry::BBEntry &E);
};

template <> struct MappingTraits<BBAddrMapYAML::PGOAnalysisMapEntry> {
  static void mapping(IO &IO, BBAddrMapYAML::PGOAnalysisMapEntry &E);
};

template <>
struct MappingTraits<BBAddrMapYAML::PGOAnalysisMapEntry::PGOBBEntry> {
  static void mapping(IO &IO,
                      BBAddrMapYAML::PGOAnalysisMapEntry::PGOBBEntry &E);
};

template <>
struct MappingTraits<
    BBAddrMapYAML::PGOAnalysisMapEntry::PGOBBEntry::SuccessorEntry> {
  static void
  mapping(IO &IO,
          BBAddrMapYAML::PGOAnalysisMapEntry::PGOBBEntry::SuccessorEntry &E);
};

} // end namespace yaml
} // end namespace llvm

#endif // LLVM_OBJECTYAML_BBADDRMAPYAML_H
