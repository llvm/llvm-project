//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the YAMLIO mappings for the format-agnostic BB address
/// map YAML types declared in BBAddrMapYAML.h.
///
//===----------------------------------------------------------------------===//

#include "llvm/ObjectYAML/BBAddrMapYAML.h"
#include "llvm/Object/BBAddrMap.h"
#include "llvm/ObjectYAML/ContiguousBlobAccumulator.h"
#include "llvm/Support/WithColor.h"

namespace llvm {
namespace yaml {

void MappingTraits<BBAddrMapYAML::BBAddrMapEntry>::mapping(
    IO &IO, BBAddrMapYAML::BBAddrMapEntry &E) {
  assert(IO.getContext() && "The IO context is not initialized");
  IO.mapRequired("Version", E.Version);
  IO.mapOptional("Feature", E.Feature, Hex16(0));
  IO.mapOptional("NumBBRanges", E.NumBBRanges);
  IO.mapOptional("BBRanges", E.BBRanges);
}

void MappingTraits<BBAddrMapYAML::BBAddrMapEntry::BBRangeEntry>::mapping(
    IO &IO, BBAddrMapYAML::BBAddrMapEntry::BBRangeEntry &E) {
  IO.mapOptional("BaseAddress", E.BaseAddress, Hex64(0));
  IO.mapOptional("NumBlocks", E.NumBlocks);
  IO.mapOptional("BBEntries", E.BBEntries);
}

void MappingTraits<BBAddrMapYAML::BBAddrMapEntry::BBEntry>::mapping(
    IO &IO, BBAddrMapYAML::BBAddrMapEntry::BBEntry &E) {
  assert(IO.getContext() && "The IO context is not initialized");
  IO.mapOptional("ID", E.ID);
  IO.mapRequired("AddressOffset", E.AddressOffset);
  IO.mapRequired("Size", E.Size);
  IO.mapRequired("Metadata", E.Metadata);
  IO.mapOptional("CallsiteEndOffsets", E.CallsiteEndOffsets);
  IO.mapOptional("Hash", E.Hash);
}

void MappingTraits<BBAddrMapYAML::PGOAnalysisMapEntry>::mapping(
    IO &IO, BBAddrMapYAML::PGOAnalysisMapEntry &E) {
  assert(IO.getContext() && "The IO context is not initialized");
  IO.mapOptional("FuncEntryCount", E.FuncEntryCount);
  IO.mapOptional("PGOBBEntries", E.PGOBBEntries);
}

void MappingTraits<BBAddrMapYAML::PGOAnalysisMapEntry::PGOBBEntry>::mapping(
    IO &IO, BBAddrMapYAML::PGOAnalysisMapEntry::PGOBBEntry &E) {
  assert(IO.getContext() && "The IO context is not initialized");
  IO.mapOptional("BBFreq", E.BBFreq);
  IO.mapOptional("PostLinkBBFreq", E.PostLinkBBFreq);
  IO.mapOptional("Successors", E.Successors);
}

void MappingTraits<
    BBAddrMapYAML::PGOAnalysisMapEntry::PGOBBEntry::SuccessorEntry>::
    mapping(IO &IO,
            BBAddrMapYAML::PGOAnalysisMapEntry::PGOBBEntry::SuccessorEntry &E) {
  assert(IO.getContext() && "The IO context is not initialized");
  IO.mapRequired("ID", E.ID);
  IO.mapRequired("BrProb", E.BrProb);
  IO.mapOptional("PostLinkBrFreq", E.PostLinkBrFreq);
}

} // end namespace yaml

namespace BBAddrMapYAML {

void encodePayload(ArrayRef<BBAddrMapEntry> Entries,
                   const std::vector<PGOAnalysisMapEntry> *PGOAnalyses,
                   yaml::ContiguousBlobAccumulator &CBA,
                   llvm::endianness Endian, unsigned AddressSize) {
  assert((AddressSize == 4 || AddressSize == 8) && "invalid address size");
  for (const auto &[Idx, E] : llvm::enumerate(Entries)) {
    // Write version and feature values.
    if (E.Version > 5)
      WithColor::warning() << "unsupported BB address map version: "
                           << static_cast<int>(E.Version)
                           << "; encoding using the most recent version";
    CBA.write(E.Version);
    if (E.Version < 5)
      CBA.write(static_cast<uint8_t>(E.Feature));
    else
      CBA.write<uint16_t>(E.Feature, Endian);
    auto FeatureOrErr = llvm::object::BBAddrMap::Features::decode(E.Feature);
    if (!FeatureOrErr) {
      // Invalid feature: warn and skip the entry.
      WithColor::warning() << toString(FeatureOrErr.takeError());
      continue;
    }
    bool MultiBBRangeFeatureEnabled = FeatureOrErr->MultiBBRange;
    bool MultiBBRange =
        MultiBBRangeFeatureEnabled ||
        (E.NumBBRanges.has_value() && E.NumBBRanges.value() != 1) ||
        (E.BBRanges && E.BBRanges->size() != 1);
    if (MultiBBRange && !MultiBBRangeFeatureEnabled)
      WithColor::warning() << "feature value(" << E.Feature
                           << ") does not support multiple BB ranges.";
    if (MultiBBRange) {
      // Write the number of basic block ranges, which is overridden by the
      // 'NumBBRanges' field when specified.
      uint64_t NumBBRanges =
          E.NumBBRanges.value_or(E.BBRanges ? E.BBRanges->size() : 0);
      CBA.writeULEB128(NumBBRanges);
    }
    if (!E.BBRanges)
      continue;
    uint64_t TotalNumBlocks = 0;
    bool EmitCallsiteEndOffsets =
        FeatureOrErr->CallsiteEndOffsets || E.hasAnyCallsiteEndOffsets();
    for (const BBAddrMapEntry::BBRangeEntry &BBR : *E.BBRanges) {
      // Write the pointer-sized base address of the range.
      if (AddressSize == 8)
        CBA.write<uint64_t>(BBR.BaseAddress, Endian);
      else
        CBA.write<uint32_t>(static_cast<uint32_t>(BBR.BaseAddress), Endian);
      // Write number of BBEntries (number of basic blocks in this basic block
      // range). This is overridden by the 'NumBlocks' YAML field when
      // specified.
      uint64_t NumBlocks =
          BBR.NumBlocks.value_or(BBR.BBEntries ? BBR.BBEntries->size() : 0);
      CBA.writeULEB128(NumBlocks);
      // Write all BBEntries in this BBRange.
      if (!BBR.BBEntries || FeatureOrErr->OmitBBEntries)
        continue;
      for (const BBAddrMapEntry::BBEntry &BBE : *BBR.BBEntries) {
        ++TotalNumBlocks;
        if (E.Version > 1)
          CBA.writeULEB128(BBE.ID);
        CBA.writeULEB128(BBE.AddressOffset);
        if (EmitCallsiteEndOffsets) {
          size_t NumCallsiteEndOffsets =
              BBE.CallsiteEndOffsets ? BBE.CallsiteEndOffsets->size() : 0;
          CBA.writeULEB128(NumCallsiteEndOffsets);
          if (BBE.CallsiteEndOffsets) {
            for (uint32_t Offset : *BBE.CallsiteEndOffsets)
              CBA.writeULEB128(Offset);
          }
        }
        CBA.writeULEB128(BBE.Size);
        CBA.writeULEB128(BBE.Metadata);
        if (FeatureOrErr->BBHash || BBE.Hash.has_value()) {
          uint64_t Hash =
              BBE.Hash.has_value() ? BBE.Hash.value() : llvm::yaml::Hex64(0);
          CBA.write<uint64_t>(Hash, Endian);
        }
      }
    }
    if (!PGOAnalyses)
      continue;
    const PGOAnalysisMapEntry &PGOEntry = PGOAnalyses->at(Idx);

    if (PGOEntry.FuncEntryCount)
      CBA.writeULEB128(*PGOEntry.FuncEntryCount);

    if (!PGOEntry.PGOBBEntries)
      continue;

    const auto &PGOBBEntries = PGOEntry.PGOBBEntries.value();
    if (TotalNumBlocks != PGOBBEntries.size()) {
      WithColor::warning() << "PGOBBEntries must be the same length as "
                              "BBEntries in the BB address map.\n"
                           << "Mismatch on function with address: "
                           << E.getFunctionAddress();
      continue;
    }

    for (const auto &PGOBBE : PGOBBEntries) {
      if (PGOBBE.BBFreq)
        CBA.writeULEB128(*PGOBBE.BBFreq);
      if (FeatureOrErr->PostLinkCfg || PGOBBE.PostLinkBBFreq.has_value())
        CBA.writeULEB128(PGOBBE.PostLinkBBFreq.value_or(0));
      if (PGOBBE.Successors) {
        CBA.writeULEB128(PGOBBE.Successors->size());
        for (const auto &[ID, BrProb, PostLinkBrFreq] : *PGOBBE.Successors) {
          CBA.writeULEB128(ID);
          CBA.writeULEB128(BrProb);
          if (FeatureOrErr->PostLinkCfg || PostLinkBrFreq.has_value())
            CBA.writeULEB128(PostLinkBrFreq.value_or(0));
        }
      }
    }
  }
}

} // end namespace BBAddrMapYAML
} // end namespace llvm
