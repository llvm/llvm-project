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
#include "llvm/ADT/STLExtras.h"
#include "llvm/Object/BBAddrMap.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/WithColor.h"

namespace llvm {

void BBAddrMapYAML::Encoder::writeAddress(uint64_t Address) {
  if (AddressSize == 8)
    writeInteger<uint64_t>(Address);
  else
    writeInteger<uint32_t>(static_cast<uint32_t>(Address));
}

void BBAddrMapYAML::Encoder::writeULEB128(uint64_t Value) {
  NumBytes += encodeULEB128(Value, OS);
}

uint64_t BBAddrMapYAML::encodePayload(
    ArrayRef<BBAddrMapYAML::BBAddrMapEntry> Entries,
    const std::vector<BBAddrMapYAML::PGOAnalysisMapEntry> *PGOAnalyses,
    BBAddrMapYAML::Encoder &E) {
  uint64_t StartBytes = E.getNumBytes();

  for (const auto &[Idx, Entry] : llvm::enumerate(Entries)) {
    if (Entry.Version > 5)
      WithColor::warning() << "unsupported BB address map version: "
                           << static_cast<int>(Entry.Version)
                           << "; encoding using the most recent version";
    E.writeInteger<uint8_t>(Entry.Version);
    if (Entry.Version < 5)
      E.writeInteger<uint8_t>(static_cast<uint8_t>(Entry.Feature));
    else
      E.writeInteger<uint16_t>(Entry.Feature);

    auto FeatureOrErr = object::BBAddrMap::Features::decode(Entry.Feature);
    bool MultiBBRangeFeatureEnabled = false;
    if (!FeatureOrErr)
      WithColor::warning() << toString(FeatureOrErr.takeError());
    else
      MultiBBRangeFeatureEnabled = FeatureOrErr->MultiBBRange;
    bool MultiBBRange =
        MultiBBRangeFeatureEnabled ||
        (Entry.NumBBRanges.has_value() && Entry.NumBBRanges.value() != 1) ||
        (Entry.BBRanges && Entry.BBRanges->size() != 1);
    if (MultiBBRange && !MultiBBRangeFeatureEnabled)
      WithColor::warning() << "feature value(" << Entry.Feature
                           << ") does not support multiple BB ranges.";
    if (MultiBBRange) {
      // Write the number of basic block ranges, which is overridden by the
      // 'NumBBRanges' field when specified.
      E.writeULEB128(Entry.NumBBRanges.value_or(
          Entry.BBRanges ? Entry.BBRanges->size() : 0));
    }
    if (!Entry.BBRanges)
      continue;
    uint64_t TotalNumBlocks = 0;
    bool EmitCallsiteEndOffsets =
        FeatureOrErr->CallsiteEndOffsets || Entry.hasAnyCallsiteEndOffsets();
    for (const BBAddrMapYAML::BBAddrMapEntry::BBRangeEntry &BBR :
         *Entry.BBRanges) {
      // Write the base address of the range.
      E.writeAddress(BBR.BaseAddress);
      // Write number of BBEntries (number of basic blocks in this basic block
      // range). This is overridden by the 'NumBlocks' YAML field when
      // specified.
      E.writeULEB128(
          BBR.NumBlocks.value_or(BBR.BBEntries ? BBR.BBEntries->size() : 0));
      if (!BBR.BBEntries || FeatureOrErr->OmitBBEntries)
        continue;
      for (const BBAddrMapYAML::BBAddrMapEntry::BBEntry &BBE : *BBR.BBEntries) {
        ++TotalNumBlocks;
        if (Entry.Version > 1)
          E.writeULEB128(BBE.ID);
        E.writeULEB128(BBE.AddressOffset);
        if (EmitCallsiteEndOffsets) {
          size_t NumCallsiteEndOffsets =
              BBE.CallsiteEndOffsets ? BBE.CallsiteEndOffsets->size() : 0;
          E.writeULEB128(NumCallsiteEndOffsets);
          if (BBE.CallsiteEndOffsets) {
            for (uint32_t Offset : *BBE.CallsiteEndOffsets)
              E.writeULEB128(Offset);
          }
        }
        E.writeULEB128(BBE.Size);
        E.writeULEB128(BBE.Metadata);
        if (FeatureOrErr->BBHash || BBE.Hash.has_value())
          E.writeInteger<uint64_t>(BBE.Hash.has_value() ? BBE.Hash.value()
                                                        : llvm::yaml::Hex64(0));
      }
    }
    if (!PGOAnalyses)
      continue;
    const BBAddrMapYAML::PGOAnalysisMapEntry &PGOEntry = PGOAnalyses->at(Idx);

    if (PGOEntry.FuncEntryCount)
      E.writeULEB128(*PGOEntry.FuncEntryCount);

    if (!PGOEntry.PGOBBEntries)
      continue;

    const auto &PGOBBEntries = PGOEntry.PGOBBEntries.value();
    if (TotalNumBlocks != PGOBBEntries.size()) {
      WithColor::warning() << "PGOBBEntries must be the same length as "
                              "BBEntries in the BB address map.\n"
                           << "Mismatch on function with address: "
                           << Entry.getFunctionAddress();
      continue;
    }

    for (const auto &PGOBBE : PGOBBEntries) {
      if (PGOBBE.BBFreq)
        E.writeULEB128(*PGOBBE.BBFreq);
      if (FeatureOrErr->PostLinkCfg || PGOBBE.PostLinkBBFreq.has_value())
        E.writeULEB128(PGOBBE.PostLinkBBFreq.value_or(0));
      if (PGOBBE.Successors) {
        E.writeULEB128(PGOBBE.Successors->size());
        for (const auto &[ID, BrProb, PostLinkBrFreq] : *PGOBBE.Successors) {
          E.writeULEB128(ID);
          E.writeULEB128(BrProb);
          if (FeatureOrErr->PostLinkCfg || PostLinkBrFreq.has_value())
            E.writeULEB128(PostLinkBrFreq.value_or(0));
        }
      }
    }
  }
  return E.getNumBytes() - StartBytes;
}

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
} // end namespace llvm
