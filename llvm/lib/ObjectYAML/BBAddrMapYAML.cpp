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
} // end namespace llvm
