//===- StableFunctionMapRecord.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// This defines the StableFunctionMapRecord structure, which provides
// functionality for managing and serializing a StableFunctionMap. It includes
// methods for serialization to and from raw and YAML streams, as well as
// utilities for merging and finalizing function maps.
//
//===---------------------------------------------------------------------===//

#ifndef LLVM_CGDATA_STABLEFUNCTIONMAPRECORD_H
#define LLVM_CGDATA_STABLEFUNCTIONMAPRECORD_H

#include "llvm/CGData/CGDataPatchItem.h"
#include "llvm/CGData/StableFunctionMap.h"
#include "llvm/ObjectYAML/YAML.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

/// The structure of the serialized stable function map is as follows:
/// - Number of unique function/module names
/// - Total size of unique function/module names for opt-in skipping
/// - Unique function/module names
/// - Padding to align to 4 bytes
/// - Number of StableFunctionEntries
/// - Hashes of each StableFunctionEntry
/// - Fixed-size fields for each StableFunctionEntry (the order is consistent
///   with the hashes above):
///   - FunctionNameId
///   - ModuleNameId
///   - InstCount
///   - Relative offset to the beginning of IndexOperandHashes for this entry
/// - Total size of variable-sized IndexOperandHashes for lazy-loading support
/// - Variable-sized IndexOperandHashes for each StableFunctionEntry:
///   - Number of IndexOperandHashes
///   - Contents of each IndexOperandHashes
///     - InstIndex
///     - OpndIndex
///     - OpndHash
struct StableFunctionMapRecord {
  std::unique_ptr<StableFunctionMap> FunctionMap;

  StableFunctionMapRecord() {
    FunctionMap = std::make_unique<StableFunctionMap>();
  }

  StableFunctionMapRecord(std::unique_ptr<StableFunctionMap> FunctionMap)
      : FunctionMap(std::move(FunctionMap)) {}

  /// A static helper function to serialize the stable function map without
  /// owning the stable function map.
  LLVM_ABI static void serialize(raw_ostream &OS,
                                 const StableFunctionMap *FunctionMap,
                                 std::vector<CGDataPatchItem> &PatchItems);

  /// A static helper function to deserialize the stable function map entry.
  /// Ptr should be pointing to the start of the fixed-sized fields of the
  /// entry when passed in.
  LLVM_ABI static void deserializeEntry(const unsigned char *Ptr,
                                        stable_hash Hash,
                                        StableFunctionMap *FunctionMap);

  /// Serialize the stable function map to a raw_ostream.
  LLVM_ABI void serialize(raw_ostream &OS,
                          std::vector<CGDataPatchItem> &PatchItems) const;

  /// Deserialize the stable function map from a raw_ostream.
  LLVM_ABI void deserialize(const unsigned char *&Ptr);

  /// Lazily deserialize the stable function map from `Buffer` starting at
  /// `Offset`. The individual stable function entry would be read lazily from
  /// `Buffer` when the function map is accessed.
  LLVM_ABI void lazyDeserialize(std::shared_ptr<MemoryBuffer> Buffer,
                                uint64_t Offset);

  /// Serialize the stable function map to a YAML stream.
  LLVM_ABI void serializeYAML(yaml::Output &YOS) const;

  /// Deserialize the stable function map from a YAML stream.
  LLVM_ABI void deserializeYAML(yaml::Input &YIS);

  /// Finalize the stable function map by trimming content.
  void finalize(bool SkipTrim = false) { FunctionMap->finalize(SkipTrim); }

  /// Merge the stable function map into this one.
  void merge(const StableFunctionMapRecord &Other) {
    FunctionMap->merge(*Other.FunctionMap);
  }

  /// \returns true if the stable function map is empty.
  bool empty() const { return FunctionMap->empty(); }

  /// Print the stable function map in a YAML format.
  void print(raw_ostream &OS = llvm::errs()) const {
    yaml::Output YOS(OS);
    serializeYAML(YOS);
  }

  /// Set whether to read stable function names from the buffer.
  /// Has no effect if the function map is read from a YAML stream.
  void setReadStableFunctionMapNames(bool Read) {
    assert(
        FunctionMap->empty() &&
        "Cannot change ReadStableFunctionMapNames after the map is populated");
    FunctionMap->ReadStableFunctionMapNames = Read;
  }

private:
  void deserialize(const unsigned char *&Ptr, bool Lazy);
};

} // namespace llvm

#endif
