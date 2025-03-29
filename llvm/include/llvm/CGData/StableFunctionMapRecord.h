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

#include "llvm/CGData/StableFunctionMap.h"
#include "llvm/ObjectYAML/YAML.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

struct StableFunctionMapRecord {
  std::unique_ptr<StableFunctionMap> FunctionMap;

  StableFunctionMapRecord() {
    FunctionMap = std::make_unique<StableFunctionMap>();
  }

  StableFunctionMapRecord(std::unique_ptr<StableFunctionMap> FunctionMap)
      : FunctionMap(std::move(FunctionMap)) {}

  /// A static helper function to serialize the stable function map without
  /// owning the stable function map.
  static void serialize(raw_ostream &OS, const StableFunctionMap *FunctionMap);

  /// Serialize the stable function map to a raw_ostream.
  void serialize(raw_ostream &OS) const;

  /// Deserialize the stable function map from a raw_ostream.
  void deserialize(const unsigned char *&Ptr);

  /// Serialize the stable function map to a YAML stream.
  void serializeYAML(yaml::Output &YOS) const;

  /// Deserialize the stable function map from a YAML stream.
  void deserializeYAML(yaml::Input &YIS);

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
};

} // namespace llvm

#endif
