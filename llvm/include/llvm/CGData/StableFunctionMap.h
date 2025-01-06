//===- StableFunctionMap.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// This defines the StableFunctionMap class, to track similar functions.
// It provides a mechanism to map stable hashes of functions to their
// corresponding metadata. It includes structures for storing function details
// and methods for managing and querying these mappings.
//
//===---------------------------------------------------------------------===//

#ifndef LLVM_CGDATA_STABLEFUNCTIONMAP_H
#define LLVM_CGDATA_STABLEFUNCTIONMAP_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/IR/StructuralHash.h"

namespace llvm {

using IndexPairHash = std::pair<IndexPair, stable_hash>;
using IndexOperandHashVecType = SmallVector<IndexPairHash>;

/// A stable function is a function with a stable hash while tracking the
/// locations of ignored operands and their hashes.
struct StableFunction {
  /// The combined stable hash of the function.
  stable_hash Hash;
  /// The name of the function.
  std::string FunctionName;
  /// The name of the module the function is in.
  std::string ModuleName;
  /// The number of instructions.
  unsigned InstCount;
  /// A vector of pairs of IndexPair and operand hash which was skipped.
  IndexOperandHashVecType IndexOperandHashes;

  StableFunction(stable_hash Hash, const std::string FunctionName,
                 const std::string ModuleName, unsigned InstCount,
                 IndexOperandHashVecType &&IndexOperandHashes)
      : Hash(Hash), FunctionName(FunctionName), ModuleName(ModuleName),
        InstCount(InstCount),
        IndexOperandHashes(std::move(IndexOperandHashes)) {}
  StableFunction() = default;
};

struct StableFunctionMap {
  /// An efficient form of StableFunction for fast look-up
  struct StableFunctionEntry {
    /// The combined stable hash of the function.
    stable_hash Hash;
    /// Id of the function name.
    unsigned FunctionNameId;
    /// Id of the module name.
    unsigned ModuleNameId;
    /// The number of instructions.
    unsigned InstCount;
    /// A map from an IndexPair to a stable_hash which was skipped.
    std::unique_ptr<IndexOperandHashMapType> IndexOperandHashMap;

    StableFunctionEntry(
        stable_hash Hash, unsigned FunctionNameId, unsigned ModuleNameId,
        unsigned InstCount,
        std::unique_ptr<IndexOperandHashMapType> IndexOperandHashMap)
        : Hash(Hash), FunctionNameId(FunctionNameId),
          ModuleNameId(ModuleNameId), InstCount(InstCount),
          IndexOperandHashMap(std::move(IndexOperandHashMap)) {}
  };

  using HashFuncsMapType =
      DenseMap<stable_hash, SmallVector<std::unique_ptr<StableFunctionEntry>>>;

  /// Get the HashToFuncs map for serialization.
  const HashFuncsMapType &getFunctionMap() const { return HashToFuncs; }

  /// Get the NameToId vector for serialization.
  const SmallVector<std::string> getNames() const { return IdToName; }

  /// Get an existing ID associated with the given name or create a new ID if it
  /// doesn't exist.
  unsigned getIdOrCreateForName(StringRef Name);

  /// Get the name associated with a given ID
  std::optional<std::string> getNameForId(unsigned Id) const;

  /// Insert a `StableFunction` object into the function map. This method
  /// handles the uniquing of string names and create a `StableFunctionEntry`
  /// for insertion.
  void insert(const StableFunction &Func);

  /// Merge a \p OtherMap into this function map.
  void merge(const StableFunctionMap &OtherMap);

  /// \returns true if there is no stable function entry.
  bool empty() const { return size() == 0; }

  enum SizeType {
    UniqueHashCount,        // The number of unique hashes in HashToFuncs.
    TotalFunctionCount,     // The number of total functions in HashToFuncs.
    MergeableFunctionCount, // The number of functions that can be merged based
                            // on their hash.
  };

  /// \returns the size of StableFunctionMap.
  /// \p Type is the type of size to return.
  size_t size(SizeType Type = UniqueHashCount) const;

  /// Finalize the stable function map by trimming content.
  void finalize(bool SkipTrim = false);

private:
  /// Insert a `StableFunctionEntry` into the function map directly. This
  /// method assumes that string names have already been uniqued and the
  /// `StableFunctionEntry` is ready for insertion.
  void insert(std::unique_ptr<StableFunctionEntry> FuncEntry) {
    assert(!Finalized && "Cannot insert after finalization");
    HashToFuncs[FuncEntry->Hash].emplace_back(std::move(FuncEntry));
  }

  /// A map from a stable_hash to a vector of functions with that hash.
  HashFuncsMapType HashToFuncs;
  /// A vector of strings to hold names.
  SmallVector<std::string> IdToName;
  /// A map from StringRef (name) to an ID.
  StringMap<unsigned> NameToId;
  /// True if the function map is finalized with minimal content.
  bool Finalized = false;

  friend struct StableFunctionMapRecord;
};

} // namespace llvm

#endif
