//===- DataAccessProf.h - Data access profile format support ---------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains support to construct and use data access profiles.
//
// For the original RFC of this pass please see
// https://discourse.llvm.org/t/rfc-profile-guided-static-data-partitioning/83744
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PROFILEDATA_DATAACCESSPROF_H_
#define LLVM_PROFILEDATA_DATAACCESSPROF_H_

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfoVariant.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/StringSaver.h"

#include <cstdint>
#include <variant>

namespace llvm {

namespace data_access_prof {
// The location of data in the source code.
struct DataLocation {
  // The filename where the data is located.
  StringRef FileName;
  // The line number in the source code.
  uint32_t Line;
};

// The data access profiles for a symbol.
struct DataAccessProfRecord {
  DataAccessProfRecord(uint64_t SymbolID, uint64_t AccessCount,
                       bool IsStringLiteral)
      : SymbolID(SymbolID), AccessCount(AccessCount),
        IsStringLiteral(IsStringLiteral) {}

  // Represents a data symbol. The semantic comes in two forms: a symbol index
  // for symbol name if `IsStringLiteral` is false, or the hash of a string
  // content if `IsStringLiteral` is true. Required.
  uint64_t SymbolID;

  // The access count of symbol. Required.
  uint64_t AccessCount;

  // True iff this is a record for string literal (symbols with name pattern
  // `.str.*` in the symbol table). Required.
  bool IsStringLiteral;

  // The locations of data in the source code. Optional.
  llvm::SmallVector<DataLocation, 0> Locations;
};

/// Encapsulates the data access profile data and the methods to operate on it.
/// This class provides profile look-up, serialization and deserialization.
class DataAccessProfData {
public:
  // SymbolID is either a string representing symbol name, or a uint64_t
  // representing the content hash of a string literal.
  using SymbolID = std::variant<StringRef, uint64_t>;
  using StringToIndexMap = llvm::MapVector<StringRef, uint64_t>;

  DataAccessProfData() : Saver(Allocator) {}

  /// Serialize profile data to the output stream.
  /// Storage layout:
  /// - Serialized strings.
  /// - The encoded hashes.
  /// - Records.
  Error serialize(ProfOStream &OS) const;

  /// Deserialize this class from the given buffer.
  Error deserialize(const unsigned char *&Ptr);

  /// Returns a pointer of profile record for \p SymbolID, or nullptr if there
  /// isn't a record. Internally, this function will canonicalize the symbol
  /// name before the lookup.
  const DataAccessProfRecord *getProfileRecord(const SymbolID SymID) const;

  /// Returns true if \p SymID is seen in profiled binaries and cold.
  bool isKnownColdSymbol(const SymbolID SymID) const;

  /// Methods to add symbolized data access profile. Returns error if duplicated
  /// symbol names or content hashes are seen. The user of this class should
  /// aggregate counters that correspond to the same symbol name or with the
  /// same string literal hash before calling 'add*' methods.
  Error setDataAccessProfile(SymbolID SymbolID, uint64_t AccessCount);
  /// Similar to the method above, for records with \p Locations representing
  /// the `filename:line` where this symbol shows up. Note because of linker's
  /// merge of identical symbols (e.g., unnamed_addr string literals), one
  /// symbol is likely to have multiple locations.
  Error setDataAccessProfile(SymbolID SymbolID, uint64_t AccessCount,
                             const llvm::SmallVector<DataLocation> &Locations);
  Error addKnownSymbolWithoutSamples(SymbolID SymbolID);

  /// Returns an iterable StringRef for strings in the order they are added.
  /// Each string may be a symbol name or a file name.
  auto getStrings() const {
    ArrayRef<std::pair<StringRef, uint64_t>> RefSymbolNames(
        StrToIndexMap.begin(), StrToIndexMap.end());
    return llvm::make_first_range(RefSymbolNames);
  }

  /// Returns array reference for various internal data structures.
  ArrayRef<
      std::pair<std::variant<StringRef, uint64_t>, DataAccessProfRecord>>
  getRecords() const {
    return Records.getArrayRef();
  }
  ArrayRef<StringRef> getKnownColdSymbols() const {
    return KnownColdSymbols.getArrayRef();
  }
  ArrayRef<uint64_t> getKnownColdHashes() const {
    return KnownColdHashes.getArrayRef();
  }

private:
  /// Serialize the symbol strings into the output stream.
  Error serializeSymbolsAndFilenames(ProfOStream &OS) const;

  /// Deserialize the symbol strings from \p Ptr and increment \p Ptr to the
  /// start of the next payload.
  Error deserializeSymbolsAndFilenames(const unsigned char *&Ptr,
                                       const uint64_t NumSampledSymbols,
                                       uint64_t NumColdKnownSymbols);

  /// Decode the records and increment \p Ptr to the start of the next payload.
  Error deserializeRecords(const unsigned char *&Ptr);

  /// A helper function to compute a storage index for \p SymbolID.
  uint64_t getEncodedIndex(const SymbolID SymbolID) const;

  // Keeps owned copies of the input strings.
  // NOTE: Keep `Saver` initialized before other class members that reference
  // its string copies and destructed after they are destructed.
  llvm::BumpPtrAllocator Allocator;
  llvm::UniqueStringSaver Saver;

  // `Records` stores the records and `SymbolToRecordIndex` maps a symbol ID to
  // its record index.
  MapVector<SymbolID, DataAccessProfRecord> Records;

  // Use MapVector to keep input order of strings for serialization and
  // deserialization.
  StringToIndexMap StrToIndexMap;
  llvm::SetVector<uint64_t> KnownColdHashes;
  llvm::SetVector<StringRef> KnownColdSymbols;
};

} // namespace data_access_prof
} // namespace llvm

#endif // LLVM_PROFILEDATA_DATAACCESSPROF_H_
