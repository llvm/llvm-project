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

#include "llvm/ADT/DenseMapInfoVariant.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/StringSaver.h"

#include <cstdint>
#include <optional>
#include <variant>

namespace llvm {

namespace memprof {

/// The location of data in the source code. Used by profile lookup API.
struct SourceLocation {
  SourceLocation(StringRef FileNameRef, uint32_t Line)
      : FileName(FileNameRef.str()), Line(Line) {}

  // Empty constructor is used in yaml conversion.
  SourceLocation() {}
  /// The filename where the data is located.
  std::string FileName;
  /// The line number in the source code.
  uint32_t Line;
};

namespace internal {

// Conceptually similar to SourceLocation except that FileNames are StringRef of
// which strings are owned by `DataAccessProfData`. Used by `DataAccessProfData`
// to represent data locations internally.
struct SourceLocationRef {
  SourceLocationRef(StringRef FileNameRef, uint32_t Line)
      : FileName(FileNameRef), Line(Line) {}
  // The filename where the data is located.
  StringRef FileName;
  // The line number in the source code.
  uint32_t Line;
};

// The data access profiles for a symbol. Used by `DataAccessProfData`
// to represent records internally.
struct DataAccessProfRecordRef {
  DataAccessProfRecordRef(uint64_t SymbolID, uint64_t AccessCount,
                          bool IsStringLiteral)
      : SymbolID(SymbolID), AccessCount(AccessCount),
        IsStringLiteral(IsStringLiteral) {}

  // Represents a data symbol. The semantic comes in two forms: a symbol index
  // for symbol name if `IsStringLiteral` is false, or the hash of a string
  // content if `IsStringLiteral` is true. For most of the symbolizable static
  // data, the mangled symbol names remain stable relative to the source code
  // and therefore used to identify symbols across binary releases. String
  // literals have unstable name patterns like `.str.N[.llvm.hash]`, so we use
  // the content hash instead. This is a required field.
  uint64_t SymbolID;

  // The access count of symbol. Required.
  uint64_t AccessCount;

  // True iff this is a record for string literal (symbols with name pattern
  // `.str.*` in the symbol table). Required.
  bool IsStringLiteral;

  // The locations of data in the source code. Optional.
  llvm::SmallVector<SourceLocationRef, 0> Locations;
};
} // namespace internal

// SymbolID is either a string representing symbol name if the symbol has
// stable mangled name relative to source code, or a uint64_t representing the
// content hash of a string literal (with unstable name patterns like
// `.str.N[.llvm.hash]`). The StringRef is owned by the class's saver object.
using SymbolHandleRef = std::variant<StringRef, uint64_t>;

// The senamtic is the same as `SymbolHandleRef` above. The strings are owned.
using SymbolHandle = std::variant<std::string, uint64_t>;

/// The data access profiles for a symbol.
struct DataAccessProfRecord {
public:
  DataAccessProfRecord(SymbolHandleRef SymHandleRef, uint64_t AccessCount,
                       ArrayRef<internal::SourceLocationRef> LocRefs)
      : AccessCount(AccessCount) {
    if (std::holds_alternative<StringRef>(SymHandleRef)) {
      SymHandle = std::get<StringRef>(SymHandleRef).str();
    } else
      SymHandle = std::get<uint64_t>(SymHandleRef);

    for (auto Loc : LocRefs)
      Locations.emplace_back(Loc.FileName, Loc.Line);
  }
  // Empty constructor is used in yaml conversion.
  DataAccessProfRecord() {}
  SymbolHandle SymHandle;
  uint64_t AccessCount;
  // The locations of data in the source code. Optional.
  SmallVector<SourceLocation> Locations;
};

/// Encapsulates the data access profile data and the methods to operate on
/// it. This class provides profile look-up, serialization and
/// deserialization.
class DataAccessProfData {
public:
  // Use MapVector to keep input order of strings for serialization and
  // deserialization.
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

  /// Returns a profile record for \p SymbolID, or std::nullopt if there
  /// isn't a record. Internally, this function will canonicalize the symbol
  /// name before the lookup.
  std::optional<DataAccessProfRecord>
  getProfileRecord(const SymbolHandleRef SymID) const;

  /// Returns true if \p SymID is seen in profiled binaries and cold.
  bool isKnownColdSymbol(const SymbolHandleRef SymID) const;

  /// Methods to set symbolized data access profile. Returns error if
  /// duplicated symbol names or content hashes are seen. The user of this
  /// class should aggregate counters that correspond to the same symbol name
  /// or with the same string literal hash before calling 'set*' methods.
  Error setDataAccessProfile(SymbolHandleRef SymbolID, uint64_t AccessCount);
  /// Similar to the method above, for records with \p Locations representing
  /// the `filename:line` where this symbol shows up. Note because of linker's
  /// merge of identical symbols (e.g., unnamed_addr string literals), one
  /// symbol is likely to have multiple locations.
  Error setDataAccessProfile(SymbolHandleRef SymbolID, uint64_t AccessCount,
                             ArrayRef<SourceLocation> Locations);
  /// Add a symbol that's seen in the profiled binary without samples.
  Error addKnownSymbolWithoutSamples(SymbolHandleRef SymbolID);

  /// The following methods return array reference for various internal data
  /// structures.
  ArrayRef<StringToIndexMap::value_type> getStrToIndexMapRef() const {
    return StrToIndexMap.getArrayRef();
  }
  ArrayRef<
      MapVector<SymbolHandleRef, internal::DataAccessProfRecordRef>::value_type>
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
                                       const uint64_t NumColdKnownSymbols);

  /// Decode the records and increment \p Ptr to the start of the next
  /// payload.
  Error deserializeRecords(const unsigned char *&Ptr);

  /// A helper function to compute a storage index for \p SymbolID.
  uint64_t getEncodedIndex(const SymbolHandleRef SymbolID) const;

  // Keeps owned copies of the input strings.
  // NOTE: Keep `Saver` initialized before other class members that reference
  // its string copies and destructed after they are destructed.
  llvm::BumpPtrAllocator Allocator;
  llvm::UniqueStringSaver Saver;

  // `Records` stores the records.
  MapVector<SymbolHandleRef, internal::DataAccessProfRecordRef> Records;

  StringToIndexMap StrToIndexMap;
  llvm::SetVector<uint64_t> KnownColdHashes;
  llvm::SetVector<StringRef> KnownColdSymbols;
};

} // namespace memprof
} // namespace llvm

#endif // LLVM_PROFILEDATA_DATAACCESSPROF_H_
