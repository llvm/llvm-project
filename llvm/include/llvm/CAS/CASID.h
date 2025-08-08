//===- llvm/CAS/CASID.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CAS_CASID_H
#define LLVM_CAS_CASID_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace llvm {

class raw_ostream;

namespace cas {

class CASID;

/// Context for CAS identifiers.
class CASContext {
  virtual void anchor();

public:
  virtual ~CASContext() = default;

  /// Get an identifer for the schema used by this CAS context. Two CAS
  /// instances should return \c true for this identifier if and only if their
  /// CASIDs are safe to compare by hash. This is used by \a
  /// CASID::equalsImpl().
  virtual StringRef getHashSchemaIdentifier() const = 0;

protected:
  /// Print \p ID to \p OS.
  virtual void printIDImpl(raw_ostream &OS, const CASID &ID) const = 0;

  friend class CASID;
};

/// Unique identifier for a CAS object.
///
/// Locally, stores an internal CAS identifier that's specific to a single CAS
/// instance. It's guaranteed not to change across the view of that CAS, but
/// might change between runs.
///
/// It also has \a CASIDContext pointer to allow comparison of these
/// identifiers. If two CASIDs are from the same CASIDContext, they can be
/// compared directly. If they are, then \a
/// CASIDContext::getHashSchemaIdentifier() is compared to see if they can be
/// compared by hash, in which case the result of \a getHash() is compared.
class CASID {
public:
  void dump() const;

  friend raw_ostream &operator<<(raw_ostream &OS, const CASID &ID) {
    ID.print(OS);
    return OS;
  }

  /// Print CASID.
  void print(raw_ostream &OS) const {
    return getContext().printIDImpl(OS, *this);
  }

  /// Return a printable string for CASID.
  std::string toString() const;

  ArrayRef<uint8_t> getHash() const {
    return arrayRefFromStringRef<uint8_t>(Hash);
  }

  friend bool operator==(const CASID &LHS, const CASID &RHS) {
    if (LHS.Context == RHS.Context)
      return LHS.Hash == RHS.Hash;

    // EmptyKey or TombstoneKey.
    if (!LHS.Context || !RHS.Context)
      return false;

    // CASIDs are equal when they have the same hash schema and same hash value.
    return LHS.Context->getHashSchemaIdentifier() ==
               RHS.Context->getHashSchemaIdentifier() &&
           LHS.Hash == RHS.Hash;
  }

  friend bool operator!=(const CASID &LHS, const CASID &RHS) {
    return !(LHS == RHS);
  }

  friend hash_code hash_value(const CASID &ID) {
    ArrayRef<uint8_t> Hash = ID.getHash();
    return hash_combine_range(Hash.begin(), Hash.end());
  }

  const CASContext &getContext() const {
    assert(Context && "Tombstone or empty key for DenseMap?");
    return *Context;
  }

  static CASID getDenseMapEmptyKey() {
    return CASID(nullptr, DenseMapInfo<StringRef>::getEmptyKey());
  }
  static CASID getDenseMapTombstoneKey() {
    return CASID(nullptr, DenseMapInfo<StringRef>::getTombstoneKey());
  }

  CASID() = delete;

  /// Create CASID from CASContext and raw hash bytes.
  static CASID create(const CASContext *Context, StringRef Hash) {
    return CASID(Context, Hash);
  }

private:
  CASID(const CASContext *Context, StringRef Hash)
      : Context(Context), Hash(Hash) {}

  const CASContext *Context;
  SmallString<32> Hash;
};

} // namespace cas

template <> struct DenseMapInfo<cas::CASID> {
  static cas::CASID getEmptyKey() { return cas::CASID::getDenseMapEmptyKey(); }

  static cas::CASID getTombstoneKey() {
    return cas::CASID::getDenseMapTombstoneKey();
  }

  static unsigned getHashValue(cas::CASID ID) {
    return (unsigned)hash_value(ID);
  }

  static bool isEqual(cas::CASID LHS, cas::CASID RHS) { return LHS == RHS; }
};

} // namespace llvm

#endif // LLVM_CAS_CASID_H
