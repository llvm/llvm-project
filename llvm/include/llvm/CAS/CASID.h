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
#include "llvm/ADT/StringRef.h"

namespace llvm {

class raw_ostream;

namespace cas {

class CASID;

/// Context for CAS identifiers.
///
/// FIXME: Rename to ObjectContext.
class CASIDContext {
  virtual void anchor();

public:
  virtual ~CASIDContext() = default;

  /// Get an identifer for the schema used by this CAS context. Two CAS
  /// instances should return \c true for this identifier if and only if their
  /// CASIDs are safe to compare by hash. This is used by \a
  /// CASID::equalsImpl().
  virtual StringRef getHashSchemaIdentifier() const = 0;

protected:
  /// Get the hash for \p ID. Implementation for \a CASID::getHash().
  virtual ArrayRef<uint8_t> getHashImpl(const CASID &ID) const = 0;

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
///
/// FIXME: Rename to ObjectID (and rename file to CASObjectID.h?).
class CASID {
public:
  void dump() const;
  void print(raw_ostream &OS) const {
    return getContext().printIDImpl(OS, *this);
  }
  friend raw_ostream &operator<<(raw_ostream &OS, const CASID &ID) {
    ID.print(OS);
    return OS;
  }
  std::string toString() const;

  ArrayRef<uint8_t> getHash() const { return getContext().getHashImpl(*this); }

  friend bool operator==(CASID LHS, CASID RHS) {
    // If it's the same CAS (or both nullptr), then the IDs are directly
    // comparable.
    if (LHS.Context == RHS.Context)
      return LHS.InternalID == RHS.InternalID;

    // Check if one CAS is nullptr, indicating a tombstone or empty key for
    // DenseMap, and return false if so.
    if (!LHS.Context || !RHS.Context)
      return false;

    // Check if the schemas match.
    if (LHS.Context->getHashSchemaIdentifier() !=
        RHS.Context->getHashSchemaIdentifier())
      return false;

    // Compare the hashes.
    return LHS.getHash() == RHS.getHash();
  }

  friend bool operator!=(CASID LHS, CASID RHS) { return !(LHS == RHS); }

  friend hash_code hash_value(CASID ID) {
    ArrayRef<uint8_t> Hash = ID.getHash();
    return hash_combine_range(Hash.begin(), Hash.end());
  }

  const CASIDContext &getContext() const {
    assert(Context && "Tombstone or empty key for DenseMap?");
    return *Context;
  }

  /// Get the internal ID. Asserts that \p ExpectedContext is the Context that
  /// this ID comes from, to help catch usage errors.
  uint64_t getInternalID(const CASIDContext &ExpectedContext) const {
    assert(&ExpectedContext == Context);
    return InternalID;
  }

  static CASID getDenseMapEmptyKey() { return CASID(-1ULL, nullptr); }
  static CASID getDenseMapTombstoneKey() { return CASID(-2ULL, nullptr); }

  static CASID getFromInternalID(const CASIDContext &Context,
                                 uint64_t InternalID) {
    return CASID(InternalID, &Context);
  }

  CASID() = delete;

private:
  CASID(uint64_t InternalID, const CASIDContext *Context)
      : InternalID(InternalID), Context(Context) {}

  bool equalsImpl(CASID RHS) const;

  uint64_t InternalID = 0;
  const CASIDContext *Context = nullptr;
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
