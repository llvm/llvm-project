//===- llvm/CAS/CASReference.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CAS_CASREFERENCE_H
#define LLVM_CAS_CASREFERENCE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {

class raw_ostream;

namespace cas {

class CASDB;

class ObjectHandle;
class ObjectRef;

/// Base class for references to things in \a CASDB.
class ReferenceBase {
protected:
  struct DenseMapEmptyTag {};
  struct DenseMapTombstoneTag {};
  static constexpr uint64_t getDenseMapEmptyRef() { return -1ULL; }
  static constexpr uint64_t getDenseMapTombstoneRef() { return -2ULL; }

public:
  /// Get an internal reference.
  uint64_t getInternalRef(const CASDB &ExpectedCAS) const {
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
    assert(CAS == &ExpectedCAS && "Extracting reference for the wrong CAS");
#endif
    return InternalRef;
  }

  unsigned getDenseMapHash() const {
    return (unsigned)llvm::hash_value(InternalRef);
  }
  bool isDenseMapEmpty() const { return InternalRef == getDenseMapEmptyRef(); }
  bool isDenseMapTombstone() const {
    return InternalRef == getDenseMapTombstoneRef();
  }
  bool isDenseMapSentinel() const {
    return isDenseMapEmpty() || isDenseMapTombstone();
  }

protected:
  void print(raw_ostream &OS, const ObjectHandle &This) const;
  void print(raw_ostream &OS, const ObjectRef &This) const;

  bool hasSameInternalRef(const ReferenceBase &RHS) const {
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
    assert(
        (isDenseMapSentinel() || RHS.isDenseMapSentinel() || CAS == RHS.CAS) &&
        "Cannot compare across CAS instances");
#endif
    return InternalRef == RHS.InternalRef;
  }

protected:
  friend class CASDB;
  ReferenceBase(const CASDB *CAS, uint64_t InternalRef, bool IsHandle)
      : InternalRef(InternalRef) {
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
    this->CAS = CAS;
#endif
    assert(InternalRef != getDenseMapEmptyRef() && "Reserved for DenseMapInfo");
    assert(InternalRef != getDenseMapTombstoneRef() &&
           "Reserved for DenseMapInfo");
  }
  explicit ReferenceBase(DenseMapEmptyTag)
      : InternalRef(getDenseMapEmptyRef()) {}
  explicit ReferenceBase(DenseMapTombstoneTag)
      : InternalRef(getDenseMapTombstoneRef()) {}

private:
  uint64_t InternalRef;

#if LLVM_ENABLE_ABI_BREAKING_CHECKS
  const CASDB *CAS = nullptr;
#endif
};

/// Reference to an object in a \a CASDB instance.
///
/// If you have an ObjectRef, you know the object exists, and you can point at
/// it from new nodes with \a CASDB::store(), but you don't know anything
/// about it. "Loading" the object is a separate step that may not have
/// happened yet, and which can fail (due to filesystem corruption) or
/// introduce latency (if downloading from a remote store).
///
/// \a CASDB::store() takes a list of these, and these are returned by \a
/// CASDB::forEachRef() and \a CASDB::readRef(), which are accessors for nodes,
/// and \a CASDB::getReference().
///
/// \a CASDB::load() will load the referenced object, and returns \a
/// ObjectHandle, a variant that knows what kind of entity it is. \a
/// CASDB::getReferenceKind() can expect the type of reference without asking
/// for unloaded objects to be loaded.
///
/// This is a wrapper around a \c uint64_t (and a \a CASDB instance when
/// assertions are on). If necessary, it can be deconstructed and reconstructed
/// using \a Reference::getInternalRef() and \a
/// Reference::getFromInternalRef(), but clients aren't expected to need to do
/// this. These both require the right \a CASDB instance.
class ObjectRef : public ReferenceBase {
  struct DenseMapTag {};

public:
  friend bool operator==(const ObjectRef &LHS, const ObjectRef &RHS) {
    return LHS.hasSameInternalRef(RHS);
  }
  friend bool operator!=(const ObjectRef &LHS, const ObjectRef &RHS) {
    return !(LHS == RHS);
  }

  /// Allow a reference to be recreated after it's deconstructed.
  static ObjectRef getFromInternalRef(const CASDB &CAS, uint64_t InternalRef) {
    return ObjectRef(CAS, InternalRef);
  }

  static ObjectRef getDenseMapEmptyKey() {
    return ObjectRef(DenseMapEmptyTag{});
  }
  static ObjectRef getDenseMapTombstoneKey() {
    return ObjectRef(DenseMapTombstoneTag{});
  }

  /// Print internal ref and/or CASID. Only suitable for debugging.
  void print(raw_ostream &OS) const { return ReferenceBase::print(OS, *this); }

  LLVM_DUMP_METHOD void dump() const;

private:
  friend class CASDB;
  friend class ReferenceBase;
  using ReferenceBase::ReferenceBase;
  ObjectRef(const CASDB &CAS, uint64_t InternalRef)
      : ReferenceBase(&CAS, InternalRef, /*IsHandle=*/false) {
    assert(InternalRef != -1ULL && "Reserved for DenseMapInfo");
    assert(InternalRef != -2ULL && "Reserved for DenseMapInfo");
  }
  explicit ObjectRef(DenseMapEmptyTag T) : ReferenceBase(T) {}
  explicit ObjectRef(DenseMapTombstoneTag T) : ReferenceBase(T) {}
  explicit ObjectRef(ReferenceBase) = delete;
};

/// Handle to a loaded object in a \a CASDB instance.
///
/// ObjectHandle encapulates a *loaded* object in the CAS. You need one
/// of these to inspect the content of an object: to look at its stored
/// data and references.
class ObjectHandle : public ReferenceBase {
public:
  friend bool operator==(const ObjectHandle &LHS, const ObjectHandle &RHS) {
    return LHS.hasSameInternalRef(RHS);
  }
  friend bool operator!=(const ObjectHandle &LHS, const ObjectHandle &RHS) {
    return !(LHS == RHS);
  }

  /// Print internal ref and/or CASID. Only suitable for debugging.
  void print(raw_ostream &OS) const { return ReferenceBase::print(OS, *this); }

  LLVM_DUMP_METHOD void dump() const;

private:
  friend class CASDB;
  friend class ReferenceBase;
  using ReferenceBase::ReferenceBase;
  explicit ObjectHandle(ReferenceBase) = delete;
  ObjectHandle(const CASDB &CAS, uint64_t InternalRef)
      : ReferenceBase(&CAS, InternalRef, /*IsHandle=*/true) {}
};

} // namespace cas

template <> struct DenseMapInfo<cas::ObjectRef> {
  static cas::ObjectRef getEmptyKey() {
    return cas::ObjectRef::getDenseMapEmptyKey();
  }

  static cas::ObjectRef getTombstoneKey() {
    return cas::ObjectRef::getDenseMapTombstoneKey();
  }

  static unsigned getHashValue(cas::ObjectRef Ref) {
    return Ref.getDenseMapHash();
  }

  static bool isEqual(cas::ObjectRef LHS, cas::ObjectRef RHS) {
    return LHS == RHS;
  }
};

} // namespace llvm

#endif // LLVM_CAS_CASREFERENCE_H
