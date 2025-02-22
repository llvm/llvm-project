//===- OnDiskGraphDB.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CAS_ONDISKGRAPHDB_H
#define LLVM_CAS_ONDISKGRAPHDB_H

#include "llvm/ADT/PointerUnion.h"
#include "llvm/CAS/OnDiskHashMappedTrie.h"

namespace llvm::cas::ondisk {

/// 8B reference.
class InternalRef {
public:
  FileOffset getFileOffset() const { return FileOffset(getRawOffset()); }

  uint64_t getRawData() const { return Data; }
  uint64_t getRawOffset() const { return Data; }

  static InternalRef getFromRawData(uint64_t Data) { return InternalRef(Data); }

  static InternalRef getFromOffset(FileOffset Offset) {
    return InternalRef(Offset.get());
  }

  friend bool operator==(InternalRef LHS, InternalRef RHS) {
    return LHS.Data == RHS.Data;
  }

private:
  InternalRef(FileOffset Offset) : Data((uint64_t)Offset.get()) {}
  InternalRef(uint64_t Data) : Data(Data) {}
  uint64_t Data;
};

/// 4B reference.
class InternalRef4B {
public:
  FileOffset getFileOffset() const { return FileOffset(Data); }

  uint32_t getRawData() const { return Data; }

  /// Shrink to 4B reference.
  static std::optional<InternalRef4B> tryToShrink(InternalRef Ref) {
    uint64_t Offset = Ref.getRawOffset();
    if (Offset > UINT32_MAX)
      return std::nullopt;

    return InternalRef4B(Offset);
  }

  operator InternalRef() const {
    return InternalRef::getFromOffset(getFileOffset());
  }

private:
  friend class InternalRef;
  InternalRef4B(uint32_t Data) : Data(Data) {}
  uint32_t Data;
};

/// Array of internal node references.
class InternalRefArrayRef {
public:
  size_t size() const { return Size; }
  bool empty() const { return !Size; }

  class iterator
      : public iterator_facade_base<iterator, std::random_access_iterator_tag,
                                    const InternalRef> {
  public:
    bool operator==(const iterator &RHS) const { return I == RHS.I; }
    InternalRef operator*() const {
      if (auto *Ref = dyn_cast<const InternalRef *>(I))
        return *Ref;
      return InternalRef(*cast<const InternalRef4B *>(I));
    }
    bool operator<(const iterator &RHS) const {
      assert(isa<const InternalRef *>(I) == isa<const InternalRef *>(RHS.I));
      if (auto *Ref = dyn_cast<const InternalRef *>(I))
        return Ref < cast<const InternalRef *>(RHS.I);
      return cast<const InternalRef4B *>(I) -
             cast<const InternalRef4B *>(RHS.I);
    }
    ptrdiff_t operator-(const iterator &RHS) const {
      assert(isa<const InternalRef *>(I) == isa<const InternalRef *>(RHS.I));
      if (auto *Ref = dyn_cast<const InternalRef *>(I))
        return Ref - cast<const InternalRef *>(RHS.I);
      return cast<const InternalRef4B *>(I) -
             cast<const InternalRef4B *>(RHS.I);
    }
    iterator &operator+=(ptrdiff_t N) {
      if (auto *Ref = dyn_cast<const InternalRef *>(I))
        I = Ref + N;
      else
        I = cast<const InternalRef4B *>(I) + N;
      return *this;
    }
    iterator &operator-=(ptrdiff_t N) {
      if (auto *Ref = dyn_cast<const InternalRef *>(I))
        I = Ref - N;
      else
        I = cast<const InternalRef4B *>(I) - N;
      return *this;
    }
    InternalRef operator[](ptrdiff_t N) const { return *(this->operator+(N)); }

    iterator() = default;

    uint64_t getOpaqueData() const { return uintptr_t(I.getOpaqueValue()); }

    static iterator fromOpaqueData(uint64_t Opaque) {
      return iterator(
          PointerUnion<const InternalRef *,
                       const InternalRef4B *>::getFromOpaqueValue((void *)
                                                                      Opaque));
    }

  private:
    friend class InternalRefArrayRef;
    explicit iterator(
        PointerUnion<const InternalRef *, const InternalRef4B *> I)
        : I(I) {}
    PointerUnion<const InternalRef *, const InternalRef4B *> I;
  };

  bool operator==(const InternalRefArrayRef &RHS) const {
    return size() == RHS.size() && std::equal(begin(), end(), RHS.begin());
  }

  iterator begin() const { return iterator(Begin); }
  iterator end() const { return begin() + Size; }

  /// Array accessor.
  InternalRef operator[](ptrdiff_t N) const { return begin()[N]; }

  bool is4B() const { return isa<const InternalRef4B *>(Begin); }
  bool is8B() const { return isa<const InternalRef *>(Begin); }

  ArrayRef<uint8_t> getBuffer() const {
    if (is4B()) {
      auto *B = cast<const InternalRef4B *>(Begin);
      return ArrayRef((const uint8_t *)B, sizeof(InternalRef4B) * Size);
    } else {
      auto *B = cast<const InternalRef *>(Begin);
      return ArrayRef((const uint8_t *)B, sizeof(InternalRef) * Size);
    }
  }

  InternalRefArrayRef(std::nullopt_t = std::nullopt) {
    // This is useful so that all the casts in the \p iterator functions can
    // operate without needing to check for a null value.
    static InternalRef PlaceHolder = InternalRef::getFromRawData(0);
    Begin = &PlaceHolder;
  }

  InternalRefArrayRef(ArrayRef<InternalRef> Refs)
      : Begin(Refs.begin()), Size(Refs.size()) {}

  InternalRefArrayRef(ArrayRef<InternalRef4B> Refs)
      : Begin(Refs.begin()), Size(Refs.size()) {}

private:
  PointerUnion<const InternalRef *, const InternalRef4B *> Begin;
  size_t Size = 0;
};

struct OnDiskContent;

/// Reference to a node. The node's data may not be stored in the database.
/// An \p ObjectID instance can only be used with the \p OnDiskGraphDB instance
/// it came from. \p ObjectIDs from different \p OnDiskGraphDB instances are not
/// comparable.
class ObjectID {
public:
  uint64_t getOpaqueData() const { return Opaque; }

  static ObjectID fromOpaqueData(uint64_t Opaque) { return ObjectID(Opaque); }

  friend bool operator==(const ObjectID &LHS, const ObjectID &RHS) {
    return LHS.Opaque == RHS.Opaque;
  }
  friend bool operator!=(const ObjectID &LHS, const ObjectID &RHS) {
    return !(LHS == RHS);
  }

private:
  explicit ObjectID(uint64_t Opaque) : Opaque(Opaque) {}
  uint64_t Opaque;
};

/// Handle for a loaded node object.
class ObjectHandle {
public:
  uint64_t getOpaqueData() const { return Opaque; }

  static ObjectHandle fromOpaqueData(uint64_t Opaque) {
    return ObjectHandle(Opaque);
  }

  friend bool operator==(const ObjectHandle &LHS, const ObjectHandle &RHS) {
    return LHS.Opaque == RHS.Opaque;
  }
  friend bool operator!=(const ObjectHandle &LHS, const ObjectHandle &RHS) {
    return !(LHS == RHS);
  }

private:
  explicit ObjectHandle(uint64_t Opaque) : Opaque(Opaque) {}
  uint64_t Opaque;
};

class object_refs_iterator
    : public iterator_facade_base<object_refs_iterator,
                                  std::random_access_iterator_tag, ObjectID> {
public:
  bool operator==(const object_refs_iterator &RHS) const { return I == RHS.I; }
  ObjectID operator*() const {
    return ObjectID::fromOpaqueData((*I).getRawData());
  }
  bool operator<(const object_refs_iterator &RHS) const { return I < RHS.I; }
  ptrdiff_t operator-(const object_refs_iterator &RHS) const {
    return I - RHS.I;
  }
  object_refs_iterator &operator+=(ptrdiff_t N) {
    I += N;
    return *this;
  }
  object_refs_iterator &operator-=(ptrdiff_t N) {
    I -= N;
    return *this;
  }
  ObjectID operator[](ptrdiff_t N) const { return *(this->operator+(N)); }

  object_refs_iterator() = default;
  object_refs_iterator(InternalRefArrayRef::iterator I) : I(I) {}

  uint64_t getOpaqueData() const { return I.getOpaqueData(); }

  static object_refs_iterator fromOpaqueData(uint64_t Opaque) {
    return InternalRefArrayRef::iterator::fromOpaqueData(Opaque);
  }

private:
  InternalRefArrayRef::iterator I;
};

using object_refs_range = llvm::iterator_range<object_refs_iterator>;

/// On-disk CAS nodes database, independent of a particular hashing algorithm.
class OnDiskGraphDB {
public:
  /// Associate data & references with a particular object ID. If there is
  /// already a record for this object the operation is a no-op. \param ID the
  /// object ID to associate the data & references with. \param Refs references
  /// \param Data data buffer.
  Error store(ObjectID ID, ArrayRef<ObjectID> Refs, ArrayRef<char> Data);

  /// \returns \p nullopt if the object associated with \p Ref does not exist.
  Expected<std::optional<ObjectHandle>> load(ObjectID Ref);

  /// \returns the hash bytes digest for the object reference.
  ArrayRef<uint8_t> getDigest(ObjectID Ref) const {
    return getDigest(getInternalRef(Ref));
  }

  /// Form a reference for the provided hash. The reference can be used as part
  /// of a CAS object even if it's not associated with an object yet.
  Expected<ObjectID> getReference(ArrayRef<uint8_t> Hash);

  /// Get an existing reference to the object \p Digest.
  ///
  /// Returns \p nullopt if the object is not stored in this CAS.
  std::optional<ObjectID> getExistingReference(ArrayRef<uint8_t> Digest);

  /// Check whether the object associated with \p Ref is stored in the CAS.
  /// Note that this function will fault-in according to the policy.
  Expected<bool> isMaterialized(ObjectID Ref);

  /// Check whether the object associated with \p Ref is stored in the CAS.
  /// Note that this function does not fault-in.
  bool containsObject(ObjectID Ref) const {
    return containsObject(Ref, /*CheckUpstream=*/true);
  }

  /// \returns the data part of the provided object handle.
  ArrayRef<char> getObjectData(ObjectHandle Node) const;

  object_refs_range getObjectRefs(ObjectHandle Node) const {
    InternalRefArrayRef Refs = getInternalRefs(Node);
    return make_range(Refs.begin(), Refs.end());
  }

  /// \returns Total size of stored objects.
  ///
  /// NOTE: There's a possibility that the returned size is not including a
  /// large object if the process crashed right at the point of inserting it.
  size_t getStorageSize() const;

  /// \returns The precentage of space utilization of hard space limits.
  ///
  /// Return value is an integer between 0 and 100 for percentage.
  unsigned getHardStorageLimitUtilization() const;

  void print(raw_ostream &OS) const;

  /// Hashing function type for validation.
  using HashingFuncT = function_ref<void(
      ArrayRef<ArrayRef<uint8_t>>, ArrayRef<char>, SmallVectorImpl<uint8_t> &)>;
  Error validate(bool Deep, HashingFuncT Hasher) const;

  /// How to fault-in nodes if an upstream database is used.
  enum class FaultInPolicy {
    /// Copy only the requested node.
    SingleNode,
    /// Copy the the entire graph of a node.
    FullTree,
  };

  /// Open the on-disk store from a directory.
  ///
  /// \param Path directory for the on-disk store. The directory will be created
  /// if it doesn't exist.
  /// \param HashName Identifier name for the hashing algorithm that is going to
  /// be used.
  /// \param HashByteSize Size for the object digest hash bytes.
  /// \param UpstreamDB Optional on-disk store to be used for faulting-in nodes
  /// if they don't exist in the primary store. The upstream store is only used
  /// for reading nodes, new nodes are only written to the primary store.
  /// \param Policy If \p UpstreamDB is provided, controls how nodes are copied
  /// to primary store. This is recorded at creation time and subsequent opens
  /// need to pass the same policy otherwise the \p open will fail.
  static Expected<std::unique_ptr<OnDiskGraphDB>>
  open(StringRef Path, StringRef HashName, unsigned HashByteSize,
       std::unique_ptr<OnDiskGraphDB> UpstreamDB = nullptr,
       FaultInPolicy Policy = FaultInPolicy::FullTree);

  ~OnDiskGraphDB();

private:
  struct IndexProxy;
  class TempFile;
  class MappedTempFile;

  enum class ObjectPresence {
    Missing,
    InPrimaryDB,
    OnlyInUpstreamDB,
  };

  ObjectPresence getObjectPresence(ObjectID Ref, bool CheckUpstream) const;

  bool containsObject(ObjectID Ref, bool CheckUpstream) const {
    switch (getObjectPresence(Ref, CheckUpstream)) {
    case ObjectPresence::Missing:
      return false;
    case ObjectPresence::InPrimaryDB:
      return true;
    case ObjectPresence::OnlyInUpstreamDB:
      return true;
    }
  }

  /// When \p load is called for a node that doesn't exist, this function tries
  /// to load it from the upstream store and copy it to the primary one.
  Expected<std::optional<ObjectHandle>> faultInFromUpstream(ObjectID PrimaryID);
  Error importFullTree(ObjectID PrimaryID, ObjectHandle UpstreamNode);
  Error importSingleNode(ObjectID PrimaryID, ObjectHandle UpstreamNode);

  Expected<IndexProxy> indexHash(ArrayRef<uint8_t> Hash);

  Error createStandaloneLeaf(IndexProxy &I, ArrayRef<char> Data);

  Expected<MappedTempFile> createTempFile(StringRef FinalPath, uint64_t Size);

  OnDiskContent getContentFromHandle(ObjectHandle H) const;

  static InternalRef getInternalRef(ObjectID Ref) {
    return InternalRef::getFromRawData(Ref.getOpaqueData());
  }
  static ObjectID getExternalReference(InternalRef Ref) {
    return ObjectID::fromOpaqueData(Ref.getRawData());
  }

  static ObjectID getExternalReference(const IndexProxy &I);

  void getStandalonePath(StringRef FileSuffix, const IndexProxy &I,
                         SmallVectorImpl<char> &Path) const;

  ArrayRef<uint8_t> getDigest(InternalRef Ref) const;
  ArrayRef<uint8_t> getDigest(const IndexProxy &I) const;

  IndexProxy getIndexProxyFromRef(InternalRef Ref) const;

  static InternalRef makeInternalRef(FileOffset IndexOffset);

  IndexProxy
  getIndexProxyFromPointer(OnDiskHashMappedTrie::const_pointer P) const;

  InternalRefArrayRef getInternalRefs(ObjectHandle Node) const;

  void recordStandaloneSizeIncrease(size_t SizeIncrease);

  std::atomic<uint64_t> &getStandaloneStorageSize();
  uint64_t getStandaloneStorageSize() const;

  OnDiskGraphDB(StringRef RootPath, OnDiskHashMappedTrie Index,
                OnDiskDataAllocator DataPool,
                std::unique_ptr<OnDiskGraphDB> UpstreamDB,
                FaultInPolicy Policy);

  /// Mapping from hash to object reference.
  ///
  /// Data type is TrieRecord.
  OnDiskHashMappedTrie Index;

  /// Storage for most objects.
  ///
  /// Data type is DataRecordHandle.
  OnDiskDataAllocator DataPool;

  void *StandaloneData; // a StandaloneDataMap.

  std::string RootPath;

  /// Optional on-disk store to be used for faulting-in nodes.
  std::unique_ptr<OnDiskGraphDB> UpstreamDB;
  FaultInPolicy FIPolicy;
};

} // namespace llvm::cas::ondisk

#endif // LLVM_CAS_ONDISKGRAPHDB_H
