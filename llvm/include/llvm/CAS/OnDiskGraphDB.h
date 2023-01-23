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

#if LLVM_ENABLE_ONDISK_CAS

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
      return InternalRef(*I.get<const InternalRef4B *>());
    }
    bool operator<(const iterator &RHS) const {
      assert(I.is<const InternalRef *>() == RHS.I.is<const InternalRef *>());
      if (auto *Ref = dyn_cast<const InternalRef *>(I))
        return Ref < RHS.I.get<const InternalRef *>();
      return I.get<const InternalRef4B *>() -
             RHS.I.get<const InternalRef4B *>();
    }
    ptrdiff_t operator-(const iterator &RHS) const {
      assert(I.is<const InternalRef *>() == RHS.I.is<const InternalRef *>());
      if (auto *Ref = dyn_cast<const InternalRef *>(I))
        return Ref - RHS.I.get<const InternalRef *>();
      return I.get<const InternalRef4B *>() -
             RHS.I.get<const InternalRef4B *>();
    }
    iterator &operator+=(ptrdiff_t N) {
      if (auto *Ref = dyn_cast<const InternalRef *>(I))
        I = Ref + N;
      else
        I = I.get<const InternalRef4B *>() + N;
      return *this;
    }
    iterator &operator-=(ptrdiff_t N) {
      if (auto *Ref = dyn_cast<const InternalRef *>(I))
        I = Ref - N;
      else
        I = I.get<const InternalRef4B *>() - N;
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

  bool is4B() const { return Begin.is<const InternalRef4B *>(); }
  bool is8B() const { return Begin.is<const InternalRef *>(); }

  ArrayRef<uint8_t> getBuffer() const {
    if (is4B()) {
      auto *B = Begin.get<const InternalRef4B *>();
      return ArrayRef((const uint8_t *)B, sizeof(InternalRef4B) * Size);
    } else {
      auto *B = Begin.get<const InternalRef *>();
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
  ObjectID getReference(ArrayRef<uint8_t> Hash);

  /// Get an existing reference to the object \p Digest.
  ///
  /// Returns \p nullopt if the object is not stored in this CAS.
  std::optional<ObjectID> getExistingReference(ArrayRef<uint8_t> Digest) const;

  /// \returns true if the object associated with \p Ref is stored in the CAS.
  bool containsObject(ObjectID Ref) const;

  /// \returns the data part of the provided object handle.
  ArrayRef<char> getObjectData(ObjectHandle Node) const;

  object_refs_range getObjectRefs(ObjectHandle Node) const {
    InternalRefArrayRef Refs = getInternalRefs(Node);
    return make_range(Refs.begin(), Refs.end());
  }

  void print(raw_ostream &OS) const;

  static Expected<std::unique_ptr<OnDiskGraphDB>>
  open(StringRef Path, StringRef HashName, unsigned HashByteSize);

  ~OnDiskGraphDB();

private:
  struct IndexProxy;
  class TempFile;
  class MappedTempFile;

  IndexProxy indexHash(ArrayRef<uint8_t> Hash);

  Error createStandaloneLeaf(IndexProxy &I, ArrayRef<char> Data);

  Expected<MappedTempFile> createTempFile(StringRef FinalPath, uint64_t Size);

  OnDiskContent getContentFromHandle(ObjectHandle H) const;

  InternalRef getInternalRef(ObjectID Ref) const {
    return InternalRef::getFromRawData(Ref.getOpaqueData());
  }
  ObjectID getExternalReference(InternalRef Ref) const {
    return ObjectID::fromOpaqueData(Ref.getRawData());
  }

  void getStandalonePath(StringRef FileSuffix, const IndexProxy &I,
                         SmallVectorImpl<char> &Path) const;

  ArrayRef<uint8_t> getDigest(InternalRef Ref) const;
  ArrayRef<uint8_t> getDigest(const IndexProxy &I) const;

  IndexProxy getIndexProxyFromRef(InternalRef Ref) const;

  InternalRef makeInternalRef(FileOffset IndexOffset) const;

  IndexProxy
  getIndexProxyFromPointer(OnDiskHashMappedTrie::const_pointer P) const;

  InternalRefArrayRef getInternalRefs(ObjectHandle Node) const;

  Expected<std::unique_ptr<MemoryBuffer>> openFile(StringRef Path);
  Expected<std::unique_ptr<MemoryBuffer>> openFileWithID(StringRef BaseDir,
                                                         ArrayRef<uint8_t> ID);

  OnDiskGraphDB(StringRef RootPath, OnDiskHashMappedTrie Index,
                OnDiskDataAllocator DataPool);

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
  std::string TempPrefix;
};

} // namespace llvm::cas::ondisk

#endif // LLVM_ENABLE_ONDISK_CAS
#endif // LLVM_CAS_ONDISKGRAPHDB_H
