//===- OnDiskCAS.cpp --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BuiltinCAS.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/CAS/BuiltinObjectHasher.h"
#include "llvm/CAS/OnDiskHashMappedTrie.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"

#define DEBUG_TYPE "on-disk-cas"

using namespace llvm;
using namespace llvm::cas;
using namespace llvm::cas::builtin;

#if LLVM_ENABLE_ONDISK_CAS
namespace {

/// Trie record data: 8B, atomic<uint64_t>
/// - 1-byte: StorageKind
/// - 1-byte: ObjectKind
/// - 6-bytes: DataStoreOffset (offset into referenced file)
class TrieRecord {
public:
  enum class StorageKind : uint8_t {
    /// Unknown object.
    Unknown = 0,

    /// vX.data: main pool, full DataStore record.
    DataPool = 1,

    /// vX.data: main pool, string with 2B size field.
    DataPoolString2B = 2,

    /// vX.<TrieRecordOffset>.data: standalone, with a full DataStore record.
    Standalone = 10,

    /// vX.<TrieRecordOffset>.leaf: standalone, just the data. File contents
    /// exactly the data content and file size matches the data size. No refs.
    StandaloneLeaf = 11,

    /// vX.<TrieRecordOffset>.leaf+0: standalone, just the data plus an
    /// extra null character ('\0'). File size is 1 bigger than the data size.
    /// No refs.
    StandaloneLeaf0 = 12,
  };

  enum class ObjectKind : uint8_t {
    /// Object: refs and data.
    Invalid = 0,

    /// Object: refs and data.
    Object = 1,

    /// String: data, no alignment guarantee, null-terminated.
    String = 4,
  };

  enum Limits : int64_t {
    // Saves files bigger than 64KB standalone instead of embedding them.
    MaxEmbeddedSize = 64LL * 1024LL - 1,
  };

  struct Data {
    StorageKind SK = StorageKind::Unknown;
    ObjectKind OK = ObjectKind::Invalid;
    FileOffset Offset;
  };

  static uint64_t pack(Data D) {
    assert(D.Offset.get() < (int64_t)(1ULL << 48));
    uint64_t Packed =
        uint64_t(D.SK) << 56 | uint64_t(D.OK) << 48 | D.Offset.get();
    assert(D.SK != StorageKind::Unknown || Packed == 0);
#ifndef NDEBUG
    Data RoundTrip = unpack(Packed);
    assert(D.SK == RoundTrip.SK);
    assert(D.OK == RoundTrip.OK);
    assert(D.Offset.get() == RoundTrip.Offset.get());
#endif
    return Packed;
  }

  static Data unpack(uint64_t Packed) {
    Data D;
    if (!Packed)
      return D;
    D.SK = (StorageKind)(Packed >> 56);
    D.OK = (ObjectKind)((Packed >> 48) & 0xFF);
    D.Offset = FileOffset(Packed & (UINT64_MAX >> 16));
    return D;
  }

  TrieRecord() : Storage(0) {}

  Data load() const { return unpack(Storage); }
  bool compare_exchange_strong(Data &Existing, Data New);

private:
  std::atomic<uint64_t> Storage;
};

class InternalRef4B;

/// 8B reference:
/// - bits  0-47: Offset
/// - bits 48-63: Reserved for other metadata.
class InternalRef {
  enum Counts : size_t {
    NumMetadataBits = 16,
    NumOffsetBits = 64 - NumMetadataBits,
  };

public:
  enum class OffsetKind {
    IndexRecord = 0,
    DataRecord = 1,
    String2B = 2,
  };

  OffsetKind getOffsetKind() const {
    return (OffsetKind)((Data >> NumOffsetBits) & UINT8_MAX);
  }

  FileOffset getFileOffset() const { return FileOffset(getRawOffset()); }

  uint64_t getRawData() const { return Data; }
  uint64_t getRawOffset() const { return Data & (UINT64_MAX >> 16); }

  static InternalRef getFromRawData(uint64_t Data) { return InternalRef(Data); }

  static InternalRef getFromOffset(OffsetKind Kind, FileOffset Offset) {
    assert((uint64_t)Offset.get() <= (UINT64_MAX >> NumMetadataBits) &&
           "Offset must fit in 6B");
    return InternalRef((uint64_t)Kind << NumOffsetBits | Offset.get());
  }

  friend bool operator==(InternalRef LHS, InternalRef RHS) {
    return LHS.Data == RHS.Data;
  }

private:
  InternalRef(OffsetKind Kind, FileOffset Offset)
      : Data((uint64_t)Kind << NumOffsetBits | Offset.get()) {
    assert(Offset.get() == getFileOffset().get());
    assert(Kind == getOffsetKind());
  }
  InternalRef(uint64_t Data) : Data(Data) {}
  uint64_t Data;
};

/// 4B reference:
/// - bits  0-29: Offset
/// - bits 30-31: Reserved for other metadata.
class InternalRef4B {
  enum Counts : size_t {
    NumMetadataBits = 4,
    NumOffsetBits = 32 - NumMetadataBits,
  };

public:
  using OffsetKind = InternalRef::OffsetKind;

  OffsetKind getOffsetKind() const {
    return (OffsetKind)(Data >> NumOffsetBits);
  }

  FileOffset getFileOffset() const {
    uint64_t RawOffset = Data & (UINT32_MAX >> NumMetadataBits);
    return FileOffset(RawOffset << 3);
  }

  /// Shrink to 4B reference.
  static Optional<InternalRef4B> tryToShrink(InternalRef Ref) {
    OffsetKind Kind = Ref.getOffsetKind();
    uint64_t ShiftedKind = (uint64_t)Kind << NumOffsetBits;
    if (ShiftedKind > UINT32_MAX)
      return None;

    uint64_t ShiftedOffset = Ref.getRawOffset();
    assert(isAligned(Align(8), ShiftedOffset));
    ShiftedOffset >>= 3;
    if (ShiftedOffset > (UINT32_MAX >> NumMetadataBits))
      return None;

    return InternalRef4B(ShiftedKind | ShiftedOffset);
  }

  operator InternalRef() const {
    return InternalRef::getFromOffset(getOffsetKind(), getFileOffset());
  }

private:
  friend class InternalRef;
  InternalRef4B(uint32_t Data) : Data(Data) {}
  uint32_t Data;
};

class InternalRefArrayRef {
public:
  size_t size() const { return Size; }
  bool empty() const { return !Size; }

  class iterator
      : public iterator_facade_base<iterator, std::random_access_iterator_tag,
                                    const InternalRef> {
  public:
    bool operator==(const iterator &RHS) const { return I == RHS.I; }
    const InternalRef &operator*() const {
      if (auto *Ref = I.dyn_cast<const InternalRef *>())
        return *Ref;
      LocalProxyFor4B = InternalRef(*I.get<const InternalRef4B *>());
      return *LocalProxyFor4B;
    }
    bool operator<(const iterator &RHS) const {
      if (I == RHS.I)
        return false;
      assert(I.is<const InternalRef *>() == RHS.I.is<const InternalRef *>());
      if (auto *Ref = I.dyn_cast<const InternalRef *>())
        return Ref < RHS.I.get<const InternalRef *>();
      return I.get<const InternalRef4B *>() -
             RHS.I.get<const InternalRef4B *>();
    }
    ptrdiff_t operator-(const iterator &RHS) const {
      if (I == RHS.I)
        return 0;
      assert(I.is<const InternalRef *>() == RHS.I.is<const InternalRef *>());
      if (auto *Ref = I.dyn_cast<const InternalRef *>())
        return Ref - RHS.I.get<const InternalRef *>();
      return I.get<const InternalRef4B *>() -
             RHS.I.get<const InternalRef4B *>();
    }
    iterator &operator+=(ptrdiff_t N) {
      if (!N)
        return *this;
      if (auto *Ref = I.dyn_cast<const InternalRef *>())
        I = Ref + N;
      else
        I = I.get<const InternalRef4B *>() + N;
      return *this;
    }
    iterator &operator-=(ptrdiff_t N) {
      if (!N)
        return *this;
      if (auto *Ref = I.dyn_cast<const InternalRef *>())
        I = Ref - N;
      else
        I = I.get<const InternalRef4B *>() - N;
      return *this;
    }

    iterator() = default;

  private:
    friend class InternalRefArrayRef;
    explicit iterator(
        PointerUnion<const InternalRef *, const InternalRef4B *> I)
        : I(I) {}
    PointerUnion<const InternalRef *, const InternalRef4B *> I;
    mutable Optional<InternalRef> LocalProxyFor4B;
  };

  bool operator==(const InternalRefArrayRef &RHS) const {
    return size() == RHS.size() && std::equal(begin(), end(), RHS.begin());
  }

  iterator begin() const { return iterator(Begin); }
  iterator end() const { return begin() + Size; }

  /// Array accessor.
  ///
  /// Returns a reference proxy to avoid lifetime issues, since a reference
  /// derived from a InternalRef4B lives inside the iterator.
  iterator::ReferenceProxy operator[](ptrdiff_t N) const { return begin()[N]; }

  bool is4B() const { return Begin.is<const InternalRef4B *>(); }
  bool is8B() const { return Begin.is<const InternalRef *>(); }

  ArrayRef<InternalRef> as8B() const {
    assert(is8B());
    auto *B = Begin.get<const InternalRef *>();
    return makeArrayRef(B, Size);
  }

  ArrayRef<InternalRef4B> as4B() const {
    auto *B = Begin.get<const InternalRef4B *>();
    return makeArrayRef(B, Size);
  }

  InternalRefArrayRef(NoneType = None) {}

  InternalRefArrayRef(ArrayRef<InternalRef> Refs)
      : Begin(Refs.begin()), Size(Refs.size()) {}

  InternalRefArrayRef(ArrayRef<InternalRef4B> Refs)
      : Begin(Refs.begin()), Size(Refs.size()) {}

private:
  PointerUnion<const InternalRef *, const InternalRef4B *> Begin;
  size_t Size = 0;
};

class InternalRefVector {
public:
  void push_back(InternalRef Ref) {
    if (NeedsFull)
      return FullRefs.push_back(Ref);
    if (Optional<InternalRef4B> Small = InternalRef4B::tryToShrink(Ref))
      return SmallRefs.push_back(*Small);
    NeedsFull = true;
    assert(FullRefs.empty());
    FullRefs.reserve(SmallRefs.size() + 1);
    for (InternalRef4B Small : SmallRefs)
      FullRefs.push_back(Small);
    FullRefs.push_back(Ref);
    SmallRefs.clear();
  }

  operator InternalRefArrayRef() const {
    assert(SmallRefs.empty() || FullRefs.empty());
    return NeedsFull ? InternalRefArrayRef(FullRefs)
                     : InternalRefArrayRef(SmallRefs);
  }

private:
  bool NeedsFull = false;
  SmallVector<InternalRef4B> SmallRefs;
  SmallVector<InternalRef> FullRefs;
};

/// DataStore record data: 8B + size? + refs? + data + 0
/// - 8-bytes: Header
/// - {0,4,8}-bytes: DataSize     (may be packed in Header)
/// - {0,4,8}-bytes: NumRefs      (may be packed in Header)
/// - NumRefs*{4,8}-bytes: Refs[] (end-ptr is 8-byte aligned)
/// - <data>
/// - 1-byte: 0-term
struct DataRecordHandle {
  /// NumRefs storage: 4B, 2B, 1B, or 0B (no refs). Or, 8B, for alignment
  /// convenience to avoid computing padding later.
  enum class NumRefsFlags : uint8_t {
    Uses0B = 0U,
    Uses1B = 1U,
    Uses2B = 2U,
    Uses4B = 3U,
    Uses8B = 4U,
    Max = Uses8B,
  };

  /// DataSize storage: 8B, 4B, 2B, or 1B.
  enum class DataSizeFlags {
    Uses1B = 0U,
    Uses2B = 1U,
    Uses4B = 2U,
    Uses8B = 3U,
    Max = Uses8B,
  };

  enum class TrieOffsetFlags {
    /// TrieRecord storage: 6B or 4B.
    Uses6B = 0U,
    Uses4B = 1U,
    Max = Uses4B,
  };

  /// Kind of ref stored in Refs[]: InternalRef or InternalRef4B.
  enum class RefKindFlags {
    InternalRef = 0U,
    InternalRef4B = 1U,
    Max = InternalRef4B,
  };

  enum Counts : int {
    NumRefsShift = 0,
    NumRefsBits = 3,
    DataSizeShift = NumRefsShift + NumRefsBits,
    DataSizeBits = 2,
    TrieOffsetShift = DataSizeShift + DataSizeBits,
    TrieOffsetBits = 1,
    RefKindShift = TrieOffsetShift + TrieOffsetBits,
    RefKindBits = 1,
  };
  static_assert(((UINT32_MAX << NumRefsBits) & (uint32_t)NumRefsFlags::Max) ==
                    0,
                "Not enough bits");
  static_assert(((UINT32_MAX << DataSizeBits) & (uint32_t)DataSizeFlags::Max) ==
                    0,
                "Not enough bits");
  static_assert(((UINT32_MAX << TrieOffsetBits) &
                 (uint32_t)TrieOffsetFlags::Max) == 0,
                "Not enough bits");
  static_assert(((UINT32_MAX << RefKindBits) & (uint32_t)RefKindFlags::Max) ==
                    0,
                "Not enough bits");

  struct LayoutFlags {
    NumRefsFlags NumRefs;
    DataSizeFlags DataSize;
    TrieOffsetFlags TrieOffset;
    RefKindFlags RefKind;

    static uint64_t pack(LayoutFlags LF) {
      unsigned Packed = ((unsigned)LF.NumRefs << NumRefsShift) |
                        ((unsigned)LF.DataSize << DataSizeShift) |
                        ((unsigned)LF.TrieOffset << TrieOffsetShift) |
                        ((unsigned)LF.RefKind << RefKindShift);
#ifndef NDEBUG
      LayoutFlags RoundTrip = unpack(Packed);
      assert(LF.NumRefs == RoundTrip.NumRefs);
      assert(LF.DataSize == RoundTrip.DataSize);
      assert(LF.TrieOffset == RoundTrip.TrieOffset);
      assert(LF.RefKind == RoundTrip.RefKind);
#endif
      return Packed;
    }
    static LayoutFlags unpack(uint64_t Storage) {
      assert(Storage <= UINT8_MAX && "Expect storage to fit in a byte");
      LayoutFlags LF;
      LF.NumRefs =
          (NumRefsFlags)((Storage >> NumRefsShift) & ((1U << NumRefsBits) - 1));
      LF.DataSize = (DataSizeFlags)((Storage >> DataSizeShift) &
                                    ((1U << DataSizeBits) - 1));
      LF.TrieOffset = (TrieOffsetFlags)((Storage >> TrieOffsetShift) &
                                        ((1U << TrieOffsetBits) - 1));
      LF.RefKind =
          (RefKindFlags)((Storage >> RefKindShift) & ((1U << RefKindBits) - 1));
      return LF;
    }
  };

  /// Header layout:
  /// - 1-byte:      LayoutFlags
  /// - 1-byte:      1B size field
  /// - {0,2}-bytes: 2B size field
  /// - {4,6}-bytes: TrieRecordOffset
  struct Header {
    uint64_t Packed;
  };

  struct Input {
    FileOffset TrieRecordOffset;
    InternalRefArrayRef Refs;
    ArrayRef<char> Data;
  };

  LayoutFlags getLayoutFlags() const {
    return LayoutFlags::unpack(H->Packed >> 56);
  }
  FileOffset getTrieRecordOffset() const {
    if (getLayoutFlags().TrieOffset == TrieOffsetFlags::Uses4B)
      return FileOffset(H->Packed & UINT32_MAX);
    return FileOffset(H->Packed & (UINT64_MAX >> 16));
  }

  uint64_t getDataSize() const;
  void skipDataSize(LayoutFlags LF, int64_t &RelOffset) const;
  uint32_t getNumRefs() const;
  void skipNumRefs(LayoutFlags LF, int64_t &RelOffset) const;
  int64_t getRefsRelOffset() const;
  int64_t getDataRelOffset() const;

  static uint64_t getTotalSize(uint64_t DataRelOffset, uint64_t DataSize) {
    return DataRelOffset + DataSize + 1;
  }
  uint64_t getTotalSize() const {
    return getDataRelOffset() + getDataSize() + 1;
  }

  struct Layout {
    explicit Layout(const Input &I);

    LayoutFlags Flags{};
    uint64_t TrieRecordOffset = 0;
    uint64_t DataSize = 0;
    uint32_t NumRefs = 0;
    int64_t RefsRelOffset = 0;
    int64_t DataRelOffset = 0;
    uint64_t getTotalSize() const {
      return DataRecordHandle::getTotalSize(DataRelOffset, DataSize);
    }
  };

  InternalRefArrayRef getRefs() const {
    assert(H && "Expected valid handle");
    auto *BeginByte = reinterpret_cast<const char *>(H) + getRefsRelOffset();
    size_t Size = getNumRefs();
    if (!Size)
      return InternalRefArrayRef();
    if (getLayoutFlags().RefKind == RefKindFlags::InternalRef4B)
      return makeArrayRef(reinterpret_cast<const InternalRef4B *>(BeginByte),
                          Size);
    return makeArrayRef(reinterpret_cast<const InternalRef *>(BeginByte), Size);
  }

  ArrayRef<char> getData() const {
    assert(H && "Expected valid handle");
    return makeArrayRef(reinterpret_cast<const char *>(H) + getDataRelOffset(),
                        getDataSize());
  }

  static DataRecordHandle create(function_ref<char *(size_t Size)> Alloc,
                                 const Input &I);
  static Expected<DataRecordHandle>
  createWithError(function_ref<Expected<char *>(size_t Size)> Alloc,
                  const Input &I);
  static DataRecordHandle construct(char *Mem, const Input &I);

  static DataRecordHandle get(const char *Mem) {
    return DataRecordHandle(
        *reinterpret_cast<const DataRecordHandle::Header *>(Mem));
  }

  explicit operator bool() const { return H; }
  const Header &getHeader() const { return *H; }

  DataRecordHandle() = default;
  explicit DataRecordHandle(const Header &H) : H(&H) {}

private:
  static DataRecordHandle constructImpl(char *Mem, const Input &I,
                                        const Layout &L);
  const Header *H = nullptr;
};

Expected<DataRecordHandle> DataRecordHandle::createWithError(
    function_ref<Expected<char *>(size_t Size)> Alloc, const Input &I) {
  Layout L(I);
  if (Expected<char *> Mem = Alloc(L.getTotalSize()))
    return constructImpl(*Mem, I, L);
  else
    return Mem.takeError();
}

DataRecordHandle
DataRecordHandle::create(function_ref<char *(size_t Size)> Alloc,
                         const Input &I) {
  Layout L(I);
  return constructImpl(Alloc(L.getTotalSize()), I, L);
}

struct String2BHandle {
  /// Header layout:
  /// - 2-bytes: Length
  struct Header {
    uint16_t Length;
  };

  uint64_t getLength() const { return H->Length; }

  StringRef getString() const { return toStringRef(getArray()); }
  ArrayRef<char> getArray() const {
    assert(H && "Expected valid handle");
    return makeArrayRef(reinterpret_cast<const char *>(H + 1), getLength());
  }

  static String2BHandle create(function_ref<char *(size_t Size)> Alloc,
                               StringRef String) {
    assert(String.size() <= UINT16_MAX);
    char *Mem = Alloc(sizeof(Header) + String.size() + 1);
    Header *H = new (Mem) Header{(uint16_t)String.size()};
    llvm::copy(String, Mem + sizeof(Header));
    Mem[sizeof(Header) + String.size()] = 0;
    return String2BHandle(*H);
  }

  static String2BHandle get(const char *Mem) {
    return String2BHandle(
        *reinterpret_cast<const String2BHandle::Header *>(Mem));
  }

  explicit operator bool() const { return H; }
  const Header &getHeader() const { return *H; }

  String2BHandle() = default;
  explicit String2BHandle(const Header &H) : H(&H) {}

private:
  const Header *H = nullptr;
};

struct OnDiskContent;
class StandaloneDataInMemory {
public:
  OnDiskContent getContent() const;

  /// FIXME: Should be mapped_file_region instead of MemoryBuffer to drop a
  /// layer of indirection.
  std::unique_ptr<MemoryBuffer> Region;
  InternalRef Ref;
  TrieRecord::StorageKind SK;
  StandaloneDataInMemory(std::unique_ptr<MemoryBuffer> Region, InternalRef Ref,
                         TrieRecord::StorageKind SK)
      : Region(std::move(Region)), Ref(Ref), SK(SK) {
    assert(Ref.getOffsetKind() == InternalRef::OffsetKind::IndexRecord);
#ifndef NDEBUG
    bool IsStandalone = false;
    switch (SK) {
    case TrieRecord::StorageKind::Standalone:
    case TrieRecord::StorageKind::StandaloneLeaf:
    case TrieRecord::StorageKind::StandaloneLeaf0:
      IsStandalone = true;
      break;
    default:
      break;
    }
    assert(IsStandalone);
#endif
  }
};

struct InternalHandle {
  InternalRef getRef() const { return DirectRef ? *DirectRef : SDIM->Ref; }

  uint64_t getRawData() const {
    if (DirectRef) {
      uint64_t Raw = DirectRef->getRawData();
      assert(!(Raw & 0x1));
      return Raw;
    }
    uint64_t Raw = reinterpret_cast<uintptr_t>(SDIM);
    assert(!(Raw & 0x1));
    return Raw | 1;
  }

  explicit InternalHandle(InternalRef DirectRef) : DirectRef(DirectRef) {
    assert(DirectRef.getOffsetKind() != InternalRef::OffsetKind::IndexRecord);
  }
  explicit InternalHandle(const StandaloneDataInMemory &SDIM) : SDIM(&SDIM) {
    assert(SDIM.Ref.getOffsetKind() == InternalRef::OffsetKind::IndexRecord);
  }
  Optional<InternalRef> DirectRef;
  const StandaloneDataInMemory *SDIM = nullptr;
};

/// Container for "big" objects mapped in separately.
template <size_t NumShards> class StandaloneDataMap {
  static_assert(isPowerOf2_64(NumShards), "Expected power of 2");

public:
  const StandaloneDataInMemory &insert(ArrayRef<uint8_t> Hash, InternalRef Ref,
                                       TrieRecord::StorageKind SK,
                                       std::unique_ptr<MemoryBuffer> Buffer);

  const StandaloneDataInMemory *lookup(ArrayRef<uint8_t> Hash) const;
  bool count(ArrayRef<uint8_t> Hash) const { return bool(lookup(Hash)); }

private:
  struct Shard {
    /// Needs to store a std::unique_ptr for a stable address identity.
    DenseMap<const uint8_t *, std::unique_ptr<StandaloneDataInMemory>> Map;
    mutable std::mutex Mutex;
  };
  Shard &getShard(ArrayRef<uint8_t> Hash) {
    return const_cast<Shard &>(
        const_cast<const StandaloneDataMap *>(this)->getShard(Hash));
  }
  const Shard &getShard(ArrayRef<uint8_t> Hash) const {
    static_assert(NumShards <= 256, "Expected only 8 bits of shard");
    return Shards[Hash[0] % NumShards];
  }

  Shard Shards[NumShards];
};

/// Proxy for an on-disk index record.
struct IndexProxy {
  FileOffset Offset;
  ArrayRef<uint8_t> Hash;
  TrieRecord &Ref;
};

/// Proxy for any on-disk object or raw data.
struct OnDiskContent {
  Optional<DataRecordHandle> Record;
  Optional<ArrayRef<char>> Bytes;
};

/// On-disk CAS database and action cache (the latter should be separated).
///
/// Here's a top-level description of the current layout (could expose or make
/// this configurable in the future).
///
/// Files, each with a prefix set by \a FilePrefix():
///
/// - db/<prefix>.index: a file for the "index" table, named by \a
///   getIndexTableName() and managed by \a HashMappedTrie. The contents are 8B
///   that are accessed atomically, describing the object kind and where/how
///   it's stored (including an optional file offset). See \a TrieRecord for
///   more details.
/// - db/<prefix>.data: a file for the "data" table, named by \a
///   getDataPoolTableName() and managed by \a DataStore. New objects within
///   TrieRecord::MaxEmbeddedSize are inserted here as either \a
///   TrieRecord::StorageKind::DataPool or
///   TrieRecord::StorageKind::DataPoolString2B.
///     - db/<prefix>.<offset>.data: a file storing an object outside the main
///       "data" table, named by its offset into the "index" table, with the
///       format of \a TrieRecord::StorageKind::Standalone.
///     - db/<prefix>.<offset>.leaf: a file storing a leaf node outside the
///       main "data" table, named by its offset into the "index" table, with
///       the format of \a TrieRecord::StorageKind::StandaloneLeaf.
///     - db/<prefix>.<offset>.leaf+0: a file storing a leaf object outside the
///       main "data" table, named by its offset into the "index" table, with
///       the format of \a TrieRecord::StorageKind::StandaloneLeaf0.
/// - db/<prefix>.actions: a file for the "actions" table, named by \a
///   getActionCacheTableName() and managed by \a HashMappedTrie. The contents
///   are \a CASID::getInternalID(), stored as \a ActionCacheResultT;
///   effectively, a pointer into the "index" table.
///
/// The "index", "data", and "actions" tables could be stored in a single file,
/// (using a root record that points at the two types of stores), but splitting
/// the files seems more convenient for now.
///
/// Eventually: update UniqueID/CASID to store:
/// - uint64_t: for BuiltinCAS, this is a pointer to Trie record
/// - CASDB*: for knowing how to compare, and for getHash()
///
/// Eventually: add ObjectHandle (update ObjectRef):
/// - uint64_t: for BuiltinCAS, this is a pointer to Data record
/// - CASDB*: for implementing APIs
///
/// Eventually: consider creating a StringPool for strings instead of using
/// RecordDataStore table.
/// - Lookup by prefix tree
/// - Store by suffix tree
class OnDiskCAS : public BuiltinCAS {
public:
  static StringRef getIndexTableName() {
    static const std::string Name =
        ("llvm.cas.index[" + getHashName() + "]").str();
    return Name;
  }
  static StringRef getDataPoolTableName() {
    static const std::string Name =
        ("llvm.cas.data[" + getHashName() + "]").str();
    return Name;
  }
  static StringRef getActionCacheTableName() {
    static const std::string Name =
        ("llvm.cas.actions[" + getHashName() + "->" + getHashName() + "]")
            .str();
    return Name;
  }

  static constexpr StringLiteral IndexFile = "index";
  static constexpr StringLiteral DataPoolFile = "data";
  static constexpr StringLiteral ActionCacheFile = "actions";

  static constexpr StringLiteral FilePrefix = "v5.";
  static constexpr StringLiteral FileSuffixData = ".data";
  static constexpr StringLiteral FileSuffixLeaf = ".leaf";
  static constexpr StringLiteral FileSuffixLeaf0 = ".leaf+0";

  class TempFile;
  class MappedTempFile;

  IndexProxy indexHash(ArrayRef<uint8_t> Hash);
  Expected<CASID> parseIDImpl(ArrayRef<uint8_t> Hash) final {
    return getID(indexHash(Hash));
  }

  Expected<ObjectHandle> storeImpl(ArrayRef<uint8_t> ComputedHash,
                                   ArrayRef<ObjectRef> Refs,
                                   ArrayRef<char> Data) final;

  Expected<ObjectHandle> createStandaloneLeaf(IndexProxy &I, ArrayRef<char> Data);

  Expected<ObjectHandle>
  loadOrCreateDataRecord(IndexProxy &I, TrieRecord::ObjectKind OK,
                         DataRecordHandle::Input Input);

  Expected<MappedTempFile> createTempFile(StringRef FinalPath, uint64_t Size);

  Optional<DataRecordHandle> getDataRecordForObject(ObjectHandle H) const {
    return getContentFromHandle(H).Record;
  }
  OnDiskContent getContentFromHandle(ObjectHandle H) const {
    return getContentFromHandle(getInternalHandle(H));
  }
  Expected<InternalHandle> loadContentForRef(const IndexProxy &I,
                                             TrieRecord::Data Object,
                                             InternalRef Ref);
  OnDiskContent getContentFromHandle(InternalHandle Ref) const;

  InternalRef getInternalRef(ObjectRef Ref) const {
    uint64_t Data = Ref.getInternalRef(*this);
    assert(!(Data & 1));
    return InternalRef::getFromRawData(Data);
  }
  InternalHandle getInternalHandle(ObjectHandle Handle) const {
    uint64_t Data = Handle.getInternalRef(*this);
    if (Data & 1)
      return InternalHandle(*reinterpret_cast<const StandaloneDataInMemory *>(
          Data & (-1ULL << 1)));
    return InternalHandle(InternalRef::getFromRawData(Data));
  }
  ObjectRef getExternalReference(InternalRef Ref) const {
    return ObjectRef::getFromInternalRef(*this, Ref.getRawData());
  }

  Expected<ObjectHandle> load(ObjectRef Ref) final;
  Expected<ObjectHandle> load(const IndexProxy &I, TrieRecord::Data Object,
                              InternalRef Ref);

  ObjectHandle getLoadedObject(const IndexProxy &I, TrieRecord::Data Object,
                               InternalHandle Handle) const;

  struct PooledDataRecord {
    FileOffset Offset;
    DataRecordHandle Record;
  };
  PooledDataRecord createPooledDataRecord(DataRecordHandle::Input Input);
  void getStandalonePath(TrieRecord::StorageKind SK, const IndexProxy &I,
                         SmallVectorImpl<char> &Path) const;

  CASID getID(InternalRef Ref) const;
  CASID getID(ObjectRef Ref) const final { return getID(getInternalRef(Ref)); }
  CASID getID(ObjectHandle Handle) const final {
    return getID(getInternalHandle(Handle).getRef());
  }
  Optional<FileOffset> getIndexOffset(InternalRef Ref) const;

  CASID getID(const IndexProxy &I) const {
    return getIDFromIndexOffset(I.Offset);
  }

  Optional<ObjectRef> getReference(const CASID &ID) const final;
  ObjectRef getReference(ObjectHandle Handle) const final {
    return getExternalReference(getInternalHandle(Handle).getRef());
  }

  ArrayRef<uint8_t> getHashImpl(const CASID &ID) const final {
    assert(&ID.getContext() == this && "Expected ID from this CASIDContext");
    OnDiskHashMappedTrie::const_pointer P = getInternalIndexPointer(ID);
    assert(P && "Expected to recover pointer from CASID");
    return P->Hash;
  }
  CASID getIDFromIndexOffset(FileOffset Offset) const {
    return CASID::getFromInternalID(*this, Offset.get());
  }

  OnDiskHashMappedTrie::const_pointer
  getInternalIndexPointer(InternalRef Ref) const;
  Optional<IndexProxy> getIndexProxyFromRef(InternalRef Ref) const;

  OnDiskHashMappedTrie::const_pointer
  getInternalIndexPointer(const CASID &ID) const;
  Optional<InternalRef> makeInternalRef(FileOffset IndexOffset,
                                        TrieRecord::Data Object) const;

  IndexProxy
  getIndexProxyFromPointer(OnDiskHashMappedTrie::const_pointer P) const;

  Expected<InternalRef> storeTreeEntryName(StringRef String);

  ArrayRef<char> getDataConst(ObjectHandle Node) const final;

  void print(raw_ostream &OS) const final;

  Expected<CASID> getCachedResult(CASID InputID) final;
  Error putCachedResult(CASID InputID, CASID OutputID) final;

  static Expected<std::unique_ptr<OnDiskCAS>> open(StringRef Path);

private:
  InternalRefArrayRef getRefs(ObjectHandle Node) const {
    if (Optional<DataRecordHandle> Record = getDataRecordForObject(Node))
      return Record->getRefs();
    return None;
  }
  size_t getNumRefs(ObjectHandle Node) const final {
    return getRefs(Node).size();
  }
  ObjectRef readRef(ObjectHandle Node, size_t I) const final {
    return getExternalReference(getRefs(Node)[I]);
  }
  Error forEachRef(ObjectHandle Node,
                   function_ref<Error(ObjectRef)> Callback) const final;

  StringRef getPathForID(StringRef BaseDir, CASID ID,
                         SmallVectorImpl<char> &Storage);

  Expected<std::unique_ptr<MemoryBuffer>> openFile(StringRef Path);
  Expected<std::unique_ptr<MemoryBuffer>> openFileWithID(StringRef BaseDir,
                                                         CASID ID);

  OnDiskCAS(StringRef RootPath, OnDiskHashMappedTrie Index,
            OnDiskDataAllocator DataPool, OnDiskHashMappedTrie ActionCache);

  /// Mapping from hash to object reference.
  ///
  /// Data type is TrieRecord.
  OnDiskHashMappedTrie Index;

  /// Storage for most objects.
  ///
  /// Data type is DataRecordHandle.
  OnDiskDataAllocator DataPool;

  /// Lifetime for "big" objects not in DataPool.
  ///
  /// NOTE: Could use ThreadSafeHashMappedTrie here. For now, doing something
  /// simpler on the assumption there won't be much contention since most data
  /// is not big. If there is contention, and we've already fixed ObjectProxy
  /// object handles to be cheap enough to use consistently, the fix might be
  /// to use better use of them rather than optimizing this map.
  ///
  /// FIXME: Figure out the right number of shards, if any.
  mutable StandaloneDataMap<16> StandaloneData;

  /// Action cache.
  ///
  /// FIXME: Separate out. Likely change key to be independent from CASID and
  /// stored separately.
  OnDiskHashMappedTrie ActionCache;
  using ActionCacheResultT = std::atomic<uint64_t>;

  std::string RootPath;
  std::string TempPrefix;
};

} // end anonymous namespace

constexpr StringLiteral OnDiskCAS::IndexFile;
constexpr StringLiteral OnDiskCAS::DataPoolFile;
constexpr StringLiteral OnDiskCAS::ActionCacheFile;
constexpr StringLiteral OnDiskCAS::FilePrefix;
constexpr StringLiteral OnDiskCAS::FileSuffixData;
constexpr StringLiteral OnDiskCAS::FileSuffixLeaf;
constexpr StringLiteral OnDiskCAS::FileSuffixLeaf0;

template <size_t N>
const StandaloneDataInMemory &
StandaloneDataMap<N>::insert(ArrayRef<uint8_t> Hash, InternalRef Ref,
                             TrieRecord::StorageKind SK,
                             std::unique_ptr<MemoryBuffer> Buffer) {
  auto &S = getShard(Hash);
  std::lock_guard<std::mutex> Lock(S.Mutex);
  auto &V = S.Map[Hash.data()];
  if (!V)
    V = std::make_unique<StandaloneDataInMemory>(std::move(Buffer), Ref, SK);
  return *V;
}

template <size_t N>
const StandaloneDataInMemory *
StandaloneDataMap<N>::lookup(ArrayRef<uint8_t> Hash) const {
  auto &S = getShard(Hash);
  std::lock_guard<std::mutex> Lock(S.Mutex);
  auto I = S.Map.find(Hash.data());
  if (I == S.Map.end())
    return nullptr;
  return &*I->second;
}

/// Copy of \a sys::fs::TempFile that skips RemoveOnSignal, which is too
/// expensive to register/unregister at this rate.
///
/// FIXME: Add a TempFileManager that maintains a thread-safe list of open temp
/// files and has a signal handler registerd that removes them all.
class OnDiskCAS::TempFile {
  bool Done = false;
  TempFile(StringRef Name, int FD) : TmpName(std::string(Name)), FD(FD) {}

public:
  /// This creates a temporary file with createUniqueFile.
  static Expected<TempFile> create(const Twine &Model);
  TempFile(TempFile &&Other) { *this = std::move(Other); }
  TempFile &operator=(TempFile &&Other) {
    TmpName = std::move(Other.TmpName);
    FD = Other.FD;
    Other.Done = true;
    Other.FD = -1;
    return *this;
  }

  // Name of the temporary file.
  std::string TmpName;

  // The open file descriptor.
  int FD = -1;

  // Keep this with the given name.
  Error keep(const Twine &Name);
  Error discard();

  // This checks that keep or delete was called.
  ~TempFile() { consumeError(discard()); }
};

class OnDiskCAS::MappedTempFile {
public:
  char *data() const { return Map.data(); }
  size_t size() const { return Map.size(); }

  Error discard() {
    assert(Map && "Map already destroyed");
    Map.unmap();
    return Temp.discard();
  }

  Error keep(const Twine &Name) {
    assert(Map && "Map already destroyed");
    Map.unmap();
    return Temp.keep(Name);
  }

  MappedTempFile(TempFile Temp, sys::fs::mapped_file_region Map)
      : Temp(std::move(Temp)), Map(std::move(Map)) {}

private:
  TempFile Temp;
  sys::fs::mapped_file_region Map;
};

Error OnDiskCAS::TempFile::discard() {
  Done = true;
  if (FD != -1)
    if (std::error_code EC = sys::fs::closeFile(FD))
      return errorCodeToError(EC);
  FD = -1;

  // Always try to close and remove.
  std::error_code RemoveEC;
  if (!TmpName.empty())
    if (std::error_code EC = sys::fs::remove(TmpName))
      return errorCodeToError(EC);
  TmpName = "";

  return Error::success();
}

Error OnDiskCAS::TempFile::keep(const Twine &Name) {
  assert(!Done);
  Done = true;
  // Always try to close and rename.
  std::error_code RenameEC = sys::fs::rename(TmpName, Name);

  if (!RenameEC)
    TmpName = "";

  if (std::error_code EC = sys::fs::closeFile(FD))
    return errorCodeToError(EC);
  FD = -1;

  return errorCodeToError(RenameEC);
}

Expected<OnDiskCAS::TempFile> OnDiskCAS::TempFile::create(const Twine &Model) {
  int FD;
  SmallString<128> ResultPath;
  if (std::error_code EC = sys::fs::createUniqueFile(Model, FD, ResultPath))
    return errorCodeToError(EC);

  TempFile Ret(ResultPath, FD);
  return std::move(Ret);
}

bool TrieRecord::compare_exchange_strong(Data &Existing, Data New) {
  uint64_t ExistingPacked = pack(Existing);
  uint64_t NewPacked = pack(New);
  if (Storage.compare_exchange_strong(ExistingPacked, NewPacked))
    return true;
  Existing = unpack(ExistingPacked);
  return false;
}

DataRecordHandle DataRecordHandle::construct(char *Mem, const Input &I) {
  return constructImpl(Mem, I, Layout(I));
}

DataRecordHandle DataRecordHandle::constructImpl(char *Mem, const Input &I,
                                                 const Layout &L) {
  assert(I.TrieRecordOffset && "Expected an offset into index");
  assert(L.TrieRecordOffset == (uint64_t)I.TrieRecordOffset.get() &&
         "Offset has drifted?");
  char *Next = Mem + sizeof(Header);

  // Fill in Packed and set other data, then come back to construct the header.
  uint64_t Packed = 0;
  Packed |= LayoutFlags::pack(L.Flags) << 56 | L.TrieRecordOffset;

  // Construct DataSize.
  switch (L.Flags.DataSize) {
  case DataSizeFlags::Uses1B:
    assert(I.Data.size() <= UINT8_MAX);
    Packed |= (uint64_t)I.Data.size() << 48;
    break;
  case DataSizeFlags::Uses2B:
    assert(I.Data.size() <= UINT16_MAX);
    Packed |= (uint64_t)I.Data.size() << 32;
    break;
  case DataSizeFlags::Uses4B:
    assert(isAddrAligned(Align(4), Next));
    new (Next) uint32_t(I.Data.size());
    Next += 4;
    break;
  case DataSizeFlags::Uses8B:
    assert(isAddrAligned(Align(8), Next));
    new (Next) uint64_t(I.Data.size());
    Next += 8;
    break;
  }

  // Construct NumRefs.
  //
  // NOTE: May be writing NumRefs even if there are zero refs in order to fix
  // alignment.
  switch (L.Flags.NumRefs) {
  case NumRefsFlags::Uses0B:
    break;
  case NumRefsFlags::Uses1B:
    assert(I.Refs.size() <= UINT8_MAX);
    Packed |= (uint64_t)I.Refs.size() << 48;
    break;
  case NumRefsFlags::Uses2B:
    assert(I.Refs.size() <= UINT16_MAX);
    Packed |= (uint64_t)I.Refs.size() << 32;
    break;
  case NumRefsFlags::Uses4B:
    assert(isAddrAligned(Align(4), Next));
    new (Next) uint32_t(I.Refs.size());
    Next += 4;
    break;
  case NumRefsFlags::Uses8B:
    assert(isAddrAligned(Align(8), Next));
    new (Next) uint64_t(I.Refs.size());
    Next += 8;
    break;
  }

  // Construct Refs[].
  if (!I.Refs.empty()) {
    if (L.Flags.RefKind == RefKindFlags::InternalRef4B) {
      assert(I.Refs.is4B());
      assert(isAddrAligned(Align::Of<InternalRef4B>(), Next));
      for (InternalRef4B Ref : I.Refs.as4B()) {
        new (Next) InternalRef4B(Ref);
        Next += sizeof(InternalRef4B);
      }
    } else {
      assert(I.Refs.is8B());
      assert(isAddrAligned(Align::Of<InternalRef>(), Next));
      for (InternalRef Ref : I.Refs.as8B()) {
        new (Next) InternalRef(Ref);
        Next += sizeof(InternalRef);
      }
    }
  }

  // Construct Data and the trailing null.
  assert(isAddrAligned(Align(8), Next));
  llvm::copy(I.Data, Next);
  Next[I.Data.size()] = 0;

  // Construct the header itself and return.
  Header *H = new (Mem) Header{Packed};
  DataRecordHandle Record(*H);
  assert(Record.getData() == I.Data);
  assert(Record.getNumRefs() == I.Refs.size());
  assert(Record.getRefs() == I.Refs);
  assert(Record.getLayoutFlags().DataSize == L.Flags.DataSize);
  assert(Record.getLayoutFlags().NumRefs == L.Flags.NumRefs);
  assert(Record.getLayoutFlags().RefKind == L.Flags.RefKind);
  assert(Record.getLayoutFlags().TrieOffset == L.Flags.TrieOffset);
  return Record;
}

DataRecordHandle::Layout::Layout(const Input &I) {
  // Start initial relative offsets right after the Header.
  uint64_t RelOffset = sizeof(Header);

  // Initialize the easy stuff.
  DataSize = I.Data.size();
  NumRefs = I.Refs.size();

  // Check refs size.
  Flags.RefKind =
      I.Refs.is4B() ? RefKindFlags::InternalRef4B : RefKindFlags::InternalRef;

  // Set the trie offset.
  TrieRecordOffset = (uint64_t)I.TrieRecordOffset.get();
  assert(TrieRecordOffset <= (UINT64_MAX >> 16));
  Flags.TrieOffset = TrieRecordOffset <= UINT32_MAX ? TrieOffsetFlags::Uses4B
                                                    : TrieOffsetFlags::Uses6B;

  // Find the smallest slot available for DataSize.
  bool Has1B = true;
  bool Has2B = Flags.TrieOffset == TrieOffsetFlags::Uses4B;
  if (DataSize <= UINT8_MAX && Has1B) {
    Flags.DataSize = DataSizeFlags::Uses1B;
    Has1B = false;
  } else if (DataSize <= UINT16_MAX && Has2B) {
    Flags.DataSize = DataSizeFlags::Uses2B;
    Has2B = false;
  } else if (DataSize <= UINT32_MAX) {
    Flags.DataSize = DataSizeFlags::Uses4B;
    RelOffset += 4;
  } else {
    Flags.DataSize = DataSizeFlags::Uses8B;
    RelOffset += 8;
  }

  // Find the smallest slot available for NumRefs. Never sets NumRefs8B here.
  if (!NumRefs) {
    Flags.NumRefs = NumRefsFlags::Uses0B;
  } else if (NumRefs <= UINT8_MAX && Has1B) {
    Flags.NumRefs = NumRefsFlags::Uses1B;
    Has1B = false;
  } else if (NumRefs <= UINT16_MAX && Has2B) {
    Flags.NumRefs = NumRefsFlags::Uses2B;
    Has2B = false;
  } else {
    Flags.NumRefs = NumRefsFlags::Uses4B;
    RelOffset += 4;
  }

  // Helper to "upgrade" either DataSize or NumRefs by 4B to avoid complicated
  // padding rules when reading and writing. This also bumps RelOffset.
  //
  // The value for NumRefs is strictly limited to UINT32_MAX, but it can be
  // stored as 8B. This means we can *always* find a size to grow.
  //
  // NOTE: Only call this once.
  auto GrowSizeFieldsBy4B = [&]() {
    assert(isAligned(Align(4), RelOffset));
    RelOffset += 4;

    assert(Flags.NumRefs != NumRefsFlags::Uses8B &&
           "Expected to be able to grow NumRefs8B");

    // First try to grow DataSize. NumRefs will not (yet) be 8B, and if
    // DataSize is upgraded to 8B it'll already be aligned.
    //
    // Failing that, grow NumRefs.
    if (Flags.DataSize < DataSizeFlags::Uses4B)
      Flags.DataSize = DataSizeFlags::Uses4B; // DataSize: Packed => 4B.
    else if (Flags.DataSize < DataSizeFlags::Uses8B)
      Flags.DataSize = DataSizeFlags::Uses8B; // DataSize: 4B => 8B.
    else if (Flags.NumRefs < NumRefsFlags::Uses4B)
      Flags.NumRefs = NumRefsFlags::Uses4B; // NumRefs: Packed => 4B.
    else
      Flags.NumRefs = NumRefsFlags::Uses8B; // NumRefs: 4B => 8B.
  };

  assert(isAligned(Align(4), RelOffset));
  if (Flags.RefKind == RefKindFlags::InternalRef) {
    // List of 8B refs should be 8B-aligned. Grow one of the sizes to get this
    // without padding.
    if (!isAligned(Align(8), RelOffset))
      GrowSizeFieldsBy4B();

    assert(isAligned(Align(8), RelOffset));
    RefsRelOffset = RelOffset;
    RelOffset += 8 * NumRefs;
  } else {
    // The array of 4B refs doesn't need 8B alignment, but the data will need
    // to be 8B-aligned. Detect this now, and, if necessary, shift everything
    // by 4B by growing one of the sizes.
    uint64_t RefListSize = 4 * NumRefs;
    if (!isAligned(Align(8), RelOffset + RefListSize))
      GrowSizeFieldsBy4B();
    RefsRelOffset = RelOffset;
    RelOffset += RefListSize;
  }

  assert(isAligned(Align(8), RelOffset));
  DataRelOffset = RelOffset;
}

uint64_t DataRecordHandle::getDataSize() const {
  int64_t RelOffset = sizeof(Header);
  auto *DataSizePtr = reinterpret_cast<const char *>(H) + RelOffset;
  switch (getLayoutFlags().DataSize) {
  case DataSizeFlags::Uses1B:
    return (H->Packed >> 48) & UINT8_MAX;
  case DataSizeFlags::Uses2B:
    return (H->Packed >> 32) & UINT16_MAX;
  case DataSizeFlags::Uses4B:
    return *reinterpret_cast<const uint32_t *>(DataSizePtr);
  case DataSizeFlags::Uses8B:
    return *reinterpret_cast<const uint64_t *>(DataSizePtr);
  }
}

void DataRecordHandle::skipDataSize(LayoutFlags LF, int64_t &RelOffset) const {
  if (LF.DataSize >= DataSizeFlags::Uses4B)
    RelOffset += 4;
  if (LF.DataSize >= DataSizeFlags::Uses8B)
    RelOffset += 4;
}

uint32_t DataRecordHandle::getNumRefs() const {
  LayoutFlags LF = getLayoutFlags();
  int64_t RelOffset = sizeof(Header);
  skipDataSize(LF, RelOffset);
  auto *NumRefsPtr = reinterpret_cast<const char *>(H) + RelOffset;
  switch (LF.NumRefs) {
  case NumRefsFlags::Uses0B:
    return 0;
  case NumRefsFlags::Uses1B:
    return (H->Packed >> 48) & UINT8_MAX;
  case NumRefsFlags::Uses2B:
    return (H->Packed >> 32) & UINT16_MAX;
  case NumRefsFlags::Uses4B:
    return *reinterpret_cast<const uint32_t *>(NumRefsPtr);
  case NumRefsFlags::Uses8B:
    return *reinterpret_cast<const uint64_t *>(NumRefsPtr);
  }
}

void DataRecordHandle::skipNumRefs(LayoutFlags LF, int64_t &RelOffset) const {
  if (LF.NumRefs >= NumRefsFlags::Uses4B)
    RelOffset += 4;
  if (LF.NumRefs >= NumRefsFlags::Uses8B)
    RelOffset += 4;
}

int64_t DataRecordHandle::getRefsRelOffset() const {
  LayoutFlags LF = getLayoutFlags();
  int64_t RelOffset = sizeof(Header);
  skipDataSize(LF, RelOffset);
  skipNumRefs(LF, RelOffset);
  return RelOffset;
}

int64_t DataRecordHandle::getDataRelOffset() const {
  LayoutFlags LF = getLayoutFlags();
  int64_t RelOffset = sizeof(Header);
  skipDataSize(LF, RelOffset);
  skipNumRefs(LF, RelOffset);
  uint32_t RefSize = LF.RefKind == RefKindFlags::InternalRef4B ? 4 : 8;
  RelOffset += RefSize * getNumRefs();
  return RelOffset;
}

void OnDiskCAS::print(raw_ostream &OS) const {
  OS << "on-disk-root-path: " << RootPath << "\n";

  struct PoolInfo {
    bool IsString2B;
    int64_t Offset;
  };
  SmallVector<PoolInfo> Pool;

  OS << "\n";
  OS << "index:\n";
  Index.print(OS, [&](ArrayRef<char> Data) {
    assert(Data.size() == sizeof(TrieRecord));
    assert(isAligned(Align::Of<TrieRecord>(), Data.size()));
    auto *R = reinterpret_cast<const TrieRecord *>(Data.data());
    TrieRecord::Data D = R->load();
    OS << "OK=";
    switch (D.OK) {
    case TrieRecord::ObjectKind::Invalid:
      OS << "invalid";
      break;
    case TrieRecord::ObjectKind::Object:
      OS << "object ";
      break;
    case TrieRecord::ObjectKind::String:
      OS << "string ";
      break;
    }
    OS << " SK=";
    switch (D.SK) {
    case TrieRecord::StorageKind::Unknown:
      OS << "unknown          ";
      break;
    case TrieRecord::StorageKind::DataPool:
      OS << "datapool         ";
      Pool.push_back({/*IsString2B=*/false, D.Offset.get()});
      break;
    case TrieRecord::StorageKind::DataPoolString2B:
      OS << "datapool-string2B";
      Pool.push_back({/*IsString2B=*/true, D.Offset.get()});
      break;
    case TrieRecord::StorageKind::Standalone:
      OS << "standalone-data  ";
      break;
    case TrieRecord::StorageKind::StandaloneLeaf:
      OS << "standalone-leaf  ";
      break;
    case TrieRecord::StorageKind::StandaloneLeaf0:
      OS << "standalone-leaf+0";
      break;
    }
    OS << " Offset=" << (void *)D.Offset.get();
  });
  if (Pool.empty())
    return;

  OS << "\n";
  OS << "pool:\n";
  llvm::sort(
      Pool, [](PoolInfo LHS, PoolInfo RHS) { return LHS.Offset < RHS.Offset; });
  for (PoolInfo PI : Pool) {
    OS << "- addr=" << (void *)PI.Offset << " ";
    if (PI.IsString2B) {
      auto S = String2BHandle::get(DataPool.beginData(FileOffset(PI.Offset)));
      OS << "string length=" << S.getLength();
      OS << " end="
         << (void *)(PI.Offset + sizeof(String2BHandle::Header) +
                     S.getLength() + 1)
         << "\n";
      continue;
    }
    DataRecordHandle D =
        DataRecordHandle::get(DataPool.beginData(FileOffset(PI.Offset)));
    OS << "record refs=" << D.getNumRefs() << " data=" << D.getDataSize()
       << " size=" << D.getTotalSize()
       << " end=" << (void *)(PI.Offset + D.getTotalSize()) << "\n";
  }
}

IndexProxy OnDiskCAS::indexHash(ArrayRef<uint8_t> Hash) {
  OnDiskHashMappedTrie::pointer P = Index.insertLazy(
      Hash, [](FileOffset TentativeOffset,
               OnDiskHashMappedTrie::ValueProxy TentativeValue) {
        assert(TentativeValue.Data.size() == sizeof(TrieRecord));
        assert(
            isAddrAligned(Align::Of<TrieRecord>(), TentativeValue.Data.data()));
        new (TentativeValue.Data.data()) TrieRecord();
      });
  assert(P && "Expected insertion");
  return getIndexProxyFromPointer(P);
}

IndexProxy OnDiskCAS::getIndexProxyFromPointer(
    OnDiskHashMappedTrie::const_pointer P) const {
  assert(P);
  assert(P.getOffset());
  return IndexProxy{P.getOffset(), P->Hash,
                    *const_cast<TrieRecord *>(
                        reinterpret_cast<const TrieRecord *>(P->Data.data()))};
}

Optional<ObjectRef> OnDiskCAS::getReference(const CASID &ID) const {
  OnDiskHashMappedTrie::const_pointer P = getInternalIndexPointer(ID);
  if (!P)
    return None;
  IndexProxy I = getIndexProxyFromPointer(P);
  if (Optional<InternalRef> Ref = makeInternalRef(I.Offset, I.Ref.load()))
    return getExternalReference(*Ref);
  return None;
}

OnDiskHashMappedTrie::const_pointer
OnDiskCAS::getInternalIndexPointer(const CASID &ID) const {
  // Recover the pointer from the FileOffset if ID comes from this CAS.
  if (&ID.getContext() == this) {
    OnDiskHashMappedTrie::const_pointer P =
        Index.recoverFromFileOffset(FileOffset(ID.getInternalID(*this)));
    assert(P && "Expected valid index pointer from direct lookup");
    return P;
  }

  // Fallback to a normal lookup.
  return Index.find(ID.getHash());
}

OnDiskHashMappedTrie::const_pointer
OnDiskCAS::getInternalIndexPointer(InternalRef Ref) const {
  if (Optional<FileOffset> Offset = getIndexOffset(Ref))
    return Index.recoverFromFileOffset(*Offset);
  return {};
}

Optional<IndexProxy> OnDiskCAS::getIndexProxyFromRef(InternalRef Ref) const {
  if (OnDiskHashMappedTrie::const_pointer P = getInternalIndexPointer(Ref))
    return getIndexProxyFromPointer(P);
  return None;
}

Optional<FileOffset> OnDiskCAS::getIndexOffset(InternalRef Ref) const {
  switch (Ref.getOffsetKind()) {
  case InternalRef::OffsetKind::String2B:
    return None;

  case InternalRef::OffsetKind::IndexRecord:
    return Ref.getFileOffset();

  case InternalRef::OffsetKind::DataRecord:
    return DataRecordHandle::get(DataPool.beginData(Ref.getFileOffset()))
        .getTrieRecordOffset();
  }
}

CASID OnDiskCAS::getID(InternalRef Ref) const {
  Optional<FileOffset> I = getIndexOffset(Ref);
  assert(I);
  return getIDFromIndexOffset(*I);
}

ArrayRef<char> OnDiskCAS::getDataConst(ObjectHandle Node) const {
  OnDiskContent Content = getContentFromHandle(Node);
  if (Content.Bytes)
    return *Content.Bytes;
  assert(Content.Record && "Expected record or bytes");
  return Content.Record->getData();
}

Expected<ObjectHandle> OnDiskCAS::load(ObjectRef ExternalRef) {
  InternalRef Ref = getInternalRef(ExternalRef);
  Optional<IndexProxy> I = getIndexProxyFromRef(Ref);
  if (!I)
    report_fatal_error(
        "OnDiskCAS: corrupt internal reference to unknown object");
  return load(*I, I->Ref.load(), Ref);
}

ObjectHandle OnDiskCAS::getLoadedObject(const IndexProxy &I,
                                        TrieRecord::Data Object,
                                        InternalHandle Handle) const {
  switch (Object.OK) {
  case TrieRecord::ObjectKind::Invalid:
  case TrieRecord::ObjectKind::String:
    report_fatal_error(createCorruptObjectError(getID(I)));
  case TrieRecord::ObjectKind::Object:
    return makeObjectHandle(Handle.getRawData());
  }
}

Expected<ObjectHandle>
OnDiskCAS::load(const IndexProxy &I, TrieRecord::Data Object, InternalRef Ref) {
  Optional<InternalHandle> Handle;
  if (Error E = loadContentForRef(I, Object, Ref).moveInto(Handle))
    return std::move(E);

  return getLoadedObject(I, Object, *Handle);
}

Optional<InternalRef>
OnDiskCAS::makeInternalRef(FileOffset IndexOffset,
                           TrieRecord::Data Object) const {
  switch (Object.SK) {
  case TrieRecord::StorageKind::Unknown:
    return None;

  case TrieRecord::StorageKind::DataPool:
    return InternalRef::getFromOffset(InternalRef::OffsetKind::DataRecord,
                                      Object.Offset);

  case TrieRecord::StorageKind::DataPoolString2B:
    return InternalRef::getFromOffset(InternalRef::OffsetKind::String2B,
                                      Object.Offset);

  case TrieRecord::StorageKind::Standalone:
  case TrieRecord::StorageKind::StandaloneLeaf:
  case TrieRecord::StorageKind::StandaloneLeaf0:
    return InternalRef::getFromOffset(InternalRef::OffsetKind::IndexRecord,
                                      IndexOffset);
  }
}

Expected<InternalRef> OnDiskCAS::storeTreeEntryName(StringRef String) {
  if (String.size() > UINT16_MAX)
    return createStringError(std::make_error_code(std::errc::invalid_argument),
                             "tree entry name too large for implementation");

  // Make a string.
  //
  // FIXME: Should be using a content-based trie, rather than computing a hash
  // and using a hash-based trie... otherwise this doesn't give any storage
  // savings!
  IndexProxy I = indexHash(BuiltinObjectHasher<HasherT>::hashString(String));

  // Already exists!
  TrieRecord::Data Object = I.Ref.load();
  if (Object.OK == TrieRecord::ObjectKind::String)
    return *makeInternalRef(I.Offset, Object);
  if (Object.OK != TrieRecord::ObjectKind::Invalid)
    return createCorruptStorageError();

  FileOffset Offset;
  auto Alloc = [&](size_t Size) -> char * {
    OnDiskDataAllocator::pointer P = DataPool.allocate(Size);
    Offset = P.getOffset();
    LLVM_DEBUG({
      dbgs() << "pool-alloc addr=" << (void *)Offset.get() << " size=" << Size
             << " end=" << (void *)(Offset.get() + Size) << "\n";
    });
    return P->data();
  };
  (void)String2BHandle::create(Alloc, String);

  TrieRecord::Data Existing;
  {
    TrieRecord::Data StringData{TrieRecord::StorageKind::DataPoolString2B,
                                TrieRecord::ObjectKind::String, Offset};

    if (I.Ref.compare_exchange_strong(Existing, StringData))
      return *makeInternalRef(I.Offset, StringData);
  }

  // TODO: Find a way to reuse the storage from the new-but-abandoned
  // String2BHandle.
  if (Existing.SK != TrieRecord::StorageKind::DataPoolString2B)
    return createCorruptStorageError();

  return *makeInternalRef(I.Offset, Existing);
}

void OnDiskCAS::getStandalonePath(TrieRecord::StorageKind SK,
                                  const IndexProxy &I,
                                  SmallVectorImpl<char> &Path) const {
  StringRef Suffix;
  switch (SK) {
  default:
    llvm_unreachable("Expected standalone storage kind");

  case TrieRecord::StorageKind::Standalone:
    Suffix = FileSuffixData;
    break;
  case TrieRecord::StorageKind::StandaloneLeaf0:
    Suffix = FileSuffixLeaf0;
    break;
  case TrieRecord::StorageKind::StandaloneLeaf:
    Suffix = FileSuffixLeaf;
    break;
  }

  Path.assign(RootPath.begin(), RootPath.end());
  sys::path::append(Path, FilePrefix + Twine(I.Offset.get()) + Suffix);
}

Expected<InternalHandle> OnDiskCAS::loadContentForRef(const IndexProxy &I,
                                                      TrieRecord::Data Object,
                                                      InternalRef Ref) {
  if (Ref.getOffsetKind() != InternalRef::OffsetKind::IndexRecord)
    return InternalHandle(Ref);

  assert(Ref.getOffsetKind() == InternalRef::OffsetKind::IndexRecord &&
         "Expected isContentLoaded() to check for 'IndexRecord'");

  // Only TrieRecord::StorageKind::Standalone (and variants) need to be
  // explicitly loaded.
  //
  // There's corruption if standalone objects have offsets, or if we get here
  // for something that isn't standalone.
  if (Object.Offset)
    return createCorruptObjectError(getID(I));
  switch (Object.SK) {
  default:
    return createCorruptObjectError(getID(I));
  case TrieRecord::StorageKind::Standalone:
  case TrieRecord::StorageKind::StandaloneLeaf0:
  case TrieRecord::StorageKind::StandaloneLeaf:
    break;
  }

  // Load it from disk.
  //
  // Note: Creation logic guarantees that data that needs null-termination is
  // suitably 0-padded. Requiring null-termination here would be too expensive
  // for extremely large objects that happen to be page-aligned.
  SmallString<256> Path;
  getStandalonePath(Object.SK, I, Path);
  ErrorOr<std::unique_ptr<MemoryBuffer>> OwnedBuffer = MemoryBuffer::getFile(
      Path, /*IsText=*/false, /*RequiresNullTerminator=*/false);
  if (!OwnedBuffer)
    return createCorruptObjectError(getID(I));

  return InternalHandle(
      StandaloneData.insert(I.Hash, Ref, Object.SK, std::move(*OwnedBuffer)));
}

OnDiskContent OnDiskCAS::getContentFromHandle(InternalHandle Handle) const {
  if (Handle.SDIM)
    return Handle.SDIM->getContent();

  InternalRef DirectRef = *Handle.DirectRef;
  switch (DirectRef.getOffsetKind()) {
  case InternalRef::OffsetKind::String2B: {
    auto Handle =
        String2BHandle::get(DataPool.beginData(DirectRef.getFileOffset()));
    assert(Handle.getString().end()[0] == 0 && "Null termination");
    return OnDiskContent{None, toArrayRef(Handle.getString())};
  }

  case InternalRef::OffsetKind::DataRecord: {
    auto Handle =
        DataRecordHandle::get(DataPool.beginData(DirectRef.getFileOffset()));
    assert(Handle.getData().end()[0] == 0 && "Null termination");
    return OnDiskContent{Handle, None};
  }

  case InternalRef::OffsetKind::IndexRecord:
    report_fatal_error("Invalid standalone internal handle without SDIM");
  }
}

OnDiskContent StandaloneDataInMemory::getContent() const {
  bool Leaf0 = false;
  bool Leaf = false;
  switch (SK) {
  default:
    llvm_unreachable("Storage kind must be standalone");
  case TrieRecord::StorageKind::Standalone:
    break;
  case TrieRecord::StorageKind::StandaloneLeaf0:
    Leaf = Leaf0 = true;
    break;
  case TrieRecord::StorageKind::StandaloneLeaf:
    Leaf = true;
    break;
  }

  if (Leaf) {
    assert(Region->getBuffer().drop_back(Leaf0).end()[0] == 0 &&
           "Standalone node data missing null termination");
    return OnDiskContent{None,
                         toArrayRef(Region->getBuffer().drop_back(Leaf0))};
  }

  DataRecordHandle Record = DataRecordHandle::get(Region->getBuffer().data());
  assert(Record.getData().end()[0] == 0 &&
         "Standalone object record missing null termination for data");
  return OnDiskContent{Record, None};
}

Expected<OnDiskCAS::MappedTempFile>
OnDiskCAS::createTempFile(StringRef FinalPath, uint64_t Size) {
  assert(Size && "Unexpected request for an empty temp file");
  Expected<TempFile> File = TempFile::create(FinalPath + ".%%%%%%");
  if (!File)
    return File.takeError();

  if (auto EC = sys::fs::resize_file_before_mapping_readwrite(File->FD, Size))
    return createFileError(File->TmpName, EC);

  std::error_code EC;
  sys::fs::mapped_file_region Map(sys::fs::convertFDToNativeFile(File->FD),
                                  sys::fs::mapped_file_region::readwrite, Size,
                                  0, EC);
  if (EC)
    return createFileError(File->TmpName, EC);
  return MappedTempFile(std::move(*File), std::move(Map));
}

static size_t getPageSize() {
  static int PageSize = sys::Process::getPageSizeEstimate();
  return PageSize;
}

Expected<ObjectHandle> OnDiskCAS::createStandaloneLeaf(IndexProxy &I,
                                                     ArrayRef<char> Data) {
  assert(Data.size() > TrieRecord::MaxEmbeddedSize &&
         "Expected a bigger file for external content...");

  bool Leaf0 = isAligned(Align(getPageSize()), Data.size());
  TrieRecord::StorageKind SK = Leaf0 ? TrieRecord::StorageKind::StandaloneLeaf0
                                     : TrieRecord::StorageKind::StandaloneLeaf;

  SmallString<256> Path;
  int64_t FileSize = Data.size() + Leaf0;
  getStandalonePath(SK, I, Path);

  // Write the file. Don't reuse this mapped_file_region, which is read/write.
  // Let load() pull up one that's read-only.
  Expected<MappedTempFile> File = createTempFile(Path, FileSize);
  if (!File)
    return File.takeError();
  assert(File->size() == (uint64_t)FileSize);
  llvm::copy(Data, File->data());
  if (Leaf0)
    File->data()[Data.size()] = 0;
  assert(File->data()[Data.size()] == 0);
  if (Error E = File->keep(Path))
    return std::move(E);

  // Store the object reference.
  TrieRecord::Data Existing;
  {
    TrieRecord::Data Leaf{SK, TrieRecord::ObjectKind::Object, FileOffset()};
    if (I.Ref.compare_exchange_strong(Existing, Leaf))
      return load(I, Leaf, *makeInternalRef(I.Offset, Leaf));
  }

  // If there was a race, confirm that the new value has valid storage.
  if (Existing.SK == TrieRecord::StorageKind::Unknown ||
      Existing.OK != TrieRecord::ObjectKind::Object)
    return createCorruptObjectError(getID(I));

  // Get and return the inserted leaf node.
  return load(I, Existing, *makeInternalRef(I.Offset, Existing));
}

Expected<ObjectHandle> OnDiskCAS::storeImpl(ArrayRef<uint8_t> ComputedHash,
                                            ArrayRef<ObjectRef> Refs,
                                            ArrayRef<char> Data) {
  IndexProxy I = indexHash(ComputedHash);

  // Early return in case the node exists.
  {
    TrieRecord::Data Existing = I.Ref.load();
    if (Existing.OK == TrieRecord::ObjectKind::Object)
      return load(I, Existing, *makeInternalRef(I.Offset, Existing));
    if (Existing.SK != TrieRecord::StorageKind::Unknown)
      return createCorruptObjectError(getID(I));
  }

  // Big leaf nodes.
  if (Refs.empty() && Data.size() > TrieRecord::MaxEmbeddedSize)
    return createStandaloneLeaf(I, Data);

  // TODO: Check whether it's worth checking the index for an already existing
  // object (like storeTreeImpl() does) before building up the
  // InternalRefVector.
  InternalRefVector InternalRefs;
  for (ObjectRef Ref : Refs)
    InternalRefs.push_back(getInternalRef(Ref));

  // Create the object.
  return loadOrCreateDataRecord(
      I, TrieRecord::ObjectKind::Object,
      DataRecordHandle::Input{I.Offset, InternalRefs, Data});
}

Expected<ObjectHandle>
OnDiskCAS::loadOrCreateDataRecord(IndexProxy &I, TrieRecord::ObjectKind OK,
                                  DataRecordHandle::Input Input) {
  // Compute the storage kind, allocate it, and create the record.
  TrieRecord::StorageKind SK = TrieRecord::StorageKind::Unknown;
  FileOffset PoolOffset;
  SmallString<256> Path;
  Optional<MappedTempFile> File;
  auto Alloc = [&](size_t Size) -> Expected<char *> {
    if (Size <= TrieRecord::MaxEmbeddedSize) {
      SK = TrieRecord::StorageKind::DataPool;
      OnDiskDataAllocator::pointer P = DataPool.allocate(Size);
      PoolOffset = P.getOffset();
      LLVM_DEBUG({
        dbgs() << "pool-alloc addr=" << (void *)PoolOffset.get()
               << " size=" << Size
               << " end=" << (void *)(PoolOffset.get() + Size) << "\n";
      });
      return P->data();
    }

    SK = TrieRecord::StorageKind::Standalone;
    getStandalonePath(SK, I, Path);
    if (Error E = createTempFile(Path, Size).moveInto(File))
      return std::move(E);
    return File->data();
  };
  DataRecordHandle Record;
  if (Error E =
          DataRecordHandle::createWithError(Alloc, Input).moveInto(Record))
    return std::move(E);
  assert(Record.getData().end()[0] == 0 && "Expected null-termination");
  assert(Record.getData() == Input.Data && "Expected initialization");
  assert(SK != TrieRecord::StorageKind::Unknown);
  assert(bool(File) != bool(PoolOffset) &&
         "Expected either a mapped file or a pooled offset");

  // Check for a race before calling MappedTempFile::keep().
  //
  // Then decide what to do with the file. Better to discard than overwrite if
  // another thread/process has already added this.
  TrieRecord::Data Existing = I.Ref.load();
  {
    TrieRecord::Data NewObject{SK, OK, PoolOffset};
    if (File) {
      if (Existing.SK == TrieRecord::StorageKind::Unknown) {
        // Keep the file!
        if (Error E = File->keep(Path))
          return std::move(E);
      } else {
        File.reset();
      }
    }

    // If we didn't already see a racing/existing write, then try storing the
    // new object. If that races, confirm that the new value has valid storage.
    //
    // TODO: Find a way to reuse the storage from the new-but-abandoned record
    // handle.
    if (Existing.SK == TrieRecord::StorageKind::Unknown) {
      if (I.Ref.compare_exchange_strong(Existing, NewObject))
        return load(I, NewObject, *makeInternalRef(I.Offset, NewObject));
    }
  }

  if (Existing.SK == TrieRecord::StorageKind::Unknown || Existing.OK != OK)
    return createCorruptObjectError(getID(I));

  // Load existing object.
  return load(I, Existing, *makeInternalRef(I.Offset, Existing));
}

OnDiskCAS::PooledDataRecord
OnDiskCAS::createPooledDataRecord(DataRecordHandle::Input Input) {
  FileOffset Offset;
  auto Alloc = [&](size_t Size) -> char * {
    OnDiskDataAllocator::pointer P = DataPool.allocate(Size);
    Offset = P.getOffset();
    LLVM_DEBUG({
      dbgs() << "pool-alloc addr=" << (void *)Offset.get() << " size=" << Size
             << " end=" << (void *)(Offset.get() + Size) << "\n";
    });
    return P->data();
  };
  DataRecordHandle Record = DataRecordHandle::create(Alloc, Input);
  assert(Offset && "Should always have an offset");
  return PooledDataRecord{Offset, Record};
}

Error OnDiskCAS::forEachRef(ObjectHandle Node,
                            function_ref<Error(ObjectRef)> Callback) const {
  for (InternalRef Ref : getRefs(Node))
    if (Error E = Callback(getExternalReference(Ref)))
      return E;
  return Error::success();
}

Expected<CASID> OnDiskCAS::getCachedResult(CASID InputID) {
  // Check that InputID is valid.
  //
  // FIXME: InputID check is silly; we should have a separate ActionKey.
  if (!getReference(InputID))
    return createUnknownObjectError(InputID);

  // Check the result cache.
  //
  // FIXME: Failure here should not be an error.
  OnDiskHashMappedTrie::pointer ActionP = ActionCache.find(InputID.getHash());
  if (!ActionP)
    return createResultCacheMissError(InputID);
  const uint64_t Output =
      reinterpret_cast<const ActionCacheResultT *>(ActionP->Data.data())
          ->load();

  // Return the result.
  return getID(getExternalReference(InternalRef::getFromRawData(Output)));
}

Error OnDiskCAS::putCachedResult(CASID InputID, CASID OutputID) {
  // Check that both IDs are valid.
  //
  // FIXME: InputID check is silly; we should have a separate ActionKey.
  if (!getReference(InputID))
    return createUnknownObjectError(InputID);

  Optional<InternalRef> OutputRef;
  if (Optional<ObjectRef> ExternalRef = getReference(OutputID))
    OutputRef = getInternalRef(*ExternalRef);
  else
    return createUnknownObjectError(OutputID);

  // Insert Input the result cache.
  //
  // FIXME: Consider templating OnDiskHashMappedTrie (really, renaming it to
  // OnDiskHashMappedTrieBase and adding a type-safe layer on top).
  const uint64_t Expected = OutputRef->getRawData();
  OnDiskHashMappedTrie::pointer ActionP = ActionCache.insertLazy(
      InputID.getHash(), [&](FileOffset TentativeOffset,
                             OnDiskHashMappedTrie::ValueProxy TentativeValue) {
        assert(TentativeValue.Data.size() == sizeof(ActionCacheResultT));
        assert(isAddrAligned(Align::Of<ActionCacheResultT>(),
                             TentativeValue.Data.data()));
        new (TentativeValue.Data.data()) ActionCacheResultT(Expected);
      });
  const uint64_t Observed =
      reinterpret_cast<const ActionCacheResultT *>(ActionP->Data.data())
          ->load();

  if (Expected == Observed)
    return Error::success();

  CASID ObservedID =
      getID(getExternalReference(InternalRef::getFromRawData(Observed)));
  return createResultCachePoisonedError(InputID, OutputID, ObservedID);
}

Expected<std::unique_ptr<OnDiskCAS>> OnDiskCAS::open(StringRef AbsPath) {
  if (std::error_code EC = sys::fs::create_directories(AbsPath))
    return createFileError(AbsPath, EC);

  const StringRef Slash = sys::path::get_separator();
  constexpr uint64_t MB = 1024ull * 1024ull;
  constexpr uint64_t GB = 1024ull * 1024ull * 1024ull;
  Optional<OnDiskHashMappedTrie> Index;
  if (Error E = OnDiskHashMappedTrie::create(
                    AbsPath + Slash + FilePrefix + IndexFile,
                    getIndexTableName(), sizeof(HashType) * 8,
                    /*DataSize=*/sizeof(TrieRecord), /*MaxFileSize=*/8 * GB,
                    /*MinFileSize=*/MB)
                    .moveInto(Index))
    return std::move(E);

  Optional<OnDiskDataAllocator> DataPool;
  if (Error E = OnDiskDataAllocator::create(
                    AbsPath + Slash + FilePrefix + DataPoolFile,
                    getDataPoolTableName(),
                    /*MaxFileSize=*/16 * GB, /*MinFileSize=*/MB)
                    .moveInto(DataPool))
    return std::move(E);

  Optional<OnDiskHashMappedTrie> ActionCache;
  if (Error E = OnDiskHashMappedTrie::create(
                    AbsPath + Slash + FilePrefix + ActionCacheFile,
                    getActionCacheTableName(), sizeof(HashType) * 8,
                    /*DataSize=*/sizeof(ActionCacheResultT), /*MaxFileSize=*/GB,
                    /*MinFileSize=*/MB)
                    .moveInto(ActionCache))
    return std::move(E);

  return std::unique_ptr<OnDiskCAS>(new OnDiskCAS(AbsPath, std::move(*Index),
                                                  std::move(*DataPool),
                                                  std::move(*ActionCache)));
}

OnDiskCAS::OnDiskCAS(StringRef RootPath, OnDiskHashMappedTrie Index,
                     OnDiskDataAllocator DataPool,
                     OnDiskHashMappedTrie ActionCache)
    : Index(std::move(Index)), DataPool(std::move(DataPool)),
      ActionCache(std::move(ActionCache)), RootPath(RootPath.str()) {
  SmallString<128> Temp = RootPath;
  sys::path::append(Temp, "tmp.");
  TempPrefix = Temp.str().str();
}

Expected<std::unique_ptr<CASDB>> cas::createOnDiskCAS(const Twine &Path) {
  // FIXME: An absolute path isn't really good enough. Should open a directory
  // and use openat() for files underneath.
  SmallString<256> AbsPath;
  Path.toVector(AbsPath);
  sys::fs::make_absolute(AbsPath);

  // FIXME: Remove this and update clients to do this logic.
  if (AbsPath == getDefaultOnDiskCASStableID())
    AbsPath = StringRef(getDefaultOnDiskCASPath());

  return OnDiskCAS::open(AbsPath);
}

#else /* LLVM_ENABLE_ONDISK_CAS */

Expected<std::unique_ptr<CASDB>> cas::createOnDiskCAS(const Twine &Path) {
  return createStringError(inconvertibleErrorCode(), "OnDiskCAS is disabled");
}

#endif /* LLVM_ENABLE_ONDISK_CAS */

// FIXME: Proxy not portable. Maybe also error-prone?
constexpr StringLiteral DefaultDirProxy = "/^llvm::cas::builtin::default";
constexpr StringLiteral DefaultName = "llvm.cas.builtin.default";

void cas::getDefaultOnDiskCASStableID(SmallVectorImpl<char> &Path) {
  Path.assign(DefaultDirProxy.begin(), DefaultDirProxy.end());
  llvm::sys::path::append(Path, DefaultName);
}

std::string cas::getDefaultOnDiskCASStableID() {
  SmallString<128> Path;
  getDefaultOnDiskCASStableID(Path);
  return Path.str().str();
}

void cas::getDefaultOnDiskCASPath(SmallVectorImpl<char> &Path) {
  // FIXME: Should this return 'Error' instead of hard-failing?
  if (!llvm::sys::path::cache_directory(Path))
    report_fatal_error("cannot get default cache directory");
  llvm::sys::path::append(Path, DefaultName);
}

std::string cas::getDefaultOnDiskCASPath() {
  SmallString<128> Path;
  getDefaultOnDiskCASPath(Path);
  return Path.str().str();
}
