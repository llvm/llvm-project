//===- OnDiskHashMappedTrie.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/OnDiskHashMappedTrie.h"
#include "HashMappedTrieIndexGenerator.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/CAS/MappedFileRegionBumpPtr.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"
#include <mutex>

using namespace llvm;
using namespace llvm::cas;

#if LLVM_ENABLE_ONDISK_CAS

static_assert(sizeof(size_t) == sizeof(uint64_t), "64-bit only");
static_assert(sizeof(std::atomic<int64_t>) == sizeof(uint64_t),
              "Requires lock-free 64-bit atomics");

//===----------------------------------------------------------------------===//
// Generic database data structures.
//===----------------------------------------------------------------------===//
namespace {
using MappedFileRegion = MappedFileRegionBumpPtr::RegionT;

/// Generic handle for a table.
///
/// Probably we want some table kinds for pointing at multiple tables.
/// - Probably a tree or trie type makes sense.
/// - Or a deque. Linear search is okay as long as there aren't many tables in
///   a file.
///
/// Generic table header layout:
/// - 2-bytes: TableKind
/// - 2-bytes: TableNameSize
/// - 4-bytes: TableNameRelOffset (relative to header)
class TableHandle {
public:
  enum class TableKind : uint16_t {
    HashMappedTrie = 1,
    DataAllocator = 2,
  };
  struct Header {
    TableKind Kind;
    uint16_t NameSize;
    int32_t NameRelOffset; // Relative to Header.
  };

  explicit operator bool() const { return H; }
  const Header &getHeader() const { return *H; }
  MappedFileRegion &getRegion() const { return *Region; }

  template <class T> static void check() {
    static_assert(
        std::is_same<decltype(T::Header::GenericHeader), Header>::value,
        "T::GenericHeader should be of type TableHandle::Header");
    static_assert(offsetof(typename T::Header, GenericHeader) == 0,
                  "T::GenericHeader must be the head of T::Header");
  }
  template <class T> bool is() const { return T::Kind == H->Kind; }
  template <class T> T dyn_cast() const {
    check<T>();
    if (is<T>())
      return T(*Region, *reinterpret_cast<typename T::Header *>(H));
    return T();
  }
  template <class T> T cast() const {
    assert(is<T>());
    return dyn_cast<T>();
  }

  StringRef getName() const {
    auto *Begin = reinterpret_cast<const char *>(H) + H->NameRelOffset;
    return StringRef(Begin, H->NameSize);
  }

  TableHandle() = default;
  TableHandle(MappedFileRegion &Region, Header &H) : Region(&Region), H(&H) {}
  TableHandle(MappedFileRegion &Region, intptr_t HeaderOffset)
      : TableHandle(Region,
                    *reinterpret_cast<Header *>(Region.data() + HeaderOffset)) {
  }

private:
  MappedFileRegion *Region = nullptr;
  Header *H = nullptr;
};

/// Encapsulate a database file, which:
/// - Sets/checks magic.
/// - Sets/checks version.
/// - Points at an arbitrary root table (can be changed later using a lock-free
///   algorithm).
/// - Sets up a BumpPtr for allocation.
///
/// Top-level layout:
/// - 8-bytes: Magic
/// - 8-bytes: Version
/// - 8-bytes: RootTable (16-bits: Kind; 48-bits: Offset)
/// - 8-bytes: BumpPtr
class DatabaseFile {
public:
  static constexpr uint64_t getMagic() { return 0x00FFDA7ABA53FF00ULL; }
  static constexpr uint64_t getVersion() { return 1ULL; }
  struct Header {
    uint64_t Magic;
    uint64_t Version;
    std::atomic<int64_t> RootTableOffset;
    std::atomic<int64_t> BumpPtr;
  };

  const Header &getHeader() { return *H; }
  MappedFileRegionBumpPtr &getAlloc() { return Alloc; }
  MappedFileRegion &getRegion() { return Alloc.getRegion(); }

  /// Add a table.
  ///
  /// TODO: Allow lazy construction via getOrCreate()-style API.
  void addTable(TableHandle Table);

  /// Find a table. May return null.
  std::optional<TableHandle> findTable(StringRef Name);

  static Expected<DatabaseFile>
  create(const Twine &Path, uint64_t Capacity,
         function_ref<Error(DatabaseFile &)> NewDBConstructor);

  size_t size() const { return Alloc.size(); }

private:
  static Expected<DatabaseFile>
  get(std::unique_ptr<MappedFileRegionBumpPtr> Alloc) {
    if (Error E = validate(Alloc->getRegion()))
      return std::move(E);
    return DatabaseFile(std::move(Alloc));
  }

  static Error validate(MappedFileRegion &Region);

  DatabaseFile(MappedFileRegionBumpPtr &Alloc)
      : H(reinterpret_cast<Header *>(Alloc.data())), Alloc(Alloc) {}
  DatabaseFile(std::unique_ptr<MappedFileRegionBumpPtr> Alloc)
      : DatabaseFile(*Alloc) {
    OwnedAlloc = std::move(Alloc);
  }

  Header *H = nullptr;
  MappedFileRegionBumpPtr &Alloc;
  std::unique_ptr<MappedFileRegionBumpPtr> OwnedAlloc;
};

} // end anonymous namespace

static Error createTableConfigError(std::errc ErrC, StringRef Path,
                                    StringRef TableName, const Twine &Msg) {
  return createStringError(make_error_code(ErrC),
                           Path + "[" + TableName + "]: " + Msg);
}

Expected<DatabaseFile>
DatabaseFile::create(const Twine &Path, uint64_t Capacity,
                     function_ref<Error(DatabaseFile &)> NewDBConstructor) {
  // Constructor for if the file doesn't exist.
  auto NewFileConstructor = [&](MappedFileRegionBumpPtr &Alloc) -> Error {
    if (Alloc.capacity() < sizeof(Header))
      return createTableConfigError(std::errc::argument_out_of_domain,
                                    Path.str(), "datafile",
                                    "Allocator too small for header");
    (void)new (Alloc.data()) Header{getMagic(), getVersion(), {0}, {0}};
    Alloc.initializeBumpPtr(offsetof(Header, BumpPtr));
    DatabaseFile DB(Alloc);
    return NewDBConstructor(DB);
  };

  // Get or create the file.
  MappedFileRegionBumpPtr Alloc;
  if (Error E = MappedFileRegionBumpPtr::create(Path, Capacity,
                                                offsetof(Header, BumpPtr),
                                                NewFileConstructor)
                    .moveInto(Alloc))
    return std::move(E);

  return DatabaseFile::get(
      std::make_unique<MappedFileRegionBumpPtr>(std::move(Alloc)));
}

void DatabaseFile::addTable(TableHandle Table) {
  assert(Table);
  assert(&Table.getRegion() == &getRegion());
  int64_t ExistingRootOffset = 0;
  const int64_t NewOffset =
      reinterpret_cast<const char *>(&Table.getHeader()) - getRegion().data();
  if (H->RootTableOffset.compare_exchange_strong(ExistingRootOffset, NewOffset))
    return;

  // Silently ignore attempts to set the root to itself.
  if (ExistingRootOffset == NewOffset)
    return;

  // FIXME: Fix the API so that having the same name is not an error. Instead,
  // the colliding table should just be used as-is and the client can decide
  // what to do with the new one.
  //
  // TODO: Add support for creating a chain or tree of tables (more than one at
  // all!) to avoid this error.
  TableHandle Root(getRegion(), ExistingRootOffset);
  if (Root.getName() == Table.getName())
    report_fatal_error(
        createStringError(make_error_code(std::errc::not_supported),
                          "table name collision '" + Table.getName() + "'"));
  else
    report_fatal_error(
        createStringError(make_error_code(std::errc::not_supported),
                          "cannot add new table '" + Table.getName() +
                              "'"
                              " to existing root '" +
                              Root.getName() + "'"));
}

std::optional<TableHandle> DatabaseFile::findTable(StringRef Name) {
  int64_t RootTableOffset = H->RootTableOffset.load();
  if (!RootTableOffset)
    return std::nullopt;

  TableHandle Root(getRegion(), RootTableOffset);
  if (Root.getName() == Name)
    return Root;

  // TODO: Once multiple tables are supported, need to walk to find them.
  return std::nullopt;
}

Error DatabaseFile::validate(MappedFileRegion &Region) {
  if (Region.size() < sizeof(Header))
    return createStringError(std::errc::invalid_argument,
                             "database: missing header");

  // Check the magic and version.
  auto *H = reinterpret_cast<Header *>(Region.data());
  if (H->Magic != getMagic())
    return createStringError(std::errc::invalid_argument,
                             "database: bad magic");
  if (H->Version != getVersion())
    return createStringError(std::errc::invalid_argument,
                             "database: wrong version");

  // Check the bump-ptr, which should point past the header.
  if (H->BumpPtr.load() < (int64_t)sizeof(Header))
    return createStringError(std::errc::invalid_argument,
                             "database: corrupt bump-ptr");

  return Error::success();
}

//===----------------------------------------------------------------------===//
// HashMappedTrie data structures.
//===----------------------------------------------------------------------===//

namespace {

class SubtrieHandle;
class TrieVisitor;
class SubtrieSlotValue {
public:
  explicit operator bool() const { return !isEmpty(); }
  bool isEmpty() const { return !Offset; }
  bool isData() const { return Offset > 0; }
  bool isSubtrie() const { return Offset < 0; }
  int64_t asData() const {
    assert(isData());
    return Offset;
  }
  int64_t asSubtrie() const {
    assert(isSubtrie());
    return -Offset;
  }

  FileOffset asSubtrieFileOffset() const { return FileOffset(asSubtrie()); }

  FileOffset asDataFileOffset() const { return FileOffset(asData()); }

  int64_t getRawOffset() const { return Offset; }

  static SubtrieSlotValue getDataOffset(int64_t Offset) {
    return SubtrieSlotValue(Offset);
  }

  static SubtrieSlotValue getSubtrieOffset(int64_t Offset) {
    return SubtrieSlotValue(-Offset);
  }

  static SubtrieSlotValue getDataOffset(FileOffset Offset) {
    return getDataOffset(Offset.get());
  }

  static SubtrieSlotValue getSubtrieOffset(FileOffset Offset) {
    return getDataOffset(Offset.get());
  }

  static SubtrieSlotValue getFromSlot(std::atomic<int64_t> &Slot) {
    return SubtrieSlotValue(Slot.load());
  }

  SubtrieSlotValue() = default;

private:
  friend class SubtrieHandle;
  explicit SubtrieSlotValue(int64_t Offset) : Offset(Offset) {}
  int64_t Offset = 0;
};

class HashMappedTrieHandle;

/// Subtrie layout:
/// - 2-bytes: StartBit
/// - 1-bytes: NumBits=lg(num-slots)
/// - 5-bytes: 0-pad
/// - <slots>
class SubtrieHandle {
public:
  struct Header {
    /// The bit this subtrie starts on.
    uint16_t StartBit;

    /// The number of bits this subtrie handles. It has 2^NumBits slots.
    uint8_t NumBits;

    /// 0-pad to 8B.
    uint8_t ZeroPad1B;
    uint32_t ZeroPad4B;
  };

  /// Slot storage:
  /// - zero:     Empty
  /// - positive: RecordOffset
  /// - negative: SubtrieOffset
  using SlotT = std::atomic<int64_t>;

  static int64_t getSlotsSize(uint32_t NumBits) {
    return sizeof(int64_t) * (1u << NumBits);
  }

  static int64_t getSize(uint32_t NumBits) {
    return sizeof(SubtrieHandle::Header) + getSlotsSize(NumBits);
  }

  int64_t getSize() const { return getSize(H->NumBits); }
  size_t getNumSlots() const { return Slots.size(); }

  SubtrieSlotValue load(size_t I) const {
    return SubtrieSlotValue(Slots[I].load());
  }
  void store(size_t I, SubtrieSlotValue V) {
    return Slots[I].store(V.getRawOffset());
  }

  void printHash(raw_ostream &OS, ArrayRef<uint8_t> Bytes) const;

  /// Return None on success, or the existing offset on failure.
  bool compare_exchange_strong(size_t I, SubtrieSlotValue &Expected,
                               SubtrieSlotValue New) {
    return Slots[I].compare_exchange_strong(Expected.Offset, New.Offset);
  }

  /// Sink \p V from \p I in this subtrie down to \p NewI in a new subtrie with
  /// \p NumSubtrieBits.
  ///
  /// \p UnusedSubtrie maintains a 1-item "free" list of unused subtries. If a
  /// new subtrie is created that isn't used because of a lost race, then it If
  /// it's already valid, it should be used instead of allocating a new one.
  /// should be returned as an out parameter to be passed back in the future.
  /// If it's already valid, it should be used instead of allocating a new one.
  ///
  /// Returns the subtrie that now lives at \p I.
  Expected<SubtrieHandle> sink(size_t I, SubtrieSlotValue V,
                               MappedFileRegionBumpPtr &Alloc,
                               size_t NumSubtrieBits,
                               SubtrieHandle &UnusedSubtrie, size_t NewI);

  /// Only safe if the subtrie is empty.
  void reinitialize(uint32_t StartBit, uint32_t NumBits);

  SubtrieSlotValue getOffset() const {
    return SubtrieSlotValue::getSubtrieOffset(
        reinterpret_cast<const char *>(H) - Region->data());
  }

  FileOffset getFileOffset() const { return getOffset().asSubtrieFileOffset(); }

  explicit operator bool() const { return H; }

  Header &getHeader() const { return *H; }
  uint32_t getStartBit() const { return H->StartBit; }
  uint32_t getNumBits() const { return H->NumBits; }

  static Expected<SubtrieHandle> create(MappedFileRegionBumpPtr &Alloc,
                                        uint32_t StartBit, uint32_t NumBits);

  static SubtrieHandle getFromFileOffset(MappedFileRegion &Region,
                                         FileOffset Offset) {
    return SubtrieHandle(Region, SubtrieSlotValue::getSubtrieOffset(Offset));
  }

  SubtrieHandle() = default;
  SubtrieHandle(MappedFileRegion &Region, Header &H)
      : Region(&Region), H(&H), Slots(getSlots(H)) {}
  SubtrieHandle(MappedFileRegion &Region, SubtrieSlotValue Offset)
      : SubtrieHandle(Region, *reinterpret_cast<Header *>(
                                  Region.data() + Offset.asSubtrie())) {}

private:
  MappedFileRegion *Region = nullptr;
  Header *H = nullptr;
  MutableArrayRef<SlotT> Slots;

  static MutableArrayRef<SlotT> getSlots(Header &H) {
    return MutableArrayRef(reinterpret_cast<SlotT *>(&H + 1), 1u << H.NumBits);
  }
};

/// Handle for a HashMappedTrie table.
///
/// HashMappedTrie table layout:
/// - [8-bytes: Generic table header]
/// - 1-byte: NumSubtrieBits
/// - 1-byte:  Flags (not used yet)
/// - 2-bytes: NumHashBits
/// - 4-bytes: RecordDataSize (in bytes)
/// - 8-bytes: RootTrieOffset
/// - 8-bytes: AllocatorOffset (reserved for implementing free lists)
/// - <name> '\0'
///
/// Record layout:
/// - <data>
/// - <hash>
class HashMappedTrieHandle {
public:
  static constexpr TableHandle::TableKind Kind =
      TableHandle::TableKind::HashMappedTrie;

  struct Header {
    TableHandle::Header GenericHeader;
    uint8_t NumSubtrieBits;
    uint8_t Flags; // None used yet.
    uint16_t NumHashBits;
    uint32_t RecordDataSize;
    std::atomic<int64_t> RootTrieOffset;
    std::atomic<int64_t> AllocatorOffset;
  };

  operator TableHandle() const {
    if (!H)
      return TableHandle();
    return TableHandle(*Region, H->GenericHeader);
  }

  struct RecordData {
    OnDiskHashMappedTrie::ValueProxy Proxy;
    SubtrieSlotValue Offset;
    FileOffset getFileOffset() const { return Offset.asDataFileOffset(); }
  };

  enum Limits : size_t {
    /// Seems like 65528 hash bits ought to be enough.
    MaxNumHashBytes = UINT16_MAX >> 3,
    MaxNumHashBits = MaxNumHashBytes << 3,

    /// 2^16 bits in a trie is 65536 slots. This restricts us to a 16-bit
    /// index. This many slots is suspicously large anyway.
    MaxNumRootBits = 16,

    /// 2^10 bits in a trie is 1024 slots. This many slots seems suspiciously
    /// large for subtries.
    MaxNumSubtrieBits = 10,
  };

  static constexpr size_t getNumHashBytes(size_t NumHashBits) {
    assert(NumHashBits % 8 == 0);
    return NumHashBits / 8;
  }
  static constexpr size_t getRecordSize(size_t RecordDataSize,
                                        size_t NumHashBits) {
    return RecordDataSize + getNumHashBytes(NumHashBits);
  }

  RecordData getRecord(SubtrieSlotValue Offset);
  Expected<RecordData> createRecord(MappedFileRegionBumpPtr &Alloc,
                                    ArrayRef<uint8_t> Hash);

  explicit operator bool() const { return H; }
  const Header &getHeader() const { return *H; }
  SubtrieHandle getRoot() const;
  Expected<SubtrieHandle> getOrCreateRoot(MappedFileRegionBumpPtr &Alloc);
  MappedFileRegion &getRegion() const { return *Region; }

  size_t getFlags() const { return H->Flags; }
  uint64_t getNumSubtrieBits() const { return H->NumSubtrieBits; }
  uint64_t getNumHashBits() const { return H->NumHashBits; }
  size_t getNumHashBytes() const { return getNumHashBytes(H->NumHashBits); }
  size_t getRecordDataSize() const { return H->RecordDataSize; }
  size_t getRecordSize() const {
    return getRecordSize(H->RecordDataSize, H->NumHashBits);
  }

  IndexGenerator getIndexGen(SubtrieHandle Root, ArrayRef<uint8_t> Hash) {
    assert(Root.getStartBit() == 0);
    assert(getNumHashBytes() == Hash.size());
    assert(getNumHashBits() == Hash.size() * 8);
    return IndexGenerator{Root.getNumBits(), getNumSubtrieBits(), Hash};
  }

  static Expected<HashMappedTrieHandle>
  create(MappedFileRegionBumpPtr &Alloc, StringRef Name,
         std::optional<uint64_t> NumRootBits, uint64_t NumSubtrieBits,
         uint64_t NumHashBits, uint64_t RecordDataSize);

  void
  print(raw_ostream &OS,
        function_ref<void(ArrayRef<char>)> PrintRecordData = nullptr) const;

  Error validate(
      function_ref<Error(FileOffset, OnDiskHashMappedTrie::ConstValueProxy)>
          RecordVerifier) const;

  HashMappedTrieHandle() = default;
  HashMappedTrieHandle(MappedFileRegion &Region, Header &H)
      : Region(&Region), H(&H) {}
  HashMappedTrieHandle(MappedFileRegion &Region, intptr_t HeaderOffset)
      : HashMappedTrieHandle(
            Region, *reinterpret_cast<Header *>(Region.data() + HeaderOffset)) {
  }

private:
  MappedFileRegion *Region = nullptr;
  Header *H = nullptr;
};

} // end anonymous namespace

struct OnDiskHashMappedTrie::ImplType {
  DatabaseFile File;
  HashMappedTrieHandle Trie;
};

Expected<SubtrieHandle> SubtrieHandle::create(MappedFileRegionBumpPtr &Alloc,
                                              uint32_t StartBit,
                                              uint32_t NumBits) {
  assert(StartBit <= HashMappedTrieHandle::MaxNumHashBits);
  assert(NumBits <= UINT8_MAX);
  assert(NumBits <= HashMappedTrieHandle::MaxNumRootBits);

  auto Mem = Alloc.allocate(getSize(NumBits));
  if (LLVM_UNLIKELY(!Mem))
    return Mem.takeError();
  auto *H =
      new (*Mem) SubtrieHandle::Header{(uint16_t)StartBit, (uint8_t)NumBits,
                                       /*ZeroPad1B=*/0, /*ZeroPad4B=*/0};
  SubtrieHandle S(Alloc.getRegion(), *H);
  for (auto I = S.Slots.begin(), E = S.Slots.end(); I != E; ++I)
    new (I) SlotT(0);
  return S;
}

SubtrieHandle HashMappedTrieHandle::getRoot() const {
  if (int64_t Root = H->RootTrieOffset)
    return SubtrieHandle(getRegion(), SubtrieSlotValue::getSubtrieOffset(Root));
  return SubtrieHandle();
}

Expected<SubtrieHandle>
HashMappedTrieHandle::getOrCreateRoot(MappedFileRegionBumpPtr &Alloc) {
  assert(&Alloc.getRegion() == &getRegion());
  if (SubtrieHandle Root = getRoot())
    return Root;

  int64_t Race = 0;
  auto LazyRoot = SubtrieHandle::create(Alloc, 0, H->NumSubtrieBits);
  if (LLVM_UNLIKELY(!LazyRoot))
    return LazyRoot.takeError();

  if (H->RootTrieOffset.compare_exchange_strong(
          Race, LazyRoot->getOffset().asSubtrie()))
    return *LazyRoot;

  // There was a race. Return the other root.
  //
  // TODO: Avoid leaking the lazy root by storing it in an allocator.
  return SubtrieHandle(getRegion(), SubtrieSlotValue::getSubtrieOffset(Race));
}

Expected<HashMappedTrieHandle>
HashMappedTrieHandle::create(MappedFileRegionBumpPtr &Alloc, StringRef Name,
                             std::optional<uint64_t> NumRootBits,
                             uint64_t NumSubtrieBits, uint64_t NumHashBits,
                             uint64_t RecordDataSize) {
  // Allocate.
  auto Offset = Alloc.allocateOffset(sizeof(Header) + Name.size() + 1);
  if (LLVM_UNLIKELY(!Offset))
    return Offset.takeError();

  // Construct the header and the name.
  assert(Name.size() <= UINT16_MAX && "Expected smaller table name");
  assert(NumSubtrieBits <= UINT8_MAX && "Expected valid subtrie bits");
  assert(NumHashBits <= UINT16_MAX && "Expected valid hash size");
  assert(RecordDataSize <= UINT32_MAX && "Expected smaller table name");
  auto *H = new (Alloc.getRegion().data() + *Offset)
      Header{{TableHandle::TableKind::HashMappedTrie, (uint16_t)Name.size(),
              (uint32_t)sizeof(Header)},
             (uint8_t)NumSubtrieBits,
             /*Flags=*/0,
             (uint16_t)NumHashBits,
             (uint32_t)RecordDataSize,
             /*RootTrieOffset=*/{0},
             /*AllocatorOffset=*/{0}};
  char *NameStorage = reinterpret_cast<char *>(H + 1);
  llvm::copy(Name, NameStorage);
  NameStorage[Name.size()] = 0;

  // Construct a root trie, if requested.
  HashMappedTrieHandle Trie(Alloc.getRegion(), *H);
  auto Sub = SubtrieHandle::create(Alloc, 0, *NumRootBits);
  if (LLVM_UNLIKELY(!Sub))
    return Sub.takeError();
  if (NumRootBits)
    H->RootTrieOffset = Sub->getOffset().asSubtrie();
  return Trie;
}

HashMappedTrieHandle::RecordData
HashMappedTrieHandle::getRecord(SubtrieSlotValue Offset) {
  char *Begin = Region->data() + Offset.asData();
  OnDiskHashMappedTrie::ValueProxy Proxy;
  Proxy.Data = MutableArrayRef(Begin, getRecordDataSize());
  Proxy.Hash = ArrayRef(reinterpret_cast<const uint8_t *>(Proxy.Data.end()),
                            getNumHashBytes());
  return RecordData{Proxy, Offset};
}

Expected<HashMappedTrieHandle::RecordData>
HashMappedTrieHandle::createRecord(MappedFileRegionBumpPtr &Alloc,
                                   ArrayRef<uint8_t> Hash) {
  assert(&Alloc.getRegion() == Region);
  assert(Hash.size() == getNumHashBytes());
  auto Offset = Alloc.allocateOffset(getRecordSize());
  if (LLVM_UNLIKELY(!Offset))
    return Offset.takeError();

  RecordData Record = getRecord(SubtrieSlotValue::getDataOffset(*Offset));
  llvm::copy(Hash, const_cast<uint8_t *>(Record.Proxy.Hash.begin()));
  return Record;
}

OnDiskHashMappedTrie::const_pointer
OnDiskHashMappedTrie::recoverFromHashPointer(
    const uint8_t *HashBeginPtr) const {
  // Record hashes occur immediately after data. Compute the beginning of the
  // record and check for overflow.
  const uintptr_t HashBegin = reinterpret_cast<uintptr_t>(HashBeginPtr);
  const uintptr_t RecordBegin = HashBegin - Impl->Trie.getRecordSize();
  if (HashBegin < RecordBegin)
    return const_pointer();

  // Check that it'll be a positive offset.
  const uintptr_t FileBegin =
      reinterpret_cast<uintptr_t>(Impl->File.getRegion().data());
  if (RecordBegin < FileBegin)
    return const_pointer();

  // Good enough to form an offset. Continue checking there.
  return recoverFromFileOffset(FileOffset(RecordBegin - FileBegin));
}

OnDiskHashMappedTrie::const_pointer
OnDiskHashMappedTrie::recoverFromFileOffset(FileOffset Offset) const {
  // Check alignment.
  if (!isAligned(MappedFileRegionBumpPtr::getAlign(), Offset.get()))
    return const_pointer();

  // Check bounds.
  //
  // Note: There's no potential overflow when using \c uint64_t because Offset
  // is in \c [0,INT64_MAX] and the record size is in \c [0,UINT32_MAX].
  assert(Offset.get() >= 0 && "Expected FileOffset constructor guarantee this");
  if ((uint64_t)Offset.get() + Impl->Trie.getRecordSize() >
      Impl->File.getAlloc().size())
    return const_pointer();

  // Looks okay...
  HashMappedTrieHandle::RecordData D =
      Impl->Trie.getRecord(SubtrieSlotValue::getDataOffset(Offset));
  return const_pointer(D.getFileOffset(), D.Proxy);
}

OnDiskHashMappedTrie::const_pointer
OnDiskHashMappedTrie::find(ArrayRef<uint8_t> Hash) const {
  HashMappedTrieHandle Trie = Impl->Trie;
  assert(Hash.size() == Trie.getNumHashBytes() && "Invalid hash");

  SubtrieHandle S = Trie.getRoot();
  if (!S)
    return const_pointer();

  IndexGenerator IndexGen = Trie.getIndexGen(S, Hash);
  size_t Index = IndexGen.next();
  for (;;) {
    // Try to set the content.
    SubtrieSlotValue V = S.load(Index);
    if (!V)
      return const_pointer();

    // Check for an exact match.
    if (V.isData()) {
      HashMappedTrieHandle::RecordData D = Trie.getRecord(V);
      return D.Proxy.Hash == Hash ? const_pointer(D.getFileOffset(), D.Proxy)
                                  : const_pointer();
    }

    Index = IndexGen.next();
    S = SubtrieHandle(Trie.getRegion(), V);
  }
}

/// Only safe if the subtrie is empty.
void SubtrieHandle::reinitialize(uint32_t StartBit, uint32_t NumBits) {
  assert(StartBit > H->StartBit);
  assert(NumBits <= H->NumBits);
  // Ideally would also assert that all slots are empty, but that's expensive.

  H->StartBit = StartBit;
  H->NumBits = NumBits;
}

Expected<OnDiskHashMappedTrie::pointer>
OnDiskHashMappedTrie::insertLazy(ArrayRef<uint8_t> Hash,
                                 LazyInsertOnConstructCB OnConstruct,
                                 LazyInsertOnLeakCB OnLeak) {
  HashMappedTrieHandle Trie = Impl->Trie;
  assert(Hash.size() == Trie.getNumHashBytes() && "Invalid hash");

  MappedFileRegionBumpPtr &Alloc = Impl->File.getAlloc();
  std::optional<SubtrieHandle> S;
  auto Err = Trie.getOrCreateRoot(Alloc).moveInto(S);
  if (LLVM_UNLIKELY(Err))
    return std::move(Err);

  IndexGenerator IndexGen = Trie.getIndexGen(*S, Hash);
  size_t Index = IndexGen.next();

  // FIXME: Add non-assertion based checks for data corruption that would
  // otherwise cause infinite loops in release builds, instead calling
  // report_fatal_error().
  //
  // Two loops are possible:
  // - All bits used up in the IndexGenerator because subtries are somehow
  //   linked in a cycle. Could confirm that each subtrie's start-bit
  //   follows from the start-bit and num-bits of its parent. Could also check
  //   that the generator doesn't run out of bits.
  // - Existing data matches tail of Hash but not the head (stored in an
  //   invalid spot). Probably a cheap way to check this too, but needs
  //   thought.
  std::optional<HashMappedTrieHandle::RecordData> NewRecord;
  SubtrieHandle UnusedSubtrie;
  for (;;) {
    SubtrieSlotValue Existing = S->load(Index);

    // Try to set it, if it's empty.
    if (!Existing) {
      if (!NewRecord) {
        auto Err = Trie.createRecord(Alloc, Hash).moveInto(NewRecord);
        if (LLVM_UNLIKELY(Err))
          return std::move(Err);
        if (OnConstruct)
          OnConstruct(NewRecord->Offset.asDataFileOffset(), NewRecord->Proxy);
      }

      if (S->compare_exchange_strong(Index, Existing, NewRecord->Offset))
        return pointer(NewRecord->Offset.asDataFileOffset(), NewRecord->Proxy);

      // Race means that Existing is no longer empty; fall through...
    }

    if (Existing.isSubtrie()) {
      S = SubtrieHandle(Trie.getRegion(), Existing);
      Index = IndexGen.next();
      continue;
    }

    // Check for an exact match.
    HashMappedTrieHandle::RecordData ExistingRecord = Trie.getRecord(Existing);
    if (ExistingRecord.Proxy.Hash == Hash) {
      if (NewRecord && OnLeak)
        OnLeak(NewRecord->Offset.asDataFileOffset(), NewRecord->Proxy,
               ExistingRecord.Offset.asDataFileOffset(), ExistingRecord.Proxy);
      return pointer(ExistingRecord.Offset.asDataFileOffset(),
                     ExistingRecord.Proxy);
    }

    // Sink the existing content as long as the indexes match.
    for (;;) {
      size_t NextIndex = IndexGen.next();
      size_t NewIndexForExistingContent =
          IndexGen.getCollidingBits(ExistingRecord.Proxy.Hash);

      auto Err = S->sink(Index, Existing, Alloc, IndexGen.getNumBits(),
                         UnusedSubtrie, NewIndexForExistingContent)
                     .moveInto(S);
      if (LLVM_UNLIKELY(Err))
        return std::move(Err);
      Index = NextIndex;

      // Found the difference.
      if (NextIndex != NewIndexForExistingContent)
        break;
    }
  }
}

Expected<SubtrieHandle> SubtrieHandle::sink(size_t I, SubtrieSlotValue V,
                                            MappedFileRegionBumpPtr &Alloc,
                                            size_t NumSubtrieBits,
                                            SubtrieHandle &UnusedSubtrie,
                                            size_t NewI) {
  std::optional<SubtrieHandle> NewS;
  if (UnusedSubtrie) {
    // Steal UnusedSubtrie and initialize it.
    NewS.emplace();
    std::swap(*NewS, UnusedSubtrie);
    NewS->reinitialize(getStartBit() + getNumBits(), NumSubtrieBits);
  } else {
    // Allocate a new, empty subtrie.
    auto Err = SubtrieHandle::create(Alloc, getStartBit() + getNumBits(),
                                     NumSubtrieBits)
                   .moveInto(NewS);
    if (LLVM_UNLIKELY(Err))
      return std::move(Err);
  }

  NewS->store(NewI, V);
  if (compare_exchange_strong(I, V, NewS->getOffset()))
    return *NewS; // Success!

  // Raced.
  assert(V.isSubtrie() && "Expected racing sink() to add a subtrie");

  // Wipe out the new slot so NewS can be reused and set the out parameter.
  NewS->store(NewI, SubtrieSlotValue());
  UnusedSubtrie = *NewS;

  // Return the subtrie added by the concurrent sink() call.
  return SubtrieHandle(Alloc.getRegion(), V);
}

void OnDiskHashMappedTrie::print(
    raw_ostream &OS, function_ref<void(ArrayRef<char>)> PrintRecordData) const {
  Impl->Trie.print(OS, PrintRecordData);
}

Error OnDiskHashMappedTrie::validate(
    function_ref<Error(FileOffset, ConstValueProxy)> RecordVerifier) const {
  return Impl->Trie.validate(RecordVerifier);
}

static void printHexDigit(raw_ostream &OS, uint8_t Digit) {
  if (Digit < 10)
    OS << char(Digit + '0');
  else
    OS << char(Digit - 10 + 'a');
}

static void printHexDigits(raw_ostream &OS, ArrayRef<uint8_t> Bytes,
                           size_t StartBit, size_t NumBits) {
  assert(StartBit % 4 == 0);
  assert(NumBits % 4 == 0);
  for (size_t I = StartBit, E = StartBit + NumBits; I != E; I += 4) {
    uint8_t HexPair = Bytes[I / 8];
    uint8_t HexDigit = I % 8 == 0 ? HexPair >> 4 : HexPair & 0xf;
    printHexDigit(OS, HexDigit);
  }
}

static void printBits(raw_ostream &OS, ArrayRef<uint8_t> Bytes, size_t StartBit,
                      size_t NumBits) {
  assert(StartBit + NumBits <= Bytes.size() * 8u);
  for (size_t I = StartBit, E = StartBit + NumBits; I != E; ++I) {
    uint8_t Byte = Bytes[I / 8];
    size_t ByteOffset = I % 8;
    if (size_t ByteShift = 8 - ByteOffset - 1)
      Byte >>= ByteShift;
    OS << (Byte & 0x1 ? '1' : '0');
  }
}

void SubtrieHandle::printHash(raw_ostream &OS, ArrayRef<uint8_t> Bytes) const {
  // afb[1c:00*01110*0]def
  size_t EndBit = getStartBit() + getNumBits();
  size_t HashEndBit = Bytes.size() * 8u;

  size_t FirstBinaryBit = getStartBit() & ~0x3u;
  printHexDigits(OS, Bytes, 0, FirstBinaryBit);

  size_t LastBinaryBit = (EndBit + 3u) & ~0x3u;
  OS << "[";
  printBits(OS, Bytes, FirstBinaryBit, LastBinaryBit - FirstBinaryBit);
  OS << "]";

  printHexDigits(OS, Bytes, LastBinaryBit, HashEndBit - LastBinaryBit);
}

static void appendIndexBits(std::string &Prefix, size_t Index,
                            size_t NumSlots) {
  std::string Bits;
  for (size_t NumBits = 1u; NumBits < NumSlots; NumBits <<= 1) {
    Bits.push_back('0' + (Index & 0x1));
    Index >>= 1;
  }
  for (char Ch : llvm::reverse(Bits))
    Prefix += Ch;
}

static void printPrefix(raw_ostream &OS, StringRef Prefix) {
  while (Prefix.size() >= 4) {
    uint8_t Digit;
    bool ErrorParsingBinary = Prefix.take_front(4).getAsInteger(2, Digit);
    assert(!ErrorParsingBinary);
    (void)ErrorParsingBinary;
    printHexDigit(OS, Digit);
    Prefix = Prefix.drop_front(4);
  }
  if (!Prefix.empty())
    OS << "[" << Prefix << "]";
}

LLVM_DUMP_METHOD void OnDiskHashMappedTrie::dump() const { print(dbgs()); }

static Expected<size_t> checkParameter(StringRef Label, size_t Max,
                                       std::optional<size_t> Value,
                                       std::optional<size_t> Default,
                                       StringRef Path, StringRef TableName) {
  assert(Value || Default);
  assert(!Default || *Default <= Max);
  if (!Value)
    return *Default;

  if (*Value <= Max)
    return *Value;
  return createTableConfigError(
      std::errc::argument_out_of_domain, Path, TableName,
      "invalid " + Label + ": " + Twine(*Value) + " (max: " + Twine(Max) + ")");
}

static Error checkTable(StringRef Label, size_t Expected, size_t Observed,
                        StringRef Path, StringRef TrieName) {
  if (Expected == Observed)
    return Error::success();
  return createTableConfigError(std::errc::invalid_argument, Path, TrieName,
                                "mismatched " + Label +
                                    " (expected: " + Twine(Expected) +
                                    ", observed: " + Twine(Observed) + ")");
}

size_t OnDiskHashMappedTrie::size() const { return Impl->File.size(); }
size_t OnDiskHashMappedTrie::capacity() const {
  return Impl->File.getRegion().size();
}

Expected<OnDiskHashMappedTrie>
OnDiskHashMappedTrie::create(const Twine &PathTwine, const Twine &TrieNameTwine,
                             size_t NumHashBits, uint64_t DataSize,
                             uint64_t MaxFileSize,
                             std::optional<uint64_t> NewFileInitialSize,
                             std::optional<size_t> NewTableNumRootBits,
                             std::optional<size_t> NewTableNumSubtrieBits) {
  SmallString<128> PathStorage;
  StringRef Path = PathTwine.toStringRef(PathStorage);
  SmallString<128> TrieNameStorage;
  StringRef TrieName = TrieNameTwine.toStringRef(TrieNameStorage);

  constexpr size_t DefaultNumRootBits = 10;
  constexpr size_t DefaultNumSubtrieBits = 6;

  size_t NumRootBits;
  if (Error E = checkParameter(
                    "root bits", HashMappedTrieHandle::MaxNumRootBits,
                    NewTableNumRootBits, DefaultNumRootBits, Path, TrieName)
                    .moveInto(NumRootBits))
    return std::move(E);

  size_t NumSubtrieBits;
  if (Error E = checkParameter("subtrie bits",
                               HashMappedTrieHandle::MaxNumSubtrieBits,
                               NewTableNumSubtrieBits, DefaultNumSubtrieBits,
                               Path, TrieName)
                    .moveInto(NumSubtrieBits))
    return std::move(E);

  size_t NumHashBytes = NumHashBits >> 3;
  if (Error E =
          checkParameter("hash size", HashMappedTrieHandle::MaxNumHashBits,
                         NumHashBits, std::nullopt, Path, TrieName)
              .takeError())
    return std::move(E);
  assert(NumHashBits == NumHashBytes << 3 &&
         "Expected hash size to be byte-aligned");
  if (NumHashBits != NumHashBytes << 3)
    return createTableConfigError(
        std::errc::argument_out_of_domain, Path, TrieName,
        "invalid hash size: " + Twine(NumHashBits) + " (not byte-aligned)");

  // Constructor for if the file doesn't exist.
  auto NewDBConstructor = [&](DatabaseFile &DB) -> Error {
    auto Trie =
        HashMappedTrieHandle::create(DB.getAlloc(), TrieName, NumRootBits,
                                     NumSubtrieBits, NumHashBits, DataSize);
    if (LLVM_UNLIKELY(!Trie))
      return Trie.takeError();

    DB.addTable(*Trie);
    return Error::success();
  };

  // Get or create the file.
  Expected<DatabaseFile> File =
      DatabaseFile::create(Path, MaxFileSize, NewDBConstructor);
  if (!File)
    return File.takeError();

  // Find the trie and validate it.
  //
  // TODO: Add support for creating/adding a table to an existing file.
  std::optional<TableHandle> Table = File->findTable(TrieName);
  if (!Table)
    return createTableConfigError(std::errc::argument_out_of_domain, Path,
                                  TrieName, "table not found");
  if (Error E = checkTable("table kind", (size_t)HashMappedTrieHandle::Kind,
                           (size_t)Table->getHeader().Kind, Path, TrieName))
    return std::move(E);
  auto Trie = Table->cast<HashMappedTrieHandle>();
  assert(Trie && "Already checked the kind");

  // Check the hash and data size.
  if (Error E = checkTable("hash size", NumHashBits, Trie.getNumHashBits(),
                           Path, TrieName))
    return std::move(E);
  if (Error E = checkTable("data size", DataSize, Trie.getRecordDataSize(),
                           Path, TrieName))
    return std::move(E);

  // No flags supported right now. Either corrupt, or coming from a future
  // writer.
  if (size_t Flags = Trie.getFlags())
    return createTableConfigError(std::errc::invalid_argument, Path, TrieName,
                                  "unsupported flags: " + Twine(Flags));

  // Success.
  OnDiskHashMappedTrie::ImplType Impl{DatabaseFile(std::move(*File)), Trie};
  return OnDiskHashMappedTrie(std::make_unique<ImplType>(std::move(Impl)));
}

static Error createInvalidTrieError(uint64_t Offset, const Twine &Msg) {
  return createStringError(make_error_code(std::errc::protocol_error),
                           "invalid trie at 0x" +
                               utohexstr(Offset, /*LowerCase=*/true) + ": " +
                               Msg);
}

//===----------------------------------------------------------------------===//
// TrieVisitor data structures.
//===----------------------------------------------------------------------===//

namespace {
// A vistior to traverse the Trie.
class TrieVisitor {
public:
  TrieVisitor(HashMappedTrieHandle Trie, unsigned ThreadCount = 0,
              unsigned ErrorLimit = 50)
      : Trie(Trie), ErrorLimit(ErrorLimit),
        Threads(hardware_concurrency(ThreadCount)) {}
  virtual ~TrieVisitor() = default;
  Error visit();

private:
  virtual Error visitSubTrie(StringRef Prefix, SubtrieHandle SubTrie) {
    return Error::success();
  }

  virtual Error visitSlot(unsigned I, SubtrieHandle Subtrie, StringRef Prefix,
                          SubtrieSlotValue Slot) {
    return Error::success();
  }

protected:
  HashMappedTrieHandle Trie;

private:
  Error traverseTrieNode(SubtrieHandle Node, StringRef Prefix);

  Error validateSubTrie(SubtrieHandle Node, bool IsRoot);

  void addError(Error NewError) {
    assert(NewError && "not an error");
    std::lock_guard<std::mutex> ErrorLock(Lock);
    if (NumError >= ErrorLimit) {
      // Too many errors.
      consumeError(std::move(NewError));
      return;
    }

    if (Err)
      Err = joinErrors(std::move(*Err), std::move(NewError));
    else
      Err = std::move(NewError);
    NumError++;
  }

  bool tooManyErrors() {
    std::lock_guard<std::mutex> ErrorLock(Lock);
    return (bool)Err && NumError >= ErrorLimit;
  }

  const unsigned ErrorLimit;
  std::optional<Error> Err;
  unsigned NumError = 0;
  std::mutex Lock;
  DefaultThreadPool Threads;
};

class TriePrinter : public TrieVisitor {
public:
  TriePrinter(HashMappedTrieHandle Trie, raw_ostream &OS,
              function_ref<void(ArrayRef<char>)> PrintRecordData)
      : TrieVisitor(Trie, /*ThreadCount=*/1), OS(OS),
        PrintRecordData(PrintRecordData) {}

  Error printRecords() {
    if (Records.empty())
      return Error::success();

    OS << "records\n";
    llvm::sort(Records);
    for (int64_t Offset : Records) {
      HashMappedTrieHandle::RecordData Record =
          Trie.getRecord(SubtrieSlotValue::getDataOffset(Offset));
      if (auto Err = printRecord(Record))
        return Err;
    }
    return Error::success();
  }

  Error printRecord(HashMappedTrieHandle::RecordData &Record) {
    OS << "- addr=" << (void*)Record.getFileOffset().get() << " ";
    if (PrintRecordData) {
      PrintRecordData(Record.Proxy.Data);
    } else {
      OS << "bytes=";
      ArrayRef<uint8_t> Data(
          reinterpret_cast<const uint8_t *>(Record.Proxy.Data.data()),
          Record.Proxy.Data.size());
      printHexDigits(OS, Data, 0, Data.size() * 8);
    }
    OS << "\n";
    return Error::success();
  }

  Error visitSubTrie(StringRef Prefix, SubtrieHandle SubTrie) override {
    if (Prefix.empty()) {
      OS << "root";
    } else {
      OS << "subtrie=";
      printPrefix(OS, Prefix);
    }

    OS << " addr="
       << (void *)(reinterpret_cast<const char *>(&SubTrie.getHeader()) -
                   Trie.getRegion().data());
    OS << " num-slots=" << SubTrie.getNumSlots() << "\n";
    return Error::success();
  }

  Error visitSlot(unsigned I, SubtrieHandle Subtrie, StringRef Prefix,
                  SubtrieSlotValue Slot) override {
    OS << "- index=";
    for (size_t Pad : {10, 100, 1000})
      if (I < Pad && Subtrie.getNumSlots() >= Pad)
        OS << "0";
    OS << I << " ";
    if (Slot.isSubtrie()) {
      OS << "addr=" << (void *)Slot.asSubtrie();
      OS << " subtrie=";
      printPrefix(OS, Prefix);
      OS << "\n";
      return Error::success();
    }
    HashMappedTrieHandle::RecordData Record = Trie.getRecord(Slot);
    OS << "addr=" << (void *)Record.getFileOffset().get();
    OS << " content=";
    Subtrie.printHash(OS, Record.Proxy.Hash);
    OS << "\n";
    Records.push_back(Slot.asData());
    return Error::success();
  }

private:
  raw_ostream &OS;
  function_ref<void(ArrayRef<char>)> PrintRecordData;
  SmallVector<int64_t> Records;
};

// TrieVerifier that adds additional verification on top of the basic visitor.
class TrieVerifier : public TrieVisitor {
public:
  TrieVerifier(
      HashMappedTrieHandle Trie,
      function_ref<Error(FileOffset, OnDiskHashMappedTrie::ConstValueProxy)>
          RecordVerifier)
      : TrieVisitor(Trie), RecordVerifier(RecordVerifier) {}

private:
  Error visitSubTrie(StringRef Prefix, SubtrieHandle SubTrie) final {
    return Error::success();
  }

  Error visitSlot(unsigned I, SubtrieHandle Subtrie, StringRef Prefix,
                  SubtrieSlotValue Slot) final {
    if (RecordVerifier && Slot.isData()) {
      if (!isAligned(MappedFileRegionBumpPtr::getAlign(), Slot.asData()))
        return createInvalidTrieError(Slot.asData(), "mis-aligned data entry");

      HashMappedTrieHandle::RecordData Record =
          Trie.getRecord(SubtrieSlotValue::getDataOffset(Slot.asData()));
      return RecordVerifier(Slot.asDataFileOffset(),
                            OnDiskHashMappedTrie::ConstValueProxy{
                                Record.Proxy.Hash, Record.Proxy.Data});
    }
    return Error::success();
  }

  function_ref<Error(FileOffset, OnDiskHashMappedTrie::ConstValueProxy)>
      RecordVerifier;
};
} // namespace

Error TrieVisitor::visit() {
  auto Root = Trie.getRoot();
  if (!Root)
    return Error::success();

  if (auto Err = validateSubTrie(Root, /*IsRoot=*/true))
    return Err;

  SmallVector<SubtrieHandle> Subs;
  SmallVector<std::string> Prefixes;
  const size_t NumSlots = Root.getNumSlots();
  for (size_t I = 0, E = NumSlots; I != E; ++I) {
    SubtrieSlotValue Slot = Root.load(I);
    if (!Slot)
      continue;
    uint64_t Offset = Slot.isSubtrie() ? Slot.asSubtrie() : Slot.asData();
    if (Offset >= (uint64_t)Trie.getRegion().size())
      return createInvalidTrieError(Offset, "slot points out of bound");
    std::string SubtriePrefix;
    appendIndexBits(SubtriePrefix, I, NumSlots);
    if (Slot.isSubtrie()) {
      SubtrieHandle S(Trie.getRegion(), Slot);
      Subs.push_back(S);
      Prefixes.push_back(SubtriePrefix);
    }
    if (auto Err = visitSlot(I, Root, SubtriePrefix, Slot))
      return Err;
  }

  for (size_t I = 0, E = Subs.size(); I != E; ++I) {
    Threads.async(
        [&](unsigned Idx) {
          // Don't run if there is an error already.
          if (tooManyErrors())
            return;
          if (auto Err = traverseTrieNode(Subs[Idx], Prefixes[Idx]))
            addError(std::move(Err));
        },
        I);
  }

  Threads.wait();
  if (Err)
    return std::move(*Err);
  return Error::success();
}

Error TrieVisitor::validateSubTrie(SubtrieHandle Node, bool IsRoot) {
  char *Addr = reinterpret_cast<char *>(&Node.getHeader());
  const int64_t Offset = Node.getFileOffset().get();
  if (Addr + Node.getSize() >=
      Trie.getRegion().data() + Trie.getRegion().size())
    return createInvalidTrieError(Offset, "subtrie node spans out of bound");

  if (Node.getStartBit() + Node.getNumBits() > Trie.getNumHashBits())
    return createInvalidTrieError(Offset,
                                  "subtrie represents too many hash bits");

  if (IsRoot) {
    if (Node.getStartBit() != 0)
      return createInvalidTrieError(Offset,
                                    "root node doesn't start at 0 index");

    return Error::success();
  }

  if (Node.getNumBits() > Trie.getNumSubtrieBits())
    return createInvalidTrieError(Offset, "subtrie has wrong number of slots");

  return Error::success();
}

Error TrieVisitor::traverseTrieNode(SubtrieHandle Node, StringRef Prefix) {
  if (auto Err = validateSubTrie(Node, /*IsRoot=*/false))
    return Err;

  if (auto Err = visitSubTrie(Prefix, Node))
    return Err;

  SmallVector<SubtrieHandle> Subs;
  SmallVector<std::string> Prefixes;
  const size_t NumSlots = Node.getNumSlots();
  for (size_t I = 0, E = NumSlots; I != E; ++I) {
    SubtrieSlotValue Slot = Node.load(I);
    if (!Slot)
      continue;
    uint64_t Offset = Slot.isSubtrie() ? Slot.asSubtrie() : Slot.asData();
    if (Offset >= (uint64_t)Trie.getRegion().size())
      return createInvalidTrieError(Offset, "slot points out of bound");
    std::string SubtriePrefix = Prefix.str();
    appendIndexBits(SubtriePrefix, I, NumSlots);
    if (Slot.isSubtrie()) {
      SubtrieHandle S(Trie.getRegion(), Slot);
      Subs.push_back(S);
      Prefixes.push_back(SubtriePrefix);
    }
    if (auto Err = visitSlot(I, Node, SubtriePrefix, Slot))
      return Err;
  }
  for (size_t I = 0, E = Subs.size(); I != E; ++I)
    if (auto Err = traverseTrieNode(Subs[I], Prefixes[I]))
      return Err;

  return Error::success();
}

void HashMappedTrieHandle::print(
    raw_ostream &OS, function_ref<void(ArrayRef<char>)> PrintRecordData) const {
  OS << "hash-num-bits=" << getNumHashBits()
     << " hash-size=" << getNumHashBytes()
     << " record-data-size=" << getRecordDataSize() << "\n";

  TriePrinter Printer(*this, OS, PrintRecordData);
  if (auto Err = Printer.visit())
    OS << "error: " << toString(std::move(Err)) << "\n";

  if (auto Err = Printer.printRecords())
    OS << "error: " << toString(std::move(Err)) << "\n";

  return;
}

Error HashMappedTrieHandle::validate(
    function_ref<Error(FileOffset, OnDiskHashMappedTrie::ConstValueProxy)>
        RecordVerifier) const {
  // Use the base TrieVisitor to identify the errors inside trie first.
  TrieVisitor BasicVerifier(*this);
  if (auto Err = BasicVerifier.visit())
    return Err;

  // If the trie data structure is sound, do a second pass to verify data and
  // verifier function can assume the index is correct. However, there can be
  // newly added bad entries that can still produce error.
  TrieVerifier Verifier(*this, RecordVerifier);
  return Verifier.visit();
}

//===----------------------------------------------------------------------===//
// DataAllocator data structures.
//===----------------------------------------------------------------------===//

namespace {
/// DataAllocator table layout:
/// - [8-bytes: Generic table header]
/// - 8-bytes: AllocatorOffset (reserved for implementing free lists)
/// - 8-bytes: Size for user data header
/// - <user data buffer>
///
/// Record layout:
/// - <data>
class DataAllocatorHandle {
public:
  static constexpr TableHandle::TableKind Kind =
      TableHandle::TableKind::DataAllocator;

  struct Header {
    TableHandle::Header GenericHeader;
    std::atomic<int64_t> AllocatorOffset;
    const uint64_t UserHeaderSize;
  };

  operator TableHandle() const {
    if (!H)
      return TableHandle();
    return TableHandle(*Region, H->GenericHeader);
  }

  Expected<MutableArrayRef<char>> allocate(MappedFileRegionBumpPtr &Alloc,
                                           size_t DataSize) {
    assert(&Alloc.getRegion() == Region);
    auto Ptr = Alloc.allocate(DataSize);
    if (LLVM_UNLIKELY(!Ptr))
      return Ptr.takeError();
    return MutableArrayRef(*Ptr, DataSize);
  }

  explicit operator bool() const { return H; }
  const Header &getHeader() const { return *H; }
  MappedFileRegion &getRegion() const { return *Region; }

  MutableArrayRef<uint8_t> getUserHeader() {
    return MutableArrayRef(reinterpret_cast<uint8_t *>(H + 1),
                           H->UserHeaderSize);
  }

  static Expected<DataAllocatorHandle> create(MappedFileRegionBumpPtr &Alloc,
                                              StringRef Name,
                                              uint32_t UserHeaderSize);

  DataAllocatorHandle() = default;
  DataAllocatorHandle(MappedFileRegion &Region, Header &H)
      : Region(&Region), H(&H) {}
  DataAllocatorHandle(MappedFileRegion &Region, intptr_t HeaderOffset)
      : DataAllocatorHandle(
            Region, *reinterpret_cast<Header *>(Region.data() + HeaderOffset)) {
  }

private:
  MappedFileRegion *Region = nullptr;
  Header *H = nullptr;
};

} // end anonymous namespace

struct OnDiskDataAllocator::ImplType {
  DatabaseFile File;
  DataAllocatorHandle Store;
};

Expected<DataAllocatorHandle>
DataAllocatorHandle::create(MappedFileRegionBumpPtr &Alloc, StringRef Name,
                            uint32_t UserHeaderSize) {
  // Allocate.
  auto Offset =
      Alloc.allocateOffset(sizeof(Header) + UserHeaderSize + Name.size() + 1);
  if (LLVM_UNLIKELY(!Offset))
    return Offset.takeError();

  // Construct the header and the name.
  assert(Name.size() <= UINT16_MAX && "Expected smaller table name");
  auto *H = new (Alloc.getRegion().data() + *Offset)
      Header{{TableHandle::TableKind::DataAllocator, (uint16_t)Name.size(),
              (int32_t)(sizeof(Header) + UserHeaderSize)},
             /*AllocatorOffset=*/{0},
             /*UserHeaderSize=*/UserHeaderSize};
  memset(H + 1, 0, UserHeaderSize);
  char *NameStorage = reinterpret_cast<char *>(H + 1) + UserHeaderSize;
  llvm::copy(Name, NameStorage);
  NameStorage[Name.size()] = 0;
  return DataAllocatorHandle(Alloc.getRegion(), *H);
}

Expected<OnDiskDataAllocator> OnDiskDataAllocator::create(
    const Twine &PathTwine, const Twine &TableNameTwine, uint64_t MaxFileSize,
    std::optional<uint64_t> NewFileInitialSize, uint32_t UserHeaderSize,
    function_ref<void(void *)> UserHeaderInit) {
  assert(!UserHeaderSize || UserHeaderInit);
  SmallString<128> PathStorage;
  StringRef Path = PathTwine.toStringRef(PathStorage);
  SmallString<128> TableNameStorage;
  StringRef TableName = TableNameTwine.toStringRef(TableNameStorage);

  // Constructor for if the file doesn't exist.
  auto NewDBConstructor = [&](DatabaseFile &DB) -> Error {
    auto Store =
        DataAllocatorHandle::create(DB.getAlloc(), TableName, UserHeaderSize);
    if (LLVM_UNLIKELY(!Store))
      return Store.takeError();

    DB.addTable(*Store);
    if (UserHeaderSize)
      UserHeaderInit(Store->getUserHeader().data());
    return Error::success();
  };

  // Get or create the file.
  Expected<DatabaseFile> File =
      DatabaseFile::create(Path, MaxFileSize, NewDBConstructor);
  if (!File)
    return File.takeError();

  // Find the table and validate it.
  //
  // TODO: Add support for creating/adding a table to an existing file.
  std::optional<TableHandle> Table = File->findTable(TableName);
  if (!Table)
    return createTableConfigError(std::errc::argument_out_of_domain, Path,
                                  TableName, "table not found");
  if (Error E = checkTable("table kind", (size_t)DataAllocatorHandle::Kind,
                           (size_t)Table->getHeader().Kind, Path, TableName))
    return std::move(E);
  auto Store = Table->cast<DataAllocatorHandle>();
  assert(Store && "Already checked the kind");

  // Success.
  OnDiskDataAllocator::ImplType Impl{DatabaseFile(std::move(*File)), Store};
  return OnDiskDataAllocator(std::make_unique<ImplType>(std::move(Impl)));
}

Expected<OnDiskDataAllocator::pointer>
OnDiskDataAllocator::allocate(size_t Size) {
  auto Data = Impl->Store.allocate(Impl->File.getAlloc(), Size);
  if (LLVM_UNLIKELY(!Data))
    return Data.takeError();

  return pointer(FileOffset(Data->data() - Impl->Store.getRegion().data()),
                 *Data);
}

const char *OnDiskDataAllocator::beginData(FileOffset Offset) const {
  assert(Offset);
  assert(Impl);
  assert(Offset.get() < (int64_t)Impl->File.getAlloc().size());
  return Impl->File.getRegion().data() + Offset.get();
}

MutableArrayRef<uint8_t> OnDiskDataAllocator::getUserHeader() {
  return Impl->Store.getUserHeader();
}

size_t OnDiskDataAllocator::size() const { return Impl->File.size(); }
size_t OnDiskDataAllocator::capacity() const {
  return Impl->File.getRegion().size();
}

OnDiskDataAllocator::OnDiskDataAllocator(std::unique_ptr<ImplType> Impl)
    : Impl(std::move(Impl)) {}

#else // !LLVM_ENABLE_ONDISK_CAS

struct OnDiskHashMappedTrie::ImplType {};

Expected<OnDiskHashMappedTrie>
OnDiskHashMappedTrie::create(const Twine &PathTwine, const Twine &TrieNameTwine,
                             size_t NumHashBits, uint64_t DataSize,
                             uint64_t MaxFileSize,
                             std::optional<uint64_t> NewFileInitialSize,
                             std::optional<size_t> NewTableNumRootBits,
                             std::optional<size_t> NewTableNumSubtrieBits) {
  report_fatal_error("not supported");
}

Expected<OnDiskHashMappedTrie::pointer>
OnDiskHashMappedTrie::insertLazy(ArrayRef<uint8_t> Hash,
                                 LazyInsertOnConstructCB OnConstruct,
                                 LazyInsertOnLeakCB OnLeak) {
  report_fatal_error("not supported");
}

OnDiskHashMappedTrie::const_pointer
OnDiskHashMappedTrie::recoverFromFileOffset(FileOffset Offset) const {
  report_fatal_error("not supported");
}

OnDiskHashMappedTrie::const_pointer
OnDiskHashMappedTrie::find(ArrayRef<uint8_t> Hash) const {
  report_fatal_error("not supported");
}

void OnDiskHashMappedTrie::print(
    raw_ostream &OS, function_ref<void(ArrayRef<char>)> PrintRecordData) const {
  report_fatal_error("not supported");
}

size_t OnDiskHashMappedTrie::size() const {
  report_fatal_error("not supported");
}

size_t OnDiskHashMappedTrie::capacity() const {
  report_fatal_error("not supported");
}

size_t OnDiskDataAllocator::capacity() const {
  report_fatal_error("not supported");
}

struct OnDiskDataAllocator::ImplType {};

Expected<OnDiskDataAllocator> OnDiskDataAllocator::create(
    const Twine &Path, const Twine &TableName, uint64_t MaxFileSize,
    std::optional<uint64_t> NewFileInitialSize, uint32_t UserHeaderSize,
    function_ref<void(void *)> UserHeaderInit) {
  report_fatal_error("not supported");
}

Expected<OnDiskDataAllocator::pointer>
OnDiskDataAllocator::allocate(size_t Size) {
  report_fatal_error("not supported");
}

const char *OnDiskDataAllocator::beginData(FileOffset Offset) const {
  report_fatal_error("not supported");
}

MutableArrayRef<uint8_t> OnDiskDataAllocator::getUserHeader() {
  report_fatal_error("not supported");
}

size_t OnDiskDataAllocator::size() const {
  report_fatal_error("not supported");
}

#endif // LLVM_ENABLE_ONDISK_CAS

OnDiskHashMappedTrie::OnDiskHashMappedTrie(std::unique_ptr<ImplType> Impl)
    : Impl(std::move(Impl)) {}
OnDiskHashMappedTrie::OnDiskHashMappedTrie(OnDiskHashMappedTrie &&RHS) =
    default;
OnDiskHashMappedTrie &
OnDiskHashMappedTrie::operator=(OnDiskHashMappedTrie &&RHS) = default;
OnDiskHashMappedTrie::~OnDiskHashMappedTrie() = default;

OnDiskDataAllocator::OnDiskDataAllocator(OnDiskDataAllocator &&RHS) = default;
OnDiskDataAllocator &
OnDiskDataAllocator::operator=(OnDiskDataAllocator &&RHS) = default;
OnDiskDataAllocator::~OnDiskDataAllocator() = default;
