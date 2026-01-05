//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file Implements OnDiskTrieRawHashMap.
///
//===----------------------------------------------------------------------===//

#include "llvm/CAS/OnDiskTrieRawHashMap.h"
#include "DatabaseFile.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TrieHashIndexGenerator.h"
#include "llvm/CAS/MappedFileRegionArena.h"
#include "llvm/CAS/OnDiskCASLogger.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::cas;
using namespace llvm::cas::ondisk;

#if LLVM_ENABLE_ONDISK_CAS

//===----------------------------------------------------------------------===//
// TrieRawHashMap data structures.
//===----------------------------------------------------------------------===//

namespace {

class SubtrieHandle;
class TrieRawHashMapHandle;
class TrieVisitor;

/// A value stored in the slots inside a SubTrie. A stored value can either be a
/// subtrie (encoded after negation) which is the file offset to another
/// subtrie, or it can be a fileset to a DataRecord.
class SubtrieSlotValue {
public:
  explicit operator bool() const { return !isEmpty(); }
  bool isEmpty() const { return !Offset; }
  bool isData() const { return Offset > 0; }
  bool isSubtrie() const { return Offset < 0; }
  uint64_t asData() const {
    assert(isData());
    return Offset;
  }
  uint64_t asSubtrie() const {
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
    return sizeof(int64_t) * (1ull << NumBits);
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
    SubtrieSlotValue SaveExpected(Expected);
    bool Result = Slots[I].compare_exchange_strong(Expected.Offset, New.Offset);
    if (Logger)
      Logger->log_compare_exchange_strong(Region->data(), getOffset().Offset, I,
                                          SaveExpected.Offset, New.Offset,
                                          Expected.Offset);
    return Result;
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
                               MappedFileRegionArena &Alloc,
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

  static Expected<SubtrieHandle> create(MappedFileRegionArena &Alloc,
                                        uint32_t StartBit, uint32_t NumBits,
                                        OnDiskCASLogger *Logger);

  static SubtrieHandle getFromFileOffset(MappedFileRegion &Region,
                                         FileOffset Offset,
                                         OnDiskCASLogger *Logger) {
    return SubtrieHandle(Region, SubtrieSlotValue::getSubtrieOffset(Offset),
                         Logger);
  }

  SubtrieHandle() = default;
  SubtrieHandle(MappedFileRegion &Region, Header &H, OnDiskCASLogger *Logger)
      : Region(&Region), H(&H), Slots(getSlots(H)), Logger(Logger) {}
  SubtrieHandle(MappedFileRegion &Region, SubtrieSlotValue Offset,
                OnDiskCASLogger *Logger)
      : SubtrieHandle(
            Region,
            *reinterpret_cast<Header *>(Region.data() + Offset.asSubtrie()),
            Logger) {}

private:
  MappedFileRegion *Region = nullptr;
  Header *H = nullptr;
  MutableArrayRef<SlotT> Slots;
  OnDiskCASLogger *Logger = nullptr;

  static MutableArrayRef<SlotT> getSlots(Header &H) {
    return MutableArrayRef(reinterpret_cast<SlotT *>(&H + 1),
                           1ull << H.NumBits);
  }
};

/// Handle for a TrieRawHashMap table.
///
/// TrieRawHashMap table layout:
/// - [8-bytes: Generic table header]
/// - 1-byte:  NumSubtrieBits
/// - 1-byte:  Flags (not used yet)
/// - 2-bytes: NumHashBits
/// - 4-bytes: RecordDataSize (in bytes)
/// - 8-bytes: RootTrieOffset
/// - 8-bytes: AllocatorOffset (reserved for implementing free lists)
/// - <name> '\0'
///
/// Record layout:
/// - <hash>
/// - <data>
class TrieRawHashMapHandle {
public:
  static constexpr TableHandle::TableKind Kind =
      TableHandle::TableKind::TrieRawHashMap;

  struct Header {
    TableHandle::Header GenericHeader;
    uint8_t NumSubtrieBits;
    uint8_t Flags; ///< None used yet.
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
    OnDiskTrieRawHashMap::ValueProxy Proxy;
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
  Expected<RecordData> createRecord(MappedFileRegionArena &Alloc,
                                    ArrayRef<uint8_t> Hash);

  explicit operator bool() const { return H; }
  const Header &getHeader() const { return *H; }
  SubtrieHandle getRoot() const;
  Expected<SubtrieHandle> getOrCreateRoot(MappedFileRegionArena &Alloc);
  MappedFileRegion &getRegion() const { return *Region; }

  size_t getFlags() const { return H->Flags; }
  size_t getNumSubtrieBits() const { return H->NumSubtrieBits; }
  size_t getNumHashBits() const { return H->NumHashBits; }
  size_t getNumHashBytes() const { return getNumHashBytes(H->NumHashBits); }
  size_t getRecordDataSize() const { return H->RecordDataSize; }
  size_t getRecordSize() const {
    return getRecordSize(H->RecordDataSize, H->NumHashBits);
  }

  TrieHashIndexGenerator getIndexGen(SubtrieHandle Root,
                                     ArrayRef<uint8_t> Hash) {
    assert(Root.getStartBit() == 0);
    assert(getNumHashBytes() == Hash.size());
    assert(getNumHashBits() == Hash.size() * 8);
    return TrieHashIndexGenerator{Root.getNumBits(), getNumSubtrieBits(), Hash};
  }

  static Expected<TrieRawHashMapHandle>
  create(MappedFileRegionArena &Alloc, StringRef Name,
         std::optional<uint64_t> NumRootBits, uint64_t NumSubtrieBits,
         uint64_t NumHashBits, uint64_t RecordDataSize,
         std::shared_ptr<OnDiskCASLogger> Logger);

  void
  print(raw_ostream &OS,
        function_ref<void(ArrayRef<char>)> PrintRecordData = nullptr) const;

  Error validate(
      function_ref<Error(FileOffset, OnDiskTrieRawHashMap::ConstValueProxy)>
          RecordVerifier) const;
  TrieRawHashMapHandle() = default;
  TrieRawHashMapHandle(MappedFileRegion &Region, Header &H,
                       std::shared_ptr<OnDiskCASLogger> Logger = nullptr)
      : Region(&Region), H(&H), Logger(std::move(Logger)) {}
  TrieRawHashMapHandle(MappedFileRegion &Region, intptr_t HeaderOffset,
                       std::shared_ptr<OnDiskCASLogger> Logger = nullptr)
      : TrieRawHashMapHandle(
            Region, *reinterpret_cast<Header *>(Region.data() + HeaderOffset),
            std::move(Logger)) {}

  OnDiskCASLogger *getLogger() const { return Logger.get(); }
  void setLogger(std::shared_ptr<OnDiskCASLogger> Logger) {
    this->Logger = std::move(Logger);
  }

private:
  MappedFileRegion *Region = nullptr;
  Header *H = nullptr;
  std::shared_ptr<OnDiskCASLogger> Logger;
};

} // end anonymous namespace

struct OnDiskTrieRawHashMap::ImplType {
  DatabaseFile File;
  TrieRawHashMapHandle Trie;
};

Expected<SubtrieHandle> SubtrieHandle::create(MappedFileRegionArena &Alloc,
                                              uint32_t StartBit,
                                              uint32_t NumBits,
                                              OnDiskCASLogger *Logger) {
  assert(StartBit <= TrieRawHashMapHandle::MaxNumHashBits);
  assert(NumBits <= UINT8_MAX);
  assert(NumBits <= TrieRawHashMapHandle::MaxNumRootBits);

  auto Mem = Alloc.allocate(getSize(NumBits));
  if (LLVM_UNLIKELY(!Mem))
    return Mem.takeError();
  auto *H =
      new (*Mem) SubtrieHandle::Header{(uint16_t)StartBit, (uint8_t)NumBits,
                                       /*ZeroPad1B=*/0, /*ZeroPad4B=*/0};
  SubtrieHandle S(Alloc.getRegion(), *H, Logger);
  for (auto I = S.Slots.begin(), E = S.Slots.end(); I != E; ++I)
    new (I) SlotT(0);

  if (Logger)
    Logger->log_SubtrieHandle_create(Alloc.data(), S.getOffset().Offset,
                                     StartBit, NumBits);
  return S;
}

SubtrieHandle TrieRawHashMapHandle::getRoot() const {
  if (int64_t Root = H->RootTrieOffset)
    return SubtrieHandle(getRegion(), SubtrieSlotValue::getSubtrieOffset(Root),
                         Logger.get());
  return SubtrieHandle();
}

Expected<SubtrieHandle>
TrieRawHashMapHandle::getOrCreateRoot(MappedFileRegionArena &Alloc) {
  assert(&Alloc.getRegion() == &getRegion());
  if (SubtrieHandle Root = getRoot())
    return Root;

  int64_t Race = 0;
  auto LazyRoot =
      SubtrieHandle::create(Alloc, 0, H->NumSubtrieBits, Logger.get());
  if (LLVM_UNLIKELY(!LazyRoot))
    return LazyRoot.takeError();

  if (H->RootTrieOffset.compare_exchange_strong(
          Race, LazyRoot->getOffset().asSubtrie()),
      Logger.get())
    return *LazyRoot;

  // There was a race. Return the other root.
  //
  // TODO: Avoid leaking the lazy root by storing it in an allocator.
  return SubtrieHandle(getRegion(), SubtrieSlotValue::getSubtrieOffset(Race),
                       Logger.get());
}

Expected<TrieRawHashMapHandle>
TrieRawHashMapHandle::create(MappedFileRegionArena &Alloc, StringRef Name,
                             std::optional<uint64_t> NumRootBits,
                             uint64_t NumSubtrieBits, uint64_t NumHashBits,
                             uint64_t RecordDataSize,
                             std::shared_ptr<OnDiskCASLogger> Logger) {
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
      Header{{TableHandle::TableKind::TrieRawHashMap, (uint16_t)Name.size(),
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
  TrieRawHashMapHandle Trie(Alloc.getRegion(), *H, Logger);
  auto Sub = SubtrieHandle::create(Alloc, 0, *NumRootBits, Logger.get());
  if (LLVM_UNLIKELY(!Sub))
    return Sub.takeError();
  if (NumRootBits)
    H->RootTrieOffset = Sub->getOffset().asSubtrie();
  return Trie;
}

TrieRawHashMapHandle::RecordData
TrieRawHashMapHandle::getRecord(SubtrieSlotValue Offset) {
  char *Begin = Region->data() + Offset.asData();
  OnDiskTrieRawHashMap::ValueProxy Proxy;
  Proxy.Data = MutableArrayRef(Begin, getRecordDataSize());
  Proxy.Hash = ArrayRef(reinterpret_cast<const uint8_t *>(Proxy.Data.end()),
                        getNumHashBytes());
  return RecordData{Proxy, Offset};
}

Expected<TrieRawHashMapHandle::RecordData>
TrieRawHashMapHandle::createRecord(MappedFileRegionArena &Alloc,
                                   ArrayRef<uint8_t> Hash) {
  assert(&Alloc.getRegion() == Region);
  assert(Hash.size() == getNumHashBytes());
  auto Offset = Alloc.allocateOffset(getRecordSize());
  if (LLVM_UNLIKELY(!Offset))
    return Offset.takeError();

  RecordData Record = getRecord(SubtrieSlotValue::getDataOffset(*Offset));
  llvm::copy(Hash, const_cast<uint8_t *>(Record.Proxy.Hash.begin()));

  if (Logger)
    Logger->log_HashMappedTrieHandle_createRecord(
        Alloc.data(), Record.Offset.getRawOffset(), Hash);

  return Record;
}

Expected<OnDiskTrieRawHashMap::ConstOnDiskPtr>
OnDiskTrieRawHashMap::recoverFromFileOffset(FileOffset Offset) const {
  // Check alignment.
  if (!isAligned(MappedFileRegionArena::getAlign(), Offset.get()))
    return createStringError(make_error_code(std::errc::protocol_error),
                             "unaligned file offset at 0x" +
                                 utohexstr(Offset.get(), /*LowerCase=*/true));

  // Check bounds.
  //
  // Note: There's no potential overflow when using \c uint64_t because Offset
  // is in valid offset range and the record size is in \c [0,UINT32_MAX].
  if (!validOffset(Offset) ||
      Offset.get() + Impl->Trie.getRecordSize() > Impl->File.getAlloc().size())
    return createStringError(make_error_code(std::errc::protocol_error),
                             "file offset too large: 0x" +
                                 utohexstr(Offset.get(), /*LowerCase=*/true));

  // Looks okay...
  TrieRawHashMapHandle::RecordData D =
      Impl->Trie.getRecord(SubtrieSlotValue::getDataOffset(Offset));
  return ConstOnDiskPtr(D.Proxy, D.getFileOffset());
}

OnDiskTrieRawHashMap::ConstOnDiskPtr
OnDiskTrieRawHashMap::find(ArrayRef<uint8_t> Hash) const {
  TrieRawHashMapHandle Trie = Impl->Trie;
  assert(Hash.size() == Trie.getNumHashBytes() && "Invalid hash");

  SubtrieHandle S = Trie.getRoot();
  if (!S)
    return ConstOnDiskPtr();

  TrieHashIndexGenerator IndexGen = Trie.getIndexGen(S, Hash);
  size_t Index = IndexGen.next();
  for (;;) {
    // Try to set the content.
    SubtrieSlotValue V = S.load(Index);
    if (!V)
      return ConstOnDiskPtr();

    // Check for an exact match.
    if (V.isData()) {
      TrieRawHashMapHandle::RecordData D = Trie.getRecord(V);
      return D.Proxy.Hash == Hash ? ConstOnDiskPtr(D.Proxy, D.getFileOffset())
                                  : ConstOnDiskPtr();
    }

    Index = IndexGen.next();
    S = SubtrieHandle(Trie.getRegion(), V, Trie.getLogger());
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

Expected<OnDiskTrieRawHashMap::OnDiskPtr>
OnDiskTrieRawHashMap::insertLazy(ArrayRef<uint8_t> Hash,
                                 LazyInsertOnConstructCB OnConstruct,
                                 LazyInsertOnLeakCB OnLeak) {
  TrieRawHashMapHandle Trie = Impl->Trie;
  assert(Hash.size() == Trie.getNumHashBytes() && "Invalid hash");

  MappedFileRegionArena &Alloc = Impl->File.getAlloc();
  std::optional<SubtrieHandle> S;
  auto Err = Trie.getOrCreateRoot(Alloc).moveInto(S);
  if (LLVM_UNLIKELY(Err))
    return std::move(Err);

  TrieHashIndexGenerator IndexGen = Trie.getIndexGen(*S, Hash);
  size_t Index = IndexGen.next();

  // Walk through the hash bytes and insert into correct trie position.
  std::optional<TrieRawHashMapHandle::RecordData> NewRecord;
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
        return OnDiskPtr(NewRecord->Proxy,
                         NewRecord->Offset.asDataFileOffset());

      // Race means that Existing is no longer empty; fall through...
    }

    if (Existing.isSubtrie()) {
      S = SubtrieHandle(Trie.getRegion(), Existing, Trie.getLogger());
      Index = IndexGen.next();
      continue;
    }

    // Check for an exact match.
    TrieRawHashMapHandle::RecordData ExistingRecord = Trie.getRecord(Existing);
    if (ExistingRecord.Proxy.Hash == Hash) {
      if (NewRecord && OnLeak)
        OnLeak(NewRecord->Offset.asDataFileOffset(), NewRecord->Proxy,
               ExistingRecord.Offset.asDataFileOffset(), ExistingRecord.Proxy);
      return OnDiskPtr(ExistingRecord.Proxy,
                       ExistingRecord.Offset.asDataFileOffset());
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
                                            MappedFileRegionArena &Alloc,
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
                                     NumSubtrieBits, Logger)
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
  return SubtrieHandle(Alloc.getRegion(), V, Logger);
}

void OnDiskTrieRawHashMap::print(
    raw_ostream &OS, function_ref<void(ArrayRef<char>)> PrintRecordData) const {
  Impl->Trie.print(OS, PrintRecordData);
}

Error OnDiskTrieRawHashMap::validate(
    function_ref<Error(FileOffset, ConstValueProxy)> RecordVerifier) const {
  return Impl->Trie.validate(RecordVerifier);
}

// Helper function that prints hexdigit and have a sub-byte starting position.
static void printHexDigits(raw_ostream &OS, ArrayRef<uint8_t> Bytes,
                           size_t StartBit, size_t NumBits) {
  assert(StartBit % 4 == 0);
  assert(NumBits % 4 == 0);
  for (size_t I = StartBit, E = StartBit + NumBits; I != E; I += 4) {
    uint8_t HexPair = Bytes[I / 8];
    uint8_t HexDigit = I % 8 == 0 ? HexPair >> 4 : HexPair & 0xf;
    OS << hexdigit(HexDigit, /*LowerCase=*/true);
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
    OS << hexdigit(Digit, /*LowerCase=*/true);
    Prefix = Prefix.drop_front(4);
  }
  if (!Prefix.empty())
    OS << "[" << Prefix << "]";
}

LLVM_DUMP_METHOD void OnDiskTrieRawHashMap::dump() const { print(dbgs()); }

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

size_t OnDiskTrieRawHashMap::size() const { return Impl->File.size(); }
size_t OnDiskTrieRawHashMap::capacity() const {
  return Impl->File.getRegion().size();
}

Expected<OnDiskTrieRawHashMap>
OnDiskTrieRawHashMap::create(const Twine &PathTwine, const Twine &TrieNameTwine,
                             size_t NumHashBits, uint64_t DataSize,
                             uint64_t MaxFileSize,
                             std::optional<uint64_t> NewFileInitialSize,
                             std::shared_ptr<OnDiskCASLogger> Logger,
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
                    "root bits", TrieRawHashMapHandle::MaxNumRootBits,
                    NewTableNumRootBits, DefaultNumRootBits, Path, TrieName)
                    .moveInto(NumRootBits))
    return std::move(E);

  size_t NumSubtrieBits;
  if (Error E = checkParameter("subtrie bits",
                               TrieRawHashMapHandle::MaxNumSubtrieBits,
                               NewTableNumSubtrieBits, DefaultNumSubtrieBits,
                               Path, TrieName)
                    .moveInto(NumSubtrieBits))
    return std::move(E);

  size_t NumHashBytes = NumHashBits >> 3;
  if (Error E =
          checkParameter("hash size", TrieRawHashMapHandle::MaxNumHashBits,
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
    auto Trie = TrieRawHashMapHandle::create(DB.getAlloc(), TrieName,
                                             NumRootBits, NumSubtrieBits,
                                             NumHashBits, DataSize, Logger);
    if (LLVM_UNLIKELY(!Trie))
      return Trie.takeError();

    return DB.addTable(*Trie);
  };

  // Get or create the file.
  Expected<DatabaseFile> File =
      DatabaseFile::create(Path, MaxFileSize, Logger, NewDBConstructor);
  if (!File)
    return File.takeError();

  // Find the trie and validate it.
  std::optional<TableHandle> Table = File->findTable(TrieName);
  if (!Table)
    return createTableConfigError(std::errc::argument_out_of_domain, Path,
                                  TrieName, "table not found");
  if (Error E = checkTable("table kind", (size_t)TrieRawHashMapHandle::Kind,
                           (size_t)Table->getHeader().Kind, Path, TrieName))
    return std::move(E);
  auto Trie = Table->cast<TrieRawHashMapHandle>();
  Trie.setLogger(Logger);
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
  OnDiskTrieRawHashMap::ImplType Impl{DatabaseFile(std::move(*File)), Trie};
  return OnDiskTrieRawHashMap(std::make_unique<ImplType>(std::move(Impl)));
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
/// A multi-threaded vistior to traverse the Trie.
///
/// TODO: add more sanity checks that isn't just plain data corruption. For
/// example, some ill-formed data can be constructed to form a cycle using
/// Sub-Tries and it can lead to inifinite loop when visiting (or inserting
/// data).
class TrieVisitor {
public:
  TrieVisitor(TrieRawHashMapHandle Trie, unsigned ThreadCount = 0,
              unsigned ErrorLimit = 50)
      : Trie(Trie), ErrorLimit(ErrorLimit),
        Threads(hardware_concurrency(ThreadCount)) {}
  virtual ~TrieVisitor() = default;
  Error visit();

private:
  // Virtual method to implement the action when visiting a sub-trie.
  virtual Error visitSubTrie(StringRef Prefix, SubtrieHandle SubTrie) {
    return Error::success();
  }

  // Virtual method to implement the action when visiting a slot in a trie node.
  virtual Error visitSlot(unsigned I, SubtrieHandle Subtrie, StringRef Prefix,
                          SubtrieSlotValue Slot) {
    return Error::success();
  }

protected:
  TrieRawHashMapHandle Trie;

private:
  Error traverseTrieNode(SubtrieHandle Node, StringRef Prefix);

  Error validateSubTrie(SubtrieHandle Node, bool IsRoot);

  // Helper function to capture errors when visiting the trie nodes.
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

/// A visitor that traverse and print the Trie.
class TriePrinter : public TrieVisitor {
public:
  TriePrinter(TrieRawHashMapHandle Trie, raw_ostream &OS,
              function_ref<void(ArrayRef<char>)> PrintRecordData)
      : TrieVisitor(Trie, /*ThreadCount=*/1), OS(OS),
        PrintRecordData(PrintRecordData) {}

  Error printRecords() {
    if (Records.empty())
      return Error::success();

    OS << "records\n";
    llvm::sort(Records);
    for (int64_t Offset : Records) {
      TrieRawHashMapHandle::RecordData Record =
          Trie.getRecord(SubtrieSlotValue::getDataOffset(Offset));
      if (auto Err = printRecord(Record))
        return Err;
    }
    return Error::success();
  }

  Error printRecord(TrieRawHashMapHandle::RecordData &Record) {
    OS << "- addr=" << (void *)Record.getFileOffset().get() << " ";
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
    TrieRawHashMapHandle::RecordData Record = Trie.getRecord(Slot);
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

/// TrieVerifier that adds additional verification on top of the basic visitor.
class TrieVerifier : public TrieVisitor {
public:
  TrieVerifier(
      TrieRawHashMapHandle Trie,
      function_ref<Error(FileOffset, OnDiskTrieRawHashMap::ConstValueProxy)>
          RecordVerifier)
      : TrieVisitor(Trie), RecordVerifier(RecordVerifier) {}

private:
  Error visitSubTrie(StringRef Prefix, SubtrieHandle SubTrie) final {
    return Error::success();
  }

  Error visitSlot(unsigned I, SubtrieHandle Subtrie, StringRef Prefix,
                  SubtrieSlotValue Slot) final {
    if (RecordVerifier && Slot.isData()) {
      if (!isAligned(MappedFileRegionArena::getAlign(), Slot.asData()))
        return createInvalidTrieError(Slot.asData(), "mis-aligned data entry");

      TrieRawHashMapHandle::RecordData Record =
          Trie.getRecord(SubtrieSlotValue::getDataOffset(Slot.asData()));
      return RecordVerifier(Slot.asDataFileOffset(),
                            OnDiskTrieRawHashMap::ConstValueProxy{
                                Record.Proxy.Hash, Record.Proxy.Data});
    }
    return Error::success();
  }

  function_ref<Error(FileOffset, OnDiskTrieRawHashMap::ConstValueProxy)>
      RecordVerifier;
};
} // namespace

Error TrieVisitor::visit() {
  auto Root = Trie.getRoot();
  if (!Root)
    return Error::success();

  if (auto Err = validateSubTrie(Root, /*IsRoot=*/true))
    return Err;

  if (auto Err = visitSubTrie("", Root))
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
      SubtrieHandle S(Trie.getRegion(), Slot, Trie.getLogger());
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

  if (!IsRoot &&
      Node.getStartBit() + Node.getNumBits() > Trie.getNumHashBits()) {
    return createInvalidTrieError(Offset,
                                  "subtrie represents too many hash bits");
  }

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
      SubtrieHandle S(Trie.getRegion(), Slot, Trie.getLogger());
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

void TrieRawHashMapHandle::print(
    raw_ostream &OS, function_ref<void(ArrayRef<char>)> PrintRecordData) const {
  OS << "hash-num-bits=" << getNumHashBits()
     << " hash-size=" << getNumHashBytes()
     << " record-data-size=" << getRecordDataSize() << "\n";

  TriePrinter Printer(*this, OS, PrintRecordData);
  if (auto Err = Printer.visit())
    OS << "error: " << toString(std::move(Err)) << "\n";

  if (auto Err = Printer.printRecords())
    OS << "error: " << toString(std::move(Err)) << "\n";
}

Error TrieRawHashMapHandle::validate(
    function_ref<Error(FileOffset, OnDiskTrieRawHashMap::ConstValueProxy)>
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

#else // !LLVM_ENABLE_ONDISK_CAS

struct OnDiskTrieRawHashMap::ImplType {};

Expected<OnDiskTrieRawHashMap>
OnDiskTrieRawHashMap::create(const Twine &PathTwine, const Twine &TrieNameTwine,
                             size_t NumHashBits, uint64_t DataSize,
                             uint64_t MaxFileSize,
                             std::optional<uint64_t> NewFileInitialSize,
                             std::shared_ptr<OnDiskCASLogger> Logger,
                             std::optional<size_t> NewTableNumRootBits,
                             std::optional<size_t> NewTableNumSubtrieBits) {
  return createStringError(make_error_code(std::errc::not_supported),
                           "OnDiskTrieRawHashMap is not supported");
}

Expected<OnDiskTrieRawHashMap::OnDiskPtr>
OnDiskTrieRawHashMap::insertLazy(ArrayRef<uint8_t> Hash,
                                 LazyInsertOnConstructCB OnConstruct,
                                 LazyInsertOnLeakCB OnLeak) {
  return createStringError(make_error_code(std::errc::not_supported),
                           "OnDiskTrieRawHashMap is not supported");
}

Expected<OnDiskTrieRawHashMap::ConstOnDiskPtr>
OnDiskTrieRawHashMap::recoverFromFileOffset(FileOffset Offset) const {
  return createStringError(make_error_code(std::errc::not_supported),
                           "OnDiskTrieRawHashMap is not supported");
}

OnDiskTrieRawHashMap::ConstOnDiskPtr
OnDiskTrieRawHashMap::find(ArrayRef<uint8_t> Hash) const {
  return ConstOnDiskPtr();
}

void OnDiskTrieRawHashMap::print(
    raw_ostream &OS, function_ref<void(ArrayRef<char>)> PrintRecordData) const {
}

Error OnDiskTrieRawHashMap::validate(
    function_ref<Error(FileOffset, OnDiskTrieRawHashMap::ConstValueProxy)>
        RecordVerifier) const {
  return createStringError(make_error_code(std::errc::not_supported),
                           "OnDiskTrieRawHashMap is not supported");
}

size_t OnDiskTrieRawHashMap::size() const { return 0; }
size_t OnDiskTrieRawHashMap::capacity() const { return 0; }

#endif // LLVM_ENABLE_ONDISK_CAS

OnDiskTrieRawHashMap::OnDiskTrieRawHashMap(std::unique_ptr<ImplType> Impl)
    : Impl(std::move(Impl)) {}
OnDiskTrieRawHashMap::OnDiskTrieRawHashMap(OnDiskTrieRawHashMap &&RHS) =
    default;
OnDiskTrieRawHashMap &
OnDiskTrieRawHashMap::operator=(OnDiskTrieRawHashMap &&RHS) = default;
OnDiskTrieRawHashMap::~OnDiskTrieRawHashMap() = default;
