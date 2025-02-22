//===- OnDiskGraphDB.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// On-disk CAS nodes database, independent of a particular hashing algorithm.
//
// Here's a top-level description of the current layout (could expose or make
// this configurable in the future).
//
// Files, each with a prefix set by \a FilePrefix:
//
// - db/<prefix>.index: a file for the "index" table, named by \a
//   IndexTableName and managed by \a HashMappedTrie. The contents are 8B
//   that are accessed atomically, describing the object kind and where/how
//   it's stored (including an optional file offset). See \a TrieRecord for
//   more details.
// - db/<prefix>.data: a file for the "data" table, named by \a
//   DataPoolTableName and managed by \a DataStore. New objects within
//   TrieRecord::MaxEmbeddedSize are inserted here as \a
//   TrieRecord::StorageKind::DataPool.
//     - db/<prefix>.<offset>.data: a file storing an object outside the main
//       "data" table, named by its offset into the "index" table, with the
//       format of \a TrieRecord::StorageKind::Standalone.
//     - db/<prefix>.<offset>.leaf: a file storing a leaf node outside the
//       main "data" table, named by its offset into the "index" table, with
//       the format of \a TrieRecord::StorageKind::StandaloneLeaf.
//     - db/<prefix>.<offset>.leaf+0: a file storing a leaf object outside the
//       main "data" table, named by its offset into the "index" table, with
//       the format of \a TrieRecord::StorageKind::StandaloneLeaf0.
//
// The "index", and "data" tables could be stored in a single file,
// (using a root record that points at the two types of stores), but splitting
// the files seems more convenient for now.
//
// ObjectID: this is a pointer to Trie record
//
// ObjectHandle: this is a pointer to Data record
//
// Eventually: consider creating a StringPool for strings instead of using
// RecordDataStore table.
// - Lookup by prefix tree
// - Store by suffix tree
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/OnDiskGraphDB.h"
#include "OnDiskCommon.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/CAS/OnDiskHashMappedTrie.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include <optional>

#if __has_include(<sys/mount.h>)
#include <sys/mount.h> // statfs
#endif

#define DEBUG_TYPE "on-disk-cas"

using namespace llvm;
using namespace llvm::cas;
using namespace llvm::cas::ondisk;

static constexpr StringLiteral IndexTableName = "llvm.cas.index";
static constexpr StringLiteral DataPoolTableName = "llvm.cas.data";

static constexpr StringLiteral IndexFile = "index";
static constexpr StringLiteral DataPoolFile = "data";

static constexpr StringLiteral FilePrefix = "v8.";
static constexpr StringLiteral FileSuffixData = ".data";
static constexpr StringLiteral FileSuffixLeaf = ".leaf";
static constexpr StringLiteral FileSuffixLeaf0 = ".leaf+0";

static Error createCorruptObjectError(ArrayRef<uint8_t> ID) {
  return createStringError(llvm::errc::invalid_argument,
                           "corrupt object '" + toHex(ID) + "'");
}

namespace {

/// Trie record data: 8B, atomic<uint64_t>
/// - 1-byte: StorageKind
/// - 7-bytes: DataStoreOffset (offset into referenced file)
class TrieRecord {
public:
  enum class StorageKind : uint8_t {
    /// Unknown object.
    Unknown = 0,

    /// vX.data: main pool, full DataStore record.
    DataPool = 1,

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

  static StringRef getStandaloneFileSuffix(StorageKind SK) {
    switch (SK) {
    default:
      llvm_unreachable("Expected standalone storage kind");
    case TrieRecord::StorageKind::Standalone:
      return FileSuffixData;
    case TrieRecord::StorageKind::StandaloneLeaf0:
      return FileSuffixLeaf0;
    case TrieRecord::StorageKind::StandaloneLeaf:
      return FileSuffixLeaf;
    }
  }

  enum Limits : int64_t {
    // Saves files bigger than 64KB standalone instead of embedding them.
    MaxEmbeddedSize = 64LL * 1024LL - 1,
  };

  struct Data {
    StorageKind SK = StorageKind::Unknown;
    FileOffset Offset;
  };

  static uint64_t pack(Data D) {
    assert(D.Offset.get() < (int64_t)(1ULL << 56));
    uint64_t Packed = uint64_t(D.SK) << 56 | D.Offset.get();
    assert(D.SK != StorageKind::Unknown || Packed == 0);
#ifndef NDEBUG
    Data RoundTrip = unpack(Packed);
    assert(D.SK == RoundTrip.SK);
    assert(D.Offset.get() == RoundTrip.Offset.get());
#endif
    return Packed;
  }

  static Data unpack(uint64_t Packed) {
    Data D;
    if (!Packed)
      return D;
    D.SK = (StorageKind)(Packed >> 56);
    D.Offset = FileOffset(Packed & (UINT64_MAX >> 8));
    return D;
  }

  TrieRecord() : Storage(0) {}

  Data load() const { return unpack(Storage); }
  bool compare_exchange_strong(Data &Existing, Data New);

private:
  std::atomic<uint64_t> Storage;
};

/// DataStore record data: 4B + size? + refs? + data + 0
/// - 4-bytes: Header
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
    RefKindShift = DataSizeShift + DataSizeBits,
    RefKindBits = 1,
  };
  static_assert(((UINT32_MAX << NumRefsBits) & (uint32_t)NumRefsFlags::Max) ==
                    0,
                "Not enough bits");
  static_assert(((UINT32_MAX << DataSizeBits) & (uint32_t)DataSizeFlags::Max) ==
                    0,
                "Not enough bits");
  static_assert(((UINT32_MAX << RefKindBits) & (uint32_t)RefKindFlags::Max) ==
                    0,
                "Not enough bits");

  struct LayoutFlags {
    NumRefsFlags NumRefs;
    DataSizeFlags DataSize;
    RefKindFlags RefKind;

    static uint64_t pack(LayoutFlags LF) {
      unsigned Packed = ((unsigned)LF.NumRefs << NumRefsShift) |
                        ((unsigned)LF.DataSize << DataSizeShift) |
                        ((unsigned)LF.RefKind << RefKindShift);
#ifndef NDEBUG
      LayoutFlags RoundTrip = unpack(Packed);
      assert(LF.NumRefs == RoundTrip.NumRefs);
      assert(LF.DataSize == RoundTrip.DataSize);
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
      LF.RefKind =
          (RefKindFlags)((Storage >> RefKindShift) & ((1U << RefKindBits) - 1));
      return LF;
    }
  };

  /// Header layout:
  /// - 1-byte:      LayoutFlags
  /// - 1-byte:      1B size field
  /// - {0,2}-bytes: 2B size field
  struct Header {
    using PackTy = uint32_t;
    PackTy Packed;

    static constexpr unsigned LayoutFlagsShift =
        (sizeof(PackTy) - 1) * CHAR_BIT;
  };

  struct Input {
    InternalRefArrayRef Refs;
    ArrayRef<char> Data;
  };

  LayoutFlags getLayoutFlags() const {
    return LayoutFlags::unpack(H->Packed >> Header::LayoutFlagsShift);
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
      return ArrayRef(reinterpret_cast<const InternalRef4B *>(BeginByte), Size);
    return ArrayRef(reinterpret_cast<const InternalRef *>(BeginByte), Size);
  }

  ArrayRef<char> getData() const {
    assert(H && "Expected valid handle");
    return ArrayRef(reinterpret_cast<const char *>(H) + getDataRelOffset(),
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

class StandaloneDataInMemory {
public:
  OnDiskContent getContent() const;

  /// FIXME: Should be mapped_file_region instead of MemoryBuffer to drop a
  /// layer of indirection.
  std::unique_ptr<MemoryBuffer> Region;
  TrieRecord::StorageKind SK;
  StandaloneDataInMemory(std::unique_ptr<MemoryBuffer> Region,
                         TrieRecord::StorageKind SK)
      : Region(std::move(Region)), SK(SK) {
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

/// Container for "big" objects mapped in separately.
template <size_t NumShards> class StandaloneDataMap {
  static_assert(isPowerOf2_64(NumShards), "Expected power of 2");

public:
  const StandaloneDataInMemory &insert(ArrayRef<uint8_t> Hash,
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

using StandaloneDataMapTy = StandaloneDataMap<16>;

struct InternalHandle {
  FileOffset getAsFileOffset() const { return *DataOffset; }

  uint64_t getRawData() const {
    if (DataOffset) {
      uint64_t Raw = DataOffset->get();
      assert(!(Raw & 0x1));
      return Raw;
    }
    uint64_t Raw = reinterpret_cast<uintptr_t>(SDIM);
    assert(!(Raw & 0x1));
    return Raw | 1;
  }

  explicit InternalHandle(FileOffset DataOffset) : DataOffset(DataOffset) {}
  explicit InternalHandle(uint64_t DataOffset) : DataOffset(DataOffset) {}
  explicit InternalHandle(const StandaloneDataInMemory &SDIM) : SDIM(&SDIM) {}
  std::optional<FileOffset> DataOffset;
  const StandaloneDataInMemory *SDIM = nullptr;
};

class InternalRefVector {
public:
  void push_back(InternalRef Ref) {
    if (NeedsFull)
      return FullRefs.push_back(Ref);
    if (std::optional<InternalRef4B> Small = InternalRef4B::tryToShrink(Ref))
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

} // namespace

/// Proxy for any on-disk object or raw data.
struct ondisk::OnDiskContent {
  std::optional<DataRecordHandle> Record;
  std::optional<ArrayRef<char>> Bytes;
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

/// Proxy for an on-disk index record.
struct OnDiskGraphDB::IndexProxy {
  FileOffset Offset;
  ArrayRef<uint8_t> Hash;
  TrieRecord &Ref;
};

template <size_t N>
const StandaloneDataInMemory &
StandaloneDataMap<N>::insert(ArrayRef<uint8_t> Hash, TrieRecord::StorageKind SK,
                             std::unique_ptr<MemoryBuffer> Buffer) {
  auto &S = getShard(Hash);
  std::lock_guard<std::mutex> Lock(S.Mutex);
  auto &V = S.Map[Hash.data()];
  if (!V)
    V = std::make_unique<StandaloneDataInMemory>(std::move(Buffer), SK);
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
class OnDiskGraphDB::TempFile {
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

class OnDiskGraphDB::MappedTempFile {
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

Error OnDiskGraphDB::TempFile::discard() {
  Done = true;
  if (FD != -1) {
    sys::fs::file_t File = sys::fs::convertFDToNativeFile(FD);
    if (std::error_code EC = sys::fs::closeFile(File))
      return errorCodeToError(EC);
  }
  FD = -1;

  // Always try to close and remove.
  std::error_code RemoveEC;
  if (!TmpName.empty())
    if (std::error_code EC = sys::fs::remove(TmpName))
      return errorCodeToError(EC);
  TmpName = "";

  return Error::success();
}

Error OnDiskGraphDB::TempFile::keep(const Twine &Name) {
  assert(!Done);
  Done = true;
  // Always try to close and rename.
  std::error_code RenameEC = sys::fs::rename(TmpName, Name);

  if (!RenameEC)
    TmpName = "";

  sys::fs::file_t File = sys::fs::convertFDToNativeFile(FD);
  if (std::error_code EC = sys::fs::closeFile(File))
    return errorCodeToError(EC);
  FD = -1;

  return errorCodeToError(RenameEC);
}

Expected<OnDiskGraphDB::TempFile>
OnDiskGraphDB::TempFile::create(const Twine &Model) {
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
  char *Next = Mem + sizeof(Header);

  // Fill in Packed and set other data, then come back to construct the header.
  Header::PackTy Packed = 0;
  Packed |= LayoutFlags::pack(L.Flags) << Header::LayoutFlagsShift;

  // Construct DataSize.
  switch (L.Flags.DataSize) {
  case DataSizeFlags::Uses1B:
    assert(I.Data.size() <= UINT8_MAX);
    Packed |= (Header::PackTy)I.Data.size()
              << ((sizeof(Packed) - 2) * CHAR_BIT);
    break;
  case DataSizeFlags::Uses2B:
    assert(I.Data.size() <= UINT16_MAX);
    Packed |= (Header::PackTy)I.Data.size()
              << ((sizeof(Packed) - 4) * CHAR_BIT);
    break;
  case DataSizeFlags::Uses4B:
    support::endian::write32le(Next, I.Data.size());
    Next += 4;
    break;
  case DataSizeFlags::Uses8B:
    support::endian::write64le(Next, I.Data.size());
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
    Packed |= (Header::PackTy)I.Refs.size()
              << ((sizeof(Packed) - 2) * CHAR_BIT);
    break;
  case NumRefsFlags::Uses2B:
    assert(I.Refs.size() <= UINT16_MAX);
    Packed |= (Header::PackTy)I.Refs.size()
              << ((sizeof(Packed) - 4) * CHAR_BIT);
    break;
  case NumRefsFlags::Uses4B:
    support::endian::write32le(Next, I.Refs.size());
    Next += 4;
    break;
  case NumRefsFlags::Uses8B:
    support::endian::write64le(Next, I.Refs.size());
    Next += 8;
    break;
  }

  // Construct Refs[].
  if (!I.Refs.empty()) {
    assert((L.Flags.RefKind == RefKindFlags::InternalRef4B) == I.Refs.is4B());
    ArrayRef<uint8_t> RefsBuffer = I.Refs.getBuffer();
    llvm::copy(RefsBuffer, Next);
    Next += RefsBuffer.size();
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

  // Find the smallest slot available for DataSize.
  bool Has1B = true;
  bool Has2B = true;
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
    // If we remove the need for 8B-alignment for data there is <1% savings in
    // disk storage for a clang build using MCCAS but the 8B-alignment may be
    // useful in the future so keep it for now.
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
    return (H->Packed >> ((sizeof(Header::PackTy) - 2) * CHAR_BIT)) & UINT8_MAX;
  case DataSizeFlags::Uses2B:
    return (H->Packed >> ((sizeof(Header::PackTy) - 4) * CHAR_BIT)) &
           UINT16_MAX;
  case DataSizeFlags::Uses4B:
    return support::endian::read32le(DataSizePtr);
  case DataSizeFlags::Uses8B:
    return support::endian::read64le(DataSizePtr);
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
    return (H->Packed >> ((sizeof(Header::PackTy) - 2) * CHAR_BIT)) & UINT8_MAX;
  case NumRefsFlags::Uses2B:
    return (H->Packed >> ((sizeof(Header::PackTy) - 4) * CHAR_BIT)) &
           UINT16_MAX;
  case NumRefsFlags::Uses4B:
    return support::endian::read32le(NumRefsPtr);
  case NumRefsFlags::Uses8B:
    return support::endian::read64le(NumRefsPtr);
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

Error OnDiskGraphDB::validate(bool Deep, HashingFuncT Hasher) const {
  return Index.validate([&](FileOffset Offset,
                            OnDiskHashMappedTrie::ConstValueProxy Record)
                            -> Error {
    auto formatError = [&](Twine Msg) {
      return createStringError(
          llvm::errc::illegal_byte_sequence,
          "bad record at 0x" +
              utohexstr((unsigned)Offset.get(), /*LowerCase=*/true) + ": " +
              Msg.str());
    };

    if (Record.Data.size() != sizeof(TrieRecord))
      return formatError("wrong data record size");
    if (!isAligned(Align::Of<TrieRecord>(), Record.Data.size()))
      return formatError("wrong data record alignment");

    auto *R = reinterpret_cast<const TrieRecord *>(Record.Data.data());
    TrieRecord::Data D = R->load();
    std::unique_ptr<MemoryBuffer> FileBuffer;
    if ((uint8_t)D.SK != (uint8_t)TrieRecord::StorageKind::Unknown &&
        (uint8_t)D.SK != (uint8_t)TrieRecord::StorageKind::DataPool &&
        (uint8_t)D.SK != (uint8_t)TrieRecord::StorageKind::Standalone &&
        (uint8_t)D.SK != (uint8_t)TrieRecord::StorageKind::StandaloneLeaf &&
        (uint8_t)D.SK != (uint8_t)TrieRecord::StorageKind::StandaloneLeaf0)
      return formatError("invalid record kind value");

    auto Ref = InternalRef::getFromOffset(Offset);
    auto I = getIndexProxyFromRef(Ref);

    switch (D.SK) {
    case TrieRecord::StorageKind::Unknown:
      // This could be an abandoned entry due to a termination before updating
      // the record. It can be reused by later insertion so just skip this entry
      // for now.
      return Error::success();
    case TrieRecord::StorageKind::DataPool:
      // Check offset is a postive value, and large enough to hold the
      // header for the data record.
      if (D.Offset.get() <= 0 ||
          (uint64_t)D.Offset.get() + sizeof(DataRecordHandle::Header) >=
              DataPool.size())
        return formatError("datapool record out of bound");
      break;
    case TrieRecord::StorageKind::Standalone:
    case TrieRecord::StorageKind::StandaloneLeaf:
    case TrieRecord::StorageKind::StandaloneLeaf0:
      SmallString<256> Path;
      getStandalonePath(TrieRecord::getStandaloneFileSuffix(D.SK), I, Path);
      // If need to validate the content of the file later, just load the
      // buffer here. Otherwise, just check the existance of the file.
      if (Deep) {
        auto File = MemoryBuffer::getFile(Path, /*IsText=*/false,
                                          /*RequiresNullTerminator=*/false);
        if (!File || !*File)
          return formatError("record file \'" + Path + "\' does not exist");

        FileBuffer = std::move(*File);
      } else if (!llvm::sys::fs::exists(Path))
        return formatError("record file \'" + Path + "\' does not exist");
    }

    if (!Deep)
      return Error::success();

    auto dataError = [&](Twine Msg) {
      return createStringError(llvm::errc::illegal_byte_sequence,
                               "bad data for digest \'" + toHex(I.Hash) +
                                   "\': " + Msg.str());
    };
    SmallVector<ArrayRef<uint8_t>> Refs;
    ArrayRef<char> StoredData;

    switch (D.SK) {
    case TrieRecord::StorageKind::Unknown:
      llvm_unreachable("already handled");
    case TrieRecord::StorageKind::DataPool: {
      auto DataRecord = DataRecordHandle::get(DataPool.beginData(D.Offset));
      if (DataRecord.getTotalSize() + D.Offset.get() >= DataPool.size())
        return dataError("data record span passed the end of the data pool");
      for (auto InternRef : DataRecord.getRefs()) {
        auto Index = getIndexProxyFromRef(InternRef);
        Refs.push_back(Index.Hash);
      }
      StoredData = DataRecord.getData();
      break;
    }
    case TrieRecord::StorageKind::Standalone: {
      if (FileBuffer->getBufferSize() < sizeof(DataRecordHandle::Header))
        return dataError("data record is not big enough to read the header");
      auto DataRecord = DataRecordHandle::get(FileBuffer->getBufferStart());
      if (DataRecord.getTotalSize() < FileBuffer->getBufferSize())
        return dataError(
            "data record span passed the end of the standalone file");
      for (auto InternRef : DataRecord.getRefs()) {
        auto Index = getIndexProxyFromRef(InternRef);
        Refs.push_back(Index.Hash);
      }
      StoredData = DataRecord.getData();
      break;
    }
    case TrieRecord::StorageKind::StandaloneLeaf:
    case TrieRecord::StorageKind::StandaloneLeaf0: {
      StoredData = arrayRefFromStringRef<char>(FileBuffer->getBuffer());
      if (D.SK == TrieRecord::StorageKind::StandaloneLeaf0) {
        if (!FileBuffer->getBuffer().ends_with('\0'))
          return dataError("standalone file is not zero terminated");
        StoredData = StoredData.drop_back(1);
      }
      break;
    }
    }

    SmallVector<uint8_t> ComputedHash;
    Hasher(Refs, StoredData, ComputedHash);
    if (I.Hash != ArrayRef(ComputedHash))
      return dataError("hash mismatch, got \'" + toHex(ComputedHash) +
                       "\' instead");

    return Error::success();
  });
}

void OnDiskGraphDB::print(raw_ostream &OS) const {
  OS << "on-disk-root-path: " << RootPath << "\n";

  struct PoolInfo {
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
    OS << " SK=";
    switch (D.SK) {
    case TrieRecord::StorageKind::Unknown:
      OS << "unknown          ";
      break;
    case TrieRecord::StorageKind::DataPool:
      OS << "datapool         ";
      Pool.push_back({D.Offset.get()});
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
    DataRecordHandle D =
        DataRecordHandle::get(DataPool.beginData(FileOffset(PI.Offset)));
    OS << "record refs=" << D.getNumRefs() << " data=" << D.getDataSize()
       << " size=" << D.getTotalSize()
       << " end=" << (void *)(PI.Offset + D.getTotalSize()) << "\n";
  }
}

Expected<OnDiskGraphDB::IndexProxy>
OnDiskGraphDB::indexHash(ArrayRef<uint8_t> Hash) {
  auto P = Index.insertLazy(
      Hash, [](FileOffset TentativeOffset,
               OnDiskHashMappedTrie::ValueProxy TentativeValue) {
        assert(TentativeValue.Data.size() == sizeof(TrieRecord));
        assert(
            isAddrAligned(Align::Of<TrieRecord>(), TentativeValue.Data.data()));
        new (TentativeValue.Data.data()) TrieRecord();
      });
  if (LLVM_UNLIKELY(!P))
    return P.takeError();

  assert(*P && "Expected insertion");
  return getIndexProxyFromPointer(*P);
}

OnDiskGraphDB::IndexProxy OnDiskGraphDB::getIndexProxyFromPointer(
    OnDiskHashMappedTrie::const_pointer P) const {
  assert(P);
  assert(P.getOffset());
  return IndexProxy{P.getOffset(), P->Hash,
                    *const_cast<TrieRecord *>(
                        reinterpret_cast<const TrieRecord *>(P->Data.data()))};
}

Expected<ObjectID> OnDiskGraphDB::getReference(ArrayRef<uint8_t> Hash) {
  auto I = indexHash(Hash);
  if (LLVM_UNLIKELY(!I))
    return I.takeError();
  return getExternalReference(*I);
}

ObjectID OnDiskGraphDB::getExternalReference(const IndexProxy &I) {
  return getExternalReference(makeInternalRef(I.Offset));
}

std::optional<ObjectID>
OnDiskGraphDB::getExistingReference(ArrayRef<uint8_t> Digest) {
  auto tryUpstream =
      [&](std::optional<IndexProxy> I) -> std::optional<ObjectID> {
    if (!UpstreamDB)
      return std::nullopt;
    std::optional<ObjectID> UpstreamID =
        UpstreamDB->getExistingReference(Digest);
    if (LLVM_UNLIKELY(!UpstreamID))
      return std::nullopt;
    auto Ref = expectedToOptional(indexHash(Digest));
    if (!Ref)
      return std::nullopt;
    if (!I)
      I.emplace(*Ref);
    return getExternalReference(*I);
  };

  OnDiskHashMappedTrie::const_pointer P = Index.find(Digest);
  if (!P)
    return tryUpstream(std::nullopt);
  IndexProxy I = getIndexProxyFromPointer(P);
  TrieRecord::Data Obj = I.Ref.load();
  if (Obj.SK == TrieRecord::StorageKind::Unknown)
    return tryUpstream(I);
  return getExternalReference(makeInternalRef(I.Offset));
}

OnDiskGraphDB::IndexProxy
OnDiskGraphDB::getIndexProxyFromRef(InternalRef Ref) const {
  OnDiskHashMappedTrie::const_pointer P =
      Index.recoverFromFileOffset(Ref.getFileOffset());
  if (LLVM_UNLIKELY(!P))
    report_fatal_error("OnDiskCAS: corrupt internal reference");
  return getIndexProxyFromPointer(P);
}

ArrayRef<uint8_t> OnDiskGraphDB::getDigest(InternalRef Ref) const {
  IndexProxy I = getIndexProxyFromRef(Ref);
  return I.Hash;
}

ArrayRef<uint8_t> OnDiskGraphDB::getDigest(const IndexProxy &I) const {
  return I.Hash;
}

ArrayRef<char> OnDiskGraphDB::getObjectData(ObjectHandle Node) const {
  OnDiskContent Content = getContentFromHandle(Node);
  if (Content.Bytes)
    return *Content.Bytes;
  assert(Content.Record && "Expected record or bytes");
  return Content.Record->getData();
}

InternalRefArrayRef OnDiskGraphDB::getInternalRefs(ObjectHandle Node) const {
  if (std::optional<DataRecordHandle> Record =
          getContentFromHandle(Node).Record)
    return Record->getRefs();
  return std::nullopt;
}

Expected<std::optional<ObjectHandle>>
OnDiskGraphDB::load(ObjectID ExternalRef) {
  InternalRef Ref = getInternalRef(ExternalRef);
  IndexProxy I = getIndexProxyFromRef(Ref);
  TrieRecord::Data Object = I.Ref.load();

  if (Object.SK == TrieRecord::StorageKind::Unknown) {
    if (!UpstreamDB)
      return std::nullopt;
    return faultInFromUpstream(ExternalRef);
  }

  auto toObjectHandle = [](InternalHandle H) -> ObjectHandle {
    return ObjectHandle::fromOpaqueData(H.getRawData());
  };

  if (Object.SK == TrieRecord::StorageKind::DataPool)
    return toObjectHandle(InternalHandle(Object.Offset));

  // Only TrieRecord::StorageKind::Standalone (and variants) need to be
  // explicitly loaded.
  //
  // There's corruption if standalone objects have offsets, or if we get here
  // for something that isn't standalone.
  if (Object.Offset)
    return createCorruptObjectError(getDigest(I));
  switch (Object.SK) {
  case TrieRecord::StorageKind::Unknown:
  case TrieRecord::StorageKind::DataPool:
    llvm_unreachable("unexpected storage kind");
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
  getStandalonePath(TrieRecord::getStandaloneFileSuffix(Object.SK), I, Path);
  ErrorOr<std::unique_ptr<MemoryBuffer>> OwnedBuffer = MemoryBuffer::getFile(
      Path, /*IsText=*/false, /*RequiresNullTerminator=*/false);
  if (!OwnedBuffer)
    return createCorruptObjectError(getDigest(I));

  return toObjectHandle(
      InternalHandle(static_cast<StandaloneDataMapTy *>(StandaloneData)
                         ->insert(I.Hash, Object.SK, std::move(*OwnedBuffer))));
}

Expected<bool> OnDiskGraphDB::isMaterialized(ObjectID Ref) {
  switch (getObjectPresence(Ref, /*CheckUpstream=*/true)) {
  case ObjectPresence::Missing:
    return false;
  case ObjectPresence::InPrimaryDB:
    return true;
  case ObjectPresence::OnlyInUpstreamDB:
    if (auto FaultInResult = faultInFromUpstream(Ref); !FaultInResult)
      return FaultInResult.takeError();
    return true;
  }
}

OnDiskGraphDB::ObjectPresence
OnDiskGraphDB::getObjectPresence(ObjectID ExternalRef,
                                 bool CheckUpstream) const {
  InternalRef Ref = getInternalRef(ExternalRef);
  IndexProxy I = getIndexProxyFromRef(Ref);
  TrieRecord::Data Object = I.Ref.load();
  if (Object.SK != TrieRecord::StorageKind::Unknown)
    return ObjectPresence::InPrimaryDB;
  if (!CheckUpstream || !UpstreamDB)
    return ObjectPresence::Missing;
  std::optional<ObjectID> UpstreamID =
      UpstreamDB->getExistingReference(getDigest(I));
  return UpstreamID.has_value() ? ObjectPresence::OnlyInUpstreamDB
                                : ObjectPresence::Missing;
}

InternalRef OnDiskGraphDB::makeInternalRef(FileOffset IndexOffset) {
  return InternalRef::getFromOffset(IndexOffset);
}

void OnDiskGraphDB::getStandalonePath(StringRef Suffix, const IndexProxy &I,
                                      SmallVectorImpl<char> &Path) const {
  Path.assign(RootPath.begin(), RootPath.end());
  sys::path::append(Path, FilePrefix + Twine(I.Offset.get()) + Suffix);
}

OnDiskContent OnDiskGraphDB::getContentFromHandle(ObjectHandle OH) const {
  auto getInternalHandle = [](ObjectHandle Handle) -> InternalHandle {
    uint64_t Data = Handle.getOpaqueData();
    if (Data & 1)
      return InternalHandle(*reinterpret_cast<const StandaloneDataInMemory *>(
          Data & (-1ULL << 1)));
    return InternalHandle(Data);
  };

  InternalHandle Handle = getInternalHandle(OH);
  if (Handle.SDIM)
    return Handle.SDIM->getContent();

  auto DataHandle =
      DataRecordHandle::get(DataPool.beginData(Handle.getAsFileOffset()));
  assert(DataHandle.getData().end()[0] == 0 && "Null termination");
  return OnDiskContent{DataHandle, std::nullopt};
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
    return OnDiskContent{
        std::nullopt,
        arrayRefFromStringRef<char>(Region->getBuffer().drop_back(Leaf0))};
  }

  DataRecordHandle Record = DataRecordHandle::get(Region->getBuffer().data());
  assert(Record.getData().end()[0] == 0 &&
         "Standalone object record missing null termination for data");
  return OnDiskContent{Record, std::nullopt};
}

Expected<OnDiskGraphDB::MappedTempFile>
OnDiskGraphDB::createTempFile(StringRef FinalPath, uint64_t Size) {
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

Error OnDiskGraphDB::createStandaloneLeaf(IndexProxy &I, ArrayRef<char> Data) {
  assert(Data.size() > TrieRecord::MaxEmbeddedSize &&
         "Expected a bigger file for external content...");

  bool Leaf0 = isAligned(Align(getPageSize()), Data.size());
  TrieRecord::StorageKind SK = Leaf0 ? TrieRecord::StorageKind::StandaloneLeaf0
                                     : TrieRecord::StorageKind::StandaloneLeaf;

  SmallString<256> Path;
  int64_t FileSize = Data.size() + Leaf0;
  getStandalonePath(TrieRecord::getStandaloneFileSuffix(SK), I, Path);

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
    return E;

  // Store the object reference.
  TrieRecord::Data Existing;
  {
    TrieRecord::Data Leaf{SK, FileOffset()};
    if (I.Ref.compare_exchange_strong(Existing, Leaf)) {
      recordStandaloneSizeIncrease(FileSize);
      return Error::success();
    }
  }

  // If there was a race, confirm that the new value has valid storage.
  if (Existing.SK == TrieRecord::StorageKind::Unknown)
    return createCorruptObjectError(getDigest(I));

  return Error::success();
}

Error OnDiskGraphDB::store(ObjectID ID, ArrayRef<ObjectID> Refs,
                           ArrayRef<char> Data) {
  IndexProxy I = getIndexProxyFromRef(getInternalRef(ID));

  // Early return in case the node exists.
  {
    TrieRecord::Data Existing = I.Ref.load();
    if (Existing.SK != TrieRecord::StorageKind::Unknown)
      return Error::success();
  }

  // Big leaf nodes.
  if (Refs.empty() && Data.size() > TrieRecord::MaxEmbeddedSize)
    return createStandaloneLeaf(I, Data);

  // TODO: Check whether it's worth checking the index for an already existing
  // object (like storeTreeImpl() does) before building up the
  // InternalRefVector.
  InternalRefVector InternalRefs;
  for (ObjectID Ref : Refs)
    InternalRefs.push_back(getInternalRef(Ref));

  // Create the object.

  DataRecordHandle::Input Input{InternalRefs, Data};

  // Compute the storage kind, allocate it, and create the record.
  TrieRecord::StorageKind SK = TrieRecord::StorageKind::Unknown;
  FileOffset PoolOffset;
  SmallString<256> Path;
  std::optional<MappedTempFile> File;
  std::optional<uint64_t> FileSize;
  auto AllocStandaloneFile = [&](size_t Size) -> Expected<char *> {
    getStandalonePath(TrieRecord::getStandaloneFileSuffix(
                          TrieRecord::StorageKind::Standalone),
                      I, Path);
    if (Error E = createTempFile(Path, Size).moveInto(File))
      return std::move(E);
    assert(File->size() == Size);
    FileSize = Size;
    SK = TrieRecord::StorageKind::Standalone;
    return File->data();
  };
  auto Alloc = [&](size_t Size) -> Expected<char *> {
    if (Size <= TrieRecord::MaxEmbeddedSize) {
      SK = TrieRecord::StorageKind::DataPool;
      auto P = DataPool.allocate(Size);
      if (LLVM_UNLIKELY(!P)) {
        char *NewAlloc = nullptr;
        auto NewE = handleErrors(
            P.takeError(), [&](std::unique_ptr<StringError> E) -> Error {
              if (E->convertToErrorCode() == std::errc::not_enough_memory)
                return AllocStandaloneFile(Size).moveInto(NewAlloc);
              return Error(std::move(E));
            });
        if (!NewE)
          return NewAlloc;
        return std::move(NewE);
      }
      PoolOffset = P->getOffset();
      LLVM_DEBUG({
        dbgs() << "pool-alloc addr=" << (void *)PoolOffset.get()
               << " size=" << Size
               << " end=" << (void *)(PoolOffset.get() + Size) << "\n";
      });
      return (*P)->data();
    }
    return AllocStandaloneFile(Size);
  };

  DataRecordHandle Record;
  if (Error E =
          DataRecordHandle::createWithError(Alloc, Input).moveInto(Record))
    return E;
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
    TrieRecord::Data NewObject{SK, PoolOffset};
    if (File) {
      if (Existing.SK == TrieRecord::StorageKind::Unknown) {
        // Keep the file!
        if (Error E = File->keep(Path))
          return E;
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
      if (I.Ref.compare_exchange_strong(Existing, NewObject)) {
        if (FileSize)
          recordStandaloneSizeIncrease(*FileSize);
        return Error::success();
      }
    }
  }

  if (Existing.SK == TrieRecord::StorageKind::Unknown)
    return createCorruptObjectError(getDigest(I));

  // Load existing object.
  return Error::success();
}

void OnDiskGraphDB::recordStandaloneSizeIncrease(size_t SizeIncrease) {
  getStandaloneStorageSize().fetch_add(SizeIncrease, std::memory_order_relaxed);
}

std::atomic<uint64_t> &OnDiskGraphDB::getStandaloneStorageSize() {
  MutableArrayRef<uint8_t> UserHeader = DataPool.getUserHeader();
  assert(UserHeader.size() == sizeof(std::atomic<uint64_t>));
  assert(isAddrAligned(Align(8), UserHeader.data()));
  return *reinterpret_cast<std::atomic<uint64_t> *>(UserHeader.data());
}

uint64_t OnDiskGraphDB::getStandaloneStorageSize() const {
  return const_cast<OnDiskGraphDB *>(this)->getStandaloneStorageSize().load(
      std::memory_order_relaxed);
}

size_t OnDiskGraphDB::getStorageSize() const {
  return Index.size() + DataPool.size() + getStandaloneStorageSize();
}

unsigned OnDiskGraphDB::getHardStorageLimitUtilization() const {
  unsigned IndexPercent = Index.size() * 100ULL / Index.capacity();
  unsigned DataPercent = DataPool.size() * 100ULL / DataPool.capacity();
  return std::max(IndexPercent, DataPercent);
}

static bool useSmallMappedFiles(const Twine &P) {
  // macOS tmpfs does not support sparse tails.
#if defined(__APPLE__) && __has_include(<sys/mount.h>)
  SmallString<128> PathStorage;
  StringRef Path = P.toNullTerminatedStringRef(PathStorage);
  struct statfs StatFS;
  if (statfs(Path.data(), &StatFS) != 0)
    return false;

  if (strcmp(StatFS.f_fstypename, "tmpfs") == 0)
    return true;
#endif

  return false;
}

Expected<std::unique_ptr<OnDiskGraphDB>> OnDiskGraphDB::open(
    StringRef AbsPath, StringRef HashName, unsigned HashByteSize,
    std::unique_ptr<OnDiskGraphDB> UpstreamDB, FaultInPolicy Policy) {
  if (std::error_code EC = sys::fs::create_directories(AbsPath))
    return createFileError(AbsPath, EC);

  const StringRef Slash = sys::path::get_separator();
  constexpr uint64_t MB = 1024ull * 1024ull;
  constexpr uint64_t GB = 1024ull * 1024ull * 1024ull;

  uint64_t MaxIndexSize = 12 * GB;
  uint64_t MaxDataPoolSize = 24 * GB;

  if (useSmallMappedFiles(AbsPath)) {
    MaxIndexSize = 1 * GB;
    MaxDataPoolSize = 2 * GB;
  }

  auto CustomSize = getOverriddenMaxMappingSize();
  if (!CustomSize)
    return CustomSize.takeError();
  if (*CustomSize)
    MaxIndexSize = MaxDataPoolSize = **CustomSize;

  std::optional<OnDiskHashMappedTrie> Index;
  if (Error E =
          OnDiskHashMappedTrie::create(
              AbsPath + Slash + FilePrefix + IndexFile,
              IndexTableName + "[" + HashName + "]", HashByteSize * CHAR_BIT,
              /*DataSize=*/sizeof(TrieRecord), MaxIndexSize, /*MinFileSize=*/MB)
              .moveInto(Index))
    return std::move(E);

  uint32_t UserHeaderSize = sizeof(std::atomic<uint64_t>);
  std::optional<OnDiskDataAllocator> DataPool;
  StringRef PolicyName =
      Policy == FaultInPolicy::SingleNode ? "single" : "full";
  if (Error E = OnDiskDataAllocator::create(
                    AbsPath + Slash + FilePrefix + DataPoolFile,
                    DataPoolTableName + "[" + HashName + "]" + PolicyName,
                    MaxDataPoolSize, /*MinFileSize=*/MB, UserHeaderSize,
                    [](void *UserHeaderPtr) {
                      new (UserHeaderPtr) std::atomic<uint64_t>(0);
                    })
                    .moveInto(DataPool))
    return std::move(E);
  if (DataPool->getUserHeader().size() != UserHeaderSize)
    return createStringError(llvm::errc::argument_out_of_domain,
                             "unexpected user header in '" + AbsPath + Slash +
                                 FilePrefix + DataPoolFile + "'");

  return std::unique_ptr<OnDiskGraphDB>(
      new OnDiskGraphDB(AbsPath, std::move(*Index), std::move(*DataPool),
                        std::move(UpstreamDB), Policy));
}

OnDiskGraphDB::OnDiskGraphDB(StringRef RootPath, OnDiskHashMappedTrie Index,
                             OnDiskDataAllocator DataPool,
                             std::unique_ptr<OnDiskGraphDB> UpstreamDB,
                             FaultInPolicy Policy)
    : Index(std::move(Index)), DataPool(std::move(DataPool)),
      RootPath(RootPath.str()), UpstreamDB(std::move(UpstreamDB)),
      FIPolicy(Policy) {
  /// Lifetime for "big" objects not in DataPool.
  ///
  /// NOTE: Could use ThreadSafeHashMappedTrie here. For now, doing something
  /// simpler on the assumption there won't be much contention since most data
  /// is not big. If there is contention, and we've already fixed ObjectProxy
  /// object handles to be cheap enough to use consistently, the fix might be
  /// to use better use of them rather than optimizing this map.
  ///
  /// FIXME: Figure out the right number of shards, if any.
  StandaloneData = new StandaloneDataMapTy();
}

OnDiskGraphDB::~OnDiskGraphDB() {
  delete static_cast<StandaloneDataMapTy *>(StandaloneData);
}

Error OnDiskGraphDB::importFullTree(ObjectID PrimaryID,
                                    ObjectHandle UpstreamNode) {
  // Copies the full CAS tree from upstream. Uses depth-first copying to protect
  // against the process dying during importing and leaving the database with an
  // incomplete tree. Note that if the upstream has missing nodes then the tree
  // will be copied with missing nodes as well, it won't be considered an error.

  struct UpstreamCursor {
    ObjectHandle Node;
    size_t RefsCount;
    object_refs_iterator RefI;
    object_refs_iterator RefE;
  };
  /// Keeps track of the state of visitation for current node and all of its
  /// parents.
  SmallVector<UpstreamCursor, 16> CursorStack;
  /// Keeps track of the currently visited nodes as they are imported into
  /// primary database, from current node and its parents. When a node is
  /// entered for visitation it appends its own ID, then appends referenced IDs
  /// as they get imported. When a node is fully imported it removes the
  /// referenced IDs from the bottom of the stack which leaves its own ID at the
  /// bottom, adding to the list of referenced IDs for the parent node.
  SmallVector<ObjectID, 128> PrimaryNodesStack;

  auto enqueueNode = [&](ObjectID PrimaryID, std::optional<ObjectHandle> Node) {
    PrimaryNodesStack.push_back(PrimaryID);
    if (!Node)
      return;
    auto Refs = UpstreamDB->getObjectRefs(*Node);
    CursorStack.push_back({*Node,
                           (size_t)std::distance(Refs.begin(), Refs.end()),
                           Refs.begin(), Refs.end()});
  };

  enqueueNode(PrimaryID, UpstreamNode);

  while (!CursorStack.empty()) {
    UpstreamCursor &Cur = CursorStack.back();
    if (Cur.RefI == Cur.RefE) {
      // Copy the node data into the primary store.
      // FIXME: Use hard-link or cloning if the file-system supports it and data
      // is stored into a separate file.

      // The bottom of \p PrimaryNodesStack contains the primary ID for the
      // current node plus the list of imported referenced IDs.
      assert(PrimaryNodesStack.size() >= Cur.RefsCount + 1);
      ObjectID PrimaryID = *(PrimaryNodesStack.end() - Cur.RefsCount - 1);
      auto PrimaryRefs = ArrayRef(PrimaryNodesStack)
                             .slice(PrimaryNodesStack.size() - Cur.RefsCount);
      auto Data = UpstreamDB->getObjectData(Cur.Node);
      if (Error E = store(PrimaryID, PrimaryRefs, Data))
        return E;
      // Remove the current node and its IDs from the stack.
      PrimaryNodesStack.truncate(PrimaryNodesStack.size() - Cur.RefsCount);
      CursorStack.pop_back();
      continue;
    }

    ObjectID UpstreamID = *(Cur.RefI++);
    auto PrimaryID = getReference(UpstreamDB->getDigest(UpstreamID));
    if (LLVM_UNLIKELY(!PrimaryID))
      return PrimaryID.takeError();
    if (containsObject(*PrimaryID, /*CheckUpstream=*/false)) {
      // This \p ObjectID already exists in the primary. Either it was imported
      // via \p importFullTree or the client created it, in which case the
      // client takes responsibility for how it was formed.
      enqueueNode(*PrimaryID, std::nullopt);
      continue;
    }
    Expected<std::optional<ObjectHandle>> UpstreamNode =
        UpstreamDB->load(UpstreamID);
    if (!UpstreamNode)
      return UpstreamNode.takeError();
    enqueueNode(*PrimaryID, *UpstreamNode);
  }

  assert(PrimaryNodesStack.size() == 1);
  assert(PrimaryNodesStack.front() == PrimaryID);
  return Error::success();
}

Error OnDiskGraphDB::importSingleNode(ObjectID PrimaryID,
                                      ObjectHandle UpstreamNode) {
  // Copies only a single node, it doesn't copy the referenced nodes.

  // Copy the node data into the primary store.
  // FIXME: Use hard-link or cloning if the file-system supports it and data is
  // stored into a separate file.

  auto Data = UpstreamDB->getObjectData(UpstreamNode);
  auto UpstreamRefs = UpstreamDB->getObjectRefs(UpstreamNode);
  SmallVector<ObjectID, 64> Refs;
  Refs.reserve(std::distance(UpstreamRefs.begin(), UpstreamRefs.end()));
  for (ObjectID UpstreamRef : UpstreamRefs) {
    auto Ref = getReference(UpstreamDB->getDigest(UpstreamRef));
    if (LLVM_UNLIKELY(!Ref))
      return Ref.takeError();
    Refs.push_back(*Ref);
  }

  return store(PrimaryID, Refs, Data);
}

Expected<std::optional<ObjectHandle>>
OnDiskGraphDB::faultInFromUpstream(ObjectID PrimaryID) {
  assert(UpstreamDB);

  auto UpstreamID = UpstreamDB->getReference(getDigest(PrimaryID));
  if (LLVM_UNLIKELY(!UpstreamID))
    return UpstreamID.takeError();

  Expected<std::optional<ObjectHandle>> UpstreamNode =
      UpstreamDB->load(*UpstreamID);
  if (!UpstreamNode)
    return UpstreamNode.takeError();
  if (!*UpstreamNode)
    return std::nullopt;

  if (Error E = FIPolicy == FaultInPolicy::SingleNode
                    ? importSingleNode(PrimaryID, **UpstreamNode)
                    : importFullTree(PrimaryID, **UpstreamNode))
    return std::move(E);
  return load(PrimaryID);
}
