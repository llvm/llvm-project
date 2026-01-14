//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file Implements OnDiskDataAllocator.
///
//===----------------------------------------------------------------------===//

#include "llvm/CAS/OnDiskDataAllocator.h"
#include "DatabaseFile.h"
#include "llvm/Config/llvm-config.h"

using namespace llvm;
using namespace llvm::cas;
using namespace llvm::cas::ondisk;

#if LLVM_ENABLE_ONDISK_CAS

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

  Expected<MutableArrayRef<char>> allocate(MappedFileRegionArena &Alloc,
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

  static Expected<DataAllocatorHandle>
  create(MappedFileRegionArena &Alloc, StringRef Name, uint32_t UserHeaderSize);

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
DataAllocatorHandle::create(MappedFileRegionArena &Alloc, StringRef Name,
                            uint32_t UserHeaderSize) {
  // Allocate.
  auto Offset =
      Alloc.allocateOffset(sizeof(Header) + UserHeaderSize + Name.size() + 1);
  if (LLVM_UNLIKELY(!Offset))
    return Offset.takeError();

  // Construct the header and the name.
  assert(Name.size() <= UINT16_MAX && "Expected smaller table name");
  auto *H = new (Alloc.getRegion().data() + *Offset)
      Header{{TableHandle::TableKind::DataAllocator,
              static_cast<uint16_t>(Name.size()),
              static_cast<int32_t>(sizeof(Header) + UserHeaderSize)},
             /*AllocatorOffset=*/{0},
             /*UserHeaderSize=*/UserHeaderSize};
  // Memset UserHeader.
  char *UserHeader = reinterpret_cast<char *>(H + 1);
  memset(UserHeader, 0, UserHeaderSize);
  // Write database file name (null-terminated).
  char *NameStorage = UserHeader + UserHeaderSize;
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

    if (auto E = DB.addTable(*Store))
      return E;

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

Expected<OnDiskDataAllocator::OnDiskPtr>
OnDiskDataAllocator::allocate(size_t Size) {
  auto Data = Impl->Store.allocate(Impl->File.getAlloc(), Size);
  if (LLVM_UNLIKELY(!Data))
    return Data.takeError();

  return OnDiskPtr(FileOffset(Data->data() - Impl->Store.getRegion().data()),
                   *Data);
}

Expected<ArrayRef<char>> OnDiskDataAllocator::get(FileOffset Offset,
                                                  size_t Size) const {
  assert(Offset);
  assert(Impl);
  if (Offset.get() + Size >= Impl->File.getAlloc().size())
    return createStringError(make_error_code(std::errc::protocol_error),
                             "requested size too large in allocator");
  return ArrayRef<char>{Impl->File.getRegion().data() + Offset.get(), Size};
}

MutableArrayRef<uint8_t> OnDiskDataAllocator::getUserHeader() const {
  return Impl->Store.getUserHeader();
}

size_t OnDiskDataAllocator::size() const { return Impl->File.size(); }
size_t OnDiskDataAllocator::capacity() const {
  return Impl->File.getRegion().size();
}

OnDiskDataAllocator::OnDiskDataAllocator(std::unique_ptr<ImplType> Impl)
    : Impl(std::move(Impl)) {}

#else // !LLVM_ENABLE_ONDISK_CAS

struct OnDiskDataAllocator::ImplType {};

Expected<OnDiskDataAllocator> OnDiskDataAllocator::create(
    const Twine &Path, const Twine &TableName, uint64_t MaxFileSize,
    std::optional<uint64_t> NewFileInitialSize, uint32_t UserHeaderSize,
    function_ref<void(void *)> UserHeaderInit) {
  return createStringError(make_error_code(std::errc::not_supported),
                           "OnDiskDataAllocator is not supported");
}

Expected<OnDiskDataAllocator::OnDiskPtr>
OnDiskDataAllocator::allocate(size_t Size) {
  return createStringError(make_error_code(std::errc::not_supported),
                           "OnDiskDataAllocator is not supported");
}

Expected<ArrayRef<char>> OnDiskDataAllocator::get(FileOffset Offset,
                                                  size_t Size) const {
  return createStringError(make_error_code(std::errc::not_supported),
                           "OnDiskDataAllocator is not supported");
}

MutableArrayRef<uint8_t> OnDiskDataAllocator::getUserHeader() const {
  return {};
}

size_t OnDiskDataAllocator::size() const { return 0; }
size_t OnDiskDataAllocator::capacity() const { return 0; }

#endif // LLVM_ENABLE_ONDISK_CAS

OnDiskDataAllocator::OnDiskDataAllocator(OnDiskDataAllocator &&RHS) = default;
OnDiskDataAllocator &
OnDiskDataAllocator::operator=(OnDiskDataAllocator &&RHS) = default;
OnDiskDataAllocator::~OnDiskDataAllocator() = default;
