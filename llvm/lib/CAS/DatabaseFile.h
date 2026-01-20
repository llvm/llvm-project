//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file declares the common interface for a DatabaseFile that is used to
/// implement OnDiskCAS.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CAS_DATABASEFILE_H
#define LLVM_LIB_CAS_DATABASEFILE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/CAS/MappedFileRegionArena.h"
#include "llvm/CAS/OnDiskCASLogger.h"
#include "llvm/Support/Error.h"

namespace llvm::cas::ondisk {

using MappedFileRegion = MappedFileRegionArena::RegionT;

/// Generic handle for a table.
///
/// Generic table header layout:
/// - 2-bytes: TableKind
/// - 2-bytes: TableNameSize
/// - 4-bytes: TableNameRelOffset (relative to header)
class TableHandle {
public:
  enum class TableKind : uint16_t {
    TrieRawHashMap = 1,
    DataAllocator = 2,
  };
  struct Header {
    TableKind Kind;
    uint16_t NameSize;
    int32_t NameRelOffset; ///< Relative to Header.
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
/// - Points at an arbitrary root table.
/// - Sets up a MappedFileRegionArena for allocation.
///
/// Top-level layout:
/// - 4-bytes: Magic
/// - 4-bytes: Version
/// - 8-bytes: RootTableOffset (16-bits: Kind; 48-bits: Offset)
/// - 8-bytes: BumpPtr from MappedFileRegionArena
class DatabaseFile {
public:
  static constexpr uint32_t getMagic() { return 0xDA7ABA53UL; }
  static constexpr uint32_t getVersion() { return 1UL; }
  struct Header {
    uint32_t Magic;
    uint32_t Version;
    std::atomic<int64_t> RootTableOffset;
  };

  const Header &getHeader() { return *H; }
  MappedFileRegionArena &getAlloc() { return Alloc; }
  MappedFileRegion &getRegion() { return Alloc.getRegion(); }

  /// Add a table. This is currently not thread safe and should be called inside
  /// NewDBConstructor.
  Error addTable(TableHandle Table);

  /// Find a table. May return null.
  std::optional<TableHandle> findTable(StringRef Name);

  /// Create the DatabaseFile at Path with Capacity.
  static Expected<DatabaseFile>
  create(const Twine &Path, uint64_t Capacity,
         std::shared_ptr<OnDiskCASLogger> Logger,
         function_ref<Error(DatabaseFile &)> NewDBConstructor);

  size_t size() const { return Alloc.size(); }

private:
  static Expected<DatabaseFile>
  get(std::unique_ptr<MappedFileRegionArena> Alloc) {
    if (Error E = validate(Alloc->getRegion()))
      return std::move(E);
    return DatabaseFile(std::move(Alloc));
  }

  static Error validate(MappedFileRegion &Region);

  DatabaseFile(MappedFileRegionArena &Alloc)
      : H(reinterpret_cast<Header *>(Alloc.data())), Alloc(Alloc) {}
  DatabaseFile(std::unique_ptr<MappedFileRegionArena> Alloc)
      : DatabaseFile(*Alloc) {
    OwnedAlloc = std::move(Alloc);
  }

  Header *H = nullptr;
  MappedFileRegionArena &Alloc;
  std::unique_ptr<MappedFileRegionArena> OwnedAlloc;
};

Error createTableConfigError(std::errc ErrC, StringRef Path,
                             StringRef TableName, const Twine &Msg);

Error checkTable(StringRef Label, size_t Expected, size_t Observed,
                 StringRef Path, StringRef TrieName);

} // namespace llvm::cas::ondisk

#endif
