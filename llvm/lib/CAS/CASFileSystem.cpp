//===- CASFileSystem.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/CASFileSystem.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/CAS/FileSystemCache.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/CAS/TreeSchema.h"
#include "llvm/Support/AlignOf.h"
#include "llvm/Support/Allocator.h"

using namespace llvm;
using namespace llvm::cas;

namespace {
class CASFileSystem : public ThreadSafeFileSystem {
  struct WorkingDirectoryType {
    FileSystemCache::DirectoryEntry *Entry;

    /// Mimics shell behaviour on directory changes. Not necessarily the same
    /// as \c Entry->getTreePath().
    std::string Path;
  };

public:
  using File = FileSystemCache::File;
  using Symlink = FileSystemCache::Symlink;
  using Directory = FileSystemCache::Directory;
  using DirectoryEntry = FileSystemCache::DirectoryEntry;

  class VFSFile; // Return type for vfs::FileSystem::openFileForRead().

  Error loadDirectory(DirectoryEntry &Entry);
  Error loadFile(DirectoryEntry &Entry);
  Error loadSymlink(DirectoryEntry &Entry);

  /// Look up a directory entry in the CAS, navigating trees and resolving
  /// symlinks in the parent path. If \p FollowSymlinks is true, also follows
  /// symlinks in the filename.
  Expected<DirectoryEntry *> lookupPath(StringRef Path,
                                        bool FollowSymlinks = true);

  ErrorOr<vfs::Status> status(const Twine &Path) final;
  ErrorOr<std::unique_ptr<vfs::File>> openFileForRead(const Twine &Path) final;

  Expected<const vfs::CachedDirectoryEntry *>
  getDirectoryEntry(const Twine &Path, bool FollowSymlinks) const final;

  vfs::directory_iterator dir_begin(const Twine &Dir,
                                    std::error_code &EC) final {
    auto IterOr = getDirectoryIterator(Dir);
    if (IterOr)
      return *IterOr;
    EC = IterOr.getError();
    return vfs::directory_iterator();
  }

  ErrorOr<vfs::directory_iterator> getDirectoryIterator(const Twine &Dir);

  std::error_code setCurrentWorkingDirectory(const Twine &Path) final;

  ErrorOr<std::string> getCurrentWorkingDirectory() const final {
    return WorkingDirectory.Path;
  }

  Error initialize(ObjectRef Root);

  CASFileSystem(std::shared_ptr<ObjectStore> DB, sys::path::Style PathStyle)
      : DB(*DB), OwnedDB(std::move(DB)), PathStyle(PathStyle) {}
  CASFileSystem(ObjectStore &DB, sys::path::Style PathStyle)
      : DB(DB), PathStyle(PathStyle) {}

  IntrusiveRefCntPtr<ThreadSafeFileSystem> createThreadSafeProxyFS() final {
    return makeIntrusiveRefCnt<CASFileSystem>(*this);
  }
  CASFileSystem(const CASFileSystem &FS) = default;

  ObjectStore &getCAS() const { return DB; }

private:
  ObjectStore &DB;
  std::shared_ptr<ObjectStore> OwnedDB;

  IntrusiveRefCntPtr<FileSystemCache> Cache;
  WorkingDirectoryType WorkingDirectory;
  sys::path::Style PathStyle;
};
} // namespace

class CASFileSystem::VFSFile : public vfs::File {
public:
  ErrorOr<vfs::Status> status() final { return Entry->getStatus(Name); }

  ErrorOr<std::string> getName() final { return Name; }

  /// Get the contents of the file as a \p MemoryBuffer.
  ErrorOr<std::unique_ptr<MemoryBuffer>> getBuffer(const Twine &Name, int64_t,
                                                   bool, bool) final {
    Expected<ObjectProxy> Object = DB.getProxy(*Entry->getRef());
    if (!Object)
      return errorToErrorCode(Object.takeError());
    assert(Object->getNumReferences() == 0 && "Expected a leaf node");
    SmallString<256> Storage;
    return Object->getMemoryBuffer(Name.toStringRef(Storage));
  }

  llvm::ErrorOr<std::optional<cas::ObjectRef>> getObjectRefForContent() final {
    return Entry->getRef();
  }

  /// Closes the file.
  std::error_code close() final { return std::error_code(); }

  VFSFile() = delete;
  explicit VFSFile(ObjectStore &DB, DirectoryEntry &Entry, StringRef Name)
      : DB(DB), Name(Name.str()), Entry(&Entry) {
    assert(Entry.isFile());
    assert(Entry.hasNode());
  }

private:
  ObjectStore &DB;
  std::string Name;
  DirectoryEntry *Entry;
};

Error CASFileSystem::initialize(ObjectRef Root) {
  Cache = makeIntrusiveRefCnt<FileSystemCache>(PathStyle);

  // Initial working directory is the root.
  StringRef path_separator = get_separator(PathStyle);
  WorkingDirectory.Entry = &Cache->getRoot(path_separator, Root);
  WorkingDirectory.Path = WorkingDirectory.Entry->getTreePath().str();

  // Load the root to confirm it's really a tree.
  return loadDirectory(*WorkingDirectory.Entry);
}

std::error_code CASFileSystem::setCurrentWorkingDirectory(const Twine &Path) {
  SmallString<128> Storage;
  StringRef CanonicalPath = FileSystemCache::canonicalizeWorkingDirectory(
      Path, PathStyle, WorkingDirectory.Path, Storage);

  // Read and cache all the symlinks in the path by looking it up. Return any
  // error encountered.
  Expected<DirectoryEntry *> ExpectedEntry = lookupPath(CanonicalPath);
  if (!ExpectedEntry)
    return errorToErrorCode(ExpectedEntry.takeError());

  WorkingDirectory.Path = CanonicalPath.str();
  WorkingDirectory.Entry = *ExpectedEntry;
  return std::error_code();
}

Error CASFileSystem::loadDirectory(DirectoryEntry &Parent) {
  Directory &D = Parent.asDirectory();
  if (D.isComplete())
    return Error::success();

  SmallString<128> Path = Parent.getTreePath();
  size_t ParentPathSize = Path.size();
  auto makeCachedEntry =
      [&](const NamedTreeEntry &NewEntry) -> DirectoryEntry & {
    Path.resize(ParentPathSize);
    sys::path::append(Path, PathStyle, NewEntry.getName());
    switch (NewEntry.getKind()) {
    case TreeEntry::Regular:
    case TreeEntry::Executable:
      return Cache->makeLazyFileAlreadyLocked(Parent, Path, NewEntry.getRef(),
                                              NewEntry.getKind() ==
                                                  TreeEntry::Executable);
    case TreeEntry::Symlink:
      return Cache->makeLazySymlinkAlreadyLocked(Parent, Path,
                                                 NewEntry.getRef());
    case TreeEntry::Tree:
      return Cache->makeDirectoryAlreadyLocked(Parent, Path, NewEntry.getRef());
    }
    llvm_unreachable("invalid tree type");
  };
  Expected<ObjectProxy> Object = DB.getProxy(*Parent.getRef());
  if (!Object)
    return Object.takeError();

  TreeSchema Schema(DB);
  if (!Schema.isNode(*Object))
    report_fatal_error(createStringError(
        inconvertibleErrorCode(),
        "invalid tree '" + Object->getID().toString() + "'"));

  // Lock and check for a race.
  Directory::Writer W(D);
  if (D.isComplete())
    return Error::success();

  Expected<TreeProxy> Tree = Schema.load(*Object);
  if (!Tree)
    return Tree.takeError();

  if (Error E = Tree->forEachEntry([&](const NamedTreeEntry &NewEntry) {
        D.add(makeCachedEntry(NewEntry));
        return Error::success();
      }))
    return E;
  D.IsComplete = true;
  return Error::success();
}

Error CASFileSystem::loadFile(DirectoryEntry &Entry) {
  assert(Entry.isFile());

  Expected<ObjectProxy> File = DB.getProxy(*Entry.getRef());
  if (!File)
    return File.takeError();

  Cache->finishLazyFile(Entry, File->getData().size());
  return Error::success();
}

Error CASFileSystem::loadSymlink(DirectoryEntry &Entry) {
  assert(Entry.isSymlink());

  Expected<ObjectProxy> File = DB.getProxy(*Entry.getRef());
  if (!File)
    return File.takeError();

  Cache->finishLazySymlink(Entry, File->getData());
  return Error::success();
}

ErrorOr<vfs::Status> CASFileSystem::status(const Twine &Path) {
  SmallString<128> Storage;
  StringRef PathRef = Path.toStringRef(Storage);

  // Lookup only returns an Error if there's a problem communicating with the
  // CAS, or there's data corruption.
  //
  // FIXME: Translate the error to a filesystem-like error to encapsulate the
  // user from CAS issues.
  Expected<DirectoryEntry *> ExpectedEntry = lookupPath(PathRef);
  if (!ExpectedEntry)
    return errorToErrorCode(ExpectedEntry.takeError());
  DirectoryEntry *Entry = *ExpectedEntry;

  if (!Entry->hasNode()) {
    assert(!Entry->isDirectory());
    if (Error E = Entry->isSymlink() ? loadSymlink(*Entry) : loadFile(*Entry))
      return errorToErrorCode(std::move(E));
  }

  return Entry->getStatus(PathRef);
}

Expected<const vfs::CachedDirectoryEntry *>
CASFileSystem::getDirectoryEntry(const Twine &Path, bool FollowSymlinks) const {
  SmallString<128> Storage;
  StringRef PathRef = Path.toStringRef(Storage);

  // It's not a const operation, but it's thread-safe.
  return const_cast<CASFileSystem *>(this)->lookupPath(PathRef, FollowSymlinks);
}

ErrorOr<std::unique_ptr<vfs::File>>
CASFileSystem::openFileForRead(const Twine &Path) {
  SmallString<128> Storage;
  StringRef PathRef = Path.toStringRef(Storage);

  Expected<DirectoryEntry *> ExpectedEntry = lookupPath(PathRef);
  if (!ExpectedEntry)
    return errorToErrorCode(ExpectedEntry.takeError());

  DirectoryEntry *Entry = *ExpectedEntry;
  if (!Entry->isFile())
    return std::errc::invalid_argument;

  if (!Entry->hasNode())
    if (Error E = loadFile(*Entry))
      return errorToErrorCode(std::move(E));

  return std::make_unique<VFSFile>(DB, *Entry, PathRef);
}

ErrorOr<vfs::directory_iterator>
CASFileSystem::getDirectoryIterator(const Twine &Path) {
  SmallString<128> Storage;
  StringRef PathRef = Path.toStringRef(Storage);

  Expected<DirectoryEntry *> ExpectedEntry = lookupPath(PathRef);
  if (!ExpectedEntry)
    return errorToErrorCode(ExpectedEntry.takeError());
  DirectoryEntry *Entry = *ExpectedEntry;

  if (!Entry->isDirectory())
    return std::errc::not_a_directory;

  if (Error E = loadDirectory(*Entry))
    return errorToErrorCode(std::move(E));

  return Cache->getCachedVFSDirIter(
      Entry->asDirectory(), [this](StringRef Path) { return lookupPath(Path); },
      WorkingDirectory.Path, PathRef);
}

namespace {
class DiscoveryInstanceImpl final : public FileSystemCache::DiscoveryInstance {
public:
  DiscoveryInstanceImpl(CASFileSystem &FS) : FS(FS) {}
  ~DiscoveryInstanceImpl() {}

private:
  using DirectoryEntry = FileSystemCache::DirectoryEntry;

  Expected<DirectoryEntry *> requestDirectoryEntry(DirectoryEntry &Parent,
                                                   StringRef Name) override {
    if (Parent.asDirectory().isComplete())
      return errorCodeToError(
          std::make_error_code(std::errc::no_such_file_or_directory));

    if (Error E = FS.loadDirectory(Parent))
      return std::move(E);

    CASFileSystem::Directory &D = Parent.asDirectory();
    assert(D.isComplete() && "Loaded directory should be complete");
    if (DirectoryEntry *Entry = D.lookup(Name))
      return Entry;
    return errorCodeToError(
        std::make_error_code(std::errc::no_such_file_or_directory));
  }
  Error requestSymlinkTarget(DirectoryEntry &Symlink) override {
    return FS.loadSymlink(Symlink);
  }

private:
  CASFileSystem &FS;
};
} // end anonymous namespace

Expected<CASFileSystem::DirectoryEntry *>
CASFileSystem::lookupPath(StringRef Path, bool FollowSymlinks) {
  DiscoveryInstanceImpl DI(*this);
  return Cache->lookupPath(DI, Path, *WorkingDirectory.Entry, FollowSymlinks);
}

static Expected<std::unique_ptr<CASFileSystem>>
initializeCASFileSystem(std::unique_ptr<CASFileSystem> FS,
                        const CASID &RootID) {
  std::optional<ObjectRef> Root = FS->getCAS().getReference(RootID);
  if (!Root)
    return createStringError(inconvertibleErrorCode(),
                             "cannot get reference to root FS");
  if (Error E = FS->initialize(*Root))
    return std::move(E);
  return std::move(FS);
}

Expected<std::unique_ptr<vfs::FileSystem>>
cas::createCASFileSystem(std::shared_ptr<ObjectStore> DB,
                         const CASID &RootID,
                         sys::path::Style PathStyle) {
  return initializeCASFileSystem(
      std::make_unique<CASFileSystem>(std::move(DB), PathStyle),
      RootID);
}

Expected<std::unique_ptr<vfs::FileSystem>>
cas::createCASFileSystem(ObjectStore &DB, const CASID &RootID,
                         sys::path::Style PathStyle) {
  return initializeCASFileSystem(
      std::make_unique<CASFileSystem>(DB, PathStyle), RootID);
}
