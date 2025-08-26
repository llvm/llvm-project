//===- CachingOnDiskFileSystem.cpp ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/CachingOnDiskFileSystem.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/CAS/FileSystemCache.h"
#include "llvm/CAS/HierarchicalTreeBuilder.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/Config/config.h"
#include "llvm/Support/FileSystem.h"
#include <mutex>

using namespace llvm;
using namespace llvm::cas;

const char CachingOnDiskFileSystem::ID = 0;
void CachingOnDiskFileSystem::anchor() {}

namespace {

class CachingOnDiskFileSystemImpl final : public CachingOnDiskFileSystem {
  struct WorkingDirectoryType {
    FileSystemCache::DirectoryEntry *Entry;

    /// Mimics shell behaviour on directory changes. Not necessarily the same
    /// as \c Entry->getTreePath().
    std::string Path;
  };

  class VFSFile; // Return type for vfs::FileSystem::openFileForRead().
  class TreeBuilder;

public:
  using File = FileSystemCache::File;
  using Symlink = FileSystemCache::Symlink;
  using Directory = FileSystemCache::Directory;
  using DirectoryEntry = FileSystemCache::DirectoryEntry;

  Expected<const vfs::CachedDirectoryEntry *>
  getDirectoryEntry(const Twine &Path, bool FollowSymlinks) const override;

  /// Look up a directory entry in the CAS, navigating trees and resolving
  /// symlinks in the parent path. If \p FollowSymlinks is true, also follows
  /// symlinks in the filename.
  ///
  /// If \p TrackNonRealPathEntries is given, the links in the symlink chain and
  /// the final path are passed to it as the search progresses.
  Expected<DirectoryEntry *> lookupPath(
      StringRef Path, bool FollowSymlinks = true, bool LookupOnDisk = true,
      bool ForceDisableTracking = false, bool NeedsContent = true,
      function_ref<void(DirectoryEntry &)> TrackNonRealPathEntries = nullptr);

  DirectoryEntry *makeDirectory(DirectoryEntry &Parent, StringRef TreePath);

  Expected<DirectoryEntry *> makeSymlink(DirectoryEntry &Parent,
                                         StringRef TreePath);
  Expected<DirectoryEntry *>
  makeSymlinkTo(DirectoryEntry &Parent, StringRef TreePath, StringRef Target);

  Expected<DirectoryEntry *> makeFile(DirectoryEntry &Parent,
                                      StringRef TreePath, sys::fs::file_t F,
                                      sys::fs::file_status Status);

  /// Create an entry for \p TreePath, which is contained in \p Parent. If
  /// \p KnownStatus is provided, it is used to determine which kind of entry
  /// to create (file, directory, symlink); otherwise, it is determined using
  /// status(TreePath, /*FollowSymlinks=*/false).
  Expected<DirectoryEntry *>
  makeEntry(DirectoryEntry &Parent, StringRef TreePath,
            std::optional<sys::fs::file_status> KnownStatus);

  /// Preload the real path for \p Remaining, relative to \p From.
  /// \returns The real path entry if it can be computed, an error if the path
  ///          cannot be accessed, or nullptr if the path was accessed but there
  ///          was a subsequent filesystem modification.
  Expected<FileSystemCache::DirectoryEntry *>
  preloadRealPath(DirectoryEntry &From, StringRef Remaining);

  ErrorOr<vfs::Status> statusAndFileID(const Twine &Path,
                                       std::optional<CASID> &FileID) final;
  ErrorOr<vfs::Status> status(const Twine &Path) final;
  bool exists(const Twine &Path) final;
  Expected<std::unique_ptr<CASBackedFile>>
  openCASBackedFileForReadImpl(const Twine &Path) final;
  vfs::directory_iterator dir_begin(const Twine &Dir,
                                    std::error_code &EC) final {
    auto IterOr = getDirectoryIterator(Dir);
    if (IterOr)
      return *IterOr;
    EC = IterOr.getError();
    return vfs::directory_iterator();
  }

  std::error_code getRealPath(const Twine &Path,
                              SmallVectorImpl<char> &Output) final;
  ErrorOr<vfs::directory_iterator> getDirectoryIterator(const Twine &Dir);

  std::error_code setCurrentWorkingDirectory(const Twine &Path) final;

  ErrorOr<std::string> getCurrentWorkingDirectory() const final {
    return WorkingDirectory.Path;
  }

  void trackNewAccesses() final;
  std::error_code excludeFromTracking(const Twine &Path) final;
  Expected<ObjectProxy>
  createTreeFromNewAccesses(RemapPathCallback RemapPath) final;
  Expected<ObjectProxy> createTreeFromAllAccesses() final;
  std::unique_ptr<CachingOnDiskFileSystem::TreeBuilder>
  createTreeBuilder() final;
  Error pushCachedPath(const Twine &Path, TreeBuilder &State);

  IntrusiveRefCntPtr<CachingOnDiskFileSystem> createProxyFS() final {
    return makeIntrusiveRefCnt<CachingOnDiskFileSystemImpl>(*this);
  }

  CachingOnDiskFileSystemImpl(std::shared_ptr<ObjectStore> DB)
      : CachingOnDiskFileSystem(std::move(DB)) {
    initializeWorkingDirectory();
  }
  CachingOnDiskFileSystemImpl(ObjectStore &DB) : CachingOnDiskFileSystem(DB) {
    initializeWorkingDirectory();
  }

  CachingOnDiskFileSystemImpl(const CachingOnDiskFileSystemImpl &Proxy)
      : CachingOnDiskFileSystem(Proxy), Cache(Proxy.Cache),
        WorkingDirectory(Proxy.WorkingDirectory) {}

  bool isTrackingAccess() const {
    std::lock_guard<std::mutex> Lock(TrackedAccessesMutex);
    return !TrackedAccesses.empty();
  }

  /// Track an access to \p Entry.
  ///
  /// \param Entry The path to track.
  /// \param NeedContent Whether this access required the contents of \p Entry.
  /// If no accesses need content, it can be canonicalized away in the resulting
  /// \c CASFileSystem.
  void trackAccess(const DirectoryEntry &Entry, bool NeedContent = true) {
    std::lock_guard<std::mutex> Lock(TrackedAccessesMutex);
    if (!TrackedAccesses.empty()) {
      auto &State = TrackedAccesses.back()[&Entry];
      State.NeedContent |= NeedContent;
    }
  }

private:
  void initializeWorkingDirectory();

  /// An empty CAS object.
  ObjectRef getEmptyRef() {
    if (!EmptyRef)
      if (auto Err = DB.store({}, {}).moveInto(EmptyRef))
        llvm::report_fatal_error(std::move(Err));
    return *EmptyRef;
  }

  struct TrackingState {
    uint8_t NeedContent : 1;
    TrackingState() : NeedContent(false) {}
  };

  // Cached stats. Useful for tracking everything that has been stat'ed.
  SmallVector<DenseMap<const DirectoryEntry *, TrackingState>> TrackedAccesses;
  SmallVector<DenseSet<const DirectoryEntry *>> ExcludedAccesses;
  mutable std::mutex TrackedAccessesMutex;

  IntrusiveRefCntPtr<FileSystemCache> Cache;
  WorkingDirectoryType WorkingDirectory;

  std::optional<ObjectRef> EmptyRef;
};

class DiscoveryInstanceImpl final : public FileSystemCache::DiscoveryInstance {
public:
  using DirectoryEntry = FileSystemCache::DirectoryEntry;
  DiscoveryInstanceImpl(
      CachingOnDiskFileSystemImpl &FS,
      function_ref<void(DirectoryEntry &)> TrackNonRealPathEntries,
      bool IsTrackingStats, bool LookupOnDisk);
  ~DiscoveryInstanceImpl();

private:
  Expected<DirectoryEntry *> requestDirectoryEntry(DirectoryEntry &Parent,
                                                   StringRef Name) override;
  Error requestSymlinkTarget(DirectoryEntry &Symlink) override;
  Error preloadRealPath(DirectoryEntry &Parent, StringRef Remaining) override;
  void trackNonRealPathEntry(DirectoryEntry &Entry) override;

private:
  CachingOnDiskFileSystemImpl &FS;
  function_ref<void(DirectoryEntry &)> TrackNonRealPathEntries;
  DirectoryEntry *RealPath = nullptr;
  bool IsTrackingStats;
  bool LookupOnDisk;
  bool ComputedRealPath = false;
};
} // namespace

CachingOnDiskFileSystem::CachingOnDiskFileSystem(
    std::shared_ptr<ObjectStore> DB)
    : DB(*DB), OwnedDB(std::move(DB)) {}

CachingOnDiskFileSystem::CachingOnDiskFileSystem(ObjectStore &DB) : DB(DB) {}

class CachingOnDiskFileSystemImpl::VFSFile final : public CASBackedFile {
public:
  ErrorOr<vfs::Status> status() final { return Entry->getStatus(Name); }

  ErrorOr<std::string> getName() final { return Name; }

  /// Get the contents of the file as a \p MemoryBuffer.
  ErrorOr<std::unique_ptr<MemoryBuffer>> getBuffer(const Twine &RequestedName,
                                                   int64_t, bool, bool) final {
    Expected<ObjectProxy> Object = DB.getProxy(*Entry->getRef());
    if (!Object)
      return errorToErrorCode(Object.takeError());
    assert(Object->getNumReferences() == 0 && "Expected a leaf node");
    SmallString<256> Storage;
    return Object->getMemoryBuffer(RequestedName.toStringRef(Storage));
  }

  cas::ObjectRef getObjectRefForContent() final { return *Entry->getRef(); }

  /// Closes the file.
  std::error_code close() final { return std::error_code(); }

  VFSFile() = delete;
  explicit VFSFile(ObjectStore &DB, DirectoryEntry &Entry, StringRef Name)
      : DB(DB), Entry(&Entry), Name(Name.str()) {}

private:
  ObjectStore &DB;
  DirectoryEntry *Entry;
  std::string Name;
};

void CachingOnDiskFileSystemImpl::initializeWorkingDirectory() {
  Cache = makeIntrusiveRefCnt<FileSystemCache>(sys::path::Style::native);

  // Start with root, and then initialize the current working directory to
  // match process state, ignoring errors if there's a problem.
#ifndef _WIN32
  static std::string RootPath = get_separator(sys::path::Style::native).str();
#else
  static std::string RootPath = const_cast<const char *>(getenv("SYSTEMDRIVE")) +
      get_separator(sys::path::Style::native).str();
#endif
  WorkingDirectory.Entry = &Cache->getRoot(RootPath);
  WorkingDirectory.Path = WorkingDirectory.Entry->getTreePath().str();

  SmallString<128> CWD;
  if (std::error_code EC = llvm::sys::fs::current_path(CWD)) {
    (void)EC;
    return;
  }
  std::error_code EC = setCurrentWorkingDirectory(CWD);
  (void)EC;
}

std::error_code
CachingOnDiskFileSystemImpl::setCurrentWorkingDirectory(const Twine &Path) {
  PathStorage PathStorage(Path);
  StringRef CanonicalPath = FileSystemCache::canonicalizeWorkingDirectory(
      sys::path::Style::native, WorkingDirectory.Path, PathStorage.Storage);

  // Read and cache all the symlinks in the path by looking it up. Return any
  // error encountered.
  Expected<DirectoryEntry *> ExpectedEntry = lookupPath(CanonicalPath);
  if (!ExpectedEntry)
    return errorToErrorCode(ExpectedEntry.takeError());

  WorkingDirectory.Path = CanonicalPath.str();
  WorkingDirectory.Entry = *ExpectedEntry;
  return std::error_code();
}

FileSystemCache::DirectoryEntry *
CachingOnDiskFileSystemImpl::makeDirectory(DirectoryEntry &Parent,
                                           StringRef TreePath) {
  PathStorage TreePathStorage(TreePath);
  return &Cache->makeDirectory(Parent, TreePathStorage.Path);
}

#if defined(HAVE_UNISTD_H)
// FIXME: sink into llvm::sys::fs?
#include <unistd.h>
static Error readLink(const llvm::Twine &Path, SmallVectorImpl<char> &Dest) {
  SmallString<128> PathStorage;
  StringRef P = Path.toNullTerminatedStringRef(PathStorage);
  char TargetBuffer[PATH_MAX] = {0};
  int TargetLength = ::readlink(P.data(), TargetBuffer, sizeof(TargetBuffer));
  if (TargetLength == -1)
    return errorCodeToError(std::error_code(errno, std::generic_category()));
  Dest.assign(TargetBuffer, TargetBuffer + TargetLength);
  return Error::success();
}
#else
// Use real_path implementation for platforms that doesn't have readlink().
static Error readLink(const llvm::Twine &Path, SmallVectorImpl<char> &Dest) {
  std::error_code EC = sys::fs::real_path(Path, Dest);
  return errorCodeToError(EC);
}
#endif

Expected<FileSystemCache::DirectoryEntry *>
CachingOnDiskFileSystemImpl::makeSymlink(DirectoryEntry &Parent,
                                         StringRef TreePath) {
  PathStorage TreePathStorage(TreePath);
  SmallString<128> Target;
  if (auto Err = readLink(TreePathStorage.Path, Target))
    return std::move(Err);
  return makeSymlinkTo(Parent, TreePathStorage.Path, Target);
}

Expected<FileSystemCache::DirectoryEntry *>
CachingOnDiskFileSystemImpl::makeSymlinkTo(DirectoryEntry &Parent,
                                           StringRef TreePath,
                                           StringRef Target) {
  PathStorage TreePathStorage(TreePath);
  PathStorage TargetStorage(Target);
  Expected<ObjectRef> Node = DB.storeFromString({}, TargetStorage.Path);
  if (!Node)
    return Node.takeError();
  return &Cache->makeSymlink(Parent, TreePathStorage.Path, *Node,
                             TargetStorage.Path);
}

Expected<FileSystemCache::DirectoryEntry *>
CachingOnDiskFileSystemImpl::makeFile(DirectoryEntry &Parent,
                                      StringRef TreePath, sys::fs::file_t F,
                                      sys::fs::file_status Status) {
  PathStorage TreePathStorage(TreePath);
  Expected<ObjectRef> Node = DB.storeFromOpenFile(F, Status);
  if (!Node)
    return Node.takeError();

  // Load back the data from CAS since we stored from openFile buffer.
  Expected<ObjectProxy> Handle = DB.getProxy(*Node);
  if (!Handle)
    return Handle.takeError();
  // Do not trust Status.size() in case the file is volatile.
  return &Cache->makeFile(Parent, TreePathStorage.Path, *Node,
                          Handle->getData().size(),
                          Status.permissions() & sys::fs::perms::owner_exe);
}

Expected<FileSystemCache::DirectoryEntry *>
CachingOnDiskFileSystemImpl::makeEntry(
    DirectoryEntry &Parent, StringRef TreePath,
    std::optional<sys::fs::file_status> KnownStatus) {
  assert(Parent.isDirectory() && "Expected a directory");
  PathStorage TreePathStorage(TreePath);

  // lstat is extremely slow...
  sys::fs::file_status Status;
  if (KnownStatus) {
    Status = std::move(*KnownStatus);
  } else if (auto EC = sys::fs::status(TreePathStorage.Path, Status,
                                       /*follow=*/false)) {
    return errorCodeToError(EC);
  }

  if (Status.type() == sys::fs::file_type::directory_file)
    return makeDirectory(Parent, TreePathStorage.Path);

  if (Status.type() == sys::fs::file_type::symlink_file)
    return makeSymlink(Parent, TreePathStorage.Path);

  auto F = sys::fs::openNativeFile(TreePathStorage.Path,
      sys::fs::CD_OpenExisting, sys::fs::FA_Read, sys::fs::OF_None);
  if (!F)
    return F.takeError();

  auto CloseOnExit = make_scope_exit([&F]() { sys::fs::closeFile(*F); });
  return makeFile(Parent, TreePathStorage.Path, *F, Status);
}

ErrorOr<vfs::Status>
CachingOnDiskFileSystemImpl::statusAndFileID(const Twine &Path,
                                             std::optional<CASID> &FileID) {
  FileID = std::nullopt;
  PathStorage PathStorage(Path);
  StringRef PathRef = PathStorage.Path;

  // Lookup only returns an Error if there's a problem communicating with the
  // CAS, or there's data corruption.
  //
  // FIXME: Translate the error to a filesystem-like error to encapsulate the
  // user from CAS issues.
  Expected<DirectoryEntry *> ExpectedEntry = lookupPath(PathRef);
  if (!ExpectedEntry)
    return errorToErrorCode(ExpectedEntry.takeError());

  // Errors indicate a broken symlink.
  DirectoryEntry *Entry = *ExpectedEntry;
  ErrorOr<vfs::Status> StatusOrErr = Entry->getStatus(PathRef);
  if (!StatusOrErr)
    return StatusOrErr.getError();
  if (Entry->isFile())
    FileID = DB.getID(*Entry->getRef());
  return StatusOrErr;
}

Expected<const vfs::CachedDirectoryEntry *>
CachingOnDiskFileSystemImpl::getDirectoryEntry(const Twine &Path,
                                               bool FollowSymlinks) const {
  PathStorage PathStorage(Path);

  // It's not a const operation, but it's thread-safe.
  return const_cast<CachingOnDiskFileSystemImpl *>(this)->lookupPath(
      PathStorage.Path, FollowSymlinks);
}

std::error_code
CachingOnDiskFileSystemImpl::getRealPath(const Twine &Path,
                                         SmallVectorImpl<char> &Output) {
  // We can get the real path, but it's not a const operation.
  const vfs::CachedDirectoryEntry *Entry = nullptr;
  if (Error E =
          getDirectoryEntry(Path, /*FollowSymlinks=*/true).moveInto(Entry))
    return errorToErrorCode(std::move(E));

  StringRef TreePath = Entry->getTreePath();
  Output.resize(TreePath.size());
  llvm::copy(TreePath, Output.begin());
  return std::error_code();
}

ErrorOr<vfs::Status> CachingOnDiskFileSystemImpl::status(const Twine &Path) {
  std::optional<CASID> IgnoredID;
  return statusAndFileID(Path, IgnoredID);
}

bool CachingOnDiskFileSystemImpl::exists(const Twine &Path) {
  SmallString<128> Storage;
  StringRef PathRef = Path.toStringRef(Storage);
  Expected<DirectoryEntry *> ExpectedEntry =
      lookupPath(PathRef, /*FollowSymlinks=*/true, /*LookupOnDisk=*/true,
                 /*ForceDisableTracking=*/false, /*NeedsContent=*/false);
  if (!ExpectedEntry) {
    llvm::consumeError(ExpectedEntry.takeError());
    return false;
  }
  return true;
}

Expected<std::unique_ptr<CASBackedFile>>
CachingOnDiskFileSystemImpl::openCASBackedFileForReadImpl(const Twine &Path) {
  PathStorage PathStorage(Path);
  StringRef PathRef = PathStorage.Path;

  Expected<DirectoryEntry *> ExpectedEntry = lookupPath(PathRef);
  if (!ExpectedEntry)
    return ExpectedEntry.takeError();

  DirectoryEntry *Entry = *ExpectedEntry;
  if (!Entry->isFile())
    return createFileError(Path,
                           std::make_error_code(std::errc::invalid_argument));

  return std::make_unique<VFSFile>(DB, *Entry, PathRef);
}

ErrorOr<vfs::directory_iterator>
CachingOnDiskFileSystemImpl::getDirectoryIterator(const Twine &Path) {
  PathStorage PathStorage(Path);
  StringRef PathRef = PathStorage.Path;

  Expected<DirectoryEntry *> ExpectedEntry = lookupPath(PathRef);
  if (!ExpectedEntry)
    return errorToErrorCode(ExpectedEntry.takeError());

  DirectoryEntry *Entry = *ExpectedEntry;
  if (!Entry->isDirectory())
    return std::errc::not_a_directory;

  // Walk the directory on-disk to discover entries.
  std::error_code EC;
  SmallVector<std::string> TreePaths;
  for (sys::fs::directory_iterator I(Entry->getTreePath(), EC), E;
       !EC && I != E; I.increment(EC))
    TreePaths.emplace_back(I->path());
  if (EC)
    return EC;

  // Cache all the entries.
  Directory &D = Entry->asDirectory();

  // Filter out names that we know about.
  {
    Directory::Reader R(D);
    TreePaths.erase(llvm::remove_if(TreePaths,
                                    [&D](StringRef TreePath) {
                                      return D.lookup(
                                          sys::path::filename(TreePath));
                                    }),
                    TreePaths.end());
  }

  for (StringRef TreePath : TreePaths)
    if (Error E = makeEntry(*Entry, TreePath, /*KnownStatus=*/std::nullopt)
                      .takeError())
      return errorToErrorCode(std::move(E));

  return Cache->getCachedVFSDirIter(
      D, [this](StringRef Path) { return lookupPath(Path); },
      WorkingDirectory.Path, PathRef);
}

Expected<FileSystemCache::DirectoryEntry *>
CachingOnDiskFileSystemImpl::preloadRealPath(DirectoryEntry &From,
                                             StringRef Remaining) {
  PathStorage RemainingStorage(Remaining);
  SmallString<256> ExpectedRealPath;
  ExpectedRealPath = From.getTreePath();
  sys::path::append(ExpectedRealPath, RemainingStorage.Path);

  // Most paths don't exist. Start with a stat. Profiling says this is faster
  // on Darwin when running clang-scan-deps (looks like allocation traffic in
  // ::open on stat failures). This may be platform- or even
  // OS-version-dependent though.
  {
    sys::fs::file_status Status;
    if (std::error_code EC = sys::fs::status(ExpectedRealPath, Status))
      return errorCodeToError(EC);
    if (!sys::fs::exists(Status))
      return errorCodeToError(
          std::make_error_code(std::errc::no_such_file_or_directory));

    // Don't reuse Status below since there could be a race.
  }

  SmallString<256> RealPath;
  if (std::error_code EC = sys::fs::real_path(ExpectedRealPath, RealPath,
      /* expand_tilde */false))
    return errorCodeToError(EC);

  FileSystemCache::LookupPathState State(Cache->getPathStyle(),
      Cache->getRoot(Cache->getRootPathFor(RealPath)), RealPath);

  // Advance through the cached directories. Note: no need to pass through
  // TrackNonRealPathEntries because we're navigating a real path.
  StringRef ExpectedPrefix =
      StringRef(ExpectedRealPath).drop_back(RemainingStorage.Path.size());
#ifndef _WIN32
  if (RealPath.str().starts_with(ExpectedPrefix))
#else
  if (RealPath.str().starts_with_insensitive(ExpectedPrefix))
#endif
    State = FileSystemCache::LookupPathState(
        Cache->getPathStyle(), From, RealPath.substr(ExpectedPrefix.size()));
  else
    State = Cache->lookupRealPathPrefixFromCached(
        State, /*TrackNonRealPathEntries=*/nullptr);

  // Real path is already fully cached.
  if (State.Remaining.empty())
    return State.Entry;

  // All but the last component must be directories.
  while (!State.AfterName.empty()) {
    DirectoryEntry &Entry = Cache->makeDirectory(
        *State.Entry, RealPath.substr(0, State.Name.end() - RealPath.begin()));

    // If we don't get back a directory, the disk state must have changed and
    // another thread raced. Give up on this endeavour.
    if (!Entry.isDirectory())
      return nullptr;

    State.advance(Entry);
  }

  assert(!State.Name.empty());

  // Skip all errors from here out. This is just priming the cache.

  // Check if it's a directory or not before opening because
  // openNativeFileForRead will fail on directories on Windows.
  sys::fs::file_status Status;
  if (/*std::error_code EC =*/sys::fs::status(ExpectedRealPath, Status))
    return nullptr;
  const bool IsDir = sys::fs::is_directory(Status);

  if (IsDir)
    return makeDirectory(*State.Entry, RealPath);

  auto FD = sys::fs::openNativeFileForRead(ExpectedRealPath, sys::fs::OF_None,
                                           /*RealPath=*/nullptr);
  if (!FD) {
    llvm::consumeError(FD.takeError());
    return nullptr;
  }
  auto CloseOnExit = make_scope_exit([&FD]() { sys::fs::closeFile(*FD); });

  auto F = makeFile(*State.Entry, RealPath, *FD, Status);
  if (F)
    return *F;
  llvm::consumeError(F.takeError());
  return nullptr;
}

Expected<FileSystemCache::DirectoryEntry *>
CachingOnDiskFileSystemImpl::lookupPath(
    StringRef Path, bool FollowSymlinks, bool LookupOnDisk,
    bool ForceDisableTracking, bool NeedsContent,
    function_ref<void(FileSystemCache::DirectoryEntry &)>
        TrackNonRealPathEntries) {
  PathStorage PathStorage(Path);
  bool IsTrackingStats = ForceDisableTracking ? false : isTrackingAccess();
  DiscoveryInstanceImpl DI(*this, TrackNonRealPathEntries, IsTrackingStats,
                           LookupOnDisk);
  Expected<DirectoryEntry *> ExpectedEntry =
      Cache->lookupPath(DI, PathStorage.Path, *WorkingDirectory.Entry,
                        FollowSymlinks);
  if (IsTrackingStats && ExpectedEntry && *ExpectedEntry)
    trackAccess(**ExpectedEntry, NeedsContent);
  return ExpectedEntry;
}

static TreeEntry::EntryKind
getTreeEntryKind(const FileSystemCache::DirectoryEntry &Entry) {
  switch (Entry.getKind()) {
  case FileSystemCache::DirectoryEntry::Directory:
    return TreeEntry::Tree;
  case FileSystemCache::DirectoryEntry::Symlink:
    return TreeEntry::Symlink;
  case FileSystemCache::DirectoryEntry::Regular:
    return TreeEntry::Regular;
  case FileSystemCache::DirectoryEntry::Executable:
    return TreeEntry::Executable;
  }
}

/// Push an entry to the builder, doing nothing (but returning false) for
/// directories.
static void pushEntryToBuilder(const ObjectStore &DB,
                               HierarchicalTreeBuilder &Builder,
                               const FileSystemCache::DirectoryEntry &Entry) {
  assert(!Entry.isDirectory());

  if (Entry.isSymlink()) {
    Builder.push(*Entry.getRef(), TreeEntry::Symlink, Entry.getTreePath());
    return;
  }

  Builder.push(*Entry.getRef(), getTreeEntryKind(Entry), Entry.getTreePath());
}

void CachingOnDiskFileSystemImpl::trackNewAccesses() {
  std::lock_guard<std::mutex> Lock(TrackedAccessesMutex);
  TrackedAccesses.emplace_back();
  TrackedAccesses.back().reserve(128); // Seed with a bit of runway.
  ExcludedAccesses.emplace_back();
}

std::error_code
CachingOnDiskFileSystemImpl::excludeFromTracking(const Twine &Path) {
  PathStorage PathStorage(Path);
  StringRef PathRef = PathStorage.Path;
  DirectoryEntry *Entry = nullptr;
  if (auto Err =
          lookupPath(PathRef, /*FollowSymlinks=*/true, /*LookupOnDisk=*/true,
                     /*ForceDisableTracking=*/true)
              .moveInto(Entry))
    return errorToErrorCode(std::move(Err));

  std::lock_guard<std::mutex> Lock(TrackedAccessesMutex);
  assert(!ExcludedAccesses.empty() && "missing tracking scope");
  if (!ExcludedAccesses.empty())
    ExcludedAccesses.back().insert(Entry);
  return std::error_code();
}

Expected<ObjectProxy> CachingOnDiskFileSystemImpl::createTreeFromNewAccesses(
    RemapPathCallback RemapPath) {
  DenseMap<const DirectoryEntry *, TrackingState> TrackedAccesses;
  DenseSet<const DirectoryEntry *> ExcludedAccesses;
  {
    std::lock_guard<std::mutex> Lock(TrackedAccessesMutex);
    assert(!this->TrackedAccesses.empty() && !this->ExcludedAccesses.empty() &&
           "createTreeFromNewAccesses must be paired with trackNewAccesses");
    TrackedAccesses = this->TrackedAccesses.pop_back_val();
    ExcludedAccesses = this->ExcludedAccesses.pop_back_val();
  }

  auto IsExcluded = [&](const DirectoryEntry *Entry) {
    if (ExcludedAccesses.empty())
      return false;
    do {
      if (ExcludedAccesses.contains(Entry))
        return true;
    } while ((Entry = Entry->getParent()));
    return false;
  };

  TreeSchema Schema(DB);
  if (TrackedAccesses.empty())
    return Schema.create();

  HierarchicalTreeBuilder Builder;
  for (auto &&[Entry, State] : TrackedAccesses) {
    if (IsExcluded(Entry))
      continue;

    SmallString<128> Storage;
    StringRef Path =
        RemapPath ? RemapPath(*Entry, Storage) : Entry->getTreePath();

    // FIXME: If Entry is a symbol link, the spelling of its target should be
    // remapped.
    if (Entry->isDirectory()) {
      Builder.pushDirectory(Path);
    } else {
      ObjectRef Ref = *Entry->getRef();
      // If the content is not needed, canonicalize as an empty file.
      if (Entry->isFile() && !State.NeedContent)
        Ref = getEmptyRef();
      Builder.push(Ref, getTreeEntryKind(*Entry), Path);
    }
  }

  return Builder.create(DB);
}

Expected<ObjectProxy> CachingOnDiskFileSystemImpl::createTreeFromAllAccesses() {
  std::unique_ptr<CachingOnDiskFileSystem::TreeBuilder> Builder =
      createTreeBuilder();

  // FIXME: don't know if we want this push to be recursive...
  if (Error E = Builder->push(get_separator(sys::path::Style::native)))
    return std::move(E);
  return Builder->create();
}

DiscoveryInstanceImpl::DiscoveryInstanceImpl(
    CachingOnDiskFileSystemImpl &FS,
    function_ref<void(DirectoryEntry &)> TrackNonRealPathEntries,
    bool IsTrackingStats, bool LookupOnDisk)
    : FS(FS), TrackNonRealPathEntries(TrackNonRealPathEntries),
      IsTrackingStats(IsTrackingStats), LookupOnDisk(LookupOnDisk) {}
DiscoveryInstanceImpl::~DiscoveryInstanceImpl() {}

Expected<FileSystemCache::DirectoryEntry *>
DiscoveryInstanceImpl::requestDirectoryEntry(DirectoryEntry &Parent,
                                             StringRef Name) {
  if (!LookupOnDisk)
    return nullptr;

  assert(Parent.isDirectory() && "Expected a directory");
  SmallString<256> Path(Parent.getTreePath());
  sys::path::append(Path, Name);

  // lstat is extremely slow...
  sys::fs::file_status Status;
  if (std::error_code EC = sys::fs::status(Path, Status, /*follow=*/false))
    return errorCodeToError(EC);

  DirectoryEntry *Next =
      RealPath ? RealPath->nextEntryAfterPrefix(Parent) : nullptr;
  if (!Next) {
    // We do not know the realpath, or it does not contain Parent, which
    // may indicate we are in the middle of looking up a path that contains a
    // symlink to an absolute path in the current or a subsequent component, or
    // that we hit F_GETPATH non-determinism for a hard link on Darwin. We need
    // to get the realpath of Parent + Name, which may be different from the
    // realpath of the path that's ultimately being looked up.
    auto RP = FS.preloadRealPath(Parent, Name);
    if (!RP)
      return RP.takeError();
    if (*RP)
      Next = (*RP)->nextEntryAfterPrefix(Parent);
  }
  if (Next) {
    StringRef NextName = Next->getName();
    if (Name == NextName)
      return Next;
    if (Name.equals_insensitive(NextName) || !isASCII(Name)) {
      // Might be a case-insensitive match, check if it's the same entity.
      // FIXME: put this unique id in the cache.
      sys::fs::file_status StatNext;
      if (std::error_code EC =
              sys::fs::status(Next->getTreePath(), StatNext, /*follow=*/false))
        return errorCodeToError(EC);
      if (Status.getUniqueID() == StatNext.getUniqueID()) {
        // Case-insensitive match! Create a fake symlink so that it will have
        // the correct realpath and uid.
        return FS.makeSymlinkTo(Parent, Path, NextName);
      }
    }
    // Fallthrough and create a new entry...
  }

  return FS.makeEntry(Parent, Path, Status);
}

Error DiscoveryInstanceImpl::requestSymlinkTarget(DirectoryEntry &Symlink) {
  assert(Symlink.hasNode());
  return Error::success();
}

Error DiscoveryInstanceImpl::preloadRealPath(DirectoryEntry &Parent,
                                             StringRef Remaining) {
  if (LookupOnDisk && !ComputedRealPath) {
    ComputedRealPath = true;
    auto Entry = FS.preloadRealPath(Parent, Remaining);
    if (Entry)
      RealPath = *Entry;
    return Entry.takeError();
  }
  return Error::success();
}

void DiscoveryInstanceImpl::trackNonRealPathEntry(DirectoryEntry &Entry) {
  if (TrackNonRealPathEntries)
    TrackNonRealPathEntries(Entry);
  if (IsTrackingStats)
    FS.trackAccess(Entry);
}

class CachingOnDiskFileSystemImpl::TreeBuilder final
    : public CachingOnDiskFileSystem::TreeBuilder {
public:
  /// Add \p Path to hierarchical tree-in-progress.
  ///
  /// If \p Path resolves to a symlink, its target is implicitly pushed as
  /// well.
  ///
  /// If \p Path resolves to a directory, the recursive directory contents
  /// will be pushed, implicitly pushing the targets of any contained symlinks.
  ///
  /// If \p Path does not exist, an error will be returned. If \p Path's parent
  /// path exists but the filename refers to a broken symlink, that is not an
  /// error; the symlink will be added without the target.
  Error push(const Twine &Path) final;

  Expected<ObjectProxy> create() final {
    return Builder.create(FS.getCAS());
  }

  // Push \p Entry directly to \a Builder, asserting that it's a symlink.
  void pushSymlink(const DirectoryEntry &Entry);

  // Push \p Entry to \a Builder if it's a file, to \a Worklist otherwise.
  void pushEntry(const DirectoryEntry &Entry);

  explicit TreeBuilder(CachingOnDiskFileSystemImpl &FS)
      : Builder(HierarchicalTreeBuilder()), FS(FS) {}
  HierarchicalTreeBuilder Builder;
  CachingOnDiskFileSystemImpl &FS;

  PathStorage Storage;
  SmallVector<const DirectoryEntry *> Worklist;
  DenseSet<const DirectoryEntry *> Seen;
};

std::unique_ptr<CachingOnDiskFileSystem::TreeBuilder>
CachingOnDiskFileSystemImpl::createTreeBuilder() {
  return std::make_unique<TreeBuilder>(*this);
}

void CachingOnDiskFileSystemImpl::TreeBuilder::pushSymlink(
    const DirectoryEntry &Entry) {
  assert(Entry.isSymlink());
  if (Seen.insert(&Entry).second)
    pushEntryToBuilder(FS.getCAS(), Builder, Entry);
}

void CachingOnDiskFileSystemImpl::TreeBuilder::pushEntry(
    const DirectoryEntry &Entry) {
  if (!Seen.insert(&Entry).second)
    return;
  if (Entry.isFile() || Entry.isSymlink())
    pushEntryToBuilder(FS.getCAS(), Builder, Entry);
  if (!Entry.isFile())
    Worklist.push_back(&Entry);
}

Error CachingOnDiskFileSystemImpl::TreeBuilder::push(const Twine &Path) {
  Storage = PathStorage(Path);
  StringRef PathRef = Storage.Path;

  // Look for Path without following symlinks. Failure here indicates that Path
  // does not exist or has a broken symlink in its parent path. Keep track of
  // symlinks in the parent path, but don't proactively push them to the
  // builder.
  SmallVector<const DirectoryEntry *> NonRealPathEntries;
  Expected<const DirectoryEntry *> PathEntry =
      FS.lookupPath(PathRef, /*FollowSymlinks=*/false,
                    /*LookupOnDisk=*/false, /*ForceDisableTracking=*/false,
                    /*NeedContent=*/true, [&](const DirectoryEntry &Entry) {
                      NonRealPathEntries.push_back(&Entry);
                    });
  if (!PathEntry)
    return PathEntry.takeError();

  // Finish resolving Path. If it's a symlink, recursively push its target. If
  // it's a directory, recursively push its contents. Broken symlinks here are
  // fine; we should push whatever we see.
  pushEntry(**PathEntry);

  // Push any symlinks from the parent path to the builder.
  for (const DirectoryEntry *Entry : NonRealPathEntries) {
    if (Entry->isSymlink()) {
      pushSymlink(*Entry);
      continue;
    }

    // Handle directories navigated away from with "..".
    assert(Entry->isDirectory());
    Builder.pushDirectory(Entry->getTreePath());
  }

  while (!Worklist.empty()) {
    const DirectoryEntry *Current = Worklist.pop_back_val();
    assert(!Current->isFile());
    if (Current->isSymlink()) {
      if (Expected<DirectoryEntry *> ExpectedEntry = FS.lookupPath(
              Current->getTreePath(),
              /*FollowSymlinks=*/true, /*ForceDisableTracking=*/false,
              /*NeedContent=*/true,
              /*LookupOnDisk=*/false, [this](const DirectoryEntry &Entry) {
                // Don't use pushSymlink() here since we
                // want the final entry too.
                pushEntry(Entry);
              })) {
        pushEntry(**ExpectedEntry);
      } else
        consumeError(ExpectedEntry.takeError());
      continue;
    }

    assert(Current->isDirectory());
    Directory &D = Current->asDirectory();
    D.forEachEntryUnsorted(
        [this](const DirectoryEntry &Entry) { pushEntry(Entry); });
  }
  return Error::success();
}

Expected<IntrusiveRefCntPtr<CachingOnDiskFileSystem>>
cas::createCachingOnDiskFileSystem(std::shared_ptr<ObjectStore> DB) {
  return std::make_unique<CachingOnDiskFileSystemImpl>(std::move(DB));
}

Expected<IntrusiveRefCntPtr<CachingOnDiskFileSystem>>
cas::createCachingOnDiskFileSystem(ObjectStore &DB) {
  return std::make_unique<CachingOnDiskFileSystemImpl>(DB);
}
