//===- FileSystemCache.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CAS_FILESYSTEMCACHE_H
#define LLVM_CAS_FILESYSTEMCACHE_H

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/CAS/CASReference.h"
#include "llvm/CAS/HashMappedTrie.h"
#include "llvm/CAS/ThreadSafeAllocator.h"
#include "llvm/Support/AlignOf.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/VirtualCachedDirectoryEntry.h"
#include "llvm/Support/VirtualFileSystem.h"
#include <mutex>

namespace llvm {
namespace cas {

/// Caching for lazily discovering a CAS-based filesystem.
///
/// FIXME: Extract most of this into llvm::vfs::FileSystemCache, so that it can
/// be reused by \a InMemoryFileSystem and \a RedirectingFileSystem.
class FileSystemCache : public ThreadSafeRefCountedBase<FileSystemCache> {
public:
  static constexpr unsigned MaxSymlinkDepth = 16;

  class File;
  class Symlink;
  class Directory;
  class DirectoryEntry;
  class VFSDirIterImpl;
  struct DirectoryListingInfo;

  struct LookupPathState {
    DirectoryEntry *Entry;
    StringRef Remaining;
    StringRef Name;
    StringRef AfterName;

    LookupPathState(DirectoryEntry &Entry, StringRef Remaining)
        : Entry(&Entry), Remaining(Remaining) {
      size_t Slash = Remaining.find('/');
      Name = Remaining.substr(0, Slash);
      AfterName = Slash == StringRef::npos ? "" : Remaining.drop_front(Slash);
    }

    void advance(DirectoryEntry &NewEntry) {
      // If all that's left is the path separator, need to ensure a preceding
      // symlink is followed. Return a "." to do that.
      //
      // FIXME: This logic is awkward. Probably Remaining should be an
      // Optional, this should crash if advancing to far, and users of
      // LookupPathState should be updated.
      if (AfterName.empty())
        *this = LookupPathState(NewEntry, AfterName);
      else if (AfterName == "/")
        *this = LookupPathState(NewEntry, ".");
      else
        *this = LookupPathState(NewEntry, AfterName.drop_front());
    }
    void skip() { advance(*Entry); }
  };

  class DiscoveryInstance {
  public:
    virtual ~DiscoveryInstance();

    /// Request a directory entry. The first parameter is the parent to look
    /// under, the second is the name of the entry.
    virtual Expected<DirectoryEntry *>
    requestDirectoryEntry(DirectoryEntry &Parent, StringRef Name) = 0;

    /// Request target of (lazy) symlink be filled in.
    virtual Error requestSymlinkTarget(DirectoryEntry &Symlink) = 0;

    /// Request a real path. The discovery instance should ensure this does
    /// minimal work if called multiple times during a single lookup.
    virtual Error preloadRealPath(DirectoryEntry &Parent, StringRef Remaining) {
      return Error::success();
    }

    /// Symlinks and navigated-away-from directories are passed through as the
    /// search progresses.
    virtual void trackNonRealPathEntry(DirectoryEntry &Entry) {}
  };

public:
  /// Create a directory entry and a directory at \a TreePath when \p Parent's
  /// mutex is already locked.
  ///
  /// Not thread-safe. Assumes there is a lock in place already on \p Parent's
  /// mutex.
  DirectoryEntry &makeDirectoryAlreadyLocked(DirectoryEntry &Parent,
                                             StringRef TreePath,
                                             Optional<ObjectRef> Ref);

  /// Create a directory entry for a \a Symlink without allocating it.
  ///
  /// Not thread-safe. Assumes there is a lock in place already on \p Parent's
  /// mutex.
  DirectoryEntry &makeLazySymlinkAlreadyLocked(DirectoryEntry &Parent,
                                               StringRef TreePath,
                                               ObjectRef Ref);

  /// Create a directory entry for a \a File without allocating it.
  ///
  /// Not thread-safe. Assumes there is a lock in place already on \p Parent's
  /// mutex.
  DirectoryEntry &makeLazyFileAlreadyLocked(DirectoryEntry &Parent,
                                            StringRef TreePath, ObjectRef Ref,
                                            bool IsExecutable);

  /// Create a directory entry and a directory (with no contents).
  ///
  /// Thread-safe; takes a lock on \p Parent's mutex.
  DirectoryEntry &makeDirectory(DirectoryEntry &Parent, StringRef TreePath,
                                Optional<ObjectRef> Ref = None);

  /// Create a directory entry and a symlink.
  ///
  /// Thread-safe; takes a lock on \p Parent's mutex.
  DirectoryEntry &makeSymlink(DirectoryEntry &Parent, StringRef TreePath,
                              ObjectRef Ref, StringRef Target);

  /// Create a directory entry and a file.
  ///
  /// Thread-safe; takes a lock on \p Parent's mutex.
  DirectoryEntry &makeFile(DirectoryEntry &Parent, StringRef TreePath,
                           ObjectRef Ref, size_t Size, bool IsExecutable);

  /// Fill in a lazy symlink, setting its target to \p Target.
  ///
  /// Thread-safe; takes a lock on \c SymlinkEntry.getParent()->Mutex.
  void finishLazySymlink(DirectoryEntry &SymlinkEntry, StringRef Target);

  /// Fill in a lazy file, setting its size to \p Target.
  ///
  /// Thread-safe; takes a lock on \c FileEntry.getParent()->Mutex.
  void finishLazyFile(DirectoryEntry &FileEntry, size_t Size);

  /// Look up a directory entry in the CAS, navigating trees and resolving
  /// symlinks in the parent path. If \p FollowSymlinks is true, also follows
  /// symlinks in the filename.
  ///
  /// If \p TrackNonRealPathEntries is given, symlinks and
  /// navigated-away-from directories are passed through as the search
  /// progresses.
  Expected<DirectoryEntry *> lookupPath(DiscoveryInstance &DI, StringRef Path,
                                        DirectoryEntry &WorkingDirectory,
                                        bool FollowSymlinks);

  /// Look up a directory entry in the CAS, navigating through real paths but
  /// returning early on a symlink.
  LookupPathState lookupRealPathPrefixFromCached(
      LookupPathState State,
      function_ref<void(DirectoryEntry &)> TrackNonRealPathEntries);

  /// Look up a directory entry in the CAS, navigating through real paths but
  /// returning early on a symlink.
  Expected<LookupPathState> lookupRealPathPrefixFrom(DiscoveryInstance &DI,
                                                     LookupPathState State);

  /// Lookup \p Path, knowing that \a sys::fs::real_path() was called and
  /// failed.
  Expected<LookupPathState>
  lookupInvalidRealPathPrefixFrom(DirectoryEntry &Start, StringRef Path);

  /// Look up a name inside \p From. Never checks the disk.
  DirectoryEntry *lookupNameFromCached(DirectoryEntry &Parent, StringRef Name);

  /// Look up a name on disk inside \p From.
  Expected<DirectoryEntry *> lookupOnDiskFrom(DirectoryEntry &Parent,
                                              StringRef Name);

  std::error_code setCurrentWorkingDirectory(const Twine &Path);

  static StringRef canonicalizeWorkingDirectory(const Twine &Path,
                                                StringRef WorkingDirectory,
                                                SmallVectorImpl<char> &Storage);

  DirectoryEntry &getRoot() { return *Root; }

  using LookupSymlinkPathType =
      unique_function<Expected<DirectoryEntry *>(StringRef)>;

  vfs::directory_iterator
  getCachedVFSDirIter(Directory &D, LookupSymlinkPathType LookupSymlinkPath,
                      StringRef WorkingDirectory, StringRef RequestedName);

  FileSystemCache(FileSystemCache &&) = delete;
  FileSystemCache(const FileSystemCache &) = delete;

  explicit FileSystemCache(Optional<ObjectRef> Root = None);

private:
  ThreadSafeAllocator<SpecificBumpPtrAllocator<File>> FileAlloc;
  ThreadSafeAllocator<SpecificBumpPtrAllocator<Symlink>> SymlinkAlloc;
  ThreadSafeAllocator<SpecificBumpPtrAllocator<Directory>> DirectoryAlloc;
  ThreadSafeAllocator<SpecificBumpPtrAllocator<DirectoryEntry>> EntryAlloc;
  ThreadSafeAllocator<SpecificBumpPtrAllocator<char>> TreePathAlloc;

  DirectoryEntry *Root = nullptr;
  struct {
    DirectoryEntry *Entry = nullptr;

    /// Mimics shell behaviour on directory changes. Not necessarily the same
    /// as \c Entry->getTreePath().
    std::string Path;
  } WorkingDirectory;
};

class FileSystemCache::DirectoryEntry : public vfs::CachedDirectoryEntry {
public:
  enum EntryKind {
    Regular,
    Executable,
    Symlink,
    Directory,
  };

  bool hasNode() const { return Node.load(std::memory_order_acquire); }
  bool isExecutable() const { return Kind == Executable; }
  bool isRegular() const { return Kind == Regular; }
  bool isFile() const { return isExecutable() || isRegular(); }
  bool isSymlink() const { return Kind == Symlink; }
  bool isDirectory() const { return Kind == Directory; }
  EntryKind getKind() const { return Kind; }
  DirectoryEntry *getParent() const { return Parent; }
  Optional<ObjectRef> getRef() const { return Ref; }

  sys::fs::file_type getFileType() const;

  /// Get the status with the requested name. Requires that this is not a
  /// symlink.
  ErrorOr<vfs::Status> getStatus(const Twine &RequestedName);

  FileSystemCache::Directory &asDirectory() const {
    assert(isDirectory());
    return *static_cast<FileSystemCache::Directory *>(
        Node.load(std::memory_order_acquire));
  }

  FileSystemCache::File &asFile() const {
    assert(isFile());
    return *static_cast<FileSystemCache::File *>(
        Node.load(std::memory_order_acquire));
  }

  FileSystemCache::Symlink &asSymlink() const {
    assert(isSymlink());
    return *static_cast<FileSystemCache::Symlink *>(
        Node.load(std::memory_order_acquire));
  }

  void setDirectory(FileSystemCache::Directory &D) {
    assert(isDirectory());
    void *Null = nullptr;
    (void)Node.compare_exchange_strong(Null, &D, std::memory_order_acq_rel);
  }

  void setFile(FileSystemCache::File &F) {
    assert(isFile());
    void *Null = nullptr;
    (void)Node.compare_exchange_strong(Null, &F, std::memory_order_acq_rel);
  }

  void setSymlink(FileSystemCache::Symlink &S) {
    assert(isSymlink());
    void *Null = nullptr;
    (void)Node.compare_exchange_strong(Null, &S, std::memory_order_acq_rel);
  }

  /// If \p Other is a prefix of the current entry, returns the child of
  /// \p Other that would be next in the current entry, if any.
  DirectoryEntry *nextEntryAfterPrefix(DirectoryEntry &Other) {
    DirectoryEntry *E = this, *P = Parent;
    while (P) {
      if (P == &Other)
        return E;
      E = P;
      P = P->Parent;
    }
    return nullptr;
  }

  DirectoryEntry() = delete;

  DirectoryEntry(DirectoryEntry *Parent, StringRef TreePath, EntryKind Kind,
                 Optional<ObjectRef> Ref)
      : CachedDirectoryEntry(TreePath), Parent(Parent), Kind(Kind),
        Node(nullptr), Ref(Ref) {}

private:
  DirectoryEntry *Parent;
  EntryKind Kind;
  Optional<sys::fs::UniqueID> UniqueID;
  std::atomic<void *> Node;
  Optional<ObjectRef> Ref; /// If this is a fixed tree.
};

struct FileSystemCache::DirectoryListingInfo {
  using DirectoryEntry = FileSystemCache::DirectoryEntry;
  static DirectoryEntry *getEmptyKey() {
    return DenseMapInfo<DirectoryEntry *>::getEmptyKey();
  }
  static DirectoryEntry *getTombstoneKey() {
    return DenseMapInfo<DirectoryEntry *>::getTombstoneKey();
  }
  static unsigned getHashValue(const DirectoryEntry *V) {
    return getHashValue(V->getName());
  }
  static bool isEqual(const DirectoryEntry *LHS, const DirectoryEntry *RHS) {
    return LHS == RHS;
  }

  static unsigned getHashValue(StringRef Name) {
    return DenseMapInfo<StringRef>::getHashValue(Name);
  }
  static bool isEqual(StringRef Name, const DirectoryEntry *RHS) {
    if (RHS == getEmptyKey())
      return false;
    if (RHS == getTombstoneKey())
      return false;
    return Name == RHS->getName();
  }
};

class FileSystemCache::Directory {
public:
  sys::fs::UniqueID getUniqueID() const { return UniqueID; }
  DirectoryEntry *lookup(StringRef Name) {
    auto I = Entries.find_as(Name);
    return I == Entries.end() ? nullptr : *I;
  }

  void add(DirectoryEntry &Entry) { Entries.insert(&Entry); }
  void getKnownEntries(SmallVectorImpl<DirectoryEntry *> &Result) const {
    Result.resize(Entries.size());
    llvm::copy(Entries, Result.begin());
  }
  void forEachEntryUnsorted(function_ref<void(DirectoryEntry &)> Do) {
    for (DirectoryEntry *Entry : Entries)
      Do(*Entry);
  }

  Directory()
      : IsComplete(false), IsWriting(false), NumReaders(0),
        UniqueID(vfs::getNextVirtualUniqueID()) {}

  /// True iff contents are known to be complete.
  bool isComplete() const { return IsComplete.load(); }

  std::mutex Mutex;
  std::atomic<bool> IsComplete; /// For clients to set when appropriate.

  class Reader;

  class Writer;

private:
  std::atomic<bool> IsWriting;
  std::atomic<int> NumReaders;
  DenseSet<DirectoryEntry *, DirectoryListingInfo> Entries;
  sys::fs::UniqueID UniqueID;
};

class FileSystemCache::Directory::Reader {
  std::atomic<int> *NumReaders = nullptr;

  void waitForWriter(Directory &D);

public:
  Reader() = delete;
  explicit Reader(Directory &D) {
    if (D.isComplete())
      return;
    NumReaders = &D.NumReaders;

    // Force writers to wait.
    ++D.NumReaders;
    if (D.IsWriting.load()) {
      // Take a lock if someone is already writing. This is rare. First
      // decrement NumReaders to avoid a deadlock.
      --D.NumReaders;
      std::lock_guard<std::mutex> Lock(D.Mutex);
      ++D.NumReaders;
    }
  }
  ~Reader() {
    if (NumReaders)
      --*NumReaders;
  }
};

class FileSystemCache::Directory::Writer {
  Optional<std::lock_guard<std::mutex>> Lock;

public:
  Writer() = delete;
  explicit Writer(Directory &D);
};

class FileSystemCache::File {
public:
  sys::fs::UniqueID getUniqueID() const { return UniqueID; }

  int64_t getSize() const { return Size; }

  explicit File(int64_t Size)
      : Size(Size), UniqueID(vfs::getNextVirtualUniqueID()) {}

private:
  friend class FileSystemCache;
  friend class StringMapEntryStorage<File>;

  int64_t Size;
  sys::fs::UniqueID UniqueID;
};

class FileSystemCache::Symlink {
public:
  sys::fs::UniqueID getUniqueID() const { return UniqueID; }
  StringRef getTarget() const { return Target; }

  Symlink(StringRef Target)
      : Target(Target), UniqueID(vfs::getNextVirtualUniqueID()) {}

private:
  std::string Target;
  sys::fs::UniqueID UniqueID;
};

class FileSystemCache::VFSDirIterImpl : public vfs::detail::DirIterImpl {
public:
  std::error_code increment() override;

  static std::shared_ptr<VFSDirIterImpl>
  create(LookupSymlinkPathType LookupSymlinkPath, StringRef ParentPath,
         ArrayRef<const DirectoryEntry *> Entries);

  void operator delete(void *Ptr) { ::free(Ptr); }

private:
  void setEntry();

  VFSDirIterImpl(LookupSymlinkPathType LookupSymlinkPath, StringRef ParentPath,
                 ArrayRef<const DirectoryEntry *> Entries)
      : LookupSymlinkPath(std::move(LookupSymlinkPath)), ParentPath(ParentPath),
        Entries(Entries), I(this->Entries.begin()) {
    setEntry();
  }

  LookupSymlinkPathType LookupSymlinkPath;
  StringRef ParentPath;
  ArrayRef<const DirectoryEntry *> Entries;
  ArrayRef<const DirectoryEntry *>::iterator I;
};

} // namespace cas
} // namespace llvm

#endif // LLVM_CAS_FILESYSTEMCACHE_H
