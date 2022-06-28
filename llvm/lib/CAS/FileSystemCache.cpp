//===- FileSystemCache.cpp --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/FileSystemCache.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/CAS/CASDB.h"
#include "llvm/CAS/HashMappedTrie.h"
#include "llvm/Support/AlignOf.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Path.h"
#include <mutex>

using namespace llvm;
using namespace llvm::cas;

using DirectoryEntry = FileSystemCache::DirectoryEntry;

FileSystemCache::FileSystemCache(Optional<ObjectRef> RootRef) {
  // FIXME: Only correct for posix. To generalize this (for both posix and
  // windows) we should refactor so that the root node has no name (instead of
  // "/").
  Root = new (EntryAlloc.Allocate())
      DirectoryEntry(nullptr, "/", DirectoryEntry::Directory, RootRef);
  Root->setDirectory(*new (DirectoryAlloc.Allocate()) Directory);
}

StringRef
FileSystemCache::canonicalizeWorkingDirectory(const Twine &Path,
                                              StringRef WorkingDirectory,
                                              SmallVectorImpl<char> &Storage) {
  // Not portable.
  assert(WorkingDirectory.startswith("/"));
  Path.toVector(Storage);
  if (Storage.empty())
    return WorkingDirectory;

  if (Storage[0] != '/') {
    SmallString<128> Prefix = StringRef(WorkingDirectory);
    Prefix.push_back('/');
    Storage.insert(Storage.begin(), Prefix.begin(), Prefix.end());
  }

  // Remove ".." components based on working directory string, not based on
  // real path. This matches shell behaviour.
  sys::path::remove_dots(Storage, /*remove_dot_dot=*/true,
                         sys::path::Style::posix);

  // Remove double slashes.
  int W = 0;
  bool WasSlash = false;
  for (int R = 0, E = Storage.size(); R != E; ++R) {
    bool IsSlash = Storage[R] == '/';
    if (IsSlash && WasSlash)
      continue;
    WasSlash = IsSlash;
    Storage[W++] = Storage[R];
  }
  Storage.resize(W);

  // Remove final slash.
  if (Storage.size() > 1 && Storage.back() == '/')
    Storage.pop_back();

  return StringRef(Storage.begin(), Storage.size());
}

FileSystemCache::Directory::Writer::Writer(Directory &D) {
  D.IsWriting = true;
  Lock.emplace(D.Mutex);
  while (D.NumReaders.load() > 0) // Wait for readers to finish.
    ;
}

static StringRef allocateTreePath(
    ThreadSafeAllocator<SpecificBumpPtrAllocator<char>> &TreePathAlloc,
    StringRef TreePath) {
  // Use constant strings when reasonable.
  if (TreePath.empty())
    return "";
  if (TreePath == "/")
    return "/";

  char *AllocatedTreePath = TreePathAlloc.Allocate(TreePath.size() + 1);
  llvm::copy(TreePath, AllocatedTreePath);
  AllocatedTreePath[TreePath.size()] = 0;
  return AllocatedTreePath;
}

static DirectoryEntry &makeLazyEntry(
    ThreadSafeAllocator<SpecificBumpPtrAllocator<char>> &TreePathAlloc,
    ThreadSafeAllocator<SpecificBumpPtrAllocator<DirectoryEntry>> &EntryAlloc,
    DirectoryEntry &Parent, FileSystemCache::Directory &D, StringRef TreePath,
    DirectoryEntry::EntryKind Kind, Optional<ObjectRef> Ref) {
  assert(sys::path::parent_path(TreePath) == Parent.getTreePath());
  assert(!D.lookup(sys::path::filename(TreePath)));
  assert(!D.isComplete());

  TreePath = allocateTreePath(TreePathAlloc, TreePath);
  DirectoryEntry &Entry =
      *new (EntryAlloc.Allocate()) DirectoryEntry(&Parent, TreePath, Kind, Ref);
  D.add(Entry);
  return Entry;
}

DirectoryEntry &FileSystemCache::makeLazySymlinkAlreadyLocked(
    DirectoryEntry &Parent, StringRef TreePath, ObjectRef Ref) {
  return makeLazyEntry(TreePathAlloc, EntryAlloc, Parent, Parent.asDirectory(),
                       TreePath, DirectoryEntry::Symlink, Ref);
}

DirectoryEntry &
FileSystemCache::makeLazyFileAlreadyLocked(DirectoryEntry &Parent,
                                           StringRef TreePath, ObjectRef Ref,
                                           bool IsExecutable) {
  return makeLazyEntry(
      TreePathAlloc, EntryAlloc, Parent, Parent.asDirectory(), TreePath,
      IsExecutable ? DirectoryEntry::Executable : DirectoryEntry::Regular, Ref);
}

DirectoryEntry &FileSystemCache::makeDirectory(DirectoryEntry &Parent,
                                               StringRef TreePath,
                                               Optional<ObjectRef> Ref) {
  Directory &D = Parent.asDirectory();
  Directory::Writer W(D);
  if (DirectoryEntry *Existing = D.lookup(sys::path::filename(TreePath)))
    return *Existing;

  return makeDirectoryAlreadyLocked(Parent, TreePath, Ref);
}

DirectoryEntry &FileSystemCache::makeDirectoryAlreadyLocked(
    DirectoryEntry &Parent, StringRef TreePath, Optional<ObjectRef> Ref) {
  DirectoryEntry &Entry =
      makeLazyEntry(TreePathAlloc, EntryAlloc, Parent, Parent.asDirectory(),
                    TreePath, DirectoryEntry::Directory, Ref);
  Entry.setDirectory(*new (DirectoryAlloc.Allocate()) Directory);
  return Entry;
}

DirectoryEntry &FileSystemCache::makeSymlink(DirectoryEntry &Parent,
                                             StringRef TreePath, ObjectRef Ref,
                                             StringRef Target) {
  Directory &D = Parent.asDirectory();
  Directory::Writer W(D);
  if (DirectoryEntry *Existing = D.lookup(sys::path::filename(TreePath)))
    return *Existing;

  DirectoryEntry &Entry = makeLazySymlinkAlreadyLocked(Parent, TreePath, Ref);
  Entry.setSymlink(*new (SymlinkAlloc.Allocate()) Symlink(Target));
  return Entry;
}

void FileSystemCache::finishLazySymlink(DirectoryEntry &SymlinkEntry,
                                        StringRef Target) {
  assert(SymlinkEntry.isSymlink());

  DirectoryEntry &Parent = *SymlinkEntry.getParent();
  Directory &D = Parent.asDirectory();
  Directory::Writer W(D);
  if (SymlinkEntry.hasNode())
    return;

  SymlinkEntry.setSymlink(*new (SymlinkAlloc.Allocate()) Symlink(Target));
}

DirectoryEntry &FileSystemCache::makeFile(DirectoryEntry &Parent,
                                          StringRef TreePath, ObjectRef Ref,
                                          size_t Size, bool IsExecutable) {
  Directory &D = Parent.asDirectory();
  Directory::Writer W(D);
  if (DirectoryEntry *Existing = D.lookup(sys::path::filename(TreePath)))
    return *Existing;

  DirectoryEntry &Entry =
      makeLazyFileAlreadyLocked(Parent, TreePath, Ref, IsExecutable);
  Entry.setFile(*new (FileAlloc.Allocate()) File(Size));
  return Entry;
}

void FileSystemCache::finishLazyFile(DirectoryEntry &FileEntry, size_t Size) {
  assert(FileEntry.isFile());

  DirectoryEntry &Parent = *FileEntry.getParent();
  Directory &D = Parent.asDirectory();
  Directory::Writer W(D);
  if (FileEntry.hasNode())
    return;

  FileEntry.setFile(*new (FileAlloc.Allocate()) File(Size));
}

sys::fs::file_type DirectoryEntry::getFileType() const {
  assert(!isSymlink() && "Expected symlink to be followed first");
  switch (Kind) {
  case Directory:
    return sys::fs::file_type::directory_file;
  case Regular:
  case Executable:
    return sys::fs::file_type::regular_file;
  case Symlink:
    llvm_unreachable("symlinks should be followed before getting file type");
  };
}

ErrorOr<vfs::Status> DirectoryEntry::getStatus(const Twine &RequestedName) {
  // Symlinks should be followed first. Getting here indicates a broken symlink
  // in directory iteration.
  if (Kind == Symlink)
    return std::errc::no_such_file_or_directory;

  const sys::fs::perms RegularPermissions =
      sys::fs::perms::all_read | sys::fs::perms::owner_write;

  const sys::fs::perms ExecutablePermissions =
      RegularPermissions | sys::fs::perms::all_exe;

  sys::fs::UniqueID UniqueID;
  uint64_t Size;
  sys::fs::perms Permissions;
  switch (Kind) {
  case Directory:
    Size = 0;
    Permissions = ExecutablePermissions;
    UniqueID = asDirectory().getUniqueID();
    break;
  case Regular:
  case Executable: {
    auto &F = asFile();
    Size = F.getSize();
    Permissions = isExecutable() ? ExecutablePermissions : RegularPermissions;
    UniqueID = F.getUniqueID();
    break;
  }
  case Symlink:
    llvm_unreachable("symlinks don't expose status");
  };

  return vfs::Status(RequestedName, UniqueID, sys::TimePoint<>(), /*User=*/0,
                     /*Group=*/0, Size, getFileType(), Permissions);
}

Expected<DirectoryEntry *>
FileSystemCache::lookupPath(DiscoveryInstance &DI, StringRef Path,
                            DirectoryEntry &WorkingDirectory,
                            bool FollowSymlinks) {
  assert(Root && "Expected root filesystem to exist");

  struct WorklistNode {
    StringRef Remaining;
    unsigned SymlinkDepth;
  };
  SmallVector<WorklistNode> Worklist;
  auto pushWork = [&Worklist](StringRef Work, unsigned SymlinkDepth = 0) {
    if (Work.empty())
      return false;
    Worklist.push_back({Work, SymlinkDepth});
    return true;
  };

  // Start at the current working directory, unless the path is absolute.
  DirectoryEntry *Current = &WorkingDirectory;
  if (Path.consume_front("/"))
    Current = Root;

  pushWork(Path);
  while (!Worklist.empty()) {
    assert(Current);
    auto Work = Worklist.pop_back_val();
    Expected<LookupPathState> Found =
        lookupRealPathPrefixFrom(DI, LookupPathState(*Current, Work.Remaining));
    if (!Found)
      return createFileError(Path, Found.takeError());
    Current = Found->Entry;
    StringRef Remaining = Found->Remaining;

    if (Current->getKind() != DirectoryEntry::Symlink) {
      // Give up if we're not done and we didn't hit a symlink.
      if (!Remaining.empty())
        return createFileError(
            Path, make_error_code(std::errc::no_such_file_or_directory));
      continue;
    }

    // Save the progress in the builder.
    DI.trackNonRealPathEntry(*Current);

    unsigned SymlinkDepth = Work.SymlinkDepth + 1;
    pushWork(Remaining, SymlinkDepth);

    // If the worklist is empty then we have decide whether to follow the
    // symlink, since it's in the filename part of the path.
    if (Worklist.empty() && !FollowSymlinks)
      break;

    if (SymlinkDepth > MaxSymlinkDepth)
      return createFileError(
          Path, make_error_code(std::errc::too_many_symbolic_link_levels));

    if (!Current->hasNode())
      if (Error E = DI.requestSymlinkTarget(*Current))
        return createFileError(Path, std::move(E));
    StringRef Target = Current->asSymlink().getTarget();
    if (Target.consume_front("/"))
      Current = Root;
    else
      Current = Current->getParent();
    pushWork(Target, SymlinkDepth);
  }

  // Success.
  return Current;
}

FileSystemCache::LookupPathState
FileSystemCache::lookupRealPathPrefixFromCached(
    LookupPathState State,
    function_ref<void(DirectoryEntry &)> TrackNonRealPathEntries) {
  while (!State.Remaining.empty()) {
    assert(State.Entry);

    // Stop if this isn't a tree.
    if (State.Entry->getKind() != DirectoryEntry::Directory)
      return State;

    // FIXME: Need this logic in all the iteration loops...
    if (State.Name == "" || State.Name == ".") {
      State.advance(*State.Entry);
      continue;
    }

    if (State.Name == "..") {
      if (TrackNonRealPathEntries)
        TrackNonRealPathEntries(*State.Entry);
      if (DirectoryEntry *Parent = State.Entry->getParent())
        State.Entry = Parent;
      State.skip();
      continue;
    }

    // If we can't find the name cached, give up.
    DirectoryEntry *Next = lookupNameFromCached(*State.Entry, State.Name);
    if (!Next)
      return State;

    // Found it locally. Go deeper.
    State.advance(*Next);
  }

  // Found it.
  return State;
}

Expected<FileSystemCache::LookupPathState>
FileSystemCache::lookupRealPathPrefixFrom(DiscoveryInstance &DI,
                                          LookupPathState State) {
  assert(State.Entry);
  bool LoadedRealPath = false;
  while (true) {
    State = lookupRealPathPrefixFromCached(State, [&DI](DirectoryEntry &Entry) {
      return DI.trackNonRealPathEntry(Entry);
    });

    // Success!
    if (State.Remaining.empty())
      return State;

    // Success! The real path can't go through a directory.
    if (State.Entry->getKind() != DirectoryEntry::Directory ||
        State.Entry->asDirectory().isComplete())
      return State;

    // Cache the real path to avoid unnecessary component-by-component stat
    // calls.
    if (!LoadedRealPath) {
      if (Error E = DI.preloadRealPath(*State.Entry, State.Remaining))
        return std::move(E);
      LoadedRealPath = true;
      continue;
    }

    // Read the next component from disk.
    Expected<DirectoryEntry *> Next =
        DI.requestDirectoryEntry(*State.Entry, State.Name);
    if (!Next)
      return Next.takeError();

    // Can't look any further if we're not accessing the disk.
    if (!*Next)
      return State;

    State.advance(**Next);
  }
}

DirectoryEntry *FileSystemCache::lookupNameFromCached(DirectoryEntry &Parent,
                                                      StringRef Name) {
  assert(Parent.isDirectory() && "Expected a directory");
  Directory &D = Parent.asDirectory();
  Directory::Reader R(D);
  return D.lookup(Name);
}

vfs::directory_iterator FileSystemCache::getCachedVFSDirIter(
    Directory &D, LookupSymlinkPathType LookupSymlinkPath,
    StringRef WorkingDirectory, StringRef RequestedName) {
  SmallVector<DirectoryEntry *, 0> Entries;
  {
    Directory::Reader R(D);
    D.getKnownEntries(Entries);
  }

  SmallString<128> Storage;
  if (RequestedName.empty()) {
    RequestedName = WorkingDirectory;
  } else if (!RequestedName.startswith("/")) {
    Storage.append(WorkingDirectory);
    sys::path::append(Storage, sys::path::Style::posix, RequestedName);
    RequestedName = Storage;
  }
  RequestedName = RequestedName.rtrim("/");
  std::shared_ptr<VFSDirIterImpl> DirIter = VFSDirIterImpl::create(
      std::move(LookupSymlinkPath), RequestedName, Entries);
  return vfs::directory_iterator(std::move(DirIter));
}

void FileSystemCache::VFSDirIterImpl::setEntry() {
  if (I == Entries.end()) {
    CurrentEntry = vfs::directory_entry();
    return;
  }
  const DirectoryEntry &Entry = **I;
  std::string Path = (ParentPath + "/" + Entry.getName()).str();
  if (!Entry.isSymlink()) {
    CurrentEntry = vfs::directory_entry(std::move(Path), Entry.getFileType());
    return;
  }

  // Follow the symlink.
  sys::fs::file_type Type = sys::fs::file_type::status_error;
  if (Expected<DirectoryEntry *> ExpectedTarget = LookupSymlinkPath(Path))
    Type = (*ExpectedTarget)->getFileType();
  else
    consumeError(ExpectedTarget.takeError());
  CurrentEntry = vfs::directory_entry(std::move(Path), Type);
  return;
}

std::error_code FileSystemCache::VFSDirIterImpl::increment() {
  assert(!Entries.empty() && "Incrementing past the end");
  ++I;
  setEntry();
  return std::error_code();
}

std::shared_ptr<FileSystemCache::VFSDirIterImpl>
FileSystemCache::VFSDirIterImpl::create(
    LookupSymlinkPathType LookupSymlinkPath, StringRef ParentPath,
    ArrayRef<const DirectoryEntry *> Entries) {
  // Compute where to put the entry pointers and allocate.
  size_t IterSize = sizeof(VFSDirIterImpl);
  IterSize = alignTo(IterSize, alignof(DirectoryEntry *));
  size_t EntriesSize = Entries.size() * sizeof(DirectoryEntry *);
  size_t ParentPathSize = ParentPath.size() + 1;
  size_t TotalSize = IterSize + EntriesSize + ParentPathSize;
  void *IterPtr = ::malloc(TotalSize);

  // Initialize the entries.
  const DirectoryEntry **HungOffEntries =
      new (reinterpret_cast<char *>(IterPtr) + IterSize)
          const DirectoryEntry *[Entries.size()];
  llvm::copy(Entries, HungOffEntries);
  std::sort(HungOffEntries, HungOffEntries + Entries.size(),
            [](const DirectoryEntry *LHS, const DirectoryEntry *RHS) {
              return LHS->getName() < RHS->getName();
            });

  char *HungOffParentPath = new (reinterpret_cast<char *>(HungOffEntries) +
                                 EntriesSize) char[ParentPathSize];
  llvm::copy(ParentPath, HungOffParentPath);
  HungOffParentPath[ParentPath.size()] = 0;

  // Construct the iterator, pointing at the co-allocated entries.
  return std::shared_ptr<VFSDirIterImpl>(new (IterPtr) VFSDirIterImpl(
      std::move(LookupSymlinkPath),
      StringRef(HungOffParentPath, ParentPath.size()),
      makeArrayRef(HungOffEntries, Entries.size())));
}

FileSystemCache::DiscoveryInstance::~DiscoveryInstance() {}
