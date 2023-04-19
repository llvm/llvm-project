//===- UnifiedOnDiskCache.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Encapsulates \p OnDiskGraphDB and \p OnDiskKeyValueDB instances within one
// directory while also restricting storage growth with a scheme of chaining the
// two most recent directories (primary & upstream), where the primary
// "faults-in" data from the upstream one. When the primary (most recent)
// directory exceeds its intended limit a new empty directory becomes the
// primary one.
//
// Within the top-level directory (the path that \p UnifiedOnDiskCache::open
// receives) there are directories named like this:
//
// 'v<version>.<x>'
// 'v<version>.<x+1'
// 'v<version>.<x+2>'
// ...
//
// 'version' is the version integer for this \p UnifiedOnDiskCache's scheme and
// the part after the dot is an increasing integer. The primary directory is the
// one with the highest integer and the upstream one is the directory before it.
// For example, if the sub-directories contained are:
//
// 'v1.5', 'v1.6', 'v1.7', 'v1.8'
//
// Then the primary one is 'v1.8', the upstream one is 'v1.7', and the rest are
// unused directories that can be safely deleted at any time and by any process.
//
// Contained within the top-level directory is a file named "lock" which is used
// for processes to take shared or exclusive locks for the contents of the top
// directory. While a \p UnifiedOnDiskCache is open it keeps a shared lock for
// the top-level directory; when it closes, if the primary sub-directory
// exceeded its limit, it attempts to get an exclusive lock in order to create a
// new empty primary directory; if it can't get the exclusive lock it gives up
// and lets the next \p UnifiedOnDiskCache instance that closes to attempt
// again.
//
// The downside of this scheme is that while \p UnifiedOnDiskCache is open on a
// directory, by any process, the storage size in that directory will keep
// growing unrestricted. But the major benefit is that garbage-collection can be
// triggered on a directory concurrently, at any time and by any process,
// without affecting any active readers/writers in the same process or other
// processes.
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/UnifiedOnDiskCache.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/CAS/OnDiskKeyValueDB.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

using namespace llvm;
using namespace llvm::cas;
using namespace llvm::cas::ondisk;

/// FIXME: When the version of \p DBDirPrefix is bumped up we need to figure out
/// how to handle the leftover sub-directories of the previous version, within
/// the \p UnifiedOnDiskCache::collectGarbage function.
static constexpr StringLiteral DBDirPrefix = "v1.";

Expected<ObjectID> UnifiedOnDiskCache::KVPut(ObjectID Key, ObjectID Value) {
  return KVPut(PrimaryGraphDB->getDigest(Key), Value);
}

Expected<ObjectID> UnifiedOnDiskCache::KVPut(ArrayRef<uint8_t> Key,
                                             ObjectID Value) {
  static_assert(sizeof(Value.getOpaqueData()) == sizeof(uint64_t),
                "unexpected return opaque type");
  std::array<char, sizeof(uint64_t)> ValBytes;
  support::endian::write64le(ValBytes.data(), Value.getOpaqueData());
  Expected<ArrayRef<char>> Existing = PrimaryKVDB->put(Key, ValBytes);
  if (!Existing)
    return Existing.takeError();
  assert(Existing->size() == sizeof(uint64_t));
  return ObjectID::fromOpaqueData(support::endian::read64le(Existing->data()));
}

Expected<std::optional<ObjectID>>
UnifiedOnDiskCache::KVGet(ArrayRef<uint8_t> Key) {
  std::optional<ArrayRef<char>> Value;
  if (Error E = PrimaryKVDB->get(Key).moveInto(Value))
    return std::move(E);
  if (!Value) {
    if (UpstreamKVDB)
      return faultInFromUpstreamKV(Key);
    return std::nullopt;
  }
  assert(Value->size() == sizeof(uint64_t));
  return ObjectID::fromOpaqueData(support::endian::read64le(Value->data()));
}

Expected<std::optional<ObjectID>>
UnifiedOnDiskCache::faultInFromUpstreamKV(ArrayRef<uint8_t> Key) {
  assert(UpstreamGraphDB);
  assert(UpstreamKVDB);

  std::optional<ArrayRef<char>> UpstreamValue;
  if (Error E = UpstreamKVDB->get(Key).moveInto(UpstreamValue))
    return std::move(E);
  if (!UpstreamValue)
    return std::nullopt;

  // The value is the \p ObjectID in the context of the upstream
  // \p OnDiskGraphDB instance. Translate it to the context of the primary
  // \p OnDiskGraphDB instance.
  assert(UpstreamValue->size() == sizeof(uint64_t));
  ObjectID UpstreamID = ObjectID::fromOpaqueData(
      support::endian::read64le(UpstreamValue->data()));
  ObjectID PrimaryID =
      PrimaryGraphDB->getReference(UpstreamGraphDB->getDigest(UpstreamID));
  return KVPut(Key, PrimaryID);
}

/// \returns all the 'v<version>.<x>' names of sub-directories, sorted with
/// ascending order of the integer after the dot.
static Error getAllDBDirs(StringRef Path,
                          SmallVectorImpl<std::string> &DBDirs) {
  struct DBDir {
    uint64_t Order;
    std::string Name;
  };
  SmallVector<DBDir, 6> FoundDBDirs;

  std::error_code EC;
  for (sys::fs::directory_iterator DirI(Path, EC), DirE; !EC && DirI != DirE;
       DirI.increment(EC)) {
    if (DirI->type() != sys::fs::file_type::directory_file)
      continue;
    StringRef SubDir = sys::path::filename(DirI->path());
    if (!SubDir.startswith(DBDirPrefix))
      continue;
    uint64_t Order;
    if (SubDir.substr(DBDirPrefix.size()).getAsInteger(10, Order))
      return createStringError(inconvertibleErrorCode(),
                               "unexpected directory " + DirI->path());
    FoundDBDirs.push_back({Order, std::string(SubDir)});
  }
  if (EC)
    return createFileError(Path, EC);

  llvm::sort(FoundDBDirs, [](const DBDir &LHS, const DBDir &RHS) -> bool {
    return LHS.Order <= RHS.Order;
  });
  for (DBDir &Dir : FoundDBDirs)
    DBDirs.push_back(std::move(Dir.Name));
  return Error::success();
}

/// \returns Given a sub-directory named 'v<version>.<x>', it outputs the
/// 'v<version>.<x+1>' name.
static void getNextDBDirName(StringRef DBDir, llvm::raw_ostream &OS) {
  assert(DBDir.startswith(DBDirPrefix));
  uint64_t Count;
  bool Failed = DBDir.substr(DBDirPrefix.size()).getAsInteger(10, Count);
  assert(!Failed);
  (void)Failed;
  OS << DBDirPrefix << Count + 1;
}

Expected<std::unique_ptr<UnifiedOnDiskCache>>
UnifiedOnDiskCache::open(StringRef RootPath, std::optional<uint64_t> SizeLimit,
                         StringRef HashName, unsigned HashByteSize,
                         OnDiskGraphDB::FaultInPolicy FaultInPolicy) {
  if (std::error_code EC = sys::fs::create_directories(RootPath))
    return createFileError(RootPath, EC);

  SmallString<256> PathBuf(RootPath);
  sys::path::append(PathBuf, "lock");
  int LockFD = -1;
  if (std::error_code EC = sys::fs::openFileForReadWrite(
          PathBuf, LockFD, sys::fs::CD_OpenAlways, sys::fs::OF_None))
    return createFileError(PathBuf, EC);
  assert(LockFD != -1);
  // Locking the directory using shared lock, which will prevent other processes
  // from creating a new chain (essentially while a \p UnifiedOnDiskCache
  // instance holds a shared lock the storage for the primary directory will
  // grow unrestricted).
  if (std::error_code EC = sys::fs::lockFile(LockFD, /*Exclusive=*/false))
    return createFileError(PathBuf, EC);

  SmallVector<std::string, 4> DBDirs;
  if (Error E = getAllDBDirs(RootPath, DBDirs))
    return std::move(E);
  if (DBDirs.empty())
    DBDirs.push_back((Twine(DBDirPrefix) + "1").str());

  assert(!DBDirs.empty());

  /// If there is only one directory open databases on it. If there are 2 or
  /// more directories, get the most recent directories and chain them, with the
  /// most recent being the primary one. The remaining directories are unused
  /// data than can be garbage-collected.
  std::unique_ptr<OnDiskGraphDB> UpstreamGraphDB;
  std::unique_ptr<OnDiskKeyValueDB> UpstreamKVDB;
  if (DBDirs.size() > 1) {
    StringRef UpstreamDir = *(DBDirs.end() - 2);
    PathBuf = RootPath;
    sys::path::append(PathBuf, UpstreamDir);
    if (Error E = OnDiskGraphDB::open(PathBuf, HashName, HashByteSize,
                                      /*UpstreamDB=*/nullptr, FaultInPolicy)
                      .moveInto(UpstreamGraphDB))
      return std::move(E);
    if (Error E = OnDiskKeyValueDB::open(PathBuf, HashName, HashByteSize,
                                         /*ValueName=*/"objectid",
                                         /*ValueSize=*/sizeof(uint64_t))
                      .moveInto(UpstreamKVDB))
      return std::move(E);
  }
  OnDiskGraphDB *UpstreamGraphDBPtr = UpstreamGraphDB.get();

  StringRef PrimaryDir = *(DBDirs.end() - 1);
  PathBuf = RootPath;
  sys::path::append(PathBuf, PrimaryDir);
  std::unique_ptr<OnDiskGraphDB> PrimaryGraphDB;
  if (Error E = OnDiskGraphDB::open(PathBuf, HashName, HashByteSize,
                                    std::move(UpstreamGraphDB), FaultInPolicy)
                    .moveInto(PrimaryGraphDB))
    return std::move(E);
  std::unique_ptr<OnDiskKeyValueDB> PrimaryKVDB;
  // \p UnifiedOnDiskCache does manual chaining for key-value requests,
  // including an extra translation step of the value during fault-in.
  if (Error E = OnDiskKeyValueDB::open(PathBuf, HashName, HashByteSize,
                                       /*ValueName=*/"objectid",
                                       /*ValueSize=*/sizeof(uint64_t))
                    .moveInto(PrimaryKVDB))
    return std::move(E);

  auto UniDB = std::unique_ptr<UnifiedOnDiskCache>(new UnifiedOnDiskCache());
  UniDB->RootPath = RootPath;
  UniDB->SizeLimit = SizeLimit;
  UniDB->LockFD = LockFD;
  UniDB->NeedsGarbageCollection = DBDirs.size() > 2;
  UniDB->PrimaryDBDir = PrimaryDir;
  UniDB->UpstreamGraphDB = UpstreamGraphDBPtr;
  UniDB->PrimaryGraphDB = std::move(PrimaryGraphDB);
  UniDB->UpstreamKVDB = std::move(UpstreamKVDB);
  UniDB->PrimaryKVDB = std::move(PrimaryKVDB);

  return std::move(UniDB);
}

bool UnifiedOnDiskCache::hasExceededSizeLimit() const {
  if (!SizeLimit)
    return false;
  // We allow each of the directories in the chain to reach up to half the
  // intended size limit. Check whether the primary directory has exceeded half
  // the limit or not, in order to decide whether we need to start a new chain.
  //
  // We could check the size limit against the sum of sizes of both the primary
  // and upstream directories but then if the upstream is significantly larger
  // than the intended limit, it would trigger a new chain to be created before
  // the primary has reached its own limit. Essentially in such situation we
  // prefer reclaiming the storage later in order to have more consistent cache
  // hits behavior.
  return (*SizeLimit / 2) <
         (PrimaryGraphDB->getStorageSize() + PrimaryKVDB->getStorageSize());
}

Error UnifiedOnDiskCache::close(bool CheckSizeLimit) {
  if (LockFD == -1)
    return Error::success(); // already closed.
  auto _1 = make_scope_exit([&]() {
    assert(LockFD >= 0);
    sys::fs::file_t LockFile = sys::fs::convertFDToNativeFile(LockFD);
    sys::fs::closeFile(LockFile);
    LockFD = -1;
  });

  bool ExceededSizeLimit = CheckSizeLimit ? hasExceededSizeLimit() : false;
  PrimaryKVDB.reset();
  UpstreamKVDB.reset();
  PrimaryGraphDB.reset();
  UpstreamGraphDB = nullptr;
  if (std::error_code EC = sys::fs::unlockFile(LockFD))
    return createFileError(RootPath, EC);

  if (!ExceededSizeLimit)
    return Error::success();

  // The primary directory exceeded its intended size limit. Try to get an
  // exclusive lock in order to create a new primary directory for next time
  // this \p UnifiedOnDiskCache path is opened.

  if (std::error_code EC = sys::fs::tryLockFile(
          LockFD, std::chrono::milliseconds(0), /*Exclusive=*/true)) {
    if (EC == errc::no_lock_available)
      return Error::success(); // couldn't get exclusive lock, give up.
    return createFileError(RootPath, EC);
  }
  auto _2 = make_scope_exit([&]() { sys::fs::unlockFile(LockFD); });

  // Managed to get an exclusive lock which means there are no other open
  // \p UnifiedOnDiskCache instances for the same path, so we can safely start a
  // new primary directory. To start a new primary directory we just have to
  // create a new empty directory with the next consecutive index; since this is
  // an atomic operation we will leave the top-level directory in a consistent
  // state even if the process dies during this code-path.

  SmallString<256> PathBuf(RootPath);
  raw_svector_ostream OS(PathBuf);
  OS << sys::path::get_separator();
  getNextDBDirName(PrimaryDBDir, OS);
  if (std::error_code EC = sys::fs::create_directory(PathBuf))
    return createFileError(PathBuf, EC);

  NeedsGarbageCollection = true;
  return Error::success();
}

UnifiedOnDiskCache::UnifiedOnDiskCache() = default;

UnifiedOnDiskCache::~UnifiedOnDiskCache() { consumeError(close()); }

Error UnifiedOnDiskCache::collectGarbage(StringRef Path) {
  SmallVector<std::string, 4> DBDirs;
  if (Error E = getAllDBDirs(Path, DBDirs))
    return E;
  if (DBDirs.size() <= 2)
    return Error::success(); // no unused directories.

  // FIXME: When the version of \p DBDirPrefix is bumped up we need to figure
  // out how to handle the leftover sub-directories of the previous version.

  SmallString<256> PathBuf(Path);
  for (StringRef UnusedSubDir : ArrayRef(DBDirs).drop_back(2)) {
    sys::path::append(PathBuf, UnusedSubDir);
    if (std::error_code EC = sys::fs::remove_directories(PathBuf))
      return createFileError(PathBuf, EC);
    sys::path::remove_filename(PathBuf);
  }
  return Error::success();
}
