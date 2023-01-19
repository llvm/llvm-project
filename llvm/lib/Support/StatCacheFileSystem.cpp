//===- StatCacheFileSystem.cpp - Status Caching Proxy File System ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/StatCacheFileSystem.h"

#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/OnDiskHashTable.h"

namespace llvm {
namespace vfs {

class StatCacheFileSystem::StatCacheLookupInfo {
public:
  typedef StringRef external_key_type;
  typedef StringRef internal_key_type;
  typedef llvm::sys::fs::file_status data_type;
  typedef uint32_t hash_value_type;
  typedef uint32_t offset_type;

  static bool EqualKey(const internal_key_type &a, const internal_key_type &b) {
    return a == b;
  }

  static hash_value_type ComputeHash(const internal_key_type &a) {
    return hash_value(a);
  }

  static std::pair<unsigned, unsigned>
  ReadKeyDataLength(const unsigned char *&d) {
    using namespace llvm::support;
    unsigned KeyLen = endian::readNext<uint16_t, little, unaligned>(d);
    unsigned DataLen = endian::readNext<uint16_t, little, unaligned>(d);
    return std::make_pair(KeyLen, DataLen);
  }

  static const internal_key_type &GetInternalKey(const external_key_type &x) {
    return x;
  }

  static const external_key_type &GetExternalKey(const internal_key_type &x) {
    return x;
  }

  static internal_key_type ReadKey(const unsigned char *d, unsigned n) {
    return StringRef((const char *)d, n);
  }

  static data_type ReadData(const internal_key_type &k, const unsigned char *d,
                            unsigned DataLen) {
    data_type Result;
    memcpy(&Result, d, sizeof(Result));
    return Result;
  }
};

class StatCacheFileSystem::StatCacheGenerationInfo {
public:
  typedef StringRef key_type;
  typedef const StringRef &key_type_ref;
  typedef sys::fs::file_status data_type;
  typedef const sys::fs::file_status &data_type_ref;
  typedef uint32_t hash_value_type;
  typedef uint32_t offset_type;

  /// Calculate the hash for Key
  static hash_value_type ComputeHash(key_type_ref Key) {
    return static_cast<size_t>(hash_value(Key));
  }

  /// Return the lengths, in bytes, of the given Key/Data pair.
  static std::pair<unsigned, unsigned>
  EmitKeyDataLength(raw_ostream &Out, key_type_ref Key, data_type_ref Data) {
    using namespace llvm::support;
    endian::Writer LE(Out, little);
    unsigned KeyLen = Key.size();
    unsigned DataLen = sizeof(Data);
    LE.write<uint16_t>(KeyLen);
    LE.write<uint16_t>(DataLen);
    return std::make_pair(KeyLen, DataLen);
  }

  static void EmitKey(raw_ostream &Out, key_type_ref Key, unsigned KeyLen) {
    Out.write(Key.data(), KeyLen);
  }

  /// Write Data to Out.  DataLen is the length from EmitKeyDataLength.
  static void EmitData(raw_ostream &Out, key_type_ref Key, data_type_ref Data,
                       unsigned Len) {
    Out.write((const char *)&Data, Len);
  }

  static bool EqualKey(key_type_ref Key1, key_type_ref Key2) {
    return Key1 == Key2;
  }
};

// The format of the stat cache is (pseudo-code):
//  struct stat_cache {
//    char     Magic[4];       // "STAT" or "Stat"
//    uint32_t BucketOffset;   // See BucketOffset in OnDiskHashTable.h
//    uint64_t ValidityToken;  // Platofrm specific data allowing to check
//                             // whether the cache is up-to-date.
//    uint32_t Version;        // The stat cache format version.
//    char     BaseDir[N];     // Zero terminated path to the base directory
//    < OnDiskHashtable Data > // Data for the has table. The keys are the
//                             // relative paths under BaseDir. The data is
//                             // llvm::sys::fs::file_status structures.
//  };

#define MAGIC_CASE_SENSITIVE "Stat"
#define MAGIC_CASE_INSENSITIVE "STAT"
#define STAT_CACHE_VERSION 1

namespace {
struct StatCacheHeader {
  char Magic[4];
  uint32_t BucketOffset;
  uint64_t ValidityToken;
  uint32_t Version;
  char BaseDir[1];
};
} // namespace

StatCacheFileSystem::StatCacheFileSystem(
    std::unique_ptr<MemoryBuffer> CacheFile, IntrusiveRefCntPtr<FileSystem> FS,
    bool IsCaseSensitive)
    : ProxyFileSystem(std::move(FS)), StatCacheFile(std::move(CacheFile)),
      IsCaseSensitive(IsCaseSensitive) {
  const char *CacheFileStart = StatCacheFile->getBufferStart();
  auto *Header = reinterpret_cast<const StatCacheHeader *>(CacheFileStart);

  uint32_t BucketOffset = Header->BucketOffset;
  StatCachePrefix = StringRef(Header->BaseDir);
  // HashTableStart points at the beginning of the data emitted by the
  // OnDiskHashTable.
  const unsigned char *HashTableStart = (const unsigned char *)CacheFileStart +
                                        StatCachePrefix.size() +
                                        sizeof(StatCacheHeader);
  StatCache.reset(StatCacheType::Create(
      (const unsigned char *)CacheFileStart + BucketOffset, HashTableStart,
      (const unsigned char *)CacheFileStart));
}

Expected<IntrusiveRefCntPtr<StatCacheFileSystem>>
StatCacheFileSystem::create(std::unique_ptr<MemoryBuffer> CacheBuffer,
                            IntrusiveRefCntPtr<FileSystem> FS) {
  StringRef BaseDir;
  bool IsCaseSensitive;
  bool VersionMatch;
  uint64_t ValidityToken;
  if (auto E = validateCacheFile(*CacheBuffer, BaseDir, IsCaseSensitive,
                                 VersionMatch, ValidityToken))
    return std::move(E);
  if (!VersionMatch) {
    return createStringError(inconvertibleErrorCode(),
                             CacheBuffer->getBufferIdentifier() +
                                 ": Mismatched cache file version");
  }
  return new StatCacheFileSystem(std::move(CacheBuffer), FS, IsCaseSensitive);
}

ErrorOr<Status> StatCacheFileSystem::status(const Twine &Path) {
  SmallString<180> StringPath;
  Path.toVector(StringPath);
  // If the cache is not case sensitive, do all operations on lower-cased paths.
  if (!IsCaseSensitive)
    std::transform(StringPath.begin(), StringPath.end(), StringPath.begin(),
                   toLower);

  // Canonicalize the path. This removes single dot path components,
  // but it also gets rid of repeating separators.
  llvm::sys::path::remove_dots(StringPath);

  // If on Windows, canonicalize separators.
  llvm::sys::path::make_preferred(StringPath);

  // Check if the requested path falls into the cache.
  StringRef SuffixPath(StringPath);
  if (!SuffixPath.consume_front(StatCachePrefix))
    return ProxyFileSystem::status(Path);

  auto It = StatCache->find(SuffixPath);
  if (It == StatCache->end()) {
    // We didn't find the file in the cache even though it started with the
    // cache prefix. It could be that the file doesn't exist, or the spelling
    // the path is different. `remove_dots` canonicalizes the path by removing
    // `.` and excess separators, but leaves `..` since it isn't semantically
    // preserving to remove them in the presence of symlinks. If the path
    // does not contain '..' we can safely say it doesn't exist.
    if (std::find(sys::path::begin(SuffixPath), sys::path::end(SuffixPath),
                  "..") == sys::path::end(SuffixPath)) {
      return llvm::errc::no_such_file_or_directory;
    }
    return ProxyFileSystem::status(Path);
  }

  // clang-stat-cache will record entries for broken symlnks with a default-
  // constructed Status. This will have a default-constructed UinqueID.
  if ((*It).getUniqueID() == llvm::sys::fs::UniqueID())
    return llvm::errc::no_such_file_or_directory;

  return llvm::vfs::Status::copyWithNewName(*It, Path);
}

StatCacheFileSystem::StatCacheWriter::StatCacheWriter(
    StringRef BaseDir, const sys::fs::file_status &Status, bool IsCaseSensitive,
    uint64_t ValidityToken)
    : BaseDir(IsCaseSensitive ? BaseDir.str() : BaseDir.lower()),
      IsCaseSensitive(IsCaseSensitive), ValidityToken(ValidityToken),
      Generator(new StatCacheGeneratorType()) {
  addEntry(BaseDir, Status);
  // If on Windows, canonicalize separators.
  llvm::sys::path::make_preferred(this->BaseDir);
}

StatCacheFileSystem::StatCacheWriter::~StatCacheWriter() { delete Generator; }

void StatCacheFileSystem::StatCacheWriter::addEntry(
    StringRef Path, const sys::fs::file_status &Status) {
  llvm::SmallString<128> StoredPath;

#if defined(_WIN32)
  StoredPath = Path;
  llvm::sys::path::make_preferred(StoredPath);
  Path = StoredPath;
#endif

  if (!IsCaseSensitive) {
    StoredPath = Path.lower();
    Path = StoredPath;
  }

  LLVM_ATTRIBUTE_UNUSED bool Consumed = Path.consume_front(BaseDir);
  assert(Consumed && "Path does not start with expected prefix.");

  PathStorage.emplace_back(Path.str());
  Generator->insert(PathStorage.back(), Status);
}

size_t
StatCacheFileSystem::StatCacheWriter::writeStatCache(raw_fd_ostream &Out) {
  const uint32_t Version = STAT_CACHE_VERSION;
  // Magic value.
  if (IsCaseSensitive)
    Out.write(MAGIC_CASE_SENSITIVE, 4);
  else
    Out.write(MAGIC_CASE_INSENSITIVE, 4);
  // Placeholder for BucketOffset, filled in below.
  Out.write("\0\0\0\0", 4);
  // Write out the validity token.
  Out.write((const char *)&ValidityToken, sizeof(ValidityToken));
  // Write out the version.
  Out.write((const char *)&Version, sizeof(Version));
  // Write out the base directory for the cache.
  Out.write(BaseDir.c_str(), BaseDir.size() + 1);
  // Write out the hashtable data.
  uint32_t BucketOffset = Generator->Emit(Out);
  int Size = Out.tell();
  // Move back to right after the Magic to insert BucketOffset
  Out.seek(4);
  Out.write((const char *)&BucketOffset, sizeof(BucketOffset));
  return Size;
}

Error StatCacheFileSystem::validateCacheFile(MemoryBufferRef Buffer,
                                             StringRef &BaseDir,
                                             bool &IsCaseSensitive,
                                             bool &VersionMatch,
                                             uint64_t &ValidityToken) {
  auto *Header =
      reinterpret_cast<const StatCacheHeader *>(Buffer.getBufferStart());
  if (Buffer.getBufferSize() < sizeof(StatCacheHeader) ||
      (memcmp(Header->Magic, MAGIC_CASE_INSENSITIVE, sizeof(Header->Magic)) &&
       memcmp(Header->Magic, MAGIC_CASE_SENSITIVE, sizeof(Header->Magic))) ||
      Header->BucketOffset > Buffer.getBufferSize())
    return createStringError(inconvertibleErrorCode(), "Invalid cache file");

  auto PathLen =
      strnlen(Header->BaseDir,
              Buffer.getBufferSize() - offsetof(StatCacheHeader, BaseDir));
  if (Header->BaseDir[PathLen] != 0)
    return createStringError(inconvertibleErrorCode(), "Invalid cache file");

  IsCaseSensitive = Header->Magic[1] == MAGIC_CASE_SENSITIVE[1];
  VersionMatch = Header->Version == STAT_CACHE_VERSION;
  BaseDir = StringRef(Header->BaseDir, PathLen);
  ValidityToken = Header->ValidityToken;

  return ErrorSuccess();
}

void StatCacheFileSystem::updateValidityToken(raw_fd_ostream &CacheFile,
                                              uint64_t ValidityToken) {
  CacheFile.pwrite(reinterpret_cast<char *>(&ValidityToken),
                   sizeof(ValidityToken),
                   offsetof(StatCacheHeader, ValidityToken));
}

} // namespace vfs
} // namespace llvm
