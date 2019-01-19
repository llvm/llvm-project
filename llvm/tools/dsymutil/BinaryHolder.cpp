//===-- BinaryHolder.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This program is a utility that aims to be a dropin replacement for
// Darwin's dsymutil.
//
//===----------------------------------------------------------------------===//

#include "BinaryHolder.h"
#include "llvm/Object/MachO.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
namespace dsymutil {

static std::pair<StringRef, StringRef>
getArchiveAndObjectName(StringRef Filename) {
  StringRef Archive = Filename.substr(0, Filename.find('('));
  StringRef Object = Filename.substr(Archive.size() + 1).drop_back();
  return {Archive, Object};
}

static bool isArchive(StringRef Filename) { return Filename.endswith(")"); }

static std::vector<MemoryBufferRef>
getMachOFatMemoryBuffers(StringRef Filename, MemoryBuffer &Mem,
                         object::MachOUniversalBinary &Fat) {
  std::vector<MemoryBufferRef> Buffers;
  StringRef FatData = Fat.getData();
  for (auto It = Fat.begin_objects(), End = Fat.end_objects(); It != End;
       ++It) {
    StringRef ObjData = FatData.substr(It->getOffset(), It->getSize());
    Buffers.emplace_back(ObjData, Filename);
  }
  return Buffers;
}

Error BinaryHolder::ArchiveEntry::load(StringRef Filename,
                                       TimestampTy Timestamp, bool Verbose) {
  StringRef ArchiveFilename = getArchiveAndObjectName(Filename).first;

  // Try to load archive and force it to be memory mapped.
  auto ErrOrBuff = MemoryBuffer::getFileOrSTDIN(ArchiveFilename, -1, false);
  if (auto Err = ErrOrBuff.getError())
    return errorCodeToError(Err);

  MemBuffer = std::move(*ErrOrBuff);

  if (Verbose)
    WithColor::note() << "loaded archive '" << ArchiveFilename << "'\n";

  // Load one or more archive buffers, depending on whether we're dealing with
  // a fat binary.
  std::vector<MemoryBufferRef> ArchiveBuffers;

  auto ErrOrFat =
      object::MachOUniversalBinary::create(MemBuffer->getMemBufferRef());
  if (!ErrOrFat) {
    consumeError(ErrOrFat.takeError());
    ArchiveBuffers.push_back(MemBuffer->getMemBufferRef());
  } else {
    FatBinary = std::move(*ErrOrFat);
    FatBinaryName = ArchiveFilename;
    ArchiveBuffers =
        getMachOFatMemoryBuffers(FatBinaryName, *MemBuffer, *FatBinary);
  }

  // Finally, try to load the archives.
  Archives.reserve(ArchiveBuffers.size());
  for (auto MemRef : ArchiveBuffers) {
    auto ErrOrArchive = object::Archive::create(MemRef);
    if (!ErrOrArchive)
      return ErrOrArchive.takeError();
    Archives.push_back(std::move(*ErrOrArchive));
  }

  return Error::success();
}

Error BinaryHolder::ObjectEntry::load(StringRef Filename, bool Verbose) {
  // Try to load regular binary and force it to be memory mapped.
  auto ErrOrBuff = MemoryBuffer::getFileOrSTDIN(Filename, -1, false);
  if (auto Err = ErrOrBuff.getError())
    return errorCodeToError(Err);

  MemBuffer = std::move(*ErrOrBuff);

  if (Verbose)
    WithColor::note() << "loaded object.\n";

  // Load one or more object buffers, depending on whether we're dealing with a
  // fat binary.
  std::vector<MemoryBufferRef> ObjectBuffers;

  auto ErrOrFat =
      object::MachOUniversalBinary::create(MemBuffer->getMemBufferRef());
  if (!ErrOrFat) {
    consumeError(ErrOrFat.takeError());
    ObjectBuffers.push_back(MemBuffer->getMemBufferRef());
  } else {
    FatBinary = std::move(*ErrOrFat);
    FatBinaryName = Filename;
    ObjectBuffers =
        getMachOFatMemoryBuffers(FatBinaryName, *MemBuffer, *FatBinary);
  }

  Objects.reserve(ObjectBuffers.size());
  for (auto MemRef : ObjectBuffers) {
    auto ErrOrObjectFile = object::ObjectFile::createObjectFile(MemRef);
    if (!ErrOrObjectFile)
      return ErrOrObjectFile.takeError();
    Objects.push_back(std::move(*ErrOrObjectFile));
  }

  return Error::success();
}

std::vector<const object::ObjectFile *>
BinaryHolder::ObjectEntry::getObjects() const {
  std::vector<const object::ObjectFile *> Result;
  Result.reserve(Objects.size());
  for (auto &Object : Objects) {
    Result.push_back(Object.get());
  }
  return Result;
}
Expected<const object::ObjectFile &>
BinaryHolder::ObjectEntry::getObject(const Triple &T) const {
  for (const auto &Obj : Objects) {
    if (const auto *MachO = dyn_cast<object::MachOObjectFile>(Obj.get())) {
      if (MachO->getArchTriple().str() == T.str())
        return *MachO;
    } else if (Obj->getArch() == T.getArch())
      return *Obj;
  }
  return errorCodeToError(object::object_error::arch_not_found);
}

Expected<const BinaryHolder::ObjectEntry &>
BinaryHolder::ArchiveEntry::getObjectEntry(StringRef Filename,
                                           TimestampTy Timestamp,
                                           bool Verbose) {
  StringRef ArchiveFilename;
  StringRef ObjectFilename;
  std::tie(ArchiveFilename, ObjectFilename) = getArchiveAndObjectName(Filename);

  // Try the cache first.
  KeyTy Key = {ObjectFilename, Timestamp};

  {
    std::lock_guard<std::mutex> Lock(MemberCacheMutex);
    if (MemberCache.count(Key))
      return MemberCache[Key];
  }

  // Create a new ObjectEntry, but don't add it to the cache yet. Loading of
  // the archive members might fail and we don't want to lock the whole archive
  // during this operation.
  ObjectEntry OE;

  for (const auto &Archive : Archives) {
    Error Err = Error::success();
    for (auto Child : Archive->children(Err)) {
      if (auto NameOrErr = Child.getName()) {
        if (*NameOrErr == ObjectFilename) {
          auto ModTimeOrErr = Child.getLastModified();
          if (!ModTimeOrErr)
            return ModTimeOrErr.takeError();

          if (Timestamp != sys::TimePoint<>() &&
              Timestamp != ModTimeOrErr.get()) {
            if (Verbose)
              WithColor::warning() << "member has timestamp mismatch.\n";
            continue;
          }

          if (Verbose)
            WithColor::note() << "found member in archive.\n";

          auto ErrOrMem = Child.getMemoryBufferRef();
          if (!ErrOrMem)
            return ErrOrMem.takeError();

          auto ErrOrObjectFile =
              object::ObjectFile::createObjectFile(*ErrOrMem);
          if (!ErrOrObjectFile)
            return ErrOrObjectFile.takeError();

          OE.Objects.push_back(std::move(*ErrOrObjectFile));
        }
      }
    }
    if (Err)
      return std::move(Err);
  }

  if (OE.Objects.empty())
    return errorCodeToError(errc::no_such_file_or_directory);

  std::lock_guard<std::mutex> Lock(MemberCacheMutex);
  MemberCache.try_emplace(Key, std::move(OE));
  return MemberCache[Key];
}

Expected<const BinaryHolder::ObjectEntry &>
BinaryHolder::getObjectEntry(StringRef Filename, TimestampTy Timestamp) {
  if (Verbose)
    WithColor::note() << "trying to open '" << Filename << "'\n";

  // If this is an archive, we might have either the object or the archive
  // cached. In this case we can load it without accessing the file system.
  if (isArchive(Filename)) {
    StringRef ArchiveFilename = getArchiveAndObjectName(Filename).first;
    std::lock_guard<std::mutex> Lock(ArchiveCacheMutex);
    if (ArchiveCache.count(ArchiveFilename)) {
      return ArchiveCache[ArchiveFilename].getObjectEntry(Filename, Timestamp,
                                                          Verbose);
    } else {
      ArchiveEntry &AE = ArchiveCache[ArchiveFilename];
      auto Err = AE.load(Filename, Timestamp, Verbose);
      if (Err) {
        ArchiveCache.erase(ArchiveFilename);
        // Don't return the error here: maybe the file wasn't an archive.
        llvm::consumeError(std::move(Err));
      } else {
        return ArchiveCache[ArchiveFilename].getObjectEntry(Filename, Timestamp,
                                                            Verbose);
      }
    }
  }

  // If this is an object, we might have it cached. If not we'll have to load
  // it from the file system and cache it now.
  std::lock_guard<std::mutex> Lock(ObjectCacheMutex);
  if (!ObjectCache.count(Filename)) {
    ObjectEntry &OE = ObjectCache[Filename];
    auto Err = OE.load(Filename, Verbose);
    if (Err) {
      ObjectCache.erase(Filename);
      return std::move(Err);
    }
  }

  return ObjectCache[Filename];
}

void BinaryHolder::clear() {
  std::lock_guard<std::mutex> ArchiveLock(ArchiveCacheMutex);
  std::lock_guard<std::mutex> ObjectLock(ObjectCacheMutex);
  ArchiveCache.clear();
  ObjectCache.clear();
}

} // namespace dsymutil
} // namespace llvm
