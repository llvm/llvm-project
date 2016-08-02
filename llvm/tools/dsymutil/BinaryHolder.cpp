//===-- BinaryHolder.cpp --------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This program is a utility that aims to be a dropin replacement for
// Darwin's dsymutil.
//
//===----------------------------------------------------------------------===//

#include "BinaryHolder.h"
#include "llvm/Object/MachO.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
namespace dsymutil {

Triple BinaryHolder::getTriple(const object::MachOObjectFile &Obj) {
  // If a ThumbTriple is returned, use it instead of the standard
  // one. This is because the thumb triple always allows to create a
  // target, whereas the non-thumb one might not.
  Triple ThumbTriple;
  Triple T = Obj.getArch(nullptr, &ThumbTriple);
  return ThumbTriple.getArch() ? ThumbTriple : T;
}

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

void BinaryHolder::changeBackingMemoryBuffer(
    std::unique_ptr<MemoryBuffer> &&Buf) {
  CurrentArchives.clear();
  CurrentObjectFiles.clear();
  CurrentFatBinary.reset();

  CurrentMemoryBuffer = std::move(Buf);
}

ErrorOr<std::vector<MemoryBufferRef>>
BinaryHolder::GetMemoryBuffersForFile(StringRef Filename,
                                      sys::TimeValue Timestamp) {
  if (Verbose)
    outs() << "trying to open '" << Filename << "'\n";

  // Try that first as it doesn't involve any filesystem access.
  if (auto ErrOrArchiveMembers = GetArchiveMemberBuffers(Filename, Timestamp))
    return *ErrOrArchiveMembers;

  // If the name ends with a closing paren, there is a huge chance
  // it is an archive member specification.
  if (Filename.endswith(")"))
    if (auto ErrOrArchiveMembers =
            MapArchiveAndGetMemberBuffers(Filename, Timestamp))
      return *ErrOrArchiveMembers;

  // Otherwise, just try opening a standard file. If this is an
  // archive member specifiaction and any of the above didn't handle it
  // (either because the archive is not there anymore, or because the
  // archive doesn't contain the requested member), this will still
  // provide a sensible error message.
  auto ErrOrFile = MemoryBuffer::getFileOrSTDIN(Filename);
  if (auto Err = ErrOrFile.getError())
    return Err;

  changeBackingMemoryBuffer(std::move(*ErrOrFile));
  if (Verbose)
    outs() << "\tloaded file.\n";

  auto ErrOrFat = object::MachOUniversalBinary::create(
      CurrentMemoryBuffer->getMemBufferRef());
  if (ErrOrFat.getError()) {
    // Not a fat binary must be a standard one. Return a one element vector.
    return std::vector<MemoryBufferRef>{CurrentMemoryBuffer->getMemBufferRef()};
  }

  CurrentFatBinary = std::move(*ErrOrFat);
  return getMachOFatMemoryBuffers(Filename, *CurrentMemoryBuffer,
                                  *CurrentFatBinary);
}

ErrorOr<std::vector<MemoryBufferRef>>
BinaryHolder::GetArchiveMemberBuffers(StringRef Filename,
                                      sys::TimeValue Timestamp) {
  if (CurrentArchives.empty())
    return make_error_code(errc::no_such_file_or_directory);

  StringRef CurArchiveName = CurrentArchives.front()->getFileName();
  if (!Filename.startswith(Twine(CurArchiveName, "(").str()))
    return make_error_code(errc::no_such_file_or_directory);

  // Remove the archive name and the parens around the archive member name.
  Filename = Filename.substr(CurArchiveName.size() + 1).drop_back();

  std::vector<MemoryBufferRef> Buffers;
  Buffers.reserve(CurrentArchives.size());

  for (const auto &CurrentArchive : CurrentArchives) {
    for (auto ChildOrErr : CurrentArchive->children()) {
      if (std::error_code Err = ChildOrErr.getError())
        return Err;
      const auto &Child = *ChildOrErr;
      if (auto NameOrErr = Child.getName()) {
        if (*NameOrErr == Filename) {
          if (Timestamp != sys::TimeValue::PosixZeroTime() &&
              Timestamp != Child.getLastModified()) {
            if (Verbose)
              outs() << "\tmember had timestamp mismatch.\n";
            continue;
          }
          if (Verbose)
            outs() << "\tfound member in current archive.\n";
          auto ErrOrMem = Child.getMemoryBufferRef();
          if (auto Err = ErrOrMem.getError())
            return Err;
          Buffers.push_back(*ErrOrMem);
        }
      }
    }
  }

  if (Buffers.empty())
    return make_error_code(errc::no_such_file_or_directory);
  return Buffers;
}

ErrorOr<std::vector<MemoryBufferRef>>
BinaryHolder::MapArchiveAndGetMemberBuffers(StringRef Filename,
                                            sys::TimeValue Timestamp) {
  StringRef ArchiveFilename = Filename.substr(0, Filename.find('('));

  auto ErrOrBuff = MemoryBuffer::getFileOrSTDIN(ArchiveFilename);
  if (auto Err = ErrOrBuff.getError())
    return Err;

  if (Verbose)
    outs() << "\topened new archive '" << ArchiveFilename << "'\n";

  changeBackingMemoryBuffer(std::move(*ErrOrBuff));
  std::vector<MemoryBufferRef> ArchiveBuffers;
  auto ErrOrFat = object::MachOUniversalBinary::create(
      CurrentMemoryBuffer->getMemBufferRef());
  if (ErrOrFat.getError()) {
    // Not a fat binary must be a standard one.
    ArchiveBuffers.push_back(CurrentMemoryBuffer->getMemBufferRef());
  } else {
    CurrentFatBinary = std::move(*ErrOrFat);
    ArchiveBuffers = getMachOFatMemoryBuffers(
        ArchiveFilename, *CurrentMemoryBuffer, *CurrentFatBinary);
  }

  for (auto MemRef : ArchiveBuffers) {
    auto ErrOrArchive = object::Archive::create(MemRef);
    if (auto Err = ErrOrArchive.getError())
      return Err;
    CurrentArchives.push_back(std::move(*ErrOrArchive));
  }
  return GetArchiveMemberBuffers(Filename, Timestamp);
}

ErrorOr<const object::ObjectFile &>
BinaryHolder::getObjfileForArch(const Triple &T) {
  for (const auto &Obj : CurrentObjectFiles) {
    if (const auto *MachO = dyn_cast<object::MachOObjectFile>(Obj.get())) {
      if (getTriple(*MachO).str() == T.str())
        return *MachO;
    } else if (Obj->getArch() == T.getArch())
      return *Obj;
  }

  return make_error_code(object::object_error::arch_not_found);
}

ErrorOr<std::vector<const object::ObjectFile *>>
BinaryHolder::GetObjectFiles(StringRef Filename, sys::TimeValue Timestamp) {
  auto ErrOrMemBufferRefs = GetMemoryBuffersForFile(Filename, Timestamp);
  if (auto Err = ErrOrMemBufferRefs.getError())
    return Err;

  std::vector<const object::ObjectFile *> Objects;
  Objects.reserve(ErrOrMemBufferRefs->size());

  CurrentObjectFiles.clear();
  for (auto MemBuf : *ErrOrMemBufferRefs) {
    auto ErrOrObjectFile = object::ObjectFile::createObjectFile(MemBuf);
    if (auto Err = ErrOrObjectFile.getError())
      return Err;

    Objects.push_back(ErrOrObjectFile->get());
    CurrentObjectFiles.push_back(std::move(*ErrOrObjectFile));
  }

  return std::move(Objects);
}
}
}
