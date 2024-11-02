//===-- LVReaderHandler.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This class implements the Reader Handler.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/LogicalView/LVReaderHandler.h"
#include "llvm/DebugInfo/LogicalView/Core/LVCompare.h"
#include "llvm/DebugInfo/LogicalView/Readers/LVELFReader.h"

using namespace llvm;
using namespace llvm::object;
using namespace llvm::pdb;
using namespace llvm::logicalview;

#define DEBUG_TYPE "ReaderHandler"

Error LVReaderHandler::process() {
  if (Error Err = createReaders())
    return Err;
  if (Error Err = printReaders())
    return Err;
  if (Error Err = compareReaders())
    return Err;

  return Error::success();
}

void LVReaderHandler::destroyReaders() {
  LLVM_DEBUG(dbgs() << "destroyReaders\n");
  for (const LVReader *Reader : TheReaders)
    delete Reader;
}

Error LVReaderHandler::createReader(StringRef Filename, LVReaders &Readers,
                                    PdbOrObj &Input, StringRef FileFormatName,
                                    StringRef ExePath) {
  auto CreateOneReader = [&]() -> LVReader * {
    if (Input.is<ObjectFile *>()) {
      ObjectFile &Obj = *Input.get<ObjectFile *>();
      if (Obj.isELF() || Obj.isMachO())
        return new LVELFReader(Filename, FileFormatName, Obj, W);
    }
    return nullptr;
  };

  LVReader *Reader = CreateOneReader();
  if (!Reader)
    return createStringError(errc::invalid_argument,
                             "unable to create reader for: '%s'",
                             Filename.str().c_str());

  Readers.push_back(Reader);
  return Reader->doLoad();
}

Error LVReaderHandler::handleArchive(LVReaders &Readers, StringRef Filename,
                                     Archive &Arch) {
  Error Err = Error::success();
  for (const Archive::Child &Child : Arch.children(Err)) {
    Expected<MemoryBufferRef> BuffOrErr = Child.getMemoryBufferRef();
    if (Error Err = BuffOrErr.takeError())
      return createStringError(errorToErrorCode(std::move(Err)), "%s",
                               Filename.str().c_str());
    Expected<StringRef> NameOrErr = Child.getName();
    if (Error Err = NameOrErr.takeError())
      return createStringError(errorToErrorCode(std::move(Err)), "%s",
                               Filename.str().c_str());
    std::string Name = (Filename + "(" + NameOrErr.get() + ")").str();
    if (Error Err = handleBuffer(Readers, Name, BuffOrErr.get()))
      return createStringError(errorToErrorCode(std::move(Err)), "%s",
                               Filename.str().c_str());
  }

  return Error::success();
}

Error LVReaderHandler::handleBuffer(LVReaders &Readers, StringRef Filename,
                                    MemoryBufferRef Buffer, StringRef ExePath) {
  Expected<std::unique_ptr<Binary>> BinOrErr = createBinary(Buffer);
  if (errorToErrorCode(BinOrErr.takeError())) {
    return createStringError(errc::not_supported,
                             "Binary object format in '%s' is not supported.",
                             Filename.str().c_str());
  }
  return handleObject(Readers, Filename, *BinOrErr.get());
}

Error LVReaderHandler::handleFile(LVReaders &Readers, StringRef Filename,
                                  StringRef ExePath) {
  // Convert any Windows backslashes into forward slashes to get the path.
  std::string ConvertedPath =
      sys::path::convert_to_slash(Filename, sys::path::Style::windows);
  ErrorOr<std::unique_ptr<MemoryBuffer>> BuffOrErr =
      MemoryBuffer::getFileOrSTDIN(ConvertedPath);
  if (BuffOrErr.getError()) {
    return createStringError(errc::bad_file_descriptor,
                             "File '%s' does not exist.",
                             ConvertedPath.c_str());
  }
  std::unique_ptr<MemoryBuffer> Buffer = std::move(BuffOrErr.get());
  return handleBuffer(Readers, ConvertedPath, *Buffer, ExePath);
}

Error LVReaderHandler::handleMach(LVReaders &Readers, StringRef Filename,
                                  MachOUniversalBinary &Mach) {
  for (const MachOUniversalBinary::ObjectForArch &ObjForArch : Mach.objects()) {
    std::string ObjName = (Twine(Filename) + Twine("(") +
                           Twine(ObjForArch.getArchFlagName()) + Twine(")"))
                              .str();
    if (Expected<std::unique_ptr<MachOObjectFile>> MachOOrErr =
            ObjForArch.getAsObjectFile()) {
      MachOObjectFile &Obj = **MachOOrErr;
      PdbOrObj Input = &Obj;
      if (Error Err =
              createReader(Filename, Readers, Input, Obj.getFileFormatName()))
        return Err;
      continue;
    } else
      consumeError(MachOOrErr.takeError());
    if (Expected<std::unique_ptr<Archive>> ArchiveOrErr =
            ObjForArch.getAsArchive()) {
      if (Error Err = handleArchive(Readers, ObjName, *ArchiveOrErr.get()))
        return Err;
      continue;
    } else
      consumeError(ArchiveOrErr.takeError());
  }
  return Error::success();
}

Error LVReaderHandler::handleObject(LVReaders &Readers, StringRef Filename,
                                    Binary &Binary) {
  if (PdbOrObj Input = dyn_cast<ObjectFile>(&Binary))
    return createReader(Filename, Readers, Input,
                        Input.get<ObjectFile *>()->getFileFormatName());

  if (MachOUniversalBinary *Fat = dyn_cast<MachOUniversalBinary>(&Binary))
    return handleMach(Readers, Filename, *Fat);

  if (Archive *Arch = dyn_cast<Archive>(&Binary))
    return handleArchive(Readers, Filename, *Arch);

  return createStringError(errc::not_supported,
                           "Binary object format in '%s' is not supported.",
                           Filename.str().c_str());
}

Error LVReaderHandler::createReaders() {
  LLVM_DEBUG(dbgs() << "createReaders\n");
  for (std::string &Object : Objects) {
    LVReaders Readers;
    if (Error Err = createReader(Object, Readers))
      return Err;
    TheReaders.insert(TheReaders.end(), Readers.begin(), Readers.end());
  }

  return Error::success();
}

Error LVReaderHandler::printReaders() {
  LLVM_DEBUG(dbgs() << "printReaders\n");
  if (options().getPrintExecute())
    for (LVReader *Reader : TheReaders)
      if (Error Err = Reader->doPrint())
        return Err;

  return Error::success();
}

Error LVReaderHandler::compareReaders() {
  LLVM_DEBUG(dbgs() << "compareReaders\n");
  size_t ReadersCount = TheReaders.size();
  if (options().getCompareExecute() && ReadersCount >= 2) {
    // If we have more than 2 readers, compare them by pairs.
    size_t ViewPairs = ReadersCount / 2;
    LVCompare Compare(OS);
    for (size_t Pair = 0, Index = 0; Pair < ViewPairs; ++Pair) {
      if (Error Err = Compare.execute(TheReaders[Index], TheReaders[Index + 1]))
        return Err;
      Index += 2;
    }
  }

  return Error::success();
}

void LVReaderHandler::print(raw_ostream &OS) const { OS << "ReaderHandler\n"; }
