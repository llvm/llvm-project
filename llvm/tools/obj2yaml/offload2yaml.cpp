//===------ offload2yaml.cpp - obj2yaml conversion tool ---*- C++ -------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "obj2yaml.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Object/OffloadBinary.h"
#include "llvm/ObjectYAML/OffloadYAML.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/StringSaver.h"

#include <memory>

using namespace llvm;

namespace {

void populateYAML(OffloadYAML::Binary &YAMLBinary,
                  ArrayRef<std::unique_ptr<object::OffloadBinary>> OBinaries,
                  UniqueStringSaver Saver) {
  for (const auto &OBinaryPtr : OBinaries) {
    object::OffloadBinary &OB = *OBinaryPtr;

    YAMLBinary.Members.emplace_back();
    auto &Member = YAMLBinary.Members.back();
    Member.ImageKind = OB.getImageKind();
    Member.OffloadKind = OB.getOffloadKind();
    Member.Flags = OB.getFlags();
    if (!OB.strings().empty()) {
      Member.StringEntries = std::vector<OffloadYAML::Binary::StringEntry>();
      for (const auto &StringEntry : OB.strings())
        Member.StringEntries->emplace_back(OffloadYAML::Binary::StringEntry(
            {Saver.save(StringEntry.first), Saver.save(StringEntry.second)}));
    }

    if (!OB.getImage().empty())
      Member.Content = arrayRefFromStringRef(OB.getImage());
  }
}

Expected<OffloadYAML::Binary *> dump(MemoryBufferRef Source,
                                     UniqueStringSaver Saver) {
  std::unique_ptr<OffloadYAML::Binary> YAMLBinary =
      std::make_unique<OffloadYAML::Binary>();

  YAMLBinary->Members = std::vector<OffloadYAML::Binary::Member>();

  uint64_t Offset = 0;
  while (Offset < Source.getBufferSize()) {
    MemoryBufferRef Buffer = MemoryBufferRef(
        Source.getBuffer().drop_front(Offset), Source.getBufferIdentifier());
    std::unique_ptr<MemoryBuffer> Aligned;
    if (!isAddrAligned(Align(object::OffloadBinary::getAlignment()),
                       Buffer.getBufferStart())) {
      Aligned = MemoryBuffer::getMemBufferCopy(Buffer.getBuffer(),
                                               Buffer.getBufferIdentifier());
      Buffer = *Aligned;
    }
    auto HeaderOrErr = object::OffloadBinary::extractHeader(Buffer);
    if (!HeaderOrErr)
      return HeaderOrErr.takeError();
    const object::OffloadBinary::Header *TheHeader = *HeaderOrErr;
    uint64_t Size = TheHeader->Size;
    if (TheHeader->Version >= 3 && TheHeader->InflatedSize != 0) {
      StringRef Payload = Buffer.getBuffer().take_front(Size).drop_front(
          TheHeader->EntriesOffset);
      switch (identify_magic(Payload)) {
      case file_magic::zstd:
        YAMLBinary->Compression = compression::Format::Zstd;
        break;
      case file_magic::zlib:
        YAMLBinary->Compression = compression::Format::Zlib;
        break;
      default:
        return createStringError("unknown compression format");
      }
    }
    auto BinariesOrErr = object::OffloadBinary::create(Buffer);
    if (!BinariesOrErr)
      return BinariesOrErr.takeError();

    SmallVector<std::unique_ptr<object::OffloadBinary>> &Binaries =
        *BinariesOrErr;
    populateYAML(*YAMLBinary, Binaries, Saver);

    Offset =
        alignTo(Offset + Size, Align(object::OffloadBinary::getAlignment()));
  }

  return YAMLBinary.release();
}

} // namespace

Error offload2yaml(raw_ostream &Out, MemoryBufferRef Source) {
  BumpPtrAllocator Alloc;
  UniqueStringSaver Saver(Alloc);

  Expected<OffloadYAML::Binary *> YAMLOrErr = dump(Source, Saver);
  if (!YAMLOrErr)
    return YAMLOrErr.takeError();

  std::unique_ptr<OffloadYAML::Binary> YAML(YAMLOrErr.get());
  yaml::Output Yout(Out);
  Yout << *YAML;

  return Error::success();
}
