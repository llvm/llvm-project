//===- Offloading.cpp - Utilities for handling offloading code  -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/OffloadBinary.h"

#include "llvm/ADT/StringSwitch.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/MC/StringTableBuilder.h"
#include "llvm/Object/Error.h"
#include "llvm/Support/FileOutputBuffer.h"

using namespace llvm;
using namespace llvm::object;

Expected<std::unique_ptr<OffloadBinary>>
OffloadBinary::create(MemoryBufferRef Buf) {
  if (Buf.getBufferSize() < sizeof(Header) + sizeof(Entry))
    return errorCodeToError(object_error::parse_failed);

  // Check for 0x10FF1OAD magic bytes.
  if (identify_magic(Buf.getBuffer()) != file_magic::offload_binary)
    return errorCodeToError(object_error::parse_failed);

  const char *Start = Buf.getBufferStart();
  const Header *TheHeader = reinterpret_cast<const Header *>(Start);
  const Entry *TheEntry =
      reinterpret_cast<const Entry *>(&Start[TheHeader->EntryOffset]);

  // Make sure the offsets are inside the file.
  if (TheHeader->EntryOffset > Buf.getBufferSize() ||
      TheEntry->ImageOffset > Buf.getBufferSize() ||
      TheEntry->StringOffset > Buf.getBufferSize())
    return errorCodeToError(object_error::unexpected_eof);

  return std::unique_ptr<OffloadBinary>(
      new OffloadBinary(Buf, TheHeader, TheEntry));
}

std::unique_ptr<MemoryBuffer>
OffloadBinary::write(const OffloadingImage &OffloadingData) {
  // Create a null-terminated string table with all the used strings.
  StringTableBuilder StrTab(StringTableBuilder::ELF);
  for (auto &KeyAndValue : OffloadingData.StringData) {
    StrTab.add(KeyAndValue.getKey());
    StrTab.add(KeyAndValue.getValue());
  }
  StrTab.finalize();

  uint64_t StringEntrySize =
      sizeof(StringEntry) * OffloadingData.StringData.size();

  // Make sure the image we're wrapping around is aligned as well.
  uint64_t BinaryDataSize = alignTo(sizeof(Header) + sizeof(Entry) +
                                        StringEntrySize + StrTab.getSize(),
                                    getAlignment());

  // Create the header and fill in the offsets. The entry will be directly
  // placed after the header in memory. Align the size to the alignment of the
  // header so this can be placed contiguously in a single section.
  Header TheHeader;
  TheHeader.Size = alignTo(
      BinaryDataSize + OffloadingData.Image->getBufferSize(), getAlignment());
  TheHeader.EntryOffset = sizeof(Header);
  TheHeader.EntrySize = sizeof(Entry);

  // Create the entry using the string table offsets. The string table will be
  // placed directly after the entry in memory, and the image after that.
  Entry TheEntry;
  TheEntry.TheImageKind = OffloadingData.TheImageKind;
  TheEntry.TheOffloadKind = OffloadingData.TheOffloadKind;
  TheEntry.Flags = OffloadingData.Flags;
  TheEntry.StringOffset = sizeof(Header) + sizeof(Entry);
  TheEntry.NumStrings = OffloadingData.StringData.size();

  TheEntry.ImageOffset = BinaryDataSize;
  TheEntry.ImageSize = OffloadingData.Image->getBufferSize();

  SmallVector<char> Data;
  Data.reserve(TheHeader.Size);
  raw_svector_ostream OS(Data);
  OS << StringRef(reinterpret_cast<char *>(&TheHeader), sizeof(Header));
  OS << StringRef(reinterpret_cast<char *>(&TheEntry), sizeof(Entry));
  for (auto &KeyAndValue : OffloadingData.StringData) {
    uint64_t Offset = sizeof(Header) + sizeof(Entry) + StringEntrySize;
    StringEntry Map{Offset + StrTab.getOffset(KeyAndValue.getKey()),
                    Offset + StrTab.getOffset(KeyAndValue.getValue())};
    OS << StringRef(reinterpret_cast<char *>(&Map), sizeof(StringEntry));
  }
  StrTab.write(OS);
  // Add padding to required image alignment.
  OS.write_zeros(TheEntry.ImageOffset - OS.tell());
  OS << OffloadingData.Image->getBuffer();

  // Add final padding to required alignment.
  assert(TheHeader.Size >= OS.tell() && "Too much data written?");
  OS.write_zeros(TheHeader.Size - OS.tell());
  assert(TheHeader.Size == OS.tell() && "Size mismatch");

  return MemoryBuffer::getMemBufferCopy(OS.str());
}

OffloadKind object::getOffloadKind(StringRef Name) {
  return llvm::StringSwitch<OffloadKind>(Name)
      .Case("openmp", OFK_OpenMP)
      .Case("cuda", OFK_Cuda)
      .Case("hip", OFK_HIP)
      .Default(OFK_None);
}

StringRef object::getOffloadKindName(OffloadKind Kind) {
  switch (Kind) {
  case OFK_OpenMP:
    return "openmp";
  case OFK_Cuda:
    return "cuda";
  case OFK_HIP:
    return "hip";
  default:
    return "none";
  }
}

ImageKind object::getImageKind(StringRef Name) {
  return llvm::StringSwitch<ImageKind>(Name)
      .Case("o", IMG_Object)
      .Case("bc", IMG_Bitcode)
      .Case("cubin", IMG_Cubin)
      .Case("fatbin", IMG_Fatbinary)
      .Case("s", IMG_PTX)
      .Default(IMG_None);
}

StringRef object::getImageKindName(ImageKind Kind) {
  switch (Kind) {
  case IMG_Object:
    return "o";
  case IMG_Bitcode:
    return "bc";
  case IMG_Cubin:
    return "cubin";
  case IMG_Fatbinary:
    return "fatbin";
  case IMG_PTX:
    return "s";
  default:
    return "";
  }
}
