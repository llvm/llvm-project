//===- ModuleDebugFileChecksumFragment.cpp ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/ModuleDebugFileChecksumFragment.h"

#include "llvm/DebugInfo/CodeView/CodeViewError.h"
#include "llvm/Support/BinaryStreamReader.h"

using namespace llvm;
using namespace llvm::codeview;

struct FileChecksumEntryHeader {
  using ulittle32_t = support::ulittle32_t;

  ulittle32_t FileNameOffset; // Byte offset of filename in global string table.
  uint8_t ChecksumSize;       // Number of bytes of checksum.
  uint8_t ChecksumKind;       // FileChecksumKind
                              // Checksum bytes follow.
};

Error llvm::VarStreamArrayExtractor<FileChecksumEntry>::extract(
    BinaryStreamRef Stream, uint32_t &Len, FileChecksumEntry &Item, void *Ctx) {
  BinaryStreamReader Reader(Stream);

  const FileChecksumEntryHeader *Header;
  if (auto EC = Reader.readObject(Header))
    return EC;

  Item.FileNameOffset = Header->FileNameOffset;
  Item.Kind = static_cast<FileChecksumKind>(Header->ChecksumKind);
  if (auto EC = Reader.readBytes(Item.Checksum, Header->ChecksumSize))
    return EC;

  Len = alignTo(Header->ChecksumSize + sizeof(FileChecksumEntryHeader), 4);
  return Error::success();
}

Error ModuleDebugFileChecksumFragmentRef::initialize(
    BinaryStreamReader Reader) {
  if (auto EC = Reader.readArray(Checksums, Reader.bytesRemaining()))
    return EC;

  return Error::success();
}

ModuleDebugFileChecksumFragment::ModuleDebugFileChecksumFragment()
    : ModuleDebugFragment(ModuleDebugFragmentKind::FileChecksums) {}

void ModuleDebugFileChecksumFragment::addChecksum(uint32_t StringTableOffset,
                                                  FileChecksumKind Kind,
                                                  ArrayRef<uint8_t> Bytes) {
  FileChecksumEntry Entry;
  if (!Bytes.empty()) {
    uint8_t *Copy = Storage.Allocate<uint8_t>(Bytes.size());
    ::memcpy(Copy, Bytes.data(), Bytes.size());
    Entry.Checksum = makeArrayRef(Copy, Bytes.size());
  }
  Entry.FileNameOffset = StringTableOffset;
  Entry.Kind = Kind;
  Checksums.push_back(Entry);

  // This maps the offset of this string in the string table to the offset
  // of this checksum entry in the checksum buffer.
  OffsetMap[StringTableOffset] = SerializedSize;
  assert(SerializedSize % 4 == 0);

  uint32_t Len = alignTo(sizeof(FileChecksumEntryHeader) + Bytes.size(), 4);
  SerializedSize += Len;
}

uint32_t ModuleDebugFileChecksumFragment::calculateSerializedLength() {
  return SerializedSize;
}

Error ModuleDebugFileChecksumFragment::commit(BinaryStreamWriter &Writer) {
  for (const auto &FC : Checksums) {
    FileChecksumEntryHeader Header;
    Header.ChecksumKind = uint8_t(FC.Kind);
    Header.ChecksumSize = FC.Checksum.size();
    Header.FileNameOffset = FC.FileNameOffset;
    if (auto EC = Writer.writeObject(Header))
      return EC;
    if (auto EC = Writer.writeArray(makeArrayRef(FC.Checksum)))
      return EC;
    if (auto EC = Writer.padToAlignment(4))
      return EC;
  }
  return Error::success();
}

uint32_t ModuleDebugFileChecksumFragment::mapChecksumOffset(
    uint32_t StringTableOffset) const {
  auto Iter = OffsetMap.find(StringTableOffset);
  assert(Iter != OffsetMap.end());
  return Iter->second;
}
