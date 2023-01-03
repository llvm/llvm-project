//===- DXContainer.cpp - DXContainer object file implementation -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/DXContainer.h"
#include "llvm/BinaryFormat/DXContainer.h"
#include "llvm/Object/Error.h"
#include "llvm/Support/FormatVariadic.h"

using namespace llvm;
using namespace llvm::object;

static Error parseFailed(const Twine &Msg) {
  return make_error<GenericBinaryError>(Msg.str(), object_error::parse_failed);
}

template <typename T>
static Error readStruct(StringRef Buffer, const char *Src, T &Struct) {
  // Don't read before the beginning or past the end of the file
  if (Src < Buffer.begin() || Src + sizeof(T) > Buffer.end())
    return parseFailed("Reading structure out of file bounds");

  memcpy(&Struct, Src, sizeof(T));
  // DXContainer is always little endian
  if (sys::IsBigEndianHost)
    Struct.swapBytes();
  return Error::success();
}

template <typename T>
static Error readInteger(StringRef Buffer, const char *Src, T &Val,
                         Twine Str = "structure") {
  static_assert(std::is_integral_v<T>,
                "Cannot call readInteger on non-integral type.");
  // Don't read before the beginning or past the end of the file
  if (Src < Buffer.begin() || Src + sizeof(T) > Buffer.end())
    return parseFailed(Twine("Reading ") + Str + " out of file bounds");

  // The DXContainer offset table is comprised of uint32_t values but not padded
  // to a 64-bit boundary. So Parts may start unaligned if there is an odd
  // number of parts and part data itself is not required to be padded.
  if (reinterpret_cast<uintptr_t>(Src) % alignof(T) != 0)
    memcpy(reinterpret_cast<char *>(&Val), Src, sizeof(T));
  else
    Val = *reinterpret_cast<const T *>(Src);
  // DXContainer is always little endian
  if (sys::IsBigEndianHost)
    sys::swapByteOrder(Val);
  return Error::success();
}

DXContainer::DXContainer(MemoryBufferRef O) : Data(O) {}

Error DXContainer::parseHeader() {
  return readStruct(Data.getBuffer(), Data.getBuffer().data(), Header);
}

Error DXContainer::parseDXILHeader(StringRef Part) {
  if (DXIL)
    return parseFailed("More than one DXIL part is present in the file");
  const char *Current = Part.begin();
  dxbc::ProgramHeader Header;
  if (Error Err = readStruct(Part, Current, Header))
    return Err;
  Current += offsetof(dxbc::ProgramHeader, Bitcode) + Header.Bitcode.Offset;
  DXIL.emplace(std::make_pair(Header, Current));
  return Error::success();
}

Error DXContainer::parseShaderFlags(StringRef Part) {
  if (ShaderFlags)
    return parseFailed("More than one SFI0 part is present in the file");
  uint64_t FlagValue = 0;
  if (Error Err = readInteger(Part, Part.begin(), FlagValue))
    return Err;
  ShaderFlags = FlagValue;
  return Error::success();
}

Error DXContainer::parseHash(StringRef Part) {
  if (Hash)
    return parseFailed("More than one HASH part is present in the file");
  dxbc::ShaderHash ReadHash;
  if (Error Err = readStruct(Part, Part.begin(), ReadHash))
    return Err;
  Hash = ReadHash;
  return Error::success();
}

Error DXContainer::parsePartOffsets() {
  uint32_t LastOffset =
      sizeof(dxbc::Header) + (Header.PartCount * sizeof(uint32_t));
  const char *Current = Data.getBuffer().data() + sizeof(dxbc::Header);
  for (uint32_t Part = 0; Part < Header.PartCount; ++Part) {
    uint32_t PartOffset;
    if (Error Err = readInteger(Data.getBuffer(), Current, PartOffset))
      return Err;
    if (PartOffset < LastOffset)
      return parseFailed(
          formatv(
              "Part offset for part {0} begins before the previous part ends",
              Part)
              .str());
    Current += sizeof(uint32_t);
    if (PartOffset >= Data.getBufferSize())
      return parseFailed("Part offset points beyond boundary of the file");
    // To prevent overflow when reading the part name, we subtract the part name
    // size from the buffer size, rather than adding to the offset. Since the
    // file header is larger than the part header we can't reach this code
    // unless the buffer is at least as large as a part header, so this
    // subtraction can't underflow.
    if (PartOffset >= Data.getBufferSize() - sizeof(dxbc::PartHeader::Name))
      return parseFailed("File not large enough to read part name");
    PartOffsets.push_back(PartOffset);

    dxbc::PartType PT =
        dxbc::parsePartType(Data.getBuffer().substr(PartOffset, 4));
    uint32_t PartDataStart = PartOffset + sizeof(dxbc::PartHeader);
    uint32_t PartSize;
    if (Error Err = readInteger(Data.getBuffer(),
                                Data.getBufferStart() + PartOffset + 4,
                                PartSize, "part size"))
      return Err;
    StringRef PartData = Data.getBuffer().substr(PartDataStart, PartSize);
    LastOffset = PartOffset + PartSize;
    switch (PT) {
    case dxbc::PartType::DXIL:
      if (Error Err = parseDXILHeader(PartData))
        return Err;
      break;
    case dxbc::PartType::SFI0:
      if (Error Err = parseShaderFlags(PartData))
        return Err;
      break;
    case dxbc::PartType::HASH:
      if (Error Err = parseHash(PartData))
        return Err;
      break;
    case dxbc::PartType::Unknown:
      break;
    }
  }
  return Error::success();
}

Expected<DXContainer> DXContainer::create(MemoryBufferRef Object) {
  DXContainer Container(Object);
  if (Error Err = Container.parseHeader())
    return std::move(Err);
  if (Error Err = Container.parsePartOffsets())
    return std::move(Err);
  return Container;
}

void DXContainer::PartIterator::updateIteratorImpl(const uint32_t Offset) {
  StringRef Buffer = Container.Data.getBuffer();
  const char *Current = Buffer.data() + Offset;
  // Offsets are validated during parsing, so all offsets in the container are
  // valid and contain enough readable data to read a header.
  cantFail(readStruct(Buffer, Current, IteratorState.Part));
  IteratorState.Data =
      StringRef(Current + sizeof(dxbc::PartHeader), IteratorState.Part.Size);
  IteratorState.Offset = Offset;
}
