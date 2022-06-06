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
static Error readInteger(StringRef Buffer, const char *Src, T &Val) {
  static_assert(std::is_integral<T>::value,
                "Cannot call readInteger on non-integral type.");
  assert(reinterpret_cast<uintptr_t>(Src) % alignof(T) == 0 &&
         "Unaligned read of value from buffer!");
  // Don't read before the beginning or past the end of the file
  if (Src < Buffer.begin() || Src + sizeof(T) > Buffer.end())
    return parseFailed("Reading structure out of file bounds");

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

Error DXContainer::parsePartOffsets() {
  const char *Current = Data.getBuffer().data() + sizeof(dxbc::Header);
  for (uint32_t Part = 0; Part < Header.PartCount; ++Part) {
    uint32_t PartOffset;
    if (Error Err = readInteger(Data.getBuffer(), Current, PartOffset))
      return Err;
    Current += sizeof(uint32_t);
    // We need to ensure that each part offset leaves enough space for a part
    // header. To prevent overflow, we subtract the part header size from the
    // buffer size, rather than adding to the offset. Since the file header is
    // larger than the part header we can't reach this code unless the buffer
    // is larger than the part header, so this can't underflow.
    if (PartOffset > Data.getBufferSize() - sizeof(dxbc::PartHeader))
      return parseFailed("Part offset points beyond boundary of the file");
    PartOffsets.push_back(PartOffset);
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
