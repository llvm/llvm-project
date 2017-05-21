//===-- WindowsResource.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the .res file class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/WindowsResource.h"
#include "llvm/Object/Error.h"
#include <system_error>

namespace llvm {
namespace object {

static const size_t ResourceMagicSize = 16;

static const size_t NullEntrySize = 16;

#define RETURN_IF_ERROR(X)                                                     \
  if (auto EC = X)                                                             \
    return EC;

WindowsResource::WindowsResource(MemoryBufferRef Source)
    : Binary(Binary::ID_WinRes, Source) {
  size_t LeadingSize = ResourceMagicSize + NullEntrySize;
  BBS = BinaryByteStream(Data.getBuffer().drop_front(LeadingSize),
                         support::little);
}

WindowsResource::~WindowsResource() = default;

Expected<std::unique_ptr<WindowsResource>>
WindowsResource::createWindowsResource(MemoryBufferRef Source) {
  if (Source.getBufferSize() < ResourceMagicSize + NullEntrySize)
    return make_error<GenericBinaryError>(
        "File too small to be a resource file",
        object_error::invalid_file_type);
  std::unique_ptr<WindowsResource> Ret(new WindowsResource(Source));
  return std::move(Ret);
}

Expected<ResourceEntryRef> WindowsResource::getHeadEntry() {
  Error Err = Error::success();
  auto Ref = ResourceEntryRef(BinaryStreamRef(BBS), this, Err);
  if (Err)
    return std::move(Err);
  return Ref;
}

ResourceEntryRef::ResourceEntryRef(BinaryStreamRef Ref,
                                   const WindowsResource *Owner, Error &Err)
    : Reader(Ref), OwningRes(Owner) {
  if (loadNext())
    Err = make_error<GenericBinaryError>("Could not read first entry.",
                                         object_error::unexpected_eof);
}

Error ResourceEntryRef::moveNext(bool &End) {
  // Reached end of all the entries.
  if (Reader.bytesRemaining() == 0) {
    End = true;
    return Error::success();
  }
  RETURN_IF_ERROR(loadNext());

  return Error::success();
}

Error ResourceEntryRef::loadNext() {
  uint32_t DataSize;
  RETURN_IF_ERROR(Reader.readInteger(DataSize));
  uint32_t HeaderSize;
  RETURN_IF_ERROR(Reader.readInteger(HeaderSize));
  // The data and header size ints are themselves part of the header, so we must
  // subtract them from the size.
  RETURN_IF_ERROR(
      Reader.readStreamRef(HeaderBytes, HeaderSize - 2 * sizeof(uint32_t)));
  RETURN_IF_ERROR(Reader.readStreamRef(DataBytes, DataSize));
  RETURN_IF_ERROR(Reader.padToAlignment(sizeof(uint32_t)));
  return Error::success();
}

} // namespace object
} // namespace llvm
