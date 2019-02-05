//===- Buffer.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Buffer.h"
#include "llvm-objcopy.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
#include <memory>

namespace llvm {
namespace objcopy {

Buffer::~Buffer() {}

static Error createEmptyFile(StringRef FileName) {
  // Create an empty tempfile and atomically swap it in place with the desired
  // output file.
  Expected<sys::fs::TempFile> Temp =
      sys::fs::TempFile::create(FileName + ".temp-empty-%%%%%%%");
  return Temp ? Temp->keep(FileName) : Temp.takeError();
}

Error FileBuffer::allocate(size_t Size) {
  // When a 0-sized file is requested, skip allocation but defer file
  // creation/truncation until commit() to avoid side effects if something
  // happens between allocate() and commit().
  if (Size == 0) {
    EmptyFile = true;
    return Error::success();
  }

  Expected<std::unique_ptr<FileOutputBuffer>> BufferOrErr =
      FileOutputBuffer::create(getName(), Size, FileOutputBuffer::F_executable);
  // FileOutputBuffer::create() returns an Error that is just a wrapper around
  // std::error_code. Wrap it in FileError to include the actual filename.
  if (!BufferOrErr)
    return createFileError(getName(), BufferOrErr.takeError());
  Buf = std::move(*BufferOrErr);
  return Error::success();
}

Error FileBuffer::commit() {
  if (EmptyFile)
    return createEmptyFile(getName());

  assert(Buf && "allocate() not called before commit()!");
  Error Err = Buf->commit();
  // FileOutputBuffer::commit() returns an Error that is just a wrapper around
  // std::error_code. Wrap it in FileError to include the actual filename.
  return Err ? createFileError(getName(), std::move(Err)) : std::move(Err);
}

uint8_t *FileBuffer::getBufferStart() {
  return reinterpret_cast<uint8_t *>(Buf->getBufferStart());
}

Error MemBuffer::allocate(size_t Size) {
  Buf = WritableMemoryBuffer::getNewMemBuffer(Size, getName());
  return Error::success();
}

Error MemBuffer::commit() { return Error::success(); }

uint8_t *MemBuffer::getBufferStart() {
  return reinterpret_cast<uint8_t *>(Buf->getBufferStart());
}

std::unique_ptr<WritableMemoryBuffer> MemBuffer::releaseMemoryBuffer() {
  return std::move(Buf);
}

} // end namespace objcopy
} // end namespace llvm
