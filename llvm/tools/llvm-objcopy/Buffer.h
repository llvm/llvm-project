//===- Buffer.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_OBJCOPY_BUFFER_H
#define LLVM_TOOLS_OBJCOPY_BUFFER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/MemoryBuffer.h"
#include <memory>

namespace llvm {
namespace objcopy {

// The class Buffer abstracts out the common interface of FileOutputBuffer and
// WritableMemoryBuffer so that the hierarchy of Writers depends on this
// abstract interface and doesn't depend on a particular implementation.
// TODO: refactor the buffer classes in LLVM to enable us to use them here
// directly.
class Buffer {
  StringRef Name;

public:
  virtual ~Buffer();
  virtual Error allocate(size_t Size) = 0;
  virtual uint8_t *getBufferStart() = 0;
  virtual Error commit() = 0;

  explicit Buffer(StringRef Name) : Name(Name) {}
  StringRef getName() const { return Name; }
};

class FileBuffer : public Buffer {
  std::unique_ptr<FileOutputBuffer> Buf;
  // Indicates that allocate(0) was called, and commit() should create or
  // truncate a file instead of using a FileOutputBuffer.
  bool EmptyFile = false;

public:
  Error allocate(size_t Size) override;
  uint8_t *getBufferStart() override;
  Error commit() override;

  explicit FileBuffer(StringRef FileName) : Buffer(FileName) {}
};

class MemBuffer : public Buffer {
  std::unique_ptr<WritableMemoryBuffer> Buf;

public:
  Error allocate(size_t Size) override;
  uint8_t *getBufferStart() override;
  Error commit() override;

  explicit MemBuffer(StringRef Name) : Buffer(Name) {}

  std::unique_ptr<WritableMemoryBuffer> releaseMemoryBuffer();
};

} // end namespace objcopy
} // end namespace llvm

#endif // LLVM_TOOLS_OBJCOPY_BUFFER_H
