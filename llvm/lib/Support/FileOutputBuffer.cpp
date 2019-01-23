//===- FileOutputBuffer.cpp - File Output Buffer ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utility for creating a in-memory buffer that will be written to a file.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/Path.h"
#include <system_error>

#if !defined(_MSC_VER) && !defined(__MINGW32__)
#include <unistd.h>
#else
#include <io.h>
#endif

using namespace llvm;
using namespace llvm::sys;

namespace {
// A FileOutputBuffer which creates a temporary file in the same directory
// as the final output file. The final output file is atomically replaced
// with the temporary file on commit().
class OnDiskBuffer : public FileOutputBuffer {
public:
  OnDiskBuffer(StringRef Path, fs::TempFile Temp,
               std::unique_ptr<fs::mapped_file_region> Buf)
      : FileOutputBuffer(Path), Buffer(std::move(Buf)), Temp(std::move(Temp)) {}

  uint8_t *getBufferStart() const override { return (uint8_t *)Buffer->data(); }

  uint8_t *getBufferEnd() const override {
    return (uint8_t *)Buffer->data() + Buffer->size();
  }

  size_t getBufferSize() const override { return Buffer->size(); }

  Error commit() override {
    // Unmap buffer, letting OS flush dirty pages to file on disk.
    Buffer.reset();

    // Atomically replace the existing file with the new one.
    return Temp.keep(FinalPath);
  }

  ~OnDiskBuffer() override {
    // Close the mapping before deleting the temp file, so that the removal
    // succeeds.
    Buffer.reset();
    consumeError(Temp.discard());
  }

  void discard() override {
    // Delete the temp file if it still was open, but keeping the mapping
    // active.
    consumeError(Temp.discard());
  }

private:
  std::unique_ptr<fs::mapped_file_region> Buffer;
  fs::TempFile Temp;
};

// A FileOutputBuffer which keeps data in memory and writes to the final
// output file on commit(). This is used only when we cannot use OnDiskBuffer.
class InMemoryBuffer : public FileOutputBuffer {
public:
  InMemoryBuffer(StringRef Path, MemoryBlock Buf, unsigned Mode)
      : FileOutputBuffer(Path), Buffer(Buf), Mode(Mode) {}

  uint8_t *getBufferStart() const override { return (uint8_t *)Buffer.base(); }

  uint8_t *getBufferEnd() const override {
    return (uint8_t *)Buffer.base() + Buffer.size();
  }

  size_t getBufferSize() const override { return Buffer.size(); }

  Error commit() override {
    if (FinalPath == "-") {
      llvm::outs() << StringRef((const char *)Buffer.base(), Buffer.size());
      llvm::outs().flush();
      return Error::success();
    }

    using namespace sys::fs;
    int FD;
    std::error_code EC;
    if (auto EC =
            openFileForWrite(FinalPath, FD, CD_CreateAlways, OF_None, Mode))
      return errorCodeToError(EC);
    raw_fd_ostream OS(FD, /*shouldClose=*/true, /*unbuffered=*/true);
    OS << StringRef((const char *)Buffer.base(), Buffer.size());
    return Error::success();
  }

private:
  OwningMemoryBlock Buffer;
  unsigned Mode;
};
} // namespace

static Expected<std::unique_ptr<InMemoryBuffer>>
createInMemoryBuffer(StringRef Path, size_t Size, unsigned Mode) {
  std::error_code EC;
  MemoryBlock MB = Memory::allocateMappedMemory(
      Size, nullptr, sys::Memory::MF_READ | sys::Memory::MF_WRITE, EC);
  if (EC)
    return errorCodeToError(EC);
  return llvm::make_unique<InMemoryBuffer>(Path, MB, Mode);
}

static Expected<std::unique_ptr<FileOutputBuffer>>
createOnDiskBuffer(StringRef Path, size_t Size, unsigned Mode) {
  Expected<fs::TempFile> FileOrErr =
      fs::TempFile::create(Path + ".tmp%%%%%%%", Mode);
  if (!FileOrErr)
    return FileOrErr.takeError();
  fs::TempFile File = std::move(*FileOrErr);

#ifndef _WIN32
  // On Windows, CreateFileMapping (the mmap function on Windows)
  // automatically extends the underlying file. We don't need to
  // extend the file beforehand. _chsize (ftruncate on Windows) is
  // pretty slow just like it writes specified amount of bytes,
  // so we should avoid calling that function.
  if (auto EC = fs::resize_file(File.FD, Size)) {
    consumeError(File.discard());
    return errorCodeToError(EC);
  }
#endif

  // Mmap it.
  std::error_code EC;
  auto MappedFile = llvm::make_unique<fs::mapped_file_region>(
      File.FD, fs::mapped_file_region::readwrite, Size, 0, EC);

  // mmap(2) can fail if the underlying filesystem does not support it.
  // If that happens, we fall back to in-memory buffer as the last resort.
  if (EC) {
    consumeError(File.discard());
    return createInMemoryBuffer(Path, Size, Mode);
  }

  return llvm::make_unique<OnDiskBuffer>(Path, std::move(File),
                                         std::move(MappedFile));
}

// Create an instance of FileOutputBuffer.
Expected<std::unique_ptr<FileOutputBuffer>>
FileOutputBuffer::create(StringRef Path, size_t Size, unsigned Flags) {
  // Handle "-" as stdout just like llvm::raw_ostream does.
  if (Path == "-")
    return createInMemoryBuffer("-", Size, /*Mode=*/0);

  unsigned Mode = fs::all_read | fs::all_write;
  if (Flags & F_executable)
    Mode |= fs::all_exe;

  fs::file_status Stat;
  fs::status(Path, Stat);

  // Usually, we want to create OnDiskBuffer to create a temporary file in
  // the same directory as the destination file and atomically replaces it
  // by rename(2).
  //
  // However, if the destination file is a special file, we don't want to
  // use rename (e.g. we don't want to replace /dev/null with a regular
  // file.) If that's the case, we create an in-memory buffer, open the
  // destination file and write to it on commit().
  switch (Stat.type()) {
  case fs::file_type::directory_file:
    return errorCodeToError(errc::is_a_directory);
  case fs::file_type::regular_file:
  case fs::file_type::file_not_found:
  case fs::file_type::status_error:
    return createOnDiskBuffer(Path, Size, Mode);
  default:
    return createInMemoryBuffer(Path, Size, Mode);
  }
}
