//===- FileOutputBuffer.cpp - File Output Buffer ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "llvm/Support/Path.h"
#include "llvm/Support/Signals.h"
#include <system_error>

#if !defined(_MSC_VER) && !defined(__MINGW32__)
#include <unistd.h>
#else
#include <io.h>
#endif

using llvm::sys::fs::mapped_file_region;

namespace llvm {
FileOutputBuffer::FileOutputBuffer(std::unique_ptr<mapped_file_region> R,
                                   StringRef Path, StringRef TmpPath,
                                   bool IsRegular)
    : Region(std::move(R)), FinalPath(Path), TempPath(TmpPath),
      IsRegular(IsRegular) {}

FileOutputBuffer::~FileOutputBuffer() {
  // Close the mapping before deleting the temp file, so that the removal
  // succeeds.
  Region.reset();
  sys::fs::remove(Twine(TempPath));
}

ErrorOr<std::unique_ptr<FileOutputBuffer>>
FileOutputBuffer::create(StringRef FilePath, size_t Size, unsigned Flags) {
  // Check file is not a regular file, in which case we cannot remove it.
  sys::fs::file_status Stat;
  std::error_code EC = sys::fs::status(FilePath, Stat);
  bool IsRegular = true;
  switch (Stat.type()) {
    case sys::fs::file_type::file_not_found:
      // If file does not exist, we'll create one.
      break;
    case sys::fs::file_type::regular_file: {
        // If file is not currently writable, error out.
        // FIXME: There is no sys::fs:: api for checking this.
        // FIXME: In posix, you use the access() call to check this.
      }
      break;
    case sys::fs::file_type::directory_file:
      return errc::is_a_directory;
    default:
      if (EC)
        return EC;
      IsRegular = false;
  }

  if (IsRegular) {
    // Delete target file.
    EC = sys::fs::remove(FilePath);
    if (EC)
      return EC;
  }

  SmallString<128> TempFilePath;
  int FD;
  if (IsRegular) {
    unsigned Mode = sys::fs::all_read | sys::fs::all_write;
    // If requested, make the output file executable.
    if (Flags & F_executable)
      Mode |= sys::fs::all_exe;
    // Create new file in same directory but with random name.
    EC = sys::fs::createUniqueFile(Twine(FilePath) + ".tmp%%%%%%%", FD,
                                   TempFilePath, Mode);
  } else {
    // Create a temporary file. Since this is a special file, we will not move
    // it and the new file can be in another filesystem. This avoids trying to
    // create a temporary file in /dev when outputting to /dev/null for example.
    EC = sys::fs::createTemporaryFile(sys::path::filename(FilePath), "", FD,
                                      TempFilePath);
  }

  if (EC)
    return EC;

  sys::RemoveFileOnSignal(TempFilePath);

#ifndef LLVM_ON_WIN32
  // On Windows, CreateFileMapping (the mmap function on Windows)
  // automatically extends the underlying file. We don't need to
  // extend the file beforehand. _chsize (ftruncate on Windows) is
  // pretty slow just like it writes specified amount of bytes,
  // so we should avoid calling that.
  EC = sys::fs::resize_file(FD, Size);
  if (EC)
    return EC;
#endif

  auto MappedFile = llvm::make_unique<mapped_file_region>(
      FD, mapped_file_region::readwrite, Size, 0, EC);
  int Ret = close(FD);
  if (EC)
    return EC;
  if (Ret)
    return std::error_code(errno, std::generic_category());

  std::unique_ptr<FileOutputBuffer> Buf(new FileOutputBuffer(
      std::move(MappedFile), FilePath, TempFilePath, IsRegular));
  return std::move(Buf);
}

std::error_code FileOutputBuffer::commit() {
  // Unmap buffer, letting OS flush dirty pages to file on disk.
  Region.reset();

  std::error_code EC;
  if (IsRegular) {
    // Rename file to final name.
    EC = sys::fs::rename(Twine(TempPath), Twine(FinalPath));
    sys::DontRemoveFileOnSignal(TempPath);
  } else {
    EC = sys::fs::copy_file(TempPath, FinalPath);
    std::error_code RMEC = sys::fs::remove(TempPath);
    sys::DontRemoveFileOnSignal(TempPath);
    if (RMEC)
      return RMEC;
  }

  return EC;
}
} // namespace
