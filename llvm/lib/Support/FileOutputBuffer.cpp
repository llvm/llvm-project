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

#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"


namespace llvm {


FileOutputBuffer::FileOutputBuffer(uint8_t *Start, uint8_t *End, 
                                  StringRef Path, StringRef TmpPath)
  : BufferStart(Start), BufferEnd(End) {
  FinalPath.assign(Path);
  TempPath.assign(TmpPath);
}


FileOutputBuffer::~FileOutputBuffer() {
  // If not already commited, delete buffer and remove temp file.
  if ( BufferStart != NULL ) {
    sys::fs::unmap_file_pages((void*)BufferStart, getBufferSize());
    bool Existed;
    sys::fs::remove(Twine(TempPath), Existed);
  }
}

 
error_code FileOutputBuffer::create(StringRef FilePath, 
                                    size_t Size,  
                                    OwningPtr<FileOutputBuffer> &Result,
                                    unsigned Flags) {
  // If file already exists, it must be a regular file (to be mappable).
  sys::fs::file_status Stat;
  error_code EC = sys::fs::status(FilePath, Stat);
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
    default:
      if (EC)
        return EC;
      else
        return make_error_code(errc::operation_not_permitted);
  }

  // Delete target file.
  bool Existed;
  EC = sys::fs::remove(FilePath, Existed);
  if (EC)
    return EC;
  
  // Create new file in same directory but with random name.
  SmallString<128> TempFilePath;
  int FD;
  EC = sys::fs::unique_file(Twine(FilePath) + ".tmp%%%%%%%",  
                                                FD, TempFilePath, false, 0644);
  if (EC)
    return EC;
  
  // The unique_file() interface leaks lower layers and returns a file 
  // descriptor.  There is no way to directly close it, so use this hack
  // to hand it off to raw_fd_ostream to close for us.
  {
    raw_fd_ostream Dummy(FD, /*shouldClose=*/true);
  }
  
  // Resize file to requested initial size
  EC = sys::fs::resize_file(Twine(TempFilePath), Size);
  if (EC)
    return EC;
  
  // If requested, make the output file executable.
  if ( Flags & F_executable ) {
    sys::fs::file_status Stat2;
    EC = sys::fs::status(Twine(TempFilePath), Stat2);
    if (EC)
      return EC;
    
    sys::fs::perms new_perms = Stat2.permissions();
    if ( new_perms & sys::fs::owner_read )
      new_perms |= sys::fs::owner_exe;
    if ( new_perms & sys::fs::group_read )
      new_perms |= sys::fs::group_exe;
    if ( new_perms & sys::fs::others_read )
      new_perms |= sys::fs::others_exe;
    new_perms |= sys::fs::add_perms;
    EC = sys::fs::permissions(Twine(TempFilePath), new_perms);
    if (EC)
      return EC;
  }

  // Memory map new file.
  void *Base;
  EC = sys::fs::map_file_pages(Twine(TempFilePath), 0, Size, true, Base);
  if (EC)
    return EC;
  
  // Create FileOutputBuffer object to own mapped range.
  uint8_t *Start = reinterpret_cast<uint8_t*>(Base);
  Result.reset(new FileOutputBuffer(Start, Start+Size, FilePath, TempFilePath));
                     
  return error_code::success();
}                    


error_code FileOutputBuffer::commit(int64_t NewSmallerSize) {
  // Unmap buffer, letting OS flush dirty pages to file on disk.
  void *Start = reinterpret_cast<void*>(BufferStart);
  error_code EC = sys::fs::unmap_file_pages(Start, getBufferSize());
  if (EC)
    return EC;
  
  // If requested, resize file as part of commit.
  if ( NewSmallerSize != -1 ) {
    EC = sys::fs::resize_file(Twine(TempPath), NewSmallerSize);
    if (EC)
      return EC;
  }
  
  // Rename file to final name.
  return sys::fs::rename(Twine(TempPath), Twine(FinalPath));
}


} // namespace

