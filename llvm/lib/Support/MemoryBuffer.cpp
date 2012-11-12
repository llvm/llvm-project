//===--- MemoryBuffer.cpp - Memory Buffer implementation ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the MemoryBuffer interface.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Config/config.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Errno.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/system_error.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <cerrno>
#include <new>
#include <sys/types.h>
#include <sys/stat.h>
#if !defined(_MSC_VER) && !defined(__MINGW32__)
#include <unistd.h>
#else
#include <io.h>
#ifndef S_ISFIFO
#define S_ISFIFO(x) (0)
#endif
#endif
#include <fcntl.h>
using namespace llvm;

//===----------------------------------------------------------------------===//
// MemoryBuffer implementation itself.
//===----------------------------------------------------------------------===//

MemoryBuffer::~MemoryBuffer() { }

/// init - Initialize this MemoryBuffer as a reference to externally allocated
/// memory, memory that we know is already null terminated.
void MemoryBuffer::init(const char *BufStart, const char *BufEnd,
                        bool RequiresNullTerminator) {
  assert((!RequiresNullTerminator || BufEnd[0] == 0) &&
         "Buffer is not null terminated!");
  BufferStart = BufStart;
  BufferEnd = BufEnd;
}

//===----------------------------------------------------------------------===//
// MemoryBufferMem implementation.
//===----------------------------------------------------------------------===//

/// CopyStringRef - Copies contents of a StringRef into a block of memory and
/// null-terminates it.
static void CopyStringRef(char *Memory, StringRef Data) {
  memcpy(Memory, Data.data(), Data.size());
  Memory[Data.size()] = 0; // Null terminate string.
}

/// GetNamedBuffer - Allocates a new MemoryBuffer with Name copied after it.
template <typename T>
static T *GetNamedBuffer(StringRef Buffer, StringRef Name,
                         bool RequiresNullTerminator) {
  char *Mem = static_cast<char*>(operator new(sizeof(T) + Name.size() + 1));
  CopyStringRef(Mem + sizeof(T), Name);
  return new (Mem) T(Buffer, RequiresNullTerminator);
}

namespace {
/// MemoryBufferMem - Named MemoryBuffer pointing to a block of memory.
class MemoryBufferMem : public MemoryBuffer {
public:
  MemoryBufferMem(StringRef InputData, bool RequiresNullTerminator) {
    init(InputData.begin(), InputData.end(), RequiresNullTerminator);
  }

  virtual const char *getBufferIdentifier() const LLVM_OVERRIDE {
     // The name is stored after the class itself.
    return reinterpret_cast<const char*>(this + 1);
  }

  virtual BufferKind getBufferKind() const LLVM_OVERRIDE {
    return MemoryBuffer_Malloc;
  }
};
}

/// getMemBuffer - Open the specified memory range as a MemoryBuffer.  Note
/// that InputData must be a null terminated if RequiresNullTerminator is true!
MemoryBuffer *MemoryBuffer::getMemBuffer(StringRef InputData,
                                         StringRef BufferName,
                                         bool RequiresNullTerminator) {
  return GetNamedBuffer<MemoryBufferMem>(InputData, BufferName,
                                         RequiresNullTerminator);
}

/// getMemBufferCopy - Open the specified memory range as a MemoryBuffer,
/// copying the contents and taking ownership of it.  This has no requirements
/// on EndPtr[0].
MemoryBuffer *MemoryBuffer::getMemBufferCopy(StringRef InputData,
                                             StringRef BufferName) {
  MemoryBuffer *Buf = getNewUninitMemBuffer(InputData.size(), BufferName);
  if (!Buf) return 0;
  memcpy(const_cast<char*>(Buf->getBufferStart()), InputData.data(),
         InputData.size());
  return Buf;
}

/// getNewUninitMemBuffer - Allocate a new MemoryBuffer of the specified size
/// that is not initialized.  Note that the caller should initialize the
/// memory allocated by this method.  The memory is owned by the MemoryBuffer
/// object.
MemoryBuffer *MemoryBuffer::getNewUninitMemBuffer(size_t Size,
                                                  StringRef BufferName) {
  // Allocate space for the MemoryBuffer, the data and the name. It is important
  // that MemoryBuffer and data are aligned so PointerIntPair works with them.
  size_t AlignedStringLen =
    RoundUpToAlignment(sizeof(MemoryBufferMem) + BufferName.size() + 1,
                       sizeof(void*)); // TODO: Is sizeof(void*) enough?
  size_t RealLen = AlignedStringLen + Size + 1;
  char *Mem = static_cast<char*>(operator new(RealLen, std::nothrow));
  if (!Mem) return 0;

  // The name is stored after the class itself.
  CopyStringRef(Mem + sizeof(MemoryBufferMem), BufferName);

  // The buffer begins after the name and must be aligned.
  char *Buf = Mem + AlignedStringLen;
  Buf[Size] = 0; // Null terminate buffer.

  return new (Mem) MemoryBufferMem(StringRef(Buf, Size), true);
}

/// getNewMemBuffer - Allocate a new MemoryBuffer of the specified size that
/// is completely initialized to zeros.  Note that the caller should
/// initialize the memory allocated by this method.  The memory is owned by
/// the MemoryBuffer object.
MemoryBuffer *MemoryBuffer::getNewMemBuffer(size_t Size, StringRef BufferName) {
  MemoryBuffer *SB = getNewUninitMemBuffer(Size, BufferName);
  if (!SB) return 0;
  memset(const_cast<char*>(SB->getBufferStart()), 0, Size);
  return SB;
}


/// getFileOrSTDIN - Open the specified file as a MemoryBuffer, or open stdin
/// if the Filename is "-".  If an error occurs, this returns null and fills
/// in *ErrStr with a reason.  If stdin is empty, this API (unlike getSTDIN)
/// returns an empty buffer.
error_code MemoryBuffer::getFileOrSTDIN(StringRef Filename,
                                        OwningPtr<MemoryBuffer> &result,
                                        int64_t FileSize) {
  if (Filename == "-")
    return getSTDIN(result);
  return getFile(Filename, result, FileSize);
}

error_code MemoryBuffer::getFileOrSTDIN(const char *Filename,
                                        OwningPtr<MemoryBuffer> &result,
                                        int64_t FileSize) {
  if (strcmp(Filename, "-") == 0)
    return getSTDIN(result);
  return getFile(Filename, result, FileSize);
}

//===----------------------------------------------------------------------===//
// MemoryBuffer::getFile implementation.
//===----------------------------------------------------------------------===//

namespace {
/// MemoryBufferMMapFile - This represents a file that was mapped in with the
/// sys::Path::MapInFilePages method.  When destroyed, it calls the
/// sys::Path::UnMapFilePages method.
class MemoryBufferMMapFile : public MemoryBufferMem {
public:
  MemoryBufferMMapFile(StringRef Buffer, bool RequiresNullTerminator)
    : MemoryBufferMem(Buffer, RequiresNullTerminator) { }

  ~MemoryBufferMMapFile() {
    static int PageSize = sys::Process::GetPageSize();

    uintptr_t Start = reinterpret_cast<uintptr_t>(getBufferStart());
    size_t Size = getBufferSize();
    uintptr_t RealStart = Start & ~(PageSize - 1);
    size_t RealSize = Size + (Start - RealStart);

    sys::Path::UnMapFilePages(reinterpret_cast<const char*>(RealStart),
                              RealSize);
  }

  virtual BufferKind getBufferKind() const LLVM_OVERRIDE {
    return MemoryBuffer_MMap;
  }
};
}

static error_code getMemoryBufferForStream(int FD, 
                                           StringRef BufferName,
                                           OwningPtr<MemoryBuffer> &result) {
  const ssize_t ChunkSize = 4096*4;
  SmallString<ChunkSize> Buffer;
  ssize_t ReadBytes;
  // Read into Buffer until we hit EOF.
  do {
    Buffer.reserve(Buffer.size() + ChunkSize);
    ReadBytes = read(FD, Buffer.end(), ChunkSize);
    if (ReadBytes == -1) {
      if (errno == EINTR) continue;
      return error_code(errno, posix_category());
    }
    Buffer.set_size(Buffer.size() + ReadBytes);
  } while (ReadBytes != 0);

  result.reset(MemoryBuffer::getMemBufferCopy(Buffer, BufferName));
  return error_code::success();
}

error_code MemoryBuffer::getFile(StringRef Filename,
                                 OwningPtr<MemoryBuffer> &result,
                                 int64_t FileSize,
                                 bool RequiresNullTerminator) {
  // Ensure the path is null terminated.
  SmallString<256> PathBuf(Filename.begin(), Filename.end());
  return MemoryBuffer::getFile(PathBuf.c_str(), result, FileSize,
                               RequiresNullTerminator);
}

error_code MemoryBuffer::getFile(const char *Filename,
                                 OwningPtr<MemoryBuffer> &result,
                                 int64_t FileSize,
                                 bool RequiresNullTerminator) {
  // First check that the "file" is not a directory
  bool is_dir = false;
  error_code err = sys::fs::is_directory(Filename, is_dir);
  if (err)
    return err;
  if (is_dir)
    return make_error_code(errc::is_a_directory);

  int OpenFlags = O_RDONLY;
#ifdef O_BINARY
  OpenFlags |= O_BINARY;  // Open input file in binary mode on win32.
#endif
  int FD = ::open(Filename, OpenFlags);
  if (FD == -1)
    return error_code(errno, posix_category());

  error_code ret = getOpenFile(FD, Filename, result, FileSize, FileSize,
                               0, RequiresNullTerminator);
  close(FD);
  return ret;
}

static bool shouldUseMmap(int FD,
                          size_t FileSize,
                          size_t MapSize,
                          off_t Offset,
                          bool RequiresNullTerminator,
                          int PageSize) {
  // We don't use mmap for small files because this can severely fragment our
  // address space.
  if (MapSize < 4096*4)
    return false;

  if (!RequiresNullTerminator)
    return true;


  // If we don't know the file size, use fstat to find out.  fstat on an open
  // file descriptor is cheaper than stat on a random path.
  // FIXME: this chunk of code is duplicated, but it avoids a fstat when
  // RequiresNullTerminator = false and MapSize != -1.
  if (FileSize == size_t(-1)) {
    struct stat FileInfo;
    // TODO: This should use fstat64 when available.
    if (fstat(FD, &FileInfo) == -1) {
      return error_code(errno, posix_category());
    }
    FileSize = FileInfo.st_size;
  }

  // If we need a null terminator and the end of the map is inside the file,
  // we cannot use mmap.
  size_t End = Offset + MapSize;
  assert(End <= FileSize);
  if (End != FileSize)
    return false;

  // Don't try to map files that are exactly a multiple of the system page size
  // if we need a null terminator.
  if ((FileSize & (PageSize -1)) == 0)
    return false;

  return true;
}

error_code MemoryBuffer::getOpenFile(int FD, const char *Filename,
                                     OwningPtr<MemoryBuffer> &result,
                                     uint64_t FileSize, uint64_t MapSize,
                                     int64_t Offset,
                                     bool RequiresNullTerminator) {
  static int PageSize = sys::Process::GetPageSize();

  // Default is to map the full file.
  if (MapSize == uint64_t(-1)) {
    // If we don't know the file size, use fstat to find out.  fstat on an open
    // file descriptor is cheaper than stat on a random path.
    if (FileSize == uint64_t(-1)) {
      struct stat FileInfo;
      // TODO: This should use fstat64 when available.
      if (fstat(FD, &FileInfo) == -1) {
        return error_code(errno, posix_category());
      }

      // If this is a named pipe, we can't trust the size. Create the memory
      // buffer by copying off the stream.
      if (S_ISFIFO(FileInfo.st_mode)) {
        return getMemoryBufferForStream(FD, Filename, result);
      }

      FileSize = FileInfo.st_size;
    }
    MapSize = FileSize;
  }

  if (shouldUseMmap(FD, FileSize, MapSize, Offset, RequiresNullTerminator,
                    PageSize)) {
    off_t RealMapOffset = Offset & ~(PageSize - 1);
    off_t Delta = Offset - RealMapOffset;
    size_t RealMapSize = MapSize + Delta;

    if (const char *Pages = sys::Path::MapInFilePages(FD,
                                                      RealMapSize,
                                                      RealMapOffset)) {
      result.reset(GetNamedBuffer<MemoryBufferMMapFile>(
          StringRef(Pages + Delta, MapSize), Filename, RequiresNullTerminator));
      return error_code::success();
    }
  }

  MemoryBuffer *Buf = MemoryBuffer::getNewUninitMemBuffer(MapSize, Filename);
  if (!Buf) {
    // Failed to create a buffer. The only way it can fail is if
    // new(std::nothrow) returns 0.
    return make_error_code(errc::not_enough_memory);
  }

  OwningPtr<MemoryBuffer> SB(Buf);
  char *BufPtr = const_cast<char*>(SB->getBufferStart());

  size_t BytesLeft = MapSize;
#ifndef HAVE_PREAD
  if (lseek(FD, Offset, SEEK_SET) == -1)
    return error_code(errno, posix_category());
#endif

  while (BytesLeft) {
#ifdef HAVE_PREAD
    ssize_t NumRead = ::pread(FD, BufPtr, BytesLeft, MapSize-BytesLeft+Offset);
#else
    ssize_t NumRead = ::read(FD, BufPtr, BytesLeft);
#endif
    if (NumRead == -1) {
      if (errno == EINTR)
        continue;
      // Error while reading.
      return error_code(errno, posix_category());
    }
    if (NumRead == 0) {
      assert(0 && "We got inaccurate FileSize value or fstat reported an "
                   "invalid file size.");
      *BufPtr = '\0'; // null-terminate at the actual size.
      break;
    }
    BytesLeft -= NumRead;
    BufPtr += NumRead;
  }

  result.swap(SB);
  return error_code::success();
}

//===----------------------------------------------------------------------===//
// MemoryBuffer::getSTDIN implementation.
//===----------------------------------------------------------------------===//

error_code MemoryBuffer::getSTDIN(OwningPtr<MemoryBuffer> &result) {
  // Read in all of the data from stdin, we cannot mmap stdin.
  //
  // FIXME: That isn't necessarily true, we should try to mmap stdin and
  // fallback if it fails.
  sys::Program::ChangeStdinToBinary();

  return getMemoryBufferForStream(0, "<stdin>", result);
}
