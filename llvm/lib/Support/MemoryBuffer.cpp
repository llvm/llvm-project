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
#include "llvm/System/Path.h"
#include "llvm/System/Process.h"
#include "llvm/System/Program.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <cerrno>
#include <sys/types.h>
#include <sys/stat.h>
#if !defined(_MSC_VER) && !defined(__MINGW32__)
#include <unistd.h>
#include <sys/uio.h>
#else
#include <io.h>
#endif
#include <fcntl.h>
using namespace llvm;

//===----------------------------------------------------------------------===//
// MemoryBuffer implementation itself.
//===----------------------------------------------------------------------===//

MemoryBuffer::~MemoryBuffer() {
  if (MustDeleteBuffer)
    delete [] BufferStart;
}

/// initCopyOf - Initialize this source buffer with a copy of the specified
/// memory range.  We make the copy so that we can null terminate it
/// successfully.
void MemoryBuffer::initCopyOf(const char *BufStart, const char *BufEnd) {
  size_t Size = BufEnd-BufStart;
  BufferStart = new char[Size+1];
  BufferEnd = BufferStart+Size;
  memcpy(const_cast<char*>(BufferStart), BufStart, Size);
  *const_cast<char*>(BufferEnd) = 0;   // Null terminate buffer.
  MustDeleteBuffer = true;
}

/// init - Initialize this MemoryBuffer as a reference to externally allocated
/// memory, memory that we know is already null terminated.
void MemoryBuffer::init(const char *BufStart, const char *BufEnd) {
  assert(BufEnd[0] == 0 && "Buffer is not null terminated!");
  BufferStart = BufStart;
  BufferEnd = BufEnd;
  MustDeleteBuffer = false;
}

//===----------------------------------------------------------------------===//
// MemoryBufferMem implementation.
//===----------------------------------------------------------------------===//

namespace {
class MemoryBufferMem : public MemoryBuffer {
  std::string FileID;
public:
  MemoryBufferMem(const char *Start, const char *End, const char *FID,
                  bool Copy = false)
  : FileID(FID) {
    if (!Copy)
      init(Start, End);
    else
      initCopyOf(Start, End);
  }
  
  virtual const char *getBufferIdentifier() const {
    return FileID.c_str();
  }
};
}

/// getMemBuffer - Open the specified memory range as a MemoryBuffer.  Note
/// that EndPtr[0] must be a null byte and be accessible!
MemoryBuffer *MemoryBuffer::getMemBuffer(const char *StartPtr, 
                                         const char *EndPtr,
                                         const char *BufferName) {
  return new MemoryBufferMem(StartPtr, EndPtr, BufferName);
}

/// getMemBufferCopy - Open the specified memory range as a MemoryBuffer,
/// copying the contents and taking ownership of it.  This has no requirements
/// on EndPtr[0].
MemoryBuffer *MemoryBuffer::getMemBufferCopy(const char *StartPtr, 
                                             const char *EndPtr,
                                             const char *BufferName) {
  return new MemoryBufferMem(StartPtr, EndPtr, BufferName, true);
}

/// getNewUninitMemBuffer - Allocate a new MemoryBuffer of the specified size
/// that is completely initialized to zeros.  Note that the caller should
/// initialize the memory allocated by this method.  The memory is owned by
/// the MemoryBuffer object.
MemoryBuffer *MemoryBuffer::getNewUninitMemBuffer(size_t Size,
                                                  const char *BufferName) {
  char *Buf = new char[Size+1];
  Buf[Size] = 0;
  MemoryBufferMem *SB = new MemoryBufferMem(Buf, Buf+Size, BufferName);
  // The memory for this buffer is owned by the MemoryBuffer.
  SB->MustDeleteBuffer = true;
  return SB;
}

/// getNewMemBuffer - Allocate a new MemoryBuffer of the specified size that
/// is completely initialized to zeros.  Note that the caller should
/// initialize the memory allocated by this method.  The memory is owned by
/// the MemoryBuffer object.
MemoryBuffer *MemoryBuffer::getNewMemBuffer(size_t Size,
                                            const char *BufferName) {
  MemoryBuffer *SB = getNewUninitMemBuffer(Size, BufferName);
  memset(const_cast<char*>(SB->getBufferStart()), 0, Size+1);
  return SB;
}


/// getFileOrSTDIN - Open the specified file as a MemoryBuffer, or open stdin
/// if the Filename is "-".  If an error occurs, this returns null and fills
/// in *ErrStr with a reason.  If stdin is empty, this API (unlike getSTDIN)
/// returns an empty buffer.
MemoryBuffer *MemoryBuffer::getFileOrSTDIN(const char *Filename,
                                           std::string *ErrStr,
                                           int64_t FileSize) {
  if (Filename[0] != '-' || Filename[1] != 0)
    return getFile(Filename, ErrStr, FileSize);
  MemoryBuffer *M = getSTDIN();
  if (M) return M;

  // If stdin was empty, M is null.  Cons up an empty memory buffer now.
  const char *EmptyStr = "";
  return MemoryBuffer::getMemBuffer(EmptyStr, EmptyStr, "<stdin>");
}

//===----------------------------------------------------------------------===//
// MemoryBuffer::getFile implementation.
//===----------------------------------------------------------------------===//

namespace {
/// MemoryBufferMMapFile - This represents a file that was mapped in with the
/// sys::Path::MapInFilePages method.  When destroyed, it calls the
/// sys::Path::UnMapFilePages method.
class MemoryBufferMMapFile : public MemoryBuffer {
  std::string Filename;
public:
  MemoryBufferMMapFile(const char *filename, const char *Pages, uint64_t Size)
    : Filename(filename) {
    init(Pages, Pages+Size);
  }
  
  virtual const char *getBufferIdentifier() const {
    return Filename.c_str();
  }
    
  ~MemoryBufferMMapFile() {
    sys::Path::UnMapFilePages(getBufferStart(), getBufferSize());
  }
};
}

MemoryBuffer *MemoryBuffer::getFile(const char *Filename, std::string *ErrStr,
                                    int64_t FileSize) {
  int OpenFlags = 0;
#ifdef O_BINARY
  OpenFlags |= O_BINARY;  // Open input file in binary mode on win32.
#endif
  int FD = ::open(Filename, O_RDONLY|OpenFlags);
  if (FD == -1) {
    if (ErrStr) *ErrStr = "could not open file";
    return 0;
  }
  
  // If we don't know the file size, use fstat to find out.  fstat on an open
  // file descriptor is cheaper than stat on a random path.
  if (FileSize == -1) {
    struct stat FileInfo;
    // TODO: This should use fstat64 when available.
    if (fstat(FD, &FileInfo) == -1) {
      if (ErrStr) *ErrStr = "could not get file length";
      ::close(FD);
      return 0;
    }
    FileSize = FileInfo.st_size;
  }
  
  
  // If the file is large, try to use mmap to read it in.  We don't use mmap
  // for small files, because this can severely fragment our address space. Also
  // don't try to map files that are exactly a multiple of the system page size,
  // as the file would not have the required null terminator.
  if (FileSize >= 4096*4 &&
      (FileSize & (sys::Process::GetPageSize()-1)) != 0) {
    if (const char *Pages = sys::Path::MapInFilePages(FD, FileSize)) {
      // Close the file descriptor, now that the whole file is in memory.
      ::close(FD);
      return new MemoryBufferMMapFile(Filename, Pages, FileSize);
    }
  }
  
  OwningPtr<MemoryBuffer> SB;
  SB.reset(MemoryBuffer::getNewUninitMemBuffer(FileSize, Filename));
  char *BufPtr = const_cast<char*>(SB->getBufferStart());
  
  size_t BytesLeft = FileSize;
  while (BytesLeft) {
    ssize_t NumRead = ::read(FD, BufPtr, BytesLeft);
    if (NumRead != -1) {
      BytesLeft -= NumRead;
      BufPtr += NumRead;
    } else if (errno == EINTR) {
      // try again
    } else {
      // error reading.
      close(FD);
      if (ErrStr) *ErrStr = "error reading file data";
      return 0;
    }
  }
  close(FD);
  
  return SB.take();
}

//===----------------------------------------------------------------------===//
// MemoryBuffer::getSTDIN implementation.
//===----------------------------------------------------------------------===//

namespace {
class STDINBufferFile : public MemoryBuffer {
public:
  virtual const char *getBufferIdentifier() const {
    return "<stdin>";
  }
};
}

MemoryBuffer *MemoryBuffer::getSTDIN() {
  char Buffer[4096*4];
  
  std::vector<char> FileData;
  
  // Read in all of the data from stdin, we cannot mmap stdin.
  sys::Program::ChangeStdinToBinary();
  while (size_t ReadBytes = fread(Buffer, sizeof(char), 4096*4, stdin))
    FileData.insert(FileData.end(), Buffer, Buffer+ReadBytes);

  FileData.push_back(0); // &FileData[Size] is invalid. So is &*FileData.end().
  size_t Size = FileData.size();
  if (Size <= 1)
    return 0;
  MemoryBuffer *B = new STDINBufferFile();
  B->initCopyOf(&FileData[0], &FileData[Size-1]);
  return B;
}
