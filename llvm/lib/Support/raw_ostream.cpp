//===--- raw_ostream.cpp - Implement the raw_ostream classes --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements support for bulk buffered stream output.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"
#include "llvm/System/Program.h"
#include "llvm/System/Process.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Config/config.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include <ostream>
#include <sys/stat.h>
#include <sys/types.h>

#if defined(HAVE_UNISTD_H)
# include <unistd.h>
#endif
#if defined(HAVE_FCNTL_H)
# include <fcntl.h>
#endif

#if defined(_MSC_VER)
#include <io.h>
#include <fcntl.h>
#ifndef STDIN_FILENO
# define STDIN_FILENO 0
#endif
#ifndef STDOUT_FILENO
# define STDOUT_FILENO 1
#endif
#ifndef STDERR_FILENO
# define STDERR_FILENO 2
#endif
#endif

using namespace llvm;

raw_ostream::~raw_ostream() {
  // raw_ostream's subclasses should take care to flush the buffer
  // in their destructors.
  assert(OutBufCur == OutBufStart &&
         "raw_ostream destructor called with non-empty buffer!");

  if (BufferMode == InternalBuffer)
    delete [] OutBufStart;

  // If there are any pending errors, report them now. Clients wishing
  // to avoid llvm_report_error calls should check for errors with
  // has_error() and clear the error flag with clear_error() before
  // destructing raw_ostream objects which may have errors.
  if (Error)
    llvm_report_error("IO failure on output stream.");
}

// An out of line virtual method to provide a home for the class vtable.
void raw_ostream::handle() {}

size_t raw_ostream::preferred_buffer_size() {
  // BUFSIZ is intended to be a reasonable default.
  return BUFSIZ;
}

void raw_ostream::SetBuffered() {
  // Ask the subclass to determine an appropriate buffer size.
  if (size_t Size = preferred_buffer_size())
    SetBufferSize(Size);
  else
    // It may return 0, meaning this stream should be unbuffered.
    SetUnbuffered();
}

void raw_ostream::SetBufferAndMode(char *BufferStart, size_t Size, 
                                    BufferKind Mode) {
  assert(((Mode == Unbuffered && BufferStart == 0 && Size == 0) || 
          (Mode != Unbuffered && BufferStart && Size >= 64)) &&
         "stream must be unbuffered, or have >= 64 bytes of buffer");
  // Make sure the current buffer is free of content (we can't flush here; the
  // child buffer management logic will be in write_impl).
  assert(GetNumBytesInBuffer() == 0 && "Current buffer is non-empty!");

  if (BufferMode == InternalBuffer)
    delete [] OutBufStart;
  OutBufStart = BufferStart;
  OutBufEnd = OutBufStart+Size;
  OutBufCur = OutBufStart;
  BufferMode = Mode;

  assert(OutBufStart <= OutBufEnd && "Invalid size!");
}

raw_ostream &raw_ostream::operator<<(unsigned long N) {
  // Zero is a special case.
  if (N == 0)
    return *this << '0';
  
  char NumberBuffer[20];
  char *EndPtr = NumberBuffer+sizeof(NumberBuffer);
  char *CurPtr = EndPtr;
  
  while (N) {
    *--CurPtr = '0' + char(N % 10);
    N /= 10;
  }
  return write(CurPtr, EndPtr-CurPtr);
}

raw_ostream &raw_ostream::operator<<(long N) {
  if (N <  0) {
    *this << '-';
    N = -N;
  }
  
  return this->operator<<(static_cast<unsigned long>(N));
}

raw_ostream &raw_ostream::operator<<(unsigned long long N) {
  // Output using 32-bit div/mod when possible.
  if (N == static_cast<unsigned long>(N))
    return this->operator<<(static_cast<unsigned long>(N));

  char NumberBuffer[20];
  char *EndPtr = NumberBuffer+sizeof(NumberBuffer);
  char *CurPtr = EndPtr;
  
  while (N) {
    *--CurPtr = '0' + char(N % 10);
    N /= 10;
  }
  return write(CurPtr, EndPtr-CurPtr);
}

raw_ostream &raw_ostream::operator<<(long long N) {
  if (N <  0) {
    *this << '-';
    N = -N;
  }
  
  return this->operator<<(static_cast<unsigned long long>(N));
}

raw_ostream &raw_ostream::write_hex(unsigned long long N) {
  // Zero is a special case.
  if (N == 0)
    return *this << '0';

  char NumberBuffer[20];
  char *EndPtr = NumberBuffer+sizeof(NumberBuffer);
  char *CurPtr = EndPtr;

  while (N) {
    uintptr_t x = N % 16;
    *--CurPtr = (x < 10 ? '0' + x : 'a' + x - 10);
    N /= 16;
  }

  return write(CurPtr, EndPtr-CurPtr);
}

raw_ostream &raw_ostream::operator<<(const void *P) {
  *this << '0' << 'x';

  return write_hex((uintptr_t) P);
}

void raw_ostream::flush_nonempty() {
  assert(OutBufCur > OutBufStart && "Invalid call to flush_nonempty.");
  size_t Length = OutBufCur - OutBufStart;
  OutBufCur = OutBufStart;
  write_impl(OutBufStart, Length);
}

raw_ostream &raw_ostream::write(unsigned char C) {
  // Group exceptional cases into a single branch.
  if (BUILTIN_EXPECT(OutBufCur >= OutBufEnd, false)) {
    if (BUILTIN_EXPECT(!OutBufStart, false)) {
      if (BufferMode == Unbuffered) {
        write_impl(reinterpret_cast<char*>(&C), 1);
        return *this;
      }
      // Set up a buffer and start over.
      SetBuffered();
      return write(C);
    }

    flush_nonempty();
  }

  *OutBufCur++ = C;
  return *this;
}

raw_ostream &raw_ostream::write(const char *Ptr, size_t Size) {
  // Group exceptional cases into a single branch.
  if (BUILTIN_EXPECT(OutBufCur+Size > OutBufEnd, false)) {
    if (BUILTIN_EXPECT(!OutBufStart, false)) {
      if (BufferMode == Unbuffered) {
        write_impl(Ptr, Size);
        return *this;
      }
      // Set up a buffer and start over.
      SetBuffered();
      return write(Ptr, Size);
    }

    // Write out the data in buffer-sized blocks until the remainder
    // fits within the buffer.
    do {
      size_t NumBytes = OutBufEnd - OutBufCur;
      copy_to_buffer(Ptr, NumBytes);
      flush_nonempty();
      Ptr += NumBytes;
      Size -= NumBytes;
    } while (OutBufCur+Size > OutBufEnd);
  }

  copy_to_buffer(Ptr, Size);

  return *this;
}

void raw_ostream::copy_to_buffer(const char *Ptr, size_t Size) {
  assert(Size <= size_t(OutBufEnd - OutBufCur) && "Buffer overrun!");

  // Handle short strings specially, memcpy isn't very good at very short
  // strings.
  switch (Size) {
  case 4: OutBufCur[3] = Ptr[3]; // FALL THROUGH
  case 3: OutBufCur[2] = Ptr[2]; // FALL THROUGH
  case 2: OutBufCur[1] = Ptr[1]; // FALL THROUGH
  case 1: OutBufCur[0] = Ptr[0]; // FALL THROUGH
  case 0: break;
  default:
    memcpy(OutBufCur, Ptr, Size);
    break;
  }

  OutBufCur += Size;
}

// Formatted output.
raw_ostream &raw_ostream::operator<<(const format_object_base &Fmt) {
  // If we have more than a few bytes left in our output buffer, try
  // formatting directly onto its end.
  size_t NextBufferSize = 127;
  size_t BufferBytesLeft = OutBufEnd - OutBufCur;
  if (BufferBytesLeft > 3) {
    size_t BytesUsed = Fmt.print(OutBufCur, BufferBytesLeft);
    
    // Common case is that we have plenty of space.
    if (BytesUsed <= BufferBytesLeft) {
      OutBufCur += BytesUsed;
      return *this;
    }
    
    // Otherwise, we overflowed and the return value tells us the size to try
    // again with.
    NextBufferSize = BytesUsed;
  }
  
  // If we got here, we didn't have enough space in the output buffer for the
  // string.  Try printing into a SmallVector that is resized to have enough
  // space.  Iterate until we win.
  SmallVector<char, 128> V;
  
  while (1) {
    V.resize(NextBufferSize);
    
    // Try formatting into the SmallVector.
    size_t BytesUsed = Fmt.print(V.data(), NextBufferSize);
    
    // If BytesUsed fit into the vector, we win.
    if (BytesUsed <= NextBufferSize)
      return write(V.data(), BytesUsed);
    
    // Otherwise, try again with a new size.
    assert(BytesUsed > NextBufferSize && "Didn't grow buffer!?");
    NextBufferSize = BytesUsed;
  }
}

//===----------------------------------------------------------------------===//
//  Formatted Output
//===----------------------------------------------------------------------===//

// Out of line virtual method.
void format_object_base::home() {
}

//===----------------------------------------------------------------------===//
//  raw_fd_ostream
//===----------------------------------------------------------------------===//

/// raw_fd_ostream - Open the specified file for writing. If an error
/// occurs, information about the error is put into ErrorInfo, and the
/// stream should be immediately destroyed; the string will be empty
/// if no error occurred.
raw_fd_ostream::raw_fd_ostream(const char *Filename, bool Binary, bool Force,
                               std::string &ErrorInfo) : pos(0) {
  ErrorInfo.clear();

  // Handle "-" as stdout.
  if (Filename[0] == '-' && Filename[1] == 0) {
    FD = STDOUT_FILENO;
    // If user requested binary then put stdout into binary mode if
    // possible.
    if (Binary)
      sys::Program::ChangeStdoutToBinary();
    ShouldClose = false;
    return;
  }
  
  int Flags = O_WRONLY|O_CREAT|O_TRUNC;
#ifdef O_BINARY
  if (Binary)
    Flags |= O_BINARY;
#endif
  if (!Force)
    Flags |= O_EXCL;
  FD = open(Filename, Flags, 0664);
  if (FD < 0) {
    ErrorInfo = "Error opening output file '" + std::string(Filename) + "'";
    ShouldClose = false;
  } else {
    ShouldClose = true;
  }
}

raw_fd_ostream::~raw_fd_ostream() {
  if (FD >= 0) {
    flush();
    if (ShouldClose)
      if (::close(FD) != 0)
        error_detected();
  }
}

void raw_fd_ostream::write_impl(const char *Ptr, size_t Size) {
  assert (FD >= 0 && "File already closed.");
  pos += Size;
  if (::write(FD, Ptr, Size) != (ssize_t) Size)
    error_detected();
}

void raw_fd_ostream::close() {
  assert (ShouldClose);
  ShouldClose = false;
  flush();
  if (::close(FD) != 0)
    error_detected();
  FD = -1;
}

uint64_t raw_fd_ostream::seek(uint64_t off) {
  flush();
  pos = ::lseek(FD, off, SEEK_SET);
  if (pos != off)
    error_detected();
  return pos;  
}

size_t raw_fd_ostream::preferred_buffer_size() {
#if !defined(_MSC_VER) && !defined(__MINGW32__) // Windows has no st_blksize.
  assert(FD >= 0 && "File not yet open!");
  struct stat statbuf;
  if (fstat(FD, &statbuf) == 0) {
    // If this is a terminal, don't use buffering. Line buffering
    // would be a more traditional thing to do, but it's not worth
    // the complexity.
    if (S_ISCHR(statbuf.st_mode) && isatty(FD))
      return 0;
    // Return the preferred block size.
    return statbuf.st_blksize;
  }
  error_detected();
#endif
  return raw_ostream::preferred_buffer_size();
}

raw_ostream &raw_fd_ostream::changeColor(enum Colors colors, bool bold,
                                         bool bg) {
  if (sys::Process::ColorNeedsFlush())
    flush();
  const char *colorcode =
    (colors == SAVEDCOLOR) ? sys::Process::OutputBold(bg)
    : sys::Process::OutputColor(colors, bold, bg);
  if (colorcode) {
    size_t len = strlen(colorcode);
    write(colorcode, len);
    // don't account colors towards output characters
    pos -= len;
  }
  return *this;
}

raw_ostream &raw_fd_ostream::resetColor() {
  if (sys::Process::ColorNeedsFlush())
    flush();
  const char *colorcode = sys::Process::ResetColor();
  if (colorcode) {
    size_t len = strlen(colorcode);
    write(colorcode, len);
    // don't account colors towards output characters
    pos -= len;
  }
  return *this;
}

//===----------------------------------------------------------------------===//
//  raw_stdout/err_ostream
//===----------------------------------------------------------------------===//

// Set buffer settings to model stdout and stderr behavior.
// Set standard error to be unbuffered by default.
raw_stdout_ostream::raw_stdout_ostream():raw_fd_ostream(STDOUT_FILENO, false) {}
raw_stderr_ostream::raw_stderr_ostream():raw_fd_ostream(STDERR_FILENO, false,
                                                        true) {}

// An out of line virtual method to provide a home for the class vtable.
void raw_stdout_ostream::handle() {}
void raw_stderr_ostream::handle() {}

/// outs() - This returns a reference to a raw_ostream for standard output.
/// Use it like: outs() << "foo" << "bar";
raw_ostream &llvm::outs() {
  static raw_stdout_ostream S;
  return S;
}

/// errs() - This returns a reference to a raw_ostream for standard error.
/// Use it like: errs() << "foo" << "bar";
raw_ostream &llvm::errs() {
  static raw_stderr_ostream S;
  return S;
}

/// nulls() - This returns a reference to a raw_ostream which discards output.
raw_ostream &llvm::nulls() {
  static raw_null_ostream S;
  return S;
}

//===----------------------------------------------------------------------===//
//  raw_os_ostream
//===----------------------------------------------------------------------===//

raw_os_ostream::~raw_os_ostream() {
  flush();
}

void raw_os_ostream::write_impl(const char *Ptr, size_t Size) {
  OS.write(Ptr, Size);
}

uint64_t raw_os_ostream::current_pos() { return OS.tellp(); }

//===----------------------------------------------------------------------===//
//  raw_string_ostream
//===----------------------------------------------------------------------===//

raw_string_ostream::~raw_string_ostream() {
  flush();
}

void raw_string_ostream::write_impl(const char *Ptr, size_t Size) {
  OS.append(Ptr, Size);
}

//===----------------------------------------------------------------------===//
//  raw_svector_ostream
//===----------------------------------------------------------------------===//

// The raw_svector_ostream implementation uses the SmallVector itself as the
// buffer for the raw_ostream. We guarantee that the raw_ostream buffer is
// always pointing past the end of the vector, but within the vector
// capacity. This allows raw_ostream to write directly into the correct place,
// and we only need to set the vector size when the data is flushed.

raw_svector_ostream::raw_svector_ostream(SmallVectorImpl<char> &O) : OS(O) {
  // Set up the initial external buffer. We make sure that the buffer has at
  // least 128 bytes free; raw_ostream itself only requires 64, but we want to
  // make sure that we don't grow the buffer unnecessarily on destruction (when
  // the data is flushed). See the FIXME below.
  OS.reserve(OS.size() + 128);
  SetBuffer(OS.end(), OS.capacity() - OS.size());
}

raw_svector_ostream::~raw_svector_ostream() {
  // FIXME: Prevent resizing during this flush().
  flush();
}

void raw_svector_ostream::write_impl(const char *Ptr, size_t Size) {
  assert(Ptr == OS.end() && OS.size() + Size <= OS.capacity() &&
         "Invalid write_impl() call!");

  // We don't need to copy the bytes, just commit the bytes to the
  // SmallVector.
  OS.set_size(OS.size() + Size);

  // Grow the vector if necessary.
  if (OS.capacity() - OS.size() < 64)
    OS.reserve(OS.capacity() * 2);

  // Update the buffer position.
  SetBuffer(OS.end(), OS.capacity() - OS.size());
}

uint64_t raw_svector_ostream::current_pos() { return OS.size(); }

StringRef raw_svector_ostream::str() {
  flush();
  return StringRef(OS.begin(), OS.size());
}

//===----------------------------------------------------------------------===//
//  raw_null_ostream
//===----------------------------------------------------------------------===//

raw_null_ostream::~raw_null_ostream() {
#ifndef NDEBUG
  // ~raw_ostream asserts that the buffer is empty. This isn't necessary
  // with raw_null_ostream, but it's better to have raw_null_ostream follow
  // the rules than to change the rules just for raw_null_ostream.
  flush();
#endif
}

void raw_null_ostream::write_impl(const char *Ptr, size_t Size) {
}

uint64_t raw_null_ostream::current_pos() {
  return 0;
}
