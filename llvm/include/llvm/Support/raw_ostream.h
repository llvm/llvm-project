//===--- raw_ostream.h - Raw output stream ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the raw_ostream class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_RAW_OSTREAM_H
#define LLVM_SUPPORT_RAW_OSTREAM_H

#include "llvm/ADT/StringExtras.h"
#include <cassert>
#include <cstring>
#include <string>
#include <iosfwd>

namespace llvm {
  class format_object_base;
  template <typename T>
  class SmallVectorImpl;
  
/// raw_ostream - This class implements an extremely fast bulk output stream
/// that can *only* output to a stream.  It does not support seeking, reopening,
/// rewinding, line buffered disciplines etc. It is a simple buffer that outputs
/// a chunk at a time.
class raw_ostream {
protected:
  char *OutBufStart, *OutBufEnd, *OutBufCur;
public:
  raw_ostream() {
    // Start out ready to flush.
    OutBufStart = OutBufEnd = OutBufCur = 0;
  }

  virtual ~raw_ostream() {
    delete [] OutBufStart;
  }
  
  //===--------------------------------------------------------------------===//
  // Configuration Interface
  //===--------------------------------------------------------------------===//
  
  /// SetBufferSize - Set the internal buffer size to the specified amount
  /// instead of the default.
  void SetBufferSize(unsigned Size) {
    assert(Size >= 64 &&
           "Buffer size must be somewhat large for invariants to hold");
    flush();
    
    delete [] OutBufStart;
    OutBufStart = new char[Size];
    OutBufEnd = OutBufStart+Size;
    OutBufCur = OutBufStart;
  }
  
  //===--------------------------------------------------------------------===//
  // Data Output Interface
  //===--------------------------------------------------------------------===//
  
  void flush() {
    if (OutBufCur != OutBufStart)
      flush_impl();
  }
  
  raw_ostream &operator<<(char C) {
    if (OutBufCur >= OutBufEnd)
      flush_impl();
    *OutBufCur++ = C;
    return *this;
  }
  
  raw_ostream &operator<<(unsigned char C) {
    if (OutBufCur >= OutBufEnd)
      flush_impl();
    *OutBufCur++ = C;
    return *this;
  }
  
  raw_ostream &operator<<(signed char C) {
    if (OutBufCur >= OutBufEnd)
      flush_impl();
    *OutBufCur++ = C;
    return *this;
  }
  
  raw_ostream &operator<<(const char *Str) {
    return write(Str, strlen(Str));
  }
  
  raw_ostream &operator<<(const std::string& Str) {
    return write(Str.data(), Str.length());
  }
  
  raw_ostream &operator<<(unsigned long N);
  raw_ostream &operator<<(long N);
  raw_ostream &operator<<(unsigned long long N);
  raw_ostream &operator<<(long long N);
  raw_ostream &operator<<(const void *P);
  raw_ostream &operator<<(unsigned int N) {
    return this->operator<<(static_cast<unsigned long>(N));
  }
  
  raw_ostream &operator<<(int N) {
    return this->operator<<(static_cast<long>(N));
  }

  raw_ostream &operator<<(double N) {
    return this->operator<<(ftostr(N));
  }
  
  raw_ostream &write(const char *Ptr, unsigned Size);
  
  // Formatted output, see the format() function in Support/Format.h.
  raw_ostream &operator<<(const format_object_base &Fmt);
  
  //===--------------------------------------------------------------------===//
  // Subclass Interface
  //===--------------------------------------------------------------------===//

protected:
  
  /// flush_impl - The is the piece of the class that is implemented by
  /// subclasses.  This outputs the currently buffered data and resets the
  /// buffer to empty.
  virtual void flush_impl() = 0;
  
  /// HandleFlush - A stream's implementation of flush should call this after
  /// emitting the bytes to the data sink.
  void HandleFlush() {
    if (OutBufStart == 0)
      SetBufferSize(4096);
    OutBufCur = OutBufStart;
  }
private:
  // An out of line virtual method to provide a home for the class vtable.
  virtual void handle();
};
  
//===----------------------------------------------------------------------===//
// File Output Streams
//===----------------------------------------------------------------------===//
  
/// raw_fd_ostream - A raw_ostream that writes to a file descriptor.
///
class raw_fd_ostream : public raw_ostream {
  int FD;
  bool ShouldClose;
  uint64_t pos;
public:
  /// raw_fd_ostream - Open the specified file for writing. If an
  /// error occurs, information about the error is put into ErrorInfo,
  /// and the stream should be immediately destroyed; the string will
  /// be empty if no error occurred.
  ///
  /// \param Filename - The file to open. If this is "-" then the
  /// stream will use stdout instead.
  /// \param Binary - The file should be opened in binary mode on
  /// platforms that support this distinction.
  raw_fd_ostream(const char *Filename, bool Binary, std::string &ErrorInfo);
  
  /// raw_fd_ostream ctor - FD is the file descriptor that this writes to.  If
  /// ShouldClose is true, this closes the file when 
  raw_fd_ostream(int fd, bool shouldClose) : FD(fd), ShouldClose(shouldClose) {}
  
  ~raw_fd_ostream();
    
  /// flush_impl - The is the piece of the class that is implemented by
  /// subclasses.  This outputs the currently buffered data and resets the
  /// buffer to empty.
  virtual void flush_impl();
  
  /// close - Manually flush the stream and close the file.
  void close();
  
  /// tell - Return the current offset with the file.
  uint64_t tell() {
    return pos + (OutBufCur - OutBufStart);
  }
  
  /// seek - Flushes the stream and repositions the underlying file descriptor
  ///  positition to the offset specified from the beginning of the file.
  uint64_t seek(uint64_t off);
};
  
/// raw_stdout_ostream - This is a stream that always prints to stdout.
///
class raw_stdout_ostream : public raw_fd_ostream {
  // An out of line virtual method to provide a home for the class vtable.
  virtual void handle();
public:
  raw_stdout_ostream();
};

/// raw_stderr_ostream - This is a stream that always prints to stderr.
///
class raw_stderr_ostream : public raw_fd_ostream {
  // An out of line virtual method to provide a home for the class vtable.
  virtual void handle();
public:
  raw_stderr_ostream();
};
  
/// outs() - This returns a reference to a raw_ostream for standard output.
/// Use it like: outs() << "foo" << "bar";
raw_ostream &outs();

/// errs() - This returns a reference to a raw_ostream for standard error.
/// Use it like: errs() << "foo" << "bar";
raw_ostream &errs();
  
  
//===----------------------------------------------------------------------===//
// Output Stream Adaptors
//===----------------------------------------------------------------------===//
  
/// raw_os_ostream - A raw_ostream that writes to an std::ostream.  This is a
/// simple adaptor class.
class raw_os_ostream : public raw_ostream {
  std::ostream &OS;
public:
  raw_os_ostream(std::ostream &O) : OS(O) {}
  ~raw_os_ostream();

  /// flush_impl - The is the piece of the class that is implemented by
  /// subclasses.  This outputs the currently buffered data and resets the
  /// buffer to empty.
  virtual void flush_impl();
};

/// raw_string_ostream - A raw_ostream that writes to an std::string.  This is a
/// simple adaptor class.
class raw_string_ostream : public raw_ostream {
  std::string &OS;
public:
  raw_string_ostream(std::string &O) : OS(O) {}
  ~raw_string_ostream();
  
  /// str - Flushes the stream contents to the target string and returns
  ///  the string's reference.
  std::string& str() {
    flush();
    return OS;
  }
  
  /// flush_impl - The is the piece of the class that is implemented by
  /// subclasses.  This outputs the currently buffered data and resets the
  /// buffer to empty.
  virtual void flush_impl();
};
  
/// raw_svector_ostream - A raw_ostream that writes to an SmallVector or
/// SmallString.  This is a simple adaptor class.
class raw_svector_ostream : public raw_ostream {
  SmallVectorImpl<char> &OS;
public:
  raw_svector_ostream(SmallVectorImpl<char> &O) : OS(O) {}
  ~raw_svector_ostream();
  
  /// flush_impl - The is the piece of the class that is implemented by
  /// subclasses.  This outputs the currently buffered data and resets the
  /// buffer to empty.
  virtual void flush_impl();
};
  
} // end llvm namespace

#endif
