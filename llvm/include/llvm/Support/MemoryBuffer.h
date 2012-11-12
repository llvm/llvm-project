//===--- MemoryBuffer.h - Memory Buffer Interface ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the MemoryBuffer interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_MEMORYBUFFER_H
#define LLVM_SUPPORT_MEMORYBUFFER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {

class error_code;
template<class T> class OwningPtr;

/// MemoryBuffer - This interface provides simple read-only access to a block
/// of memory, and provides simple methods for reading files and standard input
/// into a memory buffer.  In addition to basic access to the characters in the
/// file, this interface guarantees you can read one character past the end of
/// the file, and that this character will read as '\0'.
///
/// The '\0' guarantee is needed to support an optimization -- it's intended to
/// be more efficient for clients which are reading all the data to stop
/// reading when they encounter a '\0' than to continually check the file
/// position to see if it has reached the end of the file.
class MemoryBuffer {
  const char *BufferStart; // Start of the buffer.
  const char *BufferEnd;   // End of the buffer.

  MemoryBuffer(const MemoryBuffer &) LLVM_DELETED_FUNCTION;
  MemoryBuffer &operator=(const MemoryBuffer &) LLVM_DELETED_FUNCTION;
protected:
  MemoryBuffer() {}
  void init(const char *BufStart, const char *BufEnd,
            bool RequiresNullTerminator);
public:
  virtual ~MemoryBuffer();

  const char *getBufferStart() const { return BufferStart; }
  const char *getBufferEnd() const   { return BufferEnd; }
  size_t getBufferSize() const { return BufferEnd-BufferStart; }

  StringRef getBuffer() const {
    return StringRef(BufferStart, getBufferSize());
  }

  /// getBufferIdentifier - Return an identifier for this buffer, typically the
  /// filename it was read from.
  virtual const char *getBufferIdentifier() const {
    return "Unknown buffer";
  }

  /// getFile - Open the specified file as a MemoryBuffer, returning a new
  /// MemoryBuffer if successful, otherwise returning null.  If FileSize is
  /// specified, this means that the client knows that the file exists and that
  /// it has the specified size.
  static error_code getFile(StringRef Filename, OwningPtr<MemoryBuffer> &result,
                            int64_t FileSize = -1,
                            bool RequiresNullTerminator = true);
  static error_code getFile(const char *Filename,
                            OwningPtr<MemoryBuffer> &result,
                            int64_t FileSize = -1,
                            bool RequiresNullTerminator = true);

  /// getOpenFile - Given an already-open file descriptor, read the file and
  /// return a MemoryBuffer.
  static error_code getOpenFile(int FD, const char *Filename,
                                OwningPtr<MemoryBuffer> &result,
                                uint64_t FileSize = -1,
                                uint64_t MapSize = -1,
                                int64_t Offset = 0,
                                bool RequiresNullTerminator = true);

  /// getMemBuffer - Open the specified memory range as a MemoryBuffer.  Note
  /// that InputData must be null terminated if RequiresNullTerminator is true.
  static MemoryBuffer *getMemBuffer(StringRef InputData,
                                    StringRef BufferName = "",
                                    bool RequiresNullTerminator = true);

  /// getMemBufferCopy - Open the specified memory range as a MemoryBuffer,
  /// copying the contents and taking ownership of it.  InputData does not
  /// have to be null terminated.
  static MemoryBuffer *getMemBufferCopy(StringRef InputData,
                                        StringRef BufferName = "");

  /// getNewMemBuffer - Allocate a new MemoryBuffer of the specified size that
  /// is completely initialized to zeros.  Note that the caller should
  /// initialize the memory allocated by this method.  The memory is owned by
  /// the MemoryBuffer object.
  static MemoryBuffer *getNewMemBuffer(size_t Size, StringRef BufferName = "");

  /// getNewUninitMemBuffer - Allocate a new MemoryBuffer of the specified size
  /// that is not initialized.  Note that the caller should initialize the
  /// memory allocated by this method.  The memory is owned by the MemoryBuffer
  /// object.
  static MemoryBuffer *getNewUninitMemBuffer(size_t Size,
                                             StringRef BufferName = "");

  /// getSTDIN - Read all of stdin into a file buffer, and return it.
  /// If an error occurs, this returns null and sets ec.
  static error_code getSTDIN(OwningPtr<MemoryBuffer> &result);


  /// getFileOrSTDIN - Open the specified file as a MemoryBuffer, or open stdin
  /// if the Filename is "-".  If an error occurs, this returns null and sets
  /// ec.
  static error_code getFileOrSTDIN(StringRef Filename,
                                   OwningPtr<MemoryBuffer> &result,
                                   int64_t FileSize = -1);
  static error_code getFileOrSTDIN(const char *Filename,
                                   OwningPtr<MemoryBuffer> &result,
                                   int64_t FileSize = -1);
  
  
  //===--------------------------------------------------------------------===//
  // Provided for performance analysis.
  //===--------------------------------------------------------------------===//

  /// The kind of memory backing used to support the MemoryBuffer.
  enum BufferKind {
    MemoryBuffer_Malloc,
    MemoryBuffer_MMap
  };

  /// Return information on the memory mechanism used to support the
  /// MemoryBuffer.
  virtual BufferKind getBufferKind() const = 0;  
};

} // end namespace llvm

#endif
