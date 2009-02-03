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

#include "llvm/Support/DataTypes.h"
#include <string>

namespace llvm {

/// MemoryBuffer - This interface provides simple read-only access to a block
/// of memory, and provides simple methods for reading files and standard input
/// into a memory buffer.  In addition to basic access to the characters in the
/// file, this interface guarantees you can read one character past the end of
/// @verbatim the file, and that this character will read as '\0'. @endverbatim
class MemoryBuffer {
  const char *BufferStart; // Start of the buffer.
  const char *BufferEnd;   // End of the buffer.

  /// MustDeleteBuffer - True if we allocated this buffer.  If so, the
  /// destructor must know the delete[] it.
  bool MustDeleteBuffer;
protected:
  MemoryBuffer() : MustDeleteBuffer(false) {}
  void init(const char *BufStart, const char *BufEnd);
  void initCopyOf(const char *BufStart, const char *BufEnd);
public:
  virtual ~MemoryBuffer();
  
  const char *getBufferStart() const { return BufferStart; }
  const char *getBufferEnd() const   { return BufferEnd; }
  size_t getBufferSize() const { return BufferEnd-BufferStart; }
  
  /// getBufferIdentifier - Return an identifier for this buffer, typically the
  /// filename it was read from.
  virtual const char *getBufferIdentifier() const {
    return "Unknown buffer";
  }

  /// getFile - Open the specified file as a MemoryBuffer, returning a new
  /// MemoryBuffer if successful, otherwise returning null.  If FileSize is
  /// specified, this means that the client knows that the file exists and that
  /// it has the specified size.
  static MemoryBuffer *getFile(const char *Filename,
                               std::string *ErrStr = 0,
                               int64_t FileSize = -1);

  /// getMemBuffer - Open the specified memory range as a MemoryBuffer.  Note
  /// that EndPtr[0] must be a null byte and be accessible!
  static MemoryBuffer *getMemBuffer(const char *StartPtr, const char *EndPtr,
                                    const char *BufferName = "");
  
  /// getMemBufferCopy - Open the specified memory range as a MemoryBuffer,
  /// copying the contents and taking ownership of it.  This has no requirements
  /// on EndPtr[0].
  static MemoryBuffer *getMemBufferCopy(const char *StartPtr,const char *EndPtr,
                                        const char *BufferName = "");
  
  /// getNewMemBuffer - Allocate a new MemoryBuffer of the specified size that
  /// is completely initialized to zeros.  Note that the caller should
  /// initialize the memory allocated by this method.  The memory is owned by
  /// the MemoryBuffer object.
  static MemoryBuffer *getNewMemBuffer(size_t Size,
                                       const char *BufferName = "");
  
  /// getNewUninitMemBuffer - Allocate a new MemoryBuffer of the specified size
  /// that is not initialized.  Note that the caller should initialize the
  /// memory allocated by this method.  The memory is owned by the MemoryBuffer
  /// object.
  static MemoryBuffer *getNewUninitMemBuffer(size_t Size,
                                             const char *BufferName = "");
  
  /// getSTDIN - Read all of stdin into a file buffer, and return it.  This
  /// returns null if stdin is empty.
  static MemoryBuffer *getSTDIN();
  
  
  /// getFileOrSTDIN - Open the specified file as a MemoryBuffer, or open stdin
  /// if the Filename is "-".  If an error occurs, this returns null and fills
  /// in *ErrStr with a reason.  If stdin is empty, this API (unlike getSTDIN)
  /// returns an empty buffer.
  static MemoryBuffer *getFileOrSTDIN(const char *Filename,
                                      std::string *ErrStr = 0,
                                      int64_t FileSize = -1);
  
  /// getFileOrSTDIN - Open the specified file as a MemoryBuffer, or open stdin
  /// if the Filename is "-".  If an error occurs, this returns null and fills
  /// in *ErrStr with a reason.
  static MemoryBuffer *getFileOrSTDIN(const std::string &FN,
                                      std::string *ErrStr = 0,
                                      int64_t FileSize = -1) {
    return getFileOrSTDIN(FN.c_str(), ErrStr, FileSize);
  }
};

} // end namespace llvm

#endif
