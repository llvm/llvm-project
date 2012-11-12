//===---- ObjectBuffer.h - Utility class to wrap object image memory -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares a wrapper class to hold the memory into which an
// object will be generated.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_OBJECTBUFFER_H
#define LLVM_EXECUTIONENGINE_OBJECTBUFFER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/MemoryBuffer.h"

namespace llvm {

/// ObjectBuffer - This class acts as a container for the memory buffer used during
/// generation and loading of executable objects using MCJIT and RuntimeDyld.  The
/// underlying memory for the object will be owned by the ObjectBuffer instance
/// throughout its lifetime.  The getMemBuffer() method provides a way to create a
/// MemoryBuffer wrapper object instance to be owned by other classes (such as
/// ObjectFile) as needed, but the MemoryBuffer instance returned does not own the
/// actual memory it points to.
class ObjectBuffer {
public:
  ObjectBuffer() {}
  ObjectBuffer(MemoryBuffer* Buf) : Buffer(Buf) {}
  virtual ~ObjectBuffer() {}

  /// getMemBuffer - Like MemoryBuffer::getMemBuffer() this function
  /// returns a pointer to an object that is owned by the caller. However,
  /// the caller does not take ownership of the underlying memory.
  MemoryBuffer *getMemBuffer() const {
    return MemoryBuffer::getMemBuffer(Buffer->getBuffer(), "", false);
  }

  const char *getBufferStart() const { return Buffer->getBufferStart(); }
  size_t getBufferSize() const { return Buffer->getBufferSize(); }

protected:
  // The memory contained in an ObjectBuffer
  OwningPtr<MemoryBuffer> Buffer;
};

/// ObjectBufferStream - This class encapsulates the SmallVector and
/// raw_svector_ostream needed to generate an object using MC code emission
/// while providing a common ObjectBuffer interface for access to the
/// memory once the object has been generated.
class ObjectBufferStream : public ObjectBuffer {
public:
  ObjectBufferStream() : OS(SV) {}
  virtual ~ObjectBufferStream() {}

  raw_ostream &getOStream() { return OS; }
  void flush()
  {
    OS.flush();

    // Make the data accessible via the ObjectBuffer::Buffer
    Buffer.reset(MemoryBuffer::getMemBuffer(StringRef(SV.data(), SV.size()),
                                            "",
                                            false));
  }

protected:
  SmallVector<char, 4096> SV; // Working buffer into which we JIT.
  raw_svector_ostream     OS; // streaming wrapper
};

} // namespace llvm

#endif
