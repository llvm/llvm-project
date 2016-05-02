//===- StreamReader.h - Reads bytes and objects from a stream ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_RAW_STREAMREADER_H
#define LLVM_DEBUGINFO_PDB_RAW_STREAMREADER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/DebugInfo/PDB/Raw/StreamInterface.h"
#include "llvm/Support/Endian.h"

#include <string>
#include <system_error>

namespace llvm {
namespace pdb {
class StreamInterface;
class StreamReader {
public:
  StreamReader(const StreamInterface &S);

  std::error_code readBytes(MutableArrayRef<uint8_t> Buffer);
  std::error_code readInteger(uint32_t &Dest);
  std::error_code readZeroString(std::string &Dest);

  template <typename T> std::error_code readObject(T *Dest) {
    MutableArrayRef<uint8_t> Buffer(reinterpret_cast<uint8_t *>(Dest),
                                    sizeof(T));
    return readBytes(Buffer);
  }

  template <typename T> std::error_code readArray(MutableArrayRef<T> Array) {
    MutableArrayRef<uint8_t> Casted(reinterpret_cast<uint8_t*>(Array.data()), Array.size() * sizeof(T));
    return readBytes(Casted);
  }

  void setOffset(uint32_t Off) { Offset = Off; }
  uint32_t getOffset() const { return Offset; }
  uint32_t getLength() const { return Stream.getLength(); }
  uint32_t bytesRemaining() const { return getLength() - getOffset(); }

private:
  const StreamInterface &Stream;
  uint32_t Offset;
};
}
}

#endif
