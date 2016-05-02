//===- ByteStream.h - Reads stream data from a byte sequence ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_RAW_BYTESTREAM_H
#define LLVM_DEBUGINFO_PDB_RAW_BYTESTREAM_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/DebugInfo/PDB/Raw/StreamInterface.h"

#include <stdint.h>

#include <system_error>
#include <vector>

namespace llvm {
namespace pdb {
class StreamReader;
class ByteStream : public StreamInterface {
public:
  ByteStream();
  explicit ByteStream(MutableArrayRef<uint8_t> Bytes);
  explicit ByteStream(uint32_t Length);
  ~ByteStream() override;

  void reset();
  void initialize(MutableArrayRef<uint8_t> Bytes);
  void initialize(uint32_t Length);
  std::error_code initialize(StreamReader &Reader, uint32_t Length);

  std::error_code readBytes(uint32_t Offset,
                            MutableArrayRef<uint8_t> Buffer) const override;
  uint32_t getLength() const override;

  ArrayRef<uint8_t> data() const { return Data; }
  StringRef str() const;

private:
  MutableArrayRef<uint8_t> Data;
  bool Owned;
};
}
}

#endif
