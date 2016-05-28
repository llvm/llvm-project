//===- ByteStream.h - Reads stream data from a byte sequence ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_BYTESTREAM_H
#define LLVM_DEBUGINFO_CODEVIEW_BYTESTREAM_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/CodeView/StreamInterface.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <memory>

namespace llvm {
namespace codeview {
class StreamReader;

class ByteStream : public StreamInterface {
public:
  ByteStream();
  explicit ByteStream(ArrayRef<uint8_t> Data);
  ~ByteStream() override;

  Error readBytes(uint32_t Offset, uint32_t Size,
                  ArrayRef<uint8_t> &Buffer) const override;

  uint32_t getLength() const override;

  ArrayRef<uint8_t> data() const { return Data; }
  StringRef str() const;

private:
  ArrayRef<uint8_t> Data;
};

} // end namespace pdb
} // end namespace llvm

#endif // LLVM_DEBUGINFO_CODEVIEW_BYTESTREAM_H
