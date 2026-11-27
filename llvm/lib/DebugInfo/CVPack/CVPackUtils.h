//===- CVPackUtils.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helpers shared by the sources of the DebugInfoCVPack library.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_DEBUGINFO_CVPACK_CVPACKUTILS_H
#define LLVM_LIB_DEBUGINFO_CVPACK_CVPACKUTILS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/DebugInfo/CodeView/CVRecord.h"
#include "llvm/DebugInfo/CodeView/CodeViewError.h"
#include "llvm/DebugInfo/CodeView/RecordSerialization.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <vector>

namespace llvm {
namespace codeview {
namespace cvpack {

inline Error malformedRecord(const Twine &Message) {
  return make_error<CodeViewError>(Message);
}

/// Append Value in little-endian order.
template <typename T> void appendLE(std::vector<uint8_t> &Bytes, T Value) {
  uint8_t Buffer[sizeof(T)];
  support::endian::write<T, llvm::endianness::little>(Buffer, Value);
  Bytes.insert(Bytes.end(), Buffer, Buffer + sizeof(T));
}

inline void appendBytes(std::vector<uint8_t> &Bytes, ArrayRef<uint8_t> Data) {
  Bytes.insert(Bytes.end(), Data.begin(), Data.end());
}

inline void appendAlignedZeros(std::vector<uint8_t> &Bytes) {
  while ((Bytes.size() & 3) != 0)
    Bytes.push_back(0);
}

inline bool containsRange(ArrayRef<uint8_t> Bytes, uint64_t Offset,
                          uint64_t Size) {
  return Offset <= Bytes.size() && Size <= Bytes.size() - Offset;
}

/// A COFF header in a buffer, which may be written to.
template <typename HeaderT>
HeaderT &headerAt(std::vector<uint8_t> &Bytes, size_t Offset) {
  return *reinterpret_cast<HeaderT *>(Bytes.data() + Offset);
}

/// Check that Bytes is one CodeView record: a length that covers the rest.
inline Error validateRecord(ArrayRef<uint8_t> Bytes) {
  if (Bytes.size() < sizeof(RecordPrefix) || Bytes.size() > MaxRecordLength ||
      support::endian::read16le(Bytes.data()) !=
          Bytes.size() - sizeof(uint16_t))
    return malformedRecord("invalid CodeView record length");
  return Error::success();
}

} // namespace cvpack
} // namespace codeview
} // namespace llvm

#endif // LLVM_LIB_DEBUGINFO_CVPACK_CVPACKUTILS_H
