//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines ContiguousBlobAccumulator, the size-limited output buffer
/// shared by the yaml2obj emitters.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECTYAML_CONTIGUOUSBLOBACCUMULATOR_H
#define LLVM_OBJECTYAML_CONTIGUOUSBLOBACCUMULATOR_H

#include "llvm/Support/EndianStream.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace yaml {

class BinaryRef;

// This class is used to build up a contiguous binary blob while keeping
// track of an offset in the output (which notionally begins at
// `InitialOffset`).
// The blob might be limited to an arbitrary size. All attempts to write data
// are ignored and the error condition is remembered once the limit is reached.
// Such an approach allows us to simplify the code by delaying error reporting
// and doing it at a convenient time.
class ContiguousBlobAccumulator {
  const uint64_t InitialOffset;
  const uint64_t MaxSize;

  SmallVector<char, 128> Buf;
  raw_svector_ostream OS;
  Error ReachedLimitErr = Error::success();

  LLVM_ABI bool checkLimit(uint64_t Size);

public:
  ContiguousBlobAccumulator(uint64_t BaseOffset, uint64_t SizeLimit)
      : InitialOffset(BaseOffset), MaxSize(SizeLimit), OS(Buf) {}

  uint64_t tell() const { return OS.tell(); }
  uint64_t getOffset() const { return InitialOffset + OS.tell(); }
  void writeBlobToStream(raw_ostream &Out) const { Out << OS.str(); }

  Error takeLimitError() {
    // Request to write 0 bytes to check we did not reach the limit.
    checkLimit(0);
    return std::move(ReachedLimitErr);
  }

  /// \returns The new offset.
  LLVM_ABI uint64_t padToAlignment(unsigned Align);

  raw_ostream *getRawOS(uint64_t Size) {
    if (checkLimit(Size))
      return &OS;
    return nullptr;
  }

  LLVM_ABI void writeAsBinary(const BinaryRef &Bin, uint64_t N = UINT64_MAX);

  void writeZeros(uint64_t Num) {
    if (checkLimit(Num))
      OS.write_zeros(Num);
  }

  void write(const char *Ptr, size_t Size) {
    if (checkLimit(Size))
      OS.write(Ptr, Size);
  }

  void write(unsigned char C) {
    if (checkLimit(1))
      OS.write(C);
  }

  LLVM_ABI unsigned writeULEB128(uint64_t Val);

  LLVM_ABI unsigned writeSLEB128(int64_t Val);

  template <typename T> void write(T Val, llvm::endianness E) {
    if (checkLimit(sizeof(T)))
      support::endian::write<T>(OS, Val, E);
  }

  LLVM_ABI void updateDataAt(uint64_t Pos, const void *Data, size_t Size);
};

} // end namespace yaml
} // end namespace llvm

#endif // LLVM_OBJECTYAML_CONTIGUOUSBLOBACCUMULATOR_H
