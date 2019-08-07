//==--AArch64StackOffset.h ---------------------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the StackOffset class, which is used to
// describe scalable and non-scalable offsets during frame lowering.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AARCH64_AARCH64STACKOFFSET_H
#define LLVM_LIB_TARGET_AARCH64_AARCH64STACKOFFSET_H

#include "llvm/Support/MachineValueType.h"

namespace llvm {

/// StackOffset is a wrapper around scalable and non-scalable offsets and is
/// used in several functions such as 'isAArch64FrameOffsetLegal' and
/// 'emitFrameOffset()'. StackOffsets are described by MVTs, e.g.
//
///   StackOffset(1, MVT::nxv16i8)
//
/// would describe an offset as being the size of a single SVE vector.
///
/// The class also implements simple arithmetic (addition/subtraction) on these
/// offsets, e.g.
//
///   StackOffset(1, MVT::nxv16i8) + StackOffset(1, MVT::i64)
//
/// describes an offset that spans the combined storage required for an SVE
/// vector and a 64bit GPR.
class StackOffset {
  int64_t Bytes;

  explicit operator int() const;

public:
  using Part = std::pair<int64_t, MVT>;

  StackOffset() : Bytes(0) {}

  StackOffset(int64_t Offset, MVT::SimpleValueType T) : StackOffset() {
    assert(!MVT(T).isScalableVector() && "Scalable types not supported");
    *this += Part(Offset, T);
  }

  StackOffset(const StackOffset &Other) : Bytes(Other.Bytes) {}

  StackOffset &operator=(const StackOffset &) = default;

  StackOffset &operator+=(const StackOffset::Part &Other) {
    assert(Other.second.getSizeInBits() % 8 == 0 &&
           "Offset type is not a multiple of bytes");
    Bytes += Other.first * (Other.second.getSizeInBits() / 8);
    return *this;
  }

  StackOffset &operator+=(const StackOffset &Other) {
    Bytes += Other.Bytes;
    return *this;
  }

  StackOffset operator+(const StackOffset &Other) const {
    StackOffset Res(*this);
    Res += Other;
    return Res;
  }

  StackOffset &operator-=(const StackOffset &Other) {
    Bytes -= Other.Bytes;
    return *this;
  }

  StackOffset operator-(const StackOffset &Other) const {
    StackOffset Res(*this);
    Res -= Other;
    return Res;
  }

  StackOffset operator-() const {
    StackOffset Res = {};
    const StackOffset Other(*this);
    Res -= Other;
    return Res;
  }

  /// Returns the non-scalable part of the offset in bytes.
  int64_t getBytes() const { return Bytes; }

  /// Returns the offset in parts to which this frame offset can be
  /// decomposed for the purpose of describing a frame offset.
  /// For non-scalable offsets this is simply its byte size.
  void getForFrameOffset(int64_t &ByteSized) const { ByteSized = Bytes; }

  /// Returns whether the offset is known zero.
  explicit operator bool() const { return Bytes; }
};

} // end namespace llvm

#endif
