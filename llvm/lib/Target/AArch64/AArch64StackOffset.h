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
#include "llvm/Support/TypeSize.h"
#include <cassert>

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
  int64_t ScalableBytes;

  explicit operator int() const;

public:
  using Part = std::pair<int64_t, MVT>;

  StackOffset() : Bytes(0), ScalableBytes(0) {}

  StackOffset(int64_t Offset, MVT::SimpleValueType T) : StackOffset() {
    assert(MVT(T).isByteSized() && "Offset type is not a multiple of bytes");
    *this += Part(Offset, T);
  }

  StackOffset(const StackOffset &Other)
      : Bytes(Other.Bytes), ScalableBytes(Other.ScalableBytes) {}

  StackOffset &operator=(const StackOffset &) = default;

  StackOffset &operator+=(const StackOffset::Part &Other) {
    const TypeSize Size = Other.second.getSizeInBits();
    if (Size.isScalable())
      ScalableBytes += Other.first * ((int64_t)Size.getKnownMinSize() / 8);
    else
      Bytes += Other.first * ((int64_t)Size.getFixedSize() / 8);
    return *this;
  }

  StackOffset &operator+=(const StackOffset &Other) {
    Bytes += Other.Bytes;
    ScalableBytes += Other.ScalableBytes;
    return *this;
  }

  StackOffset operator+(const StackOffset &Other) const {
    StackOffset Res(*this);
    Res += Other;
    return Res;
  }

  StackOffset &operator-=(const StackOffset &Other) {
    Bytes -= Other.Bytes;
    ScalableBytes -= Other.ScalableBytes;
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

  /// Returns the scalable part of the offset in bytes.
  int64_t getScalableBytes() const { return ScalableBytes; }

  /// Returns the non-scalable part of the offset in bytes.
  int64_t getBytes() const { return Bytes; }

  /// Returns the offset in parts to which this frame offset can be
  /// decomposed for the purpose of describing a frame offset.
  /// For non-scalable offsets this is simply its byte size.
  void getForFrameOffset(int64_t &NumBytes, int64_t &NumPredicateVectors,
                         int64_t &NumDataVectors) const {
    assert(isValid() && "Invalid frame offset");

    NumBytes = Bytes;
    NumDataVectors = 0;
    NumPredicateVectors = ScalableBytes / 2;
    // This method is used to get the offsets to adjust the frame offset.
    // If the function requires ADDPL to be used and needs more than two ADDPL
    // instructions, part of the offset is folded into NumDataVectors so that it
    // uses ADDVL for part of it, reducing the number of ADDPL instructions.
    if (NumPredicateVectors % 8 == 0 || NumPredicateVectors < -64 ||
        NumPredicateVectors > 62) {
      NumDataVectors = NumPredicateVectors / 8;
      NumPredicateVectors -= NumDataVectors * 8;
    }
  }

  void getForDwarfOffset(int64_t &ByteSized, int64_t &VGSized) const {
    assert(isValid() && "Invalid frame offset");

    // VGSized offsets are divided by '2', because the VG register is the
    // the number of 64bit granules as opposed to 128bit vector chunks,
    // which is how the 'n' in e.g. MVT::nxv1i8 is modelled.
    // So, for a stack offset of 16 MVT::nxv1i8's, the size is n x 16 bytes.
    // VG = n * 2 and the dwarf offset must be VG * 8 bytes.
    ByteSized = Bytes;
    VGSized = ScalableBytes / 2;
  }

  /// Returns whether the offset is known zero.
  explicit operator bool() const { return Bytes || ScalableBytes; }

  bool isValid() const {
    // The smallest scalable element supported by scaled SVE addressing
    // modes are predicates, which are 2 scalable bytes in size. So the scalable
    // byte offset must always be a multiple of 2.
    return ScalableBytes % 2 == 0;
  }
};

} // end namespace llvm

#endif
