//===-- include/llvm/CodeGen/ByteProvider.h - Map bytes ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file
// This file implements ByteProvider. The purpose of ByteProvider is to provide
// a map between a byte of a target node and the source that provides it.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_BYTEPROVIDER_H
#define LLVM_CODEGEN_BYTEPROVIDER_H

#include "llvm/CodeGen/SelectionDAGNodes.h"
#include <optional>

namespace llvm {

/// Represents known origin of an individual byte in combine pattern. The
/// value of the byte is either constant zero, or comes from memory /
/// some other productive instruction (e.g. arithmetic instructions).
/// Bit manipulation instructions like shifts are not ByteProviders, rather
/// are used to extract Bytes.
class ByteProvider {
private:
  ByteProvider(SDValue Src, int64_t DestOffset, int64_t SrcOffset)
      : Src(Src), DestOffset(DestOffset), SrcOffset(SrcOffset) {}

public:
  // For constant zero providers Src is null. For actual providers Src is the
  // value which originally produced the relevant bits.
  SDValue Src;
  int64_t DestOffset = 0; // Load byte in DAGCombiner, unused in AMDGPU.
  int64_t SrcOffset = 0;  // Vector lane in DAGCombiner, byte in Src in AMDGPU.

  ByteProvider() = default;

  static ByteProvider getSrc(SDValue Val, int64_t ByteOffset,
                             int64_t VectorOffset) {
    return ByteProvider(Val, ByteOffset, VectorOffset);
  }

  static ByteProvider getConstantZero() { return ByteProvider(); }
  bool isConstantZero() const { return !Src; }

  bool hasSrc() const { return static_cast<bool>(Src); }

  bool hasSameSrc(const ByteProvider &Other) const { return Other.Src == Src; }

  bool operator==(const ByteProvider &Other) const {
    return hasSameSrc(Other) && Other.DestOffset == DestOffset &&
           Other.SrcOffset == SrcOffset;
  }
};

/// In a well formed or, one of the two byte providers is constant zero.
std::optional<ByteProvider>
selectOrByteProvider(const std::optional<ByteProvider> &LHS,
                     const std::optional<ByteProvider> &RHS);

enum class NarrowByteAction { Unknown, ConstantZero, FromNarrow };

/// FromNarrow keeps \p Index. \p NarrowBitWidth is not always the operand
/// width, sign_extend_inreg takes it from the VTSDNode.
NarrowByteAction classifyNarrowByte(unsigned Index, unsigned NarrowBitWidth,
                                    bool ZeroFills);

} // end namespace llvm

#endif // LLVM_CODEGEN_BYTEPROVIDER_H
