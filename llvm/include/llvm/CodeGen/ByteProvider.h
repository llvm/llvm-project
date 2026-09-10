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

#include "llvm/ADT/STLFunctionalExtras.h"
#include <optional>

namespace llvm {

class SDNode;
class SDValue;

/// Represents known origin of an individual byte in combine pattern. The
/// value of the byte is either constant zero, or comes from memory /
/// some other productive instruction (e.g. arithmetic instructions).
/// Bit manipulation instructions like shifts are not ByteProviders, rather
/// are used to extract Bytes.
class ByteProvider {
private:
  ByteProvider(SDNode *Node, unsigned ResNo, int64_t DestOffset,
               int64_t SrcOffset)
      : Node(Node), ResNo(ResNo), DestOffset(DestOffset), SrcOffset(SrcOffset) {
  }

public:
  // For constant zero providers Node is null. For actual providers Node and
  // ResNo represent the SDValue which originally produced the relevant bits.
  SDNode *Node = nullptr;
  unsigned ResNo = 0;
  // DestOffset and SrcOffset are producer defined, see DAGCombiner.cpp.
  int64_t DestOffset = 0;
  int64_t SrcOffset = 0;

  ByteProvider() = default;

  static ByteProvider getSrc(SDValue Val, int64_t ByteOffset,
                             int64_t VectorOffset);

  static ByteProvider getConstantZero() { return ByteProvider(); }
  bool isConstantZero() const { return !Node; }

  bool hasSrc() const { return Node != nullptr; }

  /// Returns the SDValue this byte comes from. Only valid if hasSrc().
  SDValue getSrc() const;

  bool hasSameSrc(const ByteProvider &Other) const {
    return Other.Node == Node && Other.ResNo == ResNo;
  }

  bool operator==(const ByteProvider &Other) const {
    return hasSameSrc(Other) && Other.DestOffset == DestOffset &&
           Other.SrcOffset == SrcOffset;
  }
};

using SDByteProviderRecurseFn =
    function_ref<std::optional<ByteProvider>(SDValue, unsigned)>;

/// Visits both operands even once one answers, because \p Recurse may have
/// side effects (DAGCombiner accumulates an and mask there).
std::optional<ByteProvider>
calculateByteProviderForOr(SDValue Op, unsigned Index,
                           SDByteProviderRecurseFn Recurse);

/// \p NarrowBitWidth is a parameter because it is not always the operand
/// width, for instance sign_extend_inreg takes it from the VTSDNode.
std::optional<ByteProvider>
calculateByteProviderForExtend(SDValue Op, unsigned Index,
                               unsigned NarrowBitWidth, bool ZeroFills,
                               SDByteProviderRecurseFn Recurse);

} // end namespace llvm

#endif // LLVM_CODEGEN_BYTEPROVIDER_H
