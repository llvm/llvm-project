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

#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/Support/DataTypes.h"
#include <optional>
#include <type_traits>

namespace llvm {

/// Represents known origin of an individual byte in combine pattern. The
/// value of the byte is either constant zero, or comes from memory /
/// some other productive instruction (e.g. arithmetic instructions).
/// Bit manipulation instructions like shifts are not ByteProviders, rather
/// are used to extract Bytes.
template <typename ISelOp> class ByteProvider {
private:
  ByteProvider(std::optional<ISelOp> Src, int64_t DestOffset, int64_t SrcOffset)
      : Src(Src), DestOffset(DestOffset), SrcOffset(SrcOffset) {}

  // TODO -- use constraint in c++20
  // Does this type correspond with an operation in selection DAG
  // Only allow classes with member function getOpcode
  template <typename U>
  using check_has_getOpcode =
      decltype(std::declval<std::remove_pointer_t<U> &>().getOpcode());

  template <typename U>
  static constexpr bool has_getOpcode =
      is_detected<check_has_getOpcode, U>::value;

public:
  // For constant zero providers Src is set to nullopt. For actual providers
  // Src represents the node which originally produced the relevant bits.
  std::optional<ISelOp> Src = std::nullopt;
  // DestOffset and SrcOffset are producer defined, see DAGCombiner.cpp and
  // SIISelLowering.cpp.
  int64_t DestOffset = 0;
  int64_t SrcOffset = 0;

  ByteProvider() = default;

  static ByteProvider getSrc(std::optional<ISelOp> Val, int64_t ByteOffset,
                             int64_t VectorOffset) {
    static_assert(has_getOpcode<ISelOp>,
                  "ByteProviders must contain an operation in selection DAG.");
    return ByteProvider(Val, ByteOffset, VectorOffset);
  }

  static ByteProvider getConstantZero() {
    return ByteProvider<ISelOp>(std::nullopt, 0, 0);
  }
  bool isConstantZero() const { return !Src; }

  bool hasSrc() const { return Src.has_value(); }

  bool hasSameSrc(const ByteProvider &Other) const { return Other.Src == Src; }

  bool operator==(const ByteProvider &Other) const {
    return Other.Src == Src && Other.DestOffset == DestOffset &&
           Other.SrcOffset == SrcOffset;
  }
};

using SDByteProviderRecurseFn =
    function_ref<std::optional<ByteProvider<SDValue>>(SDValue, unsigned)>;

/// Visits both operands even once one answers, because \p Recurse may have
/// side effects (DAGCombiner accumulates an and mask there).
inline std::optional<ByteProvider<SDValue>>
calculateByteProviderForOr(SDValue Op, unsigned Index,
                           SDByteProviderRecurseFn Recurse) {
  std::optional<ByteProvider<SDValue>> LHS = Recurse(Op.getOperand(0), Index);
  if (!LHS)
    return std::nullopt;
  std::optional<ByteProvider<SDValue>> RHS = Recurse(Op.getOperand(1), Index);
  if (!RHS)
    return std::nullopt;

  // A well formed or has two ByteProviders for each byte, one of which is
  // constant zero.
  if (LHS->isConstantZero())
    return RHS;
  if (RHS->isConstantZero())
    return LHS;
  return std::nullopt;
}

/// \p NarrowBitWidth is a parameter because it is not always the operand
/// width, for instance sign_extend_inreg takes it from the VTSDNode.
inline std::optional<ByteProvider<SDValue>>
calculateByteProviderForExtend(SDValue Op, unsigned Index,
                               unsigned NarrowBitWidth, bool ZeroFills,
                               SDByteProviderRecurseFn Recurse) {
  if (NarrowBitWidth % 8 != 0)
    return std::nullopt;

  if (Index >= NarrowBitWidth / 8) {
    if (!ZeroFills)
      return std::nullopt;
    return ByteProvider<SDValue>::getConstantZero();
  }
  return Recurse(Op.getOperand(0), Index);
}

} // end namespace llvm

#endif // LLVM_CODEGEN_BYTEPROVIDER_H
