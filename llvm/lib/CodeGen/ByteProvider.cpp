//===-- ByteProvider.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/ByteProvider.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"

using namespace llvm;

ByteProvider ByteProvider::getSrc(SDValue Val, int64_t ByteOffset,
                                  int64_t VectorOffset) {
  return ByteProvider(Val.getNode(), Val.getResNo(), ByteOffset, VectorOffset);
}

SDValue ByteProvider::getSrc() const { return SDValue(Node, ResNo); }

std::optional<ByteProvider>
llvm::calculateByteProviderForOr(SDValue Op, unsigned Index,
                                 SDByteProviderRecurseFn Recurse) {
  std::optional<ByteProvider> LHS = Recurse(Op.getOperand(0), Index);
  if (!LHS)
    return std::nullopt;
  std::optional<ByteProvider> RHS = Recurse(Op.getOperand(1), Index);
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

std::optional<ByteProvider>
llvm::calculateByteProviderForExtend(SDValue Op, unsigned Index,
                                     unsigned NarrowBitWidth, bool ZeroFills,
                                     SDByteProviderRecurseFn Recurse) {
  if (NarrowBitWidth % 8 != 0)
    return std::nullopt;

  if (Index >= NarrowBitWidth / 8) {
    if (!ZeroFills)
      return std::nullopt;
    return ByteProvider::getConstantZero();
  }
  return Recurse(Op.getOperand(0), Index);
}
