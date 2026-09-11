//===-- ByteProvider.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/ByteProvider.h"

using namespace llvm;

std::optional<ByteProvider>
llvm::selectOrByteProvider(const std::optional<ByteProvider> &LHS,
                           const std::optional<ByteProvider> &RHS) {
  if (!LHS || !RHS)
    return std::nullopt;
  if (LHS->isConstantZero())
    return RHS;
  if (RHS->isConstantZero())
    return LHS;
  return std::nullopt;
}

NarrowByteAction llvm::classifyNarrowByte(unsigned Index,
                                          unsigned NarrowBitWidth,
                                          bool ZeroFills) {
  if (NarrowBitWidth % 8 != 0)
    return NarrowByteAction::Unknown;
  if (Index < NarrowBitWidth / 8)
    return NarrowByteAction::FromNarrow;
  return ZeroFills ? NarrowByteAction::ConstantZero : NarrowByteAction::Unknown;
}
