//===-- llvm/Support/Alignment.h - Useful alignment functions ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides a class that can represent the separate destination and
// source volatilities of memory transfer intrinsics (llvm.memcpy et al).
// It provides a deprecated bool-like compatibility layer.

#ifndef LLVM_SUPPORT_MEMTRANSFERVOLATILITY_H_
#define LLVM_SUPPORT_MEMTRANSFERVOLATILITY_H_

#include "llvm/IR/Constants.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class MemTransferVolatility {
  unsigned Flags = 0;

public:
  MemTransferVolatility() = default;
  MemTransferVolatility(const ConstantInt *Int)
      : Flags(unsigned(Int->getZExtValue())) {
    assert(!(Int->getZExtValue() >> 2) && "Invalid volatilities");
  }

private:
  MemTransferVolatility(unsigned F) : Flags(F) {}

public:
  auto Dst(bool F = true) {
    assert(!(Flags & 1) && "Dst volatility already set");
    return MemTransferVolatility(Flags | (int(F) << 0));
  }
  auto Src(bool F = true) {
    assert(!(Flags & 2) && "Src volatility already set");
    return MemTransferVolatility(Flags | (int(F) << 1));
  }
  auto All(bool F = true) { return Dst(F).Src(F); }

public:
  // Compatibility layer -- make this type bool-like, but moan at you.
  LLVM_DEPRECATED("Specify separate Dst & Src volatilities instead",
                  "MemTransferVolatility().(All|Dst|Src|All)(bool)")
  MemTransferVolatility(bool isVolatile) { All(isVolatile); }
  LLVM_DEPRECATED(
      "Use isAnyVolatile, isDstVolatile or isSrcVolatile predicates instead",
      "is(Any|Dst|Src)Volatile()")
  operator bool() const { return isAnyVolatile(); }

public:
  // Create the integral constant value for the intrinsic call.
  ConstantInt *getAsInt(LLVMContext &Ctx) const {
    return ConstantInt::get(Type::getInt8Ty(Ctx), Flags);
  }

public:
  bool isAnyVolatile() const { return bool(Flags); }
  bool isDstVolatile() const { return bool(Flags & 1); }
  bool isSrcVolatile() const { return bool(Flags & 2); }
};

} // end namespace llvm

#endif // LLVM_SUPPORT_MEMTRANSFERVOLATILITY_H_
