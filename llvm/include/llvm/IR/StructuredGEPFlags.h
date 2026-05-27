//===- llvm/IR/StructuredGEPFlags.h - Structured GEP flags ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the per-index flags for llvm.structured.gep.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_STRUCTUREDGEPFLAGS_H
#define LLVM_IR_STRUCTUREDGEPFLAGS_H

namespace llvm {

/// Represents the per-index flags for llvm.structured.gep.
class StructuredGEPFlags {
  enum Flag : unsigned {
    InBoundsFlag = (1 << 0),
    NNegFlag = (1 << 1),
    UnsignedFlag = (1 << 2),
  };

  unsigned Flags;

public:
  StructuredGEPFlags() : Flags(0) {}
  explicit StructuredGEPFlags(unsigned Flags) : Flags(Flags) {}

  static StructuredGEPFlags none() { return StructuredGEPFlags(); }
  static StructuredGEPFlags inBounds() {
    return StructuredGEPFlags(InBoundsFlag);
  }
  static StructuredGEPFlags nneg() { return StructuredGEPFlags(NNegFlag); }
  static StructuredGEPFlags unsignedIndex() {
    return StructuredGEPFlags(UnsignedFlag);
  }

  static StructuredGEPFlags fromRaw(unsigned Flags) {
    return StructuredGEPFlags(Flags);
  }
  unsigned getRaw() const { return Flags; }

  bool isInBounds() const { return Flags & InBoundsFlag; }
  bool isNNeg() const { return Flags & NNegFlag; }
  bool isUnsignedIndex() const { return Flags & UnsignedFlag; }

  bool operator==(StructuredGEPFlags Other) const {
    return Flags == Other.Flags;
  }
  bool operator!=(StructuredGEPFlags Other) const { return !(*this == Other); }

  StructuredGEPFlags operator&(StructuredGEPFlags Other) const {
    return StructuredGEPFlags(Flags & Other.Flags);
  }
  StructuredGEPFlags operator|(StructuredGEPFlags Other) const {
    return StructuredGEPFlags(Flags | Other.Flags);
  }
  StructuredGEPFlags &operator&=(StructuredGEPFlags Other) {
    Flags &= Other.Flags;
    return *this;
  }
  StructuredGEPFlags &operator|=(StructuredGEPFlags Other) {
    Flags |= Other.Flags;
    return *this;
  }
};

} // end namespace llvm

#endif // LLVM_IR_STRUCTUREDGEPFLAGS_H
