//===----- AArch64ImmCheck.h -- ARM immediate range check -----*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements the ImmCheck class which supports the range-checking of
/// immediate values supplied to AArch64 SVE/SME and NEON intrinsics.
///
//===----------------------------------------------------------------------===//

#ifndef AARCH64_IMMCHECK_H
#define AARCH64_IMMCHECK_H

class ImmCheck {
  int ImmArgIdx;
  unsigned Kind;
  unsigned ElementSizeInBits;
  unsigned VecSizeInBits;
  int TypeArgIdx;

public:
  ImmCheck(int ImmArgIdx, unsigned Kind, unsigned ElementSizeInBits = 0,
           unsigned VecSizeInBits = 128, int TypeArgIdx = -1)
      : ImmArgIdx(ImmArgIdx), Kind(Kind), ElementSizeInBits(ElementSizeInBits),
        VecSizeInBits(VecSizeInBits), TypeArgIdx(TypeArgIdx) {}
  ImmCheck(const ImmCheck &Other) = default;
  ~ImmCheck() = default;

  bool operator==(const ImmCheck &other) const {
    return other.getImmArgIdx() == ImmArgIdx && other.getKind() == Kind &&
           other.getTypeArgIdx() == TypeArgIdx;
  }
  int getImmArgIdx() const { return ImmArgIdx; }
  int getTypeArgIdx() const { return TypeArgIdx; }
  unsigned getKind() const { return Kind; }
  unsigned getElementSizeInBits() const { return ElementSizeInBits; }
  unsigned getVecSizeInBits() const { return VecSizeInBits; }
};

#endif
