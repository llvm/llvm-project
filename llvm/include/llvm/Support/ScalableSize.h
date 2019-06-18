//===- ScalableSize.h - Scalable vector size info ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides a struct that can be used to query the size of IR types
// which may be scalable vectors. It provides convenience operators so that
// it can be used in much the same way as a single scalar value.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_SCALABLESIZE_H
#define LLVM_SUPPORT_SCALABLESIZE_H

namespace llvm {

class ElementCount {
public:
  unsigned Min;  // Minimum number of vector elements.
  bool Scalable; // If true, NumElements is a multiple of 'Min' determined
                 // at runtime rather than compile time.

  ElementCount(unsigned Min, bool Scalable)
  : Min(Min), Scalable(Scalable) {}

  ElementCount operator*(unsigned RHS) {
    return { Min * RHS, Scalable };
  }
  ElementCount operator/(unsigned RHS) {
    return { Min / RHS, Scalable };
  }

  bool operator==(const ElementCount& RHS) const {
    return Min == RHS.Min && Scalable == RHS.Scalable;
  }
};

} // end namespace llvm

#endif // LLVM_SUPPORT_SCALABLESIZE_H
