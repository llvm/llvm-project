//===-- OMPVersion.h - OpenMP version definition ------------------ C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the core set of OpenMP definitions and declarations.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FRONTEND_OPENMP_OMPVERSION_H
#define LLVM_FRONTEND_OPENMP_OMPVERSION_H

#include "llvm/ADT/DenseMapInfo.h"

namespace llvm {
namespace omp {
struct Version {
  using value_type = unsigned;
  constexpr explicit Version(value_type Ver = 0) : V(Ver) {}
  constexpr explicit operator value_type() const { return V; }
  constexpr explicit operator bool() const { return V != 0; }

  friend constexpr bool operator<(Version A, Version B);
  friend constexpr bool operator==(Version A, Version B);

private:
  value_type V;
};

inline constexpr bool operator==(Version A, Version B) { return A.V == B.V; }
inline constexpr bool operator!=(Version A, Version B) { return !(A == B); }
inline constexpr bool operator<(Version A, Version B) { return A.V < B.V; }
inline constexpr bool operator<=(Version A, Version B) {
  return A < B || A == B;
}
inline constexpr bool operator>(Version A, Version B) { return !(A <= B); }
inline constexpr bool operator>=(Version A, Version B) { return !(A < B); }

inline constexpr bool operator==(Version A, int B) { return A == Version(B); }
inline constexpr bool operator!=(Version A, int B) { return A != Version(B); }
inline constexpr bool operator<(Version A, int B) { return A < Version(B); }
inline constexpr bool operator<=(Version A, int B) { return A <= Version(B); }
inline constexpr bool operator>(Version A, int B) { return A > Version(B); }
inline constexpr bool operator>=(Version A, int B) { return A >= Version(B); }
} // namespace omp

template <> struct DenseMapInfo<omp::Version> {
  static unsigned getHashValue(omp::Version V) {
    using UnderlyingTy = omp::Version::value_type;
    return DenseMapInfo<UnderlyingTy>::getHashValue(
        static_cast<UnderlyingTy>(V));
  }
  static bool isEqual(omp::Version LHS, omp::Version RHS) { return LHS == RHS; }
};
} // namespace llvm

#endif // LLVM_FRONTEND_OPENMP_OMPVERSION_H
