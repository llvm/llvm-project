//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FRONTEND_DIRECTIVE_VERSION_H
#define LLVM_FRONTEND_DIRECTIVE_VERSION_H

#include "llvm/ADT/DenseMapInfo.h"

#include <limits>

namespace llvm {
namespace directive {
struct Version {
  using value_type = unsigned;
  constexpr explicit Version(value_type Ver = 0) : V(Ver) {}
  constexpr explicit operator bool() const { return V != 0; }
  constexpr value_type value() const { return V; }

  friend constexpr bool operator<(Version A, Version B);
  friend constexpr bool operator==(Version A, Version B);

  static constexpr Version Max() {
    // The version values in .td files are "int", and the max value that
    // can be specified there is 0x7fffffff.
    return Version(std::numeric_limits<int>::max());
  }

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

struct VersionRange {
  Version Min = Version(0);
  Version Max = Version::Max();

  constexpr bool operator<(const VersionRange &R) const {
    return std::tie(Min, Max) < std::tie(R.Min, R.Max);
  }
  constexpr bool isValid() const { return Min <= Max; }
  constexpr bool contains(Version V) const { return Min <= V && V <= Max; }
};
} // namespace directive

template <> struct DenseMapInfo<directive::Version> {
  static unsigned getHashValue(directive::Version V) {
    using UnderlyingTy = directive::Version::value_type;
    return DenseMapInfo<UnderlyingTy>::getHashValue(V.value());
  }
  static bool isEqual(directive::Version LHS, directive::Version RHS) {
    return LHS == RHS;
  }
};
} // namespace llvm

#endif // LLVM_FRONTEND_DIRECTIVE_VERSION_H
