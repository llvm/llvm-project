//===- Chipset.h - AMDGPU Chipset version struct ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_AMDGPU_UTILS_CHIPSET_H_
#define MLIR_DIALECT_AMDGPU_UTILS_CHIPSET_H_

#include "mlir/Support/LLVM.h"
#include <tuple>

namespace mlir::amdgpu {

/// Represents the amdgpu gfx chipset version, e.g., gfx90a, gfx942, gfx1103.
/// Note that the leading digits form a decimal number, while the last two
/// digits form a hexadecimal number. For example:
///   gfx942  --> major = 9, minor = 0x4, stepping = 0x2
///   gfx90a  --> major = 9, minor = 0x0, stepping = 0xa
///   gfx1103 --> major = 11, minor = 0x0, stepping = 0x3
struct Chipset {
  unsigned majorVersion = 0;    // The major version (decimal).
  unsigned minorVersion = 0;    // The minor version (hexadecimal).
  unsigned steppingVersion = 0; // The stepping version (hexadecimal).

  constexpr Chipset() = default;
  constexpr Chipset(unsigned major, unsigned minor, unsigned stepping)
      : majorVersion(major), minorVersion(minor), steppingVersion(stepping) {};

  /// Parses the chipset version string and returns the chipset on success, and
  /// failure otherwise.
  static FailureOr<Chipset> parse(StringRef name);

  std::tuple<unsigned, unsigned, unsigned> asTuple() const {
    return {majorVersion, minorVersion, steppingVersion};
  }

#define DEFINE_COMP_OPERATOR(OPERATOR)                                         \
  friend bool operator OPERATOR(const Chipset &lhs, const Chipset &rhs) {      \
    return lhs.asTuple() OPERATOR rhs.asTuple();                               \
  }
  DEFINE_COMP_OPERATOR(==)
  DEFINE_COMP_OPERATOR(!=)
  DEFINE_COMP_OPERATOR(<)
  DEFINE_COMP_OPERATOR(<=)
  DEFINE_COMP_OPERATOR(>)
  DEFINE_COMP_OPERATOR(>=)
#undef DEFINE_COMP_OPERATOR
};

inline bool hasOcpFp8(const Chipset &chipset) {
  return (chipset.majorVersion == 9 && chipset.minorVersion >= 5) ||
         chipset.majorVersion >= 12;
}

/// Predicates mirroring the LLVM AMDGPU `HasDot{N}Insts` features that gate
/// the `v_dot*` instructions.
inline bool hasDot1Insts(const Chipset &chipset) {
  if (chipset.majorVersion == 9)
    return chipset >= Chipset(9, 0, 6);
  if (chipset.majorVersion == 10) {
    if (chipset.minorVersion == 1)
      return chipset.steppingVersion == 1u || chipset.steppingVersion == 2u;
    return chipset.minorVersion >= 3u;
  }
  return false;
}

inline bool hasDot2Insts(const Chipset &chipset) {
  return hasDot1Insts(chipset);
}

inline bool hasDot7Insts(const Chipset &chipset) {
  return chipset.majorVersion >= 11 || hasDot1Insts(chipset);
}

inline bool hasDot8Insts(const Chipset &chipset) {
  return chipset.majorVersion >= 11;
}

inline bool hasDot9Insts(const Chipset &chipset) {
  if (chipset.majorVersion == 11)
    return true;
  return chipset.majorVersion == 12 && chipset.minorVersion == 0;
}

inline bool hasDot10Insts(const Chipset &chipset) {
  if (chipset.majorVersion == 11)
    return true;
  if (chipset.majorVersion == 12)
    return chipset.minorVersion == 0;
  return hasDot1Insts(chipset);
}

inline bool hasDot11Insts(const Chipset &chipset) {
  if (chipset.majorVersion == 11)
    return chipset.minorVersion == 7u;
  return chipset.majorVersion == 12 && chipset.minorVersion == 0;
}

inline bool hasDot12Insts(const Chipset &chipset) {
  if (chipset == Chipset(9, 5, 0))
    return true;
  if (chipset.majorVersion == 11)
    return true;
  return chipset.majorVersion == 12 && chipset.minorVersion == 0;
}

} // namespace mlir::amdgpu

#endif
