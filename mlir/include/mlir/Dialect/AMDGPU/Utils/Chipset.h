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
#include <utility>

namespace mlir::amdgpu {

/// Represents the amdgpu gfx chipset version, e.g., gfx90a, gfx942, gfx1103.
/// Note that the leading digits form a decimal number, while the last two
/// digits for a hexadecimal number. For example:
///   gfx942  --> major = 9, minor = 0x42
///   gfx90a  --> major = 9, minor = 0xa
///   gfx1103 --> major = 10, minor = 0x3
struct Chipset {
  Chipset() = default;
  Chipset(unsigned majorVersion, unsigned minorVersion)
      : majorVersion(majorVersion), minorVersion(minorVersion){};

  /// Parses the chipset version string and returns the chipset on success, and
  /// failure otherwise.
  static FailureOr<Chipset> parse(StringRef name);

  friend bool operator==(const Chipset &lhs, const Chipset &rhs) {
    return lhs.majorVersion == rhs.majorVersion &&
           lhs.minorVersion == rhs.minorVersion;
  }
  friend bool operator!=(const Chipset &lhs, const Chipset &rhs) {
    return !(lhs == rhs);
  }
  friend bool operator<(const Chipset &lhs, const Chipset &rhs) {
    return std::make_pair(lhs.majorVersion, lhs.minorVersion) <
           std::make_pair(rhs.majorVersion, rhs.minorVersion);
  }

  unsigned majorVersion = 0; // The major version (decimal).
  unsigned minorVersion = 0; // The minor version (hexadecimal).
};

} // namespace mlir::amdgpu

#endif
