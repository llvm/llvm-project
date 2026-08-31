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
#include "llvm/Support/Compiler.h"
#include <tuple>

namespace mlir::amdgpu {

/// Represents the amdgpu gfx chipset version, e.g., gfx90a, gfx942, gfx1103.
/// Note that the leading digits form a decimal number, while the last two
/// digits form a hexadecimal number. For example:
///   gfx942  --> major = 9, minor = 0x4, stepping = 0x2
///   gfx90a  --> major = 9, minor = 0x0, stepping = 0xa
///   gfx1103 --> major = 11, minor = 0x0, stepping = 0x3
///
/// \deprecated Use `mlir::ROCDL::TargetInfo` instead, and rely on target
/// features rather than about version numbers. Will be removed after one
/// release.
struct Chipset {
  unsigned majorVersion = 0;    // The major version (decimal).
  unsigned minorVersion = 0;    // The minor version (hexadecimal).
  unsigned steppingVersion = 0; // The stepping version (hexadecimal).

  constexpr Chipset() = default;
  constexpr Chipset(unsigned major, unsigned minor, unsigned stepping)
      : majorVersion(major), minorVersion(minor), steppingVersion(stepping) {};

  /// Parses the chipset version string and returns the chipset on success, and
  /// failure otherwise.
  ///
  /// \deprecated Use `ROCDL::TargetInfo::get`.
  LLVM_DEPRECATED("use ROCDL::TargetInfo::get instead", "")
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

/// \deprecated Test `llvm::AMDGPU::FEAT_OCP_FP8_CONVERSION_INSTS` on a
/// `ROCDL::TargetInfo` instead. This misses gfx11.7, which does have the OCP
/// fp8 conversions.
LLVM_DEPRECATED("test FEAT_OCP_FP8_CONVERSION_INSTS on a ROCDL::TargetInfo", "")
inline bool hasOcpFp8(const Chipset &chipset) {
  return (chipset.majorVersion == 9 && chipset.minorVersion >= 5) ||
         chipset.majorVersion >= 12;
}

} // namespace mlir::amdgpu

#endif
