//===- Chipset.h - AMDGPU Chipset version struct ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_AMDGPU_UTILS_CHIPSET_H_
#define MLIR_DIALECT_AMDGPU_UTILS_CHIPSET_H_

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include <tuple>

namespace mlir::amdgpu {

/// Represents the amdgpu gfx chipset version, e.g., gfx90a, gfx942, gfx1103.
/// Note that the leading digits form a decimal number, while the last two
/// digits for a hexadecimal number. For example:
///   gfx942  --> major = 9, minor = 0x4, stepping = 0x2
///   gfx90a  --> major = 9, minor = 0x0, stepping = 0xa
///   gfx1103 --> major = 10, minor = 0x0, stepping = 0x3
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

inline int32_t getGlobalPrefetchLLVMEncoding(amdgpu::LoadTemporalHint hint,
                                             amdgpu::Scope scope,
                                             bool isSpeculative) {
  int32_t immArg = static_cast<int32_t>(hint);

  // Note that only RT and HT can operate in both speculative and
  // non-speculative modes. The other variants (NT_RT, RT_NT, NT_HT, etc.)
  // operate only in the speculative mode and, therefore, do not require
  // toggling the least significant bit for mode changes
  // Temporal hint is encoded in lower bits - i.e. [2:0]
  if (llvm::is_contained({LoadTemporalHint::RT, LoadTemporalHint::HT}, hint))
    immArg = isSpeculative ? immArg : immArg | 1;

  // Prefetch scope level is encoded in upper bits - i.e., [4:3]
  return static_cast<int32_t>(scope) << 3 | immArg;
}

} // namespace mlir::amdgpu

#endif
