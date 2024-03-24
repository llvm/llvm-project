//===- Options.h - LLVM Target Options --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares LLVM target options.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_LLVM_OPTIONS_H
#define MLIR_TARGET_LLVM_OPTIONS_H

#include <cstdint>

namespace mlir {
namespace LLVM {
/// This enumeration represents LLVM bitcode linking flags.
enum class LinkingFlags : uint32_t {
  none = 0,
  overrideFromSrc = 1, /// Override symbols using the source definition.
  onlyNeeded = 2, /// Add only symbols referenced by the destination module.
};

/// LinkingFlags bitwise operators.
inline LinkingFlags operator|(LinkingFlags x, LinkingFlags y) {
  return static_cast<LinkingFlags>(static_cast<uint32_t>(x) |
                                   static_cast<uint32_t>(y));
}
inline LinkingFlags operator&(LinkingFlags x, LinkingFlags y) {
  return static_cast<LinkingFlags>(static_cast<uint32_t>(x) &
                                   static_cast<uint32_t>(y));
}
} // namespace LLVM
} // namespace mlir

#endif // MLIR_TARGET_LLVM_OPTIONS_H
