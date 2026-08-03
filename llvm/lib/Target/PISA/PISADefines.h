//===-- PISADefines.h - Common defines for PISA ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_PISA_PISADEFINES_H
#define LLVM_LIB_TARGET_PISA_PISADEFINES_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>

namespace llvm {
namespace PISA {

enum class Swizzle : unsigned { X, Y, Z, W, XYZW, XY, ZW, NONE };
constexpr unsigned SimdSize = 32;

// Keep these in sync with PISAInstrInfo.td!
enum class LdCacheCtrl : unsigned {
  Default = 0,
  L1c = 1,
  L1uc = 2,
  L1s = 3,
  L2c = 4,
  L2uc = 5,
  L3c = 6,
  L3uc = 7,
  ri = 8
};

enum class StCacheCtrl : unsigned {
  Default = 0,
  L1uc = 1,
  L1wb = 2,
  L1wt = 3,
  L1s = 4,
  L2uc = 5,
  L2wb = 6,
  L3uc = 7,
  L3wb = 8
};

enum class AtomCacheCtrl : unsigned { Default = 0, uc = 2, L2wb = 4, L3wb = 3 };

} // namespace PISA
} // namespace llvm

#endif // LLVM_LIB_TARGET_PISA_PISADEFINES_H
