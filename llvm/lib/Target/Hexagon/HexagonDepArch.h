//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_HEXAGON_HEXAGONDEPARCH_H
#define LLVM_LIB_TARGET_HEXAGON_HEXAGONDEPARCH_H

#include "llvm/ADT/StringSwitch.h"
#include <optional>

namespace llvm {
namespace Hexagon {
enum class ArchEnum {
  NoArch,
  Generic,
  V5,
  V55,
  V60,
  V62,
  V65,
  V66,
  V67,
  V68,
  V69,
  V71,
  V73,
  V75,
  V79
};

} // namespace Hexagon
} // namespace llvm

#endif // LLVM_LIB_TARGET_HEXAGON_HEXAGONDEPARCH_H
