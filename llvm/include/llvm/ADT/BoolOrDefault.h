//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_BOOLORDEFAULT_H
#define LLVM_ADT_BOOLORDEFAULT_H

#include <cstdint>

namespace llvm {

// A bool, or Default when the caller's default applies. Unlike
// std::optional<bool>, it does not convert to bool.
enum class BoolOrDefault : uint8_t { Default, False, True };

// Like std::optional<bool>::value_or.
constexpr bool valueOr(BoolOrDefault X, bool Default) {
  return X == BoolOrDefault::Default ? Default : X == BoolOrDefault::True;
}

} // namespace llvm

#endif // LLVM_ADT_BOOLORDEFAULT_H
