//===- llvm/Support/ErrorExtras.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_ERROREXTRAS_H
#define LLVM_SUPPORT_ERROREXTRAS_H

#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"

namespace llvm {

// LLVM formatv versions of llvm::createStringError

template <typename... Ts>
inline Error createStringErrorV(std::error_code EC, char const *Fmt,
                                Ts &&...Vals) {
  return make_error<StringError>(formatv(Fmt, std::forward<Ts>(Vals)...).str(),
                                 EC, true);
}

template <typename... Ts>
inline Error createStringErrorV(char const *Fmt, Ts &&...Vals) {
  return createStringErrorV(llvm::inconvertibleErrorCode(), Fmt,
                            std::forward<Ts>(Vals)...);
}

} // namespace llvm

#endif // LLVM_SUPPORT_ERROREXTRAS_H
