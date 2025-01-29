//===-PollyDebug.h -Provide support for debugging Polly passes-*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Functions to aid printing Debug Info of all polly passes.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_DEBUG_H
#define POLLY_DEBUG_H

#include "llvm/Support/Debug.h"

namespace polly {
using namespace llvm;
bool getPollyDebugFlag();

#ifndef NDEBUG
#define POLLY_DEBUG(X)                                                         \
  do {                                                                         \
    if (polly::getPollyDebugFlag()) {                                          \
      X;                                                                       \
    } else {                                                                   \
      DEBUG_WITH_TYPE(DEBUG_TYPE, X);                                          \
    }                                                                          \
  } while (0)
#else
#define POLLY_DEBUG(X)                                                         \
  do {                                                                         \
  } while (false)
#endif

} // namespace polly
#endif
