//===- Environment.cpp - ORC-RT executor environment access ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of the environment helpers declared in Environment.h.
//
//===----------------------------------------------------------------------===//

// Must precede any include so <stdlib.h> exposes glibc's secure_getenv.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "Environment.h"

#include <stdlib.h>

#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) ||      \
    defined(__NetBSD__) || defined(__DragonFly__)
#include <unistd.h> // for issetugid
#endif

namespace orc_rt {

const char *secureGetenv(const char *Name) {
#if defined(__GLIBC__)
  // secure_getenv returns null in "secure execution" mode, which the kernel
  // sets for set-user-ID / set-group-ID programs (and other privilege
  // transitions).
  return ::secure_getenv(Name);
#elif defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) ||    \
    defined(__NetBSD__) || defined(__DragonFly__)
  // No secure_getenv; issetugid() reports whether the process was made
  // set-user-ID / set-group-ID (or otherwise had its ids changed).
  return ::issetugid() ? nullptr : ::getenv(Name);
#else
  // We cannot verify here that the environment is trustworthy, so fail secure:
  // refuse to read the variable rather than risk honoring an attacker-supplied
  // value in a privileged process.
  //
  // TODO: Add branches for other libcs/platforms as needed (e.g. musl's
  // secure_getenv, Windows), so their environment variables aren't ignored.
  (void)Name;
  return nullptr;
#endif
}

} // namespace orc_rt
