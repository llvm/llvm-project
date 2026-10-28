//===- Environment.h - ORC-RT executor environment access -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helpers for reading the process environment.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_INTERNAL_SUPPORT_ENVIRONMENT_H
#define ORC_RT_INTERNAL_SUPPORT_ENVIRONMENT_H

#include <stdlib.h>

#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) ||      \
    defined(__NetBSD__) || defined(__DragonFly__)
#include <unistd.h> // for issetugid
#endif

#if defined(__GLIBC__)
// Declared here rather than by defining _GNU_SOURCE before <stdlib.h>: a header
// cannot set a feature macro reliably, since <stdlib.h> may already have been
// included by the time this one is reached. The signature matches glibc's,
// whose __THROW expands to noexcept in C++.
extern "C" char *secure_getenv(const char *Name) noexcept;
#endif

namespace orc_rt {

/// Like getenv, but returns null when the process is running with elevated
/// privileges (e.g. a set-user-ID or set-group-ID program), so that a variable
/// in an attacker-controlled environment cannot influence a privileged host.
///
/// Use this for any environment variable whose value has security-relevant
/// effects, e.g. choosing a file to open.
///
/// This is header-only so that Support carries no per-system objects: Support
/// is embedded into every library that ships, including the JIT-linked SPIRE,
/// so anything it compiles is code those libraries have to carry.
inline const char *secureGetenv(const char *Name) {
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

#endif // ORC_RT_INTERNAL_SUPPORT_ENVIRONMENT_H
