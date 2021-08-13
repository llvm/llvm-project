//===-- llvm/Support/Signposts.h - Interval debug annotations ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file Some OS's provide profilers that allow applications to provide custom
/// annotations to the profiler. For example, on Xcode 10 and later 'signposts'
/// can be emitted by the application and these will be rendered to the Points
/// of Interest track on the instruments timeline.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_SIGNPOSTS_H
#define LLVM_SUPPORT_SIGNPOSTS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Config/llvm-config.h"
#include <memory>

#if LLVM_SUPPORT_XCODE_SIGNPOSTS
#include <Availability.h>
#include <os/signpost.h>
#endif

#define SIGNPOSTS_AVAILABLE()                                                  \
  __builtin_available(macos 10.14, iOS 12, tvOS 12, watchOS 5, *)

namespace llvm {
class SignpostEmitterImpl;

/// Manages the emission of signposts into the recording method supported by
/// the OS.
class SignpostEmitter {
  std::unique_ptr<SignpostEmitterImpl> Impl;

public:
  SignpostEmitter();
  ~SignpostEmitter();

  bool isEnabled() const;

  /// Begin a signposted interval for a given object.
  void startInterval(const void *O, StringRef Name);

#if LLVM_SUPPORT_XCODE_SIGNPOSTS
  os_log_t &getLogger() const;
  os_signpost_id_t getSignpostForObject(const void *O);
#endif

  /// A macro to take advantage of the special format string handling
  /// in the os_signpost API. The format string substitution is
  /// deferred to the log consumer and done outside of the
  /// application.
#if LLVM_SUPPORT_XCODE_SIGNPOSTS
#define SIGNPOST_EMITTER_START_INTERVAL(SIGNPOST_EMITTER, O, ...)              \
  do {                                                                         \
    if ((SIGNPOST_EMITTER).isEnabled())                                        \
      if (SIGNPOSTS_AVAILABLE())                                               \
        os_signpost_interval_begin((SIGNPOST_EMITTER).getLogger(),             \
                                   (SIGNPOST_EMITTER).getSignpostForObject(O), \
                                   "LLVM Timers", __VA_ARGS__);                \
  } while (0)
#else
#define SIGNPOST_EMITTER_START_INTERVAL(SIGNPOST_EMITTER, O, ...)              \
  do {                                                                         \
  } while (0)
#endif

  /// End a signposted interval for a given object.
  void endInterval(const void *O);
};

} // end namespace llvm

#endif // LLVM_SUPPORT_SIGNPOSTS_H
