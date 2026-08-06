//===- Logging_printf.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of the printf logging backend declared in orc-rt-c/Logging.h.
//
//===----------------------------------------------------------------------===//

#include "orc-rt-c/Logging.h"

#include "Environment.h"

#include <algorithm>
#include <cstdarg>
#include <cstddef>
#include <cstdio>

namespace {

// The output sink, resolved once on first use: stderr, unless ORC_RT_LOG_OUTPUT
// names a file that can be opened for appending.
FILE *sink() {
  static FILE *Sink = []() -> FILE * {
    const char *Path = orc_rt::secureGetenv("ORC_RT_LOG_OUTPUT");
    if (!Path || !*Path)
      return stderr;
    FILE *F = std::fopen(Path, "a");
    if (!F) {
      std::fprintf(stderr,
                   "[orc-rt:General:ERROR] could not open ORC_RT_LOG_OUTPUT "
                   "'%s' for logging; using stderr\n",
                   Path);
      return stderr;
    }
    // Line-buffer: every message ends in a newline, so this flushes each
    // message promptly without the cost of a fully unbuffered stream.
    std::setvbuf(F, nullptr, _IOLBF, 0);
    return F;
  }();
  return Sink;
}

// The runtime level threshold, resolved once on first use from ORC_RT_LOG. A
// message is emitted only if its level is at or above this value. When
// ORC_RT_LOG is unset the threshold is WARNING: warnings and errors are shown
// by default, and info and debug are opt-in. An unrecognized value, or a level
// below the compile-time ORC_RT_LOG_LEVEL floor, falls back to the floor,
// but never more verbose than INFO.
orc_rt_log_Level runtimeLevel() {
  static orc_rt_log_Level Level = []() -> orc_rt_log_Level {
    const char *S = orc_rt::secureGetenv("ORC_RT_LOG");
    if (!S || !*S)
      return ORC_RT_LOG_LEVEL_WARNING;
    orc_rt_log_Level L = orc_rt_log_Level_parse(S);

    // Fallback for an unavailable or unrecognized request: the compile-time
    // floor, but never more verbose than INFO.
    constexpr orc_rt_log_Level Fallback =
        std::max(ORC_RT_LOG_LEVEL, ORC_RT_LOG_LEVEL_INFO);
    if (L < 0) {
      std::fprintf(sink(),
                   "[orc-rt:General:ERROR] ignoring unrecognized ORC_RT_LOG "
                   "value '%s'\n",
                   S);
      return Fallback;
    } else if (L < ORC_RT_LOG_LEVEL) {
      std::fprintf(sink(),
                   "[orc-rt:General:ERROR] requested ORC_RT_LOG = %s is "
                   "unavailable, using lowest available level ORC_RT_LOG = "
                   "%s\n",
                   S, orc_rt_log_Level_getName(Fallback));
      return Fallback;
    }
    return L;
  }();
  return Level;
}

constexpr size_t LogBufferSize = 1024;

} // namespace

extern "C" void orc_rt_log_printf(orc_rt_log_Level Level,
                                  orc_rt_log_Category Category, const char *Fmt,
                                  ...) noexcept {
  if (Level < runtimeLevel())
    return;

  char Buf[LogBufferSize];
  int P = std::snprintf(Buf, sizeof(Buf), "[orc-rt:%s:%s] ",
                        orc_rt_log_Category_getName(Category),
                        orc_rt_log_Level_getName(Level));
  if (P < 0)
    return;
  size_t Len = (size_t)P < sizeof(Buf) ? (size_t)P : sizeof(Buf) - 1;

  va_list Ap;
  va_start(Ap, Fmt);
  int B = std::vsnprintf(Buf + Len, sizeof(Buf) - Len, Fmt, Ap);
  va_end(Ap);
  if (B > 0) {
    Len += (size_t)B;
    if (Len > sizeof(Buf) - 1)
      Len = sizeof(Buf) - 1; // message was truncated to fit the buffer
  }

  // Guarantee a trailing newline, overwriting the last byte when the buffer is
  // full so that adjacent messages never run together.
  if (Len == sizeof(Buf) - 1)
    Buf[Len - 1] = '\n';
  else
    Buf[Len++] = '\n';

  // A single stdio call: stdio locks the stream per call, so the whole line is
  // written atomically with respect to other threads logging to the same sink.
  std::fwrite(Buf, 1, Len, sink());
}
