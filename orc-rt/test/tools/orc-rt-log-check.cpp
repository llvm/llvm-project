//===- orc-rt-log-check.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Regression-test helper for the ORC runtime logging backends.
//
// With no arguments it emits one log record at each level (all in the General
// category), so a lit test can check message formatting and level/output
// filtering by running it under different ORC_RT_LOG / ORC_RT_LOG_OUTPUT
// environments.
//
// The query options let the lit config gate tests on the build's logging
// configuration:
//   --print-backend         print the compiled-in backend (none/printf/os_log).
//   --print-enabled-levels  print, one per line, the levels that are actually
//                           compiled in (empty for the none backend). This
//                           reflects both the ORC_RT_LOG_LEVEL floor and the
//                           backend, i.e. exactly which ORC_RT_LOG calls emit.
//
//===----------------------------------------------------------------------===//

#include "orc-rt-c/Logging.h"

#include "orc-rt-utils/CommandLine.h"

#include <iostream>

namespace {

const char *getLoggingBackend() {
#if ORC_RT_LOG_BACKEND == ORC_RT_LOG_BACKEND_NONE
  return "none";
#elif ORC_RT_LOG_BACKEND == ORC_RT_LOG_BACKEND_PRINTF
  return "printf";
#elif ORC_RT_LOG_BACKEND == ORC_RT_LOG_BACKEND_OS_LOG
  return "os_log";
#else
#error "Unrecognized ORC_RT_LOG_BACKEND"
#endif
}

void printEnabledLevels() {
#if ORC_RT_LOG_BACKEND != ORC_RT_LOG_BACKEND_NONE
  // Stop before OFF: it is the disable sentinel, not an emittable level.
  for (orc_rt_log_Level L = ORC_RT_LOG_LEVEL; L != ORC_RT_LOG_LEVEL_OFF; ++L)
    std::cout << orc_rt_log_Level_getName(L) << "\n";
#endif
}

void emitAllLevels() {
  ORC_RT_LOG(Error, General, "error message");
  ORC_RT_LOG(Warning, General, "warning message");
  ORC_RT_LOG(Info, General, "info message");
  ORC_RT_LOG(Debug, General, "debug message");
}

} // namespace

int main(int argc, char *argv[]) {

  bool PrintBackend = false;
  bool PrintEnabledLevels = false;
  bool PrintHelp = false;
  int UID = -1;

  {
    orc_rt::CommandLineParser P;
    P.addFlag("print-backend", "Print log backend", false, PrintBackend)
        .addFlag("print-enabled-levels", "Print enabled log levels", false,
                 PrintEnabledLevels)
        .addValue("uid",
                  "Emit one marker record carrying this id (for os_log "
                  "delivery tests)",
                  -1, UID)
        .addFlag("help", "Print help", false, PrintHelp);

    if (auto Err = P.parse(argc, argv)) {
      std::cerr << "error: " << orc_rt::toString(std::move(Err)) << "\n";
      P.printHelp(std::cerr, argv[0]);
      return 1;
    }

    if ((PrintBackend && PrintEnabledLevels) || !P.positionals().empty()) {
      P.printHelp(std::cerr, argv[0]);
      return 1;
    }

    if (PrintHelp) {
      P.printHelp(std::cerr, argv[0]);
      return 0;
    }
  }

  if (PrintBackend)
    std::cout << getLoggingBackend() << "\n";

  if (PrintEnabledLevels)
    printEnabledLevels();

  if (PrintBackend || PrintEnabledLevels)
    return 0;

  if (UID != -1) {
    // Emit one record whose payload carries the caller-supplied id, so a
    // delivery test can match exactly its own record and not a stale one. The
    // id is an integer (a %d scalar), which os_log shows unredacted.
    //
    // The record also carries a runtime string logged via ORC_RT_LOG_PUB_S. On
    // the os_log backend a dynamic string argument is redacted to <private>
    // unless the public annotation is applied, so a delivery test can confirm
    // the annotation publishes the string by matching its contents rather than
    // <private>.
    const char *PublicPayload = "public-payload";
    ORC_RT_LOG(Error, General,
               "delivery marker uid=%d payload=" ORC_RT_LOG_PUB_S, UID,
               PublicPayload);
    return 0;
  }

  emitAllLevels();

  return 0;
}
