/*===------ LogLevel.h - ORC Runtime compiled-in log levels -------*- C -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* ORC_RT_LOG_ENABLED, kept apart from Logging.h so that it can be used       *|
|* without pulling in the logging backend's headers (e.g. <os/log.h>).        *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef ORC_RT_C_SUPPORT_LOGLEVEL_H
#define ORC_RT_C_SUPPORT_LOGLEVEL_H

#include "orc-rt-c/config.h"

/**
 * ORC_RT_LOG_ENABLED(Level) is 1 if log sites at Level (Error, Warning, Info or
 * Debug, as for ORC_RT_LOG) are compiled in, and 0 otherwise. Usable in #if.
 */
#define ORC_RT_LOG_ENABLED(Level)                                              \
  (ORC_RT_LOG_BACKEND != ORC_RT_LOG_BACKEND_NONE &&                            \
   ORC_RT_LOG_ENABLED_##Level() >= ORC_RT_LOG_LEVEL)

/*
 * Per-level macros are function-like so that a mistyped level is a hard error
 * rather than silently evaluating to 0.
 */
#define ORC_RT_LOG_ENABLED_Debug() ORC_RT_LOG_LEVEL_DEBUG
#define ORC_RT_LOG_ENABLED_Info() ORC_RT_LOG_LEVEL_INFO
#define ORC_RT_LOG_ENABLED_Warning() ORC_RT_LOG_LEVEL_WARNING
#define ORC_RT_LOG_ENABLED_Error() ORC_RT_LOG_LEVEL_ERROR

#endif /* ORC_RT_C_SUPPORT_LOGLEVEL_H */
