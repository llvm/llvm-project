/*===------------ Logging.h - ORC Runtime logging support ---------*- C -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* Configurable logging for the ORC runtime.                                  *|
|*                                                                            *|
|* Log a message with:                                                        *|
|*                                                                            *|
|*   ORC_RT_LOG(Level, Category, Fmt, ...)                                    *|
|*                                                                            *|
|* where Level is one of Error, Warning, Info, Debug; Category is one of the  *|
|* orc_rt_log_Category tokens (without the leading "orc_rt_log_Category_");   *|
|* and Fmt, ... are a printf-style format string and arguments, e.g.          *|
|*                                                                            *|
|*   ORC_RT_LOG(Error, General, "failed to map %zu bytes", Size);             *|
|*                                                                            *|
|* Level and Category must be compile-time tokens (not runtime values): the   *|
|* os_log backend selects its record type from the level at compile time, and *|
|* the category enum gives typo-safe, filterable subsystems.                  *|
|*                                                                            *|
|* The backend is selected at build time via ORC_RT_LOG_BACKEND. The default  *|
|* "none" backend emits no code (but still type-checks the format string and  *|
|* arguments at every call site).                                             *|
|*                                                                            *|
|* IMPORTANT: log arguments should be free of observable side effects.        *|
|* Whether they are evaluated is backend- and level-dependent, so they must   *|
|* not be relied upon to run.                                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef ORC_RT_C_LOGGING_H
#define ORC_RT_C_LOGGING_H

#include "orc-rt-c/Compiler.h"
#include "orc-rt-c/config.h"

ORC_RT_C_EXTERN_C_BEGIN

/**
 * Logging categories.
 *
 * Each category identifies a runtime subsystem. On the os_log backend a
 * category maps to an os_log category (filterable in Console / the `log`
 * tool); on the printf backend it appears in the line prefix. Add new
 * categories here as call sites require them.
 */
typedef enum {
  orc_rt_log_Category_General,

  /*
   * Count is the number of defined categories; it is not itself a valid
   * category.
   */
  orc_rt_log_Category_Count
} orc_rt_log_Category;

/**
 * Logging levels are integers. Valid values are ORC_RT_LOG_LEVEL_DEBUG,
 * ORC_RT_LOG_LEVEL_INFO, ORC_RT_LOG_LEVEL_WARNING, ORC_RT_LOG_LEVEL_ERROR, and
 * ORC_RT_LOG_LEVEL_OFF.
 */
typedef int orc_rt_log_Level;

/**
 * Returns the display name for the given category, or null if the category is
 * unrecognized.
 */
const char *
orc_rt_log_Category_getName(orc_rt_log_Category Cat) ORC_RT_C_NOTHROW;

/**
 * Returns the display name for the given log level, or null if the log level
 * is unrecognized.
 */
const char *orc_rt_log_Level_getName(orc_rt_log_Level L) ORC_RT_C_NOTHROW;

/**
 * Returns the level corresponding to the given level name, or -1 if the level
 * name is unrecognized.
 *
 * Comparison is case-insensitive.
 */
orc_rt_log_Level orc_rt_log_Level_parse(const char *Str) ORC_RT_C_NOTHROW;

/**
 * Declared but never defined: referenced only in unevaluated (sizeof) contexts
 * to type-check a printf-style format string against its arguments without
 * evaluating them or emitting any code.
 */
int orc_rt_log_formatCheck(const char *Fmt, ...) ORC_RT_C_FORMAT_PRINTF(1, 2);

/*
 * A disabled log site. Validates the category token, format string, and
 * arguments at compile time, but evaluates nothing and generates no code:
 * both operands sit in unevaluated sizeof contexts.
 */
#define ORC_RT_LOG_DISABLED(Category, ...)                                     \
  ((void)sizeof(Category), (void)sizeof(orc_rt_log_formatCheck(__VA_ARGS__)))

/*
 * ORC_RT_LOG dispatches on the level token to a per-level macro, which each
 * backend defines below to either emit a message or, when the level is
 * compiled out, to ORC_RT_LOG_DISABLED. Fmt is the first variadic argument, so
 * __VA_ARGS__ is always non-empty and no GNU comma-swallowing extension is
 * needed.
 */
#define ORC_RT_LOG(Level, Category, ...)                                       \
  ORC_RT_LOG_##Level(orc_rt_log_Category_##Category, __VA_ARGS__)

#if ORC_RT_LOG_BACKEND == ORC_RT_LOG_BACKEND_NONE

/* The none backend compiles every level out, regardless of ORC_RT_LOG_LEVEL. */
#define ORC_RT_LOG_Error(Category, ...)                                        \
  ORC_RT_LOG_DISABLED(Category, __VA_ARGS__)
#define ORC_RT_LOG_Warning(Category, ...)                                      \
  ORC_RT_LOG_DISABLED(Category, __VA_ARGS__)
#define ORC_RT_LOG_Info(Category, ...)                                         \
  ORC_RT_LOG_DISABLED(Category, __VA_ARGS__)
#define ORC_RT_LOG_Debug(Category, ...)                                        \
  ORC_RT_LOG_DISABLED(Category, __VA_ARGS__)

#elif ORC_RT_LOG_BACKEND == ORC_RT_LOG_BACKEND_PRINTF
#error "The printf logging backend is not yet implemented."
#elif ORC_RT_LOG_BACKEND == ORC_RT_LOG_BACKEND_OS_LOG
#error "The os_log logging backend is not yet implemented."
#else
#error "Unknown ORC_RT_LOG_BACKEND."
#endif

ORC_RT_C_EXTERN_C_END

#endif /* ORC_RT_C_LOGGING_H */
