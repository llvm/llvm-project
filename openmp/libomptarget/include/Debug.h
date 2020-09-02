//===------- Debug.h - Target independent OpenMP target RTL -- C++ --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Routines used to provide debug messages and information from libomptarget
// and plugin RTLs to the user.
//
// Each plugin RTL and libomptarget define TARGET_NAME and DEBUG_PREFIX for use
// when sending messages to the user. These indicate which RTL sent the message
//
// Debug and information messages are controlled by the environment variables
// LIBOMPTARGET_DEBUG and LIBOMPTARGET_INFO which is set upon initialization
// of libomptarget or the plugin RTL. 
//
// To printf a pointer in hex with a fixed width of 16 digits and a leading 0x,
// use printf("ptr=" DPxMOD "...\n", DPxPTR(ptr));
// 
// DPxMOD expands to:
//   "0x%0*" PRIxPTR
// where PRIxPTR expands to an appropriate modifier for the type uintptr_t on a
// specific platform, e.g. "lu" if uintptr_t is typedef'd as unsigned long:
//   "0x%0*lu"
// 
// Ultimately, the whole statement expands to:
//   printf("ptr=0x%0*lu...\n",  // the 0* modifier expects an extra argument
//                               // specifying the width of the output
//   (int)(2*sizeof(uintptr_t)), // the extra argument specifying the width
//                               // 8 digits for 32bit systems
//                               // 16 digits for 64bit
//   (uintptr_t) ptr);
//
//===----------------------------------------------------------------------===//
#ifndef _OMPTARGET_DEBUG_H
#define _OMPTARGET_DEBUG_H

static inline int getInfoLevel() {
  static int InfoLevel = -1;
  if (InfoLevel >= 0)
    return InfoLevel;

  if (char *EnvStr = getenv("LIBOMPTARGET_INFO"))
    InfoLevel = std::stoi(EnvStr);

  return InfoLevel;
}

static inline int getDebugLevel() {
  static int DebugLevel = -1;
  if (DebugLevel >= 0)
    return DebugLevel;

  if (char *EnvStr = getenv("LIBOMPTARGET_DEBUG"))
    DebugLevel = std::stoi(EnvStr);

  return DebugLevel;
}

#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif
#include <inttypes.h>
#undef __STDC_FORMAT_MACROS

#define DPxMOD "0x%0*" PRIxPTR
#define DPxPTR(ptr) ((int)(2 * sizeof(uintptr_t))), ((uintptr_t)(ptr))
#define GETNAME2(name) #name
#define GETNAME(name) GETNAME2(name)

// Messaging interface
#define MESSAGE0(_str)                                                         \
  do {                                                                         \
    fprintf(stderr, GETNAME(TARGET_NAME) " message: %s\n", _str);              \
  } while (0)

#define MESSAGE(_str, ...)                                                     \
  do {                                                                         \
    fprintf(stderr, GETNAME(TARGET_NAME) " message: " _str "\n", __VA_ARGS__); \
  } while (0)

#define FATAL_MESSAGE0(_num, _str)                                             \
  do {                                                                         \
    fprintf(stderr, GETNAME(TARGET_NAME) " fatal error %d: %s\n", _num, _str); \
    abort();                                                                   \
  } while (0)

#define FATAL_MESSAGE(_num, _str, ...)                                         \
  do {                                                                         \
    fprintf(stderr, GETNAME(TARGET_NAME) " fatal error %d:" _str "\n", _num,   \
            __VA_ARGS__);                                                      \
    abort();                                                                   \
  } while (0)

#define FAILURE_MESSAGE(...)                                                   \
  do {                                                                         \
    fprintf(stderr, GETNAME(TARGET_NAME) " error: ");                          \
    fprintf(stderr, __VA_ARGS__);                                              \
  } while (0)

// Debugging messages
#ifdef OMPTARGET_DEBUG
#include <stdio.h>

#define DEBUGP(prefix, ...)                                                    \
  {                                                                            \
    fprintf(stderr, "%s --> ", prefix);                                        \
    fprintf(stderr, __VA_ARGS__);                                              \
  }

#define DP(...)                                                                \
  do {                                                                         \
    if (getDebugLevel() > 0) {                                                 \
      DEBUGP(DEBUG_PREFIX, __VA_ARGS__);                                       \
    }                                                                          \
  } while (false)

#define REPORT(...)                                                            \
  do {                                                                         \
    if (getDebugLevel() > 0) {                                                 \
      DP(__VA_ARGS__);                                                         \
    } else {                                                                   \
      FAILURE_MESSAGE(__VA_ARGS__);                                            \
    }                                                                          \
  } while (false)
#else
#define DEBUGP(prefix, ...)                                                    \
  {}
#define DP(...)                                                                \
  {}
#define REPORT(...) FAILURE_MESSAGE(__VA_ARGS__);
#endif // OMPTARGET_DEBUG

#endif // _OMPTARGET_DEBUG_H
