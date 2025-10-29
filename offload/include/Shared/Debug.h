//===-- Shared/Debug.h - Target independent OpenMP target RTL -- C++ ------===//
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

#ifndef OMPTARGET_SHARED_DEBUG_H
#define OMPTARGET_SHARED_DEBUG_H

#include <atomic>
#include <mutex>
#include <string>

#include "llvm/Support/circular_raw_ostream.h"

/// 32-Bit field data attributes controlling information presented to the user.
enum OpenMPInfoType : uint32_t {
  // Print data arguments and attributes upon entering an OpenMP device kernel.
  OMP_INFOTYPE_KERNEL_ARGS = 0x0001,
  // Indicate when an address already exists in the device mapping table.
  OMP_INFOTYPE_MAPPING_EXISTS = 0x0002,
  // Dump the contents of the device pointer map at kernel exit or failure.
  OMP_INFOTYPE_DUMP_TABLE = 0x0004,
  // Indicate when an address is added to the device mapping table.
  OMP_INFOTYPE_MAPPING_CHANGED = 0x0008,
  // Print kernel information from target device plugins.
  OMP_INFOTYPE_PLUGIN_KERNEL = 0x0010,
  // Print whenever data is transferred to the device
  OMP_INFOTYPE_DATA_TRANSFER = 0x0020,
  // Print whenever data does not have a viable device counterpart.
  OMP_INFOTYPE_EMPTY_MAPPING = 0x0040,
  // Enable every flag.
  OMP_INFOTYPE_ALL = 0xffffffff,
};

inline std::atomic<uint32_t> &getInfoLevelInternal() {
  static std::atomic<uint32_t> InfoLevel;
  static std::once_flag Flag{};
  std::call_once(Flag, []() {
    if (char *EnvStr = getenv("LIBOMPTARGET_INFO"))
      InfoLevel.store(std::stoi(EnvStr));
  });

  return InfoLevel;
}

inline uint32_t getInfoLevel() { return getInfoLevelInternal().load(); }

inline uint32_t getDebugLevel() {
  static uint32_t DebugLevel = 0;
  static std::once_flag Flag{};
  std::call_once(Flag, []() {
    if (char *EnvStr = getenv("LIBOMPTARGET_DEBUG"))
      DebugLevel = std::stoi(EnvStr);
  });

  return DebugLevel;
}

#undef USED
#undef GCC_VERSION

#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif
#include <inttypes.h>
#undef __STDC_FORMAT_MACROS

#define DPxMOD "0x%0*" PRIxPTR
#define DPxPTR(ptr) ((int)(2 * sizeof(uintptr_t))), ((uintptr_t)(ptr))
#define GETNAME2(name) #name
#define GETNAME(name) GETNAME2(name)

/// Print a generic message string from libomptarget or a plugin RTL
#define MESSAGE0(_str)                                                         \
  do {                                                                         \
    fprintf(stderr, GETNAME(TARGET_NAME) " message: %s\n", _str);              \
  } while (0)

/// Print a printf formatting string message from libomptarget or a plugin RTL
#define MESSAGE(_str, ...)                                                     \
  do {                                                                         \
    fprintf(stderr, GETNAME(TARGET_NAME) " message: " _str "\n", __VA_ARGS__); \
  } while (0)

/// Print fatal error message with an error string and error identifier
#define FATAL_MESSAGE0(_num, _str)                                             \
  do {                                                                         \
    fprintf(stderr, GETNAME(TARGET_NAME) " fatal error %d: %s\n", (int)_num,   \
            _str);                                                             \
    abort();                                                                   \
  } while (0)

/// Print fatal error message with a printf string and error identifier
#define FATAL_MESSAGE(_num, _str, ...)                                         \
  do {                                                                         \
    fprintf(stderr, GETNAME(TARGET_NAME) " fatal error %d: " _str "\n",        \
            (int)_num, __VA_ARGS__);                                           \
    abort();                                                                   \
  } while (0)

/// Print a generic error string from libomptarget or a plugin RTL
#define FAILURE_MESSAGE(...)                                                   \
  do {                                                                         \
    fprintf(stderr, GETNAME(TARGET_NAME) " error: ");                          \
    fprintf(stderr, __VA_ARGS__);                                              \
  } while (0)

/// Print a generic information string used if LIBOMPTARGET_INFO=1
#define INFO_MESSAGE(_num, ...) INFO_MESSAGE_TO(stderr, _num, __VA_ARGS__)

#define INFO_MESSAGE_TO(_stdDst, _num, ...)                                    \
  do {                                                                         \
    fprintf(_stdDst, GETNAME(TARGET_NAME) " device %d info: ", (int)_num);     \
    fprintf(_stdDst, __VA_ARGS__);                                             \
  } while (0)

// Debugging messages
#ifdef OMPTARGET_DEBUG
#include <stdio.h>

#define DEBUGP(prefix, ...)                                                    \
  {                                                                            \
    fprintf(stderr, "%s --> ", prefix);                                        \
    fprintf(stderr, __VA_ARGS__);                                              \
  }

/// Emit a message for debugging
#define DP(...)                                                                \
  do {                                                                         \
    if (getDebugLevel() > 0) {                                                 \
      DEBUGP(DEBUG_PREFIX, __VA_ARGS__);                                       \
    }                                                                          \
  } while (false)

/// Emit a message for debugging or failure if debugging is disabled
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

/// Emit a message giving the user extra information about the runtime if
#define INFO(_flags, _id, ...)                                                 \
  do {                                                                         \
    if (getDebugLevel() > 0) {                                                 \
      DEBUGP(DEBUG_PREFIX, __VA_ARGS__);                                       \
    } else if (getInfoLevel() & _flags) {                                      \
      INFO_MESSAGE(_id, __VA_ARGS__);                                          \
    }                                                                          \
  } while (false)

#define DUMP_INFO(toStdOut, _flags, _id, ...)                                  \
  do {                                                                         \
    if (toStdOut) {                                                            \
      INFO_MESSAGE_TO(stdout, _id, __VA_ARGS__);                               \
    } else {                                                                   \
      INFO(_flags, _id, __VA_ARGS__);                                          \
    }                                                                          \
  } while (false)

// New macros that will allow for more granular control over debugging output
// Each message can be classified by Component, Type and Level
// Component: The broad component of the offload runtime emitting the message.
// Type: A cross-component classification of messages
// Level: The verbosity level of the message
//
// The component is pulled from the TARGET_NAME macro, Type and Level can be
// defined for each debug message but by default they are "default" and "1"
// respectively.
//
// For liboffload and plugins, use OFFLOAD_DEBUG(...)
// For libomptarget, use OPENMP_DEBUG(...)
// Constructing messages should be done using C++ stream style syntax.
//
// Usage examples:
// OFFLOAD_DEBUG("type1", 2, "This is a level 2 message of type1");
// OFFLOAD_DEBUG("Init", "This is a default level of the init type");
// OPENMP_DEBUG("This is a level 1 message of the default type");
// OFFLOAD_DEBUG("Init", 3, NumDevices << " were initialized\n");
// OFFLOAD_DEBUG("Kernel", "Starting kernel " << KernelName << " on device " <<
//               DeviceId);
//
// Message output can be controlled by setting LIBOMPTARGET_DEBUG or
// LIBOFFLOAD_DEBUG environment variables. Their syntax is as follows:
// [integer]|all|<type1>[:<level1>][,<type2>[:<level2>],...]
//
// 0 : Disable all debug messages
// all : Enable all level 1 debug messages
// integer : Set the default level for all messages
// <type> : Enable only messages of the specified type and level (more than one
//          can be specified). Components are also supported as
//          types.
// <level> : Set the verbosity level for the specified type (default is 1)
//
// Some examples:
// LIBOFFLOAD_DEBUG=1  (Print all messages of level 1 or lower)
// LIBOFFLOAD_DEBUG=5  (Print all messages of level 5 or lower)
// LIBOFFLOAD_DEBUG=init (Print messages of type "init" of level 1 or lower)
// LIBOFFLOAD_DEBUG=init:3,mapping:2 (Print messages of type "init" of level 3
//                                   or lower and messages of type "mapping" of
//                                   level 2 or lower)
// LIBOFFLOAD_DEBUG=omptarget:4, init (Print messages from component "omptarget" of
//                                   level 4 or lower and messages of type
//                                   "init" of level 1 or lower)
//
// For very specific cases where more control is needed, use OFFLOAD_DEBUG_RAW
// or OFFLOAD_DEBUG_BASE. See below for details.

namespace llvm::offload::debug {

#ifdef OMPTARGET_DEBUG

struct DebugFilter {
  StringRef Type;
  uint32_t Level;
};

struct DebugSettings {
  bool Enabled = false;
  uint32_t DefaultLevel = 1;
  llvm::SmallVector<DebugFilter> Filters;
};

/// dbgs - Return a circular-buffered debug stream.
inline llvm::raw_ostream &dbgs() {
  // Do one-time initialization in a thread-safe way.
  static struct dbgstream {
    llvm::circular_raw_ostream strm;

    dbgstream() : strm(llvm::errs(), "*** Debug Log Output ***\n", 0) {}
  } thestrm;

  return thestrm.strm;
}

inline DebugFilter parseDebugFilter(StringRef Filter) {
  size_t Pos = Filter.find(':');
  if (Pos == StringRef::npos)
    return {Filter, 1};

  StringRef Type = Filter.slice(0, Pos);
  uint32_t Level = 1;
  if (Filter.slice(Pos + 1, Filter.size()).getAsInteger(10, Level))
    Level = 1;

  return {Type, Level};
}

inline DebugSettings &getDebugSettings() {
  static DebugSettings Settings;
  static std::once_flag Flag{};
  std::call_once(Flag, []() {
    printf("Configuring debug settings\n");
    // Eventually, we probably should allow the upper layers to set
    // debug settings directly according to their own env var or
    // other methods.
    // For now, mantain compatibility with existing libomptarget env var
    // and add a liboffload independent one.
    char *Env = getenv("LIBOMPTARGET_DEBUG");
    if (!Env) {
      Env = getenv("LIBOFFLOAD_DEBUG");
      if (!Env)
        return;
    }

    StringRef EnvRef(Env);
    if (EnvRef == "0")
      return;

    Settings.Enabled = true;
    if (EnvRef.equals_insensitive("all"))
      return;

    if (!EnvRef.getAsInteger(10, Settings.DefaultLevel))
      return;

    Settings.DefaultLevel = 1;

    SmallVector<StringRef> DbgTypes;
    EnvRef.split(DbgTypes, ',', -1, false);

    for (auto &DT : DbgTypes)
      Settings.Filters.push_back(parseDebugFilter(DT));
  });

  return Settings;
}

inline bool isDebugEnabled() { return getDebugSettings().Enabled; }

inline bool shouldPrintDebug(const char *Component, const char *Type,
                             uint32_t Level) {
  const auto &Settings = getDebugSettings();
  if (!Settings.Enabled)
    return false;

  if (Settings.Filters.empty())
    return Level <= Settings.DefaultLevel;

  for (const auto &DT : Settings.Filters) {
    if (DT.Level < Level)
      continue;
    if (DT.Type.equals_insensitive(Type))
      return true;
    if (DT.Type.equals_insensitive(Component))
      return true;
  }

  return false;
}

#define OFFLOAD_DEBUG_BASE(Component, Type, Level, ...)                        \
  do {                                                                         \
    if (llvm::offload::debug::isDebugEnabled() &&                              \
        llvm::offload::debug::shouldPrintDebug(Component, Type, Level))        \
      __VA_ARGS__;                                                             \
  } while (0)

#define OFFLOAD_DEBUG_RAW(Type, Level, X)                                      \
  OFFLOAD_DEBUG_BASE(GETNAME(TARGET_NAME), Type, Level, X)

#define OFFLOAD_DEBUG_1(X)                                                     \
  OFFLOAD_DEBUG_BASE(GETNAME(TARGET_NAME), "default", 1,                       \
                     llvm::offload::debug::dbgs()                              \
                         << DEBUG_PREFIX << " --> " << X)

#define OFFLOAD_DEBUG_2(Type, X)                                               \
  OFFLOAD_DEBUG_BASE(GETNAME(TARGET_NAME), Type, 1,                            \
                     llvm::offload::debug::dbgs()                              \
                         << DEBUG_PREFIX << " --> " << X)

#define OFFLOAD_DEBUG_3(Type, Level, X)                                        \
  OFFLOAD_DEBUG_BASE(GETNAME(TARGET_NAME), Type, Level,                        \
                     llvm::offload::debug::dbgs()                              \
                         << DEBUG_PREFIX << " --> " << X)

#define OFFLOAD_SELECT(Type, Level, X, NArgs, ...) OFFLOAD_DEBUG_##NArgs

// To be used in liboffload and plugins
#define OFFLOAD_DEBUG(...) OFFLOAD_SELECT(__VA_ARGS__, 3, 2, 1)(__VA_ARGS__)

// To be used in libomptarget only
#define OPENMP_DEBUG(...) OFFLOAD_DEBUG(__VA_ARGS__)

#else

// Don't print anything if debugging is disabled
#define OFFLOAD_DEBUG_BASE(Component, Type, Level, ...)
#define OFFLOAD_DEBUG_RAW(Type, Level, X)
#define OFFLOAD_DEBUG(...)
#define OPENMP_DEBUG(...)

#endif

} // namespace llvm::offload::debug

#endif // OMPTARGET_SHARED_DEBUG_H
