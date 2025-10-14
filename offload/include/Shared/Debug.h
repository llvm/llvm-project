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
#include <sstream>
#include <string>

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

/// 32-bit field attributes controlling debug trace/dump
enum DebugInfoType : uint32_t {
  /// Generic plugin/runtime interface/management
  DEBUG_INFOTYPE_RTL = 0x0001,
  /// Generic device activity
  DEBUG_INFOTYPE_DEVICE = 0x0002,
  /// Module preparation
  DEBUG_INFOTYPE_MODULE = 0x0004,
  /// Kernel preparation and invocation
  DEBUG_INFOTYPE_KERNEL = 0x0008,
  /// Memory allocation/deallocation or related activities
  DEBUG_INFOTYPE_MEMORY = 0x0010,
  /// Data-mapping activities
  DEBUG_INFOTYPE_MAP = 0x0020,
  /// Data-copying or similar activities
  DEBUG_INFOTYPE_COPY = 0x0040,
  /// OpenMP interop
  DEBUG_INFOTYPE_INTEROP = 0x0080,
  /// Tool interface
  DEBUG_INFOTYPE_TOOL = 0x0100,
  /// Backend API tracing
  DEBUG_INFOTYPE_API = 0x0200,
  /// All
  DEBUG_INFOTYPE_ALL = 0xffffffff,
};

/// Debug option struct to support both numeric and string value
struct DebugOptionTy {
  uint32_t Level;
  uint32_t Type;
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

inline DebugOptionTy &getDebugOption() {
  static DebugOptionTy DebugOption = []() {
    DebugOptionTy OptVal{0, 0};
    char *EnvStr = getenv("LIBOMPTARGET_DEBUG");
    if (!EnvStr || *EnvStr == '0')
      return OptVal; // undefined or explicitly defined as zero
    OptVal.Level = std::atoi(EnvStr);
    if (OptVal.Level)
      return OptVal; // defined as numeric value
    struct DebugStrToBitTy {
      const char *Str;
      uint32_t Bit;
    } DebugStrToBit[] = {
        {"rtl", DEBUG_INFOTYPE_RTL},       {"device", DEBUG_INFOTYPE_DEVICE},
        {"module", DEBUG_INFOTYPE_MODULE}, {"kernel", DEBUG_INFOTYPE_KERNEL},
        {"memory", DEBUG_INFOTYPE_MEMORY}, {"map", DEBUG_INFOTYPE_MAP},
        {"copy", DEBUG_INFOTYPE_COPY},     {"interop", DEBUG_INFOTYPE_INTEROP},
        {"tool", DEBUG_INFOTYPE_TOOL},     {"api", DEBUG_INFOTYPE_API},
        {"all", DEBUG_INFOTYPE_ALL},       {nullptr, 0},
    };
    // Check string value of the option
    std::istringstream Tokens(EnvStr);
    for (std::string Token; std::getline(Tokens, Token, ',');) {
      for (int I = 0; DebugStrToBit[I].Str; I++) {
        if (Token == DebugStrToBit[I].Str) {
          OptVal.Type |= DebugStrToBit[I].Bit;
          break;
        }
      }
    }
    return OptVal;
  }();
  return DebugOption;
}

inline uint32_t getDebugLevel() { return getDebugOption().Level; }
inline uint32_t getDebugType() { return getDebugOption().Type; }

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

/// Check if debug option is turned on for `Type`
#define DPSET(Type)                                                            \
  ((getDebugType() & DEBUG_INFOTYPE_##Type) || getDebugLevel() > 0)

/// Emit a message for debugging if related to `Type`
#define DPIF(Type, ...)                                                        \
  do {                                                                         \
    if (DPSET(Type)) {                                                         \
      DEBUGP(DEBUG_PREFIX, __VA_ARGS__);                                       \
    }                                                                          \
  } while (false)

/// Emit a message for debugging
#define DP(...) DPIF(ALL, __VA_ARGS__);

/// Emit a message for debugging or failure if debugging is disabled
#define REPORT(...)                                                            \
  do {                                                                         \
    if (DPSET(ALL)) {                                                          \
      DP(__VA_ARGS__);                                                         \
    } else {                                                                   \
      FAILURE_MESSAGE(__VA_ARGS__);                                            \
    }                                                                          \
  } while (false)
#else
#define DEBUGP(prefix, ...)                                                    \
  {}
#define DPSET(Type) false
#define DPIF(Type, ...)                                                        \
  {                                                                            \
  }
#define DP(...)                                                                \
  {}
#define REPORT(...) FAILURE_MESSAGE(__VA_ARGS__);
#endif // OMPTARGET_DEBUG

#ifdef OMPTARGET_DEBUG
// Convert `OpenMPInfoType` to corresponding `DebugInfoType`
inline bool debugInfoEnabled(OpenMPInfoType InfoType) {
  switch (InfoType) {
  case OMP_INFOTYPE_KERNEL_ARGS:
    [[fallthrough]];
  case OMP_INFOTYPE_PLUGIN_KERNEL:
    return DPSET(KERNEL);
  case OMP_INFOTYPE_MAPPING_EXISTS:
    [[fallthrough]];
  case OMP_INFOTYPE_DUMP_TABLE:
    [[fallthrough]];
  case OMP_INFOTYPE_MAPPING_CHANGED:
    [[fallthrough]];
  case OMP_INFOTYPE_EMPTY_MAPPING:
    return DPSET(MAP);
  case OMP_INFOTYPE_DATA_TRANSFER:
    return DPSET(COPY);
  case OMP_INFOTYPE_ALL:
    return DPSET(ALL);
  }
}
#else
#define debugInfoEnabled(InfoType) false
#endif // OMPTARGET_DEBUG

/// Emit a message giving the user extra information about the runtime if
#define INFO(_flags, _id, ...)                                                 \
  do {                                                                         \
    if (debugInfoEnabled(_flags)) {                                            \
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

#endif // OMPTARGET_SHARED_DEBUG_H
