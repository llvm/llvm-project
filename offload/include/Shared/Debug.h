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
#include <cstdarg>
#include <mutex>
#include <string>
#include <cstdint>
#include <cstdlib>

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Format.h"
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
  // AMD-only flag values (at least for now)
  // Show kernel launches
  OMP_INFOTYPE_AMD_KERNEL_TRACE = 0x1000,
  // Enable also API-level tracing
  OMP_INFOTYPE_AMD_API_TRACE = 0x200,
  // Print whenever data does not have a viable device counterpart.
  OMP_INFOTYPE_EMPTY_MAPPING = 0x0040,
  // Print diagnostic information for users.
  OMP_INFOTYPE_USER_DIAGNOSTIC = 0x0080,
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

  static std::once_flag KTFlag{};
  std::call_once(KTFlag, []() {
    if (char *EnvStr = getenv("LIBOMPTARGET_KERNEL_TRACE")) {
      auto V = std::stoi(EnvStr);
      // Match the LIBOMPTARGET_KERNEL_TRACE values and set InfoLevel to the
      // enum values to keep backward-compatibility for
      // LIBOMPTARGET_KERNEL_TRACE
      if (V == 1)
        InfoLevel.store(OMP_INFOTYPE_AMD_KERNEL_TRACE);
      if (V == 2)
        InfoLevel.store(OMP_INFOTYPE_AMD_API_TRACE |
                        /*OMP_INFOTYPE_API_TRACE=*/0xff000000);
      if (V == 3)
        InfoLevel.store(OMP_INFOTYPE_AMD_KERNEL_TRACE |
			OMP_INFOTYPE_AMD_API_TRACE |
                        /*OMP_INFOTYPE_API_TRACE=*/0xff000000);
    }
  });

  return InfoLevel;
}

inline uint32_t getInfoLevel() { return getInfoLevelInternal().load(); }

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

/// Emit a message giving the user extra information about the runtime if
#define INFO(_flags, _id, ...)                                                 \
  do {                                                                         \
    if (::llvm::offload::debug::isDebugEnabled()) {                            \
      DP(__VA_ARGS__);                                                         \
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

namespace llvm::offload::debug {

/// A raw_ostream that tracks `\n` and print the prefix after each
/// newline. Based on raw_ldbg_ostream from Support/DebugLog.h
class LLVM_ABI odbg_ostream final : public raw_ostream {
public:
  enum IfLevel : uint32_t;
  enum OnlyLevel : uint32_t;

private:
  std::string Prefix;
  raw_ostream &Os;
  uint32_t BaseLevel;
  bool ShouldPrefixNextString;
  bool ShouldEmitNewLineOnDestruction;
  bool NeedEndNewLine = false;

  /// If the stream is muted, writes to it are ignored
  bool Muted = false;

  /// Split the line on newlines and insert the prefix before each
  /// newline. Forward everything to the underlying stream.
  void write_impl(const char *Ptr, size_t Size) final {
    if (Muted)
      return;

    NeedEndNewLine = false;
    auto Str = StringRef(Ptr, Size);
    auto Eol = Str.find('\n');
    // Handle `\n` occurring in the string, ensure to print the prefix at the
    // beginning of each line.
    while (Eol != StringRef::npos) {
      // Take the line up to the newline (including the newline).
      StringRef Line = Str.take_front(Eol + 1);
      if (!Line.empty())
        writeWithPrefix(Line);
      // We printed a newline, record here to print a prefix.
      ShouldPrefixNextString = true;
      Str = Str.drop_front(Eol + 1);
      Eol = Str.find('\n');
    }
    if (!Str.empty()) {
      writeWithPrefix(Str);
      NeedEndNewLine = true;
    }
  }
  void emitPrefix() { Os.write(Prefix.c_str(), Prefix.size()); }
  void writeWithPrefix(StringRef Str) {
    if (ShouldPrefixNextString) {
      emitPrefix();
      ShouldPrefixNextString = false;
    }
    Os.write(Str.data(), Str.size());
  }

public:
  explicit odbg_ostream(std::string Prefix, raw_ostream &Os, uint32_t BaseLevel,
                        bool ShouldPrefixNextString = true,
                        bool ShouldEmitNewLineOnDestruction = true)
      : Prefix(std::move(Prefix)), Os(Os), BaseLevel(BaseLevel),
        ShouldPrefixNextString(ShouldPrefixNextString),
        ShouldEmitNewLineOnDestruction(ShouldEmitNewLineOnDestruction) {
    SetUnbuffered();
  }
  ~odbg_ostream() final {
    if (ShouldEmitNewLineOnDestruction && NeedEndNewLine)
      Os << '\n';
  }
  odbg_ostream(const odbg_ostream &) = delete;
  odbg_ostream &operator=(const odbg_ostream &) = delete;
  odbg_ostream(odbg_ostream &&other) : Os(other.Os) {
    Prefix = std::move(other.Prefix);
    BaseLevel = other.BaseLevel;
    ShouldPrefixNextString = other.ShouldPrefixNextString;
    ShouldEmitNewLineOnDestruction = other.ShouldEmitNewLineOnDestruction;
    NeedEndNewLine = other.NeedEndNewLine;
    Muted = other.Muted;
  }

  /// Forward the current_pos method to the underlying stream.
  uint64_t current_pos() const final { return Os.tell(); }

  /// Some of the `<<` operators expect an lvalue, so we trick the type
  /// system.
  odbg_ostream &asLvalue() { return *this; }

  void shouldMute(const IfLevel Filter) { Muted = Filter > BaseLevel; }
  void shouldMute(const OnlyLevel Filter) { Muted = BaseLevel != Filter; }
};

/// dbgs - Return a circular-buffered debug stream.
[[maybe_unused]] static llvm::raw_ostream &dbgs() {
  // Do one-time initialization in a thread-safe way.
  static struct dbgstream {
    llvm::circular_raw_ostream strm;

    dbgstream() : strm(llvm::errs(), "*** Debug Log Output ***\n", 0) {}
  } thestrm;

  return thestrm.strm;
}

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

[[maybe_unused]] static DebugFilter parseDebugFilter(StringRef Filter) {
  size_t Pos = Filter.find(':');
  if (Pos == StringRef::npos)
    return {Filter, 1};

  StringRef Type = Filter.slice(0, Pos);
  uint32_t Level = 1;
  if (Filter.slice(Pos + 1, Filter.size()).getAsInteger(10, Level))
    Level = 1;

  return {Type, Level};
}

[[maybe_unused]] static DebugSettings &getDebugSettings() {
  static DebugSettings Settings;
  static std::once_flag Flag{};
  std::call_once(Flag, []() {
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

    for (auto &FilterSpec : llvm::split(EnvRef, ',')) {
      if (FilterSpec.empty())
        continue;
      Settings.Filters.push_back(parseDebugFilter(FilterSpec));
    }
  });

  return Settings;
}

inline bool isDebugEnabled() { return getDebugSettings().Enabled; }

[[maybe_unused]] static bool
shouldPrintDebug(const char *Component, const char *Type, uint32_t &Level) {
  const auto &Settings = getDebugSettings();
  if (!Settings.Enabled)
    return false;

  if (Settings.Filters.empty()) {
    if (Level <= Settings.DefaultLevel) {
      Level = Settings.DefaultLevel;
      return true;
    }
    return false;
  }

  for (const auto &DT : Settings.Filters) {
    if (DT.Level < Level)
      continue;
    if (DT.Type.equals_insensitive(Type) ||
        DT.Type.equals_insensitive(Component)) {
      Level = DT.Level;
      return true;
    }
  }

  return false;
}

/// Compute the prefix for the debug log in the form of:
/// "Component --> "
[[maybe_unused]] static std::string computePrefix(StringRef Component,
                                                  StringRef DebugType) {
  std::string Prefix;
  raw_string_ostream OsPrefix(Prefix);
  OsPrefix << Component << " --> ";
  return OsPrefix.str();
}

static inline raw_ostream &operator<<(raw_ostream &Os,
                                      const odbg_ostream::IfLevel Filter) {
  odbg_ostream &Dbg = static_cast<odbg_ostream &>(Os);
  Dbg.shouldMute(Filter);
  return Dbg;
}

static inline raw_ostream &operator<<(raw_ostream &Os,
                                      const odbg_ostream::OnlyLevel Filter) {
  odbg_ostream &Dbg = static_cast<odbg_ostream &>(Os);
  Dbg.shouldMute(Filter);
  return Dbg;
}

#define ODBG_BASE(Stream, Component, Prefix, Type, Level)                      \
  for (uint32_t RealLevel = (Level),                                           \
                _c = llvm::offload::debug::isDebugEnabled() &&                 \
                     llvm::offload::debug::shouldPrintDebug(                   \
                         (Component), (Type), RealLevel);                      \
       _c; _c = 0)                                                             \
  ::llvm::offload::debug::odbg_ostream{                                        \
      ::llvm::offload::debug::computePrefix((Prefix), (Type)), (Stream),       \
      RealLevel, /*ShouldPrefixNextString=*/true,                              \
      /*ShouldEmitNewLineOnDestruction=*/true}                                 \
      .asLvalue()

#define ODBG_STREAM(Stream, Type, Level)                                       \
  ODBG_BASE(Stream, GETNAME(TARGET_NAME), DEBUG_PREFIX, Type, Level)

#define ODBG_0() ODBG_2("default", 1)
#define ODBG_1(Type) ODBG_2(Type, 1)
#define ODBG_2(Type, Level)                                                    \
  ODBG_STREAM(llvm::offload::debug::dbgs(), Type, Level)
#define ODBG_SELECT(Type, Level, NArgs, ...) ODBG_##NArgs

// Print a debug message of a certain type and verbosity level. If no type
// or level is provided, "default" and "1" are assumed respectively.
// Usage examples:
// ODBG("type1", 2) << "This is a level 2 message of type1";
// ODBG("Init") << "This is a default level of the init type";
// ODBG() << "This is a level 1 message of the default type";
// ODBG("Init", 3) << NumDevices << " were initialized";
// ODBG("Kernel") << "Launching " << KernelName << " on device " << DeviceId;
#define ODBG(...) ODBG_SELECT(__VA_ARGS__ __VA_OPT__(, ) 2, 1, 0)(__VA_ARGS__)

// Filter the next elements in the debug stream if the current debug level is
// lower than  specified level. Example:
// ODBG("Mapping", 2) << "level 2 info "
//   << ODBG_IF_LEVEL(3) << " level 3 info" << Arg
//   << ODBG_IF_LEVEL(4) << " level 4 info" << &Arg
//   << ODBG_RESET_LEVEL() << " more level 2 info";
#define ODBG_IF_LEVEL(Level)                                                   \
  static_cast<llvm::offload::debug::odbg_ostream::IfLevel>(Level)

// Filter the next elements in the debug stream if the current debug level is
// not exactly the specified level. Example:
// ODBG() << "Starting computation "
//   << ODBG_ONLY_LEVEL(1) << "on a device"
//   << ODBG_ONLY_LEVEL(2) << "and mapping data on device" << DeviceId;
//   << ODBG_ONLY_LEVEL(3) << dumpDetailedMappingInfo(DeviceId);
#define ODBG_ONLY_LEVEL(Level)                                                 \
  static_cast<llvm::offload::debug::odbg_ostream::OnlyLevel>(Level)

// Reset the level back to the original level after ODBG_IF_LEVEL or
// ODBG_ONLY_LEVEL have been used
#define ODBG_RESET_LEVEL()                                                     \
  static_cast<llvm::offload::debug::odbg_ostream::IfLevel>(0)

// helper templates to support lambdas with different number of arguments
template <typename LambdaTy> struct LambdaHelper {
  template <typename T, typename = std::void_t<>>
  struct has_two_args : std::false_type {};
  template <typename T>
  struct has_two_args<T,
                      std::void_t<decltype(std::declval<T>().operator()(1, 2))>>
      : std::true_type {};

  static void dispatch(LambdaTy func, llvm::raw_ostream &Os, uint32_t Level) {
    if constexpr (has_two_args<LambdaTy>::value)
      func(Os, Level);
    else
      func(Os);
  }
};

#define ODBG_OS_BASE(Stream, Component, Prefix, Type, Level, Callback)         \
  if (::llvm::offload::debug::isDebugEnabled()) {                              \
    uint32_t RealLevel = (Level);                                              \
    if (::llvm::offload::debug::shouldPrintDebug((Component), (Type),          \
                                                 RealLevel)) {                 \
      ::llvm::offload::debug::odbg_ostream OS{                                 \
          ::llvm::offload::debug::computePrefix((Prefix), (Type)), (Stream),   \
          RealLevel, /*ShouldPrefixNextString=*/true,                          \
          /*ShouldEmitNewLineOnDestruction=*/true};                            \
      auto F = Callback;                                                       \
      ::llvm::offload::debug::LambdaHelper<decltype(F)>::dispatch(F, OS,       \
                                                                  RealLevel);  \
    }                                                                          \
  }

#define ODBG_OS_STREAM(Stream, Type, Level, Callback)                          \
  ODBG_OS_BASE(Stream, GETNAME(TARGET_NAME), DEBUG_PREFIX, Type, Level,        \
               Callback)
#define ODBG_OS_3(Type, Level, Callback)                                       \
  ODBG_OS_STREAM(llvm::offload::debug::dbgs(), Type, Level, Callback)
#define ODBG_OS_2(Type, Callback) ODBG_OS_3(Type, 1, Callback)
#define ODBG_OS_1(Callback) ODBG_OS_2("default", Callback)
#define ODBG_OS_SELECT(Type, Level, Callback, NArgs, ...) ODBG_OS_##NArgs
// Print a debug message of a certain type and verbosity level using a callback
// to emit the message. If no type or level is provided, "default" and "1 are
// assumed respectively.
#define ODBG_OS(...)                                                           \
  ODBG_OS_SELECT(__VA_ARGS__ __VA_OPT__(, ) 3, 2, 1)(__VA_ARGS__)

#else

inline bool isDebugEnabled() { return false; }

#define ODBG_NULL                                                              \
  for (bool _c = false; _c; _c = false)                                        \
  ::llvm::nulls()

// Don't print anything if debugging is disabled
#define ODBG_BASE(Stream, Component, Prefix, Type, Level) ODBG_NULL
#define ODBG_STREAM(Stream, Type, Level) ODBG_NULL
#define ODBG_IF_LEVEL(Level) 0
#define ODBG_ONLY_LEVEL(Level) 0
#define ODBG_RESET_LEVEL() 0
#define ODBG(...) ODBG_NULL

#define ODBG_OS_BASE(Stream, Component, Prefix, Type, Level, Callback)
#define ODBG_OS_STREAM(Stream, Type, Level, Callback)
#define ODBG_OS(...)

#endif

} // namespace llvm::offload::debug

namespace llvm::omp::target::debug {
using namespace llvm::offload::debug;

enum OmpDebugLevel : uint32_t {
  ODL_Default = 1,
  ODL_Error = ODL_Default,
  ODL_Detailed = 2,
  ODL_Verbose = 3,
  ODL_VeryVerbose = 4,
  ODL_Dumping = 5
};

/* Debug types to use in libomptarget */
constexpr const char *ODT_Init = "Init";
constexpr const char *ODT_Mapping = "Mapping";
constexpr const char *ODT_Kernel = "Kernel";
constexpr const char *ODT_DataTransfer = "DataTransfer";
constexpr const char *ODT_Sync = "Sync";
constexpr const char *ODT_Deinit = "Deinit";
constexpr const char *ODT_Error = "Error";
constexpr const char *ODT_KernelArgs = "KernelArgs";
constexpr const char *ODT_MappingExists = "MappingExists";
constexpr const char *ODT_DumpTable = "DumpTable";
constexpr const char *ODT_MappingChanged = "MappingChanged";
constexpr const char *ODT_PluginKernel = "PluginKernel";
constexpr const char *ODT_EmptyMapping = "EmptyMapping";
constexpr const char *ODT_Device = "Device";
constexpr const char *ODT_Interface = "Interface";
constexpr const char *ODT_Alloc = "Alloc";

static inline odbg_ostream reportErrorStream() {
#ifdef OMPTARGET_DEBUG
  if (::llvm::offload::debug::isDebugEnabled()) {
    uint32_t RealLevel = ODL_Error;
    if (::llvm::offload::debug::shouldPrintDebug(GETNAME(TARGET_NAME),
                                                 (ODT_Error), RealLevel))
      return odbg_ostream{
          ::llvm::offload::debug::computePrefix("DEBUG_PREFIX_FIXME:", ODT_Error),
          ::llvm::offload::debug::dbgs(), RealLevel};
    else
      return odbg_ostream{"", ::llvm::nulls(), 1};
  }
#endif
  return odbg_ostream{GETNAME(TARGET_NAME) " error: ",
                      ::llvm::offload::debug::dbgs(), ODL_Error};
}

#ifdef OMPTARGET_DEBUG
// Deprecated debug print macros
[[maybe_unused]] static std::string formatToStr(const char *format, ...) {
  va_list args;
  va_start(args, format);
  size_t len = std::vsnprintf(NULL, 0, format, args);
  va_end(args);
  llvm::SmallVector<char, 128> vec(len + 1);
  va_start(args, format);
  std::vsnprintf(&vec[0], len + 1, format, args);
  va_end(args);
  return &vec[0];
}

// helper macro to support old DP and REPORT macros with printf syntax
#define FORMAT_TO_STR(Format, ...)                                             \
  ::llvm::omp::target::debug::formatToStr(Format __VA_OPT__(, ) __VA_ARGS__)

#define DP(...) ODBG() << FORMAT_TO_STR(__VA_ARGS__);

#define REPORT_INT_OLD(...)                                                    \
  do {                                                                         \
    if (::llvm::offload::debug::isDebugEnabled()) {                            \
      ODBG(::llvm::omp::target::debug::ODT_Error,                              \
           ::llvm::omp::target::debug::ODL_Error)                              \
          << FORMAT_TO_STR(__VA_ARGS__);                                       \
    } else {                                                                   \
      FAILURE_MESSAGE(__VA_ARGS__);                                            \
    }                                                                          \
  } while (false)

// Define default format for pointers
static inline raw_ostream &operator<<(raw_ostream &Os, void *Ptr) {
  Os << ::llvm::format(DPxMOD, DPxPTR(Ptr));
  return Os;
}

#else
#define DP(...)                                                                \
  {                                                                            \
  }
#define REPORT_INT_OLD(...) FAILURE_MESSAGE(__VA_ARGS__);
#endif // OMPTARGET_DEBUG

// This is used for the new style REPORT macro
#define REPORT_INT() ::llvm::omp::target::debug::reportErrorStream()

// Make REPORT compatible with old and new syntax
#define REPORT(...) REPORT_INT##__VA_OPT__(_OLD)(__VA_ARGS__)

} // namespace llvm::omp::target::debug

#endif // OMPTARGET_SHARED_DEBUG_H
