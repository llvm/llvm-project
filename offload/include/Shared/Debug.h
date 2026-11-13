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

#ifndef DEBUG_PREFIX
#warning                                                                       \
    "Debug support was included without defining DEBUG_PREFIX. Setting it to \"Unknown\""
#define DEBUG_PREFIX "Unknown"
#endif

#include <atomic>
#include <cstdio>
#include <mutex>
#include <string>

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

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

#define INFO_VERBOSITY(_flags)                                                 \
  ::llvm::offload::debug::infoVerbosity(GETNAME(TARGET_NAME), (_flags))

/// Emit a message giving the user extra information about the runtime.
/// In OMPTARGET_DEBUG builds, messages selected by LIBOMPTARGET_DEBUG go to the
/// debug stream instead of the "info:" output.
#ifdef OMPTARGET_DEBUG
#define INFO(_level, _flags, _id, ...)                                         \
  do {                                                                         \
    const char *_infoType = ::llvm::offload::debug::infoTypeName(_flags);      \
    uint32_t _infoLevel = (_level);                                            \
    if (::llvm::offload::debug::shouldPrintDebug(GETNAME(TARGET_NAME),         \
                                                 _infoType, _infoLevel))       \
      ODBG_STREAM(::llvm::offload::debug::dbgs(), _infoType, (_level))         \
          << ::llvm::offload::debug::formatInfoMessage(__VA_ARGS__);           \
    else if (::getInfoLevel() & (_flags))                                      \
      INFO_MESSAGE(_id, __VA_ARGS__);                                          \
  } while (false)
#else
#define INFO(_level, _flags, _id, ...)                                         \
  do {                                                                         \
    if (INFO_VERBOSITY(_flags) >= (_level))                                    \
      INFO_MESSAGE(_id, __VA_ARGS__);                                          \
  } while (false)
#endif

#define DUMP_INFO(toStdOut, _flags, _id, ...)                                  \
  do {                                                                         \
    if (toStdOut) {                                                            \
      INFO_MESSAGE_TO(stdout, _id, __VA_ARGS__);                               \
    } else {                                                                   \
      INFO(1, _flags, _id, __VA_ARGS__);                                       \
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

  /// Buffer to reduce interference between different threads
  /// writing at the same time to the underlying stream.
  static constexpr size_t BufferSize = 256;
  llvm::SmallString<BufferSize> Buffer;

  // Stream to write into Buffer. Its flushed to Os upon destruction.
  llvm::raw_svector_ostream BufferStrm;

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
  void emitPrefix() { BufferStrm.write(Prefix.c_str(), Prefix.size()); }
  void writeWithPrefix(StringRef Str) {
    if (ShouldPrefixNextString) {
      emitPrefix();
      ShouldPrefixNextString = false;
    }
    BufferStrm.write(Str.data(), Str.size());
  }

public:
  explicit odbg_ostream(std::string Prefix, raw_ostream &Os, uint32_t BaseLevel,
                        bool ShouldPrefixNextString = true,
                        bool ShouldEmitNewLineOnDestruction = true)
      : Prefix(std::move(Prefix)), Os(Os), BaseLevel(BaseLevel),
        ShouldPrefixNextString(ShouldPrefixNextString),
        ShouldEmitNewLineOnDestruction(ShouldEmitNewLineOnDestruction),
        BufferStrm(Buffer) {
    SetUnbuffered();
  }
  ~odbg_ostream() final {
    if (ShouldEmitNewLineOnDestruction && NeedEndNewLine)
      BufferStrm << '\n';
    Os << BufferStrm.str();
  }
  odbg_ostream(const odbg_ostream &) = delete;
  odbg_ostream &operator=(const odbg_ostream &) = delete;
  odbg_ostream(odbg_ostream &&other) : Os(other.Os), BufferStrm(Buffer) {
    Prefix = std::move(other.Prefix);
    BaseLevel = other.BaseLevel;
    ShouldPrefixNextString = other.ShouldPrefixNextString;
    ShouldEmitNewLineOnDestruction = other.ShouldEmitNewLineOnDestruction;
    NeedEndNewLine = other.NeedEndNewLine;
    Muted = other.Muted;
    BufferStrm << other.BufferStrm.str();
  }

  /// Forward the current_pos method to the underlying stream.
  uint64_t current_pos() const final { return BufferStrm.tell(); }

  /// Some of the `<<` operators expect an lvalue, so we trick the type
  /// system.
  odbg_ostream &asLvalue() { return *this; }

  void shouldMute(const IfLevel Filter) { Muted = Filter > BaseLevel; }
  void shouldMute(const OnlyLevel Filter) { Muted = BaseLevel != Filter; }
};

/// dbgs - Return the debug stream for offload debugging (just llvm::errs()).
[[maybe_unused]] static llvm::raw_ostream &dbgs() { return llvm::errs(); }

struct DebugFilter {
  StringRef Type;
  uint32_t Level;
};

struct DebugSettings {
  bool Enabled = false;
  uint32_t DefaultLevel = 1;
  // Types/Components in this list are not printed when debug is enabled
  // unless they are explicitly requested by the user in IncludeFilters.
  llvm::SmallVector<StringRef> ExcludeFilters;
  // Types/Components in this list are printed when debug is enabled if
  // the debug level is equal or higher than the specified level.
  llvm::SmallVector<DebugFilter> IncludeFilters;
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

[[maybe_unused]] static const char *infoTypeName(uint32_t Flag);

[[maybe_unused]] static DebugSettings &getDebugSettings() {
  // Never destroyed: it is still queried during runtime teardown at exit.
  static DebugSettings &Settings = *new DebugSettings();
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

    // Messages with Type/Components added to the exclude list are not
    // not printed when debug is enabled unless they are explicitly
    // requested by the user.
    // Eventually, this should be configured from the upper layers but
    // for now we can hardcode some excluded types here like:
    // Settings.ExcludeFilters.push_back(Type);

    if (!EnvRef.getAsInteger(10, Settings.DefaultLevel))
      return;

    Settings.DefaultLevel = 1;

    for (auto &FilterSpec : llvm::split(EnvRef, ',')) {
      if (FilterSpec.empty())
        continue;
      DebugFilter Filter = parseDebugFilter(FilterSpec);

      if (Filter.Type.equals_insensitive("info")) {
        uint32_t InfoLevel = FilterSpec.contains(':') ? Filter.Level : 3;
        for (uint32_t F = OMP_INFOTYPE_KERNEL_ARGS;
             F <= OMP_INFOTYPE_EMPTY_MAPPING; F <<= 1)
          Settings.IncludeFilters.push_back({infoTypeName(F), InfoLevel});
        continue;
      }

      // Remove from ExcludeFilters if present
      Settings.ExcludeFilters.erase(
          std::remove_if(Settings.ExcludeFilters.begin(),
                         Settings.ExcludeFilters.end(),
                         [&](StringRef OutType) {
                           return OutType.equals_insensitive(Filter.Type);
                         }),
          Settings.ExcludeFilters.end());

      Settings.IncludeFilters.push_back(Filter);
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

  for (const auto &Filter : Settings.ExcludeFilters) {
    if (Filter.equals_insensitive(Type) || Filter.equals_insensitive(Component))
      return false;
  }

  if (Settings.IncludeFilters.empty()) {
    if (Level <= Settings.DefaultLevel) {
      Level = Settings.DefaultLevel;
      return true;
    }
    return false;
  }

  bool Matched = false;
  for (const auto &DT : Settings.IncludeFilters) {
    if (DT.Level < Level)
      continue;
    if (DT.Type.equals_insensitive("all") || DT.Type.equals_insensitive(Type) ||
        DT.Type.equals_insensitive(Component)) {
      Level = DT.Level;
      Matched = true;
    }
  }

  return Matched;
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

[[maybe_unused]] static const char *infoTypeName(uint32_t Flag) {
  switch (Flag) {
  case OMP_INFOTYPE_KERNEL_ARGS:
    return "KernelArgs";
  case OMP_INFOTYPE_MAPPING_EXISTS:
    return "MappingExists";
  case OMP_INFOTYPE_DUMP_TABLE:
    return "DumpTable";
  case OMP_INFOTYPE_MAPPING_CHANGED:
    return "MappingChanged";
  case OMP_INFOTYPE_PLUGIN_KERNEL:
    return "PluginKernel";
  case OMP_INFOTYPE_DATA_TRANSFER:
    return "DataCopy";
  case OMP_INFOTYPE_EMPTY_MAPPING:
    return "EmptyMapping";
  default:
    return "all";
  }
}

enum InfoLevel : uint32_t { IL_Summary = 1, IL_Detailed = 2, IL_Full = 3 };

/// Highest verbosity any matching filter grants; 0 if not selected.
[[maybe_unused]] static uint32_t infoVerbosity(const char *Component,
                                               uint32_t Flags) {
  if (::getInfoLevel() & Flags)
    return IL_Full;
  if (!isDebugEnabled())
    return 0;
  uint32_t Level = IL_Summary;
  return shouldPrintDebug(Component, infoTypeName(Flags), Level) ? Level : 0;
}

#ifdef OMPTARGET_DEBUG

// Without arguments the message is printed as-is: llvm::format would trip
// -Wformat-security.
[[maybe_unused]] static StringRef formatInfoMessage(const char *Msg) {
  return Msg;
}
template <typename... Ts>
static auto formatInfoMessage(const char *Fmt, const Ts &...Vals) {
  return ::llvm::format(Fmt, Vals...);
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
  template <typename FuncTy, typename RetTy, typename... Args>
  static constexpr size_t CountArgs(RetTy (FuncTy::*)(Args...)) {
    return sizeof...(Args);
  }
  template <typename FuncTy, typename RetTy, typename... Args>
  static constexpr size_t CountArgs(RetTy (FuncTy::*)(Args...) const) {
    return sizeof...(Args);
  }

  static constexpr size_t NArgs = CountArgs(&LambdaTy::operator());
};

template <typename LambdaTy> struct LambdaOs : public LambdaHelper<LambdaTy> {
  static void dispatch(LambdaTy func, llvm::raw_ostream &Os, uint32_t Level) {
    if constexpr (LambdaHelper<LambdaTy>::NArgs == 2)
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
      ::llvm::offload::debug::LambdaOs<decltype(F)>::dispatch(F, OS,           \
                                                              RealLevel);      \
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

// helper templates to support lambdas with different number of arguments
template <typename LambdaTy> struct LambdaIf : public LambdaHelper<LambdaTy> {
  static void dispatch(LambdaTy func, uint32_t Level) {
    if constexpr (LambdaHelper<LambdaTy>::NArgs == 1)
      func(Level);
    else
      func();
  }
};

#define ODBG_IF_BASE(Type, Level, Callback)                                    \
  if (::llvm::offload::debug::isDebugEnabled()) {                              \
    uint32_t RealLevel = (Level);                                              \
    if (::llvm::offload::debug::shouldPrintDebug(GETNAME(TARGET_NAME), (Type), \
                                                 RealLevel)) {                 \
      auto F = Callback;                                                       \
      ::llvm::offload::debug::LambdaIf<decltype(F)>::dispatch(F, RealLevel);   \
    }                                                                          \
  }

#define ODBG_IF_3(Type, Level, Callback) ODBG_IF_BASE(Type, Level, Callback)
#define ODBG_IF_2(Type, Callback) ODBG_IF_3(Type, 1, Callback)
#define ODBG_IF_1(Callback) ODBG_IF_2("default", Callback)
#define ODBG_IF_SELECT(Type, Level, Callback, NArgs, ...) ODBG_IF_##NArgs
#define ODBG_IF(...)                                                           \
  ODBG_IF_SELECT(__VA_ARGS__ __VA_OPT__(, ) 3, 2, 1)(__VA_ARGS__)

#else

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

#define ODBG_IF_BASE(Type, Level, Callback)
#define ODBG_IF(...)

#endif

// Common debug types in offload.
constexpr const char *OLDT_Init = "Init";
constexpr const char *OLDT_Kernel = "Kernel";
constexpr const char *OLDT_DataTransfer = "DataTransfer";
constexpr const char *OLDT_Sync = "Sync";
constexpr const char *OLDT_Deinit = "Deinit";
constexpr const char *OLDT_Error = "Error";
constexpr const char *OLDT_Device = "Device";
constexpr const char *OLDT_Interface = "Interface";
constexpr const char *OLDT_Alloc = "Alloc";
constexpr const char *OLDT_Tool = "Tool";
constexpr const char *OLDT_Module = "Module";

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
constexpr const char *ODT_Init = OLDT_Init;
constexpr const char *ODT_Mapping = "Mapping";
constexpr const char *ODT_Kernel = OLDT_Kernel;
constexpr const char *ODT_DataTransfer = OLDT_DataTransfer;
constexpr const char *ODT_Sync = OLDT_Sync;
constexpr const char *ODT_Deinit = OLDT_Deinit;
constexpr const char *ODT_Error = OLDT_Error;
constexpr const char *ODT_KernelArgs = "KernelArgs";
constexpr const char *ODT_MappingExists = "MappingExists";
constexpr const char *ODT_DumpTable = "DumpTable";
constexpr const char *ODT_MappingChanged = "MappingChanged";
constexpr const char *ODT_PluginKernel = "PluginKernel";
constexpr const char *ODT_EmptyMapping = "EmptyMapping";
constexpr const char *ODT_Device = OLDT_Device;
constexpr const char *ODT_Interface = OLDT_Interface;
constexpr const char *ODT_Alloc = OLDT_Alloc;
constexpr const char *ODT_Tool = OLDT_Tool;
constexpr const char *ODT_Module = OLDT_Module;
constexpr const char *ODT_Interop = "Interop";

static inline odbg_ostream reportErrorStream() {
#ifdef OMPTARGET_DEBUG
  if (::llvm::offload::debug::isDebugEnabled()) {
    uint32_t RealLevel = ODL_Error;
    if (::llvm::offload::debug::shouldPrintDebug(GETNAME(TARGET_NAME),
                                                 (ODT_Error), RealLevel))
      return odbg_ostream{
          ::llvm::offload::debug::computePrefix(DEBUG_PREFIX, ODT_Error),
          ::llvm::offload::debug::dbgs(), RealLevel};
    else
      return odbg_ostream{"", ::llvm::nulls(), 1};
  }
#endif
  return odbg_ostream{GETNAME(TARGET_NAME) " error: ",
                      ::llvm::offload::debug::dbgs(), ODL_Error};
}

#ifdef OMPTARGET_DEBUG
// Define default format for pointers
static inline raw_ostream &operator<<(raw_ostream &Os, void *Ptr) {
  Os << ::llvm::format(DPxMOD, DPxPTR(Ptr));
  return Os;
}
#endif // OMPTARGET_DEBUG

// New REPORT macro in the same style as ODBG
#define REPORT() ::llvm::omp::target::debug::reportErrorStream()

} // namespace llvm::omp::target::debug

#endif // OMPTARGET_SHARED_DEBUG_H
