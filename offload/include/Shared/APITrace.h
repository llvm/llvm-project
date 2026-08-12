//===-- Shared/APITrace.h - Tracing of offload API calls --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Prints one line per offload API call while tracing is enabled:
//
//   ---> init_device(.DeviceId = 0)-> OFFLOAD_SUCCESS (1053 us)
//
// Tracing is off until the runtime above the plugins turns it on, see
// GenericPluginTy::set_api_trace.
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_SHARED_API_TRACE_H
#define OMPTARGET_SHARED_API_TRACE_H

#include "APITypes.h"
#include "omptarget.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <atomic>
#include <chrono>
#include <type_traits>
#include <utility>

namespace llvm::offload::trace {

inline std::atomic<bool> &getTraceEnabledFlag() {
  static std::atomic<bool> Enabled = false;
  return Enabled;
}

/// The flag is read on every call so that tracing can be toggled around a
/// region of interest.
inline bool isTraceEnabled() {
  return getTraceEnabledFlag().load(std::memory_order_relaxed);
}

inline void setTraceEnabled(bool Enable) {
  getTraceEnabledFlag().store(Enable, std::memory_order_relaxed);
}

inline raw_ostream &operator<<(raw_ostream &OS, __tgt_device_binary Binary) {
  return OS << reinterpret_cast<void *>(Binary.handle);
}

inline raw_ostream &operator<<(raw_ostream &OS, OffloadStatusTy Status) {
  switch (Status) {
  case OFFLOAD_SUCCESS:
    return OS << "OFFLOAD_SUCCESS";
  case OFFLOAD_FAIL:
    return OS << "OFFLOAD_FAIL";
  }
  return OS << static_cast<int32_t>(Status);
}

inline raw_ostream &operator<<(raw_ostream &OS, TargetAllocTy Kind) {
  switch (Kind) {
  case TARGET_ALLOC_DEVICE:
    return OS << "TARGET_ALLOC_DEVICE";
  case TARGET_ALLOC_HOST:
    return OS << "TARGET_ALLOC_HOST";
  case TARGET_ALLOC_SHARED:
    return OS << "TARGET_ALLOC_SHARED";
  case TARGET_ALLOC_DEFAULT:
    return OS << "TARGET_ALLOC_DEFAULT";
  }
  return OS << static_cast<int32_t>(Kind);
}

template <typename Ty, typename = void> struct IsPrintable : std::false_type {};
template <typename Ty>
struct IsPrintable<Ty, std::void_t<decltype(std::declval<raw_ostream &>()
                                            << std::declval<const Ty &>())>>
    : std::true_type {};

/// Types with no raw_ostream overload are shown as an address, so that adding a
/// parameter to a traced entry point never breaks the build.
template <typename Ty> void printArg(raw_ostream &OS, const Ty &Arg) {
  using DecayedTy = std::decay_t<Ty>;
  if constexpr (std::is_same_v<DecayedTy, char *> ||
                std::is_same_v<DecayedTy, const char *>) {
    if (Arg)
      OS << '"' << Arg << '"';
    else
      OS << "(nullptr)";
  } else if constexpr (IsPrintable<Ty>::value) {
    OS << Arg;
  } else {
    OS << '&' << static_cast<const void *>(&Arg);
  }
}

/// Arguments are formatted on entry so that output parameters show the values
/// the caller passed in, and the line is buffered so that concurrent calls do
/// not interleave.
class CallTraceTy {
  SmallString<128> Buffer;
  raw_svector_ostream OS;
  std::chrono::steady_clock::time_point Start;
  bool Enabled;

  /// The argument names arrive as a single stringified macro argument list,
  /// e.g. "DeviceId, Size, HostPtr", which is consumed one name at a time.
  static StringRef takeName(StringRef &Names) {
    auto [Name, Rest] = Names.split(',');
    Names = Rest;
    return Name.trim();
  }

public:
  template <typename... ArgsTy>
  CallTraceTy(const char *Name, StringRef ArgNames, ArgsTy &&...Args)
      : OS(Buffer), Enabled(isTraceEnabled()) {
    if (!Enabled)
      return;

    OS << "---> " << Name << '(';
    StringRef Separator = "";
    ((OS << Separator << '.' << takeName(ArgNames) << " = ", printArg(OS, Args),
      Separator = ", "),
     ...);
    OS << ')';

    Start = std::chrono::steady_clock::now();
  }

  CallTraceTy(const CallTraceTy &) = delete;
  CallTraceTy &operator=(const CallTraceTy &) = delete;

  ~CallTraceTy() {
    if (!Enabled)
      return;

    auto Elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - Start);
    OS << " (" << Elapsed.count() << " us)\n";
    errs() << OS.str();
  }

  template <typename Ty> Ty &&result(Ty &&Result) {
    if (Enabled) {
      OS << "-> ";
      printArg(OS, Result);
    }
    return std::forward<Ty>(Result);
  }
};

} // namespace llvm::offload::trace

#define OFFLOAD_TRACE_CALL(...)                                                \
  ::llvm::offload::trace::CallTraceTy OffloadCallTrace(                        \
      __func__, #__VA_ARGS__ __VA_OPT__(, ) __VA_ARGS__)

/// Only valid where OFFLOAD_TRACE_CALL opened a scope. Optional; entry points
/// returning void just open the scope.
#define OFFLOAD_TRACE_RESULT(Result) OffloadCallTrace.result(Result)

#endif // OMPTARGET_SHARED_API_TRACE_H
