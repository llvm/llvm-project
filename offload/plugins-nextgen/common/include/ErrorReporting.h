//===- ErrorReporting.h - Helper to provide nice error messages ----- c++ -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef OFFLOAD_PLUGINS_NEXTGEN_COMMON_ERROR_REPORTING_H
#define OFFLOAD_PLUGINS_NEXTGEN_COMMON_ERROR_REPORTING_H

#include "PluginInterface.h"
#include "Shared/EnvironmentVar.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <optional>
#include <string>
#include <unistd.h>

namespace llvm {
namespace omp {
namespace target {
namespace plugin {

class ErrorReporter {

  enum ColorTy {
    Yellow = int(HighlightColor::Address),
    Green = int(HighlightColor::String),
    DarkBlue = int(HighlightColor::Tag),
    Cyan = int(HighlightColor::Attribute),
    DarkPurple = int(HighlightColor::Enumerator),
    DarkRed = int(HighlightColor::Macro),
    BoldRed = int(HighlightColor::Error),
    BoldLightPurple = int(HighlightColor::Warning),
    BoldDarkGrey = int(HighlightColor::Note),
    BoldLightBlue = int(HighlightColor::Remark),
  };

  /// The banner printed at the beginning of an error report.
  static constexpr auto ErrorBanner = "OFFLOAD ERROR: ";

  /// Return the device id as string, or n/a if not available.
  static std::string getDeviceIdStr(GenericDeviceTy *Device) {
    return Device ? std::to_string(Device->getDeviceId()) : "n/a";
  }

  /// Return a nice name for an TargetAllocTy.
  static StringRef getAllocTyName(TargetAllocTy Kind) {
    switch (Kind) {
    case TARGET_ALLOC_DEVICE_NON_BLOCKING:
    case TARGET_ALLOC_DEFAULT:
    case TARGET_ALLOC_DEVICE:
      return "device memory";
    case TARGET_ALLOC_HOST:
      return "pinned host memory";
    case TARGET_ALLOC_SHARED:
      return "managed memory";
      break;
    }
    llvm_unreachable("Unknown target alloc kind");
  }

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgcc-compat"
#pragma clang diagnostic ignored "-Wformat-security"
  /// Print \p Format, instantiated with \p Args to stderr.
  /// TODO: Allow redirection into a file stream.
  template <typename... ArgsTy>
  [[gnu::format(__printf__, 1, 2)]] static void print(const char *Format,
                                                      ArgsTy &&...Args) {
    raw_fd_ostream OS(STDERR_FILENO, false);
    OS << llvm::format(Format, Args...);
  }

  /// Print \p Format, instantiated with \p Args to stderr, but colored.
  /// TODO: Allow redirection into a file stream.
  template <typename... ArgsTy>
  [[gnu::format(__printf__, 2, 3)]] static void
  print(ColorTy Color, const char *Format, ArgsTy &&...Args) {
    raw_fd_ostream OS(STDERR_FILENO, false);
    WithColor(OS, HighlightColor(Color)) << llvm::format(Format, Args...);
  }

  /// Print \p Format, instantiated with \p Args to stderr, but colored and with
  /// a banner.
  /// TODO: Allow redirection into a file stream.
  template <typename... ArgsTy>
  [[gnu::format(__printf__, 1, 2)]] static void reportError(const char *Format,
                                                            ArgsTy &&...Args) {
    print(BoldRed, "%s", ErrorBanner);
    print(BoldRed, Format, Args...);
    print("\n");
  }
#pragma clang diagnostic pop

  static void reportError(const char *Str) { reportError("%s", Str); }
  static void print(const char *Str) { print("%s", Str); }
  static void print(StringRef Str) { print("%s", Str.str().c_str()); }
  static void print(ColorTy Color, const char *Str) { print(Color, "%s", Str); }
  static void print(ColorTy Color, StringRef Str) {
    print(Color, "%s", Str.str().c_str());
  }

  /// Pretty print a stack trace.
  static void reportStackTrace(StringRef StackTrace) {
    if (StackTrace.empty())
      return;

    SmallVector<StringRef> Lines, Parts;
    StackTrace.split(Lines, "\n", /*MaxSplit=*/-1, /*KeepEmpty=*/false);
    int Start = Lines.empty() || !Lines[0].contains("PrintStackTrace") ? 0 : 1;
    unsigned NumDigits =
        (int)(floor(log10(Lines.size() - Start - /*0*/ 1)) + 1);
    for (int I = Start, E = Lines.size(); I < E; ++I) {
      auto Line = Lines[I];
      Parts.clear();
      Line = Line.drop_while([](char C) { return std::isspace(C); });
      Line.split(Parts, " ", /*MaxSplit=*/2);
      if (Parts.size() != 3 || Parts[0].size() < 2 || Parts[0][0] != '#') {
        print("%s\n", Line.str().c_str());
        continue;
      }
      unsigned FrameIdx = std::stoi(Parts[0].drop_front(1).str());
      if (Start)
        FrameIdx -= 1;
      print(DarkPurple, "    %s", Parts[0].take_front().str().c_str());
      print(Green, "%*u", NumDigits, FrameIdx);
      print(BoldLightBlue, " %s", Parts[1].str().c_str());
      print(" %s\n", Parts[2].str().c_str());
    }
    print("\n");
  }

  /// Report information about an allocation associated with \p ATI.
  static void reportAllocationInfo(AllocationTraceInfoTy *ATI) {
    if (!ATI)
      return;

    if (!ATI->DeallocationTrace.empty()) {
      print(BoldLightPurple, "Last deallocation:\n");
      reportStackTrace(ATI->DeallocationTrace);
    }

    if (ATI->HostPtr)
      print(BoldLightPurple,
            "Last allocation of size %lu for host pointer %p:\n", ATI->Size,
            ATI->HostPtr);
    else
      print(BoldLightPurple, "Last allocation of size %lu:\n", ATI->Size);
    reportStackTrace(ATI->AllocationTrace);
    if (!ATI->LastAllocationInfo)
      return;

    unsigned I = 0;
    print(BoldLightPurple, "Prior allocations with the same base pointer:");
    while (ATI->LastAllocationInfo) {
      print("\n");
      ATI = ATI->LastAllocationInfo;
      print(BoldLightPurple, " #%u Prior deallocation of size %lu:\n", I,
            ATI->Size);
      reportStackTrace(ATI->DeallocationTrace);
      if (ATI->HostPtr)
        print(BoldLightPurple, " #%u Prior allocation for host pointer %p:\n",
              I, ATI->HostPtr);
      else
        print(BoldLightPurple, " #%u Prior allocation:\n", I);
      reportStackTrace(ATI->AllocationTrace);
      ++I;
    }
  }

  /// End the execution of the program.
  static void abortExecution() { abort(); }

public:
#define DEALLOCATION_ERROR(Format, ...)                                        \
  reportError(Format, __VA_ARGS__);                                            \
  reportStackTrace(StackTrace);                                                \
  reportAllocationInfo(ATI);                                                   \
  abortExecution();

  static void reportDeallocationOfNonAllocatedPtr(void *DevicePtr,
                                                  TargetAllocTy Kind,
                                                  AllocationTraceInfoTy *ATI,
                                                  std::string &StackTrace) {
    DEALLOCATION_ERROR("deallocation of non-allocated %s: %p",
                       getAllocTyName(Kind).data(), DevicePtr);
  }

  static void reportDeallocationOfDeallocatedPtr(void *DevicePtr,
                                                 TargetAllocTy Kind,
                                                 AllocationTraceInfoTy *ATI,
                                                 std::string &StackTrace) {
    DEALLOCATION_ERROR("double-free of %s: %p", getAllocTyName(Kind).data(),
                       DevicePtr);
  }

  static void reportDeallocationOfWrongPtrKind(void *DevicePtr,
                                               TargetAllocTy Kind,
                                               AllocationTraceInfoTy *ATI,
                                               std::string &StackTrace) {
    DEALLOCATION_ERROR("deallocation requires %s but allocation was %s: %p",
                       getAllocTyName(Kind).data(),
                       getAllocTyName(ATI->Kind).data(), DevicePtr);
#undef DEALLOCATION_ERROR
  }
};

} // namespace plugin
} // namespace target
} // namespace omp
} // namespace llvm

#endif // OFFLOAD_PLUGINS_NEXTGEN_COMMON_ERROR_REPORTING_H
