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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Frontend/OpenMP/OMP.h"
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
#ifdef __clang__ // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=77958
  [[gnu::format(__printf__, 1, 2)]]
#endif
  static void print(const char *Format, ArgsTy &&...Args) {
    raw_fd_ostream OS(STDERR_FILENO, false);
    OS << llvm::format(Format, Args...);
  }

  /// Print \p Format, instantiated with \p Args to stderr, but colored.
  /// TODO: Allow redirection into a file stream.
  template <typename... ArgsTy>
#ifdef __clang__ // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=77958
  [[gnu::format(__printf__, 2, 3)]]
#endif
  static void print(ColorTy Color, const char *Format, ArgsTy &&...Args) {
    raw_fd_ostream OS(STDERR_FILENO, false);
    WithColor(OS, HighlightColor(Color)) << llvm::format(Format, Args...);
  }

  /// Print \p Format, instantiated with \p Args to stderr, but colored and with
  /// a banner.
  /// TODO: Allow redirection into a file stream.
  template <typename... ArgsTy>
#ifdef __clang__ // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=77958
  [[gnu::format(__printf__, 1, 2)]]
#endif
  static void reportError(const char *Format, ArgsTy &&...Args) {
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
            "Last allocation of size %lu for host pointer %p -> device pointer "
            "%p:\n",
            ATI->Size, ATI->HostPtr, ATI->DevicePtr);
    else
      print(BoldLightPurple,
            "Last allocation of size %lu -> device pointer %p:\n", ATI->Size,
            ATI->DevicePtr);
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
        print(
            BoldLightPurple,
            " #%u Prior allocation for host pointer %p -> device pointer %p:\n",
            I, ATI->HostPtr, ATI->DevicePtr);
      else
        print(BoldLightPurple, " #%u Prior allocation -> device pointer %p:\n",
              I, ATI->DevicePtr);
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

  static void reportMemoryAccessError(GenericDeviceTy &Device, void *DevicePtr,
                                      std::string &ErrorStr, bool Abort) {
    reportError(ErrorStr.c_str());

    if (!Device.OMPX_TrackAllocationTraces) {
      print(Yellow, "Use '%s=true' to track device allocations\n",
            Device.OMPX_TrackAllocationTraces.getName().data());
      if (Abort)
        abortExecution();
      return;
    }
    uintptr_t Distance = false;
    auto *ATI =
        Device.getClosestAllocationTraceInfoForAddr(DevicePtr, Distance);
    if (!ATI) {
      print(Cyan,
            "No host-issued allocations; device pointer %p might be "
            "a global, stack, or shared location\n",
            DevicePtr);
      if (Abort)
        abortExecution();
      return;
    }
    if (!Distance) {
      print(Cyan, "Device pointer %p points into%s host-issued allocation:\n",
            DevicePtr, ATI->DeallocationTrace.empty() ? "" : " prior");
      reportAllocationInfo(ATI);
      if (Abort)
        abortExecution();
      return;
    }

    bool IsClose = Distance < (1L << 29L /*512MB=*/);
    print(Cyan,
          "Device pointer %p does not point into any (current or prior) "
          "host-issued allocation%s.\n",
          DevicePtr,
          IsClose ? "" : " (might be a global, stack, or shared location)");
    if (IsClose) {
      print(Cyan,
            "Closest host-issued allocation (distance %" PRIuPTR
            " byte%s; might be by page):\n",
            Distance, Distance > 1 ? "s" : "");
      reportAllocationInfo(ATI);
    }
    if (Abort)
      abortExecution();
  }

  /// Report that a kernel encountered a trap instruction.
  static void reportTrapInKernel(
      GenericDeviceTy &Device, KernelTraceInfoRecordTy &KTIR,
      std::function<bool(__tgt_async_info &)> AsyncInfoWrapperMatcher) {
    assert(AsyncInfoWrapperMatcher && "A matcher is required");

    uint32_t Idx = 0;
    for (uint32_t I = 0, E = KTIR.size(); I < E; ++I) {
      auto KTI = KTIR.getKernelTraceInfo(I);
      if (KTI.Kernel == nullptr)
        break;
      // Skip kernels issued in other queues.
      if (KTI.AsyncInfo && !(AsyncInfoWrapperMatcher(*KTI.AsyncInfo)))
        continue;
      Idx = I;
      break;
    }

    auto KTI = KTIR.getKernelTraceInfo(Idx);
    if (KTI.AsyncInfo && (AsyncInfoWrapperMatcher(*KTI.AsyncInfo))) {
      auto PrettyKernelName =
          llvm::omp::prettifyFunctionName(KTI.Kernel->getName());
      reportError("Kernel '%s'", PrettyKernelName.c_str());
    }
    reportError("execution interrupted by hardware trap instruction");
    if (KTI.AsyncInfo && (AsyncInfoWrapperMatcher(*KTI.AsyncInfo))) {
      if (!KTI.LaunchTrace.empty())
        reportStackTrace(KTI.LaunchTrace);
      else
        print(Yellow, "Use '%s=1' to show the stack trace of the kernel\n",
              Device.OMPX_TrackNumKernelLaunches.getName().data());
    }
    abort();
  }

  /// Report the kernel traces taken from \p KTIR, up to
  /// OFFLOAD_TRACK_NUM_KERNEL_LAUNCH_TRACES many.
  static void reportKernelTraces(GenericDeviceTy &Device,
                                 KernelTraceInfoRecordTy &KTIR) {
    uint32_t NumKTIs = 0;
    for (uint32_t I = 0, E = KTIR.size(); I < E; ++I) {
      auto KTI = KTIR.getKernelTraceInfo(I);
      if (KTI.Kernel == nullptr)
        break;
      ++NumKTIs;
    }
    if (NumKTIs == 0) {
      print(BoldRed, "No kernel launches known\n");
      return;
    }

    uint32_t TracesToShow =
        std::min(Device.OMPX_TrackNumKernelLaunches.get(), NumKTIs);
    if (TracesToShow == 0) {
      if (NumKTIs == 1)
        print(BoldLightPurple, "Display only launched kernel:\n");
      else
        print(BoldLightPurple, "Display last %u kernels launched:\n", NumKTIs);
    } else {
      if (NumKTIs == 1)
        print(BoldLightPurple, "Display kernel launch trace:\n");
      else
        print(BoldLightPurple,
              "Display %u of the %u last kernel launch traces:\n", TracesToShow,
              NumKTIs);
    }

    for (uint32_t Idx = 0, I = 0; I < NumKTIs; ++Idx) {
      auto KTI = KTIR.getKernelTraceInfo(Idx);
      auto PrettyKernelName =
          llvm::omp::prettifyFunctionName(KTI.Kernel->getName());
      if (NumKTIs == 1)
        print(BoldLightPurple, "Kernel '%s'\n", PrettyKernelName.c_str());
      else
        print(BoldLightPurple, "Kernel %d: '%s'\n", I,
              PrettyKernelName.c_str());
      reportStackTrace(KTI.LaunchTrace);
      ++I;
    }

    if (NumKTIs != 1) {
      print(Yellow,
            "Use '%s=<num>' to adjust the number of shown stack traces (%u "
            "now, up to %zu)\n",
            Device.OMPX_TrackNumKernelLaunches.getName().data(),
            Device.OMPX_TrackNumKernelLaunches.get(), KTIR.size());
    }
    // TODO: Let users know how to serialize kernels
  }
};

} // namespace plugin
} // namespace target
} // namespace omp
} // namespace llvm

#endif // OFFLOAD_PLUGINS_NEXTGEN_COMMON_ERROR_REPORTING_H
