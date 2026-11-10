//===- DiagnosticPrinter.cpp - OpenMP messages for offload diagnostics ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Handle diagnostic messages from liboffload.
//
//===----------------------------------------------------------------------===//

#include "DiagnosticPrinter.h"

#include "OffloadAPI.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Frontend/OpenMP/OMP.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>

using namespace llvm;
using namespace llvm::omp::target;

namespace {

class DiagnosticPrinterTy {

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

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgcc-compat"
#pragma clang diagnostic ignored "-Wformat-security"
#endif
  /// Print \p Format, instantiated with \p Args to stderr.
  /// TODO: Allow redirection into a file stream.
  template <typename... ArgsTy>
#ifdef __clang__ // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=77958
  [[gnu::format(__printf__, 1, 2)]]
#endif
  static void print(const char *Format, ArgsTy &&...Args) {
    auto &OS = llvm::errs();
    OS << llvm::format(Format, Args...);
  }

  /// Print \p Format, instantiated with \p Args to stderr, but colored.
  /// TODO: Allow redirection into a file stream.
  template <typename... ArgsTy>
#ifdef __clang__ // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=77958
  [[gnu::format(__printf__, 2, 3)]]
#endif
  static void print(ColorTy Color, const char *Format, ArgsTy &&...Args) {
    auto &OS = llvm::errs();
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
#ifdef __clang__
#pragma clang diagnostic pop
#endif

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

  /// Report information about an allocation, and the prior allocations at the
  /// same base pointer.
  static void reportAllocationInfo(const ol_allocation_info_t &Alloc) {
    if (*Alloc.DeallocationTrace) {
      print(BoldLightPurple, "Last deallocation:\n");
      reportStackTrace(Alloc.DeallocationTrace);
    }

    if (Alloc.HostPtr)
      print(BoldLightPurple,
            "Last allocation of size %lu for host pointer %p -> device pointer "
            "%p:\n",
            Alloc.Size, Alloc.HostPtr, Alloc.DevicePtr);
    else
      print(BoldLightPurple,
            "Last allocation of size %lu -> device pointer %p:\n", Alloc.Size,
            Alloc.DevicePtr);
    reportStackTrace(Alloc.AllocationTrace);
    if (!Alloc.NumPrior)
      return;

    print(BoldLightPurple, "Prior allocations with the same base pointer:");
    for (unsigned I = 0, E = Alloc.NumPrior; I < E; ++I) {
      const auto &Prior = Alloc.Prior[I];
      print("\n");
      print(BoldLightPurple, " #%u Prior deallocation of size %lu:\n", I,
            Prior.Size);
      reportStackTrace(Prior.DeallocationTrace);
      if (Prior.HostPtr)
        print(
            BoldLightPurple,
            " #%u Prior allocation for host pointer %p -> device pointer %p:\n",
            I, Prior.HostPtr, Prior.DevicePtr);
      else
        print(BoldLightPurple, " #%u Prior allocation -> device pointer %p:\n",
              I, Prior.DevicePtr);
      reportStackTrace(Prior.AllocationTrace);
    }
  }

  /// Report a deallocation error and the history of the pointer involved.
  static void reportDeallocationError(const ol_diagnostic_t &Diag) {
    reportError(Diag.Summary);
    reportStackTrace(Diag.OffendingStackTrace);
    if (Diag.Pointer && Diag.Pointer->Allocation)
      reportAllocationInfo(*Diag.Pointer->Allocation);
  }

  /// Report the kernel traces in \p Diag, up to
  /// OFFLOAD_TRACK_NUM_KERNEL_LAUNCH_TRACES many.
  static void reportKernelTraces(const ol_diagnostic_t &Diag) {
    uint32_t NumKTIs = Diag.NumKernelTraces;
    if (NumKTIs == 0) {
      print(BoldRed, "No kernel launches known\n");
      return;
    }

    uint32_t TracesToShow =
        std::min(Diag.Tracking.KernelLaunchesRequested, NumKTIs);
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

    for (uint32_t I = 0; I < NumKTIs; ++I) {
      const auto &KTI = Diag.KernelTraces[I];
      auto PrettyKernelName = llvm::omp::prettifyFunctionName(KTI.Name);
      if (NumKTIs == 1)
        print(BoldLightPurple, "Kernel '%s'\n", PrettyKernelName.c_str());
      else
        print(BoldLightPurple, "Kernel %d: '%s'\n", I,
              PrettyKernelName.c_str());
      reportStackTrace(KTI.LaunchTrace);
    }

    if (NumKTIs != 1) {
      print(Yellow,
            "Use '%s=<num>' to adjust the number of shown stack traces (%u "
            "now, up to %zu)\n",
            Diag.Tracking.KernelLaunchesEnvar,
            Diag.Tracking.KernelLaunchesRequested,
            size_t(Diag.Tracking.KernelLaunchesCapacity));
    }
    // TODO: Let users know how to serialize kernels
  }

  /// Report a device memory access fault and what is known about the faulting
  /// pointer.
  static void reportMemoryAccessError(const ol_diagnostic_t &Diag) {
    if (Diag.Tracking.KernelLaunchesKnown)
      reportKernelTraces(Diag);

    reportError(Diag.Summary);

    if (!Diag.Tracking.AllocationTracesEnabled) {
      print(Yellow, "Use '%s=true' to track device allocations\n",
            Diag.Tracking.AllocationTracesEnvar);
      return;
    }

    if (!Diag.Pointer)
      return;

    const ol_pointer_info_t &Pointer = *Diag.Pointer;
    switch (Pointer.Relation) {
    case OL_POINTER_RELATION_NO_ALLOCATIONS:
      print(Cyan,
            "No host-issued allocations; device pointer %p might be "
            "a global, stack, or shared location\n",
            Pointer.DevicePtr);
      return;
    case OL_POINTER_RELATION_INSIDE:
    case OL_POINTER_RELATION_INSIDE_PRIOR:
      print(Cyan, "Device pointer %p points into%s host-issued allocation:\n",
            Pointer.DevicePtr,
            Pointer.Relation == OL_POINTER_RELATION_INSIDE ? "" : " prior");
      if (Pointer.Allocation)
        reportAllocationInfo(*Pointer.Allocation);
      return;
    case OL_POINTER_RELATION_NEAR:
    case OL_POINTER_RELATION_FAR: {
      bool IsClose = Pointer.Relation == OL_POINTER_RELATION_NEAR;
      print(Cyan,
            "Device pointer %p does not point into any (current or prior) "
            "host-issued allocation%s.\n",
            Pointer.DevicePtr,
            IsClose ? "" : " (might be a global, stack, or shared location)");
      if (!IsClose)
        return;
      print(Cyan,
            "Closest host-issued allocation (distance %" PRIuPTR
            " byte%s; might be by page):\n",
            Pointer.Distance, Pointer.Distance > 1 ? "s" : "");
      if (Pointer.Allocation)
        reportAllocationInfo(*Pointer.Allocation);
      return;
    }
    case OL_POINTER_RELATION_LAST:
    case OL_POINTER_RELATION_FORCE_UINT32:
      // Not real relations.
      return;
    }
  }

  /// Report that a kernel encountered a trap instruction.
  static void reportTrapInKernel(const ol_diagnostic_t &Diag) {
    const ol_kernel_launch_info_t *Culprit =
        Diag.NumKernelTraces ? &Diag.KernelTraces[0] : nullptr;
    assert((!Culprit || Culprit->IsCulprit) &&
           "Expected the reported kernel to be the culprit");

    if (Culprit) {
      auto PrettyKernelName = llvm::omp::prettifyFunctionName(Culprit->Name);
      reportError("Kernel '%s'", PrettyKernelName.c_str());
    }
    reportError(Diag.Summary);
    if (!Culprit)
      return;

    if (*Culprit->LaunchTrace)
      reportStackTrace(Culprit->LaunchTrace);
    else
      print(Yellow, "Use '%s=1' to show the stack trace of the kernel\n",
            Diag.Tracking.KernelLaunchesEnvar);
  }

public:
  /// ol_diagnostic_cb_t entry point.
  static void report(const ol_diagnostic_t *Diag, void * /*UserData*/,
                     ol_diagnostic_response_t *Response) {
    Response->Reported = 1;
    switch (Diag->Kind) {
    case OL_DIAGNOSTIC_KIND_DEALLOCATION_OF_NON_ALLOCATED_PTR:
    case OL_DIAGNOSTIC_KIND_DOUBLE_FREE:
    case OL_DIAGNOSTIC_KIND_DEALLOCATION_OF_WRONG_PTR_KIND:
      reportDeallocationError(*Diag);
      return;
    case OL_DIAGNOSTIC_KIND_MEMORY_ACCESS_FAULT:
      reportMemoryAccessError(*Diag);
      return;
    case OL_DIAGNOSTIC_KIND_TRAP_IN_KERNEL:
      reportTrapInKernel(*Diag);
      return;
    case OL_DIAGNOSTIC_KIND_LAST:
    case OL_DIAGNOSTIC_KIND_FORCE_UINT32:
      llvm_unreachable("Not a real diagnostic kind");
    }
  }
};

} // namespace

const ol_diagnostic_handler_t *llvm::omp::target::getDiagnosticPrinter() {
  static const ol_diagnostic_handler_t ThePrinter = {
      &DiagnosticPrinterTy::report,
      /*UserData=*/nullptr};
  return &ThePrinter;
}
