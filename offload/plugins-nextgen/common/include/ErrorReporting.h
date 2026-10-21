//===- ErrorReporting.h - Detect and triage offload errors ----------- c++
//-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Diagnostic entry points for offload error detection.
// Nothing here formats user-visible messages beyond the one-line summary of
// what went wrong.
//
//===----------------------------------------------------------------------===//

#ifndef OFFLOAD_PLUGINS_NEXTGEN_COMMON_ERROR_REPORTING_H
#define OFFLOAD_PLUGINS_NEXTGEN_COMMON_ERROR_REPORTING_H

#include "PluginInterface.h"
#include "Shared/EnvironmentVar.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <functional>
#include <string>

namespace llvm {
namespace omp {
namespace target {
namespace plugin {

/// Builds an ol_diagnostic_t and owns the storage its pointers refer to, so the
/// diagnostic is only usable while the builder is alive.
class DiagnosticBuilderTy {
public:
  explicit DiagnosticBuilderTy(ol_diagnostic_kind_t Kind) {
    Diag.Kind = Kind;
    // The strings in a diagnostic are never null, only possibly empty.
    Diag.Summary = "";
    Diag.OffendingStackTrace = "";
  }

  /// The diagnostic being built. Its pointer fields are owned by this object.
  ol_diagnostic_t Diag{};

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgcc-compat"
#pragma clang diagnostic ignored "-Wformat-security"
#endif
  /// Set the one-line description of the problem to \p Format, instantiated
  /// with \p Args.
  template <typename... ArgsTy>
#ifdef __clang__ // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=77958
  [[gnu::format(__printf__, 2, 3)]]
#endif
  void setSummary(const char *Format, ArgsTy &&...Args) {
    SmallString<128> Summary;
    raw_svector_ostream OS(Summary);
    OS << llvm::format(Format, Args...);
    Diag.Summary = store(Summary);
  }
#ifdef __clang__
#pragma clang diagnostic pop
#endif

  /// Set the host stack trace at the point of detection.
  void setOffendingStackTrace(StringRef StackTrace) {
    Diag.OffendingStackTrace = store(StackTrace);
  }

  void setTracking(GenericDeviceTy &Device) {
    // The names are string literals, so they can be used as C strings.
    Diag.Tracking.AllocationTracesEnabled = Device.OMPX_TrackAllocationTraces;
    Diag.Tracking.AllocationTracesEnvar =
        Device.OMPX_TrackAllocationTraces.getName().data();
    Diag.Tracking.KernelLaunchesEnvar =
        Device.OMPX_TrackNumKernelLaunches.getName().data();
    Diag.Tracking.KernelLaunchesRequested =
        Device.OMPX_TrackNumKernelLaunches.get();
  }

  ol_pointer_info_t &setPointer(void *DevicePtr) {
    Pointer = ol_pointer_info_t{};
    Pointer.DevicePtr = DevicePtr;
    Pointer.Relation = OL_POINTER_RELATION_NO_ALLOCATIONS;
    Diag.Pointer = &Pointer;
    return Pointer;
  }

  void setAllocation(AllocationTraceInfoTy *ATI) {
    assert(Diag.Pointer && "No pointer to attach the allocation to");
    fill(Allocation, ATI);

    for (auto *Prior = ATI->LastAllocationInfo; Prior;
         Prior = Prior->LastAllocationInfo)
      fill(PriorAllocations.emplace_back(), Prior);

    Allocation.Prior =
        PriorAllocations.empty() ? nullptr : PriorAllocations.data();
    Allocation.NumPrior = PriorAllocations.size();
    Pointer.Allocation = &Allocation;
  }

  void addKernelTrace(const KernelTraceInfoTy &KTI, bool IsCulprit) {
    ol_kernel_launch_info_t &Info = KernelTraces.emplace_back();
    // The name belongs to the kernel, which outlives the diagnostic.
    Info.Name = KTI.Kernel->getName();
    Info.LaunchTrace = store(KTI.LaunchTrace);
    Info.IsCulprit = IsCulprit;

    Diag.KernelTraces = KernelTraces.data();
    Diag.NumKernelTraces = KernelTraces.size();
  }

private:
  /// Keep a copy of \p Str and return it as a C string. The copies live until
  /// the builder dies, and StringSaver null-terminates them.
  const char *store(StringRef Str) { return Strings.save(Str).data(); }

  /// Copy the parts of \p ATI the messaging side can report into \p Info.
  void fill(ol_allocation_info_t &Info, AllocationTraceInfoTy *ATI) {
    Info.AllocationTrace = store(ATI->AllocationTrace);
    Info.DeallocationTrace = store(ATI->DeallocationTrace);
    Info.DevicePtr = ATI->DevicePtr;
    Info.Size = ATI->Size;
    Info.Prior = nullptr;
    Info.NumPrior = 0;
  }

  /// Storage for the strings the diagnostic points into.
  BumpPtrAllocator StringAlloc;
  StringSaver Strings{StringAlloc};

  ol_pointer_info_t Pointer{};
  ol_allocation_info_t Allocation{};
  SmallVector<ol_allocation_info_t> PriorAllocations;
  SmallVector<ol_kernel_launch_info_t> KernelTraces;
};

/// Hand \p Diag to \p Handler, falling back to the built-in minimal reporting
/// if there is no handler or if the handler asks for it, then end the
/// execution.
[[noreturn]] void emitDiagnostic(const ol_diagnostic_handler_t *Handler,
                                 const ol_diagnostic_t &Diag);

class ErrorReporter {

  /// Classify how \p DevicePtr relates to the allocations known to \p Device.
  static void classifyPointer(GenericDeviceTy &Device, void *DevicePtr,
                              DiagnosticBuilderTy &Builder) {
    ol_pointer_info_t &Pointer = Builder.setPointer(DevicePtr);

    uintptr_t Distance = 0;
    auto *ATI =
        Device.getClosestAllocationTraceInfoForAddr(DevicePtr, Distance);
    if (!ATI) {
      Pointer.Relation = OL_POINTER_RELATION_NO_ALLOCATIONS;
      return;
    }

    if (!Distance) {
      Pointer.Relation = ATI->DeallocationTrace.empty()
                             ? OL_POINTER_RELATION_INSIDE
                             : OL_POINTER_RELATION_INSIDE_PRIOR;
      Builder.setAllocation(ATI);
      return;
    }

    bool IsClose = Distance < (1L << 29L /*512MB=*/);
    Pointer.Distance = Distance;
    if (!IsClose) {
      Pointer.Relation = OL_POINTER_RELATION_FAR;
      return;
    }

    Pointer.Relation = OL_POINTER_RELATION_NEAR;
    Builder.setAllocation(ATI);
  }

  /// Report the kernel launches recorded in \p KTIR, most recent first.
  static void collectKernelTraces(KernelTraceInfoRecordTy &KTIR,
                                  DiagnosticBuilderTy &Builder) {
    Builder.Diag.Tracking.KernelLaunchesKnown = 1;
    Builder.Diag.Tracking.KernelLaunchesCapacity = KTIR.size();

    for (uint32_t I = 0, E = KTIR.size(); I < E; ++I) {
      auto KTI = KTIR.getKernelTraceInfo(I);
      if (KTI.Kernel == nullptr)
        break;
      Builder.addKernelTrace(KTI, /*IsCulprit=*/false);
    }
    Builder.Diag.Tracking.KernelLaunchesAvailable =
        Builder.Diag.NumKernelTraces;
  }

public:
#define DEALLOCATION_DIAGNOSTIC(DiagKind, Format, ...)                         \
  DiagnosticBuilderTy Builder(DiagKind);                                       \
  Builder.Diag.HasDeviceId = 1;                                                \
  Builder.Diag.DeviceId = Device.getDeviceId();                                \
  Builder.setSummary(Format, __VA_ARGS__);                                     \
  Builder.setOffendingStackTrace(StackTrace);                                  \
  if (ATI) {                                                                   \
    Builder.setPointer(DevicePtr).Relation = OL_POINTER_RELATION_INSIDE;       \
    Builder.setAllocation(ATI);                                                \
  }                                                                            \
  emitDiagnostic(Device.getDiagnosticHandler(), Builder.Diag);

  [[noreturn]] static void
  reportDeallocationOfNonAllocatedPtr(GenericDeviceTy &Device, void *DevicePtr,
                                      AllocationTraceInfoTy *ATI,
                                      std::string &StackTrace) {
    DEALLOCATION_DIAGNOSTIC(
        OL_DIAGNOSTIC_KIND_DEALLOCATION_OF_NON_ALLOCATED_PTR,
        "deallocation of non-allocated pointer: %p", DevicePtr);
  }

  [[noreturn]] static void
  reportDeallocationOfDeallocatedPtr(GenericDeviceTy &Device, void *DevicePtr,
                                     AllocationTraceInfoTy *ATI,
                                     std::string &StackTrace) {
    DEALLOCATION_DIAGNOSTIC(OL_DIAGNOSTIC_KIND_DOUBLE_FREE,
                            "double-free of pointer: %p", DevicePtr);
  }

  [[noreturn]] static void reportDeallocationOfWrongPtrKind(
      GenericDeviceTy &Device, void *DevicePtr, StringRef Summary,
      AllocationTraceInfoTy *ATI, std::string &StackTrace) {
    DEALLOCATION_DIAGNOSTIC(OL_DIAGNOSTIC_KIND_DEALLOCATION_OF_WRONG_PTR_KIND,
                            "%s", Summary.str().c_str());
#undef DEALLOCATION_DIAGNOSTIC
  }

  /// Report that the device faulted accessing \p DevicePtr. If \p KTIR is given
  /// the known kernel launches are reported alongside the fault. The fault is
  /// asynchronous: there is no call to attribute it to, it is reported with
  /// the handler of the device it happened on like everything else.
  [[noreturn]] static void
  reportMemoryAccessError(GenericDeviceTy &Device, void *DevicePtr,
                          std::string &ErrorStr,
                          KernelTraceInfoRecordTy *KTIR = nullptr) {
    DiagnosticBuilderTy Builder(OL_DIAGNOSTIC_KIND_MEMORY_ACCESS_FAULT);
    Builder.Diag.HasDeviceId = 1;
    Builder.Diag.DeviceId = Device.getDeviceId();
    Builder.setSummary("%s", ErrorStr.c_str());
    Builder.setTracking(Device);

    if (KTIR)
      collectKernelTraces(*KTIR, Builder);

    // Without allocation tracking there is nothing to say about the pointer
    // beyond its value, the messaging side will suggest enabling tracking.
    if (Device.OMPX_TrackAllocationTraces)
      classifyPointer(Device, DevicePtr, Builder);
    else
      Builder.setPointer(DevicePtr);

    emitDiagnostic(Device.getDiagnosticHandler(), Builder.Diag);
  }

  /// Report that a kernel encountered a trap instruction.
  [[noreturn]] static void reportTrapInKernel(
      GenericDeviceTy &Device, KernelTraceInfoRecordTy &KTIR,
      std::function<bool(__tgt_async_info &)> AsyncInfoWrapperMatcher) {
    assert(AsyncInfoWrapperMatcher && "A matcher is required");

    DiagnosticBuilderTy Builder(OL_DIAGNOSTIC_KIND_TRAP_IN_KERNEL);
    Builder.Diag.HasDeviceId = 1;
    Builder.Diag.DeviceId = Device.getDeviceId();
    Builder.setSummary("%s",
                       "execution interrupted by hardware trap instruction");
    Builder.setTracking(Device);

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
    if (KTI.AsyncInfo && (AsyncInfoWrapperMatcher(*KTI.AsyncInfo)))
      Builder.addKernelTrace(KTI, /*IsCulprit=*/true);

    emitDiagnostic(Device.getDiagnosticHandler(), Builder.Diag);
  }
};

} // namespace plugin
} // namespace target
} // namespace omp
} // namespace llvm

#endif // OFFLOAD_PLUGINS_NEXTGEN_COMMON_ERROR_REPORTING_H
