//===- GenericProfiler.h - GenericProfiler interface for use in Plugins ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The GenericProfiler interface allows to implement profiler logic for various
// backends, such as OMPT or other tracing mechanisms.
// This enables the plugins to be agnostic of the actual high-level language
// that is implemented.
//
//===----------------------------------------------------------------------===//

#ifndef OFFLOAD_PLUGINS_NEXTGEN_COMMON_INCLUDE_GENERICPROFILER_H
#define OFFLOAD_PLUGINS_NEXTGEN_COMMON_INCLUDE_GENERICPROFILER_H

#include "Shared/APITypes.h"

#include <cstdint>
#include <functional>
#include <tuple>

namespace llvm {
namespace omp {
namespace target {
namespace plugin {

struct GenericDeviceTy;
struct GenericPluginTy;
class GenericProfilerTy;

template <typename FunT, typename... ArgsT, size_t... IdxSequence>
void callViaIndexSeq(FunT F, GenericProfilerTy *P, uint64_t StartNanos,
                     uint64_t EndNanos, std::tuple<ArgsT...> Args,
                     std::index_sequence<IdxSequence...>) {
  F(P, StartNanos, EndNanos, std::get<IdxSequence>(Args)...);
}

template <typename FunT, typename... ArgsT>
void callViaUnpack(FunT F, GenericProfilerTy *P, uint64_t StartNanos,
                   uint64_t EndNanos, std::tuple<ArgsT...> Tup) {
  callViaIndexSeq(F, P, StartNanos, EndNanos, Tup,
                  std::index_sequence_for<ArgsT...>{});
}

/***
 * Abstraction layer to implement different profiler backends.
 *
 * The plugins call into the GenericProfilerTy to handle the specific events
 * with whatever specific backend was instantiated. For now, the supported
 * backends are limited to an OMPT implementation.
 */
class GenericProfilerTy {
public:
  GenericProfilerTy() = default;
  virtual ~GenericProfilerTy() = default;

  /// Obtain a pointer to profiler-specific data, if any.
  virtual void *getProfilerSpecificData() { return nullptr; }

  virtual bool isProfilingEnabled() { return false; }

  /// Set the factors which are used to interpolate the device clock compared to
  /// the host clock. This follows a simple linear interpolation: Slope * <time>
  /// + Offset.
  void setTimeConversionFactors(double Slope, double Offset) {
    HostToDeviceSlope = Slope;
    HostToDeviceOffset = Offset;
    setTimeConversionFactorsImpl(HostToDeviceSlope, HostToDeviceOffset);
  }

  /// Hook that is called when the plugin is initialized.
  virtual void handleInit(GenericDeviceTy *Device, GenericPluginTy *Plugin) {}

  /// Hook that is called when the plugin is de-initialized.
  virtual void handleDeinit(GenericDeviceTy *Device, GenericPluginTy *Plugin) {}

  /// Hook that is called when the device image is loaded.
  virtual void handleLoadBinary(GenericDeviceTy *Device,
                                GenericPluginTy *Plugin,
                                const StringRef InputTgtImage) {}

  /// Hook that is called when memory is allcated on the device.
  virtual void handleDataAlloc(uint64_t StartNanos, uint64_t EndNanos,
                               void *HostPtr, uint64_t Size, void *Data) {}

  /// Hook that is called when memory is freed on the device.
  virtual void handleDataDelete(uint64_t StartNanos, uint64_t EndNanos,
                                void *TgtPtr, void *Data) {}

  /// TODO: Currently this is done when "generically" launcing a kernel. Should
  /// this instead be part of the launchImpl? For OMPT, I think it is generally
  /// required to have a "pre-launch" hook, and a "post-launch" hook.
  virtual void handlePreKernelLaunch(GenericDeviceTy *Device,
                                     uint32_t NumBlocks[3],
                                     __tgt_async_info *AI) {}

  /// Hook that is called when the kernel is finished to extract he specific
  /// timing info for that kernel execution.
  virtual void handleKernelCompletion(uint64_t StartNanos, uint64_t EndNanos,
                                      void *Data) {}

  /// Hook that is called when a data transfer happens to extract timing info
  /// for that transfer.
  virtual void handleDataTransfer(uint64_t StartNanos, uint64_t EndNanos,
                                  void *Data) {}

  /// Allow factors for time conversion between host and device.
  virtual void setTimeConversionFactorsImpl(double Slope, double Offset) {
    // Empty function as default implementation
  }

  /// This part of the Profiler provides measurement RAII style functionality.
  /// Use the `getXYZ` functions to obtain a handle, which will start timing on
  /// construction and stop timing on destruction.
  template <typename FnT, typename... ArgsT> class ProfTimerTy {
  public:
    ProfTimerTy(FnT &&F, GenericProfilerTy *P, GenericDeviceTy *D, ArgsT... As)
        : Fun(F), Prof(P), Dev(D), Args(As...) {
      assert(Prof && "GenericProfilerTy is null");
      assert(Dev && "GenericDeviceTy is null");
      if (Prof)
        StartTime = Prof->getDeviceTimeStamp(Dev);
    }

    ~ProfTimerTy() {
      assert(Prof && "GenericProfilerTy is null");
      assert(Dev && "GenericDeviceTy is null");
      if (Prof) {
        uint64_t EndTime = Prof->getDeviceTimeStamp(Dev);
        callViaUnpack(Fun, Prof, StartTime, EndTime, Args);
      }
    }

  private:
    FnT Fun;
    GenericProfilerTy *Prof;
    GenericDeviceTy *Dev;
    uint64_t StartTime = 0;
    std::tuple<ArgsT...> Args;
  };

  // Deduction guide for ProfTimerTy
  template <typename FnT, typename... ArgsT>
  ProfTimerTy(FnT &&, GenericProfilerTy *, ArgsT...)
      -> ProfTimerTy<FnT, ArgsT...>;

  // Mark friend to allow ProfTimerTy to access private members
  template <typename FnT, typename... ArgsT> friend class ProfTimerTy;

  /// Returns an RAII style timer, which will handle data allocation timing
  [[nodiscard]] auto getScopedDataAllocTimer(GenericDeviceTy *Dev,
                                             void *HostPtr, uint64_t Size,
                                             void *ProfData = nullptr) {
    return ProfTimerTy(
        [](GenericProfilerTy *P, auto... args) {
          assert(P && "P was noll");
          P->handleDataAlloc(args...);
        },
        this, Dev, HostPtr, Size, ProfData);
  }

  /// Returns an RAII style timer, which will handle data deletion timing
  [[nodiscard]] auto getScopedDataDeleteTimer(GenericDeviceTy *Dev,
                                              void *TgtPtr,
                                              void *ProfData = nullptr) {
    return ProfTimerTy(
        [](GenericProfilerTy *P, auto... args) {
          assert(P && "P was null");
          P->handleDataDelete(args...);
        },
        this, Dev, TgtPtr, ProfData);
  }

protected:
  /// Linear factor used in time interpolation
  double HostToDeviceSlope = 1.0;
  /// Scalar offset used in time interpolation
  double HostToDeviceOffset = .0;

private:
  /// Vendor-specific implementation to obtain device time.
  uint64_t getDeviceTimeStamp(GenericDeviceTy *D);
};
} // namespace plugin
} // namespace target
} // namespace omp
} // namespace llvm
  //
#endif
