//===- OmptProfiler.h - OMPT specific impl of GenericProfilerTy -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// OMPT specific implementation of the GenericProfilerTy class.
// This class uses the already existing implementation of OMPT to invoke
// callbacks and perform tracing.
//
//===----------------------------------------------------------------------===//

#ifndef OFFLOAD_PLUGINS_NEXTGEN_COMMON_OMPT_OMPTPROFILERTY_H
#define OFFLOAD_PLUGINS_NEXTGEN_COMMON_OMPT_OMPTPROFILERTY_H

#include "GenericProfiler.h"

#include "OmptDeviceTracing.h"
#include "OpenMP/OMPT/Callback.h"
#include "Shared/Debug.h"
#include "omp-tools.h"

#include <functional>
#include <tuple>

#pragma push_macro("DEBUG_PREFIX")
#undef DEBUG_PREFIX
#define DEBUG_PREFIX "OMPT"

extern uint64_t getSystemTimestampInNs();

namespace llvm {
namespace omp {
namespace target {
namespace plugin {
struct GenericDeviceTy;
struct GenericPluginTy;
class GenericProfilerTy;

} // namespace plugin

namespace ompt {

// From Callback.h / Callback.cpp
extern bool Initialized;

/**
 * Implements an OMPT backend for the Profiler interface used in the plugins.
 *
 * Forwards / Implements the different generic hooks with OMPT semantics.
 */
class OmptProfilerTy : public plugin::GenericProfilerTy {
public:
  /** Public members **/
  OmptProfilerTy() {

    OmptInitialized.store(false);
    // Bind the callbacks to this device's member functions
#define bindOmptCallback(Name, Type, Code)                                     \
  if (ompt::Initialized && ompt::lookupCallbackByCode) {                       \
    ompt::lookupCallbackByCode((ompt_callbacks_t)(Code),                       \
                               ((ompt_callback_t *)&(Name##_fn)));             \
    DP("class bound %s=%p\n", #Name, ((void *)(uint64_t)Name##_fn));           \
  }

    FOREACH_OMPT_DEVICE_EVENT(bindOmptCallback);
#undef bindOmptCallback

#define bindOmptTracingFunction(FunctionName)                                  \
  if (ompt::Initialized && ompt::lookupDeviceTracingFn) {                      \
    FunctionName##_fn = ompt::lookupDeviceTracingFn(#FunctionName);            \
    DP("device tracing fn bound %s=%p\n", #FunctionName,                       \
       ((void *)(uint64_t)FunctionName##_fn));                                 \
  }

    FOREACH_OMPT_DEVICE_TRACING_FN_COMMON(bindOmptTracingFunction);
#undef bindOmptTracingFunction
  }

  bool isProfilingEnabled() override;

  void handleInit(plugin::GenericDeviceTy *Device,
                  plugin::GenericPluginTy *Plugin) override;

  void handleDeinit(plugin::GenericDeviceTy *Device,
                    plugin::GenericPluginTy *Plugin) override;

  void handleLoadBinary(plugin::GenericDeviceTy *Device,
                        plugin::GenericPluginTy *Plugin,
                        const StringRef InputTgtImage) override;

  void handleDataAlloc(uint64_t StartNanos, uint64_t EndNanos, void *HostPtr,
                       uint64_t Size, void *Data) override;
  void handleDataDelete(uint64_t StartNanos, uint64_t EndNanos, void *TgtPtr,
                        void *Data) override;

  void handlePreKernelLaunch(plugin::GenericDeviceTy *Device,
                             uint32_t NumBlocks[3],
                             __tgt_async_info *AI) override;

  void handleKernelCompletion(uint64_t StartNanos, uint64_t EndNanos,
                              void *Data) override;

  void handleDataTransfer(uint64_t StartNanos, uint64_t EndNanos,
                          void *Data) override;

  void setTimeConversionFactorsImpl(double Slope, double Offset) override;

  void *getProfilerSpecificData() override {
    // TODO: This is ID is not used currently
    uint64_t Id = OmptProfDataId.fetch_add(1);
    {
      std::scoped_lock Lock(ProfilerDataMutex);
      ProfilerData[Id] = std::make_unique<OmptEventInfoTy>();
      return ProfilerData[Id].get();
    }
  }

  void freeProfilerDataEntry(OmptEventInfoTy *DataPtr) {
    std::scoped_lock Lock(ProfilerDataMutex);

    for (auto &Entry : ProfilerData)
      if (Entry.second.get() == DataPtr) {
        ProfilerData.erase(Entry.first);
        break;
      }
  }

private:
  /// Holds a unique ID for each allocation of OmptEventInfoTy
  std::atomic<uint64_t> OmptProfDataId{0};

  /// Holds memory used to store OMPT specific data and pass it down from
  /// libomptarget into the plugins.
  std::map<uint64_t, std::unique_ptr<OmptEventInfoTy>> ProfilerData;

  /// Lock to guard STL ProfilerData map
  std::mutex ProfilerDataMutex;

  /// OMPT callback functions
#define defineOmptCallback(Name, Type, Code) Name##_t Name##_fn = nullptr;
  FOREACH_OMPT_DEVICE_EVENT(defineOmptCallback)
#undef defineOmptCallback

  /// OMPT device tracing functions
#define defineOmptTracingFunction(Name) ompt_interface_fn_t Name##_fn = nullptr;
  FOREACH_OMPT_DEVICE_TRACING_FN_COMMON(defineOmptTracingFunction);
#undef defineOmptTracingFunction

  /// Internal representation for OMPT device (initialize & finalize)
  std::atomic<bool> OmptInitialized;
};
} // namespace ompt
} // namespace target
} // namespace omp
} // namespace llvm

#pragma pop_macro("DEBUG_PREFIX")

#endif
