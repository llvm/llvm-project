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
#include "OpenMP/OMPT/OmptEventInfoTy.h"
#include "Shared/Debug.h"
#include "omp-tools.h"

#include <functional>
#include <memory>
#include <mutex>
#include <tuple>

#pragma push_macro("DEBUG_PREFIX")
#undef DEBUG_PREFIX
#define DEBUG_PREFIX "OMPT"

extern uint64_t getSystemTimestampInNs();

using namespace llvm::omp::target::debug;

namespace llvm {
namespace omp {
namespace target {
namespace plugin {
struct GenericDeviceTy;
class GenericProfilerTy;

} // namespace plugin

namespace ompt {

/**
 * Implements an OMPT backend for the Profiler interface used in the plugins.
 *
 * Forwards / Implements the different generic hooks with OMPT semantics.
 */
class OmptProfilerTy : public plugin::GenericProfilerTy {
public:
  bool isProfilingEnabled() override;

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

  /// Allocate the event info that carries \p Record into the plugins for
  /// asynchronous completion. The profiler owns the returned object and frees
  /// it via freeProfilerDataEntry once the plugin completed the record.
  OmptEventInfoTy *trackAsyncRecord(ompt_record_ompt_t *Record) {
    // TODO: This is ID is not used currently
    uint64_t Id = OmptProfDataId.fetch_add(1);
    std::scoped_lock<std::mutex> Lock(ProfilerDataMutex);
    auto &Info = ProfilerData[Id];
    Info = std::make_unique<OmptEventInfoTy>();
    Info->TraceRecord = Record;
    Info->NumTeams = 0;
    return Info.get();
  }

  void freeProfilerDataEntry(OmptEventInfoTy *DataPtr) {
    std::scoped_lock<std::mutex> Lock(ProfilerDataMutex);

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
};

/// Process-wide OMPT profiler owned by libomptarget; nullptr before it has
/// been created.
OmptProfilerTy *getOmptProfiler();
} // namespace ompt
} // namespace target
} // namespace omp
} // namespace llvm

#pragma pop_macro("DEBUG_PREFIX")

#endif
