//===----RTLs/amdgpu/src/rtl.cpp - Target RTLs Implementation ----- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RTL NextGen for AMDGPU machine
//
//===----------------------------------------------------------------------===//

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <functional>
#include <mutex>
#include <string>
#include <sys/time.h>
#include <system_error>
#include <type_traits>
#include <unistd.h>
#include <unordered_map>
#include <variant>

#include "OmptCommonDefs.h"

#include "OpenMP/OMPT/Interface.h"
#include "ErrorReporting.h"
#include "Shared/APITypes.h"
#include "Shared/Debug.h"
#include "Shared/Environment.h"
#include "Shared/Utils.h"
#include "Utils/ELF.h"

#include "GlobalHandler.h"
#include "OpenMP/OMPT/Callback.h"
#include "PluginInterface.h"
#include "UtilitiesRTL.h"
#include "omptarget.h"

#include "print_tracing.h"

#include "memtype.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Frontend/OpenMP/OMPGridValues.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"

#if !defined(__BYTE_ORDER__) || !defined(__ORDER_LITTLE_ENDIAN__) ||           \
    !defined(__ORDER_BIG_ENDIAN__)
#error "Missing preprocessor definitions for endianness detection."
#endif

// The HSA headers require these definitions.
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
#define LITTLEENDIAN_CPU
#elif defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
#define BIGENDIAN_CPU
#endif

#if defined(__has_include)
#if __has_include("hsa.h")
#include "hsa.h"
#include "hsa_ext_amd.h"
#elif __has_include("hsa/hsa.h")
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#endif
#else
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#endif

using namespace llvm::omp::target;
using namespace llvm::omp::xteam_red;

// AMDGPU-specific, so not using the common ones from the device independent
// includes.
#ifdef OMPT_SUPPORT
#define OMPT_IF_TRACING_OR_ENV_VAR_ENABLED(stmts)                              \
  do {                                                                         \
    if (llvm::omp::target::ompt::TracingActive || OMPX_EnableQueueProfiling) { \
      stmts                                                                    \
    }                                                                          \
  } while (0)
#else
#define OMPT_IF_TRACING_OR_ENV_VAR_ENABLED(stmts)                              \
  do {                                                                         \
    if (OMPX_EnableQueueProfiling) {                                           \
      stmts                                                                    \
    }                                                                          \
  } while (0)
#endif

#ifdef OMPT_SUPPORT
#include "OmptDeviceTracing.h"
#include <omp-tools.h>

extern void ompt::setOmptTimestamp(uint64_t Start, uint64_t End);
extern void ompt::setOmptHostToDeviceRate(double Slope, double Offset);

/// HSA system clock frequency
double TicksToTime = 1.0;

/// Forward declare
namespace llvm {
namespace omp {
namespace target {
namespace plugin {

struct AMDGPUSignalTy;
/// Use to transport information to OMPT timing functions.
struct OmptKernelTimingArgsAsyncTy {
  hsa_agent_t Agent;
  AMDGPUSignalTy *Signal;
  double TicksToTime;
  std::unique_ptr<ompt::OmptEventInfoTy> OmptEventInfo;
};

/// Get OmptKernelTimingArgsAsyncTy from the void * used in the action
/// functions.
static OmptKernelTimingArgsAsyncTy *getOmptTimingsArgs(void *Data);

/// Returns the pair of <start, end> time for a kernel
static std::pair<uint64_t, uint64_t>
getKernelStartAndEndTime(const OmptKernelTimingArgsAsyncTy *Args);

/// Returns the pair of <start, end> time for a data transfer
static std::pair<uint64_t, uint64_t>
getCopyStartAndEndTime(const OmptKernelTimingArgsAsyncTy *Args);

/// Obtain the timing info and call the RegionInterface callback for the
/// asynchronous trace records.
static Error timeDataTransferInNsAsync(void *Data) {
  auto Args = getOmptTimingsArgs(Data);

  auto [Start, End] = getCopyStartAndEndTime(Args);

  auto OmptEventInfo = *Args->OmptEventInfo.get();
  auto RIFunc = std::get<2>(OmptEventInfo.RIFunction);
  std::invoke(RIFunc, OmptEventInfo.RegionInterface, OmptEventInfo.TraceRecord,
              Start, End);

  return Plugin::success();
}

/// Print out some debug info for the OmptEventInfoTy
static void printOmptEventInfoTy(ompt::OmptEventInfoTy &OmptEventInfo) {
  DP("OMPT-Async Trace Info: NumTeams %lu, TR %p, "
     "RegionInterface %p\n",
     OmptEventInfo.NumTeams, OmptEventInfo.TraceRecord,
     OmptEventInfo.RegionInterface);
}

/// Returns a pointer to an OmptEventInfoTy object to be used for OMPT tracing
/// or nullptr. It is the caller's duty to free the returned pointer when no
/// longer needed.
static std::unique_ptr<ompt::OmptEventInfoTy>
getOrNullOmptEventInfo(AsyncInfoWrapperTy &AsyncInfoWrapper) {
  __tgt_async_info *AI = AsyncInfoWrapper;
  if (!AI || !AI->OmptEventInfo)
    return nullptr;

  // We need to copy the content of the OmptEventInfo object to persist it
  // between multiple async operations.
  auto LocalOmptEventInfo =
      std::make_unique<ompt::OmptEventInfoTy>(*AI->OmptEventInfo);
  DP("OMPT-Async: Two times printing\n");
  printOmptEventInfoTy(*AI->OmptEventInfo);
  printOmptEventInfoTy(*LocalOmptEventInfo);
  return LocalOmptEventInfo;
}

} // namespace plugin
} // namespace target
} // namespace omp
} // namespace llvm

/// Enable/disable async copy profiling.
void setOmptAsyncCopyProfile(bool Enable) {
  hsa_status_t Status = hsa_amd_profiling_async_copy_enable(Enable);
  if (Status != HSA_STATUS_SUCCESS)
    DP("Error enabling async copy profiling\n");
}

/// Compute system timestamp conversion factor, modeled after ROCclr.
void setOmptTicksToTime() {
  uint64_t TicksFrequency = 1;
  hsa_status_t Status =
      hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &TicksFrequency);
  if (Status == HSA_STATUS_SUCCESS)
    TicksToTime = (double)1e9 / (double)TicksFrequency;
  else
    DP("Error calling hsa_system_get_info for timestamp frequency\n");
}

/// Get the current HSA-based device timestamp.
uint64_t getSystemTimestampInNs() {
  uint64_t TimeStamp = 0;
  hsa_status_t Status =
      hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP, &TimeStamp);
  if (Status != HSA_STATUS_SUCCESS)
    DP("Error calling hsa_system_get_info for timestamp\n");
  return TimeStamp * TicksToTime;
}

/// @brief Helper to get the host time
/// @return  CLOCK_REALTIME seconds as double
static double getTimeOfDay() {
  double TimeVal = .0;
  struct timeval tval;
  int rc = gettimeofday(&tval, NULL);
  if (rc) {
    // XXX: Error case: What to do?
  } else {
    TimeVal = static_cast<double>(tval.tv_sec) +
              1.0E-06 * static_cast<double>(tval.tv_usec);
  }
  return TimeVal;
}

/// Get the first timepoints on host and device.
void startH2DTimeRate(double *HTime, uint64_t *DTime) {
  *HTime = getTimeOfDay();
  *DTime = getSystemTimestampInNs();
}

/// Get the second timepoints on host and device and compute the rate
/// required for translating device time to host time.
void completeH2DTimeRate(double HostRef1, uint64_t DeviceRef1) {
  double HostRef2 = getTimeOfDay();
  uint64_t DeviceRef2 = getSystemTimestampInNs();
  // Assume host (h) timing is related to device (d) timing as
  // h = m.d + o, where m is the slope and o is the offset.
  // Calculate slope and offset from the two host and device timepoints.
  double HostDiff = HostRef2 - HostRef1;
  uint64_t DeviceDiff = DeviceRef2 - DeviceRef1;
  double Slope = DeviceDiff != 0 ? (HostDiff / DeviceDiff) : HostDiff;
  double Offset = HostRef1 - Slope * DeviceRef1;
  ompt::setOmptHostToDeviceRate(Slope, Offset);
  DP("Translate time Slope: %f Offset: %f\n", Slope, Offset);
}

#else // OMPT_SUPPORT
namespace llvm::omp::target::ompt {
struct OmptEventInfoTy {};
} // namespace llvm::omp::target::ompt
namespace llvm::omp::target::plugin {
static std::unique_ptr<ompt::OmptEventInfoTy>
getOrNullOmptEventInfo(AsyncInfoWrapperTy &AsyncInfoWrapper) {
  return nullptr;
}
} // namespace llvm::omp::target::plugin
#endif

namespace llvm {
namespace omp {
namespace target {
namespace plugin {

extern "C" {
uint64_t hostrpc_assign_buffer(hsa_agent_t Agent, hsa_queue_t *ThisQ,
                               uint32_t DeviceId,
                               hsa_amd_memory_pool_t HostMemoryPool,
                               hsa_amd_memory_pool_t DevMemoryPool);
hsa_status_t hostrpc_terminate();
__attribute__((weak)) hsa_status_t hostrpc_terminate() {
  return HSA_STATUS_SUCCESS;
}
__attribute__((weak)) uint64_t hostrpc_assign_buffer(
    hsa_agent_t, hsa_queue_t *, uint32_t DeviceId,
    hsa_amd_memory_pool_t HostMemoryPool, hsa_amd_memory_pool_t DevMemoryPool) {
  // FIXME:THIS SHOULD BE HARD FAIL
  DP("Warning: Attempting to assign hostrpc to device %u, but hostrpc library "
     "missing\n",
     DeviceId);
  return 0;
}
}

/// Forward declarations for all specialized data structures.
struct AMDGPUKernelTy;
struct AMDGPUDeviceTy;
struct AMDGPUPluginTy;
struct AMDGPUStreamTy;
struct AMDGPUEventTy;
struct AMDGPUStreamManagerTy;
struct AMDGPUEventManagerTy;
struct AMDGPUDeviceImageTy;
struct AMDGPUMemoryManagerTy;
struct AMDGPUMemoryPoolTy;

namespace utils {

/// Iterate elements using an HSA iterate function. Do not use this function
/// directly but the specialized ones below instead.
template <typename ElemTy, typename IterFuncTy, typename CallbackTy>
hsa_status_t iterate(IterFuncTy Func, CallbackTy Cb) {
  auto L = [](ElemTy Elem, void *Data) -> hsa_status_t {
    CallbackTy *Unwrapped = static_cast<CallbackTy *>(Data);
    return (*Unwrapped)(Elem);
  };
  return Func(L, static_cast<void *>(&Cb));
}

/// Iterate elements using an HSA iterate function passing a parameter. Do not
/// use this function directly but the specialized ones below instead.
template <typename ElemTy, typename IterFuncTy, typename IterFuncArgTy,
          typename CallbackTy>
hsa_status_t iterate(IterFuncTy Func, IterFuncArgTy FuncArg, CallbackTy Cb) {
  auto L = [](ElemTy Elem, void *Data) -> hsa_status_t {
    CallbackTy *Unwrapped = static_cast<CallbackTy *>(Data);
    return (*Unwrapped)(Elem);
  };
  return Func(FuncArg, L, static_cast<void *>(&Cb));
}

/// Iterate elements using an HSA iterate function passing a parameter. Do not
/// use this function directly but the specialized ones below instead.
template <typename Elem1Ty, typename Elem2Ty, typename IterFuncTy,
          typename IterFuncArgTy, typename CallbackTy>
hsa_status_t iterate(IterFuncTy Func, IterFuncArgTy FuncArg, CallbackTy Cb) {
  auto L = [](Elem1Ty Elem1, Elem2Ty Elem2, void *Data) -> hsa_status_t {
    CallbackTy *Unwrapped = static_cast<CallbackTy *>(Data);
    return (*Unwrapped)(Elem1, Elem2);
  };
  return Func(FuncArg, L, static_cast<void *>(&Cb));
}

/// Iterate agents.
template <typename CallbackTy> Error iterateAgents(CallbackTy Callback) {
  hsa_status_t Status = iterate<hsa_agent_t>(hsa_iterate_agents, Callback);
  return Plugin::check(Status, "Error in hsa_iterate_agents: %s");
}

/// Iterate ISAs of an agent.
template <typename CallbackTy>
Error iterateAgentISAs(hsa_agent_t Agent, CallbackTy Cb) {
  hsa_status_t Status = iterate<hsa_isa_t>(hsa_agent_iterate_isas, Agent, Cb);
  return Plugin::check(Status, "Error in hsa_agent_iterate_isas: %s");
}

/// Iterate memory pools of an agent.
template <typename CallbackTy>
Error iterateAgentMemoryPools(hsa_agent_t Agent, CallbackTy Cb) {
  hsa_status_t Status = iterate<hsa_amd_memory_pool_t>(
      hsa_amd_agent_iterate_memory_pools, Agent, Cb);
  return Plugin::check(Status,
                       "Error in hsa_amd_agent_iterate_memory_pools: %s");
}

extern "C" uint64_t hostrpc_assign_buffer(hsa_agent_t Agent, hsa_queue_t *ThisQ,
                                          uint32_t DeviceId,
                                          hsa_amd_memory_pool_t HostMemoryPool,
                                          hsa_amd_memory_pool_t DevMemoryPool);
extern "C" hsa_status_t hostrpc_terminate();

/// Dispatches an asynchronous memory copy.
/// Enables different SDMA engines for the dispatch in a round-robin fashion.
Error asyncMemCopy(bool UseMultipleSdmaEngines, void *Dst, hsa_agent_t DstAgent,
                   const void *Src, hsa_agent_t SrcAgent, size_t Size,
                   uint32_t NumDepSignals, const hsa_signal_t *DepSignals,
                   hsa_signal_t CompletionSignal) {
  if (!UseMultipleSdmaEngines) {
    hsa_status_t S =
        hsa_amd_memory_async_copy(Dst, DstAgent, Src, SrcAgent, Size,
                                  NumDepSignals, DepSignals, CompletionSignal);
    return Plugin::check(S, "Error in hsa_amd_memory_async_copy: %s");
  }

// This solution is probably not the best
#if !(HSA_AMD_INTERFACE_VERSION_MAJOR >= 1 &&                                  \
      HSA_AMD_INTERFACE_VERSION_MINOR >= 2)
  return Plugin::error("Async copy on selected SDMA requires ROCm 5.7");
#else
  static std::atomic<int> SdmaEngine{1};

  // This atomics solution is probably not the best, but should be sufficient
  // for now.
  // In a worst case scenario, in which threads read the same value, they will
  // dispatch to the same SDMA engine. This may result in sub-optimal
  // performance. However, I think the possibility to be fairly low.
  int LocalSdmaEngine = SdmaEngine.load(std::memory_order_acquire);
  DP("Running Async Copy on SDMA Engine: %i\n", LocalSdmaEngine);
  // This call is only avail in ROCm >= 5.7
  hsa_status_t S = hsa_amd_memory_async_copy_on_engine(
      Dst, DstAgent, Src, SrcAgent, Size, NumDepSignals, DepSignals,
      CompletionSignal, (hsa_amd_sdma_engine_id_t)LocalSdmaEngine,
      /*force_copy_on_sdma=*/true);
  // Increment to use one of two SDMA engines: 0x1, 0x2
  LocalSdmaEngine = (LocalSdmaEngine << 1) % 3;
  SdmaEngine.store(LocalSdmaEngine, std::memory_order_relaxed);

  return Plugin::check(S, "Error in hsa_amd_memory_async_copy_on_engine: %s");
#endif
}

Expected<std::string> getTargetTripleAndFeatures(hsa_agent_t Agent) {
  std::string Target;
  auto Err = utils::iterateAgentISAs(Agent, [&](hsa_isa_t ISA) {
    uint32_t Length;
    hsa_status_t Status;
    Status = hsa_isa_get_info_alt(ISA, HSA_ISA_INFO_NAME_LENGTH, &Length);
    if (Status != HSA_STATUS_SUCCESS)
      return Status;

    llvm::SmallVector<char> ISAName(Length);
    Status = hsa_isa_get_info_alt(ISA, HSA_ISA_INFO_NAME, ISAName.begin());
    if (Status != HSA_STATUS_SUCCESS)
      return Status;

    llvm::StringRef TripleTarget(ISAName.begin(), Length);
    if (TripleTarget.consume_front("amdgcn-amd-amdhsa"))
      Target = TripleTarget.ltrim('-').rtrim('\0').str();
    return HSA_STATUS_SUCCESS;
  });
  if (Err)
    return Err;
  return Target;
}

} // namespace utils

/// Utility class representing generic resource references to AMDGPU resources.
template <typename ResourceTy>
struct AMDGPUResourceRef : public GenericDeviceResourceRef {
  /// The underlying handle type for resources.
  using HandleTy = ResourceTy *;

  /// Create an empty reference to an invalid resource.
  AMDGPUResourceRef() : Resource(nullptr) {}

  /// Create a reference to an existing resource.
  AMDGPUResourceRef(HandleTy Resource) : Resource(Resource) {}

  virtual ~AMDGPUResourceRef() {}

  /// Create a new resource and save the reference. The reference must be empty
  /// before calling to this function.
  Error create(GenericDeviceTy &Device) override;

  /// Destroy the referenced resource and invalidate the reference. The
  /// reference must be to a valid resource before calling to this function.
  Error destroy(GenericDeviceTy &Device) override {
    if (!Resource)
      return Plugin::error("Destroying an invalid resource");

    if (auto Err = Resource->deinit())
      return Err;

    delete Resource;

    Resource = nullptr;
    return Plugin::success();
  }

  /// Get the underlying resource handle.
  operator HandleTy() const { return Resource; }

private:
  /// The handle to the actual resource.
  HandleTy Resource;
};

/// Class holding an HSA memory pool.
struct AMDGPUMemoryPoolTy {
  /// Create a memory pool from an HSA memory pool.
  AMDGPUMemoryPoolTy(hsa_amd_memory_pool_t MemoryPool)
      : MemoryPool(MemoryPool), GlobalFlags(0) {}

  /// Initialize the memory pool retrieving its properties.
  Error init() {
    if (auto Err = getAttr(HSA_AMD_MEMORY_POOL_INFO_SEGMENT, Segment))
      return Err;

    if (auto Err = getAttr(HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, GlobalFlags))
      return Err;

    return Plugin::success();
  }

  /// Getter of the HSA memory pool.
  hsa_amd_memory_pool_t get() const { return MemoryPool; }

  /// Indicate the segment which belongs to.
  bool isGlobal() const { return (Segment == HSA_AMD_SEGMENT_GLOBAL); }
  bool isReadOnly() const { return (Segment == HSA_AMD_SEGMENT_READONLY); }
  bool isPrivate() const { return (Segment == HSA_AMD_SEGMENT_PRIVATE); }
  bool isGroup() const { return (Segment == HSA_AMD_SEGMENT_GROUP); }

  /// Indicate if it is fine-grained memory. Valid only for global.
  bool isFineGrained() const {
    assert(isGlobal() && "Not global memory");
    return (GlobalFlags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED);
  }

  /// Indicate if it is coarse-grained memory. Valid only for global.
  bool isCoarseGrained() const {
    assert(isGlobal() && "Not global memory");
    return (GlobalFlags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED);
  }

  /// Indicate if it supports storing kernel arguments. Valid only for global.
  bool supportsKernelArgs() const {
    assert(isGlobal() && "Not global memory");
    return (GlobalFlags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT);
  }

  /// Allocate memory on the memory pool.
  Error allocate(size_t Size, void **PtrStorage) {
    hsa_status_t Status =
        hsa_amd_memory_pool_allocate(MemoryPool, Size, 0, PtrStorage);
    return Plugin::check(Status, "Error in hsa_amd_memory_pool_allocate: %s");
  }

  /// Return memory to the memory pool.
  Error deallocate(void *Ptr) {
    hsa_status_t Status = hsa_amd_memory_pool_free(Ptr);
    return Plugin::check(Status, "Error in hsa_amd_memory_pool_free: %s");
  }

  /// Returns if the \p Agent can access the memory pool.
  bool canAccess(hsa_agent_t Agent) {
    hsa_amd_memory_pool_access_t Access;
    if (hsa_amd_agent_memory_pool_get_info(
            Agent, MemoryPool, HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &Access))
      return false;
    return Access != HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
  }

  /// Allow the device to access a specific allocation.
  Error enableAccess(void *Ptr, int64_t Size,
                     const llvm::SmallVector<hsa_agent_t> &Agents) const {
#ifdef OMPTARGET_DEBUG
    for (hsa_agent_t Agent : Agents) {
      hsa_amd_memory_pool_access_t Access;
      if (auto Err =
              getAttr(Agent, HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, Access))
        return Err;

      // The agent is not allowed to access the memory pool in any case. Do not
      // continue because otherwise it result in undefined behavior.
      if (Access == HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED)
        return Plugin::error("An agent is not allowed to access a memory pool");
    }
#endif

    // We can access but it is disabled by default. Enable the access then.
    hsa_status_t Status =
        hsa_amd_agents_allow_access(Agents.size(), Agents.data(), nullptr, Ptr);
    return Plugin::check(Status, "Error in hsa_amd_agents_allow_access: %s");
  }

  Error zeroInitializeMemory(void *Ptr, size_t Size) {
    uint64_t Rounded = sizeof(uint32_t) * ((Size + 3) / sizeof(uint32_t));
    hsa_status_t Status =
        hsa_amd_memory_fill(Ptr, 0, Rounded / sizeof(uint32_t));
    return Plugin::check(Status, "Error in hsa_amd_memory_fill: %s");
  }

  /// Get attribute from the memory pool.
  template <typename Ty>
  Error getAttr(hsa_amd_memory_pool_info_t Kind, Ty &Value) const {
    hsa_status_t Status;
    Status = hsa_amd_memory_pool_get_info(MemoryPool, Kind, &Value);
    return Plugin::check(Status, "Error in hsa_amd_memory_pool_get_info: %s");
  }

  template <typename Ty>
  hsa_status_t getAttrRaw(hsa_amd_memory_pool_info_t Kind, Ty &Value) const {
    return hsa_amd_memory_pool_get_info(MemoryPool, Kind, &Value);
  }

  /// Get attribute from the memory pool relating to an agent.
  template <typename Ty>
  Error getAttr(hsa_agent_t Agent, hsa_amd_agent_memory_pool_info_t Kind,
                Ty &Value) const {
    hsa_status_t Status;
    Status =
        hsa_amd_agent_memory_pool_get_info(Agent, MemoryPool, Kind, &Value);
    return Plugin::check(Status,
                         "Error in hsa_amd_agent_memory_pool_get_info: %s");
  }

private:
  /// The HSA memory pool.
  hsa_amd_memory_pool_t MemoryPool;

  /// The segment where the memory pool belongs to.
  hsa_amd_segment_t Segment;

  /// The global flags of memory pool. Only valid if the memory pool belongs to
  /// the global segment.
  uint32_t GlobalFlags;
};

/// Class that implements a memory manager that gets memory from a specific
/// memory pool.
struct AMDGPUMemoryManagerTy : public DeviceAllocatorTy {

  /// Create an empty memory manager.
  AMDGPUMemoryManagerTy(AMDGPUPluginTy &Plugin)
      : Plugin(Plugin), MemoryPool(nullptr), MemoryManager(nullptr) {}

  /// Initialize the memory manager from a memory pool.
  Error init(AMDGPUMemoryPoolTy &MemoryPool) {
    const uint32_t Threshold = 1 << 30;
    this->MemoryManager = new MemoryManagerTy(*this, Threshold);
    this->MemoryPool = &MemoryPool;
    return Plugin::success();
  }

  /// Deinitialize the memory manager and free its allocations.
  Error deinit() {
    assert(MemoryManager && "Invalid memory manager");

    // Delete and invalidate the memory manager. At this point, the memory
    // manager will deallocate all its allocations.
    delete MemoryManager;
    MemoryManager = nullptr;

    return Plugin::success();
  }

  /// Reuse or allocate memory through the memory manager.
  Error allocate(size_t Size, void **PtrStorage) {
    assert(MemoryManager && "Invalid memory manager");
    assert(PtrStorage && "Invalid pointer storage");

    *PtrStorage = MemoryManager->allocate(Size, nullptr);
    if (*PtrStorage == nullptr)
      return Plugin::error("Failure to allocate from AMDGPU memory manager");

    return Plugin::success();
  }

  /// Release an allocation to be reused.
  Error deallocate(void *Ptr) {
    assert(Ptr && "Invalid pointer");

    if (MemoryManager->free(Ptr))
      return Plugin::error("Failure to deallocate from AMDGPU memory manager");

    return Plugin::success();
  }

private:
  /// Allocation callback that will be called once the memory manager does not
  /// have more previously allocated buffers.
  void *allocate(size_t Size, void *HstPtr, TargetAllocTy Kind) override;

  /// Deallocation callack that will be called by the memory manager.
  int free(void *TgtPtr, TargetAllocTy Kind) override {
    if (auto Err = MemoryPool->deallocate(TgtPtr)) {
      consumeError(std::move(Err));
      return OFFLOAD_FAIL;
    }
    return OFFLOAD_SUCCESS;
  }

  /// The underlying plugin that owns this memory manager.
  AMDGPUPluginTy &Plugin;

  /// The memory pool used to allocate memory.
  AMDGPUMemoryPoolTy *MemoryPool;

  /// Reference to the actual memory manager.
  MemoryManagerTy *MemoryManager;
};

/// Class implementing the AMDGPU device images' properties.
struct AMDGPUDeviceImageTy : public DeviceImageTy {
  /// Create the AMDGPU image with the id and the target image pointer.
  AMDGPUDeviceImageTy(int32_t ImageId, GenericDeviceTy &Device,
                      const __tgt_device_image *TgtImage)
      : DeviceImageTy(ImageId, Device, TgtImage) {}

  /// Prepare and load the executable corresponding to the image.
  Error loadExecutable(const AMDGPUDeviceTy &Device);

  /// Unload the executable.
  Error unloadExecutable() {
    hsa_status_t Status = hsa_executable_destroy(Executable);
    if (auto Err = Plugin::check(Status, "Error in hsa_executable_destroy: %s"))
      return Err;

    Status = hsa_code_object_destroy(CodeObject);
    return Plugin::check(Status, "Error in hsa_code_object_destroy: %s");
  }

  /// Get the executable.
  hsa_executable_t getExecutable() const { return Executable; }

  /// Get to Code Object Version of the ELF
  uint16_t getELFABIVersion() const { return ELFABIVersion; }

  /// Find an HSA device symbol by its name on the executable.
  Expected<hsa_executable_symbol_t>
  findDeviceSymbol(GenericDeviceTy &Device, StringRef SymbolName) const;

  /// Get additional info for kernel, e.g., register spill counts
  std::optional<utils::KernelMetaDataTy>
  getKernelInfo(StringRef Identifier) const {
    auto It = KernelInfoMap.find(Identifier);

    if (It == KernelInfoMap.end())
      return {};

    return It->second;
  }

  /// Does device image contain Symbol
  bool hasDeviceSymbol(GenericDeviceTy &Device, StringRef SymbolName) const;

private:
  /// The exectuable loaded on the agent.
  hsa_executable_t Executable;
  hsa_code_object_t CodeObject;
#if SANITIZER_AMDGPU
  hsa_code_object_reader_t CodeObjectReader;
#endif
  StringMap<utils::KernelMetaDataTy> KernelInfoMap;
  uint16_t ELFABIVersion;
};

/// Class implementing the AMDGPU kernel functionalities which derives from the
/// generic kernel class.
struct AMDGPUKernelTy : public GenericKernelTy {
  /// Create an AMDGPU kernel with a name and an execution mode.
  AMDGPUKernelTy(const char *Name, GenericGlobalHandlerTy &Handler)
      : GenericKernelTy(Name),
        OMPX_DisableHostExec("LIBOMPTARGET_DISABLE_HOST_EXEC", false),
        ServiceThreadDeviceBufferGlobal("service_thread_buf", sizeof(uint64_t)),
        HostServiceBufferHandler(Handler) {}

  /// Initialize the AMDGPU kernel.
  Error initImpl(GenericDeviceTy &Device, DeviceImageTy &Image) override {
    AMDGPUDeviceImageTy &AMDImage = static_cast<AMDGPUDeviceImageTy &>(Image);

    // Kernel symbols have a ".kd" suffix.
    std::string KernelName(getName());
    KernelName += ".kd";

    // Find the symbol on the device executable.
    auto SymbolOrErr = AMDImage.findDeviceSymbol(Device, KernelName);
    if (!SymbolOrErr)
      return SymbolOrErr.takeError();

    hsa_executable_symbol_t Symbol = *SymbolOrErr;
    hsa_symbol_kind_t SymbolType;
    hsa_status_t Status;

    // Retrieve different properties of the kernel symbol.
    std::pair<hsa_executable_symbol_info_t, void *> RequiredInfos[] = {
        {HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &SymbolType},
        {HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &KernelObject},
        {HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &ArgsSize},
        {HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &GroupSize},
        {HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK, &DynamicStack},
        {HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &PrivateSize}};

    for (auto &Info : RequiredInfos) {
      Status = hsa_executable_symbol_get_info(Symbol, Info.first, Info.second);
      if (auto Err = Plugin::check(
              Status, "Error in hsa_executable_symbol_get_info: %s"))
        return Err;
    }

    // Make sure it is a kernel symbol.
    if (SymbolType != HSA_SYMBOL_KIND_KERNEL)
      return Plugin::error("Symbol %s is not a kernel function");

    // TODO: Read the kernel descriptor for the max threads per block. May be
    // read from the image.

    // Get ConstWGSize for kernel from image
    ConstWGSize = Device.getDefaultNumThreads();
    std::string WGSizeName(getName());
    WGSizeName += "_wg_size";
    GlobalTy HostConstWGSize(WGSizeName, sizeof(decltype(ConstWGSize)),
                             &ConstWGSize);
    GenericGlobalHandlerTy &GHandler = Device.Plugin.getGlobalHandler();
    if (auto Err =
            GHandler.readGlobalFromImage(Device, AMDImage, HostConstWGSize)) {
      // In case it is not found, we simply stick with the defaults.
      // So we consume the error and print a debug message.
      DP("Could not load %s global from kernel image. Run with %u %u\n",
         WGSizeName.c_str(), PreferredNumThreads, MaxNumThreads);
      consumeError(std::move(Err));
      assert(PreferredNumThreads > 0 && "Prefer more than 0 threads");
      assert(MaxNumThreads > 0 && "MaxNumThreads more than 0 threads");
    } else {
      // Set the number of preferred and max threads to the ConstWGSize to get
      // the exact value for kernel launch. Exception: In generic-spmd mode, we
      // set it to the default blocksize since ConstWGSize may include the
      // master thread which is not required.
      PreferredNumThreads =
          getExecutionModeFlags() == OMP_TGT_EXEC_MODE_GENERIC_SPMD
              ? Device.getDefaultNumThreads()
              : ConstWGSize;
      MaxNumThreads = ConstWGSize;
    }

    ImplicitArgsSize =
        utils::getImplicitArgsSize(AMDImage.getELFABIVersion()); // COV 5 patch

    DP("ELFABIVersion: %d\n", AMDImage.getELFABIVersion());

    // Get additional kernel info read from image
    KernelInfo = AMDImage.getKernelInfo(getName());
    if (!KernelInfo.has_value())
      INFO(OMP_INFOTYPE_PLUGIN_KERNEL, Device.getDeviceId(),
           "Could not read extra information for kernel %s.", getName());

    NeedsHostServices =
        AMDImage.hasDeviceSymbol(Device, "__needs_host_services");
    if (NeedsHostServices && !OMPX_DisableHostExec) {
      // GenericGlobalHandlerTy * GHandler = Plugin::createGlobalHandler();
      if (auto Err = HostServiceBufferHandler.getGlobalMetadataFromDevice(
              Device, AMDImage, ServiceThreadDeviceBufferGlobal))
        return Err;
    }

    return Plugin::success();
  }

  /// Launch the AMDGPU kernel function.
  Error launchImpl(GenericDeviceTy &GenericDevice, uint32_t NumThreads,
                   uint64_t NumBlocks, KernelArgsTy &KernelArgs,
                   KernelLaunchParamsTy LaunchParams,
                   AsyncInfoWrapperTy &AsyncInfoWrapper) const override;

  /// Print more elaborate kernel launch info for AMDGPU
  Error printLaunchInfoDetails(GenericDeviceTy &GenericDevice,
                               KernelArgsTy &KernelArgs, uint32_t NumThreads,
                               uint64_t NumBlocks) const override;
  /// Print the "old" AMD KernelTrace single-line format
  void printAMDOneLineKernelTrace(GenericDeviceTy &GenericDevice,
                                  KernelArgsTy &KernelArgs, uint32_t NumThreads,
                                  uint64_t NumBlocks) const;
  /// Get group and private segment kernel size.
  uint32_t getGroupSize() const { return GroupSize; }
  uint32_t getPrivateSize() const { return PrivateSize; }
  uint16_t getConstWGSize() const { return ConstWGSize; }

  /// Get the HSA kernel object representing the kernel function.
  uint64_t getKernelObject() const { return KernelObject; }

  /// Get the size of implicitargs based on the code object version
  /// @return 56 for cov4 and 256 for cov5
  uint32_t getImplicitArgsSize() const { return ImplicitArgsSize; }

  /// Indicates whether or not we need to set up our own private segment size.
  bool usesDynamicStack() const { return DynamicStack; }

  /// Envar to disable host-exec thread creation.
  BoolEnvar OMPX_DisableHostExec;

private:
  /// The kernel object to execute.
  uint64_t KernelObject;

  /// The args, group and private segments sizes required by a kernel instance.
  uint32_t ArgsSize;
  uint32_t GroupSize;
  uint32_t PrivateSize;
  bool DynamicStack;

  /// The size of implicit kernel arguments.
  uint32_t ImplicitArgsSize;

  /// Additional Info for the AMD GPU Kernel
  std::optional<utils::KernelMetaDataTy> KernelInfo;
  /// CodeGen generate WGSize
  uint16_t ConstWGSize;

  /// Indicate whether this Kernel requires host services
  bool NeedsHostServices;

  /// Global for host service device thread buffer
  GlobalTy ServiceThreadDeviceBufferGlobal;

  /// Global handler for hostservices buffer
  GenericGlobalHandlerTy &HostServiceBufferHandler;

  /// Lower number of threads if tripcount is low. This should produce
  /// a larger number of teams if allowed by other constraints.
  std::pair<bool, uint32_t> adjustNumThreadsForLowTripCount(
      GenericDeviceTy &GenericDevice, uint32_t BlockSize,
      uint64_t LoopTripCount, uint32_t ThreadLimitClause[3]) const override {
    uint32_t NumThreads = BlockSize;

    // If there is an override already, do nothing. Note the different
    // default for Xteam Reductions.
    if (!isXTeamReductionsMode() &&
        NumThreads != GenericDevice.getDefaultNumThreads() &&
        NumThreads != ConstWGSize)
      return std::make_pair(false, NumThreads);

    if (isXTeamReductionsMode() &&
        NumThreads != llvm::omp::xteam_red::DefaultBlockSize &&
        NumThreads != ConstWGSize)
      return std::make_pair(false, NumThreads);

    // If tripcount not set or not low, do nothing.
    if ((LoopTripCount == 0) ||
        (LoopTripCount > GenericDevice.getOMPXLowTripCount()))
      return std::make_pair(false, NumThreads);

    // Environment variable present, do nothing.
    if (GenericDevice.getOMPTeamsThreadLimit() > 0)
      return std::make_pair(false, NumThreads);

    // num_threads clause present, do nothing.
    if ((ThreadLimitClause[0] > 0) && (ThreadLimitClause[0] != (uint32_t)-1))
      return std::make_pair(false, NumThreads);

    // If generic or generic-SPMD kernel, do nothing.
    if (isGenericMode() || isGenericSPMDMode())
      return std::make_pair(false, NumThreads);

    // Reduce the blocksize as long as it is above the tunable limit.
    while (NumThreads > GenericDevice.getOMPXSmallBlockSize())
      NumThreads >>= 1;

    if (NumThreads == 0)
      return std::make_pair(false, BlockSize);

    if (isXTeamReductionsMode())
      return std::make_pair(true,
                            llvm::omp::getBlockSizeAsPowerOfTwo(NumThreads));

    return std::make_pair(true, NumThreads);
  }

  /// Get the number of threads and blocks for the kernel based on the
  /// user-defined threads and block clauses.
  uint32_t getNumThreads(GenericDeviceTy &GenericDevice,
                         uint32_t ThreadLimitClause[3]) const override {
    assert(ThreadLimitClause[1] == 0 && ThreadLimitClause[2] == 0 &&
           "Multi dimensional launch not supported yet.");

    // Honor OMP_TEAMS_THREAD_LIMIT environment variable and
    // num_threads/thread_limit clause for BigJumpLoop and NoLoop kernel types.
    int32_t TeamsThreadLimitEnvVar = GenericDevice.getOMPTeamsThreadLimit();
    if (isBigJumpLoopMode() || isNoLoopMode()) {
      if (TeamsThreadLimitEnvVar > 0)
        return std::min(static_cast<int32_t>(ConstWGSize),
                        TeamsThreadLimitEnvVar);
      if ((ThreadLimitClause[0] > 0) && (ThreadLimitClause[0] != (uint32_t)-1))
        return std::min(static_cast<uint32_t>(ConstWGSize),
                        ThreadLimitClause[0]);
      return ConstWGSize;
    }

    if (isXTeamReductionsMode()) {
      if (TeamsThreadLimitEnvVar > 0 &&
          TeamsThreadLimitEnvVar <= static_cast<int32_t>(ConstWGSize))
        return llvm::omp::getBlockSizeAsPowerOfTwo(TeamsThreadLimitEnvVar);
      if (ThreadLimitClause[0] > 0 && ThreadLimitClause[0] != (uint32_t)-1 &&
          ThreadLimitClause[0] <= static_cast<int32_t>(ConstWGSize))
        return llvm::omp::getBlockSizeAsPowerOfTwo(ThreadLimitClause[0]);
      assert(((ConstWGSize & (ConstWGSize - 1)) == 0) &&
             "XTeam Reduction blocksize must be a power of two");
      return ConstWGSize;
    }

    if (ThreadLimitClause[0] > 0 && isGenericMode()) {
      if (ThreadLimitClause[0] == (uint32_t)-1)
        ThreadLimitClause[0] = PreferredNumThreads;
      else
        ThreadLimitClause[0] += GenericDevice.getWarpSize();
    }

    // Limit number of threads taking into consideration the user
    // environment variable OMP_TEAMS_THREAD_LIMIT if provided.
    uint32_t CurrentMaxNumThreads = MaxNumThreads;
    if (TeamsThreadLimitEnvVar > 0)
      CurrentMaxNumThreads = std::min(
          static_cast<uint32_t>(TeamsThreadLimitEnvVar), CurrentMaxNumThreads);

    return std::min(CurrentMaxNumThreads, (ThreadLimitClause[0] > 0)
                                              ? ThreadLimitClause[0]
                                              : PreferredNumThreads);
  }
  uint64_t getNumBlocks(GenericDeviceTy &GenericDevice,
                        uint32_t NumTeamsClause[3], uint64_t LoopTripCount,
                        uint32_t &NumThreads,
                        bool IsNumThreadsFromUser) const override {
    assert(NumTeamsClause[1] == 0 && NumTeamsClause[2] == 0 &&
           "Multi dimensional launch not supported yet.");

    const auto getNumGroupsFromThreadsAndTripCount =
        [](const uint64_t TripCount, const uint32_t NumThreads) {
          return ((TripCount - 1) / NumThreads) + 1;
        };
    uint64_t DeviceNumCUs = GenericDevice.getNumComputeUnits(); // FIXME

    if (isNoLoopMode()) {
      return LoopTripCount > 0 ? getNumGroupsFromThreadsAndTripCount(
                                     LoopTripCount, NumThreads)
                               : 1;
    }

    uint64_t NumWavesInGroup =
        (NumThreads - 1) / GenericDevice.getWarpSize() + 1;

    if (isBigJumpLoopMode()) {
      uint64_t NumGroups = 1;
      // Cannot assert a non-zero tripcount. Instead, launch with 1 team if the
      // tripcount is indeed zero.
      if (LoopTripCount > 0)
        NumGroups =
            getNumGroupsFromThreadsAndTripCount(LoopTripCount, NumThreads);

      // Honor OMP_NUM_TEAMS environment variable for BigJumpLoop kernel type.
      int32_t NumTeamsEnvVar = GenericDevice.getOMPNumTeams();
      if (NumTeamsEnvVar > 0 && NumTeamsEnvVar <= GenericDevice.getBlockLimit())
        NumGroups = std::min(static_cast<uint64_t>(NumTeamsEnvVar), NumGroups);
      // Honor num_teams clause but lower it if tripcount dictates.
      else if (NumTeamsClause[0] > 0 &&
               NumTeamsClause[0] <= GenericDevice.getBlockLimit()) {
        NumGroups =
            std::min(static_cast<uint64_t>(NumTeamsClause[0]), NumGroups);
      } else {
        // num_teams clause is not specified. Choose lower of tripcount-based
        // num-groups and a value that maximizes occupancy. At this point, aim
        // to have 16 wavefronts in a CU. Allow for override with envar.
        uint64_t MaxOccupancyFactor =
            GenericDevice.getOMPXBigJumpLoopTeamsPerCU() > 0
                ? GenericDevice.getOMPXBigJumpLoopTeamsPerCU()
                : 16 / NumWavesInGroup;
        NumGroups = std::min(NumGroups, MaxOccupancyFactor * DeviceNumCUs);

        // If the user specifies a number of teams for low trip count loops,
        // honor it.
        uint64_t LowTripCountBlocks =
            GenericDevice.getOMPXNumBlocksForLowTripcount(LoopTripCount);
        if (LowTripCountBlocks) {
          NumGroups = LowTripCountBlocks;
        }
      }
      return NumGroups;
    }

    if (isXTeamReductionsMode()) {
      // Here's the default number of teams.
      uint64_t NumGroups = DeviceNumCUs;
      // The number of teams must not exceed this upper limit.
      uint64_t MaxNumGroups = NumGroups;
      if (GenericDevice.isFastReductionEnabled()) {
        // When fast reduction is enabled, the number of teams is capped by
        // the MaxCUMultiplier constant.
        MaxNumGroups = DeviceNumCUs * llvm::omp::xteam_red::MaxCUMultiplier;
      } else {
        // When fast reduction is not enabled, the number of teams is capped
        // by the metadata that clang CodeGen created. The number of teams
        // used here must not exceed the upper limit determined during
        // CodeGen. This upper limit is not currently communicated from
        // CodeGen to the plugin. So it is re-computed here.

        // ConstWGSize is the block size that CodeGen used.
        uint32_t CUMultiplier =
            llvm::omp::xteam_red::getXteamRedCUMultiplier(ConstWGSize);
        MaxNumGroups = DeviceNumCUs * CUMultiplier;
      }

      // Honor OMP_NUM_TEAMS environment variable for XteamReduction kernel
      // type, if possible.
      int32_t NumTeamsEnvVar = GenericDevice.getOMPNumTeams();

      // Prefer num_teams clause over environment variable. There is a corner
      // case where inspite of the presence of a num_teams clause, CodeGen
      // may fail to extract it, instead using the alternative computation of
      // the number of teams. But the runtime here will still see the value
      // of the clause, so we need to check against the upper limit.
      if (NumTeamsClause[0] > 0 &&
          NumTeamsClause[0] <= GenericDevice.getBlockLimit()) {
        NumGroups =
            std::min(static_cast<uint64_t>(NumTeamsClause[0]), MaxNumGroups);
      } else if (NumTeamsEnvVar > 0 &&
                 NumTeamsEnvVar <= GenericDevice.getBlockLimit()) {
        NumGroups =
            std::min(static_cast<uint64_t>(NumTeamsEnvVar), MaxNumGroups);
      } else {
        // Ensure we don't have a large number of teams running if the tripcount
        // is low
        uint64_t NumGroupsFromTripCount = 1;
        if (LoopTripCount > 0)
          NumGroupsFromTripCount =
              getNumGroupsFromThreadsAndTripCount(LoopTripCount, NumThreads);

        // Compute desired number of groups in the absence of user input
        // based on a factor controlled by an integer env-var.
        // 0: disabled (default)
        // 1: If the number of waves is lower than the default, increase
        // the number of teams proportionally. Ideally, this would be the
        // default behavior.
        // > 1: Use as the scaling factor for the number of teams.
        // Note that the upper bound is MaxNumGroups.
        uint32_t AdjustFactor =
            GenericDevice.getOMPXAdjustNumTeamsForXteamRedSmallBlockSize();
        if (NumThreads > 0 && AdjustFactor > 0) {
          uint64_t DesiredNumGroups = NumGroups;
          if (AdjustFactor == 1) {
            DesiredNumGroups =
                DeviceNumCUs *
                (llvm::omp::xteam_red::DesiredWavesPerCU / NumWavesInGroup);
          } else {
            DesiredNumGroups = DeviceNumCUs * AdjustFactor;
          }
          NumGroups = DesiredNumGroups;
        }
        NumGroups = std::min(NumGroups, MaxNumGroups);
        NumGroups = std::min(NumGroups, NumGroupsFromTripCount);

        // If the user specifies a number of teams for low trip count loops,
        // and no num_teams clause was used, honor it.
        uint64_t LowTripCountBlocks =
            GenericDevice.getOMPXNumBlocksForLowTripcount(LoopTripCount);
        if (LowTripCountBlocks) {
          NumGroups = std::min(MaxNumGroups, LowTripCountBlocks);
        }
      }
      DP("xteam-red:NumCUs=%lu xteam-red:NumGroups=%lu\n", DeviceNumCUs,
         NumGroups);
      return NumGroups;
    }

    if (NumTeamsClause[0] > 0) {
      // TODO: We need to honor any value and consequently allow more than the
      // block limit. For this we might need to start multiple kernels or let
      // the blocks start again until the requested number has been started.
      return std::min(NumTeamsClause[0], GenericDevice.getBlockLimit());
    }

    uint64_t TripCountNumBlocks = std::numeric_limits<uint64_t>::max();
    if (LoopTripCount > 0) {
      if (isSPMDMode()) {
        // We have a combined construct, i.e. `target teams distribute
        // parallel for [simd]`. We launch so many teams so that each thread
        // will execute one iteration of the loop. round up to the nearest
        // integer
        TripCountNumBlocks = ((LoopTripCount - 1) / NumThreads) + 1;
      } else {
        assert((isGenericMode() || isGenericSPMDMode()) &&
               "Unexpected execution mode!");
        // If we reach this point, then we have a non-combined construct, i.e.
        // `teams distribute` with a nested `parallel for` and each team is
        // assigned one iteration of the `distribute` loop. E.g.:
        //
        // #pragma omp target teams distribute
        // for(...loop_tripcount...) {
        //   #pragma omp parallel for
        //   for(...) {}
        // }
        //
        // Threads within a team will execute the iterations of the `parallel`
        // loop.
        TripCountNumBlocks = LoopTripCount;
      }
    }

    auto getAdjustedDefaultNumBlocks =
        [this](GenericDeviceTy &GenericDevice,
               uint64_t DeviceNumCUs) -> uint64_t {
      if (!isGenericSPMDMode() ||
          GenericDevice.getOMPXGenericSpmdTeamsPerCU() == 0)
        return static_cast<uint64_t>(GenericDevice.getDefaultNumBlocks());
      return DeviceNumCUs * static_cast<uint64_t>(
                                GenericDevice.getOMPXGenericSpmdTeamsPerCU());
    };

    // If the loops are long running we rather reuse blocks than spawn too many.
    // Additionally, under an env-var, adjust the number of teams based on the
    // number of wave-slots in a CU that we aim to occupy.
    uint64_t AdjustedNumBlocks =
        getAdjustedDefaultNumBlocks(GenericDevice, DeviceNumCUs);
    if (GenericDevice.getOMPXAdjustNumTeamsForSmallBlockSize()) {
      uint64_t DefaultNumWavesInGroup =
          (GenericDevice.getDefaultNumThreads() - 1) /
              GenericDevice.getWarpSize() +
          1;
      AdjustedNumBlocks =
          (AdjustedNumBlocks * DefaultNumWavesInGroup) / NumWavesInGroup;
    }

    // If the user specifies a number of teams for low trip count loops, honor
    // it.
    uint64_t LowTripCountBlocks =
        GenericDevice.getOMPXNumBlocksForLowTripcount(LoopTripCount);
    if (LowTripCountBlocks) {
      return LowTripCountBlocks;
    }

    uint64_t PreferredNumBlocks = TripCountNumBlocks;
    // If the loops are long running we rather reuse blocks than spawn too many.
    if (GenericDevice.getReuseBlocksForHighTripCount())
      PreferredNumBlocks = std::min(TripCountNumBlocks, AdjustedNumBlocks);
    return std::min(PreferredNumBlocks,
                    (uint64_t)GenericDevice.getBlockLimit());
  }
};

/// Class representing an HSA signal. Signals are used to define dependencies
/// between asynchronous operations: kernel launches and memory transfers.
struct AMDGPUSignalTy {
  /// Create an empty signal.
  AMDGPUSignalTy() : HSASignal({0}), UseCount() {}
  AMDGPUSignalTy(AMDGPUDeviceTy &Device) : HSASignal({0}), UseCount() {}

  /// Initialize the signal with an initial value.
  Error init(uint32_t InitialValue = 1) {
    hsa_status_t Status =
        hsa_amd_signal_create(InitialValue, 0, nullptr, 0, &HSASignal);
    return Plugin::check(Status, "Error in hsa_signal_create: %s");
  }

  /// Deinitialize the signal.
  Error deinit() {
    hsa_status_t Status = hsa_signal_destroy(HSASignal);
    return Plugin::check(Status, "Error in hsa_signal_destroy: %s");
  }

  /// Wait until the signal gets a zero value.
  Error wait(const uint64_t ActiveTimeout = 0, RPCServerTy *RPCServer = nullptr,
             GenericDeviceTy *Device = nullptr) const {
    if (ActiveTimeout && !RPCServer) {
      hsa_signal_value_t Got = 1;
      Got = hsa_signal_wait_scacquire(HSASignal, HSA_SIGNAL_CONDITION_EQ, 0,
                                      ActiveTimeout, HSA_WAIT_STATE_ACTIVE);
      if (Got == 0)
        return Plugin::success();
    }

    // If there is an RPC device attached to this stream we run it as a server.
    uint64_t Timeout = RPCServer ? 8192 : UINT64_MAX;
    auto WaitState = RPCServer ? HSA_WAIT_STATE_ACTIVE : HSA_WAIT_STATE_BLOCKED;
    while (hsa_signal_wait_scacquire(HSASignal, HSA_SIGNAL_CONDITION_EQ, 0,
                                     Timeout, WaitState) != 0) {
      if (RPCServer && Device)
        if (auto Err = RPCServer->runServer(*Device))
          return Err;
    }
    return Plugin::success();
  }

  /// Load the value on the signal.
  hsa_signal_value_t load() const {
    return hsa_signal_load_scacquire(HSASignal);
  }

  /// Signal decrementing by one.
  void signal() {
    assert(load() > 0 && "Invalid signal value");
    hsa_signal_subtract_screlease(HSASignal, 1);
  }

  /// Reset the signal value before reusing the signal. Do not call this
  /// function if the signal is being currently used by any watcher, such as a
  /// plugin thread or the HSA runtime.
  void reset() { hsa_signal_store_screlease(HSASignal, 1); }

  /// Increase the number of concurrent uses.
  void increaseUseCount() { UseCount.increase(); }

  /// Decrease the number of concurrent uses and return whether was the last.
  bool decreaseUseCount() { return UseCount.decrease(); }

  hsa_signal_t get() const { return HSASignal; }

private:
  /// The underlying HSA signal.
  hsa_signal_t HSASignal;

  /// Reference counter for tracking the concurrent use count. This is mainly
  /// used for knowing how many streams are using the signal.
  RefCountTy<> UseCount;
};

/// Classes for holding AMDGPU signals and managing signals.
using AMDGPUSignalRef = AMDGPUResourceRef<AMDGPUSignalTy>;
using AMDGPUSignalManagerTy = GenericDeviceResourceManagerTy<AMDGPUSignalRef>;

/// Class holding an HSA queue to submit kernel and barrier packets.
struct AMDGPUQueueTy {
  /// Create an empty queue.
  AMDGPUQueueTy() : Queue(nullptr), Mutex(), NumUsers(0) {}

  /// Lazily initialize a new queue belonging to a specific agent.
  Error init(GenericDeviceTy &Device, hsa_agent_t Agent, int32_t QueueSize,
             int OMPX_EnableQueueProfiling) {
    if (Queue)
      return Plugin::success();
    hsa_status_t Status =
        hsa_queue_create(Agent, QueueSize, HSA_QUEUE_TYPE_MULTI, callbackError,
                         &Device, UINT32_MAX, UINT32_MAX, &Queue);
    OMPT_IF_TRACING_OR_ENV_VAR_ENABLED(
        hsa_amd_profiling_set_profiler_enabled(Queue, /*Enable=*/1););
    return Plugin::check(Status, "Error in hsa_queue_create: %s");
  }

  /// Deinitialize the queue and destroy its resources.
  Error deinit() {
    std::lock_guard<std::mutex> Lock(Mutex);
    // Don't bother turning OFF profiling, the queue is going away anyways.
    if (!Queue)
      return Plugin::success();
    hsa_status_t Status = hsa_queue_destroy(Queue);
    return Plugin::check(Status, "Error in hsa_queue_destroy: %s");
  }

  /// Returns the number of streams, this queue is currently assigned to.
  bool getUserCount() const { return NumUsers; }

  /// Returns if the underlying HSA queue is initialized.
  bool isInitialized() { return Queue != nullptr; }

  /// Decrement user count of the queue object.
  void removeUser() { --NumUsers; }

  /// Increase user count of the queue object.
  void addUser() { ++NumUsers; }

  /// Push a kernel launch to the queue. The kernel launch requires an output
  /// signal and can define an optional input signal (nullptr if none).
  Error pushKernelLaunch(const AMDGPUKernelTy &Kernel, void *KernelArgs,
                         uint32_t NumThreads, uint64_t NumBlocks,
                         uint32_t GroupSize, uint32_t StackSize,
                         AMDGPUSignalTy *OutputSignal,
                         AMDGPUSignalTy *InputSignal) {
    assert(OutputSignal && "Invalid kernel output signal");

    // Lock the queue during the packet publishing process. Notice this blocks
    // the addition of other packets to the queue. The following piece of code
    // should be lightweight; do not block the thread, allocate memory, etc.
    std::lock_guard<std::mutex> Lock(Mutex);
    assert(Queue && "Interacted with a non-initialized queue!");

    // Add a barrier packet before the kernel packet in case there is a pending
    // preceding operation. The barrier packet will delay the processing of
    // subsequent queue's packets until the barrier input signal are satisfied.
    // No need output signal needed because the dependency is already guaranteed
    // by the queue barrier itself.
    if (InputSignal && InputSignal->load())
      if (auto Err = pushBarrierImpl(nullptr, InputSignal))
        return Err;

    // Now prepare the kernel packet.
    uint64_t PacketId;
    hsa_kernel_dispatch_packet_t *Packet = acquirePacket(PacketId);
    assert(Packet && "Invalid packet");

    // The first 32 bits of the packet are written after the other fields
    uint16_t Setup = UINT16_C(1) << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
    Packet->workgroup_size_x = NumThreads;
    Packet->workgroup_size_y = 1;
    Packet->workgroup_size_z = 1;
    Packet->reserved0 = 0;
    Packet->grid_size_x = NumBlocks * NumThreads;
    Packet->grid_size_y = 1;
    Packet->grid_size_z = 1;
    Packet->private_segment_size =
        Kernel.usesDynamicStack() ? std::max(Kernel.getPrivateSize(), StackSize)
                                  : Kernel.getPrivateSize();
    Packet->group_segment_size = GroupSize;
    Packet->kernel_object = Kernel.getKernelObject();
    Packet->kernarg_address = KernelArgs;
    Packet->reserved2 = 0;
    Packet->completion_signal = OutputSignal->get();

    // Publish the packet. Do not modify the packet after this point.
    publishKernelPacket(PacketId, Setup, Packet);

    return Plugin::success();
  }

  /// Push a barrier packet that will wait up to two input signals. All signals
  /// are optional (nullptr if none).
  Error pushBarrier(AMDGPUSignalTy *OutputSignal,
                    const AMDGPUSignalTy *InputSignal1,
                    const AMDGPUSignalTy *InputSignal2) {
    // Lock the queue during the packet publishing process.
    std::lock_guard<std::mutex> Lock(Mutex);
    assert(Queue && "Interacted with a non-initialized queue!");

    // Push the barrier with the lock acquired.
    return pushBarrierImpl(OutputSignal, InputSignal1, InputSignal2);
  }

  /// Return the pointer to the underlying HSA queue
  hsa_queue_t *getHsaQueue() {
    assert(Queue && "HSA Queue initialized");
    return Queue;
  }

private:
  /// Push a barrier packet that will wait up to two input signals. Assumes the
  /// the queue lock is acquired.
  Error pushBarrierImpl(AMDGPUSignalTy *OutputSignal,
                        const AMDGPUSignalTy *InputSignal1,
                        const AMDGPUSignalTy *InputSignal2 = nullptr) {
    // Add a queue barrier waiting on both the other stream's operation and the
    // last operation on the current stream (if any).
    uint64_t PacketId;
    hsa_barrier_and_packet_t *Packet =
        (hsa_barrier_and_packet_t *)acquirePacket(PacketId);
    assert(Packet && "Invalid packet");

    Packet->reserved0 = 0;
    Packet->reserved1 = 0;
    Packet->dep_signal[0] = {0};
    Packet->dep_signal[1] = {0};
    Packet->dep_signal[2] = {0};
    Packet->dep_signal[3] = {0};
    Packet->dep_signal[4] = {0};
    Packet->reserved2 = 0;
    Packet->completion_signal = {0};

    // Set input and output dependencies if needed.
    if (OutputSignal)
      Packet->completion_signal = OutputSignal->get();
    if (InputSignal1)
      Packet->dep_signal[0] = InputSignal1->get();
    if (InputSignal2)
      Packet->dep_signal[1] = InputSignal2->get();

    // Publish the packet. Do not modify the packet after this point.
    publishBarrierPacket(PacketId, Packet);

    return Plugin::success();
  }

  /// Acquire a packet from the queue. This call may block the thread if there
  /// is no space in the underlying HSA queue. It may need to wait until the HSA
  /// runtime processes some packets. Assumes the queue lock is acquired.
  hsa_kernel_dispatch_packet_t *acquirePacket(uint64_t &PacketId) {
    // Increase the queue index with relaxed memory order. Notice this will need
    // another subsequent atomic operation with acquire order.
    PacketId = hsa_queue_add_write_index_relaxed(Queue, 1);

    // Wait for the package to be available. Notice the atomic operation uses
    // the acquire memory order.
    while (PacketId - hsa_queue_load_read_index_scacquire(Queue) >= Queue->size)
      ;

    // Return the packet reference.
    const uint32_t Mask = Queue->size - 1; // The size is a power of 2.
    return (hsa_kernel_dispatch_packet_t *)Queue->base_address +
           (PacketId & Mask);
  }

  /// Publish the kernel packet so that the HSA runtime can start processing
  /// the kernel launch. Do not modify the packet once this function is called.
  /// Assumes the queue lock is acquired.
  void publishKernelPacket(uint64_t PacketId, uint16_t Setup,
                           hsa_kernel_dispatch_packet_t *Packet) {
    uint32_t *PacketPtr = reinterpret_cast<uint32_t *>(Packet);

    uint16_t Header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
    Header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
    Header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;

    // Publish the packet. Do not modify the package after this point.
    uint32_t HeaderWord = Header | (Setup << 16u);
    __atomic_store_n(PacketPtr, HeaderWord, __ATOMIC_RELEASE);

    // Signal the doorbell about the published packet.
    hsa_signal_store_relaxed(Queue->doorbell_signal, PacketId);
  }

  /// Publish the barrier packet so that the HSA runtime can start processing
  /// the barrier. Next packets in the queue will not be processed until all
  /// barrier dependencies (signals) are satisfied. Assumes the queue is locked
  void publishBarrierPacket(uint64_t PacketId,
                            hsa_barrier_and_packet_t *Packet) {
    uint32_t *PacketPtr = reinterpret_cast<uint32_t *>(Packet);
    uint16_t Setup = 0;
    uint16_t Header = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
    Header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
    Header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;

    // Publish the packet. Do not modify the package after this point.
    uint32_t HeaderWord = Header | (Setup << 16u);
    __atomic_store_n(PacketPtr, HeaderWord, __ATOMIC_RELEASE);

    // Signal the doorbell about the published packet.
    hsa_signal_store_relaxed(Queue->doorbell_signal, PacketId);
  }

  /// Callack that will be called when an error is detected on the HSA queue.
  static void callbackError(hsa_status_t Status, hsa_queue_t *Source,
                            void *Data);

  /// The HSA queue.
  hsa_queue_t *Queue;

  /// Mutex to protect the acquiring and publishing of packets. For the moment,
  /// we need this mutex to prevent publishing packets that are not ready to be
  /// published in a multi-thread scenario. Without a queue lock, a thread T1
  /// could acquire packet P and thread T2 acquire packet P+1. Thread T2 could
  /// publish its packet P+1 (signaling the queue's doorbell) before packet P
  /// from T1 is ready to be processed. That scenario should be invalid. Thus,
  /// we use the following mutex to make packet acquiring and publishing atomic.
  /// TODO: There are other more advanced approaches to avoid this mutex using
  /// atomic operations. We can further investigate it if this is a bottleneck.
  std::mutex Mutex;

  /// The number of streams, this queue is currently assigned to. A queue is
  /// considered idle when this is zero, otherwise: busy.
  uint32_t NumUsers;
};

/// Struct that implements a stream of asynchronous operations for AMDGPU
/// devices. This class relies on signals to implement streams and define the
/// dependencies between asynchronous operations.
struct AMDGPUStreamTy {
private:
  /// Utility struct holding arguments for async H2H memory copies.
  struct MemcpyArgsTy {
    void *Dst;
    const void *Src;
    size_t Size;
  };

  /// Utility struct holding arguments for freeing buffers to memory managers.
  struct ReleaseBufferArgsTy {
    void *Buffer;
    AMDGPUMemoryManagerTy *MemoryManager;
  };

  /// Utility struct holding arguments for releasing signals to signal managers.
  struct ReleaseSignalArgsTy {
    AMDGPUSignalTy *Signal;
    AMDGPUSignalManagerTy *SignalManager;
  };

  /// Utility struct holding arguments for OMPT-based kernel timing.
  struct OmptKernelTimingArgsTy {
    hsa_agent_t Agent;
    AMDGPUSignalTy *Signal;
    double TicksToTime;
  };
  /// The stream is composed of N stream's slots. The struct below represents
  /// the fields of each slot. Each slot has a signal and an optional action
  /// function. When appending an HSA asynchronous operation to the stream, one
  /// slot is consumed and used to store the operation's information. The
  /// operation's output signal is set to the consumed slot's signal. If there
  /// is a previous asynchronous operation on the previous slot, the HSA async
  /// operation's input signal is set to the signal of the previous slot. This
  /// way, we obtain a chain of dependant async operations. The action is a
  /// function that will be executed eventually after the operation is
  /// completed, e.g., for releasing a buffer.
  struct StreamSlotTy {
    /// The output signal of the stream operation. May be used by the subsequent
    /// operation as input signal.
    AMDGPUSignalTy *Signal;

    /// The action that must be performed after the operation's completion. Set
    /// to nullptr when there is no action to perform.
    Error (*ActionFunction)(void *);

    /// The OMPT action that must be performed after the operation's completion.
    /// Set to nullptr when there is no action to perform.
    Error (*OmptActionFunction)(void *);

    /// Space for the action's arguments. A pointer to these arguments is passed
    /// to the action function. Notice the space of arguments is limited.
    union {
      MemcpyArgsTy MemcpyArgs;
      ReleaseBufferArgsTy ReleaseBufferArgs;
      ReleaseSignalArgsTy ReleaseSignalArgs;
    } ActionArgs;

#ifdef OMPT_SUPPORT
    /// Space for the OMPT action's arguments. A pointer to these arguments is
    /// passed to the action function.
    OmptKernelTimingArgsAsyncTy OmptKernelTimingArgsAsync;
#endif

    /// Create an empty slot.
    StreamSlotTy()
        : Signal(nullptr), ActionFunction(nullptr),
          OmptActionFunction(nullptr) {}

    /// Schedule a host memory copy action on the slot.
    Error schedHostMemoryCopy(void *Dst, const void *Src, size_t Size) {
      ActionFunction = memcpyAction;
      ActionArgs.MemcpyArgs = MemcpyArgsTy{Dst, Src, Size};
      return Plugin::success();
    }

    /// Schedule a release buffer action on the slot.
    Error schedReleaseBuffer(void *Buffer, AMDGPUMemoryManagerTy &Manager) {
      ActionFunction = releaseBufferAction;
      ActionArgs.ReleaseBufferArgs = ReleaseBufferArgsTy{Buffer, &Manager};
      return Plugin::success();
    }

    /// Schedule a signal release action on the slot.
    Error schedReleaseSignal(AMDGPUSignalTy *SignalToRelease,
                             AMDGPUSignalManagerTy *SignalManager) {
      ActionFunction = releaseSignalAction;
      ActionArgs.ReleaseSignalArgs =
          ReleaseSignalArgsTy{SignalToRelease, SignalManager};
      return Plugin::success();
    }

#ifdef OMPT_SUPPORT
    /// Schedule OMPT kernel timing on the slot.
    Error schedOmptAsyncKernelTiming(
        hsa_agent_t Agent, AMDGPUSignalTy *OutputSignal, double TicksToTime,
        std::unique_ptr<ompt::OmptEventInfoTy> OMPTData) {
      OmptActionFunction = timeKernelInNsAsync;
      OmptKernelTimingArgsAsync = OmptKernelTimingArgsAsyncTy{
          Agent, OutputSignal, TicksToTime, std::move(OMPTData)};
      return Plugin::success();
    }

    /// Schedule OMPT data transfer timing on the slot
    Error schedOmptAsyncD2HTransferTiming(
        hsa_agent_t Agent, AMDGPUSignalTy *OutputSignal, double TicksToTime,
        std::unique_ptr<ompt::OmptEventInfoTy> OmptInfoData) {
      OmptActionFunction = timeDataTransferInNsAsync;
      OmptKernelTimingArgsAsync = OmptKernelTimingArgsAsyncTy{
          Agent, OutputSignal, TicksToTime, std::move(OmptInfoData)};
      return Plugin::success();
    }
#endif

    // Perform the action if needed.
    Error performAction() {
      if (!ActionFunction
#ifdef OMPT_SUPPORT
          && !OmptActionFunction
#endif
      )
        return Plugin::success();

      // Perform the action.
      if (ActionFunction == memcpyAction) {
        if (auto Err = memcpyAction(&ActionArgs))
          return Err;
      } else if (ActionFunction == releaseBufferAction) {
        if (auto Err = releaseBufferAction(&ActionArgs))
          return Err;
      } else if (ActionFunction == releaseSignalAction) {
        if (auto Err = releaseSignalAction(&ActionArgs))
          return Err;
      } else if (ActionFunction == nullptr) {
        // For example a Device-to-Device transfer will not require a buffer
        // release and the ActionFunction will be a nullptr. Hence, we should
        // generally pass in this scenario (but still log the info).
        DP("performAction: ActionFunction was nullptr\n");
      } else {
        return Plugin::error("Unknown action function!");
      }

      // Invalidate the actions.
      ActionFunction = nullptr;

#ifdef OMPT_SUPPORT
      OMPT_IF_TRACING_ENABLED(if (OmptActionFunction) {
        if (OmptActionFunction == timeKernelInNsAsync) {
          if (auto Err = timeKernelInNsAsync(&OmptKernelTimingArgsAsync))
            return Err;
        } else if (OmptActionFunction == timeDataTransferInNsAsync) {
          if (auto Err = timeDataTransferInNsAsync(&OmptKernelTimingArgsAsync))
            return Err;
        } else {
          return Plugin::error("Unknown ompt action function!");
        }
      });

      OmptActionFunction = nullptr;
#endif

      return Plugin::success();
    }
  };

  /// The device agent where the stream was created.
  hsa_agent_t Agent;

  /// The queue that the stream uses to launch kernels.
  AMDGPUQueueTy *Queue;

  /// The manager of signals to reuse signals.
  AMDGPUSignalManagerTy &SignalManager;

  /// A reference to the associated device.
  GenericDeviceTy &Device;

  /// Array of stream slots. Use std::deque because it can dynamically grow
  /// without invalidating the already inserted elements. For instance, the
  /// std::vector may invalidate the elements by reallocating the internal
  /// array if there is not enough space on new insertions.
  std::deque<StreamSlotTy> Slots;

  /// The next available slot on the queue. This is reset to zero each time the
  /// stream is synchronized. It also indicates the current number of consumed
  /// slots at a given time.
  uint32_t NextSlot;

  /// The synchronization id. This number is increased each time the stream is
  /// synchronized. It is useful to detect if an AMDGPUEventTy points to an
  /// operation that was already finalized in a previous stream sycnhronize.
  uint32_t SyncCycle;

  /// A pointer associated with an RPC server running on the given device. If
  /// RPC is not being used this will be a null pointer. Otherwise, this
  /// indicates that an RPC server is expected to be run on this stream.
  RPCServerTy *RPCServer;

  /// Mutex to protect stream's management.
  mutable std::mutex Mutex;

  /// Timeout hint for HSA actively waiting for signal value to change
  const uint64_t StreamBusyWaitMicroseconds;

  /// Indicate to spread data transfers across all avilable SDMAs
  bool UseMultipleSdmaEngines;

  /// Use synchronous copy back.
  bool UseSyncCopyBack;

  /// Return the current number of asychronous operations on the stream.
  uint32_t size() const { return NextSlot; }

  /// Return the last valid slot on the stream.
  uint32_t last() const { return size() - 1; }

  /// Consume one slot from the stream. Since the stream uses signals on demand
  /// and releases them once the slot is no longer used, the function requires
  /// an idle signal for the new consumed slot.
  std::pair<uint32_t, AMDGPUSignalTy *> consume(AMDGPUSignalTy *OutputSignal) {
    // Double the stream size if needed. Since we use std::deque, this operation
    // does not invalidate the already added slots.
    if (Slots.size() == NextSlot)
      Slots.resize(Slots.size() * 2);

    // Update the next available slot and the stream size.
    uint32_t Curr = NextSlot++;

    // Retrieve the input signal, if any, of the current operation.
    AMDGPUSignalTy *InputSignal = (Curr > 0) ? Slots[Curr - 1].Signal : nullptr;

    // Set the output signal of the current slot.
    Slots[Curr].Signal = OutputSignal;

    return std::make_pair(Curr, InputSignal);
  }

  /// Complete all pending post actions and reset the stream after synchronizing
  /// or positively querying the stream.
  Error complete() {
    for (uint32_t Slot = 0; Slot < NextSlot; ++Slot) {
      // Take the post action of the operation if any.
      if (auto Err = Slots[Slot].performAction())
        return Err;

      // Release the slot's signal if possible. Otherwise, another user will.
      if (Slots[Slot].Signal->decreaseUseCount())
        if (auto Err = SignalManager.returnResource(Slots[Slot].Signal))
          return Err;

      Slots[Slot].Signal = nullptr;
    }

    // Reset the stream slots to zero.
    NextSlot = 0;

    // Increase the synchronization id since the stream completed a sync cycle.
    SyncCycle += 1;

    return Plugin::success();
  }

  /// Make the current stream wait on a specific operation of another stream.
  /// The idea is to make the current stream waiting on two signals: 1) the last
  /// signal of the current stream, and 2) the last signal of the other stream.
  /// Use a barrier packet with two input signals.
  Error waitOnStreamOperation(AMDGPUStreamTy &OtherStream, uint32_t Slot) {
    if (Queue == nullptr)
      return Plugin::error("Target queue was nullptr");

    /// The signal that we must wait from the other stream.
    AMDGPUSignalTy *OtherSignal = OtherStream.Slots[Slot].Signal;

    // Prevent the release of the other stream's signal.
    OtherSignal->increaseUseCount();

    // Retrieve an available signal for the operation's output.
    AMDGPUSignalTy *OutputSignal = nullptr;
    if (auto Err = SignalManager.getResource(OutputSignal))
      return Err;
    OutputSignal->reset();
    OutputSignal->increaseUseCount();

    // Consume stream slot and compute dependencies.
    auto [Curr, InputSignal] = consume(OutputSignal);

    // Setup the post action to release the signal.
    if (auto Err = Slots[Curr].schedReleaseSignal(OtherSignal, &SignalManager))
      return Err;

    // Push a barrier into the queue with both input signals.
    DP("Using Queue: %p with HSA Queue: %p\n", Queue, Queue->getHsaQueue());
    return Queue->pushBarrier(OutputSignal, InputSignal, OtherSignal);
  }

  /// Callback for running a specific asynchronous operation. This callback is
  /// used for hsa_amd_signal_async_handler. The argument is the operation that
  /// should be executed. Notice we use the post action mechanism to codify the
  /// asynchronous operation.
  static bool asyncActionCallback(hsa_signal_value_t Value, void *Args) {
    // This thread is outside the stream mutex. Make sure the thread sees the
    // changes on the slot.
    std::atomic_thread_fence(std::memory_order_acquire);

    StreamSlotTy *Slot = reinterpret_cast<StreamSlotTy *>(Args);
    assert(Slot && "Invalid slot");
    assert(Slot->Signal && "Invalid signal");

    // Peform the operation.
    if (auto Err = Slot->performAction())
      FATAL_MESSAGE(1, "Error peforming post action: %s",
                    toString(std::move(Err)).data());

    // Signal the output signal to notify the asycnhronous operation finalized.
    Slot->Signal->signal();

    // Unregister callback.
    return false;
  }

  // Callback for host-to-host memory copies. This is an asynchronous action.
  static Error memcpyAction(void *Data) {
    MemcpyArgsTy *Args = reinterpret_cast<MemcpyArgsTy *>(Data);
    assert(Args && "Invalid arguments");
    assert(Args->Dst && "Invalid destination buffer");
    assert(Args->Src && "Invalid source buffer");

    std::memcpy(Args->Dst, Args->Src, Args->Size);

    return Plugin::success();
  }

  /// Releasing a memory buffer to a memory manager. This is a post completion
  /// action. There are two kinds of memory buffers:
  ///   1. For kernel arguments. This buffer can be freed after receiving the
  ///   kernel completion signal.
  ///   2. For H2D tranfers that need pinned memory space for staging. This
  ///   buffer can be freed after receiving the transfer completion signal.
  ///   3. For D2H tranfers that need pinned memory space for staging. This
  ///   buffer cannot be freed after receiving the transfer completion signal
  ///   because of the following asynchronous H2H callback.
  ///      For this reason, This action can only be taken at
  ///      AMDGPUStreamTy::complete()
  /// Because of the case 3, all releaseBufferActions are taken at
  /// AMDGPUStreamTy::complete() in the current implementation.
  static Error releaseBufferAction(void *Data) {
    ReleaseBufferArgsTy *Args = reinterpret_cast<ReleaseBufferArgsTy *>(Data);
    assert(Args && "Invalid arguments");
    assert(Args->MemoryManager && "Invalid memory manager");
    assert(Args->Buffer && "Invalid buffer");

    // Release the allocation to the memory manager.
    return Args->MemoryManager->deallocate(Args->Buffer);
  }

  /// Releasing a signal object back to SignalManager. This is a post completion
  /// action. This action can only be taken at AMDGPUStreamTy::complete()
  static Error releaseSignalAction(void *Data) {
    ReleaseSignalArgsTy *Args = reinterpret_cast<ReleaseSignalArgsTy *>(Data);
    assert(Args && "Invalid arguments");
    assert(Args->Signal && "Invalid signal");
    assert(Args->SignalManager && "Invalid signal manager");

    // Release the signal if needed.
    if (Args->Signal->decreaseUseCount())
      if (auto Err = Args->SignalManager->returnResource(Args->Signal))
        return Err;

    return Plugin::success();
  }

#ifdef OMPT_SUPPORT
  static Error timeKernelInNsAsync(void *Data) {
    assert(Data && "Invalid data pointer in OMPT profiling");
    auto Args = getOmptTimingsArgs(Data);

    assert(Args && "Invalid args pointer in OMPT profiling");
    auto [StartTime, EndTime] = getKernelStartAndEndTime(Args);

    DP("OMPT-Async: Time kernel for asynchronous execution (Plugin): Start %lu "
       "End %lu\n",
       StartTime, EndTime);

    assert(Args->OmptEventInfo && "Invalid OEI pointer in OMPT profiling");
    auto OmptEventInfo = *Args->OmptEventInfo;
    auto RIFunc = std::get<1>(OmptEventInfo.RIFunction);

    assert(OmptEventInfo.RegionInterface &&
           "Invalid RegionInterface pointer in OMPT profiling");
    assert(OmptEventInfo.TraceRecord && "Invalid TraceRecord");
    std::invoke(RIFunc, OmptEventInfo.RegionInterface,
                OmptEventInfo.TraceRecord, OmptEventInfo.NumTeams, StartTime,
                EndTime);

    return Plugin::success();
  }
#endif

public:
  /// Create an empty stream associated with a specific device.
  AMDGPUStreamTy(AMDGPUDeviceTy &Device);

  /// Intialize the stream's signals.
  Error init() { return Plugin::success(); }

  /// Deinitialize the stream's signals.
  Error deinit() { return Plugin::success(); }

  hsa_queue_t *getHsaQueue() { return Queue->getHsaQueue(); }

  /// Attach an RPC server to this stream.
  void setRPCServer(RPCServerTy *Server) { RPCServer = Server; }

  /// Push a asynchronous kernel to the stream. The kernel arguments must be
  /// placed in a special allocation for kernel args and must keep alive until
  /// the kernel finalizes. Once the kernel is finished, the stream will release
  /// the kernel args buffer to the specified memory manager.
  Error
  pushKernelLaunch(const AMDGPUKernelTy &Kernel, void *KernelArgs,
                   uint32_t NumThreads, uint64_t NumBlocks, uint32_t GroupSize,
                   uint32_t StackSize, AMDGPUMemoryManagerTy &MemoryManager,
                   std::unique_ptr<ompt::OmptEventInfoTy> OmptInfo = nullptr) {
    if (Queue == nullptr)
      return Plugin::error("Target queue was nullptr");

    // Retrieve an available signal for the operation's output.
    AMDGPUSignalTy *OutputSignal = nullptr;
    if (auto Err = SignalManager.getResource(OutputSignal))
      return Err;
    OutputSignal->reset();
    OutputSignal->increaseUseCount();

    std::lock_guard<std::mutex> StreamLock(Mutex);

    // Consume stream slot and compute dependencies.
    auto [Curr, InputSignal] = consume(OutputSignal);

    // Setup the post action to release the kernel args buffer.
    if (auto Err = Slots[Curr].schedReleaseBuffer(KernelArgs, MemoryManager))
      return Err;

#ifdef OMPT_SUPPORT
    if (OmptInfo) {
      DP("OMPT-Async: Info in KernelTy >> TR ptr: %p\n", OmptInfo->TraceRecord);

      // OmptInfo holds function pointer to finish trace record once the kernel
      // completed.
      if (auto Err = Slots[Curr].schedOmptAsyncKernelTiming(
              Agent, OutputSignal, TicksToTime, std::move(OmptInfo)))
        return Err;
    }
#endif

    // Push the kernel with the output signal and an input signal (optional)
    DP("Using Queue: %p with HSA Queue: %p\n", Queue, Queue->getHsaQueue());
    return Queue->pushKernelLaunch(Kernel, KernelArgs, NumThreads, NumBlocks,
                                   GroupSize, StackSize, OutputSignal,
                                   InputSignal);
  }

  /// Push an asynchronous memory copy between pinned memory buffers.
  Error pushPinnedMemoryCopyAsync(
      void *Dst, const void *Src, uint64_t CopySize,
      std::unique_ptr<ompt::OmptEventInfoTy> OmptInfo = nullptr) {
    // Retrieve an available signal for the operation's output.
    AMDGPUSignalTy *OutputSignal = nullptr;
    if (auto Err = SignalManager.getResource(OutputSignal))
      return Err;
    OutputSignal->reset();
    OutputSignal->increaseUseCount();

    std::lock_guard<std::mutex> Lock(Mutex);

    // Consume stream slot and compute dependencies.
    auto [Curr, InputSignal] = consume(OutputSignal);

#ifdef OMPT_SUPPORT
    if (OmptInfo) {
      DP("OMPT-Async: Registering data timing in pushPinnedMemoryCopyAsync\n");
      // Capture the time the data transfer required for the d2h transfer.
      if (auto Err = Slots[Curr].schedOmptAsyncD2HTransferTiming(
              Agent, OutputSignal, TicksToTime, std::move(OmptInfo)))
        return Err;
    }
#endif

    // Issue the async memory copy.
    if (InputSignal && InputSignal->load()) {
      hsa_signal_t InputSignalRaw = InputSignal->get();
      return utils::asyncMemCopy(UseMultipleSdmaEngines, Dst, Agent, Src, Agent,
                                 CopySize, 1, &InputSignalRaw,
                                 OutputSignal->get());
    }

    return utils::asyncMemCopy(UseMultipleSdmaEngines, Dst, Agent, Src, Agent,
                               CopySize, 0, nullptr, OutputSignal->get());
  }

  /// Push an asynchronous memory copy device-to-host involving an unpinned
  /// memory buffer. The operation consists of a two-step copy from the
  /// device buffer to an intermediate pinned host buffer, and then, to a
  /// unpinned host buffer. Both operations are asynchronous and dependant.
  /// The intermediate pinned buffer will be released to the specified memory
  /// manager once the operation completes.
  Error pushMemoryCopyD2HAsync(
      void *Dst, const void *Src, void *Inter, uint64_t CopySize,
      AMDGPUMemoryManagerTy &MemoryManager,
      std::unique_ptr<ompt::OmptEventInfoTy> OmptInfo = nullptr) {
    // Retrieve available signals for the operation's outputs.
    AMDGPUSignalTy *OutputSignals[2] = {};
    if (auto Err = SignalManager.getResources(/*Num=*/2, OutputSignals))
      return Err;
    for (auto *Signal : OutputSignals) {
      Signal->reset();
      Signal->increaseUseCount();
    }

    std::lock_guard<std::mutex> Lock(Mutex);

    // Consume stream slot and compute dependencies.
    auto [Curr, InputSignal] = consume(OutputSignals[0]);

    // Setup the post action for releasing the intermediate buffer.
    if (auto Err = Slots[Curr].schedReleaseBuffer(Inter, MemoryManager))
      return Err;

    // Wait for kernel to finish before scheduling the asynchronous copy.
    if (UseSyncCopyBack && InputSignal && InputSignal->load())
      if (auto Err = InputSignal->wait(StreamBusyWaitMicroseconds, RPCServer, &Device))
        return Err;

#ifdef OMPT_SUPPORT

    if (OmptInfo) {
      DP("OMPT-Async: Registering data timing in pushMemoryCopyD2HAsync\n");
      // Capture the time the data transfer required for the d2h transfer.
      if (auto Err = Slots[Curr].schedOmptAsyncD2HTransferTiming(
              Agent, OutputSignals[0], TicksToTime, std::move(OmptInfo)))
        return Err;
    }
#endif

    // Issue the first step: device to host transfer. Avoid defining the input
    // dependency if already satisfied.
    if (InputSignal && InputSignal->load()) {
      hsa_signal_t InputSignalRaw = InputSignal->get();
      if (auto Err = utils::asyncMemCopy(
              UseMultipleSdmaEngines, Inter, Agent, Src, Agent, CopySize, 1,
              &InputSignalRaw, OutputSignals[0]->get()))
        return Err;
    } else {
      if (auto Err = utils::asyncMemCopy(UseMultipleSdmaEngines, Inter, Agent,
                                         Src, Agent, CopySize, 0, nullptr,
                                         OutputSignals[0]->get()))
        return Err;
    }

    // Consume another stream slot and compute dependencies.
    std::tie(Curr, InputSignal) = consume(OutputSignals[1]);
    assert(InputSignal && "Invalid input signal");

    // The std::memcpy is done asynchronously using an async handler. We store
    // the function's information in the action but it's not actually an action.
    if (auto Err = Slots[Curr].schedHostMemoryCopy(Dst, Inter, CopySize))
      return Err;

    // Make changes on this slot visible to the async handler's thread.
    std::atomic_thread_fence(std::memory_order_release);

    // Issue the second step: host to host transfer.
    hsa_status_t Status = hsa_amd_signal_async_handler(
        InputSignal->get(), HSA_SIGNAL_CONDITION_EQ, 0, asyncActionCallback,
        (void *)&Slots[Curr]);

    return Plugin::check(Status, "Error in hsa_amd_signal_async_handler: %s");
  }

  /// Push an asynchronous memory copy host-to-device involving an unpinned
  /// memory buffer. The operation consists of a two-step copy from the
  /// unpinned host buffer to an intermediate pinned host buffer, and then, to
  /// the pinned host buffer. Both operations are asynchronous and dependant.
  /// The intermediate pinned buffer will be released to the specified memory
  /// manager once the operation completes.
  Error pushMemoryCopyH2DAsync(
      void *Dst, const void *Src, void *Inter, uint64_t CopySize,
      AMDGPUMemoryManagerTy &MemoryManager,
      std::unique_ptr<ompt::OmptEventInfoTy> OmptInfo = nullptr) {
    // Retrieve available signals for the operation's outputs.
    AMDGPUSignalTy *OutputSignals[2] = {};
    if (auto Err = SignalManager.getResources(/*Num=*/2, OutputSignals))
      return Err;
    for (auto *Signal : OutputSignals) {
      Signal->reset();
      Signal->increaseUseCount();
    }

    AMDGPUSignalTy *OutputSignal = OutputSignals[0];

    std::lock_guard<std::mutex> Lock(Mutex);

    // Consume stream slot and compute dependencies.
    auto [Curr, InputSignal] = consume(OutputSignal);

    // Issue the first step: host to host transfer.
    if (InputSignal && InputSignal->load()) {
      // The std::memcpy is done asynchronously using an async handler. We store
      // the function's information in the action but it is not actually a
      // post action.
      if (auto Err = Slots[Curr].schedHostMemoryCopy(Inter, Src, CopySize))
        return Err;

      // Make changes on this slot visible to the async handler's thread.
      std::atomic_thread_fence(std::memory_order_release);

      hsa_status_t Status = hsa_amd_signal_async_handler(
          InputSignal->get(), HSA_SIGNAL_CONDITION_EQ, 0, asyncActionCallback,
          (void *)&Slots[Curr]);

      if (auto Err = Plugin::check(Status,
                                   "Error in hsa_amd_signal_async_handler: %s"))
        return Err;

      // Let's use now the second output signal.
      OutputSignal = OutputSignals[1];

      // Consume another stream slot and compute dependencies.
      std::tie(Curr, InputSignal) = consume(OutputSignal);
    } else {
      // All preceding operations completed, copy the memory synchronously.
      std::memcpy(Inter, Src, CopySize);

      // Return the second signal because it will not be used.
      OutputSignals[1]->decreaseUseCount();
      if (auto Err = SignalManager.returnResource(OutputSignals[1]))
        return Err;
    }

    // Setup the post action to release the intermediate pinned buffer.
    if (auto Err = Slots[Curr].schedReleaseBuffer(Inter, MemoryManager))
      return Err;

#ifdef OMPT_SUPPORT
    if (OmptInfo) {
      DP("OMPT-Async: Registering data timing in pushMemoryCopyH2DAsync\n");
      // Capture the time the data transfer required for the d2h transfer.
      if (auto Err = Slots[Curr].schedOmptAsyncD2HTransferTiming(
              Agent, OutputSignals[0], TicksToTime, std::move(OmptInfo)))
        return Err;
    }
#endif

    // Issue the second step: host to device transfer. Avoid defining the input
    // dependency if already satisfied.
    if (InputSignal && InputSignal->load()) {
      hsa_signal_t InputSignalRaw = InputSignal->get();
      return utils::asyncMemCopy(UseMultipleSdmaEngines, Dst, Agent, Inter,
                                 Agent, CopySize, 1, &InputSignalRaw,
                                 OutputSignal->get());
    }
    return utils::asyncMemCopy(UseMultipleSdmaEngines, Dst, Agent, Inter, Agent,
                               CopySize, 0, nullptr, OutputSignal->get());
  }

  // AMDGPUDeviceTy is incomplete here, passing the underlying agent instead
  Error pushMemoryCopyD2DAsync(
      void *Dst, hsa_agent_t DstAgent, const void *Src, hsa_agent_t SrcAgent,
      uint64_t CopySize,
      std::unique_ptr<ompt::OmptEventInfoTy> OmptInfo = nullptr) {
    AMDGPUSignalTy *OutputSignal;
    if (auto Err = SignalManager.getResources(/*Num=*/1, &OutputSignal))
      return Err;
    OutputSignal->reset();
    OutputSignal->increaseUseCount();

    std::lock_guard<std::mutex> Lock(Mutex);

    // Consume stream slot and compute dependencies.
    auto [Curr, InputSignal] = consume(OutputSignal);

#ifdef OMPT_SUPPORT
    if (OmptInfo) {
      DP("OMPT-Async: Registering data timing in pushMemoryCopyD2DAsync\n");
      // Capture the time the data transfer required for the d2h transfer.
      if (auto Err = Slots[Curr].schedOmptAsyncD2HTransferTiming(
              Agent, OutputSignal, TicksToTime, std::move(OmptInfo)))
        return Err;
    }
#endif

    // The agents need to have access to the corresponding memory
    // This is presently only true if the pointers were originally
    // allocated by this runtime or the caller made the appropriate
    // access calls.

    if (InputSignal && InputSignal->load()) {
      hsa_signal_t InputSignalRaw = InputSignal->get();
      return utils::asyncMemCopy(UseMultipleSdmaEngines, Dst, DstAgent, Src,
                                 SrcAgent, CopySize, 1, &InputSignalRaw,
                                 OutputSignal->get());
    }
    return utils::asyncMemCopy(UseMultipleSdmaEngines, Dst, DstAgent, Src,
                               SrcAgent, CopySize, 0, nullptr,
                               OutputSignal->get());
  }

  /// Synchronize with the stream. The current thread waits until all operations
  /// are finalized and it performs the pending post actions (i.e., releasing
  /// intermediate buffers).
  Error synchronize() {
    std::lock_guard<std::mutex> Lock(Mutex);

    // No need to synchronize anything.
    if (size() == 0)
      return Plugin::success();

    // Wait until all previous operations on the stream have completed.
    if (auto Err = Slots[last()].Signal->wait(StreamBusyWaitMicroseconds,
                                              RPCServer, &Device))
      return Err;

    // Reset the stream and perform all pending post actions.
    return complete();
  }

  /// Query the stream and complete pending post actions if operations finished.
  /// Return whether all the operations completed. This operation does not block
  /// the calling thread.
  Expected<bool> query() {
    std::lock_guard<std::mutex> Lock(Mutex);

    // No need to query anything.
    if (size() == 0)
      return true;

    // The last operation did not complete yet. Return directly.
    if (Slots[last()].Signal->load())
      return false;

    // Reset the stream and perform all pending post actions.
    if (auto Err = complete())
      return std::move(Err);

    return true;
  }

  const AMDGPUQueueTy *getQueue() const { return Queue; }

  /// Record the state of the stream on an event.
  Error recordEvent(AMDGPUEventTy &Event) const;

  /// Make the stream wait on an event.
  Error waitEvent(const AMDGPUEventTy &Event);

  friend struct AMDGPUStreamManagerTy;
};

/// Class representing an event on AMDGPU. The event basically stores some
/// information regarding the state of the recorded stream.
struct AMDGPUEventTy {
  /// Create an empty event.
  AMDGPUEventTy(AMDGPUDeviceTy &Device)
      : RecordedStream(nullptr), RecordedSlot(-1), RecordedSyncCycle(-1) {}

  /// Initialize and deinitialize.
  Error init() { return Plugin::success(); }
  Error deinit() { return Plugin::success(); }

  /// Record the state of a stream on the event.
  Error record(AMDGPUStreamTy &Stream) {
    std::lock_guard<std::mutex> Lock(Mutex);

    // Ignore the last recorded stream.
    RecordedStream = &Stream;

    return Stream.recordEvent(*this);
  }

  /// Make a stream wait on the current event.
  Error wait(AMDGPUStreamTy &Stream) {
    std::lock_guard<std::mutex> Lock(Mutex);

    if (!RecordedStream)
      return Plugin::error("Event does not have any recorded stream");

    // Synchronizing the same stream. Do nothing.
    if (RecordedStream == &Stream)
      return Plugin::success();

    // No need to wait anything, the recorded stream already finished the
    // corresponding operation.
    if (RecordedSlot < 0)
      return Plugin::success();

    return Stream.waitEvent(*this);
  }

protected:
  /// The stream registered in this event.
  AMDGPUStreamTy *RecordedStream;

  /// The recordered operation on the recorded stream.
  int64_t RecordedSlot;

  /// The sync cycle when the stream was recorded. Used to detect stale events.
  int64_t RecordedSyncCycle;

  /// Mutex to safely access event fields.
  mutable std::mutex Mutex;

  friend struct AMDGPUStreamTy;
};

Error AMDGPUStreamTy::recordEvent(AMDGPUEventTy &Event) const {
  std::lock_guard<std::mutex> Lock(Mutex);

  if (size() > 0) {
    // Record the synchronize identifier (to detect stale recordings) and
    // the last valid stream's operation.
    Event.RecordedSyncCycle = SyncCycle;
    Event.RecordedSlot = last();

    assert(Event.RecordedSyncCycle >= 0 && "Invalid recorded sync cycle");
    assert(Event.RecordedSlot >= 0 && "Invalid recorded slot");
  } else {
    // The stream is empty, everything already completed, record nothing.
    Event.RecordedSyncCycle = -1;
    Event.RecordedSlot = -1;
  }
  return Plugin::success();
}

Error AMDGPUStreamTy::waitEvent(const AMDGPUEventTy &Event) {
  // Retrieve the recorded stream on the event.
  AMDGPUStreamTy &RecordedStream = *Event.RecordedStream;

  std::scoped_lock<std::mutex, std::mutex> Lock(Mutex, RecordedStream.Mutex);

  // The recorded stream already completed the operation because the synchronize
  // identifier is already outdated.
  if (RecordedStream.SyncCycle != (uint32_t)Event.RecordedSyncCycle)
    return Plugin::success();

  // Again, the recorded stream already completed the operation, the last
  // operation's output signal is satisfied.
  if (!RecordedStream.Slots[Event.RecordedSlot].Signal->load())
    return Plugin::success();

  // Otherwise, make the current stream wait on the other stream's operation.
  return waitOnStreamOperation(RecordedStream, Event.RecordedSlot);
}

struct AMDGPUStreamManagerTy final
    : GenericDeviceResourceManagerTy<AMDGPUResourceRef<AMDGPUStreamTy>> {
  using ResourceRef = AMDGPUResourceRef<AMDGPUStreamTy>;
  using ResourcePoolTy = GenericDeviceResourceManagerTy<ResourceRef>;

  AMDGPUStreamManagerTy(GenericDeviceTy &Device, hsa_agent_t HSAAgent)
      : GenericDeviceResourceManagerTy(Device), Device(Device),
        OMPX_QueueTracking("LIBOMPTARGET_AMDGPU_HSA_QUEUE_BUSY_TRACKING", true),
        OMPX_EnableQueueProfiling("LIBOMPTARGET_AMDGPU_ENABLE_QUEUE_PROFILING",
                                  false),
        NextQueue(0), Agent(HSAAgent) {}

  Error init(uint32_t InitialSize, int NumHSAQueues, int HSAQueueSize) {
    Queues = std::vector<AMDGPUQueueTy>(NumHSAQueues);
    QueueSize = HSAQueueSize;
    MaxNumQueues = NumHSAQueues;
    // Initialize one queue eagerly
    if (auto Err =
            Queues.front().init(Device, Agent, QueueSize, OMPX_EnableQueueProfiling))
      return Err;

    return GenericDeviceResourceManagerTy::init(InitialSize);
  }

  /// Deinitialize the resource pool and delete all resources. This function
  /// must be called before the destructor.
  Error deinit() override {
    // De-init all queues
    for (AMDGPUQueueTy &Queue : Queues) {
      if (auto Err = Queue.deinit())
        return Err;
    }

    return GenericDeviceResourceManagerTy::deinit();
  }

  /// Get a single stream from the pool or create new resources.
  virtual Error getResource(AMDGPUStreamTy *&StreamHandle) override {
    return getResourcesImpl(1, &StreamHandle, [this](AMDGPUStreamTy *&Handle) {
      return assignNextQueue(Handle);
    });
  }

  /// Return stream to the pool.
  virtual Error returnResource(AMDGPUStreamTy *StreamHandle) override {
    return returnResourceImpl(StreamHandle, [](AMDGPUStreamTy *Handle) {
      Handle->Queue->removeUser();
      return Plugin::success();
    });
  }

  /// Enable/disable profiling of the HSA queues.
  void setOmptQueueProfile(int Enable) {
    // If queue profiling is enabled with an env-var, it means that
    // profiling is already ON and should remain so all the time.
    if (OMPX_EnableQueueProfiling)
      return;
    for (auto &Q : Queues)
      if (Q.isInitialized())
        hsa_amd_profiling_set_profiler_enabled(Q.getHsaQueue(), Enable);
  }

private:
  /// Search for and assign an prefereably idle queue to the given Stream. If
  /// there is no queue without current users, choose the queue with the lowest
  /// user count. If utilization is ignored: use round robin selection.
  inline Error assignNextQueue(AMDGPUStreamTy *Stream) {
    // Start from zero when tracking utilization, otherwise: round robin policy.
    uint32_t Index = OMPX_QueueTracking ? 0 : NextQueue++ % MaxNumQueues;

    if (OMPX_QueueTracking) {
      // Find the least used queue.
      for (uint32_t I = 0; I < MaxNumQueues; ++I) {
        // Early exit when an initialized queue is idle.
        if (Queues[I].isInitialized() && Queues[I].getUserCount() == 0) {
          Index = I;
          break;
        }

        // Update the least used queue.
        if (Queues[Index].getUserCount() > Queues[I].getUserCount())
          Index = I;
      }
    }

    // Make sure the queue is initialized, then add user & assign.
    if (auto Err =
            Queues[Index].init(Device, Agent, QueueSize, OMPX_EnableQueueProfiling))
      return Err;
    Queues[Index].addUser();
    Stream->Queue = &Queues[Index];

    return Plugin::success();
  }

  /// The device associated with this stream.
  GenericDeviceTy &Device;

  /// Envar for controlling the tracking of busy HSA queues.
  BoolEnvar OMPX_QueueTracking;

  /// Envar for controlling whether to always profile HSA queues.
  BoolEnvar OMPX_EnableQueueProfiling;

  /// The next queue index to use for round robin selection.
  uint32_t NextQueue;

  /// The queues which are assigned to requested streams.
  std::vector<AMDGPUQueueTy> Queues;

  /// The corresponding device as HSA agent.
  hsa_agent_t Agent;

  /// The maximum number of queues.
  uint32_t MaxNumQueues;

  /// The size of created queues.
  uint32_t QueueSize;
};

/// Abstract class that holds the common members of the actual kernel devices
/// and the host device. Both types should inherit from this class.
struct AMDGenericDeviceTy {
  AMDGenericDeviceTy() {}

  virtual ~AMDGenericDeviceTy() {}

  /// Create all memory pools which the device has access to and classify them.
  Error initMemoryPools() {
    // Retrieve all memory pools from the device agent(s).
    Error Err = retrieveAllMemoryPools();
    if (Err)
      return Err;

    for (AMDGPUMemoryPoolTy *MemoryPool : AllMemoryPools) {
      // Initialize the memory pool and retrieve some basic info.
      Error Err = MemoryPool->init();
      if (Err)
        return Err;

      if (!MemoryPool->isGlobal())
        continue;

      // Classify the memory pools depending on their properties.
      if (MemoryPool->isFineGrained()) {
        FineGrainedMemoryPools.push_back(MemoryPool);
        if (MemoryPool->supportsKernelArgs())
          ArgsMemoryPools.push_back(MemoryPool);
      } else if (MemoryPool->isCoarseGrained()) {
        CoarseGrainedMemoryPools.push_back(MemoryPool);
      }
    }
    return Plugin::success();
  }

  /// Destroy all memory pools.
  Error deinitMemoryPools() {
    for (AMDGPUMemoryPoolTy *Pool : AllMemoryPools)
      delete Pool;

    AllMemoryPools.clear();
    FineGrainedMemoryPools.clear();
    CoarseGrainedMemoryPools.clear();
    ArgsMemoryPools.clear();

    return Plugin::success();
  }
  AMDGPUMemoryPoolTy *getCoarseGrainedMemoryPool() {
    return CoarseGrainedMemoryPools[0];
  }

  /// Retrieve and construct all memory pools from the device agent(s).
  virtual Error retrieveAllMemoryPools() = 0;

  /// Get the device agent.
  virtual hsa_agent_t getAgent() const = 0;

protected:
  /// Array of all memory pools available to the host agents.
  llvm::SmallVector<AMDGPUMemoryPoolTy *> AllMemoryPools;

  /// Array of fine-grained memory pools available to the host agents.
  llvm::SmallVector<AMDGPUMemoryPoolTy *> FineGrainedMemoryPools;

  /// Array of coarse-grained memory pools available to the host agents.
  llvm::SmallVector<AMDGPUMemoryPoolTy *> CoarseGrainedMemoryPools;

  /// Array of kernel args memory pools available to the host agents.
  llvm::SmallVector<AMDGPUMemoryPoolTy *> ArgsMemoryPools;
};

/// Class representing the host device. This host device may have more than one
/// HSA host agent. We aggregate all its resources into the same instance.
struct AMDHostDeviceTy : public AMDGenericDeviceTy {
  /// Create a host device from an array of host agents.
  AMDHostDeviceTy(AMDGPUPluginTy &Plugin,
                  const llvm::SmallVector<hsa_agent_t> &HostAgents)
      : AMDGenericDeviceTy(), Agents(HostAgents), ArgsMemoryManager(Plugin),
        PinnedMemoryManager(Plugin) {
    assert(HostAgents.size() && "No host agent found");
  }

  /// Initialize the host device memory pools and the memory managers for
  /// kernel args and host pinned memory allocations.
  Error init() {
    if (auto Err = initMemoryPools())
      return Err;

    if (auto Err = ArgsMemoryManager.init(getArgsMemoryPool()))
      return Err;

    if (auto Err = PinnedMemoryManager.init(getFineGrainedMemoryPool()))
      return Err;

    return Plugin::success();
  }

  /// Deinitialize memory pools and managers.
  Error deinit() {
    if (auto Err = deinitMemoryPools())
      return Err;

    if (auto Err = ArgsMemoryManager.deinit())
      return Err;

    if (auto Err = PinnedMemoryManager.deinit())
      return Err;

    return Plugin::success();
  }

  /// Retrieve and construct all memory pools from the host agents.
  Error retrieveAllMemoryPools() override {
    // Iterate through the available pools across the host agents.
    for (hsa_agent_t Agent : Agents) {
      Error Err = utils::iterateAgentMemoryPools(
          Agent, [&](hsa_amd_memory_pool_t HSAMemoryPool) {
            AMDGPUMemoryPoolTy *MemoryPool =
                new AMDGPUMemoryPoolTy(HSAMemoryPool);
            AllMemoryPools.push_back(MemoryPool);
            return HSA_STATUS_SUCCESS;
          });
      if (Err)
        return Err;
    }
    return Plugin::success();
  }

  /// Get one of the host agents. Return always the first agent.
  hsa_agent_t getAgent() const override { return Agents[0]; }

  /// Get a memory pool for fine-grained allocations.
  AMDGPUMemoryPoolTy &getFineGrainedMemoryPool() {
    assert(!FineGrainedMemoryPools.empty() && "No fine-grained mempool");
    // Retrive any memory pool.
    return *FineGrainedMemoryPools[0];
  }

  AMDGPUMemoryPoolTy &getCoarseGrainedMemoryPool() {
    assert(!CoarseGrainedMemoryPools.empty() && "No coarse-grained mempool");
    // Retrive any memory pool.
    return *CoarseGrainedMemoryPools[0];
  }

  /// Get a memory pool for kernel args allocations.
  AMDGPUMemoryPoolTy &getArgsMemoryPool() {
    assert(!ArgsMemoryPools.empty() && "No kernelargs mempool");
    // Retrieve any memory pool.
    return *ArgsMemoryPools[0];
  }

  /// Getters for kernel args and host pinned memory managers.
  AMDGPUMemoryManagerTy &getArgsMemoryManager() { return ArgsMemoryManager; }
  AMDGPUMemoryManagerTy &getPinnedMemoryManager() {
    return PinnedMemoryManager;
  }

private:
  /// Array of agents on the host side.
  const llvm::SmallVector<hsa_agent_t> Agents;

  // Memory manager for kernel arguments.
  AMDGPUMemoryManagerTy ArgsMemoryManager;

  // Memory manager for pinned memory.
  AMDGPUMemoryManagerTy PinnedMemoryManager;
};

/// Class implementing the AMDGPU device functionalities which derives from the
/// generic device class.
struct AMDGPUDeviceTy : public GenericDeviceTy, AMDGenericDeviceTy {
  // Create an AMDGPU device with a device id and default AMDGPU grid values.
  AMDGPUDeviceTy(GenericPluginTy &Plugin, int32_t DeviceId, int32_t NumDevices,
                 AMDHostDeviceTy &HostDevice, hsa_agent_t Agent)
      : GenericDeviceTy(Plugin, DeviceId, NumDevices, {}), AMDGenericDeviceTy(),
        OMPX_NumQueues("LIBOMPTARGET_AMDGPU_NUM_HSA_QUEUES", 4),
        OMPX_QueueSize("LIBOMPTARGET_AMDGPU_HSA_QUEUE_SIZE", 512),
        OMPX_DefaultTeamsPerCU("LIBOMPTARGET_AMDGPU_TEAMS_PER_CU", 6),
        OMPX_GenericSpmdTeamsPerCU(
            "LIBOMPTARGET_AMDGPU_GENERIC_SPMD_TEAMS_PER_CU", 0),
        OMPX_BigJumpLoopTeamsPerCU(
            "LIBOMPTARGET_AMDGPU_BIG_JUMP_LOOP_TEAMS_PER_CU", 0),
        OMPX_LowTripCount("LIBOMPTARGET_AMDGPU_LOW_TRIPCOUNT", 2000),
        OMPX_SmallBlockSize("LIBOMPTARGET_MIN_THREADS_FOR_LOW_TRIP_COUNT", 8),
        OMPX_NumBlocksForLowTripcount("LIBOMPTARGET_BLOCKS_FOR_LOW_TRIP_COUNT",
                                      0),
        OMPX_WavesPerCUForLowTripcount(
            "LIBOMPTARGET_WAVES_PER_CU_FOR_LOW_TRIP_COUNT", 0),
        OMPX_AdjustNumTeamsForSmallBlockSize("LIBOMPTARGET_AMDGPU_ADJUST_TEAMS",
                                             0),
        OMPX_AdjustNumTeamsForXteamRedSmallBlockSize(
            "LIBOMPTARGET_AMDGPU_ADJUST_XTEAM_RED_TEAMS", 0),
        OMPX_MaxAsyncCopyBytes("LIBOMPTARGET_AMDGPU_MAX_ASYNC_COPY_BYTES",
                               1 * 1024 * 1024), // 1MB
        OMPX_InitialNumSignals("LIBOMPTARGET_AMDGPU_NUM_INITIAL_HSA_SIGNALS",
                               64),
        OMPX_ForceSyncRegions("OMPX_FORCE_SYNC_REGIONS", 0),
        OMPX_StreamBusyWait("LIBOMPTARGET_AMDGPU_STREAM_BUSYWAIT", 2000000),
        OMPX_UseMultipleSdmaEngines(
            // setting default to true here appears to solve random sdma problem
            "LIBOMPTARGET_AMDGPU_USE_MULTIPLE_SDMA_ENGINES", false),
        OMPX_ApuMaps("OMPX_APU_MAPS", false),
        OMPX_DisableUsmMaps("OMPX_DISABLE_USM_MAPS", false),
        OMPX_NoMapChecks("OMPX_DISABLE_MAPS", true),
        OMPX_StrictSanityChecks("OMPX_STRICT_SANITY_CHECKS", false),
        OMPX_SyncCopyBack("LIBOMPTARGET_SYNC_COPY_BACK", true),
        OMPX_APUPrefaultMemcopy("LIBOMPTARGET_APU_PREFAULT_MEMCOPY", "true"),
        OMPX_APUPrefaultMemcopySize("LIBOMPTARGET_APU_PREFAULT_MEMCOPY_SIZE",
                                    1 * 1024 * 1024), // 1MB
        AMDGPUStreamManager(*this, Agent), AMDGPUEventManager(*this),
        AMDGPUSignalManager(*this), Agent(Agent), HostDevice(HostDevice) {}

  ~AMDGPUDeviceTy() {}

  /// Return synchronous copy back status variable.
  bool syncCopyBack() const { return OMPX_SyncCopyBack; }

  /// Returns the maximum of HSA queues to create
  /// This reads a non-cached environment variable, don't call everywhere.
  uint32_t getMaxNumHsaQueues() const {
    // In case this environment variable is set: respect it and give it
    // precendence
    if (const char *GPUMaxHwQsEnv = getenv("GPU_MAX_HW_QUEUES")) {
      uint32_t MaxGPUHwQueues = std::atoi(GPUMaxHwQsEnv);
      if (MaxGPUHwQueues != OMPX_NumQueues)
        DP("Different numbers of maximum HSA queues specified. Using %u\n",
           MaxGPUHwQueues);

      return MaxGPUHwQueues;
    }
    // Otherwise use the regular environment variable
    return OMPX_NumQueues;
  }

  virtual uint32_t getOMPXGenericSpmdTeamsPerCU() const override {
    return OMPX_GenericSpmdTeamsPerCU;
  }
  virtual uint32_t getOMPXBigJumpLoopTeamsPerCU() const override {
    return OMPX_BigJumpLoopTeamsPerCU;
  }
  virtual uint32_t getOMPXLowTripCount() const override {
    return OMPX_LowTripCount;
  }
  virtual uint32_t getOMPXSmallBlockSize() const override {
    return OMPX_SmallBlockSize;
  }
  virtual uint32_t
  getOMPXNumBlocksForLowTripcount(uint64_t LoopTripCount) const override {
    uint32_t NumBlocks = 0;

    if (LoopTripCount > OMPX_LowTripCount)
      return NumBlocks;

    // if NumBlocksForLowTripcount is set, it has the highest priority.
    if (OMPX_NumBlocksForLowTripcount > 0) {
      NumBlocks = OMPX_NumBlocksForLowTripcount;
      DP("Small trip count loop: Using %u blocks\n", NumBlocks);
    }

    // Next, check if the waves per CU is set. This will launch a number of
    // blocks such that we only have at most OMPX_WavesPerCUForLowTripcount
    // waves per CU.
    if (OMPX_WavesPerCUForLowTripcount > 0) {
      // Compute the number of waves per block. For sizes smaller than a full
      // wave the size is 1.
      uint32_t WavesPerBlock = (uint32_t)((OMPX_SmallBlockSize - 1) / 64) + 1;
      DP("Small trip count loop: Using %u waves per block\n", WavesPerBlock);

      // We cannot return less than the number of CUs:
      if (WavesPerBlock >= OMPX_WavesPerCUForLowTripcount) {
        NumBlocks = NumComputeUnits;
        DP("Small trip count loop: Using 1 block per CU\n");
      } else {
        uint32_t BlocksPerCU =
            (uint32_t)(OMPX_WavesPerCUForLowTripcount / WavesPerBlock);
        DP("Small trip count loop: Using %u blocks per CU\n", BlocksPerCU);
        NumBlocks = (uint32_t)(BlocksPerCU * NumComputeUnits);
      }
    }

    // Adjust the number of blocks to the trip count if number of blocks x
    // threads is much larger than the loop trip count.
    if (NumBlocks) {
      if (LoopTripCount <= OMPX_SmallBlockSize)
        NumBlocks = 1;

      uint32_t MaxBlocks =
          (uint32_t)((LoopTripCount - 1) / OMPX_SmallBlockSize) + 1;
      if (NumBlocks > MaxBlocks) {
        NumBlocks = MaxBlocks;
        DP("Small trip count loop: number of blocks capped to %u to fit loop "
           "trip count\n",
           NumBlocks);
      }
    }
    return NumBlocks;
  }
  virtual uint32_t getOMPXAdjustNumTeamsForSmallBlockSize() const override {
    return OMPX_AdjustNumTeamsForSmallBlockSize;
  }
  virtual uint32_t
  getOMPXAdjustNumTeamsForXteamRedSmallBlockSize() const override {
    return OMPX_AdjustNumTeamsForXteamRedSmallBlockSize;
  }

  /// Initialize the device, its resources and get its properties.
  Error initImpl(GenericPluginTy &Plugin) override {
    // First setup all the memory pools.
    if (auto Err = initMemoryPools())
      return Err;

    OMPT_IF_ENABLED(::setOmptTicksToTime(););

#ifdef OMPT_SUPPORT
    // At init we capture two time points for host and device. The two
    // timepoints are spaced out to help smooth out their accuracy
    // differences.
    // libomp uses the CLOCK_REALTIME (via gettimeofday) to get
    // the value for omp_get_wtime. So we use the same clock here to calculate
    // the slope/offset and convert device time to omp_get_wtime via
    // translate_time.
    double HostRef1 = 0;
    uint64_t DeviceRef1 = 0;
#endif
    // Take the first timepoints.
    OMPT_IF_ENABLED(startH2DTimeRate(&HostRef1, &DeviceRef1););

    if (auto Err = preAllocateDeviceMemoryPool())
      return Err;

    char GPUName[64];
    if (auto Err = getDeviceAttr(HSA_AGENT_INFO_NAME, GPUName))
      return Err;
    ComputeUnitKind = GPUName;

    // Get the wavefront size.
    uint32_t WavefrontSize = 0;
    if (auto Err = getDeviceAttr(HSA_AGENT_INFO_WAVEFRONT_SIZE, WavefrontSize))
      return Err;
    GridValues.GV_Warp_Size = WavefrontSize;

    // Get the frequency of the steady clock. If the attribute is missing
    // assume running on an older libhsa and default to 0, omp_get_wtime
    // will be inaccurate but otherwise programs can still run.
    if (getDeviceAttrRaw(HSA_AMD_AGENT_INFO_TIMESTAMP_FREQUENCY,
                         ClockFrequency) != HSA_STATUS_SUCCESS)
      ClockFrequency = 0;

    // Load the grid values dependending on the wavefront.
    if (WavefrontSize == 32)
      GridValues = getAMDGPUGridValues<32>();
    else if (WavefrontSize == 64)
      GridValues = getAMDGPUGridValues<64>();
    else
      return Plugin::error("Unexpected AMDGPU wavefront %d", WavefrontSize);

    // To determine the correct scratch memory size per thread, we need to check
    // the device architecure generation. Hence, we slice the major GFX version
    // from the agent info (e.g. 'gfx90a' -> 9).
    StringRef Arch(ComputeUnitKind);
    unsigned GfxGen = 0u;
    if (!llvm::to_integer(Arch.slice(sizeof("gfx") - 1, Arch.size() - 2),
                          GfxGen))
      return Plugin::error("Invalid GFX architecture string");

    // TODO: Will try to eliminate this calculation, since its duplicated.
    // See: 'getMaxWaveScratchSize' in 'llvm/lib/Target/AMDGPU/GCNSubtarget.h'.
    // But we need to divide by WavefrontSize.
    // For generations pre-gfx11: use 13-bit field in units of 256-dword,
    // otherwise: 15-bit field in units of 64-dword.
    MaxThreadScratchSize = (GfxGen < 11)
                               ? ((256 * 4) / WavefrontSize) * ((1 << 13) - 1)
                               : ((64 * 4) / WavefrontSize) * ((1 << 15) - 1);

    // Get maximum number of workitems per workgroup.
    uint16_t WorkgroupMaxDim[3];
    if (auto Err =
            getDeviceAttr(HSA_AGENT_INFO_WORKGROUP_MAX_DIM, WorkgroupMaxDim))
      return Err;
    GridValues.GV_Max_WG_Size = WorkgroupMaxDim[0];

    // Get maximum number of workgroups.
    hsa_dim3_t GridMaxDim;
    if (auto Err = getDeviceAttr(HSA_AGENT_INFO_GRID_MAX_DIM, GridMaxDim))
      return Err;

    GridValues.GV_Max_Teams = GridMaxDim.x / GridValues.GV_Max_WG_Size;
    if (GridValues.GV_Max_Teams == 0)
      return Plugin::error("Maximum number of teams cannot be zero");

    // Compute the default number of teams.
    uint32_t ComputeUnits = 0;
    if (auto Err =
            getDeviceAttr(HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT, ComputeUnits))
      return Err;
    GridValues.GV_Default_Num_Teams = ComputeUnits * OMPX_DefaultTeamsPerCU;
    NumComputeUnits = ComputeUnits;

    uint32_t WavesPerCU = 0;
    if (auto Err =
            getDeviceAttr(HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU, WavesPerCU))
      return Err;
    HardwareParallelism = ComputeUnits * WavesPerCU;

    // Get maximum size of any device queues and maximum number of queues.
    uint32_t MaxQueueSize;
    if (auto Err = getDeviceAttr(HSA_AGENT_INFO_QUEUE_MAX_SIZE, MaxQueueSize))
      return Err;

    uint32_t MaxQueues;
    if (auto Err = getDeviceAttr(HSA_AGENT_INFO_QUEUES_MAX, MaxQueues))
      return Err;

    // Compute the number of queues and their size.
    OMPX_NumQueues = std::max(1U, std::min(OMPX_NumQueues.get(), MaxQueues));
    OMPX_QueueSize = std::min(OMPX_QueueSize.get(), MaxQueueSize);
    DP("Using a maximum of %u HSA queues\n", OMPX_NumQueues.get());

    // Initialize stream pool.
    if (auto Err = AMDGPUStreamManager.init(OMPX_InitialNumStreams,
                                            OMPX_NumQueues, OMPX_QueueSize))
      return Err;

    // Initialize event pool.
    if (auto Err = AMDGPUEventManager.init(OMPX_InitialNumEvents))
      return Err;

    // Initialize signal pool.
    if (auto Err = AMDGPUSignalManager.init(OMPX_InitialNumSignals))
      return Err;

    // Take the second timepoints and compute the required metadata.
    OMPT_IF_ENABLED(completeH2DTimeRate(HostRef1, DeviceRef1););

    uint32_t NumSdmaEngines = 0;
    if (auto Err =
            getDeviceAttr(HSA_AMD_AGENT_INFO_NUM_SDMA_ENG, NumSdmaEngines))
      return Err;
    DP("The number of SDMA Engines: %i\n", NumSdmaEngines);

    uint32_t NumXGmiEngines = 0;
    if (auto Err =
            getDeviceAttr(HSA_AMD_AGENT_INFO_NUM_SDMA_XGMI_ENG, NumXGmiEngines))
      return Err;
    DP("The number of XGMI Engines: %i\n", NumXGmiEngines);

    // Detect if XNACK is enabled
    auto TargeTripleAndFeaturesOrError =
        utils::getTargetTripleAndFeatures(Agent);
    if (!TargeTripleAndFeaturesOrError)
      return TargeTripleAndFeaturesOrError.takeError();
    if (static_cast<StringRef>(*TargeTripleAndFeaturesOrError)
            .contains("xnack+"))
      IsXnackEnabled = true;

    // detect if device is an APU.
    if (auto Err = checkIfAPU())
      return Err;

    // detect if device is GFX90a.
    if (auto Err = checkIfGFX90a())
      return Err;

    // detect if device is an MI300X.
    if (auto Err = checkIfMI300x())
      return Err;

    // detect special cases for MI200 and MI300A
    specialBehaviorHandling();

    // detect ROCm-specific environment variables
    // for map and zero-copy control
    // TODO: put them back in constructor
    //    readEnvVars();

    return Plugin::success();
  }

  /// Deinitialize the device and release its resources.
  Error deinitImpl() override {
    // Deinitialize the stream and event pools.
    if (auto Err = AMDGPUStreamManager.deinit())
      return Err;

    if (auto Err = AMDGPUEventManager.deinit())
      return Err;

    if (auto Err = AMDGPUSignalManager.deinit())
      return Err;

    // Close modules if necessary.
    if (!LoadedImages.empty()) {
      // Each image has its own module.
      for (DeviceImageTy *Image : LoadedImages) {
        AMDGPUDeviceImageTy &AMDImage =
            static_cast<AMDGPUDeviceImageTy &>(*Image);

        // Unload the executable of the image.
        if (auto Err = AMDImage.unloadExecutable())
          return Err;
      }
    }

    // Invalidate agent reference.
    Agent = {0};

    delete CoarseGrainMemoryTable;

    return Plugin::success();
  }

  virtual Error callGlobalConstructors(GenericPluginTy &Plugin,
                                       DeviceImageTy &Image) override {
    GenericGlobalHandlerTy &Handler = Plugin.getGlobalHandler();
    if (Handler.isSymbolInImage(*this, Image, "amdgcn.device.fini"))
      Image.setPendingGlobalDtors();

    return callGlobalCtorDtorCommon(Plugin, Image, /*IsCtor=*/true);
  }

  virtual Error callGlobalDestructors(GenericPluginTy &Plugin,
                                      DeviceImageTy &Image) override {
    if (Image.hasPendingGlobalDtors())
      return callGlobalCtorDtorCommon(Plugin, Image, /*IsCtor=*/false);
    return Plugin::success();
  }

  uint64_t getStreamBusyWaitMicroseconds() const { return OMPX_StreamBusyWait; }

  Expected<std::unique_ptr<MemoryBuffer>>
  doJITPostProcessing(std::unique_ptr<MemoryBuffer> MB) const override {

    // TODO: We should try to avoid materialization but there seems to be no
    // good linker interface w/o file i/o.
    SmallString<128> LinkerInputFilePath;
    std::error_code EC = sys::fs::createTemporaryFile("amdgpu-pre-link-jit",
                                                      "o", LinkerInputFilePath);
    if (EC)
      return Plugin::error("Failed to create temporary file for linker");

    // Write the file's contents to the output file.
    Expected<std::unique_ptr<FileOutputBuffer>> OutputOrErr =
        FileOutputBuffer::create(LinkerInputFilePath, MB->getBuffer().size());
    if (!OutputOrErr)
      return OutputOrErr.takeError();
    std::unique_ptr<FileOutputBuffer> Output = std::move(*OutputOrErr);
    llvm::copy(MB->getBuffer(), Output->getBufferStart());
    if (Error E = Output->commit())
      return std::move(E);

    SmallString<128> LinkerOutputFilePath;
    EC = sys::fs::createTemporaryFile("amdgpu-pre-link-jit", "so",
                                      LinkerOutputFilePath);
    if (EC)
      return Plugin::error("Failed to create temporary file for linker");

    const auto &ErrorOrPath = sys::findProgramByName("lld");
    if (!ErrorOrPath)
      return createStringError(inconvertibleErrorCode(),
                               "Failed to find `lld` on the PATH.");

    std::string LLDPath = ErrorOrPath.get();
    INFO(OMP_INFOTYPE_PLUGIN_KERNEL, getDeviceId(),
         "Using `%s` to link JITed amdgcn ouput.", LLDPath.c_str());

    std::string MCPU = "-plugin-opt=mcpu=" + getComputeUnitKind();
    StringRef Args[] = {LLDPath,
                        "-flavor",
                        "gnu",
                        "--no-undefined",
                        "-shared",
                        MCPU,
                        "-o",
                        LinkerOutputFilePath.data(),
                        LinkerInputFilePath.data()};

    std::string Error;
    int RC = sys::ExecuteAndWait(LLDPath, Args, std::nullopt, {}, 0, 0, &Error);
    if (RC)
      return Plugin::error("Linking optimized bitcode failed: %s",
                           Error.c_str());

    auto BufferOrErr = MemoryBuffer::getFileOrSTDIN(LinkerOutputFilePath);
    if (!BufferOrErr)
      return Plugin::error("Failed to open temporary file for lld");

    // Clean up the temporary files afterwards.
    if (sys::fs::remove(LinkerOutputFilePath))
      return Plugin::error("Failed to remove temporary output file for lld");
    if (sys::fs::remove(LinkerInputFilePath))
      return Plugin::error("Failed to remove temporary input file for lld");

    return std::move(*BufferOrErr);
  }

  /// See GenericDeviceTy::getComputeUnitKind().
  std::string getComputeUnitKind() const override { return ComputeUnitKind; }

  uint32_t getNumComputeUnits() const override { return NumComputeUnits; }

  /// Returns the clock frequency for the given AMDGPU device.
  uint64_t getClockFrequency() const override { return ClockFrequency; }

  /// Allocate and construct an AMDGPU kernel.
  Expected<GenericKernelTy &> constructKernel(const char *Name) override {
    // Allocate and construct the AMDGPU kernel.
    AMDGPUKernelTy *AMDGPUKernel = Plugin.allocate<AMDGPUKernelTy>();
    if (!AMDGPUKernel)
      return Plugin::error("Failed to allocate memory for AMDGPU kernel");

    new (AMDGPUKernel) AMDGPUKernelTy(Name, Plugin.getGlobalHandler());

    return *AMDGPUKernel;
  }

  /// Set the current context to this device's context. Do nothing since the
  /// AMDGPU devices do not have the concept of contexts.
  Error setContext() override { return Plugin::success(); }

  /// AMDGPU returns the product of the number of compute units and the waves
  /// per compute unit.
  uint64_t getHardwareParallelism() const override {
    return HardwareParallelism;
  }

  /// We want to set up the RPC server for host services to the GPU if it is
  /// availible.
  bool shouldSetupRPCServer() const override {
    return libomptargetSupportsRPC();
  }

  /// The RPC interface should have enough space for all availible parallelism.
  uint64_t requestedRPCPortCount() const override {
    return getHardwareParallelism();
  }

  /// Get the stream of the asynchronous info sructure or get a new one.
  Error getStream(AsyncInfoWrapperTy &AsyncInfoWrapper,
                  AMDGPUStreamTy *&Stream) {
    // Get the stream (if any) from the async info.
    Stream = AsyncInfoWrapper.getQueueAs<AMDGPUStreamTy *>();
    if (!Stream) {
      // There was no stream; get an idle one.
      if (auto Err = AMDGPUStreamManager.getResource(Stream))
        return Err;

      // Modify the async info's stream.
      AsyncInfoWrapper.setQueueAs<AMDGPUStreamTy *>(Stream);
    }
    return Plugin::success();
  }

  /// Load the binary image into the device and allocate an image object.
  Expected<DeviceImageTy *> loadBinaryImpl(const __tgt_device_image *TgtImage,
                                           int32_t ImageId) override {
    // Allocate and initialize the image object.
    AMDGPUDeviceImageTy *AMDImage = Plugin.allocate<AMDGPUDeviceImageTy>();
    new (AMDImage) AMDGPUDeviceImageTy(ImageId, *this, TgtImage);

    // Load the HSA executable.
    if (Error Err = AMDImage->loadExecutable(*this))
      return std::move(Err);
    return AMDImage;
  }

  /// Allocate memory on the device or related to the device.
  void *allocate(size_t Size, void *, TargetAllocTy Kind) override;

  /// Deallocate memory on the device or related to the device.
  int free(void *TgtPtr, TargetAllocTy Kind) override {
    if (TgtPtr == nullptr)
      return OFFLOAD_SUCCESS;

    AMDGPUMemoryPoolTy *MemoryPool = nullptr;
    switch (Kind) {
    case TARGET_ALLOC_DEFAULT:
    case TARGET_ALLOC_DEVICE:
    case TARGET_ALLOC_DEVICE_NON_BLOCKING:
      MemoryPool = CoarseGrainedMemoryPools[0];
      break;
    case TARGET_ALLOC_HOST:
      MemoryPool = &HostDevice.getFineGrainedMemoryPool();
      break;
    case TARGET_ALLOC_SHARED:
      MemoryPool = &HostDevice.getFineGrainedMemoryPool();
      break;
    }

    if (!MemoryPool) {
      REPORT("No memory pool for the specified allocation kind\n");
      return OFFLOAD_FAIL;
    }

    if (Error Err = MemoryPool->deallocate(TgtPtr)) {
      REPORT("%s\n", toString(std::move(Err)).data());
      return OFFLOAD_FAIL;
    }

    return OFFLOAD_SUCCESS;
  }

  /// Synchronize current thread with the pending operations on the async info.
  Error synchronizeImpl(__tgt_async_info &AsyncInfo) override {
    AMDGPUStreamTy *Stream =
        reinterpret_cast<AMDGPUStreamTy *>(AsyncInfo.Queue);
    assert(Stream && "Invalid stream");

    if (auto Err = Stream->synchronize())
      return Err;

    // Once the stream is synchronized, return it to stream pool and reset
    // AsyncInfo. This is to make sure the synchronization only works for its
    // own tasks.
    AsyncInfo.Queue = nullptr;
    return AMDGPUStreamManager.returnResource(Stream);
  }

  /// Query for the completion of the pending operations on the async info.
  Error queryAsyncImpl(__tgt_async_info &AsyncInfo) override {
    AMDGPUStreamTy *Stream =
        reinterpret_cast<AMDGPUStreamTy *>(AsyncInfo.Queue);
    assert(Stream && "Invalid stream");

    auto CompletedOrErr = Stream->query();
    if (!CompletedOrErr)
      return CompletedOrErr.takeError();

    // Return if it the stream did not complete yet.
    if (!(*CompletedOrErr))
      return Plugin::success();

    // Once the stream is completed, return it to stream pool and reset
    // AsyncInfo. This is to make sure the synchronization only works for its
    // own tasks.
    AsyncInfo.Queue = nullptr;
    return AMDGPUStreamManager.returnResource(Stream);
  }

  /// Pin the host buffer and return the device pointer that should be used for
  /// device transfers.
  Expected<void *> dataLockImpl(void *HstPtr, int64_t Size) override {
    void *PinnedPtr = nullptr;

    hsa_status_t Status =
        hsa_amd_memory_lock(HstPtr, Size, nullptr, 0, &PinnedPtr);
    if (auto Err = Plugin::check(Status, "Error in hsa_amd_memory_lock: %s\n"))
      return std::move(Err);

    return PinnedPtr;
  }

  /// Unpin the host buffer.
  Error dataUnlockImpl(void *HstPtr) override {
    hsa_status_t Status = hsa_amd_memory_unlock(HstPtr);
    return Plugin::check(Status, "Error in hsa_amd_memory_unlock: %s\n");
  }

  /// Check through the HSA runtime whether the \p HstPtr buffer is pinned.
  Expected<bool> isPinnedPtrImpl(void *HstPtr, void *&BaseHstPtr,
                                 void *&BaseDevAccessiblePtr,
                                 size_t &BaseSize) const override {
    hsa_amd_pointer_info_t Info;
    Info.size = sizeof(hsa_amd_pointer_info_t);

    hsa_status_t Status =
        hsa_amd_pointer_info(HstPtr, &Info, /*Allocator=*/nullptr,
                             /* Number of accessible agents (out) */ nullptr,
                             /* Accessible agents */ nullptr);
    if (auto Err = Plugin::check(Status, "Error in hsa_amd_pointer_info: %s"))
      return std::move(Err);

    // The buffer may be locked or allocated through HSA allocators. Assume that
    // the buffer is host pinned if the runtime reports a HSA type.
    if (Info.type != HSA_EXT_POINTER_TYPE_LOCKED &&
        Info.type != HSA_EXT_POINTER_TYPE_HSA)
      return false;

    assert(Info.hostBaseAddress && "Invalid host pinned address");
    assert(Info.agentBaseAddress && "Invalid agent pinned address");
    assert(Info.sizeInBytes > 0 && "Invalid pinned allocation size");

    // Save the allocation info in the output parameters.
    BaseHstPtr = Info.hostBaseAddress;
    BaseDevAccessiblePtr = Info.agentBaseAddress;
    BaseSize = Info.sizeInBytes;

    return true;
  }

  /// Submit data to the device (host to device transfer).
  Error dataSubmitImpl(void *TgtPtr, const void *HstPtr, int64_t Size,
                       AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    AMDGPUStreamTy *Stream = nullptr;
    void *PinnedPtr = nullptr;

    // Obtain the OMPT-related callback data
    DP("OMPT-Async: dataSubmitImpl\n");
    auto LocalOmptEventInfo = getOrNullOmptEventInfo(AsyncInfoWrapper);

    // Prefault GPU page table in XNACK-Enabled case, on APUs,
    // under the assumption that explicitly allocated memory
    // will be fully accessed and that on-the-fly individual page faults
    // perform worse than whole memory faulting.
    if (OMPX_APUPrefaultMemcopy && Size >= OMPX_APUPrefaultMemcopySize &&
        IsAPU && IsXnackEnabled)
      if (auto Err = prepopulatePageTableImpl(const_cast<void *>(HstPtr), Size))
        return Err;

    // Use one-step asynchronous operation when host memory is already pinned.
    if (void *PinnedPtr =
            PinnedAllocs.getDeviceAccessiblePtrFromPinnedBuffer(HstPtr)) {
      if (auto Err = getStream(AsyncInfoWrapper, Stream))
        return Err;
      DP("OMPT-Async: Pinned Copy\n");
      return Stream->pushPinnedMemoryCopyAsync(TgtPtr, PinnedPtr, Size,
                                               std::move(LocalOmptEventInfo));
    }

    // For large transfers use synchronous behavior.
    // If OMPT is enabled or synchronous behavior is explicitly requested:
    if (OMPX_ForceSyncRegions || Size >= OMPX_MaxAsyncCopyBytes) {
      if (AsyncInfoWrapper.hasQueue())
        if (auto Err = synchronize(AsyncInfoWrapper))
          return Err;

      hsa_status_t Status;
      Status = hsa_amd_memory_lock(const_cast<void *>(HstPtr), Size, nullptr, 0,
                                   &PinnedPtr);
      if (auto Err =
              Plugin::check(Status, "Error in hsa_amd_memory_lock: %s\n"))
        return Err;

      AMDGPUSignalTy Signal;
      if (auto Err = Signal.init())
        return Err;

      DP("OMPT-Async: Sync Copy\n");
      if (auto Err = utils::asyncMemCopy(useMultipleSdmaEngines(), TgtPtr,
                                         Agent, PinnedPtr, Agent, Size, 0,
                                         nullptr, Signal.get()))
        return Err;

      if (auto Err = Signal.wait(getStreamBusyWaitMicroseconds()))
        return Err;

#ifdef OMPT_SUPPORT
      if (LocalOmptEventInfo) {
        OmptKernelTimingArgsAsyncTy OmptKernelTimingArgsAsync{
            Agent, &Signal, TicksToTime, std::move(LocalOmptEventInfo)};
        if (auto Err = timeDataTransferInNsAsync(&OmptKernelTimingArgsAsync))
          return Err;
      }
#endif

      if (auto Err = Signal.deinit())
        return Err;

      Status = hsa_amd_memory_unlock(const_cast<void *>(HstPtr));
      return Plugin::check(Status, "Error in hsa_amd_memory_unlock: %s\n");
    }

    // Otherwise, use two-step copy with an intermediate pinned host buffer.
    AMDGPUMemoryManagerTy &PinnedMemoryManager =
        HostDevice.getPinnedMemoryManager();
    if (auto Err = PinnedMemoryManager.allocate(Size, &PinnedPtr))
      return Err;

    if (auto Err = getStream(AsyncInfoWrapper, Stream))
      return Err;

    DP("OMPT-Async: ASync Copy\n");
    return Stream->pushMemoryCopyH2DAsync(TgtPtr, HstPtr, PinnedPtr, Size,
                                          PinnedMemoryManager,
                                          std::move(LocalOmptEventInfo));
  }

  /// Retrieve data from the device (device to host transfer).
  Error dataRetrieveImpl(void *HstPtr, const void *TgtPtr, int64_t Size,
                         AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    AMDGPUStreamTy *Stream = nullptr;
    void *PinnedPtr = nullptr;

    // Obtain the OMPT-related callback data
    DP("OMPT-Async: dataRetrieveImpl\n");
    auto LocalOmptEventInfo = getOrNullOmptEventInfo(AsyncInfoWrapper);

    // Prefault GPU page table in XNACK-Enabled case, on APUs,
    // under the assumption that explicitly allocated memory
    // will be fully accessed and that on-the-fly individual page faults
    // perform worse than whole memory faulting.
    if (OMPX_APUPrefaultMemcopy && Size >= OMPX_APUPrefaultMemcopySize &&
        IsAPU && IsXnackEnabled)
      if (auto Err = prepopulatePageTableImpl(const_cast<void *>(HstPtr), Size))
        return Err;

    // Use one-step asynchronous operation when host memory is already pinned.
    if (void *PinnedPtr =
            PinnedAllocs.getDeviceAccessiblePtrFromPinnedBuffer(HstPtr)) {
      if (auto Err = getStream(AsyncInfoWrapper, Stream))
        return Err;
      DP("OMPT-Async: Pinned Copy\n");
      return Stream->pushPinnedMemoryCopyAsync(PinnedPtr, TgtPtr, Size,
                                               std::move(LocalOmptEventInfo));
    }

    // For large transfers use synchronous behavior.
    // If OMPT is enabled or synchronous behavior is explicitly requested:
    if (OMPX_ForceSyncRegions || Size >= OMPX_MaxAsyncCopyBytes) {
      if (AsyncInfoWrapper.hasQueue())
        if (auto Err = synchronize(AsyncInfoWrapper))
          return Err;

      hsa_status_t Status;
      Status = hsa_amd_memory_lock(const_cast<void *>(HstPtr), Size, nullptr, 0,
                                   &PinnedPtr);
      if (auto Err =
              Plugin::check(Status, "Error in hsa_amd_memory_lock: %s\n"))
        return Err;

      AMDGPUSignalTy Signal;
      if (auto Err = Signal.init())
        return Err;

      if (auto Err = utils::asyncMemCopy(useMultipleSdmaEngines(), PinnedPtr,
                                         Agent, TgtPtr, Agent, Size, 0, nullptr,
                                         Signal.get()))
        return Err;

      if (auto Err = Signal.wait(getStreamBusyWaitMicroseconds()))
        return Err;

#ifdef OMPT_SUPPORT
      if (LocalOmptEventInfo) {
        OmptKernelTimingArgsAsyncTy OmptKernelTimingArgsAsync{
            Agent, &Signal, TicksToTime, std::move(LocalOmptEventInfo)};
        if (auto Err = timeDataTransferInNsAsync(&OmptKernelTimingArgsAsync))
          return Err;
      }
#endif

      if (auto Err = Signal.deinit())
        return Err;

      Status = hsa_amd_memory_unlock(const_cast<void *>(HstPtr));
      return Plugin::check(Status, "Error in hsa_amd_memory_unlock: %s\n");
    }

    // Otherwise, use two-step copy with an intermediate pinned host buffer.
    AMDGPUMemoryManagerTy &PinnedMemoryManager =
        HostDevice.getPinnedMemoryManager();
    if (auto Err = PinnedMemoryManager.allocate(Size, &PinnedPtr))
      return Err;

    if (auto Err = getStream(AsyncInfoWrapper, Stream))
      return Err;

    return Stream->pushMemoryCopyD2HAsync(HstPtr, TgtPtr, PinnedPtr, Size,
                                          PinnedMemoryManager,
                                          std::move(LocalOmptEventInfo));
  }

  /// Exchange data between two devices within the plugin.
  Error dataExchangeImpl(const void *SrcPtr, GenericDeviceTy &DstGenericDevice,
                         void *DstPtr, int64_t Size,
                         AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    AMDGPUDeviceTy &DstDevice = static_cast<AMDGPUDeviceTy &>(DstGenericDevice);

    DP("OMPT-Async: dataExchangeImpl\n");
    auto LocalOmptEventInfo = getOrNullOmptEventInfo(AsyncInfoWrapper);

    // For large transfers use synchronous behavior.
    // If OMPT is enabled or synchronous behavior is explicitly requested:
    if (OMPX_ForceSyncRegions || Size >= OMPX_MaxAsyncCopyBytes) {
      if (AsyncInfoWrapper.hasQueue())
        if (auto Err = synchronize(AsyncInfoWrapper))
          return Err;

      AMDGPUSignalTy Signal;
      if (auto Err = Signal.init())
        return Err;

      if (auto Err = utils::asyncMemCopy(
              useMultipleSdmaEngines(), DstPtr, DstDevice.getAgent(), SrcPtr,
              getAgent(), (uint64_t)Size, 0, nullptr, Signal.get()))
        return Err;

      if (auto Err = Signal.wait(getStreamBusyWaitMicroseconds()))
        return Err;

#ifdef OMPT_SUPPORT
      if (LocalOmptEventInfo) {
        OmptKernelTimingArgsAsyncTy OmptKernelTimingArgsAsync{
            Agent, &Signal, TicksToTime, std::move(LocalOmptEventInfo)};
        if (auto Err = timeDataTransferInNsAsync(&OmptKernelTimingArgsAsync))
          return Err;
      }
#endif

      return Signal.deinit();
    }

    AMDGPUStreamTy *Stream = nullptr;
    if (auto Err = getStream(AsyncInfoWrapper, Stream))
      return Err;
    if (Size <= 0)
      return Plugin::success();

    return Stream->pushMemoryCopyD2DAsync(DstPtr, DstDevice.getAgent(), SrcPtr,
                                          getAgent(), (uint64_t)Size,
                                          std::move(LocalOmptEventInfo));
  }

  /// Initialize the async info for interoperability purposes.
  Error initAsyncInfoImpl(AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    // TODO: Implement this function.
    return Plugin::success();
  }

  /// Initialize the device info for interoperability purposes.
  Error initDeviceInfoImpl(__tgt_device_info *DeviceInfo) override {
    DeviceInfo->Context = nullptr;

    if (!DeviceInfo->Device)
      DeviceInfo->Device = reinterpret_cast<void *>(Agent.handle);

    return Plugin::success();
  }

  Error setCoarseGrainMemoryImpl(void *ptr, int64_t size,
                                 bool set_attr = true) override final {
    // If the table has not yet been created, check if the gpu arch is
    // MI200 and create it.
    if (!IsEquippedWithGFX90A)
      return Plugin::success();
    if (!CoarseGrainMemoryTable)
      CoarseGrainMemoryTable = new AMDGPUMemTypeBitFieldTable(
          AMDGPU_X86_64_SystemConfiguration::max_addressable_byte +
              1, // memory size
          AMDGPU_X86_64_SystemConfiguration::page_size);

    if (CoarseGrainMemoryTable->contains((const uintptr_t)ptr, size))
      return Plugin::success();

    // track coarse grain memory pages in local table for user queries.
    CoarseGrainMemoryTable->insert((const uintptr_t)ptr, size);

    if (set_attr) {
      // Ask ROCr to turn [ptr, ptr+size-1] pages to
      // coarse grain.
      hsa_amd_svm_attribute_pair_t tt;
      tt.attribute = HSA_AMD_SVM_ATTRIB_GLOBAL_FLAG;
      tt.value = HSA_AMD_SVM_GLOBAL_FLAG_COARSE_GRAINED;
      hsa_status_t err = hsa_amd_svm_attributes_set(ptr, size, &tt, 1);
      if (err != HSA_STATUS_SUCCESS) {
        return Plugin::error("Failed to switch memotry to coarse grain mode.");
      }
    }

    return Plugin::success();
  }

  uint32_t queryCoarseGrainMemoryImpl(const void *ptr,
                                      int64_t size) override final {
    // If the table has not yet been created it means that
    // no memory has yet been set to coarse grain.
    if (!CoarseGrainMemoryTable)
      return 0;

    return CoarseGrainMemoryTable->contains((const uintptr_t)ptr, size);
  }

  Error prepopulatePageTableImpl(void *ptr, int64_t size) override final {
    // Instruct runtimes that the [ptr, ptr+size-1] pages will be accessed by
    // devices but should not be migrated (only perform page faults, if needed).
    hsa_amd_svm_attribute_pair_t tt;
    tt.attribute = HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE_IN_PLACE;
    tt.value = Agent.handle;
    hsa_status_t err = hsa_amd_svm_attributes_set(ptr, size, &tt, 1);
    if (err != HSA_STATUS_SUCCESS) {
      return Plugin::error("Failed to prepopulate GPU page table.");
    }

    return Plugin::success();
  }

  /// Create an event.
  Error createEventImpl(void **EventPtrStorage) override {
    AMDGPUEventTy **Event = reinterpret_cast<AMDGPUEventTy **>(EventPtrStorage);
    return AMDGPUEventManager.getResource(*Event);
  }

  /// Destroy a previously created event.
  Error destroyEventImpl(void *EventPtr) override {
    AMDGPUEventTy *Event = reinterpret_cast<AMDGPUEventTy *>(EventPtr);
    return AMDGPUEventManager.returnResource(Event);
  }

  /// Record the event.
  Error recordEventImpl(void *EventPtr,
                        AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    AMDGPUEventTy *Event = reinterpret_cast<AMDGPUEventTy *>(EventPtr);
    assert(Event && "Invalid event");

    AMDGPUStreamTy *Stream = nullptr;
    if (auto Err = getStream(AsyncInfoWrapper, Stream))
      return Err;

    return Event->record(*Stream);
  }

  /// Make the stream wait on the event.
  Error waitEventImpl(void *EventPtr,
                      AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    AMDGPUEventTy *Event = reinterpret_cast<AMDGPUEventTy *>(EventPtr);

    AMDGPUStreamTy *Stream = nullptr;
    if (auto Err = getStream(AsyncInfoWrapper, Stream))
      return Err;

    return Event->wait(*Stream);
  }

  /// Synchronize the current thread with the event.
  Error syncEventImpl(void *EventPtr) override {
    return Plugin::error("Synchronize event not implemented");
  }

  /// Print information about the device.
  Error obtainInfoImpl(InfoQueueTy &Info) override {
    char TmpChar[1000];
    const char *TmpCharPtr = "Unknown";
    uint16_t Major, Minor;
    uint32_t TmpUInt, TmpUInt2;
    uint32_t CacheSize[4];
    size_t TmpSt;
    bool TmpBool;
    uint16_t WorkgrpMaxDim[3];
    hsa_dim3_t GridMaxDim;
    hsa_status_t Status, Status2;

    Status = hsa_system_get_info(HSA_SYSTEM_INFO_VERSION_MAJOR, &Major);
    Status2 = hsa_system_get_info(HSA_SYSTEM_INFO_VERSION_MINOR, &Minor);
    if (Status == HSA_STATUS_SUCCESS && Status2 == HSA_STATUS_SUCCESS)
      Info.add("HSA Runtime Version",
               std::to_string(Major) + "." + std::to_string(Minor));

    Info.add("HSA OpenMP Device Number", DeviceId);

    Status = getDeviceAttrRaw(HSA_AMD_AGENT_INFO_PRODUCT_NAME, TmpChar);
    if (Status == HSA_STATUS_SUCCESS)
      Info.add("Product Name", TmpChar);

    Status = getDeviceAttrRaw(HSA_AGENT_INFO_NAME, TmpChar);
    if (Status == HSA_STATUS_SUCCESS)
      Info.add("Device Name", TmpChar);

    Status = getDeviceAttrRaw(HSA_AGENT_INFO_VENDOR_NAME, TmpChar);
    if (Status == HSA_STATUS_SUCCESS)
      Info.add("Vendor Name", TmpChar);

    hsa_device_type_t DevType;
    Status = getDeviceAttrRaw(HSA_AGENT_INFO_DEVICE, DevType);
    if (Status == HSA_STATUS_SUCCESS) {
      switch (DevType) {
      case HSA_DEVICE_TYPE_CPU:
        TmpCharPtr = "CPU";
        break;
      case HSA_DEVICE_TYPE_GPU:
        TmpCharPtr = "GPU";
        break;
      case HSA_DEVICE_TYPE_DSP:
        TmpCharPtr = "DSP";
        break;
      }
      Info.add("Device Type", TmpCharPtr);
    }

    Status = getDeviceAttrRaw(HSA_AGENT_INFO_QUEUES_MAX, TmpUInt);
    if (Status == HSA_STATUS_SUCCESS)
      Info.add("Max Queues", TmpUInt);

    Status = getDeviceAttrRaw(HSA_AGENT_INFO_QUEUE_MIN_SIZE, TmpUInt);
    if (Status == HSA_STATUS_SUCCESS)
      Info.add("Queue Min Size", TmpUInt);

    Status = getDeviceAttrRaw(HSA_AGENT_INFO_QUEUE_MAX_SIZE, TmpUInt);
    if (Status == HSA_STATUS_SUCCESS)
      Info.add("Queue Max Size", TmpUInt);

    // FIXME: This is deprecated according to HSA documentation. But using
    // hsa_agent_iterate_caches and hsa_cache_get_info breaks execution during
    // runtime.
    Status = getDeviceAttrRaw(HSA_AGENT_INFO_CACHE_SIZE, CacheSize);
    if (Status == HSA_STATUS_SUCCESS) {
      Info.add("Cache");

      for (int I = 0; I < 4; I++)
        if (CacheSize[I])
          Info.add<InfoLevel2>("L" + std::to_string(I), CacheSize[I]);
    }

    Status = getDeviceAttrRaw(HSA_AMD_AGENT_INFO_CACHELINE_SIZE, TmpUInt);
    if (Status == HSA_STATUS_SUCCESS)
      Info.add("Cacheline Size", TmpUInt);

    Status = getDeviceAttrRaw(HSA_AMD_AGENT_INFO_MAX_CLOCK_FREQUENCY, TmpUInt);
    if (Status == HSA_STATUS_SUCCESS)
      Info.add("Max Clock Freq", TmpUInt, "MHz");

    Status = getDeviceAttrRaw(HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT, TmpUInt);
    if (Status == HSA_STATUS_SUCCESS)
      Info.add("Compute Units", TmpUInt);

    Status = getDeviceAttrRaw(HSA_AMD_AGENT_INFO_NUM_SIMDS_PER_CU, TmpUInt);
    if (Status == HSA_STATUS_SUCCESS)
      Info.add("SIMD per CU", TmpUInt);

    Status = getDeviceAttrRaw(HSA_AGENT_INFO_FAST_F16_OPERATION, TmpBool);
    if (Status == HSA_STATUS_SUCCESS)
      Info.add("Fast F16 Operation", TmpBool);

    Status = getDeviceAttrRaw(HSA_AGENT_INFO_WAVEFRONT_SIZE, TmpUInt2);
    if (Status == HSA_STATUS_SUCCESS)
      Info.add("Wavefront Size", TmpUInt2);

    Status = getDeviceAttrRaw(HSA_AGENT_INFO_WORKGROUP_MAX_SIZE, TmpUInt);
    if (Status == HSA_STATUS_SUCCESS)
      Info.add("Workgroup Max Size", TmpUInt);

    Status = getDeviceAttrRaw(HSA_AGENT_INFO_WORKGROUP_MAX_DIM, WorkgrpMaxDim);
    if (Status == HSA_STATUS_SUCCESS) {
      Info.add("Workgroup Max Size per Dimension");
      Info.add<InfoLevel2>("x", WorkgrpMaxDim[0]);
      Info.add<InfoLevel2>("y", WorkgrpMaxDim[1]);
      Info.add<InfoLevel2>("z", WorkgrpMaxDim[2]);
    }

    Status = getDeviceAttrRaw(
        (hsa_agent_info_t)HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU, TmpUInt);
    if (Status == HSA_STATUS_SUCCESS) {
      Info.add("Max Waves Per CU", TmpUInt);
      Info.add("Max Work-item Per CU", TmpUInt * TmpUInt2);
    }

    Status = getDeviceAttrRaw(HSA_AGENT_INFO_GRID_MAX_SIZE, TmpUInt);
    if (Status == HSA_STATUS_SUCCESS)
      Info.add("Grid Max Size", TmpUInt);

    Status = getDeviceAttrRaw(HSA_AGENT_INFO_GRID_MAX_DIM, GridMaxDim);
    if (Status == HSA_STATUS_SUCCESS) {
      Info.add("Grid Max Size per Dimension");
      Info.add<InfoLevel2>("x", GridMaxDim.x);
      Info.add<InfoLevel2>("y", GridMaxDim.y);
      Info.add<InfoLevel2>("z", GridMaxDim.z);
    }

    Status = getDeviceAttrRaw(HSA_AGENT_INFO_FBARRIER_MAX_SIZE, TmpUInt);
    if (Status == HSA_STATUS_SUCCESS)
      Info.add("Max fbarriers/Workgrp", TmpUInt);

    Info.add("Memory Pools");
    for (AMDGPUMemoryPoolTy *Pool : AllMemoryPools) {
      std::string TmpStr, TmpStr2;

      if (Pool->isGlobal())
        TmpStr = "Global";
      else if (Pool->isReadOnly())
        TmpStr = "ReadOnly";
      else if (Pool->isPrivate())
        TmpStr = "Private";
      else if (Pool->isGroup())
        TmpStr = "Group";
      else
        TmpStr = "Unknown";

      Info.add<InfoLevel2>(std::string("Pool ") + TmpStr);

      if (Pool->isGlobal()) {
        if (Pool->isFineGrained())
          TmpStr2 += "Fine Grained ";
        if (Pool->isCoarseGrained())
          TmpStr2 += "Coarse Grained ";
        if (Pool->supportsKernelArgs())
          TmpStr2 += "Kernarg ";

        Info.add<InfoLevel3>("Flags", TmpStr2);
      }

      Status = Pool->getAttrRaw(HSA_AMD_MEMORY_POOL_INFO_SIZE, TmpSt);
      if (Status == HSA_STATUS_SUCCESS)
        Info.add<InfoLevel3>("Size", TmpSt, "bytes");

      Status = Pool->getAttrRaw(HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,
                                TmpBool);
      if (Status == HSA_STATUS_SUCCESS)
        Info.add<InfoLevel3>("Allocatable", TmpBool);

      Status = Pool->getAttrRaw(HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE,
                                TmpSt);
      if (Status == HSA_STATUS_SUCCESS)
        Info.add<InfoLevel3>("Runtime Alloc Granule", TmpSt, "bytes");

      Status = Pool->getAttrRaw(
          HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALIGNMENT, TmpSt);
      if (Status == HSA_STATUS_SUCCESS)
        Info.add<InfoLevel3>("Runtime Alloc Alignment", TmpSt, "bytes");

      Status =
          Pool->getAttrRaw(HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL, TmpBool);
      if (Status == HSA_STATUS_SUCCESS)
        Info.add<InfoLevel3>("Accessable by all", TmpBool);
    }

    Info.add("ISAs");
    auto Err = utils::iterateAgentISAs(getAgent(), [&](hsa_isa_t ISA) {
      Status = hsa_isa_get_info_alt(ISA, HSA_ISA_INFO_NAME, TmpChar);
      if (Status == HSA_STATUS_SUCCESS)
        Info.add<InfoLevel2>("Name", TmpChar);

      return Status;
    });

    // Silently consume the error.
    if (Err)
      consumeError(std::move(Err));

    return Plugin::success();
  }

  /// Get the HSA system timestamps for the input signal associated with an
  /// async copy and pass the information to libomptarget
  void recordCopyTimingInNs(hsa_signal_t signal) {
    hsa_amd_profiling_async_copy_time_t time_rec;
    hsa_status_t Status =
        hsa_amd_profiling_get_async_copy_time(signal, &time_rec);
    if (Status != HSA_STATUS_SUCCESS) {
      DP("Error while getting async copy time\n");
      return;
    }
#ifdef OMPT_SUPPORT
    ompt::setOmptTimestamp(time_rec.start * TicksToTime,
                           time_rec.end * TicksToTime);
#endif
  }

  /// Returns true if auto zero-copy the best configuration for the current
  /// arch.
  /// On AMDGPUs, automatic zero-copy is turned on
  /// when running on an APU with XNACK (unified memory) support
  /// enabled. On discrete GPUs, automatic zero-copy is triggered
  /// if the user sets the environment variable OMPX_APU_MAPS=1
  /// and if XNACK is enabled. The rationale is that zero-copy
  /// is the best configuration (performance, memory footprint) on APUs,
  /// while it is often not the best on discrete GPUs.
  /// XNACK can be enabled with a kernel boot parameter or with
  /// the HSA_XNACK environment variable.
  bool useAutoZeroCopyImpl() override {
    return ((IsAPU || OMPX_ApuMaps) && IsXnackEnabled);
  }

  /// Performs sanity checks on the selected zero-copy configuration and prints
  /// diagnostic information.
  Error zeroCopySanityChecksAndDiagImpl(bool isUnifiedSharedMemory,
                                        bool isAutoZeroCopy,
                                        bool isEagerMaps) override {
    // Implementation sanity checks: either unified_shared_memory or auto
    // zero-copy, not both
    if (isUnifiedSharedMemory && isAutoZeroCopy)
      return Plugin::error("Internal runtime error: cannot be both "
                           "unified_shared_memory and auto zero-copy.");

    if (IsXnackEnabled)
      INFO(OMP_INFOTYPE_USER_DIAGNOSTIC, getDeviceId(), "XNACK is enabled.\n");
    else
      INFO(OMP_INFOTYPE_USER_DIAGNOSTIC, getDeviceId(), "XNACK is disabled.\n");
    if (isUnifiedSharedMemory)
      INFO(OMP_INFOTYPE_USER_DIAGNOSTIC, getDeviceId(),
           "Application configured to run in zero-copy using "
           "unified_shared_memory.\n");
    else if (isAutoZeroCopy)
      INFO(
          OMP_INFOTYPE_USER_DIAGNOSTIC, getDeviceId(),
          "Application configured to run in zero-copy using auto zero-copy.\n");
    if (isEagerMaps)
      INFO(OMP_INFOTYPE_USER_DIAGNOSTIC, getDeviceId(),
           "Requested pre-faulting of GPU page tables.\n");

    // Sanity checks: selecting unified_shared_memory with XNACK-Disabled
    // triggers a warning that can be turned into a fatal error using an
    // environment variable.
    if (isUnifiedSharedMemory && !IsXnackEnabled) {
      MESSAGE0(
          "Running a program that requires XNACK on a system where XNACK is "
          "disabled. This may cause problems when using an OS-allocated "
          "pointer "
          "inside a target region. "
          "Re-run with HSA_XNACK=1 to remove this warning.");
      if (OMPX_StrictSanityChecks)
        llvm_unreachable("User-requested hard stop on sanity check errors.");
    }
    return Plugin::success();
  }

  /// Getters and setters for stack and heap sizes.
  Error getDeviceStackSize(uint64_t &Value) override {
    Value = StackSize;
    return Plugin::success();
  }
  Error setDeviceStackSize(uint64_t Value) override {
    if (Value > MaxThreadScratchSize) {
      // Cap device scratch size.
      MESSAGE("Scratch memory size will be set to %d. Reason: Requested size "
              "%ld would exceed available resources.",
              MaxThreadScratchSize, Value);
      StackSize = MaxThreadScratchSize;
    } else {
      // Apply device scratch size, since it is within limits.
      StackSize = Value;
    }

    return Plugin::success();
  }
  Error getDeviceHeapSize(uint64_t &Value) override {
    Value = DeviceMemoryPoolSize;
    return Plugin::success();
  }
  Error setDeviceHeapSize(uint64_t Value) override {
    for (DeviceImageTy *Image : LoadedImages)
      if (auto Err = setupDeviceMemoryPool(Plugin, *Image, Value))
        return Err;
    DeviceMemoryPoolSize = Value;
    return Plugin::success();
  }

  Error getDeviceMemorySize(uint64_t &Value) override {
    for (AMDGPUMemoryPoolTy *Pool : AllMemoryPools) {
      if (Pool->isGlobal()) {
        hsa_status_t Status =
            Pool->getAttrRaw(HSA_AMD_MEMORY_POOL_INFO_SIZE, Value);
        return Plugin::check(Status, "Error in getting device memory size: %s");
      }
    }
    return Plugin::error("getDeviceMemorySize:: no global pool");
  }

  /// AMDGPU-specific function to get device attributes.
  template <typename Ty> Error getDeviceAttr(uint32_t Kind, Ty &Value) {
    hsa_status_t Status =
        hsa_agent_get_info(Agent, (hsa_agent_info_t)Kind, &Value);
    return Plugin::check(Status, "Error in hsa_agent_get_info: %s");
  }

  template <typename Ty>
  hsa_status_t getDeviceAttrRaw(uint32_t Kind, Ty &Value) {
    return hsa_agent_get_info(Agent, (hsa_agent_info_t)Kind, &Value);
  }

  /// Get the device agent.
  hsa_agent_t getAgent() const override { return Agent; }

  /// Get the signal manager.
  AMDGPUSignalManagerTy &getSignalManager() { return AMDGPUSignalManager; }

  /// Retrieve and construct all memory pools of the device agent.
  Error retrieveAllMemoryPools() override {
    // Iterate through the available pools of the device agent.
    return utils::iterateAgentMemoryPools(
        Agent, [&](hsa_amd_memory_pool_t HSAMemoryPool) {
          AMDGPUMemoryPoolTy *MemoryPool =
              Plugin.allocate<AMDGPUMemoryPoolTy>();
          new (MemoryPool) AMDGPUMemoryPoolTy(HSAMemoryPool);
          AllMemoryPools.push_back(MemoryPool);
          return HSA_STATUS_SUCCESS;
        });
  }

  /// Propagate the enable/disable profiling request to the StreamManager.
  void setOmptQueueProfile(int Enable) {
    AMDGPUStreamManager.setOmptQueueProfile(Enable);
  }

  /// Get the address of pointer to the preallocated device memory pool.
  void *getPreAllocatedDeviceMemoryPool() {
    return PreAllocatedDeviceMemoryPool;
  }

  /// Allocate and zero initialize a small memory pool from the coarse grained
  /// device memory of each device.
  Error preAllocateDeviceMemoryPool() {
    Error Err = retrieveAllMemoryPools();
    if (Err)
      return Plugin::error("Unable to retieve all memmory pools");

    void *DevPtr;
    for (AMDGPUMemoryPoolTy *MemoryPool : AllMemoryPools) {
      if (!MemoryPool->isGlobal())
        continue;

      if (MemoryPool->isCoarseGrained()) {
        DevPtr = nullptr;
        size_t PreAllocSize = utils::PER_DEVICE_PREALLOC_SIZE;

        Err = MemoryPool->allocate(PreAllocSize, &DevPtr);
        if (Err)
          return Plugin::error("Device memory pool preallocation failed");

        Err = MemoryPool->enableAccess(DevPtr, PreAllocSize, {getAgent()});
        if (Err)
          return Plugin::error("Preallocated device memory pool inaccessible");

        Err = MemoryPool->zeroInitializeMemory(DevPtr, PreAllocSize);
        if (Err)
          return Plugin::error(
              "Zero initialization of preallocated device memory pool failed");

        PreAllocatedDeviceMemoryPool = DevPtr;
      }
    }
    return Plugin::success();
  }

  bool useMultipleSdmaEngines() const { return OMPX_UseMultipleSdmaEngines; }

private:
  using AMDGPUEventRef = AMDGPUResourceRef<AMDGPUEventTy>;
  using AMDGPUEventManagerTy = GenericDeviceResourceManagerTy<AMDGPUEventRef>;

  /// Common method to invoke a single threaded constructor or destructor
  /// kernel by name.
  Error callGlobalCtorDtorCommon(GenericPluginTy &Plugin, DeviceImageTy &Image,
                                 bool IsCtor) {
    const char *KernelName =
        IsCtor ? "amdgcn.device.init" : "amdgcn.device.fini";
    // Perform a quick check for the named kernel in the image. The kernel
    // should be created by the 'amdgpu-lower-ctor-dtor' pass.
    GenericGlobalHandlerTy &Handler = Plugin.getGlobalHandler();
    if (IsCtor && !Handler.isSymbolInImage(*this, Image, KernelName))
      return Plugin::success();

    // Allocate and construct the AMDGPU kernel.
    AMDGPUKernelTy AMDGPUKernel(KernelName, Plugin.getGlobalHandler());
    if (auto Err = AMDGPUKernel.init(*this, Image))
      return Err;

    AsyncInfoWrapperTy AsyncInfoWrapper(*this, nullptr);

    KernelArgsTy KernelArgs = {};
    if (auto Err =
            AMDGPUKernel.launchImpl(*this, /*NumThread=*/1u,
                                    /*NumBlocks=*/1ul, KernelArgs,
                                    KernelLaunchParamsTy{}, AsyncInfoWrapper))
      return Err;

    Error Err = Plugin::success();
    AsyncInfoWrapper.finalize(Err);

    return Err;
  }

  /// Detect if current architecture is an APU.
  Error checkIfAPU() {
    // TODO: replace with ROCr API once it becomes available.
    // MI300A
    llvm::StringRef StrGfxName(ComputeUnitKind);
    IsAPU = llvm::StringSwitch<bool>(StrGfxName)
                .Case("gfx940", true)
                .Default(false);
    if (IsAPU)
      return Plugin::success();

    bool MayBeAPU = llvm::StringSwitch<bool>(StrGfxName)
                        .Case("gfx942", true)
                        .Default(false);
    if (!MayBeAPU) // not gfx90a, gfx940, gfx941, or or gfx942
      return Plugin::success();

    // Can be MI300A or MI300X
    uint32_t ChipID = 0;
    if (auto Err = getDeviceAttr(HSA_AMD_AGENT_INFO_CHIP_ID, ChipID))
      return Err;

    if (!(ChipID & 0x1))
      IsAPU = true;

    return Plugin::success();
  }

  Error checkIfGFX90a() {
    llvm::StringRef StrGfxName(ComputeUnitKind);
    IsEquippedWithGFX90A = llvm::StringSwitch<bool>(StrGfxName)
                               .Case("gfx90a", true)
                               .Default(false);
    return Plugin::success();
  }

  Error checkIfMI300x() {
    llvm::StringRef StrGfxName(ComputeUnitKind);
    IsEquippedWithMI300X = llvm::StringSwitch<bool>(StrGfxName)
                               .Case("gfx941", true)
                               .Default(false);

    if (IsEquippedWithMI300X)
      return Plugin::success();

    bool isMI300 = llvm::StringSwitch<bool>(StrGfxName)
                       .Case("gfx942", true)
                       .Default(false);
    if (!isMI300)
      return Plugin::success();

    // Can be MI300A or MI300X
    uint32_t ChipID = 0;
    if (auto Err = getDeviceAttr(HSA_AMD_AGENT_INFO_CHIP_ID, ChipID))
      return Err;

    if (ChipID & 0x1)
      IsEquippedWithMI300X = true;

    return Plugin::success();
  }

  /// Determines if
  /// - Map checks should be disabled
  /// - Coarse graining upon map on MI200 needs to be disabled.
  /// - Prefaulting GPU page tables on MI300A needs to be enabled.
  void specialBehaviorHandling() {
    if (OMPX_NoMapChecks.get() == false) {
      NoUSMMapChecks = false;
    }

    if (OMPX_DisableUsmMaps.get() == true) {
      EnableFineGrainedMemory = true;
    }
  }

  bool IsFineGrainedMemoryEnabledImpl() override final {
    return EnableFineGrainedMemory;
  }

  bool hasAPUDeviceImpl() override final { return IsAPU; }

  // TODO: move the following function in private section.
  bool hasMI300xDevice() { return IsEquippedWithMI300X; }

  /// Returns whether the device is a gfx90a.
  bool hasGfx90aDeviceImpl() override final { return IsEquippedWithGFX90A; }

  /// Returns whether AMD GPU supports unified memory in
  /// the current configuration.
  bool supportsUnifiedMemoryImpl() override final { return IsXnackEnabled; }

  /// Envar for controlling the number of HSA queues per device. High number of
  /// queues may degrade performance.
  UInt32Envar OMPX_NumQueues;

  /// Envar for controlling the size of each HSA queue. The size is the number
  /// of HSA packets a queue is expected to hold. It is also the number of HSA
  /// packets that can be pushed into each queue without waiting the driver to
  /// process them.
  UInt32Envar OMPX_QueueSize;

  /// Envar for controlling the default number of teams relative to the number
  /// of compute units (CUs) the device has:
  ///   #default_teams = OMPX_DefaultTeamsPerCU * #CUs.
  UInt32Envar OMPX_DefaultTeamsPerCU;

  /// Envar for controlling the number of teams relative to the number of
  /// compute units (CUs) for generic-SPMD kernels. 0 indicates that this value
  /// is not specified, so instead OMPX_DefaultTeamsPerCU should be used. If
  /// non-zero, the number of teams = OMPX_GenericSpmdTeamsPerCU * #CUs.
  UInt32Envar OMPX_GenericSpmdTeamsPerCU;

  /// Envar for controlling the number of teams relative to the number of
  /// compute units (CUs) for Big-Jump-Loop kernels. 0 indicates that this value
  /// is not specified. If non-zero, the number of teams =
  /// OMPX_BigJumpLoopTeamsPerCU * #CUs.
  UInt32Envar OMPX_BigJumpLoopTeamsPerCU;

  /// Envar specifying tripcount below which the blocksize should be adjusted.
  UInt32Envar OMPX_LowTripCount;

  /// Envar specifying a value till which the blocksize can be adjusted if the
  /// tripcount is low.
  UInt32Envar OMPX_SmallBlockSize;

  /// Envar for the number of blocks when the loop trip count is under the small
  /// trip count limit.
  /// The default value of 0 means that the number of blocks will be inferred by
  /// the existing getNumBlocks logic.
  UInt32Envar OMPX_NumBlocksForLowTripcount;

  /// Envar to set the number of waves per CU for small trip count loops. The
  /// number of blocks will be adjusted such that there are no more than the
  /// specified number of blocks per CU than this variable specifies. For
  /// example:
  /// Given:
  //     a GPU with CUs = 100
  ///    and OMPX_WavesPerCUForLowTripcount = 8
  ///    and a waves per block number of 4 (256 threads)
  /// The total number of blocks will be: 200
  UInt32Envar OMPX_WavesPerCUForLowTripcount;

  /// Envar to allow adjusting number of teams after small tripcount
  /// optimization. The default 0 means no adjustment of number of teams is
  /// done.
  UInt32Envar OMPX_AdjustNumTeamsForSmallBlockSize;

  /// Envar to allow scaling up the number of teams for Xteam-Reduction
  /// whenever the blocksize has been reduced from the default. The env-var
  /// default of 0 means that the scaling is not done by default.
  UInt32Envar OMPX_AdjustNumTeamsForXteamRedSmallBlockSize;

  /// Envar specifying the maximum size in bytes where the memory copies are
  /// asynchronous operations. Up to this transfer size, the memory copies are
  /// asychronous operations pushed to the corresponding stream. For larger
  /// transfers, they are synchronous transfers.
  UInt32Envar OMPX_MaxAsyncCopyBytes;

  /// Envar controlling the initial number of HSA signals per device. There is
  /// one manager of signals per device managing several pre-allocated signals.
  /// These signals are mainly used by AMDGPU streams. If needed, more signals
  /// will be created.
  UInt32Envar OMPX_InitialNumSignals;

  /// Envar to force synchronous target regions. The default 0 uses an
  /// asynchronous implementation.
  UInt32Envar OMPX_ForceSyncRegions;
  /// switching to blocked state. The default 2000000 busywaits for 2 seconds
  /// before going into a blocking HSA wait state. The unit for these variables
  /// are microseconds.
  UInt32Envar OMPX_StreamBusyWait;

  /// Use ROCm 5.7 interface for multiple SDMA engines
  BoolEnvar OMPX_UseMultipleSdmaEngines;

  /// Value of OMPX_APU_MAPS env var used to force
  /// automatic zero-copy behavior on non-APU GPUs.
  BoolEnvar OMPX_ApuMaps;

  /// Value of OMPX_DISABLE_USM_MAPS. Use on MI200
  /// systems to disable both device memory
  /// allocations and host-device memory copies upon
  /// map, and coarse graining of mapped variables.
  BoolEnvar OMPX_DisableUsmMaps;

  /// Value of OMPX_DISABLE_MAPS. Turns off map table checks
  /// in libomptarget in unified_shared_memory mode. Legacy:
  /// never turned to false (unified_shared_memory mode is
  /// currently always without map checks.
  BoolEnvar OMPX_NoMapChecks;

  /// Makes warnings turn into fatal errors
  BoolEnvar OMPX_StrictSanityChecks;

  /// Variable to hold synchronous copy back
  BoolEnvar OMPX_SyncCopyBack;

  /// On APUs, this env var indicates whether memory copy
  /// should be preceded by pre-faulting of host memory,
  /// to prevent page faults during the copy.
  BoolEnvar OMPX_APUPrefaultMemcopy;

  /// On APUs, when prefaulting host memory before a copy,
  /// this env var controls the size after which prefaulting
  /// is applied.
  UInt32Envar OMPX_APUPrefaultMemcopySize;

  /// Stream manager for AMDGPU streams.
  AMDGPUStreamManagerTy AMDGPUStreamManager;

  /// Event manager for AMDGPU events.
  AMDGPUEventManagerTy AMDGPUEventManager;

  /// Signal manager for AMDGPU signals.
  AMDGPUSignalManagerTy AMDGPUSignalManager;

  /// The agent handler corresponding to the device.
  hsa_agent_t Agent;

  /// The GPU architecture.
  std::string ComputeUnitKind;

  /// The number of CUs available in this device
  uint32_t NumComputeUnits;

  /// The frequency of the steady clock inside the device.
  uint64_t ClockFrequency;

  /// The total number of concurrent work items that can be running on the GPU.
  uint64_t HardwareParallelism;

  /// Reference to the host device.
  AMDHostDeviceTy &HostDevice;

  // Data structure used to keep track of coarse grain memory regions
  // on MI200 in unified_shared_memory programs only.
  AMDGPUMemTypeBitFieldTable *CoarseGrainMemoryTable = nullptr;

  /// Pointer to the preallocated device memory pool
  void *PreAllocatedDeviceMemoryPool;

  /// The current size of the global device memory pool (managed by us).
  uint64_t DeviceMemoryPoolSize = 1L << 29L /* 512MB */;

  /// The current size of the stack that will be used in cases where it could
  /// not be statically determined.
  /// Default: 1024, in conformity to hipLimitStackSize.
  uint32_t StackSize = 1024 /* 1 KB */;

  // The maximum scratch memory size per thread.
  // See COMPUTE_TMPRING_SIZE.WAVESIZE (divided by threads per wave).
  uint32_t MaxThreadScratchSize;

  /// Is the plugin associated with an APU?
  bool IsAPU = false;

  // Is the device an MI300X?
  bool IsEquippedWithMI300X = false;

  // Is the device an MI200?
  bool IsEquippedWithGFX90A = false;

  /// True if the system is configured with XNACK-Enabled.
  /// False otherwise.
  bool IsXnackEnabled = false;

  // Set by OMPX_DISABLE_USM_MAPS environment variable.
  // If set, fine graned memory is used for maps instead of coarse grained.
  bool EnableFineGrainedMemory = false;

  /// Set by OMPX_DISABLE_MAPS environment variable.
  // If false, map checks are performed also in unified_shared_memory mode.
  // TODO: this feature is non functional.
  bool NoUSMMapChecks = true;
};

Error AMDGPUDeviceImageTy::loadExecutable(const AMDGPUDeviceTy &Device) {
  hsa_status_t Status;
  Status = hsa_code_object_deserialize(getStart(), getSize(), "", &CodeObject);
  if (auto Err =
          Plugin::check(Status, "Error in hsa_code_object_deserialize: %s"))
    return Err;

  Status = hsa_executable_create_alt(
      HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_ZERO, "", &Executable);
  if (auto Err =
          Plugin::check(Status, "Error in hsa_executable_create_alt: %s"))
    return Err;

#if SANITIZER_AMDGPU
  Status = hsa_code_object_reader_create_from_memory(getStart(), getSize(),
                                                     &CodeObjectReader);
  if (auto Err = Plugin::check(
          Status, "Error in hsa_code_object_reader_from_memory: %s"))
    return Err;

  Status = hsa_executable_load_agent_code_object(Executable, Device.getAgent(),
                                                 CodeObjectReader, "", nullptr);
  if (auto Err =
          Plugin::check(Status, "Error in hsa_executable_load_code_object: %s"))
    return Err;
#else
  Status = hsa_executable_load_code_object(Executable, Device.getAgent(),
                                           CodeObject, "");
  if (auto Err =
          Plugin::check(Status, "Error in hsa_executable_load_code_object: %s"))
    return Err;
#endif

  Status = hsa_executable_freeze(Executable, "");
  if (auto Err = Plugin::check(Status, "Error in hsa_executable_freeze: %s"))
    return Err;

  uint32_t Result;
  Status = hsa_executable_validate(Executable, &Result);
  if (auto Err = Plugin::check(Status, "Error in hsa_executable_validate: %s"))
    return Err;

  if (Result)
    return Plugin::error("Loaded HSA executable does not validate");

  if (auto Err = utils::readAMDGPUMetaDataFromImage(
          getMemoryBuffer(), KernelInfoMap, ELFABIVersion))
    return Err;

  return Plugin::success();
}

Expected<hsa_executable_symbol_t>
AMDGPUDeviceImageTy::findDeviceSymbol(GenericDeviceTy &Device,
                                      StringRef SymbolName) const {
  AMDGPUDeviceTy &AMDGPUDevice = static_cast<AMDGPUDeviceTy &>(Device);
  hsa_agent_t Agent = AMDGPUDevice.getAgent();

  hsa_executable_symbol_t Symbol;
  hsa_status_t Status = hsa_executable_get_symbol_by_name(
      Executable, SymbolName.data(), &Agent, &Symbol);
  if (auto Err = Plugin::check(
          Status, "Error in hsa_executable_get_symbol_by_name(%s): %s",
          SymbolName.data()))
    return std::move(Err);

  return Symbol;
}

bool AMDGPUDeviceImageTy::hasDeviceSymbol(GenericDeviceTy &Device,
                                          StringRef SymbolName) const {
  AMDGPUDeviceTy &AMDGPUDevice = static_cast<AMDGPUDeviceTy &>(Device);
  hsa_agent_t Agent = AMDGPUDevice.getAgent();
  hsa_executable_symbol_t Symbol;
  hsa_status_t Status = hsa_executable_get_symbol_by_name(
      Executable, SymbolName.data(), &Agent, &Symbol);
  return (Status == HSA_STATUS_SUCCESS);
}

template <typename ResourceTy>
Error AMDGPUResourceRef<ResourceTy>::create(GenericDeviceTy &Device) {
  if (Resource)
    return Plugin::error("Creating an existing resource");

  AMDGPUDeviceTy &AMDGPUDevice = static_cast<AMDGPUDeviceTy &>(Device);

  Resource = new ResourceTy(AMDGPUDevice);

  return Resource->init();
}

AMDGPUStreamTy::AMDGPUStreamTy(AMDGPUDeviceTy &Device)
    : Agent(Device.getAgent()), Queue(nullptr),
      SignalManager(Device.getSignalManager()), Device(Device),
      // Initialize the std::deque with some empty positions.
      Slots(32), NextSlot(0), SyncCycle(0), RPCServer(nullptr),
      StreamBusyWaitMicroseconds(Device.getStreamBusyWaitMicroseconds()),
      UseMultipleSdmaEngines(Device.useMultipleSdmaEngines()),
      UseSyncCopyBack(Device.syncCopyBack()) {}

/// Class implementing the AMDGPU-specific functionalities of the global
/// handler.
struct AMDGPUGlobalHandlerTy final : public GenericGlobalHandlerTy {
  /// Get the metadata of a global from the device. The name and size of the
  /// global is read from DeviceGlobal and the address of the global is written
  /// to DeviceGlobal.
  Error getGlobalMetadataFromDevice(GenericDeviceTy &Device,
                                    DeviceImageTy &Image,
                                    GlobalTy &DeviceGlobal) override {
    AMDGPUDeviceImageTy &AMDImage = static_cast<AMDGPUDeviceImageTy &>(Image);

    // Find the symbol on the device executable.
    auto SymbolOrErr =
        AMDImage.findDeviceSymbol(Device, DeviceGlobal.getName());
    if (!SymbolOrErr)
      return SymbolOrErr.takeError();

    hsa_executable_symbol_t Symbol = *SymbolOrErr;
    hsa_symbol_kind_t SymbolType;
    hsa_status_t Status;
    uint64_t SymbolAddr;
    uint32_t SymbolSize;

    // Retrieve the type, address and size of the symbol.
    std::pair<hsa_executable_symbol_info_t, void *> RequiredInfos[] = {
        {HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &SymbolType},
        {HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS, &SymbolAddr},
        {HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SIZE, &SymbolSize}};

    for (auto &Info : RequiredInfos) {
      Status = hsa_executable_symbol_get_info(Symbol, Info.first, Info.second);
      if (auto Err = Plugin::check(
              Status, "Error in hsa_executable_symbol_get_info: %s"))
        return Err;
    }

    // Check the size of the symbol.
    if (SymbolSize != DeviceGlobal.getSize())
      return Plugin::error(
          "Failed to load global '%s' due to size mismatch (%zu != %zu)",
          DeviceGlobal.getName().data(), SymbolSize,
          (size_t)DeviceGlobal.getSize());

    // Store the symbol address on the device global metadata.
    DeviceGlobal.setPtr(reinterpret_cast<void *>(SymbolAddr));

    return Plugin::success();
  }
};

/// Class implementing the AMDGPU-specific functionalities of the plugin.
struct AMDGPUPluginTy final : public GenericPluginTy {
  /// Create an AMDGPU plugin and initialize the AMDGPU driver.
  AMDGPUPluginTy()
      : GenericPluginTy(getTripleArch()), Initialized(false),
        HostDevice(nullptr) {}

  /// This class should not be copied.
  AMDGPUPluginTy(const AMDGPUPluginTy &) = delete;
  AMDGPUPluginTy(AMDGPUPluginTy &&) = delete;

  /// Initialize the plugin and return the number of devices.
  Expected<int32_t> initImpl() override {
    hsa_status_t Status = hsa_init();
    if (Status != HSA_STATUS_SUCCESS) {
      // Cannot call hsa_success_string.
      DP("Failed to initialize AMDGPU's HSA library\n");
      return 0;
    }

    // The initialization of HSA was successful. It should be safe to call
    // HSA functions from now on, e.g., hsa_shut_down.
    Initialized = true;

    // This should probably be ASO-only
    UInt32Envar KernTrace("LIBOMPTARGET_KERNEL_TRACE", 0);
    llvm::omp::target::plugin::PrintKernelTrace = KernTrace.get();

    // Register event handler to detect memory errors on the devices.
    Status = hsa_amd_register_system_event_handler(eventHandler, this);
    if (auto Err = Plugin::check(
            Status, "Error in hsa_amd_register_system_event_handler: %s"))
      return std::move(Err);

    // List of host (CPU) agents.
    llvm::SmallVector<hsa_agent_t> HostAgents;

    // Count the number of available agents.
    auto Err = utils::iterateAgents([&](hsa_agent_t Agent) {
      // Get the device type of the agent.
      hsa_device_type_t DeviceType;
      hsa_status_t Status =
          hsa_agent_get_info(Agent, HSA_AGENT_INFO_DEVICE, &DeviceType);
      if (Status != HSA_STATUS_SUCCESS)
        return Status;

      // Classify the agents into kernel (GPU) and host (CPU) kernels.
      if (DeviceType == HSA_DEVICE_TYPE_GPU) {
        // Ensure that the GPU agent supports kernel dispatch packets.
        hsa_agent_feature_t Features;
        Status = hsa_agent_get_info(Agent, HSA_AGENT_INFO_FEATURE, &Features);
        if (Features & HSA_AGENT_FEATURE_KERNEL_DISPATCH)
          KernelAgents.push_back(Agent);
      } else if (DeviceType == HSA_DEVICE_TYPE_CPU) {
        HostAgents.push_back(Agent);
      }
      return HSA_STATUS_SUCCESS;
    });

    if (Err)
      return std::move(Err);

    int32_t NumDevices = KernelAgents.size();
    if (NumDevices == 0) {
      // Do not initialize if there are no devices.
      DP("There are no devices supporting AMDGPU.\n");
      return 0;
    }

    // There are kernel agents but there is no host agent. That should be
    // treated as an error.
    if (HostAgents.empty())
      return Plugin::error("No AMDGPU host agents");

    // Initialize the host device using host agents.
    HostDevice = allocate<AMDHostDeviceTy>();
    new (HostDevice) AMDHostDeviceTy(*this, HostAgents);

    // Setup the memory pools of available for the host.
    if (auto Err = HostDevice->init())
      return std::move(Err);

    return NumDevices;
  }

  /// Deinitialize the plugin.
  Error deinitImpl() override {
    utils::hostrpc_terminate();
    // The HSA runtime was not initialized, so nothing from the plugin was
    // actually initialized.
    if (!Initialized)
      return Plugin::success();

    if (HostDevice)
      if (auto Err = HostDevice->deinit())
        return Err;

    // Finalize the HSA runtime.
    hsa_status_t Status = hsa_shut_down();
    return Plugin::check(Status, "Error in hsa_shut_down: %s");
  }

  /// Creates an AMDGPU device.
  GenericDeviceTy *createDevice(GenericPluginTy &Plugin, int32_t DeviceId,
                                int32_t NumDevices) override {
    return new AMDGPUDeviceTy(Plugin, DeviceId, NumDevices, getHostDevice(),
                              getKernelAgent(DeviceId));
  }

  /// Creates an AMDGPU global handler.
  GenericGlobalHandlerTy *createGlobalHandler() override {
    return new AMDGPUGlobalHandlerTy();
  }

  Triple::ArchType getTripleArch() const override { return Triple::amdgcn; }

  const char *getName() const override { return GETNAME(TARGET_NAME); }

  /// Get the ELF code for recognizing the compatible image binary.
  uint16_t getMagicElfBits() const override { return ELF::EM_AMDGPU; }

  bool IsSystemSupportingManagedMemory() override final {
    bool HasManagedMemorySupport = false;
    hsa_status_t Status = hsa_system_get_info(HSA_AMD_SYSTEM_INFO_SVM_SUPPORTED,
                                              &HasManagedMemorySupport);

    if (Status != HSA_STATUS_SUCCESS)
      return false;

    return HasManagedMemorySupport;
  }

  void checkInvalidImage(__tgt_device_image *TgtImage) override final {
    utils::checkImageCompatibilityWithSystemXnackMode(TgtImage,
                                                      IsXnackEnabled());
  }

  /// Check whether the image is compatible with an AMDGPU device.
  Expected<bool> isELFCompatible(uint32_t DeviceId,
                                 StringRef Image) const override {
    // Get the associated architecture and flags from the ELF.
    auto ElfOrErr =
        ELF64LEObjectFile::create(MemoryBufferRef(Image, /*Identifier=*/""),
                                  /*InitContent=*/false);
    if (!ElfOrErr)
      return ElfOrErr.takeError();
    std::optional<StringRef> Processor = ElfOrErr->tryGetCPUName();
    if (!Processor)
      return false;

    auto TargeTripleAndFeaturesOrError =
        utils::getTargetTripleAndFeatures(getKernelAgent(DeviceId));
    if (!TargeTripleAndFeaturesOrError)
      return TargeTripleAndFeaturesOrError.takeError();
    return utils::isImageCompatibleWithEnv(Processor ? *Processor : "",
                                           ElfOrErr->getPlatformFlags(),
                                           *TargeTripleAndFeaturesOrError);
  }

  bool isDataExchangable(int32_t SrcDeviceId, int32_t DstDeviceId) override {
    return true;
  }

  /// Get the host device instance.
  AMDHostDeviceTy &getHostDevice() {
    assert(HostDevice && "Host device not initialized");
    return *HostDevice;
  }

  /// Get the kernel agent with the corresponding agent id.
  hsa_agent_t getKernelAgent(int32_t AgentId) const {
    assert((uint32_t)AgentId < KernelAgents.size() && "Invalid agent id");
    return KernelAgents[AgentId];
  }

  /// Get the list of the available kernel agents.
  const llvm::SmallVector<hsa_agent_t> &getKernelAgents() const {
    return KernelAgents;
  }

private:
  /// Event handler that will be called by ROCr if an event is detected.
  static hsa_status_t eventHandler(const hsa_amd_event_t *Event,
                                   void *PluginPtr) {
    if (Event->event_type != HSA_AMD_GPU_MEMORY_FAULT_EVENT)
      return HSA_STATUS_SUCCESS;

    SmallVector<std::string> Reasons;
    uint32_t ReasonsMask = Event->memory_fault.fault_reason_mask;
    if (ReasonsMask & HSA_AMD_MEMORY_FAULT_PAGE_NOT_PRESENT)
      Reasons.emplace_back("Page not present or supervisor privilege");
    if (ReasonsMask & HSA_AMD_MEMORY_FAULT_READ_ONLY)
      Reasons.emplace_back("Write access to a read-only page");
    if (ReasonsMask & HSA_AMD_MEMORY_FAULT_NX)
      Reasons.emplace_back("Execute access to a page marked NX");
    if (ReasonsMask & HSA_AMD_MEMORY_FAULT_HOST_ONLY)
      Reasons.emplace_back("GPU attempted access to a host only page");
    if (ReasonsMask & HSA_AMD_MEMORY_FAULT_DRAMECC)
      Reasons.emplace_back("DRAM ECC failure");
    if (ReasonsMask & HSA_AMD_MEMORY_FAULT_IMPRECISE)
      Reasons.emplace_back("Can't determine the exact fault address");
    if (ReasonsMask & HSA_AMD_MEMORY_FAULT_SRAMECC)
      Reasons.emplace_back("SRAM ECC failure (ie registers, no fault address)");
    if (ReasonsMask & HSA_AMD_MEMORY_FAULT_HANG)
      Reasons.emplace_back("GPU reset following unspecified hang");

    // If we do not know the reason, say so, otherwise remove the trailing comma
    // and space.
    if (Reasons.empty())
      Reasons.emplace_back("Unknown (" + std::to_string(ReasonsMask) + ")");

    uint32_t Node = -1;
    hsa_agent_get_info(Event->memory_fault.agent, HSA_AGENT_INFO_NODE, &Node);

    AMDGPUPluginTy &Plugin = *reinterpret_cast<AMDGPUPluginTy *>(PluginPtr);
    for (uint32_t I = 0, E = Plugin.getNumDevices();
         Node != uint32_t(-1) && I < E; ++I) {
      AMDGPUDeviceTy &AMDGPUDevice =
          reinterpret_cast<AMDGPUDeviceTy &>(Plugin.getDevice(I));
      auto KernelTraceInfoRecord =
          AMDGPUDevice.KernelLaunchTraces.getExclusiveAccessor();

      uint32_t DeviceNode = -1;
      if (auto Err =
              AMDGPUDevice.getDeviceAttr(HSA_AGENT_INFO_NODE, DeviceNode)) {
        consumeError(std::move(Err));
        continue;
      }
      if (DeviceNode != Node)
        continue;

      ErrorReporter::reportKernelTraces(AMDGPUDevice, *KernelTraceInfoRecord);
    }

    // Abort the execution since we do not recover from this error.
    FATAL_MESSAGE(1,
                  "Memory access fault by GPU %" PRIu32 " (agent 0x%" PRIx64
                  ") at virtual address %p. Reasons: %s",
                  Node, Event->memory_fault.agent.handle,
                  (void *)Event->memory_fault.virtual_address,
                  llvm::join(Reasons, ", ").c_str());

    return HSA_STATUS_ERROR;
  }

  // TODO: This duplicates code that uses the target triple and features
  // to determine if XNACK is enabled. Merge into a single implementation
  // if possible (is this info available in ROCm 5.7? This might not apply
  // to trunk).
  bool IsXnackEnabled() const {
    bool hasSystemXnackEnabled = false;
    hsa_status_t HsaStatus = hsa_system_get_info(
        HSA_AMD_SYSTEM_INFO_XNACK_ENABLED, &hasSystemXnackEnabled);
    if (HsaStatus != HSA_STATUS_SUCCESS)
      return false;

    return hasSystemXnackEnabled;
  }

  /// Indicate whether the HSA runtime was correctly initialized. Even if there
  /// is no available devices this boolean will be true. It indicates whether
  /// we can safely call HSA functions (e.g., hsa_shut_down).
  bool Initialized;

  /// Arrays of the available GPU and CPU agents. These arrays of handles should
  /// not be here but in the AMDGPUDeviceTy structures directly. However, the
  /// HSA standard does not provide API functions to retirve agents directly,
  /// only iterating functions. We cache the agents here for convenience.
  llvm::SmallVector<hsa_agent_t> KernelAgents;

  /// The device representing all HSA host agents.
  AMDHostDeviceTy *HostDevice;
};

Error AMDGPUKernelTy::launchImpl(GenericDeviceTy &GenericDevice,
                                 uint32_t NumThreads, uint64_t NumBlocks,
                                 KernelArgsTy &KernelArgs,
                                 KernelLaunchParamsTy LaunchParams,
                                 AsyncInfoWrapperTy &AsyncInfoWrapper) const {
  if (ArgsSize != LaunchParams.Size &&
      ArgsSize != LaunchParams.Size + getImplicitArgsSize())
    return Plugin::error("Mismatch of kernel arguments size");

  AMDGPUPluginTy &AMDGPUPlugin =
      static_cast<AMDGPUPluginTy &>(GenericDevice.Plugin);
  AMDHostDeviceTy &HostDevice = AMDGPUPlugin.getHostDevice();
  AMDGPUMemoryManagerTy &ArgsMemoryManager = HostDevice.getArgsMemoryManager();

  void *AllArgs = nullptr;
  if (auto Err = ArgsMemoryManager.allocate(ArgsSize, &AllArgs))
    return Err;

  // Account for user requested dynamic shared memory.
  uint32_t GroupSize = getGroupSize();
  if (uint32_t MaxDynCGroupMem = std::max(
          KernelArgs.DynCGroupMem, GenericDevice.getDynamicMemorySize())) {
    GroupSize += MaxDynCGroupMem;
  }

  uint64_t StackSize;
  if (auto Err = GenericDevice.getDeviceStackSize(StackSize))
    return Err;

  utils::AMDGPUImplicitArgsTy *ImplArgs = nullptr;
  if (ArgsSize == LaunchParams.Size + getImplicitArgsSize()) {
    // Initialize implicit arguments.
    ImplArgs = reinterpret_cast<utils::AMDGPUImplicitArgsTy *>(
        advanceVoidPtr(AllArgs, LaunchParams.Size));

    // Initialize the implicit arguments to zero.
    std::memset(ImplArgs, 0, getImplicitArgsSize());
  }

  // Copy the explicit arguments.
  // TODO: We should expose the args memory manager alloc to the common part as
  // 	   alternative to copying them twice.
  if (LaunchParams.Size)
    std::memcpy(AllArgs, LaunchParams.Data, LaunchParams.Size);

  uint64_t Buffer = 0;
  AMDGPUDeviceTy &AMDGPUDevice = static_cast<AMDGPUDeviceTy &>(GenericDevice);
  AMDGPUStreamTy *Stream = nullptr;
  if (auto Err = AMDGPUDevice.getStream(AsyncInfoWrapper, Stream))
    return Err;
  if (NeedsHostServices) {
    int32_t DevID = AMDGPUDevice.getDeviceId();
    hsa_amd_memory_pool_t HostMemPool =
        HostDevice.getFineGrainedMemoryPool().get();
    hsa_amd_memory_pool_t DeviceMemPool =
        AMDGPUDevice.getCoarseGrainedMemoryPool()->get();
    hsa_queue_t *HsaQueue = Stream->getHsaQueue();
    Buffer = utils::hostrpc_assign_buffer(AMDGPUDevice.getAgent(), HsaQueue,
                                          DevID, HostMemPool, DeviceMemPool);
    GlobalTy ServiceThreadHostBufferGlobal("service_thread_buf",
                                           sizeof(uint64_t), &Buffer);
    if (auto Err = HostServiceBufferHandler.writeGlobalToDevice(
            AMDGPUDevice, ServiceThreadHostBufferGlobal,
            ServiceThreadDeviceBufferGlobal)) {
      DP("Missing symbol %s, continue execution anyway.\n",
         ServiceThreadHostBufferGlobal.getName().data());
      consumeError(std::move(Err));
    }
    DP("Hostrpc buffer allocated at %p and service thread started\n",
       (void *)Buffer);
  } else {
    DP("No hostrpc buffer or service thread required\n");
  }

  // If this kernel requires an RPC server we attach its pointer to the stream.
  if (GenericDevice.getRPCServer())
    Stream->setRPCServer(GenericDevice.getRPCServer());

  // Only COV5 implicitargs needs to be set. COV4 implicitargs are not used.
  if (ImplArgs &&
      getImplicitArgsSize() == sizeof(utils::AMDGPUImplicitArgsTy)) {
    DP("Setting fields of ImplicitArgs for COV5\n");
    ImplArgs->BlockCountX = NumBlocks;
    ImplArgs->BlockCountY = 1;
    ImplArgs->BlockCountZ = 1;
    ImplArgs->GroupSizeX = NumThreads;
    ImplArgs->GroupSizeY = 1;
    ImplArgs->GroupSizeZ = 1;
    ImplArgs->GridDims = 1;
    ImplArgs->HeapV1Ptr =
        (uint64_t)AMDGPUDevice.getPreAllocatedDeviceMemoryPool();
    ImplArgs->DynamicLdsSize = KernelArgs.DynCGroupMem;
  }

  // Get required OMPT-related data
  auto LocalOmptEventInfo = getOrNullOmptEventInfo(AsyncInfoWrapper);

  // Push the kernel launch into the stream.
  return Stream->pushKernelLaunch(*this, AllArgs, NumThreads, NumBlocks,
                                  GroupSize, static_cast<uint32_t>(StackSize),
                                  ArgsMemoryManager,
                                  std::move(LocalOmptEventInfo));
}

void AMDGPUKernelTy::printAMDOneLineKernelTrace(GenericDeviceTy &GenericDevice,
                                                KernelArgsTy &KernelArgs,
                                                uint32_t NumThreads,
                                                uint64_t NumBlocks) const {
  auto GroupSegmentSize = (*KernelInfo).GroupSegmentList;
  auto SGPRCount = (*KernelInfo).SGPRCount;
  auto VGPRCount = (*KernelInfo).VGPRCount;
  auto SGPRSpillCount = (*KernelInfo).SGPRSpillCount;
  auto VGPRSpillCount = (*KernelInfo).VGPRSpillCount;
  // auto MaxFlatWorkgroupSize = (*KernelInfo).MaxFlatWorkgroupSize;

  // This line should print exactly as the one in the old plugin.
  fprintf(stderr,
          "DEVID: %2d SGN:%d ConstWGSize:%-4d args:%2d teamsXthrds:(%4luX%4d) "
          "reqd:(%4dX%4d) lds_usage:%uB sgpr_count:%u vgpr_count:%u "
          "sgpr_spill_count:%u vgpr_spill_count:%u tripcount:%lu rpc:%d n:%s\n",
          GenericDevice.getDeviceId(), getExecutionModeFlags(), ConstWGSize,
          KernelArgs.NumArgs, NumBlocks, NumThreads, 0, 0, GroupSegmentSize,
          SGPRCount, VGPRCount, SGPRSpillCount, VGPRSpillCount,
          KernelArgs.Tripcount, NeedsHostServices, getName());
}

Error AMDGPUKernelTy::printLaunchInfoDetails(GenericDeviceTy &GenericDevice,
                                             KernelArgsTy &KernelArgs,
                                             uint32_t NumThreads,
                                             uint64_t NumBlocks) const {
  // When LIBOMPTARGET_KERNEL_TRACE is set, print the single-line kernel trace
  // info present in the old ASO plugin, and continue with the upstream 2-line
  // info, should LIBOMPTARGET_INFO be a meaningful value, otherwise return.
  if (getInfoLevel() & OMP_INFOTYPE_AMD_KERNEL_TRACE)
    printAMDOneLineKernelTrace(GenericDevice, KernelArgs, NumThreads,
                               NumBlocks);

  // Only do all this when the output is requested
  if (!(getInfoLevel() & OMP_INFOTYPE_PLUGIN_KERNEL))
    return Plugin::success();

  // We don't have data to print additional info, but no hard error
  if (!KernelInfo.has_value())
    return Plugin::success();

  // General Info
  auto NumGroups = NumBlocks;
  auto ThreadsPerGroup = NumThreads;

  // Kernel Arguments Info
  auto ArgNum = KernelArgs.NumArgs;
  auto LoopTripCount = KernelArgs.Tripcount;

  // Details for AMDGPU kernels (read from image)
  // https://www.llvm.org/docs/AMDGPUUsage.html#code-object-v4-metadata
  auto GroupSegmentSize = (*KernelInfo).GroupSegmentList;
  auto SGPRCount = (*KernelInfo).SGPRCount;
  auto VGPRCount = (*KernelInfo).VGPRCount;
  auto SGPRSpillCount = (*KernelInfo).SGPRSpillCount;
  auto VGPRSpillCount = (*KernelInfo).VGPRSpillCount;
  auto MaxFlatWorkgroupSize = (*KernelInfo).MaxFlatWorkgroupSize;

  // Prints additional launch info that contains the following.
  // Num Args: The number of kernel arguments
  // Teams x Thrds: The number of teams and the number of threads actually
  // running.
  // MaxFlatWorkgroupSize: Maximum flat work-group size supported by the
  // kernel in work-items
  // LDS Usage: Amount of bytes used in LDS storage
  // S/VGPR Count: the number of S/V GPRs occupied by the kernel
  // S/VGPR Spill Count: how many S/VGPRs are spilled by the kernel
  // Tripcount: loop tripcount for the kernel
  INFO(OMP_INFOTYPE_PLUGIN_KERNEL, GenericDevice.getDeviceId(),
       "#Args: %d Teams x Thrds: %4lux%4u (MaxFlatWorkGroupSize: %u) LDS "
       "Usage: %uB #SGPRs/VGPRs: %u/%u #SGPR/VGPR Spills: %u/%u Tripcount: "
       "%lu\n",
       ArgNum, NumGroups, ThreadsPerGroup, MaxFlatWorkgroupSize,
       GroupSegmentSize, SGPRCount, VGPRCount, SGPRSpillCount, VGPRSpillCount,
       LoopTripCount);

  return Plugin::success();
}

template <typename... ArgsTy>
static Error Plugin::check(int32_t Code, const char *ErrFmt, ArgsTy... Args) {
  hsa_status_t ResultCode = static_cast<hsa_status_t>(Code);
  if (ResultCode == HSA_STATUS_SUCCESS || ResultCode == HSA_STATUS_INFO_BREAK)
    return Error::success();

  const char *Desc = "Unknown error";
  hsa_status_t Ret = hsa_status_string(ResultCode, &Desc);
  if (Ret != HSA_STATUS_SUCCESS)
    REPORT("Unrecognized " GETNAME(TARGET_NAME) " error code %d\n", Code);

  return createStringError<ArgsTy..., const char *>(inconvertibleErrorCode(),
                                                    ErrFmt, Args..., Desc);
}

void *AMDGPUMemoryManagerTy::allocate(size_t Size, void *HstPtr,
                                      TargetAllocTy Kind) {
  // Allocate memory from the pool.
  void *Ptr = nullptr;
  if (auto Err = MemoryPool->allocate(Size, &Ptr)) {
    consumeError(std::move(Err));
    return nullptr;
  }
  assert(Ptr && "Invalid pointer");

  // Get a list of agents that can access this memory pool.
  llvm::SmallVector<hsa_agent_t> Agents;
  llvm::copy_if(
      Plugin.getKernelAgents(), std::back_inserter(Agents),
      [&](hsa_agent_t Agent) { return MemoryPool->canAccess(Agent); });

  // Allow all valid kernel agents to access the allocation.
  if (auto Err = MemoryPool->enableAccess(Ptr, Size, Agents)) {
    REPORT("%s\n", toString(std::move(Err)).data());
    return nullptr;
  }
  return Ptr;
}

void *AMDGPUDeviceTy::allocate(size_t Size, void *, TargetAllocTy Kind) {
  if (Size == 0)
    return nullptr;

  // Find the correct memory pool.
  AMDGPUMemoryPoolTy *MemoryPool = nullptr;
  switch (Kind) {
  case TARGET_ALLOC_DEFAULT:
  case TARGET_ALLOC_DEVICE:
  case TARGET_ALLOC_DEVICE_NON_BLOCKING:
    MemoryPool = CoarseGrainedMemoryPools[0];
    break;
  case TARGET_ALLOC_HOST:
    MemoryPool = &HostDevice.getFineGrainedMemoryPool();
    break;
  case TARGET_ALLOC_SHARED:
    MemoryPool = &HostDevice.getFineGrainedMemoryPool();
    break;
  }

  if (!MemoryPool) {
    REPORT("No memory pool for the specified allocation kind\n");
    return nullptr;
  }

  // Allocate from the corresponding memory pool.
  void *Alloc = nullptr;
  if (Error Err = MemoryPool->allocate(Size, &Alloc)) {
    REPORT("%s\n", toString(std::move(Err)).data());
    return nullptr;
  }
  // FIXME: Maybe this should be guarded by hasgfx90a
  if (MemoryPool == CoarseGrainedMemoryPools[0]) {
    // printf(" Device::allocate calling setCoarseGrainMemoryImpl(Alloc, Size,
    // false)\n");
    if (auto Err = setCoarseGrainMemoryImpl(Alloc, Size, /*set_attr=*/false)) {
      REPORT("%s\n", toString(std::move(Err)).data());
      return nullptr;
    }
  }

  if (Alloc) {
    // Get a list of agents that can access this memory pool. Inherently
    // necessary for host or shared allocations Also enabled for device memory
    // to allow device to device memcpy
    llvm::SmallVector<hsa_agent_t> Agents;
    llvm::copy_if(static_cast<AMDGPUPluginTy &>(Plugin).getKernelAgents(),
                  std::back_inserter(Agents), [&](hsa_agent_t Agent) {
                    return MemoryPool->canAccess(Agent);
                  });

    // Enable all valid kernel agents to access the buffer.
    if (auto Err = MemoryPool->enableAccess(Alloc, Size, Agents)) {
      REPORT("%s\n", toString(std::move(Err)).data());
      return nullptr;
    }
  }

  return Alloc;
}

#ifdef OMPT_SUPPORT
/// Casts and validated the OMPT-related info passed to the action function.
static OmptKernelTimingArgsAsyncTy *getOmptTimingsArgs(void *Data) {
  OmptKernelTimingArgsAsyncTy *Args =
      reinterpret_cast<OmptKernelTimingArgsAsyncTy *>(Data);
  assert(Args && "Invalid argument pointer");
  assert(Args->Signal && "Invalid signal");
  assert(Args->OmptEventInfo && "Invalid OMPT Async data (nullptr)");
  assert(Args->OmptEventInfo->TraceRecord && "Invalid Trace Record Pointer");
  assert(Args->OmptEventInfo->RegionInterface &&
         "Invalid RegionInterface pointer");
  assert((!std::holds_alternative<std::monostate>(
             Args->OmptEventInfo->RIFunction)) &&
         "Unset OMPT Interface Function Pointer Set");
  return Args;
}

static std::pair<uint64_t, uint64_t>
getKernelStartAndEndTime(const OmptKernelTimingArgsAsyncTy *Args) {
  assert(Args->Signal && "Invalid AMDGPUSignal Pointer in OMPT profiling");
  hsa_amd_profiling_dispatch_time_t TimeRec;
  hsa_status_t Status = hsa_amd_profiling_get_dispatch_time(
      Args->Agent, Args->Signal->get(), &TimeRec);

  uint64_t StartTime = TimeRec.start * Args->TicksToTime;
  uint64_t EndTime = TimeRec.end * Args->TicksToTime;

  return {StartTime, EndTime};
}

static std::pair<uint64_t, uint64_t>
getCopyStartAndEndTime(const OmptKernelTimingArgsAsyncTy *Args) {
  assert(Args->Signal && "Invalid AMDGPUSignal Pointer in OMPT profiling");
  hsa_amd_profiling_async_copy_time_t TimeRec;
  hsa_status_t Status =
      hsa_amd_profiling_get_async_copy_time(Args->Signal->get(), &TimeRec);
  uint64_t StartTime = TimeRec.start * Args->TicksToTime;
  uint64_t EndTime = TimeRec.end * Args->TicksToTime;

  return {StartTime, EndTime};
}
#endif

void AMDGPUQueueTy::callbackError(hsa_status_t Status, hsa_queue_t *Source,
                                  void *Data) {
  auto &AMDGPUDevice = *reinterpret_cast<AMDGPUDeviceTy *>(Data);

  if (Status == HSA_STATUS_ERROR_EXCEPTION) {
    auto KernelTraceInfoRecord =
        AMDGPUDevice.KernelLaunchTraces.getExclusiveAccessor();
    std::function<bool(__tgt_async_info &)> AsyncInfoWrapperMatcher =
        [=](__tgt_async_info &AsyncInfo) {
          auto *Stream = reinterpret_cast<AMDGPUStreamTy *>(AsyncInfo.Queue);
          if (!Stream || !Stream->getQueue())
            return false;
          return Stream->getQueue()->Queue == Source;
        };
    ErrorReporter::reportTrapInKernel(AMDGPUDevice, *KernelTraceInfoRecord,
                                      AsyncInfoWrapperMatcher);
  }

  auto Err = Plugin::check(Status, "Received error in queue %p: %s", Source);
  FATAL_MESSAGE(1, "%s", toString(std::move(Err)).data());
}

} // namespace plugin
} // namespace target
} // namespace omp
} // namespace llvm

#ifdef OMPT_SUPPORT
namespace llvm::omp::target::plugin {

/// Enable/disable kernel profiling for the given device.
void setOmptQueueProfile(void *Device, int Enable) {
  reinterpret_cast<llvm::omp::target::plugin::AMDGPUDeviceTy *>(Device)
      ->setOmptQueueProfile(Enable);
}

} // namespace llvm::omp::target::plugin

/// Enable/disable kernel profiling for the given device.
void setGlobalOmptKernelProfile(void *Device, int Enable) {
  llvm::omp::target::plugin::setOmptQueueProfile(Device, Enable);
}

#endif

extern "C" {
llvm::omp::target::plugin::GenericPluginTy *createPlugin_amdgpu() {
  return new llvm::omp::target::plugin::AMDGPUPluginTy();
}
}
