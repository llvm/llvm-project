//===--- amdgpu/src/rtl.cpp --------------------------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RTL for AMD hsa machine
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <libelf.h>
#include <list>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

#include "impl_runtime.h"
#include "interop_hsa.h"

#include "internal.h"
#include "rt.h"

#include "DeviceEnvironment.h"
#include "get_elf_mach_gfx_name.h"
#include "omptargetplugin.h"
#include "print_tracing.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Frontend/OpenMP/OMPGridValues.h"

using namespace llvm;

// hostrpc interface, FIXME: consider moving to its own include these are
// statically linked into amdgpu/plugin if present from hostrpc_services.a,
// linked as --whole-archive to override the weak symbols that are used to
// implement a fallback for toolchains that do not yet have a hostrpc library.
extern "C" {
uint64_t hostrpc_assign_buffer(hsa_agent_t Agent, hsa_queue_t *ThisQ,
                               uint32_t DeviceId);
hsa_status_t hostrpc_init();
hsa_status_t hostrpc_terminate();

__attribute__((weak)) hsa_status_t hostrpc_init() { return HSA_STATUS_SUCCESS; }
__attribute__((weak)) hsa_status_t hostrpc_terminate() {
  return HSA_STATUS_SUCCESS;
}
__attribute__((weak)) uint64_t hostrpc_assign_buffer(hsa_agent_t, hsa_queue_t *,
                                                     uint32_t DeviceId) {
  DP("Warning: Attempting to assign hostrpc to device %u, but hostrpc library "
     "missing\n",
     DeviceId);
  return 0;
}
}

// Heuristic parameters used for kernel launch
// Number of teams per CU to allow scheduling flexibility
static const unsigned DefaultTeamsPerCU = 4;

int print_kernel_trace;

#ifdef OMPTARGET_DEBUG
#define check(msg, status)                                                     \
  if (status != HSA_STATUS_SUCCESS) {                                          \
    DP(#msg " failed\n");                                                      \
  } else {                                                                     \
    DP(#msg " succeeded\n");                                                   \
  }
#else
#define check(msg, status)                                                     \
  {}
#endif

#include "elf_common.h"

namespace hsa {
template <typename C> hsa_status_t iterate_agents(C Cb) {
  auto L = [](hsa_agent_t Agent, void *Data) -> hsa_status_t {
    C *Unwrapped = static_cast<C *>(Data);
    return (*Unwrapped)(Agent);
  };
  return hsa_iterate_agents(L, static_cast<void *>(&Cb));
}

template <typename C>
hsa_status_t amd_agent_iterate_memory_pools(hsa_agent_t Agent, C Cb) {
  auto L = [](hsa_amd_memory_pool_t MemoryPool, void *Data) -> hsa_status_t {
    C *Unwrapped = static_cast<C *>(Data);
    return (*Unwrapped)(MemoryPool);
  };

  return hsa_amd_agent_iterate_memory_pools(Agent, L, static_cast<void *>(&Cb));
}

} // namespace hsa

/// Keep entries table per device
struct FuncOrGblEntryTy {
  __tgt_target_table Table;
  std::vector<__tgt_offload_entry> Entries;
};

struct KernelArgPool {
private:
  static pthread_mutex_t Mutex;

public:
  uint32_t KernargSegmentSize;
  void *KernargRegion = nullptr;
  std::queue<int> FreeKernargSegments;

  uint32_t kernargSizeIncludingImplicit() {
    return KernargSegmentSize + sizeof(impl_implicit_args_t);
  }

  ~KernelArgPool() {
    if (KernargRegion) {
      auto R = hsa_amd_memory_pool_free(KernargRegion);
      if (R != HSA_STATUS_SUCCESS) {
        DP("hsa_amd_memory_pool_free failed: %s\n", get_error_string(R));
      }
    }
  }

  // Can't really copy or move a mutex
  KernelArgPool() = default;
  KernelArgPool(const KernelArgPool &) = delete;
  KernelArgPool(KernelArgPool &&) = delete;

  KernelArgPool(uint32_t KernargSegmentSize, hsa_amd_memory_pool_t &MemoryPool)
      : KernargSegmentSize(KernargSegmentSize) {

    // impl uses one pool per kernel for all gpus, with a fixed upper size
    // preserving that exact scheme here, including the queue<int>

    hsa_status_t Err = hsa_amd_memory_pool_allocate(
        MemoryPool, kernargSizeIncludingImplicit() * MAX_NUM_KERNELS, 0,
        &KernargRegion);

    if (Err != HSA_STATUS_SUCCESS) {
      DP("hsa_amd_memory_pool_allocate failed: %s\n", get_error_string(Err));
      KernargRegion = nullptr; // paranoid
      return;
    }

    Err = core::allow_access_to_all_gpu_agents(KernargRegion);
    if (Err != HSA_STATUS_SUCCESS) {
      DP("hsa allow_access_to_all_gpu_agents failed: %s\n",
         get_error_string(Err));
      auto R = hsa_amd_memory_pool_free(KernargRegion);
      if (R != HSA_STATUS_SUCCESS) {
        // if free failed, can't do anything more to resolve it
        DP("hsa memory poll free failed: %s\n", get_error_string(Err));
      }
      KernargRegion = nullptr;
      return;
    }

    for (int I = 0; I < MAX_NUM_KERNELS; I++) {
      FreeKernargSegments.push(I);
    }
  }

  void *allocate(uint64_t ArgNum) {
    assert((ArgNum * sizeof(void *)) == KernargSegmentSize);
    Lock L(&Mutex);
    void *Res = nullptr;
    if (!FreeKernargSegments.empty()) {

      int FreeIdx = FreeKernargSegments.front();
      Res = static_cast<void *>(static_cast<char *>(KernargRegion) +
                                (FreeIdx * kernargSizeIncludingImplicit()));
      assert(FreeIdx == pointerToIndex(Res));
      FreeKernargSegments.pop();
    }
    return Res;
  }

  void deallocate(void *Ptr) {
    Lock L(&Mutex);
    int Idx = pointerToIndex(Ptr);
    FreeKernargSegments.push(Idx);
  }

private:
  int pointerToIndex(void *Ptr) {
    ptrdiff_t Bytes =
        static_cast<char *>(Ptr) - static_cast<char *>(KernargRegion);
    assert(Bytes >= 0);
    assert(Bytes % kernargSizeIncludingImplicit() == 0);
    return Bytes / kernargSizeIncludingImplicit();
  }
  struct Lock {
    Lock(pthread_mutex_t *M) : M(M) { pthread_mutex_lock(M); }
    ~Lock() { pthread_mutex_unlock(M); }
    pthread_mutex_t *M;
  };
};
pthread_mutex_t KernelArgPool::Mutex = PTHREAD_MUTEX_INITIALIZER;

/// Use a single entity to encode a kernel and a set of flags
struct KernelTy {
  llvm::omp::OMPTgtExecModeFlags ExecutionMode;
  int16_t ConstWGSize;
  int32_t DeviceId;
  void *CallStackAddr = nullptr;
  const char *Name;

  KernelTy(llvm::omp::OMPTgtExecModeFlags ExecutionMode, int16_t ConstWgSize,
           int32_t DeviceId, void *CallStackAddr, const char *Name,
           uint32_t KernargSegmentSize,
           hsa_amd_memory_pool_t &KernArgMemoryPool,
           std::unordered_map<std::string, std::unique_ptr<KernelArgPool>>
               &KernelArgPoolMap)
      : ExecutionMode(ExecutionMode), ConstWGSize(ConstWgSize),
        DeviceId(DeviceId), CallStackAddr(CallStackAddr), Name(Name) {
    DP("Construct kernelinfo: ExecMode %d\n", ExecutionMode);

    std::string N(Name);
    if (KernelArgPoolMap.find(N) == KernelArgPoolMap.end()) {
      KernelArgPoolMap.insert(
          std::make_pair(N, std::unique_ptr<KernelArgPool>(new KernelArgPool(
                                KernargSegmentSize, KernArgMemoryPool))));
    }
  }
};

template <typename Callback> static hsa_status_t findAgents(Callback CB) {

  hsa_status_t Err =
      hsa::iterate_agents([&](hsa_agent_t Agent) -> hsa_status_t {
        hsa_device_type_t DeviceType;
        // get_info fails iff HSA runtime not yet initialized
        hsa_status_t Err =
            hsa_agent_get_info(Agent, HSA_AGENT_INFO_DEVICE, &DeviceType);

        if (Err != HSA_STATUS_SUCCESS) {
          if (print_kernel_trace > 0)
            DP("rtl.cpp: err %s\n", get_error_string(Err));

          return Err;
        }

        CB(DeviceType, Agent);
        return HSA_STATUS_SUCCESS;
      });

  // iterate_agents fails iff HSA runtime not yet initialized
  if (print_kernel_trace > 0 && Err != HSA_STATUS_SUCCESS) {
    DP("rtl.cpp: err %s\n", get_error_string(Err));
  }

  return Err;
}

static void callbackQueue(hsa_status_t Status, hsa_queue_t *Source,
                          void *Data) {
  if (Status != HSA_STATUS_SUCCESS) {
    const char *StatusString;
    if (hsa_status_string(Status, &StatusString) != HSA_STATUS_SUCCESS) {
      StatusString = "unavailable";
    }
    DP("[%s:%d] GPU error in queue %p %d (%s)\n", __FILE__, __LINE__, Source,
       Status, StatusString);
    abort();
  }
}

namespace core {
namespace {

bool checkResult(hsa_status_t Err, const char *ErrMsg) {
  if (Err == HSA_STATUS_SUCCESS)
    return true;

  REPORT("%s", ErrMsg);
  REPORT("%s", get_error_string(Err));
  return false;
}

void packetStoreRelease(uint32_t *Packet, uint16_t Header, uint16_t Rest) {
  __atomic_store_n(Packet, Header | (Rest << 16), __ATOMIC_RELEASE);
}

uint16_t createHeader() {
  uint16_t Header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  Header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
  Header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;
  return Header;
}

hsa_status_t isValidMemoryPool(hsa_amd_memory_pool_t MemoryPool) {
  bool AllocAllowed = false;
  hsa_status_t Err = hsa_amd_memory_pool_get_info(
      MemoryPool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,
      &AllocAllowed);
  if (Err != HSA_STATUS_SUCCESS) {
    DP("Alloc allowed in memory pool check failed: %s\n",
       get_error_string(Err));
    return Err;
  }

  size_t Size = 0;
  Err = hsa_amd_memory_pool_get_info(MemoryPool, HSA_AMD_MEMORY_POOL_INFO_SIZE,
                                     &Size);
  if (Err != HSA_STATUS_SUCCESS) {
    DP("Get memory pool size failed: %s\n", get_error_string(Err));
    return Err;
  }

  return (AllocAllowed && Size > 0) ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR;
}

hsa_status_t addMemoryPool(hsa_amd_memory_pool_t MemoryPool, void *Data) {
  std::vector<hsa_amd_memory_pool_t> *Result =
      static_cast<std::vector<hsa_amd_memory_pool_t> *>(Data);

  hsa_status_t Err;
  if ((Err = isValidMemoryPool(MemoryPool)) != HSA_STATUS_SUCCESS) {
    return Err;
  }

  Result->push_back(MemoryPool);
  return HSA_STATUS_SUCCESS;
}

} // namespace
} // namespace core

struct EnvironmentVariables {
  int NumTeams;
  int TeamLimit;
  int TeamThreadLimit;
  int MaxTeamsDefault;
  int DynamicMemSize;
};

template <uint32_t wavesize>
static constexpr const llvm::omp::GV &getGridValue() {
  return llvm::omp::getAMDGPUGridValues<wavesize>();
}

struct HSALifetime {
  // Wrapper around HSA used to ensure it is constructed before other types
  // and destructed after, which means said other types can use raii for
  // cleanup without risking running outside of the lifetime of HSA
  const hsa_status_t S;

  bool HSAInitSuccess() { return S == HSA_STATUS_SUCCESS; }
  HSALifetime() : S(hsa_init()) {}

  ~HSALifetime() {
    if (S == HSA_STATUS_SUCCESS) {
      hsa_status_t Err = hsa_shut_down();
      if (Err != HSA_STATUS_SUCCESS) {
        // Can't call into HSA to get a string from the integer
        DP("Shutting down HSA failed: %d\n", Err);
      }
    }
  }
};

// Handle scheduling of multiple hsa_queue's per device to
// multiple threads (one scheduler per device)
class HSAQueueScheduler {
public:
  HSAQueueScheduler() : Current(0) {}

  HSAQueueScheduler(const HSAQueueScheduler &) = delete;

  HSAQueueScheduler(HSAQueueScheduler &&Q) {
    Current = Q.Current.load();
    for (uint8_t I = 0; I < NUM_QUEUES_PER_DEVICE; I++) {
      HSAQueues[I] = Q.HSAQueues[I];
      Q.HSAQueues[I] = nullptr;
    }
  }

  // \return false if any HSA queue creation fails
  bool createQueues(hsa_agent_t HSAAgent, uint32_t QueueSize) {
    for (uint8_t I = 0; I < NUM_QUEUES_PER_DEVICE; I++) {
      hsa_queue_t *Q = nullptr;
      hsa_status_t Rc =
          hsa_queue_create(HSAAgent, QueueSize, HSA_QUEUE_TYPE_MULTI,
                           callbackQueue, NULL, UINT32_MAX, UINT32_MAX, &Q);
      if (Rc != HSA_STATUS_SUCCESS) {
        DP("Failed to create HSA queue %d\n", I);
        return false;
      }
      HSAQueues[I] = Q;
    }
    return true;
  }

  ~HSAQueueScheduler() {
    for (uint8_t I = 0; I < NUM_QUEUES_PER_DEVICE; I++) {
      if (HSAQueues[I]) {
        hsa_status_t Err = hsa_queue_destroy(HSAQueues[I]);
        if (Err != HSA_STATUS_SUCCESS)
          DP("Error destroying HSA queue");
      }
    }
  }

  // \return next queue to use for device
  hsa_queue_t *next() {
    return HSAQueues[(Current.fetch_add(1, std::memory_order_relaxed)) %
                     NUM_QUEUES_PER_DEVICE];
  }

private:
  // Number of queues per device
  enum : uint8_t { NUM_QUEUES_PER_DEVICE = 4 };
  hsa_queue_t *HSAQueues[NUM_QUEUES_PER_DEVICE] = {};
  std::atomic<uint8_t> Current;
};

/// Class containing all the device information
class RTLDeviceInfoTy : HSALifetime {
  std::vector<std::list<FuncOrGblEntryTy>> FuncGblEntries;

  struct QueueDeleter {
    void operator()(hsa_queue_t *Q) {
      if (Q) {
        hsa_status_t Err = hsa_queue_destroy(Q);
        if (Err != HSA_STATUS_SUCCESS) {
          DP("Error destroying hsa queue: %s\n", get_error_string(Err));
        }
      }
    }
  };

public:
  bool ConstructionSucceeded = false;

  // load binary populates symbol tables and mutates various global state
  // run uses those symbol tables
  std::shared_timed_mutex LoadRunLock;

  int NumberOfDevices = 0;

  /// List that contains all the kernels.
  /// FIXME: we may need this to be per device and per library.
  std::list<KernelTy> KernelsList;
  std::unordered_map<std::string /*kernel*/, std::unique_ptr<KernelArgPool>>
      KernelArgPoolMap;

  // GPU devices
  std::vector<hsa_agent_t> HSAAgents;
  std::vector<HSAQueueScheduler> HSAQueueSchedulers; // one per gpu

  // CPUs
  std::vector<hsa_agent_t> CPUAgents;

  // Device properties
  std::vector<int> ComputeUnits;
  std::vector<int> GroupsPerDevice;
  std::vector<int> ThreadsPerGroup;
  std::vector<int> WarpSize;
  std::vector<std::string> GPUName;
  std::vector<std::string> TargetID;

  // OpenMP properties
  std::vector<int> NumTeams;
  std::vector<int> NumThreads;

  // OpenMP Environment properties
  EnvironmentVariables Env;

  // OpenMP Requires Flags
  int64_t RequiresFlags;

  // Resource pools
  SignalPoolT FreeSignalPool;

  bool HostcallRequired = false;

  std::vector<hsa_executable_t> HSAExecutables;

  std::vector<std::map<std::string, atl_kernel_info_t>> KernelInfoTable;
  std::vector<std::map<std::string, atl_symbol_info_t>> SymbolInfoTable;

  hsa_amd_memory_pool_t KernArgPool;

  // fine grained memory pool for host allocations
  hsa_amd_memory_pool_t HostFineGrainedMemoryPool;

  // fine and coarse-grained memory pools per offloading device
  std::vector<hsa_amd_memory_pool_t> DeviceFineGrainedMemoryPools;
  std::vector<hsa_amd_memory_pool_t> DeviceCoarseGrainedMemoryPools;

  struct ImplFreePtrDeletor {
    void operator()(void *P) {
      core::Runtime::Memfree(P); // ignore failure to free
    }
  };

  // device_State shared across loaded binaries, error if inconsistent size
  std::vector<std::pair<std::unique_ptr<void, ImplFreePtrDeletor>, uint64_t>>
      DeviceStateStore;

  static const unsigned HardTeamLimit =
      (1 << 16) - 1; // 64K needed to fit in uint16
  static const int DefaultNumTeams = 128;

  // These need to be per-device since different devices can have different
  // wave sizes, but are currently the same number for each so that refactor
  // can be postponed.
  static_assert(getGridValue<32>().GV_Max_Teams ==
                    getGridValue<64>().GV_Max_Teams,
                "");
  static const int MaxTeams = getGridValue<64>().GV_Max_Teams;

  static_assert(getGridValue<32>().GV_Max_WG_Size ==
                    getGridValue<64>().GV_Max_WG_Size,
                "");
  static const int MaxWgSize = getGridValue<64>().GV_Max_WG_Size;

  static_assert(getGridValue<32>().GV_Default_WG_Size ==
                    getGridValue<64>().GV_Default_WG_Size,
                "");
  static const int DefaultWgSize = getGridValue<64>().GV_Default_WG_Size;

  using MemcpyFunc = hsa_status_t (*)(hsa_signal_t, void *, void *, size_t Size,
                                      hsa_agent_t, hsa_amd_memory_pool_t);
  hsa_status_t freesignalpoolMemcpy(void *Dest, void *Src, size_t Size,
                                    MemcpyFunc Func, int32_t DeviceId) {
    hsa_agent_t Agent = HSAAgents[DeviceId];
    hsa_signal_t S = FreeSignalPool.pop();
    if (S.handle == 0) {
      return HSA_STATUS_ERROR;
    }
    hsa_status_t R = Func(S, Dest, Src, Size, Agent, HostFineGrainedMemoryPool);
    FreeSignalPool.push(S);
    return R;
  }

  hsa_status_t freesignalpoolMemcpyD2H(void *Dest, void *Src, size_t Size,
                                       int32_t DeviceId) {
    return freesignalpoolMemcpy(Dest, Src, Size, impl_memcpy_d2h, DeviceId);
  }

  hsa_status_t freesignalpoolMemcpyH2D(void *Dest, void *Src, size_t Size,
                                       int32_t DeviceId) {
    return freesignalpoolMemcpy(Dest, Src, Size, impl_memcpy_h2d, DeviceId);
  }

  static void printDeviceInfo(int32_t DeviceId, hsa_agent_t Agent) {
    char TmpChar[1000];
    uint16_t Major, Minor;
    uint32_t TmpUInt;
    uint32_t TmpUInt2;
    uint32_t CacheSize[4];
    bool TmpBool;
    uint16_t WorkgroupMaxDim[3];
    hsa_dim3_t GridMaxDim;

    // Getting basic information about HSA and Device
    core::checkResult(
        hsa_system_get_info(HSA_SYSTEM_INFO_VERSION_MAJOR, &Major),
        "Error from hsa_system_get_info when obtaining "
        "HSA_SYSTEM_INFO_VERSION_MAJOR\n");
    core::checkResult(
        hsa_system_get_info(HSA_SYSTEM_INFO_VERSION_MINOR, &Minor),
        "Error from hsa_system_get_info when obtaining "
        "HSA_SYSTEM_INFO_VERSION_MINOR\n");
    printf("    HSA Runtime Version: \t\t%u.%u \n", Major, Minor);
    printf("    HSA OpenMP Device Number: \t\t%d \n", DeviceId);
    core::checkResult(
        hsa_agent_get_info(
            Agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_PRODUCT_NAME, TmpChar),
        "Error returned from hsa_agent_get_info when obtaining "
        "HSA_AMD_AGENT_INFO_PRODUCT_NAME\n");
    printf("    Product Name: \t\t\t%s \n", TmpChar);
    core::checkResult(hsa_agent_get_info(Agent, HSA_AGENT_INFO_NAME, TmpChar),
                      "Error returned from hsa_agent_get_info when obtaining "
                      "HSA_AGENT_INFO_NAME\n");
    printf("    Device Name: \t\t\t%s \n", TmpChar);
    core::checkResult(
        hsa_agent_get_info(Agent, HSA_AGENT_INFO_VENDOR_NAME, TmpChar),
        "Error returned from hsa_agent_get_info when obtaining "
        "HSA_AGENT_INFO_NAME\n");
    printf("    Vendor Name: \t\t\t%s \n", TmpChar);
    hsa_device_type_t DevType;
    core::checkResult(
        hsa_agent_get_info(Agent, HSA_AGENT_INFO_DEVICE, &DevType),
        "Error returned from hsa_agent_get_info when obtaining "
        "HSA_AGENT_INFO_DEVICE\n");
    printf("    Device Type: \t\t\t%s \n",
           DevType == HSA_DEVICE_TYPE_CPU
               ? "CPU"
               : (DevType == HSA_DEVICE_TYPE_GPU
                      ? "GPU"
                      : (DevType == HSA_DEVICE_TYPE_DSP ? "DSP" : "UNKNOWN")));
    core::checkResult(
        hsa_agent_get_info(Agent, HSA_AGENT_INFO_QUEUES_MAX, &TmpUInt),
        "Error returned from hsa_agent_get_info when obtaining "
        "HSA_AGENT_INFO_QUEUES_MAX\n");
    printf("    Max Queues: \t\t\t%u \n", TmpUInt);
    core::checkResult(
        hsa_agent_get_info(Agent, HSA_AGENT_INFO_QUEUE_MIN_SIZE, &TmpUInt),
        "Error returned from hsa_agent_get_info when obtaining "
        "HSA_AGENT_INFO_QUEUE_MIN_SIZE\n");
    printf("    Queue Min Size: \t\t\t%u \n", TmpUInt);
    core::checkResult(
        hsa_agent_get_info(Agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &TmpUInt),
        "Error returned from hsa_agent_get_info when obtaining "
        "HSA_AGENT_INFO_QUEUE_MAX_SIZE\n");
    printf("    Queue Max Size: \t\t\t%u \n", TmpUInt);

    // Getting cache information
    printf("    Cache:\n");

    // FIXME: This is deprecated according to HSA documentation. But using
    // hsa_agent_iterate_caches and hsa_cache_get_info breaks execution during
    // runtime.
    core::checkResult(
        hsa_agent_get_info(Agent, HSA_AGENT_INFO_CACHE_SIZE, CacheSize),
        "Error returned from hsa_agent_get_info when obtaining "
        "HSA_AGENT_INFO_CACHE_SIZE\n");

    for (int I = 0; I < 4; I++) {
      if (CacheSize[I]) {
        printf("      L%u: \t\t\t\t%u bytes\n", I, CacheSize[I]);
      }
    }

    core::checkResult(
        hsa_agent_get_info(Agent,
                           (hsa_agent_info_t)HSA_AMD_AGENT_INFO_CACHELINE_SIZE,
                           &TmpUInt),
        "Error returned from hsa_agent_get_info when obtaining "
        "HSA_AMD_AGENT_INFO_CACHELINE_SIZE\n");
    printf("    Cacheline Size: \t\t\t%u \n", TmpUInt);
    core::checkResult(
        hsa_agent_get_info(
            Agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_MAX_CLOCK_FREQUENCY,
            &TmpUInt),
        "Error returned from hsa_agent_get_info when obtaining "
        "HSA_AMD_AGENT_INFO_MAX_CLOCK_FREQUENCY\n");
    printf("    Max Clock Freq(MHz): \t\t%u \n", TmpUInt);
    core::checkResult(
        hsa_agent_get_info(
            Agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT,
            &TmpUInt),
        "Error returned from hsa_agent_get_info when obtaining "
        "HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT\n");
    printf("    Compute Units: \t\t\t%u \n", TmpUInt);
    core::checkResult(hsa_agent_get_info(
                          Agent,
                          (hsa_agent_info_t)HSA_AMD_AGENT_INFO_NUM_SIMDS_PER_CU,
                          &TmpUInt),
                      "Error returned from hsa_agent_get_info when obtaining "
                      "HSA_AMD_AGENT_INFO_NUM_SIMDS_PER_CU\n");
    printf("    SIMD per CU: \t\t\t%u \n", TmpUInt);
    core::checkResult(
        hsa_agent_get_info(Agent, HSA_AGENT_INFO_FAST_F16_OPERATION, &TmpBool),
        "Error returned from hsa_agent_get_info when obtaining "
        "HSA_AMD_AGENT_INFO_NUM_SIMDS_PER_CU\n");
    printf("    Fast F16 Operation: \t\t%s \n", (TmpBool ? "TRUE" : "FALSE"));
    core::checkResult(
        hsa_agent_get_info(Agent, HSA_AGENT_INFO_WAVEFRONT_SIZE, &TmpUInt2),
        "Error returned from hsa_agent_get_info when obtaining "
        "HSA_AGENT_INFO_WAVEFRONT_SIZE\n");
    printf("    Wavefront Size: \t\t\t%u \n", TmpUInt2);
    core::checkResult(
        hsa_agent_get_info(Agent, HSA_AGENT_INFO_WORKGROUP_MAX_SIZE, &TmpUInt),
        "Error returned from hsa_agent_get_info when obtaining "
        "HSA_AGENT_INFO_WORKGROUP_MAX_SIZE\n");
    printf("    Workgroup Max Size: \t\t%u \n", TmpUInt);
    core::checkResult(hsa_agent_get_info(Agent,
                                         HSA_AGENT_INFO_WORKGROUP_MAX_DIM,
                                         WorkgroupMaxDim),
                      "Error returned from hsa_agent_get_info when obtaining "
                      "HSA_AGENT_INFO_WORKGROUP_MAX_DIM\n");
    printf("    Workgroup Max Size per Dimension:\n");
    printf("      x: \t\t\t\t%u\n", WorkgroupMaxDim[0]);
    printf("      y: \t\t\t\t%u\n", WorkgroupMaxDim[1]);
    printf("      z: \t\t\t\t%u\n", WorkgroupMaxDim[2]);
    core::checkResult(hsa_agent_get_info(
                          Agent,
                          (hsa_agent_info_t)HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU,
                          &TmpUInt),
                      "Error returned from hsa_agent_get_info when obtaining "
                      "HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU\n");
    printf("    Max Waves Per CU: \t\t\t%u \n", TmpUInt);
    printf("    Max Work-item Per CU: \t\t%u \n", TmpUInt * TmpUInt2);
    core::checkResult(
        hsa_agent_get_info(Agent, HSA_AGENT_INFO_GRID_MAX_SIZE, &TmpUInt),
        "Error returned from hsa_agent_get_info when obtaining "
        "HSA_AGENT_INFO_GRID_MAX_SIZE\n");
    printf("    Grid Max Size: \t\t\t%u \n", TmpUInt);
    core::checkResult(
        hsa_agent_get_info(Agent, HSA_AGENT_INFO_GRID_MAX_DIM, &GridMaxDim),
        "Error returned from hsa_agent_get_info when obtaining "
        "HSA_AGENT_INFO_GRID_MAX_DIM\n");
    printf("    Grid Max Size per Dimension: \t\t\n");
    printf("      x: \t\t\t\t%u\n", GridMaxDim.x);
    printf("      y: \t\t\t\t%u\n", GridMaxDim.y);
    printf("      z: \t\t\t\t%u\n", GridMaxDim.z);
    core::checkResult(
        hsa_agent_get_info(Agent, HSA_AGENT_INFO_FBARRIER_MAX_SIZE, &TmpUInt),
        "Error returned from hsa_agent_get_info when obtaining "
        "HSA_AGENT_INFO_FBARRIER_MAX_SIZE\n");
    printf("    Max fbarriers/Workgrp: \t\t%u\n", TmpUInt);

    printf("    Memory Pools:\n");
    auto CbMem = [](hsa_amd_memory_pool_t Region, void *Data) -> hsa_status_t {
      std::string TmpStr;
      size_t Size;
      bool Alloc, Access;
      hsa_amd_segment_t Segment;
      hsa_amd_memory_pool_global_flag_t GlobalFlags;
      core::checkResult(
          hsa_amd_memory_pool_get_info(
              Region, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &GlobalFlags),
          "Error returned from hsa_amd_memory_pool_get_info when obtaining "
          "HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS\n");
      core::checkResult(hsa_amd_memory_pool_get_info(
                            Region, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &Segment),
                        "Error returned from hsa_amd_memory_pool_get_info when "
                        "obtaining HSA_AMD_MEMORY_POOL_INFO_SEGMENT\n");

      switch (Segment) {
      case HSA_AMD_SEGMENT_GLOBAL:
        TmpStr = "GLOBAL; FLAGS: ";
        if (HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT & GlobalFlags)
          TmpStr += "KERNARG, ";
        if (HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED & GlobalFlags)
          TmpStr += "FINE GRAINED, ";
        if (HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED & GlobalFlags)
          TmpStr += "COARSE GRAINED, ";
        break;
      case HSA_AMD_SEGMENT_READONLY:
        TmpStr = "READONLY";
        break;
      case HSA_AMD_SEGMENT_PRIVATE:
        TmpStr = "PRIVATE";
        break;
      case HSA_AMD_SEGMENT_GROUP:
        TmpStr = "GROUP";
        break;
      }
      printf("      Pool %s: \n", TmpStr.c_str());

      core::checkResult(hsa_amd_memory_pool_get_info(
                            Region, HSA_AMD_MEMORY_POOL_INFO_SIZE, &Size),
                        "Error returned from hsa_amd_memory_pool_get_info when "
                        "obtaining HSA_AMD_MEMORY_POOL_INFO_SIZE\n");
      printf("        Size: \t\t\t\t %zu bytes\n", Size);
      core::checkResult(
          hsa_amd_memory_pool_get_info(
              Region, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &Alloc),
          "Error returned from hsa_amd_memory_pool_get_info when obtaining "
          "HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED\n");
      printf("        Allocatable: \t\t\t %s\n", (Alloc ? "TRUE" : "FALSE"));
      core::checkResult(
          hsa_amd_memory_pool_get_info(
              Region, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE, &Size),
          "Error returned from hsa_amd_memory_pool_get_info when obtaining "
          "HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE\n");
      printf("        Runtime Alloc Granule: \t\t %zu bytes\n", Size);
      core::checkResult(
          hsa_amd_memory_pool_get_info(
              Region, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALIGNMENT, &Size),
          "Error returned from hsa_amd_memory_pool_get_info when obtaining "
          "HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALIGNMENT\n");
      printf("        Runtime Alloc alignment: \t %zu bytes\n", Size);
      core::checkResult(
          hsa_amd_memory_pool_get_info(
              Region, HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL, &Access),
          "Error returned from hsa_amd_memory_pool_get_info when obtaining "
          "HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL\n");
      printf("        Accessable by all: \t\t %s\n",
             (Access ? "TRUE" : "FALSE"));

      return HSA_STATUS_SUCCESS;
    };
    // Iterate over all the memory regions for this agent. Get the memory region
    // type and size
    hsa_amd_agent_iterate_memory_pools(Agent, CbMem, nullptr);

    printf("    ISAs:\n");
    auto CBIsas = [](hsa_isa_t Isa, void *Data) -> hsa_status_t {
      char TmpChar[1000];
      core::checkResult(hsa_isa_get_info_alt(Isa, HSA_ISA_INFO_NAME, TmpChar),
                        "Error returned from hsa_isa_get_info_alt when "
                        "obtaining HSA_ISA_INFO_NAME\n");
      printf("        Name: \t\t\t\t %s\n", TmpChar);

      return HSA_STATUS_SUCCESS;
    };
    // Iterate over all the memory regions for this agent. Get the memory region
    // type and size
    hsa_agent_iterate_isas(Agent, CBIsas, nullptr);
  }

  // Record entry point associated with device
  void addOffloadEntry(int32_t DeviceId, __tgt_offload_entry Entry) {
    assert(DeviceId < (int32_t)FuncGblEntries.size() &&
           "Unexpected device id!");
    FuncOrGblEntryTy &E = FuncGblEntries[DeviceId].back();

    E.Entries.push_back(Entry);
  }

  // Return true if the entry is associated with device
  bool findOffloadEntry(int32_t DeviceId, void *Addr) {
    assert(DeviceId < (int32_t)FuncGblEntries.size() &&
           "Unexpected device id!");
    FuncOrGblEntryTy &E = FuncGblEntries[DeviceId].back();

    for (auto &It : E.Entries) {
      if (It.addr == Addr)
        return true;
    }

    return false;
  }

  // Return the pointer to the target entries table
  __tgt_target_table *getOffloadEntriesTable(int32_t DeviceId) {
    assert(DeviceId < (int32_t)FuncGblEntries.size() &&
           "Unexpected device id!");
    FuncOrGblEntryTy &E = FuncGblEntries[DeviceId].back();

    int32_t Size = E.Entries.size();

    // Table is empty
    if (!Size)
      return 0;

    __tgt_offload_entry *Begin = &E.Entries[0];
    __tgt_offload_entry *End = &E.Entries[Size - 1];

    // Update table info according to the entries and return the pointer
    E.Table.EntriesBegin = Begin;
    E.Table.EntriesEnd = ++End;

    return &E.Table;
  }

  // Clear entries table for a device
  void clearOffloadEntriesTable(int DeviceId) {
    assert(DeviceId < (int32_t)FuncGblEntries.size() &&
           "Unexpected device id!");
    FuncGblEntries[DeviceId].emplace_back();
    FuncOrGblEntryTy &E = FuncGblEntries[DeviceId].back();
    E.Entries.clear();
    E.Table.EntriesBegin = E.Table.EntriesEnd = 0;
  }

  hsa_status_t addDeviceMemoryPool(hsa_amd_memory_pool_t MemoryPool,
                                   unsigned int DeviceId) {
    assert(DeviceId < DeviceFineGrainedMemoryPools.size() && "Error here.");
    uint32_t GlobalFlags = 0;
    hsa_status_t Err = hsa_amd_memory_pool_get_info(
        MemoryPool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &GlobalFlags);

    if (Err != HSA_STATUS_SUCCESS) {
      return Err;
    }

    if (GlobalFlags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) {
      DeviceFineGrainedMemoryPools[DeviceId] = MemoryPool;
    } else if (GlobalFlags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) {
      DeviceCoarseGrainedMemoryPools[DeviceId] = MemoryPool;
    }

    return HSA_STATUS_SUCCESS;
  }

  hsa_status_t setupDevicePools(const std::vector<hsa_agent_t> &Agents) {
    for (unsigned int DeviceId = 0; DeviceId < Agents.size(); DeviceId++) {
      hsa_status_t Err = hsa::amd_agent_iterate_memory_pools(
          Agents[DeviceId], [&](hsa_amd_memory_pool_t MemoryPool) {
            hsa_status_t ValidStatus = core::isValidMemoryPool(MemoryPool);
            if (ValidStatus != HSA_STATUS_SUCCESS) {
              DP("Alloc allowed in memory pool check failed: %s\n",
                 get_error_string(ValidStatus));
              return HSA_STATUS_SUCCESS;
            }
            return addDeviceMemoryPool(MemoryPool, DeviceId);
          });

      if (Err != HSA_STATUS_SUCCESS) {
        DP("[%s:%d] %s failed: %s\n", __FILE__, __LINE__,
           "Iterate all memory pools", get_error_string(Err));
        return Err;
      }
    }
    return HSA_STATUS_SUCCESS;
  }

  hsa_status_t setupHostMemoryPools(std::vector<hsa_agent_t> &Agents) {
    std::vector<hsa_amd_memory_pool_t> HostPools;

    // collect all the "valid" pools for all the given agents.
    for (const auto &Agent : Agents) {
      hsa_status_t Err = hsa_amd_agent_iterate_memory_pools(
          Agent, core::addMemoryPool, static_cast<void *>(&HostPools));
      if (Err != HSA_STATUS_SUCCESS) {
        DP("addMemoryPool returned %s, continuing\n", get_error_string(Err));
      }
    }

    // We need two fine-grained pools.
    //  1. One with kernarg flag set for storing kernel arguments
    //  2. Second for host allocations
    bool FineGrainedMemoryPoolSet = false;
    bool KernArgPoolSet = false;
    for (const auto &MemoryPool : HostPools) {
      hsa_status_t Err = HSA_STATUS_SUCCESS;
      uint32_t GlobalFlags = 0;
      Err = hsa_amd_memory_pool_get_info(
          MemoryPool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &GlobalFlags);
      if (Err != HSA_STATUS_SUCCESS) {
        DP("Get memory pool info failed: %s\n", get_error_string(Err));
        return Err;
      }

      if (GlobalFlags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) {
        if (GlobalFlags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT) {
          KernArgPool = MemoryPool;
          KernArgPoolSet = true;
        }
        HostFineGrainedMemoryPool = MemoryPool;
        FineGrainedMemoryPoolSet = true;
      }
    }

    if (FineGrainedMemoryPoolSet && KernArgPoolSet)
      return HSA_STATUS_SUCCESS;

    return HSA_STATUS_ERROR;
  }

  hsa_amd_memory_pool_t getDeviceMemoryPool(unsigned int DeviceId) {
    assert(DeviceId >= 0 && DeviceId < DeviceCoarseGrainedMemoryPools.size() &&
           "Invalid device Id");
    return DeviceCoarseGrainedMemoryPools[DeviceId];
  }

  hsa_amd_memory_pool_t getHostMemoryPool() {
    return HostFineGrainedMemoryPool;
  }

  static int readEnv(const char *Env, int Default = -1) {
    const char *EnvStr = getenv(Env);
    int Res = Default;
    if (EnvStr) {
      Res = std::stoi(EnvStr);
      DP("Parsed %s=%d\n", Env, Res);
    }
    return Res;
  }

  RTLDeviceInfoTy() {
    DP("Start initializing " GETNAME(TARGET_NAME) "\n");

    // LIBOMPTARGET_KERNEL_TRACE provides a kernel launch trace to stderr
    // anytime. You do not need a debug library build.
    //  0 => no tracing
    //  1 => tracing dispatch only
    // >1 => verbosity increase

    if (!HSAInitSuccess()) {
      DP("Error when initializing HSA in " GETNAME(TARGET_NAME) "\n");
      return;
    }

    if (char *EnvStr = getenv("LIBOMPTARGET_KERNEL_TRACE"))
      print_kernel_trace = atoi(EnvStr);
    else
      print_kernel_trace = 0;

    hsa_status_t Err = core::atl_init_gpu_context();
    if (Err != HSA_STATUS_SUCCESS) {
      DP("Error when initializing " GETNAME(TARGET_NAME) "\n");
      return;
    }

    // Init hostcall soon after initializing hsa
    hostrpc_init();

    Err = findAgents([&](hsa_device_type_t DeviceType, hsa_agent_t Agent) {
      if (DeviceType == HSA_DEVICE_TYPE_CPU) {
        CPUAgents.push_back(Agent);
      } else {
        HSAAgents.push_back(Agent);
      }
    });
    if (Err != HSA_STATUS_SUCCESS)
      return;

    NumberOfDevices = (int)HSAAgents.size();

    if (NumberOfDevices == 0) {
      DP("There are no devices supporting HSA.\n");
      return;
    }
    DP("There are %d devices supporting HSA.\n", NumberOfDevices);

    // Init the device info
    HSAQueueSchedulers.reserve(NumberOfDevices);
    FuncGblEntries.resize(NumberOfDevices);
    ThreadsPerGroup.resize(NumberOfDevices);
    ComputeUnits.resize(NumberOfDevices);
    GPUName.resize(NumberOfDevices);
    GroupsPerDevice.resize(NumberOfDevices);
    WarpSize.resize(NumberOfDevices);
    NumTeams.resize(NumberOfDevices);
    NumThreads.resize(NumberOfDevices);
    DeviceStateStore.resize(NumberOfDevices);
    KernelInfoTable.resize(NumberOfDevices);
    SymbolInfoTable.resize(NumberOfDevices);
    DeviceCoarseGrainedMemoryPools.resize(NumberOfDevices);
    DeviceFineGrainedMemoryPools.resize(NumberOfDevices);

    Err = setupDevicePools(HSAAgents);
    if (Err != HSA_STATUS_SUCCESS) {
      DP("Setup for Device Memory Pools failed\n");
      return;
    }

    Err = setupHostMemoryPools(CPUAgents);
    if (Err != HSA_STATUS_SUCCESS) {
      DP("Setup for Host Memory Pools failed\n");
      return;
    }

    for (int I = 0; I < NumberOfDevices; I++) {
      uint32_t QueueSize = 0;
      {
        hsa_status_t Err = hsa_agent_get_info(
            HSAAgents[I], HSA_AGENT_INFO_QUEUE_MAX_SIZE, &QueueSize);
        if (Err != HSA_STATUS_SUCCESS) {
          DP("HSA query QUEUE_MAX_SIZE failed for agent %d\n", I);
          return;
        }
        enum { MaxQueueSize = 4096 };
        if (QueueSize > MaxQueueSize) {
          QueueSize = MaxQueueSize;
        }
      }

      {
        HSAQueueScheduler QSched;
        if (!QSched.createQueues(HSAAgents[I], QueueSize))
          return;
        HSAQueueSchedulers.emplace_back(std::move(QSched));
      }

      DeviceStateStore[I] = {nullptr, 0};
    }

    for (int I = 0; I < NumberOfDevices; I++) {
      ThreadsPerGroup[I] = RTLDeviceInfoTy::DefaultWgSize;
      GroupsPerDevice[I] = RTLDeviceInfoTy::DefaultNumTeams;
      ComputeUnits[I] = 1;
      DP("Device %d: Initial groupsPerDevice %d & threadsPerGroup %d\n", I,
         GroupsPerDevice[I], ThreadsPerGroup[I]);
    }

    // Get environment variables regarding teams
    Env.TeamLimit = readEnv("OMP_TEAM_LIMIT");
    Env.NumTeams = readEnv("OMP_NUM_TEAMS");
    Env.MaxTeamsDefault = readEnv("OMP_MAX_TEAMS_DEFAULT");
    Env.TeamThreadLimit = readEnv("OMP_TEAMS_THREAD_LIMIT");
    Env.DynamicMemSize = readEnv("LIBOMPTARGET_SHARED_MEMORY_SIZE", 0);

    // Default state.
    RequiresFlags = OMP_REQ_UNDEFINED;

    ConstructionSucceeded = true;
  }

  ~RTLDeviceInfoTy() {
    DP("Finalizing the " GETNAME(TARGET_NAME) " DeviceInfo.\n");
    if (!HSAInitSuccess()) {
      // Then none of these can have been set up and they can't be torn down
      return;
    }
    // Run destructors on types that use HSA before
    // impl_finalize removes access to it
    DeviceStateStore.clear();
    KernelArgPoolMap.clear();
    // Terminate hostrpc before finalizing hsa
    hostrpc_terminate();

    hsa_status_t Err;
    for (uint32_t I = 0; I < HSAExecutables.size(); I++) {
      Err = hsa_executable_destroy(HSAExecutables[I]);
      if (Err != HSA_STATUS_SUCCESS) {
        DP("[%s:%d] %s failed: %s\n", __FILE__, __LINE__,
           "Destroying executable", get_error_string(Err));
      }
    }
  }
};

pthread_mutex_t SignalPoolT::mutex = PTHREAD_MUTEX_INITIALIZER;

static RTLDeviceInfoTy *DeviceInfoState = nullptr;
static RTLDeviceInfoTy &DeviceInfo() { return *DeviceInfoState; }

namespace {

int32_t dataRetrieve(int32_t DeviceId, void *HstPtr, void *TgtPtr, int64_t Size,
                     __tgt_async_info *AsyncInfo) {
  assert(AsyncInfo && "AsyncInfo is nullptr");
  assert(DeviceId < DeviceInfo().NumberOfDevices && "Device ID too large");
  // Return success if we are not copying back to host from target.
  if (!HstPtr)
    return OFFLOAD_SUCCESS;
  hsa_status_t Err;
  DP("Retrieve data %ld bytes, (tgt:%016llx) -> (hst:%016llx).\n", Size,
     (long long unsigned)(Elf64_Addr)TgtPtr,
     (long long unsigned)(Elf64_Addr)HstPtr);

  Err = DeviceInfo().freesignalpoolMemcpyD2H(HstPtr, TgtPtr, (size_t)Size,
                                             DeviceId);

  if (Err != HSA_STATUS_SUCCESS) {
    DP("Error when copying data from device to host. Pointers: "
       "host = 0x%016lx, device = 0x%016lx, size = %lld\n",
       (Elf64_Addr)HstPtr, (Elf64_Addr)TgtPtr, (unsigned long long)Size);
    return OFFLOAD_FAIL;
  }
  DP("DONE Retrieve data %ld bytes, (tgt:%016llx) -> (hst:%016llx).\n", Size,
     (long long unsigned)(Elf64_Addr)TgtPtr,
     (long long unsigned)(Elf64_Addr)HstPtr);
  return OFFLOAD_SUCCESS;
}

int32_t dataSubmit(int32_t DeviceId, void *TgtPtr, void *HstPtr, int64_t Size,
                   __tgt_async_info *AsyncInfo) {
  assert(AsyncInfo && "AsyncInfo is nullptr");
  hsa_status_t Err;
  assert(DeviceId < DeviceInfo().NumberOfDevices && "Device ID too large");
  // Return success if we are not doing host to target.
  if (!HstPtr)
    return OFFLOAD_SUCCESS;

  DP("Submit data %ld bytes, (hst:%016llx) -> (tgt:%016llx).\n", Size,
     (long long unsigned)(Elf64_Addr)HstPtr,
     (long long unsigned)(Elf64_Addr)TgtPtr);
  Err = DeviceInfo().freesignalpoolMemcpyH2D(TgtPtr, HstPtr, (size_t)Size,
                                             DeviceId);
  if (Err != HSA_STATUS_SUCCESS) {
    DP("Error when copying data from host to device. Pointers: "
       "host = 0x%016lx, device = 0x%016lx, size = %lld\n",
       (Elf64_Addr)HstPtr, (Elf64_Addr)TgtPtr, (unsigned long long)Size);
    return OFFLOAD_FAIL;
  }
  return OFFLOAD_SUCCESS;
}

// Async.
// The implementation was written with cuda streams in mind. The semantics of
// that are to execute kernels on a queue in order of insertion. A synchronise
// call then makes writes visible between host and device. This means a series
// of N data_submit_async calls are expected to execute serially. HSA offers
// various options to run the data copies concurrently. This may require changes
// to libomptarget.

// __tgt_async_info* contains a void * Queue. Queue = 0 is used to indicate that
// there are no outstanding kernels that need to be synchronized. Any async call
// may be passed a Queue==0, at which point the cuda implementation will set it
// to non-null (see getStream). The cuda streams are per-device. Upstream may
// change this interface to explicitly initialize the AsyncInfo_pointer, but
// until then hsa lazily initializes it as well.

void initAsyncInfo(__tgt_async_info *AsyncInfo) {
  // set non-null while using async calls, return to null to indicate completion
  assert(AsyncInfo);
  if (!AsyncInfo->Queue) {
    AsyncInfo->Queue = reinterpret_cast<void *>(UINT64_MAX);
  }
}
void finiAsyncInfo(__tgt_async_info *AsyncInfo) {
  assert(AsyncInfo);
  assert(AsyncInfo->Queue);
  AsyncInfo->Queue = 0;
}

// Determine launch values for kernel.
struct LaunchVals {
  int WorkgroupSize;
  int GridSize;
};
LaunchVals getLaunchVals(int WarpSize, EnvironmentVariables Env,
                         int ConstWGSize,
                         llvm::omp::OMPTgtExecModeFlags ExecutionMode,
                         int NumTeams, int ThreadLimit, uint64_t LoopTripcount,
                         int DeviceNumTeams) {

  int ThreadsPerGroup = RTLDeviceInfoTy::DefaultWgSize;
  int NumGroups = 0;

  int MaxTeams = Env.MaxTeamsDefault > 0 ? Env.MaxTeamsDefault : DeviceNumTeams;
  if (MaxTeams > static_cast<int>(RTLDeviceInfoTy::HardTeamLimit))
    MaxTeams = RTLDeviceInfoTy::HardTeamLimit;

  if (print_kernel_trace & STARTUP_DETAILS) {
    DP("RTLDeviceInfoTy::Max_Teams: %d\n", RTLDeviceInfoTy::MaxTeams);
    DP("Max_Teams: %d\n", MaxTeams);
    DP("RTLDeviceInfoTy::Warp_Size: %d\n", WarpSize);
    DP("RTLDeviceInfoTy::Max_WG_Size: %d\n", RTLDeviceInfoTy::MaxWgSize);
    DP("RTLDeviceInfoTy::Default_WG_Size: %d\n",
       RTLDeviceInfoTy::DefaultWgSize);
    DP("thread_limit: %d\n", ThreadLimit);
    DP("threadsPerGroup: %d\n", ThreadsPerGroup);
    DP("ConstWGSize: %d\n", ConstWGSize);
  }
  // check for thread_limit() clause
  if (ThreadLimit > 0) {
    ThreadsPerGroup = ThreadLimit;
    DP("Setting threads per block to requested %d\n", ThreadLimit);
    // Add master warp for GENERIC
    if (ExecutionMode ==
        llvm::omp::OMPTgtExecModeFlags::OMP_TGT_EXEC_MODE_GENERIC) {
      ThreadsPerGroup += WarpSize;
      DP("Adding master wavefront: +%d threads\n", WarpSize);
    }
    if (ThreadsPerGroup > RTLDeviceInfoTy::MaxWgSize) { // limit to max
      ThreadsPerGroup = RTLDeviceInfoTy::MaxWgSize;
      DP("Setting threads per block to maximum %d\n", ThreadsPerGroup);
    }
  }
  // check flat_max_work_group_size attr here
  if (ThreadsPerGroup > ConstWGSize) {
    ThreadsPerGroup = ConstWGSize;
    DP("Reduced threadsPerGroup to flat-attr-group-size limit %d\n",
       ThreadsPerGroup);
  }
  if (print_kernel_trace & STARTUP_DETAILS)
    DP("threadsPerGroup: %d\n", ThreadsPerGroup);
  DP("Preparing %d threads\n", ThreadsPerGroup);

  // Set default num_groups (teams)
  if (Env.TeamLimit > 0)
    NumGroups = (MaxTeams < Env.TeamLimit) ? MaxTeams : Env.TeamLimit;
  else
    NumGroups = MaxTeams;
  DP("Set default num of groups %d\n", NumGroups);

  if (print_kernel_trace & STARTUP_DETAILS) {
    DP("num_groups: %d\n", NumGroups);
    DP("num_teams: %d\n", NumTeams);
  }

  // Reduce num_groups if threadsPerGroup exceeds RTLDeviceInfoTy::Max_WG_Size
  // This reduction is typical for default case (no thread_limit clause).
  // or when user goes crazy with num_teams clause.
  // FIXME: We cant distinguish between a constant or variable thread limit.
  // So we only handle constant thread_limits.
  if (ThreadsPerGroup >
      RTLDeviceInfoTy::DefaultWgSize) //  256 < threadsPerGroup <= 1024
    // Should we round threadsPerGroup up to nearest WarpSize
    // here?
    NumGroups = (MaxTeams * RTLDeviceInfoTy::MaxWgSize) / ThreadsPerGroup;

  // check for num_teams() clause
  if (NumTeams > 0) {
    NumGroups = (NumTeams < NumGroups) ? NumTeams : NumGroups;
  }
  if (print_kernel_trace & STARTUP_DETAILS) {
    DP("num_groups: %d\n", NumGroups);
    DP("Env.NumTeams %d\n", Env.NumTeams);
    DP("Env.TeamLimit %d\n", Env.TeamLimit);
  }

  if (Env.NumTeams > 0) {
    NumGroups = (Env.NumTeams < NumGroups) ? Env.NumTeams : NumGroups;
    DP("Modifying teams based on Env.NumTeams %d\n", Env.NumTeams);
  } else if (Env.TeamLimit > 0) {
    NumGroups = (Env.TeamLimit < NumGroups) ? Env.TeamLimit : NumGroups;
    DP("Modifying teams based on Env.TeamLimit%d\n", Env.TeamLimit);
  } else {
    if (NumTeams <= 0) {
      if (LoopTripcount > 0) {
        if (ExecutionMode ==
            llvm::omp::OMPTgtExecModeFlags::OMP_TGT_EXEC_MODE_SPMD) {
          // round up to the nearest integer
          NumGroups = ((LoopTripcount - 1) / ThreadsPerGroup) + 1;
        } else if (ExecutionMode ==
                   llvm::omp::OMPTgtExecModeFlags::OMP_TGT_EXEC_MODE_GENERIC) {
          NumGroups = LoopTripcount;
        } else /* OMP_TGT_EXEC_MODE_GENERIC_SPMD */ {
          // This is a generic kernel that was transformed to use SPMD-mode
          // execution but uses Generic-mode semantics for scheduling.
          NumGroups = LoopTripcount;
        }
        DP("Using %d teams due to loop trip count %" PRIu64 " and number of "
           "threads per block %d\n",
           NumGroups, LoopTripcount, ThreadsPerGroup);
      }
    } else {
      NumGroups = NumTeams;
    }
    if (NumGroups > MaxTeams) {
      NumGroups = MaxTeams;
      if (print_kernel_trace & STARTUP_DETAILS)
        DP("Limiting num_groups %d to Max_Teams %d \n", NumGroups, MaxTeams);
    }
    if (NumGroups > NumTeams && NumTeams > 0) {
      NumGroups = NumTeams;
      if (print_kernel_trace & STARTUP_DETAILS)
        DP("Limiting num_groups %d to clause num_teams %d \n", NumGroups,
           NumTeams);
    }
  }

  // num_teams clause always honored, no matter what, unless DEFAULT is active.
  if (NumTeams > 0) {
    NumGroups = NumTeams;
    // Cap num_groups to EnvMaxTeamsDefault if set.
    if (Env.MaxTeamsDefault > 0 && NumGroups > Env.MaxTeamsDefault)
      NumGroups = Env.MaxTeamsDefault;
  }
  if (print_kernel_trace & STARTUP_DETAILS) {
    DP("threadsPerGroup: %d\n", ThreadsPerGroup);
    DP("num_groups: %d\n", NumGroups);
    DP("loop_tripcount: %ld\n", LoopTripcount);
  }
  DP("Final %d num_groups and %d threadsPerGroup\n", NumGroups,
     ThreadsPerGroup);

  LaunchVals Res;
  Res.WorkgroupSize = ThreadsPerGroup;
  Res.GridSize = ThreadsPerGroup * NumGroups;
  return Res;
}

static uint64_t acquireAvailablePacketId(hsa_queue_t *Queue) {
  uint64_t PacketId = hsa_queue_add_write_index_relaxed(Queue, 1);
  bool Full = true;
  while (Full) {
    Full =
        PacketId >= (Queue->size + hsa_queue_load_read_index_scacquire(Queue));
  }
  return PacketId;
}

int32_t runRegionLocked(int32_t DeviceId, void *TgtEntryPtr, void **TgtArgs,
                        ptrdiff_t *TgtOffsets, int32_t ArgNum, int32_t NumTeams,
                        int32_t ThreadLimit, uint64_t LoopTripcount) {
  // Set the context we are using
  // update thread limit content in gpu memory if un-initialized or specified
  // from host

  DP("Run target team region thread_limit %d\n", ThreadLimit);

  // All args are references.
  std::vector<void *> Args(ArgNum);
  std::vector<void *> Ptrs(ArgNum);

  DP("Arg_num: %d\n", ArgNum);
  for (int32_t I = 0; I < ArgNum; ++I) {
    Ptrs[I] = (void *)((intptr_t)TgtArgs[I] + TgtOffsets[I]);
    Args[I] = &Ptrs[I];
    DP("Offseted base: arg[%d]:" DPxMOD "\n", I, DPxPTR(Ptrs[I]));
  }

  KernelTy *KernelInfo = (KernelTy *)TgtEntryPtr;

  std::string KernelName = std::string(KernelInfo->Name);
  auto &KernelInfoTable = DeviceInfo().KernelInfoTable;
  if (KernelInfoTable[DeviceId].find(KernelName) ==
      KernelInfoTable[DeviceId].end()) {
    DP("Kernel %s not found\n", KernelName.c_str());
    return OFFLOAD_FAIL;
  }

  const atl_kernel_info_t KernelInfoEntry =
      KernelInfoTable[DeviceId][KernelName];
  const uint32_t GroupSegmentSize =
      KernelInfoEntry.group_segment_size + DeviceInfo().Env.DynamicMemSize;
  const uint32_t SgprCount = KernelInfoEntry.sgpr_count;
  const uint32_t VgprCount = KernelInfoEntry.vgpr_count;
  const uint32_t SgprSpillCount = KernelInfoEntry.sgpr_spill_count;
  const uint32_t VgprSpillCount = KernelInfoEntry.vgpr_spill_count;

  assert(ArgNum == (int)KernelInfoEntry.explicit_argument_count);

  /*
   * Set limit based on ThreadsPerGroup and GroupsPerDevice
   */
  LaunchVals LV =
      getLaunchVals(DeviceInfo().WarpSize[DeviceId], DeviceInfo().Env,
                    KernelInfo->ConstWGSize, KernelInfo->ExecutionMode,
                    NumTeams,      // From run_region arg
                    ThreadLimit,   // From run_region arg
                    LoopTripcount, // From run_region arg
                    DeviceInfo().NumTeams[KernelInfo->DeviceId]);
  const int GridSize = LV.GridSize;
  const int WorkgroupSize = LV.WorkgroupSize;

  if (print_kernel_trace >= LAUNCH) {
    int NumGroups = GridSize / WorkgroupSize;
    // enum modes are SPMD, GENERIC, NONE 0,1,2
    // if doing rtl timing, print to stderr, unless stdout requested.
    bool TraceToStdout = print_kernel_trace & (RTL_TO_STDOUT | RTL_TIMING);
    fprintf(TraceToStdout ? stdout : stderr,
            "DEVID:%2d SGN:%1d ConstWGSize:%-4d args:%2d teamsXthrds:(%4dX%4d) "
            "reqd:(%4dX%4d) lds_usage:%uB sgpr_count:%u vgpr_count:%u "
            "sgpr_spill_count:%u vgpr_spill_count:%u tripcount:%lu n:%s\n",
            DeviceId, KernelInfo->ExecutionMode, KernelInfo->ConstWGSize,
            ArgNum, NumGroups, WorkgroupSize, NumTeams, ThreadLimit,
            GroupSegmentSize, SgprCount, VgprCount, SgprSpillCount,
            VgprSpillCount, LoopTripcount, KernelInfo->Name);
  }

  // Run on the device.
  {
    hsa_queue_t *Queue = DeviceInfo().HSAQueueSchedulers[DeviceId].next();
    if (!Queue) {
      return OFFLOAD_FAIL;
    }
    uint64_t PacketId = acquireAvailablePacketId(Queue);

    const uint32_t Mask = Queue->size - 1; // size is a power of 2
    hsa_kernel_dispatch_packet_t *Packet =
        (hsa_kernel_dispatch_packet_t *)Queue->base_address + (PacketId & Mask);

    // packet->header is written last
    Packet->setup = UINT16_C(1) << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
    Packet->workgroup_size_x = WorkgroupSize;
    Packet->workgroup_size_y = 1;
    Packet->workgroup_size_z = 1;
    Packet->reserved0 = 0;
    Packet->grid_size_x = GridSize;
    Packet->grid_size_y = 1;
    Packet->grid_size_z = 1;
    Packet->private_segment_size = KernelInfoEntry.private_segment_size;
    Packet->group_segment_size = GroupSegmentSize;
    Packet->kernel_object = KernelInfoEntry.kernel_object;
    Packet->kernarg_address = 0;     // use the block allocator
    Packet->reserved2 = 0;           // impl writes id_ here
    Packet->completion_signal = {0}; // may want a pool of signals

    KernelArgPool *ArgPool = nullptr;
    void *KernArg = nullptr;
    {
      auto It =
          DeviceInfo().KernelArgPoolMap.find(std::string(KernelInfo->Name));
      if (It != DeviceInfo().KernelArgPoolMap.end()) {
        ArgPool = (It->second).get();
      }
    }
    if (!ArgPool) {
      DP("Warning: No ArgPool for %s on device %d\n", KernelInfo->Name,
         DeviceId);
    }
    {
      if (ArgPool) {
        assert(ArgPool->KernargSegmentSize == (ArgNum * sizeof(void *)));
        KernArg = ArgPool->allocate(ArgNum);
      }
      if (!KernArg) {
        DP("Allocate kernarg failed\n");
        return OFFLOAD_FAIL;
      }

      // Copy explicit arguments
      for (int I = 0; I < ArgNum; I++) {
        memcpy((char *)KernArg + sizeof(void *) * I, Args[I], sizeof(void *));
      }

      // Initialize implicit arguments. TODO: Which of these can be dropped
      impl_implicit_args_t *ImplArgs = reinterpret_cast<impl_implicit_args_t *>(
          static_cast<char *>(KernArg) + ArgPool->KernargSegmentSize);
      memset(ImplArgs, 0,
             sizeof(impl_implicit_args_t)); // may not be necessary
      ImplArgs->offset_x = 0;
      ImplArgs->offset_y = 0;
      ImplArgs->offset_z = 0;

      // assign a hostcall buffer for the selected Q
      if (__atomic_load_n(&DeviceInfo().HostcallRequired, __ATOMIC_ACQUIRE)) {
        // hostrpc_assign_buffer is not thread safe, and this function is
        // under a multiple reader lock, not a writer lock.
        static pthread_mutex_t HostcallInitLock = PTHREAD_MUTEX_INITIALIZER;
        pthread_mutex_lock(&HostcallInitLock);
        uint64_t Buffer = hostrpc_assign_buffer(
            DeviceInfo().HSAAgents[DeviceId], Queue, DeviceId);
        pthread_mutex_unlock(&HostcallInitLock);
        if (!Buffer) {
          DP("hostrpc_assign_buffer failed, gpu would dereference null and "
             "error\n");
          return OFFLOAD_FAIL;
        }

        DP("Implicit argument count: %d\n",
           KernelInfoEntry.implicit_argument_count);
        if (KernelInfoEntry.implicit_argument_count >= 4) {
          // Initialise pointer for implicit_argument_count != 0 ABI
          // Guess that the right implicit argument is at offset 24 after
          // the explicit arguments. In the future, should be able to read
          // the offset from msgpack. Clang is not annotating it at present.
          uint64_t Offset =
              sizeof(void *) * (KernelInfoEntry.explicit_argument_count + 3);
          if ((Offset + 8) > ArgPool->kernargSizeIncludingImplicit()) {
            DP("Bad offset of hostcall: %lu, exceeds kernarg size w/ implicit "
               "args: %d\n",
               Offset + 8, ArgPool->kernargSizeIncludingImplicit());
          } else {
            memcpy(static_cast<char *>(KernArg) + Offset, &Buffer, 8);
          }
        }

        // initialise pointer for implicit_argument_count == 0 ABI
        ImplArgs->hostcall_ptr = Buffer;
      }

      Packet->kernarg_address = KernArg;
    }

    hsa_signal_t S = DeviceInfo().FreeSignalPool.pop();
    if (S.handle == 0) {
      DP("Failed to get signal instance\n");
      return OFFLOAD_FAIL;
    }
    Packet->completion_signal = S;
    hsa_signal_store_relaxed(Packet->completion_signal, 1);

    // Publish the packet indicating it is ready to be processed
    core::packetStoreRelease(reinterpret_cast<uint32_t *>(Packet),
                             core::createHeader(), Packet->setup);

    // Since the packet is already published, its contents must not be
    // accessed any more
    hsa_signal_store_relaxed(Queue->doorbell_signal, PacketId);

    while (hsa_signal_wait_scacquire(S, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX,
                                     HSA_WAIT_STATE_BLOCKED) != 0)
      ;

    assert(ArgPool);
    ArgPool->deallocate(KernArg);
    DeviceInfo().FreeSignalPool.push(S);
  }

  DP("Kernel completed\n");
  return OFFLOAD_SUCCESS;
}

bool elfMachineIdIsAmdgcn(__tgt_device_image *Image) {
  const uint16_t AmdgcnMachineID = 224; // EM_AMDGPU may not be in system elf.h
  int32_t R = elf_check_machine(Image, AmdgcnMachineID);
  if (!R) {
    DP("Supported machine ID not found\n");
  }
  return R;
}

uint32_t elfEFlags(__tgt_device_image *Image) {
  char *ImgBegin = (char *)Image->ImageStart;
  size_t ImgSize = (char *)Image->ImageEnd - ImgBegin;

  Elf *E = elf_memory(ImgBegin, ImgSize);
  if (!E) {
    DP("Unable to get ELF handle: %s!\n", elf_errmsg(-1));
    return 0;
  }

  Elf64_Ehdr *Eh64 = elf64_getehdr(E);

  if (!Eh64) {
    DP("Unable to get machine ID from ELF file!\n");
    elf_end(E);
    return 0;
  }

  uint32_t Flags = Eh64->e_flags;

  elf_end(E);
  DP("ELF Flags: 0x%x\n", Flags);
  return Flags;
}

template <typename T> bool enforceUpperBound(T *Value, T Upper) {
  bool Changed = *Value > Upper;
  if (Changed) {
    *Value = Upper;
  }
  return Changed;
}

Elf64_Shdr *findOnlyShtHash(Elf *Elf) {
  size_t N;
  int Rc = elf_getshdrnum(Elf, &N);
  if (Rc != 0) {
    return nullptr;
  }

  Elf64_Shdr *Result = nullptr;
  for (size_t I = 0; I < N; I++) {
    Elf_Scn *Scn = elf_getscn(Elf, I);
    if (Scn) {
      Elf64_Shdr *Shdr = elf64_getshdr(Scn);
      if (Shdr) {
        if (Shdr->sh_type == SHT_HASH) {
          if (Result == nullptr) {
            Result = Shdr;
          } else {
            // multiple SHT_HASH sections not handled
            return nullptr;
          }
        }
      }
    }
  }
  return Result;
}

const Elf64_Sym *elfLookup(Elf *Elf, char *Base, Elf64_Shdr *SectionHash,
                           const char *Symname) {

  assert(SectionHash);
  size_t SectionSymtabIndex = SectionHash->sh_link;
  Elf64_Shdr *SectionSymtab =
      elf64_getshdr(elf_getscn(Elf, SectionSymtabIndex));
  size_t SectionStrtabIndex = SectionSymtab->sh_link;

  const Elf64_Sym *Symtab =
      reinterpret_cast<const Elf64_Sym *>(Base + SectionSymtab->sh_offset);

  const uint32_t *Hashtab =
      reinterpret_cast<const uint32_t *>(Base + SectionHash->sh_offset);

  // Layout:
  // nbucket
  // nchain
  // bucket[nbucket]
  // chain[nchain]
  uint32_t Nbucket = Hashtab[0];
  const uint32_t *Bucket = &Hashtab[2];
  const uint32_t *Chain = &Hashtab[Nbucket + 2];

  const size_t Max = strlen(Symname) + 1;
  const uint32_t Hash = elf_hash(Symname);
  for (uint32_t I = Bucket[Hash % Nbucket]; I != 0; I = Chain[I]) {
    char *N = elf_strptr(Elf, SectionStrtabIndex, Symtab[I].st_name);
    if (strncmp(Symname, N, Max) == 0) {
      return &Symtab[I];
    }
  }

  return nullptr;
}

struct SymbolInfo {
  void *Addr = nullptr;
  uint32_t Size = UINT32_MAX;
  uint32_t ShType = SHT_NULL;
};

int getSymbolInfoWithoutLoading(Elf *Elf, char *Base, const char *Symname,
                                SymbolInfo *Res) {
  if (elf_kind(Elf) != ELF_K_ELF) {
    return 1;
  }

  Elf64_Shdr *SectionHash = findOnlyShtHash(Elf);
  if (!SectionHash) {
    return 1;
  }

  const Elf64_Sym *Sym = elfLookup(Elf, Base, SectionHash, Symname);
  if (!Sym) {
    return 1;
  }

  if (Sym->st_size > UINT32_MAX) {
    return 1;
  }

  if (Sym->st_shndx == SHN_UNDEF) {
    return 1;
  }

  Elf_Scn *Section = elf_getscn(Elf, Sym->st_shndx);
  if (!Section) {
    return 1;
  }

  Elf64_Shdr *Header = elf64_getshdr(Section);
  if (!Header) {
    return 1;
  }

  Res->Addr = Sym->st_value + Base;
  Res->Size = static_cast<uint32_t>(Sym->st_size);
  Res->ShType = Header->sh_type;
  return 0;
}

int getSymbolInfoWithoutLoading(char *Base, size_t ImgSize, const char *Symname,
                                SymbolInfo *Res) {
  Elf *Elf = elf_memory(Base, ImgSize);
  if (Elf) {
    int Rc = getSymbolInfoWithoutLoading(Elf, Base, Symname, Res);
    elf_end(Elf);
    return Rc;
  }
  return 1;
}

hsa_status_t interopGetSymbolInfo(char *Base, size_t ImgSize,
                                  const char *SymName, void **VarAddr,
                                  uint32_t *VarSize) {
  SymbolInfo SI;
  int Rc = getSymbolInfoWithoutLoading(Base, ImgSize, SymName, &SI);
  if (Rc == 0) {
    *VarAddr = SI.Addr;
    *VarSize = SI.Size;
    return HSA_STATUS_SUCCESS;
  }
  return HSA_STATUS_ERROR;
}

template <typename C>
hsa_status_t moduleRegisterFromMemoryToPlace(
    std::map<std::string, atl_kernel_info_t> &KernelInfoTable,
    std::map<std::string, atl_symbol_info_t> &SymbolInfoTable,
    void *ModuleBytes, size_t ModuleSize, int DeviceId, C Cb,
    std::vector<hsa_executable_t> &HSAExecutables) {
  auto L = [](void *Data, size_t Size, void *CbState) -> hsa_status_t {
    C *Unwrapped = static_cast<C *>(CbState);
    return (*Unwrapped)(Data, Size);
  };
  return core::RegisterModuleFromMemory(
      KernelInfoTable, SymbolInfoTable, ModuleBytes, ModuleSize,
      DeviceInfo().HSAAgents[DeviceId], L, static_cast<void *>(&Cb),
      HSAExecutables);
}

uint64_t getDeviceStateBytes(char *ImageStart, size_t ImgSize) {
  uint64_t DeviceStateBytes = 0;
  {
    // If this is the deviceRTL, get the state variable size
    SymbolInfo SizeSi;
    int Rc = getSymbolInfoWithoutLoading(
        ImageStart, ImgSize, "omptarget_nvptx_device_State_size", &SizeSi);

    if (Rc == 0) {
      if (SizeSi.Size != sizeof(uint64_t)) {
        DP("Found device_State_size variable with wrong size\n");
        return 0;
      }

      // Read number of bytes directly from the elf
      memcpy(&DeviceStateBytes, SizeSi.Addr, sizeof(uint64_t));
    }
  }
  return DeviceStateBytes;
}

struct DeviceEnvironment {
  // initialise an DeviceEnvironmentTy in the deviceRTL
  // patches around differences in the deviceRTL between trunk, aomp,
  // rocmcc. Over time these differences will tend to zero and this class
  // simplified.
  // Symbol may be in .data or .bss, and may be missing fields, todo:
  // review aomp/trunk/rocm and simplify the following

  // The symbol may also have been deadstripped because the device side
  // accessors were unused.

  // If the symbol is in .data (aomp, rocm) it can be written directly.
  // If it is in .bss, we must wait for it to be allocated space on the
  // gpu (trunk) and initialize after loading.
  const char *sym() { return "omptarget_device_environment"; }

  DeviceEnvironmentTy HostDeviceEnv;
  SymbolInfo SI;
  bool Valid = false;

  __tgt_device_image *Image;
  const size_t ImgSize;

  DeviceEnvironment(int DeviceId, int NumberDevices, int DynamicMemSize,
                    __tgt_device_image *Image, const size_t ImgSize)
      : Image(Image), ImgSize(ImgSize) {

    HostDeviceEnv.NumDevices = NumberDevices;
    HostDeviceEnv.DeviceNum = DeviceId;
    HostDeviceEnv.DebugKind = 0;
    HostDeviceEnv.DynamicMemSize = DynamicMemSize;
    if (char *EnvStr = getenv("LIBOMPTARGET_DEVICE_RTL_DEBUG"))
      HostDeviceEnv.DebugKind = std::stoi(EnvStr);

    int Rc = getSymbolInfoWithoutLoading((char *)Image->ImageStart, ImgSize,
                                         sym(), &SI);
    if (Rc != 0) {
      DP("Finding global device environment '%s' - symbol missing.\n", sym());
      return;
    }

    if (SI.Size > sizeof(HostDeviceEnv)) {
      DP("Symbol '%s' has size %u, expected at most %zu.\n", sym(), SI.Size,
         sizeof(HostDeviceEnv));
      return;
    }

    Valid = true;
  }

  bool inImage() { return SI.ShType != SHT_NOBITS; }

  hsa_status_t beforeLoading(void *Data, size_t Size) {
    if (Valid) {
      if (inImage()) {
        DP("Setting global device environment before load (%u bytes)\n",
           SI.Size);
        uint64_t Offset = (char *)SI.Addr - (char *)Image->ImageStart;
        void *Pos = (char *)Data + Offset;
        memcpy(Pos, &HostDeviceEnv, SI.Size);
      }
    }
    return HSA_STATUS_SUCCESS;
  }

  hsa_status_t afterLoading() {
    if (Valid) {
      if (!inImage()) {
        DP("Setting global device environment after load (%u bytes)\n",
           SI.Size);
        int DeviceId = HostDeviceEnv.DeviceNum;
        auto &SymbolInfo = DeviceInfo().SymbolInfoTable[DeviceId];
        void *StatePtr;
        uint32_t StatePtrSize;
        hsa_status_t Err = interop_hsa_get_symbol_info(
            SymbolInfo, DeviceId, sym(), &StatePtr, &StatePtrSize);
        if (Err != HSA_STATUS_SUCCESS) {
          DP("failed to find %s in loaded image\n", sym());
          return Err;
        }

        if (StatePtrSize != SI.Size) {
          DP("Symbol had size %u before loading, %u after\n", StatePtrSize,
             SI.Size);
          return HSA_STATUS_ERROR;
        }

        return DeviceInfo().freesignalpoolMemcpyH2D(StatePtr, &HostDeviceEnv,
                                                    StatePtrSize, DeviceId);
      }
    }
    return HSA_STATUS_SUCCESS;
  }
};

hsa_status_t implCalloc(void **RetPtr, size_t Size, int DeviceId) {
  uint64_t Rounded = 4 * ((Size + 3) / 4);
  void *Ptr;
  hsa_amd_memory_pool_t MemoryPool = DeviceInfo().getDeviceMemoryPool(DeviceId);
  hsa_status_t Err = hsa_amd_memory_pool_allocate(MemoryPool, Rounded, 0, &Ptr);
  if (Err != HSA_STATUS_SUCCESS) {
    return Err;
  }

  hsa_status_t Rc = hsa_amd_memory_fill(Ptr, 0, Rounded / 4);
  if (Rc != HSA_STATUS_SUCCESS) {
    DP("zero fill device_state failed with %u\n", Rc);
    core::Runtime::Memfree(Ptr);
    return HSA_STATUS_ERROR;
  }

  *RetPtr = Ptr;
  return HSA_STATUS_SUCCESS;
}

bool imageContainsSymbol(void *Data, size_t Size, const char *Sym) {
  SymbolInfo SI;
  int Rc = getSymbolInfoWithoutLoading((char *)Data, Size, Sym, &SI);
  return (Rc == 0) && (SI.Addr != nullptr);
}

} // namespace

namespace core {
hsa_status_t allow_access_to_all_gpu_agents(void *Ptr) {
  return hsa_amd_agents_allow_access(DeviceInfo().HSAAgents.size(),
                                     &DeviceInfo().HSAAgents[0], NULL, Ptr);
}
} // namespace core

static hsa_status_t GetIsaInfo(hsa_isa_t isa, void *data) {
  hsa_status_t err;
  uint32_t name_len;
  err = hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME_LENGTH, &name_len);
  if (err != HSA_STATUS_SUCCESS) {
    DP("Error getting ISA info length\n");
    return err;
  }

  char TargetID[name_len];
  err = hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME, TargetID);
  if (err != HSA_STATUS_SUCCESS) {
    DP("Error getting ISA info name\n");
    return err;
  }

  auto TripleTargetID = llvm::StringRef(TargetID);
  if (TripleTargetID.consume_front("amdgcn-amd-amdhsa")) {
    DeviceInfo().TargetID.push_back(TripleTargetID.ltrim('-').str());
  }
  return HSA_STATUS_SUCCESS;
}

/// Parse a TargetID to get processor arch and feature map.
/// Returns processor subarch.
/// Returns TargetID features in \p FeatureMap argument.
/// If the \p TargetID contains feature+, FeatureMap it to true.
/// If the \p TargetID contains feature-, FeatureMap it to false.
/// If the \p TargetID does not contain a feature (default), do not map it.
StringRef parseTargetID(StringRef TargetID, StringMap<bool> &FeatureMap) {
  if (TargetID.empty())
    return llvm::StringRef();

  auto ArchFeature = TargetID.split(":");
  auto Arch = ArchFeature.first;
  auto Features = ArchFeature.second;
  if (Features.empty())
    return Arch;

  if (Features.contains("sramecc+")) {
    FeatureMap.insert(std::pair<std::string, bool>("sramecc", true));
  } else if (Features.contains("sramecc-")) {
    FeatureMap.insert(std::pair<std::string, bool>("sramecc", false));
  }
  if (Features.contains("xnack+")) {
    FeatureMap.insert(std::pair<std::string, bool>("xnack", true));
  } else if (Features.contains("xnack-")) {
    FeatureMap.insert(std::pair<std::string, bool>("xnack", false));
  }

  return Arch;
}

/// Checks if an image \p ImgInfo is compatible with current
/// system's environment \p EnvInfo
bool IsImageCompatibleWithEnv(const char *ImgInfo, std::string EnvInfo) {
  llvm::StringRef ImgTID(ImgInfo), EnvTID(EnvInfo);

  // Compatible in case of exact match
  if (ImgTID == EnvTID) {
    DP("Compatible: Exact match \t[Image: %s]\t:\t[Environment: %s]\n",
       ImgTID.data(), EnvTID.data());
    return true;
  }

  // Incompatible if Archs mismatch.
  StringMap<bool> ImgMap, EnvMap;
  StringRef ImgArch = parseTargetID(ImgTID, ImgMap);
  StringRef EnvArch = parseTargetID(EnvTID, EnvMap);

  // Both EnvArch and ImgArch can't be empty here.
  if (EnvArch.empty() || ImgArch.empty() || !ImgArch.contains(EnvArch)) {
    DP("Incompatible: Processor mismatch \t[Image: %s]\t:\t[Environment: %s]\n",
       ImgTID.data(), EnvTID.data());
    return false;
  }

  // Incompatible if image has more features than the environment, irrespective
  // of type or sign of features.
  if (ImgMap.size() > EnvMap.size()) {
    DP("Incompatible: Image has more features than the environment \t[Image: "
       "%s]\t:\t[Environment: %s]\n",
       ImgTID.data(), EnvTID.data());
    return false;
  }

  // Compatible if each target feature specified by the environment is
  // compatible with target feature of the image. The target feature is
  // compatible if the iamge does not specify it (meaning Any), or if it
  // specifies it with the same value (meaning On or Off).
  for (const auto &ImgFeature : ImgMap) {
    auto EnvFeature = EnvMap.find(ImgFeature.first());
    if (EnvFeature == EnvMap.end()) {
      DP("Incompatible: Value of Image's non-ANY feature is not matching with "
         "the Environment feature's ANY value \t[Image: %s]\t:\t[Environment: "
         "%s]\n",
         ImgTID.data(), EnvTID.data());
      return false;
    } else if (EnvFeature->first() == ImgFeature.first() &&
               EnvFeature->second != ImgFeature.second) {
      DP("Incompatible: Value of Image's non-ANY feature is not matching with "
         "the Environment feature's non-ANY value \t[Image: "
         "%s]\t:\t[Environment: %s]\n",
         ImgTID.data(), EnvTID.data());
      return false;
    }
  }

  // Image is compatible if all features of Environment are:
  //   - either, present in the Image's features map with the same sign,
  //   - or, the feature is missing from Image's features map i.e. it is
  //   set to ANY
  DP("Compatible: Target IDs are compatible \t[Image: %s]\t:\t[Environment: "
     "%s]\n",
     ImgTID.data(), EnvTID.data());
  return true;
}

extern "C" {

int32_t __tgt_rtl_init_plugin() {
  DeviceInfoState = new RTLDeviceInfoTy;
  return (DeviceInfoState && DeviceInfoState->ConstructionSucceeded)
             ? OFFLOAD_SUCCESS
             : OFFLOAD_FAIL;
}

int32_t __tgt_rtl_deinit_plugin() {
  if (DeviceInfoState)
    delete DeviceInfoState;
  return OFFLOAD_SUCCESS;
}

int32_t __tgt_rtl_is_valid_binary(__tgt_device_image *Image) {
  return elfMachineIdIsAmdgcn(Image);
}

int32_t __tgt_rtl_is_valid_binary_info(__tgt_device_image *image,
                                       __tgt_image_info *info) {
  if (!__tgt_rtl_is_valid_binary(image))
    return false;

  // A subarchitecture was not specified. Assume it is compatible.
  if (!info->Arch)
    return true;

  int32_t NumberOfDevices = __tgt_rtl_number_of_devices();

  for (int32_t DeviceId = 0; DeviceId < NumberOfDevices; ++DeviceId) {
    __tgt_rtl_init_device(DeviceId);
    hsa_agent_t agent = DeviceInfo().HSAAgents[DeviceId];
    hsa_status_t err = hsa_agent_iterate_isas(agent, GetIsaInfo, &DeviceId);
    if (err != HSA_STATUS_SUCCESS) {
      DP("Error iterating ISAs\n");
      return false;
    }
    if (!IsImageCompatibleWithEnv(info->Arch, DeviceInfo().TargetID[DeviceId]))
      return false;
  }
  DP("Image has Target ID compatible with the current environment: %s\n",
     info->Arch);
  return true;
}

int __tgt_rtl_number_of_devices() {
  // If the construction failed, no methods are safe to call
  if (DeviceInfo().ConstructionSucceeded) {
    return DeviceInfo().NumberOfDevices;
  }
  DP("AMDGPU plugin construction failed. Zero devices available\n");
  return 0;
}

int64_t __tgt_rtl_init_requires(int64_t RequiresFlags) {
  DP("Init requires flags to %ld\n", RequiresFlags);
  DeviceInfo().RequiresFlags = RequiresFlags;
  return RequiresFlags;
}

int32_t __tgt_rtl_init_device(int DeviceId) {
  hsa_status_t Err = hsa_init();
  if (Err != HSA_STATUS_SUCCESS) {
    DP("HSA Initialization Failed.\n");
    return HSA_STATUS_ERROR;
  }
  // this is per device id init
  DP("Initialize the device id: %d\n", DeviceId);

  hsa_agent_t Agent = DeviceInfo().HSAAgents[DeviceId];

  // Get number of Compute Unit
  uint32_t ComputeUnits = 0;
  Err = hsa_agent_get_info(
      Agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT,
      &ComputeUnits);
  if (Err != HSA_STATUS_SUCCESS) {
    DeviceInfo().ComputeUnits[DeviceId] = 1;
    DP("Error getting compute units : settiing to 1\n");
  } else {
    DeviceInfo().ComputeUnits[DeviceId] = ComputeUnits;
    DP("Using %d compute unis per grid\n", DeviceInfo().ComputeUnits[DeviceId]);
  }

  char GetInfoName[64]; // 64 max size returned by get info
  Err = hsa_agent_get_info(Agent, (hsa_agent_info_t)HSA_AGENT_INFO_NAME,
                           (void *)GetInfoName);
  if (Err)
    DeviceInfo().GPUName[DeviceId] = "--unknown gpu--";
  else {
    DeviceInfo().GPUName[DeviceId] = GetInfoName;
  }

  if (print_kernel_trace & STARTUP_DETAILS)
    DP("Device#%-2d CU's: %2d %s\n", DeviceId,
       DeviceInfo().ComputeUnits[DeviceId],
       DeviceInfo().GPUName[DeviceId].c_str());

  // Query attributes to determine number of threads/block and blocks/grid.
  uint16_t WorkgroupMaxDim[3];
  Err = hsa_agent_get_info(Agent, HSA_AGENT_INFO_WORKGROUP_MAX_DIM,
                           &WorkgroupMaxDim);
  if (Err != HSA_STATUS_SUCCESS) {
    DeviceInfo().GroupsPerDevice[DeviceId] = RTLDeviceInfoTy::DefaultNumTeams;
    DP("Error getting grid dims: num groups : %d\n",
       RTLDeviceInfoTy::DefaultNumTeams);
  } else if (WorkgroupMaxDim[0] <= RTLDeviceInfoTy::HardTeamLimit) {
    DeviceInfo().GroupsPerDevice[DeviceId] = WorkgroupMaxDim[0];
    DP("Using %d ROCm blocks per grid\n",
       DeviceInfo().GroupsPerDevice[DeviceId]);
  } else {
    DeviceInfo().GroupsPerDevice[DeviceId] = RTLDeviceInfoTy::HardTeamLimit;
    DP("Max ROCm blocks per grid %d exceeds the hard team limit %d, capping "
       "at the hard limit\n",
       WorkgroupMaxDim[0], RTLDeviceInfoTy::HardTeamLimit);
  }

  // Get thread limit
  hsa_dim3_t GridMaxDim;
  Err = hsa_agent_get_info(Agent, HSA_AGENT_INFO_GRID_MAX_DIM, &GridMaxDim);
  if (Err == HSA_STATUS_SUCCESS) {
    DeviceInfo().ThreadsPerGroup[DeviceId] =
        reinterpret_cast<uint32_t *>(&GridMaxDim)[0] /
        DeviceInfo().GroupsPerDevice[DeviceId];

    if (DeviceInfo().ThreadsPerGroup[DeviceId] == 0) {
      DeviceInfo().ThreadsPerGroup[DeviceId] = RTLDeviceInfoTy::MaxWgSize;
      DP("Default thread limit: %d\n", RTLDeviceInfoTy::MaxWgSize);
    } else if (enforceUpperBound(&DeviceInfo().ThreadsPerGroup[DeviceId],
                                 RTLDeviceInfoTy::MaxWgSize)) {
      DP("Capped thread limit: %d\n", RTLDeviceInfoTy::MaxWgSize);
    } else {
      DP("Using ROCm Queried thread limit: %d\n",
         DeviceInfo().ThreadsPerGroup[DeviceId]);
    }
  } else {
    DeviceInfo().ThreadsPerGroup[DeviceId] = RTLDeviceInfoTy::MaxWgSize;
    DP("Error getting max block dimension, use default:%d \n",
       RTLDeviceInfoTy::MaxWgSize);
  }

  // Get wavefront size
  uint32_t WavefrontSize = 0;
  Err =
      hsa_agent_get_info(Agent, HSA_AGENT_INFO_WAVEFRONT_SIZE, &WavefrontSize);
  if (Err == HSA_STATUS_SUCCESS) {
    DP("Queried wavefront size: %d\n", WavefrontSize);
    DeviceInfo().WarpSize[DeviceId] = WavefrontSize;
  } else {
    // TODO: Burn the wavefront size into the code object
    DP("Warning: Unknown wavefront size, assuming 64\n");
    DeviceInfo().WarpSize[DeviceId] = 64;
  }

  // Adjust teams to the env variables

  if (DeviceInfo().Env.TeamLimit > 0 &&
      (enforceUpperBound(&DeviceInfo().GroupsPerDevice[DeviceId],
                         DeviceInfo().Env.TeamLimit))) {
    DP("Capping max groups per device to OMP_TEAM_LIMIT=%d\n",
       DeviceInfo().Env.TeamLimit);
  }

  // Set default number of teams
  if (DeviceInfo().Env.NumTeams > 0) {
    DeviceInfo().NumTeams[DeviceId] = DeviceInfo().Env.NumTeams;
    DP("Default number of teams set according to environment %d\n",
       DeviceInfo().Env.NumTeams);
  } else {
    char *TeamsPerCUEnvStr = getenv("OMP_TARGET_TEAMS_PER_PROC");
    int TeamsPerCU = DefaultTeamsPerCU;
    if (TeamsPerCUEnvStr) {
      TeamsPerCU = std::stoi(TeamsPerCUEnvStr);
    }

    DeviceInfo().NumTeams[DeviceId] =
        TeamsPerCU * DeviceInfo().ComputeUnits[DeviceId];
    DP("Default number of teams = %d * number of compute units %d\n",
       TeamsPerCU, DeviceInfo().ComputeUnits[DeviceId]);
  }

  if (enforceUpperBound(&DeviceInfo().NumTeams[DeviceId],
                        DeviceInfo().GroupsPerDevice[DeviceId])) {
    DP("Default number of teams exceeds device limit, capping at %d\n",
       DeviceInfo().GroupsPerDevice[DeviceId]);
  }

  // Adjust threads to the env variables
  if (DeviceInfo().Env.TeamThreadLimit > 0 &&
      (enforceUpperBound(&DeviceInfo().NumThreads[DeviceId],
                         DeviceInfo().Env.TeamThreadLimit))) {
    DP("Capping max number of threads to OMP_TEAMS_THREAD_LIMIT=%d\n",
       DeviceInfo().Env.TeamThreadLimit);
  }

  // Set default number of threads
  DeviceInfo().NumThreads[DeviceId] = RTLDeviceInfoTy::DefaultWgSize;
  DP("Default number of threads set according to library's default %d\n",
     RTLDeviceInfoTy::DefaultWgSize);
  if (enforceUpperBound(&DeviceInfo().NumThreads[DeviceId],
                        DeviceInfo().ThreadsPerGroup[DeviceId])) {
    DP("Default number of threads exceeds device limit, capping at %d\n",
       DeviceInfo().ThreadsPerGroup[DeviceId]);
  }

  DP("Device %d: default limit for groupsPerDevice %d & threadsPerGroup %d\n",
     DeviceId, DeviceInfo().GroupsPerDevice[DeviceId],
     DeviceInfo().ThreadsPerGroup[DeviceId]);

  DP("Device %d: wavefront size %d, total threads %d x %d = %d\n", DeviceId,
     DeviceInfo().WarpSize[DeviceId], DeviceInfo().ThreadsPerGroup[DeviceId],
     DeviceInfo().GroupsPerDevice[DeviceId],
     DeviceInfo().GroupsPerDevice[DeviceId] *
         DeviceInfo().ThreadsPerGroup[DeviceId]);

  return OFFLOAD_SUCCESS;
}

static __tgt_target_table *
__tgt_rtl_load_binary_locked(int32_t DeviceId, __tgt_device_image *Image);

__tgt_target_table *__tgt_rtl_load_binary(int32_t DeviceId,
                                          __tgt_device_image *Image) {
  DeviceInfo().LoadRunLock.lock();
  __tgt_target_table *Res = __tgt_rtl_load_binary_locked(DeviceId, Image);
  DeviceInfo().LoadRunLock.unlock();
  return Res;
}

__tgt_target_table *__tgt_rtl_load_binary_locked(int32_t DeviceId,
                                                 __tgt_device_image *Image) {
  // This function loads the device image onto gpu[DeviceId] and does other
  // per-image initialization work. Specifically:
  //
  // - Initialize an DeviceEnvironmentTy instance embedded in the
  //   image at the symbol "omptarget_device_environment"
  //   Fields DebugKind, DeviceNum, NumDevices. Used by the deviceRTL.
  //
  // - Allocate a large array per-gpu (could be moved to init_device)
  //   - Read a uint64_t at symbol omptarget_nvptx_device_State_size
  //   - Allocate at least that many bytes of gpu memory
  //   - Zero initialize it
  //   - Write the pointer to the symbol omptarget_nvptx_device_State
  //
  // - Pulls some per-kernel information together from various sources and
  //   records it in the KernelsList for quicker access later
  //
  // The initialization can be done before or after loading the image onto the
  // gpu. This function presently does a mixture. Using the hsa api to get/set
  // the information is simpler to implement, in exchange for more complicated
  // runtime behaviour. E.g. launching a kernel or using dma to get eight bytes
  // back from the gpu vs a hashtable lookup on the host.

  const size_t ImgSize = (char *)Image->ImageEnd - (char *)Image->ImageStart;

  DeviceInfo().clearOffloadEntriesTable(DeviceId);

  // We do not need to set the ELF version because the caller of this function
  // had to do that to decide the right runtime to use

  if (!elfMachineIdIsAmdgcn(Image))
    return NULL;

  {
    auto Env =
        DeviceEnvironment(DeviceId, DeviceInfo().NumberOfDevices,
                          DeviceInfo().Env.DynamicMemSize, Image, ImgSize);

    auto &KernelInfo = DeviceInfo().KernelInfoTable[DeviceId];
    auto &SymbolInfo = DeviceInfo().SymbolInfoTable[DeviceId];
    hsa_status_t Err = moduleRegisterFromMemoryToPlace(
        KernelInfo, SymbolInfo, (void *)Image->ImageStart, ImgSize, DeviceId,
        [&](void *Data, size_t Size) {
          if (imageContainsSymbol(Data, Size, "needs_hostcall_buffer")) {
            __atomic_store_n(&DeviceInfo().HostcallRequired, true,
                             __ATOMIC_RELEASE);
          }
          return Env.beforeLoading(Data, Size);
        },
        DeviceInfo().HSAExecutables);

    check("Module registering", Err);
    if (Err != HSA_STATUS_SUCCESS) {
      const char *DeviceName = DeviceInfo().GPUName[DeviceId].c_str();
      const char *ElfName = get_elf_mach_gfx_name(elfEFlags(Image));

      if (strcmp(DeviceName, ElfName) != 0) {
        DP("Possible gpu arch mismatch: device:%s, image:%s please check"
           " compiler flag: -march=<gpu>\n",
           DeviceName, ElfName);
      } else {
        DP("Error loading image onto GPU: %s\n", get_error_string(Err));
      }

      return NULL;
    }

    Err = Env.afterLoading();
    if (Err != HSA_STATUS_SUCCESS) {
      return NULL;
    }
  }

  DP("AMDGPU module successfully loaded!\n");

  {
    // the device_State array is either large value in bss or a void* that
    // needs to be assigned to a pointer to an array of size device_state_bytes
    // If absent, it has been deadstripped and needs no setup.

    void *StatePtr;
    uint32_t StatePtrSize;
    auto &SymbolInfoMap = DeviceInfo().SymbolInfoTable[DeviceId];
    hsa_status_t Err = interop_hsa_get_symbol_info(
        SymbolInfoMap, DeviceId, "omptarget_nvptx_device_State", &StatePtr,
        &StatePtrSize);

    if (Err != HSA_STATUS_SUCCESS) {
      DP("No device_state symbol found, skipping initialization\n");
    } else {
      if (StatePtrSize < sizeof(void *)) {
        DP("unexpected size of state_ptr %u != %zu\n", StatePtrSize,
           sizeof(void *));
        return NULL;
      }

      // if it's larger than a void*, assume it's a bss array and no further
      // initialization is required. Only try to set up a pointer for
      // sizeof(void*)
      if (StatePtrSize == sizeof(void *)) {
        uint64_t DeviceStateBytes =
            getDeviceStateBytes((char *)Image->ImageStart, ImgSize);
        if (DeviceStateBytes == 0) {
          DP("Can't initialize device_State, missing size information\n");
          return NULL;
        }

        auto &DSS = DeviceInfo().DeviceStateStore[DeviceId];
        if (DSS.first.get() == nullptr) {
          assert(DSS.second == 0);
          void *Ptr = NULL;
          hsa_status_t Err = implCalloc(&Ptr, DeviceStateBytes, DeviceId);
          if (Err != HSA_STATUS_SUCCESS) {
            DP("Failed to allocate device_state array\n");
            return NULL;
          }
          DSS = {
              std::unique_ptr<void, RTLDeviceInfoTy::ImplFreePtrDeletor>{Ptr},
              DeviceStateBytes,
          };
        }

        void *Ptr = DSS.first.get();
        if (DeviceStateBytes != DSS.second) {
          DP("Inconsistent sizes of device_State unsupported\n");
          return NULL;
        }

        // write ptr to device memory so it can be used by later kernels
        Err = DeviceInfo().freesignalpoolMemcpyH2D(StatePtr, &Ptr,
                                                   sizeof(void *), DeviceId);
        if (Err != HSA_STATUS_SUCCESS) {
          DP("memcpy install of state_ptr failed\n");
          return NULL;
        }
      }
    }
  }

  // Here, we take advantage of the data that is appended after img_end to get
  // the symbols' name we need to load. This data consist of the host entries
  // begin and end as well as the target name (see the offloading linker script
  // creation in clang compiler).

  // Find the symbols in the module by name. The name can be obtain by
  // concatenating the host entry name with the target name

  __tgt_offload_entry *HostBegin = Image->EntriesBegin;
  __tgt_offload_entry *HostEnd = Image->EntriesEnd;

  for (__tgt_offload_entry *E = HostBegin; E != HostEnd; ++E) {

    if (!E->addr) {
      // The host should have always something in the address to
      // uniquely identify the target region.
      DP("Analyzing host entry '<null>' (size = %lld)...\n",
         (unsigned long long)E->size);
      return NULL;
    }

    if (E->size) {
      __tgt_offload_entry Entry = *E;

      void *Varptr;
      uint32_t Varsize;

      auto &SymbolInfoMap = DeviceInfo().SymbolInfoTable[DeviceId];
      hsa_status_t Err = interop_hsa_get_symbol_info(
          SymbolInfoMap, DeviceId, E->name, &Varptr, &Varsize);

      if (Err != HSA_STATUS_SUCCESS) {
        // Inform the user what symbol prevented offloading
        DP("Loading global '%s' (Failed)\n", E->name);
        return NULL;
      }

      if (Varsize != E->size) {
        DP("Loading global '%s' - size mismatch (%u != %lu)\n", E->name,
           Varsize, E->size);
        return NULL;
      }

      DP("Entry point " DPxMOD " maps to global %s (" DPxMOD ")\n",
         DPxPTR(E - HostBegin), E->name, DPxPTR(Varptr));
      Entry.addr = (void *)Varptr;

      DeviceInfo().addOffloadEntry(DeviceId, Entry);

      if (DeviceInfo().RequiresFlags & OMP_REQ_UNIFIED_SHARED_MEMORY &&
          E->flags & OMP_DECLARE_TARGET_LINK) {
        // If unified memory is present any target link variables
        // can access host addresses directly. There is no longer a
        // need for device copies.
        Err = DeviceInfo().freesignalpoolMemcpyH2D(Varptr, E->addr,
                                                   sizeof(void *), DeviceId);
        if (Err != HSA_STATUS_SUCCESS)
          DP("Error when copying USM\n");
        DP("Copy linked variable host address (" DPxMOD ")"
           "to device address (" DPxMOD ")\n",
           DPxPTR(*((void **)E->addr)), DPxPTR(Varptr));
      }

      continue;
    }

    DP("to find the kernel name: %s size: %lu\n", E->name, strlen(E->name));

    // errors in kernarg_segment_size previously treated as = 0 (or as undef)
    uint32_t KernargSegmentSize = 0;
    auto &KernelInfoMap = DeviceInfo().KernelInfoTable[DeviceId];
    hsa_status_t Err = HSA_STATUS_SUCCESS;
    if (!E->name) {
      Err = HSA_STATUS_ERROR;
    } else {
      std::string KernelStr = std::string(E->name);
      auto It = KernelInfoMap.find(KernelStr);
      if (It != KernelInfoMap.end()) {
        atl_kernel_info_t Info = It->second;
        KernargSegmentSize = Info.kernel_segment_size;
      } else {
        Err = HSA_STATUS_ERROR;
      }
    }

    // default value GENERIC (in case symbol is missing from cubin file)
    llvm::omp::OMPTgtExecModeFlags ExecModeVal =
        llvm::omp::OMPTgtExecModeFlags::OMP_TGT_EXEC_MODE_GENERIC;

    // get flat group size if present, else Default_WG_Size
    int16_t WGSizeVal = RTLDeviceInfoTy::DefaultWgSize;

    // get Kernel Descriptor if present.
    // Keep struct in sync wih getTgtAttributeStructQTy in CGOpenMPRuntime.cpp
    struct KernDescValType {
      uint16_t Version;
      uint16_t TSize;
      uint16_t WGSize;
    };
    struct KernDescValType KernDescVal;
    std::string KernDescNameStr(E->name);
    KernDescNameStr += "_kern_desc";
    const char *KernDescName = KernDescNameStr.c_str();

    void *KernDescPtr;
    uint32_t KernDescSize;
    void *CallStackAddr = nullptr;
    Err = interopGetSymbolInfo((char *)Image->ImageStart, ImgSize, KernDescName,
                               &KernDescPtr, &KernDescSize);

    if (Err == HSA_STATUS_SUCCESS) {
      if ((size_t)KernDescSize != sizeof(KernDescVal))
        DP("Loading global computation properties '%s' - size mismatch (%u != "
           "%lu)\n",
           KernDescName, KernDescSize, sizeof(KernDescVal));

      memcpy(&KernDescVal, KernDescPtr, (size_t)KernDescSize);

      // Check structure size against recorded size.
      if ((size_t)KernDescSize != KernDescVal.TSize)
        DP("KernDescVal size %lu does not match advertized size %d for '%s'\n",
           sizeof(KernDescVal), KernDescVal.TSize, KernDescName);

      DP("After loading global for %s KernDesc \n", KernDescName);
      DP("KernDesc: Version: %d\n", KernDescVal.Version);
      DP("KernDesc: TSize: %d\n", KernDescVal.TSize);
      DP("KernDesc: WG_Size: %d\n", KernDescVal.WGSize);

      if (KernDescVal.WGSize == 0) {
        KernDescVal.WGSize = RTLDeviceInfoTy::DefaultWgSize;
        DP("Setting KernDescVal.WG_Size to default %d\n", KernDescVal.WGSize);
      }
      WGSizeVal = KernDescVal.WGSize;
      DP("WGSizeVal %d\n", WGSizeVal);
      check("Loading KernDesc computation property", Err);
    } else {
      DP("Warning: Loading KernDesc '%s' - symbol not found, ", KernDescName);

      // Flat group size
      std::string WGSizeNameStr(E->name);
      WGSizeNameStr += "_wg_size";
      const char *WGSizeName = WGSizeNameStr.c_str();

      void *WGSizePtr;
      uint32_t WGSize;
      Err = interopGetSymbolInfo((char *)Image->ImageStart, ImgSize, WGSizeName,
                                 &WGSizePtr, &WGSize);

      if (Err == HSA_STATUS_SUCCESS) {
        if ((size_t)WGSize != sizeof(int16_t)) {
          DP("Loading global computation properties '%s' - size mismatch (%u "
             "!= "
             "%lu)\n",
             WGSizeName, WGSize, sizeof(int16_t));
          return NULL;
        }

        memcpy(&WGSizeVal, WGSizePtr, (size_t)WGSize);

        DP("After loading global for %s WGSize = %d\n", WGSizeName, WGSizeVal);

        if (WGSizeVal < RTLDeviceInfoTy::DefaultWgSize ||
            WGSizeVal > RTLDeviceInfoTy::MaxWgSize) {
          DP("Error wrong WGSize value specified in HSA code object file: "
             "%d\n",
             WGSizeVal);
          WGSizeVal = RTLDeviceInfoTy::DefaultWgSize;
        }
      } else {
        DP("Warning: Loading WGSize '%s' - symbol not found, "
           "using default value %d\n",
           WGSizeName, WGSizeVal);
      }

      check("Loading WGSize computation property", Err);
    }

    // Read execution mode from global in binary
    std::string ExecModeNameStr(E->name);
    ExecModeNameStr += "_exec_mode";
    const char *ExecModeName = ExecModeNameStr.c_str();

    void *ExecModePtr;
    uint32_t VarSize;
    Err = interopGetSymbolInfo((char *)Image->ImageStart, ImgSize, ExecModeName,
                               &ExecModePtr, &VarSize);

    if (Err == HSA_STATUS_SUCCESS) {
      if ((size_t)VarSize != sizeof(llvm::omp::OMPTgtExecModeFlags)) {
        DP("Loading global computation properties '%s' - size mismatch(%u != "
           "%lu)\n",
           ExecModeName, VarSize, sizeof(llvm::omp::OMPTgtExecModeFlags));
        return NULL;
      }

      memcpy(&ExecModeVal, ExecModePtr, (size_t)VarSize);

      DP("After loading global for %s ExecMode = %d\n", ExecModeName,
         ExecModeVal);

      if (ExecModeVal < 0 ||
          ExecModeVal > llvm::omp::OMP_TGT_EXEC_MODE_GENERIC_SPMD) {
        DP("Error wrong exec_mode value specified in HSA code object file: "
           "%d\n",
           ExecModeVal);
        return NULL;
      }
    } else {
      DP("Loading global exec_mode '%s' - symbol missing, using default "
         "value "
         "GENERIC (1)\n",
         ExecModeName);
    }
    check("Loading computation property", Err);

    DeviceInfo().KernelsList.push_back(
        KernelTy(ExecModeVal, WGSizeVal, DeviceId, CallStackAddr, E->name,
                 KernargSegmentSize, DeviceInfo().KernArgPool,
                 DeviceInfo().KernelArgPoolMap));
    __tgt_offload_entry Entry = *E;
    Entry.addr = (void *)&DeviceInfo().KernelsList.back();
    DeviceInfo().addOffloadEntry(DeviceId, Entry);
    DP("Entry point %ld maps to %s\n", E - HostBegin, E->name);
  }

  return DeviceInfo().getOffloadEntriesTable(DeviceId);
}

void *__tgt_rtl_data_alloc(int DeviceId, int64_t Size, void *, int32_t Kind) {
  void *Ptr = NULL;
  assert(DeviceId < DeviceInfo().NumberOfDevices && "Device ID too large");

  hsa_amd_memory_pool_t MemoryPool;
  switch (Kind) {
  case TARGET_ALLOC_DEFAULT:
    // GPU memory
    MemoryPool = DeviceInfo().getDeviceMemoryPool(DeviceId);
    break;
  case TARGET_ALLOC_HOST:
    // non-migratable memory accessible by host and device(s)
    MemoryPool = DeviceInfo().getHostMemoryPool();
    break;
  default:
    REPORT("Invalid target data allocation kind or requested allocator not "
           "implemented yet\n");
    return NULL;
  }

  hsa_status_t Err = hsa_amd_memory_pool_allocate(MemoryPool, Size, 0, &Ptr);
  DP("Tgt alloc data %ld bytes, (tgt:%016llx).\n", Size,
     (long long unsigned)(Elf64_Addr)Ptr);
  Ptr = (Err == HSA_STATUS_SUCCESS) ? Ptr : NULL;
  return Ptr;
}

int32_t __tgt_rtl_data_submit(int DeviceId, void *TgtPtr, void *HstPtr,
                              int64_t Size) {
  assert(DeviceId < DeviceInfo().NumberOfDevices && "Device ID too large");
  __tgt_async_info AsyncInfo;
  int32_t Rc = dataSubmit(DeviceId, TgtPtr, HstPtr, Size, &AsyncInfo);
  if (Rc != OFFLOAD_SUCCESS)
    return OFFLOAD_FAIL;

  return __tgt_rtl_synchronize(DeviceId, &AsyncInfo);
}

int32_t __tgt_rtl_data_submit_async(int DeviceId, void *TgtPtr, void *HstPtr,
                                    int64_t Size, __tgt_async_info *AsyncInfo) {
  assert(DeviceId < DeviceInfo().NumberOfDevices && "Device ID too large");
  if (AsyncInfo) {
    initAsyncInfo(AsyncInfo);
    return dataSubmit(DeviceId, TgtPtr, HstPtr, Size, AsyncInfo);
  }
  return __tgt_rtl_data_submit(DeviceId, TgtPtr, HstPtr, Size);
}

int32_t __tgt_rtl_data_retrieve(int DeviceId, void *HstPtr, void *TgtPtr,
                                int64_t Size) {
  assert(DeviceId < DeviceInfo().NumberOfDevices && "Device ID too large");
  __tgt_async_info AsyncInfo;
  int32_t Rc = dataRetrieve(DeviceId, HstPtr, TgtPtr, Size, &AsyncInfo);
  if (Rc != OFFLOAD_SUCCESS)
    return OFFLOAD_FAIL;

  return __tgt_rtl_synchronize(DeviceId, &AsyncInfo);
}

int32_t __tgt_rtl_data_retrieve_async(int DeviceId, void *HstPtr, void *TgtPtr,
                                      int64_t Size,
                                      __tgt_async_info *AsyncInfo) {
  assert(AsyncInfo && "AsyncInfo is nullptr");
  assert(DeviceId < DeviceInfo().NumberOfDevices && "Device ID too large");
  initAsyncInfo(AsyncInfo);
  return dataRetrieve(DeviceId, HstPtr, TgtPtr, Size, AsyncInfo);
}

int32_t __tgt_rtl_data_delete(int DeviceId, void *TgtPtr) {
  assert(DeviceId < DeviceInfo().NumberOfDevices && "Device ID too large");
  // HSA can free pointers allocated from different types of memory pool.
  hsa_status_t Err;
  DP("Tgt free data (tgt:%016llx).\n", (long long unsigned)(Elf64_Addr)TgtPtr);
  Err = core::Runtime::Memfree(TgtPtr);
  if (Err != HSA_STATUS_SUCCESS) {
    DP("Error when freeing CUDA memory\n");
    return OFFLOAD_FAIL;
  }
  return OFFLOAD_SUCCESS;
}

int32_t __tgt_rtl_run_target_team_region(int32_t DeviceId, void *TgtEntryPtr,
                                         void **TgtArgs, ptrdiff_t *TgtOffsets,
                                         int32_t ArgNum, int32_t NumTeams,
                                         int32_t ThreadLimit,
                                         uint64_t LoopTripcount) {

  DeviceInfo().LoadRunLock.lock_shared();
  int32_t Res = runRegionLocked(DeviceId, TgtEntryPtr, TgtArgs, TgtOffsets,
                                ArgNum, NumTeams, ThreadLimit, LoopTripcount);

  DeviceInfo().LoadRunLock.unlock_shared();
  return Res;
}

int32_t __tgt_rtl_run_target_region(int32_t DeviceId, void *TgtEntryPtr,
                                    void **TgtArgs, ptrdiff_t *TgtOffsets,
                                    int32_t ArgNum) {
  // use one team and one thread
  // fix thread num
  int32_t TeamNum = 1;
  int32_t ThreadLimit = 0; // use default
  return __tgt_rtl_run_target_team_region(DeviceId, TgtEntryPtr, TgtArgs,
                                          TgtOffsets, ArgNum, TeamNum,
                                          ThreadLimit, 0);
}

int32_t __tgt_rtl_run_target_team_region_async(
    int32_t DeviceId, void *TgtEntryPtr, void **TgtArgs, ptrdiff_t *TgtOffsets,
    int32_t ArgNum, int32_t NumTeams, int32_t ThreadLimit,
    uint64_t LoopTripcount, __tgt_async_info *AsyncInfo) {
  assert(AsyncInfo && "AsyncInfo is nullptr");
  initAsyncInfo(AsyncInfo);

  DeviceInfo().LoadRunLock.lock_shared();
  int32_t Res = runRegionLocked(DeviceId, TgtEntryPtr, TgtArgs, TgtOffsets,
                                ArgNum, NumTeams, ThreadLimit, LoopTripcount);

  DeviceInfo().LoadRunLock.unlock_shared();
  return Res;
}

int32_t __tgt_rtl_run_target_region_async(int32_t DeviceId, void *TgtEntryPtr,
                                          void **TgtArgs, ptrdiff_t *TgtOffsets,
                                          int32_t ArgNum,
                                          __tgt_async_info *AsyncInfo) {
  // use one team and one thread
  // fix thread num
  int32_t TeamNum = 1;
  int32_t ThreadLimit = 0; // use default
  return __tgt_rtl_run_target_team_region_async(DeviceId, TgtEntryPtr, TgtArgs,
                                                TgtOffsets, ArgNum, TeamNum,
                                                ThreadLimit, 0, AsyncInfo);
}

int32_t __tgt_rtl_synchronize(int32_t DeviceId, __tgt_async_info *AsyncInfo) {
  assert(AsyncInfo && "AsyncInfo is nullptr");

  // Cuda asserts that AsyncInfo->Queue is non-null, but this invariant
  // is not ensured by devices.cpp for amdgcn
  // assert(AsyncInfo->Queue && "AsyncInfo->Queue is nullptr");
  if (AsyncInfo->Queue) {
    finiAsyncInfo(AsyncInfo);
  }
  return OFFLOAD_SUCCESS;
}

void __tgt_rtl_print_device_info(int32_t DeviceId) {
  // TODO: Assertion to see if DeviceId is correct
  // NOTE: We don't need to set context for print device info.

  DeviceInfo().printDeviceInfo(DeviceId, DeviceInfo().HSAAgents[DeviceId]);
}

} // extern "C"
