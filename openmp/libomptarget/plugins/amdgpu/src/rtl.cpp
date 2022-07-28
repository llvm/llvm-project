//===--- amdgpu/src/rtl.cpp --------------------------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
// Notified per clause 4(b) of the license.
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
#include <dlfcn.h>
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
#include "small_pool.h"

#include "DeviceEnvironment.h"
#include "get_elf_mach_gfx_name.h"
#include "memtype.h"
#include "omptargetplugin.h"
#include "print_tracing.h"
#include "trace.h"

#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Frontend/OpenMP/OMPGridValues.h"

#include "MemoryManager.h"

#include "utils.h"

#ifdef OMPT_SUPPORT
#include <ompt_device_callbacks.h>
#define OMPT_IF_ENABLED(stmts)                                                 \
  do {                                                                         \
    if (ompt_device_callbacks.is_enabled()) {                                  \
      stmts                                                                    \
    }                                                                          \
  } while (0)
#define OMPT_IF_TRACING_ENABLED(stmts)                                         \
  do {                                                                         \
    if (ompt_device_callbacks.is_tracing_enabled()) {                          \
      stmts                                                                    \
    }                                                                          \
  } while (0)
#else
#define OMPT_IF_ENABLED(stmts)
#define OMPT_IF_TRACING_ENABLED(stmts)
#endif

/// Libomptarget function that will be used to set num_teams in trace records.
typedef void (*libomptarget_ompt_set_granted_teams_t)(uint32_t);
libomptarget_ompt_set_granted_teams_t ompt_set_granted_teams_fn = nullptr;
std::mutex granted_teams_mtx;

/// Libomptarget function that will be used to set timestamps in trace records.
typedef void (*libomptarget_ompt_set_timestamp_t)(uint64_t start, uint64_t end);
libomptarget_ompt_set_timestamp_t ompt_set_timestamp_fn = nullptr;
std::mutex ompt_set_timestamp_mtx;

SmallPoolMgrTy SmallPoolMgr;

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

// Data structure used to keep track of coarse grain memory regions
AMDGPUMemTypeBitFieldTable *coarse_grain_mem_tab = nullptr;

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

typedef enum { INIT = 1, FINI } initORfini;

typedef struct DeviceImageTy {
  int size;
  bool initfini;
  DeviceImageTy() {
    size = 0;
    initfini = false;
  }
  DeviceImageTy(int s, bool init_fini) {
    size = s;
    initfini = init_fini;
  }
  ~DeviceImageTy() {}
} Image_t;

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

std::unordered_map<std::string /*kernel*/, std::unique_ptr<KernelArgPool>>
    KernelArgPoolMap;

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
           hsa_amd_memory_pool_t &KernArgMemoryPool)
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

/// List that contains all the kernels.
/// FIXME: we may need this to be per device and per library.
std::list<KernelTy> KernelsList;

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

void launchInitFiniKernel(int32_t, void *, const size_t &, const initORfini);

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
  // header |= 1 << 8; // set barrier bit?
  Header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
  Header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;
  return Header;
}

uint16_t create_BarrierAND_header() {
  uint16_t header = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
  header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
  header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;
  return header;
}

static uint64_t acquire_available_packet_id(hsa_queue_t *Queue) {
  uint64_t packet_id = hsa_queue_add_write_index_relaxed(Queue, 1);
  bool full = true;
  while (full) {
    full =
        packet_id >= (Queue->size + hsa_queue_load_read_index_scacquire(Queue));
  }
  return packet_id;
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

  /// Enable/disable queue profiling for OMPT trace records
  void enableQueueProfiling(int enable) {
    for (uint8_t i = 0; i < NUM_QUEUES_PER_DEVICE; ++i) {
      hsa_status_t err =
          hsa_amd_profiling_set_profiler_enabled(HSAQueues[i], enable);
      if (err != HSA_STATUS_SUCCESS)
        DP("Error enabling queue profiling\n");
    }
  }

  // Get the Number of Queues per Device
  int getNumQueues() { return NUM_QUEUES_PER_DEVICE; }

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

  std::map<void *, Image_t> ImageList;
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

  /// HSA system clock frequency. Modeled after ROCclr Timestamp functionality
  double TicksToTime = 0;

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
                                      hsa_agent_t, hsa_amd_memory_pool_t,
				      bool *UserLocked);
  hsa_status_t freesignalpoolMemcpy(void *Dest, void *Src, size_t Size,
                                    MemcpyFunc Func, int32_t DeviceId,
				    hsa_signal_t &S, bool &UserLocked) {
    hsa_agent_t Agent = HSAAgents[DeviceId];
    S = FreeSignalPool.pop();
    if (S.handle == 0) {
      return HSA_STATUS_ERROR;
    }
    hsa_status_t R = Func(S, Dest, Src, Size, Agent, HostFineGrainedMemoryPool, &UserLocked);
 // FIXME   FreeSignalPool.push(S);
    return R;
  }

  hsa_status_t freesignalpoolMemcpyD2H(void *Dest, void *Src, size_t Size,
                                       int32_t DeviceId, hsa_signal_t &Signal,
				       bool &UserLocked) {
    return freesignalpoolMemcpy(Dest, Src, Size, impl_memcpy_d2h, DeviceId,
		    Signal, UserLocked);
  }

  hsa_status_t freesignalpoolMemcpyH2D(void *Dest, void *Src, size_t Size,
                                       int32_t DeviceId,  hsa_signal_t &Signal,
				       bool &UserLocked) {
    return freesignalpoolMemcpy(Dest, Src, Size, impl_memcpy_h2d, DeviceId,
		    Signal, UserLocked);
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
    // KernelArgPoolMap.clear();
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

  // Enable/disable queue profiling for all devices, consumed by OMPT trace
  // records
  void enableQueueProfiling(int enable) {
    for (int i = 0; i < NumberOfDevices; ++i)
      HSAQueueSchedulers[i].enableQueueProfiling(enable);
  }

  /// Tracker of memory allocation types.
  // tgt_rtl_data_free is not passed memory type (host or device)
  // but it is saved in this data structure
  class AMDGPUDeviceAllocatorTy : public DeviceAllocatorTy {
    int DeviceId;
    std::unordered_map<void *, TargetAllocTy> HostAllocations;

  public:
    AMDGPUDeviceAllocatorTy(int DeviceId) : DeviceId(DeviceId) {}

    void *allocate(size_t size, void *, TargetAllocTy kind) override {
      if (size == 0)
        return nullptr;
      void *ptr = nullptr;
      switch (kind) {
      case TARGET_ALLOC_DEFAULT:
      case TARGET_ALLOC_DEVICE: {
        void *devPtr;
        hsa_status_t err = device_malloc(&devPtr, size, DeviceId);
        ptr = (err == HSA_STATUS_SUCCESS) ? devPtr : nullptr;
        if (!ptr)
          REPORT("Error allocating device memory");
        break;
      }
      case TARGET_ALLOC_HOST:
      case TARGET_ALLOC_SHARED:
        ptr = malloc(size);
        if (!ptr)
          REPORT("Error allocating host memory");
        HostAllocations[ptr] = kind;
        break;
      }

      return ptr;
    }

    int dev_free(void *ptr) override {
      TargetAllocTy kind = (HostAllocations.find(ptr) == HostAllocations.end())
                               ? TARGET_ALLOC_DEFAULT
                               : TARGET_ALLOC_HOST;
      hsa_status_t err = HSA_STATUS_SUCCESS;

      err = unlock_memory(ptr);
      if (err != HSA_STATUS_SUCCESS)
        DP("Error when unlocking memory\n");

      switch (kind) {
      case TARGET_ALLOC_DEFAULT:
      case TARGET_ALLOC_DEVICE: {
        hsa_status_t err;
        err = core::Runtime::Memfree(ptr);
        if (err != HSA_STATUS_SUCCESS) {
          DP("Error when freeing device memory\n");
          return OFFLOAD_FAIL;
        }
        break;
      }
      case TARGET_ALLOC_HOST:
      case TARGET_ALLOC_SHARED:
        free(ptr);
        break;
      }
      return OFFLOAD_SUCCESS;
    }
  };

  // One device allocator per device
  std::vector<AMDGPUDeviceAllocatorTy> DeviceAllocators;

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

    // Initialize system timestamp conversion factor, modeled after ROCclr
    uint64_t ticks_frequency;
    hsa_status_t freq_err = hsa_system_get_info(
        HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &ticks_frequency);
    if (freq_err == HSA_STATUS_SUCCESS)
      TicksToTime = (double)1e9 / double(ticks_frequency);
    else {
      DP("Error calling hsa_system_get_info for timestamp frequency: %s\n",
         get_error_string(freq_err));
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

#ifdef OMPT_SUPPORT
    // TODO ompt_device_callbacks.enabled is not yet set since
    // register_callbacks on the plugin instance is not yet
    // called. Hence, unconditionally prepare devices.
    ompt_device_callbacks.prepare_devices(NumberOfDevices);
#endif

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
      DP("Device %d: Initial groupsPerDevice %d & ThreadsPerGroup %d\n", I,
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

    for (int I = 0; I < NumberOfDevices; ++I)
      DeviceAllocators.emplace_back(I);
    ConstructionSucceeded = true;
  }

  ~RTLDeviceInfoTy() {
    DP("Finalizing the " GETNAME(TARGET_NAME) " DeviceInfo.\n");

    OMPT_IF_ENABLED(for (int i = 0; i < NumberOfDevices; i++) {
      ompt_device_callbacks.ompt_callback_device_finalize(i);
    });

    if (!HSAInitSuccess()) {
      // Then none of these can have been set up and they can't be torn down
      return;
    }

    // Cleanup by unlocking all the locked pointers from the small pools
    SmallPoolTy::PtrVecTy AllPtrs = SmallPoolMgr.getAllPoolPtrs();
    for (const auto &e : AllPtrs) {
      hsa_status_t err = HSA_STATUS_SUCCESS;
      assert(already_locked(e, &err, nullptr));

      DP("Calling hsa_amd_memory_unlock in RTLDeviceInfoTy dtor for PoolPtr "
         "%p\n",
         e);

      err = hsa_amd_memory_unlock(e);
      if (err != HSA_STATUS_SUCCESS)
        DP("PoolPtr memory_unlock returned %s, continuing\n",
           get_error_string(err));
    }

    for (int i = 0; i < NumberOfDevices; i++) {
      std::map<void *, Image_t>::iterator itr = ImageList.begin();
      if (itr != ImageList.end()) {
        void *img = itr->first;
        Image_t img_attr = (itr->second);
        core::launchInitFiniKernel(i, img, img_attr.size, FINI);
        itr++;
      }
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

// Putting accesses to DeviceInfo global behind a function call prior
// to changing to use init_plugin/deinit_plugin calls
static RTLDeviceInfoTy DeviceInfoState;
static RTLDeviceInfoTy& DeviceInfo() { return DeviceInfoState; }

/// Global function for enabling/disabling queue profiling, used for OMPT trace
/// records.
void ompt_enable_queue_profiling(int enable) {
  DeviceInfo().enableQueueProfiling(enable);
}

namespace {

/// Retrieve the libomptarget function for setting timestamps in trace records
static void ensureTimestampFn() {
  std::unique_lock<std::mutex> timestamp_fn_lck(ompt_set_timestamp_mtx);
  if (ompt_set_timestamp_fn)
    return;
  void *vptr = dlsym(NULL, "libomptarget_ompt_set_timestamp");
  assert(vptr && "OMPT set timestamp entry point not found");
  ompt_set_timestamp_fn =
      reinterpret_cast<libomptarget_ompt_set_timestamp_t>(vptr);
}

/// Get the HSA system timestamps for the input signal associated with an
/// async copy and pass the information to libomptarget
static void recordCopyTimingInNs(hsa_signal_t signal) {
  hsa_amd_profiling_async_copy_time_t time_rec;
  hsa_status_t err = hsa_amd_profiling_get_async_copy_time(signal, &time_rec);
  if (err != HSA_STATUS_SUCCESS) {
    DP("Getting profiling_async_copy_time returned %s, continuing\n",
       get_error_string(err));
    return;
  }
  // Retrieve the libomptarget function pointer if required
  ensureTimestampFn();
  // No need to hold a lock
  // Factor in the frequency
  ompt_set_timestamp_fn(time_rec.start * DeviceInfo().TicksToTime,
                        time_rec.end * DeviceInfo().TicksToTime);
}

/// Get the HSA system timestamps for the input agent and signal associated
/// with a kernel dispatch and pass the information to libomptarget
static void recordKernelTimingInNs(hsa_signal_t signal, hsa_agent_t agent) {
  hsa_amd_profiling_dispatch_time_t time_rec;
  hsa_status_t err =
      hsa_amd_profiling_get_dispatch_time(agent, signal, &time_rec);
  if (err != HSA_STATUS_SUCCESS) {
    DP("Getting profiling_dispatch_time returned %s, continuing\n",
       get_error_string(err));
    return;
  }
  // Retrieve the libomptarget function pointer if required
  ensureTimestampFn();
  // No need to hold a lock
  // Factor in the frequency
  ompt_set_timestamp_fn(time_rec.start * DeviceInfo().TicksToTime,
                        time_rec.end * DeviceInfo().TicksToTime);
}

/// Get the current HSA system timestamp
static uint64_t getSystemTimestampInNs() {
  uint64_t timestamp = 0;
  hsa_status_t err = hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP, &timestamp);
  if (err != HSA_STATUS_SUCCESS) {
    DP("Error while getting system timestamp: %s\n", get_error_string(err));
  }
  return timestamp * DeviceInfo().TicksToTime;
}

/// RAII used for timing certain plugin functionality and transferring the
/// information to libomptarget
struct OmptTimestampRAII {
  OmptTimestampRAII() { OMPT_IF_TRACING_ENABLED(setStart();); }
  ~OmptTimestampRAII() { OMPT_IF_ENABLED(setTimestamp();); }

private:
  uint64_t StartTime = 0;
  void setStart() { StartTime = getSystemTimestampInNs(); }
  void setTimestamp() {
    uint64_t EndTime = getSystemTimestampInNs();
    ensureTimestampFn();
    ompt_set_timestamp_fn(StartTime, EndTime);
  }
};

// Enable delaying of memory copy completion check
// and unlocking of host pointers used in the transfer
class AMDGPUAsyncInfoDataTy {
public:
  AMDGPUAsyncInfoDataTy()
      : HstPtr(nullptr), HstOrPoolPtr(nullptr), Size(0),
        alreadyCompleted(false), userLocked(false){};
  AMDGPUAsyncInfoDataTy(hsa_signal_t signal, void *HPtr, void *EitherPtr,
                        size_t Sz, bool userLocked)
      : signal(signal), HstPtr(HPtr), HstOrPoolPtr(EitherPtr), Size(Sz),
        alreadyCompleted(false), userLocked(userLocked) {}

  AMDGPUAsyncInfoDataTy(const AMDGPUAsyncInfoDataTy &) = delete;
  AMDGPUAsyncInfoDataTy(AMDGPUAsyncInfoDataTy &&) = default; // assume noexcept

  AMDGPUAsyncInfoDataTy &operator=(const AMDGPUAsyncInfoDataTy &&tmp) {
    signal = tmp.signal;
    HstPtr = tmp.HstPtr;
    HstOrPoolPtr = tmp.HstOrPoolPtr;
    Size = tmp.Size;
    alreadyCompleted = tmp.alreadyCompleted;
    userLocked = tmp.userLocked;
    return *this;
  }

  inline hsa_signal_t getSignal() const { return signal; }

  hsa_status_t waitToComplete() {
    if (alreadyCompleted)
      return HSA_STATUS_SUCCESS;
    hsa_signal_value_t init = 1;
    hsa_signal_value_t success = 0;
    hsa_status_t err = wait_for_signal(signal, init, success);
    OMPT_IF_TRACING_ENABLED(recordCopyTimingInNs(signal););

    // Now that the operation is complete, copy data from PoolPtr
    // to HstPtr if applicable
    if (HstPtr != HstOrPoolPtr) {
      assert(HstPtr != nullptr && HstOrPoolPtr != nullptr &&
             "HstPr and PoolPtr both must be non-null");
      DP("Memcpy %lu bytes from PoolPtr %p to HstPtr %p\n", Size, HstOrPoolPtr,
         HstPtr);
      memcpy(HstPtr, HstOrPoolPtr, Size);
    }

    DeviceInfo().FreeSignalPool.push(signal);
    alreadyCompleted = true;
    return err;
  }

  hsa_status_t releaseResources() {
    if (userLocked)
      return HSA_STATUS_SUCCESS;

    // If allocated from the pool, just release the ptr to the pool without
    // unlocking it
    assert(HstPtr != nullptr && HstOrPoolPtr != nullptr &&
           "Both HstPtr and HstOrPoolPtr must be non-null");
    if (HstOrPoolPtr != HstPtr) {
      DP("Releasing %p into pool without unlocking\n", HstOrPoolPtr);
      SmallPoolMgr.releaseIntoPool(Size, HstPtr);
      return HSA_STATUS_SUCCESS;
    }
    DP("Calling hsa_amd_memory_unlock %p\n", HstPtr);
    return hsa_amd_memory_unlock(HstPtr);
  }

private:
  hsa_signal_t signal;
  /// HostPtr initially passed in from a higher layer
  void *HstPtr;
  /// HstOrPoolPtr could be what was initially passed in from a higher layer or
  // it could be a pool pointer
  void *HstOrPoolPtr;    // for delayed unlocking
  size_t Size;           // size of data
  bool alreadyCompleted; // libomptarget might call synchronize multiple times:
                         // only serve once
  bool userLocked;       // skip unlocking when user provided locked pointer
};

// Enable delaying of kernel launch completion check
class AMDGPUAsyncInfoComputeTy {
public:
  AMDGPUAsyncInfoComputeTy()
      : kernelExecutionCompleted(false), ArgPool(nullptr), kernarg(nullptr) {}
  AMDGPUAsyncInfoComputeTy(hsa_signal_t signal, hsa_agent_t agt,
                           KernelArgPool *ArgPool, void *kernarg)
      : kernelExecutionCompleted(false), signal(signal), agent(agt),
        ArgPool(ArgPool), kernarg(kernarg) {}

  ~AMDGPUAsyncInfoComputeTy() {}
  AMDGPUAsyncInfoComputeTy(const AMDGPUAsyncInfoComputeTy &) = delete;

  AMDGPUAsyncInfoComputeTy(AMDGPUAsyncInfoComputeTy &&tmp) = default;

  AMDGPUAsyncInfoComputeTy &operator=(const AMDGPUAsyncInfoComputeTy &&tmp) {
    kernelExecutionCompleted = tmp.kernelExecutionCompleted;
    signal = tmp.signal;
    agent = tmp.agent;
    ArgPool = tmp.ArgPool;
    kernarg = tmp.kernarg;
    return *this;
  }

  inline bool hasCompleted() const { return kernelExecutionCompleted; }

  hsa_status_t waitToComplete() {
    hsa_signal_value_t init = 1;
    hsa_signal_value_t success = 0;
    hsa_status_t err = wait_for_signal(signal, init, success);
    OMPT_IF_TRACING_ENABLED(recordKernelTimingInNs(signal, agent););
    DeviceInfo().FreeSignalPool.push(signal);
    assert(ArgPool);
    ArgPool->deallocate(kernarg);
    kernelExecutionCompleted = true;
    return err;
  }

private:
  // used to prevent tgt_rtl_data_retrieve to start copy before kernel has
  // finishe
  bool kernelExecutionCompleted;

  hsa_signal_t signal;
  hsa_agent_t agent;
  KernelArgPool *ArgPool; // needed for deallocation of kernarg
  void *kernarg;          // kernarg ptr used by kernel launch
};

class AMDGPUAsyncInfoQueueTy {
public:
  AMDGPUAsyncInfoQueueTy()
      : hasMapEnteringInfo(false), hasMapExitingInfo(false),
        hasKernelLaunchInfo(false),
        kernelLaunchInfo(AMDGPUAsyncInfoComputeTy()) {
    // reserve capacity for vectors: best guess or get libomptarget to tell us
    // precisely how much (modify interface)
  }

  ~AMDGPUAsyncInfoQueueTy() {
    mapEnteringInfo.clear();
    mapExitingInfo.clear();
  }

  void addMapEnteringInfo(AMDGPUAsyncInfoDataTy &&enter) {
    hasMapEnteringInfo = true;
    mapEnteringInfo.emplace_back(std::move(enter));
  }

  void addMapExitingInfo(AMDGPUAsyncInfoDataTy &&exit) {
    hasMapExitingInfo = true;
    mapExitingInfo.emplace_back(std::move(exit));
  }

  void addKernelLaunchInfo(const AMDGPUAsyncInfoComputeTy &&launch) {
    hasKernelLaunchInfo = true;
    kernelLaunchInfo = std::move(launch);
  }

  inline bool needToWaitForKernel() const {
    return hasKernelLaunchInfo && !kernelLaunchInfo.hasCompleted();
  }

  AMDGPUAsyncInfoComputeTy &getKernelInfo() { return kernelLaunchInfo; }

  // create barrierAND packets for map-entering info's
  hsa_status_t createMapEnteringDependencies(hsa_queue_t *queue);
  hsa_status_t waitForKernelCompletion();
  void waitForMapExiting();
  hsa_status_t synchronize();

private:
  // currently a literal constant in the HSA header file
  static constexpr size_t maxBarrierANDSignals =
      sizeof(hsa_barrier_and_packet_s::dep_signal) / sizeof(hsa_signal_t);
  bool hasMapEnteringInfo;
  bool hasMapExitingInfo;
  bool hasKernelLaunchInfo;

  std::vector<AMDGPUAsyncInfoDataTy> mapEnteringInfo;
  std::vector<AMDGPUAsyncInfoDataTy> mapExitingInfo;
  AMDGPUAsyncInfoComputeTy kernelLaunchInfo;

  // signals used by barrierAND packets (if needed)
  std::vector<hsa_signal_t> barrierANDCompletionSignals;
};

hsa_status_t
AMDGPUAsyncInfoQueueTy::createMapEnteringDependencies(hsa_queue_t *queue) {
  // fast track: kernel without map-entering phase
  if (!hasMapEnteringInfo)
    return HSA_STATUS_SUCCESS;

  int numBarrierANDPackets =
      mapEnteringInfo.size() / maxBarrierANDSignals +
      ((mapEnteringInfo.size() % maxBarrierANDSignals == 0) ? 0 : 1);

  // to schedule numBarrierANDPackets (n) Barrier-AND packets, we need
  // - n completion signals
  // - n packet IDs
  // - n packets
  uint64_t barrierANDMapEnteringPacketIDs[numBarrierANDPackets];
  hsa_barrier_and_packet_s
      *barrierANDMapEnteringBarrierPackets[numBarrierANDPackets];
  for (int i = 0; i < numBarrierANDPackets; i++) {
    hsa_signal_t completion_signal = DeviceInfo().FreeSignalPool.pop();
    if (completion_signal.handle == 0) {
      DP("Failed to get signal instance\n");
      return HSA_STATUS_ERROR;
    }
    barrierANDCompletionSignals.emplace_back(completion_signal);

    uint64_t packetId = core::acquire_available_packet_id(queue);
    barrierANDMapEnteringPacketIDs[i] = packetId;
    const uint32_t mask = queue->size - 1; // size is a power of 2
    hsa_barrier_and_packet_s *barrierPacket =
        (hsa_barrier_and_packet_s *)queue->base_address + (packetId & mask);

    // completion signal is not needed: can I leave this empty?
    // check at runtime
    barrierPacket->completion_signal =
        completion_signal; // link packet to its completion signal

    barrierANDMapEnteringBarrierPackets[i] = barrierPacket;
  }

  // Create input dependecies between map-entering info and BarrierAND
  // packet
  for (int i = 0, k = 0; i < mapEnteringInfo.size(); i += 5, k++) {
    // Fill Barrier-AND packet with signals from MapEntering phase
    hsa_signal_t zeroHandleSignal;
    zeroHandleSignal.handle = 0;
    barrierANDMapEnteringBarrierPackets[k]->dep_signal[0] =
        mapEnteringInfo[i].getSignal();
    barrierANDMapEnteringBarrierPackets[k]->dep_signal[1] =
        (i + 1 < mapEnteringInfo.size()) ? mapEnteringInfo[i + 1].getSignal()
                                         : zeroHandleSignal;
    barrierANDMapEnteringBarrierPackets[k]->dep_signal[2] =
        (i + 2 < mapEnteringInfo.size()) ? mapEnteringInfo[i + 2].getSignal()
                                         : zeroHandleSignal;
    barrierANDMapEnteringBarrierPackets[k]->dep_signal[3] =
        (i + 3 < mapEnteringInfo.size()) ? mapEnteringInfo[i + 3].getSignal()
                                         : zeroHandleSignal;
    barrierANDMapEnteringBarrierPackets[k]->dep_signal[4] =
        (i + 4 < mapEnteringInfo.size()) ? mapEnteringInfo[i + 4].getSignal()
                                         : zeroHandleSignal;

    // enqueue into device queue
    // Set signal to 1. It will be decremented to 0 by HSA runtime once
    // Barrier-AND packet is complete
    hsa_signal_store_relaxed(
        barrierANDMapEnteringBarrierPackets[k]->completion_signal, 1);

    // Publish the packet indicating it is ready to be processed
    core::packetStoreRelease(
        reinterpret_cast<uint32_t *>(barrierANDMapEnteringBarrierPackets[k]),
        core::create_BarrierAND_header(), 0);

    hsa_signal_store_relaxed(queue->doorbell_signal,
                             barrierANDMapEnteringPacketIDs[k]);
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t AMDGPUAsyncInfoQueueTy::waitForKernelCompletion() {
  if (!hasKernelLaunchInfo)
    return HSA_STATUS_SUCCESS;
  return kernelLaunchInfo.waitToComplete();
}

void AMDGPUAsyncInfoQueueTy::waitForMapExiting() {
  if (!hasMapExitingInfo)
    return;

  for (auto &&s : mapExitingInfo)
    s.waitToComplete();
}

hsa_status_t AMDGPUAsyncInfoQueueTy::synchronize() {
  // fast tracks:
  // - kernel with no map-entering and no map-exiting info's
  // - single data submission
  // - single data retrieval
  if (hasKernelLaunchInfo && !hasMapEnteringInfo && !hasMapExitingInfo)
    return waitForKernelCompletion();

  // in absence of kernel and map exiting info, only wait for data submit's
  if (!hasKernelLaunchInfo && hasMapEnteringInfo && !hasMapExitingInfo) {
    for(auto &&enter : mapEnteringInfo) {
      enter.waitToComplete();
      enter.releaseResources();
    }
    return HSA_STATUS_SUCCESS;
  }

  // in absence of kernel and map entering info's, only wait for data retrieve's
  if (!hasKernelLaunchInfo && !hasMapEnteringInfo && hasMapExitingInfo) {
    for(auto &&exit : mapExitingInfo) {
      exit.waitToComplete();
      exit.releaseResources();
    }
    return HSA_STATUS_SUCCESS;
  }

  // slow track: all other cases, such as any combination
  // of kernel plus map-entering or map-exiting info's
  // or both. Either have to wait for kernel (no map-exiting)
  // or for all map-exiting events
  if (!hasMapExitingInfo) {
    waitForKernelCompletion();
  } else
    waitForMapExiting();

  // finally, release all resources
  for (auto &&ent : mapEnteringInfo)
    ent.releaseResources();

  for (auto &&ext : mapExitingInfo)
    ext.releaseResources();

  for (int i = 0; i < barrierANDCompletionSignals.size(); i++)
    DeviceInfo().FreeSignalPool.push(barrierANDCompletionSignals[i]);

  return HSA_STATUS_SUCCESS;
}

/// Get a pointer from a small pool, given a host pointer
void *prepareHstPtrForDataRetrieve(size_t Size, void *HstPtr) {
  void *PoolPtr = SmallPoolMgr.allocateFromPool(Size, HstPtr);
  if (PoolPtr != nullptr) {
    DP("prepareHstPtrForDataRetrieve: HostPtr %p PoolPtr %p\n", HstPtr,
       PoolPtr);
    return PoolPtr;
  }
  return HstPtr;
}

int32_t dataRetrieve(int32_t DeviceId, void *HstPtr, void *TgtPtr, int64_t Size,
                     AMDGPUAsyncInfoDataTy &AsyncData) {
  assert(DeviceId < DeviceInfo().NumberOfDevices && "Device ID too large");
  // Return success if we are not copying back to host from target.
  if (!HstPtr)
    return OFFLOAD_SUCCESS;
  hsa_status_t Err;
  DP("Retrieve data %ld bytes, (tgt:%016llx) -> (hst:%016llx).\n", Size,
     (long long unsigned)(Elf64_Addr)TgtPtr,
     (long long unsigned)(Elf64_Addr)HstPtr);

  void *HstOrPoolPtr = prepareHstPtrForDataRetrieve(Size, HstPtr);
  assert(HstOrPoolPtr && "HstOrPoolPtr cannot be null");

  hsa_signal_t Signal;
  bool UserLocked;
  Err = DeviceInfo().freesignalpoolMemcpyD2H(HstOrPoolPtr, TgtPtr, (size_t)Size,
                                           DeviceId, Signal, UserLocked);

  if (Err != HSA_STATUS_SUCCESS) {
    DP("Error when copying data from device to host. Pointers: "
       "host = 0x%016lx, device = 0x%016lx, size = %lld\n",
       (Elf64_Addr)HstOrPoolPtr, (Elf64_Addr)TgtPtr, (unsigned long long)Size);
    return OFFLOAD_FAIL;
  }

  DP("dataRetrieve: Creating AsyncData with HostPtr %p HstOrPoolPtr %p\n",
     HstPtr, HstOrPoolPtr);

  AsyncData = std::move(
      AMDGPUAsyncInfoDataTy(Signal, HstPtr, HstOrPoolPtr, Size, UserLocked));
  DP("DONE Retrieve data %ld bytes, (tgt:%016llx) -> (hst:%016llx).\n", Size,
     (long long unsigned)(Elf64_Addr)TgtPtr,
     (long long unsigned)(Elf64_Addr)HstOrPoolPtr);
  return Err;
}

/// Get a pointer from a small pool, given a HstPtr. Perform copy-in to the pool
/// pointer since data transfer will use the pool pointer
void *prepareHstPtrForDataSubmit(size_t Size, void *HstPtr) {
  void *PoolPtr = SmallPoolMgr.allocateFromPool(Size, HstPtr);
  if (PoolPtr != nullptr) {
    DP("dataSubmit: memcpy %lu bytes from HstPtr %p to PoolPtr %p\n", Size,
       HstPtr, PoolPtr);
    memcpy(PoolPtr, HstPtr, Size);
    return PoolPtr;
  }
  return HstPtr;
}

int32_t dataSubmit(int32_t DeviceId, void *TgtPtr, void *HstPtr, int64_t Size,
                   AMDGPUAsyncInfoDataTy &AsyncData) {
  hsa_status_t Err;
  assert(DeviceId < DeviceInfo().NumberOfDevices && "Device ID too large");
  // Return success if we are not doing host to target.
  if (!HstPtr)
    return OFFLOAD_SUCCESS;

  DP("Submit data %ld bytes, (hst:%016llx) -> (tgt:%016llx).\n", Size,
     (long long unsigned)(Elf64_Addr)HstPtr,
     (long long unsigned)(Elf64_Addr)TgtPtr);

  void *HstOrPoolPtr = prepareHstPtrForDataSubmit(Size, HstPtr);
  assert(HstOrPoolPtr && "HstOrPoolPtr cannot be null");

  hsa_signal_t Signal;
  bool UserLocked;
  Err = DeviceInfo().freesignalpoolMemcpyH2D(TgtPtr, HstOrPoolPtr, (size_t)Size,
                                           DeviceId, Signal, UserLocked);
  if (Err != HSA_STATUS_SUCCESS) {
    DP("Error when copying data from host to device. Pointers: "
       "host = 0x%016lx, device = 0x%016lx, size = %lld\n",
       (Elf64_Addr)HstOrPoolPtr, (Elf64_Addr)TgtPtr, (unsigned long long)Size);
    return OFFLOAD_FAIL;
  }

  AsyncData = std::move(
      AMDGPUAsyncInfoDataTy(Signal, HstPtr, HstOrPoolPtr, Size, UserLocked));
  return Err;
}

void initAsyncInfo(__tgt_async_info *AsyncInfo) {
  assert(AsyncInfo);
  if (!AsyncInfo->Queue) {
    AsyncInfo->Queue = new AMDGPUAsyncInfoQueueTy();
  }
}
void finiAsyncInfo(__tgt_async_info *AsyncInfo) {
  assert(AsyncInfo);
  assert(AsyncInfo->Queue);
  AMDGPUAsyncInfoQueueTy *AMDAsyncInfoQueue =
      reinterpret_cast<AMDGPUAsyncInfoQueueTy *>(AsyncInfo->Queue);
  delete (AMDAsyncInfoQueue);
  AsyncInfo->Queue = nullptr;
}

// Determine launch values for ThreadsPerGroup and NumGroups.
// Outputs: treadsPerGroup, NumGroups
// Inputs: Max_Teams, MaxWgSize, Warp_Size, ExecutionMode,
//         EnvTeamLimit, EnvNumTeams, num_teams, thread_limit,
//         loop_tripcount.
void getLaunchVals(int &ThreadsPerGroup, int &NumGroups, int WarpSize,
                   EnvironmentVariables Env, int ConstWGSize, int ExecutionMode,
                   int NumTeams, int ThreadLimit, uint64_t LoopTripcount,
                   int DeviceNumTeams, int DeviceNumCUs) {
  if (ExecutionMode == llvm::omp::OMPTgtExecModeFlags::OMP_TGT_EXEC_MODE_SPMD ||
      ExecutionMode ==
          llvm::omp::OMPTgtExecModeFlags::OMP_TGT_EXEC_MODE_SPMD_NO_LOOP ||
      ExecutionMode ==
          llvm::omp::OMPTgtExecModeFlags::OMP_TGT_EXEC_MODE_SPECIAL_RED) {
    // ConstWGSize is used for communicating any command-line value to
    // the plugin. ConstWGSize will either be the default workgroup
    // size or a value set by CodeGen. If the kernel is SPMD, it means
    // that the number of threads-per-group has not been adjusted by
    // CodeGen. Since a generic mode may have been changed to
    // generic_spmd by OpenMPOpt after adjustment of
    // threads-per-group, we don't use ConstWGSize but instead start
    // with the default for both generic and generic_spmd in the
    // plugin so that any adjustment can be done again.
    ThreadsPerGroup = ConstWGSize;
  } else
    ThreadsPerGroup = RTLDeviceInfoTy::DefaultWgSize;

  if (ExecutionMode ==
      llvm::omp::OMPTgtExecModeFlags::OMP_TGT_EXEC_MODE_SPMD_NO_LOOP) {
    assert(LoopTripcount &&
           "No loop exec mode needs a non-zero loop tripcount");
    NumGroups = ((LoopTripcount - 1) / ThreadsPerGroup) + 1;
    return;
  }

  // For optimized reduction, we use as many teams as the number of CUs. This
  // must be kept in sync with CodeGen and DeviceRTL.
  if (ExecutionMode ==
      llvm::omp::OMPTgtExecModeFlags::OMP_TGT_EXEC_MODE_SPECIAL_RED) {
    assert(LoopTripcount &&
           "No loop exec mode needs a non-zero loop tripcount");
    NumGroups = DeviceNumCUs;
    return;
  }

  int MaxTeams = Env.MaxTeamsDefault > 0 ? Env.MaxTeamsDefault : DeviceNumTeams;
  if (MaxTeams > static_cast<int>(RTLDeviceInfoTy::HardTeamLimit))
    MaxTeams = RTLDeviceInfoTy::HardTeamLimit;

  if (print_kernel_trace & STARTUP_DETAILS) {
    DP("RTLDeviceInfoTy::Max_Teams: %d\n", RTLDeviceInfoTy::MaxTeams);
    DP("Max_Teams: %d\n", MaxTeams);
    DP("RTLDeviceInfoTy::Warp_Size: %d\n", WarpSize);
    DP("RTLDeviceInfoTy::MaxWgSize: %d\n", RTLDeviceInfoTy::MaxWgSize);
    DP("RTLDeviceInfoTy::DefaultWgSize: %d\n",
       RTLDeviceInfoTy::DefaultWgSize);
    DP("thread_limit: %d\n", ThreadLimit);
    DP("ThreadsPerGroup: %d\n", ThreadsPerGroup);
    DP("ConstWGSize: %d\n", ConstWGSize);
  }
  // check for thread_limit() clause
  if (ThreadLimit > 0) {
    ThreadsPerGroup = ThreadLimit;
    DP("Setting threads per block to requested %d\n", ThreadLimit);
    if (ThreadsPerGroup > RTLDeviceInfoTy::MaxWgSize) { // limit to max
      ThreadsPerGroup = RTLDeviceInfoTy::MaxWgSize;
      DP("Setting threads per block to maximum %d\n", ThreadsPerGroup);
    }
  }
  // check flat_max_work_group_size attr here
  if (ThreadsPerGroup > ConstWGSize) {
    ThreadsPerGroup = ConstWGSize;
    DP("Reduced ThreadsPerGroup to flat-attr-group-size limit %d\n",
       ThreadsPerGroup);
  }

  if (ExecutionMode ==
      llvm::omp::OMPTgtExecModeFlags::OMP_TGT_EXEC_MODE_GENERIC) {
    // Add master thread in additional warp for GENERIC mode
    // Only one additional thread is started, not an entire warp

    if (ThreadsPerGroup >= RTLDeviceInfoTy::MaxWgSize)
      // Do not exceed max number of threads: sacrifice last warp for
      // the thread master
      ThreadsPerGroup = RTLDeviceInfoTy::MaxWgSize - WarpSize + 1;
    else if (ThreadsPerGroup <= WarpSize)
      // Cap ThreadsPerGroup at WarpSize level as we need a master
      // FIXME: omp_get_num_threads still too big for thread_limit(<warpsize)
      ThreadsPerGroup = WarpSize + 1;
    else
      ThreadsPerGroup = WarpSize * (ThreadsPerGroup / WarpSize) + 1;

    DP("Adding master thread (+1)\n");
  }

  if (print_kernel_trace & STARTUP_DETAILS)
    DP("ThreadsPerGroup: %d\n", ThreadsPerGroup);
  DP("Preparing %d threads\n", ThreadsPerGroup);

  // Set default NumGroups (teams)
  if (DeviceInfo().Env.TeamLimit > 0)
    NumGroups = (MaxTeams < DeviceInfo().Env.TeamLimit)
                     ? MaxTeams
                     : DeviceInfo().Env.TeamLimit;
  else
    NumGroups = MaxTeams;
  DP("Set default num of groups %d\n", NumGroups);

  if (print_kernel_trace & STARTUP_DETAILS) {
    DP("NumGroups: %d\n", NumGroups);
    DP("num_teams: %d\n", NumTeams);
  }

  // Reduce NumGroups if ThreadsPerGroup exceeds RTLDeviceInfoTy::MaxWgSize
  // This reduction is typical for default case (no thread_limit clause).
  // or when user goes crazy with num_teams clause.
  // FIXME: We cant distinguish between a constant or variable thread limit.
  // So we only handle constant thread_limits.
  if (ThreadsPerGroup >
      RTLDeviceInfoTy::DefaultWgSize) //  256 < ThreadsPerGroup <= 1024
    // Should we round ThreadsPerGroup up to nearest WarpSize
    // here?
    NumGroups = (MaxTeams * RTLDeviceInfoTy::MaxWgSize) / ThreadsPerGroup;

  // check for num_teams() clause
  if (NumTeams > 0) {
    NumGroups = (NumTeams < NumGroups) ? NumTeams : NumGroups;
  }
  if (print_kernel_trace & STARTUP_DETAILS) {
    DP("NumGroups: %d\n", NumGroups);
    DP("Env.NumTeams %d\n", DeviceInfo().Env.NumTeams);
    DP("Env.TeamLimit %d\n", DeviceInfo().Env.TeamLimit);
  }

  if (DeviceInfo().Env.NumTeams > 0) {
    NumGroups = (DeviceInfo().Env.NumTeams < NumGroups)
                     ? DeviceInfo().Env.NumTeams
                     : NumGroups;
    DP("Modifying teams based on Env.NumTeams %d\n", DeviceInfo().Env.NumTeams);
  } else if (DeviceInfo().Env.TeamLimit > 0) {
    NumGroups = (DeviceInfo().Env.TeamLimit < NumGroups)
                     ? DeviceInfo().Env.TeamLimit
                     : NumGroups;
    DP("Modifying teams based on Env.TeamLimit%d\n", DeviceInfo().Env.TeamLimit);
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
        DP("Limiting NumGroups %d to Max_Teams %d \n", NumGroups, MaxTeams);
    }
    if (NumGroups > NumTeams && NumTeams > 0) {
      NumGroups = NumTeams;
      if (print_kernel_trace & STARTUP_DETAILS)
        DP("Limiting NumGroups %d to clause num_teams %d \n", NumGroups,
           NumTeams);
    }
  }

  // num_teams clause always honored, no matter what, unless DEFAULT is active.
  if (NumTeams > 0) {
    NumGroups = NumTeams;
    // Cap NumGroups to EnvMaxTeamsDefault if set.
    if (DeviceInfo().Env.MaxTeamsDefault > 0 &&
        NumGroups > DeviceInfo().Env.MaxTeamsDefault)
      NumGroups = DeviceInfo().Env.MaxTeamsDefault;
  }
  if (print_kernel_trace & STARTUP_DETAILS) {
    DP("ThreadsPerGroup: %d\n", ThreadsPerGroup);
    DP("NumGroups: %d\n", NumGroups);
    DP("LoopTripcount: %ld\n", LoopTripcount);
  }
  DP("Final %d NumGroups and %d ThreadsPerGroup\n", NumGroups,
     ThreadsPerGroup);

#ifdef OMPT_SUPPORT
  if (ompt_device_callbacks.is_tracing_enabled()) {
    {
      std::unique_lock<std::mutex> granted_teams_fn_lck(granted_teams_mtx);
      if (!ompt_set_granted_teams_fn) {
        void *vptr = dlsym(NULL, "libomptarget_ompt_set_granted_teams");
        assert(vptr && "OMPT set granted teams entry point not found");
        ompt_set_granted_teams_fn =
            reinterpret_cast<libomptarget_ompt_set_granted_teams_t>(vptr);
      }
    }
    // No need to hold a lock
    ompt_set_granted_teams_fn(NumGroups);
  }
#endif
}

int32_t runRegionLocked(int32_t DeviceId, void *TgtEntryPtr, void **TgtArgs,
                        ptrdiff_t *TgtOffsets, int32_t ArgNum, int32_t NumTeams,
                        int32_t ThreadLimit, uint64_t LoopTripcount,
                        AMDGPUAsyncInfoQueueTy &AsyncInfo) {
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
  int NumGroups = 0;
  int ThreadsPerGroup = 0;

  getLaunchVals(ThreadsPerGroup, NumGroups, DeviceInfo().WarpSize[DeviceId],
                DeviceInfo().Env, KernelInfo->ConstWGSize,
                KernelInfo->ExecutionMode,
                NumTeams,      // From run_region arg
                ThreadLimit,   // From run_region arg
                LoopTripcount, // From run_region arg
                DeviceInfo().NumTeams[KernelInfo->DeviceId],
                DeviceInfo().ComputeUnits[KernelInfo->DeviceId]);

  if (print_kernel_trace >= LAUNCH) {
    // enum modes are SPMD, GENERIC, NONE 0,1,2
    // if doing rtl timing, print to stderr, unless stdout requested.
    bool TraceToStdout = print_kernel_trace & (RTL_TO_STDOUT | RTL_TIMING);
    fprintf(TraceToStdout ? stdout : stderr,
            "DEVID:%2d SGN:%1d ConstWGSize:%-4d args:%2d teamsXthrds:(%4dX%4d) "
            "reqd:(%4dX%4d) lds_usage:%uB sgpr_count:%u vgpr_count:%u "
            "sgpr_spill_count:%u vgpr_spill_count:%u tripcount:%lu rpc:%d n:%s\n",
            DeviceId, KernelInfo->ExecutionMode, KernelInfo->ConstWGSize,
            ArgNum, NumGroups, ThreadsPerGroup, NumTeams, ThreadLimit,
            GroupSegmentSize, SgprCount, VgprCount, SgprSpillCount,
            VgprSpillCount, LoopTripcount, DeviceInfo().HostcallRequired,
            KernelInfo->Name);
  }

  // Run on the device.
  {
    hsa_queue_t *Queue = DeviceInfo().HSAQueueSchedulers[DeviceId].next();
    if (!Queue) {
      return OFFLOAD_FAIL;
    }

    AsyncInfo.createMapEnteringDependencies(Queue);
    uint64_t PacketId = core::acquire_available_packet_id(Queue);

    const uint32_t Mask = Queue->size - 1; // size is a power of 2
    hsa_kernel_dispatch_packet_t *Packet =
        (hsa_kernel_dispatch_packet_t *)Queue->base_address + (PacketId & Mask);

    // Packet->header is written last
    Packet->setup = UINT16_C(1) << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
    Packet->workgroup_size_x = ThreadsPerGroup;
    Packet->workgroup_size_y = 1;
    Packet->workgroup_size_z = 1;
    Packet->reserved0 = 0;
    Packet->grid_size_x = NumGroups * ThreadsPerGroup;
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
      auto It = KernelArgPoolMap.find(std::string(KernelInfo->Name));
      if (It != KernelArgPoolMap.end()) {
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
        uint64_t Buffer = hostrpc_assign_buffer(DeviceInfo().HSAAgents[DeviceId],
                                                Queue, DeviceId);
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

    // wait for completion, then free signal and kernarg
    AsyncInfo.addKernelLaunchInfo(AMDGPUAsyncInfoComputeTy(
        S, DeviceInfo().HSAAgents[DeviceId], ArgPool, KernArg));
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

        hsa_signal_t Signal;
        bool UserLocked;
        Err = DeviceInfo().freesignalpoolMemcpyH2D(StatePtr, &HostDeviceEnv,
                                                 StatePtrSize, DeviceId,
                                                 Signal, UserLocked);
        if (Err == HSA_STATUS_ERROR)
          return Err;
        AMDGPUAsyncInfoDataTy AsyncInfo(Signal, &HostDeviceEnv, &HostDeviceEnv,
                                        StatePtrSize, UserLocked);
        Err = AsyncInfo.waitToComplete();
        return Err;
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

hsa_signal_t launchBarrierANDPacket(hsa_queue_t *Queue,
                                    std::vector<hsa_signal_t> &depSignals,
                                    bool isBarrierBitSet) {
  hsa_signal_t barrier_signal = DeviceInfo().FreeSignalPool.pop();
  uint64_t barrierAndPacketId = acquire_available_packet_id(Queue);
  const uint32_t mask = Queue->size - 1;
  hsa_barrier_and_packet_t *barrier_and_packet =
      (hsa_barrier_and_packet_t *)Queue->base_address +
      (barrierAndPacketId & mask);
  memset(barrier_and_packet, 0, sizeof(hsa_barrier_and_packet_t));
  for (size_t i = 0; (i < depSignals.size()) && (depSignals.size() <= 5); i++)
    barrier_and_packet->dep_signal[i] = depSignals[i];
  int16_t header = create_BarrierAND_header();
  if (isBarrierBitSet)
    header |= 1 << 8;
  packetStoreRelease(reinterpret_cast<uint32_t *>(barrier_and_packet), header,
                       0);
  hsa_signal_store_screlease(Queue->doorbell_signal, barrierAndPacketId);
  return barrier_signal;
}

int32_t runInitFiniKernel(int DeviceId, uint16_t header,
                          const atl_kernel_info_t &entryInfo) {
  hsa_signal_t signal;
  void *kernarg_address = nullptr;

  hsa_queue_t *Queue = DeviceInfo().HSAQueueSchedulers[DeviceId].next();
  if (!Queue) {
    DP("Failed to get the queue instance.\n");
    return OFFLOAD_FAIL;
  }

  uint64_t packet_id = acquire_available_packet_id(Queue);
  const uint32_t mask = Queue->size - 1; // size is a power of 2
  hsa_kernel_dispatch_packet_t *dispatch_packet =
      (hsa_kernel_dispatch_packet_t *)Queue->base_address + (packet_id & mask);

  signal = DeviceInfo().FreeSignalPool.pop();
  if (signal.handle == 0) {
    DP("Failed to get signal instance\n");
    return OFFLOAD_FAIL;
  }

  dispatch_packet->setup |= 1 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
  dispatch_packet->workgroup_size_x = (uint16_t)1;
  dispatch_packet->workgroup_size_y = (uint16_t)1;
  dispatch_packet->workgroup_size_z = (uint16_t)1;
  dispatch_packet->grid_size_x = (uint32_t)(1 * 1);
  dispatch_packet->grid_size_y = 1;
  dispatch_packet->grid_size_z = 1;
  dispatch_packet->completion_signal = signal;
  dispatch_packet->kernel_object = entryInfo.kernel_object;
  dispatch_packet->private_segment_size = entryInfo.private_segment_size;
  dispatch_packet->group_segment_size = entryInfo.group_segment_size;
  dispatch_packet->kernarg_address = (void *)kernarg_address;
  dispatch_packet->completion_signal = signal;

  hsa_signal_store_relaxed(dispatch_packet->completion_signal, 1);

  packetStoreRelease(reinterpret_cast<uint32_t *>(dispatch_packet), header,
                       dispatch_packet->setup);

  // Increment the write index and ring the doorbell to dispatch the kernel.
  hsa_signal_store_screlease(Queue->doorbell_signal, packet_id);

  while (hsa_signal_wait_scacquire(dispatch_packet->completion_signal,
                                   HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX,
                                   HSA_WAIT_STATE_ACTIVE) != 0)
    ;
  DeviceInfo().FreeSignalPool.push(signal);
  return OFFLOAD_SUCCESS;
}

void launchInitFiniKernel(int32_t DeviceId, void *img, const size_t &size,
                          const initORfini status) {
  std::string kernelName, kernelTag;
  bool symbolExist = false;
  auto &KernelInfoTable = DeviceInfo().KernelInfoTable;
  int32_t runInitFini = OFFLOAD_FAIL;
  atl_kernel_info_t kernelInfoEntry;
  switch (status) {
  case INIT:
    kernelName = "amdgcn.device.init";
    kernelTag = "Init";
    symbolExist =
        imageContainsSymbol(img, size, (kernelName + ".kd").c_str());
    if (symbolExist && KernelInfoTable[DeviceId].find(kernelName) !=
                           KernelInfoTable[DeviceId].end()) {
      assert(DeviceInfo().ImageList[img].initfini != 0);
      kernelInfoEntry = KernelInfoTable[DeviceId][kernelName];
      assert(kernelInfoEntry.kind == "init");
      runInitFini =
          runInitFiniKernel(DeviceId, createHeader(), kernelInfoEntry);
    }
    break;

  case FINI:
    kernelName = "amdgcn.device.fini";
    kernelTag = "Fini";
    symbolExist =
        imageContainsSymbol(img, size, (kernelName + ".kd").c_str());
    if (symbolExist && KernelInfoTable[DeviceId].find(kernelName) !=
                           KernelInfoTable[DeviceId].end()) {
      assert(DeviceInfo().ImageList[img].initfini != 0);
      kernelInfoEntry = KernelInfoTable[DeviceId][kernelName];
      assert(kernelInfoEntry.kind == "fini");
      runInitFini =
          runInitFiniKernel(DeviceId, createHeader(), kernelInfoEntry);
    }
    break;

  default:
    kernelTag = "Normal";
  };

  if (runInitFini == OFFLOAD_SUCCESS) {
    DP("%s kernel launch successfull on AMDGPU Device %d for image(" DPxMOD
       ")!\n ",
       kernelTag.c_str(), DeviceId, DPxPTR(img));
  } else {
    DP("%s kernel launch failed on AMDGPU Device %d for image(" DPxMOD ")!\n ",
       kernelTag.c_str(), DeviceId, DPxPTR(img));
  }
}
} // namespace core

extern "C" {
int32_t __tgt_rtl_is_valid_binary(__tgt_device_image *Image) {
  return elfMachineIdIsAmdgcn(Image);
}

int __tgt_rtl_number_of_team_procs(int DeviceId) {
  return DeviceInfo().ComputeUnits[DeviceId];
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
  hsa_status_t Err;

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
       DeviceInfo().ComputeUnits[DeviceId], DeviceInfo().GPUName[DeviceId].c_str());

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
    DP("Using %d ROCm blocks per grid\n", DeviceInfo().GroupsPerDevice[DeviceId]);
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

  DP("Device %d: default limit for groupsPerDevice %d & ThreadsPerGroup %d\n",
     DeviceId, DeviceInfo().GroupsPerDevice[DeviceId],
     DeviceInfo().ThreadsPerGroup[DeviceId]);

  DP("Device %d: wavefront size %d, total threads %d x %d = %d\n", DeviceId,
     DeviceInfo().WarpSize[DeviceId], DeviceInfo().ThreadsPerGroup[DeviceId],
     DeviceInfo().GroupsPerDevice[DeviceId],
     DeviceInfo().GroupsPerDevice[DeviceId] *
         DeviceInfo().ThreadsPerGroup[DeviceId]);

  // Initialize memspace table to keep track of coarse grain memory regions
  // in USM mode
  if (DeviceInfo().RequiresFlags & OMP_REQ_UNIFIED_SHARED_MEMORY) {
    // TODO: add framework for multiple systems supporting unified_shared_memory
    coarse_grain_mem_tab = new AMDGPUMemTypeBitFieldTable(
        AMDGPU_X86_64_SystemConfiguration::max_addressable_byte +
            1, // memory size
        AMDGPU_X86_64_SystemConfiguration::page_size);
  }

  OMPT_IF_ENABLED(
      std::string ompt_gpu_type("AMD "); ompt_gpu_type += GetInfoName;
      const char *type = ompt_gpu_type.c_str();
      ompt_device_callbacks.ompt_callback_device_initialize(DeviceId, type););

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
    auto Env = DeviceEnvironment(DeviceId, DeviceInfo().NumberOfDevices,
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
          if (imageContainsSymbol(Data, Size, "amdgcn.device.init") &&
              imageContainsSymbol(Data, Size, "amdgcn.device.fini")) {
            DeviceInfo().ImageList.insert(
                {Image->ImageStart, Image_t(Size, true)});
          } else {
            DeviceInfo().ImageList.insert(
                {Image->ImageStart, Image_t(Size, false)});
          }

          return Env.beforeLoading(Data, Size);
        },
        DeviceInfo().HSAExecutables);

    check("Module registering", Err);
    if (Err != HSA_STATUS_SUCCESS) {
      const char *DeviceName = DeviceInfo().GPUName[DeviceId].c_str();
      const char *ElfName = get_elf_mach_gfx_name(elfEFlags(Image));

      if (strcmp(DeviceName, ElfName) != 0) {
        fprintf(stderr, "Possible gpu arch mismatch: device:%s, image:%s please check"
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

  OMPT_IF_ENABLED(const char *filename = nullptr; int64_t offset_in_file = 0;
                  void *vma_in_file = 0; size_t bytes = ImgSize;
                  void *host_addr = Image->ImageStart; void *device_addr = 0;
                  uint64_t module_id = 0; // FIXME
                  ompt_device_callbacks.ompt_callback_device_load(
                      DeviceId, filename, offset_in_file, vma_in_file, bytes,
                      host_addr, device_addr, module_id););

  core::launchInitFiniKernel(DeviceId, Image->ImageStart, ImgSize, INIT);

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
        hsa_signal_t Signal;
        bool UserLocked;
        Err = DeviceInfo().freesignalpoolMemcpyH2D(StatePtr, &Ptr, sizeof(void *),
                                                 DeviceId, Signal, UserLocked);
        if (Err != HSA_STATUS_SUCCESS) {
          DP("memcpy install of state_ptr failed\n");
          return NULL;
        }
        AMDGPUAsyncInfoDataTy AsyncInfo(Signal, &Ptr, &Ptr, sizeof(void *),
                                        UserLocked);
        Err = AsyncInfo.waitToComplete();
        if (Err != HSA_STATUS_SUCCESS)
          return NULL;
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
        hsa_signal_t Signal;
        bool UserLocked;
        Err = DeviceInfo().freesignalpoolMemcpyH2D(Varptr, E->addr,
          sizeof(void *), DeviceId, Signal, UserLocked);
        if (Err != HSA_STATUS_SUCCESS)
          DP("Error when copying USM\n");

        AMDGPUAsyncInfoDataTy AsyncInfo(Signal, E->addr, E->addr,
                                        sizeof(void *), UserLocked);
        AsyncInfo.waitToComplete();

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

    // get flat group size if present, else DefaultWgSize
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

      if (ExecModeVal < llvm::omp::OMP_TGT_EXEC_MODE_GENERIC ||
          ExecModeVal > llvm::omp::OMP_TGT_EXEC_MODE_SPECIAL_RED) {
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

    KernelsList.push_back(KernelTy(ExecModeVal, WGSizeVal, DeviceId,
                                   CallStackAddr, E->name, KernargSegmentSize,
                                   DeviceInfo().KernArgPool));
    __tgt_offload_entry Entry = *E;
    Entry.addr = (void *)&KernelsList.back();
    DeviceInfo().addOffloadEntry(DeviceId, Entry);
    DP("Entry point %ld maps to %s\n", E - HostBegin, E->name);
  }

  return DeviceInfo().getOffloadEntriesTable(DeviceId);
}

void *__tgt_rtl_data_alloc(int DeviceId, int64_t size, void *, int32_t kind) {
  void *ptr = nullptr;
  assert(DeviceId < DeviceInfo().NumberOfDevices && "Device ID too large");

  {
    // We don't have HSA-profiling timestamps for device allocation, so just get
    // the start and end system timestamps for OMPT
    OmptTimestampRAII AllocTimestamp;
    ptr = DeviceInfo().DeviceAllocators[DeviceId].allocate(size, nullptr,
                                                          (TargetAllocTy)kind);
  }
  if (kind == TARGET_ALLOC_SHARED) {
    __tgt_rtl_set_coarse_grain_mem_region(ptr, size);
  }

  DP("Tgt alloc data %ld bytes, (tgt:%016llx).\n", size,
     (long long unsigned)(Elf64_Addr)ptr);

  return ptr;
}

void *__tgt_rtl_data_lock(int DeviceId, void *TgtPtr, int64_t size) {
  void *ptr = TgtPtr;
  assert(DeviceId < DeviceInfo().NumberOfDevices && "Device ID too large");
  hsa_status_t err = HSA_STATUS_SUCCESS;

  err = lock_memory(&ptr, size);

  if (err != HSA_STATUS_SUCCESS) {
    DP("Error in tgt_rtl_data_lock\n");
    return nullptr;
  }

  DP("Tgt lock data %ld bytes, (tgt:%016llx).\n", size,
     (long long unsigned)(Elf64_Addr)ptr);

  return ptr;
}

void __tgt_rtl_data_unlock(int DeviceId, void *TgtPtr) {
  assert(DeviceId < DeviceInfo().NumberOfDevices && "Device ID too large");
  hsa_status_t err = HSA_STATUS_SUCCESS;

  err = unlock_memory(TgtPtr);

  if (err != HSA_STATUS_SUCCESS)
    DP("Error in tgt_rtl_data_unlock\n");

  DP("Tgt unlock data (tgt:%016llx).\n",
     (long long unsigned)(Elf64_Addr)TgtPtr);
}

int32_t __tgt_rtl_data_submit(int DeviceId, void *tgt_ptr, void *hst_ptr,
                              int64_t size) {
  assert(DeviceId < DeviceInfo().NumberOfDevices && "Device ID too large");
  AMDGPUAsyncInfoDataTy AsyncData;
  int32_t rc = dataSubmit(DeviceId, tgt_ptr, hst_ptr, size, AsyncData);
  if (rc != OFFLOAD_SUCCESS)
    return OFFLOAD_FAIL;

  AsyncData.waitToComplete();
  AsyncData.releaseResources();

  return rc;
}

int32_t __tgt_rtl_data_submit_async(int DeviceId, void *TgtPtr, void *HstPtr,
                                    int64_t Size, __tgt_async_info *AsyncInfo) {
  assert(DeviceId < DeviceInfo().NumberOfDevices && "Device ID too large");
  if (AsyncInfo) {
    initAsyncInfo(AsyncInfo);
    AMDGPUAsyncInfoDataTy AsyncData;
    int32_t rc = dataSubmit(DeviceId, TgtPtr, HstPtr, Size, AsyncData);
    reinterpret_cast<AMDGPUAsyncInfoQueueTy *>(AsyncInfo->Queue)
        ->addMapEnteringInfo(std::move(AsyncData));
    return rc;
  } else {
    return __tgt_rtl_data_submit(DeviceId, TgtPtr, HstPtr, Size);
  }
  return __tgt_rtl_data_submit(DeviceId, TgtPtr, HstPtr, Size);
}

int32_t __tgt_rtl_data_retrieve(int DeviceId, void *hst_ptr, void *tgt_ptr,
                                int64_t size) {
  assert(DeviceId < DeviceInfo().NumberOfDevices && "Device ID too large");
  AMDGPUAsyncInfoDataTy AsyncData;
  int32_t rc = dataRetrieve(DeviceId, hst_ptr, tgt_ptr, size, AsyncData);
  if (rc != OFFLOAD_SUCCESS)
    return OFFLOAD_FAIL;

  AsyncData.waitToComplete();
  AsyncData.releaseResources();
  return rc;
}

int32_t __tgt_rtl_data_retrieve_async(int DeviceId, void *HstPtr, void *TgtPtr,
                                      int64_t Size,
                                      __tgt_async_info *AsyncInfo) {
  assert(DeviceId < DeviceInfo().NumberOfDevices && "Device ID too large");
  if (AsyncInfo) {
    initAsyncInfo(AsyncInfo);
    AMDGPUAsyncInfoDataTy AsyncData;
    AMDGPUAsyncInfoQueueTy *AsyncInfoQueue =
        reinterpret_cast<AMDGPUAsyncInfoQueueTy *>(AsyncInfo->Queue);

    // if data retrieve call is part of target region, wait for kernel to
    // complete
    if (AsyncInfoQueue->needToWaitForKernel())
      AsyncInfoQueue->getKernelInfo().waitToComplete();
    // OMPT_END for kernel goes here

    int32_t RC = dataRetrieve(DeviceId, HstPtr, TgtPtr, Size, AsyncData);
    reinterpret_cast<AMDGPUAsyncInfoQueueTy *>(AsyncInfo->Queue)
        ->addMapExitingInfo(std::move(AsyncData));
    return RC;
  } else
    return __tgt_rtl_data_retrieve(DeviceId, HstPtr, TgtPtr, Size);
}

int32_t __tgt_rtl_data_delete(int DeviceId, void *TgtPtr) {
  assert(DeviceId < DeviceInfo().NumberOfDevices && "Device ID too large");
  // We don't have HSA-profiling timestamps for device delete, so just get the
  // start and end system timestamps for OMPT
  OmptTimestampRAII DeleteTimestamp;
  return DeviceInfo().DeviceAllocators[DeviceId].dev_free(TgtPtr);
}

int32_t __tgt_rtl_run_target_team_region(int32_t DeviceId, void *TgtEntryPtr,
                                         void **TgtArgs, ptrdiff_t *TgtOffsets,
                                         int32_t ArgNum, int32_t NumTeams,
                                         int32_t ThreadLimit,
                                         uint64_t LoopTripcount) {

  AMDGPUAsyncInfoQueueTy AsyncInfo;
  DeviceInfo().LoadRunLock.lock_shared();
  int32_t Res = runRegionLocked(DeviceId, TgtEntryPtr, TgtArgs, TgtOffsets,
                                ArgNum, NumTeams, ThreadLimit, LoopTripcount,
				AsyncInfo);

  DeviceInfo().LoadRunLock.unlock_shared();
  AsyncInfo.waitForKernelCompletion();
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
  AMDGPUAsyncInfoQueueTy *AsyncInfoQueue =
      reinterpret_cast<AMDGPUAsyncInfoQueueTy *>(AsyncInfo->Queue);

  DeviceInfo().LoadRunLock.lock_shared();
  int32_t Res = runRegionLocked(DeviceId, TgtEntryPtr, TgtArgs, TgtOffsets,
                      ArgNum, NumTeams, ThreadLimit, LoopTripcount, *AsyncInfoQueue);

  DeviceInfo().LoadRunLock.unlock_shared();
  return Res;
}

int32_t __tgt_rtl_run_target_region_async(int32_t DeviceId, void *TgtEntryPtr,
                                          void **TgtArgs, ptrdiff_t *TgtOffsets,
                                          int32_t ArgNum,
                                          __tgt_async_info *AsyncInfo) {
  assert(AsyncInfo && "AsyncInfo is nullptr");
  initAsyncInfo(AsyncInfo);

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
    AMDGPUAsyncInfoQueueTy *AMDGPUAsyncInfoQueue =
        reinterpret_cast<AMDGPUAsyncInfoQueueTy *>(AsyncInfo->Queue);
    AMDGPUAsyncInfoQueue->synchronize();
    finiAsyncInfo(AsyncInfo);
  }
  return OFFLOAD_SUCCESS;
}

// Register mapped or allocated memory (with omp_target_alloc or omp_alloc)
// as coarse grain
// \arg ptr is the base pointer of the region to be registered as coarse grain
// \arg size is the size of the memory region to be registered as coarse grain
int __tgt_rtl_set_coarse_grain_mem_region(void *ptr, int64_t size) {
  // track coarse grain memory pages in local table
  coarse_grain_mem_tab->insert((const uintptr_t)ptr, size);

  // Instruct ROCr that the [ptr, ptr+size-1] pages are
  // coarse grain
  hsa_amd_svm_attribute_pair_t tt;
  tt.attribute = HSA_AMD_SVM_ATTRIB_GLOBAL_FLAG;
  tt.value = HSA_AMD_SVM_GLOBAL_FLAG_COARSE_GRAINED;
  hsa_status_t err = hsa_amd_svm_attributes_set(ptr, size, &tt, 1);
  if (err != HSA_STATUS_SUCCESS) {
    return OFFLOAD_FAIL;
  }

  return OFFLOAD_SUCCESS;
}

// Query if [ptr, ptr+size] belongs to coarse grain memory region
int32_t __tgt_rtl_query_coarse_grain_mem_region(const void *ptr, int64_t size) {
  // if the table is not yet allocated, it means we have not yet gone through
  // an OpenMP pragma or API that would provoke intialization of the RTL
  if (!coarse_grain_mem_tab)
    return 0;

  return coarse_grain_mem_tab->contains((const uintptr_t)ptr, size);
}

// Make ptr accessible by all agents
int32_t __tgt_rtl_enable_access_to_all_agents(const void *ptr, int32_t) {
  if (!ptr)
    return OFFLOAD_FAIL;
  hsa_status_t err = hsa_amd_agents_allow_access(
      DeviceInfo().HSAAgents.size(), DeviceInfo().HSAAgents.data(), nullptr, ptr);
  if (err != HSA_STATUS_SUCCESS)
    return OFFLOAD_FAIL;
  return OFFLOAD_SUCCESS;
}
}

extern "C" {
// following are some utility functions used by hostrpc
hsa_status_t host_malloc(void **mem, size_t size) {
  return core::Runtime::HostMalloc(mem, size,
                                   DeviceInfo().HostFineGrainedMemoryPool);
}

hsa_status_t device_malloc(void **mem, size_t size, int DeviceId) {
  hsa_amd_memory_pool_t MemoryPool = DeviceInfo().getDeviceMemoryPool(DeviceId);
  return hsa_amd_memory_pool_allocate(MemoryPool, size, 0, mem);
}

hsa_status_t lock_memory(void **mem, size_t size) {
  void *lockedPtr = nullptr;
  hsa_status_t err = HSA_STATUS_SUCCESS;

  if (already_locked(*mem, &err, nullptr))
    return HSA_STATUS_SUCCESS;

  err = hsa_amd_memory_lock(*mem, size, nullptr, 0, (void **)&lockedPtr);
  if (err != HSA_STATUS_SUCCESS)
    return err;

  *mem = lockedPtr;
  return err;
}

hsa_status_t unlock_memory(void *mem) {
  hsa_status_t err = HSA_STATUS_SUCCESS;
  if (already_locked(mem, &err, nullptr))
    err = hsa_amd_memory_unlock(mem);
  return err;
}

hsa_status_t impl_free(void *mem) { return core::Runtime::Memfree(mem); }

hsa_status_t ftn_assign_wrapper(void *arg0, void *arg1, void *arg2, void *arg3,
                                void *arg4) {
  return core::Runtime::FtnAssignWrapper(arg0, arg1, arg2, arg3, arg4);
}
// This method is only used by hostrpc demo
hsa_status_t impl_memcpy_no_signal(void *dest, void *src, size_t size,
                                   int host2Device) {
  hsa_signal_t sig;
  hsa_status_t err = hsa_signal_create(0, 0, NULL, &sig);
  if (err != HSA_STATUS_SUCCESS) {
    return err;
  }

  const int deviceId = 0;
  hsa_agent_t device_agent = DeviceInfo().HSAAgents[deviceId];
  auto MemoryPool = DeviceInfo().HostFineGrainedMemoryPool;
  hsa_status_t r;
  bool userLocked;
  if (host2Device)
    r = impl_memcpy_h2d(sig, dest, src, size, device_agent, MemoryPool,
                        &userLocked);
  else
    r = impl_memcpy_d2h(sig, dest, src, size, device_agent, MemoryPool,
                        &userLocked);

  hsa_status_t rc = hsa_signal_destroy(sig);

  if (r != HSA_STATUS_SUCCESS) {
    return r;
  }
  if (rc != HSA_STATUS_SUCCESS) {
    return rc;
  }

  return HSA_STATUS_SUCCESS;
}

void __tgt_rtl_print_device_info(int32_t DeviceId) {
  // TODO: Assertion to see if DeviceId is correct
  // NOTE: We don't need to set context for print device info.

  DeviceInfo().printDeviceInfo(DeviceId, DeviceInfo().HSAAgents[DeviceId]);
}

} // extern "C"
