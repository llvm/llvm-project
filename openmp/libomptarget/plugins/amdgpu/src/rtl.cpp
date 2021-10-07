//===--- amdgpu/src/rtl.cpp --------------------------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RTL for hsa machine
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

#include "interop_hsa.h"
#include "impl_runtime.h"

#include "internal.h"
#include "rt.h"

#include "DeviceEnvironment.h"
#include "get_elf_mach_gfx_name.h"
#include "omptargetplugin.h"
#include "print_tracing.h"

#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Frontend/OpenMP/OMPGridValues.h"

// hostrpc interface, FIXME: consider moving to its own include these are
// statically linked into amdgpu/plugin if present from hostrpc_services.a,
// linked as --whole-archive to override the weak symbols that are used to
// implement a fallback for toolchains that do not yet have a hostrpc library.
extern "C" {
unsigned long hostrpc_assign_buffer(hsa_agent_t agent, hsa_queue_t *this_Q,
                                    uint32_t device_id);
hsa_status_t hostrpc_init();
hsa_status_t hostrpc_terminate();

__attribute__((weak)) hsa_status_t hostrpc_init() { return HSA_STATUS_SUCCESS; }
__attribute__((weak)) hsa_status_t hostrpc_terminate() {
  return HSA_STATUS_SUCCESS;
}
__attribute__((weak)) unsigned long
hostrpc_assign_buffer(hsa_agent_t, hsa_queue_t *, uint32_t device_id) {
  DP("Warning: Attempting to assign hostrpc to device %u, but hostrpc library "
     "missing\n",
     device_id);
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
template <typename C> hsa_status_t iterate_agents(C cb) {
  auto L = [](hsa_agent_t agent, void *data) -> hsa_status_t {
    C *unwrapped = static_cast<C *>(data);
    return (*unwrapped)(agent);
  };
  return hsa_iterate_agents(L, static_cast<void *>(&cb));
}

template <typename C>
hsa_status_t amd_agent_iterate_memory_pools(hsa_agent_t Agent, C cb) {
  auto L = [](hsa_amd_memory_pool_t MemoryPool, void *data) -> hsa_status_t {
    C *unwrapped = static_cast<C *>(data);
    return (*unwrapped)(MemoryPool);
  };

  return hsa_amd_agent_iterate_memory_pools(Agent, L, static_cast<void *>(&cb));
}

} // namespace hsa

/// Keep entries table per device
struct FuncOrGblEntryTy {
  __tgt_target_table Table;
  std::vector<__tgt_offload_entry> Entries;
};

struct KernelArgPool {
private:
  static pthread_mutex_t mutex;

public:
  uint32_t kernarg_segment_size;
  void *kernarg_region = nullptr;
  std::queue<int> free_kernarg_segments;

  uint32_t kernarg_size_including_implicit() {
    return kernarg_segment_size + sizeof(impl_implicit_args_t);
  }

  ~KernelArgPool() {
    if (kernarg_region) {
      auto r = hsa_amd_memory_pool_free(kernarg_region);
      if (r != HSA_STATUS_SUCCESS) {
        DP("hsa_amd_memory_pool_free failed: %s\n", get_error_string(r));
      }
    }
  }

  // Can't really copy or move a mutex
  KernelArgPool() = default;
  KernelArgPool(const KernelArgPool &) = delete;
  KernelArgPool(KernelArgPool &&) = delete;

  KernelArgPool(uint32_t kernarg_segment_size,
                hsa_amd_memory_pool_t &memory_pool)
      : kernarg_segment_size(kernarg_segment_size) {

    // impl uses one pool per kernel for all gpus, with a fixed upper size
    // preserving that exact scheme here, including the queue<int>

    hsa_status_t err = hsa_amd_memory_pool_allocate(
        memory_pool, kernarg_size_including_implicit() * MAX_NUM_KERNELS, 0,
        &kernarg_region);

    if (err != HSA_STATUS_SUCCESS) {
      DP("hsa_amd_memory_pool_allocate failed: %s\n", get_error_string(err));
      kernarg_region = nullptr; // paranoid
      return;
    }

    err = core::allow_access_to_all_gpu_agents(kernarg_region);
    if (err != HSA_STATUS_SUCCESS) {
      DP("hsa allow_access_to_all_gpu_agents failed: %s\n",
         get_error_string(err));
      auto r = hsa_amd_memory_pool_free(kernarg_region);
      if (r != HSA_STATUS_SUCCESS) {
        // if free failed, can't do anything more to resolve it
        DP("hsa memory poll free failed: %s\n", get_error_string(err));
      }
      kernarg_region = nullptr;
      return;
    }

    for (int i = 0; i < MAX_NUM_KERNELS; i++) {
      free_kernarg_segments.push(i);
    }
  }

  void *allocate(uint64_t arg_num) {
    assert((arg_num * sizeof(void *)) == kernarg_segment_size);
    lock l(&mutex);
    void *res = nullptr;
    if (!free_kernarg_segments.empty()) {

      int free_idx = free_kernarg_segments.front();
      res = static_cast<void *>(static_cast<char *>(kernarg_region) +
                                (free_idx * kernarg_size_including_implicit()));
      assert(free_idx == pointer_to_index(res));
      free_kernarg_segments.pop();
    }
    return res;
  }

  void deallocate(void *ptr) {
    lock l(&mutex);
    int idx = pointer_to_index(ptr);
    free_kernarg_segments.push(idx);
  }

private:
  int pointer_to_index(void *ptr) {
    ptrdiff_t bytes =
        static_cast<char *>(ptr) - static_cast<char *>(kernarg_region);
    assert(bytes >= 0);
    assert(bytes % kernarg_size_including_implicit() == 0);
    return bytes / kernarg_size_including_implicit();
  }
  struct lock {
    lock(pthread_mutex_t *m) : m(m) { pthread_mutex_lock(m); }
    ~lock() { pthread_mutex_unlock(m); }
    pthread_mutex_t *m;
  };
};
pthread_mutex_t KernelArgPool::mutex = PTHREAD_MUTEX_INITIALIZER;

std::unordered_map<std::string /*kernel*/, std::unique_ptr<KernelArgPool>>
    KernelArgPoolMap;

/// Use a single entity to encode a kernel and a set of flags
struct KernelTy {
  llvm::omp::OMPTgtExecModeFlags ExecutionMode;
  int16_t ConstWGSize;
  int32_t device_id;
  void *CallStackAddr = nullptr;
  const char *Name;

  KernelTy(llvm::omp::OMPTgtExecModeFlags _ExecutionMode, int16_t _ConstWGSize,
           int32_t _device_id, void *_CallStackAddr, const char *_Name,
           uint32_t _kernarg_segment_size,
           hsa_amd_memory_pool_t &KernArgMemoryPool)
      : ExecutionMode(_ExecutionMode), ConstWGSize(_ConstWGSize),
        device_id(_device_id), CallStackAddr(_CallStackAddr), Name(_Name) {
    DP("Construct kernelinfo: ExecMode %d\n", ExecutionMode);

    std::string N(_Name);
    if (KernelArgPoolMap.find(N) == KernelArgPoolMap.end()) {
      KernelArgPoolMap.insert(
          std::make_pair(N, std::unique_ptr<KernelArgPool>(new KernelArgPool(
                                _kernarg_segment_size, KernArgMemoryPool))));
    }
  }
};

/// List that contains all the kernels.
/// FIXME: we may need this to be per device and per library.
std::list<KernelTy> KernelsList;

template <typename Callback> static hsa_status_t FindAgents(Callback CB) {

  hsa_status_t err =
      hsa::iterate_agents([&](hsa_agent_t agent) -> hsa_status_t {
        hsa_device_type_t device_type;
        // get_info fails iff HSA runtime not yet initialized
        hsa_status_t err =
            hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);

        if (err != HSA_STATUS_SUCCESS) {
          if (print_kernel_trace > 0)
            DP("rtl.cpp: err %s\n", get_error_string(err));

          return err;
        }

        CB(device_type, agent);
        return HSA_STATUS_SUCCESS;
      });

  // iterate_agents fails iff HSA runtime not yet initialized
  if (print_kernel_trace > 0 && err != HSA_STATUS_SUCCESS) {
    DP("rtl.cpp: err %s\n", get_error_string(err));
  }

  return err;
}

static void callbackQueue(hsa_status_t status, hsa_queue_t *source,
                          void *data) {
  if (status != HSA_STATUS_SUCCESS) {
    const char *status_string;
    if (hsa_status_string(status, &status_string) != HSA_STATUS_SUCCESS) {
      status_string = "unavailable";
    }
    DP("[%s:%d] GPU error in queue %p %d (%s)\n", __FILE__, __LINE__, source,
       status, status_string);
    abort();
  }
}

namespace core {
namespace {
void packet_store_release(uint32_t *packet, uint16_t header, uint16_t rest) {
  __atomic_store_n(packet, header | (rest << 16), __ATOMIC_RELEASE);
}

uint16_t create_header() {
  uint16_t header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
  header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;
  return header;
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

  hsa_status_t err;
  if ((err = isValidMemoryPool(MemoryPool)) != HSA_STATUS_SUCCESS) {
    return err;
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

  bool success() { return S == HSA_STATUS_SUCCESS; }
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

/// Class containing all the device information
class RTLDeviceInfoTy {
  HSALifetime HSA; // First field => constructed first and destructed last
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
  std::shared_timed_mutex load_run_lock;

  int NumberOfDevices = 0;

  // GPU devices
  std::vector<hsa_agent_t> HSAAgents;
  std::vector<std::unique_ptr<hsa_queue_t, QueueDeleter>>
      HSAQueues; // one per gpu

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

  bool hostcall_required = false;

  std::vector<hsa_executable_t> HSAExecutables;

  std::vector<std::map<std::string, atl_kernel_info_t>> KernelInfoTable;
  std::vector<std::map<std::string, atl_symbol_info_t>> SymbolInfoTable;

  hsa_amd_memory_pool_t KernArgPool;

  // fine grained memory pool for host allocations
  hsa_amd_memory_pool_t HostFineGrainedMemoryPool;

  // fine and coarse-grained memory pools per offloading device
  std::vector<hsa_amd_memory_pool_t> DeviceFineGrainedMemoryPools;
  std::vector<hsa_amd_memory_pool_t> DeviceCoarseGrainedMemoryPools;

  struct implFreePtrDeletor {
    void operator()(void *p) {
      core::Runtime::Memfree(p); // ignore failure to free
    }
  };

  // device_State shared across loaded binaries, error if inconsistent size
  std::vector<std::pair<std::unique_ptr<void, implFreePtrDeletor>, uint64_t>>
      deviceStateStore;

  static const unsigned HardTeamLimit =
      (1 << 16) - 1; // 64K needed to fit in uint16
  static const int DefaultNumTeams = 128;

  // These need to be per-device since different devices can have different
  // wave sizes, but are currently the same number for each so that refactor
  // can be postponed.
  static_assert(getGridValue<32>().GV_Max_Teams ==
                    getGridValue<64>().GV_Max_Teams,
                "");
  static const int Max_Teams = getGridValue<64>().GV_Max_Teams;

  static_assert(getGridValue<32>().GV_Max_WG_Size ==
                    getGridValue<64>().GV_Max_WG_Size,
                "");
  static const int Max_WG_Size = getGridValue<64>().GV_Max_WG_Size;

  static_assert(getGridValue<32>().GV_Default_WG_Size ==
                    getGridValue<64>().GV_Default_WG_Size,
                "");
  static const int Default_WG_Size = getGridValue<64>().GV_Default_WG_Size;

  using MemcpyFunc = hsa_status_t (*)(hsa_signal_t, void *, const void *,
                                      size_t size, hsa_agent_t,
                                      hsa_amd_memory_pool_t);
  hsa_status_t freesignalpool_memcpy(void *dest, const void *src, size_t size,
                                     MemcpyFunc Func, int32_t deviceId) {
    hsa_agent_t agent = HSAAgents[deviceId];
    hsa_signal_t s = FreeSignalPool.pop();
    if (s.handle == 0) {
      return HSA_STATUS_ERROR;
    }
    hsa_status_t r = Func(s, dest, src, size, agent, HostFineGrainedMemoryPool);
    FreeSignalPool.push(s);
    return r;
  }

  hsa_status_t freesignalpool_memcpy_d2h(void *dest, const void *src,
                                         size_t size, int32_t deviceId) {
    return freesignalpool_memcpy(dest, src, size, impl_memcpy_d2h, deviceId);
  }

  hsa_status_t freesignalpool_memcpy_h2d(void *dest, const void *src,
                                         size_t size, int32_t deviceId) {
    return freesignalpool_memcpy(dest, src, size, impl_memcpy_h2d, deviceId);
  }

  // Record entry point associated with device
  void addOffloadEntry(int32_t device_id, __tgt_offload_entry entry) {
    assert(device_id < (int32_t)FuncGblEntries.size() &&
           "Unexpected device id!");
    FuncOrGblEntryTy &E = FuncGblEntries[device_id].back();

    E.Entries.push_back(entry);
  }

  // Return true if the entry is associated with device
  bool findOffloadEntry(int32_t device_id, void *addr) {
    assert(device_id < (int32_t)FuncGblEntries.size() &&
           "Unexpected device id!");
    FuncOrGblEntryTy &E = FuncGblEntries[device_id].back();

    for (auto &it : E.Entries) {
      if (it.addr == addr)
        return true;
    }

    return false;
  }

  // Return the pointer to the target entries table
  __tgt_target_table *getOffloadEntriesTable(int32_t device_id) {
    assert(device_id < (int32_t)FuncGblEntries.size() &&
           "Unexpected device id!");
    FuncOrGblEntryTy &E = FuncGblEntries[device_id].back();

    int32_t size = E.Entries.size();

    // Table is empty
    if (!size)
      return 0;

    __tgt_offload_entry *begin = &E.Entries[0];
    __tgt_offload_entry *end = &E.Entries[size - 1];

    // Update table info according to the entries and return the pointer
    E.Table.EntriesBegin = begin;
    E.Table.EntriesEnd = ++end;

    return &E.Table;
  }

  // Clear entries table for a device
  void clearOffloadEntriesTable(int device_id) {
    assert(device_id < (int32_t)FuncGblEntries.size() &&
           "Unexpected device id!");
    FuncGblEntries[device_id].emplace_back();
    FuncOrGblEntryTy &E = FuncGblEntries[device_id].back();
    // KernelArgPoolMap.clear();
    E.Entries.clear();
    E.Table.EntriesBegin = E.Table.EntriesEnd = 0;
  }

  hsa_status_t addDeviceMemoryPool(hsa_amd_memory_pool_t MemoryPool,
                                   int DeviceId) {
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
    for (int DeviceId = 0; DeviceId < Agents.size(); DeviceId++) {
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

  hsa_amd_memory_pool_t getDeviceMemoryPool(int DeviceId) {
    assert(DeviceId >= 0 && DeviceId < DeviceCoarseGrainedMemoryPools.size() &&
           "Invalid device Id");
    return DeviceCoarseGrainedMemoryPools[DeviceId];
  }

  hsa_amd_memory_pool_t getHostMemoryPool() {
    return HostFineGrainedMemoryPool;
  }

  static int readEnvElseMinusOne(const char *Env) {
    const char *envStr = getenv(Env);
    int res = -1;
    if (envStr) {
      res = std::stoi(envStr);
      DP("Parsed %s=%d\n", Env, res);
    }
    return res;
  }

  RTLDeviceInfoTy() {
    DP("Start initializing " GETNAME(TARGET_NAME) "\n");

    // LIBOMPTARGET_KERNEL_TRACE provides a kernel launch trace to stderr
    // anytime. You do not need a debug library build.
    //  0 => no tracing
    //  1 => tracing dispatch only
    // >1 => verbosity increase

    if (!HSA.success()) {
      DP("Error when initializing HSA in " GETNAME(TARGET_NAME) "\n");
      return;
    }

    if (char *envStr = getenv("LIBOMPTARGET_KERNEL_TRACE"))
      print_kernel_trace = atoi(envStr);
    else
      print_kernel_trace = 0;

    hsa_status_t err = core::atl_init_gpu_context();
    if (err != HSA_STATUS_SUCCESS) {
      DP("Error when initializing " GETNAME(TARGET_NAME) "\n");
      return;
    }

    // Init hostcall soon after initializing hsa
    hostrpc_init();

    err = FindAgents([&](hsa_device_type_t DeviceType, hsa_agent_t Agent) {
      if (DeviceType == HSA_DEVICE_TYPE_CPU) {
        CPUAgents.push_back(Agent);
      } else {
        HSAAgents.push_back(Agent);
      }
    });
    if (err != HSA_STATUS_SUCCESS)
      return;

    NumberOfDevices = (int)HSAAgents.size();

    if (NumberOfDevices == 0) {
      DP("There are no devices supporting HSA.\n");
      return;
    } else {
      DP("There are %d devices supporting HSA.\n", NumberOfDevices);
    }

    // Init the device info
    HSAQueues.resize(NumberOfDevices);
    FuncGblEntries.resize(NumberOfDevices);
    ThreadsPerGroup.resize(NumberOfDevices);
    ComputeUnits.resize(NumberOfDevices);
    GPUName.resize(NumberOfDevices);
    GroupsPerDevice.resize(NumberOfDevices);
    WarpSize.resize(NumberOfDevices);
    NumTeams.resize(NumberOfDevices);
    NumThreads.resize(NumberOfDevices);
    deviceStateStore.resize(NumberOfDevices);
    KernelInfoTable.resize(NumberOfDevices);
    SymbolInfoTable.resize(NumberOfDevices);
    DeviceCoarseGrainedMemoryPools.resize(NumberOfDevices);
    DeviceFineGrainedMemoryPools.resize(NumberOfDevices);

    err = setupDevicePools(HSAAgents);
    if (err != HSA_STATUS_SUCCESS) {
      DP("Setup for Device Memory Pools failed\n");
      return;
    }

    err = setupHostMemoryPools(CPUAgents);
    if (err != HSA_STATUS_SUCCESS) {
      DP("Setup for Host Memory Pools failed\n");
      return;
    }

    for (int i = 0; i < NumberOfDevices; i++) {
      uint32_t queue_size = 0;
      {
        hsa_status_t err = hsa_agent_get_info(
            HSAAgents[i], HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size);
        if (err != HSA_STATUS_SUCCESS) {
          DP("HSA query QUEUE_MAX_SIZE failed for agent %d\n", i);
          return;
        }
        enum { MaxQueueSize = 4096 };
        if (queue_size > MaxQueueSize) {
          queue_size = MaxQueueSize;
        }
      }

      {
        hsa_queue_t *Q = nullptr;
        hsa_status_t rc =
            hsa_queue_create(HSAAgents[i], queue_size, HSA_QUEUE_TYPE_MULTI,
                             callbackQueue, NULL, UINT32_MAX, UINT32_MAX, &Q);
        if (rc != HSA_STATUS_SUCCESS) {
          DP("Failed to create HSA queue %d\n", i);
          return;
        }
        HSAQueues[i].reset(Q);
      }

      deviceStateStore[i] = {nullptr, 0};
    }

    for (int i = 0; i < NumberOfDevices; i++) {
      ThreadsPerGroup[i] = RTLDeviceInfoTy::Default_WG_Size;
      GroupsPerDevice[i] = RTLDeviceInfoTy::DefaultNumTeams;
      ComputeUnits[i] = 1;
      DP("Device %d: Initial groupsPerDevice %d & threadsPerGroup %d\n", i,
         GroupsPerDevice[i], ThreadsPerGroup[i]);
    }

    // Get environment variables regarding teams
    Env.TeamLimit = readEnvElseMinusOne("OMP_TEAM_LIMIT");
    Env.NumTeams = readEnvElseMinusOne("OMP_NUM_TEAMS");
    Env.MaxTeamsDefault = readEnvElseMinusOne("OMP_MAX_TEAMS_DEFAULT");
    Env.TeamThreadLimit = readEnvElseMinusOne("OMP_TEAMS_THREAD_LIMIT");

    // Default state.
    RequiresFlags = OMP_REQ_UNDEFINED;

    ConstructionSucceeded = true;
  }

  ~RTLDeviceInfoTy() {
    DP("Finalizing the " GETNAME(TARGET_NAME) " DeviceInfo.\n");
    if (!HSA.success()) {
      // Then none of these can have been set up and they can't be torn down
      return;
    }
    // Run destructors on types that use HSA before
    // impl_finalize removes access to it
    deviceStateStore.clear();
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

static RTLDeviceInfoTy DeviceInfo;

namespace {

int32_t dataRetrieve(int32_t DeviceId, void *HstPtr, void *TgtPtr, int64_t Size,
                     __tgt_async_info *AsyncInfo) {
  assert(AsyncInfo && "AsyncInfo is nullptr");
  assert(DeviceId < DeviceInfo.NumberOfDevices && "Device ID too large");
  // Return success if we are not copying back to host from target.
  if (!HstPtr)
    return OFFLOAD_SUCCESS;
  hsa_status_t err;
  DP("Retrieve data %ld bytes, (tgt:%016llx) -> (hst:%016llx).\n", Size,
     (long long unsigned)(Elf64_Addr)TgtPtr,
     (long long unsigned)(Elf64_Addr)HstPtr);

  err = DeviceInfo.freesignalpool_memcpy_d2h(HstPtr, TgtPtr, (size_t)Size,
                                             DeviceId);

  if (err != HSA_STATUS_SUCCESS) {
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
  hsa_status_t err;
  assert(DeviceId < DeviceInfo.NumberOfDevices && "Device ID too large");
  // Return success if we are not doing host to target.
  if (!HstPtr)
    return OFFLOAD_SUCCESS;

  DP("Submit data %ld bytes, (hst:%016llx) -> (tgt:%016llx).\n", Size,
     (long long unsigned)(Elf64_Addr)HstPtr,
     (long long unsigned)(Elf64_Addr)TgtPtr);
  err = DeviceInfo.freesignalpool_memcpy_h2d(TgtPtr, HstPtr, (size_t)Size,
                                             DeviceId);
  if (err != HSA_STATUS_SUCCESS) {
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

bool elf_machine_id_is_amdgcn(__tgt_device_image *image) {
  const uint16_t amdgcnMachineID = 224; // EM_AMDGPU may not be in system elf.h
  int32_t r = elf_check_machine(image, amdgcnMachineID);
  if (!r) {
    DP("Supported machine ID not found\n");
  }
  return r;
}

uint32_t elf_e_flags(__tgt_device_image *image) {
  char *img_begin = (char *)image->ImageStart;
  size_t img_size = (char *)image->ImageEnd - img_begin;

  Elf *e = elf_memory(img_begin, img_size);
  if (!e) {
    DP("Unable to get ELF handle: %s!\n", elf_errmsg(-1));
    return 0;
  }

  Elf64_Ehdr *eh64 = elf64_getehdr(e);

  if (!eh64) {
    DP("Unable to get machine ID from ELF file!\n");
    elf_end(e);
    return 0;
  }

  uint32_t Flags = eh64->e_flags;

  elf_end(e);
  DP("ELF Flags: 0x%x\n", Flags);
  return Flags;
}
} // namespace

int32_t __tgt_rtl_is_valid_binary(__tgt_device_image *image) {
  return elf_machine_id_is_amdgcn(image);
}

int __tgt_rtl_number_of_devices() {
  // If the construction failed, no methods are safe to call
  if (DeviceInfo.ConstructionSucceeded) {
    return DeviceInfo.NumberOfDevices;
  } else {
    DP("AMDGPU plugin construction failed. Zero devices available\n");
    return 0;
  }
}

int64_t __tgt_rtl_init_requires(int64_t RequiresFlags) {
  DP("Init requires flags to %ld\n", RequiresFlags);
  DeviceInfo.RequiresFlags = RequiresFlags;
  return RequiresFlags;
}

namespace {
template <typename T> bool enforce_upper_bound(T *value, T upper) {
  bool changed = *value > upper;
  if (changed) {
    *value = upper;
  }
  return changed;
}
} // namespace

int32_t __tgt_rtl_init_device(int device_id) {
  hsa_status_t err;

  // this is per device id init
  DP("Initialize the device id: %d\n", device_id);

  hsa_agent_t agent = DeviceInfo.HSAAgents[device_id];

  // Get number of Compute Unit
  uint32_t compute_units = 0;
  err = hsa_agent_get_info(
      agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT,
      &compute_units);
  if (err != HSA_STATUS_SUCCESS) {
    DeviceInfo.ComputeUnits[device_id] = 1;
    DP("Error getting compute units : settiing to 1\n");
  } else {
    DeviceInfo.ComputeUnits[device_id] = compute_units;
    DP("Using %d compute unis per grid\n", DeviceInfo.ComputeUnits[device_id]);
  }

  char GetInfoName[64]; // 64 max size returned by get info
  err = hsa_agent_get_info(agent, (hsa_agent_info_t)HSA_AGENT_INFO_NAME,
                           (void *)GetInfoName);
  if (err)
    DeviceInfo.GPUName[device_id] = "--unknown gpu--";
  else {
    DeviceInfo.GPUName[device_id] = GetInfoName;
  }

  if (print_kernel_trace & STARTUP_DETAILS)
    DP("Device#%-2d CU's: %2d %s\n", device_id,
       DeviceInfo.ComputeUnits[device_id],
       DeviceInfo.GPUName[device_id].c_str());

  // Query attributes to determine number of threads/block and blocks/grid.
  uint16_t workgroup_max_dim[3];
  err = hsa_agent_get_info(agent, HSA_AGENT_INFO_WORKGROUP_MAX_DIM,
                           &workgroup_max_dim);
  if (err != HSA_STATUS_SUCCESS) {
    DeviceInfo.GroupsPerDevice[device_id] = RTLDeviceInfoTy::DefaultNumTeams;
    DP("Error getting grid dims: num groups : %d\n",
       RTLDeviceInfoTy::DefaultNumTeams);
  } else if (workgroup_max_dim[0] <= RTLDeviceInfoTy::HardTeamLimit) {
    DeviceInfo.GroupsPerDevice[device_id] = workgroup_max_dim[0];
    DP("Using %d ROCm blocks per grid\n",
       DeviceInfo.GroupsPerDevice[device_id]);
  } else {
    DeviceInfo.GroupsPerDevice[device_id] = RTLDeviceInfoTy::HardTeamLimit;
    DP("Max ROCm blocks per grid %d exceeds the hard team limit %d, capping "
       "at the hard limit\n",
       workgroup_max_dim[0], RTLDeviceInfoTy::HardTeamLimit);
  }

  // Get thread limit
  hsa_dim3_t grid_max_dim;
  err = hsa_agent_get_info(agent, HSA_AGENT_INFO_GRID_MAX_DIM, &grid_max_dim);
  if (err == HSA_STATUS_SUCCESS) {
    DeviceInfo.ThreadsPerGroup[device_id] =
        reinterpret_cast<uint32_t *>(&grid_max_dim)[0] /
        DeviceInfo.GroupsPerDevice[device_id];

    if (DeviceInfo.ThreadsPerGroup[device_id] == 0) {
      DeviceInfo.ThreadsPerGroup[device_id] = RTLDeviceInfoTy::Max_WG_Size;
      DP("Default thread limit: %d\n", RTLDeviceInfoTy::Max_WG_Size);
    } else if (enforce_upper_bound(&DeviceInfo.ThreadsPerGroup[device_id],
                                   RTLDeviceInfoTy::Max_WG_Size)) {
      DP("Capped thread limit: %d\n", RTLDeviceInfoTy::Max_WG_Size);
    } else {
      DP("Using ROCm Queried thread limit: %d\n",
         DeviceInfo.ThreadsPerGroup[device_id]);
    }
  } else {
    DeviceInfo.ThreadsPerGroup[device_id] = RTLDeviceInfoTy::Max_WG_Size;
    DP("Error getting max block dimension, use default:%d \n",
       RTLDeviceInfoTy::Max_WG_Size);
  }

  // Get wavefront size
  uint32_t wavefront_size = 0;
  err =
      hsa_agent_get_info(agent, HSA_AGENT_INFO_WAVEFRONT_SIZE, &wavefront_size);
  if (err == HSA_STATUS_SUCCESS) {
    DP("Queried wavefront size: %d\n", wavefront_size);
    DeviceInfo.WarpSize[device_id] = wavefront_size;
  } else {
    // TODO: Burn the wavefront size into the code object
    DP("Warning: Unknown wavefront size, assuming 64\n");
    DeviceInfo.WarpSize[device_id] = 64;
  }

  // Adjust teams to the env variables

  if (DeviceInfo.Env.TeamLimit > 0 &&
      (enforce_upper_bound(&DeviceInfo.GroupsPerDevice[device_id],
                           DeviceInfo.Env.TeamLimit))) {
    DP("Capping max groups per device to OMP_TEAM_LIMIT=%d\n",
       DeviceInfo.Env.TeamLimit);
  }

  // Set default number of teams
  if (DeviceInfo.Env.NumTeams > 0) {
    DeviceInfo.NumTeams[device_id] = DeviceInfo.Env.NumTeams;
    DP("Default number of teams set according to environment %d\n",
       DeviceInfo.Env.NumTeams);
  } else {
    char *TeamsPerCUEnvStr = getenv("OMP_TARGET_TEAMS_PER_PROC");
    int TeamsPerCU = DefaultTeamsPerCU;
    if (TeamsPerCUEnvStr) {
      TeamsPerCU = std::stoi(TeamsPerCUEnvStr);
    }

    DeviceInfo.NumTeams[device_id] =
        TeamsPerCU * DeviceInfo.ComputeUnits[device_id];
    DP("Default number of teams = %d * number of compute units %d\n",
       TeamsPerCU, DeviceInfo.ComputeUnits[device_id]);
  }

  if (enforce_upper_bound(&DeviceInfo.NumTeams[device_id],
                          DeviceInfo.GroupsPerDevice[device_id])) {
    DP("Default number of teams exceeds device limit, capping at %d\n",
       DeviceInfo.GroupsPerDevice[device_id]);
  }

  // Adjust threads to the env variables
  if (DeviceInfo.Env.TeamThreadLimit > 0 &&
      (enforce_upper_bound(&DeviceInfo.NumThreads[device_id],
                           DeviceInfo.Env.TeamThreadLimit))) {
    DP("Capping max number of threads to OMP_TEAMS_THREAD_LIMIT=%d\n",
       DeviceInfo.Env.TeamThreadLimit);
  }

  // Set default number of threads
  DeviceInfo.NumThreads[device_id] = RTLDeviceInfoTy::Default_WG_Size;
  DP("Default number of threads set according to library's default %d\n",
     RTLDeviceInfoTy::Default_WG_Size);
  if (enforce_upper_bound(&DeviceInfo.NumThreads[device_id],
                          DeviceInfo.ThreadsPerGroup[device_id])) {
    DP("Default number of threads exceeds device limit, capping at %d\n",
       DeviceInfo.ThreadsPerGroup[device_id]);
  }

  DP("Device %d: default limit for groupsPerDevice %d & threadsPerGroup %d\n",
     device_id, DeviceInfo.GroupsPerDevice[device_id],
     DeviceInfo.ThreadsPerGroup[device_id]);

  DP("Device %d: wavefront size %d, total threads %d x %d = %d\n", device_id,
     DeviceInfo.WarpSize[device_id], DeviceInfo.ThreadsPerGroup[device_id],
     DeviceInfo.GroupsPerDevice[device_id],
     DeviceInfo.GroupsPerDevice[device_id] *
         DeviceInfo.ThreadsPerGroup[device_id]);

  return OFFLOAD_SUCCESS;
}

namespace {
Elf64_Shdr *find_only_SHT_HASH(Elf *elf) {
  size_t N;
  int rc = elf_getshdrnum(elf, &N);
  if (rc != 0) {
    return nullptr;
  }

  Elf64_Shdr *result = nullptr;
  for (size_t i = 0; i < N; i++) {
    Elf_Scn *scn = elf_getscn(elf, i);
    if (scn) {
      Elf64_Shdr *shdr = elf64_getshdr(scn);
      if (shdr) {
        if (shdr->sh_type == SHT_HASH) {
          if (result == nullptr) {
            result = shdr;
          } else {
            // multiple SHT_HASH sections not handled
            return nullptr;
          }
        }
      }
    }
  }
  return result;
}

const Elf64_Sym *elf_lookup(Elf *elf, char *base, Elf64_Shdr *section_hash,
                            const char *symname) {

  assert(section_hash);
  size_t section_symtab_index = section_hash->sh_link;
  Elf64_Shdr *section_symtab =
      elf64_getshdr(elf_getscn(elf, section_symtab_index));
  size_t section_strtab_index = section_symtab->sh_link;

  const Elf64_Sym *symtab =
      reinterpret_cast<const Elf64_Sym *>(base + section_symtab->sh_offset);

  const uint32_t *hashtab =
      reinterpret_cast<const uint32_t *>(base + section_hash->sh_offset);

  // Layout:
  // nbucket
  // nchain
  // bucket[nbucket]
  // chain[nchain]
  uint32_t nbucket = hashtab[0];
  const uint32_t *bucket = &hashtab[2];
  const uint32_t *chain = &hashtab[nbucket + 2];

  const size_t max = strlen(symname) + 1;
  const uint32_t hash = elf_hash(symname);
  for (uint32_t i = bucket[hash % nbucket]; i != 0; i = chain[i]) {
    char *n = elf_strptr(elf, section_strtab_index, symtab[i].st_name);
    if (strncmp(symname, n, max) == 0) {
      return &symtab[i];
    }
  }

  return nullptr;
}

struct symbol_info {
  void *addr = nullptr;
  uint32_t size = UINT32_MAX;
  uint32_t sh_type = SHT_NULL;
};

int get_symbol_info_without_loading(Elf *elf, char *base, const char *symname,
                                    symbol_info *res) {
  if (elf_kind(elf) != ELF_K_ELF) {
    return 1;
  }

  Elf64_Shdr *section_hash = find_only_SHT_HASH(elf);
  if (!section_hash) {
    return 1;
  }

  const Elf64_Sym *sym = elf_lookup(elf, base, section_hash, symname);
  if (!sym) {
    return 1;
  }

  if (sym->st_size > UINT32_MAX) {
    return 1;
  }

  if (sym->st_shndx == SHN_UNDEF) {
    return 1;
  }

  Elf_Scn *section = elf_getscn(elf, sym->st_shndx);
  if (!section) {
    return 1;
  }

  Elf64_Shdr *header = elf64_getshdr(section);
  if (!header) {
    return 1;
  }

  res->addr = sym->st_value + base;
  res->size = static_cast<uint32_t>(sym->st_size);
  res->sh_type = header->sh_type;
  return 0;
}

int get_symbol_info_without_loading(char *base, size_t img_size,
                                    const char *symname, symbol_info *res) {
  Elf *elf = elf_memory(base, img_size);
  if (elf) {
    int rc = get_symbol_info_without_loading(elf, base, symname, res);
    elf_end(elf);
    return rc;
  }
  return 1;
}

hsa_status_t interop_get_symbol_info(char *base, size_t img_size,
                                     const char *symname, void **var_addr,
                                     uint32_t *var_size) {
  symbol_info si;
  int rc = get_symbol_info_without_loading(base, img_size, symname, &si);
  if (rc == 0) {
    *var_addr = si.addr;
    *var_size = si.size;
    return HSA_STATUS_SUCCESS;
  } else {
    return HSA_STATUS_ERROR;
  }
}

template <typename C>
hsa_status_t module_register_from_memory_to_place(
    std::map<std::string, atl_kernel_info_t> &KernelInfoTable,
    std::map<std::string, atl_symbol_info_t> &SymbolInfoTable,
    void *module_bytes, size_t module_size, int DeviceId, C cb,
    std::vector<hsa_executable_t> &HSAExecutables) {
  auto L = [](void *data, size_t size, void *cb_state) -> hsa_status_t {
    C *unwrapped = static_cast<C *>(cb_state);
    return (*unwrapped)(data, size);
  };
  return core::RegisterModuleFromMemory(
      KernelInfoTable, SymbolInfoTable, module_bytes, module_size,
      DeviceInfo.HSAAgents[DeviceId], L, static_cast<void *>(&cb),
      HSAExecutables);
}
} // namespace

static uint64_t get_device_State_bytes(char *ImageStart, size_t img_size) {
  uint64_t device_State_bytes = 0;
  {
    // If this is the deviceRTL, get the state variable size
    symbol_info size_si;
    int rc = get_symbol_info_without_loading(
        ImageStart, img_size, "omptarget_nvptx_device_State_size", &size_si);

    if (rc == 0) {
      if (size_si.size != sizeof(uint64_t)) {
        DP("Found device_State_size variable with wrong size\n");
        return 0;
      }

      // Read number of bytes directly from the elf
      memcpy(&device_State_bytes, size_si.addr, sizeof(uint64_t));
    }
  }
  return device_State_bytes;
}

static __tgt_target_table *
__tgt_rtl_load_binary_locked(int32_t device_id, __tgt_device_image *image);

static __tgt_target_table *
__tgt_rtl_load_binary_locked(int32_t device_id, __tgt_device_image *image);

__tgt_target_table *__tgt_rtl_load_binary(int32_t device_id,
                                          __tgt_device_image *image) {
  DeviceInfo.load_run_lock.lock();
  __tgt_target_table *res = __tgt_rtl_load_binary_locked(device_id, image);
  DeviceInfo.load_run_lock.unlock();
  return res;
}

struct device_environment {
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

  DeviceEnvironmentTy host_device_env;
  symbol_info si;
  bool valid = false;

  __tgt_device_image *image;
  const size_t img_size;

  device_environment(int device_id, int number_devices,
                     __tgt_device_image *image, const size_t img_size)
      : image(image), img_size(img_size) {

    host_device_env.NumDevices = number_devices;
    host_device_env.DeviceNum = device_id;
    host_device_env.DebugKind = 0;
    host_device_env.DynamicMemSize = 0;
#ifdef OMPTARGET_DEBUG
    if (char *envStr = getenv("LIBOMPTARGET_DEVICE_RTL_DEBUG")) {
      host_device_env.DebugKind = std::stoi(envStr);
    }
#endif

    int rc = get_symbol_info_without_loading((char *)image->ImageStart,
                                             img_size, sym(), &si);
    if (rc != 0) {
      DP("Finding global device environment '%s' - symbol missing.\n", sym());
      return;
    }

    if (si.size > sizeof(host_device_env)) {
      DP("Symbol '%s' has size %u, expected at most %zu.\n", sym(), si.size,
         sizeof(host_device_env));
      return;
    }

    valid = true;
  }

  bool in_image() { return si.sh_type != SHT_NOBITS; }

  hsa_status_t before_loading(void *data, size_t size) {
    if (valid) {
      if (in_image()) {
        DP("Setting global device environment before load (%u bytes)\n",
           si.size);
        uint64_t offset = (char *)si.addr - (char *)image->ImageStart;
        void *pos = (char *)data + offset;
        memcpy(pos, &host_device_env, si.size);
      }
    }
    return HSA_STATUS_SUCCESS;
  }

  hsa_status_t after_loading() {
    if (valid) {
      if (!in_image()) {
        DP("Setting global device environment after load (%u bytes)\n",
           si.size);
        int device_id = host_device_env.DeviceNum;
        auto &SymbolInfo = DeviceInfo.SymbolInfoTable[device_id];
        void *state_ptr;
        uint32_t state_ptr_size;
        hsa_status_t err = interop_hsa_get_symbol_info(
            SymbolInfo, device_id, sym(), &state_ptr, &state_ptr_size);
        if (err != HSA_STATUS_SUCCESS) {
          DP("failed to find %s in loaded image\n", sym());
          return err;
        }

        if (state_ptr_size != si.size) {
          DP("Symbol had size %u before loading, %u after\n", state_ptr_size,
             si.size);
          return HSA_STATUS_ERROR;
        }

        return DeviceInfo.freesignalpool_memcpy_h2d(state_ptr, &host_device_env,
                                                    state_ptr_size, device_id);
      }
    }
    return HSA_STATUS_SUCCESS;
  }
};

static hsa_status_t impl_calloc(void **ret_ptr, size_t size, int DeviceId) {
  uint64_t rounded = 4 * ((size + 3) / 4);
  void *ptr;
  hsa_amd_memory_pool_t MemoryPool = DeviceInfo.getDeviceMemoryPool(DeviceId);
  hsa_status_t err = hsa_amd_memory_pool_allocate(MemoryPool, rounded, 0, &ptr);
  if (err != HSA_STATUS_SUCCESS) {
    return err;
  }

  hsa_status_t rc = hsa_amd_memory_fill(ptr, 0, rounded / 4);
  if (rc != HSA_STATUS_SUCCESS) {
    DP("zero fill device_state failed with %u\n", rc);
    core::Runtime::Memfree(ptr);
    return HSA_STATUS_ERROR;
  }

  *ret_ptr = ptr;
  return HSA_STATUS_SUCCESS;
}

static bool image_contains_symbol(void *data, size_t size, const char *sym) {
  symbol_info si;
  int rc = get_symbol_info_without_loading((char *)data, size, sym, &si);
  return (rc == 0) && (si.addr != nullptr);
}

__tgt_target_table *__tgt_rtl_load_binary_locked(int32_t device_id,
                                                 __tgt_device_image *image) {
  // This function loads the device image onto gpu[device_id] and does other
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

  const size_t img_size = (char *)image->ImageEnd - (char *)image->ImageStart;

  DeviceInfo.clearOffloadEntriesTable(device_id);

  // We do not need to set the ELF version because the caller of this function
  // had to do that to decide the right runtime to use

  if (!elf_machine_id_is_amdgcn(image)) {
    return NULL;
  }

  {
    auto env = device_environment(device_id, DeviceInfo.NumberOfDevices, image,
                                  img_size);

    auto &KernelInfo = DeviceInfo.KernelInfoTable[device_id];
    auto &SymbolInfo = DeviceInfo.SymbolInfoTable[device_id];
    hsa_status_t err = module_register_from_memory_to_place(
        KernelInfo, SymbolInfo, (void *)image->ImageStart, img_size, device_id,
        [&](void *data, size_t size) {
          if (image_contains_symbol(data, size, "needs_hostcall_buffer")) {
            __atomic_store_n(&DeviceInfo.hostcall_required, true,
                             __ATOMIC_RELEASE);
          }
          return env.before_loading(data, size);
        },
        DeviceInfo.HSAExecutables);

    check("Module registering", err);
    if (err != HSA_STATUS_SUCCESS) {
      const char *DeviceName = DeviceInfo.GPUName[device_id].c_str();
      const char *ElfName = get_elf_mach_gfx_name(elf_e_flags(image));

      if (strcmp(DeviceName, ElfName) != 0) {
        DP("Possible gpu arch mismatch: device:%s, image:%s please check"
           " compiler flag: -march=<gpu>\n",
           DeviceName, ElfName);
      } else {
        DP("Error loading image onto GPU: %s\n", get_error_string(err));
      }

      return NULL;
    }

    err = env.after_loading();
    if (err != HSA_STATUS_SUCCESS) {
      return NULL;
    }
  }

  DP("AMDGPU module successfully loaded!\n");

  {
    // the device_State array is either large value in bss or a void* that
    // needs to be assigned to a pointer to an array of size device_state_bytes
    // If absent, it has been deadstripped and needs no setup.

    void *state_ptr;
    uint32_t state_ptr_size;
    auto &SymbolInfoMap = DeviceInfo.SymbolInfoTable[device_id];
    hsa_status_t err = interop_hsa_get_symbol_info(
        SymbolInfoMap, device_id, "omptarget_nvptx_device_State", &state_ptr,
        &state_ptr_size);

    if (err != HSA_STATUS_SUCCESS) {
      DP("No device_state symbol found, skipping initialization\n");
    } else {
      if (state_ptr_size < sizeof(void *)) {
        DP("unexpected size of state_ptr %u != %zu\n", state_ptr_size,
           sizeof(void *));
        return NULL;
      }

      // if it's larger than a void*, assume it's a bss array and no further
      // initialization is required. Only try to set up a pointer for
      // sizeof(void*)
      if (state_ptr_size == sizeof(void *)) {
        uint64_t device_State_bytes =
            get_device_State_bytes((char *)image->ImageStart, img_size);
        if (device_State_bytes == 0) {
          DP("Can't initialize device_State, missing size information\n");
          return NULL;
        }

        auto &dss = DeviceInfo.deviceStateStore[device_id];
        if (dss.first.get() == nullptr) {
          assert(dss.second == 0);
          void *ptr = NULL;
          hsa_status_t err = impl_calloc(&ptr, device_State_bytes, device_id);
          if (err != HSA_STATUS_SUCCESS) {
            DP("Failed to allocate device_state array\n");
            return NULL;
          }
          dss = {
              std::unique_ptr<void, RTLDeviceInfoTy::implFreePtrDeletor>{ptr},
              device_State_bytes,
          };
        }

        void *ptr = dss.first.get();
        if (device_State_bytes != dss.second) {
          DP("Inconsistent sizes of device_State unsupported\n");
          return NULL;
        }

        // write ptr to device memory so it can be used by later kernels
        err = DeviceInfo.freesignalpool_memcpy_h2d(state_ptr, &ptr,
                                                   sizeof(void *), device_id);
        if (err != HSA_STATUS_SUCCESS) {
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

  __tgt_offload_entry *HostBegin = image->EntriesBegin;
  __tgt_offload_entry *HostEnd = image->EntriesEnd;

  for (__tgt_offload_entry *e = HostBegin; e != HostEnd; ++e) {

    if (!e->addr) {
      // The host should have always something in the address to
      // uniquely identify the target region.
      DP("Analyzing host entry '<null>' (size = %lld)...\n",
         (unsigned long long)e->size);
      return NULL;
    }

    if (e->size) {
      __tgt_offload_entry entry = *e;

      void *varptr;
      uint32_t varsize;

      auto &SymbolInfoMap = DeviceInfo.SymbolInfoTable[device_id];
      hsa_status_t err = interop_hsa_get_symbol_info(
          SymbolInfoMap, device_id, e->name, &varptr, &varsize);

      if (err != HSA_STATUS_SUCCESS) {
        // Inform the user what symbol prevented offloading
        DP("Loading global '%s' (Failed)\n", e->name);
        return NULL;
      }

      if (varsize != e->size) {
        DP("Loading global '%s' - size mismatch (%u != %lu)\n", e->name,
           varsize, e->size);
        return NULL;
      }

      DP("Entry point " DPxMOD " maps to global %s (" DPxMOD ")\n",
         DPxPTR(e - HostBegin), e->name, DPxPTR(varptr));
      entry.addr = (void *)varptr;

      DeviceInfo.addOffloadEntry(device_id, entry);

      if (DeviceInfo.RequiresFlags & OMP_REQ_UNIFIED_SHARED_MEMORY &&
          e->flags & OMP_DECLARE_TARGET_LINK) {
        // If unified memory is present any target link variables
        // can access host addresses directly. There is no longer a
        // need for device copies.
        err = DeviceInfo.freesignalpool_memcpy_h2d(varptr, e->addr,
                                                   sizeof(void *), device_id);
        if (err != HSA_STATUS_SUCCESS)
          DP("Error when copying USM\n");
        DP("Copy linked variable host address (" DPxMOD ")"
           "to device address (" DPxMOD ")\n",
           DPxPTR(*((void **)e->addr)), DPxPTR(varptr));
      }

      continue;
    }

    DP("to find the kernel name: %s size: %lu\n", e->name, strlen(e->name));

    uint32_t kernarg_segment_size;
    auto &KernelInfoMap = DeviceInfo.KernelInfoTable[device_id];
    hsa_status_t err = interop_hsa_get_kernel_info(
        KernelInfoMap, device_id, e->name,
        HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
        &kernarg_segment_size);

    // each arg is a void * in this openmp implementation
    uint32_t arg_num = kernarg_segment_size / sizeof(void *);
    std::vector<size_t> arg_sizes(arg_num);
    for (std::vector<size_t>::iterator it = arg_sizes.begin();
         it != arg_sizes.end(); it++) {
      *it = sizeof(void *);
    }

    // default value GENERIC (in case symbol is missing from cubin file)
    llvm::omp::OMPTgtExecModeFlags ExecModeVal =
        llvm::omp::OMPTgtExecModeFlags::OMP_TGT_EXEC_MODE_GENERIC;

    // get flat group size if present, else Default_WG_Size
    int16_t WGSizeVal = RTLDeviceInfoTy::Default_WG_Size;

    // get Kernel Descriptor if present.
    // Keep struct in sync wih getTgtAttributeStructQTy in CGOpenMPRuntime.cpp
    struct KernDescValType {
      uint16_t Version;
      uint16_t TSize;
      uint16_t WG_Size;
    };
    struct KernDescValType KernDescVal;
    std::string KernDescNameStr(e->name);
    KernDescNameStr += "_kern_desc";
    const char *KernDescName = KernDescNameStr.c_str();

    void *KernDescPtr;
    uint32_t KernDescSize;
    void *CallStackAddr = nullptr;
    err = interop_get_symbol_info((char *)image->ImageStart, img_size,
                                  KernDescName, &KernDescPtr, &KernDescSize);

    if (err == HSA_STATUS_SUCCESS) {
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
      DP("KernDesc: WG_Size: %d\n", KernDescVal.WG_Size);

      if (KernDescVal.WG_Size == 0) {
        KernDescVal.WG_Size = RTLDeviceInfoTy::Default_WG_Size;
        DP("Setting KernDescVal.WG_Size to default %d\n", KernDescVal.WG_Size);
      }
      WGSizeVal = KernDescVal.WG_Size;
      DP("WGSizeVal %d\n", WGSizeVal);
      check("Loading KernDesc computation property", err);
    } else {
      DP("Warning: Loading KernDesc '%s' - symbol not found, ", KernDescName);

      // Flat group size
      std::string WGSizeNameStr(e->name);
      WGSizeNameStr += "_wg_size";
      const char *WGSizeName = WGSizeNameStr.c_str();

      void *WGSizePtr;
      uint32_t WGSize;
      err = interop_get_symbol_info((char *)image->ImageStart, img_size,
                                    WGSizeName, &WGSizePtr, &WGSize);

      if (err == HSA_STATUS_SUCCESS) {
        if ((size_t)WGSize != sizeof(int16_t)) {
          DP("Loading global computation properties '%s' - size mismatch (%u "
             "!= "
             "%lu)\n",
             WGSizeName, WGSize, sizeof(int16_t));
          return NULL;
        }

        memcpy(&WGSizeVal, WGSizePtr, (size_t)WGSize);

        DP("After loading global for %s WGSize = %d\n", WGSizeName, WGSizeVal);

        if (WGSizeVal < RTLDeviceInfoTy::Default_WG_Size ||
            WGSizeVal > RTLDeviceInfoTy::Max_WG_Size) {
          DP("Error wrong WGSize value specified in HSA code object file: "
             "%d\n",
             WGSizeVal);
          WGSizeVal = RTLDeviceInfoTy::Default_WG_Size;
        }
      } else {
        DP("Warning: Loading WGSize '%s' - symbol not found, "
           "using default value %d\n",
           WGSizeName, WGSizeVal);
      }

      check("Loading WGSize computation property", err);
    }

    // Read execution mode from global in binary
    std::string ExecModeNameStr(e->name);
    ExecModeNameStr += "_exec_mode";
    const char *ExecModeName = ExecModeNameStr.c_str();

    void *ExecModePtr;
    uint32_t varsize;
    err = interop_get_symbol_info((char *)image->ImageStart, img_size,
                                  ExecModeName, &ExecModePtr, &varsize);

    if (err == HSA_STATUS_SUCCESS) {
      if ((size_t)varsize != sizeof(llvm::omp::OMPTgtExecModeFlags)) {
        DP("Loading global computation properties '%s' - size mismatch(%u != "
           "%lu)\n",
           ExecModeName, varsize, sizeof(llvm::omp::OMPTgtExecModeFlags));
        return NULL;
      }

      memcpy(&ExecModeVal, ExecModePtr, (size_t)varsize);

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
    check("Loading computation property", err);

    KernelsList.push_back(KernelTy(ExecModeVal, WGSizeVal, device_id,
                                   CallStackAddr, e->name, kernarg_segment_size,
                                   DeviceInfo.KernArgPool));
    __tgt_offload_entry entry = *e;
    entry.addr = (void *)&KernelsList.back();
    DeviceInfo.addOffloadEntry(device_id, entry);
    DP("Entry point %ld maps to %s\n", e - HostBegin, e->name);
  }

  return DeviceInfo.getOffloadEntriesTable(device_id);
}

void *__tgt_rtl_data_alloc(int device_id, int64_t size, void *, int32_t kind) {
  void *ptr = NULL;
  assert(device_id < DeviceInfo.NumberOfDevices && "Device ID too large");

  if (kind != TARGET_ALLOC_DEFAULT) {
    REPORT("Invalid target data allocation kind or requested allocator not "
           "implemented yet\n");
    return NULL;
  }

  hsa_amd_memory_pool_t MemoryPool = DeviceInfo.getDeviceMemoryPool(device_id);
  hsa_status_t err = hsa_amd_memory_pool_allocate(MemoryPool, size, 0, &ptr);
  DP("Tgt alloc data %ld bytes, (tgt:%016llx).\n", size,
     (long long unsigned)(Elf64_Addr)ptr);
  ptr = (err == HSA_STATUS_SUCCESS) ? ptr : NULL;
  return ptr;
}

int32_t __tgt_rtl_data_submit(int device_id, void *tgt_ptr, void *hst_ptr,
                              int64_t size) {
  assert(device_id < DeviceInfo.NumberOfDevices && "Device ID too large");
  __tgt_async_info AsyncInfo;
  int32_t rc = dataSubmit(device_id, tgt_ptr, hst_ptr, size, &AsyncInfo);
  if (rc != OFFLOAD_SUCCESS)
    return OFFLOAD_FAIL;

  return __tgt_rtl_synchronize(device_id, &AsyncInfo);
}

int32_t __tgt_rtl_data_submit_async(int device_id, void *tgt_ptr, void *hst_ptr,
                                    int64_t size, __tgt_async_info *AsyncInfo) {
  assert(device_id < DeviceInfo.NumberOfDevices && "Device ID too large");
  if (AsyncInfo) {
    initAsyncInfo(AsyncInfo);
    return dataSubmit(device_id, tgt_ptr, hst_ptr, size, AsyncInfo);
  } else {
    return __tgt_rtl_data_submit(device_id, tgt_ptr, hst_ptr, size);
  }
}

int32_t __tgt_rtl_data_retrieve(int device_id, void *hst_ptr, void *tgt_ptr,
                                int64_t size) {
  assert(device_id < DeviceInfo.NumberOfDevices && "Device ID too large");
  __tgt_async_info AsyncInfo;
  int32_t rc = dataRetrieve(device_id, hst_ptr, tgt_ptr, size, &AsyncInfo);
  if (rc != OFFLOAD_SUCCESS)
    return OFFLOAD_FAIL;

  return __tgt_rtl_synchronize(device_id, &AsyncInfo);
}

int32_t __tgt_rtl_data_retrieve_async(int device_id, void *hst_ptr,
                                      void *tgt_ptr, int64_t size,
                                      __tgt_async_info *AsyncInfo) {
  assert(AsyncInfo && "AsyncInfo is nullptr");
  assert(device_id < DeviceInfo.NumberOfDevices && "Device ID too large");
  initAsyncInfo(AsyncInfo);
  return dataRetrieve(device_id, hst_ptr, tgt_ptr, size, AsyncInfo);
}

int32_t __tgt_rtl_data_delete(int device_id, void *tgt_ptr) {
  assert(device_id < DeviceInfo.NumberOfDevices && "Device ID too large");
  hsa_status_t err;
  DP("Tgt free data (tgt:%016llx).\n", (long long unsigned)(Elf64_Addr)tgt_ptr);
  err = core::Runtime::Memfree(tgt_ptr);
  if (err != HSA_STATUS_SUCCESS) {
    DP("Error when freeing CUDA memory\n");
    return OFFLOAD_FAIL;
  }
  return OFFLOAD_SUCCESS;
}

// Determine launch values for kernel.
struct launchVals {
  int WorkgroupSize;
  int GridSize;
};
launchVals getLaunchVals(int WarpSize, EnvironmentVariables Env,
                         int ConstWGSize,
                         llvm::omp::OMPTgtExecModeFlags ExecutionMode,
                         int num_teams, int thread_limit,
                         uint64_t loop_tripcount, int DeviceNumTeams) {

  int threadsPerGroup = RTLDeviceInfoTy::Default_WG_Size;
  int num_groups = 0;

  int Max_Teams =
      Env.MaxTeamsDefault > 0 ? Env.MaxTeamsDefault : DeviceNumTeams;
  if (Max_Teams > RTLDeviceInfoTy::HardTeamLimit)
    Max_Teams = RTLDeviceInfoTy::HardTeamLimit;

  if (print_kernel_trace & STARTUP_DETAILS) {
    DP("RTLDeviceInfoTy::Max_Teams: %d\n", RTLDeviceInfoTy::Max_Teams);
    DP("Max_Teams: %d\n", Max_Teams);
    DP("RTLDeviceInfoTy::Warp_Size: %d\n", WarpSize);
    DP("RTLDeviceInfoTy::Max_WG_Size: %d\n", RTLDeviceInfoTy::Max_WG_Size);
    DP("RTLDeviceInfoTy::Default_WG_Size: %d\n",
       RTLDeviceInfoTy::Default_WG_Size);
    DP("thread_limit: %d\n", thread_limit);
    DP("threadsPerGroup: %d\n", threadsPerGroup);
    DP("ConstWGSize: %d\n", ConstWGSize);
  }
  // check for thread_limit() clause
  if (thread_limit > 0) {
    threadsPerGroup = thread_limit;
    DP("Setting threads per block to requested %d\n", thread_limit);
    // Add master warp for GENERIC
    if (ExecutionMode ==
        llvm::omp::OMPTgtExecModeFlags::OMP_TGT_EXEC_MODE_GENERIC) {
      threadsPerGroup += WarpSize;
      DP("Adding master wavefront: +%d threads\n", WarpSize);
    }
    if (threadsPerGroup > RTLDeviceInfoTy::Max_WG_Size) { // limit to max
      threadsPerGroup = RTLDeviceInfoTy::Max_WG_Size;
      DP("Setting threads per block to maximum %d\n", threadsPerGroup);
    }
  }
  // check flat_max_work_group_size attr here
  if (threadsPerGroup > ConstWGSize) {
    threadsPerGroup = ConstWGSize;
    DP("Reduced threadsPerGroup to flat-attr-group-size limit %d\n",
       threadsPerGroup);
  }
  if (print_kernel_trace & STARTUP_DETAILS)
    DP("threadsPerGroup: %d\n", threadsPerGroup);
  DP("Preparing %d threads\n", threadsPerGroup);

  // Set default num_groups (teams)
  if (Env.TeamLimit > 0)
    num_groups = (Max_Teams < Env.TeamLimit) ? Max_Teams : Env.TeamLimit;
  else
    num_groups = Max_Teams;
  DP("Set default num of groups %d\n", num_groups);

  if (print_kernel_trace & STARTUP_DETAILS) {
    DP("num_groups: %d\n", num_groups);
    DP("num_teams: %d\n", num_teams);
  }

  // Reduce num_groups if threadsPerGroup exceeds RTLDeviceInfoTy::Max_WG_Size
  // This reduction is typical for default case (no thread_limit clause).
  // or when user goes crazy with num_teams clause.
  // FIXME: We cant distinguish between a constant or variable thread limit.
  // So we only handle constant thread_limits.
  if (threadsPerGroup >
      RTLDeviceInfoTy::Default_WG_Size) //  256 < threadsPerGroup <= 1024
    // Should we round threadsPerGroup up to nearest WarpSize
    // here?
    num_groups = (Max_Teams * RTLDeviceInfoTy::Max_WG_Size) / threadsPerGroup;

  // check for num_teams() clause
  if (num_teams > 0) {
    num_groups = (num_teams < num_groups) ? num_teams : num_groups;
  }
  if (print_kernel_trace & STARTUP_DETAILS) {
    DP("num_groups: %d\n", num_groups);
    DP("Env.NumTeams %d\n", Env.NumTeams);
    DP("Env.TeamLimit %d\n", Env.TeamLimit);
  }

  if (Env.NumTeams > 0) {
    num_groups = (Env.NumTeams < num_groups) ? Env.NumTeams : num_groups;
    DP("Modifying teams based on Env.NumTeams %d\n", Env.NumTeams);
  } else if (Env.TeamLimit > 0) {
    num_groups = (Env.TeamLimit < num_groups) ? Env.TeamLimit : num_groups;
    DP("Modifying teams based on Env.TeamLimit%d\n", Env.TeamLimit);
  } else {
    if (num_teams <= 0) {
      if (loop_tripcount > 0) {
        if (ExecutionMode ==
            llvm::omp::OMPTgtExecModeFlags::OMP_TGT_EXEC_MODE_SPMD) {
          // round up to the nearest integer
          num_groups = ((loop_tripcount - 1) / threadsPerGroup) + 1;
        } else if (ExecutionMode ==
                   llvm::omp::OMPTgtExecModeFlags::OMP_TGT_EXEC_MODE_GENERIC) {
          num_groups = loop_tripcount;
        } else /* OMP_TGT_EXEC_MODE_GENERIC_SPMD */ {
          // This is a generic kernel that was transformed to use SPMD-mode
          // execution but uses Generic-mode semantics for scheduling.
          num_groups = loop_tripcount;
        }
        DP("Using %d teams due to loop trip count %" PRIu64 " and number of "
           "threads per block %d\n",
           num_groups, loop_tripcount, threadsPerGroup);
      }
    } else {
      num_groups = num_teams;
    }
    if (num_groups > Max_Teams) {
      num_groups = Max_Teams;
      if (print_kernel_trace & STARTUP_DETAILS)
        DP("Limiting num_groups %d to Max_Teams %d \n", num_groups, Max_Teams);
    }
    if (num_groups > num_teams && num_teams > 0) {
      num_groups = num_teams;
      if (print_kernel_trace & STARTUP_DETAILS)
        DP("Limiting num_groups %d to clause num_teams %d \n", num_groups,
           num_teams);
    }
  }

  // num_teams clause always honored, no matter what, unless DEFAULT is active.
  if (num_teams > 0) {
    num_groups = num_teams;
    // Cap num_groups to EnvMaxTeamsDefault if set.
    if (Env.MaxTeamsDefault > 0 && num_groups > Env.MaxTeamsDefault)
      num_groups = Env.MaxTeamsDefault;
  }
  if (print_kernel_trace & STARTUP_DETAILS) {
    DP("threadsPerGroup: %d\n", threadsPerGroup);
    DP("num_groups: %d\n", num_groups);
    DP("loop_tripcount: %ld\n", loop_tripcount);
  }
  DP("Final %d num_groups and %d threadsPerGroup\n", num_groups,
     threadsPerGroup);

  launchVals res;
  res.WorkgroupSize = threadsPerGroup;
  res.GridSize = threadsPerGroup * num_groups;
  return res;
}

static uint64_t acquire_available_packet_id(hsa_queue_t *queue) {
  uint64_t packet_id = hsa_queue_add_write_index_relaxed(queue, 1);
  bool full = true;
  while (full) {
    full =
        packet_id >= (queue->size + hsa_queue_load_read_index_scacquire(queue));
  }
  return packet_id;
}

static int32_t __tgt_rtl_run_target_team_region_locked(
    int32_t device_id, void *tgt_entry_ptr, void **tgt_args,
    ptrdiff_t *tgt_offsets, int32_t arg_num, int32_t num_teams,
    int32_t thread_limit, uint64_t loop_tripcount);

int32_t __tgt_rtl_run_target_team_region(int32_t device_id, void *tgt_entry_ptr,
                                         void **tgt_args,
                                         ptrdiff_t *tgt_offsets,
                                         int32_t arg_num, int32_t num_teams,
                                         int32_t thread_limit,
                                         uint64_t loop_tripcount) {

  DeviceInfo.load_run_lock.lock_shared();
  int32_t res = __tgt_rtl_run_target_team_region_locked(
      device_id, tgt_entry_ptr, tgt_args, tgt_offsets, arg_num, num_teams,
      thread_limit, loop_tripcount);

  DeviceInfo.load_run_lock.unlock_shared();
  return res;
}

int32_t __tgt_rtl_run_target_team_region_locked(
    int32_t device_id, void *tgt_entry_ptr, void **tgt_args,
    ptrdiff_t *tgt_offsets, int32_t arg_num, int32_t num_teams,
    int32_t thread_limit, uint64_t loop_tripcount) {
  // Set the context we are using
  // update thread limit content in gpu memory if un-initialized or specified
  // from host

  DP("Run target team region thread_limit %d\n", thread_limit);

  // All args are references.
  std::vector<void *> args(arg_num);
  std::vector<void *> ptrs(arg_num);

  DP("Arg_num: %d\n", arg_num);
  for (int32_t i = 0; i < arg_num; ++i) {
    ptrs[i] = (void *)((intptr_t)tgt_args[i] + tgt_offsets[i]);
    args[i] = &ptrs[i];
    DP("Offseted base: arg[%d]:" DPxMOD "\n", i, DPxPTR(ptrs[i]));
  }

  KernelTy *KernelInfo = (KernelTy *)tgt_entry_ptr;

  std::string kernel_name = std::string(KernelInfo->Name);
  auto &KernelInfoTable = DeviceInfo.KernelInfoTable;
  if (KernelInfoTable[device_id].find(kernel_name) ==
      KernelInfoTable[device_id].end()) {
    DP("Kernel %s not found\n", kernel_name.c_str());
    return OFFLOAD_FAIL;
  }

  const atl_kernel_info_t KernelInfoEntry =
      KernelInfoTable[device_id][kernel_name];
  const uint32_t group_segment_size = KernelInfoEntry.group_segment_size;
  const uint32_t sgpr_count = KernelInfoEntry.sgpr_count;
  const uint32_t vgpr_count = KernelInfoEntry.vgpr_count;
  const uint32_t sgpr_spill_count = KernelInfoEntry.sgpr_spill_count;
  const uint32_t vgpr_spill_count = KernelInfoEntry.vgpr_spill_count;

  assert(arg_num == (int)KernelInfoEntry.num_args);

  /*
   * Set limit based on ThreadsPerGroup and GroupsPerDevice
   */
  launchVals LV =
      getLaunchVals(DeviceInfo.WarpSize[device_id], DeviceInfo.Env,
                    KernelInfo->ConstWGSize, KernelInfo->ExecutionMode,
                    num_teams,      // From run_region arg
                    thread_limit,   // From run_region arg
                    loop_tripcount, // From run_region arg
                    DeviceInfo.NumTeams[KernelInfo->device_id]);
  const int GridSize = LV.GridSize;
  const int WorkgroupSize = LV.WorkgroupSize;

  if (print_kernel_trace >= LAUNCH) {
    int num_groups = GridSize / WorkgroupSize;
    // enum modes are SPMD, GENERIC, NONE 0,1,2
    // if doing rtl timing, print to stderr, unless stdout requested.
    bool traceToStdout = print_kernel_trace & (RTL_TO_STDOUT | RTL_TIMING);
    fprintf(traceToStdout ? stdout : stderr,
            "DEVID:%2d SGN:%1d ConstWGSize:%-4d args:%2d teamsXthrds:(%4dX%4d) "
            "reqd:(%4dX%4d) lds_usage:%uB sgpr_count:%u vgpr_count:%u "
            "sgpr_spill_count:%u vgpr_spill_count:%u tripcount:%lu n:%s\n",
            device_id, KernelInfo->ExecutionMode, KernelInfo->ConstWGSize,
            arg_num, num_groups, WorkgroupSize, num_teams, thread_limit,
            group_segment_size, sgpr_count, vgpr_count, sgpr_spill_count,
            vgpr_spill_count, loop_tripcount, KernelInfo->Name);
  }

  // Run on the device.
  {
    hsa_queue_t *queue = DeviceInfo.HSAQueues[device_id].get();
    if (!queue) {
      return OFFLOAD_FAIL;
    }
    uint64_t packet_id = acquire_available_packet_id(queue);

    const uint32_t mask = queue->size - 1; // size is a power of 2
    hsa_kernel_dispatch_packet_t *packet =
        (hsa_kernel_dispatch_packet_t *)queue->base_address +
        (packet_id & mask);

    // packet->header is written last
    packet->setup = UINT16_C(1) << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
    packet->workgroup_size_x = WorkgroupSize;
    packet->workgroup_size_y = 1;
    packet->workgroup_size_z = 1;
    packet->reserved0 = 0;
    packet->grid_size_x = GridSize;
    packet->grid_size_y = 1;
    packet->grid_size_z = 1;
    packet->private_segment_size = KernelInfoEntry.private_segment_size;
    packet->group_segment_size = KernelInfoEntry.group_segment_size;
    packet->kernel_object = KernelInfoEntry.kernel_object;
    packet->kernarg_address = 0;     // use the block allocator
    packet->reserved2 = 0;           // impl writes id_ here
    packet->completion_signal = {0}; // may want a pool of signals

    KernelArgPool *ArgPool = nullptr;
    void *kernarg = nullptr;
    {
      auto it = KernelArgPoolMap.find(std::string(KernelInfo->Name));
      if (it != KernelArgPoolMap.end()) {
        ArgPool = (it->second).get();
      }
    }
    if (!ArgPool) {
      DP("Warning: No ArgPool for %s on device %d\n", KernelInfo->Name,
         device_id);
    }
    {
      if (ArgPool) {
        assert(ArgPool->kernarg_segment_size == (arg_num * sizeof(void *)));
        kernarg = ArgPool->allocate(arg_num);
      }
      if (!kernarg) {
        DP("Allocate kernarg failed\n");
        return OFFLOAD_FAIL;
      }

      // Copy explicit arguments
      for (int i = 0; i < arg_num; i++) {
        memcpy((char *)kernarg + sizeof(void *) * i, args[i], sizeof(void *));
      }

      // Initialize implicit arguments. TODO: Which of these can be dropped
      impl_implicit_args_t *impl_args =
          reinterpret_cast<impl_implicit_args_t *>(
              static_cast<char *>(kernarg) + ArgPool->kernarg_segment_size);
      memset(impl_args, 0,
             sizeof(impl_implicit_args_t)); // may not be necessary
      impl_args->offset_x = 0;
      impl_args->offset_y = 0;
      impl_args->offset_z = 0;

      // assign a hostcall buffer for the selected Q
      if (__atomic_load_n(&DeviceInfo.hostcall_required, __ATOMIC_ACQUIRE)) {
        // hostrpc_assign_buffer is not thread safe, and this function is
        // under a multiple reader lock, not a writer lock.
        static pthread_mutex_t hostcall_init_lock = PTHREAD_MUTEX_INITIALIZER;
        pthread_mutex_lock(&hostcall_init_lock);
        impl_args->hostcall_ptr = hostrpc_assign_buffer(
            DeviceInfo.HSAAgents[device_id], queue, device_id);
        pthread_mutex_unlock(&hostcall_init_lock);
        if (!impl_args->hostcall_ptr) {
          DP("hostrpc_assign_buffer failed, gpu would dereference null and "
             "error\n");
          return OFFLOAD_FAIL;
        }
      }

      packet->kernarg_address = kernarg;
    }

    hsa_signal_t s = DeviceInfo.FreeSignalPool.pop();
    if (s.handle == 0) {
      DP("Failed to get signal instance\n");
      return OFFLOAD_FAIL;
    }
    packet->completion_signal = s;
    hsa_signal_store_relaxed(packet->completion_signal, 1);

    // Publish the packet indicating it is ready to be processed
    core::packet_store_release(reinterpret_cast<uint32_t *>(packet),
                               core::create_header(), packet->setup);

    // Since the packet is already published, its contents must not be
    // accessed any more
    hsa_signal_store_relaxed(queue->doorbell_signal, packet_id);

    while (hsa_signal_wait_scacquire(s, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX,
                                     HSA_WAIT_STATE_BLOCKED) != 0)
      ;

    assert(ArgPool);
    ArgPool->deallocate(kernarg);
    DeviceInfo.FreeSignalPool.push(s);
  }

  DP("Kernel completed\n");
  return OFFLOAD_SUCCESS;
}

int32_t __tgt_rtl_run_target_region(int32_t device_id, void *tgt_entry_ptr,
                                    void **tgt_args, ptrdiff_t *tgt_offsets,
                                    int32_t arg_num) {
  // use one team and one thread
  // fix thread num
  int32_t team_num = 1;
  int32_t thread_limit = 0; // use default
  return __tgt_rtl_run_target_team_region(device_id, tgt_entry_ptr, tgt_args,
                                          tgt_offsets, arg_num, team_num,
                                          thread_limit, 0);
}

int32_t __tgt_rtl_run_target_region_async(int32_t device_id,
                                          void *tgt_entry_ptr, void **tgt_args,
                                          ptrdiff_t *tgt_offsets,
                                          int32_t arg_num,
                                          __tgt_async_info *AsyncInfo) {
  assert(AsyncInfo && "AsyncInfo is nullptr");
  initAsyncInfo(AsyncInfo);

  // use one team and one thread
  // fix thread num
  int32_t team_num = 1;
  int32_t thread_limit = 0; // use default
  return __tgt_rtl_run_target_team_region(device_id, tgt_entry_ptr, tgt_args,
                                          tgt_offsets, arg_num, team_num,
                                          thread_limit, 0);
}

int32_t __tgt_rtl_synchronize(int32_t device_id, __tgt_async_info *AsyncInfo) {
  assert(AsyncInfo && "AsyncInfo is nullptr");

  // Cuda asserts that AsyncInfo->Queue is non-null, but this invariant
  // is not ensured by devices.cpp for amdgcn
  // assert(AsyncInfo->Queue && "AsyncInfo->Queue is nullptr");
  if (AsyncInfo->Queue) {
    finiAsyncInfo(AsyncInfo);
  }
  return OFFLOAD_SUCCESS;
}

namespace core {
hsa_status_t allow_access_to_all_gpu_agents(void *ptr) {
  return hsa_amd_agents_allow_access(DeviceInfo.HSAAgents.size(),
                                     &DeviceInfo.HSAAgents[0], NULL, ptr);
}

} // namespace core
