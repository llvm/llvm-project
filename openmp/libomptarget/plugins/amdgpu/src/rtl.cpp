//===----RTLs/hsa/src/rtl.cpp - Target RTLs Implementation -------- C++ -*-===//
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
#include <dlfcn.h>
#include <elf.h>
#include <ffi.h>
#include <fstream>
#include <iostream>
#include <libelf.h>
#include <list>
#include <memory>
#include <unordered_map>
#include <vector>

// Header from ATMI interface
#include "atmi_interop_hsa.h"
#include "atmi_runtime.h"

#include "internal.h"

#include "internal.h"

#include "omptargetplugin.h"

// Get static gpu grid values from clang target-specific constants managed
// in the header file llvm/Frontend/OpenMP/OMPGridValues.h
// Copied verbatim to meet the requirement that libomptarget builds without
// a copy of llvm checked out nearby
namespace llvm {
namespace omp {
enum GVIDX {
  /// The maximum number of workers in a kernel.
  /// (THREAD_ABSOLUTE_LIMIT) - (GV_Warp_Size), might be issue for blockDim.z
  GV_Threads,
  /// The size reserved for data in a shared memory slot.
  GV_Slot_Size,
  /// The default value of maximum number of threads in a worker warp.
  GV_Warp_Size,
  /// Alternate warp size for some AMDGCN architectures. Same as GV_Warp_Size
  /// for NVPTX.
  GV_Warp_Size_32,
  /// The number of bits required to represent the max number of threads in warp
  GV_Warp_Size_Log2,
  /// GV_Warp_Size * GV_Slot_Size,
  GV_Warp_Slot_Size,
  /// the maximum number of teams.
  GV_Max_Teams,
  /// Global Memory Alignment
  GV_Mem_Align,
  /// (~0u >> (GV_Warp_Size - GV_Warp_Size_Log2))
  GV_Warp_Size_Log2_Mask,
  // An alternative to the heavy data sharing infrastructure that uses global
  // memory is one that uses device __shared__ memory.  The amount of such space
  // (in bytes) reserved by the OpenMP runtime is noted here.
  GV_SimpleBufferSize,
  // The absolute maximum team size for a working group
  GV_Max_WG_Size,
  // The default maximum team size for a working group
  GV_Default_WG_Size,
  // This is GV_Max_WG_Size / GV_WarpSize. 32 for NVPTX and 16 for AMDGCN.
  GV_Max_Warp_Number,
  /// The slot size that should be reserved for a working warp.
  /// (~0u >> (GV_Warp_Size - GV_Warp_Size_Log2))
  GV_Warp_Size_Log2_MaskL
};

static constexpr unsigned AMDGPUGpuGridValues[] = {
    448,       // GV_Threads
    256,       // GV_Slot_Size
    64,        // GV_Warp_Size
    32,        // GV_Warp_Size_32
    6,         // GV_Warp_Size_Log2
    64 * 256,  // GV_Warp_Slot_Size
    128,       // GV_Max_Teams
    256,       // GV_Mem_Align
    63,        // GV_Warp_Size_Log2_Mask
    896,       // GV_SimpleBufferSize
    1024,      // GV_Max_WG_Size,
    256,       // GV_Defaut_WG_Size
    1024 / 64, // GV_Max_WG_Size / GV_WarpSize
    63         // GV_Warp_Size_Log2_MaskL
};
} // namespace omp
} // namespace llvm

#ifndef TARGET_NAME
#define TARGET_NAME AMDHSA
#endif

int print_kernel_trace;

// Size of the target call stack struture
uint32_t TgtStackItemSize = 0;

#ifdef OMPTARGET_DEBUG
static int DebugLevel = 0;

#define GETNAME2(name) #name
#define GETNAME(name) GETNAME2(name)
#define DP(...)                                                                \
  do {                                                                         \
    if (DebugLevel > 0) {                                                      \
      DEBUGP("Target " GETNAME(TARGET_NAME) " RTL", __VA_ARGS__);              \
    }                                                                          \
  } while (false)
#else // OMPTARGET_DEBUG
#define DP(...)                                                                \
  {}
#endif // OMPTARGET_DEBUG

#undef check // Drop definition from internal.h
#ifdef OMPTARGET_DEBUG
#define check(msg, status)                                                     \
  if (status != ATMI_STATUS_SUCCESS) {                                         \
    /* fprintf(stderr, "[%s:%d] %s failed.\n", __FILE__, __LINE__, #msg);*/    \
    DP(#msg " failed\n");                                                      \
    /*assert(0);*/                                                             \
  } else {                                                                     \
    /* fprintf(stderr, "[%s:%d] %s succeeded.\n", __FILE__, __LINE__, #msg);   \
     */                                                                        \
    DP(#msg " succeeded\n");                                                   \
  }
#else
#define check(msg, status)                                                     \
  {}
#endif

#include "../../common/elf_common.c"

static bool elf_machine_id_is_amdgcn(__tgt_device_image *image) {
  const uint16_t amdgcnMachineID = 224;
  int32_t r = elf_check_machine(image, amdgcnMachineID);
  if (!r) {
    DP("Supported machine ID not found\n");
  }
  return r;
}

/// Keep entries table per device
struct FuncOrGblEntryTy {
  __tgt_target_table Table;
  std::vector<__tgt_offload_entry> Entries;
};

enum ExecutionModeType {
  SPMD,    // constructors, destructors,
           // combined constructs (`teams distribute parallel for [simd]`)
  GENERIC, // everything else
  NONE
};

struct KernelArgPool {
private:
  static pthread_mutex_t mutex;

public:
  uint32_t kernarg_segment_size;
  void *kernarg_region = nullptr;
  std::queue<int> free_kernarg_segments;

  uint32_t kernarg_size_including_implicit() {
    return kernarg_segment_size + sizeof(atmi_implicit_args_t);
  }

  ~KernelArgPool() {
    if (kernarg_region) {
      auto r = hsa_amd_memory_pool_free(kernarg_region);
      assert(r == HSA_STATUS_SUCCESS);
      ErrorCheck(Memory pool free, r);
    }
  }

  // Can't really copy or move a mutex
  KernelArgPool() = default;
  KernelArgPool(const KernelArgPool &) = delete;
  KernelArgPool(KernelArgPool &&) = delete;

  KernelArgPool(uint32_t kernarg_segment_size)
      : kernarg_segment_size(kernarg_segment_size) {

    // atmi uses one pool per kernel for all gpus, with a fixed upper size
    // preserving that exact scheme here, including the queue<int>
    {
      hsa_status_t err = hsa_amd_memory_pool_allocate(
          atl_gpu_kernarg_pools[0],
          kernarg_size_including_implicit() * MAX_NUM_KERNELS, 0,
          &kernarg_region);
      ErrorCheck(Allocating memory for the executable-kernel, err);
      core::allow_access_to_all_gpu_agents(kernarg_region);

      for (int i = 0; i < MAX_NUM_KERNELS; i++) {
        free_kernarg_segments.push(i);
      }
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
  // execution mode of kernel
  // 0 - SPMD mode (without master warp)
  // 1 - Generic mode (with master warp)
  int8_t ExecutionMode;
  int16_t ConstWGSize;
  int8_t MaxParLevel;
  int32_t device_id;
  void *CallStackAddr;
  const char *Name;

  KernelTy(int8_t _ExecutionMode, int16_t _ConstWGSize, int8_t _MaxParLevel,
           int32_t _device_id, void *_CallStackAddr, const char *_Name,
           uint32_t _kernarg_segment_size)
      : ExecutionMode(_ExecutionMode), ConstWGSize(_ConstWGSize),
        MaxParLevel(_MaxParLevel), device_id(_device_id),
        CallStackAddr(_CallStackAddr), Name(_Name) {
    DP("Construct kernelinfo: ExecMode %d\n", ExecutionMode);

    std::string N(_Name);
    if (KernelArgPoolMap.find(N) == KernelArgPoolMap.end()) {
      KernelArgPoolMap.insert(
          std::make_pair(N, std::unique_ptr<KernelArgPool>(
                                new KernelArgPool(_kernarg_segment_size))));
    }
  }
};

/// List that contains all the kernels.
/// FIXME: we may need this to be per device and per library.
std::list<KernelTy> KernelsList;

// ATMI API to get gpu and gpu memory place
static atmi_place_t get_gpu_place(int device_id) {
  return ATMI_PLACE_GPU(0, device_id);
}
static atmi_mem_place_t get_gpu_mem_place(int device_id) {
  return ATMI_MEM_PLACE_GPU_MEM(0, device_id, 0);
}

static std::vector<hsa_agent_t> find_gpu_agents() {
  std::vector<hsa_agent_t> res;

  hsa_status_t err = hsa_iterate_agents(
      [](hsa_agent_t agent, void *data) -> hsa_status_t {
        std::vector<hsa_agent_t> *res =
            static_cast<std::vector<hsa_agent_t> *>(data);

        hsa_device_type_t device_type;
        // get_info fails iff HSA runtime not yet initialized
        hsa_status_t err =
            hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
        if (print_kernel_trace > 0 && err != HSA_STATUS_SUCCESS)
          printf("rtl.cpp: err %d\n", err);
        assert(err == HSA_STATUS_SUCCESS);

        if (device_type == HSA_DEVICE_TYPE_GPU) {
          res->push_back(agent);
        }
        return HSA_STATUS_SUCCESS;
      },
      &res);

  // iterate_agents fails iff HSA runtime not yet initialized
  if (print_kernel_trace > 0 && err != HSA_STATUS_SUCCESS)
    printf("rtl.cpp: err %d\n", err);
  assert(err == HSA_STATUS_SUCCESS);
  return res;
}

static void callbackQueue(hsa_status_t status, hsa_queue_t *source,
                          void *data) {
  if (status != HSA_STATUS_SUCCESS) {
    const char *status_string;
    if (hsa_status_string(status, &status_string) != HSA_STATUS_SUCCESS) {
      status_string = "unavailable";
    }
    fprintf(stderr, "[%s:%d] GPU error in queue %p %d (%s)\n", __FILE__,
            __LINE__, source, status, status_string);
    abort();
  }
}

namespace core {
void packet_store_release(uint32_t *packet, uint16_t header, uint16_t rest) {
  __atomic_store_n(packet, header | (rest << 16), __ATOMIC_RELEASE);
}

uint16_t create_header(hsa_packet_type_t type, int barrier,
                       atmi_task_fence_scope_t acq_fence,
                       atmi_task_fence_scope_t rel_fence) {
  uint16_t header = type << HSA_PACKET_HEADER_TYPE;
  header |= barrier << HSA_PACKET_HEADER_BARRIER;
  header |= (hsa_fence_scope_t) static_cast<int>(
      acq_fence << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE);
  header |= (hsa_fence_scope_t) static_cast<int>(
      rel_fence << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);
  return header;
}
} // namespace core

/// Class containing all the device information
class RTLDeviceInfoTy {
  std::vector<std::list<FuncOrGblEntryTy>> FuncGblEntries;

public:
  int NumberOfDevices;

  // GPU devices
  std::vector<hsa_agent_t> HSAAgents;
  std::vector<hsa_queue_t *> HSAQueues; // one per gpu

  // Device properties
  std::vector<int> ComputeUnits;
  std::vector<int> GroupsPerDevice;
  std::vector<int> ThreadsPerGroup;
  std::vector<int> WarpSize;

  // OpenMP properties
  std::vector<int> NumTeams;
  std::vector<int> NumThreads;

  // OpenMP Environment properties
  int EnvNumTeams;
  int EnvTeamLimit;
  int EnvMaxTeamsDefault;

  // OpenMP Requires Flags
  int64_t RequiresFlags;

  // Resource pools
  SignalPoolT FreeSignalPool;

  static const int HardTeamLimit = 1 << 20; // 1 Meg
  static const int DefaultNumTeams = 128;
  static const int Max_Teams =
      llvm::omp::AMDGPUGpuGridValues[llvm::omp::GVIDX::GV_Max_Teams];
  static const int Warp_Size =
      llvm::omp::AMDGPUGpuGridValues[llvm::omp::GVIDX::GV_Warp_Size];
  static const int Max_WG_Size =
      llvm::omp::AMDGPUGpuGridValues[llvm::omp::GVIDX::GV_Max_WG_Size];
  static const int Default_WG_Size =
      llvm::omp::AMDGPUGpuGridValues[llvm::omp::GVIDX::GV_Default_WG_Size];

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

  RTLDeviceInfoTy() {
#ifdef OMPTARGET_DEBUG
    if (char *envStr = getenv("LIBOMPTARGET_DEBUG"))
      DebugLevel = std::stoi(envStr);
#endif // OMPTARGET_DEBUG

    // LIBOMPTARGET_KERNEL_TRACE provides a kernel launch trace to stderr
    // anytime. You do not need a debug library build.
    //  0 => no tracing
    //  1 => tracing dispatch only
    // >1 => verbosity increase
    if (char *envStr = getenv("LIBOMPTARGET_KERNEL_TRACE"))
      print_kernel_trace = atoi(envStr);
    else
      print_kernel_trace = 0;

    DP("Start initializing HSA-ATMI\n");
    atmi_status_t err = atmi_init();
    if (err != ATMI_STATUS_SUCCESS) {
      DP("Error when initializing HSA-ATMI\n");
      return;
    }

    HSAAgents = find_gpu_agents();
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
    GroupsPerDevice.resize(NumberOfDevices);
    WarpSize.resize(NumberOfDevices);
    NumTeams.resize(NumberOfDevices);
    NumThreads.resize(NumberOfDevices);

    for (int i = 0; i < NumberOfDevices; i++) {
      uint32_t queue_size = 0;
      {
        hsa_status_t err;
        err = hsa_agent_get_info(HSAAgents[i], HSA_AGENT_INFO_QUEUE_MAX_SIZE,
                                 &queue_size);
        ErrorCheck(Querying the agent maximum queue size, err);
        if (queue_size > core::Runtime::getInstance().getMaxQueueSize()) {
          queue_size = core::Runtime::getInstance().getMaxQueueSize();
        }
      }

      hsa_status_t rc = hsa_queue_create(
          HSAAgents[i], queue_size, HSA_QUEUE_TYPE_MULTI, callbackQueue, NULL,
          UINT32_MAX, UINT32_MAX, &HSAQueues[i]);
      if (rc != HSA_STATUS_SUCCESS) {
        DP("Failed to create HSA queues\n");
        return;
      }
    }

    for (int i = 0; i < NumberOfDevices; i++) {
      ThreadsPerGroup[i] = RTLDeviceInfoTy::Default_WG_Size;
      GroupsPerDevice[i] = RTLDeviceInfoTy::DefaultNumTeams;
      ComputeUnits[i] = 1;
      DP("Device %d: Initial groupsPerDevice %d & threadsPerGroup %d\n", i,
         GroupsPerDevice[i], ThreadsPerGroup[i]);
    }

    // Get environment variables regarding teams
    char *envStr = getenv("OMP_TEAM_LIMIT");
    if (envStr) {
      // OMP_TEAM_LIMIT has been set
      EnvTeamLimit = std::stoi(envStr);
      DP("Parsed OMP_TEAM_LIMIT=%d\n", EnvTeamLimit);
    } else {
      EnvTeamLimit = -1;
    }
    envStr = getenv("OMP_NUM_TEAMS");
    if (envStr) {
      // OMP_NUM_TEAMS has been set
      EnvNumTeams = std::stoi(envStr);
      DP("Parsed OMP_NUM_TEAMS=%d\n", EnvNumTeams);
    } else {
      EnvNumTeams = -1;
    }
    // Get environment variables regarding expMaxTeams
    envStr = getenv("OMP_MAX_TEAMS_DEFAULT");
    if (envStr) {
      EnvMaxTeamsDefault = std::stoi(envStr);
      DP("Parsed OMP_MAX_TEAMS_DEFAULT=%d\n", EnvMaxTeamsDefault);
    } else {
      EnvMaxTeamsDefault = -1;
    }

    // Default state.
    RequiresFlags = OMP_REQ_UNDEFINED;
  }

  ~RTLDeviceInfoTy() {
    DP("Finalizing the HSA-ATMI DeviceInfo.\n");
    KernelArgPoolMap.clear(); // calls hsa to free memory
    atmi_finalize();
  }
};

pthread_mutex_t SignalPoolT::mutex = PTHREAD_MUTEX_INITIALIZER;

// TODO: May need to drop the trailing to fields until deviceRTL is updated
struct omptarget_device_environmentTy {
  int32_t debug_level; // gets value of envvar LIBOMPTARGET_DEVICE_RTL_DEBUG
                       // only useful for Debug build of deviceRTLs
  int32_t num_devices; // gets number of active offload devices
  int32_t device_num;  // gets a value 0 to num_devices-1
};

static RTLDeviceInfoTy DeviceInfo;

namespace {

int32_t dataRetrieve(int32_t DeviceId, void *HstPtr, void *TgtPtr, int64_t Size,
                     __tgt_async_info *AsyncInfoPtr) {
  assert(AsyncInfoPtr && "AsyncInfoPtr is nullptr");
  assert(DeviceId < DeviceInfo.NumberOfDevices && "Device ID too large");
  // Return success if we are not copying back to host from target.
  if (!HstPtr)
    return OFFLOAD_SUCCESS;
  atmi_status_t err;
  DP("Retrieve data %ld bytes, (tgt:%016llx) -> (hst:%016llx).\n", Size,
     (long long unsigned)(Elf64_Addr)TgtPtr,
     (long long unsigned)(Elf64_Addr)HstPtr);
  err = atmi_memcpy(HstPtr, TgtPtr, (size_t)Size);
  if (err != ATMI_STATUS_SUCCESS) {
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
                   __tgt_async_info *AsyncInfoPtr) {
  assert(AsyncInfoPtr && "AsyncInfoPtr is nullptr");
  atmi_status_t err;
  assert(DeviceId < DeviceInfo.NumberOfDevices && "Device ID too large");
  // Return success if we are not doing host to target.
  if (!HstPtr)
    return OFFLOAD_SUCCESS;

  DP("Submit data %ld bytes, (hst:%016llx) -> (tgt:%016llx).\n", Size,
     (long long unsigned)(Elf64_Addr)HstPtr,
     (long long unsigned)(Elf64_Addr)TgtPtr);
  err = atmi_memcpy(TgtPtr, HstPtr, (size_t)Size);
  if (err != ATMI_STATUS_SUCCESS) {
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
// change this interface to explicitly initialize the async_info_pointer, but
// until then hsa lazily initializes it as well.

void initAsyncInfoPtr(__tgt_async_info *async_info_ptr) {
  // set non-null while using async calls, return to null to indicate completion
  assert(async_info_ptr);
  if (!async_info_ptr->Queue) {
    async_info_ptr->Queue = reinterpret_cast<void *>(UINT64_MAX);
  }
}
void finiAsyncInfoPtr(__tgt_async_info *async_info_ptr) {
  assert(async_info_ptr);
  assert(async_info_ptr->Queue);
  async_info_ptr->Queue = 0;
}
} // namespace

int32_t __tgt_rtl_is_valid_binary(__tgt_device_image *image) {
  return elf_machine_id_is_amdgcn(image);
}

int __tgt_rtl_number_of_devices() { return DeviceInfo.NumberOfDevices; }

int64_t __tgt_rtl_init_requires(int64_t RequiresFlags) {
  DP("Init requires flags to %ld\n", RequiresFlags);
  DeviceInfo.RequiresFlags = RequiresFlags;
  return RequiresFlags;
}

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
  if (print_kernel_trace > 1)
    fprintf(stderr, "Device#%-2d CU's: %2d\n", device_id,
            DeviceInfo.ComputeUnits[device_id]);

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
    if ((DeviceInfo.ThreadsPerGroup[device_id] >
         RTLDeviceInfoTy::Max_WG_Size) ||
        DeviceInfo.ThreadsPerGroup[device_id] == 0) {
      DP("Capped thread limit: %d\n", RTLDeviceInfoTy::Max_WG_Size);
      DeviceInfo.ThreadsPerGroup[device_id] = RTLDeviceInfoTy::Max_WG_Size;
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
    DP("Default wavefront size: %d\n",
       llvm::omp::AMDGPUGpuGridValues[llvm::omp::GVIDX::GV_Warp_Size]);
    DeviceInfo.WarpSize[device_id] =
        llvm::omp::AMDGPUGpuGridValues[llvm::omp::GVIDX::GV_Warp_Size];
  }

  // Adjust teams to the env variables
  if (DeviceInfo.EnvTeamLimit > 0 &&
      DeviceInfo.GroupsPerDevice[device_id] > DeviceInfo.EnvTeamLimit) {
    DeviceInfo.GroupsPerDevice[device_id] = DeviceInfo.EnvTeamLimit;
    DP("Capping max groups per device to OMP_TEAM_LIMIT=%d\n",
       DeviceInfo.EnvTeamLimit);
  }

  // Set default number of teams
  if (DeviceInfo.EnvNumTeams > 0) {
    DeviceInfo.NumTeams[device_id] = DeviceInfo.EnvNumTeams;
    DP("Default number of teams set according to environment %d\n",
       DeviceInfo.EnvNumTeams);
  } else {
    DeviceInfo.NumTeams[device_id] = RTLDeviceInfoTy::DefaultNumTeams;
    DP("Default number of teams set according to library's default %d\n",
       RTLDeviceInfoTy::DefaultNumTeams);
  }

  if (DeviceInfo.NumTeams[device_id] > DeviceInfo.GroupsPerDevice[device_id]) {
    DeviceInfo.NumTeams[device_id] = DeviceInfo.GroupsPerDevice[device_id];
    DP("Default number of teams exceeds device limit, capping at %d\n",
       DeviceInfo.GroupsPerDevice[device_id]);
  }

  // Set default number of threads
  DeviceInfo.NumThreads[device_id] = RTLDeviceInfoTy::Default_WG_Size;
  DP("Default number of threads set according to library's default %d\n",
     RTLDeviceInfoTy::Default_WG_Size);
  if (DeviceInfo.NumThreads[device_id] >
      DeviceInfo.ThreadsPerGroup[device_id]) {
    DeviceInfo.NumTeams[device_id] = DeviceInfo.ThreadsPerGroup[device_id];
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

typedef struct {
  void *addr = nullptr;
  uint32_t size = UINT32_MAX;
} symbol_info;

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

  res->size = static_cast<uint32_t>(sym->st_size);
  res->addr = sym->st_value + base;
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

atmi_status_t interop_get_symbol_info(char *base, size_t img_size,
                                      const char *symname, void **var_addr,
                                      uint32_t *var_size) {
  symbol_info si;
  int rc = get_symbol_info_without_loading(base, img_size, symname, &si);
  if (rc == 0) {
    *var_addr = si.addr;
    *var_size = si.size;
    return ATMI_STATUS_SUCCESS;
  } else {
    return ATMI_STATUS_ERROR;
  }
}

template <typename C>
atmi_status_t module_register_from_memory_to_place(void *module_bytes,
                                                   size_t module_size,
                                                   atmi_place_t place, C cb) {
  auto L = [](void *data, size_t size, void *cb_state) -> atmi_status_t {
    C *unwrapped = static_cast<C *>(cb_state);
    return (*unwrapped)(data, size);
  };
  return atmi_module_register_from_memory_to_place(
      module_bytes, module_size, place, L, static_cast<void *>(&cb));
}
} // namespace

static __tgt_target_table *
__tgt_rtl_load_binary_locked(int32_t device_id, __tgt_device_image *image);

__tgt_target_table *__tgt_rtl_load_binary(int32_t device_id,
                                          __tgt_device_image *image) {
  static pthread_mutex_t load_binary_mutex = PTHREAD_MUTEX_INITIALIZER;
  pthread_mutex_lock(&load_binary_mutex);
  __tgt_target_table *res = __tgt_rtl_load_binary_locked(device_id, image);
  pthread_mutex_unlock(&load_binary_mutex);
  return res;
}

__tgt_target_table *__tgt_rtl_load_binary_locked(int32_t device_id,
                                                 __tgt_device_image *image) {
  const size_t img_size = (char *)image->ImageEnd - (char *)image->ImageStart;

  DeviceInfo.clearOffloadEntriesTable(device_id);

  // We do not need to set the ELF version because the caller of this function
  // had to do that to decide the right runtime to use

  if (!elf_machine_id_is_amdgcn(image)) {
    return NULL;
  }

  omptarget_device_environmentTy host_device_env;
  host_device_env.num_devices = DeviceInfo.NumberOfDevices;
  host_device_env.device_num = device_id;
  host_device_env.debug_level = 0;
#ifdef OMPTARGET_DEBUG
  if (char *envStr = getenv("LIBOMPTARGET_DEVICE_RTL_DEBUG")) {
    host_device_env.debug_level = std::stoi(envStr);
  }
#endif

  auto on_deserialized_data = [&](void *data, size_t size) -> atmi_status_t {
    const char *device_env_Name = "omptarget_device_environment";
    symbol_info si;
    int rc = get_symbol_info_without_loading((char *)image->ImageStart,
                                             img_size, device_env_Name, &si);
    if (rc != 0) {
      DP("Finding global device environment '%s' - symbol missing.\n",
         device_env_Name);
      // no need to return FAIL, consider this is a not a device debug build.
      return ATMI_STATUS_SUCCESS;
    }
    if (si.size != sizeof(host_device_env)) {
      return ATMI_STATUS_ERROR;
    }
    DP("Setting global device environment %lu bytes\n", si.size);
    uint64_t offset = (char *)si.addr - (char *)image->ImageStart;
    void *pos = (char *)data + offset;
    memcpy(pos, &host_device_env, sizeof(host_device_env));
    return ATMI_STATUS_SUCCESS;
  };

  atmi_status_t err;
  {
    err = module_register_from_memory_to_place(
        (void *)image->ImageStart, img_size, get_gpu_place(device_id),
        on_deserialized_data);

    check("Module registering", err);
    if (err != ATMI_STATUS_SUCCESS) {
      char GPUName[64] = "--unknown gpu--";
      hsa_agent_t agent = DeviceInfo.HSAAgents[device_id];
      (void)hsa_agent_get_info(agent, (hsa_agent_info_t)HSA_AGENT_INFO_NAME,
                               (void *)GPUName);
      fprintf(stderr,
              "Possible gpu arch mismatch: %s, please check"
              " compiler: -march=<gpu> flag\n",
              GPUName);
      return NULL;
    }
  }

  DP("ATMI module successfully loaded!\n");

  // TODO: Check with Guansong to understand the below comment more thoroughly.
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
      fprintf(stderr, "Analyzing host entry '<null>' (size = %lld)...\n",
              (unsigned long long)e->size);
      return NULL;
    }

    if (e->size) {
      __tgt_offload_entry entry = *e;

      void *varptr;
      uint32_t varsize;

      err = atmi_interop_hsa_get_symbol_info(get_gpu_mem_place(device_id),
                                             e->name, &varptr, &varsize);

      if (err != ATMI_STATUS_SUCCESS) {
        DP("Loading global '%s' (Failed)\n", e->name);
        // Inform the user what symbol prevented offloading
        fprintf(stderr, "Loading global '%s' (Failed)\n", e->name);
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
        err = atmi_memcpy(varptr, e->addr, sizeof(void *));
        if (err != ATMI_STATUS_SUCCESS)
          DP("Error when copying USM\n");
        DP("Copy linked variable host address (" DPxMOD ")"
           "to device address (" DPxMOD ")\n",
           DPxPTR(*((void **)e->addr)), DPxPTR(varptr));
      }

      continue;
    }

    DP("to find the kernel name: %s size: %lu\n", e->name, strlen(e->name));

    atmi_mem_place_t place = get_gpu_mem_place(device_id);
    uint32_t kernarg_segment_size;
    err = atmi_interop_hsa_get_kernel_info(
        place, e->name, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
        &kernarg_segment_size);

    // each arg is a void * in this openmp implementation
    uint32_t arg_num = kernarg_segment_size / sizeof(void *);
    std::vector<size_t> arg_sizes(arg_num);
    for (std::vector<size_t>::iterator it = arg_sizes.begin();
         it != arg_sizes.end(); it++) {
      *it = sizeof(void *);
    }

    // default value GENERIC (in case symbol is missing from cubin file)
    int8_t ExecModeVal = ExecutionModeType::GENERIC;

    // get flat group size if present, else Default_WG_Size
    int16_t WGSizeVal = RTLDeviceInfoTy::Default_WG_Size;

    // Max parallel level
    int16_t MaxParLevVal = 0;

    // get Kernel Descriptor if present.
    // Keep struct in sync wih getTgtAttributeStructQTy in CGOpenMPRuntime.cpp
    struct KernDescValType {
      uint16_t Version;
      uint16_t TSize;
      uint16_t WG_Size;
      uint8_t Mode;
      uint8_t HostServices;
      uint8_t MaxParallelLevel;
    };
    struct KernDescValType KernDescVal;
    std::string KernDescNameStr(e->name);
    KernDescNameStr += "_kern_desc";
    const char *KernDescName = KernDescNameStr.c_str();

    void *KernDescPtr;
    uint32_t KernDescSize;
    void *CallStackAddr;
    err = interop_get_symbol_info((char *)image->ImageStart, img_size,
                                  KernDescName, &KernDescPtr, &KernDescSize);

    if (err == ATMI_STATUS_SUCCESS) {
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
      DP("KernDesc: Mode: %d\n", KernDescVal.Mode);
      DP("KernDesc: HostServices: %x\n", KernDescVal.HostServices);
      DP("KernDesc: MaxParallelLevel: %x\n", KernDescVal.MaxParallelLevel);

      // gather location of callStack and size of struct
      MaxParLevVal = KernDescVal.MaxParallelLevel;
      if (MaxParLevVal > 0) {
        uint32_t varsize;
        const char *CsNam = "omptarget_nest_par_call_stack";
        err = atmi_interop_hsa_get_symbol_info(place, CsNam, &CallStackAddr,
                                               &varsize);
        if (err != ATMI_STATUS_SUCCESS) {
          fprintf(stderr, "Addr of %s failed\n", CsNam);
          return NULL;
        }
        void *StructSizePtr;
        const char *SsNam = "omptarget_nest_par_call_struct_size";
        err = interop_get_symbol_info((char *)image->ImageStart, img_size,
                                      SsNam, &StructSizePtr, &varsize);
        if ((err != ATMI_STATUS_SUCCESS) ||
            (varsize != sizeof(TgtStackItemSize))) {
          fprintf(stderr, "Addr of %s failed\n", SsNam);
          return NULL;
        }
        memcpy(&TgtStackItemSize, StructSizePtr, sizeof(TgtStackItemSize));
        DP("Size of our struct is %d\n", TgtStackItemSize);
      }

      // Get ExecMode
      ExecModeVal = KernDescVal.Mode;
      DP("ExecModeVal %d\n", ExecModeVal);
      if (KernDescVal.WG_Size == 0) {
        KernDescVal.WG_Size = RTLDeviceInfoTy::Default_WG_Size;
        DP("Setting KernDescVal.WG_Size to default %d\n", KernDescVal.WG_Size);
      }
      WGSizeVal = KernDescVal.WG_Size;
      DP("WGSizeVal %d\n", WGSizeVal);
      check("Loading KernDesc computation property", err);
    } else {
      DP("Warning: Loading KernDesc '%s' - symbol not found, ", KernDescName);

      // Generic
      std::string ExecModeNameStr(e->name);
      ExecModeNameStr += "_exec_mode";
      const char *ExecModeName = ExecModeNameStr.c_str();

      void *ExecModePtr;
      uint32_t varsize;
      err = interop_get_symbol_info((char *)image->ImageStart, img_size,
                                    ExecModeName, &ExecModePtr, &varsize);

      if (err == ATMI_STATUS_SUCCESS) {
        if ((size_t)varsize != sizeof(int8_t)) {
          DP("Loading global computation properties '%s' - size mismatch(%u != "
             "%lu)\n",
             ExecModeName, varsize, sizeof(int8_t));
          return NULL;
        }

        memcpy(&ExecModeVal, ExecModePtr, (size_t)varsize);

        DP("After loading global for %s ExecMode = %d\n", ExecModeName,
           ExecModeVal);

        if (ExecModeVal < 0 || ExecModeVal > 1) {
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

      // Flat group size
      std::string WGSizeNameStr(e->name);
      WGSizeNameStr += "_wg_size";
      const char *WGSizeName = WGSizeNameStr.c_str();

      void *WGSizePtr;
      uint32_t WGSize;
      err = interop_get_symbol_info((char *)image->ImageStart, img_size,
                                    WGSizeName, &WGSizePtr, &WGSize);

      if (err == ATMI_STATUS_SUCCESS) {
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

    KernelsList.push_back(KernelTy(ExecModeVal, WGSizeVal, MaxParLevVal,
                                   device_id, CallStackAddr, e->name,
                                   kernarg_segment_size));
    __tgt_offload_entry entry = *e;
    entry.addr = (void *)&KernelsList.back();
    DeviceInfo.addOffloadEntry(device_id, entry);
    DP("Entry point %ld maps to %s\n", e - HostBegin, e->name);
  }

  return DeviceInfo.getOffloadEntriesTable(device_id);
}

void *__tgt_rtl_data_alloc(int device_id, int64_t size, void *) {
  void *ptr = NULL;
  assert(device_id < DeviceInfo.NumberOfDevices && "Device ID too large");
  atmi_status_t err = atmi_malloc(&ptr, size, get_gpu_mem_place(device_id));
  DP("Tgt alloc data %ld bytes, (tgt:%016llx).\n", size,
     (long long unsigned)(Elf64_Addr)ptr);
  ptr = (err == ATMI_STATUS_SUCCESS) ? ptr : NULL;
  return ptr;
}

int32_t __tgt_rtl_data_submit(int device_id, void *tgt_ptr, void *hst_ptr,
                              int64_t size) {
  assert(device_id < DeviceInfo.NumberOfDevices && "Device ID too large");
  __tgt_async_info async_info;
  int32_t rc = dataSubmit(device_id, tgt_ptr, hst_ptr, size, &async_info);
  if (rc != OFFLOAD_SUCCESS)
    return OFFLOAD_FAIL;

  return __tgt_rtl_synchronize(device_id, &async_info);
}

int32_t __tgt_rtl_data_submit_async(int device_id, void *tgt_ptr, void *hst_ptr,
                                    int64_t size,
                                    __tgt_async_info *async_info_ptr) {
  assert(device_id < DeviceInfo.NumberOfDevices && "Device ID too large");
  if (async_info_ptr) {
    initAsyncInfoPtr(async_info_ptr);
    return dataSubmit(device_id, tgt_ptr, hst_ptr, size, async_info_ptr);
  } else {
    return __tgt_rtl_data_submit(device_id, tgt_ptr, hst_ptr, size);
  }
}

int32_t __tgt_rtl_data_retrieve(int device_id, void *hst_ptr, void *tgt_ptr,
                                int64_t size) {
  assert(device_id < DeviceInfo.NumberOfDevices && "Device ID too large");
  __tgt_async_info async_info;
  int32_t rc = dataRetrieve(device_id, hst_ptr, tgt_ptr, size, &async_info);
  if (rc != OFFLOAD_SUCCESS)
    return OFFLOAD_FAIL;

  return __tgt_rtl_synchronize(device_id, &async_info);
}

int32_t __tgt_rtl_data_retrieve_async(int device_id, void *hst_ptr,
                                      void *tgt_ptr, int64_t size,
                                      __tgt_async_info *async_info_ptr) {
  assert(async_info_ptr && "async_info is nullptr");
  assert(device_id < DeviceInfo.NumberOfDevices && "Device ID too large");
  initAsyncInfoPtr(async_info_ptr);
  return dataRetrieve(device_id, hst_ptr, tgt_ptr, size, async_info_ptr);
}

int32_t __tgt_rtl_data_delete(int device_id, void *tgt_ptr) {
  assert(device_id < DeviceInfo.NumberOfDevices && "Device ID too large");
  atmi_status_t err;
  DP("Tgt free data (tgt:%016llx).\n", (long long unsigned)(Elf64_Addr)tgt_ptr);
  err = atmi_free(tgt_ptr);
  if (err != ATMI_STATUS_SUCCESS) {
    DP("Error when freeing CUDA memory\n");
    return OFFLOAD_FAIL;
  }
  return OFFLOAD_SUCCESS;
}

// Determine launch values for threadsPerGroup and num_groups.
// Outputs: treadsPerGroup, num_groups
// Inputs: Max_Teams, Max_WG_Size, Warp_Size, ExecutionMode,
//         EnvTeamLimit, EnvNumTeams, num_teams, thread_limit,
//         loop_tripcount.
void getLaunchVals(int &threadsPerGroup, int &num_groups, int ConstWGSize,
                   int ExecutionMode, int EnvTeamLimit, int EnvNumTeams,
                   int num_teams, int thread_limit, uint64_t loop_tripcount) {

  int Max_Teams = DeviceInfo.EnvMaxTeamsDefault > 0
                      ? DeviceInfo.EnvMaxTeamsDefault
                      : DeviceInfo.Max_Teams;
  if (Max_Teams > DeviceInfo.HardTeamLimit)
    Max_Teams = DeviceInfo.HardTeamLimit;

  if (print_kernel_trace > 1) {
    fprintf(stderr, "RTLDeviceInfoTy::Max_Teams: %d\n",
            RTLDeviceInfoTy::Max_Teams);
    fprintf(stderr, "Max_Teams: %d\n", Max_Teams);
    fprintf(stderr, "RTLDeviceInfoTy::Warp_Size: %d\n",
            RTLDeviceInfoTy::Warp_Size);
    fprintf(stderr, "RTLDeviceInfoTy::Max_WG_Size: %d\n",
            RTLDeviceInfoTy::Max_WG_Size);
    fprintf(stderr, "RTLDeviceInfoTy::Default_WG_Size: %d\n",
            RTLDeviceInfoTy::Default_WG_Size);
    fprintf(stderr, "thread_limit: %d\n", thread_limit);
    fprintf(stderr, "threadsPerGroup: %d\n", threadsPerGroup);
    fprintf(stderr, "ConstWGSize: %d\n", ConstWGSize);
  }
  // check for thread_limit() clause
  if (thread_limit > 0) {
    threadsPerGroup = thread_limit;
    DP("Setting threads per block to requested %d\n", thread_limit);
    if (ExecutionMode == GENERIC) { // Add master warp for GENERIC
      threadsPerGroup += RTLDeviceInfoTy::Warp_Size;
      DP("Adding master wavefront: +%d threads\n", RTLDeviceInfoTy::Warp_Size);
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
  if (print_kernel_trace > 1)
    fprintf(stderr, "threadsPerGroup: %d\n", threadsPerGroup);
  DP("Preparing %d threads\n", threadsPerGroup);

  // Set default num_groups (teams)
  if (DeviceInfo.EnvTeamLimit > 0)
    num_groups = (Max_Teams < DeviceInfo.EnvTeamLimit)
                     ? Max_Teams
                     : DeviceInfo.EnvTeamLimit;
  else
    num_groups = Max_Teams;
  DP("Set default num of groups %d\n", num_groups);

  if (print_kernel_trace > 1) {
    fprintf(stderr, "num_groups: %d\n", num_groups);
    fprintf(stderr, "num_teams: %d\n", num_teams);
  }

  // Reduce num_groups if threadsPerGroup exceeds RTLDeviceInfoTy::Max_WG_Size
  // This reduction is typical for default case (no thread_limit clause).
  // or when user goes crazy with num_teams clause.
  // FIXME: We cant distinguish between a constant or variable thread limit.
  // So we only handle constant thread_limits.
  if (threadsPerGroup >
      RTLDeviceInfoTy::Default_WG_Size) //  256 < threadsPerGroup <= 1024
    // Should we round threadsPerGroup up to nearest RTLDeviceInfoTy::Warp_Size
    // here?
    num_groups = (Max_Teams * RTLDeviceInfoTy::Max_WG_Size) / threadsPerGroup;

  // check for num_teams() clause
  if (num_teams > 0) {
    num_groups = (num_teams < num_groups) ? num_teams : num_groups;
  }
  if (print_kernel_trace > 1) {
    fprintf(stderr, "num_groups: %d\n", num_groups);
    fprintf(stderr, "DeviceInfo.EnvNumTeams %d\n", DeviceInfo.EnvNumTeams);
    fprintf(stderr, "DeviceInfo.EnvTeamLimit %d\n", DeviceInfo.EnvTeamLimit);
  }

  if (DeviceInfo.EnvNumTeams > 0) {
    num_groups = (DeviceInfo.EnvNumTeams < num_groups) ? DeviceInfo.EnvNumTeams
                                                       : num_groups;
    DP("Modifying teams based on EnvNumTeams %d\n", DeviceInfo.EnvNumTeams);
  } else if (DeviceInfo.EnvTeamLimit > 0) {
    num_groups = (DeviceInfo.EnvTeamLimit < num_groups)
                     ? DeviceInfo.EnvTeamLimit
                     : num_groups;
    DP("Modifying teams based on EnvTeamLimit%d\n", DeviceInfo.EnvTeamLimit);
  } else {
    if (num_teams <= 0) {
      if (loop_tripcount > 0) {
        if (ExecutionMode == SPMD) {
          // round up to the nearest integer
          num_groups = ((loop_tripcount - 1) / threadsPerGroup) + 1;
        } else {
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
      if (print_kernel_trace > 1)
        fprintf(stderr, "Limiting num_groups %d to Max_Teams %d \n", num_groups,
                Max_Teams);
    }
    if (num_groups > num_teams && num_teams > 0) {
      num_groups = num_teams;
      if (print_kernel_trace > 1)
        fprintf(stderr, "Limiting num_groups %d to clause num_teams %d \n",
                num_groups, num_teams);
    }
  }

  // num_teams clause always honored, no matter what, unless DEFAULT is active.
  if (num_teams > 0) {
    num_groups = num_teams;
    // Cap num_groups to EnvMaxTeamsDefault if set.
    if (DeviceInfo.EnvMaxTeamsDefault > 0 &&
        num_groups > DeviceInfo.EnvMaxTeamsDefault)
      num_groups = DeviceInfo.EnvMaxTeamsDefault;
  }
  if (print_kernel_trace > 1) {
    fprintf(stderr, "threadsPerGroup: %d\n", threadsPerGroup);
    fprintf(stderr, "num_groups: %d\n", num_groups);
    fprintf(stderr, "loop_tripcount: %ld\n", loop_tripcount);
  }
  DP("Final %d num_groups and %d threadsPerGroup\n", num_groups,
     threadsPerGroup);
}

static void *AllocateNestedParallelCallMemory(int MaxParLevel, int NumGroups,
                                              int ThreadsPerGroup,
                                              int device_id,
                                              void *CallStackAddr, int SPMD) {
  if (print_kernel_trace > 1)
    fprintf(stderr, "MaxParLevel %d SPMD %d NumGroups %d NumThrds %d\n",
            MaxParLevel, SPMD, NumGroups, ThreadsPerGroup);
  // Total memory needed is Teams * Threads * ParLevels
  size_t NestedMemSize =
      MaxParLevel * NumGroups * ThreadsPerGroup * TgtStackItemSize * 4;

  if (print_kernel_trace > 1)
    fprintf(stderr, "NestedMemSize %ld \n", NestedMemSize);
  assert(device_id < DeviceInfo.NumberOfDevices && "Device ID too large");
  void *TgtPtr = NULL;
  atmi_status_t err =
      atmi_malloc(&TgtPtr, NestedMemSize, get_gpu_mem_place(device_id));
  err = atmi_memcpy(CallStackAddr, &TgtPtr, sizeof(void *));
  if (print_kernel_trace > 2)
    fprintf(stderr, "CallSck %lx TgtPtr %lx *TgtPtr %lx \n",
            (long)CallStackAddr, (long)&TgtPtr, (long)TgtPtr);
  if (err != ATMI_STATUS_SUCCESS) {
    fprintf(stderr, "Mem not wrtten to target, err %d\n", err);
  }
  return TgtPtr; // we need to free this after kernel.
}

static uint64_t acquire_available_packet_id(hsa_queue_t *queue) {
  uint64_t packet_id = hsa_queue_add_write_index_relaxed(queue, 1);
  bool full = true;
  while (full) {
    full =
        packet_id >= (queue->size + hsa_queue_load_read_index_acquire(queue));
  }
  return packet_id;
}

int32_t __tgt_rtl_run_target_team_region(int32_t device_id, void *tgt_entry_ptr,
                                         void **tgt_args,
                                         ptrdiff_t *tgt_offsets,
                                         int32_t arg_num, int32_t num_teams,
                                         int32_t thread_limit,
                                         uint64_t loop_tripcount) {
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

  /*
   * Set limit based on ThreadsPerGroup and GroupsPerDevice
   */
  int num_groups = 0;

  int threadsPerGroup = RTLDeviceInfoTy::Default_WG_Size;

  getLaunchVals(threadsPerGroup, num_groups, KernelInfo->ConstWGSize,
                KernelInfo->ExecutionMode, DeviceInfo.EnvTeamLimit,
                DeviceInfo.EnvNumTeams,
                num_teams,     // From run_region arg
                thread_limit,  // From run_region arg
                loop_tripcount // From run_region arg
  );

  void *TgtCallStack = NULL;
  if (KernelInfo->MaxParLevel > 0)
    TgtCallStack = AllocateNestedParallelCallMemory(
        KernelInfo->MaxParLevel, num_groups, threadsPerGroup,
        KernelInfo->device_id, KernelInfo->CallStackAddr,
        KernelInfo->ExecutionMode);

  if (print_kernel_trace > 0)
    // enum modes are SPMD, GENERIC, NONE 0,1,2
    fprintf(stderr,
            "DEVID:%2d SGN:%1d ConstWGSize:%-4d args:%2d teamsXthrds:(%4dX%4d) "
            "reqd:(%4dX%4d) n:%s\n",
            device_id, KernelInfo->ExecutionMode, KernelInfo->ConstWGSize,
            arg_num, num_groups, threadsPerGroup, num_teams, thread_limit,
            KernelInfo->Name);

  // Run on the device.
  {
    hsa_queue_t *queue = DeviceInfo.HSAQueues[device_id];
    uint64_t packet_id = acquire_available_packet_id(queue);

    const uint32_t mask = queue->size - 1; // size is a power of 2
    hsa_kernel_dispatch_packet_t *packet =
        (hsa_kernel_dispatch_packet_t *)queue->base_address +
        (packet_id & mask);

    // packet->header is written last
    packet->setup = UINT16_C(1) << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
    packet->workgroup_size_x = threadsPerGroup;
    packet->workgroup_size_y = 1;
    packet->workgroup_size_z = 1;
    packet->reserved0 = 0;
    packet->grid_size_x = num_groups * threadsPerGroup;
    packet->grid_size_y = 1;
    packet->grid_size_z = 1;
    packet->private_segment_size = 0;
    packet->group_segment_size = 0;
    packet->kernel_object = 0;
    packet->kernarg_address = 0;     // use the block allocator
    packet->reserved2 = 0;           // atmi writes id_ here
    packet->completion_signal = {0}; // may want a pool of signals

    std::string kernel_name = std::string(KernelInfo->Name);
    {
      assert(KernelInfoTable[device_id].find(kernel_name) !=
             KernelInfoTable[device_id].end());
      auto it = KernelInfoTable[device_id][kernel_name];
      packet->kernel_object = it.kernel_object;
      packet->private_segment_size = it.private_segment_size;
      packet->group_segment_size = it.group_segment_size;
      assert(arg_num == (int)it.num_args);
    }

    KernelArgPool *ArgPool = nullptr;
    {
      auto it = KernelArgPoolMap.find(std::string(KernelInfo->Name));
      if (it != KernelArgPoolMap.end()) {
        ArgPool = (it->second).get();
      }
    }
    if (!ArgPool) {
      fprintf(stderr, "Warning: No ArgPool for %s on device %d\n",
              KernelInfo->Name, device_id);
    }
    {
      void *kernarg = nullptr;
      if (ArgPool) {
        assert(ArgPool->kernarg_segment_size == (arg_num * sizeof(void *)));
        kernarg = ArgPool->allocate(arg_num);
      }
      if (!kernarg) {
        printf("Allocate kernarg failed\n");
        exit(1);
      }

      // Copy explicit arguments
      for (int i = 0; i < arg_num; i++) {
        memcpy((char *)kernarg + sizeof(void *) * i, args[i], sizeof(void *));
      }

      // Initialize implicit arguments. ATMI seems to leave most fields
      // uninitialized
      atmi_implicit_args_t *impl_args =
          reinterpret_cast<atmi_implicit_args_t *>(
              static_cast<char *>(kernarg) + ArgPool->kernarg_segment_size);
      memset(impl_args, 0,
             sizeof(atmi_implicit_args_t)); // may not be necessary
      impl_args->offset_x = 0;
      impl_args->offset_y = 0;
      impl_args->offset_z = 0;

      packet->kernarg_address = kernarg;
    }

    {
      hsa_signal_t s = DeviceInfo.FreeSignalPool.pop();
      if (s.handle == 0) {
        printf("Failed to get signal instance\n");
        exit(1);
      }
      packet->completion_signal = s;
      hsa_signal_store_relaxed(packet->completion_signal, 1);
    }

    core::packet_store_release(
        reinterpret_cast<uint32_t *>(packet),
        core::create_header(HSA_PACKET_TYPE_KERNEL_DISPATCH, 0,
                            ATMI_FENCE_SCOPE_SYSTEM, ATMI_FENCE_SCOPE_SYSTEM),
        packet->setup);

    hsa_signal_store_relaxed(queue->doorbell_signal, packet_id);

    while (hsa_signal_wait_acquire(packet->completion_signal,
                                   HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX,
                                   HSA_WAIT_STATE_BLOCKED) != 0)
      ;

    assert(ArgPool);
    ArgPool->deallocate(packet->kernarg_address);
    DeviceInfo.FreeSignalPool.push(packet->completion_signal);
  }

  DP("Kernel completed\n");
  // Free call stack for nested
  if (TgtCallStack)
    atmi_free(TgtCallStack);

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
                                          __tgt_async_info *async_info_ptr) {
  assert(async_info_ptr && "async_info is nullptr");
  initAsyncInfoPtr(async_info_ptr);

  // use one team and one thread
  // fix thread num
  int32_t team_num = 1;
  int32_t thread_limit = 0; // use default
  return __tgt_rtl_run_target_team_region(device_id, tgt_entry_ptr, tgt_args,
                                          tgt_offsets, arg_num, team_num,
                                          thread_limit, 0);
}

int32_t __tgt_rtl_synchronize(int32_t device_id,
                              __tgt_async_info *async_info_ptr) {
  assert(async_info_ptr && "async_info is nullptr");

  // Cuda asserts that async_info_ptr->Queue is non-null, but this invariant
  // is not ensured by devices.cpp for amdgcn
  // assert(async_info_ptr->Queue && "async_info_ptr->Queue is nullptr");
  if (async_info_ptr->Queue) {
    finiAsyncInfoPtr(async_info_ptr);
  }
  return OFFLOAD_SUCCESS;
}
