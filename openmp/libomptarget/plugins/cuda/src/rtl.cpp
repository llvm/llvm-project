//===----RTLs/cuda/src/rtl.cpp - Target RTLs Implementation ------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RTL for CUDA machine
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cstddef>
#include <cuda.h>
#include <list>
#include <string>
#include <vector>

#include "omptargetplugin.h"

#ifndef TARGET_NAME
#define TARGET_NAME CUDA
#endif

#ifdef OMPTARGET_DEBUG
static int DebugLevel = 0;

#define GETNAME2(name) #name
#define GETNAME(name) GETNAME2(name)
#define DP(...) \
  do { \
    if (DebugLevel > 0) { \
      DEBUGP("Target " GETNAME(TARGET_NAME) " RTL", __VA_ARGS__); \
    } \
  } while (false)
#else // OMPTARGET_DEBUG
#define DP(...) {}
#endif // OMPTARGET_DEBUG

#include "../../common/elf_common.c"

// Utility for retrieving and printing CUDA error string.
#ifdef CUDA_ERROR_REPORT
#define CUDA_ERR_STRING(err)                                                   \
  do {                                                                         \
    const char *errStr;                                                        \
    cuGetErrorString(err, &errStr);                                            \
    DP("CUDA error is: %s\n", errStr);                                         \
  } while (0)
#else
#define CUDA_ERR_STRING(err)                                                   \
  {}
#endif

/// Keep entries table per device.
struct FuncOrGblEntryTy {
  __tgt_target_table Table;
  std::vector<__tgt_offload_entry> Entries;
};

enum ExecutionModeType {
  SPMD, // constructors, destructors,
        // combined constructs (`teams distribute parallel for [simd]`)
  GENERIC, // everything else
  NONE
};

/// Use a single entity to encode a kernel and a set of flags
struct KernelTy {
  CUfunction Func;

  // execution mode of kernel
  // 0 - SPMD mode (without master warp)
  // 1 - Generic mode (with master warp)
  int8_t ExecutionMode;

  KernelTy(CUfunction _Func, int8_t _ExecutionMode)
      : Func(_Func), ExecutionMode(_ExecutionMode) {}
};

/// Device envrionment data
/// Manually sync with the deviceRTL side for now, move to a dedicated header file later.
struct omptarget_device_environmentTy {
  int32_t debug_level;
};

/// List that contains all the kernels.
/// FIXME: we may need this to be per device and per library.
std::list<KernelTy> KernelsList;

/// Class containing all the device information.
class RTLDeviceInfoTy {
  std::vector<std::list<FuncOrGblEntryTy>> FuncGblEntries;

public:
  int NumberOfDevices;
  std::vector<CUmodule> Modules;
  std::vector<CUcontext> Contexts;

  // Device properties
  std::vector<int> ThreadsPerBlock;
  std::vector<int> BlocksPerGrid;
  std::vector<int> WarpSize;

  // OpenMP properties
  std::vector<int> NumTeams;
  std::vector<int> NumThreads;

  // OpenMP Environment properties
  int EnvNumTeams;
  int EnvTeamLimit;

  // OpenMP Requires Flags
  int64_t RequiresFlags;

  //static int EnvNumThreads;
  static const int HardTeamLimit = 1<<16; // 64k
  static const int HardThreadLimit = 1024;
  static const int DefaultNumTeams = 128;
  static const int DefaultNumThreads = 128;

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
  void clearOffloadEntriesTable(int32_t device_id) {
    assert(device_id < (int32_t)FuncGblEntries.size() &&
           "Unexpected device id!");
    FuncGblEntries[device_id].emplace_back();
    FuncOrGblEntryTy &E = FuncGblEntries[device_id].back();
    E.Entries.clear();
    E.Table.EntriesBegin = E.Table.EntriesEnd = 0;
  }

  RTLDeviceInfoTy() {
#ifdef OMPTARGET_DEBUG
    if (char *envStr = getenv("LIBOMPTARGET_DEBUG")) {
      DebugLevel = std::stoi(envStr);
    }
#endif // OMPTARGET_DEBUG

    DP("Start initializing CUDA\n");

    CUresult err = cuInit(0);
    if (err != CUDA_SUCCESS) {
      DP("Error when initializing CUDA\n");
      CUDA_ERR_STRING(err);
      return;
    }

    NumberOfDevices = 0;

    err = cuDeviceGetCount(&NumberOfDevices);
    if (err != CUDA_SUCCESS) {
      DP("Error when getting CUDA device count\n");
      CUDA_ERR_STRING(err);
      return;
    }

    if (NumberOfDevices == 0) {
      DP("There are no devices supporting CUDA.\n");
      return;
    }

    FuncGblEntries.resize(NumberOfDevices);
    Contexts.resize(NumberOfDevices);
    ThreadsPerBlock.resize(NumberOfDevices);
    BlocksPerGrid.resize(NumberOfDevices);
    WarpSize.resize(NumberOfDevices);
    NumTeams.resize(NumberOfDevices);
    NumThreads.resize(NumberOfDevices);

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

    // Default state.
    RequiresFlags = OMP_REQ_UNDEFINED;
  }

  ~RTLDeviceInfoTy() {
    // Close modules
    for (auto &module : Modules)
      if (module) {
        CUresult err = cuModuleUnload(module);
        if (err != CUDA_SUCCESS) {
          DP("Error when unloading CUDA module\n");
          CUDA_ERR_STRING(err);
        }
      }

    // Destroy contexts
    for (auto &ctx : Contexts)
      if (ctx) {
        CUresult err = cuCtxDestroy(ctx);
        if (err != CUDA_SUCCESS) {
          DP("Error when destroying CUDA context\n");
          CUDA_ERR_STRING(err);
        }
      }
  }
};

static RTLDeviceInfoTy DeviceInfo;

#ifdef __cplusplus
extern "C" {
#endif

int32_t __tgt_rtl_is_valid_binary(__tgt_device_image *image) {
  return elf_check_machine(image, 190); // EM_CUDA = 190.
}

int32_t __tgt_rtl_number_of_devices() { return DeviceInfo.NumberOfDevices; }

int64_t __tgt_rtl_init_requires(int64_t RequiresFlags) {
  DP("Init requires flags to %ld\n", RequiresFlags);
  DeviceInfo.RequiresFlags = RequiresFlags;
  return RequiresFlags;
}

int32_t __tgt_rtl_init_device(int32_t device_id) {

  CUdevice cuDevice;
  DP("Getting device %d\n", device_id);
  CUresult err = cuDeviceGet(&cuDevice, device_id);
  if (err != CUDA_SUCCESS) {
    DP("Error when getting CUDA device with id = %d\n", device_id);
    CUDA_ERR_STRING(err);
    return OFFLOAD_FAIL;
  }

  // Create the context and save it to use whenever this device is selected.
  err = cuCtxCreate(&DeviceInfo.Contexts[device_id], CU_CTX_SCHED_BLOCKING_SYNC,
                    cuDevice);
  if (err != CUDA_SUCCESS) {
    DP("Error when creating a CUDA context\n");
    CUDA_ERR_STRING(err);
    return OFFLOAD_FAIL;
  }

  // Query attributes to determine number of threads/block and blocks/grid.
  int maxGridDimX;
  err = cuDeviceGetAttribute(&maxGridDimX, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
                             cuDevice);
  if (err != CUDA_SUCCESS) {
    DP("Error getting max grid dimension, use default\n");
    DeviceInfo.BlocksPerGrid[device_id] = RTLDeviceInfoTy::DefaultNumTeams;
  } else if (maxGridDimX <= RTLDeviceInfoTy::HardTeamLimit) {
    DeviceInfo.BlocksPerGrid[device_id] = maxGridDimX;
    DP("Using %d CUDA blocks per grid\n", maxGridDimX);
  } else {
    DeviceInfo.BlocksPerGrid[device_id] = RTLDeviceInfoTy::HardTeamLimit;
    DP("Max CUDA blocks per grid %d exceeds the hard team limit %d, capping "
       "at the hard limit\n",
       maxGridDimX, RTLDeviceInfoTy::HardTeamLimit);
  }

  // We are only exploiting threads along the x axis.
  int maxBlockDimX;
  err = cuDeviceGetAttribute(&maxBlockDimX, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
                             cuDevice);
  if (err != CUDA_SUCCESS) {
    DP("Error getting max block dimension, use default\n");
    DeviceInfo.ThreadsPerBlock[device_id] = RTLDeviceInfoTy::DefaultNumThreads;
  } else if (maxBlockDimX <= RTLDeviceInfoTy::HardThreadLimit) {
    DeviceInfo.ThreadsPerBlock[device_id] = maxBlockDimX;
    DP("Using %d CUDA threads per block\n", maxBlockDimX);
  } else {
    DeviceInfo.ThreadsPerBlock[device_id] = RTLDeviceInfoTy::HardThreadLimit;
    DP("Max CUDA threads per block %d exceeds the hard thread limit %d, capping"
       "at the hard limit\n",
       maxBlockDimX, RTLDeviceInfoTy::HardThreadLimit);
  }

  int warpSize;
  err =
      cuDeviceGetAttribute(&warpSize, CU_DEVICE_ATTRIBUTE_WARP_SIZE, cuDevice);
  if (err != CUDA_SUCCESS) {
    DP("Error getting warp size, assume default\n");
    DeviceInfo.WarpSize[device_id] = 32;
  } else {
    DeviceInfo.WarpSize[device_id] = warpSize;
  }

  // Adjust teams to the env variables
  if (DeviceInfo.EnvTeamLimit > 0 &&
      DeviceInfo.BlocksPerGrid[device_id] > DeviceInfo.EnvTeamLimit) {
    DeviceInfo.BlocksPerGrid[device_id] = DeviceInfo.EnvTeamLimit;
    DP("Capping max CUDA blocks per grid to OMP_TEAM_LIMIT=%d\n",
        DeviceInfo.EnvTeamLimit);
  }

  DP("Max number of CUDA blocks %d, threads %d & warp size %d\n",
     DeviceInfo.BlocksPerGrid[device_id], DeviceInfo.ThreadsPerBlock[device_id],
     DeviceInfo.WarpSize[device_id]);

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
  if (DeviceInfo.NumTeams[device_id] > DeviceInfo.BlocksPerGrid[device_id]) {
    DeviceInfo.NumTeams[device_id] = DeviceInfo.BlocksPerGrid[device_id];
    DP("Default number of teams exceeds device limit, capping at %d\n",
        DeviceInfo.BlocksPerGrid[device_id]);
  }

  // Set default number of threads
  DeviceInfo.NumThreads[device_id] = RTLDeviceInfoTy::DefaultNumThreads;
  DP("Default number of threads set according to library's default %d\n",
          RTLDeviceInfoTy::DefaultNumThreads);
  if (DeviceInfo.NumThreads[device_id] >
      DeviceInfo.ThreadsPerBlock[device_id]) {
    DeviceInfo.NumTeams[device_id] = DeviceInfo.ThreadsPerBlock[device_id];
    DP("Default number of threads exceeds device limit, capping at %d\n",
        DeviceInfo.ThreadsPerBlock[device_id]);
  }

  return OFFLOAD_SUCCESS;
}

__tgt_target_table *__tgt_rtl_load_binary(int32_t device_id,
    __tgt_device_image *image) {

  // Set the context we are using.
  CUresult err = cuCtxSetCurrent(DeviceInfo.Contexts[device_id]);
  if (err != CUDA_SUCCESS) {
    DP("Error when setting a CUDA context for device %d\n", device_id);
    CUDA_ERR_STRING(err);
    return NULL;
  }

  // Clear the offload table as we are going to create a new one.
  DeviceInfo.clearOffloadEntriesTable(device_id);

  // Create the module and extract the function pointers.

  CUmodule cumod;
  DP("Load data from image " DPxMOD "\n", DPxPTR(image->ImageStart));
  err = cuModuleLoadDataEx(&cumod, image->ImageStart, 0, NULL, NULL);
  if (err != CUDA_SUCCESS) {
    DP("Error when loading CUDA module\n");
    CUDA_ERR_STRING(err);
    return NULL;
  }

  DP("CUDA module successfully loaded!\n");
  DeviceInfo.Modules.push_back(cumod);

  // Find the symbols in the module by name.
  __tgt_offload_entry *HostBegin = image->EntriesBegin;
  __tgt_offload_entry *HostEnd = image->EntriesEnd;

  for (__tgt_offload_entry *e = HostBegin; e != HostEnd; ++e) {

    if (!e->addr) {
      // We return NULL when something like this happens, the host should have
      // always something in the address to uniquely identify the target region.
      DP("Invalid binary: host entry '<null>' (size = %zd)...\n", e->size);

      return NULL;
    }

    if (e->size) {
      __tgt_offload_entry entry = *e;

      CUdeviceptr cuptr;
      size_t cusize;
      err = cuModuleGetGlobal(&cuptr, &cusize, cumod, e->name);

      if (err != CUDA_SUCCESS) {
        DP("Loading global '%s' (Failed)\n", e->name);
        CUDA_ERR_STRING(err);
        return NULL;
      }

      if (cusize != e->size) {
        DP("Loading global '%s' - size mismatch (%zd != %zd)\n", e->name,
            cusize, e->size);
        CUDA_ERR_STRING(err);
        return NULL;
      }

      DP("Entry point " DPxMOD " maps to global %s (" DPxMOD ")\n",
          DPxPTR(e - HostBegin), e->name, DPxPTR(cuptr));
      entry.addr = (void *)cuptr;

      if (DeviceInfo.RequiresFlags & OMP_REQ_UNIFIED_SHARED_MEMORY &&
          e->flags & OMP_DECLARE_TARGET_LINK) {
        // If unified memory is present any target link variables
        // can access host addresses directly. There is no longer a
        // need for device copies.
        cuMemcpyHtoD(cuptr, e->addr, sizeof(void *));
        DP("Copy linked variable host address (" DPxMOD ")"
           "to device address (" DPxMOD ")\n",
          DPxPTR(*((void**)e->addr)), DPxPTR(cuptr));
      }

      DeviceInfo.addOffloadEntry(device_id, entry);

      continue;
    }

    CUfunction fun;
    err = cuModuleGetFunction(&fun, cumod, e->name);

    if (err != CUDA_SUCCESS) {
      DP("Loading '%s' (Failed)\n", e->name);
      CUDA_ERR_STRING(err);
      return NULL;
    }

    DP("Entry point " DPxMOD " maps to %s (" DPxMOD ")\n",
        DPxPTR(e - HostBegin), e->name, DPxPTR(fun));

    // default value GENERIC (in case symbol is missing from cubin file)
    int8_t ExecModeVal = ExecutionModeType::GENERIC;
    std::string ExecModeNameStr (e->name);
    ExecModeNameStr += "_exec_mode";
    const char *ExecModeName = ExecModeNameStr.c_str();

    CUdeviceptr ExecModePtr;
    size_t cusize;
    err = cuModuleGetGlobal(&ExecModePtr, &cusize, cumod, ExecModeName);
    if (err == CUDA_SUCCESS) {
      if ((size_t)cusize != sizeof(int8_t)) {
        DP("Loading global exec_mode '%s' - size mismatch (%zd != %zd)\n",
           ExecModeName, cusize, sizeof(int8_t));
        CUDA_ERR_STRING(err);
        return NULL;
      }

      err = cuMemcpyDtoH(&ExecModeVal, ExecModePtr, cusize);
      if (err != CUDA_SUCCESS) {
        DP("Error when copying data from device to host. Pointers: "
           "host = " DPxMOD ", device = " DPxMOD ", size = %zd\n",
           DPxPTR(&ExecModeVal), DPxPTR(ExecModePtr), cusize);
        CUDA_ERR_STRING(err);
        return NULL;
      }

      if (ExecModeVal < 0 || ExecModeVal > 1) {
        DP("Error wrong exec_mode value specified in cubin file: %d\n",
           ExecModeVal);
        return NULL;
      }
    } else {
      DP("Loading global exec_mode '%s' - symbol missing, using default value "
          "GENERIC (1)\n", ExecModeName);
      CUDA_ERR_STRING(err);
    }

    KernelsList.push_back(KernelTy(fun, ExecModeVal));

    __tgt_offload_entry entry = *e;
    entry.addr = (void *)&KernelsList.back();
    DeviceInfo.addOffloadEntry(device_id, entry);
  }

  // send device environment data to the device
  {
    omptarget_device_environmentTy device_env;

    device_env.debug_level = 0;

#ifdef OMPTARGET_DEBUG
    if (char *envStr = getenv("LIBOMPTARGET_DEVICE_RTL_DEBUG")) {
      device_env.debug_level = std::stoi(envStr);
    }
#endif

    const char * device_env_Name="omptarget_device_environment";
    CUdeviceptr device_env_Ptr;
    size_t cusize;

    err = cuModuleGetGlobal(&device_env_Ptr, &cusize, cumod, device_env_Name);

    if (err == CUDA_SUCCESS) {
      if ((size_t)cusize != sizeof(device_env)) {
        DP("Global device_environment '%s' - size mismatch (%zu != %zu)\n",
            device_env_Name, cusize, sizeof(int32_t));
        CUDA_ERR_STRING(err);
        return NULL;
      }

      err = cuMemcpyHtoD(device_env_Ptr, &device_env, cusize);
      if (err != CUDA_SUCCESS) {
        DP("Error when copying data from host to device. Pointers: "
            "host = " DPxMOD ", device = " DPxMOD ", size = %zu\n",
            DPxPTR(&device_env), DPxPTR(device_env_Ptr), cusize);
        CUDA_ERR_STRING(err);
        return NULL;
      }

      DP("Sending global device environment data %zu bytes\n", (size_t)cusize);
    } else {
      DP("Finding global device environment '%s' - symbol missing.\n", device_env_Name);
      DP("Continue, considering this is a device RTL which does not accept envrionment setting.\n");
    }
  }

  return DeviceInfo.getOffloadEntriesTable(device_id);
}

void *__tgt_rtl_data_alloc(int32_t device_id, int64_t size, void *hst_ptr) {
  if (size == 0) {
    return NULL;
  }

  // Set the context we are using.
  CUresult err = cuCtxSetCurrent(DeviceInfo.Contexts[device_id]);
  if (err != CUDA_SUCCESS) {
    DP("Error while trying to set CUDA current context\n");
    CUDA_ERR_STRING(err);
    return NULL;
  }

  CUdeviceptr ptr;
  err = cuMemAlloc(&ptr, size);
  if (err != CUDA_SUCCESS) {
    DP("Error while trying to allocate %d\n", err);
    CUDA_ERR_STRING(err);
    return NULL;
  }

  void *vptr = (void *)ptr;
  return vptr;
}

int32_t __tgt_rtl_data_submit(int32_t device_id, void *tgt_ptr, void *hst_ptr,
    int64_t size) {
  // Set the context we are using.
  CUresult err = cuCtxSetCurrent(DeviceInfo.Contexts[device_id]);
  if (err != CUDA_SUCCESS) {
    DP("Error when setting CUDA context\n");
    CUDA_ERR_STRING(err);
    return OFFLOAD_FAIL;
  }

  err = cuMemcpyHtoD((CUdeviceptr)tgt_ptr, hst_ptr, size);
  if (err != CUDA_SUCCESS) {
    DP("Error when copying data from host to device. Pointers: host = " DPxMOD
       ", device = " DPxMOD ", size = %" PRId64 "\n", DPxPTR(hst_ptr),
       DPxPTR(tgt_ptr), size);
    CUDA_ERR_STRING(err);
    return OFFLOAD_FAIL;
  }
  return OFFLOAD_SUCCESS;
}

int32_t __tgt_rtl_data_retrieve(int32_t device_id, void *hst_ptr, void *tgt_ptr,
    int64_t size) {
  // Set the context we are using.
  CUresult err = cuCtxSetCurrent(DeviceInfo.Contexts[device_id]);
  if (err != CUDA_SUCCESS) {
    DP("Error when setting CUDA context\n");
    CUDA_ERR_STRING(err);
    return OFFLOAD_FAIL;
  }

  err = cuMemcpyDtoH(hst_ptr, (CUdeviceptr)tgt_ptr, size);
  if (err != CUDA_SUCCESS) {
    DP("Error when copying data from device to host. Pointers: host = " DPxMOD
        ", device = " DPxMOD ", size = %" PRId64 "\n", DPxPTR(hst_ptr),
        DPxPTR(tgt_ptr), size);
    CUDA_ERR_STRING(err);
    return OFFLOAD_FAIL;
  }
  return OFFLOAD_SUCCESS;
}

int32_t __tgt_rtl_data_delete(int32_t device_id, void *tgt_ptr) {
  // Set the context we are using.
  CUresult err = cuCtxSetCurrent(DeviceInfo.Contexts[device_id]);
  if (err != CUDA_SUCCESS) {
    DP("Error when setting CUDA context\n");
    CUDA_ERR_STRING(err);
    return OFFLOAD_FAIL;
  }

  err = cuMemFree((CUdeviceptr)tgt_ptr);
  if (err != CUDA_SUCCESS) {
    DP("Error when freeing CUDA memory\n");
    CUDA_ERR_STRING(err);
    return OFFLOAD_FAIL;
  }
  return OFFLOAD_SUCCESS;
}

int32_t __tgt_rtl_run_target_team_region(int32_t device_id, void *tgt_entry_ptr,
    void **tgt_args, ptrdiff_t *tgt_offsets, int32_t arg_num, int32_t team_num,
    int32_t thread_limit, uint64_t loop_tripcount) {
  // Set the context we are using.
  CUresult err = cuCtxSetCurrent(DeviceInfo.Contexts[device_id]);
  if (err != CUDA_SUCCESS) {
    DP("Error when setting CUDA context\n");
    CUDA_ERR_STRING(err);
    return OFFLOAD_FAIL;
  }

  // All args are references.
  std::vector<void *> args(arg_num);
  std::vector<void *> ptrs(arg_num);

  for (int32_t i = 0; i < arg_num; ++i) {
    ptrs[i] = (void *)((intptr_t)tgt_args[i] + tgt_offsets[i]);
    args[i] = &ptrs[i];
  }

  KernelTy *KernelInfo = (KernelTy *)tgt_entry_ptr;

  int cudaThreadsPerBlock;

  if (thread_limit > 0) {
    cudaThreadsPerBlock = thread_limit;
    DP("Setting CUDA threads per block to requested %d\n", thread_limit);
    // Add master warp if necessary
    if (KernelInfo->ExecutionMode == GENERIC) {
      cudaThreadsPerBlock += DeviceInfo.WarpSize[device_id];
      DP("Adding master warp: +%d threads\n", DeviceInfo.WarpSize[device_id]);
    }
  } else {
    cudaThreadsPerBlock = DeviceInfo.NumThreads[device_id];
    DP("Setting CUDA threads per block to default %d\n",
        DeviceInfo.NumThreads[device_id]);
  }

  if (cudaThreadsPerBlock > DeviceInfo.ThreadsPerBlock[device_id]) {
    cudaThreadsPerBlock = DeviceInfo.ThreadsPerBlock[device_id];
    DP("Threads per block capped at device limit %d\n",
        DeviceInfo.ThreadsPerBlock[device_id]);
  }

  int kernel_limit;
  err = cuFuncGetAttribute(&kernel_limit,
      CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, KernelInfo->Func);
  if (err == CUDA_SUCCESS) {
    if (kernel_limit < cudaThreadsPerBlock) {
      cudaThreadsPerBlock = kernel_limit;
      DP("Threads per block capped at kernel limit %d\n", kernel_limit);
    }
  }

  int cudaBlocksPerGrid;
  if (team_num <= 0) {
    if (loop_tripcount > 0 && DeviceInfo.EnvNumTeams < 0) {
      if (KernelInfo->ExecutionMode == SPMD) {
        // We have a combined construct, i.e. `target teams distribute parallel
        // for [simd]`. We launch so many teams so that each thread will
        // execute one iteration of the loop.
        // round up to the nearest integer
        cudaBlocksPerGrid = ((loop_tripcount - 1) / cudaThreadsPerBlock) + 1;
      } else {
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
        cudaBlocksPerGrid = loop_tripcount;
      }
      DP("Using %d teams due to loop trip count %" PRIu64 " and number of "
          "threads per block %d\n", cudaBlocksPerGrid, loop_tripcount,
          cudaThreadsPerBlock);
    } else {
      cudaBlocksPerGrid = DeviceInfo.NumTeams[device_id];
      DP("Using default number of teams %d\n", DeviceInfo.NumTeams[device_id]);
    }
  } else if (team_num > DeviceInfo.BlocksPerGrid[device_id]) {
    cudaBlocksPerGrid = DeviceInfo.BlocksPerGrid[device_id];
    DP("Capping number of teams to team limit %d\n",
        DeviceInfo.BlocksPerGrid[device_id]);
  } else {
    cudaBlocksPerGrid = team_num;
    DP("Using requested number of teams %d\n", team_num);
  }

  // Run on the device.
  DP("Launch kernel with %d blocks and %d threads\n", cudaBlocksPerGrid,
     cudaThreadsPerBlock);

  err = cuLaunchKernel(KernelInfo->Func, cudaBlocksPerGrid, 1, 1,
      cudaThreadsPerBlock, 1, 1, 0 /*bytes of shared memory*/, 0, &args[0], 0);
  if (err != CUDA_SUCCESS) {
    DP("Device kernel launch failed!\n");
    CUDA_ERR_STRING(err);
    return OFFLOAD_FAIL;
  }

  DP("Launch of entry point at " DPxMOD " successful!\n",
      DPxPTR(tgt_entry_ptr));

  CUresult sync_err = cuCtxSynchronize();
  if (sync_err != CUDA_SUCCESS) {
    DP("Kernel execution error at " DPxMOD "!\n", DPxPTR(tgt_entry_ptr));
    CUDA_ERR_STRING(sync_err);
    return OFFLOAD_FAIL;
  } else {
    DP("Kernel execution at " DPxMOD " successful!\n", DPxPTR(tgt_entry_ptr));
  }

  return OFFLOAD_SUCCESS;
}

int32_t __tgt_rtl_run_target_region(int32_t device_id, void *tgt_entry_ptr,
    void **tgt_args, ptrdiff_t *tgt_offsets, int32_t arg_num) {
  // use one team and the default number of threads.
  const int32_t team_num = 1;
  const int32_t thread_limit = 0;
  return __tgt_rtl_run_target_team_region(device_id, tgt_entry_ptr, tgt_args,
      tgt_offsets, arg_num, team_num, thread_limit, 0);
}

#ifdef __cplusplus
}
#endif
