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
#include <list>
#include <vector>

// Header from ATMI interface
#include "atmi_interop_hsa.h"
#include "atmi_runtime.h"
// Header from hostcall
#include "amd_hostcall.h"

#include "omptargetplugin.h"

// Get static gpu grid values from clang target-specific constants managed
// in the clang header file GpuGridValues.h
#include "llvm/Frontend/OpenMP/OMPGridValues.h"

#ifndef TARGET_NAME
#define TARGET_NAME AMDHSA
#endif

int print_kernel_trace;

// Size of the target call stack struture
int32_t TgtStackItemSize = 0;

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

/// Use a single entity to encode a kernel and a set of flags
struct KernelTy {
  atmi_kernel_t Func;

  // execution mode of kernel
  // 0 - SPMD mode (without master warp)
  // 1 - Generic mode (with master warp)
  int8_t ExecutionMode;
  int16_t ConstWGSize;
  int8_t MaxParLevel;
  int32_t device_id;
  void *CallStackAddr;
  const char *Name;

  KernelTy(atmi_kernel_t _Func, int8_t _ExecutionMode, int16_t _ConstWGSize,
           int8_t _MaxParLevel, int32_t _device_id, void *_CallStackAddr,
           const char *_Name)
      : Func(_Func), ExecutionMode(_ExecutionMode), ConstWGSize(_ConstWGSize),
        MaxParLevel(_MaxParLevel), device_id(_device_id),
        CallStackAddr(_CallStackAddr), Name(_Name) {
    DP("Construct kernelinfo: ExecMode %d\n", ExecutionMode);
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
          printf("rtl.cpp: err %d\n",err);
        assert(err == HSA_STATUS_SUCCESS);

        if (device_type == HSA_DEVICE_TYPE_GPU) {
          res->push_back(agent);
        }
        return HSA_STATUS_SUCCESS;
      },
      &res);

  // iterate_agents fails iff HSA runtime not yet initialized
  if (print_kernel_trace > 0 && err != HSA_STATUS_SUCCESS)
    printf("rtl.cpp: err %d\n",err);
  assert(err == HSA_STATUS_SUCCESS);
  return res;
}

/// Class containing all the device information
class RTLDeviceInfoTy {
  std::vector<std::list<FuncOrGblEntryTy>> FuncGblEntries;

public:
  int NumberOfDevices;

  // GPU devices
  std::vector<hsa_agent_t> HSAAgents;

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

  // static int EnvNumThreads;
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
    for (std::vector<__tgt_offload_entry>::iterator it = E.Entries.begin();
         it != E.Entries.end(); it++) {
      KernelTy *kernel_info = (KernelTy *)it->addr;
      if (kernel_info->Func.handle != 0ull)
        atmi_kernel_release(kernel_info->Func);
    }
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
    atmi_status_t err = atmi_init(ATMI_DEVTYPE_GPU);
    if (err != ATMI_STATUS_SUCCESS) {
      DP("Error when initializing HSA-ATMI\n");
      return;
    }
    // Init hostcall soon after initializing ATMI
    atmi_hostcall_init();

    HSAAgents = find_gpu_agents();
    NumberOfDevices = (int)HSAAgents.size();   

    if (NumberOfDevices == 0) {
      DP("There are no devices supporting HSA.\n");
      return;
    } else {
      DP("There are %d devices supporting HSA.\n", NumberOfDevices);
    }

    // Init the device info
    FuncGblEntries.resize(NumberOfDevices);
    ThreadsPerGroup.resize(NumberOfDevices);
    ComputeUnits.resize(NumberOfDevices);
    GroupsPerDevice.resize(NumberOfDevices);
    WarpSize.resize(NumberOfDevices);
    NumTeams.resize(NumberOfDevices);
    NumThreads.resize(NumberOfDevices);

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
    // Terminate hostcall before finalizing ATMI
    atmi_hostcall_terminate();
    atmi_finalize();
  }
};

#include "../../../src/device_env_struct.h"

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

  hsa_agent_t &agent = DeviceInfo.HSAAgents[device_id];

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

__tgt_target_table *__tgt_rtl_load_binary(int32_t device_id,
                                          __tgt_device_image *image) {
  size_t img_size = (char *)image->ImageEnd - (char *)image->ImageStart;

  DeviceInfo.clearOffloadEntriesTable(device_id);

  // We do not need to set the ELF version because the caller of this function
  // had to do that to decide the right runtime to use

  if (!elf_machine_id_is_amdgcn(image)) {
    return NULL;
  }

  atmi_status_t err;
  {
    atmi_platform_type_t platform = AMDGCN;
    err = atmi_module_register_from_memory_to_place(
        (void **)&image->ImageStart, &img_size, &platform, 1, get_gpu_place(device_id));

    check("Module registering", err);
    if (err != ATMI_STATUS_SUCCESS) {
      char GPUName[64] = "--unknown gpu--";
      hsa_agent_t &agent = DeviceInfo.HSAAgents[device_id];
      (void) hsa_agent_get_info(agent, (hsa_agent_info_t)HSA_AGENT_INFO_NAME,
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
    atmi_kernel_t kernel;
    uint32_t kernel_segment_size;
    err = atmi_interop_hsa_get_kernel_info(
        place, e->name, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
        &kernel_segment_size);

    // each arg is a void * in this openmp implementation
    uint32_t arg_num = kernel_segment_size / sizeof(void *);
    std::vector<size_t> arg_sizes(arg_num);
    for (std::vector<size_t>::iterator it = arg_sizes.begin();
         it != arg_sizes.end(); it++) {
      *it = sizeof(void *);
    }

    atmi_status_t stat = atmi_kernel_create(&kernel, arg_num, &arg_sizes[0], 1,
                                            ATMI_DEVTYPE_GPU, e->name);

    if (stat != ATMI_STATUS_SUCCESS) {
      DP("atmi_kernel_create failed %d\n", stat);
      return NULL;
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
    err = atmi_interop_hsa_get_symbol_info(place, KernDescName, &KernDescPtr,
                                           &KernDescSize);
    if (err == ATMI_STATUS_SUCCESS) {
      if ((size_t)KernDescSize != sizeof(KernDescVal))
        DP("Loading global computation properties '%s' - size mismatch (%u != "
           "%lu)\n",
           KernDescName, KernDescSize, sizeof(KernDescVal));

      err = atmi_memcpy(&KernDescVal, KernDescPtr, (size_t)KernDescSize);
      if (err != ATMI_STATUS_SUCCESS) {
        DP("Error when copying data from device to host. Pointers: "
           "host = " DPxMOD ", device = " DPxMOD ", size = %u\n",
           DPxPTR(&KernDescVal), DPxPTR(KernDescPtr), KernDescSize);
        return NULL;
      }
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
	const char * CsNam = "omptarget_nest_par_call_stack";
          err = atmi_interop_hsa_get_symbol_info(place, CsNam,
              &CallStackAddr, &varsize);
        if (err != ATMI_STATUS_SUCCESS) {
	  fprintf(stderr, "Addr of %s failed\n",CsNam);
	  return NULL;
	}
	void *StructSizePtr;
	const char * SsNam = "omptarget_nest_par_call_struct_size";
        err = atmi_interop_hsa_get_symbol_info(place, SsNam,
              &StructSizePtr, &varsize);
        if (err != ATMI_STATUS_SUCCESS) {
	  fprintf(stderr,
	    "Addr of %s failed\n", SsNam);
	  return NULL;
	}
        err = atmi_memcpy(&TgtStackItemSize, StructSizePtr, (size_t)varsize);
        if (err != ATMI_STATUS_SUCCESS) {
          DP("Error when copying data from device to host. Pointers: "
             "host = " DPxMOD ", device = " DPxMOD ", size = %u\n",
             DPxPTR(&TgtStackItemSize), DPxPTR(StructSizePtr), varsize);
          return NULL;
        }
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
      err = atmi_interop_hsa_get_symbol_info(place, ExecModeName, &ExecModePtr,
                                             &varsize);
      if (err == ATMI_STATUS_SUCCESS) {
        if ((size_t)varsize != sizeof(int8_t)) {
          DP("Loading global computation properties '%s' - size mismatch(%u != "
             "%lu)\n",
             ExecModeName, varsize, sizeof(int8_t));
          return NULL;
        }

        err = atmi_memcpy(&ExecModeVal, ExecModePtr, (size_t)varsize);
        if (err != ATMI_STATUS_SUCCESS) {
          DP("Error when copying data from device to host. Pointers: "
             "host = " DPxMOD ", device = " DPxMOD ", size = %u\n",
             DPxPTR(&ExecModeVal), DPxPTR(ExecModePtr), varsize);
          return NULL;
        }
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
      err = atmi_interop_hsa_get_symbol_info(place, WGSizeName, &WGSizePtr,
                                             &WGSize);
      if (err == ATMI_STATUS_SUCCESS) {
        if ((size_t)WGSize != sizeof(int16_t)) {
          DP("Loading global computation properties '%s' - size mismatch (%u "
             "!= "
             "%lu)\n",
             WGSizeName, WGSize, sizeof(int16_t));
          return NULL;
        }

        err = atmi_memcpy(&WGSizeVal, WGSizePtr, (size_t)WGSize);
        if (err != ATMI_STATUS_SUCCESS) {
          DP("Error when copying data from device to host. Pointers: "
             "host = " DPxMOD ", device = " DPxMOD ", size = %u\n",
             DPxPTR(&WGSizeVal), DPxPTR(WGSizePtr), WGSize);
          return NULL;
        }
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

    KernelsList.push_back(KernelTy(kernel, ExecModeVal, WGSizeVal, MaxParLevVal,
                                   device_id, CallStackAddr, e->name));
    __tgt_offload_entry entry = *e;
    entry.addr = (void *)&KernelsList.back();
    DeviceInfo.addOffloadEntry(device_id, entry);
    DP("Entry point %ld maps to %s\n", e - HostBegin, e->name);
  }

  { // send device environment here

    omptarget_device_environmentTy host_device_env;
    host_device_env.num_devices = DeviceInfo.NumberOfDevices;
    host_device_env.device_num = device_id;
    host_device_env.debug_level = 0;
#ifdef OMPTARGET_DEBUG
    if (char *envStr = getenv("LIBOMPTARGET_DEVICE_RTL_DEBUG")) {
      host_device_env.debug_level = std::stoi(envStr);
    }
#endif

    const char *device_env_Name = "omptarget_device_environment";
    void *device_env_Ptr;
    uint32_t varsize;

    atmi_mem_place_t place = get_gpu_mem_place(device_id);
    err = atmi_interop_hsa_get_symbol_info(place, device_env_Name,
                                           &device_env_Ptr, &varsize);

    if (err == ATMI_STATUS_SUCCESS) {
      if ((size_t)varsize != sizeof(host_device_env)) {
        DP("Global device_environment '%s' - size mismatch (%u != %lu)\n",
           device_env_Name, varsize, sizeof(int32_t));
        return NULL;
      }

      err = atmi_memcpy(device_env_Ptr, &host_device_env, (size_t)varsize);
      if (err != ATMI_STATUS_SUCCESS) {
        DP("Error when copying data from host to device. Pointers: "
           "host = " DPxMOD ", device = " DPxMOD ", size = %u\n",
           DPxPTR(&host_device_env), DPxPTR(device_env_Ptr), varsize);
        return NULL;
      }

      DP("Sending global device environment %lu bytes\n", (size_t)varsize);
    } else {
      DP("Finding global device environment '%s' - symbol missing.\n",
         device_env_Name);
      // no need to return NULL, consider this is a not a device debug build.
      // return NULL;
    }

    check("Sending device environment", err);
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
  __tgt_async_info async_info;
  int32_t rc = dataSubmit(device_id, tgt_ptr, hst_ptr, size, &async_info);
  if (rc != OFFLOAD_SUCCESS)
    return OFFLOAD_FAIL;

  return __tgt_rtl_synchronize(device_id, &async_info);
}

int32_t __tgt_rtl_data_submit_async(int device_id, void *tgt_ptr, void *hst_ptr,
	                      int64_t size, __tgt_async_info *async_info_ptr) {
  if (async_info_ptr)
    return dataSubmit(device_id, tgt_ptr, hst_ptr, size, async_info_ptr);

  __tgt_async_info async_info;
  int32_t rc = dataSubmit(device_id, tgt_ptr, hst_ptr, size, &async_info);
  if (rc != OFFLOAD_SUCCESS)
    return OFFLOAD_FAIL;

  return __tgt_rtl_synchronize(device_id, &async_info);
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

int32_t __tgt_rtl_data_retrieve_async(int device_id, void *hst_ptr, void *tgt_ptr,
                                int64_t size,  __tgt_async_info *async_info_ptr) {
  assert(device_id < DeviceInfo.NumberOfDevices && "Device ID too large");
  if (async_info_ptr)
    return dataRetrieve(device_id, hst_ptr, tgt_ptr, size, async_info_ptr);

  __tgt_async_info async_info;
  int32_t rc = dataRetrieve(device_id, hst_ptr, tgt_ptr, size, &async_info);
  if (rc != OFFLOAD_SUCCESS)
    return OFFLOAD_FAIL;

  return __tgt_rtl_synchronize(device_id, &async_info);
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

void retrieveDeviceEnv(int32_t device_id) {
  int err;
  const char *device_env_Name = "omptarget_device_environment";
  omptarget_device_environmentTy host_device_env;
  void *device_env_Ptr;
  uint32_t varsize;

  err = atmi_interop_hsa_get_symbol_info(
      get_gpu_mem_place(device_id), device_env_Name, &device_env_Ptr, &varsize);

  if (err == ATMI_STATUS_SUCCESS) {
    if ((size_t)varsize != sizeof(host_device_env)) {
      DP("Global device_environment '%s' - size mismatch (%u != %lu)\n",
         device_env_Name, varsize, sizeof(int32_t));
      return;
    }

    err = __tgt_rtl_data_retrieve(device_id, &host_device_env, device_env_Ptr,
                                  varsize);
    if (err != 0) {
      DP("Error when copying data from device to host . Pointers: "
         "host = " DPxMOD ", device = " DPxMOD ", size = %u\n",
         DPxPTR(&host_device_env), DPxPTR(device_env_Ptr), varsize);
      return;
    }

    DP("Retrieving device environment %lu bytes\n", (size_t)varsize);
  } else {
    DP("Retrieving device environment '%s' - symbol missing.\n",
       device_env_Name);
  }

  check("Retreiving updated device environment", err);
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
  size_t NestedMemSize = MaxParLevel * NumGroups * ThreadsPerGroup *
	                 TgtStackItemSize * 4;

  if (print_kernel_trace > 1)
    fprintf(stderr, "NestedMemSize %ld \n", NestedMemSize);
  assert(device_id < DeviceInfo.NumberOfDevices && "Device ID too large");
  void *TgtPtr = NULL;
  atmi_status_t err = atmi_malloc(&TgtPtr, NestedMemSize, get_gpu_mem_place(device_id));
  err = atmi_memcpy(CallStackAddr, &TgtPtr, sizeof(void*));
  if (print_kernel_trace > 2)
    fprintf(stderr, "CallSck %lx TgtPtr %lx *TgtPtr %lx \n",
                    (long)CallStackAddr, (long)&TgtPtr, (long)TgtPtr);
  if (err != ATMI_STATUS_SUCCESS) {
    fprintf(stderr, "Mem not wrtten to target, err %d\n", err);
  }
  return TgtPtr; // we need to free this after kernel.
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
		     KernelInfo->MaxParLevel, num_groups,
                     threadsPerGroup, KernelInfo->device_id,
                     KernelInfo->CallStackAddr,
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
  atmi_kernel_t kernel = KernelInfo->Func;
  ATMI_LPARM_1D(lparm, num_groups * threadsPerGroup);
  lparm->groupDim[0] = threadsPerGroup;
  lparm->synchronous = ATMI_TRUE;
  lparm->groupable = ATMI_FALSE;
  lparm->place = get_gpu_place(device_id);
  atmi_task_launch(lparm, kernel, &args[0]);

  DP("Kernel completed\n");
  // Free call stack for nested
  if (TgtCallStack)
    atmi_free(TgtCallStack);

  retrieveDeviceEnv(device_id);

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

int32_t __tgt_rtl_run_target_region_async(int32_t device_id, void *tgt_entry_ptr,
                                    void **tgt_args, ptrdiff_t *tgt_offsets,
                                    int32_t arg_num,
				    __tgt_async_info *async_info) {
  // use one team and one thread
  // fix thread num
  int32_t team_num = 1;
  int32_t thread_limit = 0; // use default
  return __tgt_rtl_run_target_team_region(device_id, tgt_entry_ptr, tgt_args,
                                          tgt_offsets, arg_num, team_num,
                                          thread_limit, 0);
}

int32_t __tgt_rtl_synchronize(int32_t device_id, __tgt_async_info *async_info) {
  assert(async_info && "async_info is nullptr");
  return OFFLOAD_SUCCESS;
}


