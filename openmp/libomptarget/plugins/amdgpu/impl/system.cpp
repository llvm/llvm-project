/*===--------------------------------------------------------------------------
 *              ATMI (Asynchronous Task and Memory Interface)
 *
 * This file is distributed under the MIT License. See LICENSE.txt for details.
 *===------------------------------------------------------------------------*/
#include <gelf.h>
#include <libelf.h>

#include <cassert>
#include <cstdarg>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <string>

#include "internal.h"
#include "machine.h"
#include "rt.h"

#include "msgpack.h"

#define msgpackErrorCheck(msg, status)                                         \
  if (status != 0) {                                                           \
    printf("[%s:%d] %s failed\n", __FILE__, __LINE__, #msg);                   \
    return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;                               \
  } else {                                                                     \
  }

typedef unsigned char *address;
/*
 * Note descriptors.
 */
typedef struct {
  uint32_t n_namesz; /* Length of note's name. */
  uint32_t n_descsz; /* Length of note's value. */
  uint32_t n_type;   /* Type of note. */
  // then name
  // then padding, optional
  // then desc, at 4 byte alignment (not 8, despite being elf64)
} Elf_Note;

// The following include file and following structs/enums
// have been replicated on a per-use basis below. For example,
// llvm::AMDGPU::HSAMD::Kernel::Metadata has several fields,
// but we may care only about kernargSegmentSize_ for now, so
// we just include that field in our KernelMD implementation. We
// chose this approach to replicate in order to avoid forcing
// a dependency on LLVM_INCLUDE_DIR just to compile the runtime.
// #include "llvm/Support/AMDGPUMetadata.h"
// typedef llvm::AMDGPU::HSAMD::Metadata CodeObjectMD;
// typedef llvm::AMDGPU::HSAMD::Kernel::Metadata KernelMD;
// typedef llvm::AMDGPU::HSAMD::Kernel::Arg::Metadata KernelArgMD;
// using llvm::AMDGPU::HSAMD::AccessQualifier;
// using llvm::AMDGPU::HSAMD::AddressSpaceQualifier;
// using llvm::AMDGPU::HSAMD::ValueKind;
// using llvm::AMDGPU::HSAMD::ValueType;

class KernelArgMD {
public:
  enum class ValueKind {
    HiddenGlobalOffsetX,
    HiddenGlobalOffsetY,
    HiddenGlobalOffsetZ,
    HiddenNone,
    HiddenPrintfBuffer,
    HiddenDefaultQueue,
    HiddenCompletionAction,
    HiddenMultiGridSyncArg,
    HiddenHostcallBuffer,
    Unknown
  };

  KernelArgMD()
      : name_(std::string()), typeName_(std::string()), size_(0), offset_(0),
        align_(0), valueKind_(ValueKind::Unknown) {}

  // fields
  std::string name_;
  std::string typeName_;
  uint32_t size_;
  uint32_t offset_;
  uint32_t align_;
  ValueKind valueKind_;
};

class KernelMD {
public:
  KernelMD() : kernargSegmentSize_(0ull) {}

  // fields
  uint64_t kernargSegmentSize_;
};

static const std::map<std::string, KernelArgMD::ValueKind> ArgValueKind = {
    //    Including only those fields that are relevant to the runtime.
    //    {"ByValue", KernelArgMD::ValueKind::ByValue},
    //    {"GlobalBuffer", KernelArgMD::ValueKind::GlobalBuffer},
    //    {"DynamicSharedPointer",
    //    KernelArgMD::ValueKind::DynamicSharedPointer},
    //    {"Sampler", KernelArgMD::ValueKind::Sampler},
    //    {"Image", KernelArgMD::ValueKind::Image},
    //    {"Pipe", KernelArgMD::ValueKind::Pipe},
    //    {"Queue", KernelArgMD::ValueKind::Queue},
    {"HiddenGlobalOffsetX", KernelArgMD::ValueKind::HiddenGlobalOffsetX},
    {"HiddenGlobalOffsetY", KernelArgMD::ValueKind::HiddenGlobalOffsetY},
    {"HiddenGlobalOffsetZ", KernelArgMD::ValueKind::HiddenGlobalOffsetZ},
    {"HiddenNone", KernelArgMD::ValueKind::HiddenNone},
    {"HiddenPrintfBuffer", KernelArgMD::ValueKind::HiddenPrintfBuffer},
    {"HiddenDefaultQueue", KernelArgMD::ValueKind::HiddenDefaultQueue},
    {"HiddenCompletionAction", KernelArgMD::ValueKind::HiddenCompletionAction},
    {"HiddenMultiGridSyncArg", KernelArgMD::ValueKind::HiddenMultiGridSyncArg},
    {"HiddenHostcallBuffer", KernelArgMD::ValueKind::HiddenHostcallBuffer},
    // v3
    //    {"by_value", KernelArgMD::ValueKind::ByValue},
    //    {"global_buffer", KernelArgMD::ValueKind::GlobalBuffer},
    //    {"dynamic_shared_pointer",
    //    KernelArgMD::ValueKind::DynamicSharedPointer},
    //    {"sampler", KernelArgMD::ValueKind::Sampler},
    //    {"image", KernelArgMD::ValueKind::Image},
    //    {"pipe", KernelArgMD::ValueKind::Pipe},
    //    {"queue", KernelArgMD::ValueKind::Queue},
    {"hidden_global_offset_x", KernelArgMD::ValueKind::HiddenGlobalOffsetX},
    {"hidden_global_offset_y", KernelArgMD::ValueKind::HiddenGlobalOffsetY},
    {"hidden_global_offset_z", KernelArgMD::ValueKind::HiddenGlobalOffsetZ},
    {"hidden_none", KernelArgMD::ValueKind::HiddenNone},
    {"hidden_printf_buffer", KernelArgMD::ValueKind::HiddenPrintfBuffer},
    {"hidden_default_queue", KernelArgMD::ValueKind::HiddenDefaultQueue},
    {"hidden_completion_action",
     KernelArgMD::ValueKind::HiddenCompletionAction},
    {"hidden_multigrid_sync_arg",
     KernelArgMD::ValueKind::HiddenMultiGridSyncArg},
    {"hidden_hostcall_buffer", KernelArgMD::ValueKind::HiddenHostcallBuffer},
};

// public variables -- TODO(ashwinma) move these to a runtime object?
atmi_machine_t g_atmi_machine;
ATLMachine g_atl_machine;

hsa_region_t atl_gpu_kernarg_region;
std::vector<hsa_amd_memory_pool_t> atl_gpu_kernarg_pools;
hsa_region_t atl_cpu_kernarg_region;

static std::vector<hsa_executable_t> g_executables;

std::map<std::string, std::string> KernelNameMap;
std::vector<std::map<std::string, atl_kernel_info_t>> KernelInfoTable;
std::vector<std::map<std::string, atl_symbol_info_t>> SymbolInfoTable;

bool g_atmi_initialized = false;
bool g_atmi_hostcall_required = false;

struct timespec context_init_time;
int context_init_time_init = 0;

/*
   atlc is all internal global values.
   The structure atl_context_t is defined in atl_internal.h
   Most references will use the global structure prefix atlc.
   However the pointer value atlc_p-> is equivalent to atlc.

*/

atl_context_t atlc = {.struct_initialized = false};
atl_context_t *atlc_p = NULL;

namespace core {
/* Machine Info */
atmi_machine_t *Runtime::GetMachineInfo() {
  if (!atlc.g_hsa_initialized)
    return NULL;
  return &g_atmi_machine;
}

void atl_set_atmi_initialized() {
  // FIXME: thread safe? locks?
  g_atmi_initialized = true;
}

void atl_reset_atmi_initialized() {
  // FIXME: thread safe? locks?
  g_atmi_initialized = false;
}

bool atl_is_atmi_initialized() { return g_atmi_initialized; }

void allow_access_to_all_gpu_agents(void *ptr) {
  hsa_status_t err;
  std::vector<ATLGPUProcessor> &gpu_procs =
      g_atl_machine.processors<ATLGPUProcessor>();
  std::vector<hsa_agent_t> agents;
  for (uint32_t i = 0; i < gpu_procs.size(); i++) {
    agents.push_back(gpu_procs[i].agent());
  }
  err = hsa_amd_agents_allow_access(agents.size(), &agents[0], NULL, ptr);
  ErrorCheck(Allow agents ptr access, err);
}

atmi_status_t Runtime::Initialize() {
  atmi_devtype_t devtype = ATMI_DEVTYPE_GPU;
  if (atl_is_atmi_initialized())
    return ATMI_STATUS_SUCCESS;

  if (devtype == ATMI_DEVTYPE_ALL || devtype & ATMI_DEVTYPE_GPU) {
    ATMIErrorCheck(GPU context init, atl_init_gpu_context());
  }

  atl_set_atmi_initialized();
  return ATMI_STATUS_SUCCESS;
}

atmi_status_t Runtime::Finalize() {
  // TODO(ashwinma): Finalize all processors, queues, signals, kernarg memory
  // regions
  hsa_status_t err;

  for (uint32_t i = 0; i < g_executables.size(); i++) {
    err = hsa_executable_destroy(g_executables[i]);
    ErrorCheck(Destroying executable, err);
  }

  for (uint32_t i = 0; i < SymbolInfoTable.size(); i++) {
    SymbolInfoTable[i].clear();
  }
  SymbolInfoTable.clear();
  for (uint32_t i = 0; i < KernelInfoTable.size(); i++) {
    KernelInfoTable[i].clear();
  }
  KernelInfoTable.clear();

  atl_reset_atmi_initialized();
  err = hsa_shut_down();
  ErrorCheck(Shutting down HSA, err);

  return ATMI_STATUS_SUCCESS;
}

void atmi_init_context_structs() {
  atlc_p = &atlc;
  atlc.struct_initialized = true; /* This only gets called one time */
  atlc.g_hsa_initialized = false;
  atlc.g_gpu_initialized = false;
  atlc.g_tasks_initialized = false;
}

// Implement memory_pool iteration function
static hsa_status_t get_memory_pool_info(hsa_amd_memory_pool_t memory_pool,
                                         void *data) {
  ATLProcessor *proc = reinterpret_cast<ATLProcessor *>(data);
  hsa_status_t err = HSA_STATUS_SUCCESS;
  // Check if the memory_pool is allowed to allocate, i.e. do not return group
  // memory
  bool alloc_allowed = false;
  err = hsa_amd_memory_pool_get_info(
      memory_pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,
      &alloc_allowed);
  ErrorCheck(Alloc allowed in memory pool check, err);
  if (alloc_allowed) {
    uint32_t global_flag = 0;
    err = hsa_amd_memory_pool_get_info(
        memory_pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &global_flag);
    ErrorCheck(Get memory pool info, err);
    if (HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED & global_flag) {
      ATLMemory new_mem(memory_pool, *proc, ATMI_MEMTYPE_FINE_GRAINED);
      proc->addMemory(new_mem);
      if (HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT & global_flag) {
        DEBUG_PRINT("GPU kernel args pool handle: %lu\n", memory_pool.handle);
        atl_gpu_kernarg_pools.push_back(memory_pool);
      }
    } else {
      ATLMemory new_mem(memory_pool, *proc, ATMI_MEMTYPE_COARSE_GRAINED);
      proc->addMemory(new_mem);
    }
  }

  return err;
}

static hsa_status_t get_agent_info(hsa_agent_t agent, void *data) {
  hsa_status_t err = HSA_STATUS_SUCCESS;
  hsa_device_type_t device_type;
  err = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
  ErrorCheck(Get device type info, err);
  switch (device_type) {
  case HSA_DEVICE_TYPE_CPU: {
    ;
    ATLCPUProcessor new_proc(agent);
    err = hsa_amd_agent_iterate_memory_pools(agent, get_memory_pool_info,
                                             &new_proc);
    ErrorCheck(Iterate all memory pools, err);
    g_atl_machine.addProcessor(new_proc);
  } break;
  case HSA_DEVICE_TYPE_GPU: {
    ;
    hsa_profile_t profile;
    err = hsa_agent_get_info(agent, HSA_AGENT_INFO_PROFILE, &profile);
    ErrorCheck(Query the agent profile, err);
    atmi_devtype_t gpu_type;
    gpu_type =
        (profile == HSA_PROFILE_FULL) ? ATMI_DEVTYPE_iGPU : ATMI_DEVTYPE_dGPU;
    ATLGPUProcessor new_proc(agent, gpu_type);
    err = hsa_amd_agent_iterate_memory_pools(agent, get_memory_pool_info,
                                             &new_proc);
    ErrorCheck(Iterate all memory pools, err);
    g_atl_machine.addProcessor(new_proc);
  } break;
  case HSA_DEVICE_TYPE_DSP: {
    err = HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  } break;
  }

  return err;
}

hsa_status_t get_fine_grained_region(hsa_region_t region, void *data) {
  hsa_region_segment_t segment;
  hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &segment);
  if (segment != HSA_REGION_SEGMENT_GLOBAL) {
    return HSA_STATUS_SUCCESS;
  }
  hsa_region_global_flag_t flags;
  hsa_region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags);
  if (flags & HSA_REGION_GLOBAL_FLAG_FINE_GRAINED) {
    hsa_region_t *ret = reinterpret_cast<hsa_region_t *>(data);
    *ret = region;
    return HSA_STATUS_INFO_BREAK;
  }
  return HSA_STATUS_SUCCESS;
}

/* Determines if a memory region can be used for kernarg allocations.  */
static hsa_status_t get_kernarg_memory_region(hsa_region_t region, void *data) {
  hsa_region_segment_t segment;
  hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &segment);
  if (HSA_REGION_SEGMENT_GLOBAL != segment) {
    return HSA_STATUS_SUCCESS;
  }

  hsa_region_global_flag_t flags;
  hsa_region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags);
  if (flags & HSA_REGION_GLOBAL_FLAG_KERNARG) {
    hsa_region_t *ret = reinterpret_cast<hsa_region_t *>(data);
    *ret = region;
    return HSA_STATUS_INFO_BREAK;
  }

  return HSA_STATUS_SUCCESS;
}

static hsa_status_t init_compute_and_memory() {
  hsa_status_t err;

  /* Iterate over the agents and pick the gpu agent */
  err = hsa_iterate_agents(get_agent_info, NULL);
  if (err == HSA_STATUS_INFO_BREAK) {
    err = HSA_STATUS_SUCCESS;
  }
  ErrorCheck(Getting a gpu agent, err);
  if (err != HSA_STATUS_SUCCESS)
    return err;

  /* Init all devices or individual device types? */
  std::vector<ATLCPUProcessor> &cpu_procs =
      g_atl_machine.processors<ATLCPUProcessor>();
  std::vector<ATLGPUProcessor> &gpu_procs =
      g_atl_machine.processors<ATLGPUProcessor>();
  /* For CPU memory pools, add other devices that can access them directly
   * or indirectly */
  for (auto &cpu_proc : cpu_procs) {
    for (auto &cpu_mem : cpu_proc.memories()) {
      hsa_amd_memory_pool_t pool = cpu_mem.memory();
      for (auto &gpu_proc : gpu_procs) {
        hsa_agent_t agent = gpu_proc.agent();
        hsa_amd_memory_pool_access_t access;
        hsa_amd_agent_memory_pool_get_info(
            agent, pool, HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &access);
        if (access != 0) {
          // this means not NEVER, but could be YES or NO
          // add this memory pool to the proc
          gpu_proc.addMemory(cpu_mem);
        }
      }
    }
  }

  /* FIXME: are the below combinations of procs and memory pools needed?
   * all to all compare procs with their memory pools and add those memory
   * pools that are accessible by the target procs */
  for (auto &gpu_proc : gpu_procs) {
    for (auto &gpu_mem : gpu_proc.memories()) {
      hsa_amd_memory_pool_t pool = gpu_mem.memory();
      for (auto &cpu_proc : cpu_procs) {
        hsa_agent_t agent = cpu_proc.agent();
        hsa_amd_memory_pool_access_t access;
        hsa_amd_agent_memory_pool_get_info(
            agent, pool, HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &access);
        if (access != 0) {
          // this means not NEVER, but could be YES or NO
          // add this memory pool to the proc
          cpu_proc.addMemory(gpu_mem);
        }
      }
    }
  }

  g_atmi_machine.device_count_by_type[ATMI_DEVTYPE_CPU] = cpu_procs.size();
  g_atmi_machine.device_count_by_type[ATMI_DEVTYPE_GPU] = gpu_procs.size();

  size_t num_procs = cpu_procs.size() + gpu_procs.size();
  // g_atmi_machine.devices = (atmi_device_t *)malloc(num_procs *
  // sizeof(atmi_device_t));
  atmi_device_t *all_devices = reinterpret_cast<atmi_device_t *>(
      malloc(num_procs * sizeof(atmi_device_t)));
  int num_iGPUs = 0;
  int num_dGPUs = 0;
  for (uint32_t i = 0; i < gpu_procs.size(); i++) {
    if (gpu_procs[i].type() == ATMI_DEVTYPE_iGPU)
      num_iGPUs++;
    else
      num_dGPUs++;
  }
  assert(num_iGPUs + num_dGPUs == gpu_procs.size() &&
         "Number of dGPUs and iGPUs do not add up");
  DEBUG_PRINT("CPU Agents: %lu\n", cpu_procs.size());
  DEBUG_PRINT("iGPU Agents: %d\n", num_iGPUs);
  DEBUG_PRINT("dGPU Agents: %d\n", num_dGPUs);
  DEBUG_PRINT("GPU Agents: %lu\n", gpu_procs.size());

  g_atmi_machine.device_count_by_type[ATMI_DEVTYPE_iGPU] = num_iGPUs;
  g_atmi_machine.device_count_by_type[ATMI_DEVTYPE_dGPU] = num_dGPUs;

  int cpus_begin = 0;
  int cpus_end = cpu_procs.size();
  int gpus_begin = cpu_procs.size();
  int gpus_end = cpu_procs.size() + gpu_procs.size();
  g_atmi_machine.devices_by_type[ATMI_DEVTYPE_CPU] = &all_devices[cpus_begin];
  g_atmi_machine.devices_by_type[ATMI_DEVTYPE_GPU] = &all_devices[gpus_begin];
  g_atmi_machine.devices_by_type[ATMI_DEVTYPE_iGPU] = &all_devices[gpus_begin];
  g_atmi_machine.devices_by_type[ATMI_DEVTYPE_dGPU] = &all_devices[gpus_begin];
  int proc_index = 0;
  for (int i = cpus_begin; i < cpus_end; i++) {
    all_devices[i].type = cpu_procs[proc_index].type();

    std::vector<ATLMemory> memories = cpu_procs[proc_index].memories();
    int fine_memories_size = 0;
    int coarse_memories_size = 0;
    DEBUG_PRINT("CPU memory types:\t");
    for (auto &memory : memories) {
      atmi_memtype_t type = memory.type();
      if (type == ATMI_MEMTYPE_FINE_GRAINED) {
        fine_memories_size++;
        DEBUG_PRINT("Fine\t");
      } else {
        coarse_memories_size++;
        DEBUG_PRINT("Coarse\t");
      }
    }
    DEBUG_PRINT("\nFine Memories : %d", fine_memories_size);
    DEBUG_PRINT("\tCoarse Memories : %d\n", coarse_memories_size);
    proc_index++;
  }
  proc_index = 0;
  for (int i = gpus_begin; i < gpus_end; i++) {
    all_devices[i].type = gpu_procs[proc_index].type();

    std::vector<ATLMemory> memories = gpu_procs[proc_index].memories();
    int fine_memories_size = 0;
    int coarse_memories_size = 0;
    DEBUG_PRINT("GPU memory types:\t");
    for (auto &memory : memories) {
      atmi_memtype_t type = memory.type();
      if (type == ATMI_MEMTYPE_FINE_GRAINED) {
        fine_memories_size++;
        DEBUG_PRINT("Fine\t");
      } else {
        coarse_memories_size++;
        DEBUG_PRINT("Coarse\t");
      }
    }
    DEBUG_PRINT("\nFine Memories : %d", fine_memories_size);
    DEBUG_PRINT("\tCoarse Memories : %d\n", coarse_memories_size);
    proc_index++;
  }
  proc_index = 0;
  atl_cpu_kernarg_region.handle = (uint64_t)-1;
  if (cpu_procs.size() > 0) {
    err = hsa_agent_iterate_regions(
        cpu_procs[0].agent(), get_fine_grained_region, &atl_cpu_kernarg_region);
    if (err == HSA_STATUS_INFO_BREAK) {
      err = HSA_STATUS_SUCCESS;
    }
    err = (atl_cpu_kernarg_region.handle == (uint64_t)-1) ? HSA_STATUS_ERROR
                                                          : HSA_STATUS_SUCCESS;
    ErrorCheck(Finding a CPU kernarg memory region handle, err);
  }
  /* Find a memory region that supports kernel arguments.  */
  atl_gpu_kernarg_region.handle = (uint64_t)-1;
  if (gpu_procs.size() > 0) {
    hsa_agent_iterate_regions(gpu_procs[0].agent(), get_kernarg_memory_region,
                              &atl_gpu_kernarg_region);
    err = (atl_gpu_kernarg_region.handle == (uint64_t)-1) ? HSA_STATUS_ERROR
                                                          : HSA_STATUS_SUCCESS;
    ErrorCheck(Finding a kernarg memory region, err);
  }
  if (num_procs > 0)
    return HSA_STATUS_SUCCESS;
  else
    return HSA_STATUS_ERROR_NOT_INITIALIZED;
}

hsa_status_t init_hsa() {
  if (atlc.g_hsa_initialized == false) {
    DEBUG_PRINT("Initializing HSA...");
    hsa_status_t err = hsa_init();
    ErrorCheck(Initializing the hsa runtime, err);
    if (err != HSA_STATUS_SUCCESS)
      return err;

    err = init_compute_and_memory();
    if (err != HSA_STATUS_SUCCESS)
      return err;
    ErrorCheck(After initializing compute and memory, err);

    int gpu_count = g_atl_machine.processorCount<ATLGPUProcessor>();
    KernelInfoTable.resize(gpu_count);
    SymbolInfoTable.resize(gpu_count);
    for (uint32_t i = 0; i < SymbolInfoTable.size(); i++)
      SymbolInfoTable[i].clear();
    for (uint32_t i = 0; i < KernelInfoTable.size(); i++)
      KernelInfoTable[i].clear();
    atlc.g_hsa_initialized = true;
    DEBUG_PRINT("done\n");
  }
  return HSA_STATUS_SUCCESS;
}

void init_tasks() {
  if (atlc.g_tasks_initialized != false)
    return;
  std::vector<hsa_agent_t> gpu_agents;
  int gpu_count = g_atl_machine.processorCount<ATLGPUProcessor>();
  for (int gpu = 0; gpu < gpu_count; gpu++) {
    atmi_place_t place = ATMI_PLACE_GPU(0, gpu);
    ATLGPUProcessor &proc = get_processor<ATLGPUProcessor>(place);
    gpu_agents.push_back(proc.agent());
  }
  atlc.g_tasks_initialized = true;
}

hsa_status_t callbackEvent(const hsa_amd_event_t *event, void *data) {
#if (ROCM_VERSION_MAJOR >= 3) ||                                               \
    (ROCM_VERSION_MAJOR >= 2 && ROCM_VERSION_MINOR >= 3)
  if (event->event_type == HSA_AMD_GPU_MEMORY_FAULT_EVENT) {
#else
  if (event->event_type == GPU_MEMORY_FAULT_EVENT) {
#endif
    hsa_amd_gpu_memory_fault_info_t memory_fault = event->memory_fault;
    // memory_fault.agent
    // memory_fault.virtual_address
    // memory_fault.fault_reason_mask
    // fprintf("[GPU Error at %p: Reason is ", memory_fault.virtual_address);
    std::stringstream stream;
    stream << std::hex << (uintptr_t)memory_fault.virtual_address;
    std::string addr("0x" + stream.str());

    std::string err_string = "[GPU Memory Error] Addr: " + addr;
    err_string += " Reason: ";
    if (!(memory_fault.fault_reason_mask & 0x00111111)) {
      err_string += "No Idea! ";
    } else {
      if (memory_fault.fault_reason_mask & 0x00000001)
        err_string += "Page not present or supervisor privilege. ";
      if (memory_fault.fault_reason_mask & 0x00000010)
        err_string += "Write access to a read-only page. ";
      if (memory_fault.fault_reason_mask & 0x00000100)
        err_string += "Execute access to a page marked NX. ";
      if (memory_fault.fault_reason_mask & 0x00001000)
        err_string += "Host access only. ";
      if (memory_fault.fault_reason_mask & 0x00010000)
        err_string += "ECC failure (if supported by HW). ";
      if (memory_fault.fault_reason_mask & 0x00100000)
        err_string += "Can't determine the exact fault address. ";
    }
    fprintf(stderr, "%s\n", err_string.c_str());
    return HSA_STATUS_ERROR;
  }
  return HSA_STATUS_SUCCESS;
}

atmi_status_t atl_init_gpu_context() {
  if (atlc.struct_initialized == false)
    atmi_init_context_structs();
  if (atlc.g_gpu_initialized != false)
    return ATMI_STATUS_SUCCESS;

  hsa_status_t err;
  err = init_hsa();
  if (err != HSA_STATUS_SUCCESS)
    return ATMI_STATUS_ERROR;

  if (context_init_time_init == 0) {
    clock_gettime(CLOCK_MONOTONIC_RAW, &context_init_time);
    context_init_time_init = 1;
  }

  err = hsa_amd_register_system_event_handler(callbackEvent, NULL);
    ErrorCheck(Registering the system for memory faults, err);

    init_tasks();
    atlc.g_gpu_initialized = true;
    return ATMI_STATUS_SUCCESS;
}

bool isImplicit(KernelArgMD::ValueKind value_kind) {
  switch (value_kind) {
  case KernelArgMD::ValueKind::HiddenGlobalOffsetX:
  case KernelArgMD::ValueKind::HiddenGlobalOffsetY:
  case KernelArgMD::ValueKind::HiddenGlobalOffsetZ:
  case KernelArgMD::ValueKind::HiddenNone:
  case KernelArgMD::ValueKind::HiddenPrintfBuffer:
  case KernelArgMD::ValueKind::HiddenDefaultQueue:
  case KernelArgMD::ValueKind::HiddenCompletionAction:
  case KernelArgMD::ValueKind::HiddenMultiGridSyncArg:
  case KernelArgMD::ValueKind::HiddenHostcallBuffer:
    return true;
  default:
    return false;
  }
}

static std::pair<unsigned char *, unsigned char *>
find_metadata(void *binary, size_t binSize) {
  std::pair<unsigned char *, unsigned char *> failure = {nullptr, nullptr};

  Elf *e = elf_memory(static_cast<char *>(binary), binSize);
  if (elf_kind(e) != ELF_K_ELF) {
    return failure;
  }

  size_t numpHdrs;
  if (elf_getphdrnum(e, &numpHdrs) != 0) {
    return failure;
  }

  for (size_t i = 0; i < numpHdrs; ++i) {
    GElf_Phdr pHdr;
    if (gelf_getphdr(e, i, &pHdr) != &pHdr) {
      continue;
    }
    // Look for the runtime metadata note
    if (pHdr.p_type == PT_NOTE && pHdr.p_align >= sizeof(int)) {
      // Iterate over the notes in this segment
      address ptr = (address)binary + pHdr.p_offset;
      address segmentEnd = ptr + pHdr.p_filesz;

      while (ptr < segmentEnd) {
        Elf_Note *note = reinterpret_cast<Elf_Note *>(ptr);
        address name = (address)&note[1];

        if (note->n_type == 7 || note->n_type == 8) {
          return failure;
        } else if (note->n_type == 10 /* NT_AMD_AMDGPU_HSA_METADATA */ &&
                   note->n_namesz == sizeof "AMD" &&
                   !memcmp(name, "AMD", note->n_namesz)) {
          // code object v2 uses yaml metadata, no longer supported
          return failure;
        } else if (note->n_type == 32 /* NT_AMDGPU_METADATA */ &&
                   note->n_namesz == sizeof "AMDGPU" &&
                   !memcmp(name, "AMDGPU", note->n_namesz)) {

          // n_descsz = 485
          // value is padded to 4 byte alignment, may want to move end up to
          // match
          size_t offset = sizeof(uint32_t) * 3 /* fields */
                          + sizeof("AMDGPU")   /* name */
                          + 1 /* padding to 4 byte alignment */;

          // Including the trailing padding means both pointers are 4 bytes
          // aligned, which may be useful later.
          unsigned char *metadata_start = (unsigned char *)ptr + offset;
          unsigned char *metadata_end =
              metadata_start + core::alignUp(note->n_descsz, 4);
          return {metadata_start, metadata_end};
        }
        ptr += sizeof(*note) + core::alignUp(note->n_namesz, sizeof(int)) +
               core::alignUp(note->n_descsz, sizeof(int));
      }
    }
  }

  return failure;
}

namespace {
int map_lookup_array(msgpack::byte_range message, const char *needle,
                     msgpack::byte_range *res, uint64_t *size) {
  unsigned count = 0;
  struct s : msgpack::functors_defaults<s> {
    s(unsigned &count, uint64_t *size) : count(count), size(size) {}
    unsigned &count;
    uint64_t *size;
    const unsigned char *handle_array(uint64_t N, msgpack::byte_range bytes) {
      count++;
      *size = N;
      return bytes.end;
    }
  };

  msgpack::foreach_map(message,
                       [&](msgpack::byte_range key, msgpack::byte_range value) {
                         if (msgpack::message_is_string(key, needle)) {
                           // If the message is an array, record number of
                           // elements in *size
                           msgpack::handle_msgpack<s>(value, {count, size});
                           // return the whole array
                           *res = value;
                         }
                       });
  // Only claim success if exactly one key/array pair matched
  return count != 1;
}

int map_lookup_string(msgpack::byte_range message, const char *needle,
                      std::string *res) {
  unsigned count = 0;
  struct s : public msgpack::functors_defaults<s> {
    s(unsigned &count, std::string *res) : count(count), res(res) {}
    unsigned &count;
    std::string *res;
    void handle_string(size_t N, const unsigned char *str) {
      count++;
      *res = std::string(str, str + N);
    }
  };
  msgpack::foreach_map(message,
                       [&](msgpack::byte_range key, msgpack::byte_range value) {
                         if (msgpack::message_is_string(key, needle)) {
                           msgpack::handle_msgpack<s>(value, {count, res});
                         }
                       });
  return count != 1;
}

int map_lookup_uint64_t(msgpack::byte_range message, const char *needle,
                        uint64_t *res) {
  unsigned count = 0;
  msgpack::foreach_map(message,
                       [&](msgpack::byte_range key, msgpack::byte_range value) {
                         if (msgpack::message_is_string(key, needle)) {
                           msgpack::foronly_unsigned(value, [&](uint64_t x) {
                             count++;
                             *res = x;
                           });
                         }
                       });
  return count != 1;
}

int array_lookup_element(msgpack::byte_range message, uint64_t elt,
                         msgpack::byte_range *res) {
  int rc = 1;
  uint64_t i = 0;
  msgpack::foreach_array(message, [&](msgpack::byte_range value) {
    if (i == elt) {
      *res = value;
      rc = 0;
    }
    i++;
  });
  return rc;
}

int populate_kernelArgMD(msgpack::byte_range args_element,
                         KernelArgMD *kernelarg) {
  using namespace msgpack;
  int error = 0;
  foreach_map(args_element, [&](byte_range key, byte_range value) -> void {
    if (message_is_string(key, ".name")) {
      foronly_string(value, [&](size_t N, const unsigned char *str) {
        kernelarg->name_ = std::string(str, str + N);
      });
    } else if (message_is_string(key, ".type_name")) {
      foronly_string(value, [&](size_t N, const unsigned char *str) {
        kernelarg->typeName_ = std::string(str, str + N);
      });
    } else if (message_is_string(key, ".size")) {
      foronly_unsigned(value, [&](uint64_t x) { kernelarg->size_ = x; });
    } else if (message_is_string(key, ".offset")) {
      foronly_unsigned(value, [&](uint64_t x) { kernelarg->offset_ = x; });
    } else if (message_is_string(key, ".value_kind")) {
      foronly_string(value, [&](size_t N, const unsigned char *str) {
        std::string s = std::string(str, str + N);
        auto itValueKind = ArgValueKind.find(s);
        if (itValueKind != ArgValueKind.end()) {
          kernelarg->valueKind_ = itValueKind->second;
        }
      });
    }
  });
  return error;
}
} // namespace

static hsa_status_t get_code_object_custom_metadata(void *binary,
                                                    size_t binSize, int gpu) {
  // parse code object with different keys from v2
  // also, the kernel name is not the same as the symbol name -- so a
  // symbol->name map is needed

  std::pair<unsigned char *, unsigned char *> metadata =
      find_metadata(binary, binSize);
  if (!metadata.first) {
    return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  }

  uint64_t kernelsSize = 0;
  int msgpack_errors = 0;
  msgpack::byte_range kernel_array;
  msgpack_errors =
      map_lookup_array({metadata.first, metadata.second}, "amdhsa.kernels",
                       &kernel_array, &kernelsSize);
  msgpackErrorCheck(kernels lookup in program metadata, msgpack_errors);

  for (size_t i = 0; i < kernelsSize; i++) {
    assert(msgpack_errors == 0);
    std::string kernelName;
    std::string languageName;
    std::string symbolName;

    msgpack::byte_range element;
    msgpack_errors += array_lookup_element(kernel_array, i, &element);
    msgpackErrorCheck(element lookup in kernel metadata, msgpack_errors);

    msgpack_errors += map_lookup_string(element, ".name", &kernelName);
    msgpack_errors += map_lookup_string(element, ".language", &languageName);
    msgpack_errors += map_lookup_string(element, ".symbol", &symbolName);
    msgpackErrorCheck(strings lookup in kernel metadata, msgpack_errors);

    atl_kernel_info_t info = {0, 0, 0, 0, 0, {}, {}, {}};
    size_t kernel_explicit_args_size = 0;
    uint64_t kernel_segment_size;
    msgpack_errors += map_lookup_uint64_t(element, ".kernarg_segment_size",
                                          &kernel_segment_size);
    msgpackErrorCheck(kernarg segment size metadata lookup in kernel metadata,
                      msgpack_errors);

    // create a map from symbol to name
    DEBUG_PRINT("Kernel symbol %s; Name: %s; Size: %lu\n", symbolName.c_str(),
                kernelName.c_str(), kernel_segment_size);
    KernelNameMap[symbolName] = kernelName;

    bool hasHiddenArgs = false;
    if (kernel_segment_size > 0) {
      uint64_t argsSize;
      size_t offset = 0;

      msgpack::byte_range args_array;
      msgpack_errors +=
          map_lookup_array(element, ".args", &args_array, &argsSize);
      msgpackErrorCheck(kernel args metadata lookup in kernel metadata,
                        msgpack_errors);

      info.num_args = argsSize;

      for (size_t i = 0; i < argsSize; ++i) {
        KernelArgMD lcArg;

        msgpack::byte_range args_element;
        msgpack_errors += array_lookup_element(args_array, i, &args_element);
        msgpackErrorCheck(iterate args map in kernel args metadata,
                          msgpack_errors);

        msgpack_errors += populate_kernelArgMD(args_element, &lcArg);
        msgpackErrorCheck(iterate args map in kernel args metadata,
                          msgpack_errors);

        // TODO(ashwinma): should the below population actions be done only for
        // non-implicit args?
        // populate info with sizes and offsets
        info.arg_sizes.push_back(lcArg.size_);
        // v3 has offset field and not align field
        size_t new_offset = lcArg.offset_;
        size_t padding = new_offset - offset;
        offset = new_offset;
        info.arg_offsets.push_back(lcArg.offset_);
        DEBUG_PRINT("Arg[%lu] \"%s\" (%u, %u)\n", i, lcArg.name_.c_str(),
                    lcArg.size_, lcArg.offset_);
        offset += lcArg.size_;

        // check if the arg is a hidden/implicit arg
        // this logic assumes that all hidden args are 8-byte aligned
        if (!isImplicit(lcArg.valueKind_)) {
          kernel_explicit_args_size += lcArg.size_;
        } else {
          hasHiddenArgs = true;
        }
        kernel_explicit_args_size += padding;
      }
    }

    // add size of implicit args, e.g.: offset x, y and z and pipe pointer, but
    // in ATMI, do not count the compiler set implicit args, but set your own
    // implicit args by discounting the compiler set implicit args
    info.kernel_segment_size =
        (hasHiddenArgs ? kernel_explicit_args_size : kernel_segment_size) +
        sizeof(atmi_implicit_args_t);
    DEBUG_PRINT("[%s: kernarg seg size] (%lu --> %u)\n", kernelName.c_str(),
                kernel_segment_size, info.kernel_segment_size);

    // kernel received, now add it to the kernel info table
    KernelInfoTable[gpu][kernelName] = info;
  }

  return HSA_STATUS_SUCCESS;
}

static hsa_status_t populate_InfoTables(hsa_executable_t executable,
                                        hsa_executable_symbol_t symbol,
                                        void *data) {
  int gpu = *static_cast<int *>(data);
  hsa_symbol_kind_t type;

  uint32_t name_length;
  hsa_status_t err;
  err = hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_TYPE,
                                       &type);
  ErrorCheck(Symbol info extraction, err);
  DEBUG_PRINT("Exec Symbol type: %d\n", type);
  if (type == HSA_SYMBOL_KIND_KERNEL) {
    err = hsa_executable_symbol_get_info(
        symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &name_length);
    ErrorCheck(Symbol info extraction, err);
    char *name = reinterpret_cast<char *>(malloc(name_length + 1));
    err = hsa_executable_symbol_get_info(symbol,
                                         HSA_EXECUTABLE_SYMBOL_INFO_NAME, name);
    ErrorCheck(Symbol info extraction, err);
    name[name_length] = 0;

    if (KernelNameMap.find(std::string(name)) == KernelNameMap.end()) {
      // did not find kernel name in the kernel map; this can happen only
      // if the ROCr API for getting symbol info (name) is different from
      // the comgr method of getting symbol info
      ErrorCheck(Invalid kernel name, HSA_STATUS_ERROR_INVALID_CODE_OBJECT);
    }
    atl_kernel_info_t info;
    std::string kernelName = KernelNameMap[std::string(name)];
    // by now, the kernel info table should already have an entry
    // because the non-ROCr custom code object parsing is called before
    // iterating over the code object symbols using ROCr
    if (KernelInfoTable[gpu].find(kernelName) == KernelInfoTable[gpu].end()) {
      ErrorCheck(Finding the entry kernel info table,
                 HSA_STATUS_ERROR_INVALID_CODE_OBJECT);
    }
    // found, so assign and update
    info = KernelInfoTable[gpu][kernelName];

    /* Extract dispatch information from the symbol */
    err = hsa_executable_symbol_get_info(
        symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
        &(info.kernel_object));
    ErrorCheck(Extracting the symbol from the executable, err);
    err = hsa_executable_symbol_get_info(
        symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
        &(info.group_segment_size));
    ErrorCheck(Extracting the group segment size from the executable, err);
    err = hsa_executable_symbol_get_info(
        symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
        &(info.private_segment_size));
    ErrorCheck(Extracting the private segment from the executable, err);

    DEBUG_PRINT(
        "Kernel %s --> %lx symbol %u group segsize %u pvt segsize %u bytes "
        "kernarg\n",
        kernelName.c_str(), info.kernel_object, info.group_segment_size,
        info.private_segment_size, info.kernel_segment_size);

    // assign it back to the kernel info table
    KernelInfoTable[gpu][kernelName] = info;
    free(name);
  } else if (type == HSA_SYMBOL_KIND_VARIABLE) {
    err = hsa_executable_symbol_get_info(
        symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &name_length);
    ErrorCheck(Symbol info extraction, err);
    char *name = reinterpret_cast<char *>(malloc(name_length + 1));
    err = hsa_executable_symbol_get_info(symbol,
                                         HSA_EXECUTABLE_SYMBOL_INFO_NAME, name);
    ErrorCheck(Symbol info extraction, err);
    name[name_length] = 0;

    atl_symbol_info_t info;

    err = hsa_executable_symbol_get_info(
        symbol, HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS, &(info.addr));
    ErrorCheck(Symbol info address extraction, err);

    err = hsa_executable_symbol_get_info(
        symbol, HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SIZE, &(info.size));
    ErrorCheck(Symbol info size extraction, err);

    atmi_mem_place_t place = ATMI_MEM_PLACE(ATMI_DEVTYPE_GPU, gpu, 0);
    DEBUG_PRINT("Symbol %s = %p (%u bytes)\n", name, (void *)info.addr,
                info.size);
    register_allocation(reinterpret_cast<void *>(info.addr), (size_t)info.size,
                        place);
    SymbolInfoTable[gpu][std::string(name)] = info;
    if (strcmp(name, "needs_hostcall_buffer") == 0)
      g_atmi_hostcall_required = true;
    free(name);
  } else {
    DEBUG_PRINT("Symbol is an indirect function\n");
  }
  return HSA_STATUS_SUCCESS;
}

atmi_status_t Runtime::RegisterModuleFromMemory(
    void *module_bytes, size_t module_size, atmi_place_t place,
    atmi_status_t (*on_deserialized_data)(void *data, size_t size,
                                          void *cb_state),
    void *cb_state) {
  hsa_status_t err;
  int gpu = place.device_id;
  assert(gpu >= 0);

  DEBUG_PRINT("Trying to load module to GPU-%d\n", gpu);
  ATLGPUProcessor &proc = get_processor<ATLGPUProcessor>(place);
  hsa_agent_t agent = proc.agent();
  hsa_executable_t executable = {0};
  hsa_profile_t agent_profile;

  err = hsa_agent_get_info(agent, HSA_AGENT_INFO_PROFILE, &agent_profile);
  ErrorCheck(Query the agent profile, err);
  // FIXME: Assume that every profile is FULL until we understand how to build
  // GCN with base profile
  agent_profile = HSA_PROFILE_FULL;
  /* Create the empty executable.  */
  err = hsa_executable_create(agent_profile, HSA_EXECUTABLE_STATE_UNFROZEN, "",
                              &executable);
  ErrorCheck(Create the executable, err);

  bool module_load_success = false;
  do // Existing control flow used continue, preserve that for this patch
  {
    {
      // Some metadata info is not available through ROCr API, so use custom
      // code object metadata parsing to collect such metadata info

      err = get_code_object_custom_metadata(module_bytes, module_size, gpu);
      ErrorCheckAndContinue(Getting custom code object metadata, err);

      // Deserialize code object.
      hsa_code_object_t code_object = {0};
      err = hsa_code_object_deserialize(module_bytes, module_size, NULL,
                                        &code_object);
      ErrorCheckAndContinue(Code Object Deserialization, err);
      assert(0 != code_object.handle);

      // Mutating the device image here avoids another allocation & memcpy
      void *code_object_alloc_data =
          reinterpret_cast<void *>(code_object.handle);
      atmi_status_t atmi_err =
          on_deserialized_data(code_object_alloc_data, module_size, cb_state);
      ATMIErrorCheck(Error in deserialized_data callback, atmi_err);

      /* Load the code object.  */
      err =
          hsa_executable_load_code_object(executable, agent, code_object, NULL);
      ErrorCheckAndContinue(Loading the code object, err);

      // cannot iterate over symbols until executable is frozen
    }
    module_load_success = true;
  } while (0);
  DEBUG_PRINT("Modules loaded successful? %d\n", module_load_success);
  if (module_load_success) {
    /* Freeze the executable; it can now be queried for symbols.  */
    err = hsa_executable_freeze(executable, "");
    ErrorCheck(Freeze the executable, err);

    err = hsa_executable_iterate_symbols(executable, populate_InfoTables,
                                         static_cast<void *>(&gpu));
    ErrorCheck(Iterating over symbols for execuatable, err);

    // save the executable and destroy during finalize
    g_executables.push_back(executable);
    return ATMI_STATUS_SUCCESS;
  } else {
    return ATMI_STATUS_ERROR;
  }
}

} // namespace core
