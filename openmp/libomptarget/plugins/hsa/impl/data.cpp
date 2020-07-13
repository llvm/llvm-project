/*===--------------------------------------------------------------------------
 *              ATMI (Asynchronous Task and Memory Interface)
 *
 * This file is distributed under the MIT License. See LICENSE.txt for details.
 *===------------------------------------------------------------------------*/
#include "data.h"
#include <hsa.h>
#include <hsa_ext_amd.h>
#include <stdio.h>
#include <string.h>
#include <cassert>
#include <iostream>
#include <thread>
#include <vector>
#include "atmi_runtime.h"
#include "internal.h"
#include "machine.h"
#include "rt.h"

using core::TaskImpl;
extern ATLMachine g_atl_machine;
extern hsa_signal_t IdentityCopySignal;

namespace core {
#ifndef USE_ROCR_PTR_INFO
ATLPointerTracker g_data_map;  // Track all am pointer allocations.
#endif
void allow_access_to_all_gpu_agents(void *ptr);
// std::map<void *, ATLData *> MemoryMap;

const char *getPlaceStr(atmi_devtype_t type) {
  switch (type) {
    case ATMI_DEVTYPE_CPU:
      return "CPU";
    case ATMI_DEVTYPE_GPU:
      return "GPU";
    default:
      return NULL;
  }
}

std::ostream &operator<<(std::ostream &os, const ATLData *ap) {
  atmi_mem_place_t place = ap->place();
  os << "hostPointer:" << ap->host_aliasptr() << " devicePointer:" << ap->ptr()
     << " sizeBytes:" << ap->size() << " place:(" << getPlaceStr(place.dev_type)
     << ", " << place.dev_id << ", " << place.mem_id << ")";
  return os;
}

#ifndef USE_ROCR_PTR_INFO
void ATLPointerTracker::insert(void *pointer, ATLData *p) {
  std::lock_guard<std::mutex> l(mutex_);

  DEBUG_PRINT("insert: %p + %zu\n", pointer, p->size());
  tracker_.insert(std::make_pair(ATLMemoryRange(pointer, p->size()), p));
}

void ATLPointerTracker::remove(void *pointer) {
  std::lock_guard<std::mutex> l(mutex_);
  DEBUG_PRINT("remove: %p\n", pointer);
  tracker_.erase(ATLMemoryRange(pointer, 1));
}

ATLData *ATLPointerTracker::find(const void *pointer) {
  std::lock_guard<std::mutex> l(mutex_);
  ATLData *ret = NULL;
  auto iter = tracker_.find(ATLMemoryRange(pointer, 1));
  DEBUG_PRINT("find: %p\n", pointer);
  if (iter != tracker_.end())  // found
    ret = iter->second;
  return ret;
}
#endif

ATLProcessor &get_processor_by_compute_place(atmi_place_t place) {
  int dev_id = place.device_id;
  switch (place.type) {
    case ATMI_DEVTYPE_CPU:
      return g_atl_machine.processors<ATLCPUProcessor>()[dev_id];
    case ATMI_DEVTYPE_GPU:
      return g_atl_machine.processors<ATLGPUProcessor>()[dev_id];
  }
}

ATLProcessor &get_processor_by_mem_place(atmi_mem_place_t place) {
  int dev_id = place.dev_id;
  switch (place.dev_type) {
    case ATMI_DEVTYPE_CPU:
      return g_atl_machine.processors<ATLCPUProcessor>()[dev_id];
    case ATMI_DEVTYPE_GPU:
      return g_atl_machine.processors<ATLGPUProcessor>()[dev_id];
  }
}

hsa_agent_t get_compute_agent(atmi_place_t place) {
  return get_processor_by_compute_place(place).agent();
}

static hsa_agent_t get_mem_agent(atmi_mem_place_t place) {
  return get_processor_by_mem_place(place).agent();
}

hsa_amd_memory_pool_t get_memory_pool_by_mem_place(atmi_mem_place_t place) {
  ATLProcessor &proc = get_processor_by_mem_place(place);
  return get_memory_pool(proc, place.mem_id);
}

void register_allocation(void *ptr, size_t size, atmi_mem_place_t place) {
#ifndef USE_ROCR_PTR_INFO
  ATLData *data = new ATLData(ptr, NULL, size, place, ATMI_IN_OUT);
  g_data_map.insert(ptr, data);
#else
  ATLData *data = new ATLData(ptr, NULL, size, place, ATMI_IN_OUT);

  hsa_status_t err = hsa_amd_pointer_info_set_userdata(ptr, data);
  ErrorCheck(Setting pointer info with user data, err);
#endif
  if (place.dev_type == ATMI_DEVTYPE_CPU) allow_access_to_all_gpu_agents(ptr);
  // TODO(ashwinma): what if one GPU wants to access another GPU?
}

atmi_status_t Runtime::Malloc(void **ptr, size_t size, atmi_mem_place_t place) {
  atmi_status_t ret = ATMI_STATUS_SUCCESS;
  hsa_amd_memory_pool_t pool = get_memory_pool_by_mem_place(place);
  hsa_status_t err = hsa_amd_memory_pool_allocate(pool, size, 0, ptr);
  ErrorCheck(atmi_malloc, err);
  DEBUG_PRINT("Malloced [%s %d] %p\n",
              place.dev_type == ATMI_DEVTYPE_CPU ? "CPU" : "GPU", place.dev_id,
              *ptr);
  if (err != HSA_STATUS_SUCCESS) ret = ATMI_STATUS_ERROR;

  register_allocation(*ptr, size, place);

  return ret;
}

atmi_status_t Runtime::Memfree(void *ptr) {
  atmi_status_t ret = ATMI_STATUS_SUCCESS;
  hsa_status_t err;
#ifndef USE_ROCR_PTR_INFO
  ATLData *data = g_data_map.find(ptr);
#else
  hsa_amd_pointer_info_t ptr_info;
  ptr_info.size = sizeof(hsa_amd_pointer_info_t);
  err = hsa_amd_pointer_info(reinterpret_cast<void *>(ptr), &ptr_info,
                             NULL,  /* alloc fn ptr */
                             NULL,  /* num_agents_accessible */
                             NULL); /* accessible agents */
  ErrorCheck(Checking pointer info, err);

  ATLData *data = reinterpret_cast<ATLData *>(ptr_info.userData);
#endif
  if (!data)
    ErrorCheck(Checking pointer info userData,
               HSA_STATUS_ERROR_INVALID_ALLOCATION);

#ifndef USE_ROCR_PTR_INFO
  g_data_map.remove(ptr);
#else
// is there a way to unset a userdata with AMD pointer before deleting 'data'?
#endif
  delete data;

  err = hsa_amd_memory_pool_free(ptr);
  ErrorCheck(atmi_free, err);
  DEBUG_PRINT("Freed %p\n", ptr);

  if (err != HSA_STATUS_SUCCESS || !data) ret = ATMI_STATUS_ERROR;
  return ret;
}

atmi_status_t Runtime::Memcpy(void *dest, const void *src, size_t size) {
  atmi_status_t ret;
  hsa_status_t err;

#ifndef USE_ROCR_PTR_INFO
  ATLData *volatile src_data = g_data_map.find(src);
  ATLData *volatile dest_data = g_data_map.find(dest);
#else
  hsa_amd_pointer_info_t src_ptr_info;
  hsa_amd_pointer_info_t dest_ptr_info;
  src_ptr_info.size = sizeof(hsa_amd_pointer_info_t);
  dest_ptr_info.size = sizeof(hsa_amd_pointer_info_t);
  err = hsa_amd_pointer_info(reinterpret_cast<void *>(src), &src_ptr_info,
                             NULL,  /* alloc fn ptr */
                             NULL,  /* num_agents_accessible */
                             NULL); /* accessible agents */
  ErrorCheck(Checking src pointer info, err);
  err = hsa_amd_pointer_info(reinterpret_cast<void *>(dest), &dest_ptr_info,
                             NULL,  /* alloc fn ptr */
                             NULL,  /* num_agents_accessible */
                             NULL); /* accessible agents */
  ErrorCheck(Checking dest pointer info, err);
  ATLData *volatile src_data =
      reinterpret_cast<ATLData *>(src_ptr_info.userData);
  ATLData *volatile dest_data =
      reinterpret_cast<ATLData *>(dest_ptr_info.userData);
#endif
  atmi_mem_place_t cpu = ATMI_MEM_PLACE_CPU_MEM(0, 0, 0);
  hsa_agent_t cpu_agent = get_mem_agent(cpu);
  hsa_agent_t src_agent;
  hsa_agent_t dest_agent;
  void *temp_host_ptr;
  const void *src_ptr = src;
  void *dest_ptr = dest;
  volatile Direction type;
  if (src_data && !dest_data) {
    type = Direction::ATMI_D2H;
    src_agent = get_mem_agent(src_data->place());
    dest_agent = src_agent;
    // dest_agent = cpu_agent; // FIXME: can the two agents be the GPU agent
    // itself?
    ret = atmi_malloc(&temp_host_ptr, size, cpu);
    // err = hsa_amd_agents_allow_access(1, &src_agent, NULL, temp_host_ptr);
    // ErrorCheck(Allow access to ptr, err);
    src_ptr = src;
    dest_ptr = temp_host_ptr;
  } else if (!src_data && dest_data) {
    type = Direction::ATMI_H2D;
    dest_agent = get_mem_agent(dest_data->place());
    // src_agent = cpu_agent; // FIXME: can the two agents be the GPU agent
    // itself?
    src_agent = dest_agent;
    ret = atmi_malloc(&temp_host_ptr, size, cpu);
    memcpy(temp_host_ptr, src, size);
    // FIXME: ideally lock would be the better approach, but we need to try to
    // understand why the h2d copy segfaults if we dont have the below lines
    // err = hsa_amd_agents_allow_access(1, &dest_agent, NULL, temp_host_ptr);
    // ErrorCheck(Allow access to ptr, err);
    src_ptr = (const void *)temp_host_ptr;
    dest_ptr = dest;
  } else if (!src_data && !dest_data) {
    type = Direction::ATMI_H2H;
    src_agent = cpu_agent;
    dest_agent = cpu_agent;
    src_ptr = src;
    dest_ptr = dest;
  } else {
    type = Direction::ATMI_D2D;
    src_agent = get_mem_agent(src_data->place());
    dest_agent = get_mem_agent(dest_data->place());
    src_ptr = src;
    dest_ptr = dest;
  }
  DEBUG_PRINT("Memcpy source agent: %lu\n", src_agent.handle);
  DEBUG_PRINT("Memcpy dest agent: %lu\n", dest_agent.handle);
  hsa_signal_store_release(IdentityCopySignal, 1);
  // hsa_signal_add_acq_rel(IdentityCopySignal, 1);
  err = hsa_amd_memory_async_copy(dest_ptr, dest_agent, src_ptr, src_agent,
                                  size, 0, NULL, IdentityCopySignal);
  ErrorCheck(Copy async between memory pools, err);
  hsa_signal_wait_acquire(IdentityCopySignal, HSA_SIGNAL_CONDITION_EQ, 0,
                          UINT64_MAX, ATMI_WAIT_STATE);

  // cleanup for D2H and H2D
  if (type == Direction::ATMI_D2H) {
    memcpy(dest, temp_host_ptr, size);
    ret = atmi_free(temp_host_ptr);
  } else if (type == Direction::ATMI_H2D) {
    ret = atmi_free(temp_host_ptr);
  }
  if (err != HSA_STATUS_SUCCESS || ret != ATMI_STATUS_SUCCESS)
    ret = ATMI_STATUS_ERROR;
  return ret;
}

}  // namespace core
