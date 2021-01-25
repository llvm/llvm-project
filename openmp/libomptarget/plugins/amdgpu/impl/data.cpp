/*===--------------------------------------------------------------------------
 *              ATMI (Asynchronous Task and Memory Interface)
 *
 * This file is distributed under the MIT License. See LICENSE.txt for details.
 *===------------------------------------------------------------------------*/
#include "atmi_runtime.h"
#include "internal.h"
#include "machine.h"
#include "rt.h"
#include <cassert>
#include <hsa.h>
#include <hsa_ext_amd.h>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <thread>
#include <vector>

using core::TaskImpl;
extern ATLMachine g_atl_machine;

namespace core {
void allow_access_to_all_gpu_agents(void *ptr);

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

ATLProcessor &get_processor_by_mem_place(atmi_mem_place_t place) {
  int dev_id = place.dev_id;
  switch (place.dev_type) {
  case ATMI_DEVTYPE_CPU:
    return g_atl_machine.processors<ATLCPUProcessor>()[dev_id];
  case ATMI_DEVTYPE_GPU:
    return g_atl_machine.processors<ATLGPUProcessor>()[dev_id];
  }
}

hsa_amd_memory_pool_t get_memory_pool_by_mem_place(atmi_mem_place_t place) {
  ATLProcessor &proc = get_processor_by_mem_place(place);
  return get_memory_pool(proc, place.mem_id);
}

void register_allocation(void *ptr, size_t size, atmi_mem_place_t place) {
  if (place.dev_type == ATMI_DEVTYPE_CPU)
    allow_access_to_all_gpu_agents(ptr);
}

atmi_status_t Runtime::Malloc(void **ptr, size_t size, atmi_mem_place_t place) {
  atmi_status_t ret = ATMI_STATUS_SUCCESS;
  hsa_amd_memory_pool_t pool = get_memory_pool_by_mem_place(place);
  hsa_status_t err = hsa_amd_memory_pool_allocate(pool, size, 0, ptr);
  ErrorCheck(atmi_malloc, err);
  DEBUG_PRINT("Malloced [%s %d] %p\n",
              place.dev_type == ATMI_DEVTYPE_CPU ? "CPU" : "GPU", place.dev_id,
              *ptr);
  if (err != HSA_STATUS_SUCCESS)
    ret = ATMI_STATUS_ERROR;

  register_allocation(*ptr, size, place);

  return ret;
}

atmi_status_t Runtime::Memfree(void *ptr) {
  atmi_status_t ret = ATMI_STATUS_SUCCESS;
  hsa_status_t err;
  err = hsa_amd_memory_pool_free(ptr);
  ErrorCheck(atmi_free, err);
  DEBUG_PRINT("Freed %p\n", ptr);

  if (err != HSA_STATUS_SUCCESS)
    ret = ATMI_STATUS_ERROR;
  return ret;
}

} // namespace core
