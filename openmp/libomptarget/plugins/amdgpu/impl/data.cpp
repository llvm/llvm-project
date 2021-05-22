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

namespace {
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
} // namespace

hsa_status_t register_allocation(void *ptr, size_t size,
                                 atmi_mem_place_t place) {
  if (place.dev_type == ATMI_DEVTYPE_CPU)
    return allow_access_to_all_gpu_agents(ptr);
  else
    return HSA_STATUS_SUCCESS;
}

atmi_status_t Runtime::Malloc(void **ptr, size_t size, atmi_mem_place_t place) {
  hsa_amd_memory_pool_t pool = get_memory_pool_by_mem_place(place);
  hsa_status_t err = hsa_amd_memory_pool_allocate(pool, size, 0, ptr);
  DEBUG_PRINT("Malloced [%s %d] %p\n",
              place.dev_type == ATMI_DEVTYPE_CPU ? "CPU" : "GPU", place.dev_id,
              *ptr);

  if (err == HSA_STATUS_SUCCESS) {
    err = register_allocation(*ptr, size, place);
  }

  return (err == HSA_STATUS_SUCCESS) ? ATMI_STATUS_SUCCESS : ATMI_STATUS_ERROR;
}

atmi_status_t Runtime::Memfree(void *ptr) {
  hsa_status_t err = hsa_amd_memory_pool_free(ptr);
  DEBUG_PRINT("Freed %p\n", ptr);

  return (err == HSA_STATUS_SUCCESS) ? ATMI_STATUS_SUCCESS : ATMI_STATUS_ERROR;
}

} // namespace core
