/*===--------------------------------------------------------------------------
 *              ATMI (Asynchronous Task and Memory Interface)
 *
 * This file is distributed under the MIT License. See LICENSE.txt for details.
 *===------------------------------------------------------------------------*/
#include "atmi_interop_hsa.h"
#include "internal.h"

using core::atl_is_atmi_initialized;

atmi_status_t atmi_interop_hsa_get_symbol_info(atmi_mem_place_t place,
                                               const char *symbol,
                                               void **var_addr,
                                               unsigned int *var_size) {
  /*
     // Typical usage:
     void *var_addr;
     size_t var_size;
     atmi_interop_hsa_get_symbol_addr(gpu_place, "symbol_name", &var_addr,
     &var_size);
     atmi_memcpy(host_add, var_addr, var_size);
  */

  if (!atl_is_atmi_initialized())
    return ATMI_STATUS_ERROR;
  atmi_machine_t *machine = atmi_machine_get_info();
  if (!symbol || !var_addr || !var_size || !machine)
    return ATMI_STATUS_ERROR;
  if (place.dev_id < 0 ||
      place.dev_id >= machine->device_count_by_type[place.dev_type])
    return ATMI_STATUS_ERROR;

  // get the symbol info
  std::string symbolStr = std::string(symbol);
  if (SymbolInfoTable[place.dev_id].find(symbolStr) !=
      SymbolInfoTable[place.dev_id].end()) {
    atl_symbol_info_t info = SymbolInfoTable[place.dev_id][symbolStr];
    *var_addr = reinterpret_cast<void *>(info.addr);
    *var_size = info.size;
    return ATMI_STATUS_SUCCESS;
  } else {
    *var_addr = NULL;
    *var_size = 0;
    return ATMI_STATUS_ERROR;
  }
}

atmi_status_t atmi_interop_hsa_get_kernel_info(
    atmi_mem_place_t place, const char *kernel_name,
    hsa_executable_symbol_info_t kernel_info, uint32_t *value) {
  /*
     // Typical usage:
     uint32_t value;
     atmi_interop_hsa_get_kernel_addr(gpu_place, "kernel_name",
                                  HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
                                  &val);
  */

  if (!atl_is_atmi_initialized())
    return ATMI_STATUS_ERROR;
  atmi_machine_t *machine = atmi_machine_get_info();
  if (!kernel_name || !value || !machine)
    return ATMI_STATUS_ERROR;
  if (place.dev_id < 0 ||
      place.dev_id >= machine->device_count_by_type[place.dev_type])
    return ATMI_STATUS_ERROR;

  atmi_status_t status = ATMI_STATUS_SUCCESS;
  // get the kernel info
  std::string kernelStr = std::string(kernel_name);
  if (KernelInfoTable[place.dev_id].find(kernelStr) !=
      KernelInfoTable[place.dev_id].end()) {
    atl_kernel_info_t info = KernelInfoTable[place.dev_id][kernelStr];
    switch (kernel_info) {
    case HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE:
      *value = info.group_segment_size;
      break;
    case HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE:
      *value = info.private_segment_size;
      break;
    case HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE:
      // return the size for non-implicit args
      *value = info.kernel_segment_size - sizeof(atmi_implicit_args_t);
      break;
    default:
      *value = 0;
      status = ATMI_STATUS_ERROR;
      break;
    }
  } else {
    *value = 0;
    status = ATMI_STATUS_ERROR;
  }

  return status;
}
