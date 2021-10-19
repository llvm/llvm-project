//===--- amdgpu/impl/interop_hsa.cpp ------------------------------ C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "interop_hsa.h"
#include "internal.h"

hsa_status_t interop_hsa_get_symbol_info(
    const std::map<std::string, atl_symbol_info_t> &SymbolInfoTable,
    int DeviceId, const char *symbol, void **var_addr, unsigned int *var_size) {
  /*
     // Typical usage:
     void *var_addr;
     size_t var_size;
     interop_hsa_get_symbol_addr(gpu_place, "symbol_name", &var_addr,
     &var_size);
     impl_memcpy(signal, host_add, var_addr, var_size);
  */

  if (!symbol || !var_addr || !var_size)
    return HSA_STATUS_ERROR;

  // get the symbol info
  std::string symbolStr = std::string(symbol);
  auto It = SymbolInfoTable.find(symbolStr);
  if (It != SymbolInfoTable.end()) {
    atl_symbol_info_t info = It->second;
    *var_addr = reinterpret_cast<void *>(info.addr);
    *var_size = info.size;
    return HSA_STATUS_SUCCESS;
  } else {
    *var_addr = NULL;
    *var_size = 0;
    return HSA_STATUS_ERROR;
  }
}

hsa_status_t interop_hsa_get_kernel_info(
    const std::map<std::string, atl_kernel_info_t> &KernelInfoTable,
    int DeviceId, const char *kernel_name,
    hsa_executable_symbol_info_t kernel_info, uint32_t *value) {
  /*
     // Typical usage:
     uint32_t value;
     interop_hsa_get_kernel_addr(gpu_place, "kernel_name",
                                  HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
                                  &val);
  */

  if (!kernel_name || !value)
    return HSA_STATUS_ERROR;

  hsa_status_t status = HSA_STATUS_SUCCESS;
  // get the kernel info
  std::string kernelStr = std::string(kernel_name);
  auto It = KernelInfoTable.find(kernelStr);
  if (It != KernelInfoTable.end()) {
    atl_kernel_info_t info = It->second;
    switch (kernel_info) {
    case HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE:
      *value = info.group_segment_size;
      break;
    case HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE:
      *value = info.private_segment_size;
      break;
    case HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE:
      // return the size for non-implicit args
      *value = info.kernel_segment_size - sizeof(impl_implicit_args_t);
      break;
    default:
      *value = 0;
      status = HSA_STATUS_ERROR;
      break;
    }
  } else {
    *value = 0;
    status = HSA_STATUS_ERROR;
  }

  return status;
}
