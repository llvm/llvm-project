//===--- amdgpu/impl/interop_hsa.h -------------------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef INCLUDE_INTEROP_HSA_H_
#define INCLUDE_INTEROP_HSA_H_

#include "impl_runtime.h"
#include "hsa_api.h"
#include "internal.h"

#include <map>
#include <string>

extern "C" {

hsa_status_t interop_hsa_get_symbol_info(
    const std::map<std::string, atl_symbol_info_t> &SymbolInfoTable,
    int DeviceId, const char *symbol, void **var_addr, unsigned int *var_size);

hsa_status_t interop_hsa_get_kernel_info(
    const std::map<std::string, atl_kernel_info_t> &KernelInfoTable,
    int DeviceId, const char *kernel_name, hsa_executable_symbol_info_t info,
    uint32_t *value);

}

#endif // INCLUDE_INTEROP_HSA_H_
