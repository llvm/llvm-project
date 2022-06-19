//===--- amdgpu/impl/rt.h ----------------------------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef SRC_RUNTIME_INCLUDE_RT_H_
#define SRC_RUNTIME_INCLUDE_RT_H_

#include "hsa_api.h"
#include "impl_runtime.h"
#include "internal.h"

#include <string>

namespace core {
namespace Runtime {
hsa_status_t Memfree(void *);
hsa_status_t HostMalloc(void **ptr, size_t size,
                        hsa_amd_memory_pool_t MemoryPool);

} // namespace Runtime
hsa_status_t RegisterModuleFromMemory(
    std::map<std::string, atl_kernel_info_t> &KernelInfoTable,
    std::map<std::string, atl_symbol_info_t> &SymbolInfoTable,
    void *module_bytes, size_t module_size, hsa_agent_t agent,
    hsa_status_t (*on_deserialized_data)(void *data, size_t size,
                                         void *cb_state),
    void *cb_state, std::vector<hsa_executable_t> &HSAExecutables);

} // namespace core

#endif // SRC_RUNTIME_INCLUDE_RT_H_
