//===--- amdgpu/impl/impl_runtime.h ------------------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
// Notified per clause 4(b) of the license.
//
//===----------------------------------------------------------------------===//
#ifndef INCLUDE_IMPL_RUNTIME_H_
#define INCLUDE_IMPL_RUNTIME_H_

#include "hsa_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Check if pointer ptr is already locked and return true
// if so. Return false otherwise. In case of error, return false and
// the HSA error code in err_p. err_p can be nullptr,
// in which case do not return error code
// The new pointer, accessible by the HSA agents that were passed in to the
// original lock function, is returned in the memory location pointed to by
// agentBaseAddress. If this parameter is NULL, the pointer is not returned
bool already_locked(void *ptr, hsa_status_t *err_p, void **agentBaseAddress);

// wait for signal to turn to success
hsa_status_t wait_for_signal(hsa_signal_t signal, hsa_signal_value_t init,
                             hsa_signal_value_t success);

hsa_status_t impl_module_register_from_memory_to_place(
    void *module_bytes, size_t module_size, int DeviceId,
    hsa_status_t (*on_deserialized_data)(void *data, size_t size,
                                         void *cb_state),
    void *cb_state);

hsa_status_t impl_memcpy_h2d(hsa_signal_t signal, void *deviceDest,
                             void *hostSrc, size_t size,
                             hsa_agent_t device_agent,
                             hsa_amd_memory_pool_t MemoryPool,
                             bool *user_locked);

hsa_status_t impl_memcpy_d2h(hsa_signal_t sig, void *hostDest, void *deviceSrc,
                             size_t size, hsa_agent_t device_agent,
                             hsa_amd_memory_pool_t MemoryPool,
                             bool *user_locked);

#ifdef __cplusplus
}
#endif
#endif // INCLUDE_IMPL_RUNTIME_H_
