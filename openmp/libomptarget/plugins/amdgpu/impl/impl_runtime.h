//===--- amdgpu/impl/impl_runtime.h ------------------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef INCLUDE_IMPL_RUNTIME_H_
#define INCLUDE_IMPL_RUNTIME_H_

#include "impl.h"
#include "hsa_api.h"

#ifndef TARGET_NAME
#define TARGET_NAME AMDGPU
#endif
#ifdef __cplusplus
extern "C" {
#endif

hsa_status_t impl_module_register_from_memory_to_place(
    void *module_bytes, size_t module_size, int DeviceId,
    hsa_status_t (*on_deserialized_data)(void *data, size_t size,
                                         void *cb_state),
    void *cb_state);

hsa_status_t impl_memcpy_h2d(hsa_signal_t signal, void *deviceDest,
                             const void *hostSrc, size_t size,
                             hsa_agent_t agent);

hsa_status_t impl_memcpy_d2h(hsa_signal_t sig, void *hostDest,
                             const void *deviceSrc, size_t size,
                             hsa_agent_t agent);
#ifdef __cplusplus
}
#endif

#endif // INCLUDE_IMPL_RUNTIME_H_
