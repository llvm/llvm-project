//===--- Level Zero Target RTL Implementation -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Level Zero compatibility layer enabling us to compile using new APIs.
//
//===----------------------------------------------------------------------===//

#ifndef OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0COMPAT_H
#define OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0COMPAT_H

#include "APIHelpers.h"

#include <level_zero/ze_api.h>

API_HELPER_OPTIONAL(ze_result_t, zeCommandListAppendLaunchKernelWithArguments,
                    ze_command_list_handle_t hCommandList,
                    ze_kernel_handle_t hKernel,
                    const ze_group_count_t groupCounts,
                    const ze_group_size_t groupSizes, void **pArguments,
                    const void *pNext, ze_event_handle_t hSignalEvent,
                    uint32_t numWaitEvents, ze_event_handle_t *phWaitEvents);

#endif // OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0COMPAT_H
